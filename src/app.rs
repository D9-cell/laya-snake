use crate::{
    ai::Decision,
    game::{estimate_risk_and_reachability, Dir, GameSnapshot, SnakeGame},
    scores,
};
use crossterm::{
    cursor::{Hide, MoveTo, Show},
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind},
    execute, queue,
    style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor},
    terminal::{
        self, BeginSynchronizedUpdate, Clear, ClearType, EndSynchronizedUpdate,
        EnterAlternateScreen, LeaveAlternateScreen,
    },
};
use std::{
    collections::VecDeque,
    io::{stdout, Write},
    sync::{
        mpsc::{self, Receiver, TryRecvError},
        Arc, Condvar, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

pub(crate) const SPEED_LEVELS_MS: [u64; 5] = [1500, 900, 500, 200, 0];
pub(crate) const MIN_W: usize = 10;
pub(crate) const MIN_H: usize = 8;
const SCALE_STEP_W: usize = 4;
const SCALE_STEP_H: usize = 2;
/// While a model call is in flight the "thinking" timer is redrawn this often.
const THINK_REDRAW: Duration = Duration::from_millis(100);
/// The elapsed-time clock in the corner ticks once a second.
const CLOCK_REDRAW: Duration = Duration::from_secs(1);

pub(crate) struct Request {
    snapshot: GameSnapshot,
    generation: u64,
}

struct ResultMessage {
    generation: u64,
    decision: Result<Decision, String>,
}

struct Slot {
    request: Option<Request>,
    stop: bool,
}

struct Shared {
    slot: Mutex<Slot>,
    wake: Condvar,
}

/// Runs model calls on a background thread. Only the *latest* request is kept: a newer request
/// replaces one that has not started yet, so a request can never be silently lost behind a stale
/// one (which would leave the game waiting for a result that never comes).
pub(crate) struct InferenceWorker {
    shared: Arc<Shared>,
    result_rx: Receiver<ResultMessage>,
}

impl InferenceWorker {
    pub(crate) fn new<F>(decide: F) -> Self
    where
        F: Fn(&GameSnapshot) -> Result<Decision, String> + Send + 'static,
    {
        let shared = Arc::new(Shared {
            slot: Mutex::new(Slot {
                request: None,
                stop: false,
            }),
            wake: Condvar::new(),
        });
        let (result_tx, result_rx) = mpsc::channel::<ResultMessage>();
        let worker_shared = Arc::clone(&shared);

        // Detached on purpose: quitting must not wait for a model call that is still running; the
        // thread ends with the process (or as soon as it notices `stop`).
        thread::Builder::new()
            .name("laya-worker".into())
            .spawn(move || loop {
                let request = {
                    let mut slot = worker_shared.slot.lock().unwrap_or_else(|e| e.into_inner());
                    loop {
                        if slot.stop {
                            return;
                        }
                        if let Some(r) = slot.request.take() {
                            break r;
                        }
                        slot = worker_shared
                            .wake
                            .wait(slot)
                            .unwrap_or_else(|e| e.into_inner());
                    }
                };
                let decision = decide(&request.snapshot);
                if result_tx
                    .send(ResultMessage {
                        generation: request.generation,
                        decision,
                    })
                    .is_err()
                {
                    return;
                }
            })
            .expect("failed to spawn inference thread");

        Self { shared, result_rx }
    }

    fn request(&self, game: &SnakeGame, generation: u64) {
        let mut slot = self.shared.slot.lock().unwrap_or_else(|e| e.into_inner());
        slot.request = Some(Request {
            snapshot: GameSnapshot::from_game(game),
            generation,
        });
        self.shared.wake.notify_one();
    }

    fn try_result(&self) -> Option<ResultMessage> {
        match self.result_rx.try_recv() {
            Ok(r) => Some(r),
            Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => None,
        }
    }
}

impl Drop for InferenceWorker {
    fn drop(&mut self) {
        let mut slot = self.shared.slot.lock().unwrap_or_else(|e| e.into_inner());
        slot.stop = true;
        self.shared.wake.notify_all();
    }
}

fn bar(width: usize, frac: f64, filled: char, empty: char) -> String {
    let frac = frac.clamp(0.0, 1.0);
    let n = (width as f64 * frac).round() as usize;
    std::iter::repeat_n(filled, n)
        .chain(std::iter::repeat_n(empty, width - n))
        .collect()
}

fn fit_dims(cols: u16, rows: u16, mut width: usize, mut height: usize) -> (usize, usize) {
    let avail_w = (cols as usize).saturating_sub(4 + 4 + 36) / 2;
    let avail_h = (rows as usize).saturating_sub(6 + 6);
    if avail_w > 0 {
        width = width.clamp(MIN_W, avail_w.max(MIN_W));
    }
    if avail_h > 0 {
        height = height.clamp(MIN_H, avail_h.max(MIN_H));
    }
    (width, height)
}

fn draw_text(
    out: &mut impl Write,
    x: u16,
    y: u16,
    text: &str,
    color: Color,
) -> std::io::Result<()> {
    queue!(
        out,
        MoveTo(x, y),
        SetForegroundColor(color),
        Print(text),
        ResetColor
    )
}

fn direction_glyph(direction: Dir) -> &'static str {
    match direction {
        Dir::Up => "▲",
        Dir::Down => "▼",
        Dir::Left => "◀",
        Dir::Right => "▶",
    }
}

fn trail_glyph(direction: Dir) -> &'static str {
    match direction {
        Dir::Up => "↑",
        Dir::Down => "↓",
        Dir::Left => "←",
        Dir::Right => "→",
    }
}

/// The move to execute and whether the safety shield overrode the model's top choice. The shield
/// walks the directions from most to least likely and takes the first one that is not fatal; ties
/// go to the model's own pick.
pub(crate) fn apply_shield(decision: &Decision, game: &SnakeGame, enabled: bool) -> (Dir, bool) {
    let mut ranked: Vec<Dir> = std::iter::once(decision.top)
        .chain(Dir::ALL.into_iter().filter(|d| *d != decision.top))
        .collect();
    ranked.sort_by(|a, b| {
        decision.probs[b.index()]
            .partial_cmp(&decision.probs[a.index()])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top = ranked[0];
    if !enabled {
        return (top, false);
    }
    for &direction in &ranked {
        if !game.is_fatal(direction) {
            return (direction, direction != top);
        }
    }
    (top, false)
}

/// What the UI prints about the model backend.
pub struct BrainInfo {
    pub checkpoint: String,
    pub engine: String,
    pub device: String,
}

/// `--mock` runs never read or write the saved best scores: they are not the model's.
fn is_mock(info: &BrainInfo) -> bool {
    info.checkpoint == "mock"
}

fn load_best(info: &BrainInfo, width: usize, height: usize) -> u32 {
    if is_mock(info) {
        0
    } else {
        scores::get_best(width, height)
    }
}

/// How many executed moves the move history keeps (for the web dashboard's charts and log).
pub(crate) const MOVE_HISTORY: usize = 240;
/// How many finished rounds are kept.
pub(crate) const ROUND_HISTORY: usize = 100;

/// One executed move: what the model said, what actually ran, and what it cost.
#[derive(Clone, Debug)]
pub(crate) struct MoveRecord {
    pub n: u64,
    pub round: u32,
    pub probs: [f64; 4],
    pub model_top: Dir,
    pub chosen: Dir,
    pub intervened: bool,
    pub inference_ms: f64,
    pub risk: f64,
    pub reachable: f64,
    pub score: u32,
    pub ate: bool,
}

/// One finished round.
#[derive(Clone, Debug)]
pub(crate) struct RoundRecord {
    pub round: u32,
    pub width: usize,
    pub height: usize,
    pub score: u32,
    pub length: usize,
    pub decisions: u32,
    pub interventions: u32,
    pub avg_ms: f64,
    pub duration_s: f64,
    /// "wall", "self", "board full" or "reset".
    pub cause: &'static str,
}

/// Session-wide counters across all rounds.
#[derive(Clone, Debug, Default)]
pub(crate) struct Totals {
    pub decisions: u64,
    pub inference_ms: f64,
    pub min_ms: Option<f64>,
    pub max_ms: f64,
    pub interventions: u64,
    pub food: u64,
    pub model_errors: u64,
    pub deaths_wall: u64,
    pub deaths_self: u64,
    pub dir_counts: [u64; 4],
}

pub(crate) struct App {
    pub(crate) game: SnakeGame,
    pub(crate) worker: InferenceWorker,
    pub(crate) best: u32,
    pub(crate) round_no: u32,
    pub(crate) speed_level: usize,
    pub(crate) paused: bool,
    pub(crate) shield_enabled: bool,
    pub(crate) generation: u64,
    /// A request for the current position is running (or queued) on the worker.
    pub(crate) awaiting_result: bool,
    pub(crate) think_start: Instant,
    /// The model has answered but the pacing delay has not elapsed yet.
    pub(crate) pending: Option<Decision>,
    /// Earliest time the next move may be executed (pacing between moves).
    pub(crate) ready_at: Instant,
    /// Earliest time to (re)dispatch a request; pushed out after a model error.
    pub(crate) dispatch_not_before: Instant,
    pub(crate) last_decision: Option<Decision>,
    pub(crate) last_chosen: Option<Dir>,
    pub(crate) last_intervened: bool,
    pub(crate) last_risk: f64,
    pub(crate) last_reachable: f64,
    pub(crate) shield_count: u32,
    pub(crate) decisions_count: u32,
    pub(crate) decision_times: VecDeque<Instant>,
    pub(crate) trail: VecDeque<Dir>,
    pub(crate) error_message: Option<String>,
    pub(crate) start_time: Instant,
    pub(crate) info: BrainInfo,
    pub(crate) moves: VecDeque<MoveRecord>,
    pub(crate) rounds: VecDeque<RoundRecord>,
    pub(crate) totals: Totals,
    pub(crate) round_start: Instant,
    pub(crate) round_ms_sum: f64,
    /// The current round has already been written to `rounds`.
    pub(crate) round_recorded: bool,
}

impl App {
    pub(crate) fn new(
        worker: InferenceWorker,
        info: BrainInfo,
        width: usize,
        height: usize,
    ) -> Self {
        let now = Instant::now();
        Self {
            game: SnakeGame::new(width, height),
            worker,
            best: load_best(&info, width, height),
            round_no: 1,
            speed_level: 2,
            paused: false,
            shield_enabled: true,
            generation: 0,
            awaiting_result: false,
            think_start: now,
            pending: None,
            ready_at: now,
            dispatch_not_before: now,
            last_decision: None,
            last_chosen: None,
            last_intervened: false,
            last_risk: 0.0,
            last_reachable: 0.0,
            shield_count: 0,
            decisions_count: 0,
            decision_times: VecDeque::with_capacity(12),
            trail: VecDeque::with_capacity(16),
            error_message: None,
            start_time: now,
            info,
            moves: VecDeque::with_capacity(MOVE_HISTORY),
            rounds: VecDeque::with_capacity(ROUND_HISTORY),
            totals: Totals::default(),
            round_start: now,
            round_ms_sum: 0.0,
            round_recorded: false,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.finish_round("reset");
        self.game.reset();
        self.reset_transient();
        self.round_no += 1;
    }

    fn reset_transient(&mut self) {
        let now = Instant::now();
        // Answers to requests made before this point are stale: they carry the old generation.
        self.generation += 1;
        self.awaiting_result = false;
        self.pending = None;
        self.ready_at = now;
        self.dispatch_not_before = now;
        self.last_decision = None;
        self.last_chosen = None;
        self.last_intervened = false;
        self.last_risk = 0.0;
        self.last_reachable = 0.0;
        self.shield_count = 0;
        self.decisions_count = 0;
        self.decision_times.clear();
        self.trail.clear();
        self.error_message = None;
        self.round_start = now;
        self.round_ms_sum = 0.0;
        self.round_recorded = false;
    }

    /// Write the current round to the history, once. A round with no moves is not worth a row.
    pub(crate) fn finish_round(&mut self, cause: &'static str) {
        if self.round_recorded || self.decisions_count == 0 {
            return;
        }
        self.round_recorded = true;
        if self.rounds.len() == ROUND_HISTORY {
            self.rounds.pop_front();
        }
        self.rounds.push_back(RoundRecord {
            round: self.round_no,
            width: self.game.width,
            height: self.game.height,
            score: self.game.score,
            length: self.game.snake.len(),
            decisions: self.decisions_count,
            interventions: self.shield_count,
            avg_ms: self.round_ms_sum / self.decisions_count as f64,
            duration_s: self.round_start.elapsed().as_secs_f64(),
            cause,
        });
    }

    /// Start a new round on a board of a different size.
    pub(crate) fn resize(&mut self, width: usize, height: usize) {
        if (width, height) == (self.game.width, self.game.height) {
            return;
        }
        self.finish_round("reset");
        self.game = SnakeGame::new(width, height);
        self.reset_transient();
        self.round_no += 1;
        self.best = load_best(&self.info, width, height);
    }

    fn change_scale(&mut self, delta: i32) {
        let (cols, rows) = terminal::size().unwrap_or((80, 24));
        let nw = (self.game.width as i32 + delta * SCALE_STEP_W as i32).max(MIN_W as i32) as usize;
        let nh = (self.game.height as i32 + delta * SCALE_STEP_H as i32).max(MIN_H as i32) as usize;
        let (nw, nh) = fit_dims(cols, rows, nw, nh);
        self.resize(nw, nh);
    }

    pub(crate) fn decisions_per_sec(&self) -> f64 {
        if self.decision_times.len() < 2 {
            return 0.0;
        }
        let span = self
            .decision_times
            .back()
            .unwrap()
            .duration_since(*self.decision_times.front().unwrap())
            .as_secs_f64();
        if span <= 0.0 {
            0.0
        } else {
            (self.decision_times.len() - 1) as f64 / span
        }
    }

    /// Returns false when the user asked to quit.
    fn handle_key(&mut self, key: KeyEvent) -> bool {
        match key.code {
            KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => return false,
            KeyCode::Char(' ') => self.paused = !self.paused,
            KeyCode::Char('r') | KeyCode::Char('R') => self.reset(),
            KeyCode::Char('s') | KeyCode::Char('S') => self.shield_enabled = !self.shield_enabled,
            KeyCode::Up => self.speed_level = (self.speed_level + 1).min(4),
            KeyCode::Down => self.speed_level = self.speed_level.saturating_sub(1),
            KeyCode::Char('+') | KeyCode::Char('=') => self.change_scale(1),
            KeyCode::Char('-') | KeyCode::Char('_') => self.change_scale(-1),
            _ => {}
        }
        true
    }

    /// Ask the model about the current position if nothing is in flight. The request goes out as
    /// soon as the position exists, so model time overlaps the pacing delay instead of adding to it.
    fn dispatch_if_idle(&mut self, now: Instant) -> bool {
        if self.awaiting_result
            || self.pending.is_some()
            || self.game.game_over
            || now < self.dispatch_not_before
        {
            return false;
        }
        self.worker.request(&self.game, self.generation);
        self.awaiting_result = true;
        self.think_start = now;
        true
    }

    /// Advance the game state machine. Returns true when something visible changed.
    pub(crate) fn tick(&mut self) -> bool {
        if self.paused || self.game.game_over {
            return false;
        }
        let mut changed = self.dispatch_if_idle(Instant::now());

        if self.pending.is_none() {
            if let Some(result) = self.worker.try_result() {
                if result.generation == self.generation {
                    self.awaiting_result = false;
                    changed = true;
                    match result.decision {
                        Ok(decision) => self.pending = Some(decision),
                        Err(err) => {
                            self.totals.model_errors += 1;
                            self.error_message = Some(err);
                            self.dispatch_not_before = Instant::now() + Duration::from_secs(1);
                        }
                    }
                }
            }
        }

        let now = Instant::now();
        if self.pending.is_some() && now >= self.ready_at {
            let decision = self.pending.take().unwrap();
            let (chosen, intervened) = apply_shield(&decision, &self.game, self.shield_enabled);
            let ms = decision.inference_ms;
            let model_top = decision.top;
            let probs = decision.probs;
            self.last_decision = Some(decision);
            self.last_chosen = Some(chosen);
            self.last_intervened = intervened;
            if intervened {
                self.shield_count += 1;
            }

            let (risk, reachable) = estimate_risk_and_reachability(&self.game, chosen);
            self.last_risk = risk;
            self.last_reachable = reachable;

            let head = self.game.next_head(chosen);
            let cause = if !self.game.is_fatal(chosen) {
                None
            } else if head.x < 0
                || head.y < 0
                || head.x >= self.game.width as i32
                || head.y >= self.game.height as i32
            {
                Some("wall")
            } else {
                Some("self")
            };
            let score_before = self.game.score;

            self.game.step(chosen);

            let t = &mut self.totals;
            t.decisions += 1;
            t.inference_ms += ms;
            t.min_ms = Some(t.min_ms.map_or(ms, |m| m.min(ms)));
            t.max_ms = t.max_ms.max(ms);
            t.dir_counts[chosen.index()] += 1;
            if intervened {
                t.interventions += 1;
            }
            let ate = self.game.score > score_before;
            if ate {
                t.food += 1;
            }
            match cause {
                Some("wall") => t.deaths_wall += 1,
                Some(_) => t.deaths_self += 1,
                None => {}
            }
            self.round_ms_sum += ms;
            if self.moves.len() == MOVE_HISTORY {
                self.moves.pop_front();
            }
            self.moves.push_back(MoveRecord {
                n: self.totals.decisions,
                round: self.round_no,
                probs,
                model_top,
                chosen,
                intervened,
                inference_ms: ms,
                risk,
                reachable,
                score: self.game.score,
                ate,
            });

            if self.trail.len() == 16 {
                self.trail.pop_front();
            }
            self.trail.push_back(chosen);
            self.decisions_count += 1;
            if self.decision_times.len() == 12 {
                self.decision_times.pop_front();
            }
            self.decision_times.push_back(now);

            if self.game.score > self.best {
                self.best = if is_mock(&self.info) {
                    self.game.score
                } else {
                    scores::set_best(self.game.width, self.game.height, self.game.score)
                };
            }
            if let Some(cause) = cause {
                self.finish_round(cause);
            } else if self.game.food.is_none() {
                self.game.game_over = true;
                self.finish_round("board full");
            }
            self.ready_at = now + Duration::from_millis(SPEED_LEVELS_MS[self.speed_level]);
            self.error_message = None;
            self.dispatch_if_idle(now);
            changed = true;
        }
        changed
    }

    /// True while the screen needs periodic refreshes even without any state change.
    pub(crate) fn animating(&self) -> bool {
        self.awaiting_result && !self.paused
    }

    /// Draw one whole frame into `out` (nothing is written to the terminal here).
    fn render(&self, out: &mut Vec<u8>, cols: u16, rows: u16) -> std::io::Result<()> {
        let board_w = self.game.width * 2;
        let board_h = self.game.height;
        let panel_x = 4 + board_w + 4;
        let needed_w = panel_x + 36;
        let needed_h = 6 + board_h + 6;

        queue!(out, BeginSynchronizedUpdate, Hide, Clear(ClearType::All))?;

        if (cols as usize) < needed_w || (rows as usize) < needed_h {
            draw_text(
                out,
                0,
                0,
                &format!("Terminal too small ({cols}x{rows}). Need >= {needed_w}x{needed_h}. Press '-' or resize. Q quits."),
                Color::White,
            )?;
            queue!(out, EndSynchronizedUpdate)?;
            return Ok(());
        }

        let board_top = 6u16;
        let board_left = 4u16;
        let px = panel_x as u16;

        draw_text(out, 4, 0, "laya-snake  /  live decisions", Color::DarkGrey)?;
        draw_text(out, 2, 2, "LAYA / LOCAL INTELLIGENCE", Color::White)?;
        draw_text(out, 4, 4, "S N A K E", Color::White)?;

        let horizontal = "─".repeat(board_w);
        draw_text(
            out,
            board_left - 1,
            board_top - 1,
            &format!("┌{horizontal}┐"),
            Color::DarkGrey,
        )?;
        draw_text(
            out,
            board_left - 1,
            board_top + board_h as u16,
            &format!("└{horizontal}┘"),
            Color::DarkGrey,
        )?;

        let dots = "· ".repeat(self.game.width);
        for y in 0..board_h {
            draw_text(
                out,
                board_left - 1,
                board_top + y as u16,
                "│",
                Color::DarkGrey,
            )?;
            draw_text(
                out,
                board_left + board_w as u16,
                board_top + y as u16,
                "│",
                Color::DarkGrey,
            )?;
            draw_text(
                out,
                board_left,
                board_top + y as u16,
                dots.trim_end(),
                Color::DarkGrey,
            )?;
        }

        let status = if self.paused {
            "PAUSED"
        } else if self.game.game_over {
            "GAME OVER"
        } else {
            "LIVE"
        };
        draw_text(
            out,
            cols.saturating_sub(28),
            0,
            &format!(
                "{status} · L{}/5 · {}x{}",
                self.speed_level + 1,
                self.game.width,
                self.game.height
            ),
            if self.game.game_over {
                Color::Red
            } else {
                Color::Green
            },
        )?;
        draw_text(
            out,
            30,
            4,
            &format!("ROUND {:02}", self.round_no),
            Color::DarkGrey,
        )?;
        draw_text(
            out,
            px,
            2,
            &format!("Laya  {}", self.info.checkpoint),
            Color::Green,
        )?;
        draw_text(
            out,
            px,
            3,
            &format!("{} · {} · Local", self.info.engine, self.info.device),
            Color::DarkGrey,
        )?;

        let below = board_top + board_h as u16;
        draw_text(out, board_left, below + 1, "SCORE", Color::DarkGrey)?;
        draw_text(out, board_left + 12, below + 1, "LENGTH", Color::DarkGrey)?;
        draw_text(out, board_left + 26, below + 1, "BEST", Color::DarkGrey)?;
        draw_text(
            out,
            board_left,
            below + 2,
            &format!("{:03}", self.game.score),
            Color::Green,
        )?;
        draw_text(
            out,
            board_left + 12,
            below + 2,
            &format!("{:03}", self.game.snake.len()),
            Color::White,
        )?;
        draw_text(
            out,
            board_left + 26,
            below + 2,
            &format!("{:03}", self.best),
            Color::DarkGrey,
        )?;

        let fill = self.game.board_fill_ratio();
        draw_text(
            out,
            board_left,
            below + 4,
            &bar(24, fill, '█', '-'),
            Color::Green,
        )?;
        draw_text(
            out,
            board_left + 26,
            below + 4,
            &format!("{:4.1}%", fill * 100.0),
            Color::DarkGrey,
        )?;
        let trail: Vec<&str> = self.trail.iter().map(|d| trail_glyph(*d)).collect();
        draw_text(out, board_left, below + 6, &trail.join(" "), Color::Cyan)?;

        for (i, p) in self.game.snake.iter().enumerate().rev() {
            let glyph = if i == 0 {
                direction_glyph(self.game.direction)
            } else if i % 2 == 0 {
                "▓"
            } else {
                "█"
            };
            draw_text(
                out,
                board_left + (p.x as u16) * 2,
                board_top + p.y as u16,
                glyph,
                if i == 0 { Color::White } else { Color::Green },
            )?;
        }
        if let Some(food) = self.game.food {
            draw_text(
                out,
                board_left + food.x as u16 * 2,
                board_top + food.y as u16,
                "●",
                Color::Yellow,
            )?;
        }

        draw_text(out, px, 5, "NEXT MOVE", Color::White)?;
        draw_text(out, px + 14, 5, "MODEL PROBABILITIES", Color::DarkGrey)?;

        for (i, direction) in Dir::ALL.iter().enumerate() {
            let p = self
                .last_decision
                .as_ref()
                .map(|d| d.probs[direction.index()])
                .unwrap_or(0.0);
            let marker = if self.last_chosen == Some(*direction) {
                ">"
            } else {
                " "
            };
            draw_text(out, px, 7 + i as u16, marker, Color::White)?;
            draw_text(
                out,
                px + 2,
                7 + i as u16,
                &direction.name().to_uppercase(),
                Color::DarkGrey,
            )?;
            draw_text(
                out,
                px + 10,
                7 + i as u16,
                &bar(14, p, '█', '·'),
                Color::Green,
            )?;
            draw_text(out, px + 26, 7 + i as u16, &format!("{p:.2}"), Color::White)?;
        }

        draw_text(out, px, 12, "EXECUTING", Color::DarkGrey)?;
        let exec = self.last_chosen.map(Dir::name).unwrap_or("-");
        let suffix = if self.awaiting_result {
            format!(
                "thinking... {:.1}s",
                self.think_start.elapsed().as_secs_f64()
            )
        } else if self.last_intervened {
            "shield override".into()
        } else {
            String::new()
        };
        draw_text(
            out,
            px + 12,
            12,
            &format!("{exec:<8}{suffix:<20}"),
            Color::Green,
        )?;

        draw_text(out, px, 14, "DEAD-END RISK", Color::DarkGrey)?;
        draw_text(
            out,
            px,
            15,
            &bar(24, self.last_risk, '█', '·'),
            Color::Yellow,
        )?;
        draw_text(
            out,
            px + 26,
            15,
            &format!("{:.2}", self.last_risk),
            Color::White,
        )?;
        draw_text(out, px, 17, "FOOD REACHABLE", Color::DarkGrey)?;
        draw_text(
            out,
            px,
            18,
            &bar(24, self.last_reachable, '█', '·'),
            Color::Green,
        )?;
        draw_text(
            out,
            px + 26,
            18,
            &format!("{:.2}", self.last_reachable),
            Color::White,
        )?;

        draw_text(out, px, 20, "INFERENCE", Color::DarkGrey)?;
        draw_text(
            out,
            px + 16,
            20,
            &format!(
                "{:7.1} ms",
                self.last_decision
                    .as_ref()
                    .map(|d| d.inference_ms)
                    .unwrap_or(0.0)
            ),
            Color::White,
        )?;
        draw_text(out, px, 21, "DECISIONS", Color::DarkGrey)?;
        draw_text(
            out,
            px + 16,
            21,
            &format!("{:7.2} /s", self.decisions_per_sec()),
            Color::White,
        )?;
        draw_text(out, px, 22, "OUTPUT TOKENS", Color::DarkGrey)?;
        draw_text(out, px + 16, 22, "0", Color::White)?;
        draw_text(out, px, 23, "NETWORK", Color::DarkGrey)?;
        draw_text(out, px + 16, 23, "OFFLINE", Color::Green)?;
        draw_text(out, px, 24, "ENGINE", Color::DarkGrey)?;
        draw_text(
            out,
            px + 16,
            24,
            &format!("{} {}", self.info.engine, self.info.device),
            Color::White,
        )?;

        draw_text(out, px, 26, "Laya + cycle safety", Color::DarkGrey)?;
        draw_text(out, px, 27, "Shield", Color::DarkGrey)?;
        draw_text(
            out,
            px + 10,
            27,
            if self.shield_enabled {
                "ON"
            } else {
                "OFF (raw model)"
            },
            if self.shield_enabled {
                Color::Green
            } else {
                Color::Red
            },
        )?;
        draw_text(out, px, 28, "Interventions", Color::DarkGrey)?;
        draw_text(
            out,
            px + 22,
            28,
            &format!("{:04}", self.shield_count),
            Color::Yellow,
        )?;

        if let Some(err) = &self.error_message {
            let short: String = err.chars().take(32).collect();
            draw_text(out, px, 30, &format!("model error: {short}"), Color::Red)?;
        }
        if self.game.game_over {
            queue!(out, SetAttribute(Attribute::Reverse))?;
            draw_text(
                out,
                board_left + 8,
                board_top + board_h as u16 / 2,
                " GAME OVER - R restart, Q quit ",
                Color::Red,
            )?;
            queue!(out, SetAttribute(Attribute::NoReverse))?;
        }

        let elapsed = self.start_time.elapsed().as_secs();
        draw_text(
            out,
            2,
            rows.saturating_sub(1),
            "SPACE pause  ↑/↓ speed  +/- scale  S shield  R reset  Q quit",
            Color::DarkGrey,
        )?;
        draw_text(
            out,
            cols.saturating_sub(7),
            rows.saturating_sub(1),
            &format!("{:02}:{:02}", elapsed / 60, elapsed % 60),
            Color::DarkGrey,
        )?;

        queue!(out, EndSynchronizedUpdate)?;
        Ok(())
    }
}

pub fn run<F>(decide: F, info: BrainInfo, width: usize, height: usize) -> anyhow::Result<()>
where
    F: Fn(&GameSnapshot) -> Result<Decision, String> + Send + 'static,
{
    terminal::enable_raw_mode()?;
    let mut out = stdout();
    execute!(out, EnterAlternateScreen, Hide)?;

    let result = run_inner(decide, info, width, height);

    let _ = execute!(out, Show, LeaveAlternateScreen, ResetColor);
    let _ = terminal::disable_raw_mode();
    result
}

fn run_inner<F>(decide: F, info: BrainInfo, width: usize, height: usize) -> anyhow::Result<()>
where
    F: Fn(&GameSnapshot) -> Result<Decision, String> + Send + 'static,
{
    let (cols, rows) = terminal::size()?;
    let (width, height) = fit_dims(cols, rows, width, height);
    let worker = InferenceWorker::new(decide);
    let mut app = App::new(worker, info, width, height);

    let mut frame: Vec<u8> = Vec::with_capacity(16 * 1024);
    let mut dirty = true;
    let mut last_draw = Instant::now();
    let mut first = true;

    loop {
        // Sleep in the event queue, not in a blind sleep: keys are handled the moment they arrive,
        // and while a model call is in flight we look for its answer every few milliseconds.
        let timeout = if app.animating() || app.pending.is_some() {
            Duration::from_millis(10)
        } else {
            Duration::from_millis(50)
        };
        if event::poll(timeout)? {
            loop {
                match event::read()? {
                    // Windows also reports key releases; only presses (and auto-repeat) count.
                    Event::Key(key) if key.kind != KeyEventKind::Release => {
                        if !app.handle_key(key) {
                            return Ok(());
                        }
                        dirty = true;
                    }
                    Event::Resize(..) => dirty = true,
                    _ => {}
                }
                if !event::poll(Duration::ZERO)? {
                    break;
                }
            }
        }

        if app.tick() {
            dirty = true;
        }

        let since = last_draw.elapsed();
        if first || dirty || (app.animating() && since >= THINK_REDRAW) || since >= CLOCK_REDRAW {
            let (cols, rows) = terminal::size()?;
            frame.clear();
            app.render(&mut frame, cols, rows)?;
            let mut lock = stdout().lock();
            lock.write_all(&frame)?;
            lock.flush()?;
            dirty = false;
            first = false;
            last_draw = Instant::now();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Point;

    fn decision(probs: [f64; 4], top: Dir) -> Decision {
        Decision {
            probs,
            top,
            confidence: None,
            inference_ms: 0.0,
        }
    }

    /// A model stand-in: always answers "the first safe direction" after `delay`.
    fn fake_worker(delay: Duration) -> InferenceWorker {
        InferenceWorker::new(move |snap| {
            thread::sleep(delay);
            let mut probs = [0.0; 4];
            let pick = Dir::ALL
                .into_iter()
                .find(|d| !snap.is_fatal(*d))
                .unwrap_or(Dir::Up);
            probs[pick.index()] = 1.0;
            Ok(decision(probs, pick))
        })
    }

    fn test_app(delay: Duration) -> App {
        let info = BrainInfo {
            checkpoint: "test".into(),
            engine: "fake".into(),
            device: "CPU".into(),
        };
        App::new(fake_worker(delay), info, 24, 14)
    }

    fn drive(app: &mut App, for_: Duration) {
        let end = Instant::now() + for_;
        while Instant::now() < end {
            app.tick();
            thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn fit_dims_survives_tiny_terminals() {
        // The original `clamp(MIN, avail)` panicked whenever `avail < MIN` (terminals under ~54 columns).
        for cols in [0u16, 10, 40, 53, 54, 60, 200] {
            for rows in [0u16, 5, 19, 20, 24, 80] {
                let (w, h) = fit_dims(cols, rows, 24, 14);
                assert!(w >= MIN_W && h >= MIN_H, "{cols}x{rows} -> {w}x{h}");
            }
        }
        assert_eq!(fit_dims(200, 80, 24, 14), (24, 14));
    }

    #[test]
    fn shield_takes_the_best_safe_move_and_reports_the_override() {
        let mut g = SnakeGame::new(24, 14);
        g.snake = VecDeque::from([
            Point { x: 0, y: 5 },
            Point { x: 1, y: 5 },
            Point { x: 2, y: 5 },
        ]);
        g.food = Some(Point { x: 10, y: 5 });
        // Left is a wall. The model likes it best, then Up, then Down.
        let d = decision([0.2, 0.1, 0.6, 0.1], Dir::Left);
        assert_eq!(apply_shield(&d, &g, true), (Dir::Up, true));
        assert_eq!(apply_shield(&d, &g, false), (Dir::Left, false));
        // Right runs into the snake's own body, so it is overridden too.
        let d = decision([0.1, 0.1, 0.1, 0.7], Dir::Right);
        assert_eq!(apply_shield(&d, &g, true), (Dir::Up, true));
        // A safe favourite is left alone.
        let d = decision([0.1, 0.7, 0.1, 0.1], Dir::Down);
        assert_eq!(apply_shield(&d, &g, true), (Dir::Down, false));
    }

    #[test]
    fn shield_ties_go_to_the_models_own_choice() {
        let mut g = SnakeGame::new(24, 14);
        g.food = Some(Point { x: 20, y: 2 });
        let d = decision([0.25, 0.25, 0.25, 0.25], Dir::Down);
        assert_eq!(apply_shield(&d, &g, true), (Dir::Down, false));
    }

    #[test]
    fn a_newer_request_replaces_one_that_has_not_started() {
        let seen = Arc::new(Mutex::new(Vec::<u64>::new()));
        let seen_in = Arc::clone(&seen);
        let worker = InferenceWorker::new(move |snap| {
            thread::sleep(Duration::from_millis(80));
            seen_in.lock().unwrap().push(snap.snake.len() as u64);
            Ok(decision([1.0, 0.0, 0.0, 0.0], Dir::Up))
        });
        let mut g = SnakeGame::new(24, 14);
        worker.request(&g, 1); // starts running
        thread::sleep(Duration::from_millis(20));
        g.snake.push_back(Point { x: 0, y: 0 });
        worker.request(&g, 2); // queued...
        g.snake.push_back(Point { x: 0, y: 1 });
        worker.request(&g, 3); // ...and replaced before it ever runs

        let mut generations = Vec::new();
        let end = Instant::now() + Duration::from_secs(2);
        while Instant::now() < end && generations.len() < 2 {
            if let Some(r) = worker.try_result() {
                generations.push(r.generation);
            }
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            generations,
            vec![1, 3],
            "generation 2 should have been superseded"
        );
    }

    #[test]
    fn mashing_reset_never_freezes_the_game() {
        // Regression: with the old bounded queue a third request in quick succession was dropped
        // while the app kept waiting for its answer forever.
        let mut app = test_app(Duration::from_millis(60));
        app.speed_level = 4;
        app.tick();
        for _ in 0..5 {
            app.reset();
            app.tick();
            thread::sleep(Duration::from_millis(5));
        }
        drive(&mut app, Duration::from_millis(1500));
        assert!(
            app.decisions_count >= 2,
            "game stalled: {} decisions",
            app.decisions_count
        );
    }

    #[test]
    fn model_time_overlaps_the_pacing_delay() {
        // Speed level 4 waits 200 ms between moves. With a 150 ms model, moves must come every
        // ~200 ms (pipelined), not every ~350 ms (delay + model, one after the other).
        let mut app = test_app(Duration::from_millis(150));
        app.speed_level = 3;
        drive(&mut app, Duration::from_millis(400)); // let the first moves land
        let before = app.decisions_count;
        let t0 = Instant::now();
        drive(&mut app, Duration::from_millis(2000));
        let moves = (app.decisions_count - before) as f64;
        let interval = t0.elapsed().as_secs_f64() * 1000.0 / moves;
        assert!(
            interval >= 180.0,
            "pacing delay not respected: {interval:.0} ms per move"
        );
        assert!(
            interval <= 290.0,
            "model latency is not overlapping the delay: {interval:.0} ms per move"
        );
    }

    #[test]
    fn stale_answers_after_a_reset_are_ignored() {
        let mut app = test_app(Duration::from_millis(80));
        app.speed_level = 4;
        app.tick(); // request for generation 0 goes out
        app.reset(); // generation 1
        drive(&mut app, Duration::from_millis(60));
        assert_eq!(
            app.decisions_count, 0,
            "an answer for the old position must not move the new one"
        );
        drive(&mut app, Duration::from_millis(600));
        assert!(app.decisions_count >= 1);
    }

    #[test]
    fn frame_is_a_single_synchronized_update() {
        let app = test_app(Duration::from_millis(1));
        let mut buf = Vec::new();
        app.render(&mut buf, 120, 40).unwrap();
        let text = String::from_utf8_lossy(&buf);
        assert!(
            text.starts_with("\u{1b}[?2026h"),
            "frame should open a synchronized update"
        );
        assert!(text.ends_with("\u{1b}[?2026l"), "frame should close it");
        assert!(text.contains("S N A K E"));
        let mut small = Vec::new();
        app.render(&mut small, 30, 10).unwrap();
        assert!(String::from_utf8_lossy(&small).contains("Terminal too small"));
    }
}
