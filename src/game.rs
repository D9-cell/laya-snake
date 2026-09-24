use rand::{rng, Rng};
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Dir {
    Up,
    Down,
    Left,
    Right,
}

impl Dir {
    /// Canonical order used by the UI and the prompt: up, down, left, right.
    pub const ALL: [Dir; 4] = [Dir::Up, Dir::Down, Dir::Left, Dir::Right];

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn delta(self) -> (i32, i32) {
        match self {
            Dir::Up => (0, -1),
            Dir::Down => (0, 1),
            Dir::Left => (-1, 0),
            Dir::Right => (1, 0),
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Dir::Up => "up",
            Dir::Down => "down",
            Dir::Left => "left",
            Dir::Right => "right",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point {
    fn step(self, dir: Dir) -> Point {
        let (dx, dy) = dir.delta();
        Point {
            x: self.x + dx,
            y: self.y + dy,
        }
    }
}

#[derive(Clone, Copy)]
struct Board {
    w: i32,
    h: i32,
}

impl Board {
    fn new(width: usize, height: usize) -> Self {
        Self {
            w: width as i32,
            h: height as i32,
        }
    }

    #[inline]
    fn contains(self, p: Point) -> bool {
        p.x >= 0 && p.x < self.w && p.y >= 0 && p.y < self.h
    }

    #[inline]
    fn idx(self, p: Point) -> usize {
        (p.y * self.w + p.x) as usize
    }

    fn cells(self) -> usize {
        (self.w * self.h) as usize
    }
}

/// Would moving the head to `p` end the game? `body` is head-first; the tail cell only counts when
/// the move eats (the tail stays put then).
fn fatal_at(
    board: Board,
    len: usize,
    body: impl Iterator<Item = Point>,
    food: Option<Point>,
    p: Point,
) -> bool {
    if !board.contains(p) {
        return true;
    }
    let ate = food == Some(p);
    let end = if ate { len } else { len.saturating_sub(1) };
    body.take(end).any(|x| x == p)
}

#[derive(Clone, Debug)]
pub struct SnakeGame {
    pub width: usize,
    pub height: usize,
    pub snake: VecDeque<Point>,
    pub direction: Dir,
    pub score: u32,
    pub game_over: bool,
    pub food: Option<Point>,
}

impl SnakeGame {
    pub fn new(width: usize, height: usize) -> Self {
        let mut game = Self {
            width,
            height,
            snake: VecDeque::new(),
            direction: Dir::Right,
            score: 0,
            game_over: false,
            food: None,
        };
        game.reset();
        game
    }

    fn board(&self) -> Board {
        Board::new(self.width, self.height)
    }

    pub fn reset(&mut self) {
        let cx = (self.width / 2) as i32;
        let cy = (self.height / 2) as i32;
        self.snake = VecDeque::from([
            Point { x: cx, y: cy },
            Point { x: cx - 1, y: cy },
            Point { x: cx - 2, y: cy },
        ]);
        self.direction = Dir::Right;
        self.score = 0;
        self.game_over = false;
        self.food = self.spawn_food();
    }

    pub fn head(&self) -> Point {
        self.snake[0]
    }

    /// A uniformly random free cell, or `None` when the snake fills the board.
    fn spawn_food(&self) -> Option<Point> {
        let board = self.board();
        let mut occupied = vec![false; board.cells()];
        let mut used = 0;
        for &p in &self.snake {
            if board.contains(p) && !occupied[board.idx(p)] {
                occupied[board.idx(p)] = true;
                used += 1;
            }
        }
        let free = board.cells() - used;
        if free == 0 {
            return None;
        }
        let mut pick = rng().random_range(0..free);
        for x in 0..board.w {
            for y in 0..board.h {
                let p = Point { x, y };
                if !occupied[board.idx(p)] {
                    if pick == 0 {
                        return Some(p);
                    }
                    pick -= 1;
                }
            }
        }
        None
    }

    pub fn next_head(&self, dir: Dir) -> Point {
        self.head().step(dir)
    }

    pub fn is_fatal(&self, dir: Dir) -> bool {
        fatal_at(
            self.board(),
            self.snake.len(),
            self.snake.iter().copied(),
            self.food,
            self.next_head(dir),
        )
    }

    pub fn step(&mut self, dir: Dir) {
        if self.game_over {
            return;
        }
        if self.is_fatal(dir) {
            self.game_over = true;
            self.direction = dir;
            return;
        }
        let p = self.next_head(dir);
        let ate = self.food == Some(p);
        self.snake.push_front(p);
        if ate {
            self.score += 1;
            self.food = self.spawn_food();
        } else {
            self.snake.pop_back();
        }
        self.direction = dir;
    }

    pub fn board_fill_ratio(&self) -> f64 {
        self.snake.len() as f64 / (self.width * self.height) as f64
    }
}

/// An owned copy of what the inference thread needs to see.
#[derive(Clone, Debug)]
pub struct GameSnapshot {
    pub width: usize,
    pub height: usize,
    pub snake: Vec<Point>,
    pub food: Option<Point>,
}

impl GameSnapshot {
    pub fn from_game(game: &SnakeGame) -> Self {
        Self {
            width: game.width,
            height: game.height,
            snake: game.snake.iter().copied().collect(),
            food: game.food,
        }
    }

    pub fn head(&self) -> Point {
        self.snake[0]
    }

    pub fn is_fatal(&self, dir: Dir) -> bool {
        fatal_at(
            Board::new(self.width, self.height),
            self.snake.len(),
            self.snake.iter().copied(),
            self.food,
            self.head().step(dir),
        )
    }
}

/// Reachable cell count from `start`, walking only through unblocked in-bounds cells. `start` itself
/// is counted even if it lies off the board (a fatal move), matching the original behaviour.
fn flood_fill_area(
    board: Board,
    start: Point,
    blocked: &[bool],
    seen: &mut [bool],
    queue: &mut Vec<Point>,
) -> usize {
    if board.contains(start) && blocked[board.idx(start)] {
        return 0;
    }
    seen.fill(false);
    queue.clear();
    if board.contains(start) {
        seen[board.idx(start)] = true;
    }
    queue.push(start);
    let mut count = 1;
    let mut head = 0;
    while head < queue.len() {
        let p = queue[head];
        head += 1;
        for dir in Dir::ALL {
            let n = p.step(dir);
            if board.contains(n) {
                let i = board.idx(n);
                if !blocked[i] && !seen[i] {
                    seen[i] = true;
                    count += 1;
                    queue.push(n);
                }
            }
        }
    }
    count
}

fn bfs_path_exists(
    board: Board,
    start: Point,
    goal: Point,
    blocked: &[bool],
    seen: &mut [bool],
    queue: &mut Vec<Point>,
) -> bool {
    if start == goal {
        return true;
    }
    seen.fill(false);
    queue.clear();
    if board.contains(start) {
        seen[board.idx(start)] = true;
    }
    queue.push(start);
    let mut head = 0;
    while head < queue.len() {
        let p = queue[head];
        head += 1;
        for dir in Dir::ALL {
            let n = p.step(dir);
            if n == goal {
                return true;
            }
            if board.contains(n) {
                let i = board.idx(n);
                if !blocked[i] && !seen[i] {
                    seen[i] = true;
                    queue.push(n);
                }
            }
        }
    }
    false
}

/// `(dead-end risk, food reachable)` after executing `dir`: risk is `1 - reachable area / snake
/// length`, clamped to `[0, 1]`; reachability is 1.0 when a path to the food exists.
pub fn estimate_risk_and_reachability(game: &SnakeGame, dir: Dir) -> (f64, f64) {
    let board = game.board();
    let new_head = game.next_head(dir);
    let ate = game.food == Some(new_head);

    // Body after the move, minus its head: the old body, without the tail unless we ate.
    let keep = if ate {
        game.snake.len()
    } else {
        game.snake.len().saturating_sub(1)
    };
    let body_len = keep + 1;
    let mut blocked = vec![false; board.cells()];
    for &p in game.snake.iter().take(keep) {
        if board.contains(p) {
            blocked[board.idx(p)] = true;
        }
    }

    let mut seen = vec![false; board.cells()];
    let mut queue = Vec::with_capacity(board.cells());
    let area = flood_fill_area(board, new_head, &blocked, &mut seen, &mut queue);
    let risk = (1.0 - area as f64 / body_len.max(1) as f64).clamp(0.0, 1.0);
    let reachable = game
        .food
        .map(|food| bfs_path_exists(board, new_head, food, &blocked, &mut seen, &mut queue))
        .unwrap_or(false);
    (risk, if reachable { 1.0 } else { 0.0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// The original `HashSet`-based implementation, kept as the oracle.
    mod reference {
        use super::super::{Dir, Point, SnakeGame};
        use std::collections::{HashSet, VecDeque};

        pub fn flood_fill_area(
            start: Point,
            blocked: &HashSet<Point>,
            width: usize,
            height: usize,
        ) -> usize {
            if blocked.contains(&start) {
                return 0;
            }
            let mut seen = HashSet::from([start]);
            let mut q = VecDeque::from([start]);
            while let Some(p) = q.pop_front() {
                for d in Dir::ALL {
                    let (dx, dy) = d.delta();
                    let n = Point {
                        x: p.x + dx,
                        y: p.y + dy,
                    };
                    if n.x >= 0
                        && n.x < width as i32
                        && n.y >= 0
                        && n.y < height as i32
                        && !blocked.contains(&n)
                        && seen.insert(n)
                    {
                        q.push_back(n);
                    }
                }
            }
            seen.len()
        }

        pub fn bfs_path_exists(
            start: Point,
            goal: Point,
            blocked: &HashSet<Point>,
            width: usize,
            height: usize,
        ) -> bool {
            if start == goal {
                return true;
            }
            let mut seen = HashSet::from([start]);
            let mut q = VecDeque::from([start]);
            while let Some(p) = q.pop_front() {
                for d in Dir::ALL {
                    let (dx, dy) = d.delta();
                    let n = Point {
                        x: p.x + dx,
                        y: p.y + dy,
                    };
                    if n == goal {
                        return true;
                    }
                    if n.x >= 0
                        && n.x < width as i32
                        && n.y >= 0
                        && n.y < height as i32
                        && !blocked.contains(&n)
                        && seen.insert(n)
                    {
                        q.push_back(n);
                    }
                }
            }
            false
        }

        pub fn estimate(game: &SnakeGame, dir: Dir) -> (f64, f64) {
            let new_head = game.next_head(dir);
            let ate = game.food == Some(new_head);
            let mut body_after = game.snake.clone();
            body_after.push_front(new_head);
            if !ate {
                body_after.pop_back();
            }
            let blocked: HashSet<Point> = body_after.iter().skip(1).copied().collect();
            let area = flood_fill_area(new_head, &blocked, game.width, game.height);
            let risk = (1.0 - area as f64 / body_after.len().max(1) as f64).clamp(0.0, 1.0);
            let reachable = game
                .food
                .map(|f| bfs_path_exists(new_head, f, &blocked, game.width, game.height))
                .unwrap_or(false);
            (risk, if reachable { 1.0 } else { 0.0 })
        }

        pub fn is_fatal(game: &SnakeGame, dir: Dir) -> bool {
            let p = game.next_head(dir);
            if p.x < 0 || p.x >= game.width as i32 || p.y < 0 || p.y >= game.height as i32 {
                return true;
            }
            let ate = game.food == Some(p);
            let end = if ate {
                game.snake.len()
            } else {
                game.snake.len().saturating_sub(1)
            };
            game.snake.iter().take(end).any(|&x| x == p)
        }
    }

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self, n: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) % n
        }
    }

    /// A random legal position: the snake is a self-avoiding random walk, food on a free cell.
    fn random_game(r: &mut Lcg) -> SnakeGame {
        let (w, h) = (10 + r.next(20) as usize, 8 + r.next(12) as usize);
        let mut g = SnakeGame::new(w, h);
        let target = 3 + r.next((w * h / 2) as u64) as usize;
        let mut cells = vec![Point {
            x: r.next(w as u64) as i32,
            y: r.next(h as u64) as i32,
        }];
        let mut guard = 0;
        while cells.len() < target && guard < 5000 {
            guard += 1;
            let d = Dir::ALL[r.next(4) as usize];
            let n = cells[cells.len() - 1].step(d);
            if n.x >= 0 && n.x < w as i32 && n.y >= 0 && n.y < h as i32 && !cells.contains(&n) {
                cells.push(n);
            }
        }
        g.snake = cells.into_iter().rev().collect();
        let occupied: HashSet<Point> = g.snake.iter().copied().collect();
        let free: Vec<Point> = (0..w as i32)
            .flat_map(|x| (0..h as i32).map(move |y| Point { x, y }))
            .filter(|p| !occupied.contains(p))
            .collect();
        g.food = if free.is_empty() {
            None
        } else {
            Some(free[r.next(free.len() as u64) as usize])
        };
        g
    }

    #[test]
    fn risk_reachability_and_fatality_match_reference() {
        let mut r = Lcg(1234);
        for _ in 0..3000 {
            let g = random_game(&mut r);
            let snap = GameSnapshot::from_game(&g);
            for d in Dir::ALL {
                assert_eq!(
                    estimate_risk_and_reachability(&g, d),
                    reference::estimate(&g, d),
                    "{g:?} {d:?}"
                );
                assert_eq!(g.is_fatal(d), reference::is_fatal(&g, d));
                assert_eq!(snap.is_fatal(d), reference::is_fatal(&g, d));
            }
        }
    }

    #[test]
    fn off_board_start_matches_reference() {
        // Executing a fatal move (shield off) starts the flood fill outside the board.
        let mut g = SnakeGame::new(12, 9);
        g.snake = VecDeque::from([
            Point { x: 0, y: 4 },
            Point { x: 1, y: 4 },
            Point { x: 2, y: 4 },
        ]);
        g.food = Some(Point { x: 8, y: 2 });
        for d in Dir::ALL {
            assert_eq!(
                estimate_risk_and_reachability(&g, d),
                reference::estimate(&g, d)
            );
        }
    }

    #[test]
    fn step_matches_reference_rules() {
        let mut r = Lcg(99);
        for _ in 0..300 {
            let mut g = random_game(&mut r);
            for _ in 0..40 {
                if g.game_over {
                    break;
                }
                let d = Dir::ALL[r.next(4) as usize];
                let before = g.clone();
                let fatal = reference::is_fatal(&before, d);
                g.step(d);
                assert_eq!(g.game_over, fatal);
                assert_eq!(g.direction, d);
                if fatal {
                    assert_eq!(g.snake, before.snake);
                } else {
                    let ate = before.food == Some(before.next_head(d));
                    assert_eq!(g.score, before.score + ate as u32);
                    assert_eq!(g.head(), before.next_head(d));
                    assert_eq!(g.snake.len(), before.snake.len() + ate as usize);
                    if let Some(f) = g.food {
                        assert!(!g.snake.contains(&f), "food spawned on the snake");
                    }
                }
            }
        }
    }

    #[test]
    fn food_is_uniform_over_free_cells_and_none_when_full() {
        let mut g = SnakeGame::new(4, 3);
        g.snake = (0..4)
            .flat_map(|x| (0..3).map(move |y| Point { x, y }))
            .filter(|p| !(p.x == 3 && p.y == 2))
            .collect();
        assert_eq!(g.spawn_food(), Some(Point { x: 3, y: 2 }));
        g.snake.push_back(Point { x: 3, y: 2 });
        assert_eq!(g.spawn_food(), None);

        let g = SnakeGame::new(6, 4);
        let mut counts = std::collections::HashMap::new();
        for _ in 0..12_000 {
            *counts.entry(g.spawn_food().unwrap()).or_insert(0usize) += 1;
        }
        assert_eq!(
            counts.len(),
            6 * 4 - 3,
            "every free cell should be reachable"
        );
        assert!(counts.values().all(|&c| c > 300 && c < 800), "{counts:?}");
    }
}
