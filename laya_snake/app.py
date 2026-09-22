import argparse
import curses
import queue
import threading
import time
from collections import deque

from .ai import CHECKPOINTS, LayaBrain, apply_shield
from .game import DIRS, GameSnapshot, SnakeGame, estimate_risk_and_reachability
from . import scores

SPEED_LEVELS_MS = [1500, 900, 500, 200, 0]  # extra delay after each executed move
DEFAULT_SPEED_LEVEL = 2
DIR_ORDER = ["up", "down", "left", "right"]
HEAD_GLYPH = {"up": "▲", "down": "▼", "left": "◀", "right": "▶"}
TRAIL_GLYPH = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
MIN_W, MIN_H = 10, 8
SCALE_STEP_W, SCALE_STEP_H = 4, 2


class InferenceWorker:
    """One persistent background thread, reused for the whole session, instead
    of spawning a new thread per tick. Only ever one request in flight."""

    def __init__(self, brain):
        self.brain = brain
        self.request_q = queue.Queue(maxsize=1)
        self.result_q = queue.Queue()
        self._stop = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def request(self, game, generation):
        snapshot = GameSnapshot(game)
        try:
            while True:
                self.request_q.get_nowait()
        except queue.Empty:
            pass
        self.request_q.put((snapshot, generation))

    def _loop(self):
        while not self._stop:
            try:
                snapshot, generation = self.request_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                decision = self.brain.decide(snapshot)
                self.result_q.put((generation, decision, None))
            except Exception as exc:  # keep the app alive on any model hiccup
                self.result_q.put((generation, None, exc))

    def stop(self):
        self._stop = True


def bar(width, frac, filled_ch="█", empty_ch="·"):
    frac = max(0.0, min(1.0, frac))
    n = int(round(width * frac))
    return filled_ch * n + empty_ch * (width - n)


def safe_addstr(win, y, x, text, attr=0):
    try:
        win.addstr(y, x, text, attr)
    except curses.error:
        pass


class App:
    def __init__(self, stdscr, brain, width, height):
        self.stdscr = stdscr
        self.brain = brain
        self.worker = InferenceWorker(brain)
        self.game = SnakeGame(width, height)
        self.best = scores.get_best(width, height)
        self.round_no = 1
        self.speed_level = DEFAULT_SPEED_LEVEL
        self.paused = False
        self.shield_enabled = True
        self.generation = 0
        self.awaiting_result = False
        self.think_start = 0.0
        self.next_dispatch_time = 0.0
        self.last_decision = None
        self.last_chosen = None
        self.last_intervened = False
        self.last_risk = 0.0
        self.last_reachable = 0.0
        self.shield_count = 0
        self.decisions_count = 0
        self.decision_times = deque(maxlen=12)
        self.trail = deque(maxlen=16)
        self.error_message = None
        self.start_time = time.time()
        self._chrome_key = None
        self._prev_body = None
        self._prev_food = None
        self._init_colors()

    def _init_colors(self):
        curses.curs_set(0)
        curses.start_color()
        try:
            curses.use_default_colors()
            bg = -1
        except curses.error:
            bg = curses.COLOR_BLACK
        curses.init_pair(1, curses.COLOR_GREEN, bg)
        curses.init_pair(2, curses.COLOR_GREEN, bg)
        curses.init_pair(3, curses.COLOR_YELLOW, bg)
        curses.init_pair(4, curses.COLOR_RED, bg)
        curses.init_pair(5, curses.COLOR_WHITE, bg)
        curses.init_pair(6, curses.COLOR_CYAN, bg)
        curses.init_pair(7, curses.COLOR_BLACK, bg)
        self.stdscr.bkgd(" ", curses.color_pair(0))

    def _reset_transient(self):
        self.generation += 1
        self.awaiting_result = False
        self.next_dispatch_time = 0.0
        self.last_decision = None
        self.last_chosen = None
        self.last_intervened = False
        self.shield_count = 0
        self.decisions_count = 0
        self.decision_times.clear()
        self.trail.clear()
        self.error_message = None

    def reset(self):
        self.game.reset()
        self._reset_transient()
        self.round_no += 1

    def change_scale(self, delta):
        max_y, max_x = self.stdscr.getmaxyx()
        new_w = self.game.width + delta * SCALE_STEP_W
        new_h = self.game.height + delta * SCALE_STEP_H
        new_w, new_h = fit_dims(max_x, max_y, new_w, new_h)
        if (new_w, new_h) == (self.game.width, self.game.height):
            return
        self.game = SnakeGame(new_w, new_h)
        self._reset_transient()
        self.round_no += 1
        self.best = scores.get_best(new_w, new_h)

    def decisions_per_sec(self):
        if len(self.decision_times) < 2:
            return 0.0
        span = self.decision_times[-1] - self.decision_times[0]
        if span <= 0:
            return 0.0
        return (len(self.decision_times) - 1) / span

    def handle_key(self, ch):
        if ch in (ord("q"), ord("Q")):
            return False
        if ch == ord(" "):
            self.paused = not self.paused
        elif ch in (ord("r"), ord("R")):
            self.reset()
        elif ch in (ord("s"), ord("S")):
            self.shield_enabled = not self.shield_enabled
        elif ch == curses.KEY_UP:
            self.speed_level = min(len(SPEED_LEVELS_MS) - 1, self.speed_level + 1)
        elif ch == curses.KEY_DOWN:
            self.speed_level = max(0, self.speed_level - 1)
        elif ch in (ord("+"), ord("=")):
            self.change_scale(+1)
        elif ch in (ord("-"), ord("_")):
            self.change_scale(-1)
        return True

    def tick(self):
        if self.paused or self.game.game_over:
            return
        now = time.time()
        if not self.awaiting_result and now >= self.next_dispatch_time:
            self.worker.request(self.game, self.generation)
            self.awaiting_result = True
            self.think_start = now

        try:
            gen, decision, err = self.worker.result_q.get_nowait()
        except queue.Empty:
            return

        if gen != self.generation:
            return  # stale result from before a reset/rescale

        self.awaiting_result = False
        if err is not None:
            self.error_message = f"{type(err).__name__}: {err}"
            self.next_dispatch_time = time.time() + 1.0
            return

        self.last_decision = decision
        chosen, intervened = apply_shield(decision, self.game, enabled=self.shield_enabled)
        self.last_chosen = chosen
        self.last_intervened = intervened
        if intervened:
            self.shield_count += 1
        risk, reachable = estimate_risk_and_reachability(self.game, chosen)
        self.last_risk, self.last_reachable = risk, reachable

        self.game.step(chosen)
        self.trail.append(chosen)
        self.decisions_count += 1
        self.decision_times.append(time.time())
        if self.game.score > self.best:
            self.best = scores.set_best(self.game.width, self.game.height, self.game.score)
        self.next_dispatch_time = time.time() + SPEED_LEVELS_MS[self.speed_level] / 1000.0

    def _colors(self):
        return dict(
            green=curses.color_pair(1) | curses.A_BOLD,
            green_dim=curses.color_pair(2),
            dim=curses.color_pair(2),
            yellow=curses.color_pair(3) | curses.A_BOLD,
            red=curses.color_pair(4) | curses.A_BOLD,
            white=curses.color_pair(5) | curses.A_BOLD,
            cyan=curses.color_pair(6),
            grey=curses.A_DIM,
        )

    def _layout(self, max_x, max_y):
        g = self.game
        board_w, board_h = g.width * 2, g.height
        panel_x = 4 + board_w + 4
        return dict(
            board_w=board_w, board_h=board_h, panel_x=panel_x,
            needed_w=panel_x + 36, needed_h=6 + board_h + 6,
            board_top=6, board_left=4,
        )

    def _draw_chrome(self, stdscr, L, c):
        """Everything that only changes when the board size or terminal size
        changes — drawn once per (re)size instead of every frame. This is the
        bulk of a frame's bytes (border + every grid dot), so redrawing it at
        ~15fps regardless of whether anything changed was pure waste."""
        stdscr.clear()
        board_top, board_left = L["board_top"], L["board_left"]
        board_w, board_h = L["board_w"], L["board_h"]
        px = L["panel_x"]

        safe_addstr(stdscr, 0, 4, "laya-snake  /  live decisions", c["grey"])
        safe_addstr(stdscr, 2, 2, "LAYA / LOCAL INTELLIGENCE", c["white"])
        safe_addstr(stdscr, 4, 4, "S N A K E", c["white"])

        bx0, by0 = board_left - 1, board_top - 1
        bx1, by1 = board_left + board_w, board_top + board_h
        safe_addstr(stdscr, by0, bx0, "┌" + "─" * board_w + "┐", c["grey"])
        safe_addstr(stdscr, by1, bx0, "└" + "─" * board_w + "┘", c["grey"])
        for y in range(board_top, by1):
            safe_addstr(stdscr, y, bx0, "│", c["grey"])
            safe_addstr(stdscr, y, bx1, "│", c["grey"])
        for y in range(board_h):
            for x in range(board_w // 2):
                safe_addstr(stdscr, board_top + y, board_left + x * 2, "·", c["grey"])

        stats_y = board_top + board_h + 1
        safe_addstr(stdscr, stats_y, board_left, "SCORE", c["grey"])
        safe_addstr(stdscr, stats_y, board_left + 12, "LENGTH", c["grey"])
        safe_addstr(stdscr, stats_y, board_left + 26, "BEST", c["grey"])
        safe_addstr(stdscr, stats_y + 5, board_left, "RECENT MOVES", c["grey"])

        safe_addstr(stdscr, 2, px, "Laya", c["green"])
        safe_addstr(stdscr, 2, px + 5, f"{self.brain.checkpoint}", c["grey"])
        safe_addstr(stdscr, 5, px, "NEXT MOVE", c["white"])
        safe_addstr(stdscr, 5, px + 14, "MODEL PROBABILITIES", c["grey"])
        for i, d in enumerate(DIR_ORDER):
            safe_addstr(stdscr, 7 + i, px + 2, d.upper(), c["grey"])

        exec_row = 12
        safe_addstr(stdscr, exec_row, px, "EXECUTING", c["grey"])
        safe_addstr(stdscr, exec_row + 2, px, "DEAD-END RISK", c["grey"])
        safe_addstr(stdscr, exec_row + 5, px, "FOOD REACHABLE", c["grey"])

        info_row = exec_row + 8
        for i, label in enumerate(["INFERENCE", "DECISIONS", "OUTPUT TOKENS", "NETWORK", "ENGINE"]):
            safe_addstr(stdscr, info_row + i, px, label, c["grey"])

        shield_row = info_row + 6
        safe_addstr(stdscr, shield_row, px, "Laya + cycle safety", c["grey"])
        safe_addstr(stdscr, shield_row + 1, px, "Shield", c["grey"])
        safe_addstr(stdscr, shield_row + 2, px, "Interventions", c["grey"])

        footer = "SPACE pause  ↑/↓ speed  +/- scale  S shield  R reset  Q quit"
        safe_addstr(stdscr, self.stdscr.getmaxyx()[0] - 1, 2, footer, c["grey"])

    def _draw_dynamic(self, stdscr, L, max_x, max_y, c):
        g = self.game
        board_top, board_left = L["board_top"], L["board_left"]
        board_w, board_h = L["board_w"], L["board_h"]
        px = L["panel_x"]

        status = "PAUSED" if self.paused else ("GAME OVER" if g.game_over else "LIVE")
        status_col = c["yellow"] if self.paused else (c["red"] if g.game_over else c["green"])
        tag = f"{status} · L{self.speed_level+1}/5 · {g.width}x{g.height}   "
        safe_addstr(stdscr, 0, max(4, max_x - len(tag) - 4), tag, status_col)
        safe_addstr(stdscr, 4, 30, f"ROUND {self.round_no:02d}", c["grey"])
        safe_addstr(stdscr, 3, px, f"PyTorch · {self.brain.device_str} · Local  ", c["grey"])

        # snake + food: only touch cells that actually changed since last frame
        new_body = list(g.snake)
        new_body_set = set(new_body)
        new_food = g.food
        if self._prev_body is not None:
            for (x, y) in self._prev_body:
                if (x, y) not in new_body_set and (x, y) != new_food:
                    safe_addstr(stdscr, board_top + y, board_left + x * 2, "·", c["grey"])
            if self._prev_food and self._prev_food != new_food and self._prev_food not in new_body_set:
                fx, fy = self._prev_food
                safe_addstr(stdscr, board_top + fy, board_left + fx * 2, "·", c["grey"])
        for i in range(len(new_body) - 1, 0, -1):
            x, y = new_body[i]
            glyph = "█" if i % 2 else "▓"
            safe_addstr(stdscr, board_top + y, board_left + x * 2, glyph, c["green"] if i % 2 else c["green_dim"])
        hx, hy = new_body[0]
        safe_addstr(stdscr, board_top + hy, board_left + hx * 2, HEAD_GLYPH.get(g.direction, "█"), c["white"])
        if new_food:
            fx, fy = new_food
            safe_addstr(stdscr, board_top + fy, board_left + fx * 2, "●", c["yellow"])
        self._prev_body, self._prev_food = new_body, new_food

        stats_y = board_top + board_h + 1
        safe_addstr(stdscr, stats_y + 1, board_left, f"{g.score:03d}", c["green"] | curses.A_BOLD)
        safe_addstr(stdscr, stats_y + 1, board_left + 12, f"{len(g.snake):03d}", c["white"])
        safe_addstr(stdscr, stats_y + 1, board_left + 26, f"{self.best:03d}", c["dim"])
        fill = g.board_fill_ratio()
        safe_addstr(stdscr, stats_y + 3, board_left, bar(24, fill, "█", "-"), c["green"])
        safe_addstr(stdscr, stats_y + 3, board_left + 26, f"{fill*100:4.1f}%", c["grey"])
        trail_str = " ".join(TRAIL_GLYPH[d] for d in self.trail).ljust(32)
        safe_addstr(stdscr, stats_y + 6, board_left, trail_str, c["cyan"])

        probs = (self.last_decision or {}).get("probs", {}) if self.last_decision else {}
        for i, d in enumerate(DIR_ORDER):
            row = 7 + i
            p = probs.get(d, 0.0)
            marker = ">" if d == self.last_chosen else " "
            safe_addstr(stdscr, row, px, marker, c["white"] if d == self.last_chosen else c["grey"])
            safe_addstr(stdscr, row, px + 8, bar(16, p, "█", "·"), c["green"] if d == self.last_chosen else c["dim"])
            safe_addstr(stdscr, row, px + 26, f"{p:0.2f}", c["white"])

        exec_row = 12
        exec_label = (self.last_chosen or "-").upper()
        suffix = ""
        if self.last_intervened:
            suffix = "(shield override)"
        if self.awaiting_result:
            suffix = f"thinking… {time.time() - self.think_start:0.1f}s"
        safe_addstr(stdscr, exec_row, px + 12, f"{exec_label:<8}{suffix:<20}", c["green"] | curses.A_BOLD)

        safe_addstr(stdscr, exec_row + 3, px, bar(24, self.last_risk, "█", "·"), c["yellow"] if self.last_risk > 0.5 else c["dim"])
        safe_addstr(stdscr, exec_row + 3, px + 26, f"{self.last_risk:0.2f}", c["white"])
        safe_addstr(stdscr, exec_row + 6, px, bar(24, self.last_reachable, "█", "·"), c["green"] if self.last_reachable > 0.5 else c["red"])
        safe_addstr(stdscr, exec_row + 6, px + 26, f"{self.last_reachable:0.2f}", c["white"])

        info_row = exec_row + 8
        last_ms = (self.last_decision or {}).get("inference_ms", 0.0) if self.last_decision else 0.0
        safe_addstr(stdscr, info_row, px + 16, f"{last_ms:7.1f} ms", c["white"])
        safe_addstr(stdscr, info_row + 1, px + 16, f"{self.decisions_per_sec():7.2f} /s", c["white"])
        safe_addstr(stdscr, info_row + 2, px + 16, "      0", c["white"])
        safe_addstr(stdscr, info_row + 3, px + 16, " OFFLINE", c["green"])
        safe_addstr(stdscr, info_row + 4, px + 16, f"PyTorch {self.brain.device_str}   ", c["white"])

        shield_row = info_row + 6
        shield_state = "ON             " if self.shield_enabled else "OFF (raw model)"
        safe_addstr(stdscr, shield_row + 1, px + 10, shield_state, c["green"] if self.shield_enabled else c["red"])
        safe_addstr(stdscr, shield_row + 2, px + 22, f"{self.shield_count:04d}", c["yellow"])
        if self.error_message:
            safe_addstr(stdscr, shield_row + 4, px, f"model error: {self.error_message[:28]}", c["red"])

        if g.game_over:
            msg = " GAME OVER — R restart, Q quit "
            safe_addstr(stdscr, board_top + board_h // 2, board_left + max(0, (board_w - len(msg)) // 2), msg, c["red"] | curses.A_REVERSE)

        elapsed = int(time.time() - self.start_time)
        safe_addstr(stdscr, max_y - 1, max_x - 7, f"{elapsed//60:02d}:{elapsed%60:02d}", c["grey"])

    def render(self):
        stdscr = self.stdscr
        max_y, max_x = stdscr.getmaxyx()
        L = self._layout(max_x, max_y)
        c = self._colors()

        if max_x < L["needed_w"] or max_y < L["needed_h"]:
            stdscr.erase()
            safe_addstr(stdscr, 0, 0,
                        f"Terminal too small ({max_x}x{max_y}). Need >= {L['needed_w']}x{L['needed_h']}. "
                        f"Press '-' to shrink the board or resize the window. Q to quit.")
            stdscr.refresh()
            self._chrome_key = None
            return

        key = (self.game.width, self.game.height, max_x, max_y)
        if key != self._chrome_key:
            self._draw_chrome(stdscr, L, c)
            self._chrome_key = key
            self._prev_body = None
            self._prev_food = None

        self._draw_dynamic(stdscr, L, max_x, max_y, c)
        stdscr.refresh()


def fit_dims(max_x, max_y, width, height):
    avail_w = (max_x - 4 - 4 - 36) // 2
    avail_h = max_y - 6 - 6
    if avail_w > 0:
        width = max(MIN_W, min(width, avail_w))
    if avail_h > 0:
        height = max(MIN_H, min(height, avail_h))
    return width, height


def run(stdscr, brain, width, height):
    stdscr.nodelay(True)
    stdscr.timeout(50)
    max_y, max_x = stdscr.getmaxyx()
    width, height = fit_dims(max_x, max_y, width, height)
    app = App(stdscr, brain, width, height)
    try:
        while True:
            ch = stdscr.getch()
            while ch != -1:
                if not app.handle_key(ch):
                    return
                ch = stdscr.getch()
            app.tick()
            app.render()
            time.sleep(0.02)
    finally:
        app.worker.stop()


def main():
    parser = argparse.ArgumentParser(description="Watch the Laya model play Snake, live.")
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), default="root",
                         help="which Laya checkpoint to load: root (English, most tested), "
                              "multilingual (smaller/faster), typed (long-context variant)")
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--height", type=int, default=14)
    args = parser.parse_args()

    print(f"Loading Laya [{args.checkpoint}] ({args.model})... first run also downloads "
          f"the checkpoint and can take a while.")
    brain = LayaBrain(args.model, checkpoint=args.checkpoint)
    print(f"Model loaded on {brain.device_str}. Starting live snake — press Q inside to quit.")
    time.sleep(0.4)

    curses.wrapper(lambda stdscr: run(stdscr, brain, args.width, args.height))


if __name__ == "__main__":
    main()
