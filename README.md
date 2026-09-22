# laya-snake

Watch [Laya](https://huggingface.co/convaiinnovations/laya) — a small, non-autoregressive
decision model — play Snake, live, in your terminal. Every move you see is a real
forward pass through the model: its actual per-direction probabilities, its actual
inference time, nothing scripted.

```
LAYA / LOCAL INTELLIGENCE                          LIVE · L3/5 · 24x14

S N A K E                    ROUND 01               Laya root
┌────────────────────────┐                          PyTorch · CPU · Local
│ · · · · · · · · · · ·   │
│ · · · ▓█▶ · · · · · ·   │                          NEXT MOVE   MODEL PROBABILITIES
│ · · · · · · · ● · · ·   │                          > UP        ████████········  0.51
│ · · · · · · · · · · ·   │                            DOWN      ███·············  0.19
└────────────────────────┘                             LEFT      ██··············  0.14
SCORE  LENGTH  BEST                                     RIGHT     ██··············  0.16
 012    015     028
████████················ 34.2%                       EXECUTING   UP
                                                       DEAD-END RISK    ·········· 0.02
RECENT MOVES                                          FOOD REACHABLE   █████████· 0.93
↑ ↑ → → ↓ ← ↑
                                                       INFERENCE      812.4 ms
                                                       DECISIONS        1.23 /s
                                                       Shield   ON     Interventions 0002
```

## Run it — one command, any OS

```bash
./run.sh          # Linux / macOS
run.bat           # Windows
```

That's it. `play.py` detects your OS, creates an isolated `.venv` next to itself
(never touches system Python), installs the couple of packages it needs (`torch`
+ `laya`, plus `windows-curses` on Windows), lets the model download and cache
itself on first run, then starts the game. Requires Python 3.9+, nothing else.

## Controls

| Key | Action |
|---|---|
| `SPACE` | pause / resume |
| `↑` / `↓` | speed up / slow down |
| `+` / `-` | grow / shrink the board, live |
| `S` | toggle the safety shield on/off |
| `R` | reset |
| `Q` | quit |

## What's actually on screen

- **NEXT MOVE / MODEL PROBABILITIES** — Laya's real output for this tick: one
  choice question (`up`/`down`/`left`/`right`), scored in a single forward pass.
- **Shield** — a safety wrapper that overrides the model's pick only when it's
  immediately fatal (wall or self-collision) and a safe alternative exists.
  Toggle it off to watch the model's raw, unguided choices.
- **Dead-end risk / Food reachable** — flood-fill/BFS heuristics computed in
  the game code, *not* by the model. Labeled honestly as such.
- **Best**, per board size, persisted to `~/.laya_snake/best_scores.json`.

## A real bug this project found

Early on, the snake looped instead of seeking food. Turned out the model's
probabilities were nearly food-position-independent — and with identical
option text, whichever choice was literally keyed `"down"` won ~58% of the
time regardless of order or meaning. Using direction words as option keys let
that artifact dominate the real signal.

Fix, in `laya_snake/ai.py`: neutral option keys (`k0`..`k3`), shuffled to a
random real direction on every call (so residual positional bias cancels out
across ticks instead of consistently favoring one direction), with each
option's text pre-computed as "moves closer/farther from the food" instead of
making the model do coordinate arithmetic. Verified with a controlled test:
before the fix, food placed far in any of the four directions produced
near-identical probabilities; after, the correct direction was consistently
favored.

## Project layout

```
laya_snake/
  game.py    snake mechanics + flood-fill/BFS heuristics
  ai.py      Laya wrapper: prompting, debiasing, the safety shield
  app.py     curses UI + main loop (persistent background inference thread)
  scores.py  best-score persistence, per board size
play.py      cross-platform bootstrap + entry point
run.sh / run.bat   one-command launchers
```

## Flags

```bash
python3 play.py --checkpoint multilingual   # smaller/faster mmBERT-base checkpoint
python3 play.py --width 32 --height 18      # starting board size
```
