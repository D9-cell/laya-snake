# laya-snake

Watch [Laya](https://huggingface.co/convaiinnovations/laya) — a small, non-autoregressive
decision model — play Snake, live, in your terminal. Every move you see is a real
forward pass through the model: its actual per-direction probabilities, its actual
inference time, nothing scripted.

This is the Rust implementation. It replaces the earlier Python version: one native
binary, no Python, no PyTorch, no virtualenv — and a purpose-built CPU inference engine
that runs the *same* model about 2× faster than the stock `laya` crate (1.7× once the CPU is thermally throttled).

```
LAYA / LOCAL INTELLIGENCE                          LIVE · L3/5 · 24x14

S N A K E                    ROUND 01               Laya  laya-root
┌────────────────────────────────────────────────┐  Rust/AVX2 f16 x12 · CPU · Local
│ · · · · · · · · · · · · · · · · · · · · · · · · │
│ · · · · · · ▓█▶ · · · · · · · · · · · · · · · · │  NEXT MOVE   MODEL PROBABILITIES
│ · · · · · · · · · · · · · ● · · · · · · · · · · │  > UP        ████████········  0.51
└────────────────────────────────────────────────┘    DOWN      ███·············  0.19
```

## Run it

Install [Rust](https://rustup.rs), then:

```bash
./run.sh          # Linux / macOS   (cargo run --release)
run.bat           # Windows
```

The first launch downloads the model checkpoint (~850 MB) into `models/`. You can also
point at a checkpoint you already have:

```bash
cargo run --release -- --model /path/to/laya-checkpoint
```

## Browser dashboard

```bash
cargo run --release -- --web          # then open http://127.0.0.1:8080/
cargo run --release -- --web 9000 --host 0.0.0.0
```

The same game loop and the same model, served as a web page instead of the terminal UI.
It streams the live state over Server-Sent Events and shows:

- the board, score, length, best, board fill and the last 16 moves
- the model's four probabilities for the current move, what it picked, and what was executed
  (shield overrides are called out)
- dead-end risk, food reachability, shield state and override counts
- engine, device, last inference time, decisions per second, model errors
- session totals: decisions, average / p50 / p95 / min / max inference time, food eaten,
  rounds, deaths by wall or self, shield rate, mean score
- charts: inference latency per decision, the model's top-pick probability per decision,
  score by round (hover any point for details)
- a round history table and a per-decision log

Controls are buttons on the page and the same keys as the terminal (`A` toggles
auto-restart, which starts the next round 2.5 s after a game over).

`--mock` runs the UI without the model: decisions come from a hand-written heuristic, the page
says so in a banner, and best scores are not saved. It exists for UI work on machines without
the checkpoint.

## Controls

| Key | Action |
|---|---|
| `SPACE` | pause / resume |
| `↑` / `↓` | speed up / slow down |
| `+` / `-` | grow / shrink the board, live |
| `S` | toggle the safety shield on/off |
| `R` | reset |
| `Q` / `Esc` | quit |

## Options

```text
--model <id-or-dir>        Hugging Face model id or local checkpoint directory
--checkpoint <name>        root (default) | multilingual | typed
--width <n> --height <n>   starting board size (default 24x14)
--threads <n>              CPU inference threads (default: all; or env LAYA_THREADS)
--engine <fast|candle>     fast (default) or the stock candle implementation
--bench [n]                time n decisions (default 12) and exit
--verify [n]               check the fast engine against candle on n positions and exit
--web [port]               serve the browser dashboard instead of the terminal UI (default 8080)
--host <addr>              address for --web to bind (default 127.0.0.1)
--mock                     no model: a labelled heuristic stand-in, for UI development
```

`LAYA_PROFILE=1 laya-snake --bench 24` prints a per-operation time breakdown.

## What's actually on screen

- **NEXT MOVE / MODEL PROBABILITIES** — Laya's real output for this tick: one choice
  question, scored in a single forward pass.
- **Shield** — a safety wrapper that overrides the model's pick only when it is
  immediately fatal (wall or self-collision) and a safe alternative exists. Toggle it off
  to watch the model's raw, unguided choices.
- **Dead-end risk / Food reachable** — flood-fill/BFS heuristics computed in the game
  code, *not* by the model.

The option keys sent to the model are neutral (`k0`…`k3`) and are re-mapped to
`up/down/left/right` at random on every request, so any positional bias in the checkpoint
cancels out instead of consistently favouring one direction.

## Performance

The model is unchanged. What changed is how the CPU runs it. A decision is one forward
pass of a ModernBERT-large encoder over ~113 tokens. The stock implementation spends much of that
on memory traffic and on generic tensor ops rather than on arithmetic; this engine is close to
arithmetic-bound on the same CPU:

| | stock `laya` crate (candle) | this engine |
|---|---|---|
| weights in memory | up-cast to f32: **2.4 GB** | as stored, f16, memory-mapped: **0.85 GB** |
| bytes streamed per decision | ~1.6 GB | ~0.85 GB |
| start-up | ~2–3 s (read + convert) | ~0.1 s |
| matmul | packs the weight matrix on every call | 6×2 AVX2/FMA micro-kernel, f16→f32 in registers, cache-blocked, work-stealing over column blocks |
| attention | 5+ separate tensor ops, generic mask broadcast | one fused parallel kernel (rope, scale, QKᵀ, sliding window, vector `exp` softmax, ·V) |
| layer norm | 8-pass composed fallback (candle has no fused no-bias path) | one fused pass |
| GeGLU | strided `gelu_erf`, then a strided multiply | fused, parallel, same `erff` |
| last decision-head layer | full sequence | only the 4 option-marker rows it is read at (exact) |

Measured on an i5-1235U laptop (2 P-cores + 8 E-cores, 12 threads), decisions of 113 tokens:

| decision latency (p50) | stock `laya` crate | this engine | speedup |
|---|---|---|---|
| best case (cool CPU, best decision of each run) | ~610–630 ms | ~290–300 ms | **2.1×** |
| first quarter of a 60–120 decision run | 670 ms | 312 ms | **2.1×** |
| last quarter of that run (throttled) | 1017 ms | 585 ms | **1.7×** |
| memory for the model | 2.4 GB | 0.85 GB | 2.8× less |
| model load | ~2.0 s | ~0.1 s | 20× |

These were measured with ~1.3 cores of unrelated background load (a VM and a test suite were
running); on a quiet machine both numbers are lower. Method: `laya-snake --bench N` for the new
engine and a harness around the unmodified `laya::Agent` with the identical prompts for the old one.

Things worth knowing when you measure:

- **Laptops throttle.** Sustained inference heats the package to ~100 °C and the clocks fall
  from ~3.6 GHz to ~1.6 GHz within seconds, so a long run is slower than a short one. Use
  `--bench 90` and compare the first and last quarter it prints.
- **Other load matters.** The kernels use every core; a busy machine (a VM, a test suite)
  slows them noticeably. `--threads` / `LAYA_THREADS` lets you leave some cores free.
- **Model time overlaps the pacing delay.** The next decision is requested the moment a move
  lands, so at speed levels 1–4 moves come `max(delay, inference)` apart instead of
  `delay + inference`.
- The UI redraws only when something changes (about 10 Hz while thinking) instead of clearing
  and repainting the whole screen 50 times a second, and it uses no CPU while paused.

### Same answers as the stock implementation

The engine uses the same weights (f16 → f32 is exact) and the same f32 arithmetic; only
summation order differs. `laya-snake --verify` runs both implementations on the same
positions and compares them. On 24 game positions the chosen move matched 24/24 and probabilities
agreed to within one unit in the fourth decimal. The prompt construction is a port of the
`laya` crate's `build_sequence`, and token counts match exactly.

### Platforms

The fast engine needs an x86-64 CPU with AVX2, FMA and F16C (Intel Haswell / AMD Zen or
newer, i.e. anything since about 2013). Everywhere else — other CPUs, or when built with a
GPU feature — the game uses the stock candle implementation, automatically.

```bash
cargo build --release --features cuda     # NVIDIA GPU (uses candle)
cargo build --release --features metal    # Apple GPU  (uses candle)
```

## Model

The first launch downloads the selected checkpoint into:

```text
models/
  laya-root/
  laya-multilingual/
  laya-typed-decisions/
```

Downloads go to a `.part` file and are renamed on success, so an interrupted download is
never mistaken for a finished one. The fast engine has been checked against the default
`root` checkpoint; if a checkpoint does not fit it, it falls back to candle, and
`--engine candle` forces the stock implementation.

## Tests

```bash
cargo test --release
```

Covers the SIMD kernels against scalar references (f16 conversion is checked bit-exact over
all 65 536 values), the game logic against the original `HashSet`-based algorithms, the
pipelined scheduling, and the request queue: as in the Python version, a newer request replaces a
stale queued one. (The first Rust port dropped the *new* request when its queue was full, which by
code analysis could leave the game waiting forever after rapid resets; a regression test guards it.)

## Project layout

```text
Cargo.toml
src/
├── main.rs          CLI
├── ai.rs            prompt construction, backend selection, checkpoint download
├── app.rs           terminal UI, scheduling, background inference worker
├── game.rs          game rules, BFS / flood-fill heuristics
├── scores.rs        best-score persistence
├── bench.rs         --bench and --verify
└── engine/
    ├── mod.rs       ModernBERT encoder + decision head
    ├── kernels.rs   AVX2/FMA/F16C kernels
    ├── weights.rs   memory-mapped safetensors
    └── prompt.rs    port of the laya crate's sequence builder
```

## Credits

The model is [Laya](https://huggingface.co/convaiinnovations/laya) by ConvAI Innovations.
The stock Rust inference crate is [`laya`](https://crates.io/crates/laya) (Apache-2.0); this
project uses it for configuration types and as the reference implementation, and
`src/engine/prompt.rs` is a port of its `build_sequence`.
