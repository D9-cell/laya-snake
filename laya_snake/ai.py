import random
import time

from .game import DIRS

NEUTRAL_KEYS = ["k0", "k1", "k2", "k3"]

CHECKPOINTS = {
    "root": None,               # English, ModernBERT-large 421M — default, most tested
    "multilingual": "multilingual",  # mmBERT-base 322M — smaller, faster
    "typed": "typed-decisions",  # ModernBERT-large 421M, 1024-token context
}


class LayaBrain:
    """Wraps the real Laya model. Every decision below is the model's actual
    output — nothing here is scripted or faked.

    Diagnostic note: this checkpoint carries a strong, content-independent
    bias toward whichever option is literally keyed "down" (~0.58 probability
    even with identical option text — verified by holding text fixed and
    permuting key order). Using "up"/"down"/"left"/"right" as the choice keys
    let that artifact dominate the real decision signal, which is why the
    snake looped instead of seeking food. Fix: use neutral keys ("k0".."k3")
    and shuffle which key maps to which real direction on every call, so any
    residual positional/identity bias cancels out across ticks instead of
    consistently favoring one real-world direction.
    """

    def __init__(self, model_id="convaiinnovations/laya", checkpoint="root"):
        import laya

        self.model_id = model_id
        self.checkpoint = checkpoint
        subfolder = CHECKPOINTS.get(checkpoint)
        self.agent = laya.load(model_id, subfolder=subfolder)

    @property
    def device_str(self):
        # Agent already auto-selects cuda / mps / cpu internally; we just surface it.
        dev = getattr(self.agent, "device", None)
        return str(dev).upper() if dev is not None else "?"

    def decide(self, game):
        hx, hy = game.head()
        fx, fy = game.food
        cur_dist = abs(fx - hx) + abs(fy - hy)
        dangers = {d: game.is_fatal(d) for d in DIRS}

        directions = list(DIRS.keys())
        random.shuffle(directions)
        key_to_dir = dict(zip(NEUTRAL_KEYS, directions))

        criteria = {}
        for key, d in key_to_dir.items():
            dx, dy = DIRS[d]
            nx, ny = hx + dx, hy + dy
            toward = (abs(fx - nx) + abs(fy - ny)) < cur_dist
            progress = "moves closer to the food" if toward else "moves farther from the food"
            danger = " UNSAFE: ends the game immediately." if dangers[d] else " Safe."
            criteria[key] = f"{progress}.{danger}"

        state = f"Snake head at ({hx},{hy}). Food at ({fx},{fy}). Snake length={len(game.snake)}."
        questions = {
            "move": {
                "type": "choice",
                "instructions": "Pick the option that moves closer to the food and is not marked UNSAFE.",
                "criteria": criteria,
            }
        }

        t0 = time.time()
        result = self.agent.predict(state, questions)
        inference_ms = (time.time() - t0) * 1000.0
        ans = result["answers"]["move"]
        probs = {key_to_dir[k]: v for k, v in ans["probabilities"].items()}
        top = key_to_dir[ans["choice"]]
        return {
            "probs": probs,
            "top": top,
            "confidence": ans.get("confidence"),
            "inference_ms": inference_ms,
            "dangers": dangers,
        }


def apply_shield(decision, game, enabled=True):
    """Overrides the model's pick only if it is immediately fatal and a safer
    option exists. Returns (chosen_direction, intervened). With enabled=False,
    always executes the model's raw top pick — useful to compare against."""
    ranked = sorted(decision["probs"].items(), key=lambda kv: kv[1], reverse=True)
    top_choice = ranked[0][0]
    if not enabled:
        return top_choice, False
    for direction, _p in ranked:
        if not game.is_fatal(direction):
            return direction, (direction != top_choice)
    return top_choice, False
