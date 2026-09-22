import json
import os

SCORES_PATH = os.path.join(os.path.expanduser("~"), ".laya_snake", "best_scores.json")


def _load_all():
    try:
        with open(SCORES_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def get_best(width, height):
    return _load_all().get(f"{width}x{height}", 0)


def set_best(width, height, score):
    """Persists score as the new best for this board size if it's higher.
    Returns the (possibly unchanged) best on record."""
    data = _load_all()
    key = f"{width}x{height}"
    current = data.get(key, 0)
    if score <= current:
        return current
    data[key] = score
    try:
        os.makedirs(os.path.dirname(SCORES_PATH), exist_ok=True)
        with open(SCORES_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass
    return score
