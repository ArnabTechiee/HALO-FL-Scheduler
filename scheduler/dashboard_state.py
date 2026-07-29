"""
HALO Dashboard State
----------------------
Shared read/write helper for the dashboard's state file. Used by both
the AdaptiveFedAvg strategy (writer) and the Flask dashboard app (reader).
"""

import json
import time
from pathlib import Path

STATE_PATH = Path.home() / ".halo" / "dashboard_state.json"


def _ensure_parent():
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {
            "last_updated": None,
            "current_round": 0,
            "total_rounds": None,
            "global_accuracy": None,
            "global_loss": None,
            "nodes": {},
            "history": [],
        }
    try:
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # If file is corrupted or missing, return default state
        return {
            "last_updated": None,
            "current_round": 0,
            "total_rounds": None,
            "global_accuracy": None,
            "global_loss": None,
            "nodes": {},
            "history": [],
        }


def save_state(state: dict):
    _ensure_parent()
    state["last_updated"] = time.time()
    # Write directly to avoid Windows file‑locking issues during rename.
    # The Flask dashboard reads the file occasionally; a direct write is safe.
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def update_node(node_id, **fields):
    """Merge new fields into one node's entry without clobbering the rest."""
    state = load_state()
    node_key = str(node_id)
    if node_key not in state["nodes"]:
        state["nodes"][node_key] = {}
    state["nodes"][node_key].update(fields)
    save_state(state)


def update_round_summary(round_num, total_rounds, accuracy, loss):
    state = load_state()
    state["current_round"] = round_num
    state["total_rounds"] = total_rounds
    state["global_accuracy"] = accuracy
    state["global_loss"] = loss
    state["history"].append({
        "round": round_num, "accuracy": accuracy, "loss": loss,
    })
    save_state(state)


def reset_state():
    """Call this at the very start of a run so old runs' data doesn't
    linger and confuse the dashboard."""
    save_state({
        "last_updated": None,
        "current_round": 0,
        "total_rounds": None,
        "global_accuracy": None,
        "global_loss": None,
        "nodes": {},
        "history": [],
    })