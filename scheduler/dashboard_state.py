"""
HALO Dashboard State
----------------------
Shared read/write helper for the dashboard's state file. Used by both
the AdaptiveFedAvg strategy (writer) and the Flask dashboard app (reader).

Schema:
{
  "started_at": float | None,
  "last_updated": float | None,
  "current_round": int,               # last COMPLETED round
  "round_in_progress": int | None,    # round currently training, if any
  "round_started_at": float | None,   # when the in-progress round began
  "avg_round_duration": float | None, # rolling average, seconds
  "round_durations": [float, ...],    # last few round durations (capped)
  "total_rounds": int | None,
  "global_accuracy": float | None,
  "global_loss": float | None,
  "history": [ {"round": int, "accuracy": float, "loss": float} ],
  "nodes": { ... same as before ... },
  "events": [ ... same as before ... ]
}
"""

import json
import time
from pathlib import Path

STATE_PATH = Path.home() / ".halo" / "dashboard_state.json"
MAX_EVENTS = 60
MAX_ROUND_DURATIONS = 10


def _ensure_parent():
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _default_state() -> dict:
    return {
        "started_at": None,
        "last_updated": None,
        "current_round": 0,
        "round_in_progress": None,
        "round_started_at": None,
        "round_started_num": None,
        "avg_round_duration": None,
        "round_durations": [],
        "total_rounds": None,
        "global_accuracy": None,
        "global_loss": None,
        "history": [],
        "nodes": {},
        "events": [],
    }


def load_state() -> dict:
    if not STATE_PATH.exists():
        return _default_state()
    try:
        with open(STATE_PATH, "r") as f:
            state = json.load(f)
        # Backfill any keys missing from an older state file, so upgrading
        # mid-project never crashes on a stale file.
        for k, v in _default_state().items():
            state.setdefault(k, v)
        return state
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return _default_state()


def save_state(state: dict):
    _ensure_parent()
    state["last_updated"] = time.time()
    tmp_path = STATE_PATH.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    tmp_path.replace(STATE_PATH)  # atomic swap on Windows and POSIX alike


def update_node(node_id, **fields):
    state = load_state()
    node_key = str(node_id)
    if node_key not in state["nodes"]:
        state["nodes"][node_key] = {"first_seen_round": state.get("current_round", 0)}
    state["nodes"][node_key].update(fields)
    save_state(state)


def add_event(event_type: str, node_id, message: str, round_num: int = None):
    state = load_state()
    state["events"].append({
        "round": round_num,
        "type": event_type,
        "node_id": str(node_id) if node_id is not None else None,
        "message": message,
        "timestamp": time.time(),
    })
    state["events"] = state["events"][-MAX_EVENTS:]
    save_state(state)


def mark_round_start(round_num: int):
    """Call at the very start of configure_train, before any per-node
    work, so the dashboard can show a live 'training in progress' bar
    while the round is running (not just at round boundaries).

    Computes how long the PREVIOUS round took (if any) and folds it into
    a rolling average, which the dashboard uses to estimate progress
    for the round that's now starting.

    NOTE: duration tracking is keyed off round_started_num, NOT
    round_in_progress — the latter gets cleared by update_round_summary()
    as soon as a round's evaluation finishes, which happens before the
    *next* round's mark_round_start() call. Relying on round_in_progress
    here would mean the "previous round" is always seen as already-None,
    and avg_round_duration would never populate.
    """
    state = load_state()
    now = time.time()

    prev_start = state.get("round_started_at")
    prev_round_num = state.get("round_started_num")
    if prev_start is not None and prev_round_num is not None and prev_round_num != round_num:
        duration = now - prev_start
        if duration > 0:
            durations = state.get("round_durations", [])
            durations.append(duration)
            state["round_durations"] = durations[-MAX_ROUND_DURATIONS:]
            state["avg_round_duration"] = sum(state["round_durations"]) / len(state["round_durations"])

    state["round_in_progress"] = round_num
    state["round_started_at"] = now
    state["round_started_num"] = round_num
    save_state(state)


def update_round_summary(round_num, total_rounds, accuracy, loss):
    """Call once a round's evaluation is complete."""
    state = load_state()
    if state.get("started_at") is None:
        state["started_at"] = time.time()
    state["current_round"] = round_num
    state["total_rounds"] = total_rounds
    state["global_accuracy"] = accuracy
    state["global_loss"] = loss
    # This round is no longer "in progress" once it's summarized.
    if state.get("round_in_progress") == round_num:
        state["round_in_progress"] = None
    state["history"].append({
        "round": round_num, "accuracy": accuracy, "loss": loss,
    })
    save_state(state)


def reset_state():
    state = _default_state()
    state["started_at"] = time.time()
    save_state(state)