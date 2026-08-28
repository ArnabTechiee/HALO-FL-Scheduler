"""
HALO Dashboard State
====================
Shared read/write helper for the dashboard's state file. Written by the
AdaptiveFedAvg strategy and the ServerApp; read by dashboard_app.py.

Key change vs the previous version
----------------------------------
Nodes are keyed by STABLE DEVICE IDENTITY, not by Flower's node_id.
Flower assigns a fresh node_id on every SuperNode connection, so a device
that drops and reconnects used to appear as a brand-new participant and
its old entry lingered as permanently "dropped". Keying on device_id
(hostname + partition salt, from telemetry.py) means a reconnect is an
update to an existing record, not an insert.

Writes are batched: the strategy wraps a round in `state_batch()` so a
round produces one atomic file write instead of ~15.

Schema
------
{
  "started_at": float|None, "last_updated": float|None,
  "current_round": int,                # last COMPLETED round
  "round_in_progress": int|None,       # round currently training
  "round_started_at": float|None,
  "round_started_num": int|None,
  "avg_round_duration": float|None,
  "round_durations": [float, ...],
  "total_rounds": int|None,
  "global_accuracy": float|None, "global_loss": float|None,
  "history": [{"round": int, "accuracy": float, "loss": float}],
  "run": {"dataset": str|None, "model": str|None, "mode": str|None,
          "run_id": str|None, "base_local_epochs": int|None},
  "nodes": {
     "<device_id>": {
        "device_id": int, "node_id": int, "partition_id": int|None,
        "status": "training|active|skipped|dropped|reconnected|retired",
        "score": float|None, "score_parts": {...}, "score_reason": str,
        "local_epochs": int, "covering_partitions": [int],
        "num_examples": int, "cpu_percent": float, "mem_percent_used": float,
        "battery_percent": float, "battery_plugged_in": bool,
        "network_latency_ms": float, "last_round": int, "missing_rounds": int,
        "first_seen_round": int
     }
  },
  "events": [{"round","type","node_id","message","timestamp"}]
}
"""

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

HALO_DIR = Path(os.environ.get("HALO_DIR", Path.home() / ".halo"))
STATE_PATH = Path(os.environ.get("HALO_STATE_PATH", HALO_DIR / "dashboard_state.json"))
COMPARISON_PATH = Path(os.environ.get("HALO_COMPARISON_PATH", HALO_DIR / "comparison_runs.json"))

MAX_EVENTS = 150
MAX_ROUND_DURATIONS = 10

# Batching: while a batch is open, load_state() hands back the in-memory
# working copy and save_state() only marks it dirty. One disk write per batch.
_BATCH = None
_DIRTY = False


# ──────────────────────────────────────────────────────────────────
# core io
# ──────────────────────────────────────────────────────────────────
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
        "run": {"dataset": None, "model": None, "mode": None,
                "run_id": None, "base_local_epochs": None},
        "nodes": {},
        "events": [],
    }


def _write(state: dict):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(STATE_PATH)  # atomic on Windows and POSIX


def load_state() -> dict:
    if _BATCH is not None:
        return _BATCH
    if not STATE_PATH.exists():
        return _default_state()
    try:
        with open(STATE_PATH, "r") as f:
            state = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return _default_state()
    # Backfill keys added by newer versions so a stale file never crashes.
    for k, v in _default_state().items():
        state.setdefault(k, v)
    return state


def save_state(state: dict):
    global _BATCH, _DIRTY
    state["last_updated"] = time.time()
    if _BATCH is not None:
        _BATCH = state
        _DIRTY = True
        return
    _write(state)


@contextmanager
def state_batch():
    """Collapse many updates into a single atomic write.

    Re-entrant: nesting a batch inside a batch is a no-op, so helpers can
    be called both standalone and from inside a round.
    """
    global _BATCH, _DIRTY
    if _BATCH is not None:
        yield
        return
    _BATCH = load_state()
    _DIRTY = False
    try:
        yield
    finally:
        pending, dirty = _BATCH, _DIRTY
        _BATCH, _DIRTY = None, False
        if dirty:
            _write(pending)


# ──────────────────────────────────────────────────────────────────
# node records
# ──────────────────────────────────────────────────────────────────
def update_node(key, **fields):
    """Create or update one device record. `key` is the stable device id
    (or a temporary 'n<node_id>' key before identity is known)."""
    state = load_state()
    k = str(key)
    node = state["nodes"].setdefault(k, {
        "first_seen_round": state.get("current_round", 0),
        "missing_rounds": 0,
        "covering_partitions": [],
    })
    node.update({k2: v for k2, v in fields.items() if v is not None or k2 in node})
    save_state(state)


def merge_node(old_key, new_key):
    """Move a provisional record onto its real device key once identity is
    resolved, keeping whichever value is non-null. No-op if old is absent."""
    if str(old_key) == str(new_key):
        return
    state = load_state()
    old = state["nodes"].pop(str(old_key), None)
    if old is None:
        save_state(state)
        return
    target = state["nodes"].setdefault(str(new_key), {})
    for k, v in old.items():
        if target.get(k) in (None, [], "") or k not in target:
            target[k] = v
    save_state(state)


def remove_node(key):
    state = load_state()
    state["nodes"].pop(str(key), None)
    save_state(state)


def add_event(event_type: str, key, message: str, round_num: int = None):
    state = load_state()
    state["events"].append({
        "round": round_num,
        "type": event_type,
        "node_id": str(key) if key is not None else None,
        "message": message,
        "timestamp": time.time(),
    })
    state["events"] = state["events"][-MAX_EVENTS:]
    save_state(state)


# ──────────────────────────────────────────────────────────────────
# round lifecycle
# ──────────────────────────────────────────────────────────────────
def set_run_meta(**fields):
    state = load_state()
    state["run"].update(fields)
    save_state(state)


def mark_round_start(round_num: int):
    """Call at the top of configure_train, before any per-node work.

    Folds the PREVIOUS round's duration into a rolling average so the
    dashboard can estimate progress for the round now starting. Duration
    is keyed off round_started_num, not round_in_progress — the latter is
    cleared by update_round_summary() when evaluation finishes, which
    happens before the next mark_round_start().
    """
    state = load_state()
    now = time.time()

    prev_start = state.get("round_started_at")
    prev_num = state.get("round_started_num")
    if prev_start is not None and prev_num is not None and prev_num != round_num:
        duration = now - prev_start
        if duration > 0:
            durations = state.get("round_durations", [])
            durations.append(duration)
            state["round_durations"] = durations[-MAX_ROUND_DURATIONS:]
            state["avg_round_duration"] = sum(state["round_durations"]) / len(state["round_durations"])

    if state.get("started_at") is None:
        state["started_at"] = now

    state["round_in_progress"] = round_num
    state["round_started_at"] = now
    state["round_started_num"] = round_num
    save_state(state)


def update_round_summary(round_num, total_rounds, accuracy, loss):
    """Call once a round's global evaluation completes."""
    state = load_state()
    if state.get("started_at") is None:
        state["started_at"] = time.time()

    state["current_round"] = max(round_num, state.get("current_round", 0))
    state["total_rounds"] = total_rounds
    state["global_accuracy"] = accuracy
    state["global_loss"] = loss
    if state.get("round_in_progress") == round_num:
        state["round_in_progress"] = None

    # Replace rather than duplicate if this round is already recorded
    # (round 0 is the pre-training baseline evaluation).
    for entry in state["history"]:
        if entry.get("round") == round_num:
            entry["accuracy"], entry["loss"] = accuracy, loss
            break
    else:
        state["history"].append({"round": round_num, "accuracy": accuracy, "loss": loss})
    state["history"].sort(key=lambda h: h.get("round", 0))
    save_state(state)


def reset_state():
    state = _default_state()
    state["started_at"] = time.time()
    save_state(state)


# ──────────────────────────────────────────────────────────────────
# comparison runs (adaptive vs baseline)
# ──────────────────────────────────────────────────────────────────
def record_comparison_run(label: str, mode: str, history: list):
    """Persist this run's accuracy curve into comparison_runs.json.

    mode == 'adaptive' fills series_a, anything else fills series_b, so
    running once in each mode populates the dashboard's comparison panel
    with no manual file editing.
    """
    data = {}
    if COMPARISON_PATH.exists():
        try:
            with open(COMPARISON_PATH, "r") as f:
                data = json.load(f) or {}
        except (json.JSONDecodeError, OSError):
            data = {}

    series = [
        {"round": h["round"], "accuracy": h["accuracy"], "loss": h.get("loss")}
        for h in history if h.get("accuracy") is not None
    ]
    slot, lab = ("series_a", "label_a") if mode == "adaptive" else ("series_b", "label_b")
    data[slot] = series
    data[lab] = label
    data.setdefault("series_a", [])
    data.setdefault("series_b", [])
    data["updated_at"] = time.time()

    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = COMPARISON_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(COMPARISON_PATH)
    print(f"[HALO] Recorded '{label}' into {COMPARISON_PATH}")