"""
HALO Capacity Scoring
=====================
Turns a flat telemetry snapshot (telem_* fields from telemetry.py) into a
capacity score in [0.0, 1.0] plus a per-factor breakdown.

0.0  = skip this device this round (protecting it, or it can't be reached)
1.0  = full workload

Design notes
------------
* Every field is read with .get() and a neutral default. Evaluate-only
  replies carry a partial telemetry dict, and a KeyError here would take
  down the whole strategy mid-round.
* Factors are continuous ramps, not step multipliers. A device at 61% CPU
  and one at 84% CPU used to score identically (both x0.7); now they don't,
  so local_epochs actually tracks capacity instead of bucketing.
* CPU has a floor of 0.35. Local training legitimately pegs the CPU, so a
  high reading must reduce workload without ever hard-skipping a healthy
  device.
* Two hard gates remain, because they are about device safety and
  reachability rather than throughput: a critically low unplugged battery,
  and an unreachable network.
* compute_capacity_report() returns the sub-scores so the dashboard can
  show WHY a device was throttled, not just that it was.
"""

# Weights sum to 1.0. Battery dominates: it is the factor the user actually
# feels, and the one federated learning is most often blamed for.
WEIGHTS = {"battery": 0.35, "cpu": 0.30, "memory": 0.15, "network": 0.20}

BATTERY_CRITICAL = 15     # % — unplugged below this, skip entirely
BATTERY_COMFORT = 45      # % — unplugged at/above this, no penalty
CPU_EASY = 50.0           # % — at/below this, no penalty
CPU_HARD = 95.0           # % — at/above this, floor penalty
CPU_FLOOR = 0.35
MEM_EASY = 80.0
MEM_HARD = 97.0
MEM_FLOOR = 0.30
NET_EASY = 100.0          # ms
NET_HARD = 600.0          # ms
NET_FLOOR = 0.25


def _ramp(value, easy, hard, floor):
    """1.0 at/below `easy`, `floor` at/above `hard`, linear between."""
    if value is None:
        return 1.0
    if value <= easy:
        return 1.0
    if value >= hard:
        return floor
    span = (value - easy) / float(hard - easy)
    return 1.0 - span * (1.0 - floor)


def _as_float(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_capacity_report(telemetry: dict) -> dict:
    """Full breakdown: {'score', 'parts', 'reason', 'blocked'}."""
    if not telemetry:
        return {"score": None, "parts": {}, "reason": "no telemetry yet", "blocked": False}

    cpu = _as_float(telemetry.get("telem_cpu_percent"))
    mem = _as_float(telemetry.get("telem_mem_percent_used"))
    latency = _as_float(telemetry.get("telem_network_latency_ms"))
    has_battery = bool(_as_float(telemetry.get("telem_has_battery"), 0))
    battery = _as_float(telemetry.get("telem_battery_percent"))
    plugged = bool(_as_float(telemetry.get("telem_battery_plugged_in"), 0))

    reasons = []

    # --- battery -------------------------------------------------
    if not has_battery or plugged or battery is None:
        battery_part = 1.0
    elif battery < BATTERY_CRITICAL:
        return {
            "score": 0.0,
            "parts": {"battery": 0.0, "cpu": None, "memory": None, "network": None},
            "reason": f"battery critically low ({battery:.0f}%, unplugged)",
            "blocked": True,
        }
    elif battery >= BATTERY_COMFORT:
        battery_part = 1.0
    else:
        span = (battery - BATTERY_CRITICAL) / float(BATTERY_COMFORT - BATTERY_CRITICAL)
        battery_part = 0.25 + span * 0.75
        reasons.append(f"battery {battery:.0f}% unplugged")

    # --- network -------------------------------------------------
    if latency is not None and latency < 0:
        return {
            "score": 0.0,
            "parts": {"battery": battery_part, "cpu": None, "memory": None, "network": 0.0},
            "reason": "network unreachable",
            "blocked": True,
        }
    network_part = _ramp(latency, NET_EASY, NET_HARD, NET_FLOOR)
    if latency is not None and latency > NET_EASY * 2:
        reasons.append(f"latency {latency:.0f}ms")

    # --- cpu / memory --------------------------------------------
    cpu_part = _ramp(cpu, CPU_EASY, CPU_HARD, CPU_FLOOR)
    if cpu is not None and cpu > 85:
        reasons.append(f"CPU {cpu:.0f}%")
    memory_part = _ramp(mem, MEM_EASY, MEM_HARD, MEM_FLOOR)
    if mem is not None and mem > 90:
        reasons.append(f"memory {mem:.0f}%")

    parts = {"battery": battery_part, "cpu": cpu_part,
             "memory": memory_part, "network": network_part}
    score = sum(parts[k] * WEIGHTS[k] for k in WEIGHTS)

    return {
        "score": round(max(0.0, min(1.0, score)), 3),
        "parts": {k: round(v, 3) for k, v in parts.items()},
        "reason": "; ".join(reasons) if reasons else "conditions healthy",
        "blocked": False,
    }


def compute_capacity_score(telemetry: dict):
    """Backwards-compatible scalar score. None when telemetry is absent."""
    return compute_capacity_report(telemetry)["score"]


def score_to_local_epochs(score, base_epochs: int = 2) -> int:
    """Translate a score into an actual training instruction.

    Returns 0 only for a hard-gated device. Everything else trains, at a
    reduced epoch count — a throttled device is still useful, and dropping
    it entirely biases the global model against low-power hardware.
    """
    if score is None:
        return base_epochs          # unknown device: give it a normal round
    if score <= 0.0:
        return 0
    if score < 0.35:
        return 1
    if score < 0.60:
        return max(1, int(base_epochs * 0.5))
    if score < 0.85:
        return max(1, int(base_epochs * 0.75))
    return base_epochs


if __name__ == "__main__":
    cases = {
        "Healthy laptop": {"telem_cpu_percent": 34.3, "telem_mem_percent_used": 74.7,
                           "telem_has_battery": 1, "telem_battery_percent": 92,
                           "telem_battery_plugged_in": 0, "telem_network_latency_ms": 19.5},
        "Mid-training CPU spike": {"telem_cpu_percent": 100.0, "telem_mem_percent_used": 69.5,
                                   "telem_has_battery": 1, "telem_battery_percent": 61,
                                   "telem_battery_plugged_in": 0, "telem_network_latency_ms": 405.3},
        "Low battery, unplugged": {"telem_cpu_percent": 20.0, "telem_mem_percent_used": 50.0,
                                   "telem_has_battery": 1, "telem_battery_percent": 10,
                                   "telem_battery_plugged_in": 0, "telem_network_latency_ms": 25.0},
        "Network unreachable": {"telem_cpu_percent": 20.0, "telem_mem_percent_used": 40.0,
                                "telem_has_battery": 1, "telem_battery_percent": 80,
                                "telem_battery_plugged_in": 1, "telem_network_latency_ms": -1},
        "Desktop, no battery": {"telem_cpu_percent": 30.0, "telem_mem_percent_used": 55.0,
                                "telem_has_battery": 0, "telem_network_latency_ms": 15.0},
        "Partial telemetry": {"telem_cpu_percent": 44.0},
    }
    for name, telem in cases.items():
        rep = compute_capacity_report(telem)
        print(f"{name:26} score={rep['score']}  epochs={score_to_local_epochs(rep['score'])}"
              f"  parts={rep['parts']}  ({rep['reason']})")