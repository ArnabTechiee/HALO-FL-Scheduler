"""
HALO Capacity Scoring
----------------------
Converts a raw telemetry snapshot into a single score (0.0 to 1.0)
representing how much training work a client can currently handle.

0.0 = skip this client entirely this round
1.0 = full/normal workload
Values in between = reduced workload (fewer local epochs / smaller batch)
"""

# -------------------------------------------------------------------
# TEMPORARY TEST FLAG – set to False for normal operation
# When True, every client will be skipped (score 0.0) to test Flower's
# behaviour when no clients participate in a round.
# -------------------------------------------------------------------
FORCE_SKIP_ALL = False   # <-- Disabled – normal scoring is now active

def unflatten_telemetry(flat: dict) -> dict:
    """Reverses telemetry.py's flatten_telemetry() back into the nested
    shape compute_capacity_score() expects."""
    has_battery = flat.get("telem_has_battery", 0)
    battery = None
    if has_battery:
        battery = {
            "percent": flat["telem_battery_percent"],
            "plugged_in": bool(flat["telem_battery_plugged_in"]),
        }

    latency = flat.get("telem_network_latency_ms", -1)

    return {
        "cpu_percent": flat["telem_cpu_percent"],
        "memory": {"percent_used": flat["telem_mem_percent_used"]},
        "battery": battery,
        "network_latency_ms": None if latency == -1 else latency,
    }

def compute_capacity_score(telemetry: dict) -> float:
    """
    Accepts a flat telemetry dict using the telem_* field names sent by
    the client (see client_app.py's flatten_telemetry()).
    Returns 0.0 (skip) to 1.0 (full capacity).
    """
    # -----------------------------------------------------------------
    # TEMPORARY OVERRIDE – remove this block after testing
    if FORCE_SKIP_ALL:
        return 0.0
    # -----------------------------------------------------------------

    score = 1.0

    # --- Battery ---
    has_battery = telemetry.get("telem_has_battery", False)
    if has_battery:
        battery_pct = telemetry["telem_battery_percent"]
        plugged_in = telemetry["telem_battery_plugged_in"]
        if battery_pct < 15 and not plugged_in:
            return 0.0  # hard skip — protect the device
        elif battery_pct < 30 and not plugged_in:
            score *= 0.5
    # no battery (desktop) -> skip this check entirely, no penalty

    # --- CPU ---
    cpu = telemetry["telem_cpu_percent"]
    if cpu > 85:
        score *= 0.4
    elif cpu > 60:
        score *= 0.7

    # --- Memory ---
    if telemetry["telem_mem_percent_used"] > 90:
        score *= 0.5

    # --- Network ---
    latency = telemetry["telem_network_latency_ms"]
    if latency == -1:  # sentinel for unreachable
        return 0.0
    elif latency > 300:
        score *= 0.6

    return round(max(score, 0.0), 2)


def score_to_local_epochs(score: float, base_epochs: int = 1) -> int:
    """Translates a score into an actual training instruction.
    With base_epochs=1 (your current quickstart default), this mostly
    becomes a participate/skip decision rather than a fine-grained scale
    — that's fine for a first working version."""
    if score == 0.0:
        return 0
    elif score < 1.0:
        return max(1, round(base_epochs * 0.5))
    else:
        return base_epochs


if __name__ == "__main__":
    # Quick manual tests using the new flat telem_* fields
    test_cases = [
        {
            "name": "Healthy laptop (like your real reading)",
            "telemetry": {
                "telem_cpu_percent": 34.3,
                "telem_mem_percent_used": 74.7,
                "telem_has_battery": True,
                "telem_battery_percent": 92,
                "telem_battery_plugged_in": False,
                "telem_network_latency_ms": 19.5,
            },
        },
        {
            "name": "Low battery, unplugged",
            "telemetry": {
                "telem_cpu_percent": 20.0,
                "telem_mem_percent_used": 50.0,
                "telem_has_battery": True,
                "telem_battery_percent": 10,
                "telem_battery_plugged_in": False,
                "telem_network_latency_ms": 25.0,
            },
        },
        {
            "name": "CPU maxed out",
            "telemetry": {
                "telem_cpu_percent": 95.0,
                "telem_mem_percent_used": 60.0,
                "telem_has_battery": True,
                "telem_battery_percent": 80,
                "telem_battery_plugged_in": True,
                "telem_network_latency_ms": 30.0,
            },
        },
        {
            "name": "Network unreachable",
            "telemetry": {
                "telem_cpu_percent": 20.0,
                "telem_mem_percent_used": 40.0,
                "telem_has_battery": True,
                "telem_battery_percent": 80,
                "telem_battery_plugged_in": True,
                "telem_network_latency_ms": -1,
            },
        },
        {
            "name": "Desktop (no battery)",
            "telemetry": {
                "telem_cpu_percent": 30.0,
                "telem_mem_percent_used": 55.0,
                "telem_has_battery": False,
                "telem_battery_percent": -1,  # placeholder, won't be used
                "telem_battery_plugged_in": False,
                "telem_network_latency_ms": 15.0,
            },
        },
    ]

    for case in test_cases:
        score = compute_capacity_score(case["telemetry"])
        epochs = score_to_local_epochs(score)
        print(f"{case['name']:35} -> score={score:.2f}, local_epochs={epochs}")