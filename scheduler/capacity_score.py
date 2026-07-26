"""
HALO Capacity Scoring
----------------------
Converts a raw telemetry snapshot into a single score (0.0 to 1.0)
representing how much training work a client can currently handle.

0.0 = skip this client entirely this round
1.0 = full/normal workload
Values in between = reduced workload (fewer local epochs / smaller batch)
"""


def compute_capacity_score(telemetry: dict) -> float:
    score = 1.0

    # --- Battery ---
    battery = telemetry.get("battery")
    if battery is not None:
        if battery["percent"] < 15 and not battery["plugged_in"]:
            return 0.0  # hard skip — protect the device
        elif battery["percent"] < 30 and not battery["plugged_in"]:
            score *= 0.5
    # battery is None (desktop) -> no penalty, skip this check entirely

    # --- CPU ---
    cpu = telemetry["cpu_percent"]
    if cpu > 85:
        score *= 0.4
    elif cpu > 60:
        score *= 0.7

    # --- Memory ---
    if telemetry["memory"]["percent_used"] > 90:
        score *= 0.5

    # --- Network ---
    latency = telemetry.get("network_latency_ms")
    if latency is None:
        return 0.0  # unreachable -> skip
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
    elif score < 0.5:
        return max(1, round(base_epochs * 0.5))
    else:
        return base_epochs


if __name__ == "__main__":
    # Quick manual tests using realistic and edge-case telemetry
    test_cases = [
        {
            "name": "Healthy laptop (like your real reading)",
            "telemetry": {
                "cpu_percent": 34.3,
                "memory": {"percent_used": 74.7},
                "battery": {"percent": 92, "plugged_in": False},
                "network_latency_ms": 19.5,
            },
        },
        {
            "name": "Low battery, unplugged",
            "telemetry": {
                "cpu_percent": 20.0,
                "memory": {"percent_used": 50.0},
                "battery": {"percent": 10, "plugged_in": False},
                "network_latency_ms": 25.0,
            },
        },
        {
            "name": "CPU maxed out",
            "telemetry": {
                "cpu_percent": 95.0,
                "memory": {"percent_used": 60.0},
                "battery": {"percent": 80, "plugged_in": True},
                "network_latency_ms": 30.0,
            },
        },
        {
            "name": "Network unreachable",
            "telemetry": {
                "cpu_percent": 20.0,
                "memory": {"percent_used": 40.0},
                "battery": {"percent": 80, "plugged_in": True},
                "network_latency_ms": None,
            },
        },
        {
            "name": "Desktop (no battery)",
            "telemetry": {
                "cpu_percent": 30.0,
                "memory": {"percent_used": 55.0},
                "battery": None,
                "network_latency_ms": 15.0,
            },
        },
    ]

    for case in test_cases:
        score = compute_capacity_score(case["telemetry"])
        epochs = score_to_local_epochs(score)
        print(f"{case['name']:35} -> score={score:.2f}, local_epochs={epochs}")