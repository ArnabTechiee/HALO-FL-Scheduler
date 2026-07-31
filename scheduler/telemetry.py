"""
HALO Telemetry Agent
---------------------
Reads live hardware signals (CPU load, RAM headroom, battery %, network
latency) from the laptop it runs on. This is the raw data the adaptive
scheduler will later use to decide how much training work each laptop
should get.

Includes a flattening function that converts the nested snapshot into
a flat dictionary of scalars safe for MetricRecord (int/float only).
"""

import psutil
import socket
import time
import json
import hashlib


def get_device_id(salt: str = "") -> int:
    """Stable numeric ID for this physical client, derived from its
    hostname. Unlike Flower's node_id (which changes every time the
    SuperNode process restarts), this stays constant across reconnects —
    letting the Coordinator recognize a returning device as the same one.

    `salt` distinguishes multiple client processes running on the SAME
    physical machine (e.g. two SuperNode terminals on one laptop during
    local testing) — without it, both would hash to the identical
    device_id and the Coordinator would wrongly treat them as the same
    device bouncing between connections. Pass something that's stable
    per-process across restarts but differs between processes on the
    same host — e.g. the assigned partition-id.

    On genuinely separate physical laptops, hostnames already differ,
    so the salt has no effect there and isn't required — but passing it
    unconditionally is harmless and keeps single-machine testing safe
    too, so client_app.py always supplies it.
    """
    hostname = socket.gethostname()
    key = f"{hostname}:{salt}" if salt else hostname
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def get_cpu_usage():
    """Returns current CPU usage as a percentage (0-100).
    interval=1 means it measures over 1 second for an accurate reading
    (instant reads with interval=0 can be misleading/noisy)."""
    return psutil.cpu_percent(interval=1)


def get_memory_info():
    """Returns RAM stats: total, available, and percent used."""
    mem = psutil.virtual_memory()
    return {
        "total_gb": round(mem.total / (1024 ** 3), 2),
        "available_gb": round(mem.available / (1024 ** 3), 2),
        "percent_used": mem.percent,
    }


def get_battery_info():
    """Returns battery percentage and charging status.
    Returns None if the machine has no battery (e.g. a desktop) —
    the scheduler should treat that as 'always fine, ignore battery'."""
    battery = psutil.sensors_battery()
    if battery is None:
        return None
    return {
        "percent": battery.percent,
        "plugged_in": battery.power_plugged,
        # secsleft can be -1 or a huge number when plugged in/unknown
        "minutes_left": (
            round(battery.secsleft / 60, 1)
            if battery.secsleft not in (-1, -2)
            else None
        ),
    }


def get_network_latency(host="8.8.8.8", port=53, timeout=2):
    """Measures round-trip time (ms) to a reachable host as a simple
    proxy for network quality. Uses a raw TCP connect rather than ping,
    since ping (ICMP) is sometimes blocked while TCP isn't.
    Returns None if unreachable within the timeout."""
    try:
        start = time.time()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        sock.close()
        return round((time.time() - start) * 1000, 1)  # ms
    except (socket.error, socket.timeout):
        return None


def get_telemetry_snapshot():
    """Combines all signals into a single dictionary — this is what
    gets sent to the Coordinator/scheduler in the real system."""
    return {
        "timestamp": time.time(),
        "hostname": socket.gethostname(),
        "cpu_percent": get_cpu_usage(),
        "memory": get_memory_info(),
        "battery": get_battery_info(),
        "network_latency_ms": get_network_latency(),
    }


def flatten_telemetry(snapshot: dict, device_id: int) -> dict:
    """Flattens the nested telemetry snapshot into flat scalar fields,
    since MetricRecord only supports int/float (no bools, no nested dicts,
    no None). Booleans are converted to 1/0, missing readings use -1.

    device_id is now passed in (computed once by the caller via
    get_device_id(salt=...)) rather than recomputed here, since the
    caller is the one that knows the correct salt to use (e.g. its own
    partition-id) — this function has no way to know that on its own.
    """
    battery = snapshot.get("battery")
    return {
        "telem_device_id": device_id,
        "telem_cpu_percent": snapshot["cpu_percent"],
        "telem_mem_percent_used": snapshot["memory"]["percent_used"],
        "telem_mem_available_gb": snapshot["memory"]["available_gb"],
        "telem_battery_percent": battery["percent"] if battery else -1,
        "telem_battery_plugged_in": int(battery["plugged_in"]) if battery else 0,
        "telem_has_battery": int(battery is not None),
        "telem_network_latency_ms": (
            snapshot["network_latency_ms"]
            if snapshot["network_latency_ms"] is not None
            else -1
        ),
    }


def print_snapshot_loop(interval_seconds=5):
    """Prints a fresh telemetry snapshot every few seconds — useful for
    watching how the numbers change on a real laptop over time (e.g.
    unplug the charger and watch battery/plugged_in update)."""
    print(f"Starting telemetry monitor (Ctrl+C to stop)...\n")
    try:
        while True:
            snapshot = get_telemetry_snapshot()
            print(json.dumps(snapshot, indent=2))
            print("-" * 40)
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    # Standalone test — just run: python telemetry.py
    print_snapshot_loop(interval_seconds=5)