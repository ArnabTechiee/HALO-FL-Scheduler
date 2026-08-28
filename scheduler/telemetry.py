"""
HALO Telemetry Agent
---------------------
Reads live hardware signals (CPU load, RAM headroom, battery %, network
latency) from the laptop it runs on. This is the raw data the adaptive
scheduler uses to decide how much training work each laptop should get.

Includes a flattening function that converts the nested snapshot into
a flat dictionary of scalars safe for MetricRecord (int/float only).
"""

import psutil
import socket
import time
import json
import hashlib

# Sampling window for the CPU read. Long enough to be a real measurement,
# short enough that three snapshots per round per device cost ~1s total.
CPU_SAMPLE_SECONDS = 0.3

# Consecutive rounds where every latency target was unreachable. One failed
# probe is a blip; two in a row is a genuinely offline device.
_latency_failures = 0

LATENCY_TARGETS = (("8.8.8.8", 53), ("1.1.1.1", 53))
SOFT_FAIL_LATENCY_MS = 900.0


def get_device_id(salt: str = "") -> int:
    """Stable numeric ID for this physical client, derived from its
    hostname. Unlike Flower's node_id (which changes every time the
    SuperNode process reconnects), this stays constant across reconnects —
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
    """Current CPU usage as a percentage (0-100).

    Uses a short blocking sample, NOT interval=None. Flower spawns a fresh
    ClientApp process per message, so a non-blocking read has no meaningful
    baseline to measure against — it reports CPU time since module import,
    which is milliseconds earlier. That produced exactly the garbage seen in
    the last run: 100.0 in rounds 1-2 (import raced the model/data load) and
    0.0 in round 4. interval=1 was accurate but cost a full second per call;
    0.3s is a real measurement at a third of the price.
    """
    return psutil.cpu_percent(interval=CPU_SAMPLE_SECONDS)


def get_memory_info():
    """RAM stats: total, available, and percent used."""
    mem = psutil.virtual_memory()
    return {
        "total_gb": round(mem.total / (1024 ** 3), 2),
        "available_gb": round(mem.available / (1024 ** 3), 2),
        "percent_used": mem.percent,
    }


def get_battery_info():
    """Battery percentage and charging status.

    Returns None if the machine has no battery (e.g. a desktop) — the
    scheduler treats that as 'always fine, ignore battery'.
    """
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


def get_network_latency(timeout=2):
    """Round-trip time (ms) to a reachable host, as a proxy for link quality.

    Uses a raw TCP connect rather than ping, since ICMP is often blocked
    while TCP isn't.

    Tries a second target before giving up, and reports a single total
    failure as merely slow rather than offline. This matters because the
    capacity scorer treats an unreachable network as a HARD SKIP: one
    firewalled DNS port on campus Wi-Fi or a VPN would otherwise eject
    every device in the fleet from every round. Returns None only after
    two consecutive rounds where no target answered.
    """
    global _latency_failures
    for target in LATENCY_TARGETS:
        sock = None
        try:
            start = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect(target)
            _latency_failures = 0
            return round((time.time() - start) * 1000, 1)
        except (socket.error, socket.timeout):
            continue
        finally:
            if sock is not None:
                try:
                    sock.close()
                except socket.error:
                    pass

    _latency_failures += 1
    return None if _latency_failures >= 2 else SOFT_FAIL_LATENCY_MS


def get_telemetry_snapshot():
    """All signals in one dictionary — this is what gets sent to the
    Coordinator/scheduler."""
    return {
        "timestamp": time.time(),
        "hostname": socket.gethostname(),
        "cpu_percent": get_cpu_usage(),
        "memory": get_memory_info(),
        "battery": get_battery_info(),
        "network_latency_ms": get_network_latency(),
    }


def flatten_telemetry(snapshot: dict, device_id: int) -> dict:
    """Flatten the nested snapshot into scalar fields, since MetricRecord
    only supports int/float (no bools, no nested dicts, no None). Booleans
    become 1/0; missing readings use -1.

    -1 is a SENTINEL, not a measurement. Consumers must gate on
    telem_has_battery before reading telem_battery_percent, and treat a
    negative latency as 'unreachable' rather than 'very fast'.

    device_id is passed in (computed once by the caller via
    get_device_id(salt=...)) rather than recomputed here, since only the
    caller knows the correct salt to use — its own partition-id.
    """
    battery = snapshot.get("battery")
    latency = snapshot.get("network_latency_ms")
    return {
        "telem_device_id": device_id,
        "telem_cpu_percent": snapshot["cpu_percent"],
        "telem_mem_percent_used": snapshot["memory"]["percent_used"],
        "telem_mem_available_gb": snapshot["memory"]["available_gb"],
        "telem_battery_percent": battery["percent"] if battery else -1,
        "telem_battery_plugged_in": int(battery["plugged_in"]) if battery else 0,
        "telem_has_battery": int(battery is not None),
        "telem_network_latency_ms": latency if latency is not None else -1,
    }


def print_snapshot_loop(interval_seconds=5):
    """Print a fresh snapshot every few seconds — useful for watching the
    numbers move on a real laptop (unplug the charger and watch
    battery/plugged_in update)."""
    print("Starting telemetry monitor (Ctrl+C to stop)...\n")
    try:
        while True:
            snapshot = get_telemetry_snapshot()
            print(json.dumps(snapshot, indent=2))
            print("-" * 40)
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    print("Single snapshot:")
    print(json.dumps(get_telemetry_snapshot(), indent=2))
    print("\nFlattened:")
    print(json.dumps(flatten_telemetry(get_telemetry_snapshot(),
                                       get_device_id(salt="0")), indent=2))