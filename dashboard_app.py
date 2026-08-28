"""
HALO Live Dashboard v8 — "Orbit"
=====================================================================
Run:  python dashboard_app.py
Open: http://127.0.0.1:5000

What changed vs v6
------------------
The v6 UI fought itself: DOM cards positioned by CSS grid, SVG wires
positioned by getBoundingClientRect, and particles positioned by a third
code path. Every release added another patch to keep those three in sync
(clipping, index maps, reconciliation, a console.warn tripwire).

v7 removes the class of bug instead of patching it:

  * The topology is ONE <canvas>. Aggregator, links, packets, coverage
    arcs and node dials are all drawn from the same layout pass, so a
    packet cannot drift off a wire and a wire cannot leak into a panel.
    Links are quadratic beziers; packets are sampled from the identical
    bezier. Same math, same frame.
  * Node cards moved into a "Fleet" rail as a compact roster — no more
    load-bearing geometry in the DOM.
  * Positions are spring-eased, so nodes joining or leaving mid-round
    fly out of / back into the hub instead of snapping.
  * Numbers interpolate every frame; the elapsed clock ticks locally
    between polls instead of stepping once every 1.5s.
  * Event log is still append-only and DOM-capped, keyed by
    timestamp+message.
  * Added a connection state machine: if /api/state fails, the header
    shows "Reconnecting" instead of silently freezing on stale numbers.

Backend contract is unchanged: /api/state, /api/comparison,
dashboard_state.load_state(), comparison_runs.json.
"""

import json
import os
import time
from pathlib import Path

from flask import Flask, jsonify, Response

app = Flask(__name__)

# The dashboard reads the state file written by dashboard_state.py. Reading
# the JSON directly (instead of importing that module across package
# boundaries) means the dashboard runs from any directory without a
# sys.path hack, and never crashes because the scheduler package moved.
HALO_DIR = Path(os.environ.get("HALO_DIR", Path.home() / ".halo"))
STATE_PATH = Path(os.environ.get("HALO_STATE_PATH", HALO_DIR / "dashboard_state.json"))
COMPARISON_CANDIDATES = [
    Path(os.environ.get("HALO_COMPARISON_PATH", HALO_DIR / "comparison_runs.json")),
    Path(__file__).parent / "comparison_runs.json",
]

EMPTY_STATE = {
    "started_at": None, "current_round": 0, "round_in_progress": None,
    "round_started_at": None, "avg_round_duration": None, "total_rounds": None,
    "global_accuracy": None, "global_loss": None, "history": [],
    "run": {}, "nodes": {}, "events": [],
}


def load_state() -> dict:
    if not STATE_PATH.exists():
        return dict(EMPTY_STATE)
    try:
        with open(STATE_PATH, "r") as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError):
        return dict(EMPTY_STATE)
    for k, v in EMPTY_STATE.items():
        state.setdefault(k, v)
    return state


def _load_comparison():
    for path in COMPARISON_CANDIDATES:
        if not path.exists():
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if data and (data.get("series_a") or data.get("series_b")):
            return data
    return None


def _as_float(value, default=None):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _derive_summary(state: dict) -> dict:
    nodes = state.get("nodes", {})
    counts = {"training": 0, "active": 0, "skipped": 0,
              "dropped": 0, "reconnected": 0, "retired": 0}

    total_examples = 0
    battery_vals, latency_vals, mem_vals = [], [], []
    covering_nodes = 0

    for n in nodes.values():
        st = n.get("status")
        if st in counts:
            counts[st] += 1

        if st not in ("retired", "dropped"):
            total_examples += int(_as_float(n.get("num_examples"), 0) or 0)
            for key, bucket in (("battery_percent", battery_vals),
                                ("network_latency_ms", latency_vals),
                                ("mem_percent_used", mem_vals)):
                val = _as_float(n.get(key))
                # -1 is telemetry.py's "not available" sentinel (a desktop
                # with no battery, or an unreachable network), not a reading.
                if val is not None and val >= 0:
                    bucket.append(val)

        if n.get("covering_partitions"):
            covering_nodes += 1

    started_at = state.get("started_at")
    elapsed = (time.time() - started_at) if started_at else 0
    connected = sum(1 for n in nodes.values()
                    if n.get("status") not in ("retired", "dropped"))

    progress_pct = None
    round_in_progress = state.get("round_in_progress")
    if round_in_progress is not None and state.get("round_started_at"):
        elapsed_this_round = time.time() - state["round_started_at"]
        est = state.get("avg_round_duration")
        if est and est > 0:
            progress_pct = min(97, round(elapsed_this_round / est * 100))

    def _avg(vals):
        return (sum(vals) / len(vals)) if vals else None

    def _median(vals):
        """The tile is labelled MEDIAN LINK, so give it a median. A mean of
        1026ms and 27ms is 527ms — a number describing neither device."""
        if not vals:
            return None
        ordered = sorted(vals)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2

    return {
        "status_counts": counts,
        "total_examples_this_round": total_examples,
        "elapsed_seconds": elapsed,
        "connected_nodes": connected,
        "covering_nodes": covering_nodes,
        "avg_battery": _avg(battery_vals),
        "avg_latency_ms": _avg(latency_vals),
        "median_latency_ms": _median(latency_vals),
        "avg_memory": _avg(mem_vals),
        "round_in_progress": round_in_progress,
        "round_progress_pct": progress_pct,
        "server_time": time.time(),
    }


def _state_payload():
    state = load_state()
    state["summary"] = _derive_summary(state)
    state["nodes"] = {
        nid: n for nid, n in state.get("nodes", {}).items()
        if n.get("status") != "retired"
    }
    return state


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HALO — Federated Scheduler</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{
  --void:#05070f;
  --ink:#080b16;
  --panel:rgba(19,24,42,.62);
  --panel-solid:#121728;
  --line:rgba(122,146,214,.14);
  --line-soft:rgba(122,146,214,.08);
  --text:#e9edfb;
  --dim:#848eb2;
  --faint:#525b7a;
  --signal:#6c8cff;
  --iris:#a78bfa;
  --mint:#35d6a4;
  --amber:#f5b544;
  --coral:#ff5d73;
  --sky:#38d6ee;
  --r:14px;
}

*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;width:100%;overflow:hidden}
body{
  display:flex;flex-direction:column;
  background:var(--void);
  color:var(--text);
  font-family:'Inter',system-ui,-apple-system,sans-serif;
  font-size:13px;
  -webkit-font-smoothing:antialiased;
}
::selection{background:rgba(108,140,255,.3)}

/* ambient field */
.field{position:fixed;inset:0;z-index:0;pointer-events:none}
.field::before{
  content:'';position:absolute;inset:0;
  background:
    radial-gradient(900px 600px at 22% 42%, rgba(108,140,255,.10), transparent 62%),
    radial-gradient(760px 520px at 84% 12%, rgba(167,139,250,.07), transparent 60%),
    radial-gradient(700px 700px at 60% 100%, rgba(53,214,164,.045), transparent 60%);
}
.field::after{
  content:'';position:absolute;inset:0;opacity:.45;
  background-image:
    linear-gradient(rgba(122,146,214,.045) 1px,transparent 1px),
    linear-gradient(90deg,rgba(122,146,214,.045) 1px,transparent 1px);
  background-size:64px 64px;
  mask-image:radial-gradient(ellipse at 50% 40%,#000 20%,transparent 78%);
  -webkit-mask-image:radial-gradient(ellipse at 50% 40%,#000 20%,transparent 78%);
}

/* ── header ───────────────────────────────────────────── */
.masthead{
  position:relative;z-index:10;height:58px;
  display:flex;align-items:center;gap:22px;
  padding:0 20px;
  border-bottom:1px solid var(--line);
  background:linear-gradient(180deg,rgba(8,11,22,.94),rgba(8,11,22,.74));
  backdrop-filter:blur(24px);
  flex-shrink:0;
}
.mark{display:flex;align-items:center;gap:11px;flex-shrink:0}
.mark-glyph{
  width:32px;height:32px;border-radius:10px;position:relative;
  background:conic-gradient(from 140deg,var(--signal),var(--iris),var(--sky),var(--signal));
  display:grid;place-items:center;
  box-shadow:0 0 22px rgba(108,140,255,.35);
}
.mark-glyph::after{
  content:'';position:absolute;inset:2px;border-radius:8px;background:#0a0e1c;
}
.mark-glyph span{
  position:relative;z-index:1;font-family:'Space Grotesk',sans-serif;
  font-weight:700;font-size:15px;
  background:linear-gradient(135deg,var(--signal),var(--sky));
  -webkit-background-clip:text;background-clip:text;color:transparent;
}
.mark-name{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:16px;letter-spacing:-.2px;line-height:1.1}
.mark-sub{font-size:9px;color:var(--faint);letter-spacing:1.6px;text-transform:uppercase;font-weight:600;margin-top:1px}

.masthead-div{width:1px;height:26px;background:var(--line);flex-shrink:0}

.run-id{
  font-family:'JetBrains Mono',monospace;font-size:10.5px;color:var(--faint);
  letter-spacing:.4px;white-space:nowrap;
}
.run-id b{color:var(--dim);font-weight:500}

.masthead-right{margin-left:auto;display:flex;align-items:center;gap:14px}
.clock{
  font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--dim);
  letter-spacing:.5px;font-variant-numeric:tabular-nums;
}
.wire{
  display:flex;align-items:center;gap:7px;padding:5px 12px 5px 10px;
  border-radius:999px;border:1px solid rgba(53,214,164,.22);
  background:rgba(53,214,164,.07);
  font-size:10px;font-weight:700;letter-spacing:.9px;color:var(--mint);
  transition:.3s;
}
.wire i{width:6px;height:6px;border-radius:50%;background:var(--mint);box-shadow:0 0 9px var(--mint);animation:blip 1.6s ease infinite}
.wire.down{border-color:rgba(255,93,115,.28);background:rgba(255,93,115,.08);color:var(--coral)}
.wire.down i{background:var(--coral);box-shadow:0 0 9px var(--coral);animation-duration:.7s}
.wire.done{border-color:rgba(108,140,255,.3);background:rgba(108,140,255,.08);color:var(--signal)}
.wire.done i{background:var(--signal);box-shadow:0 0 9px var(--signal);animation:none}

/* ── metric strip ─────────────────────────────────────── */
.strip{
  position:relative;z-index:9;
  display:grid;grid-template-columns:repeat(7,1fr);
  border-bottom:1px solid var(--line);
  background:rgba(9,12,24,.55);
  backdrop-filter:blur(18px);
  flex-shrink:0;
}
.tile{padding:11px 18px;border-right:1px solid var(--line-soft);position:relative;overflow:hidden}
.tile:last-child{border-right:none}
.tile-k{font-size:9px;font-weight:700;letter-spacing:1.3px;text-transform:uppercase;color:var(--faint)}
.tile-v{
  font-family:'JetBrains Mono',monospace;font-weight:700;font-size:21px;
  letter-spacing:-.6px;font-variant-numeric:tabular-nums;margin-top:3px;line-height:1.15;
}
.tile-v small{font-size:12px;font-weight:500;color:var(--faint);letter-spacing:0}
.tile-n{font-size:10px;color:var(--faint);margin-top:2px;height:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.v-signal{color:var(--signal)} .v-mint{color:var(--mint)} .v-coral{color:var(--coral)}
.v-amber{color:var(--amber)} .v-iris{color:var(--iris)} .v-sky{color:var(--sky)}
.tile-bar{position:absolute;left:0;bottom:0;height:2px;width:0;background:linear-gradient(90deg,var(--signal),var(--sky));transition:none}

/* ── shell ────────────────────────────────────────────── */
.shell{
  position:relative;z-index:5;
  display:grid;
  grid-template-columns:minmax(0,1fr) 356px;
  grid-template-rows:minmax(0,1fr) 218px;
  grid-template-areas:"stage rail" "decks rail";
  gap:12px;padding:12px;
  flex:1;min-height:0;
}

.card{
  background:var(--panel);
  border:1px solid var(--line);
  border-radius:var(--r);
  backdrop-filter:blur(20px);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.045), 0 24px 48px -28px rgba(0,0,0,.9);
  display:flex;flex-direction:column;overflow:hidden;min-height:0;
}
.card-head{
  display:flex;align-items:center;gap:10px;
  padding:11px 14px;border-bottom:1px solid var(--line-soft);flex-shrink:0;
}
.card-title{
  font-family:'Space Grotesk',sans-serif;font-size:11.5px;font-weight:600;
  letter-spacing:.8px;text-transform:uppercase;color:var(--text);
}
.card-note{margin-left:auto;font-size:9.5px;color:var(--faint);letter-spacing:.6px;font-family:'JetBrains Mono',monospace}
.tick{width:3px;height:13px;border-radius:2px;background:linear-gradient(180deg,var(--signal),var(--iris))}

/* ── topology stage ───────────────────────────────────── */
.stage{grid-area:stage;position:relative}
.stage-body{flex:1;position:relative;min-height:0}
#topo{position:absolute;inset:0;display:block;width:100%;height:100%}
.legend{
  position:absolute;left:14px;bottom:12px;display:flex;gap:14px;flex-wrap:wrap;
  padding:7px 12px;border-radius:10px;
  background:rgba(6,9,18,.6);border:1px solid var(--line-soft);backdrop-filter:blur(10px);
}
.lg{display:flex;align-items:center;gap:6px;font-size:9.5px;color:var(--dim);letter-spacing:.5px;font-weight:500}
.lg i{width:7px;height:7px;border-radius:2px}
.tip{
  position:absolute;pointer-events:none;opacity:0;transform:translate(-50%,-130%);
  padding:9px 11px;border-radius:10px;min-width:150px;
  background:rgba(7,10,20,.95);border:1px solid var(--line);
  box-shadow:0 16px 40px -12px rgba(0,0,0,.9);
  transition:opacity .14s;z-index:20;
}
.tip-t{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;margin-bottom:5px}
.tip-r{display:flex;justify-content:space-between;gap:16px;font-size:10.5px;color:var(--dim);line-height:1.6}
.tip-r b{color:var(--text);font-weight:600;font-family:'JetBrains Mono',monospace}
.tip-parts{margin-top:7px;padding-top:6px;border-top:1px solid var(--line-soft);display:flex;flex-direction:column;gap:3px}
.tip-part{display:flex;align-items:center;gap:6px;font-size:9px;color:var(--faint);text-transform:uppercase;letter-spacing:.6px}
.tip-part span{width:22px}
.tip-part i{flex:1;height:3px;border-radius:99px;background:rgba(122,146,214,.14);overflow:hidden}
.tip-part b{display:block;height:100%;border-radius:99px}
.tip-why{margin-top:6px;font-size:9.5px;color:var(--amber);line-height:1.45;max-width:190px}

/* ── decks (charts) ───────────────────────────────────── */
.decks{grid-area:decks;display:grid;grid-template-columns:1fr 1fr;gap:12px;min-height:0}
.chart-body{flex:1;min-height:0;padding:8px 12px 10px;position:relative}
.chart-body canvas{width:100%!important;height:100%!important}
.blank{
  position:absolute;inset:10px;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:6px;text-align:center;
  border:1px dashed var(--line);border-radius:10px;color:var(--faint);font-size:11.5px;line-height:1.6;
}
.blank b{color:var(--dim);font-weight:600;font-size:12px}
.blank code{font-family:'JetBrains Mono',monospace;font-size:10.5px;color:var(--sky);background:rgba(56,214,238,.08);padding:2px 6px;border-radius:5px}

/* ── rail ─────────────────────────────────────────────── */
.rail{grid-area:rail;display:flex;flex-direction:column;gap:12px;min-height:0}
.fleet{flex:1.9;min-height:0}
.logcard{flex:1;min-height:0}
.scroll{flex:1;min-height:0;overflow-y:auto;padding:10px 12px}
.scroll::-webkit-scrollbar{width:5px}
.scroll::-webkit-scrollbar-thumb{background:rgba(122,146,214,.2);border-radius:99px}
.scroll::-webkit-scrollbar-track{background:transparent}

.unit{
  border:1px solid var(--line-soft);border-radius:11px;
  padding:8px 10px;margin-bottom:7px;
  background:linear-gradient(180deg,rgba(24,30,50,.5),rgba(14,18,32,.5));
  animation:rise .34s cubic-bezier(.2,.9,.3,1) both;
  transition:border-color .25s, transform .18s;
  position:relative;overflow:hidden;
}
.unit:hover{transform:translateX(-2px)}
.unit::before{content:'';position:absolute;left:0;top:0;bottom:0;width:2px;background:var(--faint);opacity:.85}
.unit[data-s="training"]::before{background:var(--sky)}
.unit[data-s="active"]::before{background:var(--mint)}
.unit[data-s="skipped"]::before{background:var(--amber)}
.unit[data-s="dropped"]::before{background:var(--coral)}
.unit[data-s="reconnected"]::before{background:var(--iris)}
.unit[data-s="training"]{border-color:rgba(56,214,238,.24)}
.unit[data-s="dropped"]{border-color:rgba(255,93,115,.3);background:linear-gradient(180deg,rgba(44,20,28,.55),rgba(16,12,20,.5))}
.unit.leaving{animation:sink .3s ease forwards}

.unit-top{display:flex;align-items:center;gap:8px;margin-bottom:7px}
.uid{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;letter-spacing:-.2px}
.ustat{font-size:8.5px;font-weight:700;letter-spacing:1px;text-transform:uppercase;padding:2px 6px;border-radius:5px}
.ustat[data-s="training"]{color:var(--sky);background:rgba(56,214,238,.11)}
.ustat[data-s="active"]{color:var(--mint);background:rgba(53,214,164,.11)}
.ustat[data-s="skipped"]{color:var(--amber);background:rgba(245,181,68,.11)}
.ustat[data-s="dropped"]{color:var(--coral);background:rgba(255,93,115,.13)}
.ustat[data-s="reconnected"]{color:var(--iris);background:rgba(167,139,250,.12)}
.upart{margin-left:auto;font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--faint)}

.gauges{display:grid;grid-template-columns:1fr 1fr;gap:4px 9px;margin-top:5px}
.gauge{display:flex;align-items:center;gap:5px;min-width:0}
.gauge-k{font-size:8px;color:var(--faint);width:20px;font-weight:700;letter-spacing:.4px;text-transform:uppercase}
.gauge-t{flex:1;min-width:0;height:4px;border-radius:99px;background:rgba(122,146,214,.1);overflow:hidden}
.gauge-f{height:100%;border-radius:99px;transition:width .7s cubic-bezier(.2,.9,.3,1)}
.gauge-v{font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--dim);width:38px;text-align:right;font-variant-numeric:tabular-nums}
.g-mint{background:linear-gradient(90deg,var(--mint),#22b98a)}
.g-amber{background:linear-gradient(90deg,var(--amber),#d99a24)}
.g-coral{background:linear-gradient(90deg,var(--coral),#e03e56)}
.g-sky{background:linear-gradient(90deg,var(--sky),#1fa7bd)}
.g-null{background:rgba(122,146,214,.16)}
.g-iris{background:linear-gradient(90deg,var(--iris),#8b6ae0)}
.why{margin-left:auto;font-size:9px;color:var(--amber);opacity:.9;max-width:100%}

.unit-foot{
  display:flex;gap:8px;align-items:center;flex-wrap:wrap;
  margin-top:8px;padding-top:7px;border-top:1px solid var(--line-soft);
  font-family:'JetBrains Mono',monospace;font-size:9.5px;color:var(--faint);
}
.unit-foot b{color:var(--dim);font-weight:500}
.cover{
  color:var(--amber);background:rgba(245,181,68,.1);
  border:1px solid rgba(245,181,68,.22);padding:1px 6px;border-radius:5px;font-weight:700;
}

.empty-rail{color:var(--faint);font-size:11.5px;text-align:center;padding:26px 12px;line-height:1.7}

/* ── log ──────────────────────────────────────────────── */
.log{
  flex:1;min-height:0;overflow-y:auto;padding:8px 12px;
  font-family:'JetBrains Mono',monospace;font-size:10.5px;line-height:1.62;
}
.log::-webkit-scrollbar{width:5px}
.log::-webkit-scrollbar-thumb{background:rgba(122,146,214,.2);border-radius:99px}
.row{display:flex;gap:8px;padding:2.5px 0 2.5px 9px;border-left:2px solid transparent;animation:slide .3s ease both}
.row.drop{border-left-color:var(--coral)}
.row.reassign{border-left-color:var(--amber)}
.row.reconnect{border-left-color:var(--iris)}
.row.skip{border-left-color:var(--faint)}
.r-t{color:#3f4763;flex-shrink:0;font-variant-numeric:tabular-nums}
.r-g{font-size:8px;font-weight:700;padding:1px 5px;border-radius:4px;align-self:center;flex-shrink:0;letter-spacing:.5px}
.r-g.drop{background:rgba(255,93,115,.13);color:var(--coral)}
.r-g.reassign{background:rgba(245,181,68,.13);color:var(--amber)}
.r-g.reconnect{background:rgba(167,139,250,.13);color:var(--iris)}
.r-g.skip{background:rgba(122,146,214,.11);color:var(--dim)}
.r-m{color:#a9b1cb;overflow-wrap:anywhere}

/* ── toast ────────────────────────────────────────────── */
.toast{
  position:fixed;left:50%;top:74px;z-index:400;
  transform:translate(-50%,-220px);visibility:hidden;
  display:flex;align-items:center;gap:9px;
  padding:9px 16px;border-radius:999px;
  background:rgba(28,22,10,.9);border:1px solid rgba(245,181,68,.3);
  color:var(--amber);font-size:11.5px;font-weight:600;
  backdrop-filter:blur(16px);
  box-shadow:0 18px 44px -18px rgba(0,0,0,.9);
  transition:transform .45s cubic-bezier(.2,1.3,.4,1);
}
.toast.on{transform:translate(-50%,0);visibility:visible}
.toast i{width:6px;height:6px;border-radius:50%;background:var(--amber);animation:blip 1.2s infinite}

@keyframes blip{0%,100%{opacity:1}50%{opacity:.25}}
@keyframes rise{from{opacity:0;transform:translateY(9px)}to{opacity:1;transform:translateY(0)}}
@keyframes sink{to{opacity:0;transform:translateX(16px);height:0;margin:0;padding:0;border-width:0}}
@keyframes slide{from{opacity:0;transform:translateX(-5px)}to{opacity:1;transform:translateX(0)}}

@media (max-width:1400px){
  .shell{grid-template-columns:minmax(0,1fr) 300px}
  .strip{grid-template-columns:repeat(5,1fr)}
  .tile.opt{display:none}
}
@media (prefers-reduced-motion:reduce){
  *{animation-duration:.01ms!important;transition-duration:.01ms!important}
}
</style>
</head>
<body>
<div class="field"></div>

<div class="toast" id="toast"><i></i><span id="toast-msg">Partition reassigned</span></div>

<header class="masthead">
  <div class="mark">
    <div class="mark-glyph"><span>H</span></div>
    <div>
      <div class="mark-name">HALO</div>
      <div class="mark-sub">Adaptive Federated Scheduler</div>
    </div>
  </div>
  <div class="masthead-div"></div>
  <div class="run-id">strategy <b id="h-strategy">adaptive</b> · dataset <b id="h-dataset">—</b> · model <b id="h-model">—</b></div>
  <div class="masthead-right">
    <div class="clock" id="h-clock">00:00:00</div>
    <div class="wire" id="h-wire"><i></i><span id="h-wire-t">LIVE</span></div>
  </div>
</header>

<section class="strip">
  <div class="tile">
    <div class="tile-k">Round</div>
    <div class="tile-v v-signal" id="m-round">—</div>
    <div class="tile-n" id="m-round-n">waiting for scheduler</div>
    <div class="tile-bar" id="m-round-bar"></div>
  </div>
  <div class="tile">
    <div class="tile-k">Global accuracy</div>
    <div class="tile-v v-mint" id="m-acc">—</div>
    <div class="tile-n" id="m-acc-n">no evaluation yet</div>
  </div>
  <div class="tile">
    <div class="tile-k">Global loss</div>
    <div class="tile-v" id="m-loss">—</div>
    <div class="tile-n" id="m-loss-n">no evaluation yet</div>
  </div>
  <div class="tile">
    <div class="tile-k">Fleet online</div>
    <div class="tile-v v-sky" id="m-nodes">—</div>
    <div class="tile-n" id="m-nodes-n">no devices</div>
  </div>
  <div class="tile">
    <div class="tile-k">Dropped</div>
    <div class="tile-v v-coral" id="m-drop">—</div>
    <div class="tile-n" id="m-drop-n">all links healthy</div>
  </div>
  <div class="tile opt">
    <div class="tile-k">Examples in round</div>
    <div class="tile-v v-iris" id="m-ex">—</div>
    <div class="tile-n" id="m-ex-n">across contributing devices</div>
  </div>
  <div class="tile opt">
    <div class="tile-k">Median link</div>
    <div class="tile-v v-amber" id="m-lat">—</div>
    <div class="tile-n" id="m-lat-n">battery avg —</div>
  </div>
</section>

<main class="shell">

  <section class="card stage">
    <div class="card-head">
      <div class="tick"></div>
      <div class="card-title">Live topology</div>
      <div class="card-note" id="pkt-count">0 packets</div>
    </div>
    <div class="stage-body">
      <canvas id="topo"></canvas>
      <div class="tip" id="tip"></div>
      <div class="legend">
        <div class="lg"><i style="background:#38d6ee"></i>Training</div>
        <div class="lg"><i style="background:#35d6a4"></i>Ready</div>
        <div class="lg"><i style="background:#f5b544"></i>Skipped</div>
        <div class="lg"><i style="background:#ff5d73"></i>Dropped</div>
        <div class="lg"><i style="background:#a78bfa"></i>Reconnected</div>
        <div class="lg"><i style="background:#f5b544;border-radius:50%"></i>Partition takeover</div>
      </div>
    </div>
  </section>

  <section class="decks">
    <div class="card">
      <div class="card-head">
        <div class="tick"></div>
        <div class="card-title">Convergence</div>
        <div class="card-note">accuracy · loss</div>
      </div>
      <div class="chart-body">
        <canvas id="trend"></canvas>
        <div class="blank" id="trend-blank"><b>No rounds completed</b><span>The curve starts after the first aggregation.</span></div>
      </div>
    </div>
    <div class="card">
      <div class="card-head">
        <div class="tick"></div>
        <div class="card-title">Adaptive vs baseline</div>
        <div class="card-note">controlled run</div>
      </div>
      <div class="chart-body">
        <canvas id="comp" style="display:none"></canvas>
        <div class="blank" id="comp-blank">
          <b>No baseline to compare against</b>
          <span>Set <code>halo-mode = "baseline"</code> in pyproject.toml and run once.<br>Both curves are saved automatically.</span>
        </div>
      </div>
    </div>
  </section>

  <aside class="rail">
    <div class="card fleet">
      <div class="card-head">
        <div class="tick"></div>
        <div class="card-title">Fleet</div>
        <div class="card-note" id="fleet-note">—</div>
      </div>
      <div class="scroll" id="fleet">
        <div class="empty-rail">Waiting for devices to register with the coordinator.</div>
      </div>
    </div>
    <div class="card logcard">
      <div class="card-head">
        <div class="tick"></div>
        <div class="card-title">Scheduler events</div>
        <div class="card-note">streaming</div>
      </div>
      <div class="log" id="log">
        <div class="empty-rail">No drops, skips or reassignments yet.</div>
      </div>
    </div>
  </aside>
</main>

<script>
(function(){
'use strict';

/* ══ constants ══════════════════════════════════════════ */
const POLL_MS = 1500;
const MAX_LOG = 300;
const COLORS = {
  training:'#38d6ee', active:'#35d6a4', skipped:'#f5b544',
  dropped:'#ff5d73', reconnected:'#a78bfa', unknown:'#848eb2'
};
const LABEL = {
  training:'training', active:'ready', skipped:'skipped',
  dropped:'dropped', reconnected:'rejoined'
};

/* ══ dom ════════════════════════════════════════════════ */
const $ = id => document.getElementById(id);
const cv = $('topo'), ctx = cv.getContext('2d');
const tip = $('tip'), fleetEl = $('fleet'), logEl = $('log');
const toast = $('toast'), toastMsg = $('toast-msg');

/* ══ state ══════════════════════════════════════════════ */
let snapshot = null;
let nodeOrder = [];
let orbs = new Map();          // nid -> render body
let packets = [];
let packetTotal = 0;
let lastEventKey = '';
let lastToast = '';
let trendChart = null, compChart = null, trendSig = '';
let hover = null;
let online = true, failures = 0, runComplete = false;
let W = 0, H = 0, DPR = 1;
let clockBase = 0, clockAt = 0;
const units = new Map();       // nid -> fleet card element

const ease = {
  acc:{c:0,t:0}, loss:{c:0,t:0}, ex:{c:0,t:0},
  prog:{c:0,t:0}, nodes:{c:0,t:0}, drop:{c:0,t:0}, lat:{c:0,t:0}
};

/* ══ helpers ════════════════════════════════════════════ */
const num = v => (v===null||v===undefined||v==='') ? null : (isNaN(Number(v)) ? null : Number(v));
// telemetry.py sends -1 for "no battery" / "network unreachable". Those are
// sentinels, not measurements — render them as absent, never as -1%.
const gauged = v => { const n = num(v); return (n===null || n < 0) ? null : n; };
const lerp = (a,b,t) => a+(b-a)*t;
const clamp = (v,a,b) => Math.max(a,Math.min(b,v));
const shortId = id => String(id).slice(0,8);

function hhmmss(sec){
  sec = Math.max(0, Math.floor(sec));
  const h = String(Math.floor(sec/3600)).padStart(2,'0');
  const m = String(Math.floor(sec%3600/60)).padStart(2,'0');
  const s = String(sec%60).padStart(2,'0');
  return h+':'+m+':'+s;
}
function rgba(hex,a){
  const n = parseInt(hex.slice(1),16);
  return 'rgba('+(n>>16&255)+','+(n>>8&255)+','+(n&255)+','+a+')';
}

/* ══ canvas sizing ══════════════════════════════════════ */
function fit(){
  const r = cv.parentElement.getBoundingClientRect();
  DPR = Math.min(window.devicePixelRatio||1, 2);
  W = r.width; H = r.height;
  cv.width = Math.round(W*DPR);
  cv.height = Math.round(H*DPR);
  ctx.setTransform(DPR,0,0,DPR,0,0);
}
window.addEventListener('resize', fit);

/* ══ layout: where each device sits ═════════════════════ */
function hub(){ return { x: W*0.5, y: H*0.5 }; }

function relayout(){
  const c = hub();
  const n = nodeOrder.length;
  if(!n) return;
  // Ellipse, not a circle: the stage is far wider than it is tall, and a
  // circle sized to min(W,H) left most of the panel empty.
  const rings = n <= 11 ? 1 : 2;
  const rx = Math.max(130, W/2 - 96);
  const ry = Math.max(86, H/2 - 62);
  const split = rings === 1 ? n : Math.ceil(n/2);

  nodeOrder.forEach((nid,i)=>{
    const o = orbs.get(nid); if(!o) return;
    const ringIdx = (rings === 1 || i < split) ? 0 : 1;
    const idxIn = ringIdx === 0 ? i : i - split;
    const cnt = Math.max(1, ringIdx === 0 ? split : (n - split));
    const f = ringIdx === 0 ? 1 : 0.56;
    const spin = ringIdx === 0 ? -Math.PI/2 : -Math.PI/2 + Math.PI/cnt;
    const a = spin + (idxIn/cnt) * Math.PI*2;
    o.tx = c.x + Math.cos(a)*rx*f;
    o.ty = c.y + Math.sin(a)*ry*f;
    o.curve = (i%2 ? 1 : -1) * (16 + (i%3)*6);
    o.ring = ringIdx;
  });
}

/* ══ bezier: links AND packets read from this ═══════════ */
function ctrl(o){
  const c = hub();
  const mx = (c.x+o.x)/2, my = (c.y+o.y)/2;
  const dx = o.x-c.x, dy = o.y-c.y;
  const len = Math.hypot(dx,dy) || 1;
  return { x: mx + (-dy/len)*o.curve, y: my + (dx/len)*o.curve };
}
function bez(o,t){
  const c = hub(), q = ctrl(o), u = 1-t;
  return {
    x: u*u*c.x + 2*u*t*q.x + t*t*o.x,
    y: u*u*c.y + 2*u*t*q.y + t*t*o.y
  };
}

/* ══ ingest snapshot ════════════════════════════════════ */
function sync(state){
  const nodes = state.nodes || {};
  const ids = Object.keys(nodes);

  ids.forEach(id=>{
    let o = orbs.get(id);
    const c = hub();
    if(!o){
      o = { id, x:c.x, y:c.y, tx:c.x, ty:c.y, curve:0, ring:0,
            alive:1, phase:Math.random()*Math.PI*2, born:performance.now(), pop:0 };
      orbs.set(id,o);
    }
    o.data = nodes[id];
    o.status = nodes[id].status || 'active';
    o.gone = false;
  });

  orbs.forEach((o,id)=>{ if(!(id in nodes)) o.gone = true; });

  nodeOrder = ids.slice().sort((a,b)=>{
    const pa = num(nodes[a].partition_id), pb = num(nodes[b].partition_id);
    if(pa!==null && pb!==null && pa!==pb) return pa-pb;
    return a < b ? -1 : 1;
  });
  relayout();
}

/* ══ packets ════════════════════════════════════════════ */
function emit(o, up){
  packets.push({
    nid:o.id, t: up?1:0, dir: up?-1:1,
    v: 0.0055 + Math.random()*0.0045,
    color: up ? '#35d6a4' : '#6c8cff',
    r: up ? 2.2 : 2.6
  });
  packetTotal++;
}
function stepPackets(dt){
  packets = packets.filter(p=>{
    const o = orbs.get(p.nid);
    if(!o || o.status === 'dropped' || o.gone) return false;
    p.t += p.dir * p.v * dt;
    if(p.t > 1){ o.pop = 1; return false; }
    return p.t > 0;
  });
}

/* ══ drawing ════════════════════════════════════════════ */
function drawField(now){
  const c = hub();
  const rings = [0.26,0.42,0.58,0.74];
  rings.forEach((f,i)=>{
    const r = Math.min(W,H)*f*0.72;
    ctx.beginPath();
    ctx.arc(c.x,c.y,r,0,Math.PI*2);
    ctx.strokeStyle = 'rgba(122,146,214,'+(0.05 - i*0.008)+')';
    ctx.lineWidth = 1;
    ctx.setLineDash(i%2 ? [3,9] : []);
    ctx.stroke();
    ctx.setLineDash([]);
  });
  // slow sweep
  const a = (now/9000)%(Math.PI*2);
  const R = Math.min(W,H)*0.53;
  const g = ctx.createLinearGradient(c.x,c.y,c.x+Math.cos(a)*R,c.y+Math.sin(a)*R);
  g.addColorStop(0,'rgba(108,140,255,.16)');
  g.addColorStop(1,'rgba(108,140,255,0)');
  ctx.beginPath(); ctx.moveTo(c.x,c.y);
  ctx.lineTo(c.x+Math.cos(a)*R, c.y+Math.sin(a)*R);
  ctx.strokeStyle = g; ctx.lineWidth = 2; ctx.stroke();
}

function drawLinks(now){
  const c = hub();
  nodeOrder.forEach(nid=>{
    const o = orbs.get(nid); if(!o) return;
    const col = COLORS[o.status] || COLORS.unknown;
    const q = ctrl(o);
    const dropped = o.status === 'dropped';
    const dim = o.alive;

    ctx.save();
    ctx.globalAlpha = dim;

    if(dropped){
      // severed link: two stubs + a break marker
      ctx.setLineDash([5,6]);
      ctx.strokeStyle = rgba(col,.34);
      ctx.lineWidth = 1.3;
      [[0,.36],[.64,1]].forEach(seg=>{
        ctx.beginPath();
        for(let t=seg[0]; t<=seg[1]; t+=0.02){
          const p = bez(o,t);
          t===seg[0] ? ctx.moveTo(p.x,p.y) : ctx.lineTo(p.x,p.y);
        }
        ctx.stroke();
      });
      ctx.setLineDash([]);
      const m = bez(o,.5);
      ctx.strokeStyle = rgba(col,.75); ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(m.x-4,m.y-4); ctx.lineTo(m.x+4,m.y+4);
      ctx.moveTo(m.x+4,m.y-4); ctx.lineTo(m.x-4,m.y+4);
      ctx.stroke();
      ctx.restore();
      return;
    }

    // glow base
    const grad = ctx.createLinearGradient(c.x,c.y,o.x,o.y);
    grad.addColorStop(0,'rgba(108,140,255,.30)');
    grad.addColorStop(1,rgba(col,.55));

    ctx.beginPath();
    ctx.moveTo(c.x,c.y);
    ctx.quadraticCurveTo(q.x,q.y,o.x,o.y);
    ctx.strokeStyle = rgba(col,.07);
    ctx.lineWidth = 6; ctx.lineCap = 'round';
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(c.x,c.y);
    ctx.quadraticCurveTo(q.x,q.y,o.x,o.y);
    ctx.strokeStyle = grad;
    ctx.lineWidth = 1.4;
    ctx.stroke();

    if(o.status === 'training'){
      ctx.beginPath();
      ctx.moveTo(c.x,c.y);
      ctx.quadraticCurveTo(q.x,q.y,o.x,o.y);
      ctx.setLineDash([2,12]);
      ctx.lineDashOffset = -now/38;
      ctx.strokeStyle = 'rgba(255,255,255,.34)';
      ctx.lineWidth = 1.1;
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.restore();
  });
}

function drawTakeovers(now){
  const nodes = snapshot ? (snapshot.nodes||{}) : {};
  nodeOrder.forEach(nid=>{
    const o = orbs.get(nid); if(!o || !o.data) return;
    const cover = o.data.covering_partitions;
    if(!cover || !cover.length) return;

    cover.forEach(pid=>{
      const victimId = nodeOrder.find(x=>{
        const d = nodes[x];
        return d && d.status === 'dropped' && num(d.partition_id) === num(pid);
      });
      if(!victimId) return;
      const v = orbs.get(victimId); if(!v) return;

      const mx = (o.x+v.x)/2, my = (o.y+v.y)/2;
      const dx = v.x-o.x, dy = v.y-o.y, L = Math.hypot(dx,dy)||1;
      const qx = mx + (-dy/L)*40, qy = my + (dx/L)*40;

      ctx.save();
      ctx.setLineDash([6,5]);
      ctx.lineDashOffset = -now/26;
      ctx.strokeStyle = 'rgba(245,181,68,.85)';
      ctx.lineWidth = 1.9;
      ctx.beginPath();
      ctx.moveTo(v.x,v.y);
      ctx.quadraticCurveTo(qx,qy,o.x,o.y);
      ctx.stroke();
      ctx.setLineDash([]);

      // arrowhead pointing at the covering node
      const ang = Math.atan2(o.y-qy, o.x-qx);
      ctx.translate(o.x - Math.cos(ang)*26, o.y - Math.sin(ang)*26);
      ctx.rotate(ang);
      ctx.fillStyle = '#f5b544';
      ctx.beginPath();
      ctx.moveTo(0,0); ctx.lineTo(-8,4); ctx.lineTo(-8,-4);
      ctx.closePath(); ctx.fill();
      ctx.restore();

      ctx.save();
      ctx.font = '700 10px JetBrains Mono, monospace';
      ctx.fillStyle = '#f5b544';
      ctx.textAlign = 'center';
      ctx.fillText('P'+pid+' takeover', qx, qy-7);
      ctx.restore();
    });
  });
}

function drawPackets(){
  packets.forEach(p=>{
    const o = orbs.get(p.nid); if(!o) return;
    const pt = bez(o, clamp(p.t,0,1));
    const fade = p.dir>0 ? clamp(1-p.t*0.35,0,1) : clamp(p.t*1.1,0,1);
    ctx.beginPath();
    ctx.arc(pt.x,pt.y,p.r*3.2,0,Math.PI*2);
    ctx.fillStyle = rgba(p.color,.09*fade);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(pt.x,pt.y,p.r,0,Math.PI*2);
    ctx.fillStyle = rgba(p.color,.95*fade);
    ctx.fill();
  });
}

function drawHub(now){
  const c = hub();
  const pulse = 1 + Math.sin(now/900)*0.03;

  const glow = ctx.createRadialGradient(c.x,c.y,4,c.x,c.y,84);
  glow.addColorStop(0,'rgba(108,140,255,.22)');
  glow.addColorStop(1,'rgba(108,140,255,0)');
  ctx.beginPath(); ctx.arc(c.x,c.y,84,0,Math.PI*2);
  ctx.fillStyle = glow; ctx.fill();

  [[46,now/2600,'rgba(108,140,255,.5)',1.4,1.5],
   [56,-now/4200,'rgba(167,139,250,.34)',1.1,2.4],
   [66,now/6400,'rgba(56,214,238,.2)',1,3.6]].forEach(cfg=>{
    ctx.beginPath();
    ctx.arc(c.x,c.y,cfg[0]*pulse,cfg[1]%(Math.PI*2),cfg[1]%(Math.PI*2)+Math.PI/cfg[4]);
    ctx.strokeStyle = cfg[2]; ctx.lineWidth = cfg[3]; ctx.lineCap='round';
    ctx.stroke();
  });

  ctx.beginPath(); ctx.arc(c.x,c.y,38*pulse,0,Math.PI*2);
  ctx.strokeStyle = 'rgba(122,146,214,.16)'; ctx.lineWidth = 1; ctx.stroke();

  const body = ctx.createLinearGradient(c.x-30,c.y-30,c.x+30,c.y+30);
  body.addColorStop(0,'#151b31'); body.addColorStop(1,'#0a0e1c');
  ctx.beginPath(); ctx.arc(c.x,c.y,30,0,Math.PI*2);
  ctx.fillStyle = body; ctx.fill();
  ctx.strokeStyle = 'rgba(108,140,255,.45)'; ctx.lineWidth = 1.6; ctx.stroke();

  ctx.textAlign='center';
  ctx.font = '700 11px "Space Grotesk", sans-serif';
  ctx.fillStyle = '#dfe6ff';
  ctx.fillText('AGGR', c.x, c.y-1);
  ctx.font = '500 8.5px "JetBrains Mono", monospace';
  ctx.fillStyle = '#6c8cff';
  ctx.fillText('FedAvg', c.x, c.y+11);

}

function arcRing(x,y,r,frac,color,width){
  ctx.beginPath();
  ctx.arc(x,y,r,-Math.PI/2,-Math.PI/2 + Math.PI*2*clamp(frac,0,1));
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap='round';
  ctx.stroke();
}

function drawNodes(now){
  nodeOrder.forEach(nid=>{
    const o = orbs.get(nid); if(!o) return;
    const d = o.data || {};
    const col = COLORS[o.status] || COLORS.unknown;
    const R = 20;
    const isHover = hover === nid;

    ctx.save();
    ctx.globalAlpha = o.alive;

    if(o.status === 'training'){
      const ripple = (now/1100)%1;
      ctx.beginPath();
      ctx.arc(o.x,o.y,R + ripple*18,0,Math.PI*2);
      ctx.strokeStyle = rgba(col,.28*(1-ripple));
      ctx.lineWidth = 1.6; ctx.stroke();
    }
    if(o.status === 'dropped'){
      const ripple = (now/900)%1;
      ctx.beginPath();
      ctx.arc(o.x,o.y,R + ripple*14,0,Math.PI*2);
      ctx.strokeStyle = rgba(col,.34*(1-ripple));
      ctx.lineWidth = 1.4; ctx.stroke();
    }
    if(o.pop > 0){
      ctx.beginPath();
      ctx.arc(o.x,o.y,R + (1-o.pop)*16,0,Math.PI*2);
      ctx.strokeStyle = rgba(col,.5*o.pop); ctx.lineWidth = 1.6; ctx.stroke();
      o.pop = Math.max(0, o.pop - 0.03);
    }

    const halo = ctx.createRadialGradient(o.x,o.y,2,o.x,o.y,R*2.1);
    halo.addColorStop(0, rgba(col,.20));
    halo.addColorStop(1, rgba(col,0));
    ctx.beginPath(); ctx.arc(o.x,o.y,R*2.1,0,Math.PI*2);
    ctx.fillStyle = halo; ctx.fill();

    const body = ctx.createLinearGradient(o.x-R,o.y-R,o.x+R,o.y+R);
    body.addColorStop(0,'#161c30'); body.addColorStop(1,'#0b1020');
    ctx.beginPath(); ctx.arc(o.x,o.y,R,0,Math.PI*2);
    ctx.fillStyle = body; ctx.fill();
    ctx.strokeStyle = rgba(col, isHover ? .95 : .6);
    ctx.lineWidth = isHover ? 2 : 1.4; ctx.stroke();

    // outer dials: battery (left arc) and cpu (thin outer)
    const batt = gauged(d.battery_percent);
    const cpu = num(d.cpu_percent);
    if(batt !== null){
      const bc = batt < 25 ? '#ff5d73' : batt < 50 ? '#f5b544' : '#35d6a4';
      ctx.beginPath();
      ctx.arc(o.x,o.y,R+5,0,Math.PI*2);
      ctx.strokeStyle = 'rgba(122,146,214,.12)'; ctx.lineWidth = 2.4; ctx.stroke();
      arcRing(o.x,o.y,R+5, batt/100, rgba(bc,.9), 2.4);
    }
    if(cpu !== null){
      arcRing(o.x,o.y,R+9.5, cpu/100, 'rgba(108,140,255,.55)', 1.4);
    }

    // partition glyph inside
    ctx.textAlign = 'center';
    ctx.font = '700 12px "JetBrains Mono", monospace';
    ctx.fillStyle = '#dfe6ff';
    ctx.fillText(d.partition_id !== null && d.partition_id !== undefined ? 'P'+d.partition_id : '··', o.x, o.y+4);

    // caption
    ctx.font = '500 9.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#7f89ad';
    ctx.fillText(shortId(nid), o.x, o.y+R+28);
    ctx.font = '700 8px Inter, sans-serif';
    ctx.fillStyle = rgba(col,.9);
    const sub = o.status === 'dropped' && d.missing_rounds
      ? 'MISSING '+d.missing_rounds+'R' : (LABEL[o.status]||o.status).toUpperCase();
    ctx.fillText(sub, o.x, o.y+R+39);

    if(d.covering_partitions && d.covering_partitions.length){
      ctx.font = '700 8.5px "JetBrains Mono", monospace';
      ctx.fillStyle = '#f5b544';
      ctx.fillText('+P'+d.covering_partitions.join(' +P'), o.x, o.y-R-16);
    }
    ctx.restore();
  });
}

/* ══ hover / tooltip ════════════════════════════════════ */
cv.addEventListener('mousemove', e=>{
  const r = cv.getBoundingClientRect();
  const mx = e.clientX-r.left, my = e.clientY-r.top;
  let found = null;
  nodeOrder.forEach(nid=>{
    const o = orbs.get(nid); if(!o) return;
    if(Math.hypot(mx-o.x,my-o.y) < 26) found = nid;
  });
  hover = found;
  if(!found){ tip.style.opacity = 0; return; }
  const o = orbs.get(found), d = o.data || {};
  const batt = gauged(d.battery_percent), cpu = num(d.cpu_percent),
        lat = gauged(d.network_latency_ms);
  tip.innerHTML =
    '<div class="tip-t" style="color:'+(COLORS[o.status]||COLORS.unknown)+'">'+shortId(found)+' · '+(LABEL[o.status]||o.status)+'</div>'+
    row('Partition', d.partition_id!=null?('P'+d.partition_id):'—')+
    row('Score', num(d.score)!==null?num(d.score).toFixed(3):'—')+
    row('Local epochs', d.local_epochs!=null?d.local_epochs:'—')+
    row('Examples', d.num_examples!=null?Number(d.num_examples).toLocaleString():'—')+
    row('Battery', batt!==null?batt.toFixed(0)+'%'+(d.battery_plugged_in?' (charging)':''):'—')+
    row('CPU', cpu!==null?cpu.toFixed(1)+'%':'—')+
    row('Memory', num(d.mem_percent_used)!==null?num(d.mem_percent_used).toFixed(1)+'%':'—')+
    row('Latency', lat!==null?lat.toFixed(0)+' ms':'unreachable')+
    row('Last round', d.last_round!=null?d.last_round:'—')+
    (d.missing_rounds ? row('Missing for', d.missing_rounds+' round(s)') : '')+
    parts(d.score_parts)+
    (d.score_reason ? '<div class="tip-why">'+d.score_reason+'</div>' : '');
  tip.style.left = o.x+'px';
  tip.style.top = (o.y-26)+'px';
  tip.style.opacity = 1;
});
cv.addEventListener('mouseleave', ()=>{ hover=null; tip.style.opacity=0; });
function row(k,v){ return '<div class="tip-r"><span>'+k+'</span><b>'+v+'</b></div>'; }
// Show WHY a device was throttled, not just that it was.
function parts(p){
  if(!p) return '';
  const keys = ['battery','cpu','memory','network'].filter(k=>p[k]!=null);
  if(!keys.length) return '';
  return '<div class="tip-parts">'+keys.map(k=>{
    const v = clamp(p[k],0,1);
    const col = v>=.8?'#35d6a4':v>=.5?'#f5b544':'#ff5d73';
    return '<div class="tip-part"><span>'+k.slice(0,3)+'</span>'+
           '<i><b style="width:'+(v*100)+'%;background:'+col+'"></b></i></div>';
  }).join('')+'</div>';
}

/* ══ fleet rail (DOM, reconciled) ═══════════════════════ */
function bar(k, val, max, cls, suffix, digits){
  const has = val !== null;
  const pct = has ? clamp(val/max*100,0,100) : 0;
  const txt = has ? val.toFixed(digits)+suffix : '—';
  return '<div class="gauge"><span class="gauge-k">'+k+'</span>'+
         '<span class="gauge-t"><span class="gauge-f '+(has?cls:'g-null')+'" style="width:'+pct+'%"></span></span>'+
         '<span class="gauge-v">'+txt+'</span></div>';
}
function unitHTML(d){
  const st = d.status || 'active';
  const batt = gauged(d.battery_percent);
  const battCls = batt===null?'g-null':batt<25?'g-coral':batt<50?'g-amber':'g-mint';
  const cover = (d.covering_partitions && d.covering_partitions.length)
    ? '<span class="cover">+P'+d.covering_partitions.join(' +P')+'</span>' : '';
  return '<div class="unit-top">'+
      '<span class="uid">'+shortId(d.id)+'</span>'+
      '<span class="ustat" data-s="'+st+'">'+(LABEL[st]||st)+'</span>'+
      '<span class="upart">'+(d.partition_id!=null?'P'+d.partition_id:'P—')+'</span>'+
    '</div>'+
    '<div class="gauges">'+
      bar('BAT', batt, 100, battCls, '%', 0)+
      bar('CPU', num(d.cpu_percent), 100, 'g-sky', '%', 0)+
      bar('MEM', gauged(d.mem_percent_used), 100, 'g-iris', '%', 0)+
      bar('NET', gauged(d.network_latency_ms), 500, 'g-amber', 'ms', 0)+
    '</div>'+
    '<div class="unit-foot">'+
      '<span>score <b>'+(num(d.score)!==null?num(d.score).toFixed(2):'—')+'</b></span>'+
      '<span>ep <b>'+(d.local_epochs!=null?d.local_epochs:'—')+'</b></span>'+
      '<span>ex <b>'+(d.num_examples!=null?Number(d.num_examples).toLocaleString():'—')+'</b></span>'+
      cover+
      (d.score_reason && d.score_reason !== 'conditions healthy'
        ? '<span class="why">'+d.score_reason+'</span>' : '')+
    '</div>';
}
function renderFleet(){
  const nodes = snapshot ? (snapshot.nodes||{}) : {};
  if(!nodeOrder.length){
    if(!fleetEl.querySelector('.empty-rail'))
      fleetEl.innerHTML = '<div class="empty-rail">Waiting for devices to register with the coordinator.</div>';
    units.clear();
    return;
  }
  const stray = fleetEl.querySelector('.empty-rail');
  if(stray) stray.remove();

  units.forEach((el,id)=>{
    if(!(id in nodes) && !el.classList.contains('leaving')){
      el.classList.add('leaving');
      el.addEventListener('animationend', ()=>{ el.remove(); units.delete(id); }, {once:true});
    }
  });

  let prev = null;
  nodeOrder.forEach(id=>{
    const d = Object.assign({id}, nodes[id]);
    let el = units.get(id);
    if(!el){
      el = document.createElement('div');
      el.className = 'unit';
      units.set(id, el);
      el.dataset.nid = id;
      el.innerHTML = unitHTML(d);
      el.dataset.s = d.status || 'active';
      fleetEl.insertBefore(el, prev ? prev.nextSibling : fleetEl.firstChild);
    } else {
      el.innerHTML = unitHTML(d);
      el.dataset.s = d.status || 'active';
      const want = prev ? prev.nextSibling : fleetEl.firstChild;
      if(want !== el) fleetEl.insertBefore(el, want);
    }
    prev = el;
  });
}

/* ══ event log ══════════════════════════════════════════ */
const trimIds = t => String(t).replace(/\b[0-9a-f]{10,}\b/gi, m => m.slice(0,8)+'…')
                              .replace(/\b\d{10,}\b/g, m => m.slice(0,8)+'…');
const keyOf = e => (e.timestamp||'')+'|'+(e.message||'');
const TAGS = { drop:'DROP', reassign:'COVER', reconnect:'REJOIN', skip:'SKIP' };

function renderLog(events){
  if(!events || !events.length) return;
  let cut = 0;
  if(lastEventKey){
    cut = events.length;
    for(let i=events.length-1;i>=0;i--){
      if(keyOf(events[i]) === lastEventKey){ cut = i+1; break; }
      if(i===0) cut = 0;
    }
  }
  const fresh = lastEventKey ? events.slice(cut) : events;
  if(!fresh.length) return;

  const stray = logEl.querySelector('.empty-rail');
  if(stray) stray.remove();

  fresh.slice().reverse().forEach(e=>{
    const ts = e.timestamp || Date.now()/1000;
    const time = new Date(ts*1000).toLocaleTimeString('en-GB',{hour12:false});
    const div = document.createElement('div');
    div.className = 'row ' + (e.type||'');
    div.innerHTML = '<span class="r-t">'+time+'</span>'+
                    '<span class="r-g '+(e.type||'')+'">'+(TAGS[e.type]||'INFO')+'</span>'+
                    '<span class="r-m">'+trimIds(e.message||'')+'</span>';
    logEl.insertBefore(div, logEl.firstChild);
  });
  while(logEl.children.length > MAX_LOG) logEl.removeChild(logEl.lastChild);
  lastEventKey = keyOf(events[events.length-1]);
}

/* ══ metric strip ═══════════════════════════════════════ */
function applyStats(state){
  const s = state.summary || {};
  const sc = s.status_counts || {};
  const nodes = state.nodes || {};
  const total = Object.keys(nodes).length;

  const run = state.run || {};
  $('h-strategy').textContent = run.mode || 'adaptive';
  $('h-dataset').textContent = run.dataset || '—';
  $('h-model').textContent = run.model || '—';

  // current_round is the last COMPLETED round; the headline number should
  // be whatever is actually training right now, or it disagrees with the
  // subtext and with the server log.
  const done = state.current_round || 0;
  const live = s.round_in_progress;
  const shown = live != null ? live : done;
  const tr = state.total_rounds;
  $('m-round').innerHTML = shown + (tr ? '<small> / '+tr+'</small>' : '');
  const finished = !!(tr && done >= tr && live == null);
  if(finished !== runComplete){ runComplete = finished; renderWire(); }
  $('m-round-n').textContent = live != null
    ? 'training · ' + done + ' aggregated'
    : finished ? 'all rounds complete'
    : (done ? 'idle between rounds' : 'waiting for scheduler');

  ease.acc.t   = state.global_accuracy != null ? state.global_accuracy*100 : ease.acc.t;
  ease.loss.t  = state.global_loss != null ? state.global_loss : ease.loss.t;
  ease.ex.t    = s.total_examples_this_round ?? ease.ex.t;
  ease.prog.t  = s.round_progress_pct != null ? s.round_progress_pct : 0;
  ease.nodes.t = s.connected_nodes ?? 0;
  ease.drop.t  = sc.dropped ?? 0;
  ease.lat.t   = s.median_latency_ms != null ? s.median_latency_ms : ease.lat.t;

  const hist = state.history || [];
  if(hist.length > 1){
    const last = hist[hist.length-1], prev = hist[hist.length-2];
    if(last.accuracy != null && prev.accuracy != null){
      const d = (last.accuracy - prev.accuracy)*100;
      $('m-acc-n').textContent = (d>=0?'▲ +':'▼ ')+d.toFixed(2)+' pts vs previous round';
    }
    if(last.loss != null && prev.loss != null){
      const d = last.loss - prev.loss;
      $('m-loss-n').textContent = (d<=0?'▼ ':'▲ +')+Math.abs(d).toFixed(4)+' vs previous round';
    }
  }

  $('m-nodes-n').textContent = (sc.training||0)+' training · '+(sc.active||0)+' ready · '+(sc.skipped||0)+' skipped';
  $('m-drop-n').textContent = (sc.dropped||0)
    ? (s.covering_nodes||0)+' device(s) covering the gap'
    : 'all links healthy';
  $('m-ex-n').textContent = total ? 'across '+(s.connected_nodes||0)+' of '+total+' devices' : 'no devices reporting';
  $('m-lat-n').textContent = s.avg_battery != null ? 'battery avg '+s.avg_battery.toFixed(0)+'%' : 'battery avg —';
  $('fleet-note').textContent = total ? total+' devices' : '—';

  if(s.elapsed_seconds != null){ clockBase = s.elapsed_seconds; clockAt = performance.now(); }

  // partition takeover toast
  let msg = '';
  const entries = Object.entries(nodes);
  const cover = entries.find(e => e[1].covering_partitions && e[1].covering_partitions.length);
  const gone = entries.find(e => e[1].status === 'dropped');
  if(cover && gone){
    // Node records are keyed by device id and carry no `id` field — reading
    // covering.id gave an empty name every time ("reassigned to —").
    const name = shortId(cover[1].device_id != null ? cover[1].device_id : cover[0]);
    const factor = cover[1].covering_partitions.length + 1;
    msg = 'Partition '+gone[1].partition_id+' → device '+name+' · training on '+factor+'× data';
  }
  if(msg && msg !== lastToast){
    toastMsg.textContent = msg;
    toast.classList.add('on');
    lastToast = msg;
    setTimeout(()=>{ if(lastToast===msg) toast.classList.remove('on'); }, 5200);
  } else if(!msg){
    toast.classList.remove('on');
    lastToast = '';
  }
}

function paintEased(){
  $('m-acc').textContent   = ease.acc.c > 0 ? ease.acc.c.toFixed(2)+'%' : '—';
  $('m-loss').textContent  = ease.loss.c > 0 ? ease.loss.c.toFixed(4) : '—';
  $('m-nodes').textContent = Math.round(ease.nodes.c);
  $('m-drop').textContent  = Math.round(ease.drop.c);
  $('m-ex').textContent    = Math.round(ease.ex.c).toLocaleString();
  $('m-lat').innerHTML     = ease.lat.c > 0 ? Math.round(ease.lat.c)+'<small> ms</small>' : '—';
  $('m-round-bar').style.width = clamp(ease.prog.c,0,100)+'%';
  $('pkt-count').textContent = Math.round(ease.nodes.c)+' linked · '
    + Math.round(ease.ex.c).toLocaleString()+' ex · '
    + packetTotal.toLocaleString()+' pkts';
  $('h-clock').textContent = hhmmss(clockBase + (performance.now()-clockAt)/1000);
}

/* ══ charts ═════════════════════════════════════════════ */
// A 0-100 axis makes a 22% accuracy curve a flat line on the floor. Round up
// to a readable ceiling instead, with a 40% floor so early rounds don't look
// like a breakthrough.
function accCeiling(values){
  const top = Math.max(...values.filter(v => v != null && !isNaN(v)), 0);
  return Math.min(100, Math.max(40, Math.ceil(top * 1.5 / 10) * 10));
}
const GRID = 'rgba(122,146,214,.06)';
const AXIS = { color:'#525b7a', font:{ size:9, family:'JetBrains Mono' } };

function initTrend(){
  const c = $('trend').getContext('2d');
  const fill = c.createLinearGradient(0,0,0,190);
  fill.addColorStop(0,'rgba(53,214,164,.28)');
  fill.addColorStop(1,'rgba(53,214,164,0)');
  trendChart = new Chart(c,{
    type:'line',
    data:{ labels:[], datasets:[
      { label:'Accuracy %', data:[], borderColor:'#35d6a4', backgroundColor:fill,
        tension:.34, fill:true, borderWidth:2.2, pointRadius:0, pointHoverRadius:4, yAxisID:'y' },
      { label:'Loss', data:[], borderColor:'#ff5d73', backgroundColor:'transparent',
        tension:.34, fill:false, borderWidth:1.8, borderDash:[4,3], pointRadius:0, pointHoverRadius:4, yAxisID:'y1' }
    ]},
    options:{
      responsive:true, maintainAspectRatio:false, animation:{duration:400},
      interaction:{ mode:'index', intersect:false },
      plugins:{ legend:{ display:true, labels:{ color:'#848eb2', boxWidth:9, boxHeight:9, usePointStyle:true, font:{size:10} } },
                tooltip:{ backgroundColor:'rgba(7,10,20,.95)', borderColor:'rgba(122,146,214,.2)', borderWidth:1,
                          titleFont:{size:11}, bodyFont:{size:11}, padding:9 } },
      scales:{
        x:{ ticks:AXIS, grid:{ color:GRID }, border:{ display:false } },
        y:{ position:'left', min:0, max:100, ticks:Object.assign({},AXIS,{color:'#35d6a4',stepSize:25}), grid:{ color:GRID }, border:{display:false} },
        y1:{ position:'right', min:0, ticks:Object.assign({},AXIS,{color:'#ff5d73'}), grid:{ display:false }, border:{display:false} }
      }
    }
  });
}
function updateTrend(history){
  if(!trendChart || !history || !history.length) return;
  $('trend-blank').style.display = 'none';
  const sig = history.length+':'+history[history.length-1].round+':'+history[history.length-1].accuracy;
  if(sig === trendSig) return;
  trendSig = sig;
  trendChart.data.labels = history.map(h=>'R'+h.round);
  trendChart.data.datasets[0].data = history.map(h=>h.accuracy==null?null:h.accuracy*100);
  trendChart.data.datasets[1].data = history.map(h=>h.loss);
  // With 1-3 rounds a line alone is an unreadable flat bar; show the points.
  const pr = history.length < 6 ? 3.5 : 0;
  trendChart.data.datasets.forEach(ds => { ds.pointRadius = pr; });
  trendChart.options.scales.y.max = accCeiling(trendChart.data.datasets[0].data);
  trendChart.update('none');
}
let compSig = '';
function loadComparison(){
  fetch('/api/comparison').then(r=>r.json()).then(data=>{
    if(!data || (!(data.series_a||[]).length && !(data.series_b||[]).length)) return;
    const spine = (data.series_a||[]).length ? data.series_a : data.series_b;
    // Rebuilding the chart on every poll destroys and re-animates it for no
    // reason; only redraw when the file actually changed.
    const sig = JSON.stringify([data.updated_at, (data.series_a||[]).length, (data.series_b||[]).length]);
    if(sig === compSig) return;
    compSig = sig;
    $('comp-blank').style.display = 'none';
    const cvs = $('comp'); cvs.style.display = 'block';
    if(compChart) compChart.destroy();
    const c = cvs.getContext('2d');
    const fill = c.createLinearGradient(0,0,0,190);
    fill.addColorStop(0,'rgba(108,140,255,.26)');
    fill.addColorStop(1,'rgba(108,140,255,0)');
    compChart = new Chart(c,{
      type:'line',
      data:{
        labels:spine.map(p=>'R'+p.round),
        datasets:[
          { label:data.label_a||'Adaptive', data:(data.series_a||[]).map(p=>p.accuracy*100),
            borderColor:'#6c8cff', backgroundColor:fill, tension:.3, fill:true, borderWidth:2.2, pointRadius:3, pointHoverRadius:5 },
          { label:data.label_b||'Baseline', data:(data.series_b||[]).map(p=>p.accuracy*100),
            borderColor:'#848eb2', backgroundColor:'transparent', tension:.3, fill:false, borderWidth:1.8, borderDash:[5,4], pointRadius:3, pointHoverRadius:5 }
        ]
      },
      options:{
        responsive:true, maintainAspectRatio:false, animation:{duration:400},
        interaction:{ mode:'index', intersect:false },
        plugins:{ legend:{ labels:{ color:'#848eb2', boxWidth:9, boxHeight:9, usePointStyle:true, font:{size:10} } },
                  tooltip:{ backgroundColor:'rgba(7,10,20,.95)', borderColor:'rgba(122,146,214,.2)', borderWidth:1, padding:9 } },
        scales:{
          x:{ ticks:AXIS, grid:{ display:false }, border:{display:false} },
          y:{ min:0, max:accCeiling([...(data.series_a||[]).map(p=>p.accuracy*100),
                                     ...(data.series_b||[]).map(p=>p.accuracy*100)]),
              ticks:AXIS, grid:{ color:GRID }, border:{display:false} }
        }
      }
    });
  }).catch(()=>{});
}

/* ══ frame loop ═════════════════════════════════════════ */
let last = performance.now();
function frame(now){
  const dt = Math.min(64, now-last); last = now;

  // spring positions + fade
  orbs.forEach((o,id)=>{
    o.x = lerp(o.x, o.tx, 0.085);
    o.y = lerp(o.y, o.ty, 0.085);
    if(o.gone){
      o.alive = Math.max(0, o.alive - 0.05);
      if(o.alive <= 0) orbs.delete(id);
    } else if(o.alive < 1){
      o.alive = Math.min(1, o.alive + 0.06);
    }
  });

  // emit packets from healthy links
  if(nodeOrder.length){
    nodeOrder.forEach(nid=>{
      const o = orbs.get(nid); if(!o) return;
      if(o.status === 'dropped' || o.status === 'skipped') return;
      const rate = o.status === 'training' ? 0.055 : 0.014;
      if(Math.random() < rate) emit(o, false);
      if(Math.random() < rate*0.7) emit(o, true);
    });
  }
  stepPackets(dt/16.6);

  for(const k in ease){
    const e = ease[k];
    e.c = Math.abs(e.t-e.c) < 0.01 ? e.t : lerp(e.c, e.t, 0.12);
  }

  ctx.clearRect(0,0,W,H);
  drawField(now);
  drawLinks(now);
  drawPackets();
  drawTakeovers(now);
  drawHub(now);
  drawNodes(now);
  paintEased();

  requestAnimationFrame(frame);
}

/* ══ polling ════════════════════════════════════════════ */
function renderWire(){
  const el = $('h-wire');
  el.classList.toggle('down', !online);
  el.classList.toggle('done', online && runComplete);
  $('h-wire-t').textContent = !online ? 'RECONNECTING'
                            : runComplete ? 'RUN COMPLETE' : 'LIVE';
}
function setWire(up){
  if(up === online) return;
  online = up;
  renderWire();
}
function poll(){
  fetch('/api/state').then(r=>{
    if(!r.ok) throw new Error('bad status');
    return r.json();
  }).then(state=>{
    failures = 0; setWire(true);
    snapshot = state;
    sync(state);
    applyStats(state);
    renderFleet();
    renderLog(state.events || []);
    updateTrend(state.history || []);
  }).catch(()=>{
    if(++failures >= 2) setWire(false);
  });
}

/* ══ boot ═══════════════════════════════════════════════ */
fit();
initTrend();
clockAt = performance.now();
poll();
loadComparison();
setInterval(poll, POLL_MS);
setInterval(loadComparison, 15000);
requestAnimationFrame(frame);
})();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    # Served verbatim — no Jinja pass, so braces in the CSS/JS are safe.
    return Response(PAGE, mimetype="text/html")


@app.route("/api/state")
def api_state():
    return jsonify(_state_payload())


@app.route("/api/comparison")
def api_comparison():
    return jsonify(_load_comparison())


if __name__ == "__main__":
    print("HALO Dashboard v8 — http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)