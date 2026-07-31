"""
HALO Live Dashboard v5 — Final Production Build
------------------------------------------------
Run: python dashboard_app.py
Open: http://127.0.0.1:5000
"""

import sys
import os
import json
import time
sys.path.append(os.path.join(os.path.dirname(__file__), "scheduler"))

from flask import Flask, jsonify, render_template_string
from dashboard_state import load_state

app = Flask(__name__)

COMPARISON_PATH = os.path.join(os.path.dirname(__file__), "comparison_runs.json")


def _load_comparison():
    if not os.path.exists(COMPARISON_PATH):
        return None
    try:
        with open(COMPARISON_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _derive_summary(state: dict) -> dict:
    nodes = state.get("nodes", {})
    current_round = state.get("current_round", 0)
    counts = {"training": 0, "active": 0, "skipped": 0, "dropped": 0, "reconnected": 0, "retired": 0}
    total_examples = 0

    for n in nodes.values():
        st = n.get("status")
        if st in counts:
            counts[st] += 1
        if st not in ("retired", "dropped"):
            ex = n.get("num_examples") or n.get("examples") or 0
            try:
                total_examples += int(float(ex))
            except (ValueError, TypeError):
                pass

    started_at = state.get("started_at")
    elapsed = (time.time() - started_at) if started_at else 0
    connected = sum(1 for n in nodes.values() if n.get("status") not in ("retired", "dropped"))

    progress_pct = None
    round_in_progress = state.get("round_in_progress")
    if round_in_progress is not None and state.get("round_started_at"):
        elapsed_this_round = time.time() - state["round_started_at"]
        est = state.get("avg_round_duration")
        if est and est > 0:
            progress_pct = min(97, round(elapsed_this_round / est * 100))

    return {
        "status_counts": counts,
        "total_examples_this_round": total_examples,
        "elapsed_seconds": elapsed,
        "connected_nodes": connected,
        "round_in_progress": round_in_progress,
        "round_progress_pct": progress_pct,
    }


def _state_payload():
    state = load_state()
    state["summary"] = _derive_summary(state)
    state["nodes"] = {
        nid: n for nid, n in state["nodes"].items() if n.get("status") != "retired"
    }
    return state


PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HALO — Live Scheduler Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

  :root {
    --bg: #030408;
    --panel: #0e111a;
    --panel-2: #161a26;
    --border: rgba(60,70,100,0.25);
    --text: #e8ecf5;
    --muted: #6b7280;
    --accent: #5aa6ff;
    --green: #3ecf8e;
    --amber: #f0b429;
    --red: #f2545b;
    --cyan: #4fd6e0;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  html, body {
    width: 100%; height: 100%; overflow: hidden;
    background: var(--bg); color: var(--text);
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  }

  .bg-depth {
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background:
      radial-gradient(ellipse at 15% 50%, rgba(90,166,255,0.04) 0%, transparent 50%),
      radial-gradient(ellipse at 85% 20%, rgba(62,207,142,0.025) 0%, transparent 50%),
      radial-gradient(circle at 50% 50%, rgba(90,166,255,0.015) 0%, transparent 70%);
  }
  .bg-grid {
    position: fixed; inset: 0; pointer-events: none; z-index: 0; opacity: 0.3;
    background-image:
      linear-gradient(rgba(90,166,255,0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(90,166,255,0.04) 1px, transparent 1px);
    background-size: 50px 50px;
  }

  /* Top Bar */
  .topbar {
    position: relative; z-index: 200; height: 48px;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 24px;
    background: rgba(3,4,8,0.95); backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border);
  }
  .brand { display: flex; align-items: center; gap: 10px; }
  .brand-icon {
    width: 30px; height: 30px; border-radius: 8px;
    background: linear-gradient(135deg, var(--accent), var(--cyan));
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 14px; color: #000;
  }
  .brand-text { font-size: 14px; font-weight: 700; letter-spacing: 0.5px; }
  .brand-sub { font-size: 9px; color: var(--muted); text-transform: uppercase; letter-spacing: 1.2px; }

  .stats-bar { display: flex; gap: 28px; align-items: center; }
  .stat-item { display: flex; align-items: baseline; gap: 6px; }
  .stat-label { font-size: 9px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
  .stat-value { font-size: 18px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .stat-value.accent { color: var(--accent); }
  .stat-value.green { color: var(--green); }
  .stat-value.red { color: var(--red); }
  .stat-value.amber { color: var(--amber); }

  .live-badge {
    display: flex; align-items: center; gap: 6px;
    background: rgba(62,207,142,0.08); border: 1px solid rgba(62,207,142,0.2);
    padding: 4px 12px; border-radius: 999px;
  }
  .live-dot {
    width: 6px; height: 6px; border-radius: 50%; background: var(--green);
    box-shadow: 0 0 8px rgba(62,207,142,0.6);
    animation: pulse-dot 1.4s ease infinite;
  }
  .live-text { font-size: 10px; color: var(--green); font-weight: 700; letter-spacing: 0.8px; }

  /* App Shell */
  .app-shell {
    display: grid;
    grid-template-columns: 1fr 340px;
    grid-template-rows: 1fr 140px;
    grid-template-areas: "main sidebar" "bottom bottom";
    width: 100%; height: calc(100vh - 48px);
  }

  /* Main Stage — Flex layout for vertical stack */
  .main-stage {
    grid-area: main; position: relative; overflow: hidden;
    display: flex;
    align-items: center;
    padding: 0 40px;
    gap: 50px;
  }

  /* Coordinator Hub */
  .coord-hub {
    position: relative; width: 160px; height: 160px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    z-index: 10;
  }
  .coord-ring {
    position: absolute; border-radius: 50%;
    border: 1.5px solid rgba(90,166,255,0.12);
  }
  .coord-ring.r1 { inset: -10px; border-top-color: rgba(90,166,255,0.5); animation: spin 8s linear infinite; }
  .coord-ring.r2 { inset: -28px; border: 1.5px dashed rgba(90,166,255,0.1); animation: spin-reverse 16s linear infinite; }
  .coord-ring.r3 { inset: -46px; border: 1.5px solid rgba(90,166,255,0.06); border-bottom-color: rgba(90,166,255,0.25); animation: spin 24s linear infinite; }
  .coord-glow {
    position: absolute; inset: -20px; border-radius: 50%;
    background: radial-gradient(circle, rgba(90,166,255,0.18) 0%, transparent 60%);
    animation: breathe 3s ease infinite;
  }
  .coord-body {
    position: relative; width: 90px; height: 90px; border-radius: 50%;
    background: linear-gradient(145deg, #0d1117, #1a1f2e);
    border: 2.5px solid rgba(90,166,255,0.35);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    box-shadow: 0 0 60px rgba(90,166,255,0.12), inset 0 0 30px rgba(90,166,255,0.04);
  }
  .coord-icon { font-size: 28px; }
  .coord-label { font-size: 9px; font-weight: 800; color: var(--accent); margin-top: 3px; letter-spacing: 1.5px; }
  .coord-packets {
    position: absolute; bottom: -22px; left: 50%; transform: translateX(-50%);
    font-size: 9px; color: var(--muted); white-space: nowrap;
    font-family: 'JetBrains Mono', monospace;
  }

  /* SVG Layer — BEHIND cards and server */
  .conn-layer {
    position: absolute; inset: 0; z-index: 5; pointer-events: none;
    width: 100%; height: 100%; overflow: visible;
  }

  /* Nodes Grid */
  .nodes-wrap {
    position: relative; z-index: 12;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
    gap: 18px;
    align-content: center;
    max-height: 100%;
    overflow-y: auto;
    padding-right: 10px;
    flex: 1;
  }
  .nodes-wrap.vertical-stack {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    gap: 24px;
    max-width: 400px;
  }

  /* Node Card */
  .node-card {
    position: relative;
    background: linear-gradient(160deg, rgba(22,26,38,0.98), rgba(12,14,22,0.98));
    border-radius: 16px; padding: 16px;
    backdrop-filter: blur(16px);
    transition: transform 0.25s ease;
    animation: card-enter 0.5s ease both;
    box-shadow:
      0 0 0 1px rgba(255,255,255,0.06),
      0 8px 32px rgba(0,0,0,0.5);
  }
  .node-card:hover { transform: translateY(-3px); }

  .node-card.status-active {
    border: 1.5px solid rgba(62,207,142,0.35);
    box-shadow:
      0 0 0 1px rgba(62,207,142,0.1),
      0 8px 32px rgba(0,0,0,0.5),
      0 0 40px rgba(62,207,142,0.08);
  }
  .node-card.status-training {
    border: 1.5px solid rgba(79,214,224,0.4);
    box-shadow:
      0 0 0 1px rgba(79,214,224,0.12),
      0 8px 32px rgba(0,0,0,0.5),
      0 0 50px rgba(79,214,224,0.12);
    animation: card-enter 0.5s ease both, training-pulse 3s ease infinite;
  }
  .node-card.status-skipped {
    border: 1.5px solid rgba(240,180,41,0.35);
    box-shadow:
      0 0 0 1px rgba(240,180,41,0.1),
      0 8px 32px rgba(0,0,0,0.5),
      0 0 40px rgba(240,180,41,0.06);
  }
  .node-card.status-dropped {
    border: 1.5px solid rgba(242,84,91,0.5);
    background: linear-gradient(160deg, rgba(30,15,15,0.95), rgba(12,14,22,0.95));
    box-shadow:
      0 0 0 1px rgba(242,84,91,0.15),
      0 8px 32px rgba(0,0,0,0.5),
      0 0 50px rgba(242,84,91,0.12);
  }
  .node-card.status-dropped::before {
    content: ''; position: absolute; inset: -3px; border-radius: 18px;
    border: 2px solid rgba(242,84,91,0.3); pointer-events: none;
    animation: alarm-ping 1.3s ease infinite;
  }
  .node-card.status-dropped::after {
    content: '✕'; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    font-size: 48px; color: rgba(242,84,91,0.08); font-weight: 800; pointer-events: none;
  }
  .node-card.status-reconnected {
    border: 1.5px solid rgba(79,214,224,0.3);
    box-shadow:
      0 0 0 1px rgba(79,214,224,0.1),
      0 8px 32px rgba(0,0,0,0.5),
      0 0 40px rgba(79,214,224,0.06);
  }

  /* Connection port dot on card edge */
  .node-port {
    position: absolute;
    left: -3px;
    top: 50%;
    transform: translateY(-50%);
    width: 5px;
    height: 5px;
    border-radius: 50%;
    z-index: 15;
    pointer-events: none;
  }

  .node-head { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
  .node-avatar {
    width: 38px; height: 38px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center; font-size: 20px;
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
  }
  .node-title { flex: 1; min-width: 0; }
  .node-name { font-size: 12px; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .node-status-row { display: flex; align-items: center; gap: 5px; margin-top: 2px; }
  .status-dot { width: 6px; height: 6px; border-radius: 50%; }
  .status-dot.active { background: var(--green); }
  .status-dot.training { background: var(--cyan); box-shadow: 0 0 8px var(--cyan); animation: pulse-dot 1.5s infinite; }
  .status-dot.skipped { background: var(--amber); }
  .status-dot.dropped { background: var(--red); box-shadow: 0 0 8px var(--red); animation: pulse-dot 1s infinite; }
  .status-dot.reconnected { background: var(--cyan); }
  .node-status-text { font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; }
  .node-status-text.active { color: var(--green); }
  .node-status-text.training { color: var(--cyan); }
  .node-status-text.skipped { color: var(--amber); }
  .node-status-text.dropped { color: var(--red); }
  .node-status-text.reconnected { color: var(--cyan); }

  .node-badges { display: flex; gap: 5px; margin-bottom: 10px; flex-wrap: wrap; }
  .nbadge {
    font-size: 9px; font-weight: 700; padding: 3px 8px; border-radius: 6px;
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    color: var(--muted);
  }
  .nbadge.part { color: var(--accent); border-color: rgba(90,166,255,0.2); background: rgba(90,166,255,0.08); }
  .nbadge.score { color: var(--green); border-color: rgba(62,207,142,0.2); background: rgba(62,207,142,0.08); }
  .nbadge.epoch { color: var(--amber); border-color: rgba(240,180,41,0.2); background: rgba(240,180,41,0.08); }
  .nbadge.covering { color: var(--amber); border-color: rgba(240,180,41,0.25); background: rgba(240,180,41,0.1); animation: badge-pulse 2s ease infinite; }

  .metric { margin-bottom: 6px; }
  .metric:last-child { margin-bottom: 0; }
  .metric-row { display: flex; align-items: center; gap: 8px; }
  .metric-label { font-size: 9px; color: var(--muted); width: 28px; flex-shrink: 0; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
  .metric-track { flex: 1; height: 6px; background: rgba(255,255,255,0.05); border-radius: 3px; overflow: hidden; }
  .metric-fill { height: 100%; border-radius: 3px; transition: width 0.6s cubic-bezier(0.34,1.56,0.64,1); }
  .metric-fill.green { background: linear-gradient(90deg, var(--green), #2ebf7e); box-shadow: 0 0 8px rgba(62,207,142,0.2); }
  .metric-fill.amber { background: linear-gradient(90deg, var(--amber), #d49a1b); }
  .metric-fill.red { background: linear-gradient(90deg, var(--red), #d43d44); box-shadow: 0 0 8px rgba(242,84,91,0.2); }
  .metric-fill.muted { background: rgba(255,255,255,0.06); }
  .metric-val { font-size: 10px; color: var(--text); font-weight: 600; width: 50px; text-align: right; flex-shrink: 0; font-variant-numeric: tabular-nums; }
  .metric-val.muted { color: var(--muted); }

  .node-footer { margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.05); display: flex; justify-content: space-between; font-size: 9px; color: var(--muted); }
  .node-footer b { color: var(--text); font-weight: 600; }

  /* Sidebar */
  .sidebar {
    grid-area: sidebar;
    background: rgba(6,8,14,0.7);
    border-left: 1px solid var(--border);
    display: flex; flex-direction: column;
    overflow: hidden;
  }
  .side-section {
    padding: 16px 18px;
    border-bottom: 1px solid var(--border);
    display: flex; flex-direction: column;
    overflow: hidden;
  }
  .side-section.chart-section { height: 220px; flex-shrink: 0; }
  .side-section.log-section { flex: 1; min-height: 0; }

  .side-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; flex-shrink: 0; }
  .side-title { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--text); }
  .side-tag { font-size: 9px; color: var(--green); background: rgba(62,207,142,0.08); padding: 2px 8px; border-radius: 999px; font-weight: 700; }

  .chart-box { flex: 1; min-height: 0; position: relative; }
  .chart-box canvas { width: 100% !important; height: 100% !important; display: block; }

  .log-box {
    flex: 1; min-height: 0; overflow: hidden;
    background: rgba(3,4,8,0.6); border-radius: 10px;
    border: 1px solid var(--border);
    display: flex; flex-direction: column;
  }
  .log-content {
    flex: 1; overflow-y: auto; padding: 10px 12px;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 10.5px; line-height: 1.6;
  }
  .log-line { display: flex; gap: 10px; padding: 2px 0; border-left: 2px solid transparent; padding-left: 10px; margin-left: -12px; }
  .log-line.drop { border-left-color: var(--red); }
  .log-line.reassign { border-left-color: var(--amber); }
  .log-line.reconnect { border-left-color: var(--cyan); }
  .log-line.skip { border-left-color: var(--muted); }
  .log-time { color: #4a5165; flex-shrink: 0; font-variant-numeric: tabular-nums; }
  .log-tag { font-size: 8px; padding: 1px 5px; border-radius: 3px; font-weight: 700; flex-shrink: 0; align-self: center; }
  .log-tag.drop { background: rgba(242,84,91,0.12); color: var(--red); }
  .log-tag.reassign { background: rgba(240,180,41,0.12); color: var(--amber); }
  .log-tag.reconnect { background: rgba(79,214,224,0.12); color: var(--cyan); }
  .log-tag.skip { background: rgba(136,144,164,0.12); color: var(--muted); }
  .log-msg { color: #b0b5c3; overflow-wrap: break-word; }

  /* Bottom Panel */
  .bottom-panel {
    grid-area: bottom;
    background: rgba(3,4,8,0.95);
    border-top: 1px solid var(--border);
    padding: 12px 24px;
    display: flex; flex-direction: column; overflow: hidden;
  }
  .bottom-header { font-size: 10px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; flex-shrink: 0; }
  .bottom-body { flex: 1; min-height: 0; position: relative; }
  .bottom-body canvas { width: 100% !important; height: 100% !important; display: block; }
  .comparison-empty {
    color: var(--muted); font-size: 12px; padding: 24px; text-align: center;
    border: 1px dashed var(--border); border-radius: 10px; line-height: 1.5;
  }
  .comparison-empty code { background: var(--panel-2); padding: 1px 5px; border-radius: 3px; color: var(--cyan); font-size: 11px; }

  /* Toast */
  .toast {
    position: fixed; top: 60px; left: 50%;
    transform: translateX(-50%) translateY(-100px);
    background: rgba(240,180,41,0.1); border: 1px solid rgba(240,180,41,0.3);
    color: var(--amber); padding: 10px 24px; border-radius: 999px;
    font-size: 12px; font-weight: 600; backdrop-filter: blur(16px);
    transition: transform 0.4s cubic-bezier(0.34,1.56,0.64,1);
    z-index: 300; pointer-events: none;
    display: flex; align-items: center; gap: 8px;
    box-shadow: 0 4px 30px rgba(240,180,41,0.08);
  }
  .toast.show { transform: translateX(-50%) translateY(0); }

  /* Animations */
  @keyframes pulse-dot { 0%,100%{opacity:1} 50%{opacity:0.3} }
  @keyframes spin { to{transform:rotate(360deg)} }
  @keyframes spin-reverse { to{transform:rotate(-360deg)} }
  @keyframes breathe { 0%,100%{opacity:0.5;transform:scale(1)} 50%{opacity:0.9;transform:scale(1.06)} }
  @keyframes alarm-ping { 0%{transform:scale(1);opacity:0.5} 100%{transform:scale(1.1);opacity:0} }
  @keyframes card-enter { from{opacity:0;transform:translateY(12px) scale(0.96)} to{opacity:1;transform:translateY(0) scale(1)} }
  @keyframes training-pulse { 0%,100%{box-shadow:0 0 0 1px rgba(79,214,224,0.12),0 8px 32px rgba(0,0,0,0.5),0 0 50px rgba(79,214,224,0.08)} 50%{box-shadow:0 0 0 1px rgba(79,214,224,0.2),0 8px 32px rgba(0,0,0,0.5),0 0 60px rgba(79,214,224,0.15)} }
  @keyframes badge-pulse { 0%,100%{box-shadow:0 0 0 0 rgba(240,180,41,0.2)} 50%{box-shadow:0 0 0 4px rgba(240,180,41,0)} }
  @keyframes flow-dash { to { stroke-dashoffset: -36 } }
</style>
</head>
<body>

<div class="bg-depth"></div>
<div class="bg-grid"></div>

<!-- Toast -->
<div class="toast" id="toast"><span>🔁</span><span id="toast-text">Partition reassigned</span></div>

<!-- Top Bar -->
<div class="topbar">
  <div class="brand">
    <div class="brand-icon">H</div>
    <div>
      <div class="brand-text">HALO</div>
      <div class="brand-sub">Live Scheduler</div>
    </div>
  </div>
  <div class="stats-bar">
    <div class="stat-item"><span class="stat-label">Round</span><span class="stat-value accent" id="s-round">—</span></div>
    <div class="stat-item"><span class="stat-label">Accuracy</span><span class="stat-value green" id="s-acc">—</span></div>
    <div class="stat-item"><span class="stat-label">Loss</span><span class="stat-value" id="s-loss">—</span></div>
    <div class="stat-item"><span class="stat-label">Nodes</span><span class="stat-value" id="s-nodes">—</span></div>
    <div class="stat-item"><span class="stat-label">Dropped</span><span class="stat-value red" id="s-dropped">—</span></div>
    <div class="stat-item"><span class="stat-label">Examples</span><span class="stat-value" id="s-examples">—</span></div>
    <div class="live-badge"><span class="live-dot"></span><span class="live-text">LIVE</span></div>
  </div>
</div>

<!-- App Shell -->
<div class="app-shell">
  <!-- Main Stage -->
  <div class="main-stage" id="main-stage">
    <svg class="conn-layer" id="conn-layer" width="100%" height="100%"></svg>

    <!-- Coordinator -->
    <div class="coord-hub" id="coord-hub">
      <div class="coord-ring r1"></div>
      <div class="coord-ring r2"></div>
      <div class="coord-ring r3"></div>
      <div class="coord-glow"></div>
      <div class="coord-body">
        <div class="coord-icon">🖥️</div>
        <div class="coord-label">SERVER</div>
      </div>
      <div class="coord-packets" id="coord-packets">0 pkts</div>
    </div>

    <!-- Nodes Grid -->
    <div class="nodes-wrap" id="nodes-wrap"></div>
  </div>

  <!-- Sidebar -->
  <div class="sidebar">
    <div class="side-section chart-section">
      <div class="side-header">
        <span class="side-title">Training Progress</span>
        <span class="side-tag">● Real-time</span>
      </div>
      <div class="chart-box">
        <canvas id="chart"></canvas>
      </div>
    </div>
    <div class="side-section log-section">
      <div class="side-header">
        <span class="side-title">System Event Log</span>
        <span class="side-tag">● Streaming</span>
      </div>
      <div class="log-box">
        <div class="log-content" id="log-content">
          <div style="color:var(--muted);text-align:center;padding:20px 0;font-family:inherit;">Waiting for events...</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Bottom Panel -->
  <div class="bottom-panel">
    <div class="bottom-header">Adaptive vs. Baseline — Controlled Comparison</div>
    <div class="bottom-body">
      <canvas id="compChart" style="display:none;"></canvas>
      <div class="comparison-empty" id="comp-empty">
        No comparison data yet. Run a controlled Phase 5 experiment and save to <code>comparison_runs.json</code>.
      </div>
    </div>
  </div>
</div>

<script>
(function(){
  'use strict';

  const svg = document.getElementById('conn-layer');
  const nodesWrap = document.getElementById('nodes-wrap');
  const mainStage = document.getElementById('main-stage');
  const coordHub = document.getElementById('coord-hub');
  const toast = document.getElementById('toast');
  const toastText = document.getElementById('toast-text');
  const coordPackets = document.getElementById('coord-packets');

  let lastState = null;
  let particles = [];
  let travelers = [];
  let packetCounter = 0;
  let lastPacketDisplay = 0;
  let chart = null, compChart = null;
  let renderedEventCount = 0;
  let lastToastMsg = '';
  let _stageRect = null;

  const STATUS_COLORS = {
    active: '#3ecf8e', training: '#4fd6e0', skipped: '#f0b429',
    dropped: '#f2545b', reconnected: '#4fd6e0'
  };

  function fmtNum(v, d) {
    if (v == null || v === undefined || v === '') return '—';
    const n = Number(v);
    return isNaN(n) ? '—' : n.toFixed(d);
  }
  function fmtPct(v) {
    if (v == null) return '—';
    const n = Number(v);
    return isNaN(n) ? '—' : (n * 100).toFixed(2) + '%';
  }
  function createSVG(tag, attrs) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
  }

  /* ---------- Geometry ---------- */
  function getCoordCenter() {
    const c = coordHub.getBoundingClientRect();
    const a = mainStage.getBoundingClientRect();
    return { x: c.left - a.left + c.width / 2, y: c.top - a.top + c.height / 2 };
  }

  function getCoordDock(index, total) {
    const c = getCoordCenter();
    const spread = total <= 2 ? 8 : 22;
    const offset = (index - (total - 1) / 2) * spread;
    return { x: c.x + 68, y: c.y + offset };
  }

  function getCardDock(card, index, total) {
    const r = card.getBoundingClientRect();
    const a = mainStage.getBoundingClientRect();
    const centerY = r.top - a.top + r.height / 2;
    const leftX = r.left - a.left;
    const spread = total <= 2 ? 8 : 22;
    const offset = (index - (total - 1) / 2) * spread;
    return { x: leftX - 2, y: centerY + offset };
  }

  /* ---------- Node Rendering (DOM-preserving) ---------- */
  function renderNodeCard(n) {
    const st = n.status || 'active';
    const portColor = STATUS_COLORS[st] || STATUS_COLORS.active;
    const batt = n.battery_percent;
    const hasBatt = batt != null && batt !== '';
    const battVal = hasBatt ? Number(batt) : null;
    const battClass = !hasBatt ? 'muted' : battVal < 25 ? 'red' : battVal < 50 ? 'amber' : 'green';
    const cpu = n.cpu_percent;
    const hasCpu = cpu != null && cpu !== '';
    const cpuVal = hasCpu ? Number(cpu) : null;
    const net = n.network_latency_ms;
    const hasNet = net != null && net !== '';
    const netVal = hasNet ? Number(net) : null;

    const covering = (n.covering_partitions && n.covering_partitions.length)
      ? `<span class="nbadge covering">🔁 +P${n.covering_partitions.join(', P')}</span>` : '';

    return `
      <div class="node-card status-${st}" data-nid="${n.id}" data-status="${st}">
        <div class="node-port" style="background:${portColor};box-shadow:0 0 10px ${portColor},0 0 18px ${portColor}"></div>
        <div class="node-head">
          <div class="node-avatar">💻</div>
          <div class="node-title">
            <div class="node-name">Node ${n.id.slice(0,8)}…</div>
            <div class="node-status-row">
              <span class="status-dot ${st}"></span>
              <span class="node-status-text ${st}">${st}</span>
            </div>
          </div>
        </div>
        <div class="node-badges">
          <span class="nbadge part">P${n.partition_id != null ? n.partition_id : '?'}</span>
          <span class="nbadge score">S:${fmtNum(n.score, 2)}</span>
          <span class="nbadge epoch">E:${n.local_epochs != null ? n.local_epochs : '?'}</span>
          ${covering}
        </div>
        <div class="metric">
          <div class="metric-row">
            <span class="metric-label">CPU</span>
            <div class="metric-track"><div class="metric-fill ${hasCpu ? 'green' : 'muted'}" style="width:${hasCpu ? Math.min(cpuVal, 100) : 0}%"></div></div>
            <span class="metric-val ${hasCpu ? '' : 'muted'}">${fmtNum(cpu, 1)}${hasCpu ? '%' : ''}</span>
          </div>
        </div>
        <div class="metric">
          <div class="metric-row">
            <span class="metric-label">Batt</span>
            <div class="metric-track"><div class="metric-fill ${battClass}" style="width:${hasBatt ? Math.min(battVal, 100) : 0}%"></div></div>
            <span class="metric-val ${hasBatt ? '' : 'muted'}">${hasBatt ? battVal.toFixed(0) + '%' : '—'} ${n.battery_plugged_in ? '🔌' : (hasBatt ? '🔋' : '')}</span>
          </div>
        </div>
        <div class="metric">
          <div class="metric-row">
            <span class="metric-label">Net</span>
            <div class="metric-track"><div class="metric-fill ${hasNet ? 'green' : 'muted'}" style="width:${hasNet ? Math.min(netVal / 3, 100) : 0}%"></div></div>
            <span class="metric-val ${hasNet ? '' : 'muted'}">${fmtNum(net, 1)}${hasNet ? 'ms' : ''}</span>
          </div>
        </div>
        <div class="node-footer">
          <span>Ex: <b>${n.num_examples != null ? Number(n.num_examples).toLocaleString() : '—'}</b></span>
          <span>Rnd: <b>${n.last_round != null ? n.last_round : '–'}</b></span>
        </div>
      </div>
    `;
  }

  function updateCardMetrics(card, n) {
    const st = n.status || 'active';
    if (card.dataset.status !== st) {
      card.classList.remove('status-' + card.dataset.status);
      card.classList.add('status-' + st);
      card.dataset.status = st;
      const dot = card.querySelector('.status-dot');
      if (dot) dot.className = 'status-dot ' + st;
      const stText = card.querySelector('.node-status-text');
      if (stText) { stText.className = 'node-status-text ' + st; stText.textContent = st; }
      const port = card.querySelector('.node-port');
      const pc = STATUS_COLORS[st] || STATUS_COLORS.active;
      if (port) port.style.cssText = `background:${pc};box-shadow:0 0 10px ${pc},0 0 18px ${pc}`;
    }

    const rows = card.querySelectorAll('.metric-row');
    if (rows[0]) {
      const cpu = n.cpu_percent; const hasCpu = cpu != null && cpu !== ''; const v = hasCpu ? Number(cpu) : null;
      const fill = rows[0].querySelector('.metric-fill'); const val = rows[0].querySelector('.metric-val');
      fill.style.width = (hasCpu ? Math.min(v, 100) : 0) + '%';
      fill.className = 'metric-fill ' + (hasCpu ? 'green' : 'muted');
      val.textContent = fmtNum(cpu, 1) + (hasCpu ? '%' : '');
      val.className = 'metric-val ' + (hasCpu ? '' : 'muted');
    }
    if (rows[1]) {
      const batt = n.battery_percent; const hasBatt = batt != null && batt !== ''; const v = hasBatt ? Number(batt) : null;
      const cls = !hasBatt ? 'muted' : v < 25 ? 'red' : v < 50 ? 'amber' : 'green';
      const fill = rows[1].querySelector('.metric-fill'); const val = rows[1].querySelector('.metric-val');
      fill.style.width = (hasBatt ? Math.min(v, 100) : 0) + '%';
      fill.className = 'metric-fill ' + cls;
      val.textContent = (hasBatt ? v.toFixed(0) + '%' : '—') + (n.battery_plugged_in ? ' 🔌' : (hasBatt ? ' 🔋' : ''));
      val.className = 'metric-val ' + (hasBatt ? '' : 'muted');
    }
    if (rows[2]) {
      const net = n.network_latency_ms; const hasNet = net != null && net !== ''; const v = hasNet ? Number(net) : null;
      const fill = rows[2].querySelector('.metric-fill'); const val = rows[2].querySelector('.metric-val');
      fill.style.width = (hasNet ? Math.min(v / 3, 100) : 0) + '%';
      fill.className = 'metric-fill ' + (hasNet ? 'green' : 'muted');
      val.textContent = fmtNum(net, 1) + (hasNet ? 'ms' : '');
      val.className = 'metric-val ' + (hasNet ? '' : 'muted');
    }

    const footerSpans = card.querySelectorAll('.node-footer span');
    if (footerSpans[0]) footerSpans[0].innerHTML = `Ex: <b>${n.num_examples != null ? Number(n.num_examples).toLocaleString() : '—'}</b>`;
    if (footerSpans[1]) footerSpans[1].innerHTML = `Rnd: <b>${n.last_round != null ? n.last_round : '–'}</b>`;

    const badges = card.querySelector('.node-badges');
    const existingCover = badges.querySelector('.nbadge.covering');
    const coveringHTML = (n.covering_partitions && n.covering_partitions.length)
      ? `<span class="nbadge covering">🔁 +P${n.covering_partitions.join(', P')}</span>` : '';
    if (coveringHTML) {
      if (!existingCover) badges.insertAdjacentHTML('beforeend', coveringHTML);
      else if (existingCover.textContent !== '🔁 +P' + n.covering_partitions.join(', P')) existingCover.outerHTML = coveringHTML;
    } else if (existingCover) {
      existingCover.remove();
    }
  }

  function renderNodes(nodeList) {
    if (!nodeList.length) {
      nodesWrap.innerHTML = '<div style="color:var(--muted);font-size:13px;text-align:center;padding:40px;width:100%;">Waiting for nodes...</div>';
      nodesWrap.classList.remove('vertical-stack');
      return;
    }

    if (nodeList.length <= 2) {
      nodesWrap.classList.add('vertical-stack');
    } else {
      nodesWrap.classList.remove('vertical-stack');
    }

    const existing = Array.from(nodesWrap.querySelectorAll('.node-card'));
    const existingIds = existing.map(c => c.dataset.nid);
    const newIds = nodeList.map(n => n.id);

    if (JSON.stringify(existingIds) === JSON.stringify(newIds)) {
      nodeList.forEach(n => {
        const card = nodesWrap.querySelector(`.node-card[data-nid="${n.id}"]`);
        if (card) updateCardMetrics(card, n);
      });
      return;
    }

    const tempWrap = document.createElement('div');
    tempWrap.innerHTML = nodeList.map(renderNodeCard).join('');
    nodesWrap.innerHTML = '';
    Array.from(tempWrap.children).forEach(c => nodesWrap.appendChild(c));
  }

  /* ---------- SVG Connections (thin, elegant, behind cards) ---------- */
  function getStageRect() {
    if (!_stageRect) _stageRect = mainStage.getBoundingClientRect();
    return _stageRect;
  }

  function drawConnections() {
    svg.innerHTML = '';
    const defs = createSVG('defs', {});

    // Subtle glow filters
    ['green', 'red', 'amber', 'cyan', 'blue'].forEach(c => {
      const f = createSVG('filter', {id: 'glow-' + c, x: '-100%', y: '-100%', width: '300%', height: '300%'});
      const col = c==='green'?'#3ecf8e':c==='red'?'#f2545b':c==='amber'?'#f0b429':c==='cyan'?'#4fd6e0':'#5aa6ff';
      f.innerHTML = `<feGaussianBlur stdDeviation="3" result="b"/><feFlood flood-color="${col}" flood-opacity="0.30" result="c"/><feComposite in="c" in2="b" operator="in" result="d"/><feMerge><feMergeNode in="d"/><feMergeNode in="SourceGraphic"/></feMerge>`;
      defs.appendChild(f);
    });

    const mk = createSVG('marker', {id: 'arr', markerWidth: '10', markerHeight: '10', refX: '9', refY: '3', orient: 'auto'});
    mk.innerHTML = '<path d="M0,0 L9,3 L0,6 Z" fill="#f0b429"/>';
    defs.appendChild(mk);
    svg.appendChild(defs);

    const cards = Array.from(nodesWrap.querySelectorAll('.node-card'));
    if (!cards.length) return;

    const total = cards.length;
    const nodes = lastState ? (lastState.nodes || {}) : {};
    _stageRect = mainStage.getBoundingClientRect();

    // Lines to each node
    cards.forEach((card, idx) => {
      const nid = card.dataset.nid;
      const n = nodes[nid];
      if (!n) return;
      const s = getCoordDock(idx, total);
      const t = getCardDock(card, idx, total);
      const isDropped = card.dataset.status === 'dropped';
      const color = STATUS_COLORS[n.status] || STATUS_COLORS.active;

      // Layer 1: Broad faint aura
      svg.appendChild(createSVG('line', {
        x1: s.x, y1: s.y, x2: t.x, y2: t.y,
        stroke: color, 'stroke-width': 5, opacity: 0.08, 'stroke-linecap': 'round'
      }));

      // Layer 2: Background reference
      svg.appendChild(createSVG('line', {
        x1: s.x, y1: s.y, x2: t.x, y2: t.y,
        stroke: isDropped ? 'rgba(242,84,91,0.25)' : 'rgba(255,255,255,0.15)',
        'stroke-width': 1.2, 'stroke-linecap': 'round'
      }));

      if (!isDropped) {
        // Layer 3: Bright core
        svg.appendChild(createSVG('line', {
          x1: s.x, y1: s.y, x2: t.x, y2: t.y,
          stroke: color, 'stroke-width': 1.8, opacity: 0.88, 'stroke-linecap': 'round'
        }));

        // Layer 4: Animated dash
        const dash = createSVG('line', {
          x1: s.x, y1: s.y, x2: t.x, y2: t.y,
          stroke: '#ffffff', 'stroke-width': 1, opacity: 0.28,
          'stroke-dasharray': '5 14', 'stroke-linecap': 'round'
        });
        dash.style.animation = 'flow-dash 1.3s linear infinite';
        svg.appendChild(dash);
      }
    });

    // Reassignment arrows
    cards.forEach((card, idx) => {
      const nid = card.dataset.nid;
      const n = nodes[nid];
      if (!n || !n.covering_partitions || !n.covering_partitions.length) return;
      const p1 = getCardDock(card, idx, total);

      n.covering_partitions.forEach(pid => {
        const droppedIdx = cards.findIndex(c => {
          const cn = nodes[c.dataset.nid];
          return c.dataset.status === 'dropped' && cn && cn.partition_id === pid;
        });
        if (droppedIdx < 0) return;
        const p2 = getCardDock(cards[droppedIdx], droppedIdx, total);

        const mx = (p1.x + p2.x) / 2, my = (p1.y + p2.y) / 2;
        const dx = p2.x - p1.x, dy = p2.y - p1.y;
        const qx = mx + dy * 0.3, qy = my - dx * 0.3;

        const path = createSVG('path', {
          d: `M${p2.x},${p2.y} Q${qx},${qy} ${p1.x},${p1.y}`,
          fill: 'none', stroke: '#f0b429', 'stroke-width': 2.5,
          'stroke-dasharray': '5 4', opacity: 0.92, 'marker-end': 'url(#arr)', 'stroke-linecap': 'round',
          filter: 'url(#glow-amber)'
        });
        path.style.animation = 'flow-dash 0.7s linear infinite';
        svg.appendChild(path);

        const lbl = createSVG('text', {
          x: qx, y: qy - 6, fill: '#f0b429', 'font-size': '11',
          'font-weight': '700', 'text-anchor': 'middle', filter: 'url(#glow-amber)'
        });
        lbl.textContent = '+P' + pid;
        svg.appendChild(lbl);
      });
    });

    // Particles
    particles.forEach(p => {
      svg.appendChild(createSVG('circle', {
        cx: p.x, cy: p.y, r: 4, fill: p.color, opacity: p.opacity,
        filter: 'url(#glow-cyan)'
      }));
      if (p.trail) {
        p.trail.forEach((t, i) => {
          svg.appendChild(createSVG('circle', {
            cx: t.x, cy: t.y, r: 2.8 - i * 0.9, fill: p.color, opacity: t.opacity * 0.35
          }));
        });
      }
    });

    // Travelers
    travelers.forEach(t => {
      svg.appendChild(createSVG('circle', {
        cx: t.x, cy: t.y, r: 8, fill: t.color, opacity: 0.85,
        filter: 'url(#glow-cyan)'
      }));
      svg.appendChild(createSVG('circle', {
        cx: t.x, cy: t.y, r: 14, fill: 'none', stroke: t.color,
        'stroke-width': 1.5, opacity: 0.25
      }));
    });
  }

  /* ---------- Particles & Travelers ---------- */
  function spawnParticle(targetCard, fromCenter, idx, total) {
    const s = getCoordDock(idx, total);
    const t = getCardDock(targetCard, idx, total);
    const sx = fromCenter ? s.x : t.x;
    const sy = fromCenter ? s.y : t.y;
    const tx = fromCenter ? t.x : s.x;
    const ty = fromCenter ? t.y : s.y;
    particles.push({
      x: sx, y: sy, tx, ty,
      progress: 0, speed: 0.020 + Math.random() * 0.016,
      color: fromCenter ? '#4fd6e0' : '#3ecf8e', opacity: 1, trail: []
    });
    packetCounter++;
  }

  function spawnTraveler(targetCard, fromCenter, idx, total) {
    const s = getCoordDock(idx, total);
    const t = getCardDock(targetCard, idx, total);
    const sx = fromCenter ? s.x : t.x;
    const sy = fromCenter ? s.y : t.y;
    const tx = fromCenter ? t.x : s.x;
    const ty = fromCenter ? t.y : s.y;
    travelers.push({
      x: sx, y: sy, tx, ty,
      progress: 0, speed: 0.004 + Math.random() * 0.003,
      color: fromCenter ? '#5aa6ff' : '#3ecf8e'
    });
  }

  function updateParticles() {
    particles = particles.filter(p => {
      const dx = p.tx - p.x, dy = p.ty - p.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 6) return false;
      const move = Math.min(dist, 10);
      p.trail.unshift({x: p.x, y: p.y, opacity: p.opacity});
      if (p.trail.length > 2) p.trail.pop();
      p.x += (dx / dist) * move;
      p.y += (dy / dist) * move;
      p.progress += p.speed;
      p.opacity = Math.max(0, 1 - p.progress * 0.95);
      p.trail.forEach((t, i) => { t.opacity = p.opacity * (1 - i * 0.35); });
      return p.opacity > 0;
    });
  }

  function updateTravelers() {
    travelers = travelers.filter(t => {
      const dx = t.tx - t.x, dy = t.ty - t.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 10) return false;
      const move = Math.min(dist, 4.5);
      t.x += (dx / dist) * move;
      t.y += (dy / dist) * move;
      t.progress += t.speed;
      return t.progress < 1.2;
    });
  }

  /* ---------- Event Log ---------- */
  function shortenIds(text) {
    return text.replace(/\\b\\d{10,}\\b/g, m => m.slice(0, 8) + '…');
  }

  function renderEvents(events) {
    const log = document.getElementById('log-content');
    if (!events || !events.length) {
      if (renderedEventCount === 0) {
        log.innerHTML = '<div style="color:var(--muted);text-align:center;padding:20px 0;font-family:inherit;">Waiting for events...</div>';
      }
      return;
    }
    if (events.length === renderedEventCount) return;

    const icons = { drop: 'DROP', reassign: 'RASGN', reconnect: 'RECON', skip: 'SKIP' };
    const wasAtTop = log.scrollTop < 5;
    log.innerHTML = events.slice().reverse().map(e => {
      const ts = e.timestamp || (Date.now() / 1000);
      const date = new Date(ts * 1000);
      const time = date.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
      return `<div class="log-line ${e.type}">
        <span class="log-time">${time}</span>
        <span class="log-tag ${e.type}">${icons[e.type] || 'INFO'}</span>
        <span class="log-msg">${shortenIds(e.message)}</span>
      </div>`;
    }).join('');

    renderedEventCount = events.length;
    if (wasAtTop) log.scrollTop = 0;
  }

  /* ---------- Stats ---------- */
  function applyStats(state) {
    const s = state.summary || {};
    const sc = s.status_counts || {};
    document.getElementById('s-round').textContent = (state.current_round || 0) + '/' + (state.total_rounds || '–');
    document.getElementById('s-acc').textContent = fmtPct(state.global_accuracy);
    document.getElementById('s-loss').textContent = state.global_loss != null ? state.global_loss.toFixed(4) : '–';
    document.getElementById('s-nodes').textContent = s.connected_nodes ?? 0;
    document.getElementById('s-dropped').textContent = sc.dropped ?? 0;
    document.getElementById('s-examples').textContent = (s.total_examples_this_round ?? 0).toLocaleString();

    const nodeList = Object.values(state.nodes || {});
    const hasDropped = nodeList.some(n => n.status === 'dropped');
    const hasCovering = nodeList.some(n => n.covering_partitions && n.covering_partitions.length);
    let toastMsg = '';
    if (hasDropped && hasCovering) {
      const covering = nodeList.find(n => n.covering_partitions && n.covering_partitions.length);
      const dropped = nodeList.find(n => n.status === 'dropped');
      if (covering && dropped) {
        toastMsg = `Partition ${dropped.partition_id} → Node ${covering.id.slice(0, 8)}… (2× data)`;
      }
    }
    if (toastMsg && toastMsg !== lastToastMsg) {
      toastText.textContent = toastMsg;
      toast.classList.add('show');
      lastToastMsg = toastMsg;
      setTimeout(() => { if (lastToastMsg === toastMsg) toast.classList.remove('show'); }, 4000);
    } else if (!toastMsg) {
      toast.classList.remove('show');
      lastToastMsg = '';
    }
  }

  /* ---------- Charts ---------- */
  function initChart() {
    const ctx = document.getElementById('chart').getContext('2d');
    chart = new Chart(ctx, {
      type: 'line',
      data: { labels: [], datasets: [
        { label: 'Accuracy %', data: [], borderColor: '#3ecf8e', backgroundColor: 'rgba(62,207,142,0.07)',
          tension: 0.35, fill: true, pointRadius: 3, borderWidth: 2.5, yAxisID: 'y' },
        { label: 'Loss', data: [], borderColor: '#f2545b', backgroundColor: 'rgba(242,84,91,0.05)',
          tension: 0.35, fill: true, pointRadius: 3, borderWidth: 2, yAxisID: 'y1' },
      ]},
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#4a5165', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.02)' }, border: { display: false } },
          y: { position: 'left', min: 0, max: 100, ticks: { color: '#3ecf8e', font: { size: 10 }, stepSize: 25 }, grid: { color: 'rgba(255,255,255,0.02)' }, border: { display: false } },
          y1: { position: 'right', min: 0, ticks: { color: '#f2545b', font: { size: 10 } }, grid: { display: false }, border: { display: false } },
        },
      },
    });
  }

  function updateChart(history) {
    if (!chart || !history) return;
    if (history.length === chart.data.labels.length) return;
    chart.data.labels = history.map(h => 'R' + h.round);
    chart.data.datasets[0].data = history.map(h => h.accuracy == null ? null : h.accuracy * 100);
    chart.data.datasets[1].data = history.map(h => h.loss);
    chart.update('none');
  }

  function loadComparison() {
    fetch('/api/comparison').then(r => r.json()).then(data => {
      if (!data) return;
      document.getElementById('comp-empty').style.display = 'none';
      const canvas = document.getElementById('compChart');
      canvas.style.display = 'block';
      if (compChart) compChart.destroy();
      compChart = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: {
          labels: data.series_a.map(p => 'R' + p.round),
          datasets: [
            { label: data.label_a || 'Adaptive', data: data.series_a.map(p => p.accuracy * 100),
              borderColor: '#3ecf8e', backgroundColor: 'rgba(62,207,142,0.06)', tension: 0.25, fill: true, pointRadius: 3, borderWidth: 2.5 },
            { label: data.label_b || 'Baseline', data: data.series_b.map(p => p.accuracy * 100),
              borderColor: '#f2545b', backgroundColor: 'rgba(242,84,91,0.04)', tension: 0.25, fill: true, pointRadius: 3, borderWidth: 2, borderDash: [5, 4] },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { labels: { color: '#8890a4', font: { size: 11 } } } },
          scales: {
            y: { min: 0, max: 100, ticks: { color: '#4a5165', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.02)' }, border: { display: false } },
            x: { ticks: { color: '#4a5165', font: { size: 10 } }, grid: { display: false }, border: { display: false } },
          },
        },
      });
    }).catch(() => {});
  }

  /* ---------- Animation Loop ---------- */
  function animate() {
    updateParticles();
    updateTravelers();
    drawConnections();
    if (packetCounter !== lastPacketDisplay) {
      coordPackets.textContent = packetCounter + ' pkts';
      lastPacketDisplay = packetCounter;
    }
    requestAnimationFrame(animate);
  }

  /* ---------- Polling ---------- */
  function poll() {
    fetch('/api/state').then(r => r.json()).then(state => {
      lastState = state;
      applyStats(state);

      const nodes = state.nodes || {};
      const nodeList = Object.entries(nodes).map(([id, n]) => ({ ...n, id }));
      renderNodes(nodeList);

      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          _stageRect = null;
          drawConnections();
        });
      });

      renderEvents(state.events || []);
      updateChart(state.history || []);
    }).catch(err => {
      console.error('Poll error:', err);
    });
  }

  /* ---------- Spawners ---------- */
  setInterval(() => {
    if (!lastState) return;
    const cards = Array.from(nodesWrap.querySelectorAll('.node-card[data-status="active"], .node-card[data-status="training"]'));
    if (!cards.length) return;
    const total = cards.length;
    if (Math.random() > 0.35) {
      const idx = Math.floor(Math.random() * cards.length);
      spawnParticle(cards[idx], true, idx, total);
    }
    if (Math.random() > 0.35) {
      const idx = Math.floor(Math.random() * cards.length);
      spawnParticle(cards[idx], false, idx, total);
    }
  }, 200);

  setInterval(() => {
    if (!lastState) return;
    const cards = Array.from(nodesWrap.querySelectorAll('.node-card[data-status="active"], .node-card[data-status="training"]'));
    if (!cards.length) return;
    const total = cards.length;
    if (Math.random() > 0.4) {
      const idx = Math.floor(Math.random() * cards.length);
      spawnTraveler(cards[idx], Math.random() > 0.5, idx, total);
    }
  }, 800);

  /* ---------- Init ---------- */
  window.addEventListener('resize', () => {
    _stageRect = null;
    drawConnections();
  });

  initChart();
  loadComparison();
  poll();
  setInterval(poll, 1500);
  setInterval(loadComparison, 10000);
  animate();
})();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE, initial_state=_state_payload())


@app.route("/api/state")
def api_state():
    return jsonify(_state_payload())


@app.route("/api/comparison")
def api_comparison():
    return jsonify(_load_comparison())


if __name__ == "__main__":
    print("HALO Dashboard v5 running at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)