"""
HALO Live Dashboard
----------------------
Run this separately: python dashboard_app.py
Then open http://127.0.0.1:5000 in a browser.
Reads scheduler/dashboard_state.py's shared state file every couple of
seconds — completely independent of the Flower processes.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "scheduler"))

from flask import Flask, jsonify, render_template_string
from dashboard_state import load_state

app = Flask(__name__)

PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>HALO Live Dashboard</title>
    <meta http-equiv="refresh" content="0">
    <style>
        body { font-family: system-ui, sans-serif; background: #0f1117; color: #eee; padding: 20px; }
        h1 { color: #7cc4ff; }
        .summary { background: #1a1d29; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
        .nodes { display: flex; gap: 15px; flex-wrap: wrap; }
        .node-card { background: #1a1d29; border-radius: 8px; padding: 15px; width: 260px; border-left: 4px solid #555; }
        .node-card.training { border-left-color: #4caf50; }
        .node-card.skipped { border-left-color: #f44336; }
        .node-card.active { border-left-color: #4caf50; }
        .metric { display: flex; justify-content: space-between; font-size: 14px; padding: 2px 0; }
        .metric span:first-child { color: #999; }
        .reassigned { color: #ffb74d; font-size: 13px; margin-top: 8px; }
    </style>
</head>
<body>
    <h1>HALO — Live Scheduler Dashboard</h1>
    <div class="summary">
        <div class="metric"><span>Round</span><span>{{ state.current_round }} / {{ state.total_rounds }}</span></div>
        <div class="metric"><span>Global Accuracy</span><span>{{ "%.2f"|format((state.global_accuracy or 0) * 100) }}%</span></div>
        <div class="metric"><span>Global Loss</span><span>{{ "%.4f"|format(state.global_loss or 0) }}</span></div>
    </div>
    <div class="nodes">
        {% for node_id, node in state.nodes.items() %}
        <div class="node-card {{ node.get('status', '') }}">
            <strong>Node {{ node_id[:8] }}...</strong>
            <div class="metric"><span>Status</span><span>{{ node.get('status', 'unknown') }}</span></div>
            <div class="metric"><span>Score</span><span>{{ node.get('score', '—') }}</span></div>
            <div class="metric"><span>Local Epochs</span><span>{{ node.get('local_epochs', '—') }}</span></div>
            <div class="metric"><span>CPU</span><span>{{ node.get('cpu_percent', '—') }}%</span></div>
            <div class="metric"><span>Battery</span><span>{{ node.get('battery_percent', '—') }}% {{ '🔌' if node.get('battery_plugged_in') else '🔋' }}</span></div>
            <div class="metric"><span>Network</span><span>{{ node.get('network_latency_ms', '—') }}ms</span></div>
            <div class="metric"><span>Num Examples</span><span>{{ node.get('num_examples', '—') }}</span></div>
            {% if node.get('reassigned_partitions') %}
            <div class="reassigned">+ reassigned partitions: {{ node.get('reassigned_partitions') }}</div>
            {% endif %}
        </div>
        {% endfor %}
    </div>
</body>
</html>
"""


@app.route("/")
def index():
    state = load_state()
    return render_template_string(PAGE, state=state)


@app.route("/api/state")
def api_state():
    return jsonify(load_state())


if __name__ == "__main__":
    print("HALO Dashboard running at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)