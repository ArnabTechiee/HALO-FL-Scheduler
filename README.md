```markdown
# HALO-FL-Scheduler

**HALO** (Health-Aware, Loss-Tolerant Orchestrator) is a fault‑tolerant, adaptive federated learning scheduler built on top of the [Flower](https://flower.dev) framework. It extends standard `FedAvg` to handle real‑world edge‑device variability by:

- **Adapting per‑client workload** (local epochs or skip) based on live telemetry (CPU, battery, memory, network).
- **Stable reassignment of data partitions** when a node drops – the partition stays with a chosen survivor until that survivor also disappears, preventing “partition bouncing”.
- **Recognising reconnecting devices** via a stable hostname‑derived ID, restoring their original partition automatically.
- **Publishing a de‑duplicated, telemetry‑enriched event feed** to a real‑time dashboard.

This project is built as an extension to Flower’s [quickstart‑pytorch](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch) example and uses **PyTorch** with **CIFAR‑10**.

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
  - [Adaptive Capacity Scoring](#adaptive-capacity-scoring)
  - [Fault‑Tolerant Stable Reassignment](#fault-tolerant-stable-reassignment)
  - [Device Identity & Reconnection](#device-identity--reconnection)
  - [Dashboard & Event Feed](#dashboard--event-feed)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Demo](#running-the-demo)
- [Project Structure](#project-structure)
- [Results & Evaluation](#results--evaluation)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Key Features

- ✅ **Telemetry‑driven workload adaptation** – Every client sends real‑time system metrics (CPU %, battery level, memory usage, network latency). The server computes a capacity score and maps it to a number of local training epochs. Nodes under heavy load may run fewer epochs or be skipped entirely.
- ✅ **Stable partition reassignment** – When a node drops, its data partition is assigned to a **deterministic surviving node** and stays there until that survivor also disappears. This avoids the “bouncing” problem where the partition jumps between survivors each round.
- ✅ **One‑time drop events with context** – A drop is reported exactly once with a summary of the node’s last telemetry (e.g., “CPU 85%, battery 10%”), giving immediate insight into the cause.
- ✅ **De‑duplicated reassignment events** – Reassignment events are only emitted when the set of covered partitions for a survivor actually changes, preventing dashboard spam during prolonged outages.
- ✅ **Device identity tracking** – A stable hash of the hostname (`telem_device_id`) allows the system to recognise a reconnecting device even if it receives a new ephemeral `node_id`, restoring its original partition and cleaning up stale state.
- ✅ **Live dashboard** – Node states and events (drop, reassign, reconnect, skip) are streamed to a simple web dashboard for real‑time monitoring.

---

## Architecture

```text
Client App (edge device)
    │
    ├── collects telemetry (CPU, battery, memory, network, device ID)
    ├── trains on its own partition (and possibly extra partitions)
    └── sends metrics + model updates
           │
           ▼
Server App (HALO AdaptiveFedAvg)
    │
    ├── stores last‑known telemetry per node
    ├── computes capacity score → local epochs (or skip)
    ├── detects dropped nodes (heartbeat timeout)
    ├── stably reassigns partitions to survivors
    ├── recognises reconnecting devices
    └── publishes events/state to dashboard
           │
           ▼
Dashboard (Flask / simple web)
    └── displays live node states & event feed

```

The system uses Flower’s **ServerApp / ClientApp** architecture. The custom strategy (`AdaptiveFedAvg`) resides on the server side and extends `flwr.serverapp.strategy.FedAvg`.

---

## How It Works

### Adaptive Capacity Scoring

Each client sends telemetry (via `telemetry.py`) in the metrics record of its training/evaluation replies. The server uses `capacity_score.py` to compute a score between `0` and `1` based on thresholds for battery, CPU, memory, and network latency. This score is then mapped to a number of local epochs via `score_to_local_epochs()`:

* `score = 0` → **skip** this round (with a human‑readable explanation).
* `0 < score < 1` → run `max(1, int(base_local_epochs * score))` epochs.
* `score = 1` → run `base_local_epochs` epochs (default 2).

The explanation logic (`_explain_low_score`) is used to generate a dashboard message describing why a node was skipped (e.g., *“battery critically low (8%, unplugged)”*).

### Fault‑Tolerant Stable Reassignment

The strategy maintains a `partition_survivor` dictionary mapping each dropped partition to the node currently covering it. During `configure_train`:

1. It determines which nodes that own partitions (`node_to_real_partition`) are **not** present in the current list of available nodes.
2. For each newly dropped node, a **drop** event is emitted exactly once (via `notified_drops`).
3. For each dropped partition, it checks if the current survivor (`partition_survivor`) is still alive:
* If yes, the partition stays with that survivor.
* If no, a new survivor is chosen deterministically (lowest `node_id`) and the mapping is updated.


4. The survivor’s training configuration is updated to include the extra partition IDs (`extra-partition-ids`). The client then loads both its own data and the dropped partition’s data for training.

**De‑duplication** is achieved with `last_reassignment_snapshot`, which stores a `frozenset` of partitions per survivor. A reassign event is only published when the snapshot changes.

### Device Identity & Reconnection

Each client sends a `telem_device_id` (a stable hash of the hostname). The strategy tracks the mapping `device_to_node`. If a device reappears under a new `node_id` (e.g., after a reboot), the system:

* Removes the stale partition mapping for the old node.
* Discards any pending drop notification for the old node.
* Marks the old node as `retired`.
* Emits a **reconnect** event explaining that the device reconnected and its partition no longer needs covering.

This prevents the system from treating the reconnecting device as a brand‑new participant and ensures its original partition is restored.

### Dashboard & Event Feed

All node states and events are sent to a simple dashboard via `dashboard_state.py` (using `update_node` and `add_event`). The dashboard (`dashboard_app.py`) displays:

* Live per‑node metrics (CPU, battery, latency, etc.)
* A chronological event feed with icons (🔴 drop, 🔄 reassign, 🔁 reconnect, ⏭️ skip)
* Current status of each node (training, skipped, dropped, reconnected, retired)

The event feed is generic and requires no schema changes when new event types are added.

---

## Getting Started

### Prerequisites

* Python 3.9+
* pip
* Git
* (Optional) A CUDA‑capable GPU for faster training

### Installation

1. Clone the repository:
```bash
git clone [https://github.com/ArnabTechiee/HALO-FL-Scheduler.git](https://github.com/ArnabTechiee/HALO-FL-Scheduler.git)
cd HALO-FL-Scheduler

```


2. Create and activate a virtual environment (recommended):
```bash
python -m venv halo-env
source halo-env/bin/activate   # On Windows: halo-env\Scripts\activate

```


3. Install dependencies:
```bash
pip install -r requirements.txt

```


*Note: The project is built with `flwr==1.32.1`. If you use a different version, adjust the code accordingly.*

### Running the Demo

The demo runs a two‑client simulation on CIFAR‑10. To start the Flower local deployment with the adaptive strategy:

```bash
flwr run . local-deploy --stream

```

**What to expect:**

* **Round 1** – Both nodes train normally with default epochs (2).
* **Simulate a drop** – You can manually kill one of the client processes (or wait for the built‑in heartbeat timeout). The strategy will detect the drop, emit a **drop** event, and reassign the dropped partition to the surviving node.
* **Stable reassignment** – The partition stays with the survivor in subsequent rounds; no bouncing occurs.
* **Simulate a reconnect** – Restart the dropped client. The system recognises the reconnecting device, restores its original partition, and emits a **reconnect** event.
* **Adaptive workload** – You can observe a node’s `local_epochs` decrease if its CPU usage spikes (e.g., from 2 to 1 when CPU > 60%).

A live dashboard should be accessible (check the console output for the URL) showing node states and events in real time.

---

## Project Structure

```text
HALO-FL-Scheduler/
├── pytorchexample/
│   ├── adaptive_strategy.py      # Custom AdaptiveFedAvg strategy
│   ├── capacity_score.py         # Telemetry → capacity score & epoch mapping
│   ├── telemetry.py              # Client‑side telemetry collection
│   ├── dashboard_state.py        # In‑memory state & event publishing
│   └── client_app.py             # Flower ClientApp (trains, evaluates, sends telemetry)
├── scheduler/                    # (Optional) Mirror of strategy for simulation
│   └── adaptive_strategy.py
├── dashboard_app.py              # Simple Flask dashboard displaying state
├── pyproject.toml                # Flower project configuration
├── requirements.txt              # Python dependencies
└── README.md

```

*Note: Some file names may vary based on your exact implementation.*

---

## Results & Evaluation

The project was tested with the quickstart‑pytorch example using CIFAR‑10. Two runs were compared:

| Run | Behaviour | Outcome |
| --- | --- | --- |
| **Run 1 (correct)** | Node dropped after round 2. Partition stably reassigned to survivor. Device reconnected and restored partition. No duplicate events. | ✅ |
| **Run 2 (baseline)** | Partition bounced to a newly connected node even though the original survivor was alive. | ❌ (bug fixed) |

The adaptive workload successfully prevented overloading: when a node’s CPU reached 78%, its local epochs were reduced from 2 to 1.

**Final metrics** (from Run 1):

* Final accuracy: ~10% (random guessing baseline for CIFAR‑10; the focus is on system behaviour, not model performance).
* Training completed without manual intervention.

---

## Configuration

Key parameters that can be adjusted:

* `base_local_epochs` – default number of epochs for a fully healthy node (set in `AdaptiveFedAvg.__init__`).
* `fraction_train` / `fraction_evaluate` – fraction of nodes sampled per round (configured in `pyproject.toml`).
* Telemetry thresholds – in `capacity_score.py` (e.g., CPU > 85%, battery < 15% unplugged) to determine skip/low‑score conditions.

To change the dataset or model, modify `client_app.py` and the corresponding data loading utilities.

---

## Troubleshooting

* **Dashboard not showing** – Ensure `dashboard_app.py` is running and the port is accessible. Check console for the URL.
* **`telem_device_id` missing** – The client must send it in the metrics. Verify `telemetry.py` is correctly integrated.
* **Stable reassignment not working** – Check that `partition_survivor` is preserved across rounds. If using multiple server processes, consider using a persistent store (e.g., Redis).
* **Node drop not detected** – Flower’s heartbeat timeout may need to be adjusted (see `clientapp.py` heartbeat settings).

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request. For major changes, discuss first.

---

## License

[MIT](https://www.google.com/search?q=LICENSE)

---

## Contact

Arnab Mondal – [GitHub](https://github.com/ArnabTechiee)

Project Link: [https://github.com/ArnabTechiee/HALO-FL-Scheduler](https://www.google.com/search?q=https://github.com/ArnabTechiee/HALO-FL-Scheduler)

```
