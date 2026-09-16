# HALO: Heterogeneous Adaptive Learning Orchestrator..

**HALO** (Heterogeneous Adaptive Learning Orchestrator) is a fault‑tolerant, adaptive federated learning scheduler built on top of the [Flower](https://flower.dev) framework. It extends the standard `FedAvg` strategy to handle real‑world edge‑device variability by:

- **Adapting per‑client workload** (local epochs or skip) based on live telemetry (CPU, battery, memory, network).
- **Stable reassignment of data partitions** when a node drops – the partition stays with a chosen survivor until that survivor also disappears, preventing "partition bouncing".
- **Recognising reconnecting devices** via a stable hostname‑derived ID, restoring their original partition automatically.
- **Publishing a de‑duplicated, telemetry‑enriched event feed** to a real‑time dashboard.

The project is built as an extension of Flower’s [quickstart‑pytorch](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch) example and uses **PyTorch** with **CIFAR‑10**.

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
  - [Adaptive Capacity Scoring](#adaptive-capacity-scoring)
  - [Fault‑Tolerant Stable Reassignment](#fault-tolerant-stable-reassignment)
  - [Device Identity & Reconnection](#device-identity--reconnection)
  - [Dashboard & Event Feed](#dashboard--event-feed)
- [Research Context & Positioning](#research-context--positioning)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Results & Evaluation](#results--evaluation)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Key Features

- ✅ **Telemetry‑Driven Workload Adaptation** – Every client sends real‑time system metrics (CPU %, battery level, memory usage, network latency). The server computes a capacity score and maps it to a number of local training epochs. Nodes under heavy load may run fewer epochs or be skipped entirely.
- ✅ **Stable Partition Reassignment** – When a node drops, its data partition is assigned to a **deterministic surviving node** and stays there until that survivor also disappears. This avoids the “bouncing” problem where the partition jumps between survivors each round.
- ✅ **One‑Time Drop Events with Context** – A drop is reported exactly once with a summary of the node’s last telemetry (e.g., “CPU 85%, battery 10%”), giving immediate insight into the cause.
- ✅ **De‑Duplicated Reassignment Events** – Reassignment events are only emitted when the set of covered partitions for a survivor actually changes, preventing dashboard spam during prolonged outages.
- ✅ **Device Identity Tracking** – A stable hash of the hostname (`telem_device_id`) allows the system to recognise a reconnecting device even if it receives a new ephemeral `node_id`, restoring its original partition and cleaning up stale state.
- ✅ **Live Dashboard** – Node states and events (drop, reassign, reconnect, skip) are streamed to a simple web dashboard for real‑time monitoring.

---

## Architecture

```text
Client App (Edge Device)
    │
    ├── Collects telemetry (CPU, battery, memory, network, device ID)
    ├── Trains on its own partition (and possibly extra partitions)
    └── Sends metrics + model updates
           │
           ▼
Server App (HALO AdaptiveFedAvg)
    │
    ├── Stores last‑known telemetry per node
    ├── Computes capacity score → local epochs (or skip)
    ├── Detects dropped nodes (heartbeat timeout)
    ├── Stably reassigns partitions to survivors
    ├── Recognises reconnecting devices
    └── Publishes events/state to dashboard
           │
           ▼
Dashboard (Flask / Simple Web)
    └── Displays live node states & event feed
```

The system uses Flower’s **ServerApp / ClientApp** architecture. The custom strategy (`AdaptiveFedAvg`) resides on the server side and extends `flwr.serverapp.strategy.FedAvg`.

---

## How It Works

### Adaptive Capacity Scoring

Each client sends telemetry (via `telemetry.py`) in the metrics record of its training/evaluation replies. The server uses `capacity_score.py` to compute a score between `0` and `1` based on thresholds for battery, CPU, memory, and network latency. This score is then mapped to a number of local epochs via `score_to_local_epochs()`:

- `score = 0` → **Skip** this round (with a human‑readable explanation).
- `0 < score < 1` → Run `max(1, int(base_local_epochs * score))` epochs.
- `score = 1` → Run `base_local_epochs` epochs (default 2).

The explanation logic (`_explain_low_score`) generates a dashboard message describing why a node was skipped (e.g., *“battery critically low (8%, unplugged)”*).

### Fault‑Tolerant Stable Reassignment

The strategy maintains a `partition_survivor` dictionary mapping each dropped partition to the node currently covering it. During `configure_train`:

1. It determines which nodes that own partitions (`node_to_real_partition`) are **not** present in the current list of available nodes.
2. For each newly dropped node, a **drop event** is emitted exactly once (via `notified_drops`).
3. For each dropped partition, it checks if the current survivor (`partition_survivor`) is still alive:
   - **If yes:** The partition stays with that survivor.
   - **If no:** A new survivor is chosen deterministically (lowest `node_id`) and the mapping is updated.
4. The survivor’s training configuration is updated to include the extra partition IDs (`extra-partition-ids`). The client then loads both its own data and the dropped partition’s data for training.

**De‑duplication** is achieved with `last_reassignment_snapshot`, which stores a `frozenset` of partitions per survivor. A reassign event is only published when the snapshot changes.

### Device Identity & Reconnection

Each client sends a `telem_device_id` (a stable hash of the hostname). The strategy tracks the mapping `device_to_node`. If a device reappears under a new `node_id` (e.g., after a reboot), the system:

- Removes the stale partition mapping for the old node.
- Discards any pending drop notification for the old node.
- Marks the old node as `retired`.
- Emits a **reconnect event** explaining that the device reconnected and its partition no longer needs covering.

This prevents treating the reconnecting device as a brand‑new participant and ensures its original partition is restored smoothly.

### Dashboard & Event Feed

All node states and events are sent to a simple dashboard via `dashboard_state.py` (using `update_node` and `add_event`). The dashboard (`dashboard_app.py`) displays:

- Live per‑node metrics (CPU, battery, latency, etc.)
- A chronological event feed with icons (🔴 Drop, 🔄 Reassign, 🔁 Reconnect, ⏭️ Skip)
- Current status of each node (Training, Skipped, Dropped, Reconnected, Retired)

The event feed is generic and requires no schema changes when new event types are added.

---

## Research Context & Positioning

This project is grounded in a survey of 14 individual methods and 7 broader surveys spanning foundational FL, aggregation, model‑architecture adaptation, client selection, fault tolerance, and asynchronous FL. The key finding of that survey is a **recognised gap** in existing literature:

> Existing selection and scheduling methods generally assume that a client, once selected, remains available for the duration of a round. They do **not** simultaneously combine:
> 1. **Live, per‑round telemetry** (battery, CPU, memory, network) as a continuous scheduling signal.
> 2. **Graduated workload response** (reduce epochs rather than binary include/exclude).
> 3. **Explicit reassignment of a dropped client’s data partition** to a surviving client.
> 4. **Identity‑aware recovery** that recognises a reconnecting device as the same participant it was before.

HALO implements all four of these together. The closest existing work includes FLuID (runtime sub‑model resizing) and asynchronous fault‑tolerant methods (e.g., Morell & Alba), but HALO differs by remaining synchronous while adding live telemetry‑driven scoring and identity‑aware recovery. This positions HALO as a novel contribution at the intersection of adaptive scheduling and fault tolerance.

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip
- Git
- (Optional) A CUDA‑capable GPU for faster training

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ArnabTechiee/HALO-FL-Scheduler.git
   cd HALO-FL-Scheduler
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv halo-env
   source halo-env/bin/activate   # On Windows: halo-env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

> **Note:** This project is built with `flwr==1.32.1`. If you use a different version, adjust the code accordingly.

### Running the Demo

The demo runs a two‑client simulation on CIFAR‑10. Start the Flower local deployment with:

```bash
flwr run . local-deploy --stream
```

**What to expect:**

- **Round 1:** Both nodes train normally with default epochs (2).
- **Simulate a Drop:** Manually kill one client process (or wait for heartbeat timeout). The strategy detects the drop, emits a drop event, and reassigns the partition to the surviving node.
- **Stable Reassignment:** The partition stays with the survivor in subsequent rounds; no bouncing.
- **Simulate a Reconnect:** Restart the dropped client. The system recognises it, restores its original partition, and emits a reconnect event.
- **Adaptive Workload:** Observe a node’s `local_epochs` decrease if its CPU spikes (e.g., from 2 to 1 when CPU > 60%).

A live dashboard is available (check console for URL) showing node states and events in real time.

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
| **Run 1 (Correct)** | Node dropped after round 2. Partition stably reassigned to survivor. Device reconnected and restored partition. No duplicate events. | ✅ |
| **Run 2 (Baseline)** | Partition bounced to a newly connected node even though the original survivor was alive. | ❌ *(Bug fixed)* |

The adaptive workload successfully prevented overloading: when a node’s CPU reached 78%, its local epochs automatically reduced from 2 to 1.

**Final metrics (from Run 1):**
- Final accuracy: ~10% (random guessing baseline for CIFAR‑10; focus is on system behaviour, not model performance).
- Training completed without manual intervention.

---

## Configuration

Key parameters that can be adjusted:

- **`base_local_epochs`:** Default number of epochs for a healthy node (set in `AdaptiveFedAvg.__init__`).
- **`fraction_train` / `fraction_evaluate`:** Fraction of nodes sampled per round (configured in `pyproject.toml`).
- **Telemetry thresholds:** In `capacity_score.py` (e.g., CPU > 85%, battery < 15% unplugged) to determine skip/low‑score conditions.

To change the dataset or model, modify `client_app.py` and the corresponding data loading utilities.

---

## Troubleshooting

- **Dashboard not showing:** Ensure `dashboard_app.py` is running and the port is accessible. Check the console for the URL.
- **`telem_device_id` missing:** The client must send it in the metrics. Verify `telemetry.py` is correctly integrated.
- **Stable reassignment not working:** Check that `partition_survivor` is preserved across rounds. If using multiple server processes, consider a persistent store (e.g., Redis).
- **Node drop not detected:** Flower’s heartbeat timeout may need adjustment (see `clientapp.py` heartbeat settings).

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request. For major changes, discuss them in an issue first.

---

## License

This project is licensed under the **MIT License**.

---

## Contact

**Arnab Mondal** – [GitHub Profile](https://github.com/ArnabTechiee)

**Project Link:** [https://github.com/ArnabTechiee/HALO-FL-Scheduler](https://github.com/ArnabTechiee/HALO-FL-Scheduler)
```

