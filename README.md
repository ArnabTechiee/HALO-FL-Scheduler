# HALO-FL-Scheduler

**HALO** (Health-Aware, Loss-Tolerant Orchestrator) is a fault‑tolerant, adaptive federated learning scheduler built on top of the [Flower](https://flower.dev) framework. It extends the standard `FedAvg` strategy to handle real‑world edge‑device variability by:

*   **Adapting per‑client workload** (local epochs or skip) based on live telemetry (CPU, battery, memory, network).
*   **Stable reassignment of data partitions** when a node drops—the partition stays with a chosen survivor until that survivor also disappears, preventing “partition bouncing.”
*   **Recognizing reconnecting devices** via a stable hostname‑derived ID, restoring their original partition automatically.
*   **Publishing a de‑duplicated, telemetry‑enriched event feed** to a real‑time dashboard.

This project is built as an extension to Flower’s [quickstart‑pytorch](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch) example and utilizes **PyTorch** with **CIFAR‑10**.

---

## Table of Contents

1. [Key Features](#key-features)
2. [Architecture](#architecture)
3. [How It Works](#how-it-works)
    * [Adaptive Capacity Scoring](#adaptive-capacity-scoring)
    * [Fault‑Tolerant Stable Reassignment](#fault-tolerant-stable-reassignment)
    * [Device Identity & Reconnection](#device-identity--reconnection)
    * [Dashboard & Event Feed](#dashboard--event-feed)
4. [Getting Started](#getting-started)
5. [Project Structure](#project-structure)
6. [Results & Evaluation](#results--evaluation)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)
9. [Contributing](#contributing)
10. [License](#license)

---

## Key Features

*   ✅ **Telemetry‑Driven Workload Adaptation:** Every client sends real‑time system metrics (CPU %, battery level, memory usage, network latency). The server computes a capacity score and maps it to a number of local training epochs. Nodes under heavy load may run fewer epochs or be skipped entirely.
*   ✅ **Stable Partition Reassignment:** When a node drops, its data partition is assigned to a **deterministic surviving node** and stays there until that survivor also disappears. This avoids the “bouncing” problem where the partition jumps between survivors each round.
*   ✅ **One‑Time Drop Events with Context:** A drop is reported exactly once with a summary of the node’s last telemetry (e.g., “CPU 85%, battery 10%”), giving immediate insight into the cause.
*   ✅ **De‑Duplicated Reassignment Events:** Reassignment events are only emitted when the set of covered partitions for a survivor actually changes, preventing dashboard spam during prolonged outages.
*   ✅ **Device Identity Tracking:** A stable hash of the hostname (`telem_device_id`) allows the system to recognize a reconnecting device even if it receives a new ephemeral `node_id`, restoring its original partition and cleaning up stale state.
*   ✅ **Live Dashboard:** Node states and events (drop, reassign, reconnect, skip) are streamed to a simple web dashboard for real‑time monitoring.

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
    ├── Recognizes reconnecting devices
    └── Publishes events/state to dashboard
           │
           ▼
Dashboard (Flask / Simple Web)
    └── Displays live node states & event feed
