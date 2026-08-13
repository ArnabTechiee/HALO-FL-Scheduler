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
