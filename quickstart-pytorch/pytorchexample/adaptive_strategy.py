"""
HALO Adaptive Strategy with Fault-Tolerant Partition Reassignment
------------------------------------------------------------------
Extends Flower's FedAvg to:
1. Set per-client local_epochs (or skip) based on telemetry.
2. Reassign a genuinely dropped node's data partition to a STABLE
   surviving node — once a partition is assigned to a survivor, it stays
   with that survivor as long as the survivor remains connected, instead
   of bouncing between whichever node happens to be first in that
   round's message list.
3. Recognize a reconnecting device (via a stable hostname-derived ID) as
   the same physical machine, restoring its original partition instead
   of treating it as a brand-new participant.
4. Publish a detailed, de-duplicated event feed (drop / reassign /
   reconnect / skip) plus live per-node state to the dashboard.
Tested against flwr==1.32.1.
"""

from collections.abc import Iterable
from logging import INFO

from pytorchexample.capacity_score import compute_capacity_score, score_to_local_epochs
from pytorchexample.dashboard_state import update_node, add_event, mark_round_start

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
)
from flwr.common import log
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import sample_nodes


def _explain_low_score(telemetry: dict) -> str:
    """Best-effort, human-readable explanation of why a node's capacity
    score is reduced/zero, for display in the event feed. Mirrors
    capacity_score.py's thresholds for display purposes only — the
    actual scoring decision is made there, this just narrates it."""
    reasons = []
    if telemetry is None:
        return "no telemetry yet"

    battery_pct = telemetry.get("telem_battery_percent")
    plugged_in = telemetry.get("telem_battery_plugged_in")
    has_battery = telemetry.get("telem_has_battery")
    if has_battery and battery_pct is not None and not plugged_in:
        if battery_pct < 15:
            reasons.append(f"battery critically low ({battery_pct}%, unplugged)")
        elif battery_pct < 30:
            reasons.append(f"battery low ({battery_pct}%, unplugged)")

    cpu = telemetry.get("telem_cpu_percent")
    if cpu is not None:
        if cpu > 85:
            reasons.append(f"CPU very high ({cpu}%)")
        elif cpu > 60:
            reasons.append(f"CPU elevated ({cpu}%)")

    mem = telemetry.get("telem_mem_percent_used")
    if mem is not None and mem > 90:
        reasons.append(f"memory pressure ({mem}%)")

    latency = telemetry.get("telem_network_latency_ms")
    if latency == -1:
        reasons.append("network unreachable")
    elif latency is not None and latency > 300:
        reasons.append(f"high network latency ({latency}ms)")

    return "; ".join(reasons) if reasons else "conditions healthy"


class AdaptiveFedAvg(FedAvg):
    """FedAvg that adapts per-client workload using last-round telemetry,
    reassigns dropped nodes' partitions to a STABLE survivor, recognizes
    reconnecting devices by stable hostname-derived identity, and streams
    a detailed event feed + live node state to the dashboard."""

    def __init__(self, *args, base_local_epochs: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_local_epochs = base_local_epochs
        self.last_known_telemetry: dict[int, dict] = {}
        self.node_to_real_partition: dict[int, int] = {}
        self.node_to_device: dict[int, int] = {}
        self.device_to_node: dict[int, int] = {}

        # Stable reassignment bookkeeping
        self.partition_survivor: dict[int, int] = {}          # partition_id -> node_id currently covering it
        self.notified_drops: set[int] = set()                  # node_ids we've already emitted a "drop" event for
        self.last_reassignment_snapshot: dict[int, frozenset] = {}  # survivor_id -> frozenset(partitions) last announced

    # ------------------------------------------------------------------
    def _track_device_identity(self, node_id: int, metrics: dict, server_round: int):
        """Track which physical device a node_id corresponds to. If a
        device reconnects under a new node_id, clean up the stale mapping
        and publish a reconnect event for the dashboard."""
        device_id = metrics.get("telem_device_id")
        if device_id is None:
            return
        device_id = int(device_id)
        self.node_to_device[node_id] = device_id

        previous_node_id = self.device_to_node.get(device_id)
        if previous_node_id is not None and previous_node_id != node_id:
            stale_partition = self.node_to_real_partition.pop(previous_node_id, None)
            self.notified_drops.discard(previous_node_id)
            update_node(previous_node_id, status="retired")

            msg = f"Device {device_id} reconnected as node {node_id} (was {previous_node_id})"
            if stale_partition is not None:
                msg += f" — partition {stale_partition} no longer needs covering"
            print(f"[HALO] {msg}")

            update_node(node_id, status="reconnected", device_id=device_id,
                        last_round=server_round)
            add_event("reconnect", node_id, msg, server_round)

        self.device_to_node[device_id] = node_id

    def _strip_identity_fields(self, msg: Message):
        metrics = msg.content.get("metrics")
        if metrics is None:
            return
        cleaned = {
            k: v for k, v in dict(metrics).items()
            if k not in ("telem_device_id", "telem_partition_id")
        }
        msg.content["metrics"] = MetricRecord(cleaned)

    # ------------------------------------------------------------------
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid
    ) -> Iterable[Message]:
        mark_round_start(server_round)  # <-- NEW: lets the dashboard show live in-progress state

        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)
        node_ids, all_available_node_ids = sample_nodes(
            grid, self.min_available_nodes, sample_size
        )

        messages = []
        skipped = []
        for node_id in node_ids:
            telemetry = self.last_known_telemetry.get(node_id)
            if telemetry is None:
                local_epochs = self.base_local_epochs
                score = None
            else:
                score = compute_capacity_score(telemetry)
                local_epochs = score_to_local_epochs(score, self.base_local_epochs)

            partition_id = self.node_to_real_partition.get(node_id)

            if local_epochs == 0:
                reason = _explain_low_score(telemetry)
                print(f"[HALO] Round {server_round}: SKIPPING node {node_id} (score={score}, {reason})")
                update_node(node_id, score=score, local_epochs=0,
                            status="skipped", last_round=server_round,
                            partition_id=partition_id)
                add_event(
                    "skip", node_id,
                    f"Node {node_id} skipped this round — capacity score {score} ({reason})",
                    server_round,
                )
                skipped.append(node_id)
                continue

            print(f"[HALO] Round {server_round}: node {node_id} score={score}, local_epochs={local_epochs}")
            update_node(node_id, score=score, local_epochs=local_epochs,
                        status="training", last_round=server_round,
                        partition_id=partition_id, covering_partitions=[])

            per_client_config = ConfigRecord(dict(config))
            per_client_config["server-round"] = server_round
            per_client_config["local-epochs"] = local_epochs

            record = RecordDict(
                {self.arrayrecord_key: arrays, self.configrecord_key: per_client_config}
            )
            messages.append(
                Message(content=record, message_type=MessageType.TRAIN, dst_node_id=node_id)
            )

        # ---- Fault-tolerant, STABLE reassignment ----
        dropped_node_ids = set(self.node_to_real_partition) - set(all_available_node_ids)
        current_survivor_ids = {m.metadata.dst_node_id for m in messages}

        # Emit a "drop" event exactly once per node, the round it's first noticed
        newly_dropped = dropped_node_ids - self.notified_drops
        for dropped_id in newly_dropped:
            last_telem = self.last_known_telemetry.get(dropped_id, {})
            partition = self.node_to_real_partition.get(dropped_id)
            summary = (
                f"CPU {last_telem.get('telem_cpu_percent', '?')}%, "
                f"battery {last_telem.get('telem_battery_percent', '?')}%, "
                f"latency {last_telem.get('telem_network_latency_ms', '?')}ms"
            )
            msg = f"Node {dropped_id} dropped (partition {partition}) — last seen: {summary}"
            print(f"[HALO] Round {server_round}: {msg}")
            update_node(dropped_id, status="dropped", last_round=server_round)
            add_event("drop", dropped_id, msg, server_round)
            self.notified_drops.add(dropped_id)

        # Determine (stable) survivor per dropped partition
        survivor_to_partitions: dict[int, list[int]] = {}
        for dropped_id in dropped_node_ids:
            partition = self.node_to_real_partition.get(dropped_id)
            if partition is None:
                continue

            existing_survivor = self.partition_survivor.get(partition)
            if existing_survivor is not None and existing_survivor in current_survivor_ids:
                survivor_id = existing_survivor  # keep it stable — don't reassign unnecessarily
            elif current_survivor_ids:
                # New assignment needed: pick deterministically (lowest node_id)
                # so the same input state always produces the same choice.
                survivor_id = min(current_survivor_ids)
                self.partition_survivor[partition] = survivor_id
            else:
                continue  # nobody available to cover it this round

            survivor_to_partitions.setdefault(survivor_id, []).append(partition)

        for survivor_id, partitions in survivor_to_partitions.items():
            survivor_msg = next(m for m in messages if m.metadata.dst_node_id == survivor_id)
            existing_extra = list(survivor_msg.content["config"].get("extra-partition-ids", []))
            merged = sorted(set(existing_extra) | set(partitions))
            survivor_msg.content["config"]["extra-partition-ids"] = merged
            update_node(survivor_id, covering_partitions=merged)

            # Only emit an event if this exact assignment is new/changed —
            # avoids spamming an identical message every round a node stays dropped.
            snapshot = frozenset(merged)
            if self.last_reassignment_snapshot.get(survivor_id) != snapshot:
                print(
                    f"[HALO] Round {server_round}: reassigning partition(s) "
                    f"{merged} to surviving node {survivor_id}"
                )
                add_event(
                    "reassign", survivor_id,
                    f"Node {survivor_id} now covering partition(s) {merged} "
                    f"(now training on {len(merged) + 1}x its own data)",
                    server_round,
                )
                self.last_reassignment_snapshot[survivor_id] = snapshot

        log(
            INFO,
            "configure_train: %s nodes training, %s skipped on low capacity (out of %s)",
            len(messages), len(skipped), len(all_available_node_ids),
        )
        return messages

    # ------------------------------------------------------------------
    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        replies = list(replies)
        for msg in replies:
            if msg.has_error():
                continue
            metrics = msg.content.get("metrics")
            if metrics is None:
                continue
            metrics_dict = dict(metrics)
            node_id = msg.metadata.src_node_id
            self.last_known_telemetry[node_id] = metrics_dict
            self._track_device_identity(node_id, metrics_dict, server_round)

            if "telem_partition_id" in metrics_dict:
                self.node_to_real_partition[node_id] = int(metrics_dict["telem_partition_id"])

            print(f"[HALO DEBUG] node {node_id} -> {metrics_dict}")

            update_node(
                node_id,
                num_examples=metrics_dict.get("num-examples"),
                cpu_percent=metrics_dict.get("telem_cpu_percent"),
                battery_percent=metrics_dict.get("telem_battery_percent"),
                battery_plugged_in=bool(metrics_dict.get("telem_battery_plugged_in")),
                network_latency_ms=metrics_dict.get("telem_network_latency_ms"),
                partition_id=metrics_dict.get("telem_partition_id"),
                device_id=metrics_dict.get("telem_device_id"),
                status="active",
                last_round=server_round,
            )

            self._strip_identity_fields(msg)

        return super().aggregate_train(server_round, replies)

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
        replies = list(replies)
        for msg in replies:
            if msg.has_error():
                continue
            metrics = msg.content.get("metrics")
            if metrics is None:
                continue
            metrics_dict = dict(metrics)
            node_id = msg.metadata.src_node_id

            telemetry = {k: v for k, v in metrics_dict.items() if k.startswith("telem_")}
            if telemetry:
                self.last_known_telemetry[node_id] = telemetry
                self._track_device_identity(node_id, telemetry, server_round)

            if "telem_partition_id" in metrics_dict:
                self.node_to_real_partition[node_id] = int(metrics_dict["telem_partition_id"])

            self._strip_identity_fields(msg)

        return super().aggregate_evaluate(server_round, replies)