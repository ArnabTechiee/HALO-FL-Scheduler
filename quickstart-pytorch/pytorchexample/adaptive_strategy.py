"""
HALO Adaptive Strategy with Fault-Tolerant Partition Reassignment
------------------------------------------------------------------
Extends Flower's FedAvg to:
1. Set per-client local_epochs (or skip) based on telemetry.
2. Reassign a genuinely dropped node's data partition to a surviving
   node, using the client's own reported partition-id (never fabricated).
3. Recognize a reconnecting device (via a stable hostname-derived ID) as
   the same physical machine, restoring its original partition instead
   of treating it as a brand-new participant.
Tested against flwr==1.32.1.
"""

from collections.abc import Iterable
from logging import INFO

from pytorchexample.capacity_score import compute_capacity_score, score_to_local_epochs

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,          # needed for _strip_identity_fields
    RecordDict,
)
from flwr.common import log
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import sample_nodes


class AdaptiveFedAvg(FedAvg):
    """FedAvg that adapts per-client workload using last-round telemetry,
    reassigns dropped nodes' partitions to survivors, and recognizes
    reconnecting devices by a stable hostname-derived identity."""

    def __init__(self, *args, base_local_epochs: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_local_epochs = base_local_epochs
        self.last_known_telemetry: dict[int, dict] = {}

        # node_id -> the REAL partition-id that node itself reported
        self.node_to_real_partition: dict[int, int] = {}
        self.node_to_device: dict[int, int] = {}          # node_id -> device_id
        self.device_to_node: dict[int, int] = {}           # device_id -> current node_id

    def _track_device_identity(self, node_id: int, metrics: dict):
        """Track which physical device a node_id corresponds to.
        If a device reconnects under a new node_id, remove the stale
        partition mapping for the old node_id so we stop treating
        that partition as dropped."""
        device_id = metrics.get("telem_device_id")
        if device_id is None:
            return
        device_id = int(device_id)
        self.node_to_device[node_id] = device_id

        previous_node_id = self.device_to_node.get(device_id)
        if previous_node_id is not None and previous_node_id != node_id:
            # Same physical device reconnected under a new node_id.
            # Drop its old node_id's partition mapping.
            stale = self.node_to_real_partition.pop(previous_node_id, None)
            if stale is not None:
                print(f"[HALO] Device {device_id} reconnected as node {node_id} "
                      f"(was {previous_node_id}) — partition {stale} no longer dropped")
        self.device_to_node[device_id] = node_id

    def _strip_identity_fields(self, msg: Message):
        """Remove per-client identity fields that are meaningless when averaged
        (device/partition IDs) so that Flower's built-in aggregation doesn't
        blend them into a nonsense number."""
        metrics = msg.content.get("metrics")
        if metrics is None:
            return
        cleaned = {
            k: v for k, v in dict(metrics).items()
            if k not in ("telem_device_id", "telem_partition_id")
        }
        msg.content["metrics"] = MetricRecord(cleaned)

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid
    ) -> Iterable[Message]:
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

            if local_epochs == 0:
                print(f"[HALO] Round {server_round}: SKIPPING node {node_id} (score={score})")
                skipped.append(node_id)
                continue

            print(f"[HALO] Round {server_round}: node {node_id} score={score}, local_epochs={local_epochs}")

            per_client_config = ConfigRecord(dict(config))
            per_client_config["server-round"] = server_round
            per_client_config["local-epochs"] = local_epochs

            record = RecordDict(
                {self.arrayrecord_key: arrays, self.configrecord_key: per_client_config}
            )
            messages.append(
                Message(content=record, message_type=MessageType.TRAIN, dst_node_id=node_id)
            )

        # Only nodes we've heard a REAL, client-reported partition from
        # are eligible for drop detection.
        dropped_node_ids = set(self.node_to_real_partition) - set(all_available_node_ids)

        if dropped_node_ids and messages:
            survivor = messages[0]
            extra_partitions = [self.node_to_real_partition[n] for n in dropped_node_ids]
            existing_extra = list(survivor.content["config"].get("extra-partition-ids", []))
            survivor.content["config"]["extra-partition-ids"] = existing_extra + extra_partitions
            print(
                f"[HALO] Round {server_round}: reassigning partitions "
                f"{extra_partitions} from dropped nodes {dropped_node_ids} "
                f"to surviving node {survivor.metadata.dst_node_id}"
            )

        log(
            INFO,
            "configure_train: %s nodes training, %s skipped on low capacity (out of %s)",
            len(messages), len(skipped), len(all_available_node_ids),
        )
        return messages

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
            self._track_device_identity(node_id, metrics_dict)

            # Capture the real partition ID reported by the client itself.
            if "telem_partition_id" in metrics_dict:
                self.node_to_real_partition[node_id] = int(metrics_dict["telem_partition_id"])

            print(f"[HALO DEBUG] node {node_id} -> {metrics_dict}")

            # Strip identity fields *after* we have extracted what we need,
            # so the super() aggregation stays clean.
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
                self._track_device_identity(node_id, telemetry)

            # Keep partition tracking up‑to‑date from eval replies as well.
            if "telem_partition_id" in metrics_dict:
                self.node_to_real_partition[node_id] = int(metrics_dict["telem_partition_id"])

            # Strip identity fields so the aggregated evaluation metrics are clean.
            self._strip_identity_fields(msg)

        return super().aggregate_evaluate(server_round, replies)