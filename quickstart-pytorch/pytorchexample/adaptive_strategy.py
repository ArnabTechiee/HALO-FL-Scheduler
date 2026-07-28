"""
HALO Adaptive Strategy with Fault-Tolerant Partition Reassignment
------------------------------------------------------------------
Extends Flower's FedAvg to:
1. Set per-client local_epochs (or skip) based on telemetry.
2. Reassign a genuinely dropped node's data partition to a surviving
   node, so that partition's data isn't lost for the round.
Tested against flwr==1.32.1.
"""

from collections.abc import Iterable
from logging import INFO

from pytorchexample.capacity_score import compute_capacity_score, score_to_local_epochs

from flwr.app import ArrayRecord, ConfigRecord, Message, MessageType, RecordDict
from flwr.common import log
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import sample_nodes


class AdaptiveFedAvg(FedAvg):
    """FedAvg that adapts per-client workload using last-round telemetry,
    and reassigns dropped nodes' data partitions to survivors."""

    def __init__(self, *args, base_local_epochs: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_local_epochs = base_local_epochs
        self.last_known_telemetry: dict[int, dict] = {}
        # Permanent partition mapping for the whole run
        self.partition_assignment: dict[int, int] = {}
        self.next_partition_id = 0

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid
    ) -> Iterable[Message]:
        # Sample nodes for this round
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)
        node_ids, all_available_node_ids = sample_nodes(
            grid, self.min_available_nodes, sample_size
        )

        # Assign a permanent partition to any node we haven't seen before
        for node_id in all_available_node_ids:
            if node_id not in self.partition_assignment:
                self.partition_assignment[node_id] = self.next_partition_id
                self.next_partition_id += 1

        messages = []
        skipped = []
        for node_id in node_ids:
            telemetry = self.last_known_telemetry.get(node_id)
            if telemetry is None:
                # No telemetry yet -> use base epochs (round 1, or newly joined node)
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

        # --- Fault-tolerant reassignment ---
        # A node is "genuinely dropped" if it was ever assigned a partition
        # but is absent from all_available_node_ids this round (disconnected).
        dropped_node_ids = set(self.partition_assignment) - set(all_available_node_ids)

        if dropped_node_ids and messages:
            survivor = messages[0]   # reassign all dropped partitions to first survivor
            extra_partitions = [self.partition_assignment[n] for n in dropped_node_ids]
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
            # Store telemetry as-is (flat telem_* fields)
            self.last_known_telemetry[msg.metadata.src_node_id] = dict(metrics)
            print(f"[HALO DEBUG] node {msg.metadata.src_node_id} -> {self.last_known_telemetry[msg.metadata.src_node_id]}")
        return super().aggregate_train(server_round, replies)

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
        replies = list(replies)
        for msg in replies:
            if msg.has_error():
                continue
            metrics = msg.content.get("metrics")
            if metrics is None:
                continue
            # Also collect telemetry from evaluation replies (allows skipped clients to recover)
            telemetry = {k: v for k, v in dict(metrics).items() if k.startswith("telem_")}
            if telemetry:
                self.last_known_telemetry[msg.metadata.src_node_id] = telemetry
        return super().aggregate_evaluate(server_round, replies)