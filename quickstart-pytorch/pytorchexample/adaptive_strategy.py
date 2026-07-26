"""
HALO Adaptive Strategy
-----------------------
Extends Flower's FedAvg to set per-client local_epochs (or skip a client
entirely) based on that client's most recently reported telemetry.
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
    """FedAvg that adapts per-client workload using last-round telemetry."""

    def __init__(self, *args, base_local_epochs: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_local_epochs = base_local_epochs
        self.last_known_telemetry: dict[int, dict] = {}  # node_id -> flat telem_* dict

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid
    ) -> Iterable[Message]:
        if not self.last_known_telemetry:
            return super().configure_train(server_round, arrays, config, grid)

        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)

        messages = []
        skipped = []
        for node_id in node_ids:
            telemetry = self.last_known_telemetry.get(node_id)

            if telemetry is None:
                local_epochs = self.base_local_epochs
                print(f"[HALO DEBUG] node {node_id} -> no telemetry yet, using base local_epochs={local_epochs}")
            else:
                # Added raw telemetry print before scoring
                print(f"[HALO RAW] node {node_id} telemetry going into scorer: {telemetry}")
                score = compute_capacity_score(telemetry)
                local_epochs = score_to_local_epochs(score, self.base_local_epochs)
                print(f"[HALO DEBUG] node {node_id} -> score={score:.2f}, local_epochs={local_epochs}")

            if local_epochs == 0:
                skipped.append(node_id)
                continue

            per_client_config = ConfigRecord(dict(config))
            per_client_config["server-round"] = server_round
            per_client_config["local-epochs"] = local_epochs

            record = RecordDict(
                {self.arrayrecord_key: arrays, self.configrecord_key: per_client_config}
            )
            messages.append(
                Message(content=record, message_type=MessageType.TRAIN, dst_node_id=node_id)
            )

        log(
            INFO,
            "configure_train: %s nodes training, %s skipped on low capacity (out of %s)",
            len(messages), len(skipped), len(num_total),
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
            # capacity_score.py now consumes the flat telem_* fields directly —
            # no unflatten step needed, store as-is
            self.last_known_telemetry[msg.metadata.src_node_id] = dict(metrics)
            print(f"[HALO DEBUG] node {msg.metadata.src_node_id} -> {self.last_known_telemetry[msg.metadata.src_node_id]}")

        return super().aggregate_train(server_round, replies)