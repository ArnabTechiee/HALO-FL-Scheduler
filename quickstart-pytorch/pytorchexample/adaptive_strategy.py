"""
HALO Adaptive Strategy with Fault-Tolerant Partition Reassignment
=================================================================
Extends Flower's FedAvg to:

1. Set per-device local_epochs (or skip) from live telemetry.
2. Reassign a genuinely dropped device's partition to a STABLE survivor,
   and RELEASE that coverage the moment the device comes back.
3. Recognize a reconnecting device as the same machine *before* it is
   scheduled, not after the round is over.
4. Stream a de-duplicated event feed and live per-device state to the
   dashboard.

Why identity is resolved up front
---------------------------------
Flower issues a new node_id on every SuperNode connection. The previous
implementation learned a device's identity in aggregate_train — one full
round too late — so a revived SuperNode was scheduled as an unknown new
participant, and its old node_id stayed in node_to_real_partition forever.
Because dropped nodes were computed as
    set(node_to_real_partition) - set(available_node_ids)
that stale id was "dropped" for the rest of the run and its partition was
covered by a survivor every round, permanently double-weighting that data
in FedAvg.

This version sends a lightweight QUERY probe to any unrecognised node at
the top of configure_train (handled by @app.query() in client_app.py). The
probe returns the device id, partition id and a fresh pre-training
telemetry snapshot, so:
  * a reconnect is matched to its existing record immediately,
  * coverage is released in the same round the device returns,
  * round 1 has real telemetry instead of score=None.

If the probe is unavailable (older flwr, or a node that doesn't answer),
the strategy degrades to reply-time identity binding — the old behaviour —
rather than failing the run.

Tested against flwr==1.32.1.
"""

from collections.abc import Iterable
from logging import INFO, WARNING

from pytorchexample.capacity_score import compute_capacity_report, score_to_local_epochs
from pytorchexample.dashboard_state import (
    add_event,
    merge_node,
    mark_round_start,
    state_batch,
    update_node,
)

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

IDENTITY_FIELDS = ("telem_device_id", "telem_partition_id")


class AdaptiveFedAvg(FedAvg):
    """FedAvg that adapts per-device workload from telemetry, covers dropped
    partitions with a stable survivor, and tracks devices across reconnects."""

    def __init__(
        self,
        *args,
        base_local_epochs: int = 3,
        probe_timeout: float = 30.0,
        enable_probe: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.base_local_epochs = base_local_epochs
        self.probe_timeout = probe_timeout
        self.enable_probe = enable_probe
        self._probe_supported = enable_probe

        # Everything below is keyed by DEVICE id, never by node_id.
        self.device_node: dict[int, int] = {}          # device -> current node_id
        self.node_device: dict[int, int] = {}          # node_id -> device
        self.device_partition: dict[int, int] = {}     # device -> its own partition
        self.device_telemetry: dict[int, dict] = {}    # device -> last telemetry
        self.partition_survivor: dict[int, int] = {}   # partition -> covering device
        self.notified_drops: set[int] = set()          # devices already announced dropped
        self.missing_rounds: dict[int, int] = {}       # device -> consecutive absences
        self.last_cover_snapshot: dict[int, frozenset] = {}

    # ── identity ─────────────────────────────────────────────────
    @staticmethod
    def _provisional_key(node_id) -> str:
        """Dashboard key for a node whose device identity isn't known yet."""
        return f"n{node_id}"

    def _bind_identity(self, node_id: int, metrics: dict, server_round: int) -> int | None:
        """Attach a node_id to its stable device. Returns the device id."""
        raw = metrics.get("telem_device_id")
        if raw is None:
            return None
        device_id = int(raw)

        previous_node = self.device_node.get(device_id)
        if previous_node is not None and previous_node != node_id:
            # Same machine, new connection. Drop the stale node mapping so it
            # can never be counted as a separate dropped participant.
            self.node_device.pop(previous_node, None)

        self.device_node[device_id] = node_id
        self.node_device[node_id] = device_id

        partition = metrics.get("telem_partition_id")
        if partition is not None:
            self.device_partition[device_id] = int(partition)

        telemetry = {k: v for k, v in metrics.items() if k.startswith("telem_")}
        if telemetry:
            self.device_telemetry[device_id] = telemetry

        # If this device was first seen under a provisional key, fold that
        # record into the real one so the dashboard card isn't duplicated.
        merge_node(self._provisional_key(node_id), device_id)
        return device_id

    def _probe_unknown(self, grid, node_ids, server_round: int):
        """Ask unrecognised nodes who they are, before scheduling them."""
        unknown = [nid for nid in node_ids if nid not in self.node_device]
        if not unknown or not self._probe_supported:
            return

        try:
            probes = [
                Message(
                    content=RecordDict({"config": ConfigRecord({"halo-probe": 1})}),
                    message_type=MessageType.QUERY,
                    dst_node_id=nid,
                )
                for nid in unknown
            ]
            log(INFO, "HALO probe: asking %s unrecognised node(s) to identify "
                      "themselves (timeout %ss)", len(unknown), self.probe_timeout)
            replies = list(grid.send_and_receive(probes, timeout=self.probe_timeout))
        except Exception as exc:  # noqa: BLE001 - any transport/version problem
            self._probe_supported = False
            log(WARNING, "HALO identity probe unavailable (%s); "
                         "falling back to reply-time identity binding", exc)
            return

        if not replies:
            # Silence, not errors: the SuperNodes never answered the QUERY at
            # all. Most likely the installed FAB predates the @app.query()
            # handler in client_app.py — reinstall the app on every node.
            log(WARNING, "HALO probe: sent %s query message(s), received 0 replies. "
                         "Devices will be identified from their train replies "
                         "instead, one round later. Check that client_app.py has "
                         "@app.query() and that the app was reinstalled.", len(unknown))
            return

        answered, failures = 0, []
        for reply in replies:
            if reply.has_error():
                failures.append(self._error_text(reply))
                continue
            metrics = reply.content.get("metrics")
            if metrics is None:
                failures.append("reply carried no metrics record")
                continue
            if self._bind_identity(reply.metadata.src_node_id, dict(metrics), server_round):
                answered += 1
            else:
                failures.append("reply carried no telem_device_id")

        log(INFO, "HALO probe: %s reply/replies received, %s identified",
            len(replies), answered)
        if failures:
            # Silence here is why round 1 schedules every device with the
            # default epoch budget (score=None). Say what actually went wrong
            # instead of degrading quietly.
            log(WARNING, "HALO probe: %s/%s node(s) did not answer (%s). "
                         "Those devices are scheduled with defaults this round.",
                len(failures), len(unknown), "; ".join(sorted(set(failures))[:3]))

    def _handle_returns(self, available_devices: set, server_round: int):
        """A previously dropped device is back: release its coverage now."""
        for device_id in sorted(available_devices & self.notified_drops):
            node_id = self.device_node.get(device_id)
            partition = self.device_partition.get(device_id)

            self.notified_drops.discard(device_id)
            self.missing_rounds[device_id] = 0
            if partition is not None:
                self.partition_survivor.pop(partition, None)

            msg = (f"Device {device_id} reconnected as node {node_id}"
                   + (f" — partition {partition} no longer needs covering"
                      if partition is not None else ""))
            print(f"[HALO] Round {server_round}: {msg}")
            update_node(device_id, status="reconnected", node_id=node_id,
                        last_round=server_round, missing_rounds=0)
            add_event("reconnect", device_id, msg, server_round)

    def _handle_drops(self, available_devices: set, server_round: int) -> set:
        """Return the set of devices currently considered dropped."""
        dropped = set(self.device_node) - available_devices
        for device_id in sorted(dropped):
            self.missing_rounds[device_id] = self.missing_rounds.get(device_id, 0) + 1
            partition = self.device_partition.get(device_id)
            update_node(device_id, status="dropped", last_round=server_round,
                        partition_id=partition,
                        missing_rounds=self.missing_rounds[device_id])

            if device_id in self.notified_drops:
                continue  # already announced; don't spam the feed each round

            telem = self.device_telemetry.get(device_id, {})
            summary = (f"CPU {telem.get('telem_cpu_percent', '?')}%, "
                       f"battery {telem.get('telem_battery_percent', '?')}%, "
                       f"latency {telem.get('telem_network_latency_ms', '?')}ms")
            msg = (f"Device {device_id} dropped (partition {partition}) — "
                   f"last seen: {summary}")
            print(f"[HALO] Round {server_round}: {msg}")
            add_event("drop", device_id, msg, server_round)
            self.notified_drops.add(device_id)
        return dropped

    def _assign_coverage(self, dropped_devices, training_devices, messages, server_round):
        """Give each orphaned partition to a stable surviving device."""
        survivor_to_partitions: dict[int, list[int]] = {}

        for device_id in sorted(dropped_devices):
            partition = self.device_partition.get(device_id)
            if partition is None:
                continue

            incumbent = self.partition_survivor.get(partition)
            if incumbent is not None and incumbent in training_devices:
                survivor = incumbent           # stable: don't bounce it around
            elif training_devices:
                # Deterministic pick so identical state gives identical output.
                survivor = min(training_devices)
                self.partition_survivor[partition] = survivor
            else:
                continue                       # nobody left to cover it

            survivor_to_partitions.setdefault(survivor, []).append(partition)

        for survivor, partitions in survivor_to_partitions.items():
            node_id = self.device_node.get(survivor)
            message = next((m for m in messages if m.metadata.dst_node_id == node_id), None)
            if message is None:
                continue

            existing = list(message.content["config"].get("extra-partition-ids", []))
            merged = sorted(set(existing) | set(partitions))
            message.content["config"]["extra-partition-ids"] = merged
            update_node(survivor, covering_partitions=merged)

            snapshot = frozenset(merged)
            if self.last_cover_snapshot.get(survivor) != snapshot:
                print(f"[HALO] Round {server_round}: device {survivor} now covering "
                      f"partition(s) {merged}")
                add_event(
                    "reassign", survivor,
                    f"Device {survivor} now covering partition(s) {merged} "
                    f"— training on {len(merged) + 1}x its own data",
                    server_round,
                )
                self.last_cover_snapshot[survivor] = snapshot

        # Anyone who was covering and no longer is gets cleared explicitly.
        for survivor in list(self.last_cover_snapshot):
            if survivor not in survivor_to_partitions:
                self.last_cover_snapshot.pop(survivor, None)
                update_node(survivor, covering_partitions=[])

    @staticmethod
    def _error_text(msg: Message) -> str:
        err = getattr(msg, "error", None)
        if err is None:
            return "unknown error"
        return str(getattr(err, "reason", None) or err)

    def _note_failed_reply(self, msg: Message, server_round: int):
        """A node that errored out of this round is already gone.

        Waiting for the next configure_train to notice means the dashboard
        shows it as 'training' for a full round after it died — which is
        exactly what happened in round 2 of your last run.
        """
        node_id = getattr(msg.metadata, "src_node_id", None)
        device_id = self.node_device.get(node_id)
        if device_id is None or device_id in self.notified_drops:
            return

        partition = self.device_partition.get(device_id)
        update_node(device_id, status="dropped", last_round=server_round,
                    partition_id=partition)
        add_event("drop", device_id,
                  f"Device {device_id} dropped mid-round (partition {partition}) "
                  f"— {self._error_text(msg)}", server_round)
        self.notified_drops.add(device_id)
        print(f"[HALO] Round {server_round}: device {device_id} failed mid-round "
              f"— {self._error_text(msg)}")

    @staticmethod
    def _strip_identity_fields(msg: Message):
        """Keep device/partition ids out of FedAvg's weighted metric average.

        Averaging an id produces a meaningless number that then shows up in
        the aggregated MetricRecord.
        """
        metrics = msg.content.get("metrics")
        if metrics is None:
            return
        cleaned = {k: v for k, v in dict(metrics).items() if k not in IDENTITY_FIELDS}
        msg.content["metrics"] = MetricRecord(cleaned)

    # ── round configuration ──────────────────────────────────────
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid
    ) -> Iterable[Message]:
        with state_batch():
            mark_round_start(server_round)

            num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
            sample_size = max(num_nodes, self.min_train_nodes)
            node_ids, all_available_node_ids = sample_nodes(
                grid, self.min_available_nodes, sample_size
            )

            # Resolve identity BEFORE any scheduling decision.
            self._probe_unknown(grid, all_available_node_ids, server_round)

            available_devices = {
                self.node_device[nid]
                for nid in all_available_node_ids
                if nid in self.node_device
            }
            self._handle_returns(available_devices, server_round)

            messages: list[Message] = []
            training_devices: set[int] = set()
            skipped = 0

            for node_id in node_ids:
                device_id = self.node_device.get(node_id)
                key = device_id if device_id is not None else self._provisional_key(node_id)
                telemetry = self.device_telemetry.get(device_id) if device_id else None
                partition = self.device_partition.get(device_id) if device_id else None

                report = compute_capacity_report(telemetry)
                score = report["score"]
                local_epochs = score_to_local_epochs(score, self.base_local_epochs)

                if local_epochs == 0:
                    print(f"[HALO] Round {server_round}: SKIPPING device {key} "
                          f"(score={score}, {report['reason']})")
                    update_node(key, node_id=node_id, device_id=device_id,
                                partition_id=partition, score=score,
                                score_parts=report["parts"], score_reason=report["reason"],
                                local_epochs=0, status="skipped",
                                last_round=server_round, covering_partitions=[])
                    add_event("skip", key,
                              f"Device {key} skipped this round — capacity {score} "
                              f"({report['reason']})", server_round)
                    skipped += 1
                    continue

                print(f"[HALO] Round {server_round}: device {key} score={score}, "
                      f"local_epochs={local_epochs}")
                update_node(key, node_id=node_id, device_id=device_id,
                            partition_id=partition, score=score,
                            score_parts=report["parts"], score_reason=report["reason"],
                            local_epochs=local_epochs, status="training",
                            last_round=server_round, missing_rounds=0,
                            covering_partitions=[])

                per_client_config = ConfigRecord(dict(config))
                per_client_config["server-round"] = server_round
                per_client_config["local-epochs"] = local_epochs

                record = RecordDict({
                    self.arrayrecord_key: arrays,
                    self.configrecord_key: per_client_config,
                })
                messages.append(Message(content=record,
                                        message_type=MessageType.TRAIN,
                                        dst_node_id=node_id))
                if device_id is not None:
                    training_devices.add(device_id)

            dropped_devices = self._handle_drops(available_devices, server_round)
            self._assign_coverage(dropped_devices, training_devices, messages, server_round)

            log(INFO,
                "configure_train: %s nodes training, %s skipped on low capacity "
                "(out of %s available, %s device(s) dropped)",
                len(messages), skipped, len(all_available_node_ids), len(dropped_devices))
            return messages

    # ── aggregation ──────────────────────────────────────────────
    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        replies = list(replies)
        with state_batch():
            for msg in replies:
                if msg.has_error():
                    self._note_failed_reply(msg, server_round)
                    continue
                metrics = msg.content.get("metrics")
                if metrics is None:
                    continue

                metrics_dict = dict(metrics)
                node_id = msg.metadata.src_node_id
                device_id = self._bind_identity(node_id, metrics_dict, server_round)
                key = device_id if device_id is not None else self._provisional_key(node_id)

                update_node(
                    key,
                    node_id=node_id,
                    device_id=device_id,
                    partition_id=metrics_dict.get("telem_partition_id"),
                    num_examples=metrics_dict.get("num-examples"),
                    train_loss=metrics_dict.get("train_loss"),
                    cpu_percent=metrics_dict.get("telem_cpu_percent"),
                    mem_percent_used=metrics_dict.get("telem_mem_percent_used"),
                    battery_percent=metrics_dict.get("telem_battery_percent"),
                    battery_plugged_in=bool(metrics_dict.get("telem_battery_plugged_in")),
                    network_latency_ms=metrics_dict.get("telem_network_latency_ms"),
                    status="active",
                    last_round=server_round,
                    missing_rounds=0,
                )
                self._strip_identity_fields(msg)

        return super().aggregate_train(server_round, replies)

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
        replies = list(replies)
        with state_batch():
            for msg in replies:
                if msg.has_error():
                    continue
                metrics = msg.content.get("metrics")
                if metrics is None:
                    continue

                metrics_dict = dict(metrics)
                node_id = msg.metadata.src_node_id
                # Refresh telemetry here too: a device skipped from training
                # never sends a train reply, so without this its readings
                # would freeze at whatever caused it to be skipped.
                device_id = self._bind_identity(node_id, metrics_dict, server_round)
                key = device_id if device_id is not None else self._provisional_key(node_id)

                update_node(
                    key,
                    node_id=node_id,
                    device_id=device_id,
                    cpu_percent=metrics_dict.get("telem_cpu_percent"),
                    mem_percent_used=metrics_dict.get("telem_mem_percent_used"),
                    battery_percent=metrics_dict.get("telem_battery_percent"),
                    battery_plugged_in=bool(metrics_dict.get("telem_battery_plugged_in")),
                    network_latency_ms=metrics_dict.get("telem_network_latency_ms"),
                    eval_acc=metrics_dict.get("eval_acc"),
                    last_round=server_round,
                )
                self._strip_identity_fields(msg)

        return super().aggregate_evaluate(server_round, replies)