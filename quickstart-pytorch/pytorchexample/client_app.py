"""pytorchexample: A Flower / PyTorch app with HALO adaptive + reassignment."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

# --- HALO: telemetry helpers ---
from pytorchexample.telemetry import get_telemetry_snapshot, flatten_telemetry, get_device_id

# --- HALO: needed for combining reassigned partitions ---
from torch.utils.data import ConcatDataset, DataLoader

app = ClientApp()


def _identity(context: Context):
    """Stable identity for this SuperNode.

    device_id is salted with this process's own partition-id so two
    SuperNodes on the same physical machine (same hostname) never collide
    into one device_id. It survives a disconnect, which is exactly what
    lets the server recognise a revived node as the same participant.
    """
    partition_id = context.node_config["partition-id"]
    return get_device_id(salt=str(partition_id)), int(partition_id)


@app.query()
def identify(msg: Message, context: Context):
    """HALO identity probe.

    The scheduler sends this to any node it doesn't recognise, at the top of
    a round, before it decides who trains. Answering it lets the server:
      * match a reconnecting SuperNode to its existing record immediately,
        instead of a round later,
      * schedule round 1 with real telemetry instead of a blind default.

    Deliberately cheap: no model, no data loading.
    """
    device_id, partition_id = _identity(context)
    telemetry = flatten_telemetry(get_telemetry_snapshot(), device_id)
    metrics = {
        "telem_device_id": device_id,
        "telem_partition_id": partition_id,
        "halo_probe_ok": 1,
        **telemetry,
    }
    return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)


@app.train()
def train(msg: Message, context: Context):
    """Train on local data, possibly including reassigned partitions."""

    device_id, partition_id = _identity(context)

    # HALO: snapshot telemetry BEFORE training starts.
    # Sampling after train_fn() returns reports the trainer, not the device
    # — every client came back at telem_cpu_percent 100.0, which flattened
    # the capacity score across the whole fleet. This reading is what the
    # scheduler will use to plan the NEXT round, so it should describe the
    # device at rest, not mid-backprop.
    telemetry_fields = flatten_telemetry(get_telemetry_snapshot(), device_id)

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # HALO: absorb partitions the scheduler reassigned to us from dropped devices
    extra_ids = msg.content["config"].get("extra-partition-ids", [])
    if extra_ids:
        datasets = [trainloader.dataset]
        for extra_id in extra_ids:
            extra_loader, _ = load_data(extra_id, num_partitions, batch_size)
            datasets.append(extra_loader.dataset)
        trainloader = DataLoader(ConcatDataset(datasets), batch_size=batch_size, shuffle=True)
        print(f"[HALO] Training on own partition + reassigned {list(extra_ids)}")

    local_epochs = msg.content["config"].get(
        "local-epochs", context.run_config["local-epochs"]
    )

    train_loss = train_fn(
        model,
        trainloader,
        local_epochs,
        msg.content["config"]["lr"],
        device,
    )

    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "telem_device_id": device_id,
        "telem_partition_id": partition_id,
        **telemetry_fields,
    }
    content = RecordDict({"arrays": model_record, "metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    device_id, partition_id = _identity(context)

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    eval_loss, eval_acc = test_fn(model, valloader, device)

    # Refresh telemetry here too: a device that was skipped from training
    # never sends a train reply, so its readings would otherwise freeze at
    # whatever got it skipped and it could never earn its way back in.
    telemetry_fields = flatten_telemetry(get_telemetry_snapshot(), device_id)

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
        "telem_device_id": device_id,
        "telem_partition_id": partition_id,
        **telemetry_fields,
    }
    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)