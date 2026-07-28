"""pytorchexample: A Flower / PyTorch app with HALO adaptive + reassignment."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

# --- HALO: import telemetry helpers from inside the package ---
from pytorchexample.telemetry import get_telemetry_snapshot, flatten_telemetry

# --- HALO: needed for combining reassigned partitions ---
from torch.utils.data import ConcatDataset, DataLoader

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data, possibly including reassigned partitions."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the base data partition
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # HALO: if the scheduler reassigned dropped nodes' partitions to us,
    # load and combine them into one larger training set for this round
    extra_ids = msg.content["config"].get("extra-partition-ids", [])
    if extra_ids:
        datasets = [trainloader.dataset]
        for extra_id in extra_ids:
            extra_loader, _ = load_data(extra_id, num_partitions, batch_size)
            datasets.append(extra_loader.dataset)
        combined_dataset = ConcatDataset(datasets)
        trainloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
        print(f"[HALO] Training on own partition + reassigned partitions {list(extra_ids)}")

    # Call the training function with (potentially enlarged) trainloader
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

    # --- HALO: capture telemetry right after local training finishes ---
    telemetry_snapshot = get_telemetry_snapshot()
    telemetry_fields = flatten_telemetry(telemetry_snapshot)

    # 🔍 Optional debug print to verify telemetry pipeline
    # print(f"[HALO DEBUG] Telemetry sent: {telemetry_fields}")

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        **telemetry_fields,  # HALO: attach telemetry to this round's reply
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # --- HALO: refresh telemetry here too, so a client that was skipped
    # from training this round still reports current conditions — without
    # this, a skipped client's telemetry freezes forever since it never
    # trains again to send a fresh reading
    telemetry_snapshot = get_telemetry_snapshot()
    telemetry_fields = flatten_telemetry(telemetry_snapshot)

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
        **telemetry_fields,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)