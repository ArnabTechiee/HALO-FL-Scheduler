"""pytorchexample: A Flower / PyTorch app with HALO adaptive scheduling."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pytorchexample.adaptive_strategy import AdaptiveFedAvg
from pytorchexample.task import Net, load_centralized_dataset, test
from pytorchexample.dashboard_state import (
    load_state,
    record_comparison_run,
    reset_state,
    set_run_meta,
    update_round_summary,
)

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Clear stale dashboard data from a previous run
    reset_state()

    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    base_local_epochs: int = int(context.run_config.get("base-local-epochs", 3))

    # "adaptive" runs HALO; "baseline" runs stock FedAvg with fixed epochs.
    # Run once in each mode and the dashboard's comparison panel fills itself
    # in — no hand-written comparison_runs.json.
    mode: str = str(context.run_config.get("halo-mode", "adaptive")).lower()
    dataset_name: str = str(context.run_config.get("dataset-name", "CIFAR-10"))
    model_name: str = str(context.run_config.get("model-name", "CNN"))

    set_run_meta(
        dataset=dataset_name,
        model=model_name,
        mode=mode,
        base_local_epochs=base_local_epochs,
        run_id=str(context.run_id) if hasattr(context, "run_id") else None,
    )

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate on central data and push the round summary to the dashboard."""
        model = Net()
        model.load_state_dict(arrays.to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        test_dataloader = load_centralized_dataset()
        test_loss, test_acc = test(model, test_dataloader, device)

        update_round_summary(
            round_num=server_round,
            total_rounds=num_rounds,
            accuracy=test_acc,
            loss=test_loss,
        )
        return MetricRecord({"accuracy": test_acc, "loss": test_loss})

    if mode == "baseline":
        print("[HALO] Running BASELINE FedAvg (no telemetry scheduling, no reassignment)")
        strategy = FedAvg(
            fraction_evaluate=fraction_evaluate,
            min_available_nodes=1,
            min_train_nodes=1,
            min_evaluate_nodes=1,
        )
    else:
        print(f"[HALO] Running ADAPTIVE strategy (base_local_epochs={base_local_epochs})")
        strategy = AdaptiveFedAvg(
            fraction_evaluate=fraction_evaluate,
            base_local_epochs=base_local_epochs,
            min_available_nodes=1,
            min_train_nodes=1,
            min_evaluate_nodes=1,
        )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Persist this run's curve so the dashboard can plot adaptive vs baseline.
    label = "HALO adaptive" if mode == "adaptive" else "FedAvg baseline"
    try:
        record_comparison_run(label, mode, load_state().get("history", []))
    except Exception as exc:  # noqa: BLE001 - never fail a finished run on bookkeeping
        print(f"[HALO] Could not record comparison run: {exc}")

    if context.run_config["save-model"]:
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")