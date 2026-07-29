"""pytorchexample: A Flower / PyTorch app with HALO adaptive scheduling."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
# --- HALO: use the custom adaptive strategy ---
from pytorchexample.adaptive_strategy import AdaptiveFedAvg

from pytorchexample.task import Net, load_centralized_dataset, test
from pytorchexample.dashboard_state import update_round_summary, reset_state

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Clear any stale dashboard data from a previous run
    reset_state()

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Define global evaluation function that captures num_rounds for dashboard
    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate model on central data and update live dashboard."""
        model = Net()
        model.load_state_dict(arrays.to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        test_dataloader = load_centralized_dataset()
        test_loss, test_acc = test(model, test_dataloader, device)

        # Update the dashboard with the latest round summary
        update_round_summary(
            round_num=server_round,
            total_rounds=num_rounds,
            accuracy=test_acc,
            loss=test_loss,
        )

        return MetricRecord({"accuracy": test_acc, "loss": test_loss})

    # Initialize AdaptiveFedAvg strategy (uses telemetry to adapt epochs)
    strategy = AdaptiveFedAvg(
        fraction_evaluate=fraction_evaluate,
        base_local_epochs=2,
        min_available_nodes=1,
        min_train_nodes=1,
        min_evaluate_nodes=1
    )

    # Start strategy, run for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    if context.run_config["save-model"]:
        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")