"""flower_DP: Flower Example using Differential Privacy and Secure Aggregation."""

from typing import List, Tuple

from flower_DP.task import Net, get_weights

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import Driver, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import DifferentialPrivacyClientSideFixedClipping, FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    examples = [num_examples for num_examples, _ in metrics]
    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    train_accuracies = [
        num_examples * m["train_accuracy"] for num_examples, m in metrics
    ]
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]
    val_accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics]

    return {
        "train_loss": sum(train_losses) / sum(examples),
        "train_accuracy": sum(train_accuracies) / sum(examples),
        "val_loss": sum(val_losses) / sum(examples),
        "val_accuracy": sum(val_accuracies) / sum(examples),
    }


app = ServerApp()


@app.main()
def main(driver: Driver, context: Context) -> None:

    # Initialize global model
    model_weights = get_weights(Net())
    parameters = ndarrays_to_parameters(model_weights)

    # Read from config:
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]

    # Note: The fraction_fit value is configured based on the DP hyperparameter `num-sampled-clients`.
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=2,
        fit_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )

    # Config differential privacy parameters
    noise_multiplier = context.run_config["noise-multiplier"]
    clipping_norm = context.run_config["clipping-norm"]
    num_sampled_clients = context.run_config["num-sampled-clients"]

    strategy = DifferentialPrivacyClientSideFixedClipping(
        strategy,
        noise_multiplier=noise_multiplier,
        clipping_norm=clipping_norm,
        num_sampled_clients=num_sampled_clients,
    )

    # Construct the LegacyContext
    context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # Create the train/evaluate workflow
    workflow = DefaultWorkflow(
        fit_workflow=SecAggPlusWorkflow(
            num_shares=context.run_config["num-shares"],
            reconstruction_threshold=context.run_config["reconstruction-threshold"],
        )
    )

    # Execute
    workflow(driver, context)