"""flower_DP: A Flower / PyTorch app."""

from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import (
    Driver,
    LegacyContext,
    ServerApp,
    ServerConfig,
)
from flwr.server.strategy import DifferentialPrivacyClientSideFixedClipping, FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

import sys
sys.path.append('../flower/flower_DP')
from centralized import Net, get_weights


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    print(metrics)
    accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


app = ServerApp()


@app.main()
def main(driver: Driver, context: Context) -> None:

    # Initialize global model
    model_weights = get_weights(Net())
    parameters = ndarrays_to_parameters(model_weights)

    # Note: The fraction_fit value is configured based on the DP hyperparameter `num-sampled-clients`.
    strategy = FedAvg(
        fraction_fit=0.2,
        fraction_evaluate=0.0,
        min_fit_clients=20,
        fit_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )

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
        config=ServerConfig(num_rounds=3),
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