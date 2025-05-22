"""flower_normal: A Flower / PyTorch app."""

from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
# Import from task.py to initialize global model parameters
from flower_normal.task import Net, get_weights


# This function defines how metrics from multiple clients, collected during
# the centralized evaluation phase (i.e., from `FlowerClient.evaluate`), are aggregated.
# It's passed to the `evaluate_metrics_aggregation_fn` parameter of the `FedAvg` strategy.
#
# Args:
#   metrics (List[Tuple[int, Metrics]]): A list of tuples. Each tuple contains:
#     - int: The number of examples the client used for its local evaluation.
#     - Metrics (Dict[str, Scalar]): The dictionary of metrics returned by
#                                      `FlowerClient.evaluate`. In this setup, it is
#                                      expected to be {"val_accuracy": ..., "val_loss": ...}.
#
# Returns:
#   Metrics (Dict[str, Scalar]): A dictionary of aggregated metrics.
#                                In this case, {"val_accuracy": ..., "val_loss": ...}
#                                representing the weighted average of these validation metrics
#                                across all evaluating clients.
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation results from multiple clients."""
    # Multiply val_accuracy of each client by number of examples used for evaluation
    val_accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics]
    # Multiply val_loss of each client by number of examples used for evaluation
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]
    # Total number of examples used across all clients for this evaluation round
    examples = [num_examples for num_examples, _ in metrics]

    # Print the raw metrics received from clients for this evaluation round (for debugging/logging)
    print("evaluate_metrics (raw from clients):", metrics)
    
    # Compute the weighted average for val_accuracy and val_loss
    aggregated_val_accuracy = sum(val_accuracies) / sum(examples) if sum(examples) > 0 else 0
    aggregated_val_loss = sum(val_losses) / sum(examples) if sum(examples) > 0 else 0

    # Return the aggregated validation metrics
    return {"val_accuracy": aggregated_val_accuracy, "val_loss": aggregated_val_loss}


# This function is responsible for constructing the server-side components
# when the Flower server starts.
def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""

    # Read run-specific configuration (e.g., number of federated rounds)
    num_rounds = context.run_config["num-server-rounds"]

    # Initialize the global model's parameters.
    # `get_weights(Net())` creates an instance of the model and extracts its parameters.
    # `ndarrays_to_parameters` converts these NumPy arrays to Flower's internal format.
    initial_model_parameters = ndarrays_to_parameters(get_weights(Net()))

    # Define the federated learning strategy. FedAvg is a standard federated averaging strategy.
    # - fraction_fit: Fraction of available clients to use for training in each round.
    # - fraction_evaluate: Fraction of available clients to use for evaluation in each round.
    # - min_available_clients: Minimum number of clients that need to be connected for a round to start.
    # - evaluate_metrics_aggregation_fn: Specifies the function (`weighted_average`) to aggregate
    #                                    metrics from the `evaluate` phase of clients.
    # - initial_parameters: The initial parameters for the global model.
    #
    # Note on fit_metrics:
    # The metrics returned by `FlowerClient.fit()` (which are `train_loss`, `train_accuracy`,
    # `val_loss`, `val_accuracy` from `task.train`) are automatically aggregated by FedAvg
    # (typically by simple averaging if they are scalar values). These aggregated fit metrics
    # will appear in the server logs and history without needing a separate fit_metrics_aggregation_fn
    # unless more complex aggregation is required.


    # This is for multiple clients
    strategy = FedAvg(
        fraction_fit=context.run_config["fraction-fit"],
        fraction_evaluate=context.run_config["fraction-evaluate"],
        min_available_clients=2, # Example: wait for at least 2 clients
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_model_parameters,
    )

    # This is just for a single client
    # strategy = FedAvg(
    #     fraction_fit=1.0,             # Use 100% of available clients (i.e., the single one)
    #     fraction_evaluate=1.0,        # Use 100% of available clients for evaluation
    #     min_fit_clients=1,            # Ensure the one client is used for fit
    #     min_evaluate_clients=1,       # Ensure the one client is used for evaluate
    #     min_available_clients=1,      # Allow rounds to start with just one client
    #     evaluate_metrics_aggregation_fn=weighted_average, # Correctly handles single client metrics
    #     initial_parameters=initial_model_parameters,
    #     # You might not need to pass fraction_fit/evaluate from run_config anymore
    #     # if you are hardcoding to 1.0 here for the single client case.
    #     # Or, ensure your run_config for single client also sets these to 1.0.
    # )
    
    # ServerConfig defines server-level settings, like the total number of rounds.
    config = ServerConfig(num_rounds=num_rounds)

    # Return the components that will run the server application.
    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp. This is the main entry point for the Flower server.
# It uses `server_fn` to set up the server components.
app = ServerApp(server_fn=server_fn)