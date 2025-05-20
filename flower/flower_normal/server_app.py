"""flower_normal: A Flower / PyTorch app."""

from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters
from flwr.common import Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flower_normal.task import Net, get_weights



"""
# Do the weighted average of the metrics that is sent to the server
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


# Define a callback function, that will be called after each round
def on_fit_config(server_round: int) -> Metrics:
    # This is a mock up hyperparameter, that we can change based on the round
    learning_rate = 0.01

    #if server_round > 2:
    #    learning_rate = 0.005

    return {"learning_rate": learning_rate}



def server_fn(context: Context):
    # Read from config
    print("Server Num-rounds:", context.run_config["num-server-rounds"])
    

    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    # Internal convertion of the model parameters to the format used by Flower
    parameters = ndarrays_to_parameters(ndarrays)

    print("Creating server")

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=2,
        initial_parameters=parameters,
        # We are using the weighted average function to aggregate the metrics
        evaluate_metrics_aggregation_fn=weighted_average,
        # Here we are passing the custom callback
        #on_fit_config_fn=on_fit_config,
    )
    
    # Define the config (number of rounds)
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)

"""

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=context.run_config["fraction-evaluate"],
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)