"""flower_normal: A Flower / PyTorch app."""

from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters
from flwr.common import Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flower_normal.task import Net, get_weights



# Do the weighted average of the metrics that is sent to the server
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """ A function that aggregates metrics """
    
    # Since we are receiving a list of accuracies from the clients
    # Here m is the accuracy we are getting from the client
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    accuracy = sum(accuracies) / total_examples
    print(f"For each client: {metrics}")
    print(f"Weighted average accuracy: {accuracy}")
    # We are returning the average accuracy of all the clients
    return {"accuracy": accuracy}

"""
# Define a callback function, that will be called after each round
def on_fit_config(server_round: int) -> Metrics:
    # This is a mock up hyperparameter, that we can change based on the round
    learning_rate = 0.01

    #if server_round > 2:
    #    learning_rate = 0.005

    return {"learning_rate": learning_rate}


"""

def server_fn(context: Context):
    # Read from config
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
