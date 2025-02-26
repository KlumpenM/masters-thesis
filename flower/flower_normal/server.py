# Import the necessary libraries
import flwr as fl

# Define the weighted average function
def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for  num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}


# Start the server with the Flower library
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3), # Define the number of rounds
    strategy=fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average, # Define the aggregation function
    ), # Define the strategy, with default parameters
)