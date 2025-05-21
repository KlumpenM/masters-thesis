"""flower_normal: A Flower / PyTorch app."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
# Import functions and classes from task.py:
# Net: The neural network model definition.
# get_weights: Extracts model parameters.
# load_data: Loads the client-specific dataset.
# set_weights: Updates the model with new parameters.
# test: Evaluates the model.
# train: Trains the model locally.
from flower_normal.task import Net, get_weights, load_data, set_weights, test, train


# FlowerClient is a subclass of NumPyClient, which means it handles model parameters
# as NumPy arrays. This class defines the client-side logic for federated learning.
class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, local_epochs, learning_rate):
        """Initializes the FlowerClient.

        Args:
            trainloader: DataLoader for the client's training data.
            valloader: DataLoader for the client's validation data.
            local_epochs: The number of epochs to train locally.
            learning_rate: The learning rate for local training.
        """
        self.net = Net()  # Instantiate the model (defined in task.py)
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device) # Move model to the appropriate device

    def fit(self, parameters, config):
        """Trains the local model using the provided parameters and configuration.

        This method is called by the Flower framework when the server requests
        the client to perform local training.

        Args:
            parameters (NDArrays): The model parameters received from the server.
                                   These are used to update the local model before training.
            config (Config): A dictionary containing configuration values from the server
                             (e.g., batch size, learning rate - though learning_rate
                             is passed in __init__ here).

        Returns:
            Tuple[NDArrays, int, Dict[str, Scalar]]:
                - NDArrays: The updated local model parameters (after training).
                - int: The number of examples used for training (len(self.trainloader.dataset)).
                - Dict[str, Scalar]: A dictionary of metrics from the training process.
                                     In this case, it's the dictionary returned by the
                                     `train` function from `task.py` containing:
                                     {"train_loss", "train_accuracy", "val_loss", "val_accuracy"}.
        """
        # Update local model with parameters received from the server
        set_weights(self.net, parameters)

        # Perform local training using the `train` function from `task.py`
        # The `train` function also handles evaluating on the validation set (valloader)
        # and returns a dictionary of metrics.
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.device,
        )
        
        # Return the updated model parameters, number of training examples, and the metrics dict
        return get_weights(self.net), len(self.trainloader.dataset), results


    def evaluate(self, parameters, config):
        """Evaluates the local model using the provided parameters and configuration.

        This method is called by the Flower framework when the server requests
        the client to evaluate the global model (or its current local model).

        Args:
            parameters (NDArrays): The model parameters received from the server.
                                   These are used to update the local model before evaluation.
            config (Config): A dictionary containing configuration values from the server.

        Returns:
            Tuple[float, int, Dict[str, Scalar]]:
                - float: The loss calculated by the `test` function from `task.py`.
                         This is interpreted as `val_loss` by the server.
                - int: The number of examples used for evaluation (len(self.valloader.dataset)).
                - Dict[str, Scalar]: A dictionary of metrics. Here it contains:
                                     {"val_accuracy", "val_loss"}.
        """
        # Update local model with parameters received from the server
        set_weights(self.net, parameters)

        # Evaluate the model using the `test` function from `task.py` on the client's valloader
        # The `test` function returns (avg_loss, accuracy)
        val_loss, val_accuracy = test(self.net, self.valloader, self.device)
        
        # Return validation loss, number of evaluation examples, and a metrics dictionary
        # The server's `weighted_average` function will use these keys.
        return val_loss, len(self.valloader.dataset), {"val_accuracy": val_accuracy, "val_loss": val_loss}


# This function is responsible for instantiating a FlowerClient when a client node starts.
# It receives context information which can be used to configure the client.
def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp.

    Args:
        context (Context): Contains node-specific and run-specific configurations.
                           - context.node_config["partition-id"]: Client's unique ID.
                           - context.node_config["num-partitions"]: Total number of clients.
                           - context.run_config: Hyperparameters for the current run.

    Returns:
        Client: An instance of FlowerClient converted to the base Client type.
    """

    # Read node-specific configuration to determine data partition
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Read run-specific configuration for hyperparameters
    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    # Load the client-specific data partition using `load_data` from `task.py`
    trainloader, valloader = load_data(partition_id, num_partitions, batch_size)
    
    # Instantiate and return the FlowerClient
    return FlowerClient(trainloader, valloader, local_epochs, learning_rate).to_client()


# Create a Flower ClientApp. This is the main entry point for a Flower client node.
# It uses the `client_fn` to create client instances.
app = ClientApp(client_fn)