"""flower_normal: A Flower / PyTorch app."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flower_normal.task import Net, get_weights, load_data, set_weights, test, train

"""
# Define Flower Client and client_fn
# We can change it as much as we want, by it needs to be a subclass of NumPyClient
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        # Discover the device at run time (We need to do this, either in "FlowerClient" or in "client_fn")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        # We set the parameters of the local model, to the parameters received from the server
        set_weights(self.net, parameters)
        # We train the local model
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            #config["learning_rate"],
            self.device,
        )
        return (
            # The parameters of the local model (after updating)
            get_weights(self.net),
            # The size of the dataset of the client
            len(self.trainloader.dataset),
            # The metrics of the local model
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        # Recieve the parameters from the server, and apply them to the local model
        set_weights(self.net, parameters)
        # Evaluates the global model on the local dataset
        loss, accuracy = test(self.net, self.valloader, self.device)
        # Returns the loss, the amount of evaluation samples and accuracy (metrics) of the local model
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = Net()

    
    # When we are using node_config
    # Num-partitions    = How many clients are there in total
    #   - Num-partitions is equal to "options.num-supernodes" in the config file
    # Partition-id      = What is the client id of the current client
    #   - Partition-id will be set at runtime (would be any number between 0 to Num-partitions-1)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]


    # We load and partition the dataset based on the partition_id and num_partitions
    local_epochs = context.run_config["local-epochs"]
    trainloader, valloader = load_data(partition_id, num_partitions)

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)

"""

class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, local_epochs, learning_rate):
        self.net = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        #Train the model with data of this client
        set_weights(self.net, parameters)
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.device,
        )
        return get_weights(self.net), len(self.trainloader.dataset), results


    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        # The loss and accuracy here is tieed to the "weighted average" function in server_app.py
        return loss, len(self.valloader.dataset), {"accuracy": accuracy, "loss": loss}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Read run_config to fetch hyperparameters relevant to this run
    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    # Set up the data for this client
    trainloader, valloader = load_data(partition_id, num_partitions, batch_size)
    
    
    # Return Client instance
    return FlowerClient(trainloader, valloader, local_epochs, learning_rate).to_client()


# Flower ClientApp
app = ClientApp(client_fn)