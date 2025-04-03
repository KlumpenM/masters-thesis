"""flower_DP: Flower Example using Differential Privacy and Secure Aggregation."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.client.mod import fixedclipping_mod, secaggplus_mod

from flower_DP.task import Net, get_weights, load_data, set_weights, test, train


class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, local_epochs) -> None:
        self.net = Net()
        self.trainloader = trainloader
        self.testloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        results = train(
            self.net,
            self.trainloader,
            self.testloader,
            epochs=self.local_epochs,
            device=self.device,
        )
        return get_weights(self.net), len(self.trainloader.dataset), {"train_loss": results}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, self.device)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = Net()


    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    local_epochs = context.node_config["local-epochs"]

    trainloader, valloader = load_data(
        partition_id=partition_id, num_partitions=num_partitions
    )
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
    mods=[
        secaggplus_mod,
        fixedclipping_mod,
    ],
)