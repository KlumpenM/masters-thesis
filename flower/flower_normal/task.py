"""flower_normal: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

# Defines the neural network architecture (a CNN in this case).
# This model is instantiated on each client and on the server
# (though the server only uses its structure for parameter initialization).
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # Adjusted for CIFAR-10
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        
        x = torch.flatten(x, 1)  # Flatten feature maps
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)
    

# Extracts model parameters (weights and biases) from a PyTorch model.
# Used by clients to send their updated local models to the server.
# Used by the server to get initial parameters.
def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


# Sets the parameters of a PyTorch model using a list of NumPy arrays.
# Used by clients to update their local model with parameters received from the server.
def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


fds = None  # Cache FederatedDataset


# Loads and partitions the CIFAR-10 dataset for a specific client.
# Each client gets a unique partition of the training data and a corresponding validation set.
# Called by `client_fn` in `client_app.py`.
def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test (becomes validation set)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    # This 'testloader' is used as the validation set for the client
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


# Trains the provided network `net` on the `trainloader` data for a number of `epochs`.
# After training, it evaluates the model on both the `trainloader` (to get training metrics)
# and the `valloader` (to get validation metrics) by calling the `test` function.
#
# Args:
#   net: The PyTorch model to train.
#   trainloader: DataLoader for the client's training data.
#   valloader: DataLoader for the client's validation data.
#   epochs: Number of local training epochs.
#   learning_rate: Learning rate for the optimizer.
#   device: Computation device ("cpu" or "cuda").
#
# Returns:
#   A dictionary containing:
#     "train_loss": Loss on the training set after local training.
#     "train_accuracy": Accuracy on the training set after local training.
#     "val_loss": Loss on the validation set after local training.
#     "val_accuracy": Accuracy on the validation set after local training.
# This dictionary is returned by `FlowerClient.fit()` in `client_app.py`.
def train(net, trainloader, valloader, epochs, learning_rate, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()

    # Calculate metrics on training set using the `test` function
    train_loss, train_acc = test(net, trainloader, device)
    # Calculate metrics on validation set using the `test` function
    val_loss, val_acc = test(net, valloader, device)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results


# Evaluates the provided network `net` on the `testloader` data.
#
# Args:
#   net: The PyTorch model to evaluate.
#   testloader: DataLoader for the data to evaluate on (can be train, validation, or test).
#   device: Computation device ("cpu" or "cuda").
#
# Returns:
#   A tuple (avg_loss, accuracy):
#     avg_loss: The average loss over the entire testloader dataset.
#     accuracy: The accuracy over the entire testloader dataset.
# This function is called by `train()` to get train/val metrics,
# and by `FlowerClient.evaluate()` in `client_app.py` to get validation metrics
# on the global model.
def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, total_loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            # Sum batch loss, weighted by batch size (as criterion usually averages over batch)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    
    avg_loss = total_loss / len(testloader.dataset)
    accuracy = correct / len(testloader.dataset)
    return avg_loss, accuracy