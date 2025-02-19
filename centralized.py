"""
This script demonstrates central training of a neural network model.
Think of this as a server where all clients send their models to be trained.
This example focuses on classification from data, not images.
The code is adapted from the following link: https://www.youtube.com/watch?v=jOmmuzMIQ4c
"""


# Importing the required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# Check if a GPU is available and use it, if possible, otheriwse the the CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define the layers of the network
        self.conv1 = nn.Conv2d(3, 6, 5)  # First convolutional layer (input channels: 3, output channels: 6, kernel size: 5)
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer (kernel size: 2, stride: 2)
        self.conv2 = nn.Conv2d(6, 16, 5)  # Second convolutional layer (input channels: 6, output channels: 16, kernel size: 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # First fully connected layer (input features: 16*5*5, output features: 120)
        self.fc2 = nn.Linear(120, 84)  # Second fully connected layer (input features: 120, output features: 84)
        self.fc3 = nn.Linear(84, 10)  # Third fully connected layer (input features: 84, output features: 10)

    # Define the forward pass of the network
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Apply first convolution, ReLU activation, and max pooling
        x = self.pool(F.relu(self.conv2(x)))  # Apply second convolution, ReLU activation, and max pooling
        x = x.view(-1, 16 * 5 * 5)  # Flatten the tensor for the fully connected layers
        x = F.relu(self.fc1(x))  # Apply first fully connected layer and ReLU activation
        x = F.relu(self.fc2(x))  # Apply second fully connected layer and ReLU activation
        x = self.fc3(x)  # Apply third fully connected layer (output layer)
        return x  # Return the output

# Define the training function
def train(net, training_set, epochs):
    criterion = nn.CrossEntropyLoss() # Loss function for classification tasks
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for image, labels in training_set: # Loop over the training data
            optimizer.zero_grad() # Zeore the gradients
            criterion(net(image.to(DEVICE)), labels.to(DEVICE)).backward() # Compute the loss and the gradients in the backward pass
            optimizer.step() # Update the model parameters

def test(net, test_set):
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0 
    with torch.no_grad():  # Disable gradient computation for testing
        for image, labels in test_set:
            outputs = net(image.to(DEVICE))  # Forward pass: compute the model output
            loss += criterion(outputs, labels.to(DEVICE)).item()  # Accumulate the loss
            total += labels.size(0)  # Accumulate the total number of samples
            correct += (torch.max(outputs.data, 1)[1] == labels.to(DEVICE)).sum().item()  # Accumulate the number of correct predictions
    return loss / len(test_set.dataset), correct / total # Return the average loss and accuracy


def load_data():
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = CIFAR10(root='./data', train=True, download=True, transform=trf)
    test_set = CIFAR10(root='./data', train=False, download=True, transform=trf)
    return DataLoader(train_set, batch_size=32, shuffle=True), DataLoader(test_set)


def load_model():
    net = Net().to(DEVICE)
    return net

if __name__ == "__main__":
    net = load_model()  # Load the model
    train_set, test_set = load_data()  # Load the data
    train(net, train_set, 5)  # Train the model for 5 epochs
    loss, accuracy = test(net, test_set)  # Test the model
    print(f"Loss: {loss:.5f}, Accuracy: {accuracy:.3f}")  # Print the test loss and accuracy