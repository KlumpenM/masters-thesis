# Importing the require libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# Check if a GPU device is available and use it, if possible, otherwise use the CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define the layers of the network (The same base model as we used before)
        self.conv1 = nn.Conv2d(3, 6, 5) 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linaer(84, 10)
    
    # Define the forward pass of the network
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train():
    return