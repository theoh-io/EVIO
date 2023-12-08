import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDecoder(nn.Module):
    def __init__(self):
        super(SimpleDecoder, self).__init__()
        # Define the architecture
        self.fc1 = nn.Linear(512, 256)  # First fully connected layer
        self.fc2 = nn.Linear(256, 128)  # Second fully connected layer
        self.fc3 = nn.Linear(128, 64)   # Third fully connected layer
        self.fc4 = nn.Linear(64, 8)     # Output layer

    def forward(self, x):
        # Forward pass through the network
        x = F.relu(self.fc1(x))  # Activation function after first layer
        x = F.relu(self.fc2(x))  # Activation function after second layer
        x = F.relu(self.fc3(x))  # Activation function after third layer
        x = self.fc4(x)          # No activation after output layer
        return x