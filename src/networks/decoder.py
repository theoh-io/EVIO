import torch
import torch.nn as nn
import torch.nn.functional as F

#TO DO: Modularize the parameters of the layer

class SimpleDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleDecoder, self).__init__()
        # Define the architecture
        self.fc1 = nn.Linear(input_dim, 256)  # First fully connected layer
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)     # Output layer

    def forward(self, x):
        # Forward pass through the network
        x = F.relu(self.fc1(x))  # Activation function after first layer
        x = F.relu(self.fc2(x))  # Activation function after second layer
        x = self.fc3(x)          # No activation after output layer
        return x