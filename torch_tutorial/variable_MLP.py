import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, architecture):
        super(MLP, self).__init__()

        # Extract the number of input features and hidden units
        self.input_units = architecture[0]
        self.hidden_units = architecture[1:-1]
        self.output_units = architecture[-1]
        self.network = None
        self.build_model()

    def forward(self, x):
        return self.network(x)

    def build_model(self):
        # Create a list to hold the layers
        layers = []

        # Add input layer
        layers.append(nn.Linear(self.input_units, self.hidden_units[0]))
        layers.append(nn.ReLU())

        # Add hidden layers
        for i in range(len(self.hidden_units) - 1):
            layers.append(nn.Linear(self.hidden_units[i], self.hidden_units[i + 1]))
            layers.append(nn.ReLU())

        # Add output layer
        layers.append(nn.Linear(self.hidden_units[-1], self.output_units))

        # Combine the layers into a sequential module
        self.network = nn.Sequential(*layers)


# Example usage
architecture = (2, 3, 4)  # Input, Hidden, Output layers
model = MLP(architecture)
print(model)