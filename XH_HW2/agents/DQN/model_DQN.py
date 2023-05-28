import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layer=[512]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        layers = []
        layers.append(nn.Linear(state_size, hidden_layer[0]))
        layers.append(nn.ReLU())
        for i in range(1, len(hidden_layer)):
            layers.append(nn.Linear(hidden_layer[i - 1], hidden_layer[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layer[-1], action_size))
        self.fc = nn.Sequential(*layers)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.fc(state)


if __name__ == "__main__":
    net = QNetwork(128, 6, 1, [512, 128, 128])
    print(net)
