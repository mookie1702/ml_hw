import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Dueling DQN Model."""

    def __init__(self, state_size, action_size, seed, hidden_layer=[512]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layer (list): List of hidden layer sizes
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.feature_fc = nn.Sequential(
            nn.Linear(state_size, hidden_layer[0]),
            nn.ReLU(),
            nn.Linear(hidden_layer[0], hidden_layer[1]),
            nn.ReLU(),
        )

        self.value_fc = nn.Sequential(
            nn.Linear(hidden_layer[-1], hidden_layer[-1]),
            nn.ReLU(),
            nn.Linear(hidden_layer[-1], 1),
        )

        self.advantage_fc = nn.Sequential(
            nn.Linear(hidden_layer[-1], hidden_layer[-1]),
            nn.ReLU(),
            nn.Linear(hidden_layer[-1], action_size),
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        features = self.feature_fc(state)
        values = self.value_fc(features)
        advantages = self.advantage_fc(features)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


if __name__ == "__main__":
    net = QNetwork(128, 6, 1, [512, 128, 128])
    print(net)
