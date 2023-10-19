import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


class ZNetwork(nn.Module):
    def __init__(self, action_dim, state_dim, lr, hl1=256, hl2=256, n_atoms=51, v_min=-100, v_max=100):
        super(ZNetwork, self).__init__()
        self.n_atoms = n_atoms
        self.register_buffer('atoms', torch.linspace(v_min, v_max, steps=n_atoms))
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hl1_dim = hl1
        self.hl2_dim = hl2
        self.network = nn.Sequential(
            nn.Linear(state_dim, hl1),
            nn.ReLU(),
            nn.Linear(hl1, hl2),
            nn.ReLU(),
            nn.Linear(hl2, self.action_dim * self.n_atoms)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state_batch, action=None):
        """
        - Reuturns either an action or if actiton is given, its probability mass function normalized
        :param state_batch: Batch containing various states
        :param action: If specified, the pmf is returned for this action
        :return: action and/or pmf of action
        """
        logits = self.network(state_batch)
        batch_size = len(state_batch)
        # Probability mass function for EACH action, NORMALIZED with Softmax
        pmfs = torch.softmax(logits.view(batch_size, self.action_dim, self.n_atoms), dim=2)
        # Accumulate Q-values for each action, note that self.atoms are linspace between v_min and v_max
        q_values = (pmfs * self.atoms).sum(dim=2)
        if action is None:
            # Chose action in standard Q-fashion
            action = torch.argmax(q_values, 1)
        action_i_distribution = pmfs[torch.arange(batch_size), action]
        return action, action_i_distribution


if __name__ == '__main__':
    z_network = ZNetwork(action_dim=4, state_dim=11, hl1=128, hl2=128, n_atoms=51, v_min=-10, v_max=10)



