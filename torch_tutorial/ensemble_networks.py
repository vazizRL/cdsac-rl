import torch
import torch as T
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class MultiHead(nn.Module):
    def __init__(self, alpha, input_dims, max_actions, fc1_dims=64, fc2_dims=64, n_actions=2,
                 name='actor'):
        super(MultiHead, self).__init__()
        # max_actions: the tanh is multiplied by max_actions to get prefered interval for actions
        self.input_dims = T.tensor(input_dims)
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.max_actions = max_actions
        self.epsilon_stability = 1e-6

        # self.fc1 = nn.Linear(np.prod(self.input_dims, dtype=int), self.fc1_dims)
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # n_actions heads
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optim = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        # Clamp std to prevent extreme values for actions, reparam to used  according to paper yet
        sigma = T.clamp(sigma, min=self.epsilon_stability, max=1)

        return mu, sigma


if __name__ == '__main__':
    device = torch.device('cuda:0')
    shape = (15, 28*28)
    inp_data = torch.rand(shape).to(device)
    mh = MultiHead(alpha=0.001, input_dims=(28*28), max_actions=3, n_actions=3)

    # Inference
    mu, sigma = mh.forward(inp_data)

    gauss = T.distributions.Normal(mu, sigma)
    actions = gauss.rsample()
    action = T.tanh(actions) * T.tensor(3).to(device)

    log_probs = gauss.log_prob(actions)
    log_probs -= T.log(1 - action.pow(2) + 1e-6)

    # log_probs_sum = log_probs.sum(1, keepdim=True)

