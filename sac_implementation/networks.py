"""
- Here, only the network classes are implemented
- Policy network: Outputs mean and std for normal distribution, sample this to get actions (deviation from orgiginal)
"""

import os
import numpy as np
import torch
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

ckpt_dir = "C:/Users/vanya/OneDrive/Desktop/PhD_RL/RL_Framework/sac_implementation/Models"


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256, name='Critic',
                 checkpoint_dir=ckpt_dir):
        super(CriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims, dtype=T.float64)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims, dtype=T.float64)
        self.q = nn.Linear(self.fc2_dims, 1, dtype=T.float64)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        q_value = self.fc1(T.cat([state, action], dim=1))
        q_value = F.relu(q_value)
        q_value = self.fc2(q_value)
        q_value = F.relu(q_value)
        q_value = self.q(q_value)

        return q_value

    def save_checkpoint(self, chk_name=None):
        if chk_name:
            self.checkpoint_file = os.path.join(self.checkpoint_dir, chk_name + '_sac')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256, name='value', checkpoint_dir=ckpt_dir):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name + '_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims, dtype=T.float64)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims, dtype=T.float64)
        self.v = nn.Linear(self.fc2_dims, 1, dtype=T.float64)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)
        v = self.v(state_value)

        return v

    def save_checkpoint(self, chk_name=None):
        if chk_name:
            self.checkpoint_file = os.path.join(self.checkpoint_dir, chk_name + '_sac')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_actions, fc1_dims=256, fc2_dims=256, n_actions=2,
                 name='actor', checkpoint_dir=ckpt_dir):
        super(ActorNetwork, self).__init__()
        # max_actions: the tanh is multiplied by max_actions to get prefered interval for actions
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name + '_sac')
        self.max_actions = max_actions
        self.epsilon_stability = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims, dtype=T.float64)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims, dtype=T.float64)
        # Two heads
        self.mu = nn.Linear(self.fc2_dims, self.n_actions, dtype=T.float64)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions, dtype=T.float64)

        self.optim = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

        torch.autograd.set_detect_anomaly(True)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        # Clamp std to prevent extreme values for actions
        sigma = T.clamp(sigma, min=self.epsilon_stability, max=1)

        return mu, sigma

    def sample_normal(self, state, reparametrize=True):
        """
        - Returns bounded actions and its scalar value of log likelihoods sum
        :param state: State defined by observation
        :param reparametrize: The actual reparametrization proposed in the paper if True
        """
        mu, sigma = self.forward(state)

        # Not a multivariate form?
        prob = Normal(mu, sigma)

        # Sample only ONE, in torch.tensor format
        if reparametrize:
            actions = prob.rsample()
        else:
            actions = prob.sample()

        # Define custom boundary for actions
        action = T.tanh(actions) * T.tensor(self.max_actions).to(self.device)

        # Get the log probability of the (unmoified) actions according to distribution prob
        log_probs = prob.log_prob(actions)
        # Enforce action bounds and recalculate, log(\pi(a|s) = log(\mu(a|s)) - sum_{i=1}^D{log(1 - tanh^2(u_i))}
        log_probs -= T.log(1 - action.pow(2) + self.epsilon_stability)
        # For each batch, sum the provs.
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs, sigma

    def save_checkpoint(self, chk_name=None):
        if chk_name:
            self.checkpoint_file = os.path.join(self.checkpoint_dir, chk_name + '_sac')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))






