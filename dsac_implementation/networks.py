import numpy as np
import torch
import torch.nn as nn
from variable_mlp import MLP
from action_distribution import TanhGaussDistribution


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, min_log_std=-0.1, max_log_std=4,
                 hidden_layers=(256, 256, 256, 256, 256), activ=('gelu', 'gelu', 'gelu', 'gelu', 'gelu', 'gelu')):
        """
        - Modelling Q distribution as a Gaussian.
        - min/max_log_std: clip(\mathcal{T}^{\pi_{\phi'}_{mathcal{D}}}Z(s,a), Q_{\theta}(s,a) - b, Q_{\theta}(s,a) + b)
        :param min_log_std: Based on ori. config. Convert to tensor and send to device
        :param max_log_std: Base don ori. config. Convert to tensor and send to device, >=0
        :param arch: Based on DSAC paper. First dim. is obs. dim.; likely to change
        :param act: Based on DSAC paper.
        """
        super().__init__()
        inp_dim = [obs_dim + action_dim]
        self.arch = tuple(inp_dim + list(hidden_layers) + [2])
        self.q = MLP(arch=self.arch, activ=activ)
        self.min_log_std = torch.tensor(min_log_std).to(self.q.device)
        self.max_log_std = torch.tensor(max_log_std).to(self.q.device)
        self.denominator = max(abs(self.min_log_std), self.max_log_std)

    def forward(self, obs: torch.tensor, action: torch.tensor, min=False):
        """
        - Critic feed forward
        :param obs: Observation
        :param action: Action
        :param min: Not known
        :return: Logits?
        :rtype: torch.tensor
        """
        logits = self.q(torch.cat([obs, action], dim=-1))
        value_mean, log_std = torch.chunk(logits, chunks=2, dim=-1)

        # tanh=-1 with min=-1, max=0.1, then -0.1; if tanh=1, min=-1, max=0.1, then 1
        value_log_std = torch.clamp_min(self.max_log_std * torch.tanh(log_std / self.denominator), 0) + \
            torch.clamp_max(-self.min_log_std * torch.tanh(log_std / self.denominator), 0)

        return torch.cat((value_mean, value_log_std), dim=-1)


class Actor(nn.Module):
    def __init__(self, state_dim: tuple, action_dim: tuple, hidden_layers=(256, 256, 256, 256, 256),
                 activation=('gelu', 'gelu', 'gelu', 'gelu', 'gelu', 'gelu'), min_log_std=-20, max_log_std=0.5,
                 action_low_lim=-1, action_up_lim=1):
        """
        - Stochastic Policy Function Approximator
        - Std. and Mean share first layers
        - Default parameters taken from original implementation
        :param state_dim:
        :param action_dim:
        :param hidden_layers:
        :param activation:
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.arch = tuple(list(self.state_dim) + list(hidden_layers) + list(self.action_dim))
        self.activation = activation
        self.policy = MLP(arch=self.arch, activ=self.activation)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        self.register_buffer("act_low_lim", torch.from_numpy(action_low_lim))
        self.register_buffer("act_up_lim", torch.from_numpy(action_up_lim))
        self.action_distribution_cls = TanhGaussDistribution

    def get_act_distr(self, logits):
        """
        - Must be rewritten for more efficiency. Why is a distribution object instantiated each time, although
          self.act_low_lim and self.act_up_lim do not change?
        :param logits:
        :type logits:
        """
        act_dist = self.action_distribution_cls(logits, self.act_low_lim, self.act_up_lim)

        return act_dist

    def forward(self, obs):
        logits = self.policy(obs)
        action_mean, action_log_std = torch.chunk(logits, chunks=2, dim=-1)
        action_std = torch.clamp(action_log_std, self.min_log_std, self.max_log_std).exp()

        return torch.cat((action_mean, action_std), dim=-1)


if __name__ == '__main__':
    import torchvision
    import torchvision.transforms as transforms

    # Note: The first dimension is the input dimension
    mlp_model = MLP((3, 2, 2, 1), ('relu', 'relu', 'relu'))

    # # # Register Activations (NOT ACTIVATION TYPES )when in inference
    # mlp_model._layers[-2].register_forward_hook(mlp_model.get_act('last_hh'))

    # Test critic
    cr_inp = (2,)
    min_log_std = -0.1
    max_log_std = 4
    hl = (256, 256, 256, 256, 256)
    activation = ('gelu', 'gelu', 'gelu', 'gelu', 'gelu', 'gelu', 'gelu')
    cr = Critic(obs_dim=2, action_dim=2, min_log_std=min_log_std, max_log_std=max_log_std,
                hidden_layers=hl, activ=activation)

    rnd_inp = torch.rand((10, 2))




