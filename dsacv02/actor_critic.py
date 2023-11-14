import torch
import torch.nn as nn
import torch.distributions as distr
from dsacv02.mlp_gmm import MLPGMM
from dsac_old_versions.dsac_implementation.action_distribution import TanhGaussDistribution


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, min_log_std=-0.1, max_log_std=5,
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
        self.state_dim = obs_dim
        self.action_dim = action_dim
        inp_dim = [self.state_dim + self.action_dim]
        self.hidden_layers = hidden_layers
        self.arch = tuple(inp_dim + list(self.hidden_layers) + [1])
        self.activation = activ
        self.q = MLPGMM(arch=self.arch, activ=self.activation)
        self.min_log_std = torch.tensor(min_log_std).to(self.q.device)
        self.max_log_std = torch.tensor(max_log_std).to(self.q.device)
        self.denominator = max(abs(self.min_log_std), self.max_log_std)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def get_class_info(self):
        return self.state_dim, self.action_dim, self.hidden_layers, self.activation, \
               self.min_log_std.item(), self.max_log_std.item()

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
        # value_mean, log_std = torch.chunk(logits, chunks=2, dim=-1)
        value_mean, log_std = logits

        # tanh=-1 with min=-1, max=0.1, then -0.1; if tanh=1, min=-1, max=0.1, then 1
        value_log_std = torch.clamp_min(self.max_log_std * torch.tanh(log_std / self.denominator), 0) + \
            torch.clamp_max(-self.min_log_std * torch.tanh(log_std / self.denominator), 0)

        # return torch.cat((value_mean, value_log_std), dim=-1)
        return value_mean, value_log_std


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_layers=(256, 256), n_kernels=2,
                 activation=('gelu',), min_log_std=-20, max_log_std=3, action_low_lim=-1, action_up_lim=1):
        """
        - Modelled as a GMM to follow the GMM of the value distribtion function. Number of kernels must be the same
        - Stochastic Policy Function Approximator
        - Std. and Mean share first layers
        :param state_dim: Number of dimensions in observation space
        :param action_dim: Number of dimensions in action space
        :param hidden_layers: Hidden layers, format (l1_n_Nodes, l2_m_Nodes,)
        :param activation: Activations per layer
        :param min_log_std: Should be high negative value to emulate 0
        :param max_log_std: Should be lesser positive value to prevent very high std
        :param action_low_lim: Lowest action possible
        :param action_up_lim:  Highest action possible
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layers = hidden_layers
        self.n_kernels = n_kernels
        self.arch = tuple([self.state_dim] + list(self.hidden_layers) + [self.action_dim])
        self.activation = activation
        self.policy = MLPGMM(arch=self.arch, activ=self.activation)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.action_low_lim = action_low_lim
        self.action_up_lim = action_up_lim

        self.register_buffer("act_low_lim", torch.tensor(self.action_low_lim))
        self.register_buffer("act_up_lim", torch.tensor(self.action_up_lim))
        self.action_distribution_cls = TanhGaussDistribution
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def get_class_info(self):
        return self.state_dim, self.action_dim, self.hidden_layers, self.n_kernels, self.activation, \
               self.min_log_std, self.max_log_std, self.action_low_lim, self.action_up_lim

    def forward(self, obs):
        # Send to device first
        obs = torch.as_tensor(obs).to(self.device)
        logits = self.policy(obs)

        action_mean, action_log_std = logits

        # Equivalent to torch.e**(...), bound the standard deviation
        action_std = torch.clamp(action_log_std, self.min_log_std, self.max_log_std).exp()

        return action_mean, action_std

    def sample_from_action_distr(self, logits, reparameterization=False):
        # Construct the GMM
        means, stds = logits
        weights = torch.ones(self.n_kernels) / self.n_kernels
        gmm = distr.MixtureSameFamily(distr.Categorical(probs=weights), distr.Normal(means, stds))

        # Sample from the GMM
        if reparameterization:
            # TODO: Implement reparameterization trick for GMMs
            action = gmm.rsample()

    def log_prob(self):
        pass

    def mode(self):
        """
        - Mode: Value of the term that occurs the most often. Note: Can also be multi-modal
        """
        pass

    def get_entropy(self):
        pass


if __name__ == '__main__':
    # Note: The first dimension is the input dimension
    mlp_model = MLPGMM((3, 2, 2, 1), ('relu', 'relu', 'relu'))

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





