import torch
import torch.nn as nn
import torch.distributions as distr
from dsacv02.mlp_gmm import MLPGMM, MLPGMMWeighted
from dsacv02.gmm_reparameterization.mixture_same_family import ReparameterizedMixtureSameFamilyMod


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_layers=(256, 256), n_kernels=2, activ=('gelu',),
                 value_min_std=-0.1, value_max_std=5, learnable_weights=False, device='cuda:0'
                ):
        """
        - Modelling Q distribution as a Gaussian Mixed Model.
        :param state_dim: Dimension of observation space
        :param action_dim: Dimension of action space
        :param hidden_layers: Hidden layers of value approximator
        :param n_kernels: Number of kernels in the GMM
        :param activ: Activation functions, note that last nodes do not have an activation
        :param value_min_std: Minimum permissible standard deviation of the Gaussian kernels
        :param value_max_std: Maximum permissible standard deviation of the Gaussian kernels
        :param learnable_weights: Whether kernel weights of the GMMs are learnable or not
        :param device: Device on which Critic is running on
        """
        super().__init__()
        self.device = device
        self.learnable_weights = learnable_weights
        self.state_dim = state_dim
        self.action_dim = action_dim
        inp_dim = [self.state_dim + self.action_dim]
        self.hidden_layers = hidden_layers
        self.n_kernels = n_kernels
        self.arch = tuple(inp_dim + list(self.hidden_layers) + [1])
        self.activation = activ
        if self.learnable_weights:
            self.q = MLPGMMWeighted(arch=self.arch, activ=self.activation, n_kernels=self.n_kernels, device=device,
                                    multivar=False, std_bias_ini=1.0)       # 1.0
        else:
            self.q = MLPGMM(arch=self.arch, activ=self.activation, n_kernels=self.n_kernels, device=device,
                            multivar=False, std_bias_ini=1.0)               # 1.0
        self.min_std = torch.tensor(value_min_std).to(self.q.device)
        self.max_std = torch.tensor(value_max_std).to(self.q.device)
        self.denominator = max(abs(self.min_std), self.max_std)
        self.to(self.device)

    def get_class_info(self):
        return self.state_dim, self.action_dim, self.hidden_layers, self.n_kernels, self.activation, \
               self.min_std.item(), self.max_std.item(), self.learnable_weights, self.device

    def forward(self, observation: torch.tensor, action: torch.tensor, exp=False) -> torch.tensor:
        """
        - Critic feed forward
        - Simple clamping as either log- or raw value
        :param exp: Whether logits are exponentiated
        :param observation: Observation
        :param action: Chosen action (result of policy GMM)
        :return: Logits of kernels
        """
        logits = self.q(torch.cat([observation, action], dim=-1), exp=exp)
        value_mean, stds, weights = logits

        value_std = torch.clamp(stds, self.min_std, self.max_std)

        return value_mean, value_std, weights


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_layers=(256, 256), n_kernels=1,
                 activation=('gelu',), action_min_std=-20, action_max_std=3, action_low_lim=-1, action_up_lim=1,
                 learnable_weights=False, device='cuda:0'):
        """
        - Modelled as a Gauss
        - Stochastic Policy Function Approximator
        - Std. and Mean share first layers
        :param state_dim: Number of dimensions in observation space
        :param action_dim: Number of dimensions in action space
        :param hidden_layers: Hidden layers, format (l1_n_Nodes, l2_m_Nodes,)
        :param n_kernels: Deprecated, always set to 1
        :param activation: Activations per layer
        :param action_min_std: Lower bound on std; either as log- or raw-value
        :param action_max_std: Upper bound on std; either as log- or raw-value
        :param action_low_lim: Lowest action possible
        :param action_up_lim:  Highest action possible
        :param device:  Device on which actor is running
        """
        super().__init__()
        self.device = device
        self.learnable_weights = learnable_weights
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layers = hidden_layers
        self.n_kernels = n_kernels
        self.arch = tuple([self.state_dim] + list(self.hidden_layers) + [self.action_dim])
        self.activation = activation
        self.min_std = action_min_std
        self.max_std = action_max_std
        self.action_low_lim = torch.tensor(action_low_lim)
        self.action_up_lim = torch.tensor(action_up_lim)
        if self.learnable_weights:
            self.policy = MLPGMMWeighted(arch=self.arch, activ=self.activation, n_kernels=self.n_kernels,
                                         device=device, multivar=False)
        else:
            self.policy = MLPGMM(arch=self.arch, activ=self.activation, n_kernels=self.n_kernels, device=device,
                                 multivar=False)
        self.register_buffer("act_low_lim", torch.tensor(self.action_low_lim))
        self.register_buffer("act_up_lim", torch.tensor(self.action_up_lim))
        self.to(self.device)
        self.eps = 1e-7

    def get_class_info(self):
        return self.state_dim, self.action_dim, self.hidden_layers, self.n_kernels, self.activation, \
               self.min_std, self.max_std, self.action_low_lim.item(), self.action_up_lim.item(), \
               self.learnable_weights, self.device

    def forward(self, obs, exp=False):
        """
        :param obs: Must be flattened before revoking thi method
        :param exp: Whether logits are exponentiated
        :return: Means, stds, kernel weights
        """
        # Send to device first
        obs = torch.as_tensor(obs).to(self.device)

        # If self.learnable_weight=False, then kernel_weights=None
        action_mean, action_std, kernel_weights = self.policy(obs, exp=exp)

        # Note: If exp=True, self.min_std and self.max_std must also be given as log, otherwise as raw value
        action_std = torch.clamp(action_std, self.min_std, self.max_std)

        return action_mean, action_std, kernel_weights

    def sample_from_action_distr(self, locs, stds, kweights, reparameterization=False):
        """
        - Samples from self-implemented GMM with reparameterization implemented
        :param locs: Means of kernels for actions
        :param stds: Std. of kernels for actions
        :param kweights: Kernel weights
        :param reparameterization: Whether reparameterization will be performed or not
        """
        # gauss = ReparameterizedMixtureSameFamilyMod(distr.Categorical(probs=kweights), distr.Normal(locs, stds))
        gauss = distr.Normal(loc=locs, scale=stds)

        # Sample from the GMM
        if reparameterization:
            action = gauss.rsample()
        else:
            action = gauss.sample()

        # Normalize each action around 0 with tanh
        action_bounded = ((self.action_up_lim - self.action_low_lim) / 2) * torch.tanh(action) + \
                         (self.action_up_lim + self.action_low_lim) / 2

        # Calculate log_prob of new action
        log_prob_bounded = gauss.log_prob(action) - \
                           torch.log(1 + self.eps - torch.pow(torch.tanh(action), 2)) \
            - torch.log((self.action_up_lim - self.action_low_lim) / 2)
        # Under the assumptions that actions are independent from each other,
        log_prob_bounded = torch.sum(log_prob_bounded, dim=1).unsqueeze(dim=1)

        return action_bounded, log_prob_bounded

    def log_prob(self, action_bounded, k_weights: torch.tensor, locs: torch.tensor, stds: torch.tensor) -> torch.tensor:
        """
        - The bounded action has first ot be converted into its unbounded variant, only the the limited log_prob
          can be calculated
        - GMM parameters not given as log, exponentiated!
        :param action_bounded:
        :param k_weights: Weigths of the kernels in the GMM
        :param locs:  Means of the kernels
        :param stds: Standard deviations of the kernels
        :param action_bounded: Bounded actions
        """
        gmm = ReparameterizedMixtureSameFamilyMod(distr.Categorical(probs=k_weights), distr.Normal(locs, stds))
        action_unbounded = torch.atanh((1 - self.eps) * (2 * action_bounded - (self.action_up_lim + self.action_low_lim))
                                       / (self.action_up_lim - self.action_low_min))

        log_prob_ = gmm.log_prob(action_unbounded) - torch.log((self.action_up_lim - self.action_low_lim) *
                                                (1 + self.eps - torch.pow(torch.tanh(action_unbounded), 2))).sum(-1)

        return log_prob_

    def mode(self, mean: torch.tensor) -> torch.tensor:
        """
        - Mode: Value of the term that occurs the most often. Note: Can also be multi-modal
        :param mean: Mean of the distribution to be calculated
        """
        return (self.action_up_lim - self.action_low_lim) / 2 * torch.tanh(mean) + \
               (self.action_up_lim + self.action_low_lim) / 2

    def get_entropy(self):
        """
        - Entropy calculation for ReparameterizedMixtureSameFamilyMod must be implemented first
        """
        pass



