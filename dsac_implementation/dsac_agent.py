import torch
import torch.nn as nn
import time
from dsac_implementation.networks import Critic, Actor
from torch.distributions import Normal
from torch.optim import Adam

from copy import deepcopy
from tensorboard_tools import tb_tags
from typing import Tuple
from typing import Dict

critic_hidden_layers = (256, 256, 256, 256, 256)
critic_activations = ('gelu' , 'gelu', 'gelu', 'gelu', 'gelu', 'gelu')

actor_hidden_layers = (256, 256, 256, 256, 256)
actor_activations = ('gelu' , 'gelu', 'gelu', 'gelu', 'gelu', 'gelu')


class Agent:
    def __init__(self, cr_inp_dim, act_state_dim, action_dim, cr_lr, alpha_lr, action_low=-1, action_up=1,
                 cr_min_log_std=-0.1, cr_max_log_std=4, cr_hl=critic_hidden_layers, cr_activ=critic_activations,
                 act_hl=actor_hidden_layers, act_activ=actor_activations, act_min_log_std=-20, act_max_log_std=0.5,
                 buffer_size=5e5, batch_size=256, tau=0.001, alpha=0.2, act_lr=0.001, reward_scale=0.2, gamma=0.99,
                 up_interval=2, auto_alpha=True):
        """
        - Implements DSACv0.2, based on https://arxiv.org/abs/2001.02811
        :param cr_inp_dim:
        :param act_state_dim:
        :type act_state_dim:
        :param action_dim:
        :type action_dim:
        :param cr_lr:
        :type cr_lr:
        :param alpha_lr:
        :type alpha_lr:
        :param action_low:
        :type action_low:
        :param action_up:
        :type action_up:
        :param cr_min_log_std:
        :type cr_min_log_std:
        :param cr_max_log_std:
        :type cr_max_log_std:
        :param cr_hl:
        :type cr_hl:
        :param cr_activ:
        :type cr_activ:
        :param act_hl:
        :type act_hl:
        :param act_activ:
        :type act_activ:
        :param act_min_log_std:
        :type act_min_log_std:
        :param act_max_log_std:
        :type act_max_log_std:
        :param buffer_size:
        :type buffer_size:
        :param batch_size:
        :type batch_size:
        :param tau:
        :type tau:
        :param alpha:
        :type alpha:
        :param act_lr:
        :type act_lr:
        :param reward_scale:
        :type reward_scale:
        :param gamma:
        :type gamma:
        :param up_interval:
        :type up_interval:
        :param auto_alpha:
        :type auto_alpha:
        """
        self.q1: nn.Module = Critic(cr_inp_dim, cr_min_log_std, cr_max_log_std, cr_hl, cr_activ)
        self.q2: nn.Module = Critic(cr_inp_dim, cr_min_log_std, cr_max_log_std, cr_hl, cr_activ)
        self.q1_target: nn.Module = deepcopy(self.q1)
        self.q2_target: nn.Module = deepcopy(self.q2)

        self.policy: nn.Module = Actor(state_dim=act_state_dim, action_dim=action_dim, hidden_layers=act_hl,
                                       activation=act_activ, min_log_std=act_min_log_std, max_log_std=act_max_log_std,
                                       action_low_lim=action_low, action_up_lim=action_up)
        self.policy_target = deepcopy(self.policy)

        # Do not track gradients for target networks
        for p_q1_i in self.q1_target.parameters():
            p_q1_i.requires_grad = False
        for p_q2_i in self.q2_target.parameters():
            p_q2_i.requires_grad = False
        for p_pol_i in self.policy_target.parameters():
            p_pol_i.requires_grad = False

        # Create entropy coefficient
        self.log_alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))

        # Create optimizers
        self.q1_opt = Adam(self.q1.parameters(), lr=cr_lr)
        self.q2_opt = Adam(self.q2.parameters(), lr=cr_lr)
        self.policy_opt = Adam(self.policy.parameters(), lr=act_lr)
        self.alpha_optimizer = Adam([self.log_alpha], lr=alpha_lr)

        # Algorithm parameters
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = -action_dim
        self.auto_alpha = auto_alpha
        self.alpha = alpha
        self.update_interval = up_interval

    @property
    def adjustable_parameters(self):
        return (
            'gamma',
            'tau',
            'auto_alpha',
            'alpha',
            'delay_update'
        )

    def compute_q_loss(self):
        pass

    def evaluate_q(self, obs, actions, qnet):
        """
        - Sample in a standard fashion from \mathcal{Z} batch-wise
        - Only modification: Std is clamped
        :param obs: observation
        :param actions: actions
        :param qnet: Q-value approximator function to be evaluated
        :return:
        :rtype:
        """
        stocha_q = qnet(obs, actions)
        means, log_stds = stocha_q[..., 0], stocha_q[..., -1]
        stds = log_stds.exp()
        # Initiate zeros and ones tensors with shape of means and stds
        normal = Normal(torch.zeros_like(means), torch.ones_like(stds))
        z = normal.sample()
        #  Where are these hyperparameters specified?
        z = torch.clamp(z, -3, 3)
        # Due to being vectors, element-wise mutiplications
        q = means + torch.mul(z, stds)

        return means, stds, q

    def compute_gradient(self, batch: tuple, iteration: int):
        start_time = time.time()

        # Unpack batch
        states, old_actions, rewards, new_states, log_ps, dones = batch

        # Construct action distribution with reparameterization trick
        logits = self.policy(states)
        logits_mean, logits_std = torch.chunk(logits, chunks=2, dim=-1)
        # item() returns scalar as normal Python scalars
        policy_mean = torch.tanh(logits_mean).mean().item()
        policy_std = logits_std.mean().item()
        list()

        act_dist = self.policy.get_act_dist(logits)
        new_action, new_log_p = act_dist.sample(reparameterization=True)
        extended_batch = tuple(list(batch) + [new_action, new_log_p])

        # Value functions
        self.q1_opt.zero_grad()
        self.q2_opt.zero_grad()
        loss_q, q1, q2, std1, std2 = self.compute_q_loss(extended_batch)
        loss_q.backward()












