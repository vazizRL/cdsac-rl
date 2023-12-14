import torch
import torch.nn as nn
import torch.distributions as distr
import time
from dsacv02.gmm_reparameterization.mixture_same_family import ReparameterizedMixtureSameFamilyMod as RMM
from torch.optim import Adam, lr_scheduler
from dsac_old_versions.dsac_implementation.tensorboard_tools import tb_tags


class DSAC:
    def __init__(self, critic1, critic2, critic1_target, critic2_target, cr_lr_ini, cr_lr_fin, policy, policy_target,
                 actor_lr_ini, actor_lr_fin, log_alpha, alpha_lr_ini, alpha_lr_fin, t_max=50, tau=0.001, alpha=0.2,
                 reward_scale=0.2, gamma=0.99, update_interval=2, auto_alpha=True, target_entropy=-1, n_kernels=None,
                 **kwargs):
        """
        - Implements DSACv0.2, based on DRL, Cramèr Distance and GMMs
        :param critic1: First q-network in the double-Q setting
        :param critic2: Second q-network in the double-Q setting
        :param critic1_target: First q-target
        :param critic2_target: Second q-target
        :param cr_lr_ini: Initial learning rate of both q-networks
        :param cr_lr_fin: Final learning rate of both q-networks
        :param policy: Actor network
        :param policy_target: Target actor network
        :param log_alpha: Temperament, can be learnable or static
        :param actor_lr_ini: Actor initial learning rate
        :param actor_lr_fin: Actor learning rate at the end of the period
        :param alpha_lr_ini: Initial temperament learning rate, only if learnable
        :param alpha_lr_fin: Final temperament learning rate, only if learnable
        :param t_max: Horizont for learning rate schedule. After that, the learning rate doesn't change
        :param target_entropy: Target entropy in actor distribution
        :param td_bound:
        :param kwargs:
        :param tau: Soft update parameter
        :param reward_scale: Reward scaling, from SAC
        :param gamma: Discount factor
        :param up_interval:
        :param auto_alpha: Whether alpha is updated automatically
        :param n_kernels: Number of Gaussian kernels in the GMM
        """

        # Initialize device
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.q1: nn.Module = critic1
        self.q2: nn.Module = critic2
        self.q1_target = critic1_target
        self.q2_target = critic2_target

        self.policy = policy
        self.policy_target = policy_target

        self.n_kernels = n_kernels

        # Do not track gradients for target networks
        self.switch_autograd_logging(require_grad=False, models=[self.q1_target, self.q2_target, self.policy_target])

        # NOTE: log_alpha is already given as a torch tensor, with initial value specified in agent
        self.log_alpha = log_alpha

        # Assign optimizer params and create optimizers
        self.q_lr_ini = cr_lr_ini
        self.q_lr_fin = cr_lr_fin
        self.policy_lr_ini = actor_lr_ini
        self.policy_lr_fin = actor_lr_fin
        self.alpha_lr_ini = alpha_lr_ini
        self.alpha_lr_fin = alpha_lr_fin
        self.t_max = t_max

        self.q1_optimizer = Adam(self.q1.parameters(), lr=cr_lr_ini)
        self.q2_optimizer = Adam(self.q2.parameters(), lr=cr_lr_ini)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=actor_lr_ini)
        self.alpha_optimizer = Adam([self.log_alpha], lr=alpha_lr_ini)

        self.q1_lrs = None
        self.q2_lrs = None
        self.pol_lrs = None
        self.alpha_lrs = None
        self.create_lr_schedules()

        # Algorithm parameters
        self.reward_scale = reward_scale
        self.gamma = torch.tensor(gamma).to(self.device)
        self.tau = torch.tensor(tau).to(self.device)
        self.target_entropy = torch.tensor(target_entropy).to(self.device)
        self.static_alpha = torch.tensor(alpha).to(self.device)
        self.auto_alpha = auto_alpha
        self.update_interval = update_interval

    @staticmethod
    def switch_autograd_logging(require_grad, models: list):
        """
        - Either turns on or turns off logging of the gradients of models.
        - When calculating policy loss, grads for value networks must not be traced
        :param require_grad: Whether gradients are logged
        :param models: Parameterized as neutal network
        """
        for model in models:
            for para in model.parameters():
                if require_grad:
                    para.requires_grad = True
                else:
                    para.requires_grad = False

    """ 
    Internally Called 
    """

    def create_lr_schedules(self):
        """
        - Instantiate learning rate scheduler with cosine annealing
        """
        self.q1_lrs = lr_scheduler.CosineAnnealingLR(self.q1_optimizer, T_max=self.t_max, eta_min=self.q_lr_fin,
                                                     last_epoch=-1, verbose=False)
        self.q2_lrs = lr_scheduler.CosineAnnealingLR(self.q2_optimizer, T_max=self.t_max, eta_min=self.q_lr_fin,
                                                     last_epoch=-1, verbose=False)
        self.pol_lrs = lr_scheduler.CosineAnnealingLR(self.policy_optimizer, T_max=self.t_max,
                                                      eta_min=self.policy_lr_fin, last_epoch=-1, verbose=False)
        self.alpha_lrs = lr_scheduler.CosineAnnealingLR(self.alpha_optimizer, T_max=self.t_max,
                                                        eta_min=self.alpha_lr_fin, last_epoch=-1, verbose=False)

    def get_alpha(self, requires_grad=False):
        """
        - Calculates alpha from log_alpha and returns scalar or tensor depending on whether temperature regulation
          is on or off
        :param requires_grad: If True, then torch tensor is returned
        :return: alpha from class, as scalar or as tensor
        """
        if self.auto_alpha:
            alpha = self.log_alpha.exp()
            if requires_grad:
                return alpha
            else:
                # item() returns value as standard Python value
                return alpha.item()
        else:
            return self.static_alpha

    def soft_avg_update(self, net: torch, net_targ: torch):
        """
        - \tau is a very small value specifying the rate of change of the target network
        :param net:
        :param net_targ: Target network
        """
        tar_complement = 1 - self.tau
        for para, para_targ in zip(net.parameters(), net_targ.parameters()):
            para_targ.data.mul_(tar_complement)
            para_targ.data.add_(self.tau * para.data)

    def evaluate_q(self, obs, actions, qnet, exp=False, sample_from_distr=True, reparameterize=False):
        """
        - Sample in a standard fashion from \mathcal{Z} batch-wise
        - Only modification: Std is clamped
        - Note that stds can not be negative
        :param obs: observation
        :param actions: actions
        :param qnet: Q-value approximator function to be evaluated
        :param exp: Whether
        :param sample_from_distr: If False, the network means will be used to calculated
        :return: PMF, Z
        """
        # (B, K, Q)
        means, stds, kernel_weights = qnet(obs, actions, exp=exp)
        if not kernel_weights:
            kernel_weights = torch.ones(self.n_kernels) / self.n_kernels

        cat_distr = distr.Categorical(probs=kernel_weights)
        comp_distr = distr.Normal(loc=means, scale=stds)
        gmm = RMM(mixture_distribution=cat_distr, component_distribution=comp_distr)
        gmm_sample = None

        if sample_from_distr:
            batch_size = obs.shape[0]
            if reparameterize:
                gmm_sample = gmm.rsample(sample_shape=batch_size)
            else:
                gmm_sample = gmm.sample(sample_shape=batch_size)

        return gmm, gmm_sample

    def update_networks(self, iteration: int):
        # Value optimizing step
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # Update every n-th iteration
        if iteration % self.update_interval == 0:
            # Policy optimizing step
            self.policy_optimizer.step()

            # Optional alpha optimizing step
            if self.auto_alpha:
                self.alpha_optimizer.step()

            with torch.no_grad():
                self.soft_avg_update(self.q1, self.q1_target)
                self.soft_avg_update(self.q2, self.q2_target)
                self.soft_avg_update(self.policy, self.policy_target)

    def compute_target_distribution(self, rewards, dones, q_means_next, stds_next, kernel_weights, log_probs_a_next,
                                    exp=False):
        """
        - Calculates the target distribution \mathcal{Z}(\cdot|s,a) as a GMM
        :param rewards: Rewards received at time t, r_t
        :param dones: Whether s_{t+1} is a terminal state
        :param q_means_next: Next Q-means from the kernels
        :param stds_next: Next standard deviation
        :param kernel_weights: Weights of the Gaussian kernels in the GMM
        :param log_probs_a_next: Log probability of the next action
        :return: Target distribution modeles as a GMM
        """
        next_batch_size = q_means_next.shape[0]
        alpha = self.get_alpha(requires_grad=False)
        # Compute target from mean Q
        q_means_target = rewards + (1 - dones) * self.gamma * (q_means_next - alpha * log_probs_a_next)
        stds_next = (1-dones) * stds_next + torch.tensor(1e-10, dtype=torch.float64)

        cat_distr = distr.Categorical(probs=kernel_weights)
        comp_distr = distr.Normal(loc=q_means_target, scale=stds_next)
        target_distribution = RMM(mixture_distribution=cat_distr, component_distribution=comp_distr)
        target_samples = target_distribution.sample(sample_shape=next_batch_size)

        return target_distribution, target_samples

    def compute_q_loss(self, batch, bound=True):
        states, actions, rewards, states_next, _, dones = batch
        # Convert to tensors
        states = torch.as_tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32).to(self.device)
        rewards = self.reward_scale * torch.as_tensor(rewards, dtype=torch.float32).to(self.device)
        states_next = torch.as_tensor(states_next, dtype=torch.float32).to(self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32).to(self.device)

        logits_next = self.policy_target(states_next)
        action_dist_nxt = self.policy_target.get_act_distr(logits_next)
        # In evaluation and control, reparameterization trick is used
        actions_nxt, log_prob_actions_next = action_dist_nxt.sample(reparameterization=True)

        q1_means, q1_stds, _ = self.evaluate_q(obs=states, actions=actions, qnet=self.q1)
        q2_means, q2_stds, _ = self.evaluate_q(obs=states, actions=actions, qnet=self.q2)

        q1_means_next, _, q1_next_sample = self.evaluate_q(obs=states_next, actions=actions_nxt, qnet=self.q1_target)
        q2_means_next, _, q2_next_sample = self.evaluate_q(obs=states_next, actions=actions_nxt, qnet=self.q2_target)

        # Returns array of smaller mean
        q_means_next = torch.min(q1_means_next, q2_means_next)

        # Chooses the r.v. drawn from dist. that has smaller mean, double
        q_next_samples = torch.where(q1_means_next < q2_means_next, q1_next_sample, q2_next_sample)

        # q_means_next: Standard overestimation counter measure
        # q_next_samples: Modified overestimation counter measure
        targets_q1_mean, targets_z1_bound, target_z1_unbound = self.compute_target_q(
                                                                        rewards=rewards,
                                                                        dones=dones,
                                                                        q_means=q1_means.detach(),
                                                                        q_stds=q1_stds.detach(),
                                                                        q_means_next=q_means_next.detach(),
                                                                        q_next_samples=q_next_samples.detach(),
                                                                        log_probs_a_next=log_prob_actions_next.detach()
                                                                                    )
        targets_q2_mean, targets_z2_bound, target_z2_unbound = self.compute_target_q(
                                                                        rewards=rewards,
                                                                        dones=dones,
                                                                        q_means=q2_means.detach(),
                                                                        q_stds=q2_stds.detach(),
                                                                        q_means_next=q_means_next.detach(),
                                                                        q_next_samples=q_next_samples.detach(),
                                                                        log_probs_a_next=log_prob_actions_next.detach()
                                                                )
        if bound:
            """ 
            Calculate losses in bounded case
            Loss is the sum of variance-augmented standard Q-loss, variance-augmented Z-loss and log(var)
            Necessary if target is bounded?
            """

            # Weight between mean and Z ?
            # weight = 0.5 * (torch.mean(torch.pow(q1_stds.detach(), 2)) + torch.mean(torch.pow(q2_stds.detach(), 2)))
            weight = 1

            # q1 loss
            q1_loss = weight * torch.mean(
                (torch.pow(q1_means - targets_q1_mean, 2)) / (2 * torch.pow(q1_stds.detach(), 2))
                + (torch.pow(q1_means.detach() - targets_z1_bound, 2)) / (2 * torch.pow(q1_stds, 2))
                + torch.log(q1_stds)
            )

            # q2 loss
            q2_loss = weight * torch.mean(
                torch.pow(q2_means - targets_q2_mean, 2) / (2 * torch.pow(q2_stds.detach(), 2))
                + torch.pow(q2_means.detach() - targets_z2_bound, 2) / (2 * torch.pow(q2_stds, 2))
                + torch.log(q2_stds)
            )

        else:
            """Calculate losses if no bounds are enforced on target. Is identical to loss in paper"""
            q1_loss = -Normal(q1_means, q1_stds).log_prob(targets_z1_bound).mean()
            q2_loss = -Normal(q2_means, q2_stds).log_prob(targets_z2_bound).mean()

        q_loss = q1_loss + q2_loss

        return q_loss, q1_means.detach().mean(), q2_means.detach().mean(), q1_stds.detach().mean(), \
            q2_stds.detach().mean()

    def compute_policy_loss(self, reduced_batch):
        states, actions_curr_pol, log_ps_curr_pol = reduced_batch
        # Convert to tensors
        states = torch.as_tensor(states, dtype=torch.float32).to(self.device)
        actions_curr_pol = torch.as_tensor(actions_curr_pol, dtype=torch.float32).to(self.device)
        log_ps_curr_pol = torch.as_tensor(log_ps_curr_pol, dtype=torch.float32).to(self.device)

        q1_means, _, _ = self.evaluate_q(obs=states, actions=actions_curr_pol, qnet=self.q1)
        q2_means, _, _ = self.evaluate_q(obs=states, actions=actions_curr_pol, qnet=self.q2)
        # Not calculated according to Z! If Z, reparameterization is needed
        policy_loss = (self.get_alpha(requires_grad=False) *
                       log_ps_curr_pol - torch.min(q1_means, q2_means)).mean()
        entropy = -log_ps_curr_pol.detach().mean()

        return policy_loss, entropy

    def compute_alpha_loss(self, log_ps):
        loss_alpha = - self.log_alpha * (log_ps.detach() + self.target_entropy).mean()

        return loss_alpha

    def compute_gradient(self, batch: tuple, iteration: int):
        start_time = time.time()

        # Unpack batch
        states, _, _, _, _, _ = batch

        # Convert state to tensor
        states = torch.as_tensor(states, dtype=torch.float32)

        # Construct action distribution with reparameterization trick
        logits = self.policy(states)
        logits_mean, logits_std = logits
        # item() returns scalar as normal Python scalars
        policy_mean = torch.tanh(logits_mean).mean().item()
        policy_std = logits_std.mean().item()

        act_dist = self.policy.get_act_distr(logits)
        new_actions, new_log_ps = act_dist.sample(reparameterization=True)
        # extended_batch = tuple(list(batch) + [new_action, new_log_p])

        # Calculate value loss and backpropagate
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        loss_q, q1_mean, q2_mean, q1_std, q2_std = self.compute_q_loss(batch)
        loss_q.backward()

        # Switch off autograd when calculating policy loss
        models = [self.q1, self.q2]
        self.switch_autograd_logging(require_grad=False, models=models)

        # Calculate policy loss and backpropagate
        policy_batch = (states, new_actions, new_log_ps)
        self.policy_optimizer.zero_grad()
        loss_policy, entropy = self.compute_policy_loss(policy_batch)
        loss_policy.backward()

        # Switch back on autograd after calculation of policy
        self.switch_autograd_logging(require_grad=True, models=models)

        # Adjust alpha is auto-alpha is enabled
        if self.auto_alpha:
            self.alpha_optimizer.zero_grad()
            loss_alpha = self.compute_alpha_loss(log_ps=new_log_ps)
            loss_alpha.backward()

        tb_info = {
            "DSAC2/critic_avg_q1-RL iter": q1_mean.item(),
            "DSAC2/critic_avg_q2-RL iter": q2_mean.item(),
            "DSAC2/critic_avg_std1-RL iter": q1_std.item(),
            "DSAC2/critic_avg_std2-RL iter": q2_std.item(),
            tb_tags["loss_actor"]: loss_policy.item(),
            tb_tags["loss_critic"]: loss_q.item(),
            "DSAC2/policy_mean-RL iter": policy_mean,
            "DSAC2/policy_std-RL iter": policy_std,
            "DSAC2/entropy-RL iter": entropy.item(),
            "DSAC2/alpha-RL iter": self.get_alpha(requires_grad=False),
            tb_tags["alg_time"]: (time.time() - start_time) * 1000,
        }

        return tb_info

    """ 
    /Internally Called 
    """

    def get_optimizers(self):
        """
        - Necessary for saving and reconstructing the optimizers
        :return: All optimizers
        """
        return self.q1_optimizer, self.q2_optimizer, self.policy_optimizer, self.alpha_optimizer

    def get_lr_info(self):
        """
        - Necessary for saving and reconstructing the learning rate schedule
        :return: All initial and final learning rates
        """
        return self.q_lr_ini, self.q_lr_fin, self.policy_lr_ini, self.policy_lr_fin, self.alpha_lr_ini, \
            self.alpha_lr_fin

    def update(self, batch: tuple, iteration: int):
        """
        - Wrapper; Calculate gradient and perform network optimization step
        - Perform lr scheduler step
        :param batch: Mini-batch
        :param iteration: Iteration number, necessary to determine update-interval
        :return: Dict containing quantities for logging in tensorboard
        """
        tb_info = self.compute_gradient(batch=batch, iteration=iteration)
        self.update_networks(iteration)
        self.update_lrs()

        return tb_info

    def update_lrs(self):
        # print(f'Before LRs: \n q1: {self.q1_optimizer.param_groups[0]["lr"]}' +
        #                   f'\n q2: {self.q2_optimizer.param_groups[0]["lr"]}' +
        #                   f'\n pol: {self.policy_optimizer.param_groups[0]["lr"]}' +
        #                   f'\n alpha: {self.alpha_optimizer.param_groups[0]["lr"]}')
        self.q1_lrs.step()
        self.q2_lrs.step()
        self.pol_lrs.step()
        self.alpha_lrs.step()
        # print(f'After Step(): \n q1: {self.q1_optimizer.param_groups[0]["lr"]}' +
        #                   f'\n q2: {self.q2_optimizer.param_groups[0]["lr"]}' +
        #                   f'\n pol: {self.policy_optimizer.param_groups[0]["lr"]}' +
        #                   f'\n alpha: {self.alpha_optimizer.param_groups[0]["lr"]}')

    def remote_update(self, update_info: dict):
        raise NotImplementedError('The method "remote_update" is not implemented')

    def get_remote_update_info(self):
        raise NotImplementedError('The method "get_remote_update_info" is not implemented')

    @staticmethod
    def get_empty_tb_info():
        tb_info = {
            "DSAC2/critic_avg_q1-RL iter": 0,
            "DSAC2/critic_avg_q2-RL iter": 0,
            "DSAC2/critic_avg_std1-RL iter": 0,
            "DSAC2/critic_avg_std2-RL iter": 0,
            tb_tags["loss_actor"]: 0,
            tb_tags["loss_critic"]: 0,
            "DSAC2/policy_mean-RL iter": 0,
            "DSAC2/policy_std-RL iter": 0,
            "DSAC2/entropy-RL iter": 0,
            "DSAC2/alpha-RL iter": 0,
            tb_tags["alg_time"]: 0,
        }

        return tb_info



