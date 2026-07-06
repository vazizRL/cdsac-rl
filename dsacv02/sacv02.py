import torch
import time
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from dsac_old_versions.dsac_implementation.tensorboard_tools import tb_tags
from dsacv02.algov02 import RealDSAC
from dsacv02.tools import get_partial_double_q_selections, get_partial_min_q


class SAC(RealDSAC):
    def __init__(self, critic1, critic2, critic1_target, critic2_target, cr_lr_ini, cr_lr_fin, policy,
                 actor_lr_ini, actor_lr_fin, log_alpha, alpha_lr_ini, alpha_lr_fin, t_max=50, tau=0.001,
                 static_alpha=0.2, reward_scale=0.2, gamma=0.99, update_interval=2, auto_alpha=True, target_entropy=-1,
                 batch_size=None, device='cuda:0'):
        """
        - Implements DSACv0.2, based on DRL and Cramèr Distance
        - Cramer II
        :param critic1: First q-network in the double-Q setting
        :param critic2: Second q-network in the double-Q setting
        :param critic1_target: First q-target
        :param critic2_target: Second q-target
        :param cr_lr_ini: Initial learning rate of both q-networks
        :param cr_lr_fin: Final learning rate of both q-networks
        :param policy: Actor network
        :param log_alpha: Temperament, can be learnable or static
        :param actor_lr_ini: Actor initial learning rate
        :param actor_lr_fin: Actor learning rate at the end of the period
        :param alpha_lr_ini: Initial temperament learning rate, only if learnable
        :param alpha_lr_fin: Final temperament learning rate, only if learnable
        :param t_max: Horizont for learning rate schedule. After that, the learning rate doesn't change
        :param target_entropy: Target entropy in actor distribution, function of action space (s. SAC paper)
        :param tau: Soft update parameter
        :param static_alpha: Static alpha, in case that self.auto_alpha=False
        :param reward_scale: Reward scaling, from SAC
        :param gamma: Discount factor
        :param update_interval: Determines data-generation to updating ratio
        :param auto_alpha: Whether alpha is updated automatically
        :param n_kernels_act: Number of Gaussian kernels in the GMM for Actor
        :param n_kernels_cr: Number of Gaussian kernels in the GMM for Critic
        :param n_supports: Number of supports
        :param ibf: Integral bound factor for numerical calculation of Cramer loss
        """

        super().__init__(critic1=critic1, critic2=critic2, critic1_target=critic1_target,
                         critic2_target=critic2_target, cr_lr_ini=cr_lr_ini, cr_lr_fin=cr_lr_fin, policy=policy,
                         actor_lr_ini=actor_lr_ini, actor_lr_fin=actor_lr_fin, log_alpha=log_alpha,
                         alpha_lr_ini=alpha_lr_ini, alpha_lr_fin=alpha_lr_fin, t_max=t_max, tau=tau,
                         static_alpha=static_alpha, reward_scale=reward_scale, gamma=gamma,
                         update_interval=update_interval, auto_alpha=auto_alpha, target_entropy=target_entropy,
                         n_kernels_act=1, n_kernels_cr=1, n_supports=1, ibf=-1, batch_size=batch_size,
                         device=device)

    @staticmethod
    def switch_autograd_logging(require_grad, models: list):
        """
        - Either turns on or turns off logging of the gradients of models.
        - When calculating policy loss, grads for value networks must not be traced
        :param require_grad: Whether gradients are logged
        :param models: Parameterized as neural network
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
        self.q1_lr_schedule = lr_scheduler.CosineAnnealingLR(self.q1_optimizer, T_max=self.t_max, eta_min=self.q_lr_fin,
                                                             last_epoch=-1, verbose=False)
        self.q2_lr_schedule = lr_scheduler.CosineAnnealingLR(self.q2_optimizer, T_max=self.t_max, eta_min=self.q_lr_fin,
                                                             last_epoch=-1, verbose=False)
        self.pol_lr_schedule = lr_scheduler.CosineAnnealingLR(self.policy_optimizer, T_max=self.t_max,
                                                              eta_min=self.policy_lr_fin, last_epoch=-1, verbose=False)
        self.alpha_lr_schedule = lr_scheduler.CosineAnnealingLR(self.alpha_optimizer, T_max=self.t_max,
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
        :param net: Online network
        :param net_targ: Target network
        """
        tar_complement = 1 - self.tau
        for para, para_targ in zip(net.parameters(), net_targ.parameters()):
            para_targ.data.mul_(tar_complement)
            para_targ.data.add_(self.tau * para.data)

    def evaluate_q(self, obs, actions, q_net):
        """
        - Returns Q(s,a)
        :param obs: obersvations
        :param actions:  atctions
        :param q_net: NN critic
        """
        q, _, _ = q_net(obs, actions)
        # Optional: Add some funneling here to
        return q

    def update_networks(self, iteration: int):
        # Q-Value optimizing step [Every step?]
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # Update policy, alpha and targets every n-th iteration
        if iteration % self.update_interval == 0:
            # Policy optimizing step
            self.policy_optimizer.step()

            # Optional alpha optimizing step
            if self.auto_alpha:
                self.alpha_optimizer.step()

            # Target network updates
            with torch.no_grad():
                # Alternatively, q1_target can be updated with min(q1,q2)
                self.soft_avg_update(self.q1, self.q1_target)
                self.soft_avg_update(self.q2, self.q2_target)

    def compute_q_target(self, rewards, dones, q_means_next, log_probs_a_next):
        """
        - Calculates the entropy-regularized target distribution \mathcal{Z}_H(\cdot|s,a) as a GMM
        - Note: Standard deviations are set to 0 for terminal states
        :param rewards: Rewards received at time t, r_t
        :param dones: Whether s_{t+1} is a terminal state
        :param q_means_next: Next Q-means from kernels
        :param log_probs_a_next: Log probability of the next action
        :return: Target distribution modeles as a GMM
        """
        alpha = self.get_alpha(requires_grad=False)
        # Format rewards, dones
        rewards.unsqueeze_(dim=1)
        dones.unsqueeze_(dim=1)
        q_target = rewards + (1 - dones) * self.gamma * (q_means_next - alpha * log_probs_a_next)

        return q_target.detach()

    def compute_q_loss(self, batch):
        """
        - Only double-Q
        :param batch: Current learning batch
        :return: MSE loss
        """
        states, old_actions, rewards, states_next, dones = batch
        # Convert to tensors, since in main_sac_sb.py, they are stored as Python datatypes
        states = torch.as_tensor(states, dtype=torch.float64).to(self.device)
        old_actions = torch.as_tensor(old_actions, dtype=torch.float64).to(self.device)
        rewards = self.reward_scale * torch.as_tensor(rewards, dtype=torch.float64).to(self.device)
        states_next = torch.as_tensor(states_next, dtype=torch.float64).to(self.device)
        dones = torch.as_tensor(dones, dtype=torch.float64).to(self.device)

        # Probability and value of action
        action_means_next, action_stds_next, kweights_pol = self.policy(obs=states_next, exp=False)

        # The action is only used for Q loss calculation, repara=False, detach from graph
        actions_bounded_next, action_log_probs_next_bounded = self.policy.sample_from_action_distr(
                                                     locs=action_means_next, stds=action_stds_next,
                                                     kweights=kweights_pol, reparameterization=False)

        # Important: Detach; log_prob is a "view" and cant be detached in-place
        actions_bounded_next.detach_()
        action_log_probs_next_bounded = action_log_probs_next_bounded.detach()

        # Q(s',\pi(\cdot | s')) - Double Q approach
        q1_next = self.evaluate_q(obs=states_next, actions=actions_bounded_next, q_net=self.q1_target)
        q2_next = self.evaluate_q(obs=states_next, actions=actions_bounded_next, q_net=self.q2_target)
        q_next_min = get_partial_min_q(means1=q1_next, means2=q2_next)

        # From min Q(s', \pi(\cdot | s')), compute bootstrapped target
        q_target = self.compute_q_target(rewards=rewards, dones=dones, q_means_next=q_next_min,
                              log_probs_a_next=action_log_probs_next_bounded)

        # Q(s,a)
        q1_curr = self.evaluate_q(obs=states, actions=old_actions, q_net=self.q1)
        q2_curr = self.evaluate_q(obs=states, actions=old_actions, q_net=self.q2)

        # Compute loss
        loss1 = F.mse_loss(q1_curr, q_target)
        loss2 = F.mse_loss(q2_curr, q_target)

        # For TB logging
        q_mean = 0.5 * (q1_curr + q2_curr)

        return 0.5 * (loss1 + loss2), q_mean.mean()

    def compute_policy_loss(self, states, actions_curr_pol, log_ps_curr_pol, double_q=False, exp=False):
        """
        - Computes policy loss for gradient calculation
        - WARNING: Make sure that log_ps_curr_pol is NOT detached from current graph as it is the only quantity
                   with which policy gradients can be calculated
        :param states: Batch of states
        :param actions_curr_pol: Actions according to the current policy
        :param log_ps_curr_pol: Log probability of the action according to current policy
        :param double_q: Whether two Q-networks are utilized for mitigating overestimation errors
        :param exp: Deprecated
        :return: Policy loss attached to functional graph for gradient calculation
        """
        # Convert to tensors
        states = torch.as_tensor(states, dtype=torch.float64).to(self.device)
        actions_curr_pol = torch.as_tensor(actions_curr_pol, dtype=torch.float64).to(self.device)
        log_ps_curr_pol = torch.as_tensor(log_ps_curr_pol, dtype=torch.float64).to(self.device)

        if double_q:
            # DO NOT detach, otherwise, actor will not be calculated according to Q!
            q1 = self.evaluate_q(obs=states, actions=actions_curr_pol, q_net=self.q1)
            q2 = self.evaluate_q(obs=states, actions=actions_curr_pol, q_net=self.q2)
            q_min = get_partial_min_q(means1=q1, means2=q2)
        else:
            q_min = self.evaluate_q(obs=states, actions=actions_curr_pol, q_net=self.q1)

        policy_loss = (self.get_alpha(requires_grad=False) * log_ps_curr_pol - q_min).mean()
        entropy = -log_ps_curr_pol.mean().detach()

        return policy_loss, entropy

    def compute_alpha_loss(self, log_ps):
        """
        - Computes the loss of log_alpha. Alpha is put in logarithm form for higher num. stability
        - Note that self.target_entropy is given in log form. In standard form: self.target_entropy.exp()
        :param log_ps:
        :return:
        """
        loss_alpha = - self.log_alpha * (log_ps.detach() + self.target_entropy).mean()

        return loss_alpha

    def compute_gradient(self, batch: tuple, iteration: int, exp=False, double_q=False):
        start_time = time.time()

        # Unpack batch
        states, _, _, _, _ = batch

        # Convert state to tensor
        states = torch.as_tensor(states, dtype=torch.float64)

        # Construct action distribution with reparameterization trick
        means_act, stds_act, kweights_act = self.policy(obs=states, exp=exp)

        # stds_act.abs_()

        # NOTE: Only for Tensorbaord; item() returns a Python native type
        policy_mean = means_act.mean().detach().item()
        policy_std = stds_act.mean().detach().item()

        action_bounded_curr, prob_bounded_curr = self.policy.sample_from_action_distr(
            locs=means_act, stds=stds_act, kweights=kweights_act, reparameterization=True
        )

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        loss_q, means_q = self.compute_q_loss(batch=batch)
        loss_q.backward()

        loss_policy, entropy = None, None
        if iteration % self.update_interval == 0:
            # Switch off autograd when calculating policy loss
            models = [self.q1, self.q2]
            self.switch_autograd_logging(require_grad=False, models=models)
            self.policy_optimizer.zero_grad()
            loss_policy, entropy = self.compute_policy_loss(states=states, actions_curr_pol=action_bounded_curr,
                                                            log_ps_curr_pol=prob_bounded_curr, double_q=double_q,
                                                            exp=exp)
            loss_policy.backward()
            # Switch back on autograd after calculation of policy
            self.switch_autograd_logging(require_grad=True, models=models)

            if self.auto_alpha:
                self.alpha_optimizer.zero_grad()
                loss_alpha = self.compute_alpha_loss(log_ps=prob_bounded_curr)
                loss_alpha.backward()

        tb_info = {
            "DSAC2_Vals/critic_avg_value iter": means_q.mean().detach().item(),
            "DSAC2_Vals/actor_avg_action iter": policy_mean,
            "DSAC2_ActDistr/actor_avg_std iter": policy_std,
            "DSAC2_ActDistr/entropy-RL iter": entropy.detach().item(),
            "DSAC2_Alpha/alpha-RL iter": self.get_alpha(requires_grad=False),
            tb_tags["loss_actor"]: loss_policy.detach().item(),
            tb_tags["loss_critic"]: loss_q.detach().item(),
            tb_tags["alg_time"]: (time.time() - start_time) * 1000
        }

        return tb_info

    """ 
    /Internally Called 
    """

    def force_alpha_to_val(self, new_alpha: torch.nn.Parameter):
        curr_lr = self.alpha_lr_schedule.get_lr().pop(0)
        self.log_alpha = new_alpha
        self.alpha_optimizer = Adam([self.log_alpha], lr=curr_lr)
        self.alpha_lr_schedule = lr_scheduler.CosineAnnealingLR(self.alpha_optimizer, T_max=self.t_max,
                                                                eta_min=self.alpha_lr_fin, last_epoch=-1, verbose=False)

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

    def update(self, batch: tuple, iteration: int, double_q=False, exp=False):
        """
        - Wrapper; Calculate gradient and perform network optimization step
        - Perform lr scheduler step
        :param batch: Mini-batch
        :param iteration: Iteration number, necessary to determine update-interval
        :param double_q: Whether two Q networks
        :param exp: Whether logits are exponentiated
        :return: Dict containing quantities for logging in tensorboard
        """
        tb_info = self.compute_gradient(batch=batch, iteration=iteration, double_q=double_q, exp=exp)
        self.update_networks(iteration)
        self.update_lrs()

        return tb_info

    def update_lrs(self):

        self.q1_lr_schedule.step()
        self.q2_lr_schedule.step()
        self.pol_lr_schedule.step()
        self.alpha_lr_schedule.step()

    def get_empty_tb_info(self):
        # list(self.q1.q.parameters())[-2].grad.mean()
        tb_info = {
            "DSAC2_Vals/critic_avg_value iter": 0,
            "DSAC2_Vals/actor_avg_action iter": 0,
            "DSAC2_ActDistr/actor_avg_std iter": 0,
            "DSAC2_ActDistr/entropy-RL iter": 0,
            "DSAC2_Alpha/alpha-RL iter": 0,
            tb_tags["loss_actor"]: 0,
            tb_tags["loss_critic"]: 0,
            tb_tags["alg_time"]: 0,
        }

        return tb_info



