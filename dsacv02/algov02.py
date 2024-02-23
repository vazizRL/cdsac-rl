import torch
import torch.nn as nn
import torch.distributions as distr
import time
from dsacv02.gmm_reparameterization.mixture_same_family import ReparameterizedMixtureSameFamilyMod as RMM
from torch.optim import Adam, lr_scheduler
from dsac_old_versions.dsac_implementation.tensorboard_tools import tb_tags
from dsacv02.tools import cramer_torch, approx_integral_bounds, get_double_q_selections, \
     get_partial_double_q_selections


class RealDSAC:
    def __init__(self, critic1, critic2, critic1_target, critic2_target, cr_lr_ini, cr_lr_fin, policy, policy_target,
                 actor_lr_ini, actor_lr_fin, log_alpha, alpha_lr_ini, alpha_lr_fin, t_max=50, tau=0.001,
                 static_alpha=0.2,
                 reward_scale=0.2, gamma=0.99, update_interval=2, auto_alpha=True, target_entropy=-1, n_kernels=1,
                 device='cuda:0'):
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
        :param target_entropy: Target entropy in actor distribution, function of action space (s. SAC paper)
        :param tau: Soft update parameter
        :param static_alpha: Static alpha, in case that self.auto_alpha=False
        :param reward_scale: Reward scaling, from SAC
        :param gamma: Discount factor
        :param update_interval: Determines data-generation to updating ratio
        :param auto_alpha: Whether alpha is updated automatically
        :param n_kernels: Number of Gaussian kernels in the GMM
        :param kwargs:
        """

        # Initialize device
        self.device = device

        self.q1: nn.Module = critic1
        self.q2: nn.Module = critic2
        self.q1_target: nn.Module = critic1_target
        self.q2_target: nn.Module = critic2_target

        self.policy: nn.Module = policy
        self.policy_target: nn.Module = policy_target

        self.n_kernels = n_kernels

        # Do not track gradients for target networks
        self.switch_autograd_logging(require_grad=False, models=[self.q1_target, self.q2_target, self.policy_target])

        # NOTE: log_alpha is already given as a torch tensor, with initial value specified in agentv02.py
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
        # Alpha is only a simple tensor with a scalar value
        self.alpha_optimizer = Adam([self.log_alpha], lr=self.alpha_lr_ini)

        self.q1_lr_schedule = None
        self.q2_lr_schedule = None
        self.pol_lr_schedule = None
        self.alpha_lr_schedule = None
        self.create_lr_schedules()

        # Algorithm parameters
        self.reward_scale = torch.tensor(reward_scale).to(self.device)
        self.gamma = torch.tensor(gamma).to(self.device)
        self.tau = torch.tensor(tau).to(self.device)
        self.target_entropy = torch.tensor(target_entropy).to(self.device)
        self.static_alpha = torch.tensor(static_alpha).to(self.device)
        self.auto_alpha = auto_alpha
        self.update_interval = update_interval

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

    @staticmethod
    def generate_gmm_distr(means, stds, kweights, multivar=False):
        """
        TODO: Refactor to avoid if-statement
        - Generates either a multivariate or standard Gaussian Mxiture Model
        :param means: Kernel means
        :param stds: Kernel standard deviations
        :param kweights: Kernel weights
        :param multivar: CHANGE! Whether components of GMM are multivariate or not
        :return: Returns a GMM
        """
        mix_distr = distr.Categorical(probs=kweights)
        if multivar:
            comp_distr = distr.MultivariateNormal(loc=means, covariance_matrix=stds)
        else:
            comp_distr = distr.Normal(loc=means, scale=stds)
        zcal = RMM(mixture_distribution=mix_distr, component_distribution=comp_distr)

        return zcal

    def evaluate_z(self, obs, actions, znet, exp=False, sample=False, reparameterize=False):
        """
        - Evaluates Z and can return mathcal{Z} and its samples
        - Note: that stds can not be negative
        :param obs: observation
        :param actions: actions
        :param znet: Q-value distribution approximator function to be evaluated
        :param exp: Whether network outputs are exponentiated or not
        :param sample: If True, samples from mathcal{Z} are returned
        :param reparameterize: Use implicit reparameterization trick for getting the samples
        :return: Either (Z, GMM_PDF, Means, Stds, K_weights) or (None, GMM_PDF, Means, Stds, K_weights) are returned
        """
        # (B, K, Q)
        means, stds, kernel_weights = znet(obs, actions, exp=exp)
        means.squeeze_(dim=2)
        stds.squeeze_(dim=2)
        stds.abs_()
        if kernel_weights is None:
            mb_size = actions.shape[0]
            kernel_weights = torch.ones(mb_size, self.n_kernels) / self.n_kernels
        gmm = self.generate_gmm_distr(means=means, stds=stds, kweights=kernel_weights, multivar=False)

        gmm_sample = None
        if sample:
            batch_size = obs.shape[0]
            if reparameterize:
                gmm_sample = gmm.rsample()
                gmm_sample.unsqueeze(dim=1)
            else:
                gmm_sample = gmm.sample()
                gmm_sample.unsqueeze(dim=1)

        return gmm_sample, gmm, means, stds, kernel_weights

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

            with torch.no_grad():
                self.soft_avg_update(self.q1, self.q1_target)
                self.soft_avg_update(self.q2, self.q2_target)
                self.soft_avg_update(self.policy, self.policy_target)

    def compute_target_distribution(self, rewards, dones, q_means_next, stds_next, kernel_weights, log_probs_a_next):
        """
        - Calculates the entropy-regularized target distribution \mathcal{Z}_H(\cdot|s,a) as a GMM
        - Note: Standard deviations are set to 0 for terminal states
        :param rewards: Rewards received at time t, r_t
        :param dones: Whether s_{t+1} is a terminal state
        :param q_means_next: Next Q-means from kernels
        :param stds_next: Next standard deviation from kernels
        :param kernel_weights: Weights of the Gaussian kernels in the GMM
        :param log_probs_a_next: Log probability of the next action
        :return: Target distribution modeles as a GMM
        """
        next_batch_size = q_means_next.shape[0]
        alpha = self.get_alpha(requires_grad=False)
        # Compute target from mean Q.
        rewards.unsqueeze_(dim=1)
        dones.unsqueeze_(dim=1)
        q_means_target = rewards + (1 - dones) * self.gamma * (q_means_next - alpha * log_probs_a_next)
        stds_next = (1-dones) * stds_next + torch.tensor(1e-10, dtype=torch.float64)

        cat_distr = distr.Categorical(probs=kernel_weights)
        comp_distr = distr.Normal(loc=q_means_target, scale=stds_next)

        target_distribution = RMM(mixture_distribution=cat_distr, component_distribution=comp_distr)
        # Target is always used for gradient calculations, so always rsample
        target_samples = target_distribution.rsample()

        return target_distribution, target_samples

    def compute_z_loss(self, batch, double_q=False, integral_bound_factor=5, exp=False):
        """
        - Calculates the Q-distribution GMM-to-GMM cramer loss
        :param integral_bound_factor:
        :param batch: Sampled batch to learn from
        :param double_q: Whether to apply double-Q for overestimation mitigation (not cla)
        :return: Q-loss attached to functional graph for gradient calculation
        """
        states, old_actions, rewards, states_next, dones = batch
        # Convert to tensors, since in main.py, they are stored as Python datatypes
        states = torch.as_tensor(states, dtype=torch.float64).to(self.device)
        old_actions = torch.as_tensor(old_actions, dtype=torch.float64).to(self.device)
        rewards = self.reward_scale * torch.as_tensor(rewards, dtype=torch.float64).to(self.device)
        states_next = torch.as_tensor(states_next, dtype=torch.float64).to(self.device)
        dones = torch.as_tensor(dones, dtype=torch.float64).to(self.device)

        # Probability and value of action
        action_means_next, action_stds_next, kweights_pol = self.policy_target(states_next)
        action_means_next.squeeze_(dim=2)
        action_stds_next.squeeze_(dim=2)
        # The action is only used for Q loss calculation, repara=False, detach from graph
        actions_bounded_next, action_log_probs_next_bounded = self.policy_target.sample_from_action_distr(
                                                     locs=action_means_next, stds=action_stds_next,
                                                     kweights=kweights_pol, reparameterization=False)
        # Important: In-place operations
        actions_bounded_next.detach_()
        action_log_probs_next_bounded.detach_()

        # Calculate Q_{\theta}(s',a') according to q1
        _, zcal1_next, means1_next, stds1_next, kweights1_next = \
            self.evaluate_z(obs=states_next, actions=actions_bounded_next, znet=self.q1_target,
                            exp=exp, sample=False, reparameterize=True)
        if double_q:
            # Calculate current according to q1, q2 and target distributions according to q2
            _, zcal1, means1, stds1, kweights1 = self.evaluate_z(obs=states, actions=old_actions, znet=self.q1,
                                                                 exp=exp, sample=False, reparameterize=True)
            _, zcal2, means2, stds2, kweights2 = self.evaluate_z(obs=states, actions=old_actions, znet=self.q2,
                                                                 exp=exp, sample=False, reparameterize=True)
            _, zcal2_next, means2_next, stds2_next, kweights2_next = \
                self.evaluate_z(obs=states_next, actions=actions_bounded_next, znet=self.q2_target,
                                exp=exp, sample=False, reparameterize=True)

            means_min, means_next_min, stds_selected_min, stds_next_selected_min, kweights, \
                kweights_next_selected_min = get_double_q_selections(means1=means1, means2=means2,
                                                                     means1_next=means1_next,
                                                                     means2_next=means2_next, stds1=stds1, stds2=stds2,
                                                                     stds1_next=stds1_next, stds2_next=stds2_next,
                                                                     kweights1=kweights1, kweights2=kweights2,
                                                                     kweights1_next=kweights1_next,
                                                                     kweights2_next=kweights2_next)

            # Calculate current distribution
            zcal = self.generate_gmm_distr(means_min, stds_selected_min, kweights)
            z = zcal.rsample(sample_shape=batch.shape[0])

            # Calculate target distribution
            zcal_next, z_next = self.compute_target_distribution(rewards=rewards, dones=dones,
                                                                 q_means_next=means_next_min,
                                                                 stds_next=stds_next_selected_min,
                                                                 kernel_weights=kweights_next_selected_min,
                                                                 log_probs_a_next=action_log_probs_next_bounded)

            # Calculate integral bounds
            int_bound_low, int_bound_up = approx_integral_bounds(means_curr=means_min,
                                                                 means_target=means_next_min,
                                                                 stds_curr=stds_selected_min,
                                                                 stds_target=stds_next_selected_min,
                                                                 factor=integral_bound_factor,
                                                                 mean_std=True)
        else:
            # Calculate current and target distributions, NOTE: Evaluation for \mathcal{Z}(|,s',a') already done
            # before If-statement
            z, zcal, means, stds, kweights = self.evaluate_z(obs=states, actions=old_actions, znet=self.q1,
                                                             exp=exp, sample=True, reparameterize=True)
            zcal_next, z_next = self.compute_target_distribution(rewards=rewards, dones=dones, q_means_next=means1_next,
                                                                 stds_next=stds1_next, kernel_weights=kweights1_next,
                                                                 log_probs_a_next=action_log_probs_next_bounded)
            # Calculate integral bounds
            int_bound_low, int_bound_up = approx_integral_bounds(means_curr=means, means_target=means1_next,
                                                                 stds_curr=stds, stds_target=stds1_next,
                                                                 factor=integral_bound_factor,
                                                                 mean_std=True)

        # Detach integral bounds from graph
        int_bound_low.detach_(), int_bound_up.detach_()

        # Calculate loss with batch-sensitive Cràmer distance on PDFs
        q_loss = cramer_torch(pdf_target=zcal_next, pdf_curr=zcal, int_l=int_bound_low, int_u=int_bound_up,
                              spacing=1e-3, dev=self.device)

        return q_loss.mean(), means.mean(), stds.mean(), kweights

    def compute_policy_loss(self, states, actions_curr_pol, log_ps_curr_pol, double_q=False, exp=False):
        """
        - Computes policy loss for gradient calculation
        - WARNING: Make sure that log_ps_curr_pol is NOT detached from current graph as it is the only quantity
                   with which policy gradients can be calculated
        :param states: Batch of states
        :param actions_curr_pol: Actions according to the current policy
        :param log_ps_curr_pol: Log probability of the action according to current policy
        :param double_q: Whether two Q-networks are utilized for mitigating overestimation errors
        :param exp: Whether logits of networks are exponentiated or not
        :return: Policy loss attached to functional graph for gradient calculation
        """
        # Convert to tensors
        states = torch.as_tensor(states, dtype=torch.float64).to(self.device)
        actions_curr_pol = torch.as_tensor(actions_curr_pol, dtype=torch.float64).to(self.device)
        log_ps_curr_pol = torch.as_tensor(log_ps_curr_pol, dtype=torch.float64).to(self.device)

        if double_q:
            _, _, q_means1, stds1, kweights1 = self.evaluate_z(obs=states, actions=actions_curr_pol, znet=self.q1,
                                                               sample=False, exp=exp, reparameterize=False)
            _, _, q_means2, stds2, kweights2 = self.evaluate_z(obs=states, actions=actions_curr_pol, znet=self.q2,
                                                               sample=False, exp=exp, reparameterize=False)
            # Detach from Q-networks [Inplace Operation] !
            q_means1.detach_(), q_means2.detach_(), stds1.detach_(), kweights1.detach_(), stds2.detach_(),
            kweights2.detach_()

            means_min, stds_selected_min, kweights_selected_min = \
                get_partial_double_q_selections(means1=q_means1, means2=q_means2, stds1=stds1, stds2=stds2,
                                                kweights1=kweights1, kweights2=kweights2)
        else:
            _, _, means_min, stds_selected_min, kweights_selected_min = \
                self.evaluate_z(obs=states, actions=actions_curr_pol, znet=self.q1, sample=False, exp=exp,
                                reparameterize=False)

            # means_min = means_min.detach()
            means_min = means_min
            kweights_selected_min = kweights_selected_min.detach()

        gmm_mean = (means_min * kweights_selected_min).sum(dim=1)

        # Not calculated according to Z! If Z, reparameterization is needed
        gmm_mean.unsqueeze_(dim=1)

        policy_loss = (self.get_alpha(requires_grad=False) * log_ps_curr_pol - gmm_mean).mean()
        entropy = -log_ps_curr_pol.mean().detach()

        return policy_loss, entropy

    def compute_alpha_loss(self, log_ps):
        loss_alpha = - self.log_alpha * (log_ps.detach() + self.target_entropy).mean()

        return loss_alpha

    def compute_gradient(self, batch: tuple, iteration: int, exp=False):
        start_time = time.time()

        # Unpack batch
        states, _, _, _, _ = batch

        # Convert state to tensor
        states = torch.as_tensor(states, dtype=torch.float64)

        # Construct action distribution with reparameterization trick
        means_act, stds_act, kweights_act = self.policy(obs=states, exp=exp)
        means_act.squeeze_(dim=2)
        stds_act.squeeze_(dim=2)
        stds_act.abs_()

        # NOTE: Only for Tensorbaord; item() returns a Python native type
        policy_mean = means_act.mean().detach().item()
        policy_std = stds_act.mean().detach().item()

        # TODO: prob_bounded is only the logit, calculated from non-exponentiated inputs
        action_bounded_curr, prob_bounded_curr = self.policy.sample_from_action_distr(
            locs=means_act, stds=stds_act, kweights=kweights_act, reparameterization=True
        )

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()

        # Compute Z-Loss, NOTE: Check exponentiation
        loss_q, mean_q, std_mean, kweights_cr = self.compute_z_loss(batch=batch, double_q=False,
                                                                    integral_bound_factor=4, exp=False)
        loss_q.backward()

        loss_policy, entropy = None, None
        if iteration % self.update_interval == 0:
            # Switch off autograd when calculating policy loss
            # TODO: Alternatively, disconnect the leaf from the computational graph with context-manager
            #       [Before optimizer step?]
            models = [self.q1, self.q2]
            self.switch_autograd_logging(require_grad=False, models=models)
            self.policy_optimizer.zero_grad()
            loss_policy, entropy = self.compute_policy_loss(states=states, actions_curr_pol=action_bounded_curr,
                                                            log_ps_curr_pol=prob_bounded_curr, double_q=False)
            loss_policy.backward()
            # Switch back on autograd after calculation of policy
            self.switch_autograd_logging(require_grad=True, models=models)

            if self.auto_alpha:
                self.alpha_optimizer.zero_grad()
                loss_alpha = self.compute_alpha_loss(log_ps=prob_bounded_curr)
                loss_alpha.backward()

        tb_info = {
            "DSAC2_Vals/gmm_critic_avg_value iter": mean_q.detach().item(),
            "DSAC2_CrDistr/gmm_critic_avg_std iter": std_mean.detach().item(),
            "DSAC2_Vals/gmm_actor_avg_action iter": policy_mean,
            "DSAC2_ActDistr/gmm_actor_avg_std iter": policy_std,
            "DSAC2_ActDistr/entropy-RL iter": entropy.detach().item(),
            "DSAC2_Alpha/alpha-RL iter": self.get_alpha(requires_grad=False),
            tb_tags["loss_actor"]: loss_policy.detach().item(),
            tb_tags["loss_critic"]: loss_q.detach().item(),
            tb_tags["alg_time"]: (time.time() - start_time) * 1000,
        }

        for i in range(self.n_kernels):
            tb_info[f"DSAC2_ActDistr/gmm_actor_avg_k{i+1}_weight iter"] = kweights_act[:, i].mean().detach().item()
            tb_info[f"DSAC2_CrDistr/gmm_critic_avg_k{i+1}_weight iter"] = kweights_cr[:, i].mean().detach().item()

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
        self.q1_lr_schedule.step()
        self.q2_lr_schedule.step()
        self.pol_lr_schedule.step()
        self.alpha_lr_schedule.step()
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
            "DSAC2_Vals/gmm_critic_avg_value iter": 0,
            "DSAC2_CrDistr/gmm_critic_avg_std iter": 0,
            "DSAC2_CrDistr/gmm_critic_avg_k1_weight iter": 0,
            "DSAC2_CrDistr/gmm_critic_avg_k2_weight iter": 0,
            "DSAC2_Vals/gmm_actor_avg_action iter": 0,
            "DSAC2_ActDistr/gmm_actor_avg_std iter": 0,
            "DSAC2_ActDistr/gmm_actor_avg_k1_weight iter": 0,
            "DSAC2_ActDistr/gmm_actor_avg_k2_weight iter": 0,
            "DSAC2_ActDistr/entropy-RL iter": 0,
            "DSAC2_Alpha/alpha-RL iter": 0,
            tb_tags["loss_actor"]: 0,
            tb_tags["loss_critic"]: 0,
            tb_tags["alg_time"]: 0,
        }

        return tb_info



