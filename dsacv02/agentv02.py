import torch
import torch.nn as nn
from dsacv02.algov02 import RealDSAC
from dsacv02.actor_critic import Critic, Actor
from dsacv02.replay_bufferv02 import ReplayBuffer
from copy import deepcopy


class Agent:
    def __init__(self, obs_dim, action_dim, n_kernels, learnable_kweights=True, cr_lr_ini=8e-5, cr_lr_fin=1e-6,
                 act_lr_ini=5e-5, act_lr_fin=1e-6, alpha_lr_ini=5e-5, alpha_lr_fin=1e-6,
                 value_min_std=0, value_max_std=5,
                 cr_hl=(256, 256, 256, 256, 256), cr_activ=('gelu', 'gelu', 'gelu', 'gelu', 'gelu'),
                 act_min_std=-20, act_max_std=0.5,
                 act_hl=(256, 256, 256, 256, 256), act_activ=('gelu', 'gelu', 'gelu', 'gelu', 'gelu'),
                 action_low=-1, action_up=1,
                 batch_size=50, t_max=50, tau=0.001, static_alpha=0.2, reward_scale=0.2, gamma=0.99, update_interval=1,
                 auto_alpha=True, log_alpha_ini=1, memory_size=int(5e5), device='cuda:0'
                 ):
        """
        :param obs_dim: Observation dimension
        :param action_dim: Vector with length |A|
        :param cr_lr_ini: Initial critic learning rate
        :param n_kernels: Number of kernels in GMM
        :param learnable_kweights: Whether the networks for value and policy have learnable kernel weights
        :param cr_lr_fin: Final critic learning rate after applying learning rate scheduler
        :param act_lr_ini: Initial Actor learning rate
        :param act_lr_fin: Final actor learning rate
        :param alpha_lr_ini: Initial alpha learning rate
        :param alpha_lr_fin: Final alpha learning rate after applying learning scheduler
        :param value_min_std: Min. standard deviation for /mathcal{Z}, recommended to set to 0 in initial version
        :param value_max_std: Max. standard deviation for /mathcal{Z},
        :param cr_hl: Hidden layers of critic network
        :param cr_activ: All activations of critic network per layer
        :param act_min_std: Min. std. of actor output
        :param act_max_std: Max. std. of actor output
        :param act_hl: Hidden layers of actor network
        :param act_activ: All activations of actor network per layer
        :param action_low: Action low limit
        :param action_up: Action high limit
        :param batch_size: Batch size for MB-SGD
        :param t_max: Horizont for learning scheduler
        :param tau: Soft-upate factor
        :param static_alpha: Entropy temperature in static case
        :param reward_scale: Scaling the reward
        :param gamma: Discount factor
        :param update_interval: Interval for updating target networks
        :param auto_alpha: Whether alpha is a learnable parameter or static
        :param log_alpha_ini: Initial log value of alpha
        :param memory_size: Max. replay buffer size
        :param device: Device on which networks are running
        """
        self.n_kernels = n_kernels
        self.batch_size = batch_size
        self.t_max = t_max
        self.tau = tau
        self.reward_scale = reward_scale
        self.gamma = gamma
        self.update_interval = update_interval
        self.auto_alpha = auto_alpha
        self.log_alpha_ini = log_alpha_ini
        self.static_alpha = static_alpha
        self.mem_size = memory_size
        self.device = device
        self.agent_params = (self.t_max, self.tau, self.reward_scale, self.gamma, self.update_interval,
                             self.auto_alpha, self.static_alpha)

        self.q1: nn.Module = Critic(state_dim=obs_dim, action_dim=action_dim, value_min_std=value_min_std,
                                    value_max_std=value_max_std, hidden_layers=cr_hl, activ=cr_activ, device=device,
                                    learnable_weights=learnable_kweights, n_kernels=self.n_kernels)
        self.q2: nn.Module = Critic(state_dim=obs_dim, action_dim=action_dim, value_min_std=value_min_std,
                                    value_max_std=value_max_std, hidden_layers=cr_hl, activ=cr_activ, device=device,
                                    learnable_weights=learnable_kweights, n_kernels=self.n_kernels)
        self.q1_target: nn.Module = deepcopy(self.q1)
        self.q2_target: nn.Module = deepcopy(self.q2)
        self.policy: nn.Module = Actor(state_dim=obs_dim, action_dim=action_dim, hidden_layers=act_hl,
                                       activation=act_activ, action_min_std=act_min_std, action_max_std=act_max_std,
                                       action_low_lim=action_low, action_up_lim=action_up, device=device,
                                       learnable_weights=learnable_kweights, n_kernels=self.n_kernels)
        self.policy_target = deepcopy(self.policy)
        self.log_alpha = nn.Parameter(torch.tensor(log_alpha_ini, dtype=torch.float64, device=device))

        self.dsac = RealDSAC(critic1=self.q1, critic2=self.q2, critic1_target=self.q1_target,
                             critic2_target=self.q2_target, cr_lr_ini=cr_lr_ini, cr_lr_fin=cr_lr_fin,
                             policy=self.policy, policy_target=self.policy_target, log_alpha=self.log_alpha,
                             actor_lr_ini=act_lr_ini, actor_lr_fin=act_lr_fin, alpha_lr_ini=alpha_lr_ini,
                             alpha_lr_fin=alpha_lr_fin, t_max=t_max, tau=self.tau, static_alpha=self.static_alpha,
                             reward_scale=self.reward_scale, gamma=self.gamma, update_interval=self.update_interval,
                             auto_alpha=self.auto_alpha, target_entropy=-action_dim, n_kernels=n_kernels,
                             device=device)

        self.memory = ReplayBuffer(max_size=memory_size, obs_shape=(obs_dim,), n_actions=action_dim)

    def save_experience_tupel(self, state, action, reward, state_, done):
        """
        - Writes experience tuple to replay buffer
        :type state: np.ndarray
        :type action: np.ndarray
        :type reward: np.ndarray
        :type state_: np.ndarray
        :type done: np.ndarray

        """
        self.memory.store_transition(state, action, reward, state_, done)

    def learn(self, n_learning_iter: int, step_number: int):
        """
        - Chain of method revokes: learn->Update->compute_gradients -> compute_z_loss
                                                                    -> compute_policy_loss
        - After loss, update networks parameters by update_networks() and refresh learning schedule by
          update_lrs()
        :param n_learning_iter: Number of iterations in which the networks are fitted to the current experience.
                                Note that the value networks are always fitted, the fitting of the policy depends on
                                step_number
        :param step_number: Total train step number from main.py
        :return: RL learning quantities (losses, rewards, ...etc.)
        """
        if self.memory.mem_cntr < self.batch_size:
            print(f'Batch size of {self.batch_size} > Stored tupels {self.memory.mem_cntr}')
            print(f'Continue without learning')
            tb_info = self.dsac.get_empty_tb_info()
        else:
            for learning_iter_i in range(n_learning_iter):
                batch_i = self.memory.sample_buffer(batch_size=self.batch_size)
                tb_info = self.dsac.update(batch=batch_i, iteration=step_number)

        return tb_info

    def clear_replay_buffer(self):
        """
        - Sets all entries of replay buffer to zero
        :return: Error
        """
        self.memory.clear_buffer()

        return 0

    def choose_action(self, observation):
        # observation = torch.as_tensor(observation)
        action_mean, action_std, kernel_weights = self.dsac.policy.forward(obs=observation, exp=False)
        action_mean.squeeze_(dim=2)
        action_std.squeeze_(dim=2)
        action_std.abs_()
        if not self.policy.learnable_weights:
            kernel_weights = torch.ones(action_mean.shape[1]) / self.policy.n_kernels
        actions_bounded, probs_bounded = self.dsac.policy.sample_from_action_distr(locs=action_mean,
                                                                                   stds=action_std,
                                                                                   kweights=kernel_weights,
                                                                                   reparameterization=False)

        return actions_bounded.cpu().detach().numpy(), probs_bounded.cpu().detach().numpy()

    def save_checkpoint(self, iter_n: int, path: str, tar_name: str, txt_name: str, replay_txt_name: str):
        """
        - Saves: Networks, optimizers, agent meta-parameters and experiences in replay buffer
        :param iter_n: Global iteration number at which the saving is performed
        :param path: Directory in which checkpoint is saved
        :param tar_name: Checkpoint file name, saved as .tar
        :param txt_name: Agent meta-parameters file name, saved as .txt
        :param replay_txt_name: Name for numpy file storing experiences
        """
        print('Saving checkpoint...')
        complete_tar_file = path + tar_name
        complete_txt_file = path + txt_name
        complete_npy_file = path + replay_txt_name
        # Save network and optimizer parameters
        cr1_optim, cr2_optim, pol_optim, alpha_optim = self.dsac.get_optimizers()
        torch.save({
            'iter_n': iter_n,
            'cr1_state_dict': self.q1.state_dict(),
            'cr1_target_state_dict': self.q1_target.state_dict(),
            'cr1_optim_state_dict': cr1_optim.state_dict(),
            'cr1_lr_schedule_state_dit': self.dsac.q1_lr_schedule.state_dict(),
            'cr2_state_dict': self.q2.state_dict(),
            'cr2_target_state_dict': self.q2_target.state_dict(),
            'cr2_optim_state_dict': cr2_optim.state_dict(),
            'cr2_lr_schedule_state_dit': self.dsac.q2_lr_schedule.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'policy_target_state_dict': self.policy_target.state_dict(),
            'policy_optim_state_dict': pol_optim.state_dict(),
            'policy_lr_schedule_state_dict': self.dsac.pol_lr_schedule.state_dict(),
            'log_alpha_state_dict': self.log_alpha,
            'log_alpha_optim_state_dict': alpha_optim.state_dict(),
            'alpha_lr_schedule_state_dict': self.dsac.alpha_lr_schedule.state_dict()
            },
            complete_tar_file
        )
        # Save non-Pytorch parameters
        agent_meta_data = (self.batch_size, self.t_max, self.tau, self.static_alpha, self.reward_scale,
                           self.gamma,
                           self.update_interval, self.auto_alpha, self.log_alpha_ini, self.mem_size, self.device)
        actor_class_params = self.policy.get_class_info()
        critic_class_params = self.q1.get_class_info()
        learning_rates = self.dsac.get_lr_info()
        with open(complete_txt_file, 'w') as file:
            file.write(str(agent_meta_data))
            file.write('\n')
            file.write(str(actor_class_params))
            file.write('\n')
            file.write(str(critic_class_params))
            file.write('\n')
            file.write(str(learning_rates))

        # Save Replay Experiences
        self.memory.save_experiences(complete_npy_file)

    def load_checkpoint(self, path, tar_name: str, txt_name: str, replay_txt_name: str, load_experience: bool):
        """
        - Loads: Network, optimizers, agent meta-parameters and experiences in replay buffer
        :param path: Directory containing files
        :param tar_name: Checkpoint file name, saved as .tar
        :param txt_name: Agent meta-parameters file name, saved as .txt
        :param replay_txt_name: Name for numpy file storing experiences
        :param load_experience: Whether experience from old replay buffer is used
        :return:
        """
        # Load files
        complete_checkpoint = path + tar_name
        complete_meta_data = path + txt_name
        complete_npy_file = path + replay_txt_name
        checkpoint = torch.load(complete_checkpoint)
        labels = ('agent_params', 'actor_params', 'critic_params', 'learning_rates')
        data = dict()
        with open(complete_meta_data, 'r') as file:
            for label, line in zip(labels, file.readlines()):
                data[label] = eval(line)

        # Extract parameters from checkpoint
        q1_optim_state_dict = checkpoint['cr1_optim_state_dict']
        q2_optim_state_dict = checkpoint['cr2_optim_state_dict']
        policy_optim_state_dict = checkpoint['policy_optim_state_dict']
        log_alpha_optim_state_dict = checkpoint['log_alpha_optim_state_dict']

        # Agent meta-parameters and update attributes
        batch_size, t_max, tau, static_alpha, reward_scale, gamma, update_interval, auto_alpha, log_alpha_ini, \
            mem_size, device = data['agent_params']
        # Actor meta-parameters
        state_dim, action_dim, act_hl, act_n_kernels, act_activation, act_min_std, act_max_std, action_low,\
            action_high, act_learnable_weights, device = data['actor_params']
        # Critic meta-parameters
        state_dim, action_dim, cr_hl, cr_n_kernels, cr_activation, cr_min_std, cr_max_std, cr_learnable_weights, \
            device = data['critic_params']
        # Learning rates
        cr_lr_ini, cr_lr_fin, act_lr_ini, act_lr_fin, alpha_lr_ini, alpha_lr_fin = data['learning_rates']

        # Note actor and critic have same n_kernels, therefore, only the one for act is used
        self.__init__(obs_dim=state_dim, action_dim=action_dim, n_kernels=act_n_kernels,
                      learnable_kweights=act_learnable_weights,
                      cr_lr_ini=cr_lr_ini, cr_lr_fin=cr_lr_fin,
                      act_lr_ini=act_lr_ini, act_lr_fin=act_lr_fin, alpha_lr_ini=alpha_lr_ini,
                      alpha_lr_fin=alpha_lr_fin, value_min_std=cr_min_std, value_max_std=cr_max_std,
                      cr_hl=cr_hl, cr_activ=cr_activation, act_min_std=act_min_std, act_max_std=act_max_std,
                      act_hl=act_hl, act_activ=act_activation, action_low=action_low, action_up=action_high,
                      batch_size=batch_size, t_max=t_max, tau=tau, static_alpha=static_alpha, reward_scale=reward_scale,
                      gamma=gamma, update_interval=update_interval,
                      auto_alpha=auto_alpha, log_alpha_ini=log_alpha_ini, memory_size=mem_size, device=device
                      )

        # Load network, tensor params and learning rate schedule
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_target.load_state_dict(checkpoint['policy_target_state_dict'])
        self.q1.load_state_dict(checkpoint['cr1_state_dict'])
        self.q1_target.load_state_dict(checkpoint['cr1_target_state_dict'])
        self.q2.load_state_dict(checkpoint['cr2_state_dict'])
        self.q2_target.load_state_dict(checkpoint['cr2_target_state_dict'])
        self.log_alpha = checkpoint['log_alpha_state_dict']

        # self.dsac = DSAC(self.q1, self.q2, self.q1_target, self.q2_target, cr_lr_ini=cr_lr_ini, cr_lr_fin=cr_lr_fin,
        #                  policy=self.policy, policy_target=self.policy_target, log_alpha=self.log_alpha,
        #                  actor_lr_ini=act_lr_ini, actor_lr_fin=act_lr_fin, alpha_lr_ini=alpha_lr_ini,
        #                  alpha_lr_fin=alpha_lr_fin, t_max=t_max, tau=self.tau, alpha=self.static_alpha,
        #                  reward_scale=self.reward_scale, gamma=self.gamma, up_interval=self.update_interval,
        #                  auto_alpha=self.auto_alpha, target_entropy=-action_dim)

        self.dsac.q1_optimizer.load_state_dict(q1_optim_state_dict)
        self.dsac.q2_optimizer.load_state_dict(q2_optim_state_dict)
        self.dsac.policy_optimizer.load_state_dict(policy_optim_state_dict)
        self.dsac.alpha_optimizer.load_state_dict(log_alpha_optim_state_dict)

        self.dsac.q1_lr_schedule.load_state_dict(checkpoint['cr1_optim_state_dict'])
        self.dsac.q2_lr_schedule.load_state_dict(checkpoint['cr2_optim_state_dict'])
        self.dsac.pol_lr_schedule.load_state_dict(checkpoint['policy_lr_schedule_state_dict'])
        self.dsac.alpha_lr_schedule.load_state_dict(checkpoint['alpha_lr_schedule_state_dict'])

        self.dsac.log_alpha = self.log_alpha

        return self
