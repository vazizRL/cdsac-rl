import torch
import torch.nn as nn
from dsac_implementation.dsac_algo import DSAC
from dsac_implementation.networks import Critic, Actor
from dsac_implementation.replay_buffer import ReplayBuffer
from copy import deepcopy


class Agent:
    def __init__(self, obs_dim, action_dim, cr_lr_ini=8e-5, cr_lr_fin=1e-6,  act_lr_ini=5e-5,
                 act_lr_fin=1e-6, alpha_lr_ini=5e-5, alpha_lr_fin=1e-6,
                 cr_min_log_std=0, cr_max_log_std=5,
                 cr_hl=(256, 256, 256, 256, 256), cr_activ=('gelu', 'gelu', 'gelu', 'gelu', 'gelu', 'gelu'),
                 act_min_log_std=-20, act_max_log_std=0.5,
                 act_hl=(256, 256, 256, 256, 256), act_activ=('gelu', 'gelu', 'gelu', 'gelu', 'gelu', 'gelu'),
                 action_low=-1, action_up=1,
                 batch_size=50, t_max=50, tau=0.001, alpha=0.2, reward_scale=0.2, gamma=0.99, update_interval=2,
                 auto_alpha=True, memory_size=int(1e5)
                 ):
        """
        :param obs_dim: Observation dimension
        :param action_dim: Vector with length |A|
        :param cr_lr_ini: Initial critic learning rate
        :param cr_lr_fin: Final critic learning rate after applying learning rate scheduler
        :param act_lr_ini: Initial Actor learning rate
        :param act_lr_fin: Final actor learning rate
        :param alpha_lr_ini: Initial alpha learning rate
        :param alpha_lr_fin: Final alpha learning rate after applying learning scheduler
        :param cr_min_log_std: Min. standard deviation for /mathcal{Z}, recommended to set to 0 in initial version
        :param cr_max_log_std: Max. standard devation for /mathcal{Z},
        :param cr_hl: Hidden layers of critic network
        :param cr_activ: All activations of critic network per layer
        :param act_min_log_std: Min. std. of actor output
        :param act_max_log_std: Max. std. of actor output
        :param act_hl: Hidden layers of actor network
        :param act_activ: All activations of actor network per layer
        :param action_low: Action low limit
        :param action_up: Action high limit
        :param batch_size: Batch size for MB-SGD
        :param t_max: Horizont for learning scheduler
        :param tau: Soft-upate factor
        :param alpha: Entropy temperature
        :param reward_scale: Scaling the reward
        :param gamma: Discount factor
        :param update_interval: Interval for updating target networks
        :param auto_alpha: Whether alpha is a learnable parameter or static
        :param memory_size: Max. replay buffer size
        """
        self.batch_size = batch_size
        self.t_max = t_max
        self.tau = tau
        self.reward_scale = reward_scale
        self.gamma = gamma
        self.update_interval = update_interval
        self.auto_alpha = auto_alpha
        self.static_alpha = alpha
        self.agent_params = (self.t_max, self.tau, self.reward_scale, self.gamma, self.update_interval,
                             self.auto_alpha, self.static_alpha)

        self.q1: nn.Module = Critic(obs_dim=obs_dim, action_dim=action_dim, min_log_std=cr_min_log_std,
                                    max_log_std=cr_max_log_std, hidden_layers=cr_hl, activ=cr_activ)
        self.q2: nn.Module = Critic(obs_dim=obs_dim, action_dim=action_dim, min_log_std=cr_min_log_std,
                                    max_log_std=cr_max_log_std, hidden_layers=cr_hl, activ=cr_activ)
        self.q1_target: nn.Module = deepcopy(self.q1)
        self.q2_target: nn.Module = deepcopy(self.q2)
        self.policy: nn.Module = Actor(state_dim=obs_dim, action_dim=action_dim, hidden_layers=act_hl,
                                       activation=act_activ, min_log_std=act_min_log_std, max_log_std=act_max_log_std,
                                       action_low_lim=action_low, action_up_lim=action_up)
        self.policy_target = deepcopy(self.policy)
        self.log_alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))

        self.dsac = DSAC(self.q1, self.q2, self.q1_target, self.q2_target, cr_lr_ini=cr_lr_ini, cr_lr_fin=cr_lr_fin,
                         policy=self.policy, policy_target=self.policy_target, log_alpha=self.log_alpha,
                         actor_lr_ini=act_lr_ini, actor_lr_fin=act_lr_fin, alpha_lr_ini=alpha_lr_ini,
                         alpha_lr_fin=alpha_lr_fin, t_max=t_max, tau=self.tau, alpha=self.static_alpha,
                         reward_scale=self.reward_scale, gamma=self.gamma, up_interval=self.update_interval,
                         auto_alpha=self.auto_alpha, target_entropy=-action_dim)

        self.memory = ReplayBuffer(max_size=memory_size, obs_shape=(obs_dim,), n_actions=action_dim)

    def save_experience_tupel(self, state, action, reward, state_, log_p, done):
        """
        - Writes experience tuple to replay buffer
        :type state: np.ndarray
        :type action: np.ndarray
        :type reward: float
        :type state_: np.ndarray
        :type log_p: np.ndarray
        :type done: bool
        """
        self.memory.store_transition(state, action, reward, state_, log_p, done)

    def learn(self, n_learning_iter: int, step_number: int, clear_mem=False):
        for learning_iter_i in range(n_learning_iter):
            batch_i = self.memory.sample_buffer(self.batch_size)
            self.dsac.update(batch_i, iteration=step_number)

        if clear_mem:
            self.memory.clear_buffer()

        return 0

    def clear_replay_buffer(self):
        self.memory.clear_buffer()

        return 0

    def choose_action(self, observation):
        # observation = torch.as_tensor(observation)
        logits = self.dsac.policy.forward(observation)
        action_distribution = self.dsac.policy.get_act_distr(logits)
        actions, log_prob_actions = action_distribution.sample(reparameterization=False)

        return actions.cpu().detach().numpy(), log_prob_actions.cpu().detach().numpy()

    def save_checkpoint(self, epoch: int, path: str, tar_name: str, txt_name: str):
        """
        - Saves: Networks, optimizers and agent meta-parameters
        :param epoch: Epoch in which saving was performed
        :param path: Directory in which checkpoint is saved
        :param tar_name: Checkpoint file name, saved as .tar
        :param txt_name: Agent meta-parameters file name, saved as .txt
        """
        complete_tar_file = path + tar_name
        complete_txt_file = path + txt_name
        # Save network and optimizer parameters
        cr1_optim, cr2_optim, pol_optim, alpha_optim = self.dsac.get_optimizers()
        torch.save({
            'epoch': epoch,
            'cr1_state_dict': self.q1.state_dict(),
            'cr1_target_state_dict': self.q1_target.state_dict(),
            'cr1_optim_state_dict': cr1_optim.state_dict(),
            'cr2_state_dict': self.q2.state_dict(),
            'cr2_target_state_dict': self.q2_target.state_dict(),
            'cr2_optim_state_dict': cr2_optim.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'policy_optim_state_dict': pol_optim.state_dict(),
            'policy_target_state_dict': self.policy_target.state_dict(),
            'log_alpha_state_dict': self.log_alpha,
            'log_alpha_optim_state_dict': alpha_optim.state_dict()
            },
            complete_tar_file
        )
        # Save
        agent_meta_data = (self.batch_size, self.t_max, self.tau, self.static_alpha, self.reward_scale, self.gamma,
                           self.update_interval, self.auto_alpha)
        actor_meta_params = self.policy.get_class_info()
        critic_class_params = self.q1.get_class_info()
        learning_rates = self.dsac.get_lr_info()
        with open(complete_txt_file, 'w') as file:
            file.write(str(agent_meta_data))
            file.write('\n')
            file.write(str(actor_meta_params))
            file.write('\n')
            file.write(str(critic_class_params))
            file.write('\n')
            file.write(str(learning_rates))

    def load_checkpoint(self, path, tar_name: str, txt_name: str):
        # Load files
        complete_checkpoint = path + tar_name
        complete_meta_data = path + txt_name
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
        batch_size, t_max, tau, static_alpha, reward_scale, gamma, update_interval, auto_alpha = data['agent_params']
        # Actor meta-parameters
        state_dim, action_dim, act_hl, act_activation, act_min_log_std, act_max_log_std, action_low, action_high \
            = data['actor_params']
        # Critic meta-parameters
        state_dim, action_dim, cr_hl, cr_activation, cr_min_log_std, cr_max_log_std = data['critic_params']
        # Learning rates
        cr_lr_ini, cr_lr_fin, act_lr_ini, act_lr_fin, alpha_lr_ini, alpha_lr_fin = data['learning_rates']

        self.__init__(obs_dim=state_dim, action_dim=action_dim, cr_lr_ini=cr_lr_ini, cr_lr_fin=cr_lr_fin,
                      act_lr_ini=act_lr_ini, act_lr_fin=act_lr_fin, alpha_lr_ini=alpha_lr_ini,
                      alpha_lr_fin=alpha_lr_fin, cr_min_log_std=cr_min_log_std, cr_max_log_std=cr_max_log_std,
                      cr_hl=cr_hl, cr_activ=cr_activation, act_min_log_std=act_min_log_std,
                      act_max_log_std=act_max_log_std, act_hl=act_hl, act_activ=act_activation,
                      action_low=action_low, action_up=action_high, batch_size=batch_size, t_max=t_max, tau=tau,
                      alpha=static_alpha, reward_scale=reward_scale, gamma=gamma,
                      update_interval=update_interval, auto_alpha=auto_alpha)

        # Load network/tensor params
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_target.load_state_dict(checkpoint['policy_target_state_dict'])
        self.q1.load_state_dict(checkpoint['cr1_state_dict'])
        self.q1_target.load_state_dict(checkpoint['cr1_target_state_dict'])
        self.q2.load_state_dict(checkpoint['cr2_state_dict'])
        self.q2_target.load_state_dict(checkpoint['cr2_target_state_dict'])
        self.log_alpha = checkpoint['log_alpha_state_dict']

        self.dsac.q1_optimizer.load_state_dict(q1_optim_state_dict)
        self.dsac.q2_optimizer.load_state_dict(q2_optim_state_dict)
        self.dsac.policy_optimizer.load_state_dict(policy_optim_state_dict)
        self.dsac.alpha_optimizer.load_state_dict(log_alpha_optim_state_dict)
        self.dsac.create_lr_schedules()

        return self