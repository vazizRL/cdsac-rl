"""
- Todo: If C-DSAC is also adjusted: Two target_qs, min(target_q1, target_q2), update targets acc. to q1, q2
- Implemented without value function
- Implemented with Double-Q method
- Actor and Critic are given to SAC from DSACv02, but initiated if loaded
- Uses ReplayBuffer implementation from DSACv02
- Similar to SB3 implementation
"""
import torch as T
import torch.nn.functional as F
import torch.optim as optim
import os
from copy import deepcopy
from dsacv02.replay_bufferv02 import ReplayBuffer
from dsacv02.actor_critic import Actor, Critic


class Agent:
    def __init__(self, policy_net, critic1_net, critic2_net, alpha=0.0003, beta=0.0003, input_dims=(8,), env=None, gamma=0.99,
                 n_actions=2, max_size=int(1e6), tau=0.005, batch_size=256, reward_scale=2,
                 auto_temp=False, temp_log_ini=-2, omega=0.0003, static_temp=1):
        """
        - Simple SAC agent, optionally entropy coefficient can be trained
        :param policy_net: Actor network
        :param critic1_net: Critic1 in Double-Q
        :param critic2_net: Critic2 in Double-Q
        :param alpha: Actor LR
        :param beta: Critic LR
        :param input_dims:
        :param env: Environment, must have gym architecture
        :param gamma: Discounting factor
        :param n_actions: Number of actions
        :param max_size: Max size of replay buffer
        :param tau: Soft update parameter
        :param batch_size: Mini batch size
        :param reward_scale:
        :param auto_temp: Whether
        :param temp_log_ini: If auto_temp, then this will be initial LOG value of temperature
        :param omega: Temp LR
        :param static_temp: If auto_alpha is False, then this value will be used for training the value network
        """
        self.gamma = gamma
        self.tau = tau
        self.mem_size = max_size
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.alpha = alpha
        self.beta = beta
        self.policy = policy_net
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=alpha)
        self.policy_target = deepcopy(self.policy)

        self.q1 = critic1_net
        self.q1_optimizer = optim.Adam(self.q1.parameter(), lr=beta)

        self.q2 = critic2_net
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=beta)

        self.q_target = deepcopy(self.q1)

        self.scale = reward_scale

        # Temperature Settings
        self.static_temp = static_temp
        self.target_entropy = -self.n_actions
        self.auto_temp = auto_temp
        self.temp_log_ini = temp_log_ini
        # self.temp = T.nn.Parameter(T.tensor(T.e**ini_temp, dtype=T.float64, device=self.actor.device))
        self.temp_log = T.nn.Parameter(T.tensor(temp_log_ini, dtype=T.float64, device=self.policy.device))
        self.temp_optim = optim.Adam([self.temp_log], lr=omega)
        self.omega = omega

        self.empty_tb_data = {'SACQ/q1_val': 0,
                              'SACQ/q2_val': 0,
                              'SACLoss/critic_loss': 0,
                              'SACLoss/actor_loss': 0,
                              'SACLoss/value_loss': 0,
                              'SACPolicy/avg. policy std plain': 0,
                              'SACPolicy/avg. policy std repara': 0
                              }

        self.generic_kw = T.tensor(1, device=self.policy.device)

    def get_temperature(self):
        if self.auto_temp:
            temp = self.temp_log.exp()
        else:
            temp = self.static_temp
        return temp

    def choose_action(self, observation):
        """
        - Chooses action for actor class as implemented in dsac2
        :param observation: (s,a,r,s')
        :return: action
        """
        state = T.tensor([observation]).to(self.policy.device)
        means, stds, _ = self.policy(state)
        actions, _ = self.policy.sample_from_action_distr(loc=means, stds=stds, kweights=_, reparametrize=False)

        # Send it to CPU, detach from graph, turn into np.ndarray and take zeroth element
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def soft_avg_update(self, net: T, net_targ: T):
        """
        - \tau is a very small value specifying the rate of change of the target network
        :param net: Online network
        :param net_targ: Target network
        """
        tar_complement = 1 - self.tau
        for para, para_targ in zip(net.parameters(), net_targ.parameters()):
            para_targ.data.mul_(tar_complement)
            para_targ.data.add_(self.tau * para.data)

    def update_target_parameters(self):
        """
        - Implements soft and hard updates of the target soft value function
        :param tau: Update weight
        """
        # SB: Update targets according to onlines:
        # self.soft_avg_update(self.q1, self.q1_target)
        # self.soft_avg_update(self.q2, self.q2_target)
        self.soft_avg_update(self.q1, self.q_target)
        self.soft_avg_update(self.policy, self.policy_target)

    def get_soft_value_target(self, rewards, states_new):
        """
        - Detach action, Q(s',a')
        - Use actor and critic target networks for target values
        :param rewards: Reward sampled from Replay Buffer
        :param states_new: Next step after state sampled from Replay Buffer
        """
        with T.no_grad():
            actions_next_mean, actions_next_std, _ = self.policy_target(states_new)
            actions_next, actions_next_log = self.policy_target.sample_from_action_distr(locs=actions_next_mean,
                                                stds=actions_next_std, kweights=self.generic_kw,
                                                reparameterization=False)

            q_val, _, _ = self.q_target(states_new, actions_next)
            omega = self.get_temperature()

        target = rewards + self.gamma * (q_val - omega * actions_next_log)

        return target

    def learn(self):
        """
        - Q and Pi updates
        """
        if self.memory.mem_cntr < self.batch_size:
            print('Memory < Batch Size, Learning not initiated')
            return self.empty_tb_data

        state, old_actions, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        # Transform numpy into PyTorch tensors
        state = T.tensor(state, dtype=T.float64).to(self.policy.device)
        old_actions = T.tensor(old_actions, dtype=T.float64).to(self.policy.device)
        reward = T.tensor(reward, dtype=T.float64).to(self.policy.device)
        state_next = T.tensor(new_state, dtype=T.float64).to(self.policy.device)
        done = T.tensor(done, dtype=T.float64).to(self.policy.device)

        """Calculate Critic Loss """
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()

        q_hat = self.get_soft_value_target(rewards=reward, states_new=state_next)

        q1_old_policy, _, _ = self.q1(state, old_actions)
        q2_old_policy, _, _ = self.q2(state, old_actions)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.q1.optimizer.step()
        self.q2.optimizer.step()

        """ Calculate Actor Loss """
        actions_mean, actions_std, _ = self.policy_target(state)
        actions_online, actions_online_log = self.policy_target.sample_from_action_distr(
            locs=actions_mean,
            stds=actions_std,
            kweights=self.generic_kw,
            reparameterization=True)

        q1_online_pol, _, _ = self.q1(state, actions_online)
        q2_online_pol, _, _ = self.q2(state, actions_online)
        q_online_pol_min = T.min(q1_online_pol, q2_online_pol)

        actor_loss = actions_online_log - q_online_pol_min
        # Get mean of the batch
        actor_loss = T.mean(actor_loss)
        self.policy_optimizer.zero_grad()
        # actor_loss.backward(retain_graph=True)
        actor_loss.backward()
        self.policy_optimizer.step()
        self.update_target_parameters()

        """ Compute Entropy Coefficient Loss"""
        if self.auto_temp:
            self.temp_optim.zero_grad()
            loss_temp = - self.temp_log * (actions_online_log.detach() + self.target_entropy).mean()
            loss_temp.backward()
            self.temp_optim.step()

        """ Tensorboard Quantities"""
        tb_info = {'SACQ/q1_val': q1_online_pol.mean().detach(),
                   'SACQ/q2_val': q2_online_pol.mean().detach(),
                   'SACLoss/critic_loss': critic_loss.detach().item(),
                   'SACLoss/actor_loss': actor_loss.detach().item(),
                   'SACPolicy/avg. policy STD': actions_std.mean().detach().item(),
                   'SACPolicy/avg. policy log entropy': actions_online_log.mean().detach().item(),
                   'SACPolicy/temperature': self.temp_log.mean().detach(),
                   }

        return tb_info

    """Save and Load Functions"""
    def save_models(self, iter_n: int, path: str, tar_name: str, txt_name: str, replay_txt_name: str):
        """
        - Saves: Networks, optimizers, agent meta-parameters and experiences in replay buffer
        :param iter_n: Global iteration number at which the saving is performed
        :param path: Directory in which checkpoint is saved
        :param tar_name: Checkpoint file name, saved as .tar
        :param txt_name: Agent meta-parameters file name, saved as .txt
        :param replay_txt_name: Name for numpy file storing experiences
        """
        print('. . . saving models . . .')
        complete_tar_file = path + '/' + tar_name
        complete_txt_file = path + '/' + txt_name
        complete_npy_file = path + '/' + replay_txt_name
        T.save({
            'iter_n': iter_n,
            'q1_state_dict': self.q1.state_dict(),
            'q1_optim_state_dict': self.q1_optimizer.state_dict(),

            'q2_state_dict': self.q2.state_dict(),
            'q2_optim_state_dict': self.q2_optimizer.state_dict(),

            'q_target_state_dict': self.q_target.state_dict(),

            'policy_state_dict': self.policy.state_dict(),
            'policy_target_state_dict': self.policy_target.state_dict(),
            'policy_optim_state_dict': self.policy_optimizer.state_dict(),
            'temp_log_parameter': self.temp_log,
            'temp_log_parameter_optim': self.temp_optim.state_dict(),
            },
            complete_tar_file
        )

        agent_meta_data = (self.batch_size, self.tau, self.alpha, self.beta, self.omega,  self.static_temp, self.scale,
                           self.gamma, self.auto_temp, self.temp_log_ini, self.mem_size, self.policy.device)
        actor_class_params = self.policy.get_class_info()
        critic_class_params = self.q1.get_class_info()
        learning_rates = (self.alpha, self.beta)

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

    def load_models(self, path, tar_name: str, txt_name: str, replay_npy_name: str, load_experience: bool):
        """
        - Loads: Network, optimizers, agent meta-parameters and experiences in replay buffer
        :param path: Directory containing files
        :param tar_name: Checkpoint file name, saved as .tar
        :param txt_name: Agent meta-parameters file name, saved as .txt
        :param replay_npy_name: Name for numpy file storing experiences
        :param load_experience: Whether experience from old replay buffer is used
        :return:
        """
        print('. . . loading modles . . .')
        # Load files
        complete_checkpoint = path + tar_name
        complete_meta_data = path + txt_name
        complete_npy_file = path + replay_npy_name
        checkpoint = T.load(complete_checkpoint)
        labels = ('agent_params', 'actor_params', 'critic_params')
        data = dict()
        with open(complete_meta_data, 'r') as file:
            for label, line in zip(labels, file.readlines()):
                data[label] = eval(line)

        # Agent Params
        batch_size, tau, alpha, beta, omega, static_temp, scale, gamma, auto_temp, temp_log_ini, mem_size,\
                device = data['agent_params']
        state_dim, action_dim, act_hl, n_kernels_act, act_activation, act_min_std, act_max_std, action_low,\
            action_high, act_learnable_weights, device = data['actor_params']
        state_dim, action_dim, cr_hl, n_kernels_cr, cr_activation, cr_min_std, cr_max_std, cr_learnable_weights, \
                device = data['critic_params']

        # Initiate policy and critic networks
        policy = Actor(state_dim, action_dim, hidden_layers=act_hl, n_kernels=n_kernels_act,
                 activation=act_activation, action_min_std=act_min_std, action_max_std=act_max_std,
                       action_low_lim=action_low, action_up_lim=action_high,
                 learnable_weights=act_learnable_weights, device=device)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        q1 = Critic(state_dim, action_dim, hidden_layers=cr_hl, n_kernels=n_kernels_cr, activ=cr_activation,
                 value_min_std=cr_min_std, value_max_std=cr_max_std, learnable_weights=cr_learnable_weights,
                    device=device)
        q1.load_state_dict(checkpoint['q1_state_dict'])
        q2 = Critic(state_dim, action_dim, hidden_layers=cr_hl, n_kernels=n_kernels_cr, activ=cr_activation,
                 value_min_std=cr_min_std, value_max_std=cr_max_std, learnable_weights=cr_learnable_weights,
                    device=device)
        q2.load_state_dict(checkpoint['q2_state_dict'])

        self.__init__(policy_net=policy, critic1_net=q1, critic2_net=q2, alpha=alpha, beta=beta, input_dims=state_dim,
                      env=None, gamma=gamma, n_actions=action_dim, max_size=mem_size, tau=tau, batch_size=batch_size,
                      reward_scale=scale, auto_temp=auto_temp, temp_log_ini=temp_log_ini, omega=omega,
                      static_temp=static_temp)

        # Load target networks after they have been initialized by constructor
        self.policy_target.load_state_dict(checkpoint['policy_target_state_dict'])
        self.q_target.load_state_dict(checkpoint['q_target_state_dict'])

        self.temp_log = checkpoint['temp_log_parameter']

        # Load Optimizers State Dict
        self.policy_optimizer = self.policy_optimizer.load_state_dict(checkpoint['policy_optim_state_dict'])
        self.q1_optimizer = self.q1_optimizer.load_state_dict(checkpoint['q1_optim_state_dict'])
        self.q2_optimizer = self.q2_optimizer.load_state_dict(checkpoint['q2_optim_state_dict'])
        if load_experience:
            self.memory.load_experiences(replay_experiences_path=complete_npy_file)








