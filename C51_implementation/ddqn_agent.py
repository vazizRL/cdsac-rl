import random

import torch
import numpy as np
from C51_implementation.replay_buffer_C51 import ReplayBuffer
from C51_implementation.networks import ZNetwork


class Agent:
    def __init__(self, input_dim, action_dim, n_atoms, lr, start_eps, end_eps, duration, gamma, batch_size, fc1_dim=256,
                 fc2_dim=256, v_min=-100, v_max=100, eps_dec=5e-4, max_mem=1e5):
        self.input_dim = input_dim
        self.actions_dim = action_dim
        self.action_space = [i for i in range(action_dim)]
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.lr = lr
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.duration = duration
        self.gamma = gamma
        self.batch_size = batch_size
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.eps_dec = eps_dec
        self.max_mem = max_mem
        self.memory = ReplayBuffer(max_size=self.max_mem, input_shape=input_dim, n_actions=action_dim)
        self.z_network = ZNetwork(action_dim=action_dim, state_dim=input_dim, lr=self.lr, hl1=self.fc1_dim,
                                  hl2=self.fc2_dim, n_atoms=self.n_atoms, v_min=v_min, v_max=v_max)
        self.delta_z = self.z_network.atoms[1] - self.z_network.atoms[0]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.empty_tb_info = {'Q-distribution': 0, 'loss': 0}

        # Administrative
        self.agent_parameters =

    def send_to_device(self, tensors: tuple):
        rel_tens = list()
        for ten in tensors:
            ten = torch.tensor(ten).to(self.device)
            rel_tens.append(ten)
        return rel_tens

    def linear_eps_schedule(self, t: int):
        """
        - Implements a linar epsilon decaying rate
        :param t: Current iteration
        :return: Current lr
        """
        slope = (self.end_eps - self.start_eps) / self.duration
        return max(slope * t + self.start_eps, self.end_eps)

    def remember(self, state, action, reward, state_new, done):
        """
        - Method to store the transitions
        :param state: Observed state
        :param action: Action in state
        :param reward: Reward received in state
        :param state_new: Transition state
        :param done: Terminal flag
        """
        self.memory.store_transition(state=state, action=action, reward=reward, state_=state_new, done=done)

        return 0

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            print(f'Not enough memories stored: Memory: {self.memory.mem_cntr}, batch size: {self.batch_size}')
            return self.empty_tb_info

        self.z_network.optimizer.zero_grad()

        states, actions, rewards, states_new, dones, mem_idxs = \
            self.send_to_device(*self.memory.sample_buffer(self.batch_size))

        with torch.no_grad():
            _, next_pmfs = self.z_network(states_new)
            next_atoms = rewards + self.gamma * self.z_network.atoms * (1 - dones)

            ''' Projection step '''
            tz = next_atoms.clamp(self.v_min, self.v_max)

            # Measure how many "bins" the target distr. is away from v_min
            b = (tz - self.v_min) / self.delta_z
            # Bin can only be 0 or max. number of atoms away from v_min
            low = b.floor().clamp(0, self.n_atoms - 1)
            up = b.ceil().clamp(0, self.n_atoms - 1)

            # (low == up) handles the case where bj is exactly an integer
            d_m_l = (up + (low == up).float() - b) * next_pmfs
            d_m_u = (b - low) * next_pmfs
            target_pmfs = torch.zeros_like(next_pmfs)
            for i in range(target_pmfs.size(dim=0)):
                target_pmfs[i].index_add_(0, low[i].long(), d_m_l[i])
                target_pmfs[i].index_add_(0, up[i].long(), d_m_u[i])

        # Calculate loss and optimize for one step
        _, old_pmfs = self.z_network(states, actions.flatten())
        loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()
        loss.backward()
        self.z_network.optimizer.step()

        return old_pmfs.detach(), loss.detach()

    def choose_action(self, state, curr_iter: int):
        state, = self.send_to_device((state,))
        epsilon = self.linear_eps_schedule(curr_iter)
        if random.random() < epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions, pmf = self.z_network(state)
            action = actions.cpu().numpy()

        return action

    def save_checkpoint(self):
        """
        - Save checkpoint, including hyperparameters
        """
        pass

    def load_checkpoint(self):
        pass




