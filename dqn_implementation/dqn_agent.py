import torch as T
import numpy as np
from replay_buffer import ReplayBuffer
from networks import Network


class Agent:
    def __init__(self, lr, input_dims, n_actions, gamma, epsilon, batch_size, fc1_dims=256, fc2_dims=256,
                 eps_end=0.01, eps_dec=5e-4, max_mem=1000):
        super(Agent).__init__()
        self.max_mem = max_mem
        self.q_eval = Network(lr=lr, input_dims=input_dims, fc1_dims=fc1_dims, fc2_dims=fc2_dims, n_actions=n_actions)
        self.memory = ReplayBuffer(max_size=max_mem, input_shape=input_dims, n_actions=n_actions)
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.learn_iter = 0
        self.empty_tb_info = {'DQN/q_val': 0,
                              'DQN/loss': 0}

    @staticmethod
    def send_to_device(tensors: tuple, device: str):
        tens = list()
        for ten in tensors:
            ten = T.tensor(ten).to(device)
            tens.append(ten)
        return tens

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        """
        - Choose action with the highest value
        :param state: Observation
        :return: Greedy or random action from the action space
        """
        if np.random.random() > self.epsilon:
            state = T.tensor([state]).to(self.q_eval.device)
            actions = self.q_eval(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            print(f'Stored memories: {self.memory.mem_cntr} < {self.batch_size}')
            return self.empty_tb_info

        self.q_eval.optimizer.zero_grad()
        states, actions, rewards, states_, dones, batch_idx = self.memory.sample_buffer(self.batch_size)
        states, actions, rewards, states_, dones = self.send_to_device((states, actions, rewards, states_, dones),
                                                                       device=self.q_eval.device)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        actions = T.max(actions, dim=1)

        # q_curr = self.q_eval(states)[batch_index, actions]
        q_curr = self.q_eval(states)[batch_index, actions[0].int()]
        q_next = self.q_eval(states_)
        q_next[dones] = 0.0

        # [0] index: Just get the value
        q_target = rewards + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.q_eval.loss(q_target, q_curr).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end else self.eps_end
        self.learn_iter += 1

        tb_info = {'DQN/q_val': q_curr.mean_target().detach(),
                   'DQN/loss': loss.mean_target().detach()}

        return tb_info

    def save_checkpoint(self, iter_n: int, path: str, tar_name: str):
        """
        - Saves: Networks, optimizers and agent meta-parameters
        :param epoch: Epoch in which saving was performed
        :param path: Directory in which checkpoint is saved
        :param tar_name: Checkpoint file name, saved as .tar
        :param txt_name: Agent meta-parameters file name, saved as .txt
        """
        print('Save checkpoint...')
        complete_tar_file = tar_name
        T.save({
            'iter_n': iter_n,
            'q_state_dict': self.q_eval.state_dict(),
            'optimizer_state_dict': self.q_eval.optimizer.state_dict(),
        },
            complete_tar_file
        )

    def load_models(self, path, tar_name, env):
        # Load files
        complete_checkpoint =  tar_name
        checkpoint = T.load(complete_checkpoint)

        # Extract parameters from checkpoint
        iter_n = checkpoint['iter_n']
        q_state_dict = checkpoint['q_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']

        self.__init__(1e-3, env.observation_space.shape, env.action_space.n, gamma=0.99, epsilon=0.0,
                      batch_size=64, fc1_dims=128, fc2_dims=128,
                      eps_end=0.0, eps_dec=5e-4, max_mem=1000)

        # Load network, tensor params and learning rate schedule
        self.q_eval.load_state_dict(q_state_dict)
        self.q_eval.optimizer.load_state_dict(optimizer_state_dict)
        self.learn_iter = iter_n

