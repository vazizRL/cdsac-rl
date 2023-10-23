import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        """
        - Should be agnostic of RL algorithm in use
        :param max_size: Max. number of transitions to be stored
        :param input_shape: Observation space shape
        :param n_actions: Number of actions
        """
        # n_actions: Number of components of continuous actions
        # input_shape: Observation space shape
        self.mem_size = max_size
        # Keeps track of first available memory
        self.mem_cntr = 0

        # (s, a, r, s') in separate attributes
        self.state_memory = np.zeros(shape=(self.mem_size, *input_shape), dtype=np.float64)
        self.new_state_memory = np.zeros(shape=(self.mem_size, *input_shape), dtype=np.float64)
        self.action_memory = np.zeros((self.mem_size, 1), dtype=np.float64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float64)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        # Find at which place new data should be stored, old ones are overwritten
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        # How many memories have been stored including the overwritten ones
        max_mem = min(self.mem_cntr, self.mem_size)

        # Chose indices of size batch_size up to max indices
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones, batch









