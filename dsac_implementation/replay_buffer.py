import numpy as np
import sys


class ReplayBuffer:
    def __init__(self, max_size, obs_shape, n_actions):
        # n_actions: Number of components of continuous actions
        # input_shape: Observation space shape
        self.mem_size = max_size
        # Keeps track of first available memory
        self.mem_cntr = 0

        # (s, a, r, s') in separate attributes
        self.state_memory = np.zeros(shape=(self.mem_size, *obs_shape), dtype=np.float64)
        self.new_state_memory = np.zeros(shape=(self.mem_size, *obs_shape), dtype=np.float64)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float64)
        self.log_p = np.zeros(self.mem_size, dtype=np.float64)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def __get_RAM__(self):
        return int(sys.getsizeof(self))

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, state_: np.ndarray,
                         log_p: np.ndarray, done: bool):
        # Find at which place new data should be stored, old ones are overwritten
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.log_p[index] = log_p
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def add_batch(self, samples: list):
        """
        - Wrapper method to store batches of samples
        :param samples:
        """
        for sample in samples:
            self.store_transition(*sample)

    def sample_buffer(self, batch_size: int):
        """
        - Old implementation of samlping, agnostic to ML module
        :param batch_size:
        :type batch_size:
        :return: One batch of (s,a,r,s', log_ps, done)
        :rtype: tuple(np.ndarrays)
        """
        # How many memories have been stored including the overwritten ones
        max_mem = min(self.mem_cntr, self.mem_size)

        # Chose indices of size batch_size up to max indices
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        log_ps = self.log_p[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, log_ps, dones


if __name__ == '__main__':
    replay = ReplayBuffer(100000, obs_shape=(2, 2), n_actions=3)
    replay.__get_RAM__()





