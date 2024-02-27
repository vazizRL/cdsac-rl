import numpy as np
import sys


class ReplayBuffer:
    def __init__(self, max_size, obs_shape, n_actions):
        # n_actions: Number of components of continuous actions
        # input_shape: Observation space shape
        self.mem_size = int(max_size)
        # Keeps track of first available memory
        self.mem_cntr = 0

        # (s, a, r, s') in separate attributes
        self.state_memory = np.zeros(shape=(self.mem_size, *obs_shape), dtype=np.float64)
        self.new_state_memory = np.zeros(shape=(self.mem_size, *obs_shape), dtype=np.float64)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float64)
        # self.log_p = np.zeros(self.mem_size, dtype=np.float64)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def __get_RAM__(self):
        return int(sys.getsizeof(self))

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, state_: np.ndarray,
                        done: bool):
        """
        - Stores a single (s,a,r,s',d) tuple in the replay buffer
        :param state: State to be saved
        :param action: Associated action to be saved
        :param reward: Associated reward to be saved
        :param state_: Associated new stae to be saved
        :param done: Associated done to be saved
        """
        # Find at which place new data should be stored, old ones are overwritten
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        # self.log_p[index] = log_p
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
        :param batch_size: Size to be sampled
        :return: One batch of (s,a,r,s', done)
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
        # log_ps = self.log_p[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

    def save_experiences(self, path_name: str):
        """
        - Saves the current state of the replay buffer as a np.ndarray
        :param path_name: Path + name of the .np file; Note: Must end with '.np'
        """
        experiences = (self.state_memory, self.action_memory, self.reward_memory, self.new_state_memory,
                       self.terminal_memory)
        experiences = np.asarray(experiences, dtype=np.float64)
        np.save(path_name, experiences)

        return 0

    def load_experiences(self, replay_experiences_path):
        """
        - Load all (s,a,r,s',d)
        :param replay_experiences_path: All tuples accumulated in one .npy file. Provide name
        """
        replay_experiences = np.load(replay_experiences_path)
        states, actions, rewards, states_, dones = replay_experiences
        self.mem_cntr = states.shape[0]
        self.state_memory = states
        self.action_memory = actions
        self.reward_memory = rewards
        self.new_state_memory = states_
        self.terminal_memory = dones

    def clear_buffer(self):
        self.state_memory.fill(0.0)
        self.new_state_memory.fill(0.0)
        self.action_memory.fill(0.0)
        self.reward_memory.fill(0.0)
        # self.log_p.fill(0.0)
        self.terminal_memory.fill(0.0)


if __name__ == '__main__':
    replay = ReplayBuffer(100000, obs_shape=(2, 2), n_actions=3)
    replay.__get_RAM__()





