import numpy as np
import sys
import os

class ReplayBuffer:
    def __init__(self, max_size, obs_shape, n_actions):
        # n_actions: Number of components of continuous actions
        # input_shape: Observation space shape
        self.mem_size = int(max_size)
        # Keeps track of first available memory
        self.mem_cntr = 0
        self.new_exp = 0

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

    def get_mem_dict(self):
        arrays = {'_state': self.state_memory,
                  '_action': self.action_memory,
                  '_reward': self.reward_memory,
                  '_state_next': self.new_state_memory,
                  '_terminal': self.terminal_memory,
                  }
        return arrays

    def save_experiences(self, path_name: str, all=False):
        """
        - Appends newly added experiences to the byte stream
        - Expand rewards and dones by one axis for homogeneity
        :param all: If True, it deactivates the append mode and writes all experiences
        :param path_name: Path + name of the .np file; Note: Must end with '.pkl'
        """
        arrays = self.get_mem_dict()

        if all:
            for name, arr in arrays.items():
                arr_name_i = path_name + name
                with open(arr_name_i, "wb") as f:
                    np.save(f, arr[0:self.mem_cntr])
                    f.flush()
                    os.fsync(f.fileno())

        else:
            for name, arr in arrays.items():
                arr_name_i = path_name + name
                with open(arr_name_i, "wb") as f:
                    np.save(f, arr[self.new_exp:self.mem_cntr])
                    f.flush()
                    os.fsync(f.fileno())

            self.new_exp = self.mem_cntr

        return 0

    def load_experiences(self, replay_experiences_path):
        """
        Todo: Save method change, change load method
        - Load all (s,a,r,s',d)
        - Loads chunks of experience streams and concatenates them
        :param replay_experiences_path: All tuples accumulated in one .npy file. Provide name
        """
        arrays =  self.get_mem_dict()

        for name, arr in arrays.items():
            npy_path = replay_experiences_path + 'mem' + name
            loaded_i = np.load(npy_path)
            loaded_size = len(loaded_i)
            arrays[name][:loaded_size] = loaded_i

        self.mem_cntr = loaded_size

        # with np.load(replay_experiences_path) as data:
        #     loaded_size = len(data['state_mem'])
        #     self.state_memory[:loaded_size] = data['state_mem']
        #     self.action_memory[:loaded_size] = data['action_mem']
        #     self.reward_memory[:loaded_size] = data['reward_mem']
        #     self.new_state_memory[:loaded_size] = data['state_next_mem']
        #     self.terminal_memory[:loaded_size] = data['terminal_mem']
        #
        # self.mem_cntr = loaded_size

        return 0

    def clear_buffer(self):
        self.state_memory.fill(0.0)
        self.new_state_memory.fill(0.0)
        self.action_memory.fill(0.0)
        self.reward_memory.fill(0.0)
        # self.log_p.fill(0.0)
        self.terminal_memory.fill(0.0)


if __name__ == '__main__':
    import os
    curr_dir = os.getcwd()
    file_name = 'Replay_Buffer_Test.npy'
    replay = ReplayBuffer(max_size=100, obs_shape=(1,), n_actions=1)
    states = np.array([0, 1])
    actions = np.array([-0.5, 0.5])
    rewards = np.array([-1, 0])
    states_new = np.array([3, 3])
    dones = np.array([0, 0])

    # Must be stored individually
    replay.store_transition(states[0], actions[0], rewards[0], states_new[0], dones[0])
    replay.store_transition(states[1], actions[1], rewards[1], states_new[1], dones[1])

    # Save replay
    complete_path = curr_dir + '/' + file_name
    replay.save_experiences(complete_path)

    # Load replay
    replay_new = ReplayBuffer(max_size=10, obs_shape=(1,), n_actions=1)
    replay_new.load_experiences(complete_path)

    replay.__get_RAM__()





