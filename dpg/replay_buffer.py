import numpy as np
import pickle
from sys import getsizeof


class ReplayBuffer:
    def __init__(self, max_size, state_shape, action_shape):
        self.max_size = max_size
        self.mem_size = 0
        self.new_mem_from = 0
        self.state_mem = np.zeros(shape=(self.max_size, *state_shape), dtype=np.float32)
        self.state_next_mem = np.zeros(shape=(self.max_size, *state_shape), dtype=np.float32)
        self.action_mem = np.zeros(shape=(self.max_size, *action_shape), dtype=np.float32)
        self.reward_mem = np.zeros(shape=(self.max_size, 1), dtype=np.float32)
        self.done_mem = np.zeros(shape=(self.max_size, 1), dtype=np.float32)

    @property
    def ram_usage_bytes(self):
        return int(getsizeof(self)) + self.state_mem.nbytes + self.state_next_mem.nbytes + self.action_mem.nbytes + \
            self.reward_mem.nbytes + self.done_mem.nbytes

    @property
    def ram_usage_mb(self):
        return self.ram_usage_bytes / (1024 ** 2)

    def sample(self, batch_size):
        upper = min(self.mem_size, self.max_size)
        batch_idx = np.random.randint(low=0, high=upper + 1, size=batch_size, dtype=np.int32)

        return self.state_mem[batch_idx], self.action_mem[batch_idx], self.reward_mem[batch_idx], \
            self.state_next_mem[batch_idx], self.done_mem[batch_idx]

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, state_next: np.ndarray,
                         done: np.ndarray):
        """
        - Store one experience tuple (s,a,r,s',done)
        """
        save_idx = self.mem_size % self.max_size
        self.state_mem[save_idx] = state
        self.action_mem[save_idx] = action
        self.reward_mem[save_idx] = reward
        self.state_next_mem[save_idx] = state_next
        self.done_mem[save_idx] = done

        self.mem_size += 1

        return 0

    def store_batch(self, experience_batch):
        """
        - Store batches of transitions
        :param experience_batch:
        :type experience_batch:
        """
        for sample in experience_batch:
            self.store_transition(*sample)

        return 0

    def save(self, path_name: str, all=False):
        """
         - Appends newly added experiences to the byte stream
         - Expand rewards and dones by one axis for homogeneity
         :param all: If True, it deactivates the append mode and writes all experiences
         :param path_name: Path + name of the .np file; Note: Must end with '.pkl'
         """
        save_until = self.mem_size
        if self.mem_size >= self.max_size:
            save_until = self.max_size - 1

        if all:
            experiences = (self.state_mem[0:save_until],
                           self.action_mem[0:save_until],
                           self.reward_mem[0:save_until],
                           self.state_next_mem[0:save_until],
                           self.done_mem[0:save_until])

            with open(path_name, mode='wb') as file:
                pickle.dump(experiences, file)
        else:
            experiences = (self.state_mem[self.new_mem_from:self.mem_size],
                           self.action_mem[self.new_mem_from:self.mem_size],
                           self.reward_mem[self.new_mem_from:self.mem_size],
                           self.state_next_mem[self.new_mem_from:self.mem_size],
                           self.done_mem[self.new_mem_from:self.mem_size])

            with open(path_name, mode='ab') as file:
                pickle.dump(experiences, file)

            self.new_mem_from = self.mem_size

        return 0

    def load_transitions(self, path: str):
        """
        - Load all (s,a,r,s',d)
        - Loads chunks of experience streams and concatenates them
        :param replay_experiences_path: All tuples accumulated in one .npy file. Provide name
        """

        conc = list()
        with open(path, 'rb') as file:
            while True:
                try:
                    row = pickle.load(file)
                    conc.append(row)
                except EOFError:
                    break

        states_last = conc[0][0]
        actions_last = conc[0][1]
        rewards_last = conc[0][2]
        states_next_last = conc[0][3]
        dones_last = conc[0][4]
        for next_chunk in conc[1:]:
            states_i, actions_i, rewards_i, states_next_i, dones_i = next_chunk

            states_last = np.concatenate((states_last, states_i))
            actions_last = np.concatenate((actions_last, actions_i))
            rewards_last = np.concatenate((rewards_last, rewards_i))
            states_next_last = np.concatenate((states_next_last, states_next_i))
            dones_last = np.concatenate((dones_last, dones_i))

        replay_len = states_last.shape[0]
        # If more than mem_size experiences are stored, make sure that the recent ones are in the buffer
        if replay_len > self.mem_size:
            states_last = states_last[-self.mem_size:]
            actions_last = actions_last[-self.mem_size:]
            rewards_last = rewards_last[-self.mem_size:]
            states_next_last = states_next_last[-self.mem_size:]
            dones_last = dones_last[-self.mem_size:]
            # For indexing max. value
            replay_len = self.mem_size

        self.state_mem[:replay_len] = states_last
        self.action_mem[:replay_len] = actions_last
        self.reward_mem[:replay_len] = rewards_last
        self.state_next_mem[:replay_len] = states_next_last
        self.done_mem[:replay_len] = dones_last

        return 0


# Sanity Check
if __name__ == '__main__':
    replay_size = int(1e6)
    obs_dim = (3, 5)
    action_dim = (5,)

    replay = ReplayBuffer(max_size=replay_size, state_shape=obs_dim, action_shape=action_dim)
    print(replay.ram_usage_mb)
