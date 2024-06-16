import numpy as np


class LinearEnv:
    def __init__(self, size, right_gets_reward=False, stochasticity=0):
        """
        - Lienar environment, modified to work also for continuous actions
        :param size: Number of cells or length of field
        :param right_gets_reward: Terminal state utmost right emits reward 1
        :param stochasticity: Randomness of rewards in the terminal states
        """
        self.size = size
        self.matrix = np.arange(0, self.size, 1)
        self.pos = np.random.choice(self.matrix[1:-1], size=1)
        self.pos = np.asarray(self.pos, dtype=np.float64)
        self.reward_range = [-1, 1]
        self.stochasticity = stochasticity
        self.reward_left = 1
        self.reward_right = -1
        if right_gets_reward:
            self.reward_left *= -1
            self.reward_right *= -1
        self.rewards_arr = np.asarray([self.reward_left, self.reward_right])
        self.p_terminal_left = np.asarray([1-stochasticity, stochasticity])
        self.p_terminal_right = np.asarray([stochasticity, 1-stochasticity])

    def step(self, action):
        self.pos += action
        if self.pos <= 0:
            if self.stochasticity:
                reward = np.random.choice(self.rewards_arr, p=self.p_terminal_left).item()
            else:
                reward = self.reward_left
            done = True
        elif self.pos >= self.size-1:
            if self.stochasticity:
                reward = np.random.choice(self.rewards_arr, p=self.p_terminal_right).item()
            else:
                reward = self.reward_right
            done = True
        else:
            reward = 0
            done = False
        state_next = self.pos
        state_next = state_next / self.size

        return state_next, reward, done, None, None

    def reset(self):
        self.pos = np.random.choice(self.matrix[1:-1], size=1)
        self.pos = np.asarray(self.pos, dtype=np.float64)
        state = self.pos
        state = state / self.size

        return state


if __name__ == '__main__':
    env = LinearEnv(size=10, right_gets_reward=False, stochasticity=0)


