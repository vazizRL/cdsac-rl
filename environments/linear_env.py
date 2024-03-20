import numpy as np


class LinearEnv:
    def __init__(self, size):
        self.size = size
        self.matrix = np.arange(0, self.size, 1)
        self.pos = np.random.choice(self.matrix[1:-1], size=1)
        self.reward_range = [-1, 1]

    def step(self, action):
        self.pos += action
        if self.pos == 0:
            reward = 1
            done = True
        elif self.pos == self.size-1:
            reward = -1
            done = True
        else:
            reward = 0
            done = False
        state_next = self.pos
        state_next = state_next / self.size

        return state_next, reward, done, None, None

    def reset(self):
        self.pos = np.random.choice(self.matrix[1:-1], size=1)
        state = self.pos
        state = state / self.size

        return state


if __name__ == '__main__':
    env = LinearEnv(size=5)


