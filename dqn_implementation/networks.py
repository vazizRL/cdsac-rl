import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Network, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.output = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.output(x)

        return actions


if __name__ == '__main__':
    q_eval = Network(lr=1e-3, input_dims=(5,), fc1_dims=256, fc2_dims=256, n_actions=3)
    rnd_state = T.rand(2, 5).to('cuda:0')
    output = q_eval(rnd_state)
    print(output)

