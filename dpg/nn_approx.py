import torch
import torch as T
import torch.nn.functional as F
import torch.distributions as distr
from torch.func import functional_call, jacrev


class Policy(T.nn.Module):
    def __init__(self, arch=(10, 10, 10), lr=.001, name='Actor', dev='cuda', act_min=-1, act_max=1):
        super(Policy, self).__init__()
        self.act_min = act_min
        self.act_max = act_max
        self.module_dict = T.nn.ModuleDict()
        self.params = None
        self.device = dev
        self.arch = arch
        self.layers = list()
        self.name = name
        self.build_network()
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def build_network(self):
        nn_depth = len(self.arch)
        for idx in range(nn_depth - 1):
            layer_i = T.nn.Linear(self.arch[idx], self.arch[idx+1], dtype=T.float32)
            self.layers.append(layer_i)
            self.module_dict.update({f'layer_{idx}': layer_i})

        self.params = dict(self.named_parameters())

        return 0

    def funnel_action(self, action):
        return ((self.act_max - self.act_min) / 2) * torch.tanh(action) + \
                 (self.act_max + self.act_min) / 2

    def sample_action(self, loc, std):
        """
        - Used for behaviour policy \beta
        :param loc: Action mean
        :param std: Action std
        :return: Bounded action
        """
        act_gauss = distr.Normal(loc=loc, scale=std)
        action = act_gauss.sample()
        # Todo: Clipping might make sense here!
        action_bounded = torch.clamp(action, self.act_min, self.act_max)

        return action_bounded

    def forward(self, state):
        action = self.layers[0](state)
        action = F.relu(action)
        for layer_i in self.layers[1:-1]:
            action = layer_i(action)
            action = F.relu(action)
        action = self.layers[-1](action)
        # Scale action to be in [self.act_min, self.act_max]
        action = ((self.act_max - self.act_min) / 2) * torch.tanh(action) + \
            (self.act_max + self.act_min) / 2

        return action

    def fcall(self, params, state):
        """
        - Same as the method self.forward, but layers treated as one function call
        - To be used for per-layer Jacobian calculation
        :param state: Input / State Featrues
        """
        return functional_call(self, params, (state,))

    def get_jacobian(self, state):
        """
        :param state: Input
        :return:  batch_size x m x n Jacobianjacrev
        """
        # Per-layer jacobian
        j = jacrev(self.fcall, argnums=0)(self.params, state)

        # Extract and flatten weights/biases on the fly
        tensor_wb = [v.flatten(start_dim=2) for v in j.values()]
        # Concatenate along last dimension
        tensor_wb = T.concatenate(tensor_wb, dim=-1)

        return tensor_wb

    def save_chk_p(self, chk_name=None):
        T.save(self.state_dict(), chk_name)

        return 0


# NN Sanity Test
if __name__ == '__main__':
    # Arch.
    inp_dim = 2
    arch = (inp_dim, 10, 10, 1)
    device = 'cuda:0'
    # Training
    learning_rate = 0.01
    epochs = 40
    batch_size = 128
    # NN
    nn = Policy(arch=arch, lr=learning_rate, name='Test', dev=device)
    # Test Setup
    label = T.tensor([i for i in range(arch[-1])], dtype=T.float32).to(device)
    # Train loop
    for e in range(epochs):
        rnd = T.randn((batch_size, inp_dim), dtype=T.float32, device=device)
        input = T.tensor(rnd, dtype=T.float32).to(device)
        nn.optimizer.zero_grad()
        pred = nn.forward(state=input)
        loss = (pred - label)**2
        loss = loss.mean()
        loss.backward()
        nn.optimizer.step()
        print(f'Finished epoch {e} \n')

    print('\n ##### Pred after training ##### ')
    input = T.tensor(rnd, dtype=T.float32).to(device)
    pred = nn.forward(state=input)
    print(pred)

    # Get the Jacobian, argnums specified the Jacobian of what argument
    params = dict(nn.named_parameters())
    j = jacrev(nn.fcall, argnums=0)(params, input)
    j_wb_layers = list()
    print('Per-layer shape of current NN:')
    for k in j.keys():
        print(j[k].shape)
        j_wb_layers.append(j[k])

    # Confirm correct calculation of jacobian
    inp_sin = input[0]
    # ID for single scalar
    name = 'module_dict.layer_0.weight'
    idx = (0, 0)
    eps = 1e-3
    params_eps = {k: v.clone() for k, v in params.items()}
    params_eps[name][idx] += eps
    # Manual gradient between params and params_eps
    y0 = nn.fcall(params, inp_sin)
    y1 = nn.fcall(params_eps, inp_sin)
    grad = (y1 - y0) / eps
    # Extract the correct entry in Jacobian
    j_single = j[name][0, :, idx[0], idx[1]]
    # Output
    print(f'Manual Grad: {grad}')
    print(f'Jacobian Entry: {j_single}')

    # Concatenation scheme: w_0.flatten(), b_0, ... w_n.flatten(), b_n
    tensor_wb = list()
    for i in range(0, len(j_wb_layers), 2):
        w_i = j_wb_layers[i]
        # Orig. w_i shape: [batch, n_action, inp_dim, out_dim], collapse the last two
        w_flat = w_i.flatten(start_dim=-2, end_dim=-1)
        b_i = j_wb_layers[i + 1]
        # Create lists
        tensor_wb.append(w_flat)
        tensor_wb.append(b_i)
    # Now put together the full Jacobian Tensor
    j_full_ten = T.concatenate(tensor_wb, dim=-1)

    # Test Jacobian method
    j_ten_meth = nn.get_jacobian(input)
