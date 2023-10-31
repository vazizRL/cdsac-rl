import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, arch: tuple, activ: tuple, n_kernels: int):
        """
        - Network with n means and n stds; n: Number of kernels
        :param arch: Soecified architecture (number of layers and nodes)
        :param activ: Activations per layer
        :param n_kernels: Number of kernels of the GMM.
        """
        super().__init__()
        self.module_dict = nn.ModuleDict()
        self._arch = arch
        self._layers = list()
        self.activ_str = activ
        self._activ = dict()
        self._n_kernels = n_kernels
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.check_arch()
        self.build_layers()

    @property
    def activ(self):
        return self._activ
    """"""
    @activ.setter
    def activ(self, new_activ: torch.func):
        self._activ = new_activ

    @property
    def loss(self):
        return self._loss
    """"""
    @loss.setter
    def loss(self, new_loss: str):
        self._loss = new_loss

    @staticmethod
    def get_activ_func_from_str(act_name: str):
        if hasattr(F, act_name):
            return getattr(F, act_name)
        else:
            raise ValueError(f'Activation functionw "{act_name}" not known')

    def check_arch(self):
        """
        - Checks if number of activations matches with the architecture
        - Note that last nodes shouldn't be funneled!
        """
        if len(self._arch) - 2 != len(self.activ_str):
            raise AssertionError(f'Number of layers and specified activations do not match!')

    def get_activ(self, name):
        def hook(model, input, output):
            self._activ[name] = output.to('cpu').detach()
        return hook

    def build_layers(self):
        layer_id = 1

        next_element_list = list(self._arch)[1:-1] + [None]
        for arch_i, next_arch in zip(self._arch, next_element_list):
            if next_arch:
                layer = nn.Linear(arch_i, next_arch, dtype=torch.float64).to(self.device)
                self._layers.append(layer)
                self.module_dict.update({'layer_id_' + str(layer_id): layer})
                layer_id += 1

        means = list()
        stds = list()
        for n in range(self._n_kernels):
            mean = nn.Linear(self._arch[-2], self._arch[-1], dtype=torch.float64).to(self.device)
            std = nn.Linear(self._arch[-2], self._arch[-1], dtype=torch.float64).to(self.device)
            self.module_dict.update({f'mean_{n+1}': mean, f'std_{n+1}': std})
            means.append(mean)
            stds.append(std)
        self._layers.append((means, stds))

        return 0

    def forward(self, x):
        ffd = torch.as_tensor(x, dtype=torch.float64)
        for act_idx, layer_i in enumerate(self._layers[:-1]):
            func = self.get_activ_func_from_str(self.activ_str[act_idx])
            ffd = func(layer_i(ffd))

        means = torch.tensor([], dtype=torch.float64).to(self.device)
        stds = torch.tensor([], dtype=torch.float64).to(self.device)
        for mean_output_i, std_output_i in zip(self._layers[-1][0], self._layers[-1][1]):
            mean_i = mean_output_i(ffd)
            std_i = std_output_i(ffd)
            means = torch.cat((means, mean_i), dim=1)
            stds = torch.cat((stds, std_i), dim=1)

        return means, stds


if __name__ == '__main__':
    from torchsummary import summary

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    arch = (11, 64, 32, 1)
    activations = ('relu', 'relu')
    n_kernels = 3

    # Define some input
    inp = torch.ones(11, 11) * torch.arange(11)
    inp = inp.to(device)

    # Instantiate Model
    critic = MLP(arch=arch, activ=activations, n_kernels=n_kernels)

    # Output of model
    means, stds = critic(inp)

    # Print Shapes of outputs
    print(f'Shape of means: {means.shape} \n {means} \n')
    print(f'Shape of stds: {stds.shape} \n {stds} \n')

    print(f'Model summary: {summary(critic, (11,))}\nActivation: {critic.activ_str}')
