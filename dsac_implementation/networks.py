import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary


class MLP(nn.Module):
    def __init__(self, arch: tuple, act: tuple):
        super().__init__()
        self._arch = arch
        self._layers = list()
        self._act_str = act
        self._act = dict()
        self.build_layers()
        self.network = nn.Sequential(*self._layers)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    @property
    def act(self):
        return self._act
    """"""
    @act.setter
    def act(self, new_act: torch.func):
        self._act = new_act

    @property
    def loss(self):
        return self._loss
    """"""
    @loss.setter
    def loss(self, new_loss: str):
        self._loss = new_loss

    @staticmethod
    def get_act_func_from_str(act_name: str):
        if hasattr(F, act_name):
            return getattr(F, act_name)
        else:
            raise ValueError(f'Activation functionw "{act_name}" not known')

    def get_act(self, name):
        def hook(model, input, output):
            self._act[name] = output.to('cpu').detach()
        return hook

    def build_layers(self):
        next_element_list = list(self._arch)[1:] + [None]
        for arch_i, next_arch in zip(self._arch, next_element_list):
            if next_arch:
                layer = nn.Linear(arch_i, next_arch)
                self._layers.append(layer)
        return 0

    def forward(self, x):
        ffd = x
        for act_idx, layer_i in enumerate(self._layers):
            func = self.get_act_func_from_str(self._act_str[act_idx])
            ffd = func(layer_i(ffd))
        return ffd


class Critic(nn.Module):
    def __init__(self, min_log_std=-0.1, max_log_std=4, arch=(1, 256, 256, 256, 256, 256),
                 act=('gelu', 'gelu', 'gelu', 'gelu', 'gelu')):
        """
        - Modelling Q distribution as a Gaussian.
        - min/max_log_std: clip(\mathcal{T}^{\pi_{\phi'}_{mathcal{D}}}Z(s,a), Q_{\theta}(s,a) - b, Q_{\theta}(s,a) + b)
        :param min_log_std: Based on ori. config. Convert to tensor and send to device
        :param max_log_std: Base don ori. config. Convert to tensor and send to device
        :param arch: Based on DSAC paper. First dim. is obs. dim.; likely to change
        :param act: Based on DSAC paper.
        """
        super().__init__()
        self.q = MLP(arch=arch, act=act)
        self.min_log_std = torch.tensor(min_log_std).to(self.q.device)
        self.max_log_std = torch.tensor(max_log_std).to(self.q.device)
        self.denominator = max(abs(self.min_log_std), self.max_log_std)

    def forward(self, obs, act, min=False):
        logits = self.q(torch.cat([obs, act], dim=-1))
        value_mean, log_std = torch.chunk(logits, chunks=2, dim=-1)

        value_log_std = torch.clamp_min(self.max_log_std * torch.tanh(log_std / self.denominator), 0) + \
            torch.clamp_max(-self.min_log_std * torch.tanh(log_std / self.denominator), 0)

        return torch.cat((value_mean, value_log_std), dim=-1)


if __name__ == '__main__':
    import torchvision
    import torchvision.transforms as transforms

    # Note: The first dimension is the input dimension
    mlp_model = MLP((3, 2, 2, 1), ('relu', 'relu', 'relu'))

    # # # Register Activations (NOT ACTIVATION TYPES )when in inference
    # mlp_model._layers[-2].register_forward_hook(mlp_model.get_act('last_hh'))





