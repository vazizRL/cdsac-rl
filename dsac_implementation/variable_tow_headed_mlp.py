import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, arch: tuple, activ: tuple):
        super().__init__()
        self.module_dict = nn.ModuleDict()
        self._arch = arch
        self._layers = list()
        self._activ_str = activ
        self._activ = dict()
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
        if len(self._arch) - 1 != len(self._activ_str):
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
        mean = nn.Linear(self._arch[-2], self._arch[-1], dtype=torch.float64).to(self.device)
        std = nn.Linear(self._arch[-2], self._arch[-1], dtype=torch.float64).to(self.device)
        self.module_dict.update({'head_1': mean, 'head_2': std})
        self._layers.append((mean, std))
        return 0

    def forward(self, x):
        ffd = torch.as_tensor(x, dtype=torch.float64)
        for act_idx, layer_i in enumerate(self._layers[:-1]):
            func = self.get_activ_func_from_str(self._activ_str[act_idx])
            ffd = func(layer_i(ffd))
        func = self.get_activ_func_from_str(self._activ_str[-1])
        means = func(self._layers[-1][0](ffd))
        stds = func(self._layers[-1][1](ffd))
        return means, stds





