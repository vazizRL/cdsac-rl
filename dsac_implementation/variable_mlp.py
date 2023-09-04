import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, arch: tuple, activ: tuple):
        super().__init__()
        self._arch = arch
        self._layers = list()
        self._activ_str = activ
        self._activ = dict()
        self.build_layers()
        self.network = nn.Sequential(*self._layers)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

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

    def get_activ(self, name):
        def hook(model, input, output):
            self._activ[name] = output.to('cpu').detach()
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
            func = self.get_activ_func_from_str(self._activ_str[act_idx])
            ffd = func(layer_i(ffd))
        return ffd