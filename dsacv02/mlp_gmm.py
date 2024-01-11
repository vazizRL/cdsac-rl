import torch
import torch.nn as nn
import torch.nn.functional as F
from dsacv02.tools import calc_size_co_matrix


class MLPGMM(nn.Module):
    def __init__(self, arch: tuple, activ: tuple, n_kernels: int, multivar=False):
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
        self.build_layers(multivar)

    @property
    def activ(self):
        return self._activ

    @activ.setter
    def activ(self, new_activ: torch.func):
        self._activ = new_activ

    @property
    def loss(self):
        return self._loss

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

    def build_layers(self, multivar):
        layer_id = 1

        next_element_list = list(self._arch)[1:-1] + [None]
        for arch_i, next_arch in zip(self._arch, next_element_list):
            if next_arch:
                layer = nn.Linear(arch_i, next_arch, dtype=torch.float64).to(self.device)
                self._layers.append(layer)
                self.module_dict.update({'layer_id_' + str(layer_id): layer})
                layer_id += 1

        means_layer = list()
        stds_layer = list()
        for mean_idx in range(self._n_kernels):
            mean_layer = nn.Linear(self._arch[-2], self._arch[-1], dtype=torch.float64).to(self.device)
            self.module_dict.update({f'mean_log{mean_idx + 1}': mean_layer})
            means_layer.append(mean_layer)

        for std_idx in range(self._n_kernels):

            std_layer = nn.Linear(self._arch[-2], self._arch[-1], dtype=torch.float64).to(self.device)
            self.module_dict.update({f'std_log{std_idx + 1}': std_layer})
            stds_layer.append(std_layer)

        self._layers.append((means_layer, stds_layer))

        return 0

    def forward(self, x, exp=False):
        """
        - Feed forward method
        :param x: Input
        :param exp: Whether return is exponentiated
        :return: Return shape (Batch, n_Kernel, n_Actions)
        """
        ffd = torch.as_tensor(x, dtype=torch.float64).to(self.device)
        for act_idx, layer_i in enumerate(self._layers[:-1]):
            func = self.get_activ_func_from_str(self.activ_str[act_idx])
            ffd = func(layer_i(ffd))

        means_logits = torch.tensor([], dtype=torch.float64).to(self.device)
        stds_logits = torch.tensor([], dtype=torch.float64).to(self.device)

        for mean_log_output_i, std_log_output_i in zip(self._layers[-1][0], self._layers[-1][1]):
            mean_logit_i = mean_log_output_i(ffd)
            std_logit_i = std_log_output_i(ffd)
            means_logits = torch.cat((means_logits, mean_logit_i), dim=1)
            stds_logits = torch.cat((stds_logits, std_logit_i), dim=1)

        # Rows: Kernels, Columns: Actions
        means_logits = means_logits.view(-1, self._n_kernels, self._arch[-1])
        stds_logits = means_logits.view(-1, self._n_kernels, self._arch[-1])

        if exp:
            means_logits = means_logits.exp()
            stds_logits = stds_logits.exp()

        return means_logits, stds_logits, None


class MLPGMMWeighted(MLPGMM):
    def __init__(self, arch: tuple, activ: tuple, n_kernels: int, multivar=False):
        super(MLPGMMWeighted, self).__init__(arch, activ, n_kernels, multivar=multivar)

    def build_layers(self, multivar):
        layer_id = 1

        next_element_list = list(self._arch)[1:-1] + [None]
        for arch_i, next_arch in zip(self._arch, next_element_list):
            if next_arch:
                layer = nn.Linear(arch_i, next_arch, dtype=torch.float64).to(self.device)
                self._layers.append(layer)
                self.module_dict.update({'layer_id_' + str(layer_id): layer})
                layer_id += 1

        means_layers = list()
        stds_layers = list()
        for mean_idx in range(self._n_kernels):
            mean_layer_i = nn.Linear(self._arch[-2], self._arch[-1], dtype=torch.float64).to(self.device)
            self.module_dict.update({f'mean_{mean_idx + 1}': mean_layer_i})
            means_layers.append(mean_layer_i)

        for std_idx in range(self._n_kernels):
            std_logit_i = nn.Linear(self._arch[-2], self._arch[-1], dtype=torch.float64).to(self.device)
            self.module_dict.update({f'std_{std_idx + 1}': std_logit_i})
            stds_layers.append(std_logit_i)

        kernel_weights_layer = nn.Linear(self._arch[-2], self._n_kernels, dtype=torch.float64).to(self.device)
        self.module_dict.update({f'kernel_weights_layer': kernel_weights_layer})

        self._layers.append((means_layers, stds_layers, kernel_weights_layer))

        return 0

    def forward(self, x, exp=False):
        """
        - Feed forward method
        :param x: Input
        :param exp: Whether return is exponentiated
        :return: Return shape (Batch, n_Kernel, n_Actions)
        """
        ffd = torch.as_tensor(x, dtype=torch.float64).to(self.device)
        for act_idx, layer_i in enumerate(self._layers[:-1]):
            func = self.get_activ_func_from_str(self.activ_str[act_idx])
            ffd = func(layer_i(ffd))

        means_logits = torch.tensor([], dtype=torch.float64).to(self.device)
        stds_logits = torch.tensor([], dtype=torch.float64).to(self.device)
        for mean_output_i, std_output_i in zip(self._layers[-1][0], self._layers[-1][1]):
            mean_i_logit = mean_output_i(ffd)
            std_i_logit = std_output_i(ffd)
            means_logits = torch.cat((means_logits, mean_i_logit), dim=1)
            stds_logits = torch.cat((stds_logits, std_i_logit), dim=1)

        # Logits of k weights
        k_weights_logits = self._layers[-1][2](ffd)

        # Rows: Kernels, Columns: Actions
        means_logits = means_logits.view(-1, self._n_kernels, self._arch[-1])
        stds_logits = stds_logits.view(-1, self._n_kernels, self._arch[-1])

        # Exponentiate all quantities
        if exp:
            means_logits = means_logits.exp()
            stds_logits = stds_logits.exp()
            k_weights_logits = k_weights_logits.exp()

        k_weights_soft = F.softmax(k_weights_logits, dim=1)
        return means_logits, stds_logits, k_weights_soft


if __name__ == '__main__':
    from torchsummary import summary

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    critic_arch = (11, 64, 32, 1)          # 11 is the batch size
    activations = ('relu', 'relu')
    n_kernels = 5

    # Define some input
    inp = torch.ones(11, 11) * torch.arange(11)
    inp = inp.to(device)
    # Create random tensor with shape 11x11
    inp_rnd = torch.randn(11, 11)

    '''
    Test for one dimensional output
    '''
    critic = MLPGMM(arch=critic_arch, activ=activations, n_kernels=n_kernels)

    # Output of model
    means, stds, _ = critic(inp)

    # Print Shapes of outputs
    print(f'Shape of means: {means.shape} \n {means} \n')
    print(f'Shape of stds: {stds.shape} \n {stds} \n')

    print(f'Critic summary: {summary(critic, (11,))}\nCritic Activation: {critic.activ_str}\n')

    '''
    Test for multi-dimensional output-
    '''
    # Rows: Kernels, Actions: dim. 2
    arch_mulo = (11, 64, 32, 3)
    activ_mulo = ('relu', 'relu')
    actor = MLPGMM(arch=arch_mulo, activ=activ_mulo, n_kernels=n_kernels)
    means_mulo, stds_mulo, _ = actor(inp)

    # Print Shapes of outputs
    print(f'Shape of means_mulo: {means_mulo.shape} \n {means_mulo} \n')
    print(f'Shape of stds_mulo: {stds_mulo.shape} \n {stds_mulo} \n')

    print(f'Actor summary: {summary(actor, (11,))}\nActor Activation: {actor.activ_str}')


    '''
    Test MLPGMMWeighted
    '''
    arch_mulo_w = (11, 64, 32, 3)
    active_mulo_w = ('gelu', 'gelu')
    actor_w = MLPGMMWeighted(arch=arch_mulo_w, activ=active_mulo_w, n_kernels=n_kernels)
    means_mulo_w, stds_mulo_w, weights = actor_w(inp)
    # Print Shapes of outputs
    print(f'Shape of means_mulo (version with weight output): {means_mulo_w.shape} \n {means_mulo_w} \n')
    print(f'Shape of stds_mulo (version with weight output): {stds_mulo_w.shape} \n {stds_mulo_w} \n')
    print(f'Shape of kernel weights: {weights.shape} \n{weights}')
    print(f'Actor summary (version with weight output): {summary(actor_w, (arch_mulo_w[0],))}'
          f'\nActor Activation: {actor_w.activ_str}')


