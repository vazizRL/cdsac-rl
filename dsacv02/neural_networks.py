import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from dsacv02.tools import calc_size_co_matrix


class MLPGMM(nn.Module):
    def __init__(self, arch: tuple, activ: tuple, n_kernels: int, device: str, multivar=False, std_bias_ini=None):
        """
        - Network with n means and n stds; n: Number of kernels
        :param arch: Specifies architecture (number of layers and nodes)
        :param activ: Activations per layer
        :param n_kernels: With n=1, the GMM reduces to a Gauss
        :param device: Specifies device on which to run the MLP
        :param multivar: Whether MLP models univariate or multivariate GMM
        """
        super().__init__()
        self.module_dict = nn.ModuleDict()
        self._arch = arch
        self._layers = list()
        self.activ_str = activ
        self._activ = dict()
        self._n_kernels = n_kernels
        self.device = device
        self.check_arch()
        # Calculate size of necessary output nodes for covariance matrix
        self.covar_out_size = calc_size_co_matrix(self._arch[-1]).item().__int__()
        self.multivar = multivar
        self.build_layers(std_bias_ini=std_bias_ini)

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

    def build_layers(self, std_bias_ini=None):
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
            if self.multivar:
                std_layer = nn.Linear(self._arch[-2], self.covar_out_size, dtype=torch.float64).to(self.device)
            else:
                std_layer = nn.Linear(self._arch[-2], self._arch[-1], dtype=torch.float64).to(self.device)
            if std_bias_ini:
                # Set initial bias
                std_layer.bias = nn.Parameter(torch.tensor(std_bias_ini, dtype=torch.float64, device=self.device))
            self.module_dict.update({f'std_matrix_log{std_idx + 1}': std_layer})
            stds_layer.append(std_layer)

        self._layers.append((means_layer, stds_layer))

        return 0

    def forward(self, x, exp=False):
        """
        - Feed forward method for multivariate GMMs
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

        if exp:
            means_logits = means_logits.exp()
            stds_logits = stds_logits.exp()
        else:
            stds_logits.abs_()

        # kweights = torch.ones((x.shape[0], 1, 1)) / self._n_kernels
        kweights = None

        return means_logits, stds_logits, kweights


class MLPGMMWeighted(MLPGMM):
    def __init__(self, arch: tuple, activ: tuple, n_kernels: int, device: str, multivar=False, std_bias_ini=None):
        super(MLPGMMWeighted, self).__init__(arch, activ, n_kernels, device, multivar=multivar,
                                             std_bias_ini=std_bias_ini)

    def build_layers(self, std_bias_ini=None):
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
            if self.multivar:
                std_logit_i = nn.Linear(self._arch[-2], self.covar_out_size, dtype=torch.float64).to(self.device)
            else:
                std_logit_i = nn.Linear(self._arch[-2], self._arch[-1], dtype=torch.float64).to(self.device)
            if std_bias_ini:
                # Set initial bias
                std_logit_i.bias = nn.Parameter(torch.tensor(std_bias_ini, dtype=torch.float64, device=self.device))
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
        if self.multivar:
            stds_logits = stds_logits.view(-1, self._n_kernels, self.covar_out_size)
        else:
            stds_logits = stds_logits.view(-1, self._n_kernels, self._arch[-1])

        # Exponentiate all quantities or take absolute value of std
        if exp:
            means_logits = means_logits.exp()
            stds_logits = stds_logits.exp()
            k_weights_logits = k_weights_logits.exp()
        else:
            stds_logits.abs_()

        k_weights_soft = F.softmax(k_weights_logits, dim=1)
        return means_logits, stds_logits, k_weights_soft


class SingleOutputNN(nn.Module):
    def __init__(self, arch: tuple, activ: tuple, device: str):
        super().__init__()
        self.module_dict = nn.ModuleDict()
        self._arch = arch
        self._activ = activ
        self._layers = list()
        self.check_arch()
        self.device = device
        self.build_layers()
        self.to(self.device)

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
        if len(self._arch) - 2 != len(self._activ):
            raise AssertionError(f'Number of layers and specified activations do not match!')

    def build_layers(self):
        nn_depth = len(self._arch)
        for idx in range(nn_depth - 1):
            layer_i = torch.nn.Linear(self._arch[idx], self._arch[idx+1], dtype=torch.float64)
            self._layers.append(layer_i)
            self.module_dict.update({f'layer_{idx}': layer_i})

        return 0

    def forward(self, inp, exp=False):
        # Convert to torch.array and Send to GPU
        ffd = torch.as_tensor(inp, dtype=torch.float64).to(self.device)
        for act_idx, layer_i in enumerate(self._layers[:-1]):
            func = self.get_activ_func_from_str(self._activ[act_idx])
            ffd = func(layer_i(ffd))
        ffd = self._layers[-1](ffd)

        return ffd, -1, None


if __name__ == '__main__':
    from torchsummary import summary

    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    test_single = False
    test_multi = False
    test_weighted = False
    test_single_output = True

    n_kernels = 1

    # Define some input
    inp = torch.ones(11, 11) * torch.arange(11)
    inp = inp.to(dev)
    # Create random tensor with shape 11x11
    x = torch.randn(11, 11)

    '''
    Test for one dimensional output
    '''
    if test_single:
        critic_arch = (11, 64, 32, 1)  # 11 is the batch size
        activations = ('relu', 'relu')
        critic_z = MLPGMM(arch=critic_arch, activ=activations, n_kernels=n_kernels, device=dev)

        # Output of model
        means, stds, _ = critic_z(inp)

        # Print Shapes of outputs
        print(f'Shape of means: {means.shape} \n {means} \n')
        print(f'Shape of stds: {stds.shape} \n {stds} \n')

        print(f'Critic summary: {summary(critic_z, (11,))}\nCritic Activation: {critic_z.activ_str}\n')

    '''
    Test for multi-dimensional output-
    '''
    if test_multi:
        # Rows: Kernels, Actions: dim. 2
        arch_mulo = (11, 64, 32, 3)
        activ_mulo = ('relu', 'relu')
        actor = MLPGMM(arch=arch_mulo, activ=activ_mulo, n_kernels=n_kernels, device=dev)
        means_mulo, stds_mulo, _ = actor(inp)

        # Print Shapes of outputs
        print(f'Shape of means_mulo: {means_mulo.shape} \n {means_mulo} \n')
        print(f'Shape of stds_mulo: {stds_mulo.shape} \n {stds_mulo} \n')

        print(f'Actor summary: {summary(actor, (11,))}\nActor Activation: {actor.activ_str}')

    '''
    Test MLPGMMWeighted
    '''
    if test_multi:
        arch_mulo_w = (11, 64, 32, 3)
        active_mulo_w = ('gelu', 'gelu')
        actor_w = MLPGMMWeighted(arch=arch_mulo_w, activ=active_mulo_w, n_kernels=n_kernels, device=dev)
        means_mulo_w, stds_mulo_w, weights = actor_w(inp)
        # Print Shapes of outputs
        print(f'Shape of means_mulo (version with weight output): {means_mulo_w.shape} \n {means_mulo_w} \n')
        print(f'Shape of stds_mulo (version with weight output): {stds_mulo_w.shape} \n {stds_mulo_w} \n')
        print(f'Shape of kernel weights: {weights.shape} \n{weights}')
        print(f'Actor summary (version with weight output): {summary(actor_w, (arch_mulo_w[0],))}'
              f'\nActor Activation: {actor_w.activ_str}')

    '''
    Test SingleOutput NN
    '''
    if test_single_output:
        inp_dim = 20
        arch_q = (inp_dim, 64, 64, 1)
        activations_q = ('relu', 'relu')
        critic_q = SingleOutputNN(arch=arch_q, activ=activations_q, device=dev)
        # Train loop
        learning_rate = 0.001
        optim_q = Adam(critic_q.parameters(), lr=learning_rate)
        optim_q.zero_grad()
        n_data = 50
        batch_s = 5
        iterations = 1000
        curr_iter = 0
        labels = torch.unsqueeze(torch.arange(n_data, dtype=torch.float64), dim=1).to(dev)
        x = torch.randn(n_data, inp_dim, dtype=torch.float64)
        while curr_iter < iterations:
            rnd_sample_idx = torch.randint(0, n_data, (batch_s,))
            x_batch = x[rnd_sample_idx]
            y_batch = labels[rnd_sample_idx]
            pred, std, kw = critic_q(x_batch)
            loss = F.mse_loss(pred, y_batch)
            loss.backward()
            optim_q.step()
            optim_q.zero_grad()
            curr_iter += 1
        idx_samples = torch.as_tensor([5, 10, 44])
        pred = critic_q(x[idx_samples])
        print(f'Label: {idx_samples}, prediction. {pred.cpu()}')