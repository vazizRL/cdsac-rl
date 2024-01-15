import torch
import torch.distributions as distr
from torch.distributions.constraints import positive_definite
from dsacv02.gmm_reparameterization.mixture_same_family import ReparameterizedMixtureSameFamilyMod as RMM


def calc_n_outputs(n_actions: int):
    """
    - Calculates necessary number of outputs as a function of action space dimension
    :param n_actions: Size of action space
    """
    n_actions = torch.as_tensor(n_actions)
    return n_actions + torch.floor(0.5 * (n_actions ** 2 - n_actions))


def gen_own_multi_gmm_distr(means, cov_matrices, kweights):
    """
    - Generate a GMM with own implementation
    - Batch agnostic until now
    :param means: One set of means
    :param cov_matrices: One set of stds
    :param kweights: One set of kernel weights
    :return: Own implementation of GMM with reparameterization allowed
    """
    cat_distr = distr.Categorical(probs=kweights)
    comp_distr = distr.MultivariateNormal(loc=means, covariance_matrix=cov_matrices)
    gmm_distr = RMM(mixture_distribution=cat_distr, component_distribution=comp_distr)

    return gmm_distr

"""
Test random multivariate GMM for two actions
n_actions = 2
n_kernels = 3
batch_size = 1
"""
# Inner elements: The two actions that are drawn simultaneously
means = torch.tensor([[-4., -3.], [-1., 0.], [2., 3.]])
cov1 = torch.tensor([[1., 0.5], [0.5, 1.]])
cov2 = torch.tensor([[1., 0.5], [0.5, 1.]])
cov3 = torch.tensor([[1., 0.5], [0.5, 1.]])
covariances = torch.cat((cov1.unsqueeze(0), cov2.unsqueeze(0), cov3.unsqueeze(0)), dim=0)
kweights = torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32)

multi_gmm_own = gen_own_multi_gmm_distr(means=means, cov_matrices=covariances, kweights=kweights)

"""
Test batch-wise random multivariate GMM for two actions
batch_size = 5
n_actions = 2
n_kernels = 3
"""
means_batch = torch.randn(size=(5, 3, 2))
diag_elements = torch.randn(size=(5, 3, 2))
# Shape: (5, 3, 2, 2)
covariances_batch = torch.diag_embed(diag_elements)
covariances_batch.abs_()
kweights_batch = torch.nn.functional.softmax(torch.rand(size=(5, 3)))
multi_gmm_own_batch = gen_own_multi_gmm_distr(means=means_batch, cov_matrices=covariances_batch,
                                              kweights=kweights_batch)

"""
Test batch-wise random multivariate GMM for two actions
batch_size = 5
n_actions = 1
n_kernels = 3
"""
means_batch_act1 = torch.randn(size=(5, 3, 1))
diag_elements_act1 = torch.randn(size=(5, 3, 1))
# Shape: (5, 3, 2, 2)
covariances_batch_act1 = torch.diag_embed(diag_elements_act1)
covariances_batch_act1.abs_()
kweights_batch_act1 = torch.nn.functional.softmax(torch.rand(size=(5, 3)))
multi_gmm_own_batch_act1 = gen_own_multi_gmm_distr(means=means_batch_act1, cov_matrices=covariances_batch_act1,
                                                   kweights=kweights_batch_act1)

# Your tensor representing one half of a symmetric matrix
half_tensor = torch.tensor([1, 2, 3, 4, 5, 6])

# Create a zeros matrix
symmetric_matrix = torch.zeros((3, 3), dtype=half_tensor.dtype)

# Fill the upper triangular part of the matrix with your tensor values
symmetric_matrix = symmetric_matrix.triu()  # Upper triangular part
symmetric_matrix = symmetric_matrix + symmetric_matrix.t()  # Make it symmetric

# If your tensor has a special structure, you can directly use it
# symmetric_matrix = torch.triu(half_tensor.view(3, 3))

print(symmetric_matrix)

