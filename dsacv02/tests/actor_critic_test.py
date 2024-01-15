"""
Note: The first dimension is the input dimension
"""
import torch
from dsacv02.actor_critic import Actor, Critic
from dsacv02.gmm_reparameterization.mixture_same_family import ReparameterizedMixtureSameFamilyMod as RMM
from dsacv02.tools import calc_size_co_matrix

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Input Data
batch_size = 5
state_dim = 11
n_actions = 3

# Shared network parameters
hls = (128, 128)
n_k = 3
activation = ('gelu', 'gelu')
lr_weights = True

# Critic-specific parameters
val_min_log_std = -20  # \approx 0
val_max_log_std = 6

# Actor-specific parameters
action_min_log_std = -20  # 0
action_max_log_std = 1  # 2.71
action_low = -1
action_up = 1

states = torch.ones(batch_size, state_dim) * torch.arange(batch_size).unsqueeze(dim=1)
actions = torch.arange(batch_size).unsqueeze(dim=1)
actions = torch.ones(size=(batch_size, n_actions)) * actions

states = states.to(device)
actions = actions.to(device)

"""
Test Actor Class
"""
actor = Actor(state_dim=state_dim, action_dim=n_actions, hidden_layers=hls, n_kernels=n_k, activation=activation,
              action_min_std=action_min_log_std, action_max_std=action_max_log_std, action_low_lim=action_low,
              action_up_lim=action_up, learnable_weights=lr_weights)
means_act, stds_act, kernel_weights_act = actor(states)

means_act.exp_()
stds_act.exp_()

"""
Build Batch-Independent Covariance Matrix from Test Tensor
"""
covar_mat = torch.zeros(n_actions, n_actions)
# Get indices for upper triangular half
cov_elements = calc_size_co_matrix(n_actions=n_actions)
test_tensor = torch.arange(cov_elements, dtype=torch.float32)
rnd_matrix = torch.randn((n_actions, n_actions), dtype=test_tensor.dtype) + 1
# Indices for the upper triangle
triu_row_idx, triu_col_idx = rnd_matrix.triu().nonzero().t()
# Construct covariance matrix
covar_mat[triu_row_idx, triu_col_idx] = test_tensor
covar_mat = covar_mat + covar_mat.t()
# Halven the diagonal
covar_mat[range(n_actions), range(n_actions)] /= 2

"""
Build Batch-Aware Multivariate GMM from Actor outputs
"""
covar_mat_batch = torch.zeros(batch_size, n_k, n_actions, n_actions, dtype=stds_act.dtype).to(device)
# Get indices for upper triangular half
rnd_matrix = torch.randn((n_actions, n_actions), dtype=stds_act.dtype).to(device) + 1
# Indices for the upper triangle
triu_row_idx, triu_col_idx = rnd_matrix.triu().nonzero().t()
# Construct covariance matrix
# covar_mat_batch[batch_size, n_k, triu_row_idx, triu_col_idx] = stds_act
covar_mat_batch[:, :, triu_row_idx, triu_col_idx] = stds_act

# TODO: Make sure to build a positive definite matrix with stds_act

# Transpose last two elements: (0, 1, 3, 2)
covar_mat_batch = covar_mat_batch.permute(0, 1, 3, 2) + covar_mat_batch
# Halven the diagonals

covar_mat_batch.diagonal(dim1=-2, dim2=-1)[:] = covar_mat_batch.diagonal(dim1=-2, dim2=-1)

# Construct
cat_distr = torch.distributions.Categorical(probs=kernel_weights_act)
comp_distr = torch.distributions.MultivariateNormal(loc=means_act, covariance_matrix=covar_mat_batch)
gmm_distr_multi = RMM(mixture_distribution=cat_distr, component_distribution=comp_distr)

# Univariate GMM, used in the same procedure

"""
Test Critic
"""
critic = Critic(state_dim=state_dim, action_dim=n_actions, hidden_layers=hls, n_kernels=n_k, activ=activation,
                value_min_std=val_min_log_std, value_max_std=val_max_log_std, learnable_weights=lr_weights)

means_cr, stds_cr, kernel_weights_cr = critic(observation=states, action=actions)
print(f'Shape of means: {means_cr.shape}')




