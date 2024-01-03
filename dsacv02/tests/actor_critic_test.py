"""
Note: The first dimension is the input dimension
"""
import torch
import torch.distributions as distr
from dsacv02.actor_critic import Actor, Critic
from dsacv02.gmm_reparameterization.mixture_same_family import ReparameterizedMixtureSameFamilyMod as RMM


def generate_gmm_distr(means, stds, kweights):
    cat_distr = distr.Categorical(probs=kweights)
    comp_distr = distr.Normal(loc=means, scale=stds)
    zcal = RMM(mixture_distribution=cat_distr, component_distribution=comp_distr)

    return zcal


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Input Data
batch_size = 5
state_dim = 11
action_dim = 2

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
actions = torch.ones(size=(batch_size, action_dim)) * actions

states = states.to(device)
actions = actions.to(device)

"""
Test Actor Class
"""
actor = Actor(state_dim=state_dim, action_dim=action_dim, hidden_layers=hls, n_kernels=n_k, activation=activation,
              action_min_std=action_min_log_std, action_max_std=action_max_log_std, action_low_lim=action_low,
              action_up_lim=action_up, learnable_weights=lr_weights)
means_act, stds_act, kernel_weights_act = actor(states)

"""
Test Critic
"""
critic = Critic(state_dim=state_dim, action_dim=action_dim, hidden_layers=hls, n_kernels=n_k, activ=activation,
                value_min_std=val_min_log_std, value_max_std=val_max_log_std, learnable_weights=lr_weights)

means_cr, stds_cr, kernel_weights_cr = critic(observation=states, action=actions)
print(f'Shape of means: {means_cr.shape}')

"""
Construct a GMM from critic parameters
"""
stds_cr.abs_()
for idx_cr, (mean_cr_i, std_cr_i, kernel_weight_cr_i) in enumerate(zip(means_cr, stds_cr, kernel_weights_cr)):
    gmm_i = generate_gmm_distr(means=mean_cr_i.squeeze(), stds=std_cr_i.squeeze(), kweights=kernel_weight_cr_i.squeeze())
    print(f'Sample from Critic GMM {idx_cr}: {gmm_i.rsample()}')
print('----------------\n')
""" 
Construct a GMM from actor parameters
"""
# In-place operation, in addition, separate for actions
stds_act.abs_()
# Action 1, sample_i_a1[0].requires_grad is False, sample_i_a1[1].requires_grad is True
for idx_act, (mean_act_i, std_act_i, kernel_weight_act_i) in enumerate(zip(means_act, stds_act, kernel_weights_act)):
    sample_i_a1 = actor.sample_from_action_distr(locs=mean_act_i.squeeze()[:, 0], stds=std_act_i.squeeze()[:, 0],
                                                 k_weights=kernel_weight_act_i.squeeze(), reparameterization=True)
    print(f'Sample for action 1 from actor GMM {idx_act}: {sample_i_a1}')

    # For first action: mean_act_i.squeeze()[:,0]
print('----------------\n')

# Action 2
for idx_act, (mean_act_i, std_act_i, kernel_weight_act_i) in enumerate(zip(means_act, stds_act, kernel_weights_act)):
    sample_i_a2 = actor.sample_from_action_distr(locs=mean_act_i.squeeze()[:, 1], stds=std_act_i.squeeze()[:, 1],
                                                 k_weights=kernel_weight_act_i.squeeze(), reparameterization=True)
    print(f'Sample for action 2 from actor GMM {idx_act}: {sample_i_a2}')
