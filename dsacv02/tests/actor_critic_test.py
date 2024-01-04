"""
Note: The first dimension is the input dimension
"""
import torch
import torch.distributions as distr
import time
from dsacv02.actor_critic import Actor, Critic
from dsacv02.gmm_reparameterization.mixture_same_family import ReparameterizedMixtureSameFamilyMod as RMM


def generate_gmm_distr(mean, std, kweight):
    cat_distr = distr.Categorical(probs=kweight)
    comp_distr = distr.Normal(loc=mean, scale=std)
    gmm_distr = RMM(mixture_distribution=cat_distr, component_distribution=comp_distr)

    return gmm_distr


def check_sampling(mean, std, kweight, n_sampling):
    own_gmm1_act1 = generate_gmm_distr(mean=mean, std=std, kweight=kweight)
    pytorch_gmm1_act1 = distr.MixtureSameFamily(mixture_distribution=distr.Categorical(probs=kweight),
                                                component_distribution=distr.Normal(loc=mean, scale=std))
    own_samples = own_gmm1_act1.sample((n_sampling,))
    pytorch_samples = pytorch_gmm1_act1.sample((n_sampling,))

    print(
        f'Element 1, Action 1. Mean by analysis: {means_act[0][:, 0].mean()}; Mean by own gmm samples: '
        f'{own_samples.mean()}; Mean by pytorch gmm samples: {pytorch_samples.mean()}')


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
start_cr_loop = time.perf_counter()
gmms_cr = []
for idx_cr, (mean_cr_i, std_cr_i, kernel_weight_cr_i) in enumerate(zip(means_cr, stds_cr, kernel_weights_cr)):
    gmm_i = generate_gmm_distr(mean=mean_cr_i.squeeze(), std=std_cr_i.squeeze(),
                               kweight=kernel_weight_cr_i.squeeze())
    gmms_cr.append(gmm_i)
    # print(f'Sample from Critic GMM {idx_cr}: {gmm_i.rsample()}')
end_cr_loop = time.perf_counter()
print(f'\n Calculation time for critic loop: {end_cr_loop - start_cr_loop}')
# print('----------------\n')
""" 
Construct a GMM from actor parameters
"""
# In-place operation, in addition, separate for actions
stds_act.abs_()
start_act_loop = time.perf_counter()
gmms_act = []
actions1_gmm = []
# Action 1, sample_i_a1[0].requires_grad is False, sample_i_a1[1].requires_grad is True
for idx_act, (mean_act_i, std_act_i, kernel_weight_act_i) in enumerate(zip(means_act, stds_act, kernel_weights_act)):
    action1_i, action1_log_prob_i = actor.sample_from_action_distr(locs=mean_act_i.squeeze()[:, 0],
                                                                   stds=std_act_i.squeeze()[:, 0],
                                                                   k_weights=kernel_weight_act_i.squeeze(),
                                                                   reparameterization=True)
    gmm_act_i = generate_gmm_distr(mean=mean_act_i.squeeze()[:, 0], std=std_act_i.squeeze()[:, 0],
                                   kweight=kernel_weight_act_i.squeeze())
    actions1_gmm.append(gmm_act_i)

    # print(f'Sample for action 1 from actor GMM {idx_act}: {action1_i} with log_prob {action1_log_prob_i}')

gmms_act.append(actions1_gmm)
#vprint('----------------\n')

# Action 2
actions2_gmm = []
for idx_act, (mean_act_i, std_act_i, kernel_weight_act_i) in enumerate(zip(means_act, stds_act, kernel_weights_act)):
    action2_i, action2_log_prob_i = actor.sample_from_action_distr(locs=mean_act_i.squeeze()[:, 1],
                                                                   stds=std_act_i.squeeze()[:, 1],
                                                                   k_weights=kernel_weight_act_i.squeeze(),
                                                                   reparameterization=True)

    gmm_act_i = generate_gmm_distr(mean=mean_act_i.squeeze()[:, 1], std=std_act_i.squeeze()[:, 1],
                                   kweight=kernel_weight_act_i.squeeze())
    actions2_gmm.append(gmm_act_i)
    # print(f'Sample for action 2 from actor GMM {idx_act}: {action2_i} with log_prob {action2_log_prob_i}')
# print('----------------\n')
gmms_act.append(actions2_gmm)
end_act_loop = time.perf_counter()
print(f'Calculation time for actor loops over all actions: {end_act_loop - start_act_loop}')
'''
Check sampling with standard implementation and own implementation
'''
mean1_act1 = means_act[0][:, 0]
std1_act1 = stds_act[0][:, 0]
kweight1_act1 = kernel_weights_act[0]
check_sampling(mean=mean1_act1, std=std1_act1, kweight=kweight1_act1, n_sampling=10_000)


