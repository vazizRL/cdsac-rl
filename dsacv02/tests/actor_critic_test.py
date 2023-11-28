"""
Note: The first dimension is the input dimension
"""
import torch
from dsacv02.actor_critic import Actor, Critic


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

