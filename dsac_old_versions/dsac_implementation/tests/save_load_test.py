import torch
import torch as T
import os
from dsac_implementation.networks import Actor, Critic
from dsac_old_versions.dsac_implementation.dsac_agent import Agent

"""Define Global Constans"""
# Define env. constants
state_dim = 2
action_dim = 3

# Define actor constants
actor_hl = (2, 2)
actor_activ = ('gelu', 'gelu', 'gelu')
actor_min_log_std = -2
actor_max_log_std = 2
action_low_lim = -1
action_high_lim = 1

# Define critic constants
cr_min_log_std = 0.1
cr_max_log_std = 5

cr_hl = (2, 2)
cr_activ = ('gelu', 'gelu', 'gelu')

# Define agent constants
actor_lr_ini = 5e-5
actor_lr_fin = 1e-6
cr_lr_ini = 5e-5
cr_lr_fin = 1e-6
alpha_lr_ini = 5e-5
alpha_lr_fin = 1e-6
t_max = 50
tau = 0.001
alpha = 0.2
reward_scale = 0.2
gamma = 0.99
update_interval = 2
auto_alpha = True

# Define inputs
device = torch.device('cuda:0')
rnd_state = T.tensor((2, 2), dtype=torch.float32).to(device)
rnd_action = T.tensor((1.5, 1.5, 2), dtype=torch.float32).to(device)


def create_actor_test():
    actor = Actor(state_dim, action_dim, actor_hl, actor_activ, actor_min_log_std, actor_max_log_std,
                  action_low_lim, action_high_lim)
    output = actor(rnd_state)
    print(f'Before saving - Random state: {rnd_state}; Output: {output}')

    return actor, output


def save_actor_and_reload(actor, input):
    rnd_tensor = input
    curr_dir = os.getcwd() + '/'
    path = curr_dir + 'model_tensor.tar'

    torch.save({
        'input': rnd_tensor,
        'actor_state_dict': actor.state_dict()},
        f=path)

    # Initialize new actor
    actor_loaded = Actor(1, 2, (2, 2), ('gelu', 'gelu', 'gelu'), -2, 2, -1, 1)
    checkpoint = torch.load(path)
    actor_loaded.load_state_dict(checkpoint['actor_state_dict'])
    input_loaded = checkpoint['input']

    output_new = actor_loaded(input_loaded)
    print(f'After saving - Input: {input_loaded}, Output: {output_new}')


def create_critic_test():
    critic = Critic(obs_dim=state_dim, action_dim=action_dim, min_log_std=cr_min_log_std, max_log_std=cr_max_log_std,
                    hidden_layers=cr_hl, activ=cr_activ)
    output = critic(obs=rnd_state, action=rnd_action)
    print(f'Before saving - State: {rnd_state} ;Action: {rnd_action}; Output: {output}')

    return critic, output


def create_agent():
    agent = Agent(obs_dim=state_dim, action_dim=action_dim, cr_lr_ini=cr_lr_ini, cr_lr_fin=cr_lr_fin,
                  act_lr_ini=actor_lr_ini, act_lr_fin=actor_lr_fin, alpha_lr_ini=alpha_lr_ini,
                  alpha_lr_fin=alpha_lr_fin, cr_min_log_std=cr_min_log_std, cr_max_log_std=cr_max_log_std,
                  cr_hl=cr_hl, cr_activ=cr_activ, act_min_log_std=actor_min_log_std, act_max_log_std=actor_max_log_std,
                  act_hl=actor_hl, act_activ=actor_activ, action_low=action_low_lim, action_up=action_high_lim,
                  t_max=t_max, tau=tau, alpha=alpha, reward_scale=reward_scale, gamma=gamma,
                  update_interval=update_interval, auto_alpha=auto_alpha)

    return agent


def test_saving_loading_complete():
    pass


if __name__ == '__main__':
    critic, cr_output = create_critic_test()
    actor, ac_output = create_actor_test()
    agent = create_agent()
    print(f'Policy minimum std before loading: {agent.policy.min_log_std}')

    # path = os.getcwd() + '/'
    # # agent.save_checkpoint(epoch=1, path=path, tar_name='test_models.tar', txt_name='agent_params.txt')
    # new_agent = agent.load_checkpoint(path=path, tar_name='test_models.tar', txt_name='agent_params.txt')
    # print(f'Policy minimum std after loading: {agent.policy.min_log_std}')


