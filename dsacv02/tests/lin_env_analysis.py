import os
import torch
import matplotlib.pyplot as plt
from dsacv02.agentv02 import Agent
from dsacv02.tools import smoothing
from environments.linear_env import LinearEnv
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

''' Agent'''
ACTION_DIM = 1
OBSERVATION_DIM = 1
N_KERNELS = 1

''' Environment constants '''
N_CELLS = 10

'''
Loading Parameters
'''
env_name = 'LinearEnv'
event_name = 'NoTargetSigma_SmallAlpha_2'
curr_dir = os.getcwd()
loading_path = curr_dir + '/' + 'DSAC_Runs' + '/' + env_name + '/' + event_name + '/'
tar_name = 'best_performance.tar'
meta_name = 'agent_meta.txt'
replay_name = 'replay_buffer.pkl'


if __name__ == '__main__':
    render = True
    DEVICE = 'cuda:0'
    env = LinearEnv(size=N_CELLS)
    # Highly reduced agent instantiation, since all parameters might be replaced
    agent = Agent(obs_dim=OBSERVATION_DIM, action_dim=ACTION_DIM, device=DEVICE, n_kernels_cr=1, n_kernels_act=1)
    agent.load_checkpoint(path=loading_path, tar_name=tar_name, txt_name=meta_name, replay_npy_name=replay_name,
                          load_experience=True)

    obs = torch.arange(start=0.1, end=0.9, step=0.1).unsqueeze(dim=1).to(DEVICE)
    actions_left = torch.tensor([-1]*(N_CELLS-2)).unsqueeze(dim=1).to(DEVICE)
    actions_right = torch.tensor([1]*(N_CELLS-2)).unsqueeze(dim=1).to(DEVICE)

    # Check for Double-Q
    path_settings = loading_path + 'Settings.txt'
    configs = dict()
    with open(path_settings, 'r') as file:
        for line in file.readlines():
            if '=' in line:
                config_str_end = line.find('=') - 1
                last_t = line.rfind('\t')
                val = line[last_t:].strip()
                try:
                    val = eval(val)
                except:
                    print(f'Exception occured for {val}')
                configs[line[0:config_str_end]] = val
    double_q = configs['DOUBLE_Q']

    # Set Q2 = Q1 is not double_q
    if double_q:
        m1_left, std1_left, kw1_left = agent.q1_target(obs, actions_left)
        m1_right, std1_right, kw1_right = agent.q1_target(obs, actions_right)

        m2_left, std2_left, kw2_left = agent.q2_target(obs, actions_left)
        m2_right, std2_right, kw2_right = agent.q2_target(obs, actions_right)
    else:
        m1_left, std1_left, kw1_left = agent.q1_target(obs, actions_left)
        m1_right, std1_right, kw1_right = agent.q1_target(obs, actions_right)

        m2_left, std2_left, kw2_left = m1_left, std1_left, kw1_left
        m2_right, std2_right, kw2_right = m1_right, std1_right, kw1_right

    '''
    Plot Stds
    '''
    cell_nr = torch.arange(1, 9, 1)
    # Left Action
    std1_left = std1_left.detach().cpu().numpy().squeeze()
    std2_left = std2_left.detach().cpu().numpy().squeeze()
    std_avg_left = 0.5 * (std1_left + std2_left)
    plt.plot(cell_nr, std_avg_left, label='Left action')

    # Right Action
    std1_right = std1_right.detach().cpu().numpy().squeeze()
    std2_right = std2_right.detach().cpu().numpy().squeeze()
    std_avg_right = 0.5 * (std1_right + std2_right)
    plt.plot(cell_nr, std_avg_right, label='Right action')

    plt.title('Uncertainty of Agent for All States in EnvLin')
    plt.xlabel('Cell Number')
    plt.ylabel('Standard Deviation')
    plt.legend()
    plt.show(block=True)









