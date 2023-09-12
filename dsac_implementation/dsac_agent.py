import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from dsac_implementation.dsac_algo import DSAC
from dsac_implementation.networks import Critic, Actor
from copy import deepcopy


class Agent:
    def __init__(self, obs_dim, action_dim, cr_lr_ini=8e-5, cr_lr_fin=1e-6,  act_lr_ini=5e-5,
                 act_lr_fin=1e-6, alpha_lr_ini=5e-5, alpha_lr_fin=1e-6,
                 cr_min_log_std=-0.1, cr_max_log_std=5,
                 cr_hl=(256, 256, 256, 256, 256), cr_activ=('gelu', 'gelu', 'gelu', 'gelu', 'gelu', 'gelu'),
                 act_min_log_std=-20, act_max_log_std=0.5,
                 act_hl=(256, 256, 256, 256, 256), act_activ=('gelu', 'gelu', 'gelu', 'gelu', 'gelu', 'gelu'),
                 action_low=-1, action_up=1,
                 t_max=50, tau=0.001, alpha=0.2, reward_scale=0.2, gamma=0.99, update_interval=2, auto_alpha=True,
                 ):
        self.q1: nn.Module = Critic(obs_dim=obs_dim, action_dim=action_dim, min_log_std=cr_min_log_std,
                                    max_log_std=cr_max_log_std, hidden_layers=cr_hl, activ=cr_activ)
        self.q2: nn.Module = Critic(obs_dim=obs_dim, action_dim=action_dim, min_log_std=cr_min_log_std,
                                    max_log_std=cr_max_log_std, hidden_layers=cr_hl, activ=cr_activ)
        self.q1_target: nn.Module = deepcopy(self.q1)
        self.q2_target: nn.Module = deepcopy(self.q2)

        self.policy: nn.Module = Actor(state_dim=obs_dim, action_dim=action_dim, hidden_layers=act_hl,
                                       activation=act_activ, min_log_std=act_min_log_std, max_log_std=act_max_log_std,
                                       action_low_lim=action_low, action_up_lim=action_up)
        self.policy_target = deepcopy(self.policy)

        self.log_alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))

        self.dsac = DSAC(self.q1, self.q2, self.q1_target, self.q2_target, cr_lr_ini=cr_lr_ini, cr_lr_fin=cr_lr_fin,
                         policy=self.policy, policy_target=self.policy_target, log_alpha=self.log_alpha,
                         actor_lr_ini=act_lr_ini, actor_lr_fin=act_lr_fin, alpha_lr_ini=alpha_lr_ini,
                         alpha_lr_fin=alpha_lr_fin, t_max=t_max, tau=tau, alpha=alpha, reward_scale=reward_scale,
                         gamma=gamma, up_interval=update_interval, auto_alpha=auto_alpha, target_entropy=-action_dim)

    def remember(self):
        pass

    def learn(self):
        pass

    def choose_action(self):
        pass

    def save_models(self, epoch: int, path_file: str):
        cr1_optim, cr2_optim, pol_optim, alpha_optim = self.dsac.get_optimizers()
        torch.save({
            'epoch': epoch,
            'cr1_state_dict': self.q1.state_dict(),
            'cr1_optim_state_dict': cr1_optim.state_dict(),
            'cr2_state_dict': self.q2.state_dict(),
            'cr2_optim_state_dict': cr2_optim.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'log_alpha_state_dict': self.log_alpha.state_dict(),
        })

    def load_models(self):
        pass
