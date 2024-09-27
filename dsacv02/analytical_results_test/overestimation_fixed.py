import torch
import os
import time
import matplotlib.pyplot as plt
from dsacv02.tools import cramer_optim_1k, get_normal_supports
from C_nach_Q_reg import get_b_integral, partial_c_partial_q
from torch.distributions.normal import Normal
from math import sqrt, floor
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def compute_delta_psi(mu_curr, std_curr, supports, lr, cramer_supp, tar_noisy, tar_true, nabla=1, dev='cpu'):
    """
    - Computes the Cramer overestimation based on the difference between psi true loss and noisy loss
    - Loss calculated: curr-tar_true, curr-tar_noisy
    :param mu_curr: Mean of current distribution
    :param std_curr: Std. dev of current distribution
    :param supports: Supports of double integral
    :param lr: Learning rate
    :param cramer_supp: Supports for cramer of optimized cramer calculation
    :param tar_noisy: Noisy target distribution
    :param tar_true: True target distribution
    :param nabla: Gradients of Q w.r.t its parameters
    :param dev: device
    """
    pdf_curr = Normal(loc=mu_curr, scale=std_curr)

    b_tilde = get_b_integral(pdf_curr=pdf_curr, pdf_tar=tar_noisy, supports_integral=supports)
    b_star = get_b_integral(pdf_curr=pdf_curr, pdf_tar=tar_true, supports_integral=supports)

    # psi_tilde = (2 * b_tilde) / ()
    # psi_star = (2 * b_star) / ()

    psi_delta = (2/std_curr) * (b_tilde - b_star)

    return lr * psi_delta * nabla


if __name__ == '__main__':
    device = 'cpu'
    learning_rate = 1.0
    ''' Numerical Configuration'''
    # Double Integral Configuration
    outer_var_std_l = -310
    outer_var_std_u = 310
    stepsize_var_std = 0.01     # 0.05
    supp_outer_var_std = torch.arange(outer_var_std_l, outer_var_std_u, stepsize_var_std)
    # Cramer Settings
    n_supp = 31
    ibf = 15
    supp_cr = get_normal_supports(batch_size=1, n_kernels=1, n_supp=n_supp, integral_bound_factor=ibf,
                                  dev=device)

    ''' Current Range '''
    # Mu Fixed
    mu_curr_fixed = torch.tensor(-100.0, device=device)
    # Standard Deviation Range
    std_l = 0.1
    std_u = 5.0
    std_stepsize = 0.25          # Old: 0.5
    std_range = torch.arange(std_l, std_u, std_stepsize, device=device)
    # Interval
    # std_range = focus_on_interval(arr=std_range, focus_interval=(3, 15), old_step_size=std_stepsize,
    #                               refined_step_size=0.5, device=device)

    ''' Initialize Targets Varying H and True Target'''
    # Values
    mu_tar_true, std_tar_true = -10.0, 20.0
    mu_tar_high1, std_tar_high1 = 50.0, 20.0
    mu_tar_high2, std_tar_high2 = 150.0, 20.0
    mu_tar_high3, std_tar_high3 = 200.0, 20.0
    # Initialize Fixed True Targets
    mean_tar_true = torch.as_tensor(mu_tar_true, device=device) # .unsqueeze(dim=0)
    std_tar_true = torch.as_tensor(std_tar_true, device=device) # .unsqueeze(dim=0)
    distr_tar_true = Normal(loc=mean_tar_true, scale=std_tar_true)
    # Initialize Fixed Target Tensors
    mu_tar_high1 = torch.as_tensor(mu_tar_high1, device=device) # .unsqueeze(dim=0)
    mu_tar_high2 = torch.as_tensor(mu_tar_high2, device=device) # .unsqueeze(dim=0)
    mu_tar_high3 = torch.as_tensor(mu_tar_high3, device=device) # .unsqueeze(dim=0)
    std_tar_high1 = torch.as_tensor(std_tar_high1, device=device) # .unsqueeze(dim=0)
    std_tar_high2 = torch.as_tensor(std_tar_high2, device=device) # .unsqueeze(dim=0)
    std_tar_high3 = torch.as_tensor(std_tar_high3, device=device) # .unsqueeze(dim=0)

    distr_tar_high1 = Normal(loc=mu_tar_high1, scale=std_tar_high1)
    distr_tar_high2 = Normal(loc=mu_tar_high2, scale=std_tar_high2)
    distr_tar_high3 = Normal(loc=mu_tar_high3, scale=std_tar_high3)
    # List of Target Distributions
    mu_tar_noisy_lists = [mu_tar_high1, mu_tar_high2, mu_tar_high3]
    std_tar_noisy_lists = [std_tar_high1, std_tar_high2, std_tar_high3]
    target_distributions_noisy = [distr_tar_high1, distr_tar_high2, distr_tar_high3]

    ''' Perform Evaluations '''
    deltas_all_tars = list()
    for mu_tar_noisy_i, std_tar_i, tar_distr in zip(mu_tar_noisy_lists, std_tar_noisy_lists, target_distributions_noisy):
        deltas = list()
        for std_i in std_range:
            delta_i = compute_delta_psi(mu_curr=mu_curr_fixed, std_curr=std_i, supports=supp_outer_var_std, lr=learning_rate,
                                        cramer_supp=supp_cr, tar_noisy=tar_distr, tar_true=distr_tar_true,
                                        nabla=1, dev=device)
            deltas.append(delta_i)
        deltas_all_tars.append(deltas)

    '''Save Parameters'''
    ts = time.time()
    curr_path = os.getcwd()
    saving_path = curr_path + '/' + 'Overestimation_' + str(ts)
    # os.mkdir(saving_path)

    ''' Plot and Save Graphs '''
    # Plot Varying Mu with Fixed Target
    plt.rcParams['figure.figsize'] = (30, 12)
    # for idx, deltas in enumerate(deltas_all_tars):
    plt.plot(std_range, deltas_all_tars[0], label=f'Delta(s,a) Curve - TarNoisy: Mu={mu_tar_high1.item()}, '
                                                  f'Std={std_tar_high1.item()} - TarTrue: Mu={mu_tar_true}, '
                                                  f'Std={std_tar_true}')
    # plt.plot(std_range, deltas_all_tars[1], label=f'Delta(s,a) Curve - TarNoisy: Mu={mu_tar_high2.item()}, '
    #                                              f'Std={std_tar_high2.item()}')
    # plt.plot(std_range, deltas_all_tars[2], label=f'Delta(s,a) Curve - TarNoisy: Mu={mu_tar_high3.item()}, '
    #                                              f'Std={std_tar_high3.item()}')
    plt.title(f'Delta(s,a) - Curr: Mu={mu_curr_fixed}, StdRange={std_l}-{std_u}')
    plt.xlabel('Curr. Std')
    plt.ylabel('Delta(s,a)')
    # plt.ylim((-5, 0))
    plt.legend()
    # plt.savefig(saving_path + '/' + 'VaryingCurrStd_Delta.png')
    plt.show(block=True)



