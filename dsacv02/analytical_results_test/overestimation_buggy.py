import torch
import os
import time
import matplotlib.pyplot as plt
from dsacv02.tools import cramer_optim_1k, get_normal_supports
from torch.distributions.normal import Normal
from math import sqrt, floor
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def focus_on_interval(arr, focus_interval: tuple, old_step_size: float, refined_step_size: float, device):
    int_i, int_ii = focus_interval
    int_i = floor(int_i / old_step_size)
    int_ii = floor(int_ii / old_step_size)
    arr_i = arr[:int_i]
    arr_ii = arr[int_ii:]

    refined_interval = torch.arange(int_i, int_ii, refined_step_size, device=device)

    new_arr = torch.cat((arr_i, refined_interval, arr_ii), dim=0)

    return new_arr


def mu_expr(dt, mu_curr, sigma_curr, mu_tar, sigma_tar, dev='cpu'):
    """
    - Fixed version
    :param dt: Incremental Supports
    :param mu_curr: Mean of current Gauss
    :param sigma_curr: Standard dev. of current Gauss
    :param mu_tar: Mean of target Gauss
    :param sigma_tar: Standard dev. of target Gauss
    :param nabla: Gradient of parameterized \mu_{\theta}
    :return:
    """
    term_old = torch.e**(-0.5*((dt-mu_curr)/sigma_curr)**2) * (dt - mu_curr)
    term_fixed_eq = (1/(sigma_curr * sqrt(2*torch.pi))) * torch.e**(-0.5*((dt-mu_curr)/sigma_curr)**2) \
                      - ((1 / (sigma_tar * sqrt(2 * torch.pi))) * torch.e**(-0.5*((dt-mu_tar)/sigma_tar)**2))
    return term_old * term_fixed_eq


def get_inner_int(inner_supports, mu_curr, sigma_curr, mu_tar, sigma_tar):
    dy_inner_supports = mu_expr(dt=inner_supports, mu_curr=mu_curr, sigma_curr=sigma_curr, mu_tar=mu_tar,
                                sigma_tar=sigma_tar)

    return torch.trapz(y=dy_inner_supports, x=inner_supports)


def get_outer_int(outer_supports, mu_curr, std_curr, mu_tar, std_tar):
    dy_inner_list = list()
    for idx, _ in enumerate(outer_supports):
        dy_inner_i = get_inner_int(inner_supports=outer_supports[:idx], mu_curr=mu_curr, sigma_curr=std_curr,
                                   mu_tar=mu_tar, sigma_tar=std_tar)
        dy_inner_list.append(dy_inner_i)
    dy_inner_expr = torch.as_tensor(dy_inner_list)
    dy_outer = torch.trapz(y=dy_inner_expr, x=outer_supports)

    return dy_outer


def compute_delta_c(mu_curr, std_curr, mu_tar_true, std_tar_true, mu_tar_noisy, std_tar_noisy,
                    outer_supp, lr, cramer_supp, tar_noisy, tar_true, nabla=1, dev='cpu'):
    """
    - Computes the Cramer overestimation based on the difference between true loss and noisy loss
    - Loss calculated: curr-tar_true, curr-tar_noisy
    :param mu_curr: Mean of current distribution
    :param std_curr: Std. dev of current distribution
    :param mu_tar_true: Mean of true target
    :param std_tar_true: Std. dev. of true target
    :param mu_tar_noisy: Mean of noisy target
    :param std_tar_noisy: Std. dev. of noisy target
    :param outer_supp: Supports of double integral
    :param lr: Learning rate
    :param cramer_supp: Supports for cramer of optimized cramer calculation
    :param tar_noisy: Noisy target distribution
    :param tar_true: True target distribution
    :param nabla: Gradients of Q w.r.t its parameters
    :param dev: device
    """
    pdf_curr = Normal(loc=mu_curr, scale=std_curr)

    b_tilde = 1 / cramer_optim_1k(pdf_target=tar_noisy, pdf_curr=pdf_curr, standard_supp=cramer_supp,
                                  n_kernels=1, dev=dev)
    b_star = 1 / cramer_optim_1k(pdf_target=tar_true, pdf_curr=pdf_curr, standard_supp=cramer_supp,
                                 n_kernels=1, dev=dev)

    double_integral_tilde = get_outer_int(outer_supports=outer_supp, mu_curr=mu_curr, std_curr=std_curr,
                                          mu_tar=mu_tar_noisy, std_tar=std_tar_noisy)
    double_integral_star = get_outer_int(outer_supports=outer_supp, mu_curr=mu_curr, std_curr=std_curr,
                                         mu_tar=mu_tar_true, std_tar=std_tar_true)

    p_bar = b_tilde * double_integral_tilde
    q_bar = b_star * double_integral_star

    diff_pq = p_bar - q_bar

    frac = 1 / (std_curr**3 * sqrt(2*torch.pi))

    return lr * frac * diff_pq * nabla


if __name__ == '__main__':
    device = 'cpu'
    learning_rate = 1.0
    ''' Numerical Configuration'''
    # Double Integral Configuration
    outer_var_std_l = -230
    outer_var_std_u = 480
    stepsize_var_std = 0.05
    supp_outer_var_std = torch.arange(outer_var_std_l, outer_var_std_u, stepsize_var_std)
    # Cramer Settings
    n_supp = 31
    ibf = 15
    supp_cr = get_normal_supports(batch_size=1, n_kernels=1, n_supp=n_supp, integral_bound_factor=ibf,
                                  dev=device)

    ''' Current Range '''
    # Mu Fixed
    mu_curr_fixed = torch.tensor(0.0, device=device)
    # Standard Deviation Range
    std_l = 0.01
    std_u = 0.1
    std_stepsize = 0.005          # Old: 0.5
    std_range = torch.arange(std_l, std_u, std_stepsize, device=device)
    # Interval
    std_range = focus_on_interval(arr=std_range, focus_interval=(3, 15), old_step_size=std_stepsize,
                                  refined_step_size=0.5, device=device)

    ''' Initialize Targets Varying H and True Target'''
    # Values
    mu_tar_true, std_tar_true = 10.0, 1.0
    mu_tar_high1, std_tar_high1 = 100.0, 10.0
    mu_tar_high2, std_tar_high2 = 120.0, 20.0
    mu_tar_high3, std_tar_high3 = 150.0, 50.0
    # Initialize Fixed True Targets
    mean_tar_true = torch.as_tensor(mu_tar_true, device=device).unsqueeze(dim=0)
    std_tar_true = torch.as_tensor(std_tar_true, device=device).unsqueeze(dim=0)
    distr_tar_true = Normal(loc=mean_tar_true, scale=std_tar_true)
    # Initialize Fixed Target Tensors
    mu_tar_high1 = torch.as_tensor(mu_tar_high1, device=device).unsqueeze(dim=0)
    mu_tar_high2 = torch.as_tensor(mu_tar_high2, device=device).unsqueeze(dim=0)
    mu_tar_high3 = torch.as_tensor(mu_tar_high3, device=device).unsqueeze(dim=0)
    std_tar_high1 = torch.as_tensor(std_tar_high1, device=device).unsqueeze(dim=0)
    std_tar_high2 = torch.as_tensor(std_tar_high2, device=device).unsqueeze(dim=0)
    std_tar_high3 = torch.as_tensor(std_tar_high3, device=device).unsqueeze(dim=0)

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
            delta_i = compute_delta_c(mu_curr=mu_curr_fixed, std_curr=std_i, mu_tar_true=mu_tar_true,
                                      std_tar_true=std_tar_true, mu_tar_noisy=mu_tar_noisy_i,
                                      std_tar_noisy=std_tar_i, outer_supp=supp_outer_var_std, lr=learning_rate,
                                      cramer_supp=supp_cr, tar_noisy=tar_distr, tar_true=distr_tar_true,
                                      nabla=1, dev=device)
            deltas.append(delta_i)
        deltas_all_tars.append(deltas)

    '''Save Parameters'''
    ts = time.time()
    curr_path = os.getcwd()
    saving_path = curr_path + '/' + 'Overestimation_' + str(ts)
    os.mkdir(saving_path)

    ''' Plot and Save Graphs '''
    # Plot Varying Mu with Fixed Target
    plt.rcParams['figure.figsize'] = (30, 12)
    # for idx, deltas in enumerate(deltas_all_tars):
    plt.plot(std_range, deltas_all_tars[0], label=f'Delta(s,a) Curve - TarNoisy: Mu={mu_tar_high1.item()}, '
                                                  f'Std={std_tar_high1.item()}')
    plt.plot(std_range, deltas_all_tars[1], label=f'Delta(s,a) Curve - TarNoisy: Mu={mu_tar_high2.item()}, '
                                                  f'Std={std_tar_high2.item()}')
    plt.plot(std_range, deltas_all_tars[2], label=f'Delta(s,a) Curve - TarNoisy: Mu={mu_tar_high3.item()}, '
                                                  f'Std={std_tar_high3.item()}')
    plt.title(f'Delta(s,a) - Curr: Mu={mu_curr_fixed}, StdRange={std_l}-{std_u}')
    plt.xlabel('Curr. Std')
    plt.ylabel('Delta(s,a)')
    # plt.ylim((-5, 0))
    plt.legend()
    plt.savefig(saving_path + '/' + 'VaryingCurrStd_Delta.png')
    plt.show(block=True)



