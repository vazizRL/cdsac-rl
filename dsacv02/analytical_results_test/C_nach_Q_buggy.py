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

    refined_interval = torch.arange(int_i + 1e-10, int_ii, refined_step_size, device=device)

    new_arr = torch.cat((arr_i, refined_interval, arr_ii), dim=0)

    return new_arr


def mu_expr(dt, mu_curr, sigma_curr, mu_tar, sigma_tar):
    """
    - Fixed version
    :param dt: Incremental Supports
    :param mu_curr: Mean of current Gauss
    :param sigma_curr: Standard dev. of current Gauss
    :param mu_tar: Mean of target Gauss
    :param sigma_tar: Standard dev. of target Gauss
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


def partial_c_partial_q(mu_curr, std_curr, mu_tar, std_tar, pdf_tar, outer_supp, lr, cramer_supp, nabla=1, dev='cpu'):

    pdf_curr = Normal(loc=mu_curr, scale=std_curr)

    b = 1 / cramer_optim_1k(pdf_target=pdf_tar, pdf_curr=pdf_curr, standard_supp=cramer_supp,
                            n_kernels=1, dev=dev)

    delta_double_inte = get_outer_int(outer_supports=outer_supp, mu_curr=mu_curr, std_curr=std_curr,
                                      mu_tar=mu_tar, std_tar=std_tar)

    frac = -1 / (std_curr**3 * sqrt(2*torch.pi))

    return lr * frac * b * delta_double_inte *nabla


if __name__ == '__main__':
    device = 'cpu'
    learning_rate = 1.0
    ''' Numerical Configuration'''
    # Double Integral Configuration
    outer_var_std_l = -180
    outer_var_std_u = 180
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
    std_u = 20.0
    std_stepsize = 0.1          # Old: 0.5
    std_range = torch.arange(std_l, std_u, std_stepsize, device=device)
    # Focus- Interval
    # std_range = focus_on_interval(arr=std_range, focus_interval=(0.01, 0.7), old_step_size=std_stepsize,
    #                               refined_step_size=0.05, device=device)

    ''' Initialize Target '''
    # Values
    mu_tar, std_tar = 5.0, 1.0
    # Initialize Fixed True Targets
    mean_tar = torch.as_tensor(mu_tar, device=device).unsqueeze(dim=0)
    std_tar = torch.as_tensor(std_tar, device=device).unsqueeze(dim=0)
    distr_tar = Normal(loc=mean_tar, scale=std_tar)

    ''' Perform Evaluations '''
    derivatives = list()
    for std_i in std_range:
        dCdQ = partial_c_partial_q(mu_curr=mu_curr_fixed, std_curr=std_i, mu_tar=mu_tar,
                                   std_tar=std_tar, pdf_tar=distr_tar, outer_supp=supp_outer_var_std, lr=learning_rate,
                                   cramer_supp=supp_cr, nabla=1, dev=device)
        derivatives.append(dCdQ)

    '''Save Parameters'''
    ts = time.time()
    curr_path = os.getcwd()
    saving_path = curr_path + '/' + 'dCdQ_' + str(ts)
    os.mkdir(saving_path)

    ''' Plot and Save Graphs '''
    # Plot Varying Mu with Fixed Target
    plt.rcParams['figure.figsize'] = (30, 12)
    # for idx, deltas in enumerate(deltas_all_tars):
    plt.plot(std_range, derivatives, label=f'dCdQ Curve - Tar: Mu={mean_tar.item()}, '
                                           f'Std={std_tar.item()}')

    plt.title(f'dCdQ - Curr: Mu={mu_curr_fixed}, StdRange={std_l}-{std_u}')
    plt.xlabel('Curr. Std')
    plt.ylabel('dCdQ')
    # plt.ylim((-5, 0))
    plt.legend()
    plt.savefig(saving_path + '/' + 'VaryingCurrStd_dCdQ.png')
    plt.show(block=True)



