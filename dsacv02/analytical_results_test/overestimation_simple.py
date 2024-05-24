import torch
import os
import time
import matplotlib.pyplot as plt
from dsacv02.tools import cramer_optim_1k, get_normal_supports
from torch.distributions.normal import Normal
from math import sqrt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def mu_expr(dt, mu, sigma, nabla=1, dev='cpu'):
    """
    -
    :param dt: Incremental Supports
    :param mu: Mean
    :param sigma: Standard Dev.
    :param nabla: Gradient of parameterized \mu_{\theta}
    :return:
    """
    return torch.e**(-0.5*((dt-mu)/sigma)**2) * (dt - mu) * nabla


def get_inner_int(inner_supports, mu_curr, sigma_curr, nabla=1):
    dy_inner_supports = mu_expr(dt=inner_supports, mu=mu_curr, sigma=sigma_curr, nabla=nabla)

    return torch.trapz(y=dy_inner_supports, x=inner_supports)


def get_outer_int(outer_supports, mu_curr, std_curr):
    dy_inner_list = list()
    for idx, _ in enumerate(outer_supports):
        dy_inner_i = get_inner_int(inner_supports=outer_supports[:idx], mu_curr=mu_curr, sigma_curr=std_curr, nabla=1)
        dy_inner_list.append(dy_inner_i)
    dy_inner_expr = torch.as_tensor(dy_inner_list)
    dy_outer = torch.trapz(y=dy_inner_expr, x=outer_supports)

    return dy_outer


def compute_delta_c(mu_curr, std_curr, outer_supp, lr, cramer_supp, tar_noisy, tar_true, nabla=1, dev='cpu'):
    pdf_curr = Normal(loc=mu_curr, scale=std_curr)
    cramer_loss_noisy = cramer_optim_1k(pdf_target=tar_noisy, pdf_curr=pdf_curr, standard_supp=cramer_supp,
                                        n_kernels=1, dev=dev)
    cramer_loss_true = cramer_optim_1k(pdf_target=tar_true, pdf_curr=pdf_curr, standard_supp=cramer_supp,
                                       n_kernels=1, dev=dev)

    double_int = get_outer_int(outer_supports=outer_supp, mu_curr=mu_curr, std_curr=std_curr)
    cramer_diff = cramer_loss_noisy - cramer_loss_true
    frac = 1 / (std_curr**3 * sqrt(2*torch.pi))

    return double_int * lr * frac * cramer_diff * nabla


if __name__ == '__main__':
    device = 'cpu'
    learning_rate = 1.0
    ''' Numerical Configuration'''
    # Double Integral Configuration
    outer_var_std_l = -810
    outer_var_std_u = 810
    stepsize_var_std = 0.05
    supp_outer_var_std = torch.arange(outer_var_std_l, outer_var_std_u, stepsize_var_std)
    # Cramer Settings
    n_supp = 31
    ibf = 15
    supp_cr = get_normal_supports(batch_size=1, n_kernels=1, n_supp=n_supp, integral_bound_factor=ibf,
                                  dev=device)

    ''' Current Range '''
    # Mu Fixed
    mu_fixed = torch.tensor(0.0, device=device)
    # Standard Deviation Range
    std_l = 0.1
    std_u = 200.0
    std_stepsize = 20.0          # Old: 0.5
    std_range = torch.arange(std_l, std_u, std_stepsize, device=device)

    ''' Initialize Targets Varying H and True Target'''
    # Values
    mu_tar_true, std_tar_true = 10.0, 1.0
    mu_tar_high1, std_tar_high1 = 15.0, 30.0
    mu_tar_high2, std_tar_high2 = 15.0, 35.0
    mu_tar_high3, std_tar_high3 = 15.0, 200.0
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
    target_distributions_noisy = [distr_tar_high1, distr_tar_high2, distr_tar_high3]

    ''' Perform Evaluations '''
    deltas_all_tars = list()
    for tar_distr in target_distributions_noisy:
        deltas = list()
        for std_i in std_range:
            delta_i = compute_delta_c(mu_curr=mu_fixed, std_curr=std_i, outer_supp=supp_outer_var_std, lr=learning_rate,
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
    plt.title(f'Delta(s,a) - Curr: Mu={mu_fixed}, StdRange={std_l}-{std_u}')
    plt.xlabel('Curr. Std')
    plt.ylabel('Delta(s,a)')
    # plt.ylim((-5, 0))
    plt.legend()
    plt.savefig(saving_path + '/' + 'VaryingCurrStd_Delta.png')
    plt.show(block=True)



