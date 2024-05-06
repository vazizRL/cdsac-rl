import torch
import os
import time
import matplotlib.pyplot as plt
from dsacv02.tools import cramer_optim_1k, get_normal_supports
from torch.distributions.normal import Normal
from math import sqrt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def mu_expr(dt, mu, sigma, nabla=1):
    """
    -
    :param dt:
    :param mu:
    :param sigma:
    :param nabla:
    :return:
    """
    return torch.e**(-0.5*((dt-mu)/sigma)**2) * (dt - mu) * nabla


def get_inner_exp(inner_supports, mu, sigma, nabla=1):
    dy_inner_supports = mu_expr(dt=inner_supports, mu=mu, sigma=sigma, nabla=nabla)

    return torch.trapz(y=dy_inner_supports, x=inner_supports)


def p_c_p_mu(pdf_curr, pdf_tar, cramer_supports, outer_supports, mu, sigma, wrt_tar_params=False):
    """
    - Can be used to calculate partial derivative of Cramer los w.r.t either target or parameterized \mu
    :param pdf_curr: Current probability density function
    :param pdf_tar: Target probability density function
    :param cramer_supports: Supports for calculation of cramer support
    :param outer_supports: Outer supports determine inner supports!
    :param mu: Means
    :param sigma: Standard deviations
    :param wrt_tar_params: If gradients are calculated w.r.t. target parameters, then this parameter is True
    """

    d_c = cramer_optim_1k(pdf_curr=pdf_curr, pdf_target=pdf_tar, standard_supp=cramer_supports)
    b = 1 / d_c
    if wrt_tar_params:
        b *= -1
    factor = b * (1/(sigma**3 * sqrt(2*torch.pi)))

    dy_inner_list = list()
    for idx, _ in enumerate(outer_supports):
        dy_inner_i = get_inner_exp(inner_supports=outer_supports[:idx], mu=mu, sigma=sigma, nabla=1)
        dy_inner_list.append(dy_inner_i)
    dy_inner_expr = torch.as_tensor(dy_inner_list)
    dy_outer = torch.trapz(y=dy_inner_expr, x=outer_supports)

    return factor * dy_outer


if __name__ == '__main__':
    device = 'cpu'

    ''' Define Supports '''
    # Supports for var_mu
    outer_var_mu_l = -55
    outer_var_mu_u = 45
    stepsize_var_mu = 0.05
    supports_outer_var_mu = torch.arange(outer_var_mu_l, outer_var_mu_u, stepsize_var_mu)
    # Supports for var_std
    outer_var_std_l = -110
    outer_var_std_u = 150
    stepsize_var_std = 0.05
    supports_outer_var_std = torch.arange(outer_var_std_l, outer_var_std_u, stepsize_var_std)

    ''' Current Distribution / Variables '''
    # Mu Range
    mu_l = -30
    mu_u = 20
    mu_stepsize = 0.5
    mu_range = torch.arange(mu_l, mu_u, mu_stepsize, device=device)
    # Std Range
    std_l = 0.1
    std_u = 30.0
    std_stepsize = 0.5
    std_range = torch.arange(std_l, std_u, std_stepsize, device=device)
    '''Current Distribution / Fixed '''
    # Mu Fixed - For VarStd
    mu_fixed = 20.0
    n_std = int((std_u - std_l)/std_stepsize) + 1
    mu_range_fixed = torch.ones(n_std) * torch.as_tensor(mu_fixed, device=device)
    # Std Fixed - For VarMu
    std_fixed = 1.0
    n_mu = int((mu_u - mu_l)/mu_stepsize)
    std_range_fixed = torch.ones(n_mu) * torch.as_tensor(std_fixed, device=device)

    ''' Target Distribution '''
    # Fixed
    mean_tar = torch.as_tensor(10.0, device=device).unsqueeze(dim=0)
    std_tar = torch.as_tensor(3.0, device=device).unsqueeze(dim=0)
    distr_tar = Normal(loc=mean_tar, scale=std_tar)

    ''' Cramer Settings '''
    n_supp = 31
    ibf = 10
    supp_cr = get_normal_supports(batch_size=1, n_kernels=1, n_supp=n_supp, integral_bound_factor=ibf,
                                  dev=device)

    ''' Calculate dC/dMu w.r.t. current distribution, vary Mu'''
    grad_list_var_mu = list()
    for mu, std_f in zip(mu_range, std_range_fixed):
        distr_curr_var_mu_i = Normal(loc=mu, scale=std_f)
        grad_var_mu_i = p_c_p_mu(pdf_curr=distr_curr_var_mu_i, pdf_tar=distr_tar, cramer_supports=supp_cr,
                                 outer_supports=supports_outer_var_mu, mu=mu, sigma=std_f, wrt_tar_params=False)
        grad_list_var_mu.append(grad_var_mu_i)

    ''' Calculate dC/dMu w.r.t. current distribution, vary Std'''
    grad_list_var_std = list()
    for mu_f, std in zip(mu_range_fixed, std_range):
        distr_curr_var_std_i = Normal(loc=mu_f, scale=std)
        grad_var_std_i = p_c_p_mu(pdf_curr=distr_curr_var_std_i, pdf_tar=distr_tar, cramer_supports=supp_cr,
                                  outer_supports=supports_outer_var_std, mu=mu_f, sigma=std, wrt_tar_params=False)
        grad_list_var_std.append(grad_var_std_i)

    '''Save Parameters'''
    ts = time.time()
    curr_path = os.getcwd()
    saving_path = curr_path + '/' + 'MuTest_' + str(ts)
    os.mkdir(saving_path)

    ''' Plot and Save Graphs '''
    # Plot Varying Mu with Fixed Target
    plt.rcParams['figure.figsize'] = (30, 12)
    plt.plot(mu_range, grad_list_var_mu, label='dC/dMu Curve')
    plt.title(f'dC/dMu - Fixed Target, Var. Mean Curr. - Mean: {mean_tar.item()}; Std: {std_tar.item()}')
    plt.xlabel('Curr. Mu')
    plt.ylabel('dC/dMu')
    # plt.ylim((-5, 0))
    plt.legend()
    plt.savefig(saving_path + '/' + 'Varying_CurrMu.png')
    plt.show(block=False)
    # Plot Varying Std with Fixed Target
    plt.figure()
    plt.rcParams['figure.figsize'] = (30, 12)
    plt.plot(std_range, grad_list_var_std, label='dc/dMu Curve')
    plt.title(f'dC/dMu - Fixed Target, Var. Std Curr. - Mean: {mean_tar.item()}; Std: {std_tar.item()}')
    plt.xlabel('Curr. Std')
    plt.ylabel('dC/dMu')
    plt.legend()
    plt.savefig(saving_path + '/' + 'Varying_CurrSTD.png')
    plt.show(block=True)






