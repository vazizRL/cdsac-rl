import torch
import os
import time
import matplotlib.pyplot as plt
from dsacv02.tools import get_normal_supports
from torch.distributions.normal import Normal
from math import sqrt, floor
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def calculate_b_expr(pdf_curr: Normal, pdf_tar: Normal, dx):
    """
    - B without integral
    :param pdf_curr: Current normal distribution
    :param pdf_tar: Target normal distribution
    :param dx: Step for integration
    """
    return (pdf_curr.cdf(dx) - pdf_tar.cdf(dx)) * pdf_curr.log_prob(dx).exp()


def get_b_integral(pdf_curr, pdf_tar, supports_integral):
    dy = calculate_b_expr(pdf_curr=pdf_curr, pdf_tar=pdf_tar, dx=supports_integral)
    integration = torch.trapz(y=dy, x=supports_integral)

    return integration


def partial_c_partial_q(mu_curr, std_curr, pdf_tar, supports, lr, cramer_supp, nabla=1, dev='cpu'):
    pdf_curr = Normal(loc=mu_curr, scale=std_curr)
    b = get_b_integral(pdf_curr=pdf_curr, pdf_tar=pdf_tar, supports_integral=supports)
    # c = cramer_optim_1k(pdf_target=pdf_tar, pdf_curr=pdf_curr, standard_supp=cramer_supp, n_kernels=1, dev=dev)
    frac = b
    print(f'b is {b}')

    return - (1/std_curr) * frac * lr * nabla, b


if __name__ == '__main__':
    device = 'cpu'
    learning_rate = 0.003
    ''' Numerical Configuration'''
    # Double Integral Configuration
    outer_var_std_l = -300
    outer_var_std_u = 300
    stepsize_var_std = 0.005
    supp_var_std = torch.arange(outer_var_std_l, outer_var_std_u, stepsize_var_std)
    # Cramer Settings
    n_supp = 31
    ibf = 15
    supp_cr = get_normal_supports(batch_size=1, n_kernels=1, n_supp=n_supp, integral_bound_factor=ibf,
                                  dev=device)

    ''' Current Range '''
    # Mu Fixed
    mu_curr_fixed = torch.tensor(100.0, device=device)
    # Standard Deviation Range
    std_l = 0.1
    std_u = 15.0
    std_stepsize = 0.001          # Old: 0.5
    std_range = torch.arange(std_l, std_u, std_stepsize, device=device)
    # Focus- Interval
    # std_range = focus_on_interval(arr=std_range, focus_interval=(0.01, 0.7), old_step_size=std_stepsize,
    #                               refined_step_size=0.05, device=device)

    ''' Initialize Target '''
    # Values
    mu_tar, std_tar = -100.0, 30.0     # 100, 10.0
    # Initialize Fixed True Targets
    mean_tar = torch.as_tensor(mu_tar, device=device).unsqueeze(dim=0)
    std_tar = torch.as_tensor(std_tar, device=device).unsqueeze(dim=0)
    distr_tar = Normal(loc=mean_tar, scale=std_tar)

    ''' Perform Evaluations '''
    derivatives = list()
    b_hist = list()
    for std_i_curr in std_range:
        dCdQ, b = partial_c_partial_q(mu_curr=mu_curr_fixed, std_curr=std_i_curr, pdf_tar=distr_tar, supports=supp_var_std,
                                   lr=1, cramer_supp=supp_cr, nabla=1, dev='cpu')
        derivatives.append(dCdQ)
        b_hist.append(b)

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



