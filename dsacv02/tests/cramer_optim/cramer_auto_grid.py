import torch
import gc
import torch.optim as optim
import torch.distributions as distr
import os
import time
from scipy import integrate
from dsacv02.gmm_reparameterization.mixture_same_family import ReparameterizedMixtureSameFamilyMod as RMM
from dsacv02.mlp_gmm import MLPGMM, MLPGMMWeighted
from copy import deepcopy, copy


def cramer_py_test(pdf_target: torch.tensor, pdf_curr: torch.tensor, int_l, int_u, spacing, dev='cpu'):
    """
    - The integration limits should NOT be too far off form the lowest and highest point of the function!
    - Optional: Define an interval to focus on, in case of rapid
    - Implementation:
        1. Define the supports
        2. Calculate the difference squared
        3. Integrate over all dx
    """
    # Discretize for numerical integration
    steps = int((int_u - int_l) / spacing)
    dx = torch.linspace(int_l, int_u, steps=steps).to(dev)
    dx.unsqueeze_(dim=1).unsqueeze_(dim=2)

    dy_curr_cdf = pdf_curr.cdf(dx)
    dy_target_cdf = pdf_target.cdf(dx)

    cramer = torch.trapz((dy_target_cdf - dy_curr_cdf)**2, dx=spacing)
    cramer = cramer.sum(dim=0)

    return cramer**0.5


def cramer_py_test_deac(pdf_target: torch.tensor, pdf_curr: torch.tensor, int_l, int_u, spacing, dev='cpu'):
    """
    - The integration limits should NOT be too far off form the lowest and highest point of the function!
    - Optional: Define an interval to focus on, in case of rapid
    - Implementation:
        1. Define the supports
        2. Calculate the difference squared
        3. Integrate over all dx
    """
    # Discretize for numerical integration
    steps = int((int_u - int_l) / spacing)
    dx = torch.linspace(int_l, int_u, steps=steps).to(dev)
    dx.unsqueeze_(dim=1).unsqueeze_(dim=2)

    dy_curr_cdf_re = pdf_curr.cdf(dx).reshape(20, 1, 22000)
    dy_target_cdf_re = pdf_target.cdf(dx).reshape(20, 1, 22000)
    cramer_re = torch.trapz((dy_target_cdf_re - dy_curr_cdf_re)**2, dx=spacing)
    cramer_re.sqrt_()
    cramer_re = cramer_re.mean()

    return cramer_re


# def cramer_py_test(pdf_target: torch.tensor, pdf_curr: torch.tensor, int_l, int_u, spacing, dev='cpu'):
#     """
#     - The integration limits should NOT be too far off form the lowest and highest point of the function!
#     - Optional: Define an interval to focus on, in case of rapid
#     - Implementation:
#         1. Define the supports
#         2. Calculate the difference squared
#         3. Integrate over all dx
#     """
#     # Discretize for numerical integration
#     steps = int((int_u - int_l) / spacing)
#     dx = torch.linspace(int_l, int_u, steps=steps).to(dev)
#     dx.unsqueeze_(dim=1).unsqueeze_(dim=2)
#
#     dy_curr_cdf = pdf_curr.cdf(dx)
#     dy_target_cdf = pdf_target.cdf(dx)
#
#     cramer = torch.trapz((dy_target_cdf - dy_curr_cdf)**2, dx=spacing, dim=0)
#     cramer.sqrt_()
#     cramer = cramer.mean()
#
#     return cramer


# def cramer_py_test(pdf_target: torch.tensor, pdf_curr: torch.tensor, int_l, int_u, spacing, dev='cpu'):
#     """
#     - The integration limits should NOT be too far off form the lowest and highest point of the function!
#     - Optional: Define an interval to focus on, in case of rapid changes
#     - Implementation:
#         1. Define the supports
#         2. Calculate the difference squared
#         3. Integrate over all dx
#     """
#     # Tar
#     means_tar = pdf_target.component_distribution.loc
#     stds_tar = pdf_target.component_distribution.scale
#     means_tar_sorted = means_tar.sort()
#     means_tar_sorted_idx = means_tar_sorted.indices
#     stds_tar_sorted = torch.gather(stds_tar, dim=1, index=means_tar_sorted_idx)
#
#     stds_tar_sorted_op = torch.clone(stds_tar_sorted)
#     stds_tar_sorted_op[:, 1] = stds_tar_sorted_op[:, 1] * (-1)
#     kernel_borders_tar = means_tar_sorted.values + 3.1 * stds_tar_sorted_op
#     kernel_distance_large_tar = kernel_borders_tar[:, 0] < kernel_borders_tar[:, 1]
#
#     # Curr
#     means_curr = pdf_curr.component_distribution.loc
#     stds_curr = pdf_curr.component_distribution.scale
#     means_curr_sorted = means_curr.sort()
#     means_curr_sorted_idx = means_curr_sorted.indices
#     stds_curr_sorted = torch.gather(stds_curr, dim=1, index=means_curr_sorted_idx)
#
#     stds_curr_sorted_op = torch.clone(stds_curr_sorted)
#     stds_curr_sorted_op[:, 1] = stds_curr_sorted_op[:, 1] * (-1)
#     kernel_borders_curr = means_curr_sorted.values + 3.1 * stds_curr_sorted_op
#     kernel_distance_large_curr = kernel_borders_curr[:, 0] < kernel_borders_curr[:, 1]
#
#     # Discretize for numerical integration
#     steps = int((int_u - int_l) / spacing)
#     dx = torch.linspace(int_l, int_u, steps=steps).to(dev)
#     dx.unsqueeze_(dim=1).unsqueeze_(dim=2)
#
#     dy_curr_cdf = pdf_curr.cdf(dx)
#     dy_target_cdf = pdf_target.cdf(dx)
#
#     cramer = torch.trapz((dy_target_cdf - dy_curr_cdf)**2, dx=spacing)
#     cramer = cramer.sum(dim=0)
#
#     return cramer**0.5


def generate_gmm(locs: torch.tensor, scales: torch.tensor, kweights: torch.tensor):
    # Test mysterious symmetry
    gmm = RMM(distr.Categorical(probs=kweights), distr.Normal(locs, scales))

    return gmm


def calculate_cdf(pdf: torch.tensor, supp_l, supp_u, spacing, dev='cpu'):
    steps = int((supp_u - supp_l) / spacing)
    dx = torch.linspace(supp_l, supp_u, steps=steps).to(dev)
    dx.unsqueeze_(dim=1).unsqueeze_(dim=2)

    cdf_curve = pdf.cdf(dx).mean(dim=2)

    return cdf_curve


if __name__ == '__main__':
    curr_path = os.getcwd()
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cuda:0'

    # Network parameters
    arch = (1, 10, 1)
    activ = ('gelu',)
    n_kernels = 2
    multivar = False
    learnable_weights = True
    kweights_fixed = torch.ones(n_kernels, dtype=torch.float64) / n_kernels
    kweights_fixed.to(device)

    # Initialize networks and optimizers for standard and improved cramer calculation
    if learnable_weights:
        gmm_approx_std = MLPGMMWeighted(arch=arch, activ=activ, n_kernels=n_kernels, device=device, multivar=multivar)
    else:
        gmm_approx_std = MLPGMM(arch=arch, activ=activ, n_kernels=n_kernels, device=device, multivar=multivar)
    optimizer_std = optim.Adam(gmm_approx_std.parameters(), lr=0.001)

    gmm_approx_imp = deepcopy(gmm_approx_std)
    optimizer_imp = optim.Adam(gmm_approx_imp.parameters(), lr=0.001)

    # Define Inputs, uniformly sampled from [-1,1]
    N_DATAPOINTS = 6000
    MB_SIZE = 20
    EPISODES_STD = 6
    EPISODES_IMP = 20
    INT_L = -6
    INT_U = 16
    n_mb = int(N_DATAPOINTS / MB_SIZE)
    input_total = torch.randn(size=(N_DATAPOINTS, 1)).to(device)
    batches = input_total.view(n_mb, MB_SIZE, 1)

    # Target distribution batches
    mean_target = (torch.ones(size=(MB_SIZE, 1)) * torch.tensor([2.0, 10.0], dtype=torch.float64)).to(device)
    std_target = (torch.ones(size=(MB_SIZE, 1)) * torch.tensor([1.0, 1.0], dtype=torch.float64)).to(device)
    kweight_target = (torch.ones(size=(MB_SIZE, 1)) * torch.tensor([0.5, 0.5], dtype=torch.float64)).to(device)
    distr_target = generate_gmm(locs=mean_target, scales=std_target, kweights=kweight_target)

    '''
    Standard Cramer Calculation
    '''
    # Logging array initialization
    means_history_std = torch.tensor([[0, 0]], dtype=torch.float64, device=device)
    stds_history_std = torch.tensor([[0, 0]], dtype=torch.float64, device=device)
    kweights_history_std = torch.tensor([[0, 0]], dtype=torch.float64, device=device)
    cramer_loss_history_std = torch.tensor([], dtype=torch.float64, device=device)
    for i in range(EPISODES_STD):
        for batch in batches:
            means_std, stds_std, kweights_std = gmm_approx_std(batch)
            stds_std.abs_()
            means_std.squeeze_(dim=2)
            stds_std.squeeze_(dim=2)

            # Log Means, Stds and Kweights
            means_history_std = torch.cat((means_history_std, means_std.mean(dim=0).unsqueeze(dim=0)), dim=0)
            stds_history_std = torch.cat((stds_history_std, stds_std.mean(dim=0).unsqueeze(dim=0)), dim=0)
            kweights_history_std = torch.cat((kweights_history_std, kweights_std.mean(dim=0).unsqueeze(dim=0)), dim=0)

            gmm_preds_std = generate_gmm(locs=means_std, scales=stds_std, kweights=kweights_std)

            time_cramer_calc_start = time.perf_counter()
            loss_batch_std = cramer_py_test(pdf_curr=gmm_preds_std, pdf_target=distr_target, int_l=INT_L, int_u=INT_U,
                                            spacing=0.001, dev=device)
            time_cramer_calc_end = time.perf_counter()
            cramer_loss_history_std = torch.cat((cramer_loss_history_std,
                                                 loss_batch_std.unsqueeze(dim=0).unsqueeze(dim=1)), dim=0)

            optimizer_std.zero_grad()

            time_grad_calc_start = time.perf_counter()
            loss_batch_std.mean().backward()
            time_grad_calc_end = time.perf_counter()

            optimizer_std.step()
        print(f'Episode {i} finished')

    '''
    Improved Cramer Calculation
    '''
    pass

    '''
    Save data
    '''
    # Make parent directory
    ts = time.time()
    save_path = curr_path + '/' + str(ts)
    os.mkdir(save_path)

    # Means, Kweights and Stds for Standard Version
    torch.save(means_history_std.detach(), save_path + '/' + 'means_history_std')
    torch.save(stds_history_std.detach(), save_path + '/' + 'stds_history_std')
    torch.save(kweights_history_std.detach(), save_path + '/' + 'kweights_history_std')
    torch.save(cramer_loss_history_std.detach(), save_path + '/' + 'cramer_loss_history_std')

    # Means Kweights and Stds for Improved Version
    pass

    # Save reference CDf serving as target
    cdf_ref = calculate_cdf(distr_target, supp_l=INT_L, supp_u=INT_U, spacing=0.01, dev=device)
    torch.save(cdf_ref.detach(), save_path + '/' + 'cdf_ref')
    # Save CDFs with last output of gmm approximators
    cdf_gmm_pred_std = calculate_cdf(gmm_preds_std, supp_l=INT_L, supp_u=INT_U, spacing=0.01, dev=device)
    torch.save(cdf_gmm_pred_std.detach(), save_path + '/' + 'cdf_gmm_pred_std')








