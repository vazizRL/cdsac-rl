"""
Testing effect of updating with high-entrpoy vs. low-entropy target
"""
import torch
import torch.optim as optim
import torch.distributions as distr
import os
import time
import matplotlib.pyplot as plt
from dsacv02.gmm_reparameterization.mixture_same_family import ReparameterizedMixtureSameFamilyMod as RMM
from dsacv02.mlp_gmm import MLPGMM, MLPGMMWeighted
from copy import deepcopy
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def cramer_distance(pdf_target: torch.tensor, pdf_curr: torch.tensor, int_l, int_u, spacing, dev='cpu'):
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


def improved_cramer_distance(pdf_target: torch.tensor, pdf_curr: torch.tensor, int_l, int_u, spacing, dev='cpu'):
    """
    - Improved implemented to be scalable
    - Implementation:
        1. Define the supports
        2. Calculate the difference squred
        3. Integrate over all dx
    """
    # Discretize for numerical integration
    steps = int((int_u - int_l) / spacing)
    dx = torch.linspace(int_l, int_u, steps=steps).to(dev)

    dy_curr_cdf = pdf_curr.cdf(dx)
    dy_target_cdf = pdf_target.cdf(dx)

    cramer = torch.trapz((dy_target_cdf - dy_curr_cdf)**2, dx=spacing)

    return cramer**0.5


def generate_gmm(locs: torch.tensor, scales: torch.tensor, kweights: torch.tensor):
    # Test mysterious symmetry
    gmm = RMM(distr.Categorical(probs=kweights), distr.Normal(locs, scales))

    return gmm


def calculate_cdf(pdf: torch.tensor, supp_l, supp_u, spacing, dev='cpu'):
    steps = int((supp_u - supp_l) / spacing)
    dx = torch.linspace(supp_l, supp_u, steps=steps).to(dev)
    dx.unsqueeze_(dim=1).unsqueeze_(dim=2)

    cdf_curve = pdf.cdf(dx)

    return cdf_curve


def approx_integral_bounds(means_curr: torch.tensor, means_target: torch.tensor, stds_curr: torch.tensor,
                           stds_target: torch.tensor, factor, mean_std=False):
    """
    - Approximates numerically relevant integration bounds when given current and target GMM meta-parameters
    :param means_curr: Means of the current GMM
    :param means_target: Means of the target GMM
    :param stds_curr: Standard deviations of the current GMM
    :param stds_target: Standard deviations of the target GMM
    :param factor: Value to me multiplied by mean or highest std
    :param mean_std: Mean of the current and target stds of the GMMs
    :return: relevant lower and higher integration bounds
    """
    means = torch.cat((means_curr, means_target), dim=0)
    l_mean = means.min()
    h_mean = means.max()

    stds = torch.cat((stds_curr, stds_target), dim=0)

    if mean_std:
        bar_std = stds.mean()
        l_integral = l_mean - factor * bar_std
        h_integral = h_mean + factor * bar_std
    else:
        h_std = stds.max()
        l_integral = l_mean - factor * h_std
        h_integral = h_mean + factor * h_std

    return l_integral, h_integral


def draw_cdfs(cdf_pred, cdf_target, block=False):
    plt.figure()
    cdf_ref_np = cdf_pred.detach().sum(dim=2).cpu().numpy()
    cdf_target_np = cdf_target.detach().sum(dim=2).cpu().numpy()
    plt.plot(cdf_ref_np, label='cdf_pred')
    plt.plot(cdf_target_np, label='cdf_target')

    # Add labels and legend
    plt.xlabel('Supports')
    plt.ylabel('Probability')
    plt.title('CDF: Pred vs Target')
    plt.legend()

    # Show the plot
    plt.show(block=block)


if __name__ == '__main__':
    # Control parameters
    save_path = os.getcwd() + '/' + 'improved_cramer'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    graph_cdfs = True

    # Network parameters
    arch = (1, 10, 1)
    activ = ('gelu',)
    n_kernels = 2
    multivar = True
    learnable_weights = True
    kweights_fix = torch.ones(n_kernels, dtype=torch.float64) / n_kernels
    kweights_fix.to(device)

    # Train parameters
    mb_size = 64
    epochs_ref = 2
    epochs_tar = 6

    # Initialize input
    n_datapoints = 64000
    n_mb = int(n_datapoints / mb_size)
    input_total = torch.randn(size=(n_datapoints, 1)).to(device)
    batches = input_total.view(n_mb, mb_size, 1)

    # Reference distribution
    mean_ref = (torch.ones(size=(mb_size, 1)) * torch.tensor([1.0, 1.0], dtype=torch.float64)).to(device)
    std_ref = (torch.ones(size=(mb_size, 1)) * torch.tensor([1.0, 1.0], dtype=torch.float64)).to(device)
    kweight_ref = (torch.ones(size=(mb_size, 1)) * torch.tensor([0.5, 0.5], dtype=torch.float64)).to(device)
    distr_ref = generate_gmm(locs=mean_ref, scales=std_ref, kweights=kweight_ref)

    # Target distribution
    mean_tar = (torch.ones(size=(mb_size, 1)) * torch.tensor([5.0, 6.0], dtype=torch.float64)).to(device)
    # mean_tar = (torch.ones(size=(mb_size, 1)) * torch.tensor([5.0, 50.0], dtype=torch.float64)).to(device)
    std_tar = (torch.ones(size=(mb_size, 1)) * torch.tensor([1.0, 1.0], dtype=torch.float64)).to(device)
    kweight_tar = (torch.ones(size=(mb_size, 1)) * torch.tensor([0.5, 0.5], dtype=torch.float64)).to(device)
    distr_tar = generate_gmm(locs=mean_tar, scales=std_tar, kweights=kweight_tar)

    # cdf_tar = calculate_cdf(distr_tar, supp_l=-10, supp_u=40, spacing=0.01, dev=device)
    # draw_cdfs(cdf_tar, cdf_tar, True)

    # Initialize reference network and optimizer
    if learnable_weights:
        gmm_ref = MLPGMMWeighted(arch=arch, activ=activ, n_kernels=n_kernels, device=device, multivar=multivar)
    else:
        gmm_ref = MLPGMM(arch=arch, activ=activ, n_kernels=n_kernels, device=device, multivar=multivar)
    optimizer_ref = optim.Adam(gmm_ref.parameters(), lr=0.001)

    '''
    Fit gmm_approx to reference distribution
    '''
    for epoch in range(epochs_ref):
        for batch in batches:
            pred_means, pred_stds, kweights = gmm_ref(batch)
            pred_stds.abs_()
            pred_means.squeeze_(dim=2)
            pred_stds.squeeze_(dim=2)
            if learnable_weights:
                pred_gmm_i = generate_gmm(locs=pred_means, scales=pred_stds,
                                          kweights=kweights)
            else:
                pred_gmm_i = generate_gmm(locs=pred_means.squeeze(), scales=pred_stds.squeeze(),
                                          kweights=kweights_fix)
            # Loss data_i
            l_ref, u_ref = approx_integral_bounds(means_curr=pred_means, means_target=mean_ref, stds_curr=pred_stds,
                                                  stds_target=std_ref, factor=5)
            l_ref.detach_().item()
            u_ref.detach_().item()
            cramer_py_loss_i = cramer_distance(pdf_target=distr_ref, pdf_curr=pred_gmm_i, int_l=l_ref,
                                               int_u=u_ref, spacing=0.01, dev=device)

            # Loss in SGD for cramer_py_loss_i
            optimizer_ref.zero_grad()
            cramer_py_loss_i.backward()
            optimizer_ref.step()
        print(f'Finished episode: {epoch + 1} with cramer loss {cramer_py_loss_i}')

    if graph_cdfs:
        cdf_ref = calculate_cdf(pdf=pred_gmm_i, supp_l=l_ref, supp_u=u_ref, spacing=0.01, dev=device)
        cdf_tar = calculate_cdf(pdf=distr_ref, supp_l=l_ref, supp_u=u_ref, spacing=0.01, dev=device)
        draw_cdfs(cdf_pred=cdf_ref, cdf_target=cdf_tar, block=True)
    '''
    Fit gmm_approx to target
    '''
    print('\n --------------------------- \n')
    gmm_tar = deepcopy(gmm_ref)
    optimizer_tar = optim.Adam(gmm_tar.parameters(), lr=0.01)

    for epoch in range(epochs_tar):
        for batch in batches:
            pred_means, pred_stds, kweights = gmm_tar(batch)
            pred_stds.abs_()
            pred_means.squeeze_(dim=2)
            pred_stds.squeeze_(dim=2)
            if learnable_weights:
                pred_gmm_tar_i = generate_gmm(locs=pred_means, scales=pred_stds,
                                              kweights=kweights)
            else:
                pred_gmm_tar_i = generate_gmm(locs=pred_means.squeeze(), scales=pred_stds.squeeze(),
                                              kweights=kweights_fix)
            # Loss data_i
            l_tar, u_tar = approx_integral_bounds(means_curr=pred_means, means_target=mean_tar, stds_curr=pred_stds,
                                                  stds_target=std_tar, factor=5)
            l_tar.detach_().item()
            u_tar.detach_().item()
            cramer_py_loss_tar_i = cramer_distance(pdf_target=distr_tar, pdf_curr=pred_gmm_tar_i, int_l=l_tar,
                                                   int_u=u_tar, spacing=0.001, dev=device)

            # Loss in SGD for cramer_py_loss_i
            optimizer_tar.zero_grad()
            cramer_py_loss_tar_i.backward()
            optimizer_tar.step()
        print(f'Finished episode: {epoch + 1} with cramer loss {cramer_py_loss_tar_i}')

    if graph_cdfs:
        cdf_pred = calculate_cdf(pdf=pred_gmm_tar_i, supp_l=l_tar, supp_u=u_tar, spacing=0.01, dev=device)
        cdf_tar = calculate_cdf(pdf=distr_tar, supp_l=l_tar, supp_u=u_tar, spacing=0.01, dev=device)
        draw_cdfs(cdf_pred=cdf_pred, cdf_target=cdf_tar, block=True)






