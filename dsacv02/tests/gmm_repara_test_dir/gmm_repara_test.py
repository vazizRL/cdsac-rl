"""
This experiment is performed to observe whether convergence is guaranted with samples from GMM
and using the KL-Loss
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distr
import os
import time
from scipy import integrate
from dsacv02.gmm_reparameterization.mixture_same_family import ReparameterizedMixtureSameFamilyMod as RMM
from dsacv02.mlp_gmm import MLPGMM, MLPGMMWeighted
from copy import deepcopy


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

    # Initialize network and optimizer
    if learnable_weights:
        gmm_approx_ref = MLPGMMWeighted(arch=arch, activ=activ, n_kernels=n_kernels, device=device, multivar=multivar)
    else:
        gmm_approx_ref = MLPGMM(arch=arch, activ=activ, n_kernels=n_kernels, device=device, multivar=multivar)
    optimizer_ref = optim.Adam(gmm_approx_ref.parameters(), lr=0.001)

    # Define Inputs, uniformly sampled from [-1,1]
    n_datapoints = 6000
    mb_size = 10
    n_mb = int(n_datapoints / mb_size)
    input_total = torch.randn(size=(n_datapoints, 1)).to(device)
    batches = input_total.view(n_mb, mb_size, 1)

    # Referece distribution
    mean_ref = (torch.ones(size=(mb_size, 1)) * torch.tensor([0.0, 1.0], dtype=torch.float64)).to(device)
    std_ref = (torch.ones(size=(mb_size, 1)) * torch.tensor([1.0, 1.0], dtype=torch.float64)).to(device)
    kweight_ref = (torch.ones(size=(mb_size, 1)) * torch.tensor([0.5, 0.5], dtype=torch.float64)).to(device)
    distr_ref = generate_gmm(locs=mean_ref, scales=std_ref, kweights=kweight_ref)

    # Target distribution
    mean_tar = (torch.ones(size=(mb_size, 1)) * torch.tensor([6.0, 6.0], dtype=torch.float64)).to(device)
    std_tar = (torch.ones(size=(mb_size, 1)) * torch.tensor([1.0, 1.0], dtype=torch.float64)).to(device)
    kweight_tar = (torch.ones(size=(mb_size, 1)) * torch.tensor([0.5, 0.5], dtype=torch.float64)).to(device)
    distr_tar = generate_gmm(locs=mean_tar, scales=std_tar, kweights=kweight_tar)

    # Train parameters
    episodes_ref = 2
    episodes_tar = 15

    '''
    Fit GMM with cramer on reference distribution
    '''
    for i in range(episodes_ref):
        for batch in batches:
            gmm_means_ref, gmm_stds_ref, gmm_kweights_ref = gmm_approx_ref(batch)
            gmm_stds_ref.abs_()
            gmm_means_ref.squeeze_(dim=2)
            gmm_stds_ref.squeeze_(dim=2)
            gmm_preds_ref = generate_gmm(locs=gmm_means_ref, scales=gmm_stds_ref, kweights=gmm_kweights_ref)
            loss_batch = cramer_distance(pdf_curr=gmm_preds_ref, pdf_target=distr_ref, int_l=-3, int_u=13, spacing=0.001,
                                         dev=device)
            optimizer_ref.zero_grad()
            loss_batch.mean().backward()
            optimizer_ref.step()
        print(f'Episode ref: {i+1} finished')

    '''
    Fit reference to target distribution through samples with implicit reparameterization technique for GMMs
    '''
    gmm_approx_tar = deepcopy(gmm_approx_ref)
    optimizer_tar = optim.Adam(gmm_approx_tar.parameters(), lr=0.001)
    kl_loss_tar = torch.tensor([], dtype=torch.float64, device=device)
    for i in range(episodes_tar):
        for batch in batches:
            gmm_means_tar, gmm_stds_tar, gmm_kweights_tar = gmm_approx_tar(batch)
            gmm_stds_tar.abs_()
            gmm_means_tar.squeeze_(dim=2)
            gmm_stds_tar.squeeze_(dim=2)
            gmm_preds_tar = generate_gmm(locs=gmm_means_tar, scales=gmm_stds_tar, kweights=gmm_kweights_tar)
            # Samples
            samples_approx_tar = gmm_preds_tar.rsample()
            # samples_approx_tar_log_probs = gmm_preds_tar.log_prob(samples_approx_tar)
            # Loss
            loss_batch_tar = distr_tar.log_prob(samples_approx_tar) * (-1)
            # Optimize
            optimizer_tar.zero_grad()
            loss_batch_tar_mean = loss_batch_tar.mean()
            kl_loss_tar = torch.cat((kl_loss_tar, loss_batch_tar_mean.unsqueeze(dim=0)), dim=0)
            loss_batch_tar_mean.backward()
            optimizer_tar.step()
        print(f'Episode tar: {i+1} finished')

    '''
    Save Data
    '''
    # Make parent directory
    ts = time.time()
    save_path = curr_path + '/' + 'Test_' + str(ts)
    os.mkdir(save_path)

    # Save CDFs with last output of gmm approximators
    cdf_gmm_pred_ref = calculate_cdf(gmm_preds_ref, supp_l=-5, supp_u=13, spacing=0.01, dev=device)
    cdf_ref = calculate_cdf(distr_ref, supp_l=-5, supp_u=13, spacing=0.01, dev=device)
    cdf_gmm_pred_tar = calculate_cdf(gmm_preds_tar, supp_l=-5, supp_u=13, spacing=0.01, dev=device)
    cdf_tar = calculate_cdf(distr_tar, supp_l=-5, supp_u=13, spacing=0.01, dev=device)

    torch.save(cdf_gmm_pred_ref.detach(), save_path + '/' + 'cdf_gmm_pred_ref')
    torch.save(cdf_ref.detach(), save_path + '/' + 'cdf_ref')
    torch.save(cdf_tar.detach(), save_path + '/' + 'cdf_tar')
    torch.save(cdf_gmm_pred_tar.detach(), save_path + '/' + 'cdf_gmm_pred_tar')
    torch.save(kl_loss_tar.detach(), save_path + '/' + 'kl_loss')

    # Save target approximator means, stds, kweights
    # torch.save()
