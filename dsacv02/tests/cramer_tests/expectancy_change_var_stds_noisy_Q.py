"""
Testing effect of updating current distribution with target that has same means but random stds
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distr
import os
import time
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
    expected_mean1 = 2.0
    expected_mean2 = 15.0
    noise_uni = 1
    mean_tar = (torch.ones(size=(mb_size, 1)) * torch.tensor([expected_mean1, expected_mean2], dtype=torch.float64)).to(device)
    std_tar_var = (torch.ones(size=(mb_size, 1)) * torch.tensor([1.0, 1.0], dtype=torch.float64)).to(device)
    kweight_tar = (torch.ones(size=(mb_size, 1)) * torch.tensor([0.5, 0.5], dtype=torch.float64)).to(device)
    distr_tar = generate_gmm(locs=mean_tar, scales=std_tar_var, kweights=kweight_tar)
    std_interval_l, std_interval_u = 1.0, 6.0

    # Train parameters
    episodes_ref = 4
    episodes_tar = 16

    '''
    Fit reference
    '''
    for i in range(episodes_ref):
        for batch in batches:
            means_ref, stds_ref, kweights_ref = gmm_approx_ref(batch)
            stds_ref.abs_()
            means_ref.squeeze_(dim=2)
            stds_ref.squeeze_(dim=2)
            # gmm_preds = generate_gmm(locs=means, scales=stds, kweights=kweights.unsqueeze(dim=2))
            gmm_preds_ref = generate_gmm(locs=means_ref, scales=stds_ref, kweights=kweights_ref)
            loss_batch_ref = cramer_distance(pdf_curr=gmm_preds_ref, pdf_target=distr_ref, int_l=-6, int_u=6,
                                             spacing=0.001, dev=device)
            optimizer_ref.zero_grad()
            loss_batch_ref.mean().backward()
            optimizer_ref.step()
        print(f'Episode {i} finished')
    print(f'Last cramer loss of ref: {loss_batch_ref.mean()}')

    '''
    Fit to target with changing stds and noisy Q
    '''
    gmm_approx_tar = deepcopy(gmm_approx_ref)
    optimizer_tar = optim.Adam(gmm_approx_tar.parameters(), lr=0.001)
    means_history_tar = torch.tensor([[0, 0]], dtype=torch.float64, device=device)
    stds_history_tar = torch.tensor([[0, 0]], dtype=torch.float64, device=device)
    kweights_history_tar = torch.tensor([[0, 0]], dtype=torch.float64, device=device)
    cramer_loss_history_tar = torch.tensor([], dtype=torch.float64, device=device)
    for i in range(episodes_tar):
        for batch in batches:
            means_pred_tar, stds_pred_tar, kweights_pred_tar = gmm_approx_tar(batch)
            stds_pred_tar.abs_()
            means_pred_tar.squeeze_(dim=2)
            stds_pred_tar.squeeze_(dim=2)

            means_history_tar = torch.cat((means_history_tar, means_pred_tar.mean(dim=0).unsqueeze(dim=0)), dim=0)
            stds_history_tar = torch.cat((stds_history_tar, stds_pred_tar.mean(dim=0).unsqueeze(dim=0)), dim=0)
            kweights_history_tar = torch.cat((kweights_history_tar, kweights_pred_tar.mean(dim=0).unsqueeze(dim=0)), dim=0)

            gmm_preds_tar = generate_gmm(locs=means_pred_tar, scales=stds_pred_tar, kweights=kweights_pred_tar)
            # Works only for 2 Kernels!
            noise_mean1, noise_mean2 = (noise_uni - (-noise_uni)) * torch.rand((2,)) - noise_uni
            noisy_means_tar = (torch.ones(size=(mb_size, 1)) *
                               torch.tensor([expected_mean1 + noise_mean1, expected_mean2 + noise_mean2],
                               dtype=torch.float64)).to(device)
            std_tar_k1, std_tar_k2 = (std_interval_u - std_interval_l) * torch.rand((2,)) + std_interval_l
            std_tar_var = (torch.ones(size=(mb_size, 1)) * torch.tensor([std_tar_k1, std_tar_k2],
                                                                        dtype=torch.float64)).to(device)
            distr_tar = generate_gmm(locs=noisy_means_tar, scales=std_tar_var, kweights=kweight_tar)

            loss_batch_tar = cramer_distance(pdf_curr=gmm_preds_tar, pdf_target=distr_tar, int_l=-15, int_u=45,
                                             spacing=0.001, dev=device)
            loss_batch_mean_tar = loss_batch_tar.mean()
            cramer_loss_history_tar = torch.cat((cramer_loss_history_tar, loss_batch_mean_tar.unsqueeze(dim=0).unsqueeze(dim=1)),
                                                dim=0)

            optimizer_tar.zero_grad()
            loss_batch_mean_tar.backward()
            optimizer_tar.step()
        print(f'Episode {i} finished')
    print(f'Last cramer loss of ref: {loss_batch_mean_tar.mean()}')

    '''
    Save
    '''
    # Make parent directory
    ts = time.time()
    save_path = curr_path + '/' + f'VarStd_{std_interval_l}-{std_interval_u}_Mod1_NoisyQ' + str(ts)
    os.mkdir(save_path)

    # Save CDFs with last output of gmm approximators
    cdf_gmm_pred_ref = calculate_cdf(gmm_preds_ref, supp_l=-8, supp_u=13, spacing=0.01, dev=device)
    cdf_ref = calculate_cdf(distr_ref, supp_l=-8, supp_u=13, spacing=0.01, dev=device)
    torch.save(cdf_gmm_pred_ref.detach(), save_path + '/' + 'cdf_gmm_pred_ref')
    torch.save(cdf_ref.detach(), save_path + '/' + 'cdf_ref')

    # Save Means, Kernel Weights, Stds and Cramer Loss for Target Fitting
    torch.save(means_history_tar.detach(), save_path + '/' + 'means_history_tar')
    torch.save(stds_history_tar.detach(), save_path + '/' + 'stds_history_tar')
    torch.save(kweights_history_tar.detach(), save_path + '/' + 'kweights_history_tar')
    torch.save(cramer_loss_history_tar.detach(), save_path + '/' + 'cramer_loss_history_tar')


