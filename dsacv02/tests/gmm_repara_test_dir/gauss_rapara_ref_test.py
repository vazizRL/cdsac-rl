import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distr
import os
import time
from sac_implementation.networks import ActorNetwork
from dsacv02.mlp_gmm import MLPGMM, MLPGMMWeighted
from copy import deepcopy


def generate_gauss(locs: torch.tensor, scales: torch.tensor):
    # Test mysterious symmetry
    gauss = distr.Normal(loc=locs, scale=scales)

    return gauss


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
    # dx.unsqueeze_(dim=1).unsqueeze_(dim=2)

    dy_curr_cdf = pdf_curr.cdf(dx)
    dy_target_cdf = pdf_target.cdf(dx)

    cramer = torch.trapz((dy_target_cdf - dy_curr_cdf)**2, dx=spacing)
    # cramer = cramer.sum(dim=1)

    return cramer**0.5


def calculate_cdf(pdf: torch.tensor, supp_l, supp_u, spacing, dev='cpu'):
    steps = int((supp_u - supp_l) / spacing)
    dx = torch.linspace(supp_l, supp_u, steps=steps).to(dev)
    # dx.unsqueeze_(dim=1).unsqueeze_(dim=2)

    cdf_curve = pdf.cdf(dx).mean(dim=0)

    return cdf_curve


if __name__ == '__main__':
    curr_path = os.getcwd()
    device = 'cuda:0'

    # Train parameters
    episodes_ref = 4
    episodes_tar = 9

    n_datapoints = 6000
    mb_size = 10
    n_mb = int(n_datapoints / mb_size)
    input_total = torch.randn(size=(n_datapoints, 1), dtype=torch.float64).to(device)
    batches = input_total.view(n_mb, mb_size, 1)

    gauss_approx_ref = ActorNetwork(alpha=0.001, input_dims=(1,), max_actions=20, fc1_dims=10, fc2_dims=10, n_actions=1,
                                    name='actor', checkpoint_dir='dummy_name')

    # Reference distribution
    mean_ref = (torch.ones(size=(mb_size, 1)) * torch.tensor([1.0], dtype=torch.float64)).to(device)
    std_ref = (torch.ones(size=(mb_size, 1)) * torch.tensor([1.0], dtype=torch.float64)).to(device)
    distr_ref = generate_gauss(locs=mean_ref, scales=std_ref)

    # Target distribution
    mean_tar = (torch.ones(size=(mb_size, 1)) * torch.tensor([6.0], dtype=torch.float64)).to(device)
    std_tar = (torch.ones(size=(mb_size, 1)) * torch.tensor([3.0], dtype=torch.float64)).to(device)
    distr_tar = generate_gauss(locs=mean_tar, scales=std_tar)

    '''
    Fit to reference distribution with cramer
    '''
    for i in range(episodes_ref):
        for batch in batches:
            gauss_means_ref, gauss_stds_ref = gauss_approx_ref(batch)
            gauss_stds_ref.abs_()

            gauss_preds_ref = generate_gauss(locs=gauss_means_ref, scales=gauss_stds_ref)
            loss_batch = cramer_distance(pdf_curr=gauss_preds_ref, pdf_target=distr_ref, int_l=-9, int_u=9,
                                         spacing=0.001, dev=device)
            gauss_approx_ref.optim.zero_grad()
            loss_batch.mean().backward()
            gauss_approx_ref.optim.step()
        print(f'Episode ref: {i+1} finished')
    print(f'End Cramer loss: {loss_batch.mean()}')

    '''
    Fit to target with Gauss Sampling via Reparameterizaiton
    '''
    gauss_approx_tar = deepcopy(gauss_approx_ref)
    optimizer_tar = optim.Adam(gauss_approx_tar.parameters(), lr=0.001)
    kl_loss_tar = torch.tensor([], dtype=torch.float64, device=device)
    for i in range(episodes_tar):
        for batch in batches:
            gauss_means_tar, gauss_stds_tar = gauss_approx_tar(batch)
            gauss_stds_tar.abs_()
            gauss_preds_tar = generate_gauss(locs=gauss_means_tar, scales=gauss_stds_tar)
            # Samples
            samples_approx_tar = gauss_preds_tar.rsample()
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
    Save 
    '''
    ts = time.time()
    save_path = curr_path + '/' + 'Test_GaussRef_' + str(ts)
    os.mkdir(save_path)

    # CDFs
    int_bounds = 20
    cdf_pred_ref = calculate_cdf(gauss_preds_ref, supp_l=-int_bounds, supp_u=int_bounds, spacing=0.01, dev=device)
    cdf_ref = calculate_cdf(distr_ref, supp_l=-int_bounds, supp_u=int_bounds, spacing=0.01, dev=device)
    cdf_tar = calculate_cdf(distr_tar, supp_l=-int_bounds, supp_u=int_bounds, spacing=0.01, dev=device)
    cdf_pred_tar = calculate_cdf(gauss_preds_tar, supp_l=-int_bounds, supp_u=int_bounds, spacing=0.01, dev=device)

    torch.save(cdf_pred_ref.detach(), save_path + '/' + 'cdf_gauss_pred_ref')
    torch.save(cdf_ref.detach(), save_path + '/' + 'cdf_ref')
    torch.save(cdf_tar.detach(), save_path + '/' + 'cdf_tar')
    torch.save(cdf_pred_tar.detach(), save_path + '/' + 'cdf_gauss_pred_tar')
    torch.save(kl_loss_tar.detach(), save_path + '/' + 'kl_loss')
