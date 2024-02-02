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


def cramer_from_pdf(pdf_target: torch.tensor, pdf_curr: torch.tensor, int_l, int_u, points=(-100, 100),
                    dev='cuda:0'):
    """
    - The integration limits should NOT be too far off form the lowest and highest point of the function!
    """
    c: tuple = integrate.quad(
        lambda x: (pdf_target.cdf(torch.tensor([x]).to(dev)) - pdf_curr.cdf(torch.tensor([x]).to(dev))) ** 2,
        int_l, int_u, points=points
    )

    distance, error_est = c

    return distance**0.5, error_est


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


def generate_gmm(locs: torch.tensor, scales: torch.tensor, kweights: torch.tensor):
    # Test mysterious symmetry
    gmm = RMM(distr.Categorical(probs=kweights), distr.Normal(locs, scales))

    return gmm


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
        gmm_approx = MLPGMMWeighted(arch=arch, activ=activ, n_kernels=n_kernels, device=device, multivar=multivar)
    else:
        gmm_approx = MLPGMM(arch=arch, activ=activ, n_kernels=n_kernels, device=device, multivar=multivar)
    optimizer = optim.Adam(gmm_approx.parameters(), lr=0.001)

    # Define Inputs, uniformly sampled from [-1,1]
    n_datapoints = 6000
    mb_size = 10
    n_mb = int(n_datapoints / mb_size)
    input_total = torch.randn(size=(n_datapoints, 1)).to(device)
    batches = input_total.view(n_mb, mb_size, 1)

    # Target distribution
    mean_target = (torch.ones(size=(mb_size, 1)) * torch.tensor([2.0, 8.0], dtype=torch.float64)).to(device)
    # mean_target = (torch.ones(size=(mb_size, n_kernels)) * torch.tensor([2.0, 8.0], dtype=torch.float64)).to(device)
    # mean_target.unsqueeze_(dim=2)
    std_target = (torch.ones(size=(mb_size, 1)) * torch.tensor([1.0, 1.0], dtype=torch.float64)).to(device)
    # std_target = (torch.ones(size=(mb_size, n_kernels)) * torch.tensor([1.0, 1.0], dtype=torch.float64)).to(device)
    # std_target.unsqueeze_(dim=2)
    kweight_target = (torch.ones(size=(mb_size, 1)) * torch.tensor([0.2, 0.8], dtype=torch.float64)).to(device)
    # kweight_target = (torch.ones(size=(mb_size, n_kernels)) * torch.tensor([0.2, 0.8], dtype=torch.float64)).to(device)
    # kweight_target.unsqueeze_(dim=2)
    distr_target = generate_gmm(locs=mean_target, scales=std_target, kweights=kweight_target)

    # Train parameters
    # episodes = 10
    episodes = 5

    # # Put target distribtuion in batch format
    # distr_target_batch = torch.ones(size=(20, 1, 1)) * distr_target

    for i in range(episodes):
        for batch in batches:
            means, stds, kweights = gmm_approx(batch)
            stds.abs_()
            means.squeeze_(dim=2)
            stds.squeeze_(dim=2)
            # gmm_preds = generate_gmm(locs=means, scales=stds, kweights=kweights.unsqueeze(dim=2))
            gmm_preds = generate_gmm(locs=means, scales=stds, kweights=kweights)
            loss_batch = cramer_py_test(pdf_curr=gmm_preds, pdf_target=distr_target, int_l=-3, int_u=13, spacing=0.001,
                                        dev=device)
            optimizer.zero_grad()
            loss_batch.mean().backward()
            optimizer.step()
        print(f'Episode {i} finished')

    means_trained, stds_trained, kweights_trained = gmm_approx(batches[0])
    means_trained.squeeze_(dim=2)
    stds_trained.squeeze_(dim=2)
    print(f'Means after training: {means_trained},\nStds after training:{stds_trained},'
          f'\nKweights after training: {kweights_trained}\n')
    print(f'Last measured batch loss: {loss_batch.mean()}')

    gmm_fin0 = generate_gmm(means_trained, stds_trained.abs(), kweights_trained)
    print(f'Samples from GMM: {gmm_fin0.sample()}')




