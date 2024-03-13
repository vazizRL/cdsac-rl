"""
Testing effect of updating with high-entrpoy vs. low-entropy target
"""
import torch
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

    dy_curr_cdf = pdf_curr.cdf(dx)
    dy_target_cdf = pdf_target.cdf(dx)

    cramer = torch.trapz((dy_target_cdf - dy_curr_cdf)**2, dx=spacing)
    cramer = cramer.sum(dim=0)

    return cramer**0.5


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

    if pdf_curr.batch_shape.__len__():
        batch_size = pdf_curr.batch_shape[0]
    else:
        batch_size = 1
    dy_curr_cdf_re = pdf_curr.cdf(dx).reshape(batch_size, 1, dx.shape[0])
    dy_target_cdf_re = pdf_target.cdf(dx).reshape(batch_size, 1, dx.shape[0])
    cramer_re = torch.trapz((dy_target_cdf_re - dy_curr_cdf_re)**2, dx=spacing)
    cramer_re.sqrt_()
    cramer_re = cramer_re.mean()

    return cramer_re


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
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Train parameters
    learning_rate = 0.001
    epochs = 7
    epochs_low_e = 10
    epochs_high_e = 10
    mb_size = 5
    spacing = 0.01

    # Boundaries
    int_ref_l = -12
    int_ref_u = 12
    int_low_l = -10
    int_low_u = 25
    int_high_l = -10
    int_high_u = 25

    # Reference distribution - Low E
    mean_ref_low_e = torch.ones(size=(mb_size, 1)).to(device) * torch.tensor([0.0, 1.0], dtype=torch.float64).to(device)
    std_ref_low_e = torch.ones(size=(mb_size, 1)).to(device) * torch.tensor([1.0, 1.0], dtype=torch.float64).to(device)
    kweight_ref_low_e = torch.ones(size=(mb_size, 1)).to(device) * torch.tensor([0.5, 0.5], dtype=torch.float64).to(device)
    distr_ref_low_e = generate_gmm(locs=mean_ref_low_e, scales=std_ref_low_e, kweights=kweight_ref_low_e)

    # Reference distribution - High E
    mean_ref_high_e = torch.ones(size=(mb_size, 1)).to(device) * torch.tensor([0.0, 1.0], dtype=torch.float64).to(device)
    std_ref_high_e = torch.ones(size=(mb_size, 1)).to(device) * torch.tensor([3.0, 3.0], dtype=torch.float64).to(device)
    kweight_ref_high_e = torch.ones(size=(mb_size, 1)).to(device) * torch.tensor([0.5, 0.5], dtype=torch.float64).to(device)
    distr_ref_high_e = generate_gmm(locs=mean_ref_high_e, scales=std_ref_high_e, kweights=kweight_ref_high_e)

    # High-entropy Distribution
    mean_high_e = torch.ones(size=(mb_size, 1)).to(device) * torch.tensor([2.0, 15.0], dtype=torch.float64).to(device)
    std_high_e = torch.ones(size=(mb_size, 1)).to(device) * torch.tensor([3.0, 3.0], dtype=torch.float64).to(device)
    kweight_high_e = torch.ones(size=(mb_size, 1)).to(device) * torch.tensor([0.5, 0.5], dtype=torch.float64).to(device)
    distr_high_e = generate_gmm(locs=mean_high_e, scales=std_high_e, kweights=kweight_high_e)

    # Low-entropy Distribution
    mean_low_e = torch.ones(size=(mb_size, 1)).to(device) * torch.tensor([2.0, 15.0], dtype=torch.float64).to(device)
    std_low_e = torch.ones(size=(mb_size, 1)).to(device) * torch.tensor([1.0, 1.0], dtype=torch.float64).to(device)
    kweight_low_e = torch.ones(size=(mb_size, 1)).to(device) * torch.tensor([0.5, 0.5], dtype=torch.float64).to(device)
    distr_low_e = generate_gmm(locs=mean_low_e, scales=std_low_e, kweights=kweight_low_e)

    # Network parameters
    arch = (1, 10, 1)
    activ = ('gelu',)
    n_kernels = 2
    multivar = True
    learnable_weights = True
    kweights_fix = torch.ones(n_kernels, dtype=torch.float64) / n_kernels
    kweights_fix.to(device)

    # Initialize network and optimizer
    if learnable_weights:
        gmm_approx_ref_low_e = \
            MLPGMMWeighted(arch=arch, activ=activ, n_kernels=n_kernels, device=device, multivar=multivar)
        gmm_approx_ref_high_e = deepcopy(gmm_approx_ref_low_e)
    else:
        gmm_approx_ref_low_e = MLPGMM(arch=arch, activ=activ, n_kernels=n_kernels, device=device, multivar=multivar)
        gmm_approx_ref_high_e = deepcopy(gmm_approx_ref_low_e)
    optimizer_ref_low_e = optim.Adam(gmm_approx_ref_low_e.parameters(), lr=learning_rate)
    optimizer_ref_high_e = optim.Adam(gmm_approx_ref_high_e.parameters(), lr=learning_rate)

    # Initialize input
    n_datapoints = 6000
    n_mb = int(n_datapoints / mb_size)
    input_total = torch.randn(size=(n_datapoints, 1)).to(device)
    batches = input_total.view(n_mb, mb_size, 1)

    '''
    Fit Low Entropy Reference
    '''
    # Fit gmm_approx to ref low_e
    for epoch in range(epochs):
        for batch in batches:
            pred_means_ref_low_e, pred_stds_ref_low_e, kweights_ref_low_e = gmm_approx_ref_low_e(batch)
            pred_stds_ref_low_e.abs_()
            pred_means_ref_low_e.squeeze_(dim=2)
            pred_stds_ref_low_e.squeeze_(dim=2)
            if learnable_weights:
                pred_gmm_ref_low_e = generate_gmm(locs=pred_means_ref_low_e.squeeze(),
                                                  scales=pred_stds_ref_low_e.squeeze(),
                                                  kweights=kweights_ref_low_e.squeeze())
            else:
                pred_gmm_ref_low_e = generate_gmm(locs=pred_means_ref_low_e.squeeze(),
                                                  scales=pred_stds_ref_low_e.squeeze(),
                                                  kweights=kweights_fix)
            # Loss batch_i
            cramer_py_loss_ref_low_e = cramer_py_test(pdf_target=distr_ref_low_e, pdf_curr=pred_gmm_ref_low_e,
                                                      int_l=int_ref_l, int_u=int_ref_u, spacing=spacing, dev=device)

            # Loss in MB-GD for cramer_py_loss_i
            optimizer_ref_low_e.zero_grad()
            cramer_py_loss_ref_low_e.backward()
            optimizer_ref_low_e.step()
        print(f'Finished episode for ref low_e: {epoch + 1}')

    # Test if trained correctly by probing mean of outputs: pred_means and pred_stds
    preds_means_all, preds_stds_all, _ = gmm_approx_ref_low_e(input_total)
    print(f'Avg. Ref preds means for low_e: {preds_means_all.mean(dim=0)}')
    print(f'Avg. Ref preds stds for low_e: {preds_stds_all.mean(dim=0)}')
    # Evaluate CDFs for: Ref, Low_H, High_H
    cdf_ref_low_e = calculate_cdf(pred_gmm_ref_low_e, supp_l=int_ref_l, supp_u=int_ref_u, spacing=spacing, dev=device)

    ''' 
    Fit High Entropy Reference
    '''
    # Fit gmm_approx to ref high_e
    for epoch in range(epochs):
        for batch in batches:
            pred_means_ref_high_e, pred_stds_ref_high_e, kweights_ref_high_e = gmm_approx_ref_high_e(batch)
            pred_stds_ref_high_e.abs_()
            pred_means_ref_high_e.squeeze_(dim=2)
            pred_stds_ref_high_e.squeeze_(dim=2)
            if learnable_weights:
                pred_gmm_ref_high_e = generate_gmm(locs=pred_means_ref_high_e.squeeze(),
                                                   scales=pred_stds_ref_high_e.squeeze(),
                                                   kweights=kweights_ref_high_e.squeeze())
            else:
                pred_gmm_ref_high_e = generate_gmm(locs=pred_means_ref_high_e.squeeze(),
                                                   scales=pred_stds_ref_high_e.squeeze(),
                                                   kweights=kweights_fix)
            # Loss batch_i
            cramer_py_loss_ref_high_e = cramer_py_test(pdf_target=distr_ref_high_e, pdf_curr=pred_gmm_ref_high_e,
                                                       int_l=int_ref_l, int_u=int_ref_u, spacing=spacing, dev=device)

            # Loss in MB-GD for cramer_py_loss_i
            optimizer_ref_high_e.zero_grad()
            cramer_py_loss_ref_high_e.backward()
            optimizer_ref_high_e.step()
        print(f'Finished episode for ref high_e: {epoch + 1}')

    # Test if trained correctly by probing mean of outputs: pred_means and pred_stds
    preds_means_high_all, preds_stds_high_all, _ = gmm_approx_ref_high_e(input_total)
    print(f'Avg. Ref preds means for low_e: {preds_means_high_all.mean(dim=0)}')
    print(f'Avg. Ref preds stds for low_e: {preds_stds_high_all.mean(dim=0)}')
    # Evaluate CDFs for: Ref, Low_H, High_H
    cdf_ref_high_e = calculate_cdf(pred_gmm_ref_high_e, supp_l=int_ref_l, supp_u=int_ref_u, spacing=spacing, dev=device)

    cdf_low_target = calculate_cdf(distr_low_e, supp_l=int_low_l, supp_u=int_low_u, spacing=spacing, dev=device)
    cdf_high_target = calculate_cdf(distr_high_e, supp_l=int_high_l, supp_u=int_high_u, spacing=spacing, dev=device)

    '''
    Deep copy the trained network (trained for means=[1.0, 0.0], stds=[1.0, 1.0])
    '''
    gmm_approx_low_e_from_low_ref = deepcopy(gmm_approx_ref_low_e)
    gmm_approx_low_e_from_high_ref = deepcopy(gmm_approx_ref_high_e)

    gmm_approx_high_e_from_low_ref = deepcopy(gmm_approx_ref_low_e)
    gmm_approx_high_e_from_high_ref = deepcopy(gmm_approx_ref_high_e)

    '''
    Measure change from low-to-low
    '''
    optimizer_low_low = optim.Adam(gmm_approx_low_e_from_low_ref.parameters(), lr=learning_rate)
    means_history_low_low = torch.tensor([], dtype=torch.float64, device=device)
    stds_history_low_low = torch.tensor([], dtype=torch.float64, device=device)
    dc_history_low_low = torch.tensor([], dtype=torch.float64, device=device)
    kweights_history_low_low = torch.tensor([], dtype=torch.float64, device=device)
    for epoch in range(epochs_low_e):
        for batch in batches:
            means_history_batch_low_low = torch.tensor([], dtype=torch.float64, device=device)
            stds_history_batch_low_low = torch.tensor([], dtype=torch.float64, device=device)
            dc_history_batch_low_low = torch.tensor([], dtype=torch.float64, device=device)
            kweights_history_batch_low_low = torch.tensor([], dtype=torch.float64, device=device)

            # kweights = None
            pred_means_low_low, pred_stds_low_low, kweights_low_low = gmm_approx_low_e_from_low_ref(batch)
            pred_stds_low_low.abs_()
            pred_means_low_low.squeeze_(dim=2)
            pred_stds_low_low.squeeze_(dim=2)
            # Log preds
            means_history_batch_low_low = torch.cat((means_history_batch_low_low, pred_means_low_low), dim=0)
            stds_history_batch_low_low = torch.cat((stds_history_batch_low_low, pred_stds_low_low), dim=0)
            kweights_history_batch_low_low = torch.cat((kweights_history_batch_low_low, kweights_low_low), dim=0)

            if learnable_weights:
                pred_gmm_low_low = generate_gmm(locs=pred_means_low_low.squeeze(), scales=pred_stds_low_low.squeeze(),
                                                kweights=kweights_low_low.squeeze())
            else:
                pred_gmm_low_low = generate_gmm(locs=pred_means_low_low.squeeze(), scales=pred_stds_low_low.squeeze(),
                                                kweights=kweights_fix)
            cramer_py_loss_low_low = cramer_py_test(pdf_target=distr_low_e, pdf_curr=pred_gmm_low_low, int_l=int_low_l,
                                                    int_u=int_low_u, spacing=spacing, dev=device)
            # Log loss
            dc_history_batch_low_low = torch.cat((dc_history_batch_low_low,
                                                  cramer_py_loss_low_low.unsqueeze(dim=0).unsqueeze(dim=1)), dim=0)

            optimizer_low_low.zero_grad()
            cramer_py_loss_low_low.backward()
            optimizer_low_low.step()

            # Log mean and std change
            means_history_low_low = torch.cat((means_history_low_low, means_history_batch_low_low), dim=0)
            stds_history_low_low = torch.cat((stds_history_low_low, stds_history_batch_low_low), dim=0)
            kweights_history_low_low = torch.cat((kweights_history_low_low, kweights_history_batch_low_low),
                                                 dim=0)
            dc_history_low_low = torch.cat((dc_history_low_low, dc_history_batch_low_low), dim=0)
        print(f'Finished episode low-low: {epoch + 1}')

    '''
    Measure change from high-to-low
    '''
    optimizer_high_low = optim.Adam(gmm_approx_low_e_from_high_ref.parameters(), lr=learning_rate)
    means_history_high_low = torch.tensor([], dtype=torch.float64, device=device)
    stds_history_high_low = torch.tensor([], dtype=torch.float64, device=device)
    dc_history_high_low = torch.tensor([], dtype=torch.float64, device=device)
    kweights_history_high_low = torch.tensor([], dtype=torch.float64, device=device)
    for epoch in range(epochs_low_e):
        for batch in batches:
            means_history_batch_high_low = torch.tensor([], dtype=torch.float64, device=device)
            stds_history_batch_high_low = torch.tensor([], dtype=torch.float64, device=device)
            dc_history_batch_high_low = torch.tensor([], dtype=torch.float64, device=device)
            kweights_history_batch_high_low = torch.tensor([], dtype=torch.float64, device=device)

            # kweights = None
            pred_means_high_low, pred_stds_high_low, kweights_high_low = gmm_approx_low_e_from_high_ref(batch)
            pred_stds_high_low.abs_()
            pred_means_high_low.squeeze_(dim=2)
            pred_stds_high_low.squeeze_(dim=2)
            # Log preds
            means_history_batch_high_low = torch.cat((means_history_batch_high_low, pred_means_high_low), dim=0)
            stds_history_batch_high_low = torch.cat((stds_history_batch_high_low, pred_stds_high_low), dim=0)
            kweights_history_batch_high_low = torch.cat((kweights_history_batch_high_low, kweights_high_low), dim=0)

            if learnable_weights:
                pred_gmm_high_low = generate_gmm(locs=pred_means_high_low.squeeze(),
                                                 scales=pred_stds_high_low.squeeze(),
                                                 kweights=kweights_high_low.squeeze())
            else:
                pred_gmm_high_low = generate_gmm(locs=pred_means_high_low.squeeze(),
                                                 scales=pred_stds_high_low.squeeze(),
                                                 kweights=kweights_fix)
            cramer_py_loss_high_low = cramer_py_test(pdf_target=distr_low_e, pdf_curr=pred_gmm_high_low,
                                                     int_l=int_low_l, int_u=int_low_u, spacing=spacing, dev=device)
            # Log loss
            dc_history_batch_high_low = torch.cat((dc_history_batch_high_low,
                                                  cramer_py_loss_high_low.unsqueeze(dim=0).unsqueeze(dim=1)), dim=0)

            optimizer_high_low.zero_grad()
            cramer_py_loss_high_low.backward()
            optimizer_high_low.step()

            # Log mean and std change
            means_history_high_low = torch.cat((means_history_high_low, means_history_batch_high_low), dim=0)
            stds_history_high_low = torch.cat((stds_history_high_low, stds_history_batch_high_low), dim=0)
            kweights_history_high_low = torch.cat((kweights_history_high_low, kweights_history_batch_high_low),
                                                 dim=0)
            dc_history_low_low = torch.cat((dc_history_low_low, dc_history_batch_high_low), dim=0)
        print(f'Finished episode high-low: {epoch + 1}')

    '''
    Measure change for from low-to-high
    '''
    optimizer_low_high = optim.Adam(gmm_approx_high_e_from_low_ref.parameters(), lr=learning_rate)
    means_history_low_high = torch.tensor([], dtype=torch.float64, device=device)
    stds_history_low_high = torch.tensor([], dtype=torch.float64, device=device)
    dc_history_low_high = torch.tensor([], dtype=torch.float64, device=device)
    kweights_history_low_high = torch.tensor([], dtype=torch.float64, device=device)
    for epoch in range(epochs_high_e):
        for batch in batches:
            means_history_batch_low_high = torch.tensor([], dtype=torch.float64, device=device)
            stds_history_batch_low_high = torch.tensor([], dtype=torch.float64, device=device)
            dc_history_batch_low_high = torch.tensor([], dtype=torch.float64, device=device)
            kweights_history_batch_low_high = torch.tensor([], dtype=torch.float64, device=device)

            pred_means_low_high, pred_stds_low_high, kweights_low_high = gmm_approx_high_e_from_low_ref(batch)
            pred_stds_low_high.abs_()
            pred_means_low_high.squeeze_(dim=2)
            pred_stds_low_high.squeeze_(dim=2)
            # Log preds
            means_history_batch_low_high = torch.cat((means_history_batch_low_high, pred_means_low_high), dim=0)
            stds_history_batch_low_high = torch.cat((stds_history_batch_low_high, pred_stds_low_high), dim=0)
            kweights_history_batch_low_high = torch.cat((kweights_history_batch_low_high, kweights_low_high), dim=0)

            if learnable_weights:
                pred_gmm_low_high = generate_gmm(locs=pred_means_low_high.squeeze(),
                                                 scales=pred_stds_low_high.squeeze(),
                                                 kweights=kweights_low_high.squeeze())
            else:
                pred_gmm_low_high = generate_gmm(locs=pred_means_low_high.squeeze(),
                                                 scales=pred_stds_low_high.squeeze(),
                                                 kweights=kweights_fix)

            cramer_py_loss_low_high = cramer_py_test(pdf_target=distr_high_e, pdf_curr=pred_gmm_low_high,
                                                     int_l=int_high_l, int_u=int_high_u, spacing=spacing, dev=device)
            # Log loss
            dc_history_batch_low_high = torch.cat((dc_history_batch_low_high,
                                                   cramer_py_loss_low_high.unsqueeze(dim=0).unsqueeze(dim=1)), dim=0)

            optimizer_low_high.zero_grad()
            cramer_py_loss_low_high.backward()
            optimizer_low_high.step()

            # Log mean and std change
            means_history_low_high = torch.cat((means_history_low_high, means_history_batch_low_high), dim=0)
            stds_history_low_high = torch.cat((stds_history_low_high, stds_history_batch_low_high), dim=0)
            dc_history_low_high = torch.cat((dc_history_low_high, dc_history_batch_low_high), dim=0)
            kweights_history_low_high = torch.cat((kweights_history_low_high, kweights_history_batch_low_high),
                                                  dim=0)
        print(f'Finished episode low-high: {epoch + 1}')

    '''
    Measure change from high-to-high
    '''
    optimizer_high_high = optim.Adam(gmm_approx_high_e_from_high_ref.parameters(), lr=learning_rate)
    means_history_high_high = torch.tensor([], dtype=torch.float64, device=device)
    stds_history_high_high = torch.tensor([], dtype=torch.float64, device=device)
    dc_history_high_high = torch.tensor([], dtype=torch.float64, device=device)
    kweights_history_high_high = torch.tensor([], dtype=torch.float64, device=device)
    for epoch in range(epochs_high_e):
        for batch in batches:
            means_history_batch_high_high = torch.tensor([], dtype=torch.float64, device=device)
            stds_history_batch_high_high = torch.tensor([], dtype=torch.float64, device=device)
            dc_history_batch_high_high = torch.tensor([], dtype=torch.float64, device=device)
            kweights_history_batch_high_high = torch.tensor([], dtype=torch.float64, device=device)

            pred_means_high_high, pred_stds_high_high, kweights_high_high = gmm_approx_high_e_from_high_ref(batch)
            pred_stds_high_high.abs_()
            pred_means_high_high.squeeze_(dim=2)
            pred_stds_high_high.squeeze_(dim=2)
            # Log preds
            means_history_batch_high_high = torch.cat((means_history_batch_high_high, pred_means_high_high), dim=0)
            stds_history_batch_high_high = torch.cat((stds_history_batch_high_high, pred_stds_high_high), dim=0)
            kweights_history_batch_high_high = torch.cat((kweights_history_batch_high_high, kweights_high_high), dim=0)

            if learnable_weights:
                pred_gmm_high_high = generate_gmm(locs=pred_means_high_high.squeeze(),
                                                  scales=pred_stds_high_high.squeeze(),
                                                  kweights=kweights_high_high.squeeze())
            else:
                pred_gmm_high_high = generate_gmm(locs=pred_means_high_high.squeeze(),
                                                  scales=pred_stds_high_high.squeeze(),
                                                  kweights=kweights_fix)

            cramer_py_loss_high_high = cramer_py_test(pdf_target=distr_high_e, pdf_curr=pred_gmm_high_high,
                                                      int_l=int_high_l, int_u=int_high_u, spacing=spacing, dev=device)
            # Log loss
            dc_history_batch_high_high = torch.cat((dc_history_batch_high_high,
                                                    cramer_py_loss_high_high.unsqueeze(dim=0).unsqueeze(dim=1)), dim=0)

            optimizer_high_high.zero_grad()
            cramer_py_loss_high_high.backward()
            optimizer_high_high.step()

            # Log mean and std change
            means_history_high_high = torch.cat((means_history_high_high, means_history_batch_high_high), dim=0)
            stds_history_high_high = torch.cat((stds_history_high_high, stds_history_batch_high_high), dim=0)
            dc_history_high_high = torch.cat((dc_history_high_high, dc_history_batch_high_high), dim=0)
            kweights_history_high_high = torch.cat((kweights_history_high_high, kweights_history_batch_high_high),
                                                   dim=0)
        print(f'Finished episode high-high: {epoch + 1}')

    '''
    Calculate CDFs: Retro-Fitted GMM for low and high entropy
    '''
    cdf_low_low_post = calculate_cdf(pred_gmm_low_low, supp_l=int_low_l, supp_u=int_low_u, spacing=spacing, dev=device)
    cdf_high_low_post = calculate_cdf(pred_gmm_high_low, supp_l=int_low_l, supp_u=int_low_u,
                                      spacing=spacing, dev=device)
    cdf_low_high_post = calculate_cdf(pred_gmm_low_high, supp_l=int_low_l, supp_u=int_low_u,
                                      spacing=spacing, dev=device)
    cdf_high_high_post = calculate_cdf(pred_gmm_high_high, supp_l=int_high_l, supp_u=int_high_u,
                                       spacing=spacing, dev=device)

    '''
    Save Data
    '''
    ts = time.time()
    save_path = curr_path + '/' + str(ts)
    os.mkdir(save_path)

    # Save the logged tensor
    torch.save(means_history_low_low, save_path + '/' + 'means_history_low')
    torch.save(stds_history_low_low, save_path + '/' + 'stds_history_low')
    torch.save(kweights_history_low_low, save_path + '/' + 'kweights_history_low')
    torch.save(dc_history_low_low, save_path + '/' + 'dc_history_low')

    torch.save(means_history_high_high, save_path + '/' + 'means_history_high')
    torch.save(stds_history_high_high, save_path + '/' + 'stds_history_high')
    torch.save(kweights_history_high_high, save_path + '/' + 'kweights_history_high')
    torch.save(dc_history_high_high, save_path + '/' + 'dc_history_high')

    # Save Ref CDFs
    torch.save(cdf_ref_low_e, save_path + '/' + 'cdf_ref_low')
    torch.save(cdf_ref_high_e, save_path + '/' + 'cdf_ref_high')

    # Save Target CDFs
    torch.save(cdf_low_target, save_path + '/' + 'cdf_low_target')
    torch.save(cdf_high_target, save_path + '/' + 'cdf_high_target')

    # Save post-training CDFs
    torch.save(cdf_low_low_post, save_path + '/' + 'cdf_low_low_post')
    torch.save(cdf_low_high_post, save_path + '/' + 'cdf_low_high_post')
    torch.save(cdf_high_low_post, save_path + '/' + 'cdf_high_low_post')
    torch.save(cdf_high_high_post, save_path + '/' + 'cdf_high_high_post')



