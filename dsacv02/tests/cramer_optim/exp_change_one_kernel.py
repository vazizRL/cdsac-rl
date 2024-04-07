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
from dsacv02.gmm_reparameterization.normal_stable import NormalStable
from copy import deepcopy

# torch.autograd.set_detect_anomaly(True)


def MSE_1k(pdf_target: torch.tensor, pdf_curr: torch.tensor, n_supp, dev='cpu'):
    """
    - Test Method calculating the MSE
    """
    # Dynamically Determine Supports for Current + Target
    batch_size = pdf_target.component_distribution.loc.shape[0]
    difference = pdf_curr.component_distribution.loc - pdf_target.component_distribution.loc
    difference_mean = difference**2 / batch_size
    mse = difference_mean.sum()

    return mse


def cramer_1k(pdf_target: torch.tensor, pdf_curr: torch.tensor, n_supp, dev='cpu'):
    """
    - Dynamic Supports Cramer For 1 Kernel
    - Batch-wise
    - int_l \approx \mu - 3.1*\sigma; int_u \approx \mu + 3.1*\sigma
    - Padding in method cdf() of RMM is deactivate, do not add additional dimension to dx
    - Implementation:
        1. Define the supports with constant n_steps
        2. Calculate the difference squared
        3. Integrate over all dx
    """
    # Meta parameters
    steps_idx = torch.arange(start=1, end=n_supp + 1, step=1).to(dev)
    n_supp = torch.tensor(n_supp, device=dev)
    # Dynamically Determine Supports for Current + Target
    int_l_curr = pdf_curr.component_distribution.loc - 10 * pdf_curr.component_distribution.scale
    int_u_curr = pdf_curr.component_distribution.loc + 10 * pdf_curr.component_distribution.scale
    int_l_tar = pdf_target.component_distribution.loc - 10 * pdf_target.component_distribution.scale
    int_u_tar = pdf_target.component_distribution.loc + 10 * pdf_target.component_distribution.scale

    # Diff Current + Target
    diff_curr = torch.abs(int_u_curr - int_l_curr)
    delta_mb_curr = diff_curr / n_supp
    # delta_mb_curr.unsqueeze_(dim=1).unsqueeze_(dim=2)
    delta_mb_curr.unsqueeze_(dim=1)
    diff_tar = torch.abs(int_u_tar - int_l_tar)
    delta_mb_tar = diff_tar / n_supp
    delta_mb_tar.unsqueeze_(dim=1)

    # Calculate \Delta x for all supports
    dx_mb_diff_curr = steps_idx * delta_mb_curr
    dx_mb_diff_tar = steps_idx * delta_mb_tar

    # Calculate Supports for Current + Target and Concatenate
    dx_mb_curr = torch.ones((1, n_supp), device=dev) * int_l_curr.unsqueeze(dim=1) + dx_mb_diff_curr
    dx_mb_tar = torch.ones((1, n_supp), device=dev) * int_l_tar.unsqueeze(dim=1) + dx_mb_diff_tar
    dx_mb_singular = torch.cat((dx_mb_curr, dx_mb_tar), dim=1)
    dx_mb_singular, _ = dx_mb_singular.sort()

    dy_curr_mb = pdf_curr.cdf(dx_mb_singular)
    dy_target_mb = pdf_target.cdf(dx_mb_singular)

    cramer_re = torch.trapz(y=(dy_target_mb - dy_curr_mb) ** 2, x=dx_mb_singular) + 1e-55
    cramer_re.sqrt_()
    # cramer_re = cramer_re.sqrt()
    cramer_re = cramer_re.mean()

    return cramer_re


def generate_gmm(locs: torch.tensor, scales: torch.tensor, kweights: torch.tensor):
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
    epochs = 10                  # Old: 7
    epochs_low_e = 12
    epochs_high_e = 12
    mb_size = 20
    # Number of Supports per Kernel
    n_eval_points = 30
    graph_spacing = 0.01
    graph_l = -60
    graph_u = 75

    # Reference distribution - Low E
    mean_ref_low_e = torch.ones(size=(mb_size,)).to(device) * torch.tensor([0.0], dtype=torch.float64).to(device)
    std_ref_low_e = torch.ones(size=(mb_size,)).to(device) * torch.tensor([1.0], dtype=torch.float64).to(device)
    kweight_ref_low_e = torch.ones(size=(mb_size,)).to(device) * torch.tensor([1.0], dtype=torch.float64).to(device)
    distr_ref_low_e = generate_gmm(locs=mean_ref_low_e, scales=std_ref_low_e, kweights=kweight_ref_low_e)

    # Reference distribution - High E
    mean_ref_high_e = torch.ones(size=(mb_size,)).to(device) * torch.tensor([0.0], dtype=torch.float64).to(device)
    std_ref_high_e = torch.ones(size=(mb_size,)).to(device) * torch.tensor([15.0], dtype=torch.float64).to(device)
    kweight_ref_high_e = torch.ones(size=(mb_size,)).to(device) * torch.tensor([1.0], dtype=torch.float64).to(device)
    distr_ref_high_e = generate_gmm(locs=mean_ref_high_e, scales=std_ref_high_e, kweights=kweight_ref_high_e)

    # High-entropy Distribution Target, Standard: [2.0, 15.0] with 3 STD
    mean_high_e = torch.ones(size=(mb_size,)).to(device) * torch.tensor([15.0], dtype=torch.float64).to(device)
    std_high_e = torch.ones(size=(mb_size,)).to(device) * torch.tensor([15.0], dtype=torch.float64).to(device)
    kweight_high_e = torch.ones(size=(mb_size,)).to(device) * torch.tensor([1.0], dtype=torch.float64).to(device)
    distr_high_e = generate_gmm(locs=mean_high_e, scales=std_high_e, kweights=kweight_high_e)

    # Low-entropy Distribution, Standard: [2.0, 15.0] with 1 STD
    mean_low_e = torch.ones(size=(mb_size,)).to(device) * torch.tensor([15.0], dtype=torch.float64).to(device)
    std_low_e = torch.ones(size=(mb_size,)).to(device) * torch.tensor([1.0], dtype=torch.float64).to(device)
    kweight_low_e = torch.ones(size=(mb_size,)).to(device) * torch.tensor([1.0], dtype=torch.float64).to(device)
    distr_low_e = generate_gmm(locs=mean_low_e, scales=std_low_e, kweights=kweight_low_e)

    # Network parameters
    arch = (1, 256, 256, 1)
    activ = ('gelu', 'gelu')
    n_kernels = 1
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
    start_ref_low = time.perf_counter()
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
            cramer_py_loss_ref_low_e = cramer_1k(pdf_target=distr_ref_low_e,
                                                 pdf_curr=pred_gmm_ref_low_e,
                                                 n_supp=n_eval_points, dev=device)

            # Loss in MB-GD for cramer_py_loss_i
            optimizer_ref_low_e.zero_grad()
            cramer_py_loss_ref_low_e.backward()
            optimizer_ref_low_e.step()
        print(f'Finished episode for ref low_e: {epoch + 1}')
    end_ref_low = time.perf_counter()
    time_ref_low = end_ref_low - start_ref_low
    print(f'Optimized Time Ref Low.: {time_ref_low}')

    # Test if trained correctly by probing mean of outputs: pred_means and pred_stds
    preds_means_all, preds_stds_all, _ = gmm_approx_ref_low_e(input_total)
    print(f'Avg. Ref preds means for low_e: {preds_means_all.mean(dim=0)}')
    print(f'Avg. Ref preds stds for low_e: {preds_stds_all.mean(dim=0)}')

    # Evaluate CDFs for: Ref_Low
    cdf_ref_low_e = calculate_cdf(pred_gmm_ref_low_e, supp_l=graph_l, supp_u=graph_u, spacing=graph_spacing,
                                  dev=device)

    ''' 
    Fit High Entropy Reference
    '''
    # Fit gmm_approx to ref high_e
    start_ref_high = time.perf_counter()
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
            cramer_py_loss_ref_high_e = cramer_1k(pdf_target=distr_ref_high_e, pdf_curr=pred_gmm_ref_high_e,
                                                  n_supp=n_eval_points, dev=device)

            # Loss in MB-GD for cramer_py_loss_i
            optimizer_ref_high_e.zero_grad()
            cramer_py_loss_ref_high_e.backward()
            optimizer_ref_high_e.step()
        print(f'Finished episode for ref high_e: {epoch + 1}')
    end_ref_high = time.perf_counter()
    time_ref_high = end_ref_high - start_ref_high
    print(f'Optimized Time Ref High: {time_ref_high}')

    # Test if trained correctly by probing mean of outputs: pred_means and pred_stds
    preds_means_high_all, preds_stds_high_all, _ = gmm_approx_ref_high_e(input_total)
    print(f'Avg. Ref preds means for low_e: {preds_means_high_all.mean(dim=0)}')
    print(f'Avg. Ref preds stds for low_e: {preds_stds_high_all.mean(dim=0)}')
    # Evaluate CDFs for: Ref, Low_H, High_H
    cdf_ref_high_e = calculate_cdf(pred_gmm_ref_high_e, supp_l=graph_l, supp_u=graph_u, spacing=graph_spacing, dev=device)

    cdf_low_target = calculate_cdf(distr_low_e, supp_l=graph_l, supp_u=graph_u, spacing=graph_spacing, dev=device)
    cdf_high_target = calculate_cdf(distr_high_e, supp_l=graph_l, supp_u=graph_u, spacing=graph_spacing, dev=device)

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
    start_low_low = time.perf_counter()
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
            # MB Cramer Loss
            cramer_py_loss_low_low = cramer_1k(pdf_target=distr_low_e, pdf_curr=pred_gmm_low_low,
                                               n_supp=n_eval_points, dev=device)
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
    end_low_low = time.perf_counter()
    time_low_low = end_low_low - start_low_low
    print(f'Optimized Time Low-Low: {time_low_low}')

    '''
    Measure change from high-to-low
    '''
    optimizer_high_low = optim.Adam(gmm_approx_low_e_from_high_ref.parameters(), lr=learning_rate)
    means_history_high_low = torch.tensor([], dtype=torch.float64, device=device)
    stds_history_high_low = torch.tensor([], dtype=torch.float64, device=device)
    dc_history_high_low = torch.tensor([], dtype=torch.float64, device=device)
    kweights_history_high_low = torch.tensor([], dtype=torch.float64, device=device)
    start_high_low = time.perf_counter()
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
            # MB Cramer Loss
            cramer_py_loss_high_low = cramer_1k(pdf_target=distr_low_e, pdf_curr=pred_gmm_high_low,
                                                n_supp=n_eval_points, dev=device)
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
            dc_history_high_low = torch.cat((dc_history_high_low, dc_history_batch_high_low), dim=0)
        print(f'Finished episode high-low: {epoch + 1}')
    end_high_low = time.perf_counter()
    time_high_low = end_high_low - start_high_low
    print(f'Optimized High Low: {time_high_low}')

    '''
    Measure change for from low-to-high
    '''
    optimizer_low_high = optim.Adam(gmm_approx_high_e_from_low_ref.parameters(), lr=learning_rate)
    means_history_low_high = torch.tensor([], dtype=torch.float64, device=device)
    stds_history_low_high = torch.tensor([], dtype=torch.float64, device=device)
    dc_history_low_high = torch.tensor([], dtype=torch.float64, device=device)
    kweights_history_low_high = torch.tensor([], dtype=torch.float64, device=device)
    start_low_high = time.perf_counter()
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

            # MB Cramer Loss
            cramer_py_loss_low_high = cramer_1k(pdf_target=distr_high_e, pdf_curr=pred_gmm_low_high,
                                                n_supp=n_eval_points, dev=device)
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
    end_low_high = time.perf_counter()
    time_low_high = end_low_high - start_low_high
    print(f'Time Low High: {time_low_high}')

    '''
    Measure change from high-to-high
    '''
    optimizer_high_high = optim.Adam(gmm_approx_high_e_from_high_ref.parameters(), lr=learning_rate)
    means_history_high_high = torch.tensor([], dtype=torch.float64, device=device)
    stds_history_high_high = torch.tensor([], dtype=torch.float64, device=device)
    dc_history_high_high = torch.tensor([], dtype=torch.float64, device=device)
    kweights_history_high_high = torch.tensor([], dtype=torch.float64, device=device)
    start_high_high = time.perf_counter()
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

            # Cramer MB Loss
            cramer_py_loss_high_high = cramer_1k(pdf_target=distr_high_e, pdf_curr=pred_gmm_high_high,
                                                 n_supp=n_eval_points, dev=device)

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
    end_high_high = time.perf_counter()
    time_high_high = end_high_high - start_high_high
    print(f'Time High High: {time_high_high}')

    '''
    Calculate CDFs: Retro-Fitted GMM for low and high entropy
    '''
    cdf_low_low_post = calculate_cdf(pred_gmm_low_low, supp_l=graph_l, supp_u=graph_u, spacing=graph_spacing, dev=device)
    cdf_high_low_post = calculate_cdf(pred_gmm_high_low, supp_l=graph_l, supp_u=graph_u, spacing=graph_spacing, dev=device)
    cdf_low_high_post = calculate_cdf(pred_gmm_low_high, supp_l=graph_l, supp_u=graph_u, spacing=graph_spacing, dev=device)
    cdf_high_high_post = calculate_cdf(pred_gmm_high_high, supp_l=graph_l, supp_u=graph_u, spacing=graph_spacing, dev=device)

    '''
    Save Data
    '''
    ts = time.time()
    save_path = curr_path + '/' + str(ts)
    os.mkdir(save_path)

    # Save logs from low-to-low
    torch.save(means_history_low_low, save_path + '/' + 'means_history_low_low')
    torch.save(stds_history_low_low, save_path + '/' + 'stds_history_low_low')
    torch.save(kweights_history_low_low, save_path + '/' + 'kweights_history_low_low')
    torch.save(dc_history_low_low, save_path + '/' + 'dc_history_low_low')
    # Save logs from high-to-low
    torch.save(means_history_high_low, save_path + '/' + 'means_history_high_low')
    torch.save(stds_history_high_low, save_path + '/' + 'stds_history_high_low')
    torch.save(kweights_history_high_low, save_path + '/' + 'kweights_history_high_low')
    torch.save(dc_history_high_low, save_path + '/' + 'dc_history_high_low')

    # Save logs from low-to-high
    torch.save(means_history_low_high, save_path + '/' + 'means_history_low_high')
    torch.save(stds_history_low_high, save_path + '/' + 'stds_history_low_high')
    torch.save(kweights_history_low_high, save_path + '/' + 'kweights_history_low_high')
    torch.save(dc_history_low_high, save_path + '/' + 'dc_history_low_high')
    # Save logs from high-to-high
    torch.save(means_history_high_high, save_path + '/' + 'means_history_high_high')
    torch.save(stds_history_high_high, save_path + '/' + 'stds_history_high_high')
    torch.save(kweights_history_high_high, save_path + '/' + 'kweights_history_high_high')
    torch.save(dc_history_high_high, save_path + '/' + 'dc_history_high_high')

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

    # Save Times
    with open(save_path + '/' + 'Times.txt', 'w') as file:
        file.write('Time Ref Low:     ' + str(time_ref_low))
        file.write('\n')
        file.write('Time Ref High:     ' + str(time_ref_high))
        file.write('\n')
        file.write('Time Low Low:     ' + str(time_low_low))
        file.write('\n')
        file.write('Time High Low:     ' + str(time_high_low))
        file.write('\n')
        file.write('Time Low High:     ' + str(time_low_high))
        file.write('\n')
        file.write('Time High High:     ' + str(time_high_high))


