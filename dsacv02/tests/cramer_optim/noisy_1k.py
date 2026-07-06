"""
- This script serves to analyze a NN-distribution with one kernel fitted on a noisy target
-
"""
import torch
import torch.optim as optim
import torch.distributions as distr
import os
import time
import matplotlib.pyplot as plt
from dsacv02.neural_networks import MLPGMM, MLPGMMWeighted
from copy import deepcopy
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# torch.autograd.set_detect_anomaly(True)


def get_normal_supports(batch_size: int, n_kernels: int, n_supp=30, integral_bound_factor=10, dev='cuda:0'):
    """
    - Calculates all Supports in a Linear Fashion for Normal Gaussian Distribution
    :param batch_size: Number of MBs
    :param n_kernels: NUmber of Kernels in GMM
    :param n_supp: Number of Supports desired
    :param integral_bound_factor: \mathbf{0} \plusminus \mathbf{integral_bound_factor}
    :param dev: GPU/CPU
    :return: Supports for Normal Gaussian; equidistant
    """
    ones = [1] * n_kernels

    normal_means = torch.ones(size=(batch_size, 1), dtype=torch.float64) * torch.zeros(n_kernels, dtype=torch.float64)
    normal_stds = torch.ones(size=(batch_size, 1), dtype=torch.float64) * torch.tensor(ones, dtype=torch.float64)

    steps_idx = torch.arange(start=1, end=n_supp + 1, step=1, dtype=torch.float64)
    n_supp = torch.tensor(n_supp)
    int_l_normal = normal_means - integral_bound_factor * normal_stds
    int_u_normal = normal_means + integral_bound_factor * normal_stds

    # Diff Current + Target
    diff_normal = torch.abs(int_u_normal - int_l_normal)
    delta_mb_normal = diff_normal / n_supp
    delta_mb_normal.unsqueeze_(dim=2)

    # Calculate \Delta x for all supports
    dx_mb_diff_normal = steps_idx * delta_mb_normal

    # Calculate All Supports for Normal Gaussian
    dx_mb_normal = torch.ones((1, n_supp)) * int_l_normal.unsqueeze(dim=2) + dx_mb_diff_normal
    dx_mb_normal = dx_mb_normal.to(dev)

    if n_kernels == 1:
        dx_mb_normal.squeeze_(dim=1)

    return dx_mb_normal


def cramer_optim_1k(pdf_target: torch.tensor, pdf_curr: torch.tensor, standard_supp,  dev='cpu'):
    """
    - Dynamic Supports for 1-Kernel
    - Batch-wise
    - Padding in method cdf() of RMM is deactivate, do not add additional dimension to dx
    - Implementation:
        1. Define the supports with constant n_steps
        2. Calculate the difference squared
        3. Integrate over all dx
    """
    dx_mb_curr = pdf_curr.loc + pdf_curr.scale * standard_supp
    dx_mb_curr = dx_mb_curr.detach()
    dx_mb_tar = pdf_target.loc + pdf_target.scale * standard_supp
    dx_mb_tar = dx_mb_tar.detach()

    dx_mb_singular = torch.cat((dx_mb_curr, dx_mb_tar), dim=1).to(dev)

    dx_singular_sorted, _ = dx_mb_singular.sort()

    dy_curr_mb = pdf_curr.cdf(dx_singular_sorted)
    dy_target_mb = pdf_target.cdf(dx_singular_sorted)

    cramer_re = torch.trapz(y=(dy_target_mb - dy_curr_mb) ** 2, x=dx_singular_sorted) + 1e-55
    cramer_re.sqrt_()
    cramer_re = cramer_re.mean()

    return cramer_re


def generate_norm_distr(locs: torch.tensor, scales: torch.tensor):
    normal_distr = distr.Normal(loc=locs, scale=scales)

    return normal_distr


if __name__ == '__main__':
    curr_path = os.getcwd()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Train parameters
    learning_rate = 0.001
    epochs = 7                  # Old: 7
    epochs_high = 5           # Old: 10
    epochs_high_e = 5          # Old: 10
    mb_size = 50

    # Generate noise distributions
    epsilon_std_l = 1
    epsilon_std_h = 5
    locs = torch.ones(size=(mb_size,)).to(device) * torch.tensor([0.0], dtype=torch.float64).to(device)
    stds_low = torch.ones(size=(mb_size,)).to(device) * torch.tensor([epsilon_std_l], dtype=torch.float64).to(device)
    stds_high = torch.ones(size=(mb_size,)).to(device) * torch.tensor([epsilon_std_h], dtype=torch.float64).to(device)
    noise_distr_l = generate_norm_distr(locs=locs, scales=stds_low)
    noise_distr_h = generate_norm_distr(locs=locs, scales=stds_high)

    # Network parameters
    arch = (1, 256, 256, 1)
    activ = ('relu', 'relu')
    n_kernels = 1
    multivar = False
    learnable_weights = False
    # Number of Supports per Kernel
    n_eval_points = 30
    ibf = 10
    cramer_supports = get_normal_supports(batch_size=mb_size, n_kernels=n_kernels, n_supp=n_eval_points,
                                          integral_bound_factor=ibf, dev=device)
    graph_spacing = 0.01
    graph_l = -40
    graph_u = 85

    # Reference distribution
    mean_ref = torch.ones(size=(mb_size,)).to(device) * torch.tensor([0.0], dtype=torch.float64).to(device)
    mean_ref.unsqueeze_(dim=1)
    std_ref = torch.ones(size=(mb_size,)).to(device) * torch.tensor([1.0], dtype=torch.float64).to(device)
    std_ref.unsqueeze_(dim=1)
    distr_ref = generate_norm_distr(locs=mean_ref, scales=std_ref)

    # Target Distribution
    mean_tar = 12
    std_tar = 1.0

    # Gauss Supports for Graphing Result
    dx = torch.arange(-8, 32, 0.01, device=device)

    # Initialize network and optimizer
    if learnable_weights:
        norm_approx_ref = \
            MLPGMMWeighted(arch=arch, activ=activ, n_kernels=n_kernels, device=device, multivar=multivar)
    else:
        norm_approx_ref = MLPGMM(arch=arch, activ=activ, n_kernels=n_kernels, device=device, multivar=multivar)
    optimizer_ref = optim.Adam(norm_approx_ref.parameters(), lr=learning_rate)

    # Initialize input
    n_datapoints = 6000
    n_mb = int(n_datapoints / mb_size)
    input_total = torch.randn(size=(n_datapoints, 1)).to(device)
    batches = input_total.view(n_mb, mb_size, 1)


    '''
    Fit Reference 
    '''
    # Fit gmm_approx to ref
    start_ref = time.perf_counter()
    for epoch in range(epochs):
        for batch in batches:
            pred_means_ref, pred_stds_ref, kweights_ref = norm_approx_ref(batch)
            pred_stds_ref.abs_()
            pred_means_ref.squeeze_(dim=2)
            pred_stds_ref.squeeze_(dim=2)
            pred_norm_ref = generate_norm_distr(locs=pred_means_ref,
                                                scales=pred_stds_ref,
                                                )
            # Loss batch_i
            cramer_py_loss_ref = cramer_optim_1k(pdf_target=distr_ref, pdf_curr=pred_norm_ref,
                                                 standard_supp=cramer_supports, dev=device)

            # Loss in MB-GD for cramer_py_loss_i
            optimizer_ref.zero_grad()
            cramer_py_loss_ref.backward()
            optimizer_ref.step()
        print(f'Finished episode for ref: {epoch + 1}')
    end_ref = time.perf_counter()
    time_ref = end_ref - start_ref
    print(f'Optimized Time Ref Low.: {time_ref}')

    # Test if trained correctly by probing mean of outputs: pred_means and pred_stds
    preds_means_all, preds_stds_all, _ = norm_approx_ref(input_total)
    print(f'Avg. Ref preds means for low_e: {preds_means_all.mean(dim=0)}')
    print(f'Avg. Ref preds stds for low_e: {preds_stds_all.mean(dim=0)}')

    '''
    Deep copy the trained network (trained for means=[1.0, 0.0], stds=[1.0, 1.0])
    '''
    norm_approx_to_l = deepcopy(norm_approx_ref)
    norm_approx_to_h = deepcopy(norm_approx_ref)

    '''
    Fit to low noise target
    '''
    optimizer_low = optim.Adam(norm_approx_to_l.parameters(), lr=learning_rate)
    means_history_low = torch.tensor([], dtype=torch.float64, device=device)
    stds_history_low = torch.tensor([], dtype=torch.float64, device=device)
    dc_history_low = torch.tensor([], dtype=torch.float64, device=device)
    kweights_history_low = torch.tensor([], dtype=torch.float64, device=device)
    start_low = time.perf_counter()
    for epoch in range(epochs_high):
        for batch in batches:
            means_history_batch_low = torch.tensor([], dtype=torch.float64, device=device)
            stds_history_batch_low = torch.tensor([], dtype=torch.float64, device=device)
            dc_history_batch_low = torch.tensor([], dtype=torch.float64, device=device)
            kweights_history_batch_low = torch.tensor([], dtype=torch.float64, device=device)

            # kweights = None
            pred_means_low, pred_stds_low, kweights_low_low = norm_approx_to_l(batch)
            pred_stds_low.abs_()
            pred_means_low.squeeze_(dim=2)
            pred_stds_low.squeeze_(dim=2)
            # Log preds
            means_history_batch_low = torch.cat((means_history_batch_low, pred_means_low), dim=0)
            stds_history_batch_low = torch.cat((stds_history_batch_low, pred_stds_low), dim=0)
            kweights_history_batch_low = torch.cat((kweights_history_batch_low, kweights_low_low), dim=0)

            pred_norm_low = generate_norm_distr(locs=pred_means_low,
                                                scales=pred_stds_low,
                                                )
            # Generate noisy target
            mean_tar_noisy_l = torch.ones(size=(mb_size,)).to(device) * torch.tensor([mean_tar], dtype=torch.float64).to(device) \
                + noise_distr_l.sample()
            mean_tar_noisy_l.unsqueeze_(dim=1)
            std_tar_noisy_l = torch.ones(size=(mb_size,)).to(device) * torch.tensor([std_tar], dtype=torch.float64).to(device)
            std_tar_noisy_l.unsqueeze_(dim=1)
            distr_tar_noisy_l = generate_norm_distr(locs=mean_tar_noisy_l, scales=std_tar_noisy_l)
            # MB Cramer Loss
            cramer_py_loss_low_low = cramer_optim_1k(pdf_target=distr_tar_noisy_l, pdf_curr=pred_norm_low,
                                                     standard_supp=cramer_supports, dev=device)
            # Log loss
            dc_history_batch_low = torch.cat((dc_history_batch_low,
                                              cramer_py_loss_low_low.unsqueeze(dim=0).unsqueeze(dim=1)), dim=0)

            optimizer_low.zero_grad()
            cramer_py_loss_low_low.backward()
            optimizer_low.step()

            # Log mean and std change
            means_history_low = torch.cat((means_history_low, means_history_batch_low), dim=0)
            stds_history_low = torch.cat((stds_history_low, stds_history_batch_low), dim=0)
            kweights_history_low = torch.cat((kweights_history_low, kweights_history_batch_low),
                                             dim=0)
            dc_history_low = torch.cat((dc_history_low, dc_history_batch_low), dim=0)
        print(f'Finished episode low-low: {epoch + 1}')
    end_low_low = time.perf_counter()
    time_low_low = end_low_low - start_low
    print(f'Optimized Time Low-Low: {time_low_low}')

    '''
    Fit to high noise targets
    '''
    optimizer_high = optim.Adam(norm_approx_to_h.parameters(), lr=learning_rate)
    means_history_high = torch.tensor([], dtype=torch.float64, device=device)
    stds_history_high = torch.tensor([], dtype=torch.float64, device=device)
    dc_history_high = torch.tensor([], dtype=torch.float64, device=device)
    kweights_history_high = torch.tensor([], dtype=torch.float64, device=device)
    start_high = time.perf_counter()
    for epoch in range(epochs_high):
        for batch in batches:
            means_history_batch_high = torch.tensor([], dtype=torch.float64, device=device)
            stds_history_batch_high = torch.tensor([], dtype=torch.float64, device=device)
            dc_history_batch_high = torch.tensor([], dtype=torch.float64, device=device)
            kweights_history_batch_high = torch.tensor([], dtype=torch.float64, device=device)

            # kweights = None
            pred_means_high, pred_stds_high, kweights_high = norm_approx_to_h(batch)
            pred_stds_high.abs_()
            pred_means_high.squeeze_(dim=2)
            pred_stds_high.squeeze_(dim=2)
            # Log preds
            means_history_batch_high = torch.cat((means_history_batch_high, pred_means_high), dim=0)
            stds_history_batch_high = torch.cat((stds_history_batch_high, pred_stds_high), dim=0)
            kweights_history_batch_high = torch.cat((kweights_history_batch_high, kweights_high), dim=0)

            pred_norm_high_low = generate_norm_distr(locs=pred_means_high,
                                                     scales=pred_stds_high,
                                                     )
            # Generate noisy target
            mean_tar_noisy_h = torch.ones(size=(mb_size,)).to(device) * torch.tensor([mean_tar],
                                          dtype=torch.float64).to(device) + noise_distr_h.sample()
            mean_tar_noisy_h.unsqueeze_(dim=1)
            std_tar_noisy_h = torch.ones(size=(mb_size,)).to(device) * torch.tensor([std_tar], dtype=torch.float64).to(device)
            std_tar_noisy_h.unsqueeze_(dim=1)
            distr_tar_noisy_h = generate_norm_distr(locs=mean_tar_noisy_h, scales=std_tar_noisy_h)
            # MB Cramer Loss
            cramer_py_loss_high = cramer_optim_1k(pdf_target=distr_tar_noisy_h, pdf_curr=pred_norm_high_low,
                                                  standard_supp=cramer_supports, dev=device)
            # Log loss
            dc_history_batch_high = torch.cat((dc_history_batch_high,
                                               cramer_py_loss_high.unsqueeze(dim=0).unsqueeze(dim=1)), dim=0)

            optimizer_high.zero_grad()
            cramer_py_loss_high.backward()
            optimizer_high.step()

            # Log mean and std change
            means_history_high = torch.cat((means_history_high, means_history_batch_high), dim=0)
            stds_history_high = torch.cat((stds_history_high, stds_history_batch_high), dim=0)
            kweights_history_high = torch.cat((kweights_history_high, kweights_history_batch_high),
                                              dim=0)
            dc_history_high = torch.cat((dc_history_high, dc_history_batch_high), dim=0)
        print(f'Finished episode high-low: {epoch + 1}')
    end_high = time.perf_counter()
    time_high = end_high - start_high
    print(f'Optimized High Low: {time_high}')

    ''' 
    Generate Distribution Prediction and Construct PDF
    '''
    mean_ref, std_ref, kw_ref = norm_approx_ref(input_total)
    mean_ref, std_ref = mean_ref.mean(), std_ref.mean()
    mean_l, std_l, kw_l = norm_approx_to_l(input_total)
    mean_l, std_l = mean_l.mean(), std_l.mean()
    mean_h, std_h, kw_h = norm_approx_to_h(input_total)
    mean_h, std_h = mean_h.mean(), std_h.mean()

    distr_ref = generate_norm_distr(locs=mean_ref, scales=std_ref)
    distr_l = generate_norm_distr(locs=mean_l, scales=std_l)
    distr_h = generate_norm_distr(locs=mean_h, scales=std_h)

    probs_ref = distr_ref.log_prob(dx).exp()
    probs_l = distr_l.log_prob(dx).exp()
    probs_h = distr_h.log_prob(dx).exp()

    '''
    Graph PDFs: Ref, Low and High
    '''
    plt.rcParams['figure.figsize'] = (30, 12)
    plt.plot(dx.cpu().detach().numpy(), probs_ref.cpu().detach().numpy(), label='Fitted on Ref')
    plt.plot(dx.cpu().detach().numpy(), probs_l.cpu().detach().numpy(), label='Ref Fitted on Low Noisy')
    plt.plot(dx.cpu().detach().numpy(), probs_h.cpu().detach().numpy(), label='Ref Fitted on High Noisy')

    plt.xlabel('Supports')
    plt.ylabel('Probabilitiy')
    plt.title(f'Tar: Mu_{mean_tar}, Std_{std_tar} | Ref: Mu_{mean_ref.cpu().item()}, '
              f'Std_{std_ref.cpu().item()} | Noise: L_{epsilon_std_l}, H_{epsilon_std_h}')
    plt.legend()

    '''
    Save Graph and Show It
    '''
    # Create Folder
    ts = time.time()
    save_path = curr_path + '/NoisyTar1k_' + str(ts)
    os.mkdir(save_path)
    # Save Pyplot Figs
    plt.savefig(save_path + '/' + 'NoisyTargets1k.png')
    plt.show(block=True)



