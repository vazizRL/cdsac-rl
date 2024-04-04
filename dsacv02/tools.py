"""
Collections of tools to be sused in Z-DSAC
"""
import torch
import numpy as np
from scipy import integrate
from typing import List


def cramer_from_pdf(pdf_target: torch.tensor, pdf_curr: torch.tensor, int_l, int_u, points=(-100, 100)):
    """
    - Only for testing purposes
    - Calculates the
    - The integration limits should NOT be too far off form the lowest and highest point of the function!
    :param pdf_target: Probability density function of target distribution
    :param pdf_curr: Probability density function of reference distribution
    :param int_l: Lower integral bound
    :param int_u: Upper integral bound
    :param points: Points to focus on, if not, then points=(int_l, int_u)
    :return: Returns distance and error estimation of the calculated distance
    """
    distance, error_est = integrate.quad(
        lambda x: (pdf_target.cdf(torch.tensor([x])).numpy() - pdf_curr.cdf(torch.tensor([x])).numpy()) ** 2,
        int_l, int_u, points=points
    )

    return distance, error_est


def cramer_torch(pdf_target: torch.tensor, pdf_curr: torch.tensor, int_l, int_u, spacing, dev='cpu'):
    """
    - WARNING: DEPRECATED
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
    cramer_re = torch.trapz((dy_target_cdf_re - dy_curr_cdf_re)**2, dx=spacing) + 1e-55
    cramer_re.sqrt_()
    cramer_re = cramer_re.mean()

    return cramer_re


def cramer_optim(pdf_target: torch.tensor, pdf_curr: torch.tensor, n_supp, integral_bound_factor=10, dev='cpu'):
    """
    - Dynamic Supports
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
    int_l_curr = pdf_curr.component_distribution.loc - integral_bound_factor * pdf_curr.component_distribution.scale
    int_u_curr = pdf_curr.component_distribution.loc + integral_bound_factor * pdf_curr.component_distribution.scale
    int_l_tar = pdf_target.component_distribution.loc - integral_bound_factor * pdf_target.component_distribution.scale
    int_u_tar = pdf_target.component_distribution.loc + integral_bound_factor * pdf_target.component_distribution.scale

    # Diff Current + Target
    diff_curr = torch.abs(int_u_curr - int_l_curr)
    delta_mb_curr = diff_curr / n_supp
    delta_mb_curr.unsqueeze_(dim=2)
    diff_tar = torch.abs(int_u_tar - int_l_tar)
    delta_mb_tar = diff_tar / n_supp
    delta_mb_tar.unsqueeze_(dim=2)

    # Calculate \Delta x for all supports
    dx_mb_diff_curr = steps_idx * delta_mb_curr
    dx_mb_diff_tar = steps_idx * delta_mb_tar

    # Calculate Supports for Current + Target and Concatenate
    dx_mb_curr = torch.ones((1, n_supp), device=dev) * int_l_curr.unsqueeze(dim=2) + dx_mb_diff_curr
    dx_mb_tar = torch.ones((1, n_supp), device=dev) * int_l_tar.unsqueeze(dim=2) + dx_mb_diff_tar
    dx_mb_singular = torch.cat((dx_mb_curr, dx_mb_tar), dim=2)

    dx_singular_flat, _ = dx_mb_singular.flatten(start_dim=1).unsqueeze(dim=1).sort()
    dx_mb_double = torch.cat((dx_singular_flat, dx_singular_flat), dim=1)

    dy_curr_mb = pdf_curr.cdf_mod(dx_mb_double)
    dy_target_mb = pdf_target.cdf_mod(dx_mb_double)

    cramer_re = torch.trapz(y=(dy_target_mb - dy_curr_mb) ** 2, x=dx_singular_flat.squeeze(dim=1)) + 1e-55
    cramer_re.sqrt_()
    cramer_re = cramer_re.mean()

    return cramer_re


def approx_integral_bounds(means_curr: torch.tensor, means_target: torch.tensor, stds_curr: torch.tensor,
                           stds_target: torch.tensor, factor, mean_std=False):
    """
    - WARNING: DEPRECATED
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


def sequential_iterator(n):
    for i in range(n):
        yield n


def get_double_q_selections(means1, means2, means1_next, means2_next, stds1, stds2, stds1_next, stds2_next,
                            kweights1, kweights2, kweights1_next, kweights2_next):
    """
    - Calculates min. values per entry from two pairs of Q-arrays (means, means_next) and selects entries of other
      four pairs (stds, stds_next, kweights, kweights_next) according to selection
    :param means1: Q1
    :param means2: Q2
    :param means1_next: Q1 next
    :param means2_next: Q2 next
    :param stds1: Std. dev. 1
    :param stds2: Std. dev. 2
    :param stds1_next: Std. dev. 1 next
    :param stds2_next: Std. dev. 2 next
    :param kweights1: Kernel weights 1
    :param kweights2: Kernel weights 2
    :param kweights1_next: Kernel weights 1 next
    :param kweights2_next: Kernel weights 2 next
    """
    # Calculate min. Q and average of stds
    stack_means = torch.stack([means1, means2])
    means_min, means_min_idx = torch.min(stack_means, dim=0)
    means_min_idx = torch.as_tensor(means_min_idx, dtype=torch.bool)

    stack_means_next = torch.stack([means1_next, means2_next])
    means_next_min, means_next_min_idx = torch.min(stack_means_next, dim=0)
    means_next_min_idx = torch.as_tensor(means_next_min_idx, dtype=torch.bool)

    # Get stds and kweights corresponding to mean_min and mean_next_min
    # Reverse stack, 1 = 2nd array; 0 = 1st array
    stds_selected_min = torch.zeros_like(stds1)
    stds_selected_min[means_min_idx] = stds2[means_min_idx]
    stds_selected_min[~means_min_idx] = stds1[~means_min_idx]

    stds_next_selected_min = torch.zeros_like(stds1_next)
    stds_next_selected_min[means_next_min_idx] = stds2_next[means_next_min_idx]
    stds_next_selected_min[~means_next_min_idx] = stds1_next[~means_next_min_idx]

    kweights_selected_min = torch.zeros_like(kweights1)
    kweights_selected_min[means_min_idx] = kweights2[means_min_idx]
    kweights_selected_min[~means_min_idx] = kweights1[~means_min_idx]

    kweights_next_selected_min = torch.zeros_like(kweights1_next)
    kweights_next_selected_min[means_next_min_idx] = kweights2_next[means_next_min_idx]
    kweights_next_selected_min[~means_next_min_idx] = kweights1_next[~means_next_min_idx]

    return means_min, means_next_min, stds_selected_min, stds_next_selected_min, kweights_selected_min, \
        kweights_next_selected_min


def get_partial_double_q_selections(means1, means2, stds1, stds2, kweights1, kweights2):
    """
    - Calculates min. values per entry from two pairs of Q-arrays (means, means_next) and selects entries of other
      four pairs (stds, stds_next, kweights, kweights_next) according to selection
    :param means1: Q1
    :param means2: Q2
    :param stds1: Std. dev. 1
    :param stds2: Std. dev. 2
    :param kweights1: Kernel weights 1
    :param kweights2: Kernel weights 2

    """
    # Calculate min. Q and average of stds
    stack_means = torch.stack([means1, means2])
    means_min, means_min_idx = torch.min(stack_means, dim=0)
    means_min_idx = torch.as_tensor(means_min_idx, dtype=torch.bool)

    # Get stds and kweights corresponding to mean_min and mean_next_min
    # Reverse stack, 1 = 2nd array; 0 = 1st array
    stds_selected_min = torch.zeros_like(stds1)
    stds_selected_min[means_min_idx] = stds2[means_min_idx]
    stds_selected_min[~means_min_idx] = stds1[~means_min_idx]

    kweights_selected_min = torch.zeros_like(kweights1)
    kweights_selected_min[means_min_idx] = kweights2[means_min_idx]
    kweights_selected_min[~means_min_idx] = kweights1[~means_min_idx]

    return means_min, stds_selected_min, kweights_selected_min


def calc_size_co_matrix(n_actions: int):
    """
    - For multivariate distributions
    - Calculates necessary number of elements of a covariance matrix as function of the action space size
    :param n_actions: Size of action space
    """
    n_actions = torch.as_tensor(n_actions)
    return n_actions + torch.floor(0.5 * (n_actions ** 2 - n_actions))


def smoothing(scalars, weight, last=0, iter=0):
    """
    EMA implementation according to tensorboard
    https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/
    vz_line_chart2/line-chart.ts#L699
    """
    smoothed = []
    for next_val in scalars:
        last = last * weight + (np.array(1, dtype=np.float64) - weight) * next_val
        iter += 1
        # de-bias
        debias_weight = np.array(1.0, dtype=np.float64)
        if weight != 1:
            debias_weight = 1 - np.power(weight, iter)
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)

    return smoothed, iter, last


def smooth_ref(scalars, weight):
    """
    EMA implementation according to tensorboard
    https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/
    vz_line_chart2/line-chart.ts#L699
    """
    last = 0
    smoothed = []
    num_acc = 0
    for next_val in scalars:
        last = last * weight + (np.array(1, dtype=np.float64) - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = np.array(1.0, dtype=np.float64)
        if weight != 1:
            debias_weight = 1 - np.power(weight, num_acc)
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)

    return smoothed


def get_supports(mb, means, sigmas):
    pass


if __name__ == '__main__':
    print(f'The action dim required for one action is: {calc_size_co_matrix(2)}')
    means = torch.randn(10)
    sigmas = torch.randn(10)