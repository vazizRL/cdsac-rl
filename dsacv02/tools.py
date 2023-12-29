"""
Collections of tools to be sused in Z-DSAC
"""
import torch
from scipy import integrate


def cramer_from_pdf(pdf_target: torch.tensor, pdf_curr: torch.tensor, int_l, int_u, points=(-100, 100)):
    """
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


def approx_integral_bounds(means_curr: torch.tensor, means_target: torch.tensor, stds_curr: torch.tensor,
                           stds_target: torch.tensor, factor, mean_std=False):
    """
    - Approximates useful integration bounds when given current and target GMM parameters
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
