"""
Collections of tools to be sused in Z-DSAC
"""
import torch
import numpy as np
from scipy import integrate
from dsacv02.gmm_reparameterization.mixture_same_family import ReparameterizedMixtureSameFamilyMod as RMM


def generate_gauss_distr(means, stds, multivar=False, kweights=None):
    """
    TODO: Refactor to avoid if-statement
    - Generates either a multivariate or standard Gaussian
    :param means: Means
    :param stds: Standard deviations
    :param kweights: Not used here;
    :param multivar: CHANGE! Whether components of GMM are multivariate or not
    :return: Returns a GMM
    """
    if multivar:
        zcal = torch.distributions.MultivariateNormal(loc=means, covariance_matrix=stds)
    else:
        zcal = torch.distributions.Normal(loc=means, scale=stds)

    return zcal


def sampler_1k(distr, reparameterize):
    """
    - Function to sample from Gaussian - Batch-Wise
    :param distr: GMM object with one kernel
    :param reparameterize: If samples are used for backpropagation, reparameterization trick must be used
    :param batch_size: Batch-size; gauss_distr.component_distribution.loc
    """
    if reparameterize:
        # gmm_sample = distr.rsample((batch_size,))
        # gmm_sample.unsqueeze_(dim=1)
        gmm_sample = distr.rsample()
    else:
        # gmm_sample = distr.sample((batch_size,))
        # gmm_sample.unsqueeze_(dim=1)
        gmm_sample = distr.sample()

    return gmm_sample


def rsampler_1k(distr):
    """
    - Sampling with reparameterization for 1 kernel
    :param distr:
    :param batch_size:
    :return:
    """
    r_sample = distr.rsample()

    return r_sample


def cramer_optim_1k(pdf_target: torch.tensor, pdf_curr: torch.tensor, standard_supp, n_kernels=None, dev='cpu'):
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


def generate_gmm_distr_multi(means, stds, kweights, multivar=False):
    """
    TODO: Refactor to avoid if-statement
    - Generates either a multivariate or standard Gaussian Mxiture Model
    :param means: Kernel means
    :param stds: Kernel standard deviations
    :param kweights: Kernel weights
    :param multivar: CHANGE! Whether components of GMM are multivariate or not
    :return: Returns a GMM
    """
    mix_distr = torch.distributions.Categorical(probs=kweights)
    if multivar:
        comp_distr = torch.distributions.MultivariateNormal(loc=means, covariance_matrix=stds)
    else:
        comp_distr = torch.distributions.Normal(loc=means, scale=stds)
    zcal = RMM(mixture_distribution=mix_distr, component_distribution=comp_distr)

    return zcal


def cramer_optim_multi(pdf_target: torch.tensor, pdf_curr: torch.tensor, n_kernels, standard_supp=None, dev='cpu'):
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
    dx_mb_curr = pdf_curr.component_distribution.loc.unsqueeze(dim=2) + \
        pdf_curr.component_distribution.scale.unsqueeze(dim=2) * standard_supp
    dx_mb_curr = dx_mb_curr.detach()
    dx_mb_tar = pdf_target.component_distribution.loc.unsqueeze(dim=2) + \
        pdf_target.component_distribution.scale.unsqueeze(dim=2) * standard_supp
    dx_mb_tar = dx_mb_tar.detach()

    dx_mb_singular = torch.cat((dx_mb_curr, dx_mb_tar), dim=2)

    dx_singular_flat, _ = dx_mb_singular.flatten(start_dim=1).unsqueeze(dim=1).sort()
    # dx_mb_multi = torch.cat((dx_singular_flat, dx_singular_flat), dim=1)
    dx_mb_multi = dx_singular_flat * torch.ones(n_kernels, device=dev).unsqueeze(dim=1)

    dy_curr_mb = pdf_curr.cdf_mod(dx_mb_multi)
    dy_target_mb = pdf_target.cdf_mod(dx_mb_multi)

    cramer_re = torch.trapz(y=(dy_target_mb - dy_curr_mb) ** 2, x=dx_singular_flat.squeeze(dim=1)) + 1e-45
    cramer_re.sqrt_()
    cramer_re = cramer_re.mean()

    return cramer_re


def sampler_multi(distr, reparameterize):
    """
    - Function to sample from GMM - Batch-Wise
    :param distr: GMM object
    :param reparameterize: If samples are used for backpropagation, reparameterization trick must be used
    :param batch_size: No need to specify in multi-kernel, since rsample() samples automatically batch-wise
    """
    if reparameterize:
        gmm_sample = distr.rsample()
        gmm_sample.unsqueeze_(dim=1)
    else:
        gmm_sample = distr.sample()
        gmm_sample.unsqueeze_(dim=1)

    return gmm_sample


def rsampler_multi(distr):
    r_sample = distr.rsample()

    return r_sample


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


def cramer_torch_deac(pdf_target: torch.tensor, pdf_curr: torch.tensor, int_l, int_u, spacing, dev='cpu'):
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


def cramer_optim_deac(pdf_target: torch.tensor, pdf_curr: torch.tensor, n_supp, n_kernels, integral_bound_factor=10,
                      dev='cpu'):
    """
    - WARNING: DEPRECATED
    - Dynamic Supports
    - Batch-wise
    - int_l \approx \mu - 3.1*\sigma; int_u \approx \mu + 3.1*\sigma
    - Padding in method cdf() of RMM is deactivate, do not add additional dimension to dx
    - Implementation:
        1. Define the supports with constant n_steps
        2. Calculate the difference squared
        3. Integrate over all dx
        :param pdf_target: Target probability density function
        :param pdf_curr:  Current probability density function
        :param n_supp: Number of supports to approximate each kernel
        :param n_kernels: Number of kernels in PDF
        :param integral_bound_factor: bounds = \mu +/- ibf*\sigma
        :param dev: device to run on
        :return: Returns Cramer loss calculated by dynamic supports
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
    dx_mb_multi = dx_singular_flat * torch.ones(n_kernels, device=dev).unsqueeze(dim=1)

    dy_curr_mb = pdf_curr.cdf_mod(dx_mb_multi)
    dy_target_mb = pdf_target.cdf_mod(dx_mb_multi)

    cramer_optim_ret = torch.trapz(y=(dy_target_mb - dy_curr_mb) ** 2, x=dx_singular_flat.squeeze(dim=1)) + 1e-55
    cramer_optim_ret.sqrt_()
    cramer_optim_ret = cramer_optim_ret.mean()

    return cramer_optim_ret


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


def eval_agent(env, agent, discrete: bool, act_dim: int, obs_dim: int, max_iter: int):
    """
    - Evaluation rollout for non-vectorized environment; 1 episode
    :param env: Environment instance
    :param agent: RL agent
    :param discrete: Whether action space is continuous or discrete
    :param act_dim: Action space dimension
    :param obs_dim: Observation space dimension
    :param max_iter: Maximal allowed iterations per episode
    """
    done = False
    observation, _ = env.reset()
    observation = np.expand_dims(observation, axis=0)
    reward_episode = 0
    episode_iter = 0
    while not done:
        action, prob_action = agent.choose_deterministic_action(observation)
        if discrete:
            if act_dim == 1:
                action = 0 if action <= 0 else 1
            else:
                # Single action per time-step for LunarLander-v2
                action = np.argmax(action)
        else:
            if act_dim == 1:
                action = action.squeeze(axis=1)
            else:
                action = action.squeeze().tolist()
        observation_, reward, done, info, _ = env(action)
        reward_episode += reward
        observation_ = observation_.reshape((1, obs_dim))

        if episode_iter > max_iter:
            done = True

        episode_iter += 1

    return reward_episode


def eval_agent_vec():
    pass


if __name__ == '__main__':
    n_actions = 1
    print(f'The action dim required for {n_actions} action is: {calc_size_co_matrix(n_actions)}')
    means = torch.randn(10)
    sigmas = torch.randn(10)