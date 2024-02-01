"""
Hyperparameters, MLP architecture...etc. defined in actor_critic_test.py
"""
import time
import torch.distributions as distr
from actor_critic_test import *


def generate_own_gmm_distr(mean, std, kweight):
    """
    - Generate a GMM with own implementation
    - Batch agnostic until now
    :param mean: One set of means
    :param std: One set of stds
    :param kweight: One set of kernel weights
    :return: Own implementation of GMM with reparameterization allowed
    """
    cat_distr = distr.Categorical(probs=kweight)
    comp_distr = distr.Normal(loc=mean, scale=std)
    gmm_distr = RMM(mixture_distribution=cat_distr, component_distribution=comp_distr)

    return gmm_distr


def generate_py_gmm_distr(means, stds, kweights):
    """
    - Pytorch implementation with which GMMs will be generated
    :param means:
    :param stds:
    :type stds:
    :param kweights:
    :type kweights:
    :return:
    :rtype:
    """
    cat_distr = distr.Categorical(probs=kweights)
    comp_distr = distr.Normal(loc=means, scale=stds)
    gmm_distr = distr.MixtureSameFamily(mixture_distribution=cat_distr, component_distribution=comp_distr)

    return gmm_distr


def check_sampling(means, stds, kweights, n_sampling):
    own_gmm1_act1 = generate_own_gmm_distr(mean=means, std=stds, kweight=kweights)
    pytorch_gmm1_act1 = distr.MixtureSameFamily(mixture_distribution=distr.Categorical(probs=kweights),
                                                component_distribution=distr.Normal(loc=means, scale=stds))
    own_samples = own_gmm1_act1.sample((n_sampling,))
    pytorch_samples = pytorch_gmm1_act1.sample((n_sampling,))

    print(
        f'Element 1, Action 1. Mean by analysis: {means_act[0][:, 0].mean_target()}; Mean by own gmm samples: '
        f'{own_samples.mean()}; Mean by pytorch gmm samples: {pytorch_samples.mean()}')

"""
Construct a GMM from critic parameters
"""
stds_cr.abs_()
start_cr_loop = time.perf_counter()
gmms_cr = []
for idx_cr, (mean_cr_i, std_cr_i, kernel_weight_cr_i) in enumerate(zip(means_cr, stds_cr, kernel_weights_cr)):
    gmm_i = generate_own_gmm_distr(mean=mean_cr_i.squeeze(), std=std_cr_i.squeeze(),
                                   kweight=kernel_weight_cr_i.squeeze())
    gmms_cr.append(gmm_i)
    # print(f'Sample from Critic GMM {idx_cr}: {gmm_i.rsample()}')
end_cr_loop = time.perf_counter()
print(f'\n Calculation time for critic loop: {end_cr_loop - start_cr_loop}')
# print('----------------\n')
""" 
Construct a GMM from actor parameters
"""
# In-place operation, in addition, separate for actions
stds_act.abs_()
start_act_loop = time.perf_counter()
gmms_act = []
actions1_gmm = []
actions2_gmm = []
# Action 1 + Action 2 in same loop
for idx_act, (mean_act_i, std_act_i, kernel_weight_act_i) in enumerate(zip(means_act, stds_act, kernel_weights_act)):
    action1_i, action1_log_prob_i = actor.sample_from_action_distr(locs=mean_act_i[:, 0],
                                                                   stds=std_act_i[:, 0],
                                                                   k_weights=kernel_weight_act_i,
                                                                   reparameterization=True)
    action2_i, action2_log_prob_i = actor.sample_from_action_distr(locs=mean_act_i[:, 1],
                                                                   stds=std_act_i[:, 1],
                                                                   k_weights=kernel_weight_act_i,
                                                                   reparameterization=True)

    gmm_act1_i = generate_own_gmm_distr(mean=mean_act_i[:, 0], std=std_act_i[:, 0],
                                        kweight=kernel_weight_act_i)
    gmm_act2_i = generate_own_gmm_distr(mean=mean_act_i[:, 1], std=std_act_i[:, 1],
                                        kweight=kernel_weight_act_i)
    actions1_gmm.append(gmm_act1_i)
    actions2_gmm.append(gmm_act2_i)
    # print(f'Sample for action 2 from actor GMM {idx_act}: {action2_i} with log_prob {action2_log_prob_i}')
    # print(f'Sample for action 1 from actor GMM {idx_act}: {action1_i} with log_prob {action1_log_prob_i}')
gmms_act.append(actions1_gmm)
gmms_act.append(actions2_gmm)
end_act_loop = time.perf_counter()
print(f'Calculation time for actor loops over all actions: {end_act_loop - start_act_loop}')

'''
Check sampling with standard implementation and own implementation
'''
mean1_act1 = means_act[0][:, 0]
std1_act1 = stds_act[0][:, 0]
kweight1_act1 = kernel_weights_act[0]
check_sampling(means=mean1_act1, stds=std1_act1, kweights=kweight1_act1, n_sampling=10_000)

"""
Test generation of batch-wise GMMs with PyTorch implementation
"""
# Critic
py_gmm_batch_cr = generate_py_gmm_distr(means=means_cr.squeeze(), stds=stds_cr.squeeze(), kweights=kernel_weights_cr)
own_gmm_batch_cr = generate_own_gmm_distr(mean=means_cr.squeeze(), std=stds_cr.squeeze(), kweight=kernel_weights_cr)
# Actor: Multivariate components of the GMM are required!
# multi_component = distr.MultivariateNormal(loc=means_act, covariance_matrix=)
# py_gmm_batch_act = generate_py_gmm_distr(means=means_act, stds=stds_act, kweights=kernel_weights_act)



