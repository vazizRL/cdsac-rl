import time
import torch
import torch.distributions as distr
from dsacv02.gmm_reparameterization.mixture_same_family import ReparameterizedMixtureSameFamilyMod as RMM


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


"""Test Sampling Order of GMMs"""
mb_size = 10
means = torch.ones(mb_size).unsqueeze(dim=1) * torch.tensor([0.0, 10.0], dtype=torch.float64)
stds = torch.ones(mb_size).unsqueeze(dim=1) * torch.tensor([1.0, 1.0], dtype=torch.float64)
kweights = torch.ones(mb_size).unsqueeze(dim=1) * torch.tensor([0.5, 0.5], dtype=torch.float64)
gmm = generate_own_gmm_distr(mean=means, std=stds, kweight=kweights)
