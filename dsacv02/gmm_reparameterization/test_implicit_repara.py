import torch
import torch.distributions as distr
from mixture_same_family import ReparameterizedMixtureSameFamilyMod
from normal_stable import NormalStable
from math import prod

n_kernels = 2
mixture_probs = torch.ones(n_kernels) / n_kernels
sample_shape = (5,)

locs = torch.tensor([-1., 1.])
stds = torch.ones(n_kernels)

mixture1 = distr.Categorical(probs=mixture_probs)
components1 = NormalStable(loc=locs, scale=stds)
gmm1 = ReparameterizedMixtureSameFamilyMod(mixture_distribution=mixture1,
                                          component_distribution=components1)

extra_ndims = 2
components2 = NormalStable(loc=locs.reshape(locs.shape + (1,) * extra_ndims),
                           scale=stds.reshape(stds.shape + (1,) * extra_ndims))
components2 = distr.Independent(components2, extra_ndims)
gmm2 = ReparameterizedMixtureSameFamilyMod(mixture_distribution=mixture1,
                                           component_distribution=components2)
torch.manual_seed(123456)
X1 = gmm1.rsample(sample_shape=(5,))
Z1 = gmm1._distributional_transform(X1)
for x1, z1 in zip(X1, Z1):
    print(f'Prob of rsample {x1}: {gmm1.log_prob(x1)}')
    print(f'Prob. of distirubtional transform {z1}: {gmm1.log_prob(z1)}\n')

torch.manual_seed(123456)
X2 = gmm2.rsample(sample_shape=(5,))
Z2 = gmm2._distributional_transform(X2)
#
# assert torch.allclose(X1, X2.squeeze())
# assert torch.allclose(Z1, Z2.squeeze())

# # Check if multivariate runs
# mixture3 = distr.Categorical(probs=mixture_probs[0])
# components3 = NormalStable(loc=locs.T, scale=stds.T)
# components3 = distr.Independent(components3, 1)
# mog3 = ReparameterizedMixtureSameFamilyMod(mixture_distribution=mixture3,
#                                            component_distribution=components3)
#
# X3 = mog3.rsample(sample_shape=(5,))
# # TODO: add test for multivariate