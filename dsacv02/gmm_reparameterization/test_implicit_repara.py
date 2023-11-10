from mixture_same_family import *
from normal_stable import *


if __name__ == '__main__':
    cat_distr = distr.Categorical(probs=torch.tensor([0.5, 0.5]))
    component_distr = distr.Normal(loc=torch.tensor([-1.0, 1.0]), scale=torch.tensor([1.0, 1.0]))
    mix_mod = MixtureSameFamilyMod(mixture_distribution=cat_distr, component_distribution=component_distr)


