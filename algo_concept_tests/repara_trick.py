import numpy as np
import torch
import torch.distributions as distr


def create_gmm(means: tuple, variances: tuple):
    n_kernels = len(means)
    weights = torch.ones(n_kernels, dtype=torch.float64) / n_kernels
    means = torch.as_tensor(means, dtype=torch.float64)
    variances = torch.as_tensor(variances, dtype=torch.float64)
    gmm = distr.MixtureSameFamily(distr.Categorical(probs=weights), distr.Normal(means, variances))

    return gmm


if __name__ == '__main__':
    N_SAMPLES = 10000
    MEAN = 0.0
    STD = 1.0
    norm = distr.Normal(torch.tensor([MEAN]), torch.tensor([STD]))

    X = list()
    X_re = list()
    for i in range(N_SAMPLES):
        X.append(norm.sample())
        X_re.append(norm.rsample())
    X = torch.as_tensor(X, dtype=torch.float64)
    X_re = torch.as_tensor(X_re, dtype=torch.float64)

    # Variance of standard sampling
    sqr_diff = (X-MEAN)**2
    var = sqr_diff.sum(dim=0) / N_SAMPLES

    # Variance of reparameterized sampling
    sqr_diff_re = (X_re-MEAN)**2
    var_re = sqr_diff_re.sum(dim=0) / N_SAMPLES

    # Print STDs calculated from samples
    std = var**0.5
    std_re = var_re**0.5
    print(f'Calculated standard deviation from samples for normal sampling: {std}\n')
    print(f'Calculated standard deviation from samples for reparameterized sampling: {std_re}\n')

    # Print Means calculated from samples
    mean = X.sum(dim=0) / N_SAMPLES
    mean_re = X_re.sum(dim=0) / N_SAMPLES
    print(f'Calculated mean from samples for normal sampling: {mean}\n')
    print(f'Calculated mean from samples for reparameterized sampling: {mean_re}\n')

    ''' 
    Test resulting Gauss parameters from GMMs
    '''
    means = (-10.0, 10.0)
    gmm = create_gmm(means=means, variances=(1.0, 1.0))
    X_gmm = list()
    for i in range(N_SAMPLES):
        X_gmm.append(gmm.sample())
    X_gmm = torch.as_tensor(X_gmm, dtype=torch.float64)
    mean_gmm = sum(means)/ len(means)
    sqr_diff_gmm = (X_gmm - mean_gmm)**2
    var_gmm = sqr_diff_gmm.sum(dim=0) / N_SAMPLES
    std_gmm = var_gmm**0.5
    print(f'Calculated standard deviation from samples for GMM normal sampling: {std_gmm}\n')

    '''
    Samples from GMM and probability according to selected kernel
    '''
    repeat = 50
    for i in range(repeat):
        component, sample = gmm.sample()
        probs_kernels = gmm.component_distribution.log_prob(sample)
        print(f'Sample: {sample} from component: {component} with prob {probs_kernels[0].exp()}')






