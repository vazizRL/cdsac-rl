import torch
import torch.distributions as distr
import numpy as np
from scipy import integrate
from dsacv02.gmm_reparameterization.mixture_same_family import ReparameterizedMixtureSameFamilyMod


def integrate_gauss(mean: float, std: float, int_l, int_u):
    # Calculate \int_a^b {\mu_Y(t)t dt} numerically
    e = torch.e
    normal = distr.Normal(loc=torch.tensor(mean), scale=torch.tensor(std))
    integral_norm = integrate.quad(lambda x: e**(normal.log_prob(torch.tensor(x))), int_l, int_u)
    print(f'The area of the Gauss calculated with integrate.quad() from {int_l} to {int_u} is: {integral_norm}')
    integral_norm, err = integral_norm
    return integral_norm, err


if __name__ == '__main__':
    gmm_normal_same_action = False
    mean = 0.0
    std = 0.6
    batch_size = 50
    gauss = distr.Normal(mean, std)
    actions_unbounded = gauss.rsample((batch_size,)).unsqueeze(dim=1)
    probs_unbounded = gauss.log_prob(actions_unbounded)

    # Bound actions
    lb, ub = torch.tensor([-1.]), torch.tensor([1.])
    actions_bounded = torch.clamp(actions_unbounded, lb, ub)

    # Probability of unbounded actions according to DSAC-paper; Shape: (Batch, Action)
    ones = torch.ones(batch_size).unsqueeze(dim=1)
    # probs_bounded = gauss.log_prob(actions_unbounded) - torch.log(ones - torch.tanh(actions_unbounded) ** 2) - \
    #                 torch.log((ub-lb)/2)
    probs_bounded = gauss.log_prob(actions_unbounded) - torch.log(ones - torch.tanh(actions_unbounded) ** 2)

    for action, prob_unbounded, action_bounded, prob_bounded in \
            zip(actions_unbounded, probs_unbounded, actions_bounded, probs_bounded):
        print(f'Original action {action}; original prob density {prob_unbounded.exp()}; bounded action {action_bounded}; '
              f'bounded prob density {prob_bounded.exp()}')

    '''
    Test Recalculation of log_prob with own GMM
    '''
    weights = torch.tensor([0.5, 0.5])
    means = torch.tensor([-1.0, 1.0])
    stds = torch.tensor([1.0, 1.0])
    mixture_weights = distr.Categorical(weights)
    kernels = distr.Normal(loc=means, scale=stds)
    gmm = ReparameterizedMixtureSameFamilyMod(mixture_weights, kernels)

    # Get log probs and
    if gmm_normal_same_action:
        gmm_actions_unbounded = actions_unbounded
    else:
        gmm_actions_unbounded = gmm.sample((batch_size,)).unsqueeze(dim=1)

    gmm_log_probs_unbounded = gmm.log_prob(gmm_actions_unbounded)
    ones = torch.ones(batch_size).unsqueeze(dim=1)

    # GMM lb, up and bounded actions
    gmm_lb, gmm_ub = torch.tensor([-1.0]), torch.tensor([1.0])
    gmm_actions_bounded = torch.clamp(gmm_actions_unbounded, lb, ub)

    gmm_log_probs_bounded = gmm_log_probs_unbounded - torch.log(ones - torch.tanh(gmm_actions_unbounded) ** 2) - \
                    torch.log((ub-lb)/2)

    print('\n\n')
    for gmm_action_unbounded, gmm_log_prob_unbounded, gmm_action_bounded, gmm_log_prob_bounded in \
            zip(gmm_actions_unbounded, gmm_log_probs_unbounded, gmm_actions_bounded, gmm_log_probs_bounded):
        print(
            f'Original action {gmm_action_unbounded}; original prob density {gmm_log_prob_unbounded.exp()}; '
            f'bounded action {gmm_action_bounded}; bounded prob density {gmm_log_prob_bounded.exp()}')





