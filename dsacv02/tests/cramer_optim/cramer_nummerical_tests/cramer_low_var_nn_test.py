import torch as t
import matplotlib.pyplot as plt
import os
from dsacv02.tools import cramer_optim_1k, get_normal_supports, generate_gauss_distr
from dsacv02.actor_critic import Critic
from torch.optim import Adam
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def quadratic(x, a, h, k, batch_size, dev='cpu'):
    x = t.as_tensor(x, device=dev)
    val = -a * ((x - h) ** 2) + k

    return val.expand(batch_size).unsqueeze(dim=1)


def uniform_sampling(low, high, batch_size, dim, dev='cpu'):

    return low + 2 * t.rand(batch_size, dim, device=dev) * high


def gauss(mean, std, dev='cpu'):

    return t.normal(mean=mean, std=std)


def MSE_1k(pdf_target: t.tensor, pdf_curr: t.tensor):
    """
    - Test Method calculating the MSE
    """
    # Dynamically Determine Supports for Current + Target
    batch_size = pdf_target.loc.shape[0]
    difference = pdf_curr.loc - pdf_target.loc
    difference_mean = difference**2 / batch_size
    mse = difference_mean.sum()

    return mse


if __name__ == '__main__':
    DEV = 'cuda:0'
    # Training parameters
    ITERATIONS = 10000
    # ITERATIONS = 19999
    LR = 3e-4
    # Log elements
    LOG_FREQ = 1
    cr_means = list()
    cr_stds = list()
    cr_loss = list()
    # Observation Input: Ant: Low, High = -inf, inf
    HIDDEN = (256, 256)
    ACTIVE = ('relu', 'relu')
    # Batch
    BATCH_SIZE = 256
    # Standard supports
    N_SUPPORTS, IBF = 32, 15
    STANDARD_SUPP = get_normal_supports(batch_size=BATCH_SIZE, n_kernels=1, n_supp=N_SUPPORTS,
                                        integral_bound_factor=IBF, dev=DEV)

    OBS_N, OBS_MEAN_LOW, OBS_MEAN_HIGH, OBS_STD = 27, t.tensor(-7, device=DEV), t.tensor(10, device=DEV), \
                                                  t.tensor(1.66, device=DEV).expand(BATCH_SIZE).unsqueeze(dim=1)

    # Policy Input
    ACT_N, ACT_MEAN_LOW, ACT_MEAN_HIGH, ACT_STD = 8, t.tensor(-1, device=DEV), t.tensor(1, device=DEV), \
                                                  t.tensor(0.13, device=DEV).expand(BATCH_SIZE).unsqueeze(dim=1)

    # Instantiate Critic
    CRITIC_MIN_STD = 0.0000000001
    # CRITIC_MAX_STD = 1000
    CRITIC_MAX_STD = 10000000000
    EXPONENTIATE = True
    critic = Critic(state_dim=OBS_N, action_dim=ACT_N, hidden_layers=HIDDEN, n_kernels=1, activ=ACTIVE,
                    value_min_std=CRITIC_MIN_STD, value_max_std=CRITIC_MAX_STD, learnable_weights=False, device=DEV)
    # NN Optimizer
    cr_adam = Adam(critic.parameters(), lr=LR)

    # Target Distribution
    TAR_STD_CONST = t.tensor(1e-3, device=DEV).expand(BATCH_SIZE).unsqueeze(dim=1)
    # tar_means = t.tensor(100, device=DEV).expand(BATCH_SIZE).unsqueeze(dim=1)
    if EXPONENTIATE:
        TAR_STD_CONST = t.nn.functional.softplus(input=TAR_STD_CONST, beta=1, threshold=2)

    for iter in range(ITERATIONS):
        # Get Means
        obs_mean = uniform_sampling(low=OBS_MEAN_LOW, high=OBS_MEAN_HIGH, batch_size=BATCH_SIZE, dim=OBS_N, dev=DEV)
        acts_mean = uniform_sampling(low=ACT_MEAN_LOW, high=ACT_MEAN_HIGH, batch_size=BATCH_SIZE, dim=ACT_N, dev=DEV)
        # Get obs and acts
        obs = gauss(mean=obs_mean, std=OBS_STD, dev=DEV)
        acts = gauss(mean=acts_mean, std=ACT_STD, dev=DEV)
        # Get batch-wise prediction
        pred_cr_means, pred_cr_stds, _ = critic(observation=obs, action=acts, exp=EXPONENTIATE)

        tar_means = quadratic(x=iter, a=1.8e-6, h=10000, k=180, batch_size=BATCH_SIZE, dev=DEV)
        # tar_means = quadratic(x=iter, a=1e-4, h=10000, k=10500, batch_size=BATCH_SIZE, dev=DEV) # Steep!
        # tar_stds_var = quadratic(x=iter, a=1.3e-7, h=10000, k=13, batch_size=BATCH_SIZE, dev=DEV) + 1e-10
        tar_stds_var = quadratic(x=iter, a=1e-8, h=10000, k=1, batch_size=BATCH_SIZE, dev=DEV) + 1e-10  # Very flat

        if EXPONENTIATE:
            pred_cr_means = t.nn.functional.softplus(input=pred_cr_means, beta=1, threshold=2)
            pred_cr_stds = t.nn.functional.softplus(input=pred_cr_stds, beta=1, threshold=2)
            tar_means = t.nn.functional.softplus(input=tar_means, beta=1, threshold=2)
            tar_stds_var = t.nn.functional.softplus(input=tar_stds_var, beta=1, threshold=2)

        # Distributions
        pred_cr_distr = generate_gauss_distr(means=pred_cr_means, stds=pred_cr_stds, multivar=False, kweights=None)
        # tar_cr_distr_var = generate_gauss_distr(means=tar_means, stds=TAR_STD_CONST, multivar=False, kweights=None)
        tar_cr_distr_var = generate_gauss_distr(means=tar_means, stds=tar_stds_var, multivar=False, kweights=None)

        loss_var = cramer_optim_1k(pdf_target=tar_cr_distr_var, pdf_curr=pred_cr_distr,
                                   standard_supp=STANDARD_SUPP, n_kernels=1, dev=DEV)
        # loss_var = MSE_1k(pdf_target=tar_cr_distr_var, pdf_curr=pred_cr_distr)

        cr_adam.zero_grad()
        loss_var.backward()
        cr_adam.step()

        # Logging function
        if iter % LOG_FREQ == 0:
            cr_loss.append(loss_var.cpu().item())
            cr_means.append(pred_cr_means.mean().cpu().item())
            cr_stds.append(pred_cr_stds.mean().cpu().item())

    # Calculate True Mean and Std
    means_true = list()
    stds_true = list()
    for j in range(ITERATIONS):
        means_true_i = quadratic(x=j, a=1.8e-6, h=10000, k=180, batch_size=1, dev='cpu')
        # means_true_i = quadratic(x=j, a=1e-4, h=10000, k=10500, batch_size=1, dev='cpu') # Steep!
        # stds_true_i = t.tensor(0.001, device='cpu')
        # stds_true_i = quadratic(x=j, a=1.3e-7, h=10000, k=13, batch_size=1, dev='cpu')
        stds_true_i = quadratic(x=j, a=1e-8, h=10000, k=1, batch_size=1, dev='cpu') + 1e-10
        if EXPONENTIATE:
            means_true_i = t.nn.functional.softplus(input=means_true_i, beta=1, threshold=2)
            stds_true_i = t.nn.functional.softplus(input=stds_true_i, beta=1, threshold=2)

        means_true.append(means_true_i.item())
        stds_true.append(stds_true_i.item())

    # Plot Cr Loss
    plt.plot(cr_loss, label='Critic Loss')
    plt.title(f'Critic Loss')
    plt.xlabel('Iterations')
    plt.grid(visible=True, which='both', color='grey', linewidth=0.3)
    # plt.xlim((RANGE_HIGH, RANGE_LOW))
    plt.ylabel('Critic Loss')
    plt.legend()
    plt.show(block=False)
    # Plot Cr Val
    plt.figure()
    plt.plot(cr_means, color='blue', label='Predicted Means')
    plt.plot(means_true, color='red', label='True Means')
    plt.title('Critic Vals')
    plt.xlabel('Iterations')
    plt.grid(visible=True, which='both', color='grey', linewidth=0.3)
    plt.ylabel('Values')
    plt.legend()
    plt.show(block=False)
    # Plot Cr Std
    plt.figure()
    plt.plot(cr_stds, color='blue', label='Predicted Stds')
    plt.plot(stds_true, color='red', label='True Stds')
    plt.title(f'Critic Stds')
    plt.xlabel('Iterations')
    plt.grid(visible=True, which='both', color='grey', linewidth=0.3)
    plt.ylabel('Stds')
    plt.legend()
    plt.show(block=True)
    print('Finished Printing Cramer Loss Calculations')

