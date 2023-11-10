import numpy as np
import torch
import torch.distributions as dist
from scipy.stats import energy_distance
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy import integrate
from torch.distributions import Normal


def normal_pdf_scaled(sprt, mean, sig, scale):
    z = 1/(sig*np.sqrt(2*np.pi))
    dist = np.e**(-0.5*((sprt-mean)/sig)**2)

    return z*dist*scale


def energy_d(supports, p_target, p_curr, s_size):
    """
    - For uni-variate only!
    :param supports: Supports of the distributoin
    :param p_target: Probability for bin i of target
    :param p_curr: Probability for bin i of current
    :param s_size: Sample size
    :return: Square root of energy distance"
    """
    t_samples = np.random.choice(supports, size=s_size, p=p_target).astype(np.float64)
    t_samples_prime = np.random.choice(supports, size=s_size, p=p_target).astype(np.float64)
    samples = np.random.choice(supports, size=s_size, p=p_curr).astype(np.float64)
    samples_prime = np.random.choice(supports, size=s_size, p=p_curr).astype(np.float64)

    # Only for P \in \R^d, d=2, otherwise ||\cdot||_p
    diff_t_t = np.abs(t_samples - t_samples_prime)
    diff_c_c = np.abs(samples - samples_prime)
    diff_t_c = np.abs(t_samples - samples)
    e_distance = 2*diff_t_c.mean() - diff_t_t.mean() - diff_c_c.mean()

    return e_distance


def cramer_from_pdf(pdf_target: torch.tensor, pdf_curr: torch.tensor, int_l, int_u, points=(-100, 100)):
    """
    - The integration limits should NOT be too far off form the lowest and highest point of the function!
    """
    distance, error_est = integrate.quad(
        lambda x: (pdf_target.cdf(torch.tensor([x])).numpy() - pdf_curr.cdf(torch.tensor([x])).numpy()) ** 2,
        int_l, int_u, points=points
    )

    return distance**0.5, error_est


def probe_gmm(gmm):
    """
    - Samples from some gmm and
    :param gmm:
    :type gmm:
    """
    for i in range(10):
        sample = gmm.sample()
        print(f'Probability for {sample} is {torch.e**gmm.log_prob(sample)}')
    print('\n')


def sym_distance_from_origin(lim=50, means_sym=10.0):
    # Define Gauss at origin with std=1
    target_distribution = Normal(torch.tensor([0]), torch.tensor([1]))
    # Test mysterious symmetry
    gmmi = dist.MixtureSameFamily(dist.Categorical(probs=torch.tensor([0.5, 0.5])),
                                  dist.Normal(torch.tensor([-means_sym, means_sym]), torch.tensor([1.0, 1.0])))
    l_sym = cramer_from_pdf(target_distribution, gmmi, int_l=-lim, int_u=lim, points=[10*-means_sym, 10*means_sym])
    print(f'Symmetrical cramer distance for means {-means_sym, means_sym} is: {l_sym}. Calculated with limits:'
          f'{-lim, lim}')

    return gmmi


def integral_of_cont_pdf(mean: float, std: float):
    # Calculate \int_a^b {\mu_Y(t)t dt} nummerically, a=0, b=20
    e = torch.e
    normal = Normal(loc=torch.tensor(mean), scale=torch.tensor(std))
    integral_norm = integrate.quad(lambda x: e**(normal.log_prob(torch.tensor(x))) * torch.tensor(x), 0.0, 20.0)
    print(f'The area of the Gauss calculated with integrate.quad() from 0 to 20 is: {integral_norm}')

    return 0


def integral_of_cont_gmm(means: torch.tensor, stds: torch.tensor):
    e = torch.e
    n_kernels = len(means)
    weights = torch.ones(n_kernels) * (1/n_kernels)
    gmm = dist.MixtureSameFamily(dist.Categorical(probs=weights), dist.Normal(means, stds))
    integral_gmm = integrate.quad(lambda x: e**(gmm.log_prob(torch.tensor(x))) * torch.tensor(x),
                                  a=-50, b=50, points=(-10*means[0], 10*means[0]))
    print(f'The area of the GMM with means {means} and stds {stds} '
          f'calculated with integrate.quad() is {integral_gmm}')

    return gmm


if __name__ == '__main__':
    # Example usage:
    p = np.array([0.1, 0.2, 0.3, 0.4])
    q = np.array([0.2, 0.25, 0.3, 0.25])
    ed = energy_distance(p, q)
    print(f"Energy Distance: {ed}")

    # m
    S_SIZE = 1100
    MIN_SUPPORT = -20
    MAX_SUPPORT = 20
    supports = np.arange(MIN_SUPPORT, MAX_SUPPORT, 0.02)

    # Scale is Std.
    target_pdf = norm.pdf(supports, loc=0, scale=1)
    target_pdf_cont = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    # Instantiate PDFs and normalize
    pdfs = list()
    samples = list()
    pdfs_cont = list()
    for i in range(-10, 11, 1):
        pdf_i = norm.pdf(supports, loc=i, scale=1)
        pdf_i_cont = Normal(torch.tensor([float(i)]), torch.tensor([1.0]))
        pdf_i /= pdf_i.sum()
        sample_i = np.random.choice(supports, size=S_SIZE, p=pdf_i)
        pdfs.append(pdf_i)
        pdfs_cont.append(pdf_i_cont)
        samples.append(sample_i)

    # Normalize target PDF
    target_pdf /= target_pdf.sum()
    target_samples = np.random.choice(supports, size=S_SIZE, p=target_pdf)

    # Calculate distance between distributions given by \hat{P}_m and \hat{Q}_m for samples
    for idx, samples_i in enumerate(samples):
        l_i = energy_distance(target_samples, samples_i)
        print(f'For mean {-10 + idx} of Normal distribution with std 1, the distance is: {l_i}')
        # Own implementation
        l_i_own = energy_d(supports=supports, p_target=target_pdf, p_curr=pdfs[idx], s_size=S_SIZE)
        print(f'Own implementation of energy distance: {l_i_own}, its square root: {l_i_own**0.5}')
        l_cramer_pdf, err = cramer_from_pdf(target_pdf_cont, pdfs_cont[idx], int_l=-50, int_u=50)
        print(f'Cramer distance from PDF: {l_cramer_pdf} with max. error: {err}')


    # Define a GMM in 1d with n Kernels in PyTorch
    means = torch.arange(-10.0, 11.0, 1.0)
    variances = torch.ones(21, dtype=torch.float64)
    weights = torch.ones(21) / 21

    # Categorical: Prob. of which kernel is "used"
    gmm = dist.MixtureSameFamily(dist.Categorical(probs=weights), dist.Normal(means, variances))

    # Distance between target_pdf_cont and gmm (should also be cont)
    for i in range(-10, 11, 1):
        # Define gmm
        means = torch.tensor([0, i], dtype=torch.float64)
        variances = torch.ones(2, dtype=torch.float64)
        weights = torch.tensor([0.5, 0.5])
        gmm = dist.MixtureSameFamily(dist.Categorical(probs=weights), dist.Normal(means, variances))
        cramer_dist = cramer_from_pdf(pdf_target=target_pdf_cont, pdf_curr=gmm, int_l=-50, int_u=50)
        print(f'For GMM means {means}, distance to target is: {cramer_dist} \n')

    # Print symetric distance and get bimodal distribution
    gmm_sym = sym_distance_from_origin(lim=50, means_sym=10.0)

    # Print probes and their probabilities according to bimodal distribution
    probe_gmm(gmm_sym)

    # Print integral of a continuous pdf with integrate.quad() method
    integral_of_cont_pdf(10.0, 1.0)

    # Print integral of symmetrical gmm
    means_gmm = torch.tensor([6.0, 4.0])
    stds_gmm = torch.tensor([5.0, 1.0])
    sym_gmm = integral_of_cont_gmm(means=means_gmm, stds=stds_gmm)





