import numpy as np
import torch
from scipy.stats import energy_distance
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy import integrate
from torch.distributions import Normal


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


def cramer_from_pdf(pdf_target: torch.tensor, pdf_curr: torch.tensor, int_l, int_u):
    distance, error_est = integrate.quad(
        lambda x: (pdf_target.cdf(torch.tensor([x])).numpy() - pdf_curr.cdf(torch.tensor([x])).numpy()) ** 2,
        int_l, int_u
    )

    return distance**0.5, error_est

# Example usage:
p = np.array([0.1, 0.2, 0.3, 0.4])
q = np.array([0.2, 0.25, 0.3, 0.25])
ed = energy_distance(p, q)
print(f"Energy Distance: {ed}")


if __name__ == '__main__':
    # m
    S_SIZE = 100000
    MIN_SUPPORT = -20
    MAX_SUPPORT = 20
    supports = np.arange(MIN_SUPPORT, MAX_SUPPORT, 0.02)

    # Scale is Std.
    target_pdf = norm.pdf(supports, loc=0, scale=1)
    target_pdf_cont = Normal(torch.tensor([0]), torch.tensor([1]))

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
        print(f'For mean {idx-abs(MIN_SUPPORT)} of Normal distribution with std 1, the distance is: {l_i}')
        # Own implementation
        l_i_own = energy_d(supports=supports, p_target=target_pdf, p_curr=pdfs[idx], s_size=S_SIZE)
        print(f'Own implementation of energy distance: {l_i_own}, its square root: {l_i_own**0.5}')
        l_cramer_pdf, err = cramer_from_pdf(target_pdf_cont, pdfs_cont[idx], int_l=-50, int_u=50)
        print(f'Cramer distance from PDF: {l_cramer_pdf} with max. error: {err} \n')


