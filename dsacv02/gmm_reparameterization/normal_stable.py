import numpy as np
import torch
from torch.distributions.normal import Normal


class NormalStable(Normal):

    def _standardize(self, x):
        return (x - self.loc) * self.scale.reciprocal()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ndtr(self._standardize(value))


'''
Based on implementations in Scipy, Tensorflow Probability and issues in  
https://github.com/pytorch/pytorch/issues/52973#issuecomment-787587188
'''

# Lower boud values. Chosen by observing where support of ndtr appears to be zero. Then made more safe by expansion
LOGNDTR_FLOAT64_LOWER = -20.
LOGNDTR_FLOAT32_LOWER = -10.

# Upper bound values chosen by which value sof 'x' log[cdf(x)]=0, after which the appr Log[1-cdf(-x)] is used
LOGNDTR_FLOAT64_UPPER = 8.
LOGNDTR_FLOAT32_UPPER = 5.


def ndtr(value: torch.Tensor):
    """
    - Gaussian CDF
    - erfc(x) = 1 - erf(x)
    - erf(x) = 2/(\sqrt{\pi}) \int_0^{x}{e^{-t^2}dt}
    :param value: Value up to which the integral is calculated
    """
    sqrt_half = torch.sqrt(torch.tensor(0.5, dtype=value.dtype))
    x = value * sqrt_half
    z = torch.abs(x)
    y = 0.5 * torch.erfc(z)
    output = torch.where(z < sqrt_half, 0.5 + 0.5 * torch.erf(x), torch.where(x > 0, 1-y, y))

    return output


def log_ndtr(value: torch.tensor):
    """
    - Standard Gaussian log-cumulative distribution function
    - Based on TFP and SciPy implementations
    :param value: Value up to which the integral is calculated
    """
    dtype = value.dtype
    if dtype == torch.float64:
        lower, upper = LOGNDTR_FLOAT64_LOWER, LOGNDTR_FLOAT64_UPPER
    elif dtype == torch.float32:
        lower, upper = LOGNDTR_FLOAT32_LOWER, LOGNDTR_FLOAT32_UPPER
    else:
        raise TypeError(f'dtype={dtype} is not supported')

    # When x < lower, perform a fixed series asymptotic expansion,
    conditonal_inner = torch.where(value >= lower, torch.log(ndtr(value)), log_ndtr_series(value))
    output = torch.where(value > upper, -ndtr(-value), conditonal_inner)

    return output


def log_ndtr_series(value: torch.tensor, num_terms=3):
    """
    - Asymptotic series expansion of log of normal CDF at value
    - Based on SciPy implementation
    :param value: Integral up to value ofr CDF
    :param num_terms: Number of expansions
    """
    value_sq = value**2
    t1 = -0.5 * (np.log(2*np.pi) + value_sq) - torch.log(-value)
    t2 = torch.zeros_like(value)
    value_even_power = value_sq.clone()
    double_fac = 1
    multiplier = -1
    for n in range(1, num_terms+1):
        t2.add_(multiplier * double_fac / value_even_power)
        value_even_power.mul_(value_sq)
        double_fac *= (2 * n - 1)
        multiplier *= -1
    return t1 + torch.log1p(t2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.special as ss
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

    x = torch.linspace(-30, 10, 40000, dtype=torch.float32)
    out = log_ndtr(x)
    plt.plot(x.numpy(), abs(out.numpy() - ss.log_ndtr(x.numpy())), label='abs(PyTorch - SciPy) (float32)')
    plt.legend()
    plt.show()

    x = torch.linspace(-30, 10, 40000, dtype=torch.float64)
    out = log_ndtr(x)
    plt.plot(x.numpy(), abs(out.numpy() - ss.log_ndtr(x.numpy())), label='abs(PyTorch - SciPy) (float64)')
    plt.legend()
    plt.show()

    x = torch.linspace(-30, 10, 40000, dtype=torch.float32)
    plt.plot(x.numpy(), abs(torch.distributions.Normal(0, 1).cdf(x).numpy() - ss.ndtr(x.numpy())), label='Old')
    plt.plot(x.numpy(), abs(ndtr(x).numpy() - ss.ndtr(x.numpy())), label='New')
    plt.title('Float32')
    plt.legend()
    plt.show()

    x = torch.linspace(-30, 10, 40000, dtype=torch.float64)
    plt.plot(x.numpy(), abs(torch.distributions.Normal(0, 1).cdf(x).numpy() - ss.ndtr(x.numpy())), label='Old')
    plt.plot(x.numpy(), abs(ndtr(x).numpy() - ss.ndtr(x.numpy())), label='New')
    plt.title('Float64')
    plt.legend()
    plt.show()


