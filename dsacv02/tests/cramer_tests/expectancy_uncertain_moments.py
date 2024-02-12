"""
These epxeriments are performed to observe whether convergence happens to means if
stds of targets are noisy / random
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distr
import os
import time
from scipy import integrate
from dsacv02.gmm_reparameterization.mixture_same_family import ReparameterizedMixtureSameFamilyMod as RMM
from dsacv02.mlp_gmm import MLPGMM, MLPGMMWeighted
from copy import deepcopy


def cramer_py_test(pdf_target: torch.tensor, pdf_curr: torch.tensor, int_l, int_u, spacing, dev='cpu', points=None):
    """
    - The integration limits should NOT be too far off form the lowest and highest point of the function!
    - Optional: Define an interval to focus on, in case of rapid
    - Implementation:
        1. Define the supports
        2. Calculate the difference squred
        3. Integrate over all dx
    """
    # Discretize for numerical integration
    steps = int((int_u - int_l) / spacing)
    dx = torch.linspace(int_l, int_u, steps=steps).to(dev)

    dy_curr_cdf = pdf_curr.cdf(dx)
    dy_target_cdf = pdf_target.cdf(dx)

    cramer = torch.trapz((dy_target_cdf - dy_curr_cdf)**2, dx=spacing)

    return cramer**0.5