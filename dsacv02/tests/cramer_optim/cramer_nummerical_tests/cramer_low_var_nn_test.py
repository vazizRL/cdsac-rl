import torch
import torch as t
import matplotlib.pyplot as plt
import os
from dsacv02.tools import cramer_optim_1k, get_normal_supports, generate_gauss_distr
from dsacv02.mlp_gmm import MLPGMM
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == '__main__':
    ARCH = (16, 256, 256, 1)
    ACTIVE = ('relu', 'relu')

    BATCH_SIZE = 256
    OBS_N, OBS_LOW, OBS_HIGH = 27, -5, 10

    # Input: Ant: Low, High = -inf, inf
    # Mean 0, std: 85% 0.5, 15% 2.5, progress: std 1.44, mean slightly positive 0.1
    # As training progresses, mean gets bigger and std more volatile; Mean: 0.23 Std: 1.66

    inp = t.rand(size=(BATCH_SIZE, OBS_N))

