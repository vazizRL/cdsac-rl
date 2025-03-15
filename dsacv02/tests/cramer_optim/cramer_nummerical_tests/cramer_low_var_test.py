import torch
import torch as t
import matplotlib.pyplot as plt
import os
from dsacv02.tools import cramer_optim_1k, get_normal_supports, generate_gauss_distr
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    DEVICE = 'cuda:0'
    RANGE_LOW, RANGE_HIGH = 0.0001, 0.01
    MEAN_CURR, MEAN_TAR, STD_TAR = torch.as_tensor([[0]], device=DEVICE), torch.as_tensor([[10]], device=DEVICE), \
                                   torch.as_tensor([[0.01]], device=DEVICE)
    BATCH_SIZE = 256
    N_SUPPORTS, IBF = 32, 15
    STANDARD_SUPP = get_normal_supports(batch_size=BATCH_SIZE, n_kernels=1, n_supp=N_SUPPORTS,
                                        integral_bound_factor=IBF, dev=DEVICE)
    STEPS = 101
    var_range = t.linspace(start=RANGE_HIGH, end=RANGE_LOW, steps=STEPS)
    # Shape: (BATCH, 1)
    distr_target = generate_gauss_distr(means=MEAN_TAR, stds=STD_TAR, kweights=None, multivar=False)

    cramer_distances = []
    for var_i in var_range:
        std_i = torch.as_tensor([[var_i]], device=DEVICE)
        distr_curr = generate_gauss_distr(means=MEAN_CURR, stds=std_i, kweights=None, multivar=False)
        cramer_i = cramer_optim_1k(pdf_target=distr_target, pdf_curr=distr_curr, n_kernels=1,
                                   standard_supp=STANDARD_SUPP, dev=DEVICE)
        print(cramer_i)
        cramer_distances.append(cramer_i.cpu().item())

    x_axis = var_range.cpu().tolist()
    plt.plot(x_axis, cramer_distances)
    plt.title(f'CURR ({MEAN_CURR.cpu().item()}, VAR) and TAR ({MEAN_TAR.cpu().item(), STD_TAR.cpu().item()})')
    plt.xlabel('Standard Deviation')
    plt.grid(visible=True, which='both', color='grey', linewidth=0.3)
    # plt.xlim((RANGE_HIGH, RANGE_LOW))
    plt.ylabel(f'Cramer Distance')
    plt.show(block=True)

    print('Finished Printing Cramer Loss Calculations')
