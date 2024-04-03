import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal


def cramer_torch_pointwise(pdf_target: torch.tensor, pdf_curr: torch.tensor, int_l, int_u, spacing, dev='cpu'):
    """
    - The integration limits should NOT be too far off form the lowest and highest point of the function!
    - Optional: Define an interval to focus on, in case of rapid
    - Implementation:
        1. Define the supports
        2. Calculate the difference squared
        3. Integrate over all dx
    """
    # Discretize for numerical integration
    diff = torch.abs(int_u - int_l)
    n_steps = torch.tensor(51, device=dev)
    delta_mb = diff / n_steps

    steps_idx = torch.arange(start=1, end=n_steps+1, step=1).to(dev)
    steps_tensor = steps_idx * delta_mb

    # Calculate Supports with correct stepsizes
    dx_mb = torch.ones((1, n_steps), device=dev) + steps_tensor

    dy_curr_mb = pdf_curr.cdf(dx_mb)
    dy_target_mb = pdf_target.cdf(dx_mb)
    cramer_pnt = torch.trapz(y=(dy_target_mb - dy_curr_mb)**2, x=dx_mb) + 1e-55
    cramer_pnt.sqrt_()
    # cramer_re = cramer_re.mean()

    return cramer_pnt, dx_mb


def inner_int(x, m, s):
    """
    - Necessary for approximation of the outer integral
    """
    # e_pow = torch.tensor(torch.e**(-0.5*(((dt-m)/s)**2)))
    # e_fac = ((dt - m)**2/s**2)
    # inner_int = torch.trapz(y=torch.trapz(y=(e_pow - e_fac * e_pow), x=dt), x=dt)
    # return inner_int
    return 0


def pCps(pdf_curr: torch.tensor, pdf_tar: torch.tensor, m: torch.tensor, s: torch.tensor, dev='cuda:0',
         invert=False):

    int_l = m - 4*s
    int_u = m + 4*s

    if invert:
        inv_fac = torch.tensor(-1, device=dev)
    else:
        inv_fac = torch.tensor(1, device=dev)
    c_loss, dt = cramer_torch_pointwise(pdf_curr=pdf_curr, pdf_target=pdf_tar, int_l=int_l, int_u=int_u, spacing=0.01,
                                        dev=dev)
    b = torch.tensor(0.5 * (1/torch.sqrt(c_loss)), device=dev)
    frac = (1 / (s**2 * torch.sqrt(2*torch.tensor(torch.pi)))).to(dev) * inv_fac

    outer_int = torch.trapz(y=inner_int, x=dt)

    res = 2*b*frac*outer_int

    return res


if __name__ == '__main__':
    # Mini batch size
    mb_size = 10
    device = 'cuda:0'

    '''
    Parameters
    '''
    # Curr
    mean_curr = 0.0
    std_curr = 1.0
    # Tar
    mean_tar = 10.0

    '''
    PDFs
    '''
    # Curr
    mu_curr = torch.ones(size=(mb_size, 1)).to(device) * torch.tensor([mean_curr], dtype=torch.float64).to(device)
    sigma_curr = torch.ones(size=(mb_size, 1)).to(device) * torch.tensor([std_curr], dtype=torch.float64).to(device)
    normal_curr = Normal(loc=mu_curr, scale=sigma_curr)
    # Target
    mu_tar = torch.ones(size=(mb_size, 1)).to(device) * torch.tensor([0.0], dtype=torch.float64).to(device)
    sigma_tar = torch.arange(start=1, end=mb_size+1, step=1).unsqueeze(dim=1).to(device)
    normal_tar = Normal(loc=mu_tar, scale=sigma_tar)

    inner_int_curr = pCps(pdf_curr=normal_curr, pdf_tar=normal_tar, m=mu_tar, s=sigma_tar, dev='cuda:0',
                          invert=False)

