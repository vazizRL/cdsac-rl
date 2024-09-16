import torch
from dsacv02.tools import cramer_optim_1k, get_normal_supports

if __name__ == '__main__':
    mean_curr = torch.tensor([[0.0]])
    std_curr = torch.tensor([[0.01]])
    mean_tar = torch.tensor([[100.0]])
    std_tar = torch.tensor([[0.01]])

    supp = get_normal_supports(batch_size=1, n_kernels=1,
                        n_supp=31, integral_bound_factor=15, dev='cpu')
    distr_curr = torch.distributions.Normal(loc=mean_curr, scale=std_curr)
    distr_tar = torch.distributions.Normal(loc=mean_tar, scale=std_tar)

    cr_dis = cramer_optim_1k(pdf_target=distr_tar, pdf_curr=distr_curr, standard_supp=supp)
    print(cr_dis)
