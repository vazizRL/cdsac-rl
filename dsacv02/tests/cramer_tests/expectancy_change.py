"""
Testing effect of updating with high-entrpoy vs. low-entropy target
"""
import torch
import torch.optim as optim
import torch.distributions as distr
import os
import time
from scipy import integrate
from dsacv02.gmm_reparameterization.mixture_same_family import ReparameterizedMixtureSameFamilyMod as RMM
from dsacv02.mlp_gmm import MLPGMM, MLPGMMWeighted
from copy import deepcopy


def cramer_from_pdf(pdf_target: torch.tensor, pdf_curr: torch.tensor, int_l, int_u, points=(-100, 100),
                    dev='cuda:0'):
    """
    - The integration limits should NOT be too far off form the lowest and highest point of the function!
    """
    c: tuple = integrate.quad(
        lambda x: (pdf_target.cdf(torch.tensor([x]).to(dev)) - pdf_curr.cdf(torch.tensor([x]).to(dev))) ** 2,
        int_l, int_u, points=points
    )

    distance, error_est = c

    return distance**0.5, error_est


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


def generate_gmm(locs: torch.tensor, scales: torch.tensor, kweights: torch.tensor):
    # Test mysterious symmetry
    gmm = RMM(distr.Categorical(probs=kweights), distr.Normal(locs, scales))

    return gmm


def calculate_cdf(pdf: torch.tensor, supp_l, supp_u, spacing, dev='cpu'):
    steps = int((supp_u - supp_l) / spacing)
    dx = torch.linspace(supp_l, supp_u, steps=steps).to(dev)

    cdf_curve = pdf.cdf(dx)

    return cdf_curve


if __name__ == '__main__':
    curr_path = os.getcwd()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # Reference distribution
    mean_ref = torch.tensor([0.0, 1.0], dtype=torch.float64).to(device)
    std_ref = torch.tensor([3.0, 3.0], dtype=torch.float64).to(device)
    kweight_ref = torch.tensor([0.5, 0.5], dtype=torch.float64).to(device)
    distr_ref = generate_gmm(locs=mean_ref, scales=std_ref, kweights=kweight_ref)

    # High-entropy Distribution
    mean_high_e = torch.tensor([2.0, 15.0], dtype=torch.float64).to(device)
    std_high_e = torch.tensor([3.0, 3.0], dtype=torch.float64).to(device)
    kweight_high_e = torch.tensor([0.5, 0.5], dtype=torch.float64).to(device)
    distr_high_e = generate_gmm(locs=mean_high_e, scales=std_high_e, kweights=kweight_high_e)

    # Low-entropy Distribution
    mean_low_e = torch.tensor([2.0, 15.0], dtype=torch.float64).to(device)
    std_low_e = torch.tensor([1.0, 1.0], dtype=torch.float64).to(device)
    kweight_low_e = torch.tensor([0.5, 0.5], dtype=torch.float64).to(device)
    distr_low_e = generate_gmm(locs=mean_low_e, scales=std_low_e, kweights=kweight_low_e)

    # # Calculate Distances
    # cramer_high_e = cramer_from_pdf(pdf_target=distr_high_e, pdf_curr=distr_ref, int_l=-8, int_u=20, points=(-6, 20))
    # cramer_low_e = cramer_from_pdf(pdf_target=distr_low_e, pdf_curr=distr_ref, int_l=-6, int_u=20, points=(-3, 15))

    # Network parameters
    arch = (1, 10, 1)
    activ = ('gelu',)
    n_kernels = 2
    multivar = True
    learnable_weights = True
    kweights_fix = torch.ones(n_kernels, dtype=torch.float64) / n_kernels
    kweights_fix.to(device)

    # Train parameters
    learning_rates = 0.001
    epochs = 4
    epochs_low_e = 1
    epochs_high_e = 1

    # Initialize network and optimizer
    if learnable_weights:
        gmm_approx = MLPGMMWeighted(arch=arch, activ=activ, n_kernels=n_kernels, device=device, multivar=multivar)
    else:
        gmm_approx = MLPGMM(arch=arch, activ=activ, n_kernels=n_kernels, device=device, multivar=multivar)
    optimizer_ref = optim.Adam(gmm_approx.parameters(), lr=learning_rates)

    # Initialize input
    n_datapoints = 6000
    mb_size = 20
    n_mb = int(n_datapoints / mb_size)
    input_total = torch.randn(size=(n_datapoints, 1)).to(device)
    batches = input_total.view(n_mb, mb_size, 1)

    # Fit gmm_approx to output normal distributio
    for epoch in range(epochs):
        for batch in batches:
            for data_i in batch:
                # kweights = None
                data_i.unsqueeze_(dim=1)
                pred_means, pred_stds, kweights = gmm_approx(data_i)
                pred_stds.abs_()
                if learnable_weights:
                    pred_gmm_i = generate_gmm(locs=pred_means.squeeze(), scales=pred_stds.squeeze(),
                                              kweights=kweights.squeeze())
                else:
                    pred_gmm_i = generate_gmm(locs=pred_means.squeeze(), scales=pred_stds.squeeze(),
                                              kweights=kweights_fix)
                # Loss data_i
                cramer_py_loss_i = cramer_py_test(pdf_target=distr_ref, pdf_curr=pred_gmm_i, int_l=-10, int_u=10,
                                                  spacing=0.01, points=(-5, 5), dev=device)
                # cramer_own_loss_i = cramer_from_pdf(pdf_target=distr_ref, pdf_curr=pred_gmm_i, int_l=-10, int_u=10,
                #                              points=(-5, 5))
                # print(f'cramer_py: {cramer_py_loss_i}\n cramer_own: {cramer_own_loss_i}')

                # Loss in SGD for cramer_py_loss_i
                optimizer_ref.zero_grad()
                cramer_py_loss_i.backward()
                optimizer_ref.step()
        print(f'Finished episode: {epoch + 1}')

    # Test if trained correctly by probing mean of outputs: pred_means and pred_stds
    preds_means_all, preds_stds_all, _ = gmm_approx(input_total)
    print(f'The mean of the two kernels for the expectancy is: {preds_means_all.mean(dim=0)}')
    print(f'The mean of the two kernels for the std is: {preds_stds_all.mean(dim=0)}')

    # Evaluate CDFs for: Ref, Low_H, High_H
    cdf_pre = calculate_cdf(pred_gmm_i, supp_l=-8, supp_u=12, spacing=0.001, dev=device)
    cdf_low_target = calculate_cdf(distr_low_e, supp_l=-8, supp_u=12, spacing=0.001, dev=device)
    cdf_high_target = calculate_cdf(distr_high_e, supp_l=-8, supp_u=12, spacing=0.001, dev=device)

    '''
    Deep copy the trained network (trained for means=[1.0, 0.0], stds=[1.0, 1.0])
    '''
    gmm_approx_low_e = deepcopy(gmm_approx)
    gmm_approx_high_e = deepcopy(gmm_approx)

    '''
    #
    Measure change for low entropy target distribution
    #
    '''
    optimizer_low = optim.Adam(gmm_approx_low_e.parameters(), lr=learning_rates)
    means_history_low_e_target = torch.tensor([], dtype=torch.float64, device=device)
    stds_history_low_e_target = torch.tensor([], dtype=torch.float64, device=device)
    dc_history_low_e_target = torch.tensor([], dtype=torch.float64, device=device)
    kweights_history_low_e_target = torch.tensor([], dtype=torch.float64, device=device)
    for epoch in range(epochs_low_e):
        for batch in batches:
            means_history_batch_low_e = torch.tensor([], dtype=torch.float64, device=device)
            stds_history_batch_low_e = torch.tensor([], dtype=torch.float64, device=device)
            dc_history_batch_low_e = torch.tensor([], dtype=torch.float64, device=device)
            kweights_history_batch_low_e = torch.tensor([], dtype=torch.float64, device=device)
            for data_i in batch:
                # kweights = None
                data_i.unsqueeze_(dim=1)
                pred_means_low_e, pred_stds_low_e, kweights_low_e = gmm_approx_low_e(data_i)
                # Log preds
                means_history_batch_low_e = torch.cat((means_history_batch_low_e, pred_means_low_e), dim=0)
                stds_history_batch_low_e = torch.cat((stds_history_batch_low_e, pred_stds_low_e), dim=0)
                kweights_history_batch_low_e = torch.cat((kweights_history_batch_low_e, kweights_low_e), dim=0)
                pred_stds_low_e.abs_()
                if learnable_weights:
                    pred_gmm_low_i = generate_gmm(locs=pred_means_low_e.squeeze(), scales=pred_stds_low_e.squeeze(),
                                                  kweights=kweights_low_e.squeeze())
                else:
                    pred_gmm_low_i = generate_gmm(locs=pred_means_low_e.squeeze(), scales=pred_stds_low_e.squeeze(),
                                                  kweights=kweights_fix)
                cramer_py_loss_low_i = cramer_py_test(pdf_target=distr_low_e, pdf_curr=pred_gmm_low_i, int_l=-10,
                                                      int_u=10, spacing=0.01, points=(-5, 5), dev=device)
                # Log loss
                dc_history_batch_low_e = torch.cat((dc_history_batch_low_e,
                                                    cramer_py_loss_low_i.unsqueeze(dim=0).unsqueeze(dim=1)), dim=0)

                optimizer_low.zero_grad()
                cramer_py_loss_low_i.backward()
                optimizer_low.step()

            # Log mean and std change
            means_history_low_e_target = torch.cat((means_history_low_e_target, means_history_batch_low_e), dim=0)
            stds_history_low_e_target = torch.cat((stds_history_low_e_target, stds_history_batch_low_e), dim=0)
            kweights_history_low_e_target = torch.cat((kweights_history_low_e_target, kweights_history_batch_low_e),
                                                      dim=0)
            dc_history_low_e_target = torch.cat((dc_history_low_e_target, dc_history_batch_low_e), dim=0)
        print(f'Finished episode low entropy: {epoch + 1}')

    '''
    Measure change for high entropy target distribution
    '''
    optimizer_high = optim.Adam(gmm_approx_high_e.parameters(), lr=learning_rates)
    means_history_high_e_target = torch.tensor([], dtype=torch.float64, device=device)
    stds_history_high_e_target = torch.tensor([], dtype=torch.float64, device=device)
    dc_history_high_e_target = torch.tensor([], dtype=torch.float64, device=device)
    kweights_history_high_e_target = torch.tensor([], dtype=torch.float64, device=device)
    for epoch in range(epochs_high_e):
        for batch in batches:
            means_history_batch_high_e = torch.tensor([], dtype=torch.float64, device=device)
            stds_history_batch_high_e = torch.tensor([], dtype=torch.float64, device=device)
            dc_history_batch_high_e = torch.tensor([], dtype=torch.float64, device=device)
            kweights_history_batch_high_e = torch.tensor([], dtype=torch.float64, device=device)
            for data_i in batch:
                # kweights = None
                data_i.unsqueeze_(dim=1)
                pred_means_high_e, pred_stds_high_e, kweights_high_e = gmm_approx_high_e(data_i)
                # Log preds
                means_history_batch_high_e = torch.cat((means_history_batch_high_e, pred_means_high_e), dim=0)
                stds_history_batch_high_e = torch.cat((stds_history_batch_high_e, pred_stds_high_e), dim=0)
                kweights_history_batch_high_e = torch.cat((kweights_history_batch_high_e, kweights_high_e), dim=0)
                pred_stds_high_e.abs_()
                if learnable_weights:
                    pred_gmm_high_i = generate_gmm(locs=pred_means_high_e.squeeze(), scales=pred_stds_high_e.squeeze(),
                                                   kweights=kweights_high_e.squeeze())
                else:
                    pred_gmm_high_i = generate_gmm(locs=pred_means_high_e.squeeze(), scales=pred_stds_high_e.squeeze(),
                                                   kweights=kweights_fix)

                cramer_py_loss_high_i = cramer_py_test(pdf_target=distr_high_e, pdf_curr=pred_gmm_high_i, int_l=-10,
                                                       int_u=12, spacing=0.01, points=(-5, 5), dev=device)
                # Log loss
                dc_history_batch_high_e = torch.cat((dc_history_batch_high_e,
                                                     cramer_py_loss_high_i.unsqueeze(dim=0).unsqueeze(dim=1)), dim=0)

                optimizer_high.zero_grad()
                cramer_py_loss_high_i.backward()
                optimizer_high.step()

            # Log mean and std change
            means_history_high_e_target = torch.cat((means_history_high_e_target, means_history_batch_high_e), dim=0)
            stds_history_high_e_target = torch.cat((stds_history_high_e_target, stds_history_batch_high_e), dim=0)
            dc_history_high_e_target = torch.cat((dc_history_high_e_target, dc_history_batch_high_e), dim=0)
            kweights_history_high_e_target = torch.cat((kweights_history_high_e_target, kweights_history_batch_high_e),
                                                       dim=0)
        print(f'Finished episode high entropy: {epoch + 1}')

    '''
    Calculate CDFs: Retro-Fitted GMM for low and high entropy
    '''
    cdf_low_post = calculate_cdf(pred_gmm_low_i, supp_l=-8, supp_u=12, spacing=0.001, dev=device)
    cdf_high_post = calculate_cdf(pred_gmm_high_i, supp_l=-8, supp_u=12, spacing=0.001, dev=device)

    '''
    Save Data
    '''
    ts = time.time()
    save_path = curr_path + '/' + str(ts)
    os.mkdir(save_path)

    # Save the logged tensor
    torch.save(means_history_low_e_target, save_path + '/' + 'means_history_low')
    torch.save(stds_history_low_e_target, save_path + '/' + 'stds_history_low')
    torch.save(kweights_history_low_e_target, save_path + '/' + 'kweights_history_low')
    torch.save(dc_history_low_e_target, save_path + '/' + 'dc_history_low')

    torch.save(means_history_high_e_target, save_path + '/' + 'means_history_high')
    torch.save(stds_history_high_e_target, save_path + '/' + 'stds_history_high')
    torch.save(kweights_history_high_e_target, save_path + '/' + 'kweights_history_high')
    torch.save(dc_history_high_e_target, save_path + '/' + 'dc_history_high')

    # Save CDFs
    torch.save(cdf_pre, save_path + '/' + 'cdf_ref')
    torch.save(cdf_low_target, save_path + '/' + 'cdf_low_target')
    torch.save(cdf_high_target, save_path + '/' + 'cdf_high_target')

    torch.save(cdf_low_post, save_path + '/' + 'cdf_low_post')
    torch.save(cdf_high_post, save_path + '/' + 'cdf_high_post')



