import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

curr_path = os.getcwd()
main_dir = '/Learnable_Weights'
dir_name = 'Means_[2,10]_Stds_[1,1]_SUM_MB20_r2_C'
# loading_path = curr_path + '/' + main_dir + '/' + dir_name
loading_path = curr_path + '/' + dir_name

means_std = torch.load(loading_path + '/' + 'means_history_std').cpu().detach().numpy()
stds_std = torch.load(loading_path + '/' + 'stds_history_std').cpu().detach().numpy()
kweights_std = torch.load(loading_path + '/' + 'kweights_history_std').cpu().detach().numpy()
loss_std = torch.load(loading_path + '/' + 'cramer_loss_history_std').cpu().detach().numpy()
cdf_std = torch.load(loading_path + '/' + 'cdf_gmm_pred_std').cpu().detach().numpy()

cdf_ref = torch.load(loading_path + '/' + 'cdf_ref').cpu().detach().numpy()

'''
Graph quantities for standard cramer calculation
'''
# Means
mean_std_k1 = means_std[:, 0]
mean_std_k2 = means_std[:, 1]

plt.plot(mean_std_k1, label='mean_k1_stdC')
plt.plot(mean_std_k2, label='mean_k2_stdC')


plt.xlabel('Train iterations')
plt.ylabel('Kernel means')
plt.title('Change in Expectancy For Cramer Loss Calculation - Usual Numerical Approach')
plt.legend()
plt.show(block=False)

# Stds
plt.figure()
stds_stda_k1 = stds_std[:, 0]
stds_stda_k2 = stds_std[:, 1]

plt.plot(stds_stda_k1, label='std_k1_stdC')
plt.plot(stds_stda_k2, label='std_k2_stdC')

plt.xlabel('Train iterations')
plt.ylabel('Kernel Stds')
plt.title('Change in Std.Deviation For Cramer Loss Calculation - Usual Numerical Approach')
plt.legend()
plt.show(block=False)

# Kernel Weights
plt.figure()
kweight_std_k1 = kweights_std[:, 0]
kweight_std_k2 = kweights_std[:, 1]

plt.plot(kweight_std_k1, label='kweights_k1_stdC')
plt.plot(kweight_std_k2, label='kweights_k2_stdC')

plt.xlabel('Train iterations')
plt.ylabel('Kernel weights')
plt.title('Change in Kweights For Cramer Loss Calculation - Usual Numerical Approach')
plt.legend()
plt.show(block=False)

# Loss
plt.figure()
plt.plot(loss_std.squeeze(), label='stdC_loss')

plt.xlabel('Train iterations')
plt.ylabel('Cramer loss')
plt.title('Change in Cramer Loss - Usual Numerical Approach')
plt.legend()

plt.show(block=False)

# CDF Ref / CDF_Pred_Std / CDF_Pred_Imp
plt.figure()
plt.plot(cdf_ref, label='CDF_Ref')
plt.plot(cdf_std, label='CDF_Pred_Std')

plt.xlabel('Train iterations')
plt.ylabel('CDF')
plt.title('Fitted CDFs - Ref vs. Usual vs. Imp')
plt.legend()
plt.show(block=True)
