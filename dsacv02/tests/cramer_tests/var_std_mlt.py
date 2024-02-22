import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

curr_path = os.getcwd()
mod2 = True
sub_dir = 'Tests'
dir_name = 'VarStd_1.0-6.0_Mod1_NoisyQ_ini_r3'
loading_path = curr_path + '/' + sub_dir + '/' + dir_name

'''
Load Data
'''
# Load CDF
cdf_ref = torch.load(loading_path + '/' + 'cdf_ref').cpu().numpy()
cdf_gmm_pred_ref = torch.load(loading_path + '/' + 'cdf_gmm_pred_ref').cpu().numpy()
# Load Loss
cramer_loss_history_tar = torch.load(loading_path + '/' + 'cramer_loss_history_tar').cpu().numpy()
# Load Means, Stds and Weights
means_history_tar = torch.load(loading_path + '/' + 'means_history_tar').cpu().numpy()
stds_history_tar = torch.load(loading_path + '/' + 'stds_history_tar').cpu().numpy()
kweights_history_tar = torch.load(loading_path + '/' + 'kweights_history_tar').cpu().numpy()

'''
Graph Reference 
'''
plt.plot(cdf_ref, label='cdf_ref')
plt.plot(cdf_gmm_pred_ref, label='cdf_gmm_pred_ref')

# Add labels and legend
plt.xlabel('Train iterations')
plt.ylabel('P <= X')
plt.title('CDF vs. Fitted CDF - Reference')
plt.legend()

# Show the plot
plt.show(block=False)
plt.figure()

'''
Graph Cramer Loss
'''
plt.plot(cramer_loss_history_tar, label='cramer_loss_tar')

# Add labels and legend
plt.xlabel('Train iterations')
plt.ylabel('Cramer Loss')
plt.title('Cramer Loss for Fitting to Target from Reference')
plt.legend()

# Show the plot
plt.show(block=False)
plt.figure()

'''
Graph Change in Means
'''
means_k1 = means_history_tar[:, 0]
means_k2 = means_history_tar[:, 1]

plt.plot(means_k1, label='mean_k1')
plt.plot(means_k2, label='mean_k2')

# Add labels and legend
plt.xlabel('Train iterations')
plt.ylabel('Kernel means')
plt.title('Change in Expectancy per Kernel')
plt.legend()
# Show the plot
plt.show(block=False)
plt.figure()

'''
Graph Change in Stds
'''
stds_k1 = stds_history_tar[:, 0]
stds_k2 = stds_history_tar[:, 1]

plt.plot(stds_k1, label='std_k1')
plt.plot(stds_k2, label='std_k2')

# Add labels and legend
plt.xlabel('Train iterations')
plt.ylabel('Kernel stds')
plt.title('Change in Std per Kernel')
plt.legend()
# Show the plot
plt.show(block=False)
plt.figure()

'''
Graph Change in Kernel Weights
'''
weight_k1 = kweights_history_tar[:, 0]
weight_k2 = kweights_history_tar[:, 1]

plt.plot(weight_k1, label='weight_k1')
plt.plot(weight_k2, label='weight_k2')

# Add labels and legend
plt.xlabel('Train iterations')
plt.ylabel('Kernel Weights')
plt.title('Change in Weight of Kernels')
plt.legend()
# Show the plot
plt.show(block=True)
plt.figure()

