import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

curr_path = os.getcwd()
mod2 = True
# sub_dir = '/Learnable_Weights'
sub_dir = 'Gauss_Ref'
dir_name = 'Test_GaussRef_mod1_maxstd_10_diff_std_B'
loading_path = curr_path + '/' + sub_dir + '/' + dir_name

'''
Load Data
'''
cdf_ref = torch.load(loading_path + '/' + 'cdf_ref').cpu().numpy()
cdf_gauss_pred_ref = torch.load(loading_path + '/' + 'cdf_gauss_pred_ref').cpu().numpy()
cdf_tar = torch.load(loading_path + '/' + 'cdf_tar').cpu().numpy()
cdf_gauss_pred_tar = torch.load(loading_path + '/' + 'cdf_gauss_pred_tar').cpu().numpy()
kl_loss = torch.load(loading_path + '/' + 'kl_loss').cpu().numpy()

'''
Make CDF graphs of cdf_ref and cdf_gmm_pred_ref
'''
plt.plot(cdf_ref, label='cdf_ref')
plt.plot(cdf_gauss_pred_ref, label='cdf_gauss_pred_ref')

# Add labels and legend
plt.xlabel('Train iterations')
plt.ylabel('Cumulative Probability')
plt.title('CDF - Reference and Prediction of Reference')
plt.legend()

# Show the plot
plt.show(block=False)
plt.figure()

'''
Make CDF graphs of cdf_tar and cdf_gmm_pred_tar
'''
plt.plot(cdf_tar, label='cdf_tar')
plt.plot(cdf_gauss_pred_tar, label='cdf_gauss_pred_tar')

# Add labels and legend
plt.xlabel('Train iterations')
plt.ylabel('Cumulative Probability')
plt.title('CDF - Target and Prediction of Target from Reference Distribution')
plt.legend()

# Show the plot
plt.show(block=False)
plt.figure()

'''
Make Target loss graph
'''
plt.plot(kl_loss, label='KL Loss for target - Gauss')

# Add labels and legend
plt.xlabel('Train iterations')
plt.ylabel('KL Loss')
plt.title('KL Loss')
plt.legend()

# Show the plot
plt.show(block=True)


