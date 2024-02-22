import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

curr_path = os.getcwd()
main_dir = '/Learnable_Weights'
dir_name = 'r19_mod3_H'
loading_path = curr_path + '/' + main_dir + '/' + dir_name
# loading_path = curr_path + '/' + dir_name

means_history_low = torch.load(loading_path + '/' + 'means_history_low').cpu().detach().numpy()
stds_history_low = torch.load(loading_path + '/' + 'stds_history_low').cpu().detach().numpy()
kweights_history_low = torch.load(loading_path + '/' + 'kweights_history_low').cpu().detach().numpy()

means_history_high = torch.load(loading_path + '/' + 'means_history_high').cpu().detach().numpy()
stds_history_high = torch.load(loading_path + '/' + 'stds_history_high').cpu().detach().numpy()
kweights_history_high = torch.load(loading_path + '/' + 'kweights_history_high').cpu().detach().numpy()

dc_history_low = torch.load(loading_path + '/' + 'dc_history_low').cpu().detach().numpy()
dc_history_high = torch.load(loading_path + '/' + 'dc_history_high').cpu().detach().numpy()

cdf_ref = torch.load(loading_path + '/' + 'cdf_ref').cpu().detach().numpy()
cdf_low_target = torch.load(loading_path + '/' + 'cdf_low_target').cpu().detach().numpy()
cdf_high_target = torch.load(loading_path + '/' + 'cdf_high_target').cpu().detach().numpy()
cdf_low_post = torch.load(loading_path + '/' + 'cdf_low_post').cpu().detach().numpy()
cdf_high_post = torch.load(loading_path + '/' + 'cdf_high_post').cpu().detach().numpy()

'''
Change in low entropy means
'''
mean_low_k1 = means_history_low[:, 0]
mean_low_k2 = means_history_low[:, 1]

plt.plot(mean_low_k1, label='low_k1')
plt.plot(mean_low_k2, label='low_k2')

# Add labels and legend
plt.xlabel('Train iterations')
plt.ylabel('Kernel means')
plt.title('Change in expectancy')
plt.legend()

# Show the plot
plt.show(block=False)

'''
Change in high entropy means
'''
mean_high_k1 = means_history_high[:, 0]
mean_high_k2 = means_history_high[:, 1]

plt.plot(mean_high_k1, label='high_k1')
plt.plot(mean_high_k2, label='high_k2')

# Add labels and legend
plt.xlabel('Train iterations')
plt.ylabel('Kernel means')
plt.title('Change in expectancy')
plt.legend()

# Show the plot
plt.show(block=False)

'''
Change in low entropy stds
'''
plt.figure()
std_low_k1 = stds_history_low[:, 0]
std_low_k2 = stds_history_low[:, 1]

plt.plot(std_low_k1, label='std_low_k1')
plt.plot(std_low_k2, label='std_low_k2')

# Add labels and legend
plt.xlabel('Train iterations')
plt.ylabel('Kernel Stds')
plt.title('Change in std')
plt.legend()

# Show the plot
plt.show(block=False)

'''
Change in high entropy stds
'''
std_high_k1 = stds_history_high[:, 0]
std_high_k2 = stds_history_high[:, 1]

plt.plot(std_high_k1, label='std_high_k1')
plt.plot(std_high_k2, label='std_high_k2')

# Add labels and legend
plt.xlabel('Train iterations')
plt.ylabel('Kernel Stds')
plt.title('Change in std')
plt.legend()

# Show the plot
plt.show(block=False)

'''
Change in cramer loss per iteration
'''
plt.figure()
plt.plot(dc_history_low, label='dc_low')
plt.plot(dc_history_high, label='dc_high')

# Add labels and legend
plt.xlabel('Train iterations')
plt.ylabel('Cramer loss')
plt.title('Change in Cramer Loss')
plt.legend()

# Show the plot
plt.show(block=False)

'''
Change in kernel weights per iteration
'''
k1_low = kweights_history_low[:, 0]
k2_low = kweights_history_low[:, 1]
k1_high = kweights_history_high[:, 0]
k2_high = kweights_history_high[:, 1]
plt.figure()
plt.plot(k1_low, label='kweights1_low')
plt.plot(k2_low, label='kweights2_low')
plt.plot(k1_high, label='kweights1_high')
plt.plot(k2_high, label='kweights2_high')

# Add labels and legend
plt.xlabel('Train iterations')
plt.ylabel('Kernel Weight')
plt.title('Kernel Weighting')
plt.legend()

# Show the plot
plt.show(block=False)

'''
Show CDFs pre training
'''
plt.figure()
plt.plot(cdf_ref, label='cdf_ref')
plt.plot(cdf_low_target, label='cdf_low_target')
plt.plot(cdf_high_target, label='cdf_high_target')

# Add labels and legend
plt.xlabel('Supports')
plt.ylabel('Probability')
plt.title('CDF: Reference with Targets')
plt.legend()

# Show the plot
plt.show(block=False)

'''
Show Low-H CDF post training
'''
plt.figure()
plt.plot(cdf_low_post, label='cdf_low_post')
plt.plot(cdf_low_target, label='cdf_low_target')

# Add labels and legend
plt.xlabel('Supports')
plt.ylabel('Probability')
plt.title('CDF Retro-Fitted on Low-H Target')
plt.legend()

# Show the plot
plt.show(block=False)

'''
Show CDFs post training
'''
plt.figure()
plt.plot(cdf_high_post, label='cdf_high_post')
plt.plot(cdf_high_target, label='cdf_high_target')

# Add labels and legend
plt.xlabel('Supports')
plt.ylabel('Probability')
plt.title('CDF Retro-Fitted on High-H Target')
plt.legend()

# Show the plot
plt.show()



