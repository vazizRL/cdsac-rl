import torch
import matplotlib.pyplot as plt
import os
from dsacv02.tools import smoothing
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

one_kernel = True
curr_path = os.getcwd()
main_dir = '/VAR_REF_OPTI'
dir_name = 'test'
loading_path = curr_path + '/' + main_dir + '/' + dir_name
# loading_path = curr_path + '/' + dir_name
smoothing_weight_means = 0.85

# Load low-low Values
means_history_low_low = torch.load(loading_path + '/' + 'means_history_low_low').cpu().detach().numpy()
means_history_low_low, _, _ = smoothing(scalars=means_history_low_low, weight=smoothing_weight_means)
means_history_low_low = torch.as_tensor(means_history_low_low)
stds_history_low_low = torch.load(loading_path + '/' + 'stds_history_low_low').cpu().detach().numpy()
kweights_history_low_low = torch.load(loading_path + '/' + 'kweights_history_low_low').cpu().detach().numpy()
dc_history_low_low = torch.load(loading_path + '/' + 'dc_history_low_low').cpu().detach().numpy()
# Load high-low Values
means_history_high_low = torch.load(loading_path + '/' + 'means_history_high_low').cpu().detach().numpy()
means_history_high_low, _, _ = smoothing(scalars=means_history_high_low, weight=smoothing_weight_means)
means_history_high_low = torch.as_tensor(means_history_high_low)
stds_history_high_low = torch.load(loading_path + '/' + 'stds_history_high_low').cpu().detach().numpy()
kweights_history_high_low = torch.load(loading_path + '/' + 'kweights_history_high_low').cpu().detach().numpy()
dc_history_high_low = torch.load(loading_path + '/' + 'dc_history_high_low').cpu().detach().numpy()

# Load low-high
means_history_low_high = torch.load(loading_path + '/' + 'means_history_low_high').cpu().detach().numpy()
means_history_low_high, _, _ = smoothing(scalars=means_history_low_high, weight=smoothing_weight_means)
means_history_low_high = torch.as_tensor(means_history_low_high)
stds_history_low_high = torch.load(loading_path + '/' + 'stds_history_low_high').cpu().detach().numpy()
kweights_history_low_high = torch.load(loading_path + '/' + 'kweights_history_low_high').cpu().detach().numpy()
dc_history_low_high = torch.load(loading_path + '/' + 'dc_history_low_high').cpu().detach().numpy()
# Load high-high
means_history_high_high = torch.load(loading_path + '/' + 'means_history_high_high').cpu().detach().numpy()
means_history_high_high, _, _ = smoothing(scalars=means_history_high_high, weight=smoothing_weight_means)
means_history_high_high = torch.as_tensor(means_history_high_high)
stds_history_high_high = torch.load(loading_path + '/' + 'stds_history_high_high').cpu().detach().numpy()
kweights_history_high_high = torch.load(loading_path + '/' + 'kweights_history_high_high').cpu().detach().numpy()
dc_history_high_high = torch.load(loading_path + '/' + 'dc_history_high_high').cpu().detach().numpy()

# Load Ref CDFs
cdf_ref_low = torch.load(loading_path + '/' + 'cdf_ref_low').cpu().detach().numpy()
cdf_ref_high = torch.load(loading_path + '/' + 'cdf_ref_high').cpu().detach().numpy()
# Load Target CDFs
cdf_low_target = torch.load(loading_path + '/' + 'cdf_low_target').cpu().detach().numpy()
cdf_high_target = torch.load(loading_path + '/' + 'cdf_high_target').cpu().detach().numpy()
# Load CDF Preds for all 4 cases
cdf_low_low = torch.load(loading_path + '/' + 'cdf_low_low_post').cpu().detach().numpy()
cdf_high_low = torch.load(loading_path + '/' + 'cdf_high_low_post').cpu().detach().numpy()
cdf_low_high = torch.load(loading_path + '/' + 'cdf_low_high_post').cpu().detach().numpy()
cdf_high_high = torch.load(loading_path + '/' + 'cdf_high_high_post').cpu().detach().numpy()


'''
Changes in means for l-l and l-h
'''
mean_low_low_k1 = means_history_low_low[:, 0]
mean_low_high_k1 = means_history_low_high[:, 0]

if one_kernel:
    plt.plot(mean_low_low_k1, label='low_low_k1')
    plt.plot(mean_low_high_k1, label='low_high_k1', alpha=0.8)
else:
    mean_low_high_k2 = means_history_low_high[:, 1]
    mean_low_low_k2 = means_history_low_low[:, 1]
    plt.plot(mean_low_low_k1, label='low_low_k1')
    plt.plot(mean_low_high_k1, label='low_high_k1')
    plt.plot(mean_low_low_k2, label='low_low_k2')
    plt.plot(mean_low_high_k2, label='low_high_k2')

# Add labels and legend
plt.xlabel('Train Iterations')
plt.ylabel('Kernel Means')
plt.title('Change in Expectancy - From Low-E Ref.')
plt.legend()
# Show the plot
plt.savefig(loading_path + '/' + 'Means_From_L.png')
plt.show(block=False)

'''
Changes in Means for h-l and h-h
'''
plt.figure()
mean_high_low_k1 = means_history_high_low[:, 0]
mean_high_high_k1 = means_history_high_high[:, 0]

if one_kernel:
    plt.plot(mean_high_low_k1, label='high_low_k1')
    plt.plot(mean_high_high_k1, label='high_high_k1', alpha=0.8)
else:
    mean_high_low_k2 = means_history_high_low[:, 1]
    mean_high_high_k2 = means_history_high_high[:, 1]
    plt.plot(mean_high_low_k1, label='high_low_k1')
    plt.plot(mean_high_high_k1, label='high_high_k1')
    plt.plot(mean_high_low_k2, label='high_low_k2')
    plt.plot(mean_high_high_k2, label='high_high_k2')

# Add labels and legend
plt.xlabel('Train Iterations')
plt.ylabel('Kernel Means')
plt.title('Change in Expectancy - From High-E Ref.')
plt.legend()
# Show the plot
plt.savefig(loading_path + '/' + 'Means_From_H.png')
plt.show(block=False)

'''
Changes Stds for l-l and l-h
'''
plt.figure()
stds_low_low_k1 = stds_history_low_low[:, 0]
stds_low_high_k1 = stds_history_low_high[:, 0]

if one_kernel:
    plt.plot(stds_low_low_k1, label='low_low_k1')
    plt.plot(stds_low_high_k1, label='low_high_k1', alpha=0.8)
else:
    stds_low_low_k2 = stds_history_low_low[:, 1]
    stds_low_high_k2 = stds_history_low_high[:, 1]
    plt.plot(stds_low_low_k1, label='low_low_k1')
    plt.plot(stds_low_low_k2, label='low_low_k2')
    plt.plot(stds_low_high_k1, label='low_high_k1')
    plt.plot(stds_low_high_k2, label='low_high_k2')

# Add labels and legend
plt.xlabel('Train Iterations')
plt.ylabel('Kernel Stds')
plt.title('Change in Stds - From Low-E Ref.')
plt.legend()
# Show the plot
plt.savefig(loading_path + '/' + 'STDs_From_L.png')
plt.show(block=False)

'''
Changes Stds for h-l and h-h
'''
plt.figure()
stds_high_low_k1 = stds_history_high_low[:, 0]
stds_high_high_k1 = stds_history_high_high[:, 0]

if one_kernel:
    plt.plot(stds_high_low_k1, label='high_low_k1')
    plt.plot(stds_high_high_k1, label='high_high_k1', alpha=0.8)
else:
    stds_high_low_k2 = stds_history_high_low[:, 1]
    stds_high_high_k2 = stds_history_high_high[:, 1]
    plt.plot(stds_high_low_k1, label='high_low_k1')
    plt.plot(stds_high_low_k2, label='high_low_k2')
    plt.plot(stds_high_high_k1, label='high_high_k1')
    plt.plot(stds_high_high_k2, label='high_high_k2')

# Add labels and legend
plt.xlabel('Train Iterations')
plt.ylabel('Kernel Stds')
plt.title('Change in Stds - From High-E Ref.')
plt.legend()
# Show the plot
plt.savefig(loading_path + '/' + 'STDs_From_H.png')
plt.show(block=False)

'''
Changes Kweights for l-l and l-h
'''
if not one_kernel:
    plt.figure()
    kweights_low_low_k1 = kweights_history_low_low[:, 0]
    kweights_low_high_k1 = kweights_history_low_high[:, 0]
    kweights_low_low_k2 = kweights_history_low_low[:, 1]
    kweights_low_high_k2 = kweights_history_low_high[:, 1]
    plt.plot(kweights_low_low_k1, label='low_low_k1')
    plt.plot(kweights_low_high_k1, label='low_high_k1')
    plt.plot(kweights_low_low_k2, label='low_low_k2')
    plt.plot(kweights_low_high_k2, label='low_high_k2')
    # Add labels and legend
    plt.xlabel('Train Iterations')
    plt.ylabel('Kernel Weights')
    plt.title('Change in Kweights - From Low-E Ref.')
    plt.legend()
    # Show the plot
    plt.savefig(loading_path + '/' + 'Kweights_From_L.png')
    plt.show(block=False)

'''
Changes Kweights for h-l and h-h
'''
if not one_kernel:
    plt.figure()
    kweights_high_low_k1 = kweights_history_high_low[:, 0]
    kweights_high_high_k1 = kweights_history_high_high[:, 0]
    kweights_high_low_k2 = kweights_history_high_low[:, 1]
    kweights_high_high_k2 = kweights_history_high_high[:, 1]
    plt.plot(kweights_high_low_k1, label='high_low_k1')
    plt.plot(kweights_high_high_k1, label='high_high_k1')
    plt.plot(kweights_high_low_k2, label='high_low_k2')
    plt.plot(kweights_high_high_k2, label='high_high_k2')
    # Add labels and legend
    plt.xlabel('Train Iterations')
    plt.ylabel('Kernel Weights')
    plt.title('Change in Kweights - From High-E Ref')
    plt.legend()
    # Show the plot
    plt.savefig(loading_path + '/' + 'Kweights_From_H.png')
    plt.show(block=False)

'''
Changes Losses for l-l and l-h
'''
plt.figure()
plt.plot(dc_history_low_low.squeeze(), label='dc_low_low')
plt.plot(dc_history_low_high.squeeze(), label='dc_low_high', alpha=0.8)

# Add labels and legend
plt.xlabel('Train iterations')
plt.ylabel('Cramer loss')
plt.title('Change in Cramer Loss - From Low-E Ref.')
plt.legend()
# Show the plot
plt.savefig(loading_path + '/' + 'Loss_From_L.png')
plt.show(block=False)

'''
Changes Losses for h-l and h-h
'''
plt.figure()
plt.plot(dc_history_high_low.squeeze(), label='dc_high_low')
plt.plot(dc_history_high_high.squeeze(), label='dc_high_high', alpha=0.8)

# Add labels and legend
plt.xlabel('Train iterations')
plt.ylabel('Cramer loss')
plt.title('Change in Cramer Loss - From High-E Ref')
plt.legend()
# Show the plot
plt.savefig(loading_path + '/' + 'Loss_From_H.png')
plt.show(block=False)

'''
Changes in Means for l-l and h-l
'''
plt.figure()
if one_kernel:
    plt.plot(mean_low_low_k1, label='low_low_k1')
    plt.plot(mean_high_low_k1, label='high_low_k1', alpha=0.8)
else:
    plt.plot(mean_low_low_k1, label='low_low_k1')
    plt.plot(mean_high_low_k1, label='high_low_k1')
    plt.plot(mean_low_low_k2, label='low_low_k2')
    plt.plot(mean_high_low_k2, label='high_low_k2')

# Add labels and legend
plt.xlabel('Train Iterations')
plt.ylabel('Kernel Means')
plt.title('Change in Expectancy - From L/H to L-Target')
plt.legend()
# Show the plot
plt.savefig(loading_path + '/' + 'Means_LH_to_L.png')
plt.show(block=False)

'''
Changes in Means for l-h and h-h
'''
plt.figure()
if one_kernel:
    plt.plot(mean_low_high_k1, label='low_high_k1')
    plt.plot(mean_high_high_k1, label='high_high_k1', alpha=0.8)
else:
    plt.plot(mean_low_high_k1, label='low_high_k1')
    plt.plot(mean_high_high_k1, label='high_high_k1')
    plt.plot(mean_low_high_k2, label='low_high_k2')
    plt.plot(mean_high_high_k2, label='high_high_k2')

# Add labels and legend
plt.xlabel('Train Iterations')
plt.ylabel('Kernel Means')
plt.title('Change in Expectancy - From L/H to H-Target')
plt.legend()
# Show the plot
# Show the plot
plt.savefig(loading_path + '/' + 'Means_LH_to_H.png')
plt.show(block=False)

'''
Changes in Stds for l-l and h-l
'''
plt.figure()
if one_kernel:
    plt.plot(stds_low_low_k1, label='low_low_k1')
    plt.plot(stds_high_low_k1, label='high_low_k1', alpha=0.8)
else:
    plt.plot(stds_low_low_k1, label='low_low_k1')
    plt.plot(stds_high_low_k1, label='high_low_k1')
    plt.plot(stds_low_low_k2, label='low_low_k2')
    plt.plot(stds_high_low_k2, label='high_low_k2')

# Add labels and legend
plt.xlabel('Train Iterations')
plt.ylabel('Kernel Stds')
plt.title('Change in Stds - From L/H to L-Target')
plt.legend()
# Show the plot
# Show the plot
plt.savefig(loading_path + '/' + 'STDs_LH_to_L.png')
plt.show(block=False)

'''
Changes in Stds for l-h and h-h
'''
plt.figure()
if one_kernel:
    plt.plot(stds_low_high_k1, label='low_high_k1')
    plt.plot(stds_high_high_k1, label='high_high_k1', alpha=0.8)
else:
    plt.plot(stds_low_high_k1, label='low_high_k1')
    plt.plot(stds_high_high_k1, label='high_high_k1')
    plt.plot(stds_low_high_k2, label='low_high_k2')
    plt.plot(stds_high_high_k2, label='high_high_k2')

# Add labels and legend
plt.xlabel('Train Iterations')
plt.ylabel('Kernel Stds')
plt.title('Change in Stds - From L/H to H-Target')
plt.legend()
# Show the plot
plt.savefig(loading_path + '/' + 'STDs_LH_to_H.png')
plt.show(block=False)

'''
Changes in Kweights for l-l and h-l
'''
if not one_kernel:
    plt.figure()
    plt.plot(kweights_low_low_k1, label='low_low_k1')
    plt.plot(kweights_high_low_k1, label='high_low_k1')
    plt.plot(kweights_low_low_k2, label='low_low_k2')
    plt.plot(kweights_high_low_k2, label='high_low_k2')

    # Add labels and legend
    plt.xlabel('Train Iterations')
    plt.ylabel('Kernel Weights')
    plt.title('Change in Kweights - From L/H to L-Target')
    plt.legend()
    # Show the plot
    plt.savefig(loading_path + '/' + 'Kweights_LH_to_L.png')
    plt.show(block=False)

'''
Changes in Kweights for l-h and h-h
'''
if not one_kernel:
    plt.figure()
    plt.plot(kweights_low_high_k1, label='low_high_k1')
    plt.plot(kweights_high_high_k1, label='high_high_k1')
    plt.plot(kweights_low_high_k2, label='low_high_k2')
    plt.plot(kweights_high_high_k2, label='high_high_k2')

    # Add labels and legend
    plt.xlabel('Train Iterations')
    plt.ylabel('Kernel Weights')
    plt.title('Change in Kweights - From L/H to H-Target')
    plt.legend()
    # Show the plot
    plt.savefig(loading_path + '/' + 'Kweights_LH_to_H.png')
    plt.show(block=False)


'''
Show CDFs pre training
'''
plt.figure()
plt.plot(cdf_ref_low, label='cdf_ref_low')
plt.plot(cdf_ref_high, label='cdf_ref_high')
plt.plot(cdf_low_target, label='cdf_target_low')
plt.plot(cdf_high_target, label='cdf_target_high')

# Add labels and legend
plt.xlabel('Supports')
plt.ylabel('Cum. Probability')
plt.title('CDF: Reference with Targets - After Ref. Fitting')
plt.legend()
# Show the plot
plt.savefig(loading_path + '/' + 'CDFPre.png')
plt.show(block=False)

'''
Show Low-H CDF post training
'''
plt.figure()
plt.plot(cdf_low_low, label='cdf_low_low')
plt.plot(cdf_high_low, label='cdf_high_low')
plt.plot(cdf_low_target, label='cdf_low_target')

# Add labels and legend
plt.xlabel('Supports')
plt.ylabel('Probability')
plt.title('CDF - From L/H Ref to Low Target')
plt.legend()
# Show the plot
plt.savefig(loading_path + '/' + 'CDFPost_LH_to_L.png')
plt.show(block=False)

'''
Show High-H CDF post training
'''
plt.figure()
plt.plot(cdf_low_high, label='cdf_low_high')
plt.plot(cdf_high_high, label='cdf_high_high')
plt.plot(cdf_high_target, label='cdf_target_high')
# Add labels and legend
plt.xlabel('Supports')
plt.ylabel('Probability')
plt.title('CDF - From L/H Ref to High Target')
plt.legend()
# Show the plot
plt.savefig(loading_path + '/' + 'CDFPost_LH_to_H.png')
plt.show(block=True)



