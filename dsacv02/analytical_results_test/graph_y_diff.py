import os
import numpy as np
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Interval Loging
cntr = 1
interval = 3000

# Path to Y-Diff
run_name = 'Y-Diff_1717172078.367713'
curr_dir = os.getcwd() + '/'
y_diff_dsac_path = curr_dir + run_name + '/' + 'y_diff_dsac.npy'
y_diff_sac_path = curr_dir + run_name + '/' + 'y_diff_sac.npy'

# Load Arrays
y_diff_dsac_hist = np.load(y_diff_dsac_path)
y_diff_sac_hist = np.load(y_diff_sac_path)
# For x-axis values
x_ticks = [i+1 for i in range(8)]

# Graph
for dsac_curve, sac_curve in zip(y_diff_dsac_hist, y_diff_sac_hist):
    curr_iter = interval * cntr
    label_dsac_i = f'DSAC_Iter_{curr_iter}'
    label_sac_i = f'SAC_Iter_{curr_iter}'
    plt.plot(x_ticks, dsac_curve, label=label_dsac_i)
    plt.plot(x_ticks, sac_curve, label=label_sac_i)

    plt.xticks(ticks=x_ticks)
    plt.xlabel('Cell_Nr')
    plt.ylabel('Difference: hat{Y} - Y^*')
    plt.title('Difference between true and app. Y - DSAC v. SAC - \pi^*')
    plt.legend()
    # Show the plot
    plt.savefig(curr_dir + run_name  + '/' + 'Y_Diff_DSAC_vs_SAC.png')

    cntr += 1

plt.show(block=True)






