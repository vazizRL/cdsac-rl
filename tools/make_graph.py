import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# sns.set_style("darkgrid", {'axes.grid': True, 'axes.edgecolor': 'white', "grid.color": "0.8"})
sns.set_style("darkgrid", {'axes.grid': True, 'axes.edgecolor': 'black'})

if __name__ == '__main__':
    # Load the dictionaries and store in list
    path = r"C:\Users\vanya\OneDrive\Desktop\PhD_RL\RL_Framework\tools\Walker-v4\Walker-v4_RO_Events".replace('\\', '/')
    files_name = os.listdir(path)
    dict_list = list()
    for file_i in files_name:
        file_i = path + '/' + file_i
        if not os.path.isdir(file_i):
            with open(file_i, 'rb') as file:
                dict_list.append(pickle.load(file))

    # Create value list for 1M iterations
    complete_values = list()
    for i in range(1000):
        value_interval = list()
        idx = (i+1) * 1000
        for log_i in dict_list:
            value_interval.append(log_i[idx])
        complete_values.append(value_interval)

    # Compute Means
    complete_values = np.asarray(complete_values)
    means = complete_values.mean(axis=1)

    # Compute Stds
    stds = complete_values.std(axis=1)
    ''' Plot and Save Graphs '''
    x_supports = [(i+1)*1000 for i in range(1000)]
    # Plot Varying Mu with Fixed Target
    plt.rcParams['figure.figsize'] = (6, 4)

    # for idx, deltas in enumerate(deltas_all_tars):
    plt.plot(x_supports, means, label=f'C-DSAC', color='darkorange')
    plt.fill_between(x_supports, means - (stds / 2), means + (stds / 2), color='orange', alpha=0.6)
    # plt.plot(x_supports, means, label=f'SAC', color='firebrick')
    # plt.fill_between(x_supports, means - (stds / 2), means + (stds / 2), color='red', alpha=0.3)
    plt.xlim(left=0)
    # plt.title(f'Average Reward')
    plt.xlabel('million steps')
    plt.ylabel('average return')
    # plt.ylim((-5, 0))
    plt.legend()
    # plt.savefig(saving_path + '/' + 'VaryingCurrStd_Delta.png')
    plt.show(block=True)