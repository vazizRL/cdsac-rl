import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore as zscore_calc
from tools import smoothing
from tensorboard.backend.event_processing import event_accumulator
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def find_event_files(root_dir, pattern_prefix="events.out.tfevents"):
    """
    Walk through root_dir and return list of paths to TensorBoard event files.
    """
    event_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith(pattern_prefix):
                event_files.append(dirpath)
    return event_files


def save_tb_graphs(logdir, output_dir, std_thld=3, n_kernels_act=1, n_kernels_cr=1):
    """
    - Tool to graph TB logs
    :param logdir: path to log dir
    :param output_dir: Name of graph dir to be created
    :param std_thld: Threshold for determining outliers in the graph
    :param n_kernels_act: n kernels actor
    :param n_kernels_cr: n kernels critic

    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir)

    # Get event dirs
    event_dirs = find_event_files(root_dir=logdir)

    event_accs = list()
    iter_list = list()
    for event_dir_i in event_dirs:
        # Load TensorBoard log files
        event_acc = event_accumulator.EventAccumulator(event_dir_i, compression_bps=None)
        # event_acc = event_accumulator.EventAccumulator(logdir, size_guidance=(15000,))
        event_acc.Reload()
        # Log the iterations, last step is the iteration number for this event
        iter_list.append(event_acc.Scalars('Rewards/Reward_Eval')[-1].step)
        # Append to events list
        event_accs.append(event_acc)
    max_iter = max(iter_list)

    # Get the TensorBoard graph
    kernel_names_act = list()
    kernel_names_act_output = list()
    kernel_names_cr = list()
    kernel_names_cr_output = list()
    for i in range(n_kernels_act):
        kernel_names_act.append(f'DSAC2_ActDistr/gmm_actor_avg_k{i+1}_weight iter')
        kernel_names_act_output.append(f'Actor_Kweights{i+1}.png')
    for i in range(n_kernels_cr):
        kernel_names_cr.append(f'DSAC2_CrDistr/gmm_critic_avg_k{i+1}_weight iter')
        kernel_names_cr_output.append(f'Critic_Kweights{i+1}.png')

    graph_names = ['DSAC2_ActDistr/entropy-RL iter', *kernel_names_act, 'DSAC2_ActDistr/gmm_actor_avg_std iter',
                   'DSAC2_Alpha/alpha-RL iter', *kernel_names_cr, 'DSAC2_CrDistr/gmm_critic_avg_std iter',
                   'DSAC2_CrDistr/gmm_critic_std_std iter', 'DSAC2_Vals/gmm_actor_avg_action iter',
                   'DSAC2_Vals/gmm_critic_avg_value iter', 'Loss/Actor loss-RL iter', 'Loss/Critic loss-RL iter',
                   'DSAC2_Sensitivity/q1_act_sen', 'DSAC2_Sensitivity/q2_act_sen',
                   'Rewards/Reward_Eval', 'Rewards/Reward_Training', 'Metric_Analysis/loss_raw_mean',
                   'Metric_Analysis/loss_raw_std',  'Metric_Analysis/loss_raw_max', 'Metric_Analysis/loss_raw_min',
                   'Metric_Analysis/dsq_supp_max_mean', 'Metric_Analysis/dsq_supp_min_mean',
                   'Metric_Analysis/dsq_supp_max', 'Metric_Analysis/dsq_supp_min', 'Time/Algorithm time [ms]-RL iter',
                   "dC:dQ/cramer_fac_mean", "dC:dQ/cramer_fac_max", "dC:dQ/cramer_fac_min", "dC:dQ/energy_fac_mean",
                   "dC:dQ/energy_fac_max", "dC:dQ/energy_fac_min"
                   ]

    file_save_names = ['Actor_Entropy.png', *kernel_names_act_output, 'Actor_Std.png', 'Alpha_Val.png',
                       *kernel_names_cr_output, 'Critic_Std.png', 'Critic_Std_Std.png', 'Actor_Val.png',
                       'Critic_Val.png', 'Actor_Loss.png', 'Critic_Loss.png', 'Q1_Action_Sensitivity.png',
                       'Q2_Action_Sensitivity.png', 'Reward_Eval.png', 'Reward_Training.png', 'Loss_Raw_Mean.png',
                       'Loss_Raw_Std.png',  'Loss_Raw_Max.png', 'Loss_Raw_Min.png', 'DSQ_Supp_Max_Mean.png',
                       'DSQ_Supp_Min_Mean.png', 'DSQ_Supp_Max.png', 'DSQ_Supp_Min.png', 'Time_per_Iter.png',
                       'dCdQ_Cr_Fac_Mean.png', 'dCdQ_Cr_Fac_Max.png', 'dCdQ_Cr_Fac_Min.png', 'dCdQ_En_Fac_Mean.png',
                       'dCdQ_En_Fac_Max.png', 'dCdQ_En_Fac_Min.png']

    for graph_name, file_save_name in zip(graph_names, file_save_names):
        # all values across events
        all_values = list()     # Nested
        all_smoothed_lists = list()
        all_supps = list()
        curve_names = list()
        for event_acc_i, event_dir_i, iter_i in zip(event_accs, event_dirs, iter_list):
            curve_name_event_i = event_dir_i.split('\\')[-1]
            curve_names.append(curve_name_event_i)
            values_i = list()
            if graph_name not in event_acc_i.Tags()['scalars']:
                print(f'{graph_name} not found in {event_dir_i}. . . Continue . . .')
                continue
            graph = event_acc_i.Scalars(graph_name)
            x_label = 'Iterations'
            for data_i in graph:
                values_i.append(data_i.value)
            smoothed_values, _, _, smoothed_list = smoothing(scalars=values_i, weight=0.99, iter=0, last=0)
            # Convert event vals and smoothed vals into np.ndarray
            values_i = np.asarray(values_i)
            smoothed_list = np.asarray(smoothed_list)
            # Calculate z-score based onr aw value
            zscores_i = zscore_calc(values_i)
            zscores_i = np.where(np.isnan(zscores_i), 0, zscores_i)
            mask_nan = np.abs(zscores_i) > std_thld
            # Set to nan if over threshold
            values_i[mask_nan] = np.nan
            smoothed_list[mask_nan] = np.nan
            # step_size from iters and generation of supports
            values_i_len = len(values_i)
            step_size = iter_i / values_i_len
            supps = np.arange(0, iter_i, step_size)
            all_values.append(values_i)
            all_smoothed_lists.append(smoothed_list)
            all_supps.append(supps)

        # Skip this graph if it doesn't exist in the TB logifle
        if len(values_i) == 0:
            continue

        # Get colormap, alt: 'jet', 'rainbow', 'cool'
        cmap = plt.cm.get_cmap('jet')

        # Initiate smooth figures
        len_iterates = len(all_supps)
        plt.figure(figsize=(10, 6))
        c_idx = 0
        for supps, smoothed_list, curve_name_i in zip(all_supps, all_smoothed_lists, curve_names):
            # plot
            color = cmap(c_idx / len_iterates)
            plt.plot(supps, smoothed_list, label=curve_name_i, color=color)
            c_idx += 1
        # Ticks
        plt.xlim(0, max_iter)
        if graph_name == 'Rewards/Reward_Eval' or graph_name == 'Rewards/Reward_Training':
            max_val = -10000
            max_val_smoothed = -10000
            for val_event_i, val_event_smoothing_i in zip(all_values, all_smoothed_lists):
                curr_max = max(val_event_i)
                curr_max_smoothed = max(val_event_smoothing_i)
                if curr_max > max_val:
                    max_val = curr_max
                if curr_max_smoothed > max_val_smoothed:
                    max_val_smoothed = curr_max_smoothed
            plt.ylim(-150, max_val + max_val * 0.1 )
        plt.xticks(np.linspace(0, max_iter, 10, dtype=int))
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0), useOffset=None, useLocale=None,
                             useMathText=True)
        plt.grid(visible=True, which='both', color='grey', linewidth=0.3)
        # Add labels and legend
        plt.xlabel(x_label)
        plt.ylabel('Smoothed_Values')
        plt.title(graph_name)
        plt.legend()
        plt.savefig(output_dir + '/' + file_save_name)

        # Initiate raw figures
        plt.figure(figsize=(10, 6))
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0), useOffset=None, useLocale=None,
                             useMathText=True)
        # Ticks
        plt.xlim(0, max_iter)
        if graph_name == 'Rewards/Reward_Eval' or graph_name == 'Rewards/Reward_Training':
            plt.ylim(-150, max_val + max_val * 0.1)
        plt.xticks(np.linspace(0, max_iter, 10, dtype=int))
        c_idx = 0
        for supps, values_i, curve_name_i in zip(all_supps, all_values, curve_names):
            # plot
            color = cmap(c_idx / len_iterates)
            plt.plot(supps, values_i, label=curve_name_i, color=color)
            c_idx += 1
        plt.grid(visible=True, which='both', color='grey', linewidth=0.3)
        # Add labels and legend
        plt.xlabel(x_label)
        plt.ylabel('Raw_Values')
        plt.title(graph_name)
        plt.legend()
        plt.savefig(output_dir + '/' + file_save_name[:-4] + '_Raw.png')

    print('Finished')


def compute_avg_rewards(logdir, output_dir):
    """
    - Computes the avg. rewards up to min(len(rewards))
    :param logdir:
    :param output_dir:
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir)

    # Get event dirs
    event_dirs = find_event_files(root_dir=logdir)

    event_accs = list()
    iter_list = list()
    for event_dir_i in event_dirs:
        # Load TensorBoard log files
        event_acc = event_accumulator.EventAccumulator(event_dir_i, compression_bps=None)
        # event_acc = event_accumulator.EventAccumulator(logdir, size_guidance=(15000,))
        event_acc.Reload()
        # Append to events list
        event_accs.append(event_acc)

    graph_names = ['Rewards/Reward_Eval', 'Rewards/Reward_Training']
    file_save_names = ['Avg_Reward_Eval.png', 'Avg_Reward_Training.png']

    color_idx = 2
    # Each graph per n events
    for graph_name, file_save_name in zip(graph_names, file_save_names):
        # Shape: (n_events x values)
        curve_names = list()
        all_values = list()
        min_len = float('inf')
        # Fix graphs for all events
        for event_acc_i, event_dir_i in zip(event_accs, event_dirs):
            curve_name_event_i = event_dir_i.split('\\')[-1]
            curve_names.append(curve_name_event_i)
            values_i = list()
            if graph_name not in event_acc_i.Tags()['scalars']:
                print(f'{graph_name} not found in {event_dir_i}. . . Continue . . .')
                continue
            graph = event_acc_i.Scalars(graph_name)
            for data_i in graph:
                values_i.append(data_i.value)
            # Convert
            values_i = np.asarray(values_i)
            all_values.append(values_i)
            # Get supports
            min_len = min(min_len, len(values_i))
        # Smallest value cut-off
        for idx in range(len(all_values)):
            all_values[idx] = all_values[idx][:min_len]
        all_values = np.asarray(all_values)
        # Smallest element determines cut-off
        # all_values = all_values[:, :min_len]
        # Compute average
        mean_values = all_values.mean(axis=0)
        mean_values_len = mean_values.shape[0]
        iter_n = event_acc.Scalars('Rewards/Reward_Eval')[-1].step
        step_size = iter_n / mean_values_len
        supps = np.arange(0, iter_n, step_size)

        # Get colormap, alt: 'jet', 'rainbow', 'cool'
        cmap = plt.cm.get_cmap('jet')

        # Instantiate figures
        plt.figure(figsize=(10, 6))
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0), useOffset=None, useLocale=None,
                             useMathText=True)
        # Tick
        plt.xlim(0, iter_n)
        plt.xticks(np.linspace(0, iter_n, 11, dtype=int))
        color_val = cmap(1 / color_idx)
        color_idx += 1
        # Graph average curve
        plt.plot(supps, mean_values, label=graph_name, color=color_val)
        plt.grid(visible=True, which='both', color='black', linewidth=0.3)
        # Add labels and legend
        plt.xlabel(xlabel='Iterations')
        plt.ylabel('Avg_Value')
        plt.title(graph_name)
        plt.legend()
        plt.savefig(output_dir + '/' + file_save_name)

    print('Finished')


if __name__ == '__main__':
    std_threshold_lib = 1000
    std_threshold_cons = 1
    log_path = r"C:\Users\vanya\OneDrive\Desktop\PhD_RL\RL_Framework\dsacv02\tests\DSAC_Runs_Optim_III\ant-v4_sac"
    output_path_lib = log_path + r'/' + 'graphs'
    output_path_lib_avg = log_path + r'/' + 'average_rewards'
    output_path_cons = log_path + r'/' + 'graphs_filtered'
    # save_tb_graphs(logdir=log_path, output_dir=output_path_lib, n_kernels_act=1, n_kernels_cr=1,
    #                std_thld=std_threshold_lib)
    # save_tb_graphs(logdir=log_path, output_dir=output_path_cons, n_kernels_act=1, n_kernels_cr=1,
    #                std_thld=std_threshold_cons)
    compute_avg_rewards(logdir=log_path, output_dir=output_path_lib_avg)
