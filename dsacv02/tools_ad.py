import os
import matplotlib.pyplot as plt
from tools import smoothing
from tensorboard.backend.event_processing import event_accumulator
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def save_tensorboard_graphs(logdir, output_dir, n_kernels_act=1, n_kernels_cr=1):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir)

    # Load TensorBoard log file
    event_acc = event_accumulator.EventAccumulator(logdir, compression_bps=None)
    # event_acc = event_accumulator.EventAccumulator(logdir, size_guidance=(15000,))
    event_acc.Reload()

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
                   'DSAC2_Vals/gmm_actor_avg_action iter', 'DSAC2_Vals/gmm_critic_avg_value iter',
                   'Loss/Actor loss-RL iter', 'Loss/Critic loss-RL iter', 'Rewards/Reward_Eval',
                   'Rewards/Reward_Training', 'Time/Algorithm time [ms]-RL iter'
                   ]
    file_save_names = ['Actor_Loss.png', *kernel_names_act_output, 'Actor_Std.png', 'Alpha_Val.png',
                       *kernel_names_cr_output, 'Critic_Std.png', 'Actor_Val.png', 'Critic_Val.png', 'Actor_Loss.png',
                       'Critic_Loss.png', 'Reward_Eval.png', 'Reward_Training', 'Time_per_Iter.png']

    for graph_name, file_save_name in zip(graph_names, file_save_names):
        values_i = list()
        graph = event_acc.Scalars(graph_name)
        x_label = 'Iterations'
        if graph_name == 'Reward':
            x_label = 'Episode'
        for data_i in graph:
            values_i.append(data_i.value)
        smoothed_values, _, _ = smoothing(scalars=values_i, weight=0.99, iter=0, last=0)
        plt.plot(smoothed_values)
        plt.grid(visible=True, which='both', color='grey', linewidth=0.3)
        # Add labels and legend
        plt.xlabel(x_label)
        plt.ylabel('Smoothed_Values')
        plt.title(graph_name)
        plt.legend()
        plt.savefig(output_dir + '/' + file_save_name)
        plt.figure()

        plt.plot(values_i)
        plt.grid(visible=True, which='both', color='grey', linewidth=0.3)
        # Add labels and legend
        plt.xlabel(x_label)
        plt.ylabel('Raw_Values')
        plt.title(graph_name)
        plt.legend()
        plt.savefig(output_dir + '/' + file_save_name[:-4] + '_Raw.png')
        plt.figure()

    print('Finished')


