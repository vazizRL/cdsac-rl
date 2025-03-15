import matplotlib.pyplot as plt
from dsacv02.tools_ad import save_tensorboard_graphs


if __name__ == '__main__':
    N_KERNELS_ACT = 1
    N_KERNELS_CR = 1
    curr_dir = r"C:\Users\vanya\OneDrive\Desktop\PhD_RL\RL_Framework\dsacv02\cramer_ii\Critic_Parameterizatoin_Test_Ant"
    output_dir = curr_dir + r"\graphs"
    plt.rcParams['figure.figsize'] = (30, 12)           # Old: (20, 8)
    # Example usage:
    logdir = curr_dir
    save_tensorboard_graphs(logdir, output_dir, n_kernels_act=N_KERNELS_ACT, n_kernels_cr=N_KERNELS_CR)