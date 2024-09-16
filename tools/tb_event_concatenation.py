'''
Script to concatenate two quantities of TB logging file
'''
import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_events(file_path, tags):
    event_acc = EventAccumulator(file_path)
    event_acc.Reload()
    events = []
    for tag in tags:
        if tag in event_acc.Tags()["scalars"]:
            scalar_events = event_acc.Scalars(tag)
            events.extend((event.step, event.value) for event in scalar_events)
    return events


def write_combined_events(output_file_path, combined_events):
    writer = SummaryWriter(output_file_path)
    for step, value in combined_events:
        writer.add_scalar("Reward_All", value, step)
    writer.close()


if __name__ == '__main__':
    # Paths to your event files
    PATH_1 = r"C:\Users\vanya\OneDrive\Desktop\PhD_RL\RL_Framework\dsacv02\rm_colab_run_checkpoint1/".replace("\\", "/")
    PATH_2 = r"C:\Users\vanya\OneDrive\Desktop\PhD_RL\RL_Framework\dsacv02\rm_colab_run_checkpoint2/".replace("\\", "/")
    OUTPUT_DIR = "C:/Users/vanya/OneDrive/Desktop/Temp/TB"

    # Load events from both TensorBoard event files
    events_1 = load_events(PATH_1, ["Rewards/Reward_Eval"])
    events_2 = load_events(PATH_2, ["Rewards/Reward_Eval"])

    # Get only y_values
    events_1_y = list()
    events_2_y = list()

    # Concatenate

    ''' Plot and Save Graphs '''
    # Plot Varying Mu with Fixed Target
    plt.rcParams['figure.figsize'] = (30, 12)
    # for idx, deltas in enumerate(deltas_all_tars):
    plt.plot(events_1_y, label='C_DSAC')
    plt.plot(events_2_y, label='SAC')

    plt.title('SAC vs. C-DSAC')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    # plt.ylim(top=520)
    plt.legend()
    plt.savefig(OUTPUT_DIR + '/' + 'Rewards.png')
    plt.show(block=True)



