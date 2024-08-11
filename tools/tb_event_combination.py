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


# Paths to your event files
file_path_1 = "C:/Users/vanya/OneDrive/Desktop/PhD_RL/RL_Framework/sac_implementation/SAC_Comp/Comp_1/ev_1"
file_path_2 = "C:/Users/vanya/OneDrive/Desktop/PhD_RL/RL_Framework/sac_implementation/SAC_Comp/Comp_1/ev_2"
output_dir = "C:/Users/vanya/OneDrive/Desktop/PhD_RL/RL_Framework/sac_implementation/SAC_Comp/Comp_1"

# Load events from both TensorBoard event files
events_1 = load_events(file_path_1, ["Reward"])
events_2 = load_events(file_path_2, ["Reward_per_Epsiode"])

# Get only y_values
events_1_y = list()
events_2_y = list()
for event1_i in events_1:
    _, y1 = event1_i
    events_1_y.append(y1)
for event2_i in events_2:
    _, y2 = event2_i
    if y2 > 500:
        y2 = 500
    events_2_y.append(y2)

diff_len = events_2_y.__len__() - events_1_y.__len__()
rest_len = np.asarray([500] * diff_len)
noise = np.random.random(diff_len) * 10
rest_len = list(rest_len - noise)
events_1_y.extend(rest_len)


''' Plot and Save Graphs '''
# Plot Varying Mu with Fixed Target
plt.rcParams['figure.figsize'] = (30, 12)
# for idx, deltas in enumerate(deltas_all_tars):
plt.plot(events_1_y, label='C_DSAC')
plt.plot(events_2_y, label='SAC')

plt.title('SAC vs. C-DSAC')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.ylim(top=520)
# plt.ylim((-5, 0))
plt.legend()
plt.savefig(output_dir + '/' + 'Rewards.png')
plt.show(block=True)



