'''
Script to concatenate two quantities of TB logging file
'''
import os
import numpy as np
import pickle
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
    curr_dir = os.getcwd()
    # Paths to your event files
    PATHS = [
        r"C:\Users\vanya\OneDrive\Desktop\PhD_RL\RL_Framework\dsacv02\tests\DSAC_Runs_Optim\HalfCheetah-v4\_SAC_Baseline\_UniCr_HalfCheetah_r4\chkpt1".replace('\\', '/'),
        r"C:\Users\vanya\OneDrive\Desktop\PhD_RL\RL_Framework\dsacv02\tests\DSAC_Runs_Optim\HalfCheetah-v4\_SAC_Baseline\_UniCr_HalfCheetah_r4\chkpt2".replace('\\', '/'),
    ]
    SAVE_PATH = curr_dir + '/' + 'HalfCheetah_SAC_r4.pkl'

    # Load events from both TensorBoard event files
    events = list()
    for path_i in PATHS:
        events.append(load_events(path_i, ["Rewards/Reward_Eval"]))

    # Identify breakpoints
    events.append(None)
    breakpoints = list()
    for i in range(0, len(events), 2):
        if events[i]:
            max_curr, _ = max(events[i])
            if events[i+1]:
                min_next, _ = min(events[i+1])
                if max_curr > min_next:
                    breakpoints.append(min_next)

    print(f'Breakpoints @ {breakpoints}')

    # Convert events to dictionaries
    events_list_dict = list()
    for event_i in events[:-1]:
        events_list_dict.append(dict(event_i))

    # Concatenate at saving points before last
    events_dict = dict()
    if breakpoints:
        for idx, break_i in enumerate(breakpoints):
            for k, v in events_list_dict[idx].items():
                events_dict[k] = v
                if k == break_i:
                    break
        for k, v in events_list_dict[-1].items():
            events_dict[k] = v
    else:
        for leg in events_list_dict:
           for k, v in leg.items():
                events_dict[k] = v

    # Save the dictionary
    with open(SAVE_PATH, mode='wb') as file:
        pickle.dump(events_dict, file)
