"""
Implemented only for Windows
"""
import tensorboard
import tensorboard.backend.application
import numpy as np
import pandas as pd
import os
import time
import webbrowser
import platform
import signal

from tensorboard.backend.event_processing import event_accumulator
from typing import Union


DEFAULT_TB_PORT = 6001
# Tensorboard Tags
tb_tags = {
    'TAR of RL iteration': 'Evaluation/1. TAR-RL iter',
    'TAR of total time': 'valuation/2. TAR-Total time [s]',
    'TAR of collected samples': 'Evaluation/3. TAR-Collected sample',
    'TAR of replay samples': 'Evaluation/4. TAR-Replay samples',
    'Buffer RAM of RL iteration': 'RAM/RAM [MB]-RL iter',
    'loss_actor': 'Loss/Actor loss-RL iter',
    'loss_critic': 'Loss/Critic loss-RL iter',
    'alg_time': 'Time/Algorithm time [ms]-RL iter',
    'sampler_time': 'Time/Sampler time [ms]-RL iter',
    'critic_avg_value': 'Train/Critic avg value-RL iter',
}


""" Helper Function"""


def get_pids_windows(port: int):
    """
    - Gets process ID of servie running on port
    :param port: Port
    :return: List of PIDS corresponding to the port/service
    """
    with os.popen(f'netstat -aon|findstr {port}') as res:
        res = res.read().split('\n')
    results = list()
    for line in res:
        temp = [i for i in line.split(' ') if i != '']
        if len(temp) > 4:
            results.append(temp[4])     # Try also temp[3]
    return list(set(results))


def kill_pid_windows(pids):
    """
    - Kills a process in windows
    """
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGINT)
        except:
            print(f'Could not kill PIDS {pids}')


def kill_port(port=DEFAULT_TB_PORT):
    """
    - Gets PID of net service and closes port by closing service
    :param port: Port on which net service is running
    """
    pids = get_pids_windows(port)
    kill_pid_windows(pids)


""" /Helper Functions"""


def read_tensorboard(path) -> dict:
    tensorboard.backend.application.logger.setLevel('ERROR')
    ev_acc = event_accumulator.EventAccumulator(path)
    ev_acc.Reload()
    valid_key_list = ev_acc.scalars.Keys()

    output_dict = dict()
    for key in valid_key_list:
        event_list = ev_acc.scalars.Items(key)
        x, y = [], []
        for event in event_list:
            x.append(event.step)
            y.append(event.value)

        data_dict = {'x': np.array(x), 'y': np.array(y)}
        output_dict[key] = data_dict

    return output_dict


def start_tensorboard(logdir, port=DEFAULT_TB_PORT):
    kill_port(port)
    cmd_line = f'start /b cmd.exe /k "tensorboard --logdir {logdir} --port {port}'
    os.system(cmd_line)
    time.sleep(5)

    webbrowser.open(f'http://localhost:{port}/')

    return 0


def add_scalars(tb_info, writer, step):
    for key, value in tb_info.items():
        writer.add_scalar(key, value, step)


def save_csv(path, step, value):
    """
    - Saves data in two columns: Step-Value
    :param path: Save directory
    :param step: Iteration step
    :param value: Value
    """
    data_frame = pd.DataFrame({'Step': step, 'Value': value})
    data_frame.to_csv(path, index=False, sep=',')


def save_tb_to_csv(path):
    data_dict = read_tensorboard(path)
    for data_name in data_dict.keys():
        data_name_format = data_name.replace(r'\\', '/').replace('/', '_')
        csv_dir = os.path.join(path, 'data')
        file_path = os.path.join(csv_dir, f'{data_name_format}.csv')
        os.makedirs(csv_dir, exist_ok=True)

        save_csv(file_path, step=data_dict[data_name]['x'], value=data_dict[data_name]['y'])



