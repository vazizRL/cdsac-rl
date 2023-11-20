"""
This script serves to analyze some expressions and methods used in the DSAC Critic Code
"""
import torch


def value_clamping(min_std, max_std, stds):
    """
    - Requirements:
        - min_std >= 0
    :param min_std:
    :param max_std:
    :param stds:
    :return:
    """
    denominator = max(abs(min_std), max_std)
    value_std = torch.clamp_min(max_std * torch.tanh(stds / denominator), 0) + \
                torch.clamp_max(min_std * torch.tanh(stds / denominator), 0)

    return value_std


def own_value_clamping(min_std, max_std, stds):
    value_stds = torch.clamp(stds, min_std, max_std)

    return value_stds


if __name__ == '__main__':
    std_min = -3.0
    std_max = 3.0
    stds_pos = torch.arange(5)
    stds_neg = torch.flip(torch.arange(5) * -1, dims=(0,))
    stds_ = torch.cat((stds_neg[:-1], stds_pos), dim=0)

    # Between 0 and 1
    stds_01 = torch.linspace(0.1, 0.9, 3)
    stds_ = torch.cat((stds_, stds_01))

    clamped_value = value_clamping(min_std=std_min, max_std=std_max, stds=stds_)
    print(f'Max and min: {(std_min, std_max)}; Raw Value: {stds_}; Clamped Value: {clamped_value}\n')


    """
    Own implementation of clamping
    """
    own_clamped = own_value_clamping(min_std=std_min, max_std=std_max, stds=stds_)
    print(f'Own clamping. Max and min: {(std_min, std_max)}; Raw Value: {stds_}; Clamped Value: {own_clamped}\n')
