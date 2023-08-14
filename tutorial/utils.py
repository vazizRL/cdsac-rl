import numpy as np


def dim_shower(layer_arch: tuple,  input_dim: tuple, stride: int, fc=None):
    """
    - Gives output per layer specified with layer_arch, stride and filter_size
    - Indiactes invalid configurations
    :param layer_arch: Format: ([type]: str, config: tuple)
    :param input_dim: Input dimensions to CNN without channels
    :param stride: Stride
    :param fc: Input of fully connected at flatten
    """
    # Assuming w = h
    # Formula:  Output dimension =      ((input dimension - kernel size) / stride) + 1
    # 	        Pooling Reduction: 		Output = flat(input_dimension / (pooling^(n_pooling)))
    current_dim = input_dim
    cnn_nodes = -1
    for layder_idx, layer_config in enumerate(layer_arch):
        if layer_config[0] == 'conv':
            kernel_red_w = current_dim[0] - layer_config[1][0]
            kernel_red_h = current_dim[1] - layer_config[1][1]
            current_dim = ((np.int_(kernel_red_w / stride)) + 1, (np.int_(kernel_red_h / stride)) + 1)
            cnn_nodes = layer_config[2]
        elif layer_config[0] == 'pool':
            rounded_w = np.int_(current_dim[0] / layer_config[1][0])
            rounded_h = np.int_(current_dim[1] / layer_config[1][1])
            current_dim = (rounded_w, rounded_h)
        else:
            raise ValueError(f'{layer_config[0]} type not known!')
        print(f'Layer {layder_idx}: Channels {cnn_nodes} @ {current_dim} ')

    last_layer_cnn_sum = cnn_nodes * current_dim[0] * current_dim[1]
    print(f'Sum of nodes of CNN last layer: {last_layer_cnn_sum}')
    if fc:
        if fc != last_layer_cnn_sum:
            print(f'INVALID CONFIG! FC inupt is {fc}, sum of nodes of last CNN: {last_layer_cnn_sum}')

    return 0


if __name__ == '__main__':
    arch = (('conv', (3, 3), 16), ('pool', (2, 2)), ('conv', (3, 3), 16), ('pool', (2, 2)))
    inp_nodes = (3, 32, 32)
    dim_shower(layer_arch=arch, input_dim=(32, 32), stride=1)

