import torch
import numpy as np
from gym.spaces import Box


class LinearV(torch.nn.Module):
    def __init__(self, n_params, dev):
        super(LinearV, self).__init__()
        self.device = dev
        self.n_params = n_params
        self.params = torch.nn.Parameter(torch.empty(self.n_params, 1, dtype=torch.float32), requires_grad=False)
        self.init_params()
        self.to(self.device)

    def forward(self, phi_s):
        return torch.matmul(phi_s, self.params)

    def update_params(self, lr, td_err, phi_s):
        loss_stepped = lr * td_err * phi_s
        self.params.add_(torch.unsqueeze(loss_stepped, dim=-1).mean(dim=0))

        return 0

    def init_params(self):
        # torch.nn.init.uniform(self.params, a=-1, b=1)
        torch.nn.init.xavier_uniform(self.params)

        return 0


class TileEncoder2D:
    def __init__(self, n_bins, frac_bin_width_multi, n_tiles: int, box: Box):
        """
        - Box example: arr([-1.2, -0.07], [0.6, 0.07], (2,) np.float32)
        :param n_bins:
        :param n_tiles:
        :param box:
        """
        self.n_tiles = n_tiles
        self.low_d1 = box.low[0]
        self.low_d2 = box.low[1]
        self.high_d1 = box.high[0]
        self.high_d2 = box.high[1]

        self.n_bins = n_bins
        self.frac_bin_multi = frac_bin_width_multi
        self.delta_1 = (self.high_d1 - self.low_d1) / self.n_bins
        self.delta_2 = (self.high_d2 - self.low_d2) / self.n_bins
        self.eps1 = self.delta_1 / self.frac_bin_multi
        self.eps2 = self.delta_2 / self.frac_bin_multi

    def encode(self, obs_arr: np.ndarray):
        """
        :param obs_arr:
        """
        # obs_arr: batch_s x obs_dim
        batch_size = obs_arr.shape[0]
        encoded = np.zeros(shape=(batch_size, self.n_tiles, self.n_bins, self.n_bins))

        # Note: Can be vectorized, but too lazy
        for i in range(self.n_tiles):
            # Get (batch_s,) for each dimension
            d1 = obs_arr[:, 0]
            d2 = obs_arr[:, 1]
            off1 = self.eps1 * i
            off2 = self.eps2 * i

            d1, d2 = d1 + off1, d2 + off2

            idx1 = np.minimum(np.floor((d1 - self.low_d1) / self.delta_1), self.n_bins - 1).astype(np.int16)
            idx2 = np.minimum(np.floor((d2 - self.low_d2) / self.delta_2), self.n_bins - 1).astype(np.int16)

            range_arr = np.arange(batch_size)
            encoded[range_arr, i, idx1, idx2] = 1

        return encoded


# Sanity Test
if __name__ == '__main__':
    # Test Encoder
    # obs = np.array([-1.2, -0.07], dtype=np.float32)
    # obs = np.array([0.35, 0.032], dtype=np.float32)
    obs = np.array([-1.2, -0.065], dtype=np.float32)
    n_bins = 12
    n_tiles = 6
    b = Box(np.array([-1.2, -0.07]), np.array([0.6, 0.07]), (2,), np.float32)
    encoder = TileEncoder2D(n_bins=n_bins, frac_bin_width_multi=3, n_tiles=n_tiles, box=b)
    tiles = encoder.encode(obs_arr=obs)
    # print(tiles)

    # Manual Example
    low1, high1 = -1.2, 0.6
    low2, high2 = -0.07, 0.07
    # Construct empty n_bins x n_bins array
    tile_i = np.zeros(shape=(n_bins, n_bins), dtype=np.float32)
    # Define epsilon
    eps1, eps2 = 0.1, 0.005
    # Example: phi = (-0.4, 0.01)
    # d1, d2 = -1.2, -0.07
    # Eps should be some fraction of the bin width
    d1, d2 = -1.2 + eps1, -0.07 + eps2
    # Compute indexes
    delta1 = (high1 - low1) / n_bins
    delta2 = (high2 - low2) / n_bins
    idx1 = min(round((d1 - low1) / delta1), n_bins - 1)
    idx2 = min(round((d2 - low2) / delta2), n_bins - 1)

    tile_i[idx1, idx2] = 1
    # print(tile_i)

    # Test value network
    device = 'cpu'
    n_parameters = n_bins**2 * n_tiles
    value = LinearV(n_params=n_parameters, dev=device)
    phi_test = torch.as_tensor(tiles, dtype=torch.float32)
    phi_test = torch.unsqueeze(phi_test.flatten(), dim=0)
    res = value(phi_test)
    print(res.shape)
    print(res)






