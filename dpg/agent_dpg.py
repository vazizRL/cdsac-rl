import torch

from critic import LinearQ
from nn_approx import Policy
from v import LinearV, TileEncoder2D
from replay_buffer import ReplayBuffer
from numpy import squeeze


class Agent:
    def __init__(self, state_dim: tuple, act_dim: tuple, policy_arch: tuple, tile_n_bins: int,
                 tile_width_multi, tiles_n, tile_box, replay_size: int, action_min: float, action_max: float,
                 batch_size: int, lr: float, gamma: float, dev='cuda:0'):
        self.state_dim = state_dim
        self.action_dim = act_dim
        self.policy_arch = policy_arch
        self.tile_n_bins = tile_n_bins
        self.tile_width_multi = tile_width_multi,
        self.tiles_n = tiles_n
        self.space_box = tile_box
        self.space_encoder = TileEncoder2D(n_bins=self.tile_n_bins, frac_bin_width_multi=self.tile_width_multi,
                                           n_tiles=self.tiles_n, box=self.space_box)
        self.value_n_params = self.tile_n_bins**2 * self.tiles_n
        self.replay_size = replay_size
        self.action_min = action_min
        self.action_max = action_max
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = torch.tensor(gamma, dtype=torch.float32)
        self.device = dev
        self.memory = ReplayBuffer(max_size=self.replay_size, state_shape=self.state_dim, action_shape=self.action_dim)
        self.policy = Policy(arch=self.policy_arch, lr=self.lr, name='NN_Actor', dev=dev, act_min=self.action_min,
                             act_max=self.action_max)
        self.critic_n_params = sum(p.numel() for p in self.policy.parameters())
        self.critic = LinearQ(n_params=self.critic_n_params, dev=self.device)
        self.value = LinearV(n_params=self.value_n_params, dev=self.device)
        self.gamma.to(self.device)

    def learn(self):
        """
        - Learning actor, critic
        """
        # Sample from replay buffer and convert into torch array
        state, actions_old, reward, state_next, done = self.memory.sample(self.batch_size)
        # Encode spaces flatten, last dims and turn into batch_s x tile_params
        phi_s = torch.as_tensor(self.space_encoder.encode(obs_arr=state), dtype=torch.float32).to(self.device)
        phi_s = phi_s.flatten(start_dim=1, end_dim=3).to(self.device)
        phi_s_next = torch.as_tensor(self.space_encoder.encode(obs_arr=state_next), dtype=torch.float32).to(self.device)
        phi_s_next = phi_s_next.flatten(start_dim=1, end_dim=3).to(self.device)
        # Convert tuple elements into Torch Tensor
        state, actions_old, reward, state_next, done = \
            torch.as_tensor(state, dtype=torch.float32, device=self.device), \
            torch.as_tensor(actions_old, dtype=torch.float32, device=self.device),\
            torch.as_tensor(reward, dtype=torch.float32, device=self.device),\
            torch.as_tensor(state_next, dtype=torch.float32, device=self.device), \
            torch.as_tensor(done, dtype=torch.float32, device=self.device)

        # Actions, current actions (batch_s x m), reshape to (batch_s x 1 x m)
        actions_old = torch.unsqueeze(actions_old, dim=1)
        # Evaluate value networks for s and s_next
        val = self.value(phi_s=phi_s)
        val_next = self.value(phi_s=phi_s_next)

        # Critic Update, first target
        # Q(s_{t+1}, \mu_{\theta}(s_{t+1})) = (a - \mu_{\theta}(s_{t+1}))^T ... where a=\mu_{\theta}(s_{t+1})
        actions_next = self.policy(state_next)
        actions_next = torch.unsqueeze(actions_next, dim=1)  # (batch_s x 1 x m)
        actions_next = actions_next.detach()
        # tanh now in functional stck of policy class
        pol_jaco_next = self.policy.get_jacobian(state_next)
        pol_jaco_next.detach_()
        phi_sa_next = torch.matmul(actions_next, pol_jaco_next)
        q_next = self.critic(phi_sa_next) + val_next
        q_target = reward + (1 - done) * self.gamma * q_next
        # Current Q(s,a), start with obtaining jacobian of policy
        pol_jaco = self.policy.get_jacobian(state=state)  # (batch x m x n)
        pol_jaco.detach_()
        phi_sa = torch.matmul(actions_old, pol_jaco)     # (batch x 1 x n)
        q_current = self.critic(phi_sa) + val

        # TD-error, collapse batch
        td_error = q_target - q_current
        # td_error = torch.clamp(q_target - q_current, -10, 10)
        critic_loss = self.critic.update_params(lr=self.lr, td_err=td_error, phi_sa=phi_sa.transpose(1, 2))

        # Value Update
        self.value.update_params(lr=self.lr, td_err=td_error, phi_s=phi_s)

        # Actor Update
        # Reset policy optimizer
        self.policy.optimizer.zero_grad()
        cr_params = torch.unsqueeze(self.critic.params, dim=0)  # (1 x n x 1)
        q_grad = torch.matmul(pol_jaco, cr_params)              # (batch_s x m x 1)
        actions_curr = self.policy(state)                       # (batch_s x m)
        actions_curr = torch.unsqueeze(actions_curr, dim=1)     # (batch_s x 1 x m)
        # Form the scalar from actions and q_grad and use torch's autodiff
        pol_gain = torch.matmul(actions_curr, q_grad)
        # PyTorch perform  gradient descent by default, so invertsigns
        # pol_loss = pol_loss.mean()
        pol_gain = -1 * pol_gain.mean()
        pol_gain.backward()
        # Optional gradient clipping
        torch.nn.utils.clip_grad_norm(self.policy.parameters(), max_norm=1.0)
        self.policy.optimizer.step()

        tb_info = {
            "Losses/Actor_Loss": pol_gain.detach().item(),
            "Losses/Critic_Loss": critic_loss.detach().item(),
            "Losses/TD_Error_Mean": td_error.mean().cpu().item(),
            "Losses/TD_Error_Abs_Mean": td_error.abs().mean().cpu().item(),
            "Values/State": val.mean().detach().item(),
            "Values/State-Action": q_current.mean().detach().item(),
        }

        return tb_info

    def choose_action(self, state, std):
        """
        - Returns bounded sampled action in [ACT_MIN, ACT_MAX] using Policy class
        :param state:
        :param std:
        :return:
        """
        # Convert to PyTorch Tensors and dsend to GPU if necessary
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        std = torch.as_tensor(std, dtype=torch.float32, device=self.device)
        action_mean = self.policy(state).detach()
        action_bounded = self.policy.sample_action(loc=action_mean, std=std)

        return squeeze(action_bounded.cpu().numpy(), axis=0)

    def choose_deterministic_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action_deter = torch.squeeze(self.policy(state), dim=0)
        action_deter = action_deter.detach().cpu().numpy()

        return action_deter

    def remember(self, state, action, reward, state_next, done):
        self.memory.store_transition(state=state, action=action, reward=reward, state_next=state_next, done=done)

        return 0

    def save(self):
        raise NotImplementedError

    def get_empty_tb_info(self):
        tb_info = {
            "Losses/Actor_Loss": 0,
            "Losses/Critic_Loss": 0,
            "Losses/TD_Error_Mean": 0,
            "Losses/TD_Error_Abs_Mean": 0,
            "Values/State": 0,
            "Values/State-Action": 0,
        }

        return tb_info


# Sanity Check
if __name__ == '__main__':
    import numpy as np
    from gym.spaces import Box
    dims_low = np.array([0., 0.])
    dims_high = np.array([1., 1.])
    b = Box(low=dims_low, high=dims_high, shape=(2,), dtype=np.float32)
    enc = TileEncoder2D(n_bins=10, frac_bin_width_multi=3, n_tiles=8, box=b)
    obs = np.array([0, 1])
    encoded = enc.encode(obs_arr=obs)

