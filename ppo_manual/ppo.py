import matplotlib.pyplot as plt
import torch
import tqdm
from collections import defaultdict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, set_exploration_mode
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE


# Define Hyperparameters
device = 'cpu' if not torch.has_cuda else 'cuda:0'
# Numbe of nodes in each layer
num_cells = 256
lr = 3e-4
# Max value the gradients are allowed to have
max_grad_norm = 1.0


# Data collection parameters
frame_skip = 1
frames_per_batch = 10000 // frame_skip
total_frames = 50_000 // frame_skip


# PPO Parameters
sub_batch_size = 64     # For inner loop
num_epochs = 2          # 10
clip_epsilon = (0.2)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4


# Create environment from gym
base_env = GymEnv('InvertedDoublePendulum-v4', device=device, frame_skip=frame_skip)

# Create env-wrapper with transformations
env = TransformedEnv(
    base_env,
    Compose(
        ObservationNorm(in_keys=['observation']),
        DoubleToFloat(in_keys=['observation']),
        StepCounter(),
    ),
)

env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
print(env.transform[0].loc)

rollout = env.rollout(5)
# Automatically saved in TensorDict
print('rollout of three steps:', rollout)
print('Shape of the rollout Tensorict:', rollout.batch_size)


# Define Policy and Wrap
actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
    NormalParamExtractor(),
)

policy_module = TensorDictModule(
    actor_net, in_keys=['observation'], out_keys=['loc', 'scale']
)
# Generate distribution from parameters and sample from it
policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=['loc', 'scale'],
    distribution_class=TanhNormal,
    distribution_kwargs={
        'min': env.action_spec.space.minimum,
        'max': env.action_spec.space.maximum,
    },
    return_log_prob=True,
)

# Define Value Network and Wrap
value_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(1, device=device)
)

value_module = ValueOperator(
    module=value_net,
    in_keys=['observation'],
)

# Test Wrappers for PyTorch environments
print('Running policy:', policy_module(env.reset()))
print('Running value:', value_module(env.reset()))


# Write Data Collector, returns TensorDict
collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device
)


# Implement Replay Buffer
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)


# Load PPO loss from torchrl
advantage_module = GAE(gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True)
loss_module = ClipPPOLoss(
    actor=policy_module,
    critic=value_module,
    advantage_key='advantage',
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # Default values
    value_target_key=advantage_module.value_target_key,
    critic_coef=1.0,
    gamma=0.99,
    loss_critic_type='smooth_l1'
)
optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, total_frames // frames_per_batch, 0.0)


# Implement training loop
logs = defaultdict(list)
pbar = tqdm.tqdm(total=total_frames*frame_skip)
eval_str = ''

# Iterate over collector until total number of frames
for i, tensordict_data in enumerate(collector):
    # Learn from batch of data
    for _ in range(num_epochs):
        # Compute advantage, value network is updated in each iteration
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals['loss_objective']
                + loss_vals['loss_critic']
                + loss_vals['loss_entropy']
            )
            # OPtimization: backward, grad clipping and optimization step
            loss_value.backward()
            # Optional but good practice to keep gradient norm bounded
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

    logs['reward'].append(tensordict_data['next', 'reward'].mean().item())
    pbar.update(tensordict_data.numel() * frame_skip)
    cum_reward_str = f'average reward={logs["reward"][-1]:4.4f} (init={logs["reward"][0]: 4.4f})'
    logs['step_count'].append(tensordict_data['step_count'].max().item())
    stepcount_str = f'Step count (max): {logs["step_count"][-1]}'
    logs['lr'].append(optim.param_groups[0]['lr'])
    lr_str = f'lr policy: {logs["lr"][-1]: 4.4f}'
    if i % 10 == 0:
        # Evaluate policy evrey 10 batches of data
        # Evaluation: Execute policy without exploration
        # Rollour method of env can take a policy as argument
        # It will execute this policy at each step
        with set_exploration_mode('mean'), torch.no_grad():
            eval_rollout = env.rollout(1000, policy_module)
            logs['eval reward'].append(eval_rollout['next', 'reward'].mean().item())
            logs['eval reward sum'].append(eval_rollout['next', 'reward'].sum().item())
            logs['eval step count'].append(eval_rollout['step_count'].max().item())
            eval_str = (
                f'eval cumulative reward: {logs["eval reward sum"][-1]: 4.4f}'
                f'(init: {logs["eval reward sum"][0]: 4.4f}'
                f'(eval step_count: {logs["eval step count"][-1]}'
            )
            del eval_rollout
    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

    scheduler.step()


plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.plot(logs['reward'])
plt.title('Average training rewards')

plt.subplot(2, 2, 2)
plt.plot(logs['step_count'])
plt.title('Max training step count ')

plt.subplot(2, 2, 3)
plt.plot(logs['eval reward sum'])
plt.title('Sum of eval reward')

plt.subplot(2, 2, 4)
plt.plot(logs['eval step count'])
plt.title('Max eval step count')


plt.show()
