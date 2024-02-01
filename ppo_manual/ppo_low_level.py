"""
Implementation notes
- Episode length: T=200
- Memory indices = [0, 1, ..., 19], fixed size
- Batches start at multiples of batch_size [0, 5, 10, 15]
- Shuffle memories then take batch size chunks of 5
- In this version: Two distinct networks instead of shared inputs, makes imp. of loss easier
- Critic evaluates states (not state-action)
- Actor:
    - Network outputs probs (softmax) for a distribution, use probs and sample for calc. the log probabilities
    - Exploration due to nature of distribution
- Track states, actions, rewards, dones, values, log probs
- Perform 4 epochs of updates on each batch

- PPO update rule: Based on ratio of new policy to old (can use logs) (eq. (6))
    - Keep track of selecting actions at each time for the logs prob
    - Advantage: Good of each state, based on GAE: Exponential smoothing; TD(\lambda) to reduce variance
        - Base don GAE, lambda hepls to reduce variance
        - Implementation: Nested for loop.
    - Clip the TRPO loss eq. (7)
- Critic loss:
    - return = advantage + ciritic value from memory
    - L_{critic} = MSE(return - critic value (from network)
- Total Loss:
    - L_t(\theta) = \hat{\double{E}} [L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c2 S_{\pi_{\theta}}(s_t)]]
        - Note that we are doing gradient ascent
        - c_2 = 0, only important if the actor and critic share same the input network
        - c_1 = 0.5
- Data Structures needed
    - Class for replay buffer -> listsu
    - Class for actor network, class for critic network
    - Class for agent (ties eyerything toegether)
    - Main loop to train and evaluate
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        """
        - List of integers corresponding to memory
        - Batch size chunks of these indices
        - Shuffle the batch
        """
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), \
            np.array(self.actions), \
            np.array(self.probs), \
            np.array(self.vals), \
            np.array(self.rewards), \
            np.array(self.dones), \
            batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = list()
        self.probs = list()
        self.actions = list()
        self.rewards = list()
        self.dones = list()
        self.vals = list()


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dim=56, fc2_dim=56):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.getcwd() + '/' + 'ppo_models/actor.pth'
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dim),
            nn.ReLU(),
            nn.Linear(fc1_dim, fc2_dim),
            nn.ReLU(),
            nn.Linear(fc2_dim, n_actions),
            # Applies softmax to the last dimension, i.e. nn.Linear(fc2_dim, n_actions)
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)

        return dist

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=56, fc2_dims=56):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.getcwd() + '/' + 'ppo_models/critic.pth'
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, policy_clip=0.2, batch_size=64, N=2048,
                 n_epochs=10, gae_lambda=0.9):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)

        self.memory = PPOMemory(batch_size=batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state=state, action=action, probs=probs, vals=vals, reward=reward,
                                 done=done)

    def save_models(self):
        print('...saving models...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('...loading models...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        # Samples from torch.distributions.categorical.Cateogircal
        action = dist.sample()

        # Get log prob for the action taken (reverse sampling)
        # Squeezing might be optional for PyTorch
        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr, reward_arr, done_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            disc_reward = 0

            for t in range(len(reward_arr)-1):
                discount = 1.0
                adv_t = 0

                # Test

                disc_factor = 1
                disc_factor *= self.gamma
                disc_reward += disc_factor * reward_arr[t]
                # /Test

                # Advantage from t to k
                for k in range(t, len(reward_arr)-1):
                    adv_t += discount * (reward_arr[k] + self.gamma * values[k+1] * (1-int(done_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = adv_t
            advantage = torch.tensor(advantage).to(self.actor.device)

            values = torch.tensor(values).to(self.actor.device)
            for batch in batches:
                # Conver to tensor and get mini-batch size of data
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                # New actions and values
                new_dist = self.actor(states)

                # NOTE: After each iteration, the critic network is updated, so new_critic_value != values[batch]
                new_critic_value = torch.squeeze(self.critic(states))

                # Get the prob of old actions according to new distribution ?
                new_probs = new_dist.log_prob(actions)

                prob_ratio = new_probs.exp() / old_probs.exp()
                # Alt.: (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weigted_clipped_prob = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * \
                    advantage[batch]

                # Actor loss
                actor_loss = -torch.min(weighted_probs, weigted_clipped_prob)

                # Calculate loss
                returns = advantage[batch] + values[batch]
                # returns = disc_reward
                critic_loss = (returns - new_critic_value)**2
                critic_loss = critic_loss.mean_target()
                total_loss = actor_loss + 0.5 * critic_loss

                # Perform an optimization network
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                # Convert to scalar
                total_loss.sum().backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()








