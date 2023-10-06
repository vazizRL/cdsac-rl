"""
- Implemented as original with target value network and Q-minimization trick
"""
import os
import torch as T
import torch.nn.functional as F
import numpy as np
from replay_buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork
import torch.optim as optim


class Agent:
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=(8,), env=None, gamma=0.99, n_actions=2,
                 max_size=int(1e6), tau=0.005, layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions, name='actor',
                                  max_actions=env.action_space.high, fc1_dims=layer1_size, fc2_dims=layer2_size)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='critic_1', fc1_dims=layer1_size,
                                      fc2_dims=layer2_size)
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='critic_2', fc1_dims=layer1_size,
                                      fc2_dims=layer2_size)

        self.value = ValueNetwork(beta, input_dims, name='value', fc1_dims=layer1_size, fc2_dims=layer2_size)
        self.target_value = ValueNetwork(beta, input_dims, name='target_value', fc1_dims=layer1_size,
                                         fc2_dims=layer2_size)

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

        self.empty_tb_data = {'SACQ/q1_val': 0,
                              'SACQ/q2_val': 0,
                              'SACLoss/critic_loss': 0,
                              'SACLoss/actor_loss': 0,
                              'SACLoss/value_loss': 0,
                              'SACPolicy/avg. policy std plain': 0,
                              'SACPolicy/avg. policy std repara': 0
                              }

    def choose_action(self, observation):
        state = T.tensor([observation]).to(self.actor.device)
        actions, _ , _ = self.actor.sample_normal(state, reparametrize=False)

        # Send it to CPU, detach from graph, turn into np.ndarray and take zeroth element
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        """
        - Implements soft and hard updates of the target soft value function
        :param tau: Update weight
        """
        if tau is None:
            tau = self.tau

        # Retrieve parameters including name as string as gen.
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        # Name as key, weights as values
        target_value_dict = dict(target_value_params)
        value_dict = dict(value_params)

        for name in value_dict:
            # Apply soft copy rule per layer
            value_dict[name] = tau * value_dict[name].clone() + (1-tau) * target_value_dict[name].clone()

        # Apply moved weights to the target
        self.target_value.load_state_dict(value_dict)

    def save_models(self):
        print('. . . saving models . . .')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('. . . loading modles . . .')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        """
        - Implements the update rules of SAC that defines this algorithm
        """
        if self.memory.mem_cntr < self.batch_size:
            print('Memory < Batch Size, Learning not initiated')
            return self.empty_tb_data

        state, old_actions, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        # Transform numpy into PyTorch tensors
        reward = T.tensor(reward, dtype=T.float64).to(self.actor.device)
        done = T.tensor(done, dtype=T.float64).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float64).to(self.actor.device)
        state = T.tensor(state, dtype=T.float64).to(self.actor.device)
        old_actions = T.tensor(old_actions, dtype=T.float64).to(self.actor.device)

        # Collapse dimension along batch_size
        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)

        # If done is True, then the value of next state after done is 0
        value_[T.tensor(done, dtype=bool)] = 0.0

        """ Calculate value loss """
        # According to current policy
        curr_actions, log_probs, pol_std = self.actor.sample_normal(state, reparametrize=False)
        log_probs = log_probs.nan_to_num()
        log_probs = log_probs.view(-1)

        # Get min. Q-value for state and action under current policy
        q1_new_policy = self.critic_1.forward(state, curr_actions)
        q2_new_policy = self.critic_2.forward(state, curr_actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        # MSE of the batch (mean of the square)
        value_loss = 0.5 * F.mse_loss(value, value_target)
        # retain_graph: Do not discard the graph calculation
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        """Calculate Critic Loss """
        # According to Fujomo, chose min target and calculate mean of Q-losses, here: Same implementaiton
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, old_actions).view(-1)
        q2_old_policy = self.critic_2.forward(state, old_actions).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        """ Calculate Actor Loss """
        # Note: In Paper, Critic loss is calculated first
        curr_actions_rep, log_probs_rep, pol_std_rep = self.actor.sample_normal(state, reparametrize=True)
        log_probs_rep = log_probs_rep.nan_to_num()
        log_probs_rep = log_probs_rep.view(-1)
        q1_new_policy_rep = self.critic_1.forward(state, curr_actions_rep)
        q2_new_policy_rep = self.critic_2.forward(state, curr_actions_rep)
        critic_value_rep = T.min(q1_new_policy_rep, q2_new_policy_rep)
        critic_value_rep = critic_value_rep.view(-1)

        actor_loss = log_probs_rep - critic_value_rep
        # Get mean of the batch
        actor_loss = T.mean(actor_loss)
        self.actor.optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optim.step()

        self.update_network_parameters()

        tb_info = {'SACQ/q1_val': q1_new_policy.mean().detach(),
                   'SACQ/q2_val': q2_new_policy.mean().detach(),
                   'SACLoss/critic_loss': critic_loss,
                   'SACLoss/actor_loss': actor_loss,
                   'SACLoss/value_loss': value_loss,
                   'SACPolicy/avg. policy std plain': T.mean(pol_std).detach(),
                   'SACPolicy/avg. policy std repara': T.mean(pol_std_rep).detach()
                   }

        return tb_info























