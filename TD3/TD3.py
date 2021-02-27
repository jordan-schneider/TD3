import copy
import logging
from typing import List, Optional, Sequence, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


def make_linear_layers(n_nodes: Sequence[int]) -> nn.ModuleList:
    assert len(n_nodes) >= 2, "Must have at least an input and output node set."
    return nn.ModuleList([nn.Linear(n_nodes[i], n_nodes[i + 1]) for i in range(len(n_nodes) - 1)])


def relu_tanh_forward(x: torch.Tensor, layers: Sequence[nn.Module]) -> torch.Tensor:
    out = x
    for layer in layers[:-1]:
        out = F.relu(layer(out))
    out = torch.tanh(layers[-1](out))
    return out


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, **kwargs):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def forward(self, state) -> torch.Tensor:
        raise NotImplementedError()


class LinearActor(Actor, nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, max_action: float, layers: Sequence[int] = [256, 256]
    ):
        super(LinearActor, self).__init__(state_dim, action_dim, max_action)

        layer_sizes = [state_dim] + list(layers) + [action_dim]
        logging.debug(f"Actor layers sizes are {layer_sizes}")
        self.layers = make_linear_layers(layer_sizes)
        logging.debug(f"There are {len(self.layers)} layers.")

        self.max_action = max_action

    def forward(self, state) -> torch.Tensor:
        return relu_tanh_forward(state, self.layers)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, state, action) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def Q1(self, state, action) -> torch.Tensor:
        raise NotImplementedError()


class LinearCritic(Critic):
    def __init__(self, state_dim: int, action_dim: int, layers: Sequence[int] = [256, 256]):
        super(LinearCritic, self).__init__(state_dim, action_dim)

        self.q1_layers = make_linear_layers([state_dim + action_dim] + list(layers) + [1])
        self.q2_layers = make_linear_layers([state_dim + action_dim] + list(layers) + [1])

    def forward(self, state, action) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)

        q1 = relu_tanh_forward(sa, self.q1_layers)
        q2 = relu_tanh_forward(sa, self.q2_layers)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return relu_tanh_forward(sa, self.q1_layers)


class TD3(object):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        actor_type: Type[Actor] = LinearActor,
        critic_type: Type[Critic] = LinearCritic,
        actor_kwargs: dict = dict(),
        critic_kwargs: dict = dict(),
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        writer: Optional[SummaryWriter] = None,
        log_weight_period: int = int(1e3),
    ):

        self.actor = actor_type(state_dim, action_dim, max_action, **actor_kwargs).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = critic_type(state_dim, action_dim, **critic_kwargs).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

        self.writer = writer
        self.writer_iter = 0

        self.log_weight_period = log_weight_period

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size: int = 100):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        q1_loss = F.mse_loss(current_Q1, target_Q)
        q2_loss = F.mse_loss(current_Q2, target_Q)
        critic_loss = q1_loss + q2_loss
        if self.writer is not None:
            self.writer.add_histogram("Critic/Q1/Value", current_Q1, self.writer_iter)
            self.writer.add_histogram("Critic/Q2/Value", current_Q2, self.writer_iter)
            self.writer.add_histogram("Critic/target_Q", target_Q, self.writer_iter)
            self.writer.add_scalar("Critic/Q1/loss", q1_loss, self.writer_iter)
            self.writer.add_scalar("Critic/Q/loss", q2_loss, self.writer_iter)
            self.writer.add_scalar("Critic/loss", critic_loss, self.writer_iter)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.log_weight_period == 0:
            self.log_critic()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            if self.writer is not None:
                self.writer.add_scalar("Actor/loss", actor_loss, self.writer_iter)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if self.total_it % self.log_weight_period == 0:
                self.log_actor()
        self.writer_iter += 1

    def log_actor(self):
        if self.writer is None:
            raise ValueError("Tensorboard writer not provided")

        for i, layer in enumerate(self.actor.layers):
            self.writer.add_histogram(f"Actor/W/{i}", layer.weight, self.writer_iter)

    def log_critic(self):
        if self.writer is None:
            raise ValueError("Tensorboard writer not provided")
        for i, layer in enumerate(self.critic.q1_layers):
            self.writer.add_histogram(f"Critic/Q1/W/{i}", layer.weight, self.writer_iter)

        for i, layer in enumerate(self.critic.q2_layers):
            self.writer.add_histogram(f"Critic/Q2/W/{i}", layer.weight, self.writer_iter)

    def save(self, filename: str):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename: str):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
