"""A small hybrid-action PPO teaching implementation for the public toy environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.direction_logits = nn.Linear(hidden_dim, 2)
        self.magnitude_mean = nn.Linear(hidden_dim, 1)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.tensor([-0.5]))

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Shared layers 先把 state 轉成 feature。
        features = self.body(states)
        # Actor 有兩個 head：Categorical direction 與 Gaussian latent magnitude。
        direction_logits = self.direction_logits(features)
        magnitude_mean = self.magnitude_mean(features)
        std = self.log_std.exp().expand_as(magnitude_mean)
        value = self.critic(features).squeeze(-1)
        return direction_logits, magnitude_mean, std, value


@dataclass
class Rollout:
    # Rollout 是 Agent 與 environment 互動後暫存的一批經驗。
    states: list[np.ndarray]
    directions: list[int]
    magnitudes: list[float]
    rewards: list[float]
    dones: list[bool]
    log_probs: list[float]
    values: list[float]


class PPOAgent:
    def __init__(self, state_dim: int, hidden_dim: int, actor_lr: float, critic_lr: float, seed: int):
        torch.manual_seed(seed)
        self.model = ActorCritic(state_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": list(self.model.body.parameters())
                    + list(self.model.direction_logits.parameters())
                    + list(self.model.magnitude_mean.parameters())
                    + [self.model.log_std],
                    "lr": actor_lr,
                },
                {"params": self.model.critic.parameters(), "lr": critic_lr},
            ]
        )

    @staticmethod
    def _sigmoid_squashed_log_prob(dist: Normal, pre_sigmoid_magnitudes: torch.Tensor) -> torch.Tensor:
        """Compute log pi(m|s) after transforming a Normal sample with sigmoid."""
        magnitudes = torch.sigmoid(pre_sigmoid_magnitudes)
        log_det_jacobian = torch.log(magnitudes * (1.0 - magnitudes) + 1e-6)
        return (dist.log_prob(pre_sigmoid_magnitudes) - log_det_jacobian).sum(dim=-1)

    @torch.no_grad()
    def act(self, state: np.ndarray, deterministic: bool = False) -> tuple[int, float, float, float]:
        """Select hybrid actions and return their joint log probability and state value."""
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        direction_logits, magnitude_mean, std, value = self.model(state_t)
        direction_dist = Categorical(logits=direction_logits)
        magnitude_dist = Normal(magnitude_mean, std)
        direction_index = direction_logits.argmax(dim=-1) if deterministic else direction_dist.sample()
        pre_sigmoid_magnitude = magnitude_mean if deterministic else magnitude_dist.sample()
        magnitude = torch.sigmoid(pre_sigmoid_magnitude)
        direction_log_prob = direction_dist.log_prob(direction_index)
        magnitude_log_prob = self._sigmoid_squashed_log_prob(magnitude_dist, pre_sigmoid_magnitude)
        joint_log_prob = direction_log_prob + magnitude_log_prob
        direction = -1 if int(direction_index.item()) == 0 else 1
        return direction, float(magnitude.item()), float(joint_log_prob.item()), float(value.item())

    def update(
        self,
        rollout: Rollout,
        gamma: float,
        gae_lambda: float,
        clip_epsilon: float,
        entropy_coef: float,
        epochs: int,
        batch_size: int,
    ) -> float:
        # 將 rollout 轉成 tensor，準備計算 advantage 與 PPO loss。
        states = torch.as_tensor(np.asarray(rollout.states), dtype=torch.float32)
        direction_indices = ((torch.as_tensor(np.asarray(rollout.directions), dtype=torch.long) + 1) // 2)
        magnitudes = torch.as_tensor(np.asarray(rollout.magnitudes), dtype=torch.float32).unsqueeze(1)
        old_log_probs = torch.as_tensor(rollout.log_probs, dtype=torch.float32)
        rewards = np.asarray(rollout.rewards, dtype=np.float32)
        dones = np.asarray(rollout.dones, dtype=np.float32)
        values = np.asarray(rollout.values + [0.0], dtype=np.float32)

        # GAE 將多步 TD residual 加權累積，以估計 action 相對於 Critic baseline 的 advantage。
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            delta = rewards[index] + gamma * values[index + 1] * (1.0 - dones[index]) - values[index]
            gae = delta + gamma * gae_lambda * (1.0 - dones[index]) * gae
            advantages[index] = gae
        returns = advantages + values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        losses: list[float] = []
        for _ in range(epochs):
            order = torch.randperm(len(states))
            for start in range(0, len(states), batch_size):
                idx = order[start : start + batch_size]
                direction_logits, magnitude_mean, std, value = self.model(states[idx])
                direction_dist = Categorical(logits=direction_logits)
                magnitude_dist = Normal(magnitude_mean, std)
                # Recover the latent Normal sample so old and new policies score
                # the same sigmoid-transformed magnitude and discrete direction.
                bounded_magnitudes = magnitudes[idx].clamp(1e-6, 1.0 - 1e-6)
                pre_sigmoid_magnitudes = torch.logit(bounded_magnitudes)
                direction_log_probs = direction_dist.log_prob(direction_indices[idx])
                magnitude_log_probs = self._sigmoid_squashed_log_prob(magnitude_dist, pre_sigmoid_magnitudes)
                log_probs = direction_log_probs + magnitude_log_probs
                # 新策略／舊策略的機率比，是 PPO clipping 的核心。
                ratio = (log_probs - old_log_probs[idx]).exp()
                adv = torch.as_tensor(advantages, dtype=torch.float32)[idx]
                target_returns = torch.as_tensor(returns, dtype=torch.float32)[idx]
                # clip 防止一次更新太大，讓 training 比較穩定。
                policy_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv).mean()
                value_loss = 0.5 * (target_returns - value).pow(2).mean()
                # Categorical entropy is analytic. The sigmoid-squashed Normal
                # entropy is estimated from a fresh reparameterized sample.
                discrete_entropy = direction_dist.entropy()
                continuous_entropy = -self._sigmoid_squashed_log_prob(magnitude_dist, magnitude_dist.rsample())
                entropy_bonus = (discrete_entropy + continuous_entropy).mean()
                loss = policy_loss + value_loss - entropy_coef * entropy_bonus
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else 0.0
