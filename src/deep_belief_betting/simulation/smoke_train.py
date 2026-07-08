from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from deep_belief_betting.simulation.env_factory import make_vector_env
from deep_belief_betting.simulation.parameters import Parameters


@dataclass
class SmokeTrainConfig:
    """Small config for smoke training."""

    num_envs: int = 4
    num_updates: int = 10
    rollout_steps: int = 32
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    gamma: float = 0.99
    device: str = "cpu"


class SmallPolicyValueNet(nn.Module):
    """Tiny shared backbone for smoke tests."""

    def __init__(self, obs_dim: int, hidden_dim: int, num_actions: int):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return policy logits and state values."""
        x = self.backbone(obs)
        logits = self.policy_head(x)
        values = self.value_head(x).squeeze(-1)
        return logits, values


def masked_categorical_sample(logits: torch.Tensor, action_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample from a masked categorical distribution."""
    # push invalid actions far away
    masked_logits = logits.masked_fill(action_mask == 0, -1e9)
    dist = torch.distributions.Categorical(logits=masked_logits)
    actions = dist.sample()
    log_probs = dist.log_prob(actions)
    return actions, log_probs


def compute_returns(
    rewards: List[torch.Tensor],
    dones: List[torch.Tensor],
    last_values: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute simple discounted returns."""
    returns: List[torch.Tensor] = []
    running = last_values

    for reward_t, done_t in zip(reversed(rewards), reversed(dones)):
        running = reward_t + gamma * running * (1.0 - done_t)
        returns.append(running)

    returns.reverse()
    return torch.stack(returns, dim=0)


def run_smoke_training(params: Parameters, config: SmokeTrainConfig) -> None:
    """Run a tiny masked actor critic style smoke test."""
    device = torch.device(config.device)
    env = make_vector_env(params=params, num_envs=config.num_envs, belief_dim=0)

    obs, infos = env.reset(seed=params.seed)
    obs_dim = obs.shape[-1]
    num_actions = int(env.single_action_space.n)

    model = SmallPolicyValueNet(
        obs_dim=obs_dim,
        hidden_dim=config.hidden_dim,
        num_actions=num_actions,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    for update_idx in range(config.num_updates):
        obs_buf: List[torch.Tensor] = []
        action_buf: List[torch.Tensor] = []
        log_prob_buf: List[torch.Tensor] = []
        reward_buf: List[torch.Tensor] = []
        done_buf: List[torch.Tensor] = []
        value_buf: List[torch.Tensor] = []

        for step_idx in range(config.rollout_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

            # vector env info gives batched arrays
            action_mask_np = infos["action_mask"]
            action_mask_t = torch.as_tensor(action_mask_np, dtype=torch.int64, device=device)

            logits_t, values_t = model(obs_t)
            actions_t, log_probs_t = masked_categorical_sample(logits_t, action_mask_t)

            next_obs, rewards, terminated, truncated, next_infos = env.step(actions_t.cpu().numpy())
            dones = np.logical_or(terminated, truncated).astype(np.float32)

            obs_buf.append(obs_t)
            action_buf.append(actions_t)
            log_prob_buf.append(log_probs_t)
            reward_buf.append(torch.as_tensor(rewards, dtype=torch.float32, device=device))
            done_buf.append(torch.as_tensor(dones, dtype=torch.float32, device=device))
            value_buf.append(values_t)

            obs = next_obs
            infos = next_infos

        with torch.no_grad():
            last_obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            _, last_values = model(last_obs_t)

        returns = compute_returns(
            rewards=reward_buf,
            dones=done_buf,
            last_values=last_values,
            gamma=config.gamma,
        )

        values = torch.stack(value_buf, dim=0)
        log_probs = torch.stack(log_prob_buf, dim=0)

        advantages = returns - values
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = 0.5 * advantages.pow(2).mean()
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        mean_reward = torch.stack(reward_buf, dim=0).sum(dim=0).mean().item()
        print(
            f"update={update_idx:03d} "
            f"loss={loss.item():.4f} "
            f"policy_loss={policy_loss.item():.4f} "
            f"value_loss={value_loss.item():.4f} "
            f"mean_rollout_reward={mean_reward:.4f}"
        )

    env.close()


if __name__ == "__main__":
    params = Parameters.from_yaml("configs/default.yaml")
    config = SmokeTrainConfig()
    run_smoke_training(params=params, config=config)