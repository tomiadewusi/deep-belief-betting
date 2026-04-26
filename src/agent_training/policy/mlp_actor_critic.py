from __future__ import annotations
import torch
import torch.nn as nn


def _create_backbone(obs_dim: int, hidden_dim: int, num_layers: int) -> nn.Module:
    layers = []
    for _ in range(num_layers):
        layers.append(nn.Linear(obs_dim, hidden_dim))
        layers.append(nn.ReLU())
        obs_dim = hidden_dim
    return nn.Sequential(*layers)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int, num_layers: int, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.backbone = _create_backbone(obs_dim, hidden_dim, num_layers)
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.backbone(obs)
        logits = self.policy_head(z)
        values = self.value_head(z).squeeze(-1)
        return logits, values