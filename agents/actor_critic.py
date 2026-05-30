"""Shared-trunk Gaussian Actor-Critic for PPO."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """MLP actor-critic with a Gaussian policy head and a scalar value head.

    Parameters
    ----------
    obs_dim:
        Dimension of the flattened observation vector.
    act_dim:
        Dimension of the continuous action vector.
    hidden:
        Width of each hidden layer (two layers are used).
    device:
        Torch device for inference.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int = 256,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        with torch.no_grad():
            # bias the terminate dimension negative so the agent continues early on
            self.mean_head.bias[-1] = -1.0

    def _dist_value(self, obs: torch.Tensor):
        h = self.trunk(obs)
        dist = Normal(self.mean_head(h), torch.exp(self.log_std))
        return dist, self.value_head(h).squeeze(-1)

    @torch.no_grad()
    def act(
        self, obs_np: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, float, float]:
        """Sample (or take the mean of) an action given a numpy observation.

        Returns
        -------
        action: np.ndarray
        log_prob: float
        value: float
        """
        obs = torch.as_tensor(obs_np, device=self.device).unsqueeze(0)
        dist, value = self._dist_value(obs)
        action = dist.mean if deterministic else dist.sample()
        return (
            action.squeeze(0).cpu().numpy(),
            dist.log_prob(action).sum(-1).item(),
            value.item(),
        )

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return log-probs, entropies, and values for a batch of (obs, action) pairs."""
        dist, value = self._dist_value(obs)
        return dist.log_prob(actions).sum(-1), dist.entropy().sum(-1), value
