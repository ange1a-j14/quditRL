"""
Reward functions for QuditEnv.

All reward functions must match the RewardFn signature:
    (U_current, U_target, n_pulses) -> float

Higher reward is better.  Distance-style rewards use 0 for a perfect match;
fidelity-style rewards use 1.
"""

from __future__ import annotations

import math
from typing import Callable

import torch


RewardFn = Callable[[torch.Tensor, torch.Tensor, int], float]
"""Type alias for reward functions.
(U_current: torch.Tensor, U_target: torch.Tensor, n_pulses: int) -> float
Both unitaries are shape ``(d, d)``; n_pulses is the number of pulses applied
so far this episode. Output is float; higher is better.
"""


def unitary_distance(U_current: torch.Tensor, U_target: torch.Tensor, n_pulses: int = 0) -> float:
    """Negative element-wise L1 distance between two unitaries.

    F(V, U) = -\Sigma_{i,j} |V_{ij} - U_{ij}|

    A perfect match produces 0; all other values are negative.
    """
    return -torch.sum((U_current - U_target).abs()).item()

def unitary_fidelity(U_current: torch.Tensor, U_target: torch.Tensor, n_pulses: int = 0) -> float:
    """
    Average gate fidelity for unitaries in SU(d).
    F = |Tr(U_current^† U_target)|² / d²  ∈ [0, 1]
    Perfect match → 1.0
    Advantages of unitary fidelity: bounded, phase-invariant
    Measures 'overlap' between two operators
    Consistent with standard quantum system performance reporting
    """
    d = U_current.shape[0]
    overlap = torch.trace(U_current.conj().T @ U_target)
    return (overlap.abs() ** 2 / (d * d)).item()

def frobenius_distance(U_current: torch.Tensor, U_target: torch.Tensor, n_pulses: int = 0) -> float:
    """Negative Frobenius norm: -||U - V||_F.
    Perfect match → 0.0
    """
    return -torch.linalg.norm(U_current - U_target).item()

REWARD_TYPES: dict[str, RewardFn] = {
    "l1": unitary_distance,
    "fidelity": unitary_fidelity,
    "frobenius": frobenius_distance,
}

def get_potential(name: str) -> RewardFn:
    if name not in REWARD_TYPES:
        raise ValueError(f"Unknown potential {name!r}. Choose from {list(REWARD_TYPES)}")
    return REWARD_TYPES[name]

def penalized_distance(step_penalty: float) -> RewardFn:
    """Reward factory: L1 distance with a flat per-pulse cost.

    r(V, U, n) = -||V - U||_1  -  step_penalty * n_pulses

    Parameters
    ----------
    step_penalty:
        Cost deducted for every pulse applied this episode.
    """
    def _reward(U_current: torch.Tensor, U_target: torch.Tensor, n_pulses: int) -> float:
        dist = -torch.sum((U_current - U_target).abs()).item()
        return dist - step_penalty * n_pulses
    return _reward


def penalized_fidelity(step_penalty: float) -> RewardFn:
    """Reward factory: gate fidelity with a flat per-pulse cost.

    r(V, U, n) = F(V, U) - step_penalty * n_pulses

    A perfect match with no pulse penalty yields 1.0. Higher is better.
    """
    def _reward(U_current: torch.Tensor, U_target: torch.Tensor, n_pulses: int) -> float:
        return unitary_fidelity(U_current, U_target) - step_penalty * n_pulses
    return _reward


def neg_log_infidelity(U_current: torch.Tensor, U_target: torch.Tensor, n_pulses: int = 0) -> float:
    """Negative log infidelity: -log(1 - F + eps).

    Sharpens the signal near the solution: as F -> 1 the reward grows large,
    rewarding the final approach to the target far more than raw fidelity.
    """
    eps = 1e-9
    fidelity = unitary_fidelity(U_current, U_target)
    return -math.log(max(1.0 - fidelity, eps))


def make_reward(name: str, step_penalty: float = 0.0) -> RewardFn:
    """Build a training reward by name, composing an optional per-pulse cost.

    Parameters
    ----------
    name:
        ``"l1"``            — negative L1 distance (smooth far-field gradient)
        ``"fidelity"``      — gate fidelity in [0, 1]
        ``"log-infidelity"`` — -log(1 - F), sharper near the solution
    step_penalty:
        Cost deducted for every pulse applied this episode (0 = none).
    """
    base_rewards: dict[str, RewardFn] = {
        "l1": unitary_distance,
        "fidelity": unitary_fidelity,
        "log-infidelity": neg_log_infidelity,
    }
    if name not in base_rewards:
        raise ValueError(
            f"Unknown reward {name!r}. Choose from {list(base_rewards)}"
        )
    base = base_rewards[name]
    if step_penalty <= 0.0:
        return base

    def _reward(U_current: torch.Tensor, U_target: torch.Tensor, n_pulses: int) -> float:
        return base(U_current, U_target, n_pulses) - step_penalty * n_pulses
    return _reward
