"""

Rewawrd functions should follow the input output format of RewardFn

e.g. ``unitary_distance``: negative L1 distance between two unitaries.
  Suitable as both a per-step and terminal reward.
"""

from __future__ import annotations

from typing import Callable

import torch


RewardFn = Callable[[torch.Tensor, torch.Tensor], float]
"""Type alias for reward functions.
(U_current: torch.Tensor, U_target: torch.Tensor) -> float
Both are shape ``(d, d)`` and output is float. Higher reward => better.
"""


def unitary_distance(U_current: torch.Tensor, U_target: torch.Tensor) -> float:
    """Negative element-wise L1 distance between two unitaries.

F(V, U) = -\Sigma_{i,j} |V_{ij} - U_{ij}|

    A perfect match produces 0; all other values are negative.
    neg loss so higher is better

    Parameters
    ----------
    U_current: current unitary
    U_target: target unitary
    """
    return -torch.sum((U_current - U_target).abs()).item()
