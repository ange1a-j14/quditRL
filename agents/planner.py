"""Amortized pulse-sequence planner.

A feedforward network that maps a target unitary to a fixed-length pulse
sequence in one shot. Trained end-to-end by backprop through a differentiable
rollout (see algos/amortized.py) against the gate infidelity.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PulseSequencePlanner(nn.Module):
    """MLP mapping a flattened target unitary to a fixed-length pulse sequence.

    Parameters
    ----------
    d:
        Qudit dimension. Input is the real/imag parts of U_target, length
        ``2*d*d``; each pulse has ``d`` parameters (``d-1`` phases + 1 angle).
    seq_len:
        Number of pulses emitted.
    hidden:
        Width of each hidden layer.
    """

    def __init__(self, d: int, seq_len: int, hidden: int = 256) -> None:
        super().__init__()
        self.d = d
        self.seq_len = seq_len
        in_dim = 2 * d * d
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pulse_head = nn.Linear(hidden, seq_len * d)
        # Identity-block init: start with ~zero pulse params so the composed
        # unitary is ~I. This places the planner next to the easy (near-identity)
        # curriculum targets with a strong, non-vanishing gradient, avoiding the
        # barren plateau that stalls deeper/higher-d rollouts at the 1/d^2 floor.
        nn.init.zeros_(self.pulse_head.bias)
        with torch.no_grad():
            self.pulse_head.weight.mul_(1e-2)

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(phis, thetas)`` for a batch of targets.

        feats:
            Shape ``(B, 2*d*d)``.

        Returns
        -------
        phis:
            Shape ``(B, T, d-1)`` drive phases.
        thetas:
            Shape ``(B, T)`` rotation angles.
        """
        h = self.trunk(feats)
        raw = self.pulse_head(h).reshape(-1, self.seq_len, self.d)
        phis = raw[..., : self.d - 1]
        thetas = raw[..., self.d - 1]
        return phis, thetas
