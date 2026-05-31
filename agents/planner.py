"""Amortized pulse-sequence planner.

A feedforward network that maps a target unitary to a full fixed-length pulse
sequence in one shot. Trained end-to-end by backprop through a differentiable
rollout (see algos/amortized.py), so it learns to emit pulses that synthesize
the target rather than imitating a teacher.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PulseSequencePlanner(nn.Module):
    """MLP mapping a flattened target unitary to pulse parameters.

    Parameters
    ----------
    d:
        Qudit dimension. Input is the real/imag parts of U_target, length
        ``2*d*d``; each pulse has ``d`` parameters (``d-1`` phases + 1 angle).
    seq_len:
        Number of pulses the planner emits.
    hidden:
        Width of each hidden layer.
    """

    def __init__(self, d: int, seq_len: int, hidden: int = 256) -> None:
        super().__init__()
        self.d = d
        self.seq_len = seq_len
        in_dim = 2 * d * d
        out_dim = seq_len * d
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(
        self, feats: torch.Tensor, seq_len: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(phis, thetas)`` for a batch of flattened targets.

        feats:
            Shape ``(B, 2*d*d)``.

        Returns
        -------
        phis:
            Shape ``(B, T, d-1)`` drive phases.
        thetas:
            Shape ``(B, T)`` rotation angles.
        """
        t = seq_len if seq_len is not None else self.seq_len
        raw = self.net(feats).reshape(-1, t, self.d)
        phis = raw[..., : self.d - 1]
        thetas = raw[..., self.d - 1]
        return phis, thetas
