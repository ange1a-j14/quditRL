"""Amortized pulse-sequence planner with a learned halting head.

A feedforward network that maps a target unitary to a full pulse sequence of
length up to ``seq_len`` in one shot, plus a per-step halt logit. Trained
end-to-end by backprop through a differentiable rollout (see algos/amortized.py)
with a PonderNet-style halting distribution, so it learns both the pulses and
when to stop. A per-pulse penalty pushes the halting distribution toward shorter
sequences without any discrete/non-differentiable termination.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PulseSequencePlanner(nn.Module):
    """MLP mapping a flattened target unitary to pulse params + halt logits.

    Parameters
    ----------
    d:
        Qudit dimension. Input is the real/imag parts of U_target, length
        ``2*d*d``; each pulse has ``d`` parameters (``d-1`` phases + 1 angle).
    seq_len:
        Maximum number of pulses; the halt head decides where to stop.
    hidden:
        Width of each hidden layer.
    """

    def __init__(self, d: int, seq_len: int, hidden: int = 256) -> None:
        super().__init__()
        self.d = d
        self.seq_len = seq_len
        in_dim = 2 * d * d
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pulse_head = nn.Linear(hidden, seq_len * d)
        self.halt_head = nn.Linear(hidden, seq_len)

    def forward(
        self, feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(phis, thetas, halt_logits)`` for a batch of targets.

        feats:
            Shape ``(B, 2*d*d)``.

        Returns
        -------
        phis:
            Shape ``(B, T, d-1)`` drive phases.
        thetas:
            Shape ``(B, T)`` rotation angles.
        halt_logits:
            Shape ``(B, T)`` per-step conditional halt logits.
        """
        h = self.trunk(feats)
        raw = self.pulse_head(h).reshape(-1, self.seq_len, self.d)
        phis = raw[..., : self.d - 1]
        thetas = raw[..., self.d - 1]
        halt_logits = self.halt_head(h)
        return phis, thetas, halt_logits
