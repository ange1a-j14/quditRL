"""Amortized differentiable pulse planner for qudit gate synthesis.

Unlike PPO/CEM, this trains a single network that maps a target unitary to a
fixed-length pulse sequence, and optimizes it by backpropagating the analytic
gradient of the infidelity through a differentiable rollout (``torch.matrix_exp``).

This is the same analytic-gradient signal the notebook's direct optimizer uses
to reach ~1e-4 infidelity, but amortized: one network solves any target in a
single forward pass instead of re-optimizing each target from scratch.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics import RunLogger, plot_run


TargetSampler = Callable[[], np.ndarray]


@dataclass
class AmortizedConfig:
    d: int
    seq_len: int
    iters: int = 20000
    batch_targets: int = 256
    lr: float = 1e-3
    loss: str = "infidelity"
    eval_interval: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "amortized"
    checkpoint_meta: dict[str, Any] = field(default_factory=dict)


def _jx_offdiag(d: int, device: str) -> torch.Tensor:
    """Nearest-neighbor J_x off-diagonal couplings (same as make_jx_NN)."""
    k = torch.arange(d - 1, dtype=torch.float32, device=device)
    return 0.5 * torch.sqrt((k + 1) * (d - 1 - k))


def compose_unitaries(phis: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
    """Differentiably compose a batch of unitaries from pulse parameters.

    Parameters
    ----------
    phis:
        Shape ``(B, T, d-1)`` drive phases per pulse.
    thetas:
        Shape ``(B, T)`` rotation angles per pulse.

    Returns
    -------
    Complex tensor ``(B, d, d)`` = ``D_T @ ... @ D_1 @ I``, autograd-connected
    to ``phis``/``thetas``.
    """
    batch, seq_len, d_minus_1 = phis.shape
    d = d_minus_1 + 1
    device = phis.device
    off_diag = _jx_offdiag(d, device).to(torch.cfloat)
    idx = torch.arange(d - 1, device=device)

    U = torch.eye(d, dtype=torch.cfloat, device=device).expand(batch, d, d).clone()
    for step in range(seq_len):
        phase = torch.exp(-1j * phis[:, step, :].to(torch.cfloat))  # (B, d-1)
        H = torch.zeros((batch, d, d), dtype=torch.cfloat, device=device)
        H[:, idx, idx + 1] = off_diag * phase
        H[:, idx + 1, idx] = off_diag * phase.conj()
        theta = thetas[:, step].to(torch.cfloat)
        pulse = torch.matrix_exp(-1j * theta[:, None, None] * H)
        U = pulse @ U
    return U


def batch_fidelity(U_pred: torch.Tensor, U_target: torch.Tensor) -> torch.Tensor:
    """Phase-invariant gate fidelity per item: |Tr(U_pred^† U_target)|^2 / d^2."""
    d = U_pred.shape[-1]
    overlap = torch.einsum("bij,bij->b", U_pred.conj(), U_target)
    return overlap.abs().square() / (d * d)


def _targets_to_tensor(targets: list[np.ndarray], device: str) -> torch.Tensor:
    arr = np.stack(targets).astype(np.complex64)
    return torch.as_tensor(arr, dtype=torch.cfloat, device=device)


def _flatten_targets(U_target: torch.Tensor) -> torch.Tensor:
    """Flatten complex targets to real features (B, 2*d*d) for the planner."""
    return torch.cat([U_target.real, U_target.imag], dim=-1).reshape(U_target.shape[0], -1)


def _loss_from_fidelity(fidelity: torch.Tensor, U_pred: torch.Tensor, U_target: torch.Tensor, name: str) -> torch.Tensor:
    if name == "infidelity":
        return (1.0 - fidelity).mean()
    if name == "log-infidelity":
        return torch.log(torch.clamp(1.0 - fidelity, min=1e-9)).mean()
    if name == "l1":
        return (U_pred - U_target).abs().sum(dim=(1, 2)).mean()
    raise ValueError(f"Unknown loss {name!r}. Choose infidelity|log-infidelity|l1")


@torch.no_grad()
def evaluate(planner, eval_targets: torch.Tensor, seq_len: int) -> float:
    """Mean fidelity of the planned sequences on a fixed target set."""
    feats = _flatten_targets(eval_targets)
    phis, thetas = planner(feats, seq_len)
    U_pred = compose_unitaries(phis, thetas)
    return float(batch_fidelity(U_pred, eval_targets).mean().item())


def train(
    sampler: TargetSampler,
    planner,
    cfg: AmortizedConfig,
    eval_targets: list[np.ndarray] | None = None,
) -> None:
    """Train the amortized planner by differentiable infidelity minimization."""
    planner.to(cfg.device)
    optimizer = torch.optim.Adam(planner.parameters(), lr=cfg.lr)

    eval_t = (
        _targets_to_tensor(eval_targets, cfg.device)
        if eval_targets
        else None
    )

    print(
        f"Amortized | d={cfg.d} seq_len={cfg.seq_len} | "
        f"{cfg.iters} iters x {cfg.batch_targets} targets | "
        f"loss={cfg.loss} | {cfg.checkpoint_name}"
    )

    with RunLogger(cfg.checkpoint_name) as logger:
        for it in range(1, cfg.iters + 1):
            batch = [sampler() for _ in range(cfg.batch_targets)]
            U_target = _targets_to_tensor(batch, cfg.device)
            feats = _flatten_targets(U_target)

            phis, thetas = planner(feats, cfg.seq_len)
            U_pred = compose_unitaries(phis, thetas)
            fidelity = batch_fidelity(U_pred, U_target)
            loss = _loss_from_fidelity(fidelity, U_pred, U_target, cfg.loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Only log/evaluate on the eval cadence (and the final step) to keep
            # the metrics CSV small over many thousands of gradient steps.
            if it % cfg.eval_interval != 0 and it != cfg.iters:
                continue

            train_fid = float(fidelity.mean().item())
            ef = evaluate(planner, eval_t, cfg.seq_len) if eval_t is not None else None
            if ef is not None:
                print(
                    f"[iter {it:5d}] train_fid={train_fid:.4f}  "
                    f"loss={loss.item():.5f}  eval_fid={ef:.4f}"
                )

            logger.log(
                iter=it,
                timestep=it * cfg.batch_targets,
                episodes=cfg.batch_targets,
                train_fidelity=train_fid,
                train_pulses=cfg.seq_len,
                eval_fidelity=ef,
                eval_pulses=cfg.seq_len if ef is not None else None,
            )

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"{cfg.checkpoint_name}.pt")
    torch.save({"state_dict": planner.state_dict(), **cfg.checkpoint_meta}, path)
    print(f"Saved amortized planner to {path}")
    plot_run(os.path.join("output", f"{cfg.checkpoint_name}.csv"))
