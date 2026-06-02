"""Amortized differentiable pulse planner for qudit gate synthesis.

A single network maps a target unitary to a fixed-length pulse sequence,
optimized by backpropagating the analytic gradient of the infidelity through a
differentiable rollout (``torch.matrix_exp``). This is the recipe that reaches
~1e-3 infidelity on d=3 Haar targets.

For higher dimensions (d>=4) plain Haar training gets stuck in the
target-independent ``1/d^2`` minimum (the planner ignores the target). A
target-difficulty curriculum fixes this: train on random pulse-circuit targets
of annealed depth (easy, guaranteed reachable, near identity) before graduating
to the real target distribution, so the network first learns to condition on the
target and then hardens.
"""

from __future__ import annotations

import math
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
    # Target-difficulty curriculum: train on random pulse-circuit targets of
    # annealed depth before graduating to the real target distribution.
    target_curriculum: bool = False
    curriculum_frac: float = 0.5
    curriculum_start_pulses: int = 1
    curriculum_end_pulses: int | None = None
    # Cosine LR decay to eta_min = lr * lr_min_frac over training; lowers the
    # end-of-training infidelity plateau and the eval noise.
    lr_min_frac: float = 0.01
    # Max global grad norm; keeps the deep matrix_exp rollout stable at high d.
    grad_clip: float = 1.0
    eval_interval: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "amortized"
    checkpoint_meta: dict[str, Any] = field(default_factory=dict)


def _jx_offdiag(d: int, device: str) -> torch.Tensor:
    """Nearest-neighbor J_x off-diagonal couplings (same as make_jx_NN)."""
    k = torch.arange(d - 1, dtype=torch.float32, device=device)
    return 0.5 * torch.sqrt((k + 1) * (d - 1 - k))


def compose_final_unitary(phis: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
    """Differentiably compose the full-sequence unitary from pulse parameters.

    Parameters
    ----------
    phis:
        Shape ``(B, T, d-1)`` drive phases per pulse.
    thetas:
        Shape ``(B, T)`` rotation angles per pulse.

    Returns
    -------
    Complex tensor ``(B, d, d)`` = ``D_T @ ... @ D_1 @ I``, autograd-connected.
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


def _random_circuit_targets(batch: int, d: int, depth: int, device: str) -> torch.Tensor:
    """Sample reachable SU(d) targets as random pulse circuits of given depth.

    Built on-device with the same (no-grad) rollout, so every target is exactly
    reachable in ``depth`` pulses. Small depth keeps targets near identity, which
    gives the planner an easy, strongly-conditioned learning signal.
    """
    phis = (torch.rand(batch, depth, d - 1, device=device) * 2 - 1) * math.pi
    thetas = torch.rand(batch, depth, device=device) * math.pi
    with torch.no_grad():
        return compose_final_unitary(phis, thetas)


def _scheduled_curriculum_depth(cfg: AmortizedConfig, it: int) -> int | None:
    """Curriculum circuit depth at iteration ``it``; None once graduated to real targets."""
    ramp_iters = max(1, int(cfg.iters * cfg.curriculum_frac))
    if it > ramp_iters:
        return None
    end = cfg.curriculum_end_pulses or cfg.seq_len
    progress = (it - 1) / max(1, ramp_iters - 1)
    depth = round(cfg.curriculum_start_pulses + progress * (end - cfg.curriculum_start_pulses))
    return max(1, min(int(depth), cfg.seq_len))


def _loss_from_fidelity(fidelity: torch.Tensor, U_pred: torch.Tensor, U_target: torch.Tensor, name: str) -> torch.Tensor:
    if name == "infidelity":
        return (1.0 - fidelity).mean()
    if name == "log-infidelity":
        return torch.log(torch.clamp(1.0 - fidelity, min=1e-9)).mean()
    if name == "l1":
        return (U_pred - U_target).abs().sum(dim=(1, 2)).mean()
    raise ValueError(f"Unknown loss {name!r}. Choose infidelity|log-infidelity|l1")


@torch.no_grad()
def evaluate(planner, eval_targets: torch.Tensor) -> tuple[float, float]:
    """Fixed-length eval: apply all seq_len pulses; return (mean_fid, n_pulses)."""
    feats = _flatten_targets(eval_targets)
    phis, thetas = planner(feats)
    U_pred = compose_final_unitary(phis, thetas)
    return float(batch_fidelity(U_pred, eval_targets).mean().item()), float(phis.shape[1])


def train(
    sampler: TargetSampler,
    planner,
    cfg: AmortizedConfig,
    eval_targets: list[np.ndarray] | None = None,
) -> None:
    """Train the fixed-length amortized planner (optionally with curriculum)."""
    planner.to(cfg.device)
    optimizer = torch.optim.Adam(planner.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.iters, eta_min=cfg.lr * cfg.lr_min_frac
    )

    eval_t = (
        _targets_to_tensor(eval_targets, cfg.device)
        if eval_targets
        else None
    )

    print(
        f"Amortized | d={cfg.d} pulses={cfg.seq_len} | "
        f"{cfg.iters} iters x {cfg.batch_targets} targets | "
        f"loss={cfg.loss} | {cfg.checkpoint_name}"
    )
    if cfg.target_curriculum:
        end = cfg.curriculum_end_pulses or cfg.seq_len
        print(
            "Target curriculum | random-circuit depth "
            f"{cfg.curriculum_start_pulses} -> {end} over "
            f"{cfg.curriculum_frac:g} of training, then real targets"
        )

    with RunLogger(cfg.checkpoint_name) as logger:
        for it in range(1, cfg.iters + 1):
            depth = _scheduled_curriculum_depth(cfg, it) if cfg.target_curriculum else None
            if depth is not None:
                U_target = _random_circuit_targets(cfg.batch_targets, cfg.d, depth, cfg.device)
            else:
                batch = [sampler() for _ in range(cfg.batch_targets)]
                U_target = _targets_to_tensor(batch, cfg.device)
            feats = _flatten_targets(U_target)

            phis, thetas = planner(feats)
            U_pred = compose_final_unitary(phis, thetas)
            fidelity = batch_fidelity(U_pred, U_target)
            loss = _loss_from_fidelity(fidelity, U_pred, U_target, cfg.loss)

            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(planner.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            if it % cfg.eval_interval != 0 and it != cfg.iters:
                continue

            train_fid = float(fidelity.mean().item())
            ef, ep = (None, None)
            if eval_t is not None:
                ef, ep = evaluate(planner, eval_t)
                depth_str = f"depth={depth}" if depth is not None else "depth=real"
                print(
                    f"[iter {it:5d}] {depth_str}  train_fid={train_fid:.4f}  "
                    f"loss={loss.item():.5f}  eval_fid={ef:.4f}"
                )

            logger.log(
                iter=it,
                timestep=it * cfg.batch_targets,
                episodes=cfg.batch_targets,
                train_fidelity=train_fid,
                train_pulses=cfg.seq_len,
                eval_fidelity=ef,
                eval_pulses=ep,
            )

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"{cfg.checkpoint_name}.pt")
    torch.save({"state_dict": planner.state_dict(), **cfg.checkpoint_meta}, path)
    print(f"Saved amortized planner to {path}")
    plot_run(os.path.join("output", f"{cfg.checkpoint_name}.csv"))
