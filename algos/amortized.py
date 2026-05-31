"""Amortized differentiable pulse planner for qudit gate synthesis.

Unlike PPO/CEM, this trains a single network that maps a target unitary to a
pulse sequence (up to ``seq_len`` pulses) plus a learned halting distribution,
optimized by backpropagating the analytic gradient of the infidelity through a
differentiable rollout (``torch.matrix_exp``).

Variable length is handled PonderNet-style: the rollout exposes every prefix
``U_1..U_T``, the halt head defines a distribution ``p_k`` over where to stop,
and the loss is the expected ``infidelity + pulse_penalty * k`` under ``p_k``.
A larger ``pulse_penalty`` shifts ``p_k`` toward shorter sequences, all fully
differentiable (no discrete terminate action, so no RL-style collapse).
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
    pulse_penalty: float = 0.0
    eval_interval: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "amortized"
    checkpoint_meta: dict[str, Any] = field(default_factory=dict)


def _jx_offdiag(d: int, device: str) -> torch.Tensor:
    """Nearest-neighbor J_x off-diagonal couplings (same as make_jx_NN)."""
    k = torch.arange(d - 1, dtype=torch.float32, device=device)
    return 0.5 * torch.sqrt((k + 1) * (d - 1 - k))


def compose_prefix_unitaries(phis: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
    """Differentiably compose every prefix unitary from pulse parameters.

    Parameters
    ----------
    phis:
        Shape ``(B, T, d-1)`` drive phases per pulse.
    thetas:
        Shape ``(B, T)`` rotation angles per pulse.

    Returns
    -------
    Complex tensor ``(B, T, d, d)`` where entry ``k`` is ``D_k @ ... @ D_1 @ I``
    (the unitary after applying ``k+1`` pulses), autograd-connected to inputs.
    """
    batch, seq_len, d_minus_1 = phis.shape
    d = d_minus_1 + 1
    device = phis.device
    off_diag = _jx_offdiag(d, device).to(torch.cfloat)
    idx = torch.arange(d - 1, device=device)

    U = torch.eye(d, dtype=torch.cfloat, device=device).expand(batch, d, d).clone()
    prefixes = []
    for step in range(seq_len):
        phase = torch.exp(-1j * phis[:, step, :].to(torch.cfloat))  # (B, d-1)
        H = torch.zeros((batch, d, d), dtype=torch.cfloat, device=device)
        H[:, idx, idx + 1] = off_diag * phase
        H[:, idx + 1, idx] = off_diag * phase.conj()
        theta = thetas[:, step].to(torch.cfloat)
        pulse = torch.matrix_exp(-1j * theta[:, None, None] * H)
        U = pulse @ U
        prefixes.append(U)
    return torch.stack(prefixes, dim=1)  # (B, T, d, d)


def prefix_fidelity(U_prefix: torch.Tensor, U_target: torch.Tensor) -> torch.Tensor:
    """Phase-invariant fidelity per prefix: ``(B, T)`` for ``U_prefix (B,T,d,d)``."""
    d = U_prefix.shape[-1]
    overlap = torch.einsum("btij,bij->bt", U_prefix.conj(), U_target)
    return overlap.abs().square() / (d * d)


def halt_distribution(halt_logits: torch.Tensor) -> torch.Tensor:
    """PonderNet-style stop distribution ``p_k`` over steps from halt logits.

    ``lambda_k = sigmoid(halt_logits_k)`` is the conditional probability of
    stopping at step ``k`` given the chain reached it; ``p_k`` is the resulting
    unconditional stop probability. The final step absorbs all remaining mass so
    ``sum_k p_k = 1``.
    """
    lam = torch.sigmoid(halt_logits)  # (B, T)
    one_minus = 1.0 - lam
    incl = torch.cumprod(one_minus, dim=1)
    ones = torch.ones_like(incl[:, :1])
    survive = torch.cat([ones, incl[:, :-1]], dim=1)  # prob of reaching step k
    p = lam * survive
    # Force a halt at the last step: it takes the remaining survival mass.
    p = torch.cat([p[:, :-1], survive[:, -1:]], dim=1)
    return p


def _targets_to_tensor(targets: list[np.ndarray], device: str) -> torch.Tensor:
    arr = np.stack(targets).astype(np.complex64)
    return torch.as_tensor(arr, dtype=torch.cfloat, device=device)


def _flatten_targets(U_target: torch.Tensor) -> torch.Tensor:
    """Flatten complex targets to real features (B, 2*d*d) for the planner."""
    return torch.cat([U_target.real, U_target.imag], dim=-1).reshape(U_target.shape[0], -1)


def _infidelity_term(fidelity: torch.Tensor, U_prefix: torch.Tensor, U_target: torch.Tensor, name: str) -> torch.Tensor:
    """Per-prefix base objective (B, T), lower is better, before pulse penalty."""
    if name == "infidelity":
        return 1.0 - fidelity
    if name == "log-infidelity":
        return torch.log(torch.clamp(1.0 - fidelity, min=1e-9))
    if name == "l1":
        return (U_prefix - U_target[:, None]).abs().sum(dim=(2, 3))
    raise ValueError(f"Unknown loss {name!r}. Choose infidelity|log-infidelity|l1")


def _pulse_counts(seq_len: int, device: str) -> torch.Tensor:
    """1-based pulse count for each prefix step: tensor of shape (T,)."""
    return torch.arange(1, seq_len + 1, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(planner, eval_targets: torch.Tensor) -> tuple[float, float]:
    """Deterministic eval: stop at argmax halt step; return (mean_fid, mean_pulses)."""
    feats = _flatten_targets(eval_targets)
    phis, thetas, halt_logits = planner(feats)
    prefixes = compose_prefix_unitaries(phis, thetas)
    fids = prefix_fidelity(prefixes, eval_targets)  # (B, T)
    p = halt_distribution(halt_logits)  # (B, T)
    kstar = p.argmax(dim=1)  # (B,)
    rows = torch.arange(fids.shape[0], device=fids.device)
    chosen_fid = fids[rows, kstar]
    pulses = (kstar + 1).float()
    return float(chosen_fid.mean().item()), float(pulses.mean().item())


def train(
    sampler: TargetSampler,
    planner,
    cfg: AmortizedConfig,
    eval_targets: list[np.ndarray] | None = None,
) -> None:
    """Train the amortized planner with a differentiable halting objective."""
    planner.to(cfg.device)
    optimizer = torch.optim.Adam(planner.parameters(), lr=cfg.lr)
    counts = _pulse_counts(cfg.seq_len, cfg.device)  # (T,)

    eval_t = (
        _targets_to_tensor(eval_targets, cfg.device)
        if eval_targets
        else None
    )

    print(
        f"Amortized | d={cfg.d} max_pulses={cfg.seq_len} | "
        f"{cfg.iters} iters x {cfg.batch_targets} targets | "
        f"loss={cfg.loss} pulse_penalty={cfg.pulse_penalty} | "
        f"{cfg.checkpoint_name}"
    )

    with RunLogger(cfg.checkpoint_name) as logger:
        for it in range(1, cfg.iters + 1):
            batch = [sampler() for _ in range(cfg.batch_targets)]
            U_target = _targets_to_tensor(batch, cfg.device)
            feats = _flatten_targets(U_target)

            phis, thetas, halt_logits = planner(feats)
            prefixes = compose_prefix_unitaries(phis, thetas)
            fids = prefix_fidelity(prefixes, U_target)            # (B, T)
            p = halt_distribution(halt_logits)                    # (B, T)

            base = _infidelity_term(fids, prefixes, U_target, cfg.loss)
            objective = base + cfg.pulse_penalty * counts[None, :]
            # Expected objective under the learned stop distribution.
            loss = (p * objective).sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if it % cfg.eval_interval != 0 and it != cfg.iters:
                continue

            # Report the deployed behavior: fidelity/pulses at the chosen stop.
            exp_pulses = float((p * counts[None, :]).sum(dim=1).mean().item())
            train_fid = float((p * fids).sum(dim=1).mean().item())
            ef, ep = (None, None)
            if eval_t is not None:
                ef, ep = evaluate(planner, eval_t)
                print(
                    f"[iter {it:5d}] train_fid={train_fid:.4f}  "
                    f"E[pulses]={exp_pulses:4.1f}  loss={loss.item():.5f}  "
                    f"eval_fid={ef:.4f}  eval_pulses={ep:4.1f}"
                )

            logger.log(
                iter=it,
                timestep=it * cfg.batch_targets,
                episodes=cfg.batch_targets,
                train_fidelity=train_fid,
                train_pulses=exp_pulses,
                eval_fidelity=ef,
                eval_pulses=ep,
            )

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"{cfg.checkpoint_name}.pt")
    torch.save({"state_dict": planner.state_dict(), **cfg.checkpoint_meta}, path)
    print(f"Saved amortized planner to {path}")
    plot_run(os.path.join("output", f"{cfg.checkpoint_name}.csv"))
