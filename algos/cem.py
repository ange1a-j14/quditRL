"""Cross-entropy trajectory optimizer for qudit pulse synthesis.

This is an optimal-control baseline rather than a learned policy.  For each
target unitary, it directly searches over a fixed-length sequence of pulse
parameters and logs the best sequence found.  If this succeeds where PPO fails,
the issue is policy optimization/exploration; if this also fails, the action
parameterization or target difficulty is the likely bottleneck.

1. Maintain a Gaussian distribution over full pulse sequences.
    For d=5 and seq_len=10, each pulse has d parameters:
    4 phases + theta, for a 10 x 5 optimization variable.
2. Sample many candidate pulse sequences.
    Example: --cem-population 128 samples 128 full sequences.
3. Simulate every candidate. It composes:
U = D_10 @ ... @ D_2 @ D_1 @ I
and measures fidelity to the target.
4. Keep the best candidates.
    Example: --cem-elites 16 keeps the 16 highest-fidelity sequences.
5. Refit the Gaussian toward those elites.
    The mean moves toward good pulse sequences, and std shrinks around them.
6. Repeat.
    Example: --cem-iters 25 means 25 rounds of sample/score/refit.

PPO failing for noisy longer rewards, Cross-entropy should help.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from environment.hamiltonian import HamiltonianConfig
from metrics import RunLogger, plot_run


TargetSampler = Callable[[], np.ndarray]


@dataclass
class CEMConfig:
    """Hyperparameters for per-target CEM trajectory search."""

    d: int
    h_config: HamiltonianConfig = HamiltonianConfig.NEAREST_NEIGHBORS
    # Number of independent training targets to solve and log.
    n_targets: int = 200
    # Fixed pulse sequence length.  Defaults to 2*d to match random-pulses.
    seq_len: int | None = None
    # Candidate sequences sampled per CEM iteration and elites retained.
    population: int = 128
    elites: int = 16
    cem_iters: int = 25
    eval_interval: int = 5
    # Initial search width for pulse parameters; min_std prevents collapse.
    init_std: float = 1.0
    min_std: float = 0.03
    # Blend old and elite-fit distributions to keep updates stable.
    smoothing: float = 0.2
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_name: str = "cem"


def _wrap_phases(x: np.ndarray) -> np.ndarray:
    """Map arbitrary phase samples back to the physical [-pi, pi) range."""
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def _bounded_actions(raw: np.ndarray) -> np.ndarray:
    """Project Gaussian samples into valid pulse parameters."""
    actions = raw.copy()
    # Each action is [phi_0, ..., phi_{d-2}, theta].
    actions[..., :-1] = _wrap_phases(actions[..., :-1])
    actions[..., -1] = np.clip(actions[..., -1], 0.0, np.pi)
    return actions


def _make_jx_offdiag(d: int, device: str) -> torch.Tensor:
    """Return the nearest-neighbor J_x off-diagonal couplings."""
    k = torch.arange(d - 1, dtype=torch.float32, device=device)
    return 0.5 * torch.sqrt((k + 1) * (d - 1 - k))


def _batch_rollout_fidelities(
    actions: np.ndarray,
    target: np.ndarray,
    h_config: HamiltonianConfig,
    device: str,
) -> np.ndarray:
    """Evaluate a full CEM population in parallel.

    The input has shape (population, seq_len, d).  For d-level systems each
    pulse has d parameters: d-1 phases plus one rotation angle.  We construct
    the Hamiltonian and matrix exponential for every candidate pulse in a
    batch, compose the candidate unitaries, then return fidelities to the
    target.
    """
    if h_config != HamiltonianConfig.NEAREST_NEIGHBORS:
        raise NotImplementedError(
            "CEM currently supports nearest-neighbor H only."
        )

    actions_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
    target_t = torch.as_tensor(target, dtype=torch.cfloat, device=device)
    population, seq_len, action_dim = actions_t.shape
    d = action_dim
    off_diag = _make_jx_offdiag(d, device)
    U = torch.eye(d, dtype=torch.cfloat, device=device)
    U = U.expand(population, d, d)
    U = U.clone()

    for step in range(seq_len):
        phis = actions_t[:, step, :-1]
        theta = actions_t[:, step, -1]
        phases = torch.exp(-1j * phis.to(torch.cfloat))

        # Build H_rot(phi) for every candidate at this sequence position.
        H = torch.zeros((population, d, d), dtype=torch.cfloat, device=device)
        idx = torch.arange(d - 1, device=device)
        H[:, idx, idx + 1] = off_diag * phases
        H[:, idx + 1, idx] = off_diag * phases.conj()

        # D(phi, theta) = exp(-i * theta * H_rot(phi)).
        pulse = torch.matrix_exp(-1j * theta[:, None, None] * H)
        U = pulse @ U

    overlaps = torch.einsum("bij,ji->b", U.conj().transpose(-2, -1), target_t)
    fidelities = overlaps.abs().square() / (d * d)
    return fidelities.detach().cpu().numpy()


def optimize_target(
    target: np.ndarray,
    cfg: CEMConfig,
    rng: np.random.Generator,
) -> tuple[float, int]:
    """Return the best fidelity found for one target."""
    seq_len = cfg.seq_len if cfg.seq_len is not None else 2 * cfg.d
    action_dim = cfg.d

    # CEM maintains a diagonal Gaussian over the entire pulse sequence.
    mean = np.zeros((seq_len, action_dim), dtype=np.float32)
    mean[..., -1] = np.pi / 2
    std = np.full_like(mean, cfg.init_std)
    std[..., :-1] = np.pi

    best_fid = 0.0
    for _ in range(cfg.cem_iters):
        # Sample candidate trajectories, evaluate them, and keep the elites.
        raw = rng.normal(mean, std, size=(cfg.population, seq_len, action_dim))
        candidates = _bounded_actions(raw.astype(np.float32))
        fidelities = _batch_rollout_fidelities(
            candidates,
            target,
            cfg.h_config,
            cfg.device,
        )

        elite_idx = np.argsort(fidelities)[-cfg.elites:]
        elites = candidates[elite_idx]
        elite_mean = elites.mean(axis=0)
        elite_std = elites.std(axis=0)

        # Refit the search distribution toward high-performing sequences.
        mean = cfg.smoothing * mean + (1.0 - cfg.smoothing) * elite_mean
        std = cfg.smoothing * std + (1.0 - cfg.smoothing) * elite_std
        std = np.maximum(std, cfg.min_std)
        best_fid = max(best_fid, float(fidelities[elite_idx[-1]]))

    return best_fid, seq_len


def _evaluate_targets(
    targets: list[np.ndarray],
    cfg: CEMConfig,
    rng: np.random.Generator,
) -> tuple[float, float]:
    fids, lens = [], []
    for target in targets:
        fid, seq_len = optimize_target(target, cfg, rng)
        fids.append(fid)
        lens.append(seq_len)
    return float(np.mean(fids)), float(np.mean(lens))


def train(
    sampler: TargetSampler,
    cfg: CEMConfig,
    eval_targets: list[np.ndarray] | None = None,
) -> None:
    rng = np.random.default_rng(cfg.seed)
    seq_len = cfg.seq_len if cfg.seq_len is not None else 2 * cfg.d
    # Treat candidate rollouts like "timesteps" so existing plotting works.
    evals_per_target = cfg.population * cfg.cem_iters
    print(
        "CEM | "
        f"{cfg.n_targets} targets x {cfg.cem_iters} iters x "
        f"{cfg.population} candidates | seq_len={seq_len} | "
        f"{cfg.checkpoint_name}"
    )

    with RunLogger(cfg.checkpoint_name) as logger:
        for it in range(1, cfg.n_targets + 1):
            # Each row is an independent optimization, not policy state.
            fid, pulses = optimize_target(sampler(), cfg, rng)

            ef, ep = None, None
            if eval_targets and it % cfg.eval_interval == 0:
                ef, ep = _evaluate_targets(eval_targets, cfg, rng)
                print(f"    eval | mean_fid={ef:.4f}  pulses={ep:5.2f}")

            timestep = it * evals_per_target
            print(
                f"[target {it:4d}] evals={timestep:7d}  "
                f"train_fid={fid:.4f}  pulses={pulses:5.2f}"
            )
            logger.log(
                iter=it,
                timestep=timestep,
                episodes=1,
                train_fidelity=fid,
                train_pulses=pulses,
                eval_fidelity=ef,
                eval_pulses=ep,
            )

    plot_run(os.path.join("output", f"{cfg.checkpoint_name}.csv"))
