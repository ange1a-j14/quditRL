"""Behavior cloning from CEM-generated pulse trajectories.

This trains the actor as an amortized solver: CEM solves each sampled target
from scratch, then the policy learns to imitate the optimized pulse sequence
conditioned on (U_current, U_target).  The result can be used directly or as a
warm start for PPO/SAC-style fine tuning.

Trial code: did not deliver much improved performance on medium sized run.
    
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from algos.cem import CEMConfig, optimize_target
from environment.hamiltonian import HamiltonianConfig, displacement_pulse
from metrics import RunLogger, plot_run
from wrappers import flatten_obs


TargetSampler = Callable[[], np.ndarray]


@dataclass
class BCConfig:
    d: int
    n_targets: int = 200
    batch_size: int = 256
    updates_per_target: int = 20
    lr: float = 3e-4
    terminate_value: float = 1.0
    continue_value: float = -1.0
    eval_interval: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "bc"
    checkpoint_meta: dict[str, Any] = field(default_factory=dict)


def _unitary_to_obs(U: torch.Tensor) -> np.ndarray:
    arr = U.detach().cpu().numpy()
    return np.stack([arr.real, arr.imag], axis=0).astype(np.float32)


def _trajectory_examples(
    target: np.ndarray,
    pulse_actions: np.ndarray,
    cfg: BCConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a teacher pulse sequence into supervised policy examples."""
    U_target = torch.tensor(target, dtype=torch.cfloat)
    U_current = torch.eye(cfg.d, dtype=torch.cfloat)
    obss, actions = [], []

    for pulse_action in pulse_actions:
        obs = {
            "U_current": _unitary_to_obs(U_current),
            "U_target": _unitary_to_obs(U_target),
        }
        action = np.empty(cfg.d + 1, dtype=np.float32)
        action[: cfg.d] = pulse_action
        action[cfg.d] = cfg.continue_value
        obss.append(flatten_obs(obs))
        actions.append(action)

        phis = torch.tensor(pulse_action[:-1], dtype=torch.float32)
        theta = torch.tensor(pulse_action[-1], dtype=torch.float32)
        U_current = displacement_pulse(
            phis,
            theta,
            HamiltonianConfig.NEAREST_NEIGHBORS,
        ) @ U_current

    # Teach the actor to end after imitating the full teacher sequence.
    obs = {
        "U_current": _unitary_to_obs(U_current),
        "U_target": _unitary_to_obs(U_target),
    }
    action = np.zeros(cfg.d + 1, dtype=np.float32)
    action[cfg.d] = cfg.terminate_value
    obss.append(flatten_obs(obs))
    actions.append(action)

    return np.asarray(obss, np.float32), np.asarray(actions, np.float32)


@torch.no_grad()
def evaluate(env, policy, targets: list[np.ndarray]) -> tuple[float, float]:
    fids, lens = [], []
    for U in targets:
        obs, _ = env.reset(options={"U_target": U})
        done = False
        while not done:
            action, _, _ = policy.act(flatten_obs(obs), deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        fids.append(info["fidelity"])
        lens.append(info["n_pulses"])
    return float(np.mean(fids)), float(np.mean(lens))


def _update_policy(
    policy,
    optimizer: torch.optim.Optimizer,
    obs_buf: list[np.ndarray],
    action_buf: list[np.ndarray],
    cfg: BCConfig,
    rng: np.random.Generator,
) -> float:
    losses = []
    obs_arr = np.asarray(obs_buf, dtype=np.float32)
    action_arr = np.asarray(action_buf, dtype=np.float32)
    n = len(obs_arr)

    for _ in range(cfg.updates_per_target):
        idx = rng.integers(0, n, size=min(cfg.batch_size, n))
        obs = torch.as_tensor(obs_arr[idx], device=cfg.device)
        target_actions = torch.as_tensor(action_arr[idx], device=cfg.device)
        pred_actions = policy.mean_action(obs)
        loss = F.mse_loss(pred_actions, target_actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses))


def train(
    sampler: TargetSampler,
    env,
    policy,
    cfg: BCConfig,
    teacher_cfg: CEMConfig,
    eval_targets: list[np.ndarray] | None = None,
) -> None:
    rng = np.random.default_rng(teacher_cfg.seed)
    policy.to(cfg.device)
    policy.device = cfg.device
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    obs_buf: list[np.ndarray] = []
    action_buf: list[np.ndarray] = []
    evals_per_target = teacher_cfg.population * teacher_cfg.cem_iters

    print(
        "BC | "
        f"{cfg.n_targets} teacher targets | "
        f"CEM {teacher_cfg.cem_iters}x{teacher_cfg.population} | "
        f"{cfg.checkpoint_name}"
    )

    with RunLogger(cfg.checkpoint_name) as logger:
        for it in range(1, cfg.n_targets + 1):
            target = sampler()
            teacher_fid, pulses, teacher_actions = optimize_target(
                target,
                teacher_cfg,
                rng,
            )
            obss, actions = _trajectory_examples(target, teacher_actions, cfg)
            obs_buf.extend(obss)
            action_buf.extend(actions)
            loss = _update_policy(
                policy,
                optimizer,
                obs_buf,
                action_buf,
                cfg,
                rng,
            )

            train_fid, train_pulses = evaluate(env, policy, [target])

            ef, ep = None, None
            if eval_targets and it % cfg.eval_interval == 0:
                ef, ep = evaluate(env, policy, eval_targets)
                print(f"    eval | mean_fid={ef:.4f}  pulses={ep:5.2f}")

            print(
                f"[target {it:4d}] teacher_fid={teacher_fid:.4f}  "
                f"teacher_pulses={pulses:5.2f}  "
                f"policy_fid={train_fid:.4f}  bc_loss={loss:.5f}"
            )
            logger.log(
                iter=it,
                timestep=it * evals_per_target,
                episodes=1,
                train_fidelity=train_fid,
                train_pulses=train_pulses,
                eval_fidelity=ef,
                eval_pulses=ep,
            )

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"{cfg.checkpoint_name}.pt")
    checkpoint = {"state_dict": policy.state_dict(), **cfg.checkpoint_meta}
    torch.save(checkpoint, path)
    print(f"Saved behavior-cloned policy to {path}")
    plot_run(os.path.join("output", f"{cfg.checkpoint_name}.csv"))
