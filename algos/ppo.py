"""PPO training algorithm for QuditEnv.

Exposes a single ``train(env, policy, cfg)`` entry point so the caller only
needs to build the env and policy, then hand them here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from wrappers import flatten_obs


@dataclass
class PPOConfig:
    rollout_steps: int = 2048
    update_epochs: int = 10
    minibatch: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    device: str = "cpu"
    total_timesteps: int = 1_000_000
    eval_interval: int = 5          # evaluate every N iterations
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "ppo"
    checkpoint_meta: dict[str, Any] = field(default_factory=dict)


def _compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[float],
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last = delta + gamma * gae_lambda * nonterminal * last
        adv[t] = last
    return adv, adv + np.asarray(values, dtype=np.float32)


def _collect_rollout(env, policy, state: dict, rollout_steps: int):
    obs = state["obs"]
    obss, acts, logps, rews, dones, vals = [], [], [], [], [], []
    ep_dist, ep_len = [], []
    for _ in range(rollout_steps):
        flat = flatten_obs(obs)
        action, logp, value = policy.act(flat)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        obss.append(flat); acts.append(action); logps.append(logp)
        rews.append(reward); dones.append(float(done)); vals.append(value)
        if done:
            ep_dist.append(info["distance"]); ep_len.append(info["n_pulses"])
            obs, _ = env.reset()
    state["obs"] = obs
    last_value = policy.act(flatten_obs(obs))[2]
    buf = dict(
        obs=np.asarray(obss, np.float32),
        act=np.asarray(acts, np.float32),
        logp=np.asarray(logps, np.float32),
        rew=np.asarray(rews, np.float32),
        done=np.asarray(dones, np.float32),
        val=np.asarray(vals, np.float32),
    )
    return buf, last_value, ep_dist, ep_len


def _ppo_update(policy, optimizer, buf: dict, adv: np.ndarray, ret: np.ndarray, cfg: PPOConfig):
    obs = torch.as_tensor(buf["obs"], device=cfg.device)
    actions = torch.as_tensor(buf["act"], device=cfg.device)
    old_logp = torch.as_tensor(buf["logp"], device=cfg.device)
    adv_t = torch.as_tensor(adv, device=cfg.device)
    ret_t = torch.as_tensor(ret, device=cfg.device)
    idx = np.arange(obs.shape[0])
    for _ in range(cfg.update_epochs):
        np.random.shuffle(idx)
        for s in range(0, len(idx), cfg.minibatch):
            mb = idx[s:s + cfg.minibatch]
            a = adv_t[mb]
            a = (a - a.mean()) / (a.std() + 1e-8)
            logp, entropy, value = policy.evaluate_actions(obs[mb], actions[mb])
            ratio = torch.exp(logp - old_logp[mb])
            pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a).mean()
            vf = 0.5 * ((value - ret_t[mb]) ** 2).mean()
            loss = pg + cfg.vf_coef * vf - cfg.ent_coef * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()


@torch.no_grad()
def evaluate(env, policy, targets: list) -> tuple[float, float]:
    """Greedy rollout on each target; returns (mean_distance, mean_pulses)."""
    dists, lens = [], []
    for U in targets:
        obs, info = env.reset(options={"U_target": U})
        done = False
        while not done:
            action, _, _ = policy.act(flatten_obs(obs), deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        dists.append(info["distance"]); lens.append(info["n_pulses"])
    return float(np.mean(dists)), float(np.mean(lens))


def train(env, policy, cfg: PPOConfig, eval_targets: list | None = None) -> None:
    """Run the PPO training loop.

    Parameters
    ----------
    env:
        A fully-wrapped Gymnasium environment (TargetSampling + ProgressReward).
    policy:
        An ActorCritic (or any agent with matching .act / .evaluate_actions API).
    cfg:
        PPOConfig with all hyperparameters.
    eval_targets:
        Optional list of target unitaries for periodic greedy evaluation.
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)
    obs, _ = env.reset()
    state = {"obs": obs}
    n_iters = cfg.total_timesteps // cfg.rollout_steps

    print(f"PPO | {n_iters} iters x {cfg.rollout_steps} steps | {cfg.checkpoint_name}")

    for it in range(1, n_iters + 1):
        buf, last_value, ep_dist, ep_len = _collect_rollout(env, policy, state, cfg.rollout_steps)
        adv, ret = _compute_gae(buf["rew"], buf["val"], buf["done"], last_value, cfg.gamma, cfg.gae_lambda)
        _ppo_update(policy, optimizer, buf, adv, ret, cfg)

        d_ = np.mean(ep_dist) if ep_dist else float("nan")
        l_ = np.mean(ep_len) if ep_len else float("nan")
        print(f"[iter {it:4d}] episodes={len(ep_dist):3d}  train_dist={d_:7.3f}  pulses={l_:5.2f}")

        if eval_targets and it % cfg.eval_interval == 0:
            ed, el = evaluate(env, policy, eval_targets)
            print(f"    eval | mean_dist={ed:7.3f}  mean_pulses={el:5.2f}")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"{cfg.checkpoint_name}.pt")
    torch.save({"state_dict": policy.state_dict(), **cfg.checkpoint_meta}, path)
    print(f"Saved policy to {path}")
