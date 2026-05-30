"""
Evaluate a trained QuditEnv policy checkpoint.

Loads a checkpoint written by train.py, rebuilds the matching policy, and runs
greedy (deterministic) episodes on a set of held-out target unitaries, reporting
the final L1 distance and number of pulses for each.

Usage:
    python eval.py checkpoints/ppo_d3_random-pulses_seed0.pt
    python eval.py checkpoints/ppo_d4_haar_seed0.pt --n-haar 5
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from train import (
    ActorCritic, QuditEnv, TargetSampling, HamiltonianConfig,
    flatten_obs, qft, haar_unitary,
)


@torch.no_grad()
def run_episode(env, policy, U):
    """Greedy rollout on one target; returns (final_distance, n_pulses)."""
    obs, info = env.reset(options={"U_target": U})
    done = False
    while not done:
        action, _, _ = policy.act(flatten_obs(obs), deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info["distance"], info["n_pulses"]


def main():
    p = argparse.ArgumentParser(description="Evaluate a QuditEnv policy checkpoint")
    p.add_argument("checkpoint", help="path to a .pt file saved by train.py")
    p.add_argument("--n-haar", type=int, default=3, help="number of random SU(d) eval targets")
    p.add_argument("--seed", type=int, default=12345, help="seed for the eval target set")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, weights_only=True)
    d = ckpt["d"]
    policy = ActorCritic(4 * d * d, d + 1)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    print(f"Loaded {args.checkpoint}  (d={d}, target={ckpt.get('target')}, seed={ckpt.get('seed')})")

    # Eval only reads info["distance"] / info["n_pulses"], so no reward shaping
    # is needed; targets are injected via options={"U_target": U}.
    env = TargetSampling(
        QuditEnv(d=d, h_config=HamiltonianConfig.NEAREST_NEIGHBORS),
        sampler=lambda: qft(d),  # unused; reset() is always given an explicit target
    )

    rng = np.random.default_rng(args.seed)
    targets = [("qft", qft(d))] + [(f"haar{i}", haar_unitary(d, rng)) for i in range(args.n_haar)]

    print(f"{'target':>8}  {'final_dist':>11}  {'pulses':>6}")
    dists, lens = [], []
    for name, U in targets:
        dist, n = run_episode(env, policy, U)
        dists.append(dist)
        lens.append(n)
        print(f"{name:>8}  {dist:11.3f}  {n:6.0f}")
    print(f"{'mean':>8}  {np.mean(dists):11.3f}  {np.mean(lens):6.2f}")


if __name__ == "__main__":
    main()
