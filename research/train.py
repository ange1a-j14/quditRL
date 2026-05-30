"""Training entrypoint for QuditEnv.

Wires together an environment, a policy, and a training algorithm.
To add a new model or algorithm, implement it in agents/ or algos/ and
add a case in the --algo / --model dispatch below.

Usage:
    python train.py --d 3 --target random-pulses
    python train.py --d 4 --target haar --hidden 512
    python train.py --d 4 --target haar --step-penalty 0.05
"""

from __future__ import annotations

import argparse
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment import QuditEnv, HamiltonianConfig
from environment.rewards import penalized_distance
from agents import ActorCritic
from algos.ppo import PPOConfig, train as train_ppo
from wrappers import TargetSampling, ProgressReward
from samplers import make_sampler, qft, haar_unitary

ALGOS = ["ppo"]
MODELS = ["actor-critic"]


def build_env(args):
    reward_fn = penalized_distance(args.step_penalty) if args.step_penalty > 0.0 else None
    kwargs = dict(d=args.d, h_config=HamiltonianConfig.NEAREST_NEIGHBORS)
    if reward_fn is not None:
        kwargs["reward_fn"] = reward_fn
    base = QuditEnv(**kwargs)
    return ProgressReward(TargetSampling(base, make_sampler(args.target, args.d, args.seed)))


def build_policy(args):
    obs_dim = 4 * args.d * args.d
    act_dim = args.d + 1
    if args.model == "actor-critic":
        return ActorCritic(obs_dim, act_dim, hidden=args.hidden)
    raise ValueError(f"Unknown model: {args.model!r}")


def build_eval_targets(args):
    rng = np.random.default_rng(12345)
    return [qft(args.d)] + [haar_unitary(args.d, rng) for _ in range(3)]


def main():
    p = argparse.ArgumentParser(description="Train a policy on QuditEnv")
    p.add_argument("--d", type=int, default=4, help="qudit levels")
    p.add_argument("--target", choices=["haar", "random-pulses", "qft"], default="haar")
    p.add_argument("--algo", choices=ALGOS, default="ppo")
    p.add_argument("--model", choices=MODELS, default="actor-critic")
    p.add_argument("--hidden", type=int, default=256, help="hidden layer width")
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--step-penalty", type=float, default=0.0, help="cost per pulse (0 = none)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = build_env(args)
    policy = build_policy(args)
    eval_targets = build_eval_targets(args)

    name = f"{args.algo}_{args.model}_d{args.d}_{args.target}_h{args.hidden}_seed{args.seed}"
    cfg = PPOConfig(
        total_timesteps=args.total_timesteps,
        checkpoint_name=name,
        checkpoint_meta=dict(
            d=args.d,
            target=args.target,
            hidden=args.hidden,
            step_penalty=args.step_penalty,
            seed=args.seed,
        ),
    )

    if args.algo == "ppo":
        train_ppo(env, policy, cfg, eval_targets=eval_targets)
    else:
        raise ValueError(f"Unknown algo: {args.algo!r}")


if __name__ == "__main__":
    main()
