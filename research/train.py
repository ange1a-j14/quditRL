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
from environment.rewards import make_reward
from agents import ActorCritic, PulseSequencePlanner
from algos.amortized import AmortizedConfig, train as train_amortized
from algos.bc import BCConfig, train as train_bc
from algos.cem import CEMConfig, train as train_cem
from algos.ppo import PPOConfig, train as train_ppo
from wrappers import TargetSampling, ProgressReward
from samplers import make_sampler, qft, haar_unitary

ALGOS = ["ppo", "cem", "bc", "amortized"]
MODELS = ["actor-critic"]


def build_env(args):
    reward_fn = make_reward(args.reward, args.step_penalty)
    kwargs = dict(
        d=args.d,
        h_config=HamiltonianConfig.NEAREST_NEIGHBORS,
        reward_fn=reward_fn,
        max_pulses=args.max_pulses,
    )
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
    if args.target == "haar":
        return [qft(args.d)] + [haar_unitary(args.d, rng) for _ in range(3)]
    if args.target == "qft":
        return [qft(args.d)]
    sampler = make_sampler(args.target, args.d, seed=12345)
    return [sampler() for _ in range(4)]


def main():
    p = argparse.ArgumentParser(description="Train a policy on QuditEnv")
    p.add_argument("--d", type=int, default=4, help="qudit levels")
    p.add_argument("--target", choices=["haar", "random-pulses", "qft"], default="haar")
    p.add_argument("--algo", choices=ALGOS, default="ppo")
    p.add_argument("--model", choices=MODELS, default="actor-critic")
    p.add_argument("--hidden", type=int, default=256, help="hidden layer width")
    p.add_argument("--max-pulses", type=int, default=None, help="max pulses per episode")
    p.add_argument("--total-timesteps", type=int, default=5_000_000)
    p.add_argument(
        "--reward",
        choices=["l1", "fidelity", "log-infidelity"],
        default="l1",
        help="training reward signal (fidelity is always logged/eval'd)",
    )
    p.add_argument("--step-penalty", type=float, default=0.0, help="cost per pulse (0 = none)")
    p.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="discount factor; 1.0 lets progress reward value a future "
             "recovery as much as an early gain (no penalty for stepping "
             "away to come back)",
    )
    p.add_argument("--cem-targets", type=int, default=200)
    p.add_argument("--cem-iters", type=int, default=25)
    p.add_argument("--cem-population", type=int, default=128)
    p.add_argument("--cem-elites", type=int, default=16)
    p.add_argument("--cem-seq-len", type=int, default=None)
    p.add_argument("--bc-targets", type=int, default=200)
    p.add_argument("--bc-batch-size", type=int, default=256)
    p.add_argument("--bc-updates-per-target", type=int, default=20)
    p.add_argument("--bc-lr", type=float, default=3e-4)
    p.add_argument("--seq-len", type=int, default=None, help="amortized pulse sequence length (default 2*d)")
    p.add_argument("--batch-targets", type=int, default=256, help="amortized targets per gradient step")
    p.add_argument("--amortized-iters", type=int, default=20000, help="amortized gradient steps")
    p.add_argument("--lr", type=float, default=1e-3, help="amortized learning rate")
    p.add_argument(
        "--loss",
        choices=["infidelity", "log-infidelity", "l1"],
        default="infidelity",
        help="amortized training loss",
    )
    p.add_argument(
        "--pulse-penalty",
        type=float,
        default=0.0,
        help="amortized per-pulse cost; >0 makes the halting head exit early",
    )
    p.add_argument(
        "--min-terminate-pulses-start",
        type=int,
        default=10,
        help="initial minimum pulses before terminate is allowed",
    )
    p.add_argument(
        "--min-terminate-pulses-end",
        type=int,
        default=4,
        help="final minimum pulses before terminate is allowed",
    )
    p.add_argument(
        "--min-terminate-anneal-frac",
        type=float,
        default=0.5,
        help="fraction of training used to anneal the terminate pulse floor",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    eval_targets = build_eval_targets(args)
    if args.algo == "cem":
        seq = args.cem_seq_len if args.cem_seq_len is not None else 2 * args.d
        name = f"{args.algo}_d{args.d}_{args.target}_seq{seq}_seed{args.seed}"
    elif args.algo == "bc":
        seq = args.cem_seq_len if args.cem_seq_len is not None else 2 * args.d
        name = (
            f"{args.algo}_{args.model}_d{args.d}_{args.target}_"
            f"seq{seq}_h{args.hidden}_seed{args.seed}"
        )
    elif args.algo == "amortized":
        seq = args.seq_len if args.seq_len is not None else 2 * args.d
        name = (
            f"{args.algo}_d{args.d}_{args.target}_"
            f"seq{seq}_h{args.hidden}_seed{args.seed}_"
            f"pulsepen_{args.pulse_penalty:g}"
        )
    else:
        name = f"{args.algo}_{args.model}_d{args.d}_{args.target}_h{args.hidden}_seed{args.seed}"

    if args.algo == "amortized":
        seq_len = args.seq_len if args.seq_len is not None else 2 * args.d
        planner = PulseSequencePlanner(args.d, seq_len, hidden=args.hidden)
        cfg = AmortizedConfig(
            d=args.d,
            seq_len=seq_len,
            iters=args.amortized_iters,
            batch_targets=args.batch_targets,
            lr=args.lr,
            loss=args.loss,
            pulse_penalty=args.pulse_penalty,
            checkpoint_name=name,
            checkpoint_meta=dict(
                d=args.d,
                target=args.target,
                hidden=args.hidden,
                seq_len=seq_len,
                loss=args.loss,
                pulse_penalty=args.pulse_penalty,
                seed=args.seed,
            ),
        )
        train_amortized(
            make_sampler(args.target, args.d, args.seed),
            planner,
            cfg,
            eval_targets=eval_targets,
        )
        return

    if args.algo == "cem":
        cfg = CEMConfig(
            d=args.d,
            n_targets=args.cem_targets,
            seq_len=args.cem_seq_len,
            population=args.cem_population,
            elites=args.cem_elites,
            cem_iters=args.cem_iters,
            seed=args.seed,
            checkpoint_name=name,
        )
        train_cem(
            make_sampler(args.target, args.d, args.seed),
            cfg,
            eval_targets=eval_targets,
        )
        return

    if args.algo == "bc":
        env = build_env(args)
        policy = build_policy(args)
        teacher_cfg = CEMConfig(
            d=args.d,
            seq_len=args.cem_seq_len,
            population=args.cem_population,
            elites=args.cem_elites,
            cem_iters=args.cem_iters,
            seed=args.seed,
        )
        cfg = BCConfig(
            d=args.d,
            n_targets=args.bc_targets,
            batch_size=args.bc_batch_size,
            updates_per_target=args.bc_updates_per_target,
            lr=args.bc_lr,
            checkpoint_name=name,
            checkpoint_meta=dict(
                d=args.d,
                target=args.target,
                hidden=args.hidden,
                reward=args.reward,
                max_pulses=args.max_pulses,
                seed=args.seed,
                teacher="cem",
                cem_seq_len=args.cem_seq_len,
                cem_population=args.cem_population,
                cem_elites=args.cem_elites,
                cem_iters=args.cem_iters,
            ),
        )
        train_bc(
            make_sampler(args.target, args.d, args.seed),
            env,
            policy,
            cfg,
            teacher_cfg,
            eval_targets=eval_targets,
        )
        return

    env = build_env(args)
    policy = build_policy(args)
    cfg = PPOConfig(
        total_timesteps=args.total_timesteps,
        gamma=args.gamma,
        min_terminate_pulses_start=args.min_terminate_pulses_start,
        min_terminate_pulses_end=args.min_terminate_pulses_end,
        min_terminate_anneal_frac=args.min_terminate_anneal_frac,
        checkpoint_name=name,
        checkpoint_meta=dict(
            d=args.d,
            target=args.target,
            hidden=args.hidden,
            reward=args.reward,
            gamma=args.gamma,
            step_penalty=args.step_penalty,
            max_pulses=args.max_pulses,
            min_terminate_pulses_start=args.min_terminate_pulses_start,
            min_terminate_pulses_end=args.min_terminate_pulses_end,
            min_terminate_anneal_frac=args.min_terminate_anneal_frac,
            seed=args.seed,
        ),
    )

    if args.algo == "ppo":
        train_ppo(env, policy, cfg, eval_targets=eval_targets)
    else:
        raise ValueError(f"Unknown algo: {args.algo!r}")


if __name__ == "__main__":
    main()
