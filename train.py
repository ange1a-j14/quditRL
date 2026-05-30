"""
PPO training loop for QuditEnv — a goal-conditioned policy for qudit gate synthesis.

Instead of optimizing pulse parameters separately for each target (the SGD
baseline in research/unitary_torch.ipynb), this trains one policy that emits a
pulse sequence conditioned on the target unitary in its observation, so it
generalizes across targets.

Reward (see milestone):
    r_t = (Phi_t - Phi_{t-1}) - step_penalty*[pulse beyond d]    Phi = -L1 distance
  - task term: per-step improvement in distance (milestone Sec. 3 metric); it
    sums to total progress from the identity, but is dense so PPO learns faster.
  - cost term: the first d pulses are free (the O(d) target), and each pulse
    beyond d is charged, implementing the milestone's Sec. 4 goal of "penalizing
    the agent for requiring more than d rounds" (shorter sequences -> less
    decoherence). Raise --step-penalty for stronger pressure toward fewer pulses.

Two environment-specific points:
  - QuditEnv.reset() needs a U_target; TargetSampling supplies a fresh one each
    episode so the policy is goal-conditioned.
  - Every pulse has det = 1 (H_rot is traceless), so U_current stays in SU(d);
    targets must be SU(d) too or the distance can never reach 0. Both samplers
    below return SU(d) matrices.

Usage:
    python train.py --d 3 --target random-pulses    # sanity: reachable targets
    python train.py --d 4 --target haar             # the real problem
    python train.py --d 4 --step-penalty 0.1        # push harder for fewer pulses
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym
from scipy.linalg import dft

from environment import QuditEnv, HamiltonianConfig
from environment.hamiltonian import displacement_pulse, make_jx

# PPO hyperparameters (fixed; expose via argparse only what you actually tune)
ROLLOUT_STEPS = 2048
UPDATE_EPOCHS = 10
MINIBATCH = 256
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP = 0.2
VF_COEF = 0.5
ENT_COEF = 0.0
LR = 3e-4
MAX_GRAD_NORM = 0.5
HIDDEN = 256
DEVICE = "cpu"
CHECKPOINT_DIR = "checkpoints"


# ----------------------------------------------------------------------------
# Target samplers (all return SU(d), i.e. reachable by the det = 1 pulses)
# ----------------------------------------------------------------------------

def qft(d: int) -> np.ndarray:
    """Normalised d-dimensional Quantum Fourier Transform (det = 1)."""
    mat = dft(d, scale="sqrtn")
    return (mat / np.linalg.det(mat) ** (1 / d)).astype(np.complex64)


def haar_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random unitary projected onto SU(d)."""
    z = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    q = q * (np.diagonal(r) / np.abs(np.diagonal(r)))   # Haar phase fix
    return (q / np.linalg.det(q) ** (1 / d)).astype(np.complex64)


def make_sampler(mode: str, d: int, seed: int):
    """Zero-arg callable returning a fresh training target."""
    rng = np.random.default_rng(seed)
    if mode == "haar":
        return lambda: haar_unitary(d, rng)
    if mode == "qft":
        return lambda: qft(d)
    if mode == "random-pulses":
        jx = make_jx(d)
        def sample():
            U = torch.eye(d, dtype=torch.cfloat)
            for _ in range(2 * d):  # reachable in <= 2d pulses by construction
                phis = torch.tensor(rng.uniform(-np.pi, np.pi, d - 1), dtype=torch.float32)
                theta = torch.tensor(rng.uniform(0, np.pi), dtype=torch.float32)
                U = displacement_pulse(jx, phis, theta) @ U
            return U.numpy().astype(np.complex64)
        return sample
    raise ValueError(f"Unknown target mode: {mode!r}")


# ----------------------------------------------------------------------------
# Wrappers: goal sampling + the reward described in the module docstring
# ----------------------------------------------------------------------------

class TargetSampling(gym.Wrapper):
    """Draw a fresh U_target each reset; pass options={'U_target': U} to fix it."""

    def __init__(self, env, sampler):
        super().__init__(env)
        self.sampler = sampler

    def reset(self, *, seed=None, options=None):
        U = options["U_target"] if options and "U_target" in options else self.sampler()
        return self.env.reset(U_target=U, seed=seed)


class ProgressReward(gym.Wrapper):
    """Per-step improvement reward with a penalty for exceeding the O(d) budget.

        r_t = (Phi_t - Phi_{t-1}) - step_penalty * [pulse beyond d applied]

    Phi = -L1 distance. The first d pulses are free (the O(d) target from the
    milestone); only pulses *beyond* d are charged, implementing Sec. 4's
    "penalizing the agent for requiring more than d rounds". A flat per-pulse
    charge instead collapses training to immediate termination, since quitting
    becomes an easier return gain than the small early improvements.
    """

    def __init__(self, env, step_penalty: float):
        super().__init__(env)
        self.step_penalty = step_penalty
        self.free_pulses = env.unwrapped.d
        self._phi_prev = 0.0

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._phi_prev = -info["distance"]
        return obs, info

    def step(self, action):
        obs, reward_raw, terminated, truncated, info = self.env.step(action)
        phi = reward_raw  # == -L1 distance to target
        reward = phi - self._phi_prev
        pulse_applied = float(action[self.env.unwrapped.d]) <= 0.0
        if pulse_applied and info["n_pulses"] > self.free_pulses:
            reward -= self.step_penalty
        self._phi_prev = phi
        return obs, reward, terminated, truncated, info


def flatten_obs(obs: dict) -> np.ndarray:
    return np.concatenate([obs["U_current"].ravel(), obs["U_target"].ravel()]).astype(np.float32)


# ----------------------------------------------------------------------------
# Actor-critic policy
# ----------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Shared-trunk Gaussian policy + value head over the flattened observation."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
        )
        self.mean_head = nn.Linear(HIDDEN, act_dim)
        self.value_head = nn.Linear(HIDDEN, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        with torch.no_grad():
            self.mean_head.bias[-1] = -1.0  # bias toward continuing early on

    def _dist_value(self, obs):
        h = self.trunk(obs)
        dist = Normal(self.mean_head(h), torch.exp(self.log_std))
        return dist, self.value_head(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs_np, deterministic=False):
        obs = torch.as_tensor(obs_np, device=DEVICE).unsqueeze(0)
        dist, value = self._dist_value(obs)
        action = dist.mean if deterministic else dist.sample()
        return action.squeeze(0).cpu().numpy(), dist.log_prob(action).sum(-1).item(), value.item()

    def evaluate_actions(self, obs, actions):
        dist, value = self._dist_value(obs)
        return dist.log_prob(actions).sum(-1), dist.entropy().sum(-1), value


# ----------------------------------------------------------------------------
# PPO core
# ----------------------------------------------------------------------------

def compute_gae(rewards, values, dones, last_value):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + GAMMA * next_value * nonterminal - values[t]
        last = delta + GAMMA * GAE_LAMBDA * nonterminal * last
        adv[t] = last
    return adv, adv + np.asarray(values, dtype=np.float32)


def collect_rollout(env, policy, state):
    obs = state["obs"]
    obss, acts, logps, rews, dones, vals = [], [], [], [], [], []
    ep_dist, ep_len = [], []
    for _ in range(ROLLOUT_STEPS):
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
        obs=np.asarray(obss, np.float32), act=np.asarray(acts, np.float32),
        logp=np.asarray(logps, np.float32), rew=np.asarray(rews, np.float32),
        done=np.asarray(dones, np.float32), val=np.asarray(vals, np.float32),
    )
    return buf, last_value, ep_dist, ep_len


def ppo_update(policy, optimizer, buf, adv, ret):
    obs = torch.as_tensor(buf["obs"], device=DEVICE)
    actions = torch.as_tensor(buf["act"], device=DEVICE)
    old_logp = torch.as_tensor(buf["logp"], device=DEVICE)
    adv = torch.as_tensor(adv, device=DEVICE)
    ret = torch.as_tensor(ret, device=DEVICE)
    idx = np.arange(obs.shape[0])
    for _ in range(UPDATE_EPOCHS):
        np.random.shuffle(idx)
        for s in range(0, len(idx), MINIBATCH):
            mb = idx[s:s + MINIBATCH]
            a = adv[mb]
            a = (a - a.mean()) / (a.std() + 1e-8)
            logp, entropy, value = policy.evaluate_actions(obs[mb], actions[mb])
            ratio = torch.exp(logp - old_logp[mb])
            pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a).mean()
            vf = 0.5 * ((value - ret[mb]) ** 2).mean()
            loss = pg + VF_COEF * vf - ENT_COEF * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()


@torch.no_grad()
def evaluate(env, policy, targets):
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


# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="PPO training for QuditEnv")
    p.add_argument("--d", type=int, default=4, help="qudit levels")
    p.add_argument("--target", choices=["haar", "random-pulses", "qft"], default="haar")
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--step-penalty", type=float, default=0.05, help="cost per pulse")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base = QuditEnv(d=args.d, h_config=HamiltonianConfig.NEAREST_NEIGHBORS)
    env = ProgressReward(TargetSampling(base, make_sampler(args.target, args.d, args.seed)),
                         step_penalty=args.step_penalty)

    policy = ActorCritic(4 * args.d * args.d, args.d + 1).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR, eps=1e-5)

    eval_rng = np.random.default_rng(12345)
    eval_targets = [qft(args.d)] + [haar_unitary(args.d, eval_rng) for _ in range(3)]

    obs, _ = env.reset()
    state = {"obs": obs}
    n_iters = args.total_timesteps // ROLLOUT_STEPS
    print(f"PPO | d={args.d} target={args.target} step_penalty={args.step_penalty} | "
          f"{n_iters} iters x {ROLLOUT_STEPS} steps")

    for it in range(1, n_iters + 1):
        buf, last_value, ep_dist, ep_len = collect_rollout(env, policy, state)
        adv, ret = compute_gae(buf["rew"], buf["val"], buf["done"], last_value)
        ppo_update(policy, optimizer, buf, adv, ret)
        d_ = np.mean(ep_dist) if ep_dist else float("nan")
        l_ = np.mean(ep_len) if ep_len else float("nan")
        print(f"[iter {it:4d}] episodes={len(ep_dist):3d}  train_dist={d_:7.3f}  pulses={l_:5.2f}")
        if it % 5 == 0:
            ed, el = evaluate(env, policy, eval_targets)
            print(f"    eval | mean_dist={ed:7.3f}  mean_pulses={el:5.2f}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"ppo_d{args.d}_{args.target}_seed{args.seed}.pt")
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "d": args.d,
            "target": args.target,
            "step_penalty": args.step_penalty,
            "seed": args.seed,
        },
        path,
    )
    print(f"Saved policy to {path}")


if __name__ == "__main__":
    main()
