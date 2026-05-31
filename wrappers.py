"""Gymnasium wrappers and observation utilities for QuditEnv."""

from __future__ import annotations

import numpy as np
import gymnasium as gym


class TargetSampling(gym.Wrapper):
    """Draw a fresh U_target each reset via a zero-arg sampler callable.

    Pass ``options={"U_target": U}`` to ``reset()`` to override the sampler
    and fix a specific target (used during evaluation).
    """

    def __init__(self, env, sampler):
        super().__init__(env)
        self.sampler = sampler

    def reset(self, *, seed=None, options=None):
        U = (
            options["U_target"]
            if options and "U_target" in options
            else self.sampler()
        )
        return self.env.reset(U_target=U, seed=seed)


class ProgressReward(gym.Wrapper):
    """Convert the raw per-step reward into a dense improvement signal.

        r_t = Phi_t - Phi_{t-1}

    where Phi is the raw reward/potential from the env. This keeps the signal
    dense so the agent receives feedback on every pulse rather than only at
    termination.
    """

    def __init__(self, env):
        super().__init__(env)
        self._phi_prev = 0.0

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._phi_prev = info["potential"]
        return obs, info

    def step(self, action):
        obs, reward_raw, terminated, truncated, info = self.env.step(action)
        reward = reward_raw - self._phi_prev
        self._phi_prev = reward_raw
        return obs, reward, terminated, truncated, info


def flatten_obs(obs: dict) -> np.ndarray:
    """Flatten a QuditEnv Dict observation to a 1-D float32 array.

    Concatenates the ravel of U_current and U_target, each stored as
    (2, d, d) real/imag arrays, giving a vector of length 4*d*d.
    """
    return np.concatenate(
        [obs["U_current"].ravel(), obs["U_target"].ravel()]
    ).astype(np.float32)
