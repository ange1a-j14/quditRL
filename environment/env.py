"""
QuditEnv: a goal-conditioned gymnasium environment for qudit gate synthesis.

The agent controls a d-level qudit by sequentially choosing displacement
pulses D(\phi, \theta) = exp(-i \theta H_rot(\phi)) to compose a unitary U_current that
approximates a target U_target supplied at the start of each episode.

State / observation
-------------------
    s_t = (U_current, U_target)


Both complex (d\timesd) matrices are represented as real float arrays of shape
(2, d, d) — axis 0 indexes [real part, imaginary part] — and returned
together in a gymnasium Dict observation space.

Action:
    a_t ∈ R^{d+1}

    a_t[:d-1]  — phase vector \phi = (\phi_0, …, \phi_{d-2}), one per transition
    a_t[d-1]   — rotation angle \theta
    a_t[d]     — terminate signal: if > 0, end the episode without applying
                 a pulse and collect the terminal reward
                 this sets up early exit later in experiments

Transition:
    U_{t+1} = D(\phi, \theta) @ U_t

Reward:
    Every step (continue or terminal): reward_fn(U_current, U_target)
    Defaults to unitary_distance = -\Sigma_{i,j} |V_ij - U_ij|.

END event:
    terminated — agent chose terminate signal > 0
    truncated  — step count reached max_steps without termination (added for debugging)
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from .hamiltonian import (
    HamiltonianConfig,
    displacement_pulse,
    make_jx,
)
from .rewards import RewardFn, unitary_distance


def _unitary_to_obs(U: torch.Tensor) -> np.ndarray:
    """Convert a complex (d, d) torch tensor to a float32 (2, d, d) numpy array."""
    arr = U.detach().numpy()
    return np.stack([arr.real, arr.imag], axis=0).astype(np.float32)


class QuditEnv(gym.Env):
    """Goal-conditioned RL environment for qudit gate synthesis.

    Each episode targets a different unitary U_target, supplied via reset(U_target=...)`.  
    The agent applies sequential displacement
    pulses until it either chooses to terminate or exhausts max_steps.

    Parameters:
    d: Number of qudit levels (>= 2).
    h_config: Hamiltonian coupling
    reward_fn:
        See rewards for secham, defaults to unitary_distance.
    max_steps:
        Maximum number of pulses per episode before forced truncation.
        Defaults to 10 * d.

    Observation space:
    gymnasium.spaces.Dict with keys:

    - "U_current": Box(shape=(2, d, d), dtype=float32) — real and
      imaginary parts of the current composed unitary.
    - "U_target": Box(shape=(2, d, d), dtype=float32) — real and
      imaginary parts of the target unitary.

    Action space:
    Box(shape=(d+1,), dtype=float32, low=-inf, high=inf)
    - action[:d-1] — phases \phi (radians, unconstrained)
    - action[d-1]  — rotation angle \theta
    - action[d]    — terminate signal (episode ends without pulse if > 0)
        end action ends game
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        d: int,
        h_config: HamiltonianConfig = HamiltonianConfig.NEAREST_NEIGHBORS,
        reward_fn: RewardFn = unitary_distance,
        max_steps: Optional[int] = None,
    ) -> None:
        super().__init__()

        if d < 2:
            raise ValueError(f"d must be at least 2, got {d}.")

        self.d = d
        self.h_config = h_config
        self.reward_fn = reward_fn
        self.max_steps = max_steps if max_steps is not None else 10 * d

        # J_x spin operator: encodes the correct Rabi frequencies for each
        # transition via jmat((d-1)/2, 'x'), exactly as in the notebook.
        self._jx = make_jx(d)

        # Setup gym spaces using the gymnasium mod
        unitary_box = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2, d, d),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "U_current": unitary_box,
                "U_target": unitary_box,
            }
        )

        # action = [\phi_0, …, \phi_{d-2}, \theta, terminate_signal]
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(d + 1,), # d+1 for the phis (d-1), theta, and terminate signal
            dtype=np.float32,
        )

        # Internal state — initialised in reset().
        self._U_current: torch.Tensor = torch.eye(d, dtype=torch.cfloat)
        self._U_target: torch.Tensor = torch.eye(d, dtype=torch.cfloat)
        self._n_pulses: int = 0

    # ------------------------------------------------------------------
    # Core GYM API

    def reset(
        self,
        *,
        U_target: np.ndarray,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Begin a new episode targeting U_target.

        Parameters
        ----------
        U_target:
            Complex numpy array of shape (d, d) representing the target
            unitary for this episode.  Must be (approximately) unitary.
        seed:
            Optional RNG seed passed to the parent class.
        options:
            Unused; accepted for API compatibility.

        Returns
        -------
        obs:
            Initial observation with U_current = I_d.
        info:
            {"distance": float, "n_pulses": int}
        """
        super().reset(seed=seed)

        # sanity check
        U_target = np.asarray(U_target, dtype=np.complex64)
        if U_target.shape != (self.d, self.d):
            raise ValueError(
                f"U_target must have shape ({self.d}, {self.d}), "
                f"got {U_target.shape}."
            )

        # set the target and current unitaries
        self._U_target = torch.tensor(U_target, dtype=torch.cfloat)
        # initialized to identity matrix as U is transition matrix,
        # I is blank slate for the transition matrix
        self._U_current = torch.eye(self.d, dtype=torch.cfloat)
        self._n_pulses = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Apply one action (pulse or Terminate) and advance the environment.

        action:
            Float array of shape (d+1,).  See class docstring for layout.

        Returns: obs, reward, terminated, truncated, info
            Standard  from gymnasium step return tuple.
        """
        if self._U_target is None:
            raise RuntimeError("Call reset() before step().")

        # if terminate, end the round
        terminate_signal = float(action[self.d])

        if terminate_signal > 0.0 or self._n_pulses >= self.max_steps:
            # Agent chose to end the episode — collect terminal reward
            # or back stop reward
            reward = self.reward_fn(self._U_current, self._U_target, self._n_pulses)
            return self._get_obs(), reward, True, False, self._get_info()

        # Apply pulse(s)
        phis = torch.tensor(action[: self.d - 1], dtype=torch.float32)
        theta = torch.tensor(action[self.d - 1], dtype=torch.float32)

        pulse = displacement_pulse(self._jx, phis, theta, self.h_config)
        self._U_current = pulse @ self._U_current
        self._n_pulses += 1

        # DO not move above as it needs to be after the pulse is applied
        reward = self.reward_fn(self._U_current, self._U_target, self._n_pulses)
        return self._get_obs(), reward, False, False, self._get_info()

    def _get_obs(self) -> dict[str, np.ndarray]:
        return {
            "U_current": _unitary_to_obs(self._U_current),
            "U_target": _unitary_to_obs(self._U_target),
        }

    def _get_info(self) -> dict[str, Any]:
        distance = -unitary_distance(self._U_current, self._U_target)
        return {
            "distance": distance,
            "n_pulses": self._n_pulses,
        }
