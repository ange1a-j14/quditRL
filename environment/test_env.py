"""
Smoke test for QuditEnv.

Run from the project root with:
    python environment/test_env.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.linalg import dft

from environment import QuditEnv, HamiltonianConfig, unitary_distance


def qdft(d: int) -> np.ndarray:
    """Normalised d-dimensional Quantum Fourier Transform (det = 1)."""
    mat = dft(d, scale="sqrtn")
    det = np.linalg.det(mat)
    return mat / (det ** (1 / d))


def run_episode(env: QuditEnv, U_target: np.ndarray, max_random_steps: int = 6) -> None:
    obs, info = env.reset(U_target=U_target)
    print(f"  reset   | distance={info['distance']:.4f}  n_pulses={info['n_pulses']}")

    for step in range(max_random_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"  step {step+1:02d} | reward={reward:+.4f}  distance={info['distance']:.4f}"
            f"  terminated={terminated}  truncated={truncated}"
        )
        if terminated or truncated:
            break

    # Force a termination via the terminate signal
    if not (terminated or truncated):
        action = env.action_space.sample()
        action[-1] = 1.0
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"  term    | reward={reward:+.4f}  distance={info['distance']:.4f}"
            f"  terminated={terminated}"
        )


def test_observation_shapes(env: QuditEnv, U_target: np.ndarray) -> None:
    obs, _ = env.reset(U_target=U_target)
    d = env.d

    assert "U_current" in obs and "U_target" in obs, "Missing observation keys"
    assert obs["U_current"].shape == (2, d, d), f"Bad U_current shape: {obs['U_current'].shape}"
    assert obs["U_target"].shape == (2, d, d), f"Bad U_target shape: {obs['U_target'].shape}"
    assert obs["U_current"].dtype == np.float32
    assert obs["U_target"].dtype == np.float32

    # At reset U_current = I, so real part is identity, imaginary part is zero
    np.testing.assert_allclose(obs["U_current"][0], np.eye(d), atol=1e-6)
    np.testing.assert_allclose(obs["U_current"][1], np.zeros((d, d)), atol=1e-6)
    print("  observation shape/dtype checks passed")


def test_action_space(env: QuditEnv) -> None:
    assert env.action_space.shape == (env.d + 1,), "Bad action space shape"
    print("  action space shape check passed")


def test_truncation(env: QuditEnv, U_target: np.ndarray) -> None:
    env2 = QuditEnv(d=env.d, max_steps=3)
    env2.reset(U_target=U_target)
    terminated = False
    for _ in range(10):
        action = env2.action_space.sample()
        action[-1] = -1.0  # never terminate voluntarily
        _, _, terminated, _, _ = env2.step(action)
        if terminated:
            break
    assert terminated, "Environment should have terminated after max_steps"
    print("  max_steps termination check passed")


def test_wrong_shape(env: QuditEnv) -> None:
    try:
        env.reset(U_target=np.eye(env.d + 1, dtype=np.complex64))
        raise AssertionError("Should have raised ValueError for wrong shape")
    except ValueError:
        pass
    print("  wrong U_target shape raises ValueError — passed")


def main() -> None:
    for d in [3, 4, 5]:
        print(f"\n{'='*55}")
        print(f" d = {d}  |  QFT target  |  NEAREST_NEIGHBORS")
        print(f"{'='*55}")

        U_target = qdft(d)
        env = QuditEnv(d=d)

        print("\n[unit checks]")
        test_observation_shapes(env, U_target)
        test_action_space(env)
        test_truncation(env, U_target)
        test_wrong_shape(env)

        print("\n[random episode]")
        run_episode(env, U_target)

    print("\n[multiple targets in same env]")
    env = QuditEnv(d=4)
    for label, U in [("QFT d=4", qdft(4)), ("Identity d=4", np.eye(4, dtype=complex))]:
        print(f"  target: {label}")
        obs, info = env.reset(U_target=U)
        action = env.action_space.sample()
        action[-1] = 1.0
        _, reward, terminated, _, info = env.step(action)
        print(f"    immediate terminate: reward={reward:+.4f}  distance={info['distance']:.4f}")

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
