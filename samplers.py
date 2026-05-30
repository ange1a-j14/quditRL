"""Target unitary samplers for QuditEnv training.

All samplers return SU(d) matrices (det = 1) so that the target is always
reachable by the displacement pulses, which also have det = 1.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import dft

from environment.hamiltonian import displacement_pulse, make_jx


def qft(d: int) -> np.ndarray:
    """Normalised d-dimensional Quantum Fourier Transform projected onto SU(d)."""
    mat = dft(d, scale="sqrtn")
    return (mat / np.linalg.det(mat) ** (1 / d)).astype(np.complex64)


def haar_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random unitary projected onto SU(d)."""
    z = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    q = q * (np.diagonal(r) / np.abs(np.diagonal(r)))  # Haar phase fix
    return (q / np.linalg.det(q) ** (1 / d)).astype(np.complex64)


def make_sampler(mode: str, d: int, seed: int):
    """Return a zero-arg callable that draws a fresh SU(d) training target.

    Parameters
    ----------
    mode:
        ``"haar"``          — Haar-random SU(d) (general case)
        ``"qft"``           — always the QFT (fixed target)
        ``"random-pulses"`` — composed from ≤2d random pulses (guaranteed reachable)
    d:
        Qudit dimension.
    seed:
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    if mode == "haar":
        return lambda: haar_unitary(d, rng)
    if mode == "qft":
        return lambda: qft(d)
    if mode == "random-pulses":
        jx = make_jx(d)
        def _sample():
            U = torch.eye(d, dtype=torch.cfloat)
            for _ in range(2 * d):
                phis = torch.tensor(rng.uniform(-np.pi, np.pi, d - 1), dtype=torch.float32)
                theta = torch.tensor(rng.uniform(0, np.pi), dtype=torch.float32)
                U = displacement_pulse(jx, phis, theta) @ U
            return U.numpy().astype(np.complex64)
        return _sample
    raise ValueError(f"Unknown target mode: {mode!r}")
