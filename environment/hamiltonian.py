"""
Hamiltonian construction for multi-tone qudit control.

Physics reference: Shi et al., Nat. Commun. 17, 1911 (2026).
  - Eq. 5: rotatig-frame Hamiltonian H_rot(\phi)
  - Eq. 6: Rabi frequencies \Omega_k = \Omega√(k(d-k)) that make H_rot(\phi=0) = \Omega·J_x

The on-resonance (\delta_k = 0) rotating-frame Hamiltonian for a d-level qudit
driven by d-1 tones, one per adjacent transition |k⟩↔|k+1⟩, is:

    H_rot(\phi) = J_x ⊙ (diag(exp(-i\phi), +1) + diag(exp(+i\phi), -1))

where J_x is the spin -(d-1)/2 angular momentum operator whose off-diagonal
elems encode the Rabi frequencies \Omega_k directly (no free Omega parameter)
J_x is obtained from QuTiP's jmat, exactly as in the unitary_torch notebook.

A single displacement pulse is then:
    D(\phi, \theta) = exp(-i \theta H_rot(\phi))
"""

from __future__ import annotations

import enum

import numpy as np
import torch
from scipy.linalg import expm


class HamiltonianConfig(enum.Enum):
    # Only adjacent transitions |k⟩↔|k+1⟩ are driven (tridiagonal H)
    # Uses the multi-tone spin-displacement scheme of Shi et al. and is
    # the physically realistic regime for most qudit platforms.
    NEAREST_NEIGHBORS = "nearest_neighbors"

    # Couples |k⟩↔|k+2⟩ transitions
    NEXT_NEAREST_NEIGHBORS = "next_nearest_neighbors"

    # Couples all transitions |k⟩↔|j⟩ (k ≠ j)
    FULLY_CONNECTED = "fully_connected"

    # build_hamiltonian() will raise NotImplementedError for unimplemented connectivities


def make_jx_NN(d: int) -> torch.Tensor:
    """Build the J_x spin operator for a d-level qudit analytically.

    Computes the spin-(d-1)/2 angular momentum operator J_x using the
    closed-form off-diagonal elements:

        J_x[k, k+1] = 0.5 * sqrt((k+1) * (d-1-k))   for k = 0, …, d-2

    Equivalent to qutip's jmat((d-1)/2, 'x') but with no external dependency (causing longer load times)

    Parameters
    ----------
    d: Number of qudit levels (>= 2).

    Returns
    -------
        Jx operator, shape (d, d), dtype cfloat.
    """
    k = np.arange(d - 1, dtype=np.float64)
    off_diag = 0.5 * np.sqrt((k + 1) * (d - 1 - k))
    jx = np.diag(off_diag, 1) + np.diag(off_diag, -1)
    return torch.tensor(jx, dtype=torch.cfloat)

def make_jx_NNN(d: int) -> torch.Tensor:
    """
    Builds uniform coupling operator for next nearest-neighbor transitions
        
    Parameters
    ----------
    d : Number of qudit levels (>= 3).

    Returns
    -------
        Symmetric operator of shape (d, d), dtype cfloat, with 0.5
        on diagonals ±2 and 0 elsewhere.
    """
    off_diag = np.ones(d - 2) * 0.5  # same prefactor as NN for scale consistency
    jx_nnn = np.diag(off_diag, 2) + np.diag(off_diag, -2)
    return torch.tensor(jx_nnn, dtype=torch.cfloat)

def make_jx_FC(d: int) -> torch.Tensor:
    """
    Builds uniform coupling operator for for all transitions |k⟩↔|j⟩ (k ≠ j)
        
    Parameters
    ----------
    d : Number of qudit levels (>= 2).

    Returns
    -------
        Symmetric operator of shape (d, d), dtype cfloat, with 0.5 on all
        off-diagonal elements and 0 on the diagonal.
    """
    jx_fc = 0.5 * (torch.ones(d, d, dtype=torch.cfloat) - torch.eye(d, dtype=torch.cfloat))
    return jx_fc


def build_hamiltonian(
    phis: torch.Tensor,
    h_config: HamiltonianConfig = HamiltonianConfig.NEAREST_NEIGHBORS,
) -> torch.Tensor:
    
    """Construct the rotating-frame Hamiltonian H_rot(\phi) for one pulse.

        pulse_ham = jx * (diag(exp(-i\phi), +1) + diag(exp(+i\phi), -1))

    Parameters
    ----------
    phis:
        Phase vector \phi of shape (d-1,), real-valued.  Each entry \phi_k is the
        drive phase on transition |k⟩↔|k+1⟩.
    h_config:
        Coupling topology.

    Returns
    -------
    torch.Tensor of shape (d, d), dtype=torch.cfloat
        The Hermitian Hamiltonian H_rot(\phi).
    """
    
    if h_config == HamiltonianConfig.NEAREST_NEIGHBORS:
        # Hamiltonian: H_rot(\phi) = J_x * (diag(exp(-i\phi), +1) + diag(exp(+i\phi), -1))
        # where \phi is the real (d-1,) phase vector for each transition.
        d = phis.shape[0] + 1
        jx = make_jx_NN(d)
        phase_vec = torch.exp(-1j * phis.to(dtype=torch.cfloat))
        # diag(+1): ofset upper diagonal with exp(-i\phi) on |k⟩↔|k+1⟩ transitions
        # diag(-1): ofset lower diagonal with exp(+i\phi) (the conjugate)
        # see H diagram in Shi paper
        phase_matrix = (
            torch.diag(phase_vec, diagonal=1)
            + torch.diag(phase_vec.conj(), diagonal=-1)
        )
        return jx * phase_matrix
    
    elif h_config == HamiltonianConfig.NEXT_NEAREST_NEIGHBORS:
        # Couples |k⟩↔|k+2⟩ transitions
        d = phis.shape[0] + 2
        jx = make_jx_NNN(d)
        phase_vec = torch.exp(-1j * phis.to(dtype=torch.cfloat))
        phase_matrix = (
            torch.diag(phase_vec, diagonal=2)
            + torch.diag(phase_vec.conj(), diagonal=-2)
        )
        return jx * phase_matrix
    
    elif h_config == HamiltonianConfig.FULLY_CONNECTED:
        # Couples all transitions |k⟩↔|j⟩ (k ≠ j)
        n = phis.shape[0]
        d = int((1 + np.sqrt(1 + 8 * n)) / 2)
        jx = make_jx_FC(d)
        phase_matrix = torch.zeros((d, d), dtype=torch.cfloat, device=phis.device)
        
        idx = 0
        for k in range(d):
            for j in range(k+1, d):
                # Fill upper triangle with phases
                phase_matrix[k, j] = torch.exp(-1j * phis[idx])
                phase_matrix[j, k] = torch.exp(1j * phis[idx])  # Hermitian
                idx += 1
        
        return jx * phase_matrix
    
    else:
        raise NotImplementedError(
            f"HamiltonianConfig.{h_config.name} is not yet implemented."
        )


def displacement_pulse(
    phis: torch.Tensor,
    theta: torch.Tensor,
    h_config: HamiltonianConfig = HamiltonianConfig.NEAREST_NEIGHBORS,
) -> torch.Tensor:
    """compiutes single displacement pulse unitary 
        D(\phi, \theta) = exp(-i \theta H_rot(\phi)).

    Parameters
    ----------
    phis:  Phase vector of shape (d-1,), real-valued.
    theta: Scalar rotation angle (pulse duration \times Rabi frequency).
    h_config: Coupling method.

    Returns
    -------
        Unitary matrix for this pulse.
    """
    H = build_hamiltonian(phis, h_config)
    H_np = (-1j * float(theta) * H).numpy()
    return torch.tensor(expm(H_np), dtype=torch.cfloat)
