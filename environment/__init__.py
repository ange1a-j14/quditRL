"""
QuditEnv:
    gymnasium-compatible environment.  Instantiate with hardware config;

HamiltonianConfig:
    Enum controlling the Hamiltonian coupling topology (just NEAREST_NEIGHBORS for now)

RewardFn:
    Type alias for a callable that takes two arguments:
        - U_current: a torch.Tensor representing the current unitary operator (d,d)
        - U_target: a torch.Tensor representing the target unitary operator (d,d)
    must returns a float reward value. Used for plugging in custom reward functions in QuditEnv.

unitary_distance:
    helper func for dist calculation on matrices.
    Negative L1 element-wise distance F(V, U) = -\Sigma|V_ij - U_ij|.

shaped_improvement:
    Per-step potential-based reward \Delta = F(U_after) - F(U_before).

make_jx:
    buildsJ_x spin operator for a d-level qudit.
     off-diagonal elements are the physically correct Rabi frequencies.
"""

from .env import QuditEnv
from .hamiltonian import HamiltonianConfig
from .rewards import RewardFn, unitary_distance

__all__ = [
    "QuditEnv",
    "HamiltonianConfig",
    "RewardFn",
    "unitary_distance",
]
