__all__ = ["factory", "stiffness_fn"]
from jax import Array, vmap
import jax.numpy as jnp
from jsrm.math_utils import blk_diag
import numpy as onp
from typing import Dict, Tuple, Union

from .planar_pcs import factory as planar_pcs_factory

def factory(
    *args, **kwargs
):
    return planar_pcs_factory(
        *args, stiffness_fn=stiffness_fn, **kwargs
    )

def _compute_stiffness_matrix_for_segment(
    l: Array, r: Array, r_cham_in: Array, r_cham_out: Array, varphi_cham: Array, E: Array
):
    # cross-sectional area [m²] of the material
    A_mat = jnp.pi * r ** 2 + 2 * r_cham_in ** 2 * varphi_cham - 2 * r_cham_out ** 2 * varphi_cham
    # second moment of area [m⁴] of the material
    Ib_mat = jnp.pi * r ** 4 / 4 + r_cham_in ** 4 * varphi_cham / 2 - r_cham_out ** 4 * varphi_cham / 2
    # poisson ratio of the material
    nu = 0.0
    # shear modulus
    G = E / (2 * (1 + nu))

    # bending stiffness [Nm²]
    Sbe = Ib_mat * E * l
    # shear stiffness [N]
    Ssh = 4 / 3 * A_mat * G * l
    # axial stiffness [N]
    Sax = A_mat * E * l

    S = jnp.diag(jnp.stack([Sbe, Ssh, Sax], axis=0))

    return S

def stiffness_fn(
    params: Dict[str, Array],
    B_xi: Array,
    formulate_in_strain_space: bool = False,
) -> Array:
    """
    Compute the stiffness matrix of the system.
    Args:
        params: Dictionary of robot parameters
        B_xi: Strain basis matrix
        formulate_in_strain_space: whether to formulate the elastic matrix in the strain space
    Returns:
        K: elastic matrix of shape (n_q, n_q) if formulate_in_strain_space is False or (n_xi, n_xi) otherwise
    """
    # stiffness matrix of shape (num_segments, 3, 3)
    S = vmap(
    _compute_stiffness_matrix_for_segment
    )(
        params["l"], params["r"], params["r_cham_in"], params["r_cham_out"], params["varphi_cham"], params["E"]
    )
    # we define the elastic matrix of shape (n_xi, n_xi) as K(xi) = K @ xi where K is equal to
    K = blk_diag(S)

    if not formulate_in_strain_space:
        K = B_xi.T @ K @ B_xi

    return K