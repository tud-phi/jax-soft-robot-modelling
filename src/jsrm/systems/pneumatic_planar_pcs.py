__all__ = ["factory", "stiffness_fn"]
from jax import Array, vmap
import jax.numpy as jnp
from jsrm.math_utils import blk_diag
import numpy as onp
from typing import Callable, Dict, Optional, Tuple, Union

from .planar_pcs import factory as planar_pcs_factory

def factory(
    num_segments: int,
    *args,
    segment_actuation_selector: Optional[Array] = None,
    **kwargs
):
    """
    Factory function for the planar PCS.
    Args:
        num_segments: number of segments
        segment_actuation_selector: actuation selector for the segments as boolean array of shape (num_segments,)
            True entries signify that the segment is actuated, False entries signify that the segment is passive
    Returns:
    """
    if segment_actuation_selector is None:
        segment_actuation_selector = jnp.ones(num_segments, dtype=bool)

    # number of input pressures
    actuation_dim = segment_actuation_selector.sum() * 2

    # matrix that maps the (possibly) underactuated actuation space to a full actuation space
    actuation_basis = jnp.zeros((2 * num_segments, actuation_dim))
    actuation_basis_cumsum = jnp.cumsum(segment_actuation_selector)
    for i in range(num_segments):
        j = int(actuation_basis_cumsum[i].item()) - 1
        if segment_actuation_selector[i].item() is True:
            actuation_basis = actuation_basis.at[2 * i, j].set(1.0)
            actuation_basis = actuation_basis.at[2 * i + 1, j + 1].set(1.0)

    def actuation_mapping_fn(
        forward_kinematics_fn: Callable,
        jacobian_fn: Callable,
        params: Dict[str, Array],
        B_xi: Array,
        q: Array,
    ) -> Array:
        """
        Returns the actuation matrix that maps the actuation space to the configuration space.
        Args:
            forward_kinematics_fn: function to compute the forward kinematics
            jacobian_fn: function to compute the Jacobian
            params: dictionary with robot parameters
            B_xi: strain basis matrix
            q: configuration of the robot
        Returns:
            A: actuation matrix of shape (n_xi, n_act) where n_xi is the number of strains and
                n_act is the number of actuators
        """
        # map the configurations to strains
        xi = B_xi @ q

        # number of strains
        n_xi = xi.shape[0]

        # all segment bases and tips
        sms = jnp.concat([jnp.zeros((1,)), jnp.cumsum(params["l"])], axis=0)
        print("sms =\n", sms)

        # compute the poses of all segment tips
        chi_sms = vmap(forward_kinematics_fn, in_axes=(None, None, 0))(params, q, sms)

        # compute the Jacobian for all segment tips
        J_sms = vmap(jacobian_fn, in_axes=(None, None, 0))(params, q, sms)

        def compute_actuation_matrix_for_segment(
            r_cham_in: Array, r_cham_out: Array, varphi_cham: Array,
            chi_pe: Array, chi_de: Array,
            J_pe: Array, J_de: Array, xi: Array
        ) -> Array:
            """
            Compute the actuation matrix for a single segment.
            Args:
                r_cham_in: inner radius of each segment chamber
                r_cham_out: outer radius of each segment chamber
                varphi_cham: sector angle of each segment chamber
                chi_pe: pose of the proximal end (i.e., the base) of the segment as array of shape (3,)
                chi_de: pose of the distal end (i.e., the tip) of the segment as array of shape (3,)
                J_pe: Jacobian of the proximal end of the segment as array of shape (3, n_q)
                J_de: Jacobian of the distal end of the segment as array of shape (3, n_q)
                xi: strains of the segment
            Returns:
                A_sm: actuation matrix of shape (n_xi, 2)
            """
            # rotation matrix from the robot base to the segment base
            R_pe = jnp.array([[jnp.cos(chi_pe[2]), -jnp.sin(chi_pe[2])], [jnp.sin(chi_pe[2]), jnp.cos(chi_pe[2])]])
            # rotation matrix from the robot base to the segment tip
            R_de = jnp.array([[jnp.cos(chi_de[2]), -jnp.sin(chi_de[2])], [jnp.sin(chi_de[2]), jnp.cos(chi_de[2])]])


            # compute the actuation matrix for a single segment
            A_sm = jnp.zeros((n_xi, 2))
            return A_sm

        A_sms = vmap(compute_actuation_matrix_for_segment)(chi_sms, J_sms, xi)

        A = jnp.zeros((n_xi, 2 * num_segments))

        # apply the actuation_basis
        A = A @ actuation_basis

        return A

    return planar_pcs_factory(
        *args, stiffness_fn=stiffness_fn, actuation_mapping_fn=actuation_mapping_fn, **kwargs
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