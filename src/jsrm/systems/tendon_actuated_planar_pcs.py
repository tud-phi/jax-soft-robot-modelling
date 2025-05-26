__all__ = ["factory", "stiffness_fn"]
from jax import Array, lax, vmap
import jax.numpy as jnp
from jsrm.math_utils import blk_diag
import numpy as onp
from typing import Callable, Dict, Optional, Tuple, Union

from .planar_pcs import factory as planar_pcs_factory, stifffness_fn

def factory(
    num_segments: int,
    *args,
    segment_actuation_selector: Optional[Array] = None,
    **kwargs
):
    """
    Factory function for the tendon-driven planar PCS.
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
        xi_eq: Array,
        q: Array,
    ) -> Array:
        """
        Returns the actuation matrix that maps the actuation space to the configuration space.
        Args:
            forward_kinematics_fn: function to compute the forward kinematics
            jacobian_fn: function to compute the Jacobian
            params: dictionary with robot parameters
            B_xi: strain basis matrix
            xi_eq: equilibrium strains as array of shape (n_xi,)
            q: configuration of the robot
        Returns:
            A: actuation matrix of shape (n_xi, n_act) where n_xi is the number of strains and
                n_act is the number of actuators
        """
        # extract the parameters
        l = params["l"]
        # map the configuration to the strains
        xi = xi_eq + B_xi @ q

        def compute_actuation_matrix_for_segment(
            segment_idx, d_sm: Array,
        ) -> Array:
            """
            Compute the actuation matrix for a single segment.
            We assume that each segment is actuated by num_segment_tendons that are routed at a distance of d from the segment's backbone, 
            respectively, and attached to the segment's distal end. We assume that the motor is located at the base of the robot and that the 
            tendons are routed through all proximal segments.
            The control inputs u1 and u2 are the tensions (i.e., forces) applied by the two tendons.

            Args:
                segment_idx: index of the segment
                d_sm: distance of the tendons from the segment's backbone (shape: (num_segment_tendons,))
            Returns:
                A_sm: actuation matrix of shape (n_xi, num_segment_tendons)
            """
            num_segment_tendons = d_sm.shape[0]
        
            A_sm = []
            for j in range(num_segment_tendons):
                d = d_sm[j]
                
                """
                A_d = []
                for i in range(0, segment_idx + 1):
                    # length of the i-th segment
                    l_i = l[i]
                    # strain of the i-th segment
                    xi_i = xi[3 * i:3 * (i + 1)]
                    A_d.append(jnp.array([
                        d * l_i * jnp.sqrt(xi_i[1]**2 + xi_i[2]**2),  # the derivative of the tendon length with respect to the bending strain of the i-th segment
                        l_i * xi_i[1] * (1 + d * xi_i[0]) / jnp.sqrt(xi_i[1]**2 + xi_i[2]**2),  # the derivative of the tendon length with respect to the shear strain of the i-th segment
                        l_i * xi_i[2] * (1 + d * xi_i[0]) / jnp.sqrt(xi_i[1]**2 + xi_i[2]**2),  # the derivative of the tendon length with respect to the axial strain of the i-th segment
                    ]))
                """
                def compute_A_d(l_i: Array, xi_i: Array) -> Array:
                    print("l_i.shape", l_i.shape, "xi_i.shape", xi_i.shape)
                    sigma_norm = jnp.sqrt(xi_i[1] ** 2 + xi_i[2] ** 2)
                    return jnp.array([
                        d * l_i * sigma_norm,
                        l_i * xi_i[1] * (1 + d * xi_i[0]) / sigma_norm,
                        l_i * xi_i[2] * (1 + d * xi_i[0]) / sigma_norm,
                    ])
                A_d = vmap(compute_A_d)(l[:j+1], xi[: 3 * (j + 1)].reshape(-1, 3))
                
                # concatenate the derivatives for all segments
                A_d = jnp.concatenate(A_d, axis=0)
                A_sm.append(A_d)

            # stack the actuation matrices for all tendons
            A_sm = jnp.stack(A_sm, axis=1)
            print("A_sm.shape", A_sm.shape)

            return A_sm

        segment_indices = jnp.arange(num_segments)
        A_sms = vmap(compute_actuation_matrix_for_segment)(
            segment_indices, params["d"],
        )
        print("A_sms.shape", A_sms.shape)
        # concatenate the actuation matrices for all tendons
        A = jnp.concatenate(A_sms, axis=1)
        print("A.shape", A.shape)

        # apply the actuation_basis
        A = A @ actuation_basis

        return A

    return planar_pcs_factory(
        *args, actuation_mapping_fn=actuation_mapping_fn, **kwargs
    )
