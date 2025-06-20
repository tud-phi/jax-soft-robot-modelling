import jax
from jax import Array, jit, lax, vmap
from jax import numpy as jnp
from pathlib import Path
from typing import Dict, Tuple
import jax.scipy.integrate as jscipy
from quadax import GaussKronrodRule


from jsrm.systems.utils import (
    compute_strain_basis,
    compute_planar_stiffness_matrix,
    gauss_quadrature
)
from jsrm.math_utils import blk_diag, blk_concat
from jsrm.utils.lie_operators import (
    vec_SE2_to_xi_SE3,
    Tangent_gn_SE3, 
    Adjoint_gn_SE3, 
    Adjoint_g_SE3,
    compute_weighted_sums, 
    adjoint_SE3
)

import jsrm
from jsrm.systems import planar_pcs

global_eps = 1e-6  # Small number to avoid singularities in the bending strain

def apply_eps_to_bend_strains(xi: Array, _eps: float) -> Array:
    """
    Add a small number to the bending strain to avoid singularities
    """
    xi_reshaped = xi.reshape((-1, 3))

    xi_bend_sign = jnp.sign(xi_reshaped[:, 0])
    
    # set zero sign to 1 (i.e. positive)
    xi_bend_sign = jnp.where(xi_bend_sign == 0, 1, xi_bend_sign)
    
    # add eps to the bending strain (i.e. the first column)
    sigma_b_epsed = lax.select(
        jnp.abs(xi_reshaped[:, 0]) < _eps,
        xi_bend_sign * _eps,
        xi_reshaped[:, 0],
    )
    xi_epsed = jnp.stack(
        [
            sigma_b_epsed,
            xi_reshaped[:, 1],
            xi_reshaped[:, 2],
        ],
        axis=1,
    )

    # Flatten the array
    xi_epsed = xi_epsed.flatten()

    return xi_epsed

def classify_segment(
    params: Dict[str, Array], 
    s: Array
    ) -> Tuple[Array, Array]:
    """
    Classify the point along the robot to the corresponding segment.

    Args:
        params (Dict[str, Array]): dictionary of robot parameters
        s (Array): point coordinate along the robot in the interval [0, L].

    Returns:
        Tuple[Array, Array]: 
            - segment_idx (Array): index of the segment where the point is located
            - s_segment (Array): point coordinate along the segment in the interval [0, l_segment]
    """
    l = params["l"]
    
    # Compute the cumulative length of the segments starting with 0
    l_cum = jnp.cumsum(jnp.concatenate([jnp.array([0.0]), l]))
    
    # Classify the point along the robot to the corresponding segment
    segment_idx = jnp.clip(jnp.sum(s > l_cum) - 1, 0, len(l) - 1)
    
    # Compute the point coordinate along the segment in the interval [0, l_segment]
    s_segment = s - l_cum[segment_idx]

    return segment_idx, s_segment.squeeze(), l_cum

def chi_fn_xi(
    params: Dict[str, Array], 
    xi: Array, 
    s: Array
) -> Array:
    """
    Compute the pose of the robot at a given point along its length with respect to the strain vector.

    Args:
        params (Dict[str, Array]): dictionary of robot parameters.
        xi (Array): strain vector of the robot.
        s (Array): point coordinate along the robot in the interval [0, L].

    Returns:
        Array: pose of the robot at the point s in the interval [0, L].
    """
    th0 = jnp.array(params["th0"])  # initial angle of the robot
    l = params["l"] # length of each segment [m]
    
    # Classify the point along the robot to the corresponding segment
    segment_idx, s_local, l_cum = classify_segment(params, s)
        
    chi_O = jnp.array([0.0, 0.0, th0])  # Initial pose of the robot

    # Iteration function
    def chi_i(
        i: int, 
        chi_prev: Array
    ) -> Array:
        th_prev = chi_prev[2]  # Extract previous orientation angle from val
        p_prev = chi_prev[:2]  # Extract previous position from val
        
        # Extract strains for the current segment
        kappa = xi[3 * i + 0]
        sigma_x = xi[3 * i + 1]
        sigma_y = xi[3 * i + 2]
        
        # Compute the length of the current segment to integrate
        l_i = jnp.where(i == segment_idx, s_local, l[i])
        
        # Compute the orientation angle for the current segment
        dth = kappa * l_i  # Angle increment for the current segment
        th = th_prev + dth

        # Compute the integrals for the transformation matrix       
        int_cos_th = ((jnp.sin(th) - jnp.sin(th_prev)) / kappa).reshape(())
        int_sin_th = ((jnp.cos(th_prev) - jnp.cos(th)) / kappa).reshape(())
        
        # Transformation matrix
        R = jnp.array([[int_cos_th, -int_sin_th],
                [int_sin_th, int_cos_th]])

        # Compute the position
        p = p_prev + (R @ jnp.array([sigma_x, sigma_y]).T).T
        
        return jnp.concatenate([p, jnp.array([th])])

    chi, chi_list = lax.scan(
        f = lambda carry, i: (chi_i(i, carry), chi_i(i, carry)),
        init = chi_O, 
        xs = jnp.arange(num_segments + 1))

    return chi_list[segment_idx]

def forward_kinematics_fn(
    params: Dict[str, Array], 
    q: Array, 
    s: Array, 
    eps: float = global_eps
    ) -> Array:
    """
    Compute the forward kinematics of the robot.

    Args:
        params (Dict[str, Array]): dictionary of robot parameters.
        q (Array): configuration vector of the robot.
        s (Array): point coordinate along the robot in the interval [0, L].
        eps (float, optional): small number to avoid singularities. Defaults to global_eps.

    Returns:
        Array: pose of the robot at the point s in the interval [0, L].
            The pose is represented as a 3D vector [x, y, theta] where (x, y) is the position
            and theta is the orientation angle.
    """        
    # Map the configuration to the strains
    xi = xi_eq + B_xi @ q
            
    # Add a small number to the bending strain to avoid singularities
    xi = apply_eps_to_bend_strains(xi, eps)
    
    chi = chi_fn_xi(params, xi, s)
    
    return chi

def J_autodiff(
    params: Dict[str, Array], 
    xi: Array, 
    s: Array,
    eps: float = global_eps
) -> Array:
    """
    Compute the Jacobian of the forward kinematics function with respect to the strain vector.

    Args:
        params (Dict[str, Array]): dictionary of robot parameters.
        xi (Array): strain vector of the robot.
        s (Array): point coordinate along the robot in the interval [0, L].
        eps (float, optional): small number to avoid singularities. Defaults to global_eps.

    Returns:
        Array: Jacobian of the forward kinematics function with respect to the strain vector.
    """
    # Add a small number to the bending strain to avoid singularities
    xi = apply_eps_to_bend_strains(xi, eps)
    
    # Compute the Jacobian of chi_fn with respect to xi
    J = jax.jacobian(lambda xi: chi_fn_xi(params, xi, s))(xi)

    # apply the strain basis to the Jacobian
    J = J @ B_xi

    return J

def tilde_SE3( 
    vec3:Array
)-> Array:
    """
    Computes the tilde operator of SE(3) for a 3D vector.
    
    Args:
        vec (Array): array-like, shape (3,1)
            A 3-dimensional vector.
    
    Returns:
        Array: shape (3, 3)
            A 3x3 matrix representing the tilde operator of the input vector.
    """
    vec3 = vec3.reshape(-1)  # Ensure vec3 is a 1D array
    
    # Extract components of the vector
    x, y, z = vec3.flatten()
    
    # Use JAX's array creation for better performance
    Mtilde = jnp.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])
    return Mtilde

def Adjoint_gn_SE3_inv(
    params: Dict[str, Array],
    xi_n: Array,
    l_nprev: float,
    s :float,
    )-> Array:
    """
    Computes the adjoint representation of a position of a points at s (general curvilinear coordinate)
    along a rod in SE(3) deformed ine the current segment according to a strain vector xi_n.
    
    If s is a point of the n-th segment, this function use the lenght from the origin of the rod to the begining of the n-th segment.

    Args:
        xi_n (Array): array-like, shape (6,1)
            A 6-dimensional vector representing the screw in SE(3).
        l_nprev (float): 
            The length from the origin of the rod to the beginning of the n-th segment.
        s (float):
            The curvilinear coordinate along the rod, representing the position of a point in the n-th segment.

    Returns:
        Array: shape (6, 6)
            A 6x6 matrix representing the adjoint transformation of the input screw vector at the specified position.
    """
    px, py, theta = chi_fn_xi(params, xi_n, s)  # Get the pose at point s
    Rinv = jnp.array([             # Rotation matrix around the z-axis
        [jnp.cos(-theta), -jnp.sin(-theta), 0],
        [jnp.sin(-theta), jnp.cos(-theta), 0],
        [0, 0, 1]
    ])
    p = jnp.array([px, py, 0])  # Position vector at point s in the n-th segment
    
    ptilde = tilde_SE3(p)  # Compute the tilde operator of the position vector
    
    Adjoint_inv = jnp.block([
        [Rinv, -Rinv @ ptilde],
        [jnp.zeros((3, 3)), Rinv]
    ])
    
    return Adjoint_inv


# def J_explicit(
#     params: Dict[str, Array],
#     xi: Array,
#     s: Array,
#     eps: float = global_eps
# ) -> Array:
#     """
#     Compute the Jacobian of the forward kinematics function with respect to the strain vector at a given point s.

#     Args:
#         params (Dict[str, Array]): dictionary of robot parameters.
#         xi (Array): strain vector of the robot.
#         s (Array): point coordinate along the robot in the interval [0, L_tot].
#         eps (float, optional): small number to avoid singularities. Defaults to global_eps.

#     Returns:
#         Array: Jacobian of the forward kinematics function with respect to the strain vector using explicit expression in SE(3).
#     """
#     # Add a small number to the bending strain to avoid singularities
#     xi = apply_eps_to_bend_strains(xi, eps)
    
#     # Classify the point along the robot to the corresponding segment
#     segment_idx, _, l_cum = classify_segment(params, s)
    
#     # Indices to extract/inject kappa_z, sigma_x, sigma_y from xi
#     SE2_to_SE3_indices = (2, 3, 4)  
    
#     # Initial condition
#     xi_SE3_0 = vec_SE2_to_xi_SE3(xi[0:3], SE2_to_SE3_indices)
    
#     # TODO: check if we better use inv or solve
#     # Ad_g0_inv_L0 = jnp.linalg.inv(Adjoint_gn_SE3(xi_SE3_0, l_cum[0], l_cum[1]))
#     # Ad_g0_inv_s = jnp.linalg.inv(Adjoint_gn_SE3(xi_SE3_0, l_cum[0], s))
#     Ad_g0_inv_L0 = jnp.linalg.solve(Adjoint_gn_SE3(xi_SE3_0, l_cum[0], l_cum[1]), jnp.eye(6))
#     Ad_g0_inv_s = jnp.linalg.solve(Adjoint_gn_SE3(xi_SE3_0, l_cum[0], s), jnp.eye(6))
#     Ad_g0_inv_L0_2 = Adjoint_gn_SE3_inv(params, xi, l_cum[0], l_cum[1])
#     Ad_g0_inv_s_2 = Adjoint_gn_SE3_inv(params, xi, l_cum[0], s)
    
#     print("Ad_g0_inv_L0: ", Ad_g0_inv_L0)
#     print("Ad_g0_inv_L0_2: ", Ad_g0_inv_L0_2)
    
#     print("Ad_g0_inv_s: ", Ad_g0_inv_s)
#     print("Ad_g0_inv_s_2: ", Ad_g0_inv_s_2)
    
#     T_g0_L0 = Tangent_gn_SE3(xi_SE3_0, l_cum[0], l_cum[1])
#     T_g0_s = Tangent_gn_SE3(xi_SE3_0, l_cum[0], s)

#     mat_0_L0 = Ad_g0_inv_L0 @ T_g0_L0
#     mat_0_s = Ad_g0_inv_s @ T_g0_s
    
#     J_0_L0 = jnp.zeros((num_segments, 6, 6)).at[0].set(mat_0_L0)
#     J_0_s = jnp.zeros((num_segments, 6, 6)).at[0].set(mat_0_s)
    
#     tuple_J_0 = (J_0_L0, J_0_s)
    
#     # Iteration function
#     def J_i(
#         tuple_J_prev: Array,
#         i: int
#     ) -> Array:
#         J_prev_Lprev, _ = tuple_J_prev
        
#         start_index = 3 * i
#         xi_i = lax.dynamic_slice(xi, (start_index,), (3,))
#         xi_SE3_i = vec_SE2_to_xi_SE3(xi_i, SE2_to_SE3_indices)
        
#         # TODO: check if we better use inv or solve
#         # Ad_gi_inv_Li = jnp.linalg.inv(Adjoint_gn_SE3(xi_SE3_i, l_cum[i], l_cum[i+1]))
#         # Ad_gi_inv_s = jnp.linalg.inv(Adjoint_gn_SE3(xi_SE3_i, l_cum[i], s))
#         Ad_gi_inv_Li = jnp.linalg.solve(Adjoint_gn_SE3(xi_SE3_i, l_cum[i], l_cum[i+1]), jnp.eye(6))
#         Ad_gi_inv_s = jnp.linalg.solve(Adjoint_gn_SE3(xi_SE3_i, l_cum[i], s), jnp.eye(6))
        
#         T_gi_Li = Tangent_gn_SE3(xi_SE3_i, l_cum[i], l_cum[i+1])
#         T_gi_s = Tangent_gn_SE3(xi_SE3_i, l_cum[i], s)
        
#         mat_i_Li = Ad_gi_inv_Li @ T_gi_Li
#         mat_i_s = Ad_gi_inv_s @ T_gi_s
        
#         # TODO: check if we better use vmap or einsum and dynamic_update_slice or set
#         # J_new_s = (vmap(lambda j: Ad_gi_inv_s@j)(J_prev_Lprev)).at[i].set(mat_i_s)
#         J_new_s = lax.dynamic_update_slice( 
#             jnp.einsum('ij, njk->nik', Ad_gi_inv_s, J_prev_Lprev),
#             mat_i_s[jnp.newaxis, ...],
#             (i, 0, 0)
#         )
#         J_new_Li = lax.dynamic_update_slice(
#             jnp.einsum('ij, njk->nik', Ad_gi_inv_Li, J_prev_Lprev),
#             mat_i_Li[jnp.newaxis, ...],
#             (i, 0, 0)
#         )
        
#         tuple_J_new = (J_new_Li, J_new_s)
        
#         return tuple_J_new, J_new_s # We accumulate J_new_s

#     _, J_array = lax.scan(
#         f = J_i,
#         init = tuple_J_0, 
#         xs = jnp.arange(1, num_segments))
    
#     # Add the initial condition to the Jacobian array
#     J_array = jnp.concatenate([J_0_s[jnp.newaxis, ...], J_array], axis=0)
    
#     # Extract the Jacobian for the segment that contains the point s
#     J_segment_SE3_local = lax.dynamic_index_in_dim(J_array, segment_idx, axis=0, keepdims=False)
    
#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     # From local to global frame : applying the rotation of the pose at point s
    
#     # Get the pose at point s
#     px, py, theta = chi_fn_xi(params, xi, s)  
#     # Convert the pose to SE(3) representation
#     g_s = jnp.eye(4)  # Initialize as identity matrix
#     R = jnp.array([             # Rotation matrix around the z-axis
#         [jnp.cos(theta), -jnp.sin(theta)],
#         [jnp.sin(theta), jnp.cos(theta) ]
#     ])
#     g_s = g_s.at[:2, :2].set(R)  # Set rotation part as a 2D rotation matrix around the z-axis
#     Adjoint_g_s = Adjoint_g_SE3(g_s)
#     # For each segment, compute the Jacobian in SE(3) coordinates in global frame J_i_global = Adjoint_g_s @ J_i_local
#     J_segment_SE3_global = jnp.einsum('ij,njk->nik', Adjoint_g_s, J_segment_SE3_local)  # shape: (n_segments, 6, 6)
    
#     # Extracted matrix to switch back from SE(3) to SE(2)
#     interest_coordinates = [2, 3, 4]
#     interest_strain = [2, 3, 4]
#     reordered_columns = [1, 2, 0]
    
#     J_global = J_segment_SE3_global[:, interest_coordinates][:, :, interest_strain][:, reordered_columns, :] # shape: (n_segments, 3, 3)
#     J_global = blk_concat(J_global) # shape: (n_segments*3, 3)
#     J_global = J_global @ B_xi
    
#     return J_global

def J_explicit(
        params: Dict[str, Array],
        xi: Array,
        s: Array,
        eps: float = global_eps
    ) -> Array:
        """
        Compute the Jacobian of the forward kinematics function with respect to the strain vector at a given point s.

        Args:
            params (Dict[str, Array]): dictionary of robot parameters.
            xi (Array): strain vector of the robot.
            s (Array): point coordinate along the robot in the interval [0, L_tot].
            eps (float, optional): small number to avoid singularities. Defaults to global_eps.

        Returns:
            Array: Jacobian of the forward kinematics function with respect to the strain vector using numeric expression in SE(3).
        """
        # Add a small number to the bending strain to avoid singularities
        xi = apply_eps_to_bend_strains(xi, eps)
        
        # Classify the point along the robot to the corresponding segment
        segment_idx, _, l_cum = classify_segment(params, s)
        
        # Indices to extract/inject kappa_z, sigma_x, sigma_y from xi
        SE2_to_SE3_indices = (2, 3, 4)  
        
        # Initial condition
        xi_SE3_0 = vec_SE2_to_xi_SE3(xi[0:3], SE2_to_SE3_indices)
        
        # TODO: check if we better use inv or solve
        # Ad_g0_inv_L0 = jnp.linalg.inv(Adjoint_gn_SE3(xi_SE3_0, l_cum[0], l_cum[1]))
        # Ad_g0_inv_s = jnp.linalg.inv(Adjoint_gn_SE3(xi_SE3_0, l_cum[0], s))
        
        # px_L0, py_L0, theta_L0 = chi_fn_xi(params, xi, l_cum[1])
        # px_s, py_s, theta_s = chi_fn_xi(params, xi, s)
        
        # # Convert the pose to SE(3) representation
        # R_L0_inv = jnp.array([             # Rotation matrix around the z-axis
        #     [jnp.cos(-theta_L0), -jnp.sin(-theta_L0), 0],
        #     [jnp.sin(-theta_L0), jnp.cos(-theta_L0), 0 ], 
        #     [0, 0, 1]
        # ])
        # R_s_inv = jnp.array([             # Rotation matrix around the z-axis
        #     [jnp.cos(-theta_s), -jnp.sin(-theta_s), 0],
        #     [jnp.sin(-theta_s), jnp.cos(-theta_s), 0 ], 
        #     [0, 0, 1]
        # ])
        # p_L0 = jnp.array([px_L0, py_L0, 0])
        # p_s = jnp.array([px_s, py_s, 0])
        # ptilde_L0 = tilde_SE3(p_L0)
        # ptilde_s = tilde_SE3(p_s)
        # Adjoint_inv_L0 = jnp.block([
        #     [R_L0_inv, -R_L0_inv @ ptilde_L0],
        #     [jnp.zeros((3, 3)), R_L0_inv]
        # ])
        # Adjoint_inv_s = jnp.block([
        #     [R_s_inv, -R_s_inv @ ptilde_s],
        #     [jnp.zeros((3, 3)), R_s_inv]
        # ])
        
        Ad_g0_inv_L0 = jnp.linalg.solve(Adjoint_gn_SE3(xi_SE3_0, l_cum[0], l_cum[1]), jnp.eye(6))
        Ad_g0_inv_s = jnp.linalg.solve(Adjoint_gn_SE3(xi_SE3_0, l_cum[0], s), jnp.eye(6))
        
        T_g0_L0 = Tangent_gn_SE3(xi_SE3_0, l_cum[0], l_cum[1])
        T_g0_s = Tangent_gn_SE3(xi_SE3_0, l_cum[0], s)

        mat_0_L0 = Ad_g0_inv_L0 @ T_g0_L0
        mat_0_s = Ad_g0_inv_s @ T_g0_s
        
        J_0_L0 = jnp.zeros((num_segments, 6, 6)).at[0].set(mat_0_L0)
        J_0_s = jnp.zeros((num_segments, 6, 6)).at[0].set(mat_0_s)
        
        tuple_J_0 = (J_0_L0, J_0_s)
        
        # Iteration function
        def J_i(
            tuple_J_prev: Array,
            i: int
        ) -> Array:
            J_prev_Lprev, _ = tuple_J_prev
            
            start_index = 3 * i
            xi_i = lax.dynamic_slice(xi, (start_index,), (3,))
            xi_SE3_i = vec_SE2_to_xi_SE3(xi_i, SE2_to_SE3_indices)
            
            # TODO: check if we better use inv or solve
            # Ad_gi_inv_Li = jnp.linalg.inv(Adjoint_gn_SE3(xi_SE3_i, l_cum[i], l_cum[i+1]))
            # Ad_gi_inv_s = jnp.linalg.inv(Adjoint_gn_SE3(xi_SE3_i, l_cum[i], s))
            Ad_gi_inv_Li = jnp.linalg.solve(Adjoint_gn_SE3(xi_SE3_i, l_cum[i], l_cum[i+1]), jnp.eye(6))
            Ad_gi_inv_s = jnp.linalg.solve(Adjoint_gn_SE3(xi_SE3_i, l_cum[i], s), jnp.eye(6))
            
            T_gi_Li = Tangent_gn_SE3(xi_SE3_i, l_cum[i], l_cum[i+1])
            T_gi_s = Tangent_gn_SE3(xi_SE3_i, l_cum[i], s)
            
            mat_i_Li = Ad_gi_inv_Li @ T_gi_Li
            mat_i_s = Ad_gi_inv_s @ T_gi_s
            
            J_new_s = lax.dynamic_update_slice( 
                jnp.einsum('ij, njk->nik', Ad_gi_inv_s, J_prev_Lprev),
                mat_i_s[jnp.newaxis, ...],
                (i, 0, 0)
            )
            J_new_Li = lax.dynamic_update_slice(
                jnp.einsum('ij, njk->nik', Ad_gi_inv_Li, J_prev_Lprev),
                mat_i_Li[jnp.newaxis, ...],
                (i, 0, 0)
            )
            
            tuple_J_new = (J_new_Li, J_new_s)
            
            return tuple_J_new, J_new_s # We accumulate J_new_s

        _, J_array = lax.scan(
            f = J_i,
            init = tuple_J_0, 
            xs = jnp.arange(1, num_segments))
        
        # Add the initial condition to the Jacobian array
        J_array = jnp.concatenate([J_0_s[jnp.newaxis, ...], J_array], axis=0)
        
        # Extract the Jacobian for the segment that contains the point s
        J_segment_SE3_local = lax.dynamic_index_in_dim(J_array, segment_idx, axis=0, keepdims=False)
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # From local to global frame : applying the rotation of the pose at point s
        
        # Get the pose at point s
        px, py, theta = chi_fn_xi(params, xi, s)  
        # Convert the pose to SE(3) representation
        R = jnp.array([             # Rotation matrix around the z-axis
            [jnp.cos(theta), -jnp.sin(theta)],
            [jnp.sin(theta), jnp.cos(theta) ]
        ])
        
        g_s = jnp.block([
            [R, jnp.zeros((2, 2))],
            [jnp.zeros((2, 2)), jnp.eye(2)]
        ])
        Adjoint_g_s = Adjoint_g_SE3(g_s)
        # For each segment, compute the Jacobian in SE(3) coordinates in global frame J_i_global = Adjoint_g_s @ J_i_local
        J_segment_SE3_global = jnp.einsum('ij,njk->nik', Adjoint_g_s, J_segment_SE3_local)  # shape: (n_segments, 6, 6)
        
        # Extracted matrix to switch back from SE(3) to SE(2)
        interest_coordinates = [2, 3, 4]
        interest_strain = [2, 3, 4]
        reordered_columns = [1, 2, 0]
        
        J_global = J_segment_SE3_global[:, interest_coordinates][:, :, interest_strain][:, reordered_columns, :] # shape: (n_segments, 3, 3)
        J_global = blk_concat(J_global) # shape: (n_segments*3, 3)
        J_global = J_global @ B_xi
        
        return J_global

# @jit
def J_Jd_autodiff(
    params: Dict[str, Array], 
    xi: Array, 
    xi_d: Array,
    s: Array, 
    eps: float = global_eps
) -> Tuple[Array, Array]:
    """
    Compute the Jacobian of the forward kinematics function and its time-derivative.

    Args:
        params (Dict[str, Array]): dictionary of robot parameters.
        xi (Array): strain vector of the robot.
        xi_d (Array): velocity vector of the robot.
        s (Array): point coordinate along the robot in the interval [0, L].
        eps (float, optional): small number to avoid singularities. Defaults to global_eps.

    Returns:
        Tuple[Array, Array]: 
            - J (Array): Jacobian of the forward kinematics function.
            - J_d (Array): time-derivative of the Jacobian of the forward kinematics function.
    """
    # Add a small number to the bending strain to avoid singularities
    xi = apply_eps_to_bend_strains(xi, eps)
    
    # Compute the Jacobian of chi_fn with respect to xi TODO
    J = jax.jacobian(lambda xi: chi_fn_xi(params, xi, s))(xi)
    # TODO: change jacobian to not use autodifferentiation
    
    dJ_dxi = jax.jacobian(J)(xi)  
    # TODO: change jacobian to not use autodifferentiation
    J_d = jnp.tensordot(dJ_dxi, xi_d, axes=([2], [0]))

    # apply the strain basis to the Jacobian
    J = J @ B_xi
    
    # apply the strain basis to the time-derivative of the Jacobian
    J_d = J_d @ B_xi

    return J, J_d

def J_Jd_explicit(
    params: Dict[str, Array],
    xi: Array,
    xi_d: Array,
    s: Array,
    eps: float = global_eps
) -> Array:
    """
    Compute the Jacobian and its derivative with respect to the strain vector at a given point s.

    Args:
        params (Dict[str, Array]): dictionary of robot parameters.
        xi (Array): strain vector of the robot.
        xi_d (Array): velocity vector of the robot.
        s (Array): point coordinate along the robot in the interval [0, L].
        eps (float, optional): small number to avoid singularities. Defaults to global_eps.

    Returns:
        Tuple[Array, Array]: 
            - J (Array): Jacobian of the forward kinematics function.
            - J_d (Array): time-derivative of the Jacobian of the forward kinematics function.
    """
    # Add a small number to the bending strain to avoid singularities
    xi = apply_eps_to_bend_strains(xi, eps)
    
    # Classify the point along the robot to the corresponding segment
    segment_idx, _, l_cum = classify_segment(params, s)
    
    # Indices to extract/inject kappa_z, sigma_x, sigma_y from xi
    SE2_to_SE3_indices = (2, 3, 4)

    # =================================
    # Computation of the Jacobian
    
    # Initial condition
    xi_SE3_0 = vec_SE2_to_xi_SE3(xi[0:3], SE2_to_SE3_indices)
    
    # TODO: check if we better use inv or solve
    # Ad_g0_inv_L0 = jnp.linalg.inv(Adjoint_gn_SE3(xi_SE3_0, l_cum[0], l_cum[1]))
    # Ad_g0_inv_s = jnp.linalg.inv(Adjoint_gn_SE3(xi_SE3_0, l_cum[0], s))
    Ad_g0_inv_L0 = jnp.linalg.solve(Adjoint_gn_SE3(xi_SE3_0, l_cum[0], l_cum[1]), jnp.eye(6))
    Ad_g0_inv_s = jnp.linalg.solve(Adjoint_gn_SE3(xi_SE3_0, l_cum[0], s), jnp.eye(6))
    
    T_g0_L0 = Tangent_gn_SE3(xi_SE3_0, l_cum[0], l_cum[1])
    T_g0_s = Tangent_gn_SE3(xi_SE3_0, l_cum[0], s)

    mat_0_L0 = Ad_g0_inv_L0 @ T_g0_L0
    mat_0_s = Ad_g0_inv_s @ T_g0_s
    
    J_0_L0 = jnp.zeros((num_segments, 6, 6)).at[0].set(mat_0_L0)
    J_0_s = jnp.zeros((num_segments, 6, 6)).at[0].set(mat_0_s)
    
    tuple_J_0 = (J_0_L0, J_0_s)
    
    # Iteration function
    def J_i(
        tuple_J_prev: Array,
        i: int
    ) -> Array:
        J_prev_Lprev, _ = tuple_J_prev
        
        start_index = 3 * i
        xi_i = lax.dynamic_slice(xi, (start_index,), (3,))
        xi_SE3_i = vec_SE2_to_xi_SE3(xi_i, SE2_to_SE3_indices)
        
        # TODO: check if we better use inv or solve
        # Ad_gi_inv_Li = jnp.linalg.inv(Adjoint_gn_SE3(xi_SE3_i, l_cum[i], l_cum[i+1]))
        # Ad_gi_inv_s = jnp.linalg.inv(Adjoint_gn_SE3(xi_SE3_i, l_cum[i], s))
        Ad_gi_inv_Li = jnp.linalg.solve(Adjoint_gn_SE3(xi_SE3_i, l_cum[i], l_cum[i+1]), jnp.eye(6))
        Ad_gi_inv_s = jnp.linalg.solve(Adjoint_gn_SE3(xi_SE3_i, l_cum[i], s), jnp.eye(6))
        
        T_gi_Li = Tangent_gn_SE3(xi_SE3_i, l_cum[i], l_cum[i+1])
        T_gi_s = Tangent_gn_SE3(xi_SE3_i, l_cum[i], s)
        
        mat_i_Li = Ad_gi_inv_Li @ T_gi_Li
        mat_i_s = Ad_gi_inv_s @ T_gi_s
        
        # TODO: check if we better use vmap or einsum and dynamic_update_slice or set
        # J_new_s = (vmap(lambda j: Ad_gi_inv_s@j)(J_prev_Lprev)).at[i].set(mat_i_s)
        J_new_s = lax.dynamic_update_slice( 
            jnp.einsum('ij, njk->nik', Ad_gi_inv_s, J_prev_Lprev),
            mat_i_s[jnp.newaxis, ...],
            (i, 0, 0)
        )
        J_new_Li = lax.dynamic_update_slice(
            jnp.einsum('ij, njk->nik', Ad_gi_inv_Li, J_prev_Lprev),
            mat_i_Li[jnp.newaxis, ...],
            (i, 0, 0)
        )
        
        tuple_J_new = (J_new_Li, J_new_s)
        
        return tuple_J_new, J_new_s # We accumulate J_new_s

    _, J_array = lax.scan(
        f = J_i,
        init = tuple_J_0, 
        xs = jnp.arange(1, num_segments))
    
    # Add the initial condition to the Jacobian array
    J_array = jnp.concatenate([J_0_s[jnp.newaxis, ...], J_array], axis=0)
    
    # Extract the Jacobian for the segment that contains the point s
    J_segment_SE3_local = lax.dynamic_index_in_dim(J_array, segment_idx, axis=0, keepdims=False)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # From local to global frame : applying the rotation of the pose at point s
    
    # Get the pose at point s
    px, py, theta = chi_fn_xi(params, xi, s)  
    # Convert the pose to SE(3) representation
    g_s = jnp.eye(4)  # Initialize as identity matrix
    R = jnp.array([             # Rotation matrix around the z-axis
        [jnp.cos(theta), -jnp.sin(theta)],
        [jnp.sin(theta), jnp.cos(theta) ]
    ])
    g_s = g_s.at[:2, :2].set(R)  # Set rotation part as a 2D rotation matrix around the z-axis
    Adjoint_g_s = Adjoint_g_SE3(g_s)
    # For each segment, compute the Jacobian in SE(3) coordinates in global frame J_i_global = Adjoint_g_s @ J_i_local
    J_segment_SE3_global = jnp.einsum('ij,njk->nik', Adjoint_g_s, J_segment_SE3_local)  # shape: (n_segments, 6, 6)
    
    # =================================
    # Computation of the time-derivative of the Jacobian
        
    idx_range = jnp.arange(num_segments)
    
    xi_d_i = vmap(
        lambda i: lax.dynamic_slice(xi_d, (3 * i,), (3,))
    )(idx_range) # shape: (num_segments, 3)
    xi_d_SE3_i = vmap(
        lambda xi_d_i : vec_SE2_to_xi_SE3(xi_d_i, SE2_to_SE3_indices)
    )(xi_d_i) # shape: (num_segments, 6)
    S_i = vmap(
        lambda i:lax.dynamic_index_in_dim(J_segment_SE3_global, i, axis=0, keepdims=False)
    )(idx_range) # shape: (num_segments, 6, 6)
    sum_Sj_xi_d_j = vmap(
        lambda i, xi_d_SE3_i: compute_weighted_sums(J_segment_SE3_global, xi_d_SE3_i, i)
    )(idx_range, xi_d_SE3_i) # shape: (num_segments, 6)
    adjoint_sum = vmap(adjoint_SE3)(sum_Sj_xi_d_j) # shape: (num_segments, 6, 6)
    
    # Compute the time-derivative of the Jacobian
    J_d_segment_SE3_local = jnp.einsum('ijk, ikl->ijl', adjoint_sum, S_i) # shape: (num_segments, 6, 6)
    
    # Replace the elements of J_d_segment_SE3 for i > segment_idx by null matrices
    J_d_segment_SE3_local = jnp.where(
        jnp.arange(num_segments)[:, None, None] > segment_idx,
        jnp.zeros_like(J_d_segment_SE3_local),
        J_d_segment_SE3_local
    )
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # From local to global frame : applying the rotation of the pose at point s
    
    # Get the pose at point s
    px, py, theta = chi_fn_xi(params, xi, s)  
    # Convert the pose to SE(3) representation
    g_s = jnp.eye(4)  # Initialize as identity matrix
    R = jnp.array([             # Rotation matrix around the z-axis
        [jnp.cos(theta), -jnp.sin(theta)],
        [jnp.sin(theta), jnp.cos(theta) ]
    ])
    g_s = g_s.at[:2, :2].set(R)  # Set rotation part as a 2D rotation matrix around the z-axis
    Adjoint_g_s = Adjoint_g_SE3(g_s)
    # For each segment, compute the Jacobian in SE(3) coordinates in global frame J_i_global = Adjoint_g_s @ J_i_local
    J_d_segment_SE3_global = jnp.einsum('ij,njk->nik', Adjoint_g_s, J_d_segment_SE3_local)  # shape: (n_segments, 6, 6)
    
    # =================================
    # Switch back from SE(3) to SE(2)
    
    # Extracted matrix 
    interest_coordinates = [2, 3, 4]
    interest_strain = [2, 3, 4]
    reordered_columns = [1, 2, 0]
    
    J = J_segment_SE3_global[:, interest_coordinates][:, :, interest_strain][:, reordered_columns, :] # shape: (n_segments, 3, 3)
    J = blk_concat(J) # shape: (n_segments*3, 3)    
    
    J_d = J_d_segment_SE3_global[:, interest_coordinates][:, :, interest_strain][:, reordered_columns, :] # shape: (n_segments, 3, 3)
    J_d = blk_concat(J_d) # shape: (n_segments*3, 3)
    
    # Apply the strain basis to the Jacobian and its time-derivative
    J = J @ B_xi    
    J_d = J_d @ B_xi
    
    return J, J_d
 
def U_g_fn_xi(
    params: Dict[str, Array], 
    xi: Array, 
    eps: float = global_eps
) -> Array:
    """
    Compute the gravity vector of the robot.

    Args:
        params (Dict[str, Array]): dictionary of robot parameters.
        xi (Array): strain vector of the robot.
        eps (float, optional): small number to avoid singularities. Defaults to global_eps.

    Returns:
        Array: gravity vector of the robot.
    """
    # Add a small number to the bending strain to avoid singularities
    xi = apply_eps_to_bend_strains(xi, eps)
    
    # Extract the parameters
    g = params["g"] # gravity vector [m/s^2]
    rho = params["rho"] # density of each segment [kg/m^3]
    l = params["l"] # length of each segment [m]
    r = params["r"] # radius of each segment [m]
    
    # Usefull derived quantitie
    A = jnp.pi * r**2  # cross-sectional area of each segment [m^2]
    
    l_cum = jnp.cumsum(jnp.concatenate([jnp.array([0.0]), l]))
    # Compute each integral
    def compute_integral(i):
        if integration_type == "gauss":
            Xs, Ws, nGauss = gauss_quadrature(N_GQ=param_integration, a=l_cum[i], b=l_cum[i + 1])
            chi_s = vmap(lambda s: chi_fn_xi(params, xi, s))(Xs)
            p_s = chi_s[:, :2]
            integrand = -rho[i] * A[i] * jnp.einsum("ij,j->i", p_s, g)
            
            # Compute the integral
            integral = jnp.sum(Ws * integrand)
        elif integration_type == 'quadax':
            rule = GaussKronrodRule(order=param_integration)
            def integrand(s):
                chi_s = chi_fn_xi(params, xi, s)
                p_s = chi_s[:2]
                return -rho[i] * A[i] * jnp.dot(p_s, g)
            
            # Compute the integral
            integral, _, _, _ = rule.integrate(integrand, l_cum[i], l_cum[i+1], args=())
            
        elif integration_type == "trapezoid":
            xs = jnp.linspace(l_cum[i], l_cum[i + 1], param_integration)
            chi_s = vmap(lambda s: chi_fn_xi(params, xi, s))(xs)
            p_s = chi_s[:, :2]
            integrand = -rho[i] * A[i] * jnp.einsum("ij,j->i", p_s, g)
            
            # Compute the integral
            integral = jscipy.integrate.trapezoid(integrand, x=xs)
        
        return integral
    
    # Compute the cumulative integral
    indices = jnp.arange(num_segments)
    integrals = vmap(compute_integral)(indices)

    U_g = jnp.sum(integrals)
    
    return U_g
 
def G_fn_xi_autodiff(
    params: Dict[str, Array], 
    xi: Array
) -> Array:
    """
    Compute the gravity vector of the robot.

    Args:
        params (Dict[str, Array]): dictionary of robot parameters.
        xi (Array): strain vector of the robot.

    Returns:
        Array: gravity vector of the robot.
    """
    
    G = jax.jacobian(lambda xi: U_g_fn_xi(params, xi))(xi)
    return G

def extract_Jp_Jo(
    J: Array
) -> Tuple[Array, Array]:
    """
    Extract the Jacobian of the axial and bending strains.

    Args:
        J (Array): Jacobian of the forward kinematics function.

    Returns:
        Tuple[Array, Array]: 
            - Jp (Array): Jacobian of the axial strains.
            - Jo (Array): Jacobian of the bending strains.
    """
    # Extract the Jacobian of the axial and bending strains
    Jp = J[:, :2, :]
    Jo = J[:, 2:, :]
    
    return Jp, Jo
 
def G_fn_xi_explicit(
    params: Dict[str, Array], 
    xi: Array,
    eps: float = global_eps
) -> Array:
    """
    Compute the gravity vector of the robot.

    Args:
        params (Dict[str, Array]): dictionary of robot parameters.
        xi (Array): strain vector of the robot.
        eps (float, optional): small number to avoid singularities. Defaults to global_eps.

    Returns:
        Array: gravity vector of the robot.
    """
    
    # Add a small number to the bending strain to avoid singularities
    xi = apply_eps_to_bend_strains(xi, eps)
    
    # Extract the parameters
    g = params["g"] # gravity vector [m/s^2]
    rho = params["rho"] # density of each segment [kg/m^3]
    l = params["l"] # length of each segment [m]
    r = params["r"] # radius of each segment [m]
    
    # Usefull derived quantitie
    A = jnp.pi * r**2  # cross-sectional area of each segment [m^2]
    
    l_cum = jnp.cumsum(jnp.concatenate([jnp.array([0.0]), l]))
    # Compute each integral
    def compute_integral(i):
        if integration_type == "gauss":
            Xs, Ws, nGauss = gauss_quadrature(N_GQ=param_integration, a=l_cum[i], b=l_cum[i + 1])
            
            J_all = vmap(lambda s: J_explicit(params, xi, s))(Xs)
            Jp_all, _ = extract_Jp_Jo(J_all) # shape: (nGauss, n_segments, 3, 3)
            print("J_all:", J_all[0].shape)
            print("Jp_all:", Jp_all[0].shape)
            print("einsum", jnp.einsum("nij,nik->njk", Jp_all, Jp_all)[0].shape)
            
            # Compute the integrand
            integrand = -rho[i] * A[i] * jnp.einsum("ijk,j->ik", Jp_all, g)
            
            # Multiply each element of integrand by the corresponding weight
            weighted_integrand = jnp.einsum("i, ij->ij", Ws, integrand)

            # Compute the integral
            integral = jnp.sum(weighted_integrand, axis=0)  # sum over the Gauss points
        elif integration_type == 'quadax':
            rule = GaussKronrodRule(order=param_integration)
            def integrand(s):
                J = J_explicit(params, xi, s)
                Jp = J[:2, :]
                Jo = J[2:, :]
                print("J:", J)
                print("Jp:", Jp)
                print("Jo:", Jo)
                return -rho[i] * A[i] * jnp.dot(g, Jp)  # shape: (k,)

            # Compute the integral
            integral, _, _, _ = rule.integrate(integrand, l_cum[i], l_cum[i+1], args=())
            
        elif integration_type == "trapezoid":
            xs = jnp.linspace(l_cum[i], l_cum[i + 1], param_integration)
            
            J_all = vmap(lambda s: J_explicit(params, xi, s))(xs)
            Jp_all, _ = extract_Jp_Jo(J_all) # shape: (nGauss, n_segments, 3, 3)
            
            # Compute the integrand
            integrand = -rho[i] * A[i] * jnp.einsum("ijk,j->ik", Jp_all, g)
            
            # Multiply each element of integrand by the corresponding weight
            weighted_integrand = jnp.einsum("i, ij->ij", Ws, integrand)

            # Compute the integral
            integral = jnp.sum(weighted_integrand, axis=0)  # sum over the Gauss points
        
        return integral
    
    # Compute the cumulative integral
    indices = jnp.arange(num_segments)
    integrals = vmap(compute_integral)(indices)

    G = jnp.sum(integrals, axis=0)  # sum over the segments
    return G

if __name__ == "__main__":
    num_segments = 2
    n_xi = 3 * num_segments

    # by default, set the axial rest strain (local y-axis) along the entire rod to 1.0
    rest_strain_reshaped = jnp.zeros((n_xi,)).reshape((-1, 3))
    rest_strain_reshaped = rest_strain_reshaped.at[:, -1].set(1.0)
    xi_eq = rest_strain_reshaped.flatten()
    
    strain_selector = jnp.ones((n_xi,), dtype=bool)
    
    B_xi = compute_strain_basis(strain_selector)
    
    rho = 1070 * jnp.ones((num_segments,))      # Volumetric density of Dragon Skin 20 [kg/m^3]
    params = {
        "th0": jnp.array(0.0),                  # Initial orientation angle [rad]
        "l": 1.0 * jnp.ones((num_segments,)),  # Length of each segment [m]
        "r": 2e-2 * jnp.ones((num_segments,)),  # Radius of each segment [m]
        "rho": rho,
        "g": jnp.array([0.0, 9.81]),            # Gravity vector [m/s^2]
        "E": 2e3 * jnp.ones((num_segments,)),   # Elastic modulus [Pa]
        "G": 1e3 * jnp.ones((num_segments,)),   # Shear modulus [Pa]
    }
    params["D"] = 1e-3 * jnp.diag(              # Damping matrix [Ns/m]
        (jnp.repeat(
            jnp.array([[1e0, 1e3, 1e3]]), num_segments, axis=0
        ) * params["l"][:, None]).flatten()
    )
    
    jnp.set_printoptions(suppress=True, precision=7)
    
    q = jnp.array([1, 0.0, 0.0] * num_segments)
    print("Configuration vector:", q)
    xi = xi_eq + B_xi @ q
    print("Strain vector:", xi)
    s = 3.5
    print("Point along the robot:", s)
    
    if num_segments < 2:
        sym_exp_filepath = (
            Path(jsrm.__file__).parent
            / "symbolic_expressions"
            / f"planar_pcs_ns-{num_segments}.dill"
        )    
        strain_basis, forward_kinematics_fn, dynamical_matrices_fn, auxiliary_fns = (
            planar_pcs.factory(sym_exp_filepath, strain_selector)
        )
        
        print("\nComputing jacobian with symbolic expressions...")
        J_symbolic = auxiliary_fns["jacobian_fn"]    
        J_symbolic_val = J_symbolic(params, q, s)
        print("Jacobian (symbolic):\n", J_symbolic_val)
    
    print("\nComputing jacobian with explicit expressions...")
    J_explicit_val = J_explicit(params, xi, s)
    print("Jacobian (explicit):\n", J_explicit_val)  
    
    print("\nComputing jacobian with autodiff...")
    J_autodiff_val = J_autodiff(params, xi, s)
    print("Jacobian (autodiff):\n", J_autodiff_val)
    
    # integration_type = "quadax"  # or "trapezoid"
    # param_integration = 21  # Number of integration points for the trapezoid rule or Gauss quadrature
    
    # print("\nComputing gravity ...")
    # U_g = U_g_fn_xi(params, xi)
    # print("U_g:\n", U_g)
    
    # print("\nComputing gravity vector with autodiff...")
    # G_fn_xi_autodiff_val = G_fn_xi_autodiff(params, xi)
    # print("Gravity vector (autodiff):\n", G_fn_xi_autodiff_val)
    
    # print("\nComputing gravity vector with explicit expressions...")
    # G_fn_xi_explicit_val = G_fn_xi_explicit(params, xi)
    # print("Gravity vector (explicit):\n", G_fn_xi_explicit_val)
    
    integration_type = "gauss"  # or "trapezoid"
    param_integration = 5  # Number of integration points for the trapezoid rule or Gauss quadrature
    
    print("\nComputing gravity ...")
    U_g = U_g_fn_xi(params, xi)
    print("U_g:\n", U_g)
    
    print("\nComputing gravity vector with autodiff...")
    G_fn_xi_autodiff_val = G_fn_xi_autodiff(params, xi)
    print("Gravity vector (autodiff):\n", G_fn_xi_autodiff_val)
    
    print("\nComputing gravity vector with explicit expressions...")
    G_fn_xi_explicit_val = G_fn_xi_explicit(params, xi)
    print("Gravity vector (explicit):\n", G_fn_xi_explicit_val)
    
    # xi_d = jnp.array([0.01, 0.02, 0.03] * num_segments)  # Example velocity vector
    # print("\nVelocity vector:\n", xi_d)
    # J_autodiff_val, Jd_autodiff_val = J_Jd_autodiff(params, xi, xi_d, s)
    # print("\nJacobian time-derivative (autodiff):\n", Jd_autodiff_val)
    # J_explicit_val, Jd_explicit_val = J_Jd_explicit(params, xi, xi_d, s)
    # print("\nJacobian time-derivative (explicit):\n", Jd_explicit_val)


    
    # operational_space_dynamical_matrices_fn = auxiliary_fns["operational_space_dynamical_matrices_fn"]
    # def J_symbolic(
    #     params: Dict[str, Array],
    #     q: Array, 
    #     s: Array
    # )->Array:
    #     return operational_space_dynamical_matrices_fn(
    #         params,
    #         q
    #         q_d = None,
    #         s,
    #         B = None,  # Not used in this context
    #         C = None  # Not used in this context
    #         )(2)
    # J_symb_val = J_symb(params, q, s)
    # # J_d_symb_val = J_d_symb(params, q, s)
    
bool_autodiff = False  # Set to True to use autodifferentiation, False to use explicit expressions
if bool_autodiff:
    jacobian_fn_xi = J_autodiff
    J_Jd = J_Jd_autodiff
else:
    jacobian_fn_xi = J_explicit
    J_Jd = J_Jd_explicit

# @jit
def jacobian_fn(
    params: Dict[str, Array], 
    q: Array, 
    s: Array, 
    eps: float = global_eps
) -> Array:
    """
    Compute the Jacobian of the forward kinematics function with respect to the configuration vector q.

    Args:
        params (Dict[str, Array]): dictionary of robot parameters.
        q (Array): configuration vector of the robot.
        s (Array): point coordinate along the robot in the interval [0, L].
        eps (float, optional): small number to avoid singularities in the bending strain. Defaults to global_eps.

    Returns:
        Array: Jacobian of the forward kinematics function with respect to the strain vector.
    """
    
    # Map the configuration to the strains
    xi = xi_eq + B_xi @ q

    # Compute the Jacobian of chi_fn with respect to xi
    J = jacobian_fn_xi(params, xi, s)
    return J


