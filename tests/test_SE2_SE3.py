from jax import Array, jit, lax, vmap
from jax import jacobian, grad
from jax import scipy as jscipy
from jax import numpy as jnp
from quadax import GaussKronrodRule

import numpy as onp
from typing import Callable, Dict, Tuple, Optional

from jsrm.math_utils import blk_diag, blk_concat
from jsrm.utils.lie_operators import ( # To use SE(3)
    vec_SE2_to_xi_SE3,
    Tangent_gn_SE3, 
    Adjoint_gn_SE3,
    Adjoint_gn_SE3_inv,
    Adjoint_g_SE3,
    adjoint_SE3,
)
from jsrm.utils.lie_operators import ( # To use SE(2)
    Tangent_gn_SE2, 
    Adjoint_gn_SE2,
    Adjoint_gn_SE2_inv,
    Adjoint_g_SE2, 
    adjoint_SE2,
)
from jsrm.utils.lie_operators import (
    compute_weighted_sums,
)
from jax import random

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
    l_cum = jnp.cumsum(jnp.concatenate([jnp.zeros(1), l]))
    
    # Classify the point along the robot to the corresponding segment
    segment_idx = jnp.clip(jnp.sum(s > l_cum) - 1, 0, len(l) - 1)
    
    # Compute the point coordinate along the segment in the interval [0, l_segment]
    s_segment = s - l_cum[segment_idx]

    return segment_idx, s_segment.squeeze(), l_cum

#@jit
def chi_fn(
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
    th0 = params["th0"]  # initial angle of the robot
    l = params["l"] # length of each segment [m]
    
    # Classify the point along the robot to the corresponding segment
    segment_idx, s_local, _ = classify_segment(params, s)
    
    chi_0 = jnp.concatenate([jnp.zeros(2), th0[None]])  # Initial pose of the robot #TODO

    # Iteration function
    def chi_i(
        i: int, 
        chi_prev: Array
    ) -> Array:
        th_prev = chi_prev[2]  # Extract previous orientation angle from val
        p_prev = chi_prev[:2]  # Extract previous position from val
        
        # Extract strains for the current segment
        kappa = xi[3 * i + 0]       # Bending strain
        sigma_x = xi[3 * i + 1]     # Shear strain
        sigma_y = xi[3 * i + 2]     # Axial strain
        
        # Compute the length of the current segment to integrate
        l_i = jnp.where(i == segment_idx, s_local, l[i])
        
        # Compute the orientation angle for the current segment
        dth = kappa * l_i  # Angle increment for the current segment
        th = th_prev + dth

        # Compute the integrals for the transformation matrix
        int_cos_th = ((jnp.sin(th) - jnp.sin(th_prev)) / kappa).reshape(())
        int_sin_th = ((jnp.cos(th_prev) - jnp.cos(th)) / kappa).reshape(())
        
        # Transformation matrix
        R = jnp.stack([
            jnp.stack([int_cos_th, -int_sin_th]),
            jnp.stack([int_sin_th,  int_cos_th])
        ])

        # Compute the position
        p = p_prev + R @ jnp.stack([sigma_x, sigma_y], axis=-1)            
        
        return jnp.concatenate([p, th[None]])

    chi, chi_list = lax.scan(
        f = lambda carry, i: (chi_i(i, carry), chi_i(i, carry)),
        init = chi_0, 
        xs = jnp.arange(num_segments + 1))

    return chi_list[segment_idx]

#@jit
def J_explicit_global(
    params: Dict[str, Array],
    xi: Array,
    s: Array
) -> Array:
    """
    Compute the inertial-frame Jacobian of the forward kinematics function with respect to the strain vector at a given point s.

    Args:
        params (Dict[str, Array]): dictionary of robot parameters.
        xi (Array): strain vector of the robot.
        s (Array): point coordinate along the robot in the interval [0, L_tot].

    Returns:
        Array: inertial-frame Jacobian of the forward kinematics function with respect to the strain vector using explicit expression in SE(2).
    """
    
    # Classify the point along the robot to the corresponding segment
    segment_idx, _, l_cum = classify_segment(params, s) 
    
    # Initial condition
    xi_SE2_0 = xi[0:3]

    Ad_g0_inv_L0 = Adjoint_gn_SE2_inv(xi_SE2_0, l_cum[0], l_cum[1])
    Ad_g0_inv_s = Adjoint_gn_SE2_inv(xi_SE2_0, l_cum[0], s)
    
    T_g0_L0 = Tangent_gn_SE2(xi_SE2_0, l_cum[0], l_cum[1])
    T_g0_s = Tangent_gn_SE2(xi_SE2_0, l_cum[0], s)

    mat_0_L0 = Ad_g0_inv_L0 @ T_g0_L0
    mat_0_s = Ad_g0_inv_s @ T_g0_s
    
    J_0_L0 = jnp.concatenate([mat_0_L0[None, :, :], jnp.zeros((num_segments - 1, 3, 3))], axis=0)
    J_0_s = jnp.concatenate([mat_0_s[None, :, :], jnp.zeros((num_segments - 1, 3, 3))], axis=0)
    
    tuple_J_0 = (J_0_L0, J_0_s)
    
    # Iteration function
    def J_i(
        tuple_J_prev: Array,
        i: int
    ) -> Array:
        J_prev_Lprev, _ = tuple_J_prev
        
        start_index = 3 * i
        xi_SE2_i = lax.dynamic_slice(xi, (start_index,), (3,))
        
        Ad_gi_inv_Li = Adjoint_gn_SE2_inv(xi_SE2_i, l_cum[i], l_cum[i+1])
        Ad_gi_inv_s = Adjoint_gn_SE2_inv(xi_SE2_i, l_cum[i], s)
        
        T_gi_Li = Tangent_gn_SE2(xi_SE2_i, l_cum[i], l_cum[i+1])
        T_gi_s = Tangent_gn_SE2(xi_SE2_i, l_cum[i], s)
        
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
    J_segment_SE2_local = lax.dynamic_index_in_dim(J_array, segment_idx, axis=0, keepdims=False)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # From local to global frame : applying the rotation of the pose at point s
    
    # Get the pose at point s
    _, _, theta = chi_fn(params, xi, s)  
    # Convert the pose to SE(3) representation
    c, s = jnp.cos(theta), jnp.sin(theta)
    R = jnp.stack([
        jnp.stack([c, -s]),
        jnp.stack([s,  c])
    ])
    g_s = jnp.block([
        [R, jnp.zeros((2, 1))],
        [jnp.zeros((1, 2)), jnp.eye(1)]
    ])
    Adjoint_g_s = Adjoint_g_SE2(g_s)
    # For each segment, compute the Jacobian in SE(3) coordinates in global frame J_i_global = Adjoint_g_s @ J_i_local
    J_segment_SE2_global = jnp.einsum('ij,njk->nik', Adjoint_g_s, J_segment_SE2_local)  # shape: (n_segments, 6, 6)
    
    reordered_lines = [1, 2, 0]
    
    J_global = J_segment_SE2_global[:, reordered_lines, :] # shape: (n_segments, 3, 3)
    J_global = blk_concat(J_global) # shape: (n_segments*3, 3)
    J_global = J_global @ B_xi
    
    return J_global

#@jit
def J_explicit_global_SE3(
    params: Dict[str, Array],
    xi: Array,
    s: Array
) -> Array:
    """
    Compute the inertial-frame Jacobian of the forward kinematics function with respect to the strain vector at a given point s.

    Args:
        params (Dict[str, Array]): dictionary of robot parameters.
        xi (Array): strain vector of the robot.
        s (Array): point coordinate along the robot in the interval [0, L_tot].

    Returns:
        Array: inertial-frame Jacobian of the forward kinematics function with respect to the strain vector using explicit expression in SE(3).
    """
    
    # Classify the point along the robot to the corresponding segment
    segment_idx, _, l_cum = classify_segment(params, s)
    
    # Indices to extract/inject kappa_z, sigma_x, sigma_y from xi
    SE2_to_SE3_indices = (2, 3, 4)  
    
    # Initial condition
    xi_SE3_0 = vec_SE2_to_xi_SE3(xi[0:3], SE2_to_SE3_indices)

    Ad_g0_inv_L0 = Adjoint_gn_SE3_inv(xi_SE3_0, l_cum[0], l_cum[1])
    Ad_g0_inv_s = Adjoint_gn_SE3_inv(xi_SE3_0, l_cum[0], s)
    
    
    T_g0_L0 = Tangent_gn_SE3(xi_SE3_0, l_cum[0], l_cum[1])
    T_g0_s = Tangent_gn_SE3(xi_SE3_0, l_cum[0], s)

    mat_0_L0 = Ad_g0_inv_L0 @ T_g0_L0
    mat_0_s = Ad_g0_inv_s @ T_g0_s
    
    J_0_L0 = jnp.concatenate([mat_0_L0[None, :, :], jnp.zeros((num_segments - 1, 6, 6))], axis=0)
    J_0_s = jnp.concatenate([mat_0_s[None, :, :], jnp.zeros((num_segments - 1, 6, 6))], axis=0)
    
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
        
        Ad_gi_inv_Li = Adjoint_gn_SE3_inv(xi_SE3_i, l_cum[i], l_cum[i+1])
        Ad_gi_inv_s = Adjoint_gn_SE3_inv(xi_SE3_i, l_cum[i], s)
        
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
    _, _, theta = chi_fn(params, xi, s)  
    # Convert the pose to SE(3) representation
    c, s = jnp.cos(theta), jnp.sin(theta)
    R = jnp.stack([
        jnp.stack([c, -s]),
        jnp.stack([s,  c])
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
    reordered_lines = [1, 2, 0]
    
    J_global = J_segment_SE3_global[:, interest_coordinates][:, :, interest_strain][:, reordered_lines, :] # shape: (n_segments, 3, 3)
    J_global = blk_concat(J_global) # shape: (n_segments*3, 3)
    J_global = J_global @ B_xi
    
    return J_global

#@jit
def J_explicit_local(
    params: Dict[str, Array],
    xi: Array,
    s: Array
) -> Array:
    """
    Compute the body-frame Jacobian of the forward kinematics function with respect to the strain vector at a given point s.

    Args:
        params (Dict[str, Array]): dictionary of robot parameters.
        xi (Array): strain vector of the robot.
        s (Array): point coordinate along the robot in the interval [0, L_tot].

    Returns:
        Array: body-frame Jacobian of the forward kinematics function with respect to the strain vector using explicit expression in SE(2).
    """
    
    # Classify the point along the robot to the corresponding segment
    segment_idx, _, l_cum = classify_segment(params, s) 
    
    # Initial condition
    xi_SE2_0 = xi[0:3]

    Ad_g0_inv_L0 = Adjoint_gn_SE2_inv(xi_SE2_0, l_cum[0], l_cum[1])
    Ad_g0_inv_s = Adjoint_gn_SE2_inv(xi_SE2_0, l_cum[0], s)
    
    T_g0_L0 = Tangent_gn_SE2(xi_SE2_0, l_cum[0], l_cum[1])
    T_g0_s = Tangent_gn_SE2(xi_SE2_0, l_cum[0], s)

    mat_0_L0 = Ad_g0_inv_L0 @ T_g0_L0
    mat_0_s = Ad_g0_inv_s @ T_g0_s
    
    J_0_L0 = jnp.concatenate([mat_0_L0[None, :, :], jnp.zeros((num_segments - 1, 3, 3))], axis=0)
    J_0_s = jnp.concatenate([mat_0_s[None, :, :], jnp.zeros((num_segments - 1, 3, 3))], axis=0)
    
    tuple_J_0 = (J_0_L0, J_0_s)
    
    # Iteration function
    def J_i(
        tuple_J_prev: Array,
        i: int
    ) -> Array:
        J_prev_Lprev, _ = tuple_J_prev
        
        start_index = 3 * i
        xi_SE2_i = lax.dynamic_slice(xi, (start_index,), (3,))
        
        Ad_gi_inv_Li = Adjoint_gn_SE2_inv(xi_SE2_i, l_cum[i], l_cum[i+1])
        Ad_gi_inv_s = Adjoint_gn_SE2_inv(xi_SE2_i, l_cum[i], s)
        
        T_gi_Li = Tangent_gn_SE2(xi_SE2_i, l_cum[i], l_cum[i+1])
        T_gi_s = Tangent_gn_SE2(xi_SE2_i, l_cum[i], s)
        
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
    J_segment_SE2_local = lax.dynamic_index_in_dim(J_array, segment_idx, axis=0, keepdims=False)
    
    reordered_lines = [1, 2, 0]
    
    J_local = J_segment_SE2_local[:, reordered_lines, :]
    J_local = blk_concat(J_local) # shape: (n_segments*3, 3)
    J_local = J_local @ B_xi
    
    return J_local

#@jit
def J_explicit_local_SE3(
    params: Dict[str, Array],
    xi: Array,
    s: Array
) -> Array:
    """
    Compute the body-frame Jacobian of the forward kinematics function with respect to the strain vector at a given point s.

    Args:
        params (Dict[str, Array]): dictionary of robot parameters.
        xi (Array): strain vector of the robot.
        s (Array): point coordinate along the robot in the interval [0, L_tot].

    Returns:
        Array: body-frame Jacobian of the forward kinematics function with respect to the strain vector using explicit expression in SE(3).
    """
    
    # Classify the point along the robot to the corresponding segment
    segment_idx, _, l_cum = classify_segment(params, s)
    
    # Indices to extract/inject kappa_z, sigma_x, sigma_y from xi
    SE2_to_SE3_indices = (2, 3, 4)  
    
    # Initial condition
    xi_SE3_0 = vec_SE2_to_xi_SE3(xi[0:3], SE2_to_SE3_indices)
    
    Ad_g0_inv_L0 = Adjoint_gn_SE3_inv(xi_SE3_0, l_cum[0], l_cum[1])
    Ad_g0_inv_s = Adjoint_gn_SE3_inv(xi_SE3_0, l_cum[0], s)
    
    T_g0_L0 = Tangent_gn_SE3(xi_SE3_0, l_cum[0], l_cum[1])
    T_g0_s = Tangent_gn_SE3(xi_SE3_0, l_cum[0], s)

    mat_0_L0 = Ad_g0_inv_L0 @ T_g0_L0
    mat_0_s = Ad_g0_inv_s @ T_g0_s
    
    J_0_L0 = jnp.concatenate([mat_0_L0[None, :, :], jnp.zeros((num_segments - 1, 6, 6))], axis=0)
    J_0_s = jnp.concatenate([mat_0_s[None, :, :], jnp.zeros((num_segments - 1, 6, 6))], axis=0)
    
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
        
        Ad_gi_inv_Li = Adjoint_gn_SE3_inv(xi_SE3_i, l_cum[i], l_cum[i+1])
        Ad_gi_inv_s = Adjoint_gn_SE3_inv(xi_SE3_i, l_cum[i], s)
        
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
    
    # Extracted matrix to switch back from SE(3) to SE(2)
    interest_coordinates = [2, 3, 4]
    interest_strain = [2, 3, 4]
    reordered_lines = [1, 2, 0]
    
    J_local = J_segment_SE3_local[:, interest_coordinates][:, :, interest_strain][:, reordered_lines, :] # shape: (n_segments, 3, 3)
    J_local = blk_concat(J_local) # shape: (n_segments*3, 3)
    J_local = J_local @ B_xi
    
    return J_local

def compare_jacobians(params, s: float, key=random.PRNGKey(0)):
    """
    Compare les jacobiens locaux et globaux calculés via SE(2) et SE(3).
    Affiche les différences numériques.
    
    Args:
        params (dict): paramètres du robot, incluant 'num_segments' et 'L_tot'.
        s (float): position le long de la tige.
        key (jax.random.PRNGKey): clé pour la génération aléatoire.
    """
    global num_segments, B_xi

    # Préparer la taille du vecteur de strain
    dim_se2 = 3 * num_segments

    # Exemple de base de changement B_xi (identité ici, à adapter si besoin)
    B_xi = jnp.eye(dim_se2)

    # Générer un vecteur de strain aléatoire (compatible SE(2))
    xi = random.normal(key, (dim_se2,))

    # Calculer les jacobiens
    J_local_SE2 = J_explicit_local(params, xi, s)
    J_local_SE3 = J_explicit_local_SE3(params, xi, s)
    J_global_SE2 = J_explicit_global(params, xi, s)
    J_global_SE3 = J_explicit_global_SE3(params, xi, s)
    
    # # Afficher les jacobiens
    print("===== JACOBIENS LOCAUX =====")
    print("J_local (SE(2)):")
    print(J_local_SE2)
    print("J_local (SE(3)):")
    print(J_local_SE3)
    # print("===== JACOBIENS GLOBAUX =====")
    print("J_global (SE(2)):")
    print(J_global_SE2)
    print("J_global (SE(3)):")
    print(J_global_SE3)
    print("==================================")

    # Comparer les résultats
    diff_local = jnp.linalg.norm(J_local_SE2 - J_local_SE3)
    diff_global = jnp.linalg.norm(J_global_SE2 - J_global_SE3)

    print("===== COMPARAISON JACOBIENS =====")
    print(f"Norme différence J_local (SE(2) vs SE(3))  : {diff_local:.3e}")
    print(f"Norme différence J_global (SE(2) vs SE(3)) : {diff_global:.3e}")
    print("==================================")

    return {
        "J_local_SE2": J_local_SE2,
        "J_local_SE3": J_local_SE3,
        "J_global_SE2": J_global_SE2,
        "J_global_SE3": J_global_SE3
    }

num_segments = 5  # Nombre de segments du robot
params = {
    'num_segments': 5,
    'l': jnp.ones(num_segments) * 0.2,  # Longueur de chaque segment
    'L_tot': 1.0,  # Longueur totale du robot
    'th0': jnp.array(1)  # Angle initial de chaque segment
}

s = 0.75  # Position d’évaluation
resultats = compare_jacobians(params, s)
