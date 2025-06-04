from jax import Array, jit, lax, vmap
from jax import jacobian, grad
from jax import scipy as jscipy
from jax import numpy as jnp
from quadax import GaussKronrodRule

import numpy as onp
from typing import Callable, Dict, Tuple, Optional

from .utils import (
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

def factory(
    num_segments: int, 
    strain_selector: Array=None,
    xi_eq: Optional[Array] = None,
    stiffness_fn: Optional[Callable] = None,
    actuation_mapping_fn: Optional[Callable] = None,
    global_eps: float = 1e-6, 
    integration_type: str = "gauss-legendre",
    param_integration: int = None,
    jacobian_type: str = "explicit",
    ) -> Tuple[
    Array,
    Callable[[Dict[str, Array], Array, Array], Array],
    Callable[
        [Dict[str, Array], Array, Array],
        Tuple[Array, Array, Array, Array, Array, Array],
    ],
    Dict[str, Callable],
    ]:
    """
    Factory function to create the forward kinematics function for a planar robot.
    This function computes the forward kinematics of a planar robot with a given number of segments.

    Args:
        num_segments (int): number of segments in the robot.
        strain_selector (Array, optional): strain selector array of shape (3 * num_segments, )
            specifying which strain components are active by setting them to True or False. 
            Defaults to None.
        xi_eq (Array, optional): equilibrium strain vector of shape (3 * num_segments, ). 
            Defaults to 1 for the axial strain and 0 for the bending and shear strains.
        stiffness_fn (Callable, optional): function to compute the stiffness matrix. 
            Defaults to None.
        actuation_mapping_fn (Callable, optional): function to compute the actuation mapping. 
            Defaults to None.
        global_eps (float, optional): small number to avoid singularities. 
            Defaults to 1e-6.
        integration_type (str, optional): type of integration to use: "gauss-legendre", "gauss-kronrad" or "trapezoid". 
            Defaults to "gauss-legendre" for Gaussian quadrature.
        param_integration (int, optional): parameter for the integration method. 
            If None, it is set to 30 for Gaussian quadrature and 1000 for trapezoidal integration.
        jacobian_type (str, optional): type of Jacobian to compute: "explicit" or "autodiff".
            Defaults to "explicit" for explicit Jacobian computation. 

    Returns:
        Callable: forward kinematics function that takes in parameters and configuration vector
            and returns the pose of the robot at a given point along its length.
    """
    
    # =======================================================================================================================
    # Initialize parameters if not provided
    # ====================================================
    # Number of segments
    if not isinstance(num_segments, int):
        raise ValueError(f"num_segments must be an integer, but got {type(num_segments)}")
    if num_segments < 1:
        raise ValueError(f"num_segments must be greater than 0, but got {num_segments}")
    
    # Max number of degrees of freedom = size of the strain vector
    n_xi = 3 * num_segments
    
    # Strain basis matrix
    if strain_selector is None:
        # activate all strains (i.e. bending, shear, and axial)
        strain_selector = jnp.ones((n_xi,), dtype=bool)
    if not isinstance(strain_selector, jnp.ndarray):
        if isinstance(strain_selector, list):
            strain_selector = jnp.array(strain_selector)
        else:
            raise TypeError(f"strain_selector must be a jnp.ndarray, but got {type(strain_selector).__name__}")
    strain_selector = strain_selector.flatten()
    if strain_selector.shape[0] != n_xi:
        raise ValueError(
            f"strain_selector must have the same shape as the strain vector, but got {strain_selector.shape[0]} instead of {n_xi}"
        )
    if not jnp.issubdtype(strain_selector.dtype, jnp.bool_):
        raise TypeError(
            f"strain_selector must be a boolean array, but got {strain_selector.dtype}"
        )
    
    # Rest strain
    if xi_eq is None:
        xi_eq = jnp.zeros((n_xi,))
        # By default, set the axial rest strain (local y-axis) along the entire rod to 1.0
        rest_strain_reshaped = xi_eq.reshape((-1, 3))
        rest_strain_reshaped = rest_strain_reshaped.at[:, -1].set(1.0)
        xi_eq = rest_strain_reshaped.flatten()
    if not isinstance(xi_eq, jnp.ndarray):
        if isinstance(xi_eq, list):
            xi_eq = jnp.array(xi_eq)
        else:
            raise TypeError(f"xi_eq must be a jnp.ndarray, but got {type(xi_eq).__name__}")
    xi_eq = xi_eq.flatten()
    if xi_eq.shape[0] != n_xi:
        raise ValueError(
            f"xi_eq must have the same shape as the strain vector, but got {xi_eq.shape[0]} instead of {n_xi}"
        )
    if not jnp.issubdtype(xi_eq.dtype, jnp.floating):
        if not jnp.issubdtype(xi_eq.dtype, jnp.integer):
            raise TypeError(
                f"xi_eq must be a floating point array, but got {xi_eq.dtype}"
            )
        else:
            xi_eq = xi_eq.astype(jnp.float32)    
    
    # Stiffness function
    compute_stiffness_matrix_for_all_segments_fn = vmap(
        compute_planar_stiffness_matrix
    )
    if stiffness_fn is None:
        def stiffness_fn(
            params: Dict[str, Array], 
            B_xi: Array, 
            formulate_in_strain_space: bool = False
        ) -> Array:
            """
            Compute the stiffness matrix of the system.
            Args:
                params: Dictionary of robot parameters
                B_xi: Strain basis matrix
                formulate_in_strain_space: 
                    whether to formulate the elastic matrix in the strain space
            Returns:
                S: elastic matrix of shape (n_q, n_q) if formulate_in_strain_space is False or (n_xi, n_xi) otherwise
            """
            # length of the segments
            l = params["l"]
            # cross-sectional area and second moment of area
            A = jnp.pi * params["r"] ** 2
            Ib = A**2 / (4 * jnp.pi)

            # elastic and shear modulus
            E, G = params["E"], params["G"]
            # stiffness matrix of shape (num_segments, 3, 3)
            S_sms = compute_stiffness_matrix_for_all_segments_fn(l, A, Ib, E, G)
            # we define the elastic matrix of shape (n_xi, n_xi) as K(xi) = K @ xi where K is equal to
            S = blk_diag(S_sms)

            if not formulate_in_strain_space:
                S = B_xi.T @ S @ B_xi

            return S
    if not isinstance(stiffness_fn, Callable):
        raise TypeError(f"stiffness_fn must be a callable, but got {type(stiffness_fn).__name__}")

    # Actuation mapping function
    if actuation_mapping_fn is None: 
        def actuation_mapping_fn(
            forward_kinematics_fn: Callable,
            jacobian_fn: Callable,
            params: Dict[str, Array],
            B_xi: Array,
            q: Array,
        ) -> Array:
            """
            Returns the actuation matrix that maps the actuation space to the configuration space.
            Assumes the fully actuated and identity actuation matrix.
            Args:
                forward_kinematics_fn: function to compute the forward kinematics
                jacobian_fn: function to compute the Jacobian
                params: dictionary with robot parameters
                B_xi: strain basis matrix
                q: configuration of the robot
            Returns:
                A: actuation matrix of shape (n_xi, n_xi) where n_xi is the number of strains.
            """
            A = B_xi.T @ jnp.identity(n_xi) @ B_xi

            return A
    if not isinstance(actuation_mapping_fn, Callable):
        raise TypeError(f"actuation_mapping_fn must be a callable, but got {type(actuation_mapping_fn).__name__}")
    
    if integration_type == "gauss-legendre":
        if param_integration is None:
            param_integration = 30
    elif integration_type == "gauss-kronrad":
        if param_integration is None:
            param_integration = 15
        if param_integration not in [15, 21, 31, 41, 51, 61]:
            raise ValueError(
                f"param_integration must be one of [15, 21, 31, 41, 51, 61] for gauss-kronrad integration, but got {param_integration}"
            )
    elif integration_type == "trapezoid":
        if param_integration is None:
            param_integration = 1000
    else:
        raise ValueError(f"integration_type must be either 'gauss-legendre', 'gauss-kronrad' or 'trapezoid', but got {integration_type}")
    
    if jacobian_type not in ["explicit", "autodiff"]:
        raise ValueError(f"jacobian_type must be either 'explicit' or 'autodiff', but got {jacobian_type}")
    
    # =======================================================================================================================
    # Define the functions
    # ====================================================
    
    # Compute the strain basis matrix
    B_xi = compute_strain_basis(strain_selector)
    
    @jit
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
    
    @jit
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
        segment_idx, s_local, _ = classify_segment(params, s)
            
        chi_O = jnp.array([0.0, 0.0, th0])  # Initial pose of the robot

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

    @jit
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
    
    @jit
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
        J = jacobian(lambda xi: chi_fn_xi(params, xi, s))(xi)

        # apply the strain basis to the Jacobian
        J = J @ B_xi

        return J

    @jit
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
            Array: Jacobian of the forward kinematics function with respect to the strain vector using explicit expression in SE(3).
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

    @jit
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
        
        # Compute the Jacobian of chi_fn with respect to xi
        J = jacobian(lambda xi: chi_fn_xi(params, xi, s))(xi)
        
        dJ_dxi = jacobian(J)(xi)  
        J_d = jnp.tensordot(dJ_dxi, xi_d, axes=([2], [0]))

        # apply the strain basis to the Jacobian
        J = J @ B_xi
        
        # apply the strain basis to the time-derivative of the Jacobian
        J_d = J_d @ B_xi

        return J, J_d

    @jit
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

    if jacobian_type == "explicit":
        jacobian_fn_xi = J_explicit
        J_Jd = J_Jd_explicit
    elif jacobian_type == "autodiff":
        jacobian_fn_xi = J_autodiff
        J_Jd = J_Jd_autodiff

    @jit
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
                
        # Add a small number to the bending strain to avoid singularities
        xi = apply_eps_to_bend_strains(xi, eps)

        # Compute the Jacobian of chi_fn with respect to xi
        J = jacobian_fn_xi(params, xi, s)
        return J

    @jit
    def B_fn_xi(
        params: Dict[str, Array], 
        xi: Array
    ) -> Array:
        """
        Compute the mass / inertia matrix of the robot.

        Args:
            params (Dict[str, Array]): dictionary of robot parameters.
            xi (Array): strain vector of the robot.

        Returns:
            Array: mass / inertia matrix of the robot.
        """
        # Extract the parameters
        rho = params["rho"] # density of each segment [kg/m^3]
        l = params["l"] # length of each segment [m]
        r = params["r"] # radius of each segment [m]
        
        # Usefull derived quantities
        A = jnp.pi * r**2  # cross-sectional area of each segment [m^2]
        Ib = A**2 / (4 * jnp.pi) # second moment of area of each segment [m^4]

        l_cum = jnp.cumsum(jnp.concatenate([jnp.array([0.0]), l]))
        # Compute each integral
        def compute_integral(i):
            if integration_type == "gauss-legendre":
                Xs, Ws, nGauss = gauss_quadrature(N_GQ=param_integration, a=l_cum[i], b=l_cum[i + 1])
                
                J_all = vmap(lambda s: jacobian_fn_xi(params, xi, s))(Xs)
                Jp_all = J_all[:, :2, :]
                Jo_all = J_all[:, 2:, :]

                integrand_JpT_Jp = jnp.einsum("nij,nik->njk", Jp_all, Jp_all)
                integrand_JoT_Jo = jnp.einsum("nij,nik->njk", Jo_all, Jo_all)
                
                integral_Jp = jnp.sum(Ws[:, None, None] * integrand_JpT_Jp, axis=0)
                integral_Jo = jnp.sum(Ws[:, None, None] * integrand_JoT_Jo, axis=0)
                
                integral = rho[i] * A[i] * integral_Jp + rho[i] * Ib[i] * integral_Jo
                
            elif integration_type == "gauss-kronrad":
                rule = GaussKronrodRule(order=param_integration)
                def integrand(s):
                    J = jacobian_fn_xi(params, xi, s)
                    Jp = J[:2, :]
                    Jo = J[2:, :]
                    return rho[i] * A[i] * Jp.T @ Jp + rho[i] * Ib[i] * Jo.T @ Jo

                integral, _, _, _ = rule.integrate(integrand, l_cum[i], l_cum[i+1], args=())

            elif integration_type == "trapezoid":
                xs = jnp.linspace(l_cum[i], l_cum[i + 1], param_integration)
                
                J_all = vmap(lambda s: jacobian_fn_xi(params, xi, s))(xs)
                Jp_all = J_all[:, :2, :]
                Jo_all = J_all[:, 2:, :]

                integrand_JpT_Jp = jnp.einsum("nij,nik->njk", Jp_all, Jp_all)
                integrand_JoT_Jo = jnp.einsum("nij,nik->njk", Jo_all, Jo_all)
                
                integral_Jp = jscipy.integrate.trapezoid(integrand_JpT_Jp, x=xs, axis=0)
                integral_Jo = jscipy.integrate.trapezoid(integrand_JoT_Jo, x=xs, axis=0) 
                
                integral = rho[i] * A[i] * integral_Jp + rho[i] * Ib[i] * integral_Jo
                
            return integral
        
        # Compute the cumulative integral
        indices = jnp.arange(num_segments) 
        integrals = vmap(compute_integral)(indices)
        
        B = jnp.sum(integrals, axis=0)
        
        return B
        
    @jit
    def B_C_fn_xi(
        params: Dict[str, Array], 
        xi: Array, 
        xi_d: Array
    ) -> Tuple[Array, Array]:
        """
        Compute the mass / inertia matrix of the robot and the Coriolis / centrifugal matrix of the robot.

        Args:
            params (Dict[str, Array]): dictionary of robot parameters.
            xi (Array): strain vector of the robot.
            xi_d (Array): velocity vector of the robot.

        Returns:
            Tuple[Array, Array]: 
                - B (Array): mass / inertia matrix of the robot.
                - C (Array): Coriolis / centrifugal matrix of the robot.
        """

        # Compute the mass / inertia matrix
        B = B_fn_xi(params, xi)

        # Compute the Christoffel symbols
        def christoffel_symbol(i, j, k):
            return 0.5 * (
                grad(lambda x: B[i, j])(xi)[k]
                + grad(lambda x: B[i, k])(xi)[j]
                - grad(lambda x: B[j, k])(xi)[i]
            )

        # Compute the Coriolis / centrifugal matrix
        C = jnp.zeros_like(B)
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                for k in range(B.shape[0]):
                    C = C.at[i, j].add(christoffel_symbol(i, j, k) * xi_d[k])

        return B, C
    
    @jit 
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
            if integration_type == "gauss-legendre":
                Xs, Ws, nGauss = gauss_quadrature(N_GQ=param_integration, a=l_cum[i], b=l_cum[i + 1])
                chi_s = vmap(lambda s: chi_fn_xi(params, xi, s))(Xs)
                p_s = chi_s[:, :2]
                integrand = -rho[i] * A[i] * jnp.einsum("ij,j->i", p_s, g)
                
                # Compute the integral
                integral = jnp.sum(Ws * integrand)
                
            elif integration_type == "gauss-kronrad":
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
    
    @jit 
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
        
        G = jacobian(lambda xi: U_g_fn_xi(params, xi))(xi)
        return G
    
    @jit 
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
            if integration_type == "gauss-legendre":
                Xs, Ws, nGauss = gauss_quadrature(N_GQ=param_integration, a=l_cum[i], b=l_cum[i + 1])
                
                J_all = vmap(lambda s: J_explicit(params, xi, s))(Xs)
                Jp_all = J_all[:, :2, :] # shape: (nGauss, n_segments, 3, 3)
                
                # Compute the integrand
                integrand = -rho[i] * A[i] * jnp.einsum("ijk,j->ik", Jp_all, g)
                
                # Multiply each element of integrand by the corresponding weight
                weighted_integrand = jnp.einsum("i, ij->ij", Ws, integrand)

                # Compute the integral
                integral = jnp.sum(weighted_integrand, axis=0)  # sum over the Gauss points
                
            elif integration_type == "gauss-kronrad":
                rule = GaussKronrodRule(order=param_integration)
                def integrand(s):
                    J = J_explicit(params, xi, s)
                    Jp = J[:2, :]
                    return -rho[i] * A[i] * jnp.dot(g, Jp)

                # Compute the integral
                integral, _, _, _ = rule.integrate(integrand, l_cum[i], l_cum[i+1], args=())
                
            elif integration_type == "trapezoid":
                xs = jnp.linspace(l_cum[i], l_cum[i + 1], param_integration)
                
                J_all = vmap(lambda s: J_explicit(params, xi, s))(xs)
                Jp_all = J_all[:, :2, :] # shape: (nGauss, n_segments, 3, 3)
                
                # Compute the integrand
                integrand = -rho[i] * A[i] * jnp.einsum("ijk,j->ik", Jp_all, g)
                
                # Multiply each element of integrand by the corresponding weight
                weighted_integrand = jnp.einsum("i, ij->ij", Ws, integrand)

                # Compute the integral
                integral = jnp.sum(weighted_integrand, axis=0)
            
            return integral
        
        # Compute the cumulative integral
        indices = jnp.arange(num_segments)
        integrals = vmap(compute_integral)(indices)

        G = jnp.sum(integrals, axis=0)  # sum over the segments
        return G
    
    if jacobian_type == "explicit":
        G_fn_xi = G_fn_xi_explicit
    elif jacobian_type == "autodiff":
        G_fn_xi = G_fn_xi_autodiff
    
    @jit
    def dynamical_matrices_fn(
        params: Dict[str, Array], 
        q: Array, 
        q_d: Array, 
        eps: float = 1e4 * global_eps
        ) -> Tuple[Array, Array, Array, Array, Array, Array]:
        """
        Compute the dynamical matrices of the robot.
        
        Args:
            params (Dict[str, Array]): dictionary of robot parameters.
            q (Array): configuration vector of the robot.
            q_d (Array): velocity vector of the robot.
            eps (float, optional): small number to avoid singularities. Defaults to 1e4 * global_eps.
            
        Returns:
            Tuple: 
            - B (Array): mass / inertia matrix of the robot. (shape: (n_q, n_q))
            - C (Array): Coriolis / centrifugal matrix of the robot. (shape: (n_q, n_q))
            - G (Array): gravity vector of the robot. (shape: (n_q,))
            - K (Array): elastic vector of the robot. (shape: (n_q,))
            - D (Array): dissipative matrix of the robot. (shape: (n_q, n_q))
            - alpha (Array): actuation matrix of the robot. (shape: (n_q, n_tau))
        """
        # Map the configuration to the strains
        xi = xi_eq + B_xi @ q
        xi_d = B_xi @ q_d
                
        # Add a small number to the bending strain to avoid singularities
        xi = apply_eps_to_bend_strains(xi, eps)
        
        # Compute the stiffness matrix
        K = stiffness_fn(params, B_xi, formulate_in_strain_space=True)
        # Apply the strain basis to the stiffness matrix
        K = B_xi.T @ K @ (xi - xi_eq) # evaluate K(xi) = K @ xi
        
        # Compute the actuation matrix
        A = actuation_mapping_fn(
            forward_kinematics_fn, jacobian_fn, params, B_xi, q
        )
        # Apply the strain basis to the actuation matrix
        alpha = A
        
        # Dissipative matrix
        D = params.get("D", jnp.zeros((n_xi, n_xi)))
        # Apply the strain basis to the dissipative matrix
        D = B_xi.T @ D @ B_xi
        
        B, C = B_C_fn_xi(params, xi, xi_d)

        G = B_xi.T @ G_fn_xi(params, xi).squeeze()
        
        return B, C, G, K, D, alpha    
    
    @jit
    def kinetic_energy_fn(
        params: Dict[str, Array], 
        q: Array, 
        q_d: Array
        ) -> Array:
        """
        Compute the kinetic energy of the system.
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (n_q, )
            q_d: generalized velocities of shape (n_q, )
        Returns:
            T: kinetic energy of shape ()
        """
        B, _, _, _, _, _ = dynamical_matrices_fn(params, q=q, q_d=q_d)

        # Kinetic energy
        T = (0.5 * q_d.T @ B @ q_d).squeeze()

        return T

    @jit
    def potential_energy_fn(
        params: Dict[str, Array], 
        q: Array, 
        eps: float = 1e4 * global_eps
    ) -> Array:
        """
        Compute the potential energy of the system.
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (n_q, )
            eps: small number to avoid singularities (e.g., division by zero)
        Returns:
            U: potential energy of shape ()
        """
        # map the configuration to the strains
        xi = xi_eq + B_xi @ q
        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains(xi, eps)

        # compute the stiffness matrix
        K = stiffness_fn(params, B_xi, formulate_in_strain_space=True)
        # elastic energy
        U_K = 0.5 * (xi - xi_eq).T @ K @ (xi - xi_eq)  # evaluate K(xi) = K @ xi

        # gravitational potential energy
        U_G = U_g_fn_xi(params, xi_epsed)

        # total potential energy
        U = (U_G + U_K).squeeze()

        return U

    @jit
    def energy_fn(
        params: Dict[str, Array], 
        q: Array, 
        q_d: Array) -> Array:
        """
        Compute the total energy of the system.
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (n_q, )
            q_d: generalized velocities of shape (n_q, )
        Returns:
            E: total energy of shape ()
        """
        T = kinetic_energy_fn(params, q, q_d)
        U = potential_energy_fn(params, q)
        E = T + U

        return E

    @jit
    def operational_space_dynamical_matrices_fn(
        params: Dict[str, Array],
        q: Array,
        q_d: Array,
        s: Array,
        B: Array,
        C: Array,
        operational_space_selector: Tuple = (True, True, True),
        eps: float = 1e4 * global_eps,
    ) -> Tuple[Array, Array, Array, Array, Array]:
        """
        Compute the dynamics in operational space.
        The implementation is based on Chapter 7.8 of "Modelling, Planning and Control of Robotics" by
        Siciliano, Sciavicco, Villani, Oriolo.
        Args:
            params: dictionary of parameters
            q: generalized coordinates of shape (n_q,)
            q_d: generalized velocities of shape (n_q,)
            s: point coordinate along the robot in the interval [0, L].
            B: inertia matrix in the generalized coordinates of shape (n_q, n_q)
            C: coriolis matrix derived with Christoffer symbols in the generalized coordinates of shape (n_q, n_q)
            operational_space_selector: tuple of shape (3,) to select the operational space variables.
                For examples, (True, True, False) selects only the positional components of the operational space.
            eps: small number to avoid singularities (e.g., division by zero)
        Returns:
            Lambda: inertia matrix in the operational space of shape (3, 3)
            mu: matrix with corioli and centrifugal terms in the operational space of shape (3, 3)
            J: Jacobian of the Cartesian pose with respect to the generalized coordinates of shape (3, n_q)
            J_d: time-derivative of the Jacobian of the end-effector pose with respect to the generalized coordinates
                of shape (3, n_q)
            JB_pinv: Dynamically-consistent pseudo-inverse of the Jacobian. Allows the mapping of torques
                from the generalized coordinates to the operational space: f = JB_pinv.T @ tau_q
                Shape (n_q, 3)
        """
        ## map the configuration to the strains
        xi = xi_eq + B_xi @ q
        xi_d = B_xi @ q_d
        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains(xi, eps)

        # classify the point along the robot to the corresponding segment
        segment_idx, s_segment, l_cum = classify_segment(params, s)

        # make operational_space_selector a boolean array
        operational_space_selector = onp.array(operational_space_selector, dtype=bool)

        # Jacobian and its time-derivative
        J, J_d = J_Jd(params, xi_epsed, xi_d, s_segment)
        J = jnp.squeeze(J)
        J_d = jnp.squeeze(J_d)

        J = J[operational_space_selector, :]
        J_d = J_d[operational_space_selector, :]

        # inverse of the inertia matrix in the configuration space
        B_inv = jnp.linalg.inv(B)

        Lambda = jnp.linalg.inv(
            J @ B_inv @ J.T
        )  # inertia matrix in the operational space
        mu = Lambda @ (
            J @ B_inv @ C - J_d
        )  # coriolis and centrifugal matrix in the operational space

        JB_pinv = (
            B_inv @ J.T @ Lambda
        )  # dynamically-consistent pseudo-inverse of the Jacobian

        return Lambda, mu, J, J_d, JB_pinv

    auxiliary_fns = {
        "apply_eps_to_bend_strains": apply_eps_to_bend_strains,
        "classify_segment": classify_segment,
        "stiffness_fn": stiffness_fn,
        "actuation_mapping_fn": actuation_mapping_fn,
        "jacobian_fn": jacobian_fn,
        "kinetic_energy_fn": kinetic_energy_fn,
        "potential_energy_fn": potential_energy_fn,
        "energy_fn": energy_fn,
        "operational_space_dynamical_matrices_fn": operational_space_dynamical_matrices_fn,
    }

    return B_xi, forward_kinematics_fn, dynamical_matrices_fn, auxiliary_fns