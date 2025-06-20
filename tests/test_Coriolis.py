from jax import jit, lax, vmap, Array, grad, jacobian
import jax.numpy as jnp
from jax import scipy as jscipy
from jsrm.math_utils import blk_diag, blk_concat
from quadax import GaussKronrodRule
from jsrm.systems.utils import (
    compute_strain_basis,
    compute_planar_stiffness_matrix,
    gauss_quadrature
)
from jsrm.utils.lie_operators import ( # To use SE(3)
    vec_SE2_to_xi_SE3,
    Tangent_gn_SE3, 
    Adjoint_gn_SE3,
    Adjoint_gn_SE3_inv,
    Adjoint_g_SE3,
    adjoint_SE3,
    adjoint_star_SE3,
)
from jsrm.utils.lie_operators import ( # To use SE(2)
    Tangent_gn_SE2, 
    Adjoint_gn_SE2,
    Adjoint_gn_SE2_inv,
    Adjoint_g_SE2, 
    adjoint_SE2,
    adjoint_star_SE2,
)
from jsrm.utils.lie_operators import (
    compute_weighted_sums,
)
from typing import Callable, Dict, Tuple, Optional
    
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

@jit
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

def J_autodiff(
    params: Dict[str, Array], 
    xi: Array, 
    s: Array
) -> Array:
    """
    Compute the Jacobian of the forward kinematics function with respect to the strain vector.

    Args:
        params (Dict[str, Array]): dictionary of robot parameters.
        xi (Array): strain vector of the robot.
        s (Array): point coordinate along the robot in the interval [0, L].

    Returns:
        Array: Jacobian of the forward kinematics function with respect to the strain vector.
    """
    # Compute the Jacobian of chi_fn with respect to xi
    J = jacobian(lambda _xi: chi_fn(params, _xi, s))(xi)

    return J

def J_Jd_explicit(
    params: Dict[str, Array],
    xi: Array,
    xi_d: Array,
    s: Array
) -> Array:
    #TODO correct
    """
    Compute the Jacobian and its derivative with respect to the strain vector at a given point s.

    Args:
        params (Dict[str, Array]): dictionary of robot parameters.
        xi (Array): strain vector of the robot.
        xi_d (Array): velocity vector of the robot.
        s (Array): point coordinate along the robot in the interval [0, L].

    Returns:
        Tuple[Array, Array]: 
            - J (Array): Jacobian of the forward kinematics function.
            - J_d (Array): time-derivative of the Jacobian of the forward kinematics function.
    """
    # Classify the point along the robot to the corresponding segment
    segment_idx, _, l_cum = classify_segment(params, s)
    
    # Indices to extract/inject kappa_z, sigma_x, sigma_y from xi
    SE2_to_SE3_indices = (2, 3, 4)

    # =================================
    # Computation of the Jacobian
    
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
    
    # =================================
    # Computation of the time-derivative of the Jacobian
        
    idx_range = jnp.arange(num_segments)
    
    xi_d_i = vmap(
        lambda _i: lax.dynamic_slice(xi_d, (3 * _i,), (3,))
    )(idx_range) # shape: (num_segments, 3)
    xi_d_SE3_i = vmap(
        lambda _xi_d_i: vec_SE2_to_xi_SE3(_xi_d_i, SE2_to_SE3_indices).squeeze()
    )(xi_d_i) # shape: (num_segments, 6)
    S_i = vmap(
        lambda _i: lax.dynamic_index_in_dim(J_segment_SE3_local, _i, axis=0, keepdims=False)
    )(idx_range) # shape: (num_segments, 6, 6)
    sum_Sj_xi_d_j = compute_weighted_sums(J_segment_SE3_local, xi_d_SE3_i, num_segments) # shape: (num_segments, 6)
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
    J_d_segment_SE3_global = jnp.einsum('ij,njk->nik', Adjoint_g_s, J_d_segment_SE3_local)  # shape: (n_segments, 6, 6)
    
    # =================================
    # Switch back from SE(3) to SE(2)
    
    # Extracted matrix 
    interest_coordinates = [2, 3, 4]
    interest_strain = [2, 3, 4]
    reordered_lines = [1, 2, 0]
    
    J_local = J_segment_SE3_local[:, interest_coordinates][:, :, interest_strain][:, reordered_lines, :] # shape: (n_segments, 3, 3)
    J_local = blk_concat(J_local) # shape: (n_segments*3, 3)    
    
    J_d_local = J_d_segment_SE3_local[:, interest_coordinates][:, :, interest_strain][:, reordered_lines, :] # shape: (n_segments, 3, 3)
    J_d_local = blk_concat(J_d_local) # shape: (n_segments*3, 3)
    
    J_global = J_segment_SE3_global[:, interest_coordinates][:, :, interest_strain][:, reordered_lines, :] # shape: (n_segments, 3, 3)
    J_global = blk_concat(J_global) # shape: (n_segments*3, 3)    
    
    J_d_global = J_d_segment_SE3_global[:, interest_coordinates][:, :, interest_strain][:, reordered_lines, :] # shape: (n_segments, 3, 3)
    J_d_global = blk_concat(J_d_global) # shape: (n_segments*3, 3)
    
    # # Apply the strain basis to the Jacobian and its time-derivative
    # J_global = J_global @ B_xi    
    # J_d_global = J_d_global @ B_xi
    
    # return J_global, J_d_global
    
    
    return J_local, J_d_local
    
def compute_Coriolis_matrix(params, xi, xi_d, integration_type="gauss-legendre", param_integration=5):
    """
    Compute the mass / inertia matrix of the robot.

    Args:
        params (Dict[str, Array]): dictionary of robot parameters.
        xi (Array): strain vector of the robot.

    Returns:
        Array: mass / inertia matrix of the robot.
    """
    
    
    print("========== C ==========")
    # Extract the parameters
    rho = params["rho"] # density of each segment [kg/m^3]
    l = params["l"] # length of each segment [m]
    r = params["r"] # radius of each segment [m]
    
    # Usefull derived quantities
    A = jnp.pi * r**2  # cross-sectional area of each segment [m^2]
    Ib = A**2 / (4 * jnp.pi) # second moment of area of each segment [m^4]

    l_cum = jnp.cumsum(jnp.concatenate([jnp.zeros(1), l]))
    
    # ==================================================================
    # B Inertia matrix
    # ===================================
    # Compute each integral
    def compute_integral_B(i):
        if integration_type == "gauss-legendre":
            Xs, Ws, nGauss = gauss_quadrature(N_GQ=param_integration, a=l_cum[i], b=l_cum[i + 1])
            
            J_all = vmap(lambda s: J_Jd_explicit(params, xi, xi_d, s)[0])(Xs)
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
                J = J_Jd_explicit(params, xi, xi_d, s)[0]
                Jp = J[:2, :]
                Jo = J[2:, :]
                return rho[i] * A[i] * Jp.T @ Jp + rho[i] * Ib[i] * Jo.T @ Jo

            integral, _, _, _ = rule.integrate(integrand, l_cum[i], l_cum[i+1], args=())

        elif integration_type == "trapezoid":
            xs = jnp.linspace(l_cum[i], l_cum[i + 1], param_integration)
            
            J_all = vmap(lambda s: J_Jd_explicit(params, xi, xi_d, s)[0])(xs)
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
    integrals = vmap(compute_integral_B)(indices)
    
    B1 = jnp.sum(integrals, axis=0)
    print("B1 explicit:\n", B1)

    reordered_lines_return = jnp.array([2, 0, 1])
    # ==================================================================
    # C Coriolis matrix
    # ===================================    
    # Compute each integral
    def compute_integral_C(i):
        if integration_type == "gauss-legendre":
            Xs, Ws, nGauss = gauss_quadrature(N_GQ=param_integration, a=l_cum[i], b=l_cum[i + 1])
            
            J_all, J_d_all = vmap(lambda s: J_Jd_explicit(params, xi, xi_d, s))(Xs) #[[Jp1],[Jp2],[Jo]]
            J_all = J_all[:, reordered_lines_return, :]                             #[[Jo],[Jp1],[Jp2]]
            J_d_all = J_d_all[:, reordered_lines_return, :]                         #[[Jo_d],[Jp1_d],[Jp2_d]]
            M_a = rho[i] * jnp.diag(jnp.array([Ib[i], A[i], A[i]]))                 #[[O],[P],[P]]
            
            integrand_C = vmap(
                lambda _J_i, _J_d_i: (
                    _J_i.T @ (adjoint_star_SE2((_J_i@xi_d)) @ M_a @ _J_i + M_a @ _J_d_i)
                )
            )(J_all, J_d_all)
            
            integral_C = jnp.sum(Ws[:, None, None] * integrand_C, axis=0)
            
        return integral_C
    
    # Compute the cumulative integral
    indices = jnp.arange(num_segments) 
    integrals = vmap(compute_integral_C)(indices)
    
    C1 = jnp.sum(integrals, axis=0)
    print("C1 explicit GVS:")
    # print(C1)
    print(C1@xi_d)
    
    # Not using J_d
    for n in range(num_segments):
        for m in range(num_segments):
            if m > n:
                continue
            C_block = jnp.zeros((3, 3))

            for i in range(max(n, m), num_segments):
                if integration_type == "gauss-legendre":
                    Xs, Ws, _ = gauss_quadrature(N_GQ=param_integration, a=l_cum[i], b=l_cum[i + 1])

                    def integrand_C(X):
                        J, _ = J_Jd_explicit(params, xi, xi_d, X)
                        J = J[reordered_lines_return, :]  # Reorder to [[Jo], [Jp1], [Jp2]] to be able to multiply by xi_d
                        S_n = J[:, 3*n:3*(n+1)]
                        S_m = J[:, 3*m:3*(m+1)]
                        M_a = rho[i] * jnp.diag(jnp.array([Ib[i], A[i], A[i]]))
                        sum_xi_S = sum(
                             J[:, 3*j:3*(j+1)] @ xi_d[3*j:3*(j+1)] for j in range(m, i+1)
                        )

                        return S_n.T @ (adjoint_star_SE2(J @ xi_d) @ M_a - M_a @ adjoint_SE2(sum_xi_S) ) @ S_m
                        
                    integrand_C_vals = vmap(integrand_C)(Xs)

                    C_block += jnp.sum(Ws[:, None, None] * integrand_C_vals, axis=0)

            C1 = C1.at[3*n:3*(n+1), 3*m:3*(m+1)].set(C_block)
    print("C1 explicit Discrete Cosserat:")
    # print(C1)
    print(C1@xi_d)
    
    def B_autodiff_fn(
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

        l_cum = jnp.cumsum(jnp.concatenate([jnp.zeros(1), l]))
        # Compute each integral
        def compute_integral(i):
            if integration_type == "gauss-legendre":
                Xs, Ws, nGauss = gauss_quadrature(N_GQ=param_integration, a=l_cum[i], b=l_cum[i + 1])
                
                J_all = vmap(lambda s: J_autodiff(params, xi, s))(Xs)
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
                    J = J_autodiff(params, xi, s)
                    Jp = J[:2, :]
                    Jo = J[2:, :]
                    return rho[i] * A[i] * Jp.T @ Jp + rho[i] * Ib[i] * Jo.T @ Jo

                integral, _, _, _ = rule.integrate(integrand, l_cum[i], l_cum[i+1], args=())

            elif integration_type == "trapezoid":
                xs = jnp.linspace(l_cum[i], l_cum[i + 1], param_integration)
                
                J_all = vmap(lambda s: J_autodiff(params, xi, s))(xs)
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
    
    # def B_explicit_fn(
    #     params: Dict[str, Array], 
    #     xi: Array
    # ) -> Array:
    #     """
    #     Compute the mass / inertia matrix of the robot.

    #     Args:
    #         params (Dict[str, Array]): dictionary of robot parameters.
    #         xi (Array): strain vector of the robot.

    #     Returns:
    #         Array: mass / inertia matrix of the robot.
    #     """
    #     # Extract the parameters
    #     rho = params["rho"] # density of each segment [kg/m^3]
    #     l = params["l"] # length of each segment [m]
    #     r = params["r"] # radius of each segment [m]
        
    #     # Usefull derived quantities
    #     A = jnp.pi * r**2  # cross-sectional area of each segment [m^2]
    #     Ib = A**2 / (4 * jnp.pi) # second moment of area of each segment [m^4]

    #     l_cum = jnp.cumsum(jnp.concatenate([jnp.zeros(1), l]))
    #     # Compute each integral
    #     def compute_integral(i):
    #         if integration_type == "gauss-legendre":
    #             Xs, Ws, nGauss = gauss_quadrature(N_GQ=param_integration, a=l_cum[i], b=l_cum[i + 1])
                
    #             J_all = vmap(lambda s: J_Jd_explicit(params, xi, xi_d, s)[0])(Xs)
    #             Jp_all = J_all[:, :2, :]
    #             Jo_all = J_all[:, 2:, :]

    #             integrand_JpT_Jp = jnp.einsum("nij,nik->njk", Jp_all, Jp_all)
    #             integrand_JoT_Jo = jnp.einsum("nij,nik->njk", Jo_all, Jo_all)
                
    #             integral_Jp = jnp.sum(Ws[:, None, None] * integrand_JpT_Jp, axis=0)
    #             integral_Jo = jnp.sum(Ws[:, None, None] * integrand_JoT_Jo, axis=0)
                
    #             integral = rho[i] * A[i] * integral_Jp + rho[i] * Ib[i] * integral_Jo
                
    #         elif integration_type == "gauss-kronrad":
    #             rule = GaussKronrodRule(order=param_integration)
    #             def integrand(s):
    #                 J = J_Jd_explicit(params, xi, xi_d, s)[0]
    #                 Jp = J[:2, :]
    #                 Jo = J[2:, :]
    #                 return rho[i] * A[i] * Jp.T @ Jp + rho[i] * Ib[i] * Jo.T @ Jo

    #             integral, _, _, _ = rule.integrate(integrand, l_cum[i], l_cum[i+1], args=())

    #         elif integration_type == "trapezoid":
    #             xs = jnp.linspace(l_cum[i], l_cum[i + 1], param_integration)
                
    #             J_all = vmap(lambda s: J_Jd_explicit(params, xi, xi_d, s)[0])(xs)
    #             Jp_all = J_all[:, :2, :]
    #             Jo_all = J_all[:, 2:, :]

    #             integrand_JpT_Jp = jnp.einsum("nij,nik->njk", Jp_all, Jp_all)
    #             integrand_JoT_Jo = jnp.einsum("nij,nik->njk", Jo_all, Jo_all)
                
    #             integral_Jp = jscipy.integrate.trapezoid(integrand_JpT_Jp, x=xs, axis=0)
    #             integral_Jo = jscipy.integrate.trapezoid(integrand_JoT_Jo, x=xs, axis=0) 
                
    #             integral = rho[i] * A[i] * integral_Jp + rho[i] * Ib[i] * integral_Jo
                
    #         return integral
        
    #     # Compute the cumulative integral
    #     indices = jnp.arange(num_segments) 
    #     integrals = vmap(compute_integral)(indices)
        
    #     B = jnp.sum(integrals, axis=0)
        
    #     return B
    
    # print("==========B Autodiff==========")
    # B2 = B_autodiff_fn(params, xi)
    # print("B2: \n", B2)
    # print("==========B Explicit==========")
    # B3 = B_explicit_fn(params, xi)
    # print("B3: \n", B3)
    # print("=======AUTODIFF ON EXPLICIT==========")
    
    # def christoffel_fn(i, j, k):
    #     dB_ij = grad(lambda x: B_explicit_fn(params, x)[i, j])(xi)[k]
    #     dB_ik = grad(lambda x: B_explicit_fn(params, x)[i, k])(xi)[j]
    #     dB_jk = grad(lambda x: B_explicit_fn(params, x)[j, k])(xi)[i]
    #     return 0.5 * (dB_ij + dB_ik - dB_jk)
    
    # n = B1.shape[0]  # Number of segments * 3 (for SE(2) coordinates)
    # # ===========================================
    # # For loop without LAX version
    # C2 = jnp.zeros_like(B2)  # Initialize the Coriolis matrix
    # for i in range(n):
    #     for j in range(n):
    #         for k in range(n):
    #             christoffel_symbol = christoffel_fn(i, j, k)
    #             C2 = C2.at[i,j].add(christoffel_symbol * xi_d[k])
    # print("C2 for loop without LAX :\n", C2)
    
    # # ===========================================
    # # For loop with LAX version
    # C2 = jnp.zeros_like(B2)
    # def body_i(i, C2):
    #     def body_j(j, C2):
    #         def body_k(k, acc):
    #             christoffel_symbol = christoffel_fn(i, j, k)
    #             coeff = christoffel_symbol * xi_d[k]
    #             return acc + coeff
    #         C_ij = lax.fori_loop(0, n, body_k, 0.0)
    #         return C2.at[i, j].set(C_ij)
    #     return lax.fori_loop(0, n, body_j, C2)
    # C2 = lax.fori_loop(0, n, body_i, C2)
    # print("C2 for loop with LAX:\n", C2)
    
    # # =========================================== To complete TODO
    # # Vmap version
    # def C_ij(i, j):
    #     cs_k = vmap(lambda k: christoffel_fn(i, j, k))(jnp.arange(n))
    #     return jnp.dot(cs_k, xi_d)
    # C2 = vmap(lambda i: vmap(lambda j: C_ij(i, j))(jnp.arange(n)))(jnp.arange(n))
    # print("C2 vmap version:\n", C2)
    
    # # =========================================== To complete TODO
    # # LAX map version
    # def C_ij(i, j, xi_d, n):
    #     cs_k = lax.map(lambda k: christoffel_fn(i, j, k), jnp.arange(n))
    #     return jnp.dot(cs_k, xi_d)
    # C2 = jnp.stack(
    #         jnp.stack(
    #             lax.map(
    #                 lambda i: lax.map(
    #                     lambda j: C_ij(i, j, xi_d, n),
    #                     jnp.arange(n)
    #                 ),
    #                 jnp.arange(n))
    #             )
    #         )
    # print("C2 LAX map version:\n", C2)
    
    # print("=======AUTODIFF ON AUTODIFF==========")
    
    def christoffel_fn(i, j, k):
        dB_ij = grad(lambda x: B_autodiff_fn(params, x)[i, j])(xi)[k]
        dB_ik = grad(lambda x: B_autodiff_fn(params, x)[i, k])(xi)[j]
        dB_jk = grad(lambda x: B_autodiff_fn(params, x)[j, k])(xi)[i]
        return 0.5 * (dB_ij + dB_ik - dB_jk)
    n = B1.shape[0]  # Number of segments * 3 (for SE(2) coordinates)
    # # ===========================================
    # # For loop without LAX version
    # C3 = jnp.zeros_like(B3)  # Initialize the Coriolis matrix
    # for i in range(n):
    #     for j in range(n):
    #         for k in range(n):
    #             C3 = C3.at[i,j].add(christoffel_fn(i, j, k)*xi_d[k])
    # print("C3 for loop without LAX :\n", C3)
    
    # # ===========================================
    # # For loop with LAX version
    # C3 = jnp.zeros_like(B3)
    # def body_i(i, C3):
    #     def body_j(j, C3):
    #         def body_k(k, acc):
    #             christoffel_symbol = christoffel_fn(i, j, k)
    #             coeff = christoffel_symbol * xi_d[k]
    #             return acc + coeff
    #         C_ij = lax.fori_loop(0, n, body_k, 0.0)
    #         return C3.at[i, j].set(C_ij)
    #     return lax.fori_loop(0, n, body_j, C3)
    # C3 = lax.fori_loop(0, n, body_i, C3)
    # print("C3 for loop with LAX:\n", C3)
    
    # =========================================== To complete TODO
    # Vmap version
    def C_ij(i, j):
        cs_k = vmap(lambda k: christoffel_fn(i, j, k))(jnp.arange(n))
        return jnp.dot(cs_k, xi_d)
    C3 = vmap(lambda i: vmap(lambda j: C_ij(i, j))(jnp.arange(n)))(jnp.arange(n))
    print("C3 vmap version:")
    # print(C3)
    print(C3@xi_d)
    
    # # =========================================== To complete TODO
    # # LAX map version
    # def C_ij(i, j, xi_d, n):
    #     cs_k = lax.map(lambda k: christoffel_fn(i, j, k), jnp.arange(n))
    #     return jnp.dot(cs_k, xi_d)
    # C3 = jnp.stack(
    #         jnp.stack(
    #             lax.map(
    #                 lambda i: lax.map(
    #                     lambda j: C_ij(i, j, xi_d, n),
    #                     jnp.arange(n)
    #                 ),
    #                 jnp.arange(n))
    #             )
    #         )
    # print("C3 LAX map version:\n", C3)
    
    return None

num_segments = 2  # Nombre de segments du robot
params = {
    'l': jnp.ones(num_segments) * 0.2,  # Longueur de chaque segment
    'r': jnp.ones(num_segments) * 0.01,  # Rayon de chaque segment
    'th0': jnp.array(1),  # Angle initial de chaque segment
    'rho': jnp.ones(num_segments) * 1000,  # Densité de chaque segment [kg/m^3]
}

xi = jnp.array([0.1, 0.2, 0.3]*num_segments)  # Vecteur de déformation
xi_d = jnp.array([0.01, 0.02, 0.03]*num_segments)  # Vecteur de vitesse
compute_Coriolis_matrix(params, xi, xi_d, integration_type="gauss-legendre", param_integration=5)
compute_Coriolis_matrix(params, xi, xi_d, integration_type="gauss-legendre", param_integration=5)