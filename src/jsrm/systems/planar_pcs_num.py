import dill
from jax import Array, jit, lax, vmap
from jax import jacobian, grad
from jax import scipy as jscipy
from jax import numpy as jnp
import numpy as onp
import sympy as sp
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union, Optional

from .utils import (
    compute_strain_basis,
    compute_planar_stiffness_matrix,
    gauss_quadrature
)
from jsrm.math_utils import blk_diag

def factory(
    num_segments: int, 
    strain_selector: Array=None,
    xi_eq: Optional[Array] = None,
    stiffness_fn: Optional[Callable] = None,
    actuation_mapping_fn: Optional[Callable] = None,
    global_eps: float = 1e-6, 
    integration_type: str = "gauss",
    param_integration: int = None
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
            specifying which strain components are active by setting them to True or False. Defaults to None.
        xi_eq (Array, optional): equilibrium strain vector of shape (3 * num_segments, ). Defaults to 1 for the axial strain and 0 for the bending and shear strains.
        stiffness_fn (Callable, optional): function to compute the stiffness matrix. Defaults to None.
        actuation_mapping_fn (Callable, optional): function to compute the actuation mapping. Defaults to None.
        global_eps (float, optional): small number to avoid singularities. Defaults to 1e-6.
        integration_type (str, optional): type of integration to use: "gauss" or "trapezoid". Defaults to "gauss" for Gaussian quadrature. 
        param_integration (int, optional): parameter for the integration method. If None, it is set to 30 for Gaussian quadrature and 1000 for trapezoidal integration.

    Returns:
        Callable: forward kinematics function that takes in parameters and configuration vector
            and returns the pose of the robot at a given point along its length.
    """
    
    n_xi = 3 * num_segments  # number of degrees of freedom
    
    # =======================================================================================================================
    # Initialize parameters if not provided
    # ====================================================
    # Strain basis matrix
    if strain_selector is None:
        strain_selector = jnp.ones((n_xi,), dtype=bool)
    else:
        assert strain_selector.shape == (n_xi,)
        
    # Rest strain
    if xi_eq is None:
        xi_eq = jnp.zeros((n_xi,))
        # By default, set the axial rest strain (local y-axis) along the entire rod to 1.0
        rest_strain_reshaped = xi_eq.reshape((-1, 3))
        rest_strain_reshaped = rest_strain_reshaped.at[:, -1].set(1.0)
        xi_eq = rest_strain_reshaped.flatten()
    else:
        assert xi_eq.shape == (n_xi,)
    
    # Stiffness function
    compute_stiffness_matrix_for_all_segments_fn = vmap(
        compute_planar_stiffness_matrix
    )
    if stiffness_fn is None:
        def stiffness_fn(
            params: Dict[str, Array], B_xi: Array, formulate_in_strain_space: bool = False
        ) -> Array:
            """
            Compute the stiffness matrix of the system.
            Args:
                params: Dictionary of robot parameters
                B_xi: Strain basis matrix
                formulate_in_strain_space: whether to formulate the elastic matrix in the strain space
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
    
    if integration_type == "gauss":
        if param_integration is None:
            param_integration = 30
    elif integration_type == "trapezoid":
        if param_integration is None:
            param_integration = 1000
    else:
        raise ValueError("integration_type must be either 'gauss' or 'trapezoid'")
    
    # =======================================================================================================================
    # Define the functions
    # ====================================================
    
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
        
        # cumsum of the segment lengths
        l_cum = jnp.cumsum(params["l"])
        # add zero to the beginning of the array
        l_cum_padded = jnp.concatenate([jnp.array([0.0]), l_cum], axis=0)
        # determine in which segment the point is located
        # use argmax to find the last index where the condition is true
        segment_idx = (
            l_cum.shape[0] - 1 - jnp.argmax((s >= l_cum_padded[:-1])[::-1]).astype(int)
        )
        # point coordinate along the segment in the interval [0, l_segment]
        s_segment = s - l_cum_padded[segment_idx]

        return segment_idx, s_segment
    
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
        segment_idx, s_segment = classify_segment(params, s)
            
        chi_O = jnp.array([0.0, 0.0, th0])  # Initial pose of the robot
        
        # Iteration function
        def chi_i(
            i: int, 
            chi_prev: Array
        ) -> Array:
            th_prev = chi_prev[2]  # Extract previous orientation angle from val
            p_prev = chi_prev[:2]  # Extract previous position from val
            
            # Extract strains for the current segment
            kappa = xi[3 * i]
            sigma_x = xi[3 * i + 1]
            sigma_y = xi[3 * i + 2]

            # Compute the orientation angle for the current segment
            dth = lax.cond(i == segment_idx,
                           lambda : (s_segment * kappa).reshape(()),
                           lambda : (l[i] * kappa).reshape(())
                           )
            th = th_prev + dth

            # Compute the integrals for the transformation matrix
            int_cos_th = lax.cond(kappa == 0, 
                lambda : lax.cond(i==segment_idx, 
                                    lambda : (-s_segment * jnp.sin(th_prev)).reshape(()), 
                                    lambda : (-l[i] * jnp.sin(th_prev)).reshape(())
                                    ), 
                lambda : ((jnp.sin(th) - jnp.sin(th_prev)) / kappa).reshape(())
                )
            int_sin_th = lax.cond(kappa == 0,
                lambda : lax.cond(i==segment_idx, 
                                    lambda : (s_segment * jnp.cos(th_prev)).reshape(()), 
                                    lambda : (l[i] * jnp.cos(th_prev)).reshape(())
                                    ), 
                lambda : ((jnp.cos(th_prev) - jnp.cos(th)) / kappa).reshape(())
                )
            
            # Transformation matrix
            R = jnp.array([[int_cos_th, -int_sin_th],
                   [int_sin_th, int_cos_th]])

            # Compute the position
            p = p_prev + (R @ jnp.array([sigma_x, sigma_y]).T).T
            
            return jnp.concatenate([p, jnp.array([th])])
        
        def chi_scan_body(carry, i):
            chi = chi_i(i, carry)
            output = chi
            return chi_i(i, carry), output

        # Currently: Scan too many elements to calculate i = segment_idx in particular, 
        # without having to pass segment_idx as an argument (dynamic). (JAX does not support dynamic shapes).
        # TODO: check if there is a way to avoid this.
        chi, chi_list = lax.scan(
            f = chi_scan_body, 
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
    def jacobian_fn_xi(
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
        J = jacobian(lambda xi: chi_fn_xi(params, xi, s))(xi)

        # apply the strain basis to the Jacobian
        J = J @ B_xi

        return J
    
    @jit
    def J_Jd(
        params: Dict[str, Array], 
        xi: Array, 
        xi_d: Array,
        s: Array
    ) -> Tuple[Array, Array]:
        """
        Compute the Jacobian of the forward kinematics function and its time-derivative.

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
        
        B_0 = jnp.zeros((n_xi, n_xi)) # Initialize the mass / inertia matrix

        l_succ = jnp.cumsum(jnp.concatenate([jnp.array([0.0]), l]))
        # Iteration function
        def B_i(
            i: int, 
            B_i_prev: Array
        ) -> Array:
            
            if integration_type == "gauss":
                Xs, Ws, nGauss = gauss_quadrature(N_GQ=param_integration, a=l_succ[i], b=l_succ[i + 1])
                
                J_all = vmap(lambda s: jacobian_fn_xi(params, xi, s))(Xs)
                Jp_all = J_all[:, :2, :]
                Jo_all = J_all[:, 2:, :]

                integrand_JpT_Jp = jnp.einsum("nij,nik->njk", Jp_all, Jp_all)
                integrand_JoT_Jo = jnp.einsum("nij,nik->njk", Jo_all, Jo_all)
                
                integral_Jp = jnp.sum(Ws[:, None, None] * integrand_JpT_Jp, axis=0)
                integral_Jo = jnp.sum(Ws[:, None, None] * integrand_JoT_Jo, axis=0)
            else:
                s_values = jnp.linspace(l_succ[i], l_succ[i + 1], param_integration)
                
                J_all = vmap(lambda s: jacobian_fn_xi(params, xi, s))(s_values)
                Jp_all = J_all[:, :2, :]
                Jo_all = J_all[:, 2:, :]

                integrand_JpT_Jp = jnp.einsum("nij,nik->njk", Jp_all, Jp_all)
                integrand_JoT_Jo = jnp.einsum("nij,nik->njk", Jo_all, Jo_all)
                
                integral_Jp = jscipy.integrate.trapezoid(integrand_JpT_Jp, x=s_values, axis=0)
                integral_Jo = jscipy.integrate.trapezoid(integrand_JoT_Jo, x=s_values, axis=0)  
             
            B = B_i_prev + rho[i] * A[i] * integral_Jp + rho[i] * Ib[i] * integral_Jo
            return B
        
        # Loop over the segments
        B = lax.fori_loop(
            lower=0,
            upper=num_segments,
            body_fun=lambda i, val: B_i(i, val),
            init_val=B_0,
        )
        
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
        # Extract the parameters
        g = params["g"] # gravity vector [m/s^2]
        rho = params["rho"] # density of each segment [kg/m^3]
        l = params["l"] # length of each segment [m]
        r = params["r"] # radius of each segment [m]
        
        # Usefull derived quantitie
        A = jnp.pi * r**2  # cross-sectional area of each segment [m^2]

        U_g_0 = 0 # Initialize the mass / inertia matrix
        
        l_succ = jnp.cumsum(jnp.concatenate([jnp.array([0.0]), l]))
        # Iteration function
        def U_g_i(
            i: int, 
            U_g_i_prev: Array
        ) -> Array:
            if integration_type == "gauss":
                Xs, Ws, nGauss = gauss_quadrature(N_GQ=param_integration, a=l_succ[i], b=l_succ[i + 1])
                
                chi_s = vmap(lambda s: chi_fn_xi(params, xi, s))(Xs)
                p_s = chi_s[:,:2]
                
                integrand = -rho[i] * A[i] * jnp.dot(p_s, g)
                integral = jnp.sum(Ws * integrand)
            else:
                xs = jnp.linspace(l_succ[i], l_succ[i+1], param_integration)
                
                chi_s = vmap(lambda s: chi_fn_xi(params, xi, s))(xs)
                p_s = chi_s[:,:2]
                
                integrand = -rho[i] * A[i] * jnp.dot(p_s, g)
                integral = jscipy.integrate.trapezoid(integrand, x=xs)
            
            U_g = U_g_i_prev + integral
            return U_g
        
        # Loop over the segments
        U_g = lax.fori_loop(
            lower=0,
            upper=num_segments,
            body_fun=lambda i, val: U_g_i(i, val),
            init_val=U_g_0,
        )
        
        return U_g
    
    @jit 
    def G_fn_xi(
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
    def dynamical_matrices_fn(
        params: Dict[str, Array], 
        q: Array, 
        q_d: Array, 
        eps: float = 1e4 * global_eps
        ):
        # -> Tuple[Array, Array, Array, Array, Array, Array]: TODO
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
        K = B_xi.T @ K @ (xi - xi_eq) # evaluate K(xi) = K @ xi #TODO: check this
        
        # Compute the actuation matrix
        A = actuation_mapping_fn(
            forward_kinematics_fn, jacobian_fn, params, B_xi, q
        )
        # Apply the strain basis to the actuation matrix
        alpha = A #TODO: why no B_xi.T @ A @ B_xi?
        
        # Dissipative matrix
        D = params.get("D", jnp.zeros((n_xi, n_xi)))
        # Apply the strain basis to the dissipative matrix
        D = B_xi.T @ D @ B_xi
        
        B, C = B_C_fn_xi(params, xi, xi_d)
        
        B = B_xi.T @ B @ B_xi
        C = B_xi.T @ C @ B_xi
        G = B_xi.T @ G_fn_xi(params, xi).squeeze()
        
        return B, C, G, K, D, alpha    
    

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
        B, C, G, K, D, alpha = dynamical_matrices_fn(params, q=q, q_d=q_d)

        # Kinetic energy
        T = (0.5 * q_d.T @ B @ q_d).squeeze()

        return T

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
        segment_idx, s_segment = classify_segment(params, s)

        # make operational_space_selector a boolean array
        operational_space_selector = onp.array(operational_space_selector, dtype=bool)

        # Jacobian and its time-derivative
        # J = jacobian_fn_xi(params, xi_epsed, s_segment).squeeze() #TODO: check this
        J, J_d = J_Jd(params, xi_epsed, xi_d, s_segment).squeeze()
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
        # return J#TODO: check this

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