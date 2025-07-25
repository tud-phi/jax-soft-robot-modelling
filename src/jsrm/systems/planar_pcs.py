from jax import Array, lax, vmap
from jax import numpy as jnp

from typing import Callable, Dict, Tuple, Optional

import equinox as eqx

from .utils import (
    compute_strain_basis,
    compute_planar_stiffness_matrix,
    gauss_quadrature,
    scale_gaussian_quadrature,
)
from jsrm.math_utils import blk_diag
import jsrm.utils.lie_algebra as lie
from jsrm.utils.lie_operators import (
    compute_weighted_sums,
)

from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, PIDController, ConstantStepSize


class PlanarPCSNum(eqx.Module):
    # Robot parameters
    th0: Array  # Initial orientation angle [rad]
    g: Array  # Gravitational acceleration vector

    L: Array  # Length of the segments
    L_cum: Array  # Cumulative length of the segments
    r: Array  # Radius of the segments
    rho: Array
    E: Array  # Young's modulus of the segments
    G: Array  # Shear modulus of the segments
    D: Array  # Damping coefficient of the segments

    global_eps: float = jnp.finfo(jnp.float64).eps

    stiffness_fn: Callable = eqx.static_field()
    actuation_mapping_fn: Callable = eqx.static_field()

    num_segments: int = eqx.static_field()
    num_gauss_points: int = eqx.static_field()  #
    num_strains: int = eqx.static_field()  # Number of strains (3 * num_segments)

    xi_star: Array  # Rest configuration strain
    B_xi: Array  # Strain basis matrix
    num_selected_strains: Array  # Number of selected strains

    Xs: Array  # Gauss nodes
    Ws: Array  # Gauss weights

    def __init__(
        self,
        num_segments: int,
        params: Dict[str, Array],
        order_gauss: int = 5,
        strain_selector: Optional[Array] = None,
        xi_star: Optional[Array] = None,
        stiffness_fn: Optional[Callable] = None,
        actuation_mapping_fn: Optional[Callable] = None,
    ) -> "PlanarPCSNum":
        """
        Initialize the PlanarPCSNum class.

        Args:
            num_segments (int):
                Number of segments in the robot.
            params (Dict[str, Array]):
                Dictionary containing the robot parameters:
                - "th0": Initial orientation angle [rad]
                - "l": Length of each segment [m]
                - "r": Radius of each segment [m]
                - "rho": Density of each segment [kg/m^3]
                - "g": Gravitational acceleration vector [m/s^2]
                - "E": Elastic modulus of each segment [Pa]
                - "G": Shear modulus of each segment [Pa]
            order_gauss (int, optional):
                Order of the Gauss-Legendre quadrature for integration over each segment.
                Defaults to 5.
            strain_selector (Optional[Array], optional):
                Boolean array of shape (3 * num_segments,) specifying which strain components are active.
                Defaults to all strains active (i.e. all True).
            xi_star (Optional[Array], optional):
                Rest strain of shape (3 * num_segments,).
                Defaults to 0.0 for bending and shear strains, and 1.0 for axial strain (along local y-axis).
            stiffness_fn (Optional[Callable], optional):
                Function to compute the stiffness matrix.
                Defaults to a function that computes the stiffness matrix based on the parameters.
            actuation_mapping_fn (Optional[Callable], optional):
                Function to compute the actuation mapping.
                Defaults to identity mapping.

        """
        # Number of segments
        if not isinstance(num_segments, int):
            raise TypeError(
                f"num_segments must be an integer, got {type(num_segments).__name__}"
            )
        if num_segments < 1:
            raise ValueError(f"num_segments must be at least 1, got {num_segments}")
        self.num_segments = num_segments

        num_strains = 3 * num_segments
        self.num_strains = num_strains

        # ================================================================
        # Robot parameters

        # Initial orientation angle
        try:
            th0 = params["th0"]
        except KeyError:
            raise KeyError("Parameter 'th0' is required in params dictionary.")
        if not (isinstance(th0, (float, int, jnp.ndarray))):
            raise TypeError(
                f"th0 must be a float, int, or an array, got {type(th0).__name__}"
            )
        th0 = jnp.asarray(th0, dtype=jnp.float64)
        self.th0 = th0

        # Gravitational acceleration vector
        try:
            g = params["g"]
        except KeyError:
            raise KeyError("Parameter 'g' is required in params dictionary.")
        if not (isinstance(g, (list, jnp.ndarray))):
            raise TypeError(f"g must be a list or an array, got {type(g).__name__}")
        g = jnp.asarray(g, dtype=jnp.float64)
        if g.size != 2:
            raise ValueError(f"g must be a vector of shape (2,), got {g.size}")
        self.g = jnp.concatenate(
            [jnp.zeros(1), g]
        )  # Add a zero for the orientation angle

        # Lengths of the segments
        try:
            L = params["l"]
        except KeyError:
            raise KeyError("Parameter 'l' is required in params dictionary.")
        if not (isinstance(L, (list, jnp.ndarray))):
            raise TypeError(f"l must be a list or an array, got {type(L).__name__}")
        L = jnp.asarray(L, dtype=jnp.float64)
        if L.shape != (num_segments,):
            raise ValueError(f"l must have shape ({num_segments},), got {L.shape}")
        self.L = L

        L_cum = jnp.cumsum(jnp.concatenate([jnp.zeros(1), self.L]))
        self.L_cum = L_cum

        # Radius of the segments
        try:
            r = params["r"]
        except KeyError:
            raise KeyError("Parameter 'r' is required in params dictionary.")
        if not (isinstance(r, (list, jnp.ndarray))):
            raise TypeError(f"r must be a list or an array, got {type(r).__name__}")
        r = jnp.asarray(r, dtype=jnp.float64)
        if r.shape != (num_segments,):
            raise ValueError(f"r must have shape ({num_segments},), got {r.shape}")
        self.r = r

        # Densities of the segments
        try:
            rho = params["rho"]
        except KeyError:
            raise KeyError("Parameter 'rho' is required in params dictionary.")
        if not (isinstance(rho, (list, jnp.ndarray))):
            raise TypeError(f"rho must be a list or an array, got {type(rho).__name__}")
        rho = jnp.asarray(rho, dtype=jnp.float64)
        if rho.shape != (num_segments,):
            raise ValueError(f"rho must have shape ({num_segments},), got {rho.shape}")
        self.rho = rho

        # Elastic modulus of the segments
        try:
            E = params["E"]
        except KeyError:
            raise KeyError("Parameter 'E' is required in params dictionary.")
        if not (isinstance(E, (list, jnp.ndarray))):
            raise TypeError(f"E must be a list or an array, got {type(E).__name__}")
        E = jnp.asarray(E, dtype=jnp.float64)
        if E.shape != (num_segments,):
            raise ValueError(f"E must have shape ({num_segments},), got {E.shape}")
        self.E = E

        # Shear modulus of the segments
        try:
            G = params["G"]
        except KeyError:
            raise KeyError("Parameter 'G' is required in params dictionary.")
        if not (isinstance(G, (list, jnp.ndarray))):
            raise TypeError(f"G must be a list or an array, got {type(G).__name__}")
        G = jnp.asarray(G, dtype=jnp.float64)
        if G.shape != (num_segments,):
            raise ValueError(f"G must have shape ({num_segments},), got {G.shape}")
        self.G = G

        # Damping matrix of the robot
        try:
            D = params["D"]
        except KeyError:
            raise KeyError("Parameter 'D' is required in params dictionary.")
        if not (isinstance(D, (list, jnp.ndarray))):
            raise TypeError(f"D must be a list or an array, got {type(D).__name__}")
        D = jnp.asarray(D, dtype=jnp.float64)
        expected_D_shape = (num_strains, num_strains)
        if D.shape != expected_D_shape:
            raise ValueError(f"D must have shape {expected_D_shape}, got {D.shape}")
        self.D = D

        # ================================================================
        # Order of Gauss-Legendre quadrature
        if not isinstance(order_gauss, int):
            raise TypeError(
                f"order_gauss must be an integer, got {type(order_gauss).__name__}"
            )
        if order_gauss < 1:
            raise ValueError(f"param_integration must be at least 1, got {order_gauss}")
        Xs, Ws, num_gauss_points = gauss_quadrature(order_gauss, a=0.0, b=1.0)
        self.Xs = Xs
        self.Ws = Ws
        self.num_gauss_points = num_gauss_points

        # ================================================================
        # Strain basis matrix
        if strain_selector is None:
            strain_selector = jnp.ones(num_strains, dtype=bool)
        else:
            if not isinstance(strain_selector, (list, jnp.ndarray)):
                raise TypeError(
                    f"strain_selector must be a list or an array, got {type(strain_selector).__name__}"
                )
            strain_selector = jnp.asarray(strain_selector)
            if not jnp.issubdtype(strain_selector.dtype, jnp.bool_):
                raise TypeError(
                    f"strain_selector must be a boolean array, got {strain_selector.dtype}"
                )
            if strain_selector.size != num_strains:
                raise ValueError(
                    f"strain_selector must have {num_strains} elements, got {strain_selector.size}"
                )
            strain_selector = strain_selector.reshape(num_strains)
        self.B_xi = compute_strain_basis(strain_selector)

        self.num_selected_strains = jnp.sum(strain_selector)

        # Rest configuration strain
        if xi_star is None:
            xi_star = jnp.tile(
                jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64), (num_segments, 1)
            ).reshape(num_strains)
        else:
            if not isinstance(xi_star, (list, jnp.ndarray)):
                raise TypeError(
                    f"xi_star must be a list or an array, got {type(xi_star).__name__}"
                )
            xi_star = jnp.asarray(xi_star)
            if xi_star.size != num_strains:
                raise ValueError(
                    f"xi_star must have {num_strains} elements, got {xi_star.size}"
                )
            xi_star = xi_star.reshape(num_strains)
        self.xi_star = xi_star

        # Stiffness function
        if stiffness_fn is None:
            compute_stiffness_matrix_for_all_segments_fn = vmap(
                compute_planar_stiffness_matrix
            )

            def stiffness_fn(
                formulate_in_strain_space: bool = False,
            ) -> Array:
                L = self.L
                r = self.r
                E = self.E
                G = self.G

                # cross-sectional area and second moment of area
                A = jnp.pi * r**2
                Ib = A**2 / (4 * jnp.pi)

                # stiffness matrix of shape (num_segments, 3, 3)
                S_sms = compute_stiffness_matrix_for_all_segments_fn(L, A, Ib, E, G)
                # we define the elastic matrix of shape (num_strains, num_strains) as K(xi) = K @ xi where K is equal to
                S = blk_diag(S_sms)

                if not formulate_in_strain_space:
                    S = self.B_xi.T @ S @ self.B_xi

                return S
        else:
            if not callable(stiffness_fn):
                raise TypeError(
                    f"stiffness_fn must be a callable, got {type(stiffness_fn).__name__}"
                )
        self.stiffness_fn = stiffness_fn

        # Actuation mapping function
        if actuation_mapping_fn is None:

            def actuation_mapping_fn(q: Array, tau: Array) -> Array:
                A = self.B_xi.T @ jnp.identity(self.num_strains) @ self.B_xi
                alpha = A @ tau
                return alpha
        else:
            if not callable(actuation_mapping_fn):
                raise TypeError(
                    f"actuation_mapping_fn must be a callable, got {type(actuation_mapping_fn).__name__}"
                )
        self.actuation_mapping_fn = actuation_mapping_fn

    def classify_segment(
        self,
        s: Array,
    ) -> Tuple[Array, Array]:
        """
        Classify the point along the robot to the corresponding segment.

        Args:
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            segment_idx (Array): index of the segment where the point is located
            s_segment (Array): point coordinate along the segment in the interval [0, l_segment]
        """

        # Classify the point along the robot to the corresponding segment
        segment_idx = jnp.clip(jnp.sum(s > self.L_cum) - 1, 0, self.num_segments - 1)

        # Compute the point coordinate along the segment in the interval [0, l_segment]
        s_local = s - self.L_cum[segment_idx]

        return segment_idx, s_local

    def strain_fn(
        self,
        q: Array,
    ) -> Array:
        """
        Compute the strain vector from the generalized coordinates.

        Args:
            q (Array): generalized coordinates of shape (num_selected_strains,).

        Returns:
            xi (Array): strain vector of shape (num_selected_strains,)
        """
        xi = self.B_xi @ q + self.xi_star

        return xi

    def chi_fn(
        self,
        xi: Array,
        s: Array,
    ) -> Array:
        """
        Compute the forward kinematics of the robot.

        Args:
            xi (Array): strain vector of shape (3*num_segments,) where each row corresponds to a segment
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            chi_s (Array): forward kinematics of the robot at point s, shape (3,) : [theta, x, y]
        """
        xi = xi.reshape(self.num_segments, 3)

        segment_idx, s_local = self.classify_segment(s)

        chi_0 = jnp.concatenate(
            [self.th0[None], jnp.zeros(2)]
        )  # Initial configuration [theta, x, y]

        # Iteration function
        def chi_i(chi_prev: Array, i: int) -> Array:
            th_prev = chi_prev[0]
            p_prev = chi_prev[1:]

            kappa_i = xi[i, 0]
            sigmas_i = xi[i, 1:]

            l_i = jnp.where(i == segment_idx, s_local, self.L[i])

            th = th_prev + kappa_i * l_i

            int_cos_th = jnp.where(
                jnp.abs(kappa_i) < self.global_eps,
                l_i * jnp.cos(th_prev),
                (jnp.sin(th) - jnp.sin(th_prev)) / kappa_i,
            )
            int_sin_th = jnp.where(
                jnp.abs(kappa_i) < self.global_eps,
                l_i * jnp.sin(th_prev),
                -(jnp.cos(th) - jnp.cos(th_prev)) / kappa_i,
            )

            R = jnp.stack(
                [
                    jnp.stack([int_cos_th, -int_sin_th]),
                    jnp.stack([int_sin_th, int_cos_th]),
                ]
            )

            p = p_prev + R @ sigmas_i

            chi = jnp.concatenate([th[None], p])

            return chi, chi

        _, chi_list = lax.scan(f=chi_i, init=chi_0, xs=jnp.arange(self.num_segments))

        chi_s = chi_list[segment_idx]

        return chi_s

    def forward_kinematics_fn(
        self,
        q: Array,
        s: Array,
    ) -> Array:
        """
        Compute the forward kinematics of the robot at a point s along the robot.

        Args:
            q (Array): generalized coordinates of shape (num_selected_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            chi (Array): forward kinematics of the robot at point s, shape (3,) : [theta, x, y]
        """
        xi = self.strain_fn(q)

        chi = self.chi_fn(xi, s)

        return chi

    def J_local_for_computation(self, q: Array, s: Array) -> Array:
        """
        Compute the Jacobian of the forward kinematics at a point s along the robot.

        Args:
            q (Array): generalized coordinates of shape (num_selected_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_local_for_computation (Array): Jacobian of the forward kinematics at point s, shape (num_segments, 3, 3)
            where each row corresponds to a segment.
        """
        xi = self.strain_fn(q).reshape(self.num_segments, 3)

        # Classify the point along the robot to the corresponding segment
        segment_idx, s_local = self.classify_segment(s)

        # Initial condition
        xi_0 = xi[0]
        L_0 = self.L[0]

        Ad_g0_inv_L0 = lie.Adjoint_gi_se2_inv(xi_0, L_0, eps=self.global_eps)
        Ad_g0_inv_s = lie.Adjoint_gi_se2_inv(xi_0, s_local, eps=self.global_eps)

        T_g0_L0 = lie.Tangent_gi_se2(xi_0, L_0, eps=self.global_eps)
        T_g0_s = lie.Tangent_gi_se2(xi_0, s_local, eps=self.global_eps)

        mat_0_L0 = Ad_g0_inv_L0 @ T_g0_L0
        mat_0_s = Ad_g0_inv_s @ T_g0_s

        J_0_L0 = jnp.concatenate(
            [mat_0_L0[None, :, :], jnp.zeros((self.num_segments - 1, 3, 3))], axis=0
        )
        J_0_s = jnp.concatenate(
            [mat_0_s[None, :, :], jnp.zeros((self.num_segments - 1, 3, 3))], axis=0
        )

        tuple_J_0 = (J_0_L0, J_0_s)

        # Iteration function
        def J_i(tuple_J_prev: Array, i: int) -> Tuple[Tuple[Array, Array], Array]:
            J_prev_Lprev, _ = tuple_J_prev

            xi_i = xi[i]

            Ad_gi_inv_Li = lie.Adjoint_gi_se2_inv(xi_i, self.L[i], eps=self.global_eps)
            Ad_gi_inv_s = lie.Adjoint_gi_se2_inv(xi_i, s_local, eps=self.global_eps)

            T_gi_Li = lie.Tangent_gi_se2(xi_i, self.L[i], eps=self.global_eps)
            T_gi_s = lie.Tangent_gi_se2(xi_i, s_local, eps=self.global_eps)

            mat_i_Li = Ad_gi_inv_Li @ T_gi_Li
            mat_i_s = Ad_gi_inv_s @ T_gi_s

            J_i_s = lax.dynamic_update_slice(
                jnp.einsum("ij, njk->nik", Ad_gi_inv_s, J_prev_Lprev),
                mat_i_s[jnp.newaxis, ...],
                (i, 0, 0),
            )
            J_i_Li = lax.dynamic_update_slice(
                jnp.einsum("ij, njk->nik", Ad_gi_inv_Li, J_prev_Lprev),
                mat_i_Li[jnp.newaxis, ...],
                (i, 0, 0),
            )

            return (J_i_Li, J_i_s), J_i_s

        indices_links = jnp.arange(1, self.num_segments)

        _, J_array = lax.scan(f=J_i, init=tuple_J_0, xs=indices_links)

        # Add the initial condition to the Jacobian array
        J_array = jnp.concatenate([J_0_s[jnp.newaxis, ...], J_array], axis=0)

        # Extract the Jacobian for the segment that contains the point s
        J_local_for_computation = lax.dynamic_index_in_dim(
            J_array, segment_idx, axis=0, keepdims=False
        )

        return J_local_for_computation

    def final_size_jacobian(self, J_full: Array) -> Array:
        """
        Convert the Jacobian or its derivative from the full computation form to the selected strains form.

        Args:
            J_full (Array): Full Jacobian of shape (num_segments, 3, 3)

        Returns:
            J_selected (Array): Jacobian for the selected strains of shape (3, num_selected_strains)
        """
        J_final = J_full.transpose(1, 0, 2).reshape(3, self.num_strains) @ self.B_xi

        return J_final

    def jacobian_bodyframe_fn(self, q: Array, s: Array) -> Array:
        """
        Compute the Jacobian of the forward kinematics at a point s along the robot in the body frame.

        Args:
            q (Array): generalized coordinates of shape (num_selected_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_local (Array): Jacobian of the forward kinematics at point s in the body frame, shape (3, num_selected_strains)
        """
        J_local_for_computation = self.J_local_for_computation(q, s)

        J_local = self.final_size_jacobian(J_local_for_computation)

        return J_local

    def jacobian_inertialframe_fn(self, q: Array, s: Array) -> Array:
        """
        Compute the Jacobian of the forward kinematics at a point s along the robot in the inertial frame.

        Args:
            q (Array): generalized coordinates of shape (num_selected_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_global (Array): Jacobian of the forward kinematics at point s in the inertial frame, shape (3, num_selected_strains)
        """
        J_local_for_computation = self.J_local_for_computation(q, s)

        xi_i = self.strain_fn(q)

        chi = self.chi_fn(xi_i, s)
        theta = chi[0]
        g_i = lie.exp_SE2(
            jnp.stack([theta, 0.0, 0.0])
        )  # SE(2) transformation at point s
        Adj_gi = lie.Adjoint_g_SE2(
            g_i
        )  # Adjoint representation of the SE(2) transformation

        J_global_for_computation = jnp.einsum(
            "ij, njk -> nik",
            Adj_gi,
            J_local_for_computation,
        )

        J_global = self.final_size_jacobian(J_global_for_computation)

        return J_global

    def J_Jd_for_computation(
        self, q: Array, q_d: Array, s: Array
    ) -> Tuple[Array, Array]:
        """
        Compute the Jacobian and its time-derivative for the forward kinematics at a point s along the robot.

        Args:
            q (Array): generalized coordinates of shape (num_selected_strains,).
            q_d (Array): time-derivative of the generalized coordinates of shape (num_selected_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_local_for_computation (Array): Jacobian of the forward kinematics at point s, shape (num_segments, 3, 3)
            J_d_local_for_computation (Array): Time-derivative of the Jacobian at point s, shape (num_segments, 3, 3)
        """
        xi_d = (self.B_xi @ q_d).reshape(self.num_segments, 3)

        # Classify the point along the robot to the corresponding segment
        segment_idx, _ = self.classify_segment(s)

        J_local_for_computation = self.J_local_for_computation(q, s)

        # =================================
        # Computation of the time-derivative of the Jacobian

        idx_range = jnp.arange(self.num_segments)
        J_i = vmap(
            lambda i: lax.dynamic_index_in_dim(
                J_local_for_computation, i, axis=0, keepdims=False
            )
        )(idx_range)  # shape: (num_segments, 3, 3)
        sum_Jj_xi_d_j = compute_weighted_sums(
            J_local_for_computation, xi_d, self.num_segments
        )  # shape: (num_segments, 3)
        adjoint_sum = vmap(lie.adjoint_se2)(
            sum_Jj_xi_d_j
        )  # shape: (num_segments, 3, 3)

        # Compute the time-derivative of the Jacobian
        J_d_local_for_computation = jnp.einsum(
            "ijk, ikl->ijl", adjoint_sum, J_i
        )  # shape: (num_segments, 3, 3)

        # Replace the elements of J_d_segment_SE2 for i > segment_idx by null matrices
        J_d_local_for_computation = jnp.where(
            jnp.arange(self.num_segments)[:, None, None] > segment_idx,
            jnp.zeros_like(J_d_local_for_computation),
            J_d_local_for_computation,
        )

        return J_local_for_computation, J_d_local_for_computation

    def jacobian_and_derivative_bodyframe_fn(
        self, q: Array, q_d: Array, s: Array
    ) -> Tuple[Array, Array]:
        """
        Compute the Jacobian and its time-derivative for the forward kinematics at a point s along the robot in the body frame.

        Args:
            q (Array): generalized coordinates of shape (num_selected_strains,).
            q_d (Array): time-derivative of the generalized coordinates of shape (num_selected_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_local (Array): Jacobian of the forward kinematics at point s in the body frame, shape (3, num_selected_strains)
            J_d_local (Array): Time-derivative of the Jacobian at point s in the body frame, shape (3, num_selected_strains)
        """
        J_local_for_computation, J_d_local_for_computation = self.J_Jd_for_computation(
            q, q_d, s
        )

        J_local = self.final_size_jacobian(J_local_for_computation)
        J_d_local = self.final_size_jacobian(J_d_local_for_computation)

        return J_local, J_d_local

    def jacobian_and_derivative_inertialframe_fn(
        self, q: Array, q_d: Array, s: Array
    ) -> Tuple[Array, Array]:
        """
        Compute the Jacobian and its time-derivative for the forward kinematics at a point s along the robot in the inertial frame.

        Args:
            q (Array): generalized coordinates of shape (num_selected_strains,).
            q_d (Array): time-derivative of the generalized coordinates of shape (num_selected_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].

        Returns:
            J_global (Array): Jacobian of the forward kinematics at point s in the inertial frame, shape (3, num_selected_strains)
            J_d_global (Array): Time-derivative of the Jacobian at point s in the inertial frame, shape (3, num_selected_strains)
        """
        J_local_for_computation, J_d_local_for_computation = self.J_Jd_for_computation(
            q, q_d, s
        )

        xi_i = self.strain_fn(q)
        chi = self.chi_fn(xi_i, s)
        theta = chi[0]
        g_i = lie.exp_SE2(
            jnp.stack([theta, 0.0, 0.0])
        )  # SE(2) transformation at point s
        Adj_gi = lie.Adjoint_g_SE2(g_i)

        J_global_for_computation = jnp.einsum(
            "ijk, ikl -> ijl",
            Adj_gi,
            J_local_for_computation,
        )
        J_d_global_for_computation = jnp.einsum(
            "ijk, ikl -> ijl",
            Adj_gi,
            J_d_local_for_computation,
        )

        J_global = self.final_size_jacobian(J_global_for_computation)
        J_d_global = self.final_size_jacobian(J_d_global_for_computation)

        return J_global, J_d_global

    # ==========================================
    # Useful functions for the system

    def local_cross_sectional_area(self, i: int) -> Array:
        """
        Compute the local cross-sectional area for the i-th segment.

        Args:
            i (int): index of the segment

        Returns:
            A_i (Array): local cross-sectional area of the i-th segment
        """
        A_i = jnp.pi * self.r[i] ** 2  # Cross-sectional area
        return A_i

    def local_mass_matrix(self, i) -> Array:
        """
        Compute the local mass matrix for the i-th segment.

        Args:
            i (int): index of the segment
        Returns:
            M_i (Array): local mass matrix of shape (3, 3) for the i-th segment
        """
        rho_i = self.rho[i]
        A_i = self.Local_cross_sectional_area(i)  # Cross-sectional area
        I_i = A_i**2 / (4 * jnp.pi)  # Second moment of area

        M_i = rho_i * jnp.diag(jnp.array([I_i, A_i, A_i]))
        return M_i

    # ===========================================
    # Dynamical matrices computation

    def inertia_full_matrix(
        self,
        q: Array,
    ) -> Array:
        """
        Compute the full inertia matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_selected_strains,).

        Returns:
            B_full (Array): Full inertia matrix of shape (num_strains, num_strains).
        """

        def B_i(i):
            Xs_scaled, Ws_scaled = scale_gaussian_quadrature(
                self.Xs, self.Ws, self.L_cum[i], self.L_cum[i + 1]
            )
            M_i = self.local_mass_matrix(i)

            def B_j(j):
                Xs_j = Xs_scaled[j]
                Ws_j = Ws_scaled[j]
                J_j = self.jacobian_bodyframe_fn(q, Xs_j)
                return Ws_j * J_j.T @ M_i @ J_j

            B_blocks_i = vmap(B_j)(jnp.arange(self.num_gauss_points))

            # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
            # B_blocks_i = jnp.stack([B_j(j) for j in range(self.num_gauss_points)], axis=0)

            return B_blocks_i

        B_blocks_tot = vmap(B_i)(jnp.arange(self.num_segments))

        # # For debugging purposes, you can uncomment the following line to see the step-by-step computation
        # B_blocks_tot = jnp.stack([B_i(i) for i in range(self.num_segments)], axis=0)

        B_full = jnp.sum(
            B_blocks_tot, axis=(0, 1)
        )  # Sum over segments and Gauss points

        return B_full

    def inertia_matrix(
        self,
        q: Array,
    ) -> Array:
        """
        Compute the inertia matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_selected_strains,).

        Returns:
            B (Array): Inertia matrix of shape (num_selected_strains, num_selected_strains).
        """
        B_full = self.inertia_full_matrix(q)

        B = self.B_xi.T @ B_full @ self.B_xi

        return B

    def coriolis_full_matrix(
        self,
        q: Array,
        q_d: Array,
    ) -> Array:
        """
        Compute the full Coriolis matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_selected_strains,).
            q_d (Array): time-derivative of the generalized coordinates of shape (num_selected_strains,).

        Returns:
            C_full (Array): Full Coriolis matrix of shape (num_strains, num_strains).
        """

        def C_i(i):
            Xs_scaled, Ws_scaled = scale_gaussian_quadrature(
                self.Xs, self.Ws, self.L_cum[i], self.L_cum[i + 1]
            )
            M_i = self.local_mass_matrix(i)

            def C_j(j):
                Xs_j = Xs_scaled[j]
                Ws_j = Ws_scaled[j]
                J_j, J_d_j = self.jacobian_and_derivative_bodyframe_fn(q, q_d, Xs_j)
                return Ws_j * (
                    J_j.T @ (M_i @ J_d_j + lie.coadjoint_se2(J_j @ q_d) @ M_i @ J_j)
                )

            C_blocks_i = vmap(C_j)(jnp.arange(self.num_gauss_points))

            return C_blocks_i

        C_blocks_tot = vmap(C_i)(jnp.arange(self.num_segments))

        C_full = jnp.sum(C_blocks_tot, axis=(0, 1))

        return C_full

    def coriolis_matrix(
        self,
        q: Array,
        q_d: Array,
    ) -> Array:
        """
        Compute the Coriolis matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_selected_strains,).
            q_d (Array): time-derivative of the generalized coordinates of shape (num_selected_strains,).

        Returns:
            C (Array): Coriolis matrix of shape (num_selected_strains, num_selected_strains).
        """
        C_full = self.coriolis_full_matrix(q, q_d)

        C = self.B_xi.T @ C_full @ self.B_xi

        return C

    def gravitational_full_vector(
        self,
        q: Array,
    ) -> Array:
        """
        Compute the full gravitational vector of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_selected_strains,).

        Returns:
            G (Array): Full gravitational vector of shape (num_strains,).
        """

        def G_i(i):
            Xs_scaled, Ws_scaled = scale_gaussian_quadrature(
                self.Xs, self.Ws, self.L_cum[i], self.L_cum[i + 1]
            )
            M_i = self.local_mass_matrix(i)

            def G_j(j):
                Xs_j = Xs_scaled[j]
                Ws_j = Ws_scaled[j]
                Ad_g_inv_j = lie.Adjoint_g_inv_SE2(
                    lie.exp_SE2(self.forward_kinematics_fn(q, Xs_j))
                )
                J_j = self.jacobian_bodyframe_fn(q, Xs_j)
                return -Ws_j * J_j.T @ M_i @ Ad_g_inv_j @ self.g

            G_blocks_segment_i = vmap(G_j)(jnp.arange(self.num_gauss_points))

            return G_blocks_segment_i

        G_blocks_tot = vmap(G_i)(jnp.arange(self.num_segments))

        G_full = jnp.sum(
            G_blocks_tot, axis=(0, 1)
        )  # Sum over links and quadrature points

        return G_full

    def gravitational_vector(
        self,
        q: Array,
    ) -> Array:
        """
        Compute the gravitational vector of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_selected_strains,).

        Returns:
            G (Array): Gravitational vector of shape (num_selected_strains,).
        """
        G_full = self.gravitational_full_vector(q)

        G = self.B_xi.T @ G_full

        return G

    def stiffness_full_matrix(
        self,
    ) -> Array:
        """
        Compute the full stiffness matrix of the robot.

        Returns:
            K_full (Array): Full stiffness matrix of shape (num_strains, num_strains).
        """
        K_full = self.stiffness_fn(formulate_in_strain_space=True)

        return K_full

    def stiffness_matrix(
        self,
        q: Array,
    ) -> Array:
        """
        Compute the stiffness matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_selected_strains,).

        Returns:
            K (Array): Stiffness matrix of shape (num_selected_strains, num_selected_strains).
        """
        K = self.stiffness_full_matrix()
        K = self.B_xi.T @ K @ self.B_xi @ q

        return K

    def damping_full_matrix(
        self,
    ) -> Array:
        """
        Compute the full damping matrix of the robot.

        Args:
            None

        Returns:
            D (Array): Full damping matrix of shape (num_strains, num_strains).
        """
        D_full = self.D

        return D_full

    def damping_matrix(
        self,
    ) -> Array:
        """
        Compute the damping matrix of the robot.

        Args:
            None

        Returns:
            D (Array): Damping matrix of shape (num_selected_strains, num_selected_strains).
        """
        D_full = self.damping_full_matrix()

        D = self.B_xi.T @ D_full @ self.B_xi

        return D

    def actuation_matrix(
        self,
        q: Array,
        actuation_args: Optional[Tuple] = None,
    ) -> Array:
        """
        Compute the actuation matrix of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_selected_strains,).
            actuation_args (Tuple, optional): Additional arguments for the actuation mapping function, if any.

        Returns:
            alpha (Array): Actuation matrix of shape (num_selected_strains, num_selected_strains).
        """
        alpha = self.actuation_mapping_fn(q, *actuation_args)

        return alpha

    def dynamical_matrices(
        self,
        q: Array,
        q_d: Array,
        actuation_args: Optional[Tuple] = None,
    ) -> Tuple[Array, Array, Array, Array, Array, Array]:
        """
        Compute the dynamical matrices of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_selected_strains,).
            q_d (Array): time-derivative of the generalized coordinates of shape (num_selected_strains,).
            actuation_args (Tuple, optional): Additional arguments for the actuation mapping function, if any.

        Returns:
            B (Array): Inertia matrix of shape (num_selected_strains, num_selected_strains).
            C (Array): Coriolis matrix of shape (num_selected_strains, num_selected_strains).
            G (Array): Gravitational vector of shape (num_selected_strains,).
            K (Array): Stiffness matrix of shape (num_selected_strains, num_selected_strains).
            D (Array): Damping matrix of shape (num_selected_strains, num_selected_strains).
            A (Array): Actuation matrix of shape (num_selected_strains, num_selected_strains).
        """
        B = self.inertia_matrix(q)
        C = self.coriolis_matrix(q, q_d)
        G = self.gravitational_vector(q)
        K = self.stiffness_matrix(q)
        D = self.damping_matrix()
        A = self.actuation_matrix(q, actuation_args)

        return B, C, G, K, D, A

    def kinetic_energy(
        self,
        q: Array,
        q_d: Array,
    ) -> float:
        """
        Compute the kinetic energy of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_selected_strains,).
            q_d (Array): time-derivative of the generalized coordinates of shape (num_selected_strains,).

        Returns:
            T (float): Kinetic energy of the robot.
        """
        B = self.inertia_matrix(q)
        T = 0.5 * q_d.T @ B @ q_d

        return T

    def elastic_energy(
        self,
        q: Array,
    ) -> float:
        """
        Compute the elastic energy of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_selected_strains,).

        Returns:
            U_K (float): Elastic energy of the robot.
        """
        K_full = self.stiffness_full_matrix()
        U_K = 0.5 * (self.B_xi @ q).T @ K_full @ (self.B_xi @ q)

        return U_K

    def gravitational_energy(
        self,
        q: Array,
    ) -> float:
        """
        Compute the gravitational energy of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_selected_strains,).

        Returns:
            U_G (float): Gravitational energy of the robot.
        """

        def U_G_i(i):
            Xs_scaled, Ws_scaled = scale_gaussian_quadrature(
                self.Xs, self.Ws, self.L_cum[i], self.L_cum[i + 1]
            )
            rho_i = self.rho[i]
            A_i = self.local_cross_sectional_area(i)  # Cross-sectional area

            def U_G_j(j):
                Xs_j = Xs_scaled[j]
                Ws_j = Ws_scaled[j]
                p_j = (
                    self.forward_kinematics_fn(q, Xs_j).at[0].set(0.0)
                )  # Set the orientation angle to 0 for gravitational energy computation
                return -Ws_j * rho_i * A_i * jnp.dot(p_j, self.g)

            U_G_blocks_segment_i = vmap(U_G_j)(jnp.arange(self.num_gauss_points))

            return U_G_blocks_segment_i

        U_G_blocks_tot = vmap(U_G_i)(jnp.arange(self.num_segments))

        U_G = jnp.sum(U_G_blocks_tot, axis=(0, 1))  # Sum over segments and Gauss points

        return U_G

    def potential_energy(
        self,
        q: Array,
    ) -> float:
        """
        Compute the potential energy of the robot.

        Args:
            q (Array): generalized coordinates of shape (num_selected_strains,).

        Returns:
            U (float): Potential energy of the robot.
        """
        U_K = self.elastic_energy(q)
        U_G = self.gravitational_energy(q)

        return U_K + U_G

    def total_energy(
        self,
        q: Array,
        q_d: Array,
    ) -> float:
        """
        Compute the total energy of the robot, which is the sum of kinetic and potential energy.

        Args:
            q (Array): generalized coordinates of shape (num_selected_strains,).
            q_d (Array): time-derivative of the generalized coordinates of shape (num_selected_strains,).

        Returns:
            E (float): Total energy of the robot.
        """
        T = self.kinetic_energy(q, q_d)
        U = self.potential_energy(q)
        E = T + U
        return E

    def operational_space_dynamical_matrices_fn(
        self,
        q: Array,
        q_d: Array,
        s: Array,
        operational_space_selector: Tuple = (True, True, True),
    ) -> Tuple[Array, Array, Array, Array, Array]:
        """
        Compute the operational space dynamical matrices for the robot at a point s along the robot.

        Args:
            q (Array): generalized coordinates of shape (num_selected_strains,).
            q_d (Array): time-derivative of the generalized coordinates of shape (num_selected_strains,).
            s (Array): point coordinate along the robot in the interval [0, L].
            operational_space_selector (Tuple): Selector for the operational space dimensions.
                Default is (True, True, True) for all dimensions.

        Returns:
            Lambda (Array): Inertia matrix in the operational space, shape (num_operational_space_dims, num_operational_space_dims).
            mu (Array): Coriolis and centrifugal matrix in the operational space, shape (num_operational_space_dims,).
            J (Array): Jacobian of the forward kinematics at point s in the body frame, shape (num_operational_space_dims, num_selected_strains).
            J_d (Array): Time-derivative of the Jacobian at point s in the body frame, shape (num_operational_space_dims, num_selected_strains).
            JB_pinv (Array): Dynamically-consistent pseudo-inverse of the Jacobian, shape (num_selected_strains, num_operational_space_dims).
        """
        # classify the point along the robot to the corresponding segment
        _, s_local = self.classify_segment(s)

        # make operational_space_selector a boolean array
        operational_space_selector = jnp.array(operational_space_selector, dtype=bool)

        # Jacobian and its time-derivative
        J, J_d = self.jacobian_and_derivative_bodyframe_fn(q, q_d, s_local)

        J = J[operational_space_selector, :]
        J_d = J_d[operational_space_selector, :]

        # inverse of the inertia matrix in the configuration space
        B = self.inertia_matrix(q)
        B_inv = jnp.linalg.inv(B)
        C = self.coriolis_matrix(q, q_d)

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

    @eqx.filter_jit
    def forward_dynamics(
        self,
        t: float,
        y: Array,
        actuation_args: Optional[Tuple] = None,
    ) -> Array:
        """
        Forward dynamics function.

        Args:
            t (float): Current time.
            y (Array): State vector containing configuration and velocity.
                Shape is (2 * num_strains,).
            actuation_args (Tuple, optional): Additional arguments for the actuation mapping function.
                Default is None.
        Returns:
            dydt: Time derivative of the state vector.
        """

        q, q_d = jnp.split(
            y, 2
        )  # Split the state vector into configuration and velocity

        B, C, G, K, D, A = self.dynamical_matrices(q, q_d, actuation_args)

        B_inv = jnp.linalg.inv(B)  # Inverse of the inertia matrix
        q_dd = B_inv @ (-C @ q_d - G - K - D @ q_d + A)  # Compute the acceleration

        dydt = jnp.concatenate([q_d, q_dd])

        return dydt

    def resolve_upon_time(
        self,
        q0: Array,
        qd0: Array,
        actuation_args: Optional[Tuple] = None,
        t0: Optional[float] = 0.0,
        t1: Optional[float] = 10.0,
        dt: Optional[float] = 1e-4,
        skip_steps: Optional[int] = 0,
        stepsize_controller: Optional[PIDController] = ConstantStepSize(),
        max_steps: Optional[int] = None,
    ) -> Tuple[Array, Array, Array]:
        """
        Resolve the system dynamics over time using Diffrax.

        Args:
            q0 (Array): Initial configuration (strains).
            qd0 (Array): Initial velocity (strains).
            actuation_args (Tuple, optional): Additional arguments for the actuation function.
                Default is None (no actuation).
            t0 (float, optionnal): Initial time.
                Default is 0.0.
            t1 (float, optionnal): Final time.
                Default is 10.0.
            dt (float, optionnal): Time step for the solver.
                Default is 1e-4.
            skip_steps (int, optionnal): Number of steps to skip in the output.
                This allows to reduce the number of saved time points.
                Default is 0.
            stepsize_controller (PIDController, optional): Stepsize controller for the solver.
                Default is ConstantStepSize().
            max_steps (int, optional): Maximum number of steps for the solver.
                Default is None (no limit).

        Returns:
            ts (Array): Time points at which the solution is saved.
            qs (Array): Configuration (strains) at the saved time points.
            qds (Array): Velocity (strains) at the saved time points.
        """
        y0 = jnp.concatenate([q0, qd0])  # Initial state vector

        term = ODETerm(self.forward_dynamics)

        solver = Tsit5()  # Runge-Kutta 5(4) method

        t = jnp.arange(t0, t1, dt)  # Time points for the solution
        saveat = SaveAt(ts=t[::skip_steps])  # Save at specified time points

        sol = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t[0],
            t1=t[-1],
            dt0=dt,
            y0=y0,
            args=actuation_args,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
        )

        ts = sol.ts
        # Extract the configuration and velocity from the solution
        y_out = sol.ys
        qs = y_out[:, : self.num_strains]  # Configuration (strains)
        qds = y_out[:, self.num_strains :]  # Velocity (strains)

        return ts, qs, qds
