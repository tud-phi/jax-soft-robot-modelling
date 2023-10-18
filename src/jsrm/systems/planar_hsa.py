import dill
from jax import Array, jit, lax, vmap
from jax import numpy as jnp
import sympy as sp
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

from .utils import (
    concatenate_params_syms,
    compute_strain_basis,
    compute_planar_stiffness_matrix,
)
from jsrm.math_utils import blk_diag


def factory(
    sym_exp_filepath: Union[str, Path],
    strain_selector: Array = None,
    eps: float = 1e-6,
) -> Tuple[
    Callable[[Dict[str, Array], Array, Array], Array],
    Callable[[Dict[str, Array], Array, Array], Array],
    Callable[[Dict[str, Array], Array], Array],
    Callable[[Dict[str, Array], Array], Array],
    Callable[[Dict[str, Array], Array], Array],
    Dict,
]:
    """
    Create jax functions from file containing symbolic expressions.
    Args:
        sym_exp_filepath: path to file containing symbolic expressions
        strain_selector: array of shape (n_xi, ) with boolean values indicating which components of the
                strain are active / non-zero
        eps: small number to avoid division by zero
    Returns:
        forward_kinematics_virtual_backbone_fn: function that returns the chi vector of shape (3, n_q) with the
            positions and orientations of the virtual backbone
        forward_kinematics_end_effector_fn: function that returns the pose of the end effector of shape (3, )
        jacobian_end_effector_fn: function that returns the Jacobian of the end effector of shape (3, n_q)
        inverse_kinematics_end_effector_fn: function that returns the generalized coordinates for a given end-effector pose
        dynamical_matrices_fn: function that returns the B, C, G, K, D, and alpha matrices
        sys_helpers: dictionary of helper functions / variables. Includes for examples:
            B_xi: strain basis matrix of shape (3 * num_segments, n_q)
            forward_kinematics_rod_fn: function that returns the chi vector of shape (3, n_q) with the
                positions and orientations of the rod
            forward_kinematics_platform_fn: function that returns the chi vector of shape (3, n_q) with the positions
                and orientations of the platform
            operational_space_dynamical_matrices_fn: function that returns Lambda, nu, and JB_pinv describing
                the dynamics in operational space
    """
    # load saved symbolic data
    sym_exps = dill.load(open(str(sym_exp_filepath), "rb"))

    # symbols for robot parameters
    params_syms = sym_exps["params_syms"]

    num_segments = len(params_syms["l"])
    num_rods_per_segment = len(params_syms["rout"]) // num_segments

    @jit
    def select_params_for_lambdify_fn(params: Dict[str, Array]) -> List[Array]:
        """
        Select the parameters for lambdify
        Args:
            params: Dictionary of robot parameters
        Returns:
            params_for_lambdify: list of with each robot parameter
        """
        params_for_lambdify = []
        for params_key, params_vals in sorted(params.items()):
            if params_key in params_syms.keys():
                for param in params_vals.flatten():
                    params_for_lambdify.append(param)
        return params_for_lambdify

    # concatenate the robot params symbols
    params_syms_cat = concatenate_params_syms(params_syms)

    # number of degrees of freedom
    n_xi = len(sym_exps["state_syms"]["xi"])

    # compute the strain basis
    if strain_selector is None:
        strain_selector = jnp.ones((n_xi,), dtype=bool)
    else:
        assert strain_selector.shape == (n_xi,)
    B_xi = compute_strain_basis(strain_selector)

    # concatenate the list of state symbols
    state_syms_cat = sym_exps["state_syms"]["xi"] + sym_exps["state_syms"]["xi_d"]

    # lambdify symbolic expressions
    chiv_lambda_sms = []
    # iterate through symbolic expressions for each segment
    for chiv_exp in sym_exps["exps"]["chiv_sms"]:
        chiv_lambda = sp.lambdify(
            params_syms_cat
            + sym_exps["state_syms"]["xi"]
            + [sym_exps["state_syms"]["s"]],
            chiv_exp,
            "jax",
        )
        chiv_lambda_sms.append(chiv_lambda)

    chir_lambda_sms = []
    # iterate through symbolic expressions for each segment
    for chir_exp in sym_exps["exps"]["chir_sms"]:
        chir_lambda = sp.lambdify(
            params_syms_cat
            + sym_exps["state_syms"]["xi"]
            + [sym_exps["state_syms"]["s"]],
            chir_exp,
            "jax",
        )
        chir_lambda_sms.append(chir_lambda)

    chip_lambda_sms = []
    # iterate through symbolic expressions for each segment
    for chip_exp in sym_exps["exps"]["chip_sms"]:
        chip_lambda = sp.lambdify(
            params_syms_cat + sym_exps["state_syms"]["xi"],
            chip_exp,
            "jax",
        )
        chip_lambda_sms.append(chip_lambda)

    # end-effector kinematics
    chiee_lambda = sp.lambdify(
        params_syms_cat + sym_exps["state_syms"]["xi"],
        sym_exps["exps"]["chiee"],
        "jax",
    )
    Jee_lambda = sp.lambdify(
        params_syms_cat + sym_exps["state_syms"]["xi"],
        sym_exps["exps"]["Jee"],
        "jax",
    )
    Jee_d_lambda = sp.lambdify(
        params_syms_cat + sym_exps["state_syms"]["xi"] + sym_exps["state_syms"]["xi_d"],
        sym_exps["exps"]["Jee_d"],
        "jax",
    )

    B_lambda = sp.lambdify(
        params_syms_cat + sym_exps["state_syms"]["xi"], sym_exps["exps"]["B"], "jax"
    )
    C_lambda = sp.lambdify(
        params_syms_cat + state_syms_cat, sym_exps["exps"]["C"], "jax"
    )
    G_lambda = sp.lambdify(
        params_syms_cat + sym_exps["state_syms"]["xi"], sym_exps["exps"]["G"], "jax"
    )
    K_lambda = sp.lambdify(
        params_syms_cat + sym_exps["state_syms"]["xi"], sym_exps["exps"]["K"], "jax"
    )
    D_lambda = sp.lambdify(params_syms_cat, sym_exps["exps"]["D"], "jax")
    alpha_lambda = sp.lambdify(
        params_syms_cat + sym_exps["state_syms"]["xi"] + sym_exps["state_syms"]["phi"],
        sym_exps["exps"]["alpha"],
        "jax",
    )

    @jit
    def beta_fn(params: Dict[str, Array], vxi: Array) -> Array:
        """
        Map the generalized coordinates to the strains in the physical rods
        Args:
            params: Dictionary of robot parameters
            vxi: strains of the virtual backbone of shape (n_xi, )
        Returns:
            pxi: strains in the physical rods of shape (num_segments, num_rods_per_segment, 3)
        """
        # strains of the virtual rod
        vxi = vxi.reshape((num_segments, 1, -1))

        pxi = jnp.repeat(vxi, num_rods_per_segment, axis=1)
        psigma_a = (
            pxi[:, :, 2]
            + params["roff"] * jnp.repeat(vxi, num_rods_per_segment, axis=1)[..., 0]
        )
        pxi = pxi.at[:, :, 2].set(psigma_a)

        return pxi

    @jit
    def beta_inv_fn(params: Dict[str, Array], pxi: Array) -> Array:
        """
        Map the strains in the physical rods to the strains of the virtual backbone
        Args:
            params: Dictionary of robot parameters
            pxi: strains in the physical rods of shape (num_segments, num_rods_per_segment, 3)
        Returns:
            vxi: strains of the virtual backbone of shape (n_xi, )
        """
        vxi = jnp.mean(pxi, axis=1)
        vxi = vxi.at[:, 2].set(
            vxi[:, 2] - jnp.mean(params["roff"] * pxi[..., 0], axis=1)
        )
        vxi = vxi.flatten()

        return vxi

    @jit
    def rest_strains_fn(params: Dict[str, Array]) -> Array:
        """
        Compute the rest strains of the virtual backbone
        Args:
            params: Dictionary of robot parameters

        Returns:
            vxi_eq: rest strains of the virtual backbone of shape (n_xi, )
        """
        # rest strains of the physical rods
        pxi_eq = jnp.zeros((num_segments, num_rods_per_segment, 3))
        pxi_eq = pxi_eq.at[:, :, 0].set(params["kappa_b_eq"])
        pxi_eq = pxi_eq.at[:, :, 1].set(params["sigma_sh_eq"])
        pxi_eq = pxi_eq.at[:, :, 2].set(params["sigma_a_eq"])

        # map the rest strains from the physical rods to the virtual backbone
        vxi_eq = beta_inv_fn(params, pxi_eq)
        return vxi_eq

    @jit
    def configuration_to_strains_fn(params: Dict[str, Array], q: Array) -> Array:
        """
        Map the generalized coordinates to the strains in the virtual backbone
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (n_q, )
        Returns:
            xi: strains of the virtual backbone of shape (n_xi, )
        """
        # rest strains of the virtual backbone
        xi_eq = rest_strains_fn(params)

        # map the configuration to the strains
        xi = xi_eq + B_xi @ q

        return xi

    @jit
    def apply_eps_to_bend_strains_fn(xi: Array, _eps: float) -> Array:
        """
        Add a small number to the bending strain to avoid singularities
        """
        xi_reshaped = xi.reshape((-1, 3))

        xi_epsed = xi_reshaped
        xi_bend_sign = jnp.sign(xi_reshaped[:, 0])
        # set zero sign to 1 (i.e. positive)
        xi_bend_sign = jnp.where(xi_bend_sign == 0, 1, xi_bend_sign)
        # add eps to the bending strain (i.e. the first column)
        xi_epsed = xi_epsed.at[:, 0].add(xi_bend_sign * _eps)

        # flatten the array
        xi_epsed = xi_epsed.flatten()

        return xi_epsed

    @jit
    def forward_kinematics_virtual_backbone_fn(
        params: Dict[str, Array], q: Array, s: Array
    ) -> Array:
        """
        Evaluate the forward kinematics the virtual backbone
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (n_q, )
            s: point coordinate along the rod in the interval [0, L].
        Returns:
            chi: pose of the backbone point in Cartesian-space with shape (3, )
                Consists of [p_x, p_y, theta]
                where p_x is the x-position, p_y is the y-position,
                and theta is the planar orientation with respect to the x-axis
        """
        # map the configuration to the strains
        xi = configuration_to_strains_fn(params, q)

        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains_fn(xi, eps)

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

        # convert the dictionary of parameters to a list, which we can pass to the lambda function
        params_for_lambdify = select_params_for_lambdify_fn(params)

        chi = lax.switch(
            segment_idx, chiv_lambda_sms, *params_for_lambdify, *xi_epsed, s_segment
        ).squeeze()

        return chi

    @jit
    def forward_kinematics_rod_fn(
        params: Dict[str, Array], q: Array, s: Array, rod_idx: Array
    ) -> Array:
        """
        Evaluate the forward kinematics of the physical rods
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (n_q, )
            s: point coordinate along the rod in the interval [0, L].
            rod_idx: index of the rod. If there are two rods per segment, then rod_idx can be 0 or 1.
        Returns:
            chir: pose of the rod centerline point in Cartesian-space with shape (3, )
                Consists of [p_x, p_y, theta]
                where p_x is the x-position, p_y is the y-position,
                and theta is the planar orientation with respect to the x-axis
        """
        # map the configuration to the strains
        xi = configuration_to_strains_fn(params, q)

        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains_fn(xi, eps)

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

        # convert the dictionary of parameters to a list, which we can pass to the lambda function
        params_for_lambdify = select_params_for_lambdify_fn(params)

        chir_lambda_sms_idx = segment_idx * num_rods_per_segment + rod_idx
        chir = lax.switch(
            chir_lambda_sms_idx,
            chir_lambda_sms,
            *params_for_lambdify,
            *xi_epsed,
            s_segment,
        ).squeeze()

        return chir

    @jit
    def forward_kinematics_platform_fn(
        params: Dict[str, Array], q: Array, segment_idx: Array
    ) -> Array:
        """
        Evaluate the forward kinematics the platform
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (n_q, )
            segment_idx: index of the segment
        Returns:
            chip: pose of the CoG of the platform in Cartesian-space with shape (3, )
                Consists of [p_x, p_y, theta]
                where p_x is the x-position, p_y is the y-position,
                and theta is the planar orientation with respect to the x-axis
        """
        # map the configuration to the strains
        xi = configuration_to_strains_fn(params, q)

        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains_fn(xi, eps)

        # convert the dictionary of parameters to a list, which we can pass to the lambda function
        params_for_lambdify = select_params_for_lambdify_fn(params)

        chip = lax.switch(
            segment_idx, chip_lambda_sms, *params_for_lambdify, *xi_epsed
        ).squeeze()

        return chip

    @jit
    def forward_kinematics_end_effector_fn(params: Dict[str, Array], q: Array) -> Array:
        """
        Evaluate the forward kinematics of the end-effector
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (n_q, )
        Returns:
            chiee: pose of the end-effector in Cartesian-space of shape (3, )
                Consists of [p_x, p_y, theta]
                where p_x is the x-position, p_y is the y-position,
                and theta is the planar orientation with respect to the x-axis
        """
        # map the configuration to the strains
        xi = configuration_to_strains_fn(params, q)
        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains_fn(xi, eps)

        # convert the dictionary of parameters to a list, which we can pass to the lambda function
        params_for_lambdify = select_params_for_lambdify_fn(params)
        # evaluate the symbolic expression
        chiee = chiee_lambda(*params_for_lambdify, *xi_epsed).squeeze()

        return chiee

    @jit
    def jacobian_end_effector_fn(params: Dict[str, Array], q: Array) -> Array:
        """
        Evaluate the Jacobian of the end-effector
        Args:
           params: Dictionary of robot parameters
           q: generalized coordinates of shape (n_q, )
        Returns:
           Jee: the Jacobian of the end-effector pose with respect to the generalized coordinates.
                Jee is an array of shape (3, n_q).
        """
        # map the configuration to the strains
        xi = configuration_to_strains_fn(params, q)
        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains_fn(xi, eps)

        # convert the dictionary of parameters to a list, which we can pass to the lambda function
        params_for_lambdify = select_params_for_lambdify_fn(params)
        # evaluate the symbolic expression
        Jee = Jee_lambda(*params_for_lambdify, *xi_epsed)

        return Jee

    @jit
    def inverse_kinematics_end_effector_fn(
        params: Dict[str, Array], chiee: Array
    ) -> Array:
        """
        Evaluates the inverse kinematics for a given end-effector pose.
            Important: only works for one segment!
        Args:
           params: Dictionary of robot parameters
           chiee: pose of the end-effector in Cartesian-space of shape (3, )
        Returns:
           q: generalized coordinates of shape (n_q, )
        """
        assert num_segments == 1, "Inverse kinematics only works for one segment!"

        # height of platform
        hp = params["pcudim"][0, 1]
        # length of the proximal rod caps
        lpc = params["lpc"][0]
        # length of the distal rod caps
        ldc = params["ldc"][0]

        # compute the pose of the proximal end of the virtual backbone
        vchi_pe = jnp.array([0, lpc, 0.0])

        # compute the pose of the distal end of the virtual backbone
        vchi_de = chiee + jnp.array(
            [jnp.sin(chiee[2]) * (ldc + hp), -jnp.cos(chiee[2]) * (ldc + hp), 0.0]
        )

        # offset the pose of the distal end of the virtual backbone by the pose of the proximal end
        vchi_pe_to_de = vchi_de - vchi_pe
        px, py, th = vchi_pe_to_de[0], vchi_pe_to_de[1], vchi_pe_to_de[2]

        # add small eps for numerical stability
        th_sign = jnp.sign(th)
        # set zero sign to 1 (i.e. positive)
        th_sign = jnp.where(th_sign == 0, 1, th_sign)
        # add eps to the bending strain (i.e. the first column)
        th_epsed = th + th_sign * eps

        # compute the inverse kinematics for the virtual backbone
        vxi = (
            th_epsed
            / (2 * params["l"].sum())
            * jnp.array(
                [
                    2,
                    py - (px * jnp.sin(th_epsed)) / (jnp.cos(th_epsed) - 1),
                    -px - (py * jnp.sin(th_epsed)) / (jnp.cos(th_epsed) - 1),
                ]
            )
        )

        # rest strains of the virtual backbone
        vxi_eq = rest_strains_fn(params)

        # map the strains to the generalized coordinates
        q = jnp.linalg.pinv(B_xi) @ (vxi - vxi_eq)

        return q

    @jit
    def dynamical_matrices_fn(
        params: Dict[str, Array],
        q: Array,
        q_d: Array,
        phi: Array = jnp.zeros((num_segments * num_rods_per_segment,)),
    ) -> Tuple[Array, Array, Array, Array, Array, Array]:
        """
        Compute the dynamical matrices of the system.
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (n_q, )
            q_d: generalized velocities of shape (n_q, )
            phi: motor positions / twist angles of shape (num_segments * num_rods_per_segment, )
        Returns:
            B: mass / inertia matrix of shape (n_q, n_q)
            C: coriolis / centrifugal matrix of shape (n_q, n_q)
            G: gravity vector of shape (n_q, )
            K: elastic vector of shape (n_q, )
            D: dissipative matrix of shape (n_q, n_q)
            alpha: actuation vector acting on the generalized coordinates of shape (n_q, )
        """
        # map the configuration to the strains
        xi = configuration_to_strains_fn(params, q)
        xi_d = B_xi @ q_d

        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains_fn(xi, 1e4 * eps)

        # evaluate the symbolic expressions
        params_for_lambdify = select_params_for_lambdify_fn(params)
        B = B_lambda(*params_for_lambdify, *xi_epsed)
        C_xi = C_lambda(*params_for_lambdify, *xi_epsed, *xi_d)
        G = G_lambda(*params_for_lambdify, *xi_epsed).squeeze()
        K = K_lambda(*params_for_lambdify, *xi_epsed).squeeze()
        D = D_lambda(*params_for_lambdify)
        alpha = alpha_lambda(*params_for_lambdify, *xi_epsed, *phi).squeeze()

        # apply the strain basis
        B = B_xi.T @ B @ B_xi
        C = B_xi.T @ C_xi @ B_xi
        G = B_xi.T @ G
        K = B_xi.T @ K
        D = B_xi.T @ D @ B_xi
        alpha = B_xi.T @ alpha

        return B, C, G, K, D, alpha

    def operational_space_dynamical_matrices_fn(
        params: Dict[str, Array], q: Array, q_d: Array, B: Array, C: Array
    ):
        """
        Compute the dynamics in operational space.
        The implementation is based on Chapter 7.8 of "Modelling, Planning and Control of Robotics" by
        Siciliano, Sciavicco, Villani, Oriolo.
        Args:
            params: dictionary of parameters
            q: generalized coordinates of shape (n_q,)
            q_d: generalized velocities of shape (n_q,)
            B: inertia matrix in the generalized coordinates of shape (n_q, n_q)
            C: coriolis matrix derived with Christoffer symbols in the generalized coordinates of shape (n_q, n_q)

        Returns:
            Lambda: inertia matrix in the operational space of shape (n_x, n_x)
            nu: coriolis vector in the operational space of shape (n_x, )
            JB_pinv: Dynamically-consistent pseudo-inverse of the Jacobian. Allows the mapping of torques
                from the generalized coordinates to the operational space: f = JB_pinv.T @ tau_q
                Shape (n_q, n_x)
        """
        # map the configuration to the strains
        xi = configuration_to_strains_fn(params, q)
        xi_d = B_xi @ q_d
        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains_fn(xi, eps)

        # convert the dictionary of parameters to a list, which we can pass to the lambda function
        params_for_lambdify = select_params_for_lambdify_fn(params)

        # end-effector Jacobian and its time-derivative
        Jee = Jee_lambda(*params_for_lambdify, *xi_epsed)
        Jee_d = Jee_d_lambda(*params_for_lambdify, *xi_epsed, *xi_d)

        # inverse of the inertia matrix in the configuration space
        B_inv = jnp.linalg.inv(B)

        Lambda = jnp.linalg.inv(Jee @ B_inv @ Jee.T)
        nu = Lambda @ (Jee @ B_inv @ C - Jee_d) @ q_d

        JB_pinv = B_inv @ Jee.T @ Lambda

        return Lambda, nu, JB_pinv

    sys_helpers = {
        "eps": eps,
        "select_params_for_lambdify_fn": select_params_for_lambdify_fn,
        "beta_fn": beta_fn,
        "beta_inv_fn": beta_inv_fn,
        "rest_strains_fn": rest_strains_fn,
        "B_xi": B_xi,
        "configuration_to_strains_fn": configuration_to_strains_fn,
        "apply_eps_to_bend_strains_fn": apply_eps_to_bend_strains_fn,
        "forward_kinematics_rod_fn": forward_kinematics_rod_fn,
        "forward_kinematics_platform_fn": forward_kinematics_platform_fn,
        "operational_space_dynamical_matrices_fn": operational_space_dynamical_matrices_fn,
    }

    return (
        forward_kinematics_virtual_backbone_fn,
        forward_kinematics_end_effector_fn,
        jacobian_end_effector_fn,
        inverse_kinematics_end_effector_fn,
        dynamical_matrices_fn,
        sys_helpers,
    )


def ode_factory(
    dynamical_matrices_fn: Callable,
    params: Dict[str, Array],
    consider_underactuation_model: bool = True,
) -> Callable[[float, Array], Array]:
    """
    Make an ODE function of the form ode_fn(t, x) -> x_dot.
    This function assumes a constant torque input (i.e. zero-order hold).
    Args:
        dynamical_matrices_fn: Callable that returns B, C, G, K, D, alpha_fn. Needs to conform to the signature:
            dynamical_matrices_fn(params, q, q_d) -> Tuple[B, C, G, K, D, A]
            where q and q_d are the configuration and velocity vectors, respectively,
            B is the inertia matrix of shape (n_q, n_q),
            C is the Coriolis matrix of shape (n_q, n_q),
            G is the gravity vector of shape (n_q, ),
            K is the stiffness vector of shape (n_q, ),
            D is the damping matrix of shape (n_q, n_q),
            alpha_fn is a function to compute the actuation vector of shape (n_q). It has the following signature:
                alpha_fn(phi) -> tau_q where phi is the twist angle vector of shape (n_phi, )
        params: Dictionary with robot parameters
        consider_underactuation_model: If True, the underactuation model is considered. Otherwise, the fully-actuated
            model is considered with the identity matrix as the actuation matrix.
    Returns:
        ode_fn: ODE function of the form ode_fn(t, x) -> x_dot
    """
    num_rods = params["rout"].shape[0] * params["rout"].shape[1]

    @jit
    def ode_fn(t: float, x: Array, u: Array) -> Array:
        """
        ODE of the dynamical Lagrangian system.
        Args:
            t: time
            x: state vector of shape (2 * n_q, )
            u: input to the system.
                - if consider_underactuation_model is True, then this is an array of shape (n_phi) with
                    motor positions / twist angles of the proximal end of the rods
                - if consider_underactuation_model is False, then this is an array of shape (n_q) with
                    the generalized torques
        Returns:
            x_d: time-derivative of the state vector of shape (2 * n_q, )
        """
        n_q = x.shape[0] // 2
        q, q_d = x[:n_q], x[n_q:]

        if consider_underactuation_model is True:
            phi = u
            B, C, G, K, D, tau_q = dynamical_matrices_fn(params, q, q_d, phi)
        else:
            B, C, G, K, D, _ = dynamical_matrices_fn(
                params, q, q_d, jnp.zeros((num_rods,))
            )
            tau_q = u

        # inverse of B
        B_inv = jnp.linalg.inv(B)

        # compute the acceleration
        q_dd = B_inv @ (tau_q - C @ q_d - G - K - D @ q_d)

        x_d = jnp.concatenate([x[n_q:], q_dd])

        return x_d

    return ode_fn
