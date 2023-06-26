import dill
import jax
from jax import Array, debug, jit, lax, vmap
from jax import numpy as jnp
import sympy as sp
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple, Union

from .utils import compute_strain_basis, compute_planar_stiffness_matrix
from jsrm.math_utils import blk_diag
from jsrm.systems import euler_lagrangian


def factory(
    filepath: Union[str, Path],
    strain_selector: Array = None,
    xi_eq: Array = None,
    eps: float = 1e-6,
) -> Tuple[
    Array,
    Callable[[Dict[str, Array], Array, Array], Array],
    Callable[[Dict[str, Array], Array, Array, Array], Array],
    Callable[[Dict[str, Array], Array, Array], Array],
    Callable[[Dict[str, Array], Array], Array],
    Callable[[Dict[str, Array], Array], Array],
    Callable[[Dict[str, Array], Array], Array],
    Callable[
        [Dict[str, Array], Array, Array],
        Tuple[Array, Array, Array, Array, Array, Array],
    ],
    Callable[
        [Dict[str, Array], Array, Array, Array, Array],
        Tuple[Array, Array, Array],
    ],
]:
    """
    Create jax functions from file containing symbolic expressions.
    Args:
        filepath: path to file containing symbolic expressions
        strain_selector: array of shape (3, ) with boolean values indicating which components of the
                strain are active / non-zero
        xi_eq: array of shape (3 * num_segments) with the rest strains of the rod
        eps: small number to avoid division by zero
    Returns:
        B_xi: strain basis matrix of shape (3 * num_segments, n_q)
        forward_kinematics_virtual_backbone_fn: function that returns the chi vector of shape (3, n_q) with the
            positions and orientations of the virtual backbone
        forward_kinematics_rod_fn: function that returns the chi vector of shape (3, n_q) with the
            positions and orientations of the rod
        forward_kinematics_platform_fn: function that returns the chi vector of shape (3, n_q) with the positions
            and orientations of the platform
        forward_kinematics_end_effector_fn: function that returns the pose of the end effector of shape (3, )
        jacobian_end_effector_fn: function that returns the Jacobian of the end effector of shape (3, n_q)
        inverse_kinematics_end_effector_fn: function that returns the generalized coordinates for a given end-effector pose
        dynamical_matrices_fn: function that returns the B, C, G, K, D, and alpha matrices
        operational_space_dynamical_matrices_fn: function that returns Lambda, nu, and JB_pinv describing
            the dynamics in operational space
    """
    # load saved symbolic data
    sym_exps = dill.load(open(str(filepath), "rb"))

    # symbols for robot parameters
    params_syms = sym_exps["params_syms"]

    num_segments = len(params_syms["l"])
    num_rods_per_segment = len(params_syms["rout"]) // num_segments

    @jit
    def select_params_for_lambdify(params: Dict[str, Array]) -> List[Array]:
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
    params_syms_cat = []
    for params_key, params_sym in sorted(params_syms.items()):
        if type(params_sym) in [list, tuple]:
            params_syms_cat += params_sym
        else:
            params_syms_cat.append(params_sym)

    # number of degrees of freedom
    n_xi = len(sym_exps["state_syms"]["xi"])

    # compute the strain basis
    if strain_selector is None:
        strain_selector = jnp.ones((n_xi,), dtype=bool)
    else:
        assert strain_selector.shape == (n_xi,)
    B_xi = compute_strain_basis(strain_selector)

    # initialize the rest strain
    if xi_eq is None:
        xi_eq = jnp.zeros((n_xi,))
        # by default, set the axial rest strain (local y-axis) along the entire rod to 1.0
        rest_strain_reshaped = xi_eq.reshape((-1, 3))
        rest_strain_reshaped = rest_strain_reshaped.at[:, -1].set(1.0)
        xi_eq = rest_strain_reshaped.flatten()
    else:
        assert xi_eq.shape == (n_xi,)

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

    compute_stiffness_matrix_for_all_rods_fn = vmap(
        vmap(compute_planar_stiffness_matrix, in_axes=(0, 0, 0, 0), out_axes=0),
        in_axes=(0, 0, 0, 0),
        out_axes=0,
    )

    @jit
    def apply_eps_to_bend_strains(xi: Array, _eps: float) -> Array:
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
        xi = xi_eq + B_xi @ q

        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains(xi, eps)

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
        params_for_lambdify = select_params_for_lambdify(params)

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
        xi = xi_eq + B_xi @ q

        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains(xi, eps)

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
        params_for_lambdify = select_params_for_lambdify(params)

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
        xi = xi_eq + B_xi @ q

        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains(xi, eps)

        # convert the dictionary of parameters to a list, which we can pass to the lambda function
        params_for_lambdify = select_params_for_lambdify(params)

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
        xi = xi_eq + B_xi @ q
        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains(xi, eps)

        # convert the dictionary of parameters to a list, which we can pass to the lambda function
        params_for_lambdify = select_params_for_lambdify(params)
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
        xi = xi_eq + B_xi @ q
        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains(xi, eps)

        # convert the dictionary of parameters to a list, which we can pass to the lambda function
        params_for_lambdify = select_params_for_lambdify(params)
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

        # map the strains to the generalized coordinates
        q = jnp.linalg.pinv(B_xi) @ (vxi - xi_eq)

        return q

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
            tau_q: actuation torque acting on the generalized coordinates of shape (n_q, )
        """
        # map the configuration to the strains
        xi = xi_eq + B_xi @ q
        xi_d = B_xi @ q_d

        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains(xi, 1e4 * eps)

        # the strains of the physical rods as array of shape (num_segments, num_rods_per_segment, 3)
        pxi = beta_fn(params, xi_epsed)
        pxi_eq = beta_fn(params, xi_eq)  # equilibrium strains of the physical rods

        # number of segments
        num_segments = params["rout"].shape[0]
        num_rods_per_segment = params["rout"].shape[1]

        # printed (i.e. original) length of each segment
        l = params["l"]  # shape (num_segments, )
        # offset of rods from the centerline
        roff = params["roff"]  # shape (num_segments, num_rods_per_segment)
        # handedness of the rods
        h = params["h"]  # shape (num_segments, num_rods_per_segment)

        # reshape phi and l to be of shape (num_segments, num_rods_per_segment)
        phi_per_rod = phi.reshape(num_segments, num_rods_per_segment)
        l_per_rod = jnp.repeat(
            l.reshape(num_segments, 1), axis=1, repeats=num_rods_per_segment
        )

        # change in the rest length
        varepsilon = params["C_varepsilon"] * h / l_per_rod * phi_per_rod

        # cross-sectional area and second moment of area for bending
        A = jnp.pi * (params["rout"] ** 2 - params["rin"] ** 2)
        Ib = jnp.pi / 4 * (params["rout"] ** 4 - params["rin"] ** 4)

        # nominal elastic and shear modulus
        Ehat, Ghat = params["E"], params["G"]
        # difference between the current modulus and the nominal modulus
        Edelta = params["C_E"] * h / l_per_rod * phi_per_rod
        Gdelta = params["C_G"] * h / l_per_rod * phi_per_rod
        # current elastic and shear modulus
        E = Ehat + Edelta
        G = Ghat + Gdelta

        # stiffness matrix of shape (num_segments, 3)
        Shat = compute_stiffness_matrix_for_all_rods_fn(A, Ib, Ehat, Ghat)
        Sdelta = compute_stiffness_matrix_for_all_rods_fn(A, Ib, Edelta, Gdelta)
        S = Shat + Sdelta

        # Jacobian of the strain of the physical HSA rods with respect to the configuration variables
        J_beta = jnp.zeros((num_segments, num_rods_per_segment, 3, 3))
        J_beta = J_beta.at[..., 0, 0].set(1.0)
        J_beta = J_beta.at[..., 1, 1].set(1.0)
        J_beta = J_beta.at[..., 2, 2].set(1.0)
        J_beta = J_beta.at[..., 2, 0].set(roff)

        # we define the elastic matrix of the physical rods of shape (n_xi, n_xi) as K(xi) = K @ xi where K is equal to
        vK = vmap(  # vmap over the segments
            vmap(  # vmap over the rods of each segment
                lambda _J_beta, _Shat, _pxi, _pxi_eq: _J_beta.T
                @ _Shat
                @ (_pxi - _pxi_eq),
                in_axes=(0, 0, 0, 0),
                out_axes=0,
            ),
            in_axes=(0, 0, 0, 0),
            out_axes=0,
        )(
            J_beta, Shat, pxi, pxi_eq
        )  # shape (num_segments, num_rods_per_segment, 3)
        # sum the elastic forces over all rods of each segment
        K = jnp.sum(vK, axis=1).flatten()  # shape (n_xi, )

        # damping coefficients of shape (num_segments, num_rods_per_segment, 3)
        zeta = params.get("zeta", jnp.zeros((num_segments, num_rods_per_segment, 3)))
        vD = vmap(  # vmap over the segments
            vmap(  # vmap over the rods of each segment
                lambda _J_beta, _zeta: _J_beta.T @ jnp.diag(_zeta) @ _J_beta,
                in_axes=(0, 0),
                out_axes=0,
            ),
            in_axes=(0, 0),
            out_axes=0,
        )(
            J_beta, zeta
        )  # shape (num_segments, num_rods_per_segment, 3, 3)
        # dissipative matrix
        D = blk_diag(jnp.sum(vD, axis=1))  # shape (n_xi, n_xi)

        # actuation strain in the physical rods
        pxiphi = jnp.zeros_like(pxi)
        # consider axial strain generated by twisting rod
        pxiphi = pxiphi.at[..., 2].set(varepsilon)

        # compute the actuation torque on the strain of the virtual backbone
        tau_xi = vmap(
            vmap(
                lambda _J_beta, _S, _Sdelta, _pxi, _pxi_eq, _pxiphi: _J_beta.T
                @ (-_Sdelta @ (_pxi - _pxi_eq) + _S @ _pxiphi),
                in_axes=(0, 0, 0, 0, 0, 0),
                out_axes=0,
            ),
            in_axes=(0, 0, 0, 0, 0, 0),
            out_axes=0,
        )(
            J_beta, S, Sdelta, pxi, pxi_eq, pxiphi
        )  # shape (num_segments, num_rods_per_segment, 3)
        tau_xi = tau_xi.sum(
            axis=1
        ).flatten()  # sum over all the rods and then flatten over all the segments

        # apply the strain basis
        params_for_lambdify = select_params_for_lambdify(params)
        B = B_xi.T @ B_lambda(*params_for_lambdify, *xi_epsed) @ B_xi
        C_xi = C_lambda(*params_for_lambdify, *xi_epsed, *xi_d)
        C = B_xi.T @ C_xi @ B_xi
        G = B_xi.T @ G_lambda(*params_for_lambdify, *xi_epsed).squeeze()

        # apply the strain basis to the elastic, and dissipative matrices and the actuation torques
        K = B_xi.T @ K
        D = B_xi.T @ D @ B_xi
        tau_q = B_xi.T @ tau_xi @ B_xi

        # def alpha_fn(phi: Array) -> Array:
        #     """
        #     Compute the actuation vector as a function of the twist angles phi.
        #         tau_q = alpha(phi)
        #
        #     Args:
        #         phi: twist angles of shape (num_rods)
        #
        #     Returns:
        #         tau_q: torque applied on the generalized coordinates
        #     """
        #
        #     _phi = phi.reshape(num_segments, num_rods_per_segment)
        #     _l = jnp.repeat(l.reshape(num_segments, 1), axis=1, repeats=num_rods_per_segment)
        #
        #     # change in the rest length
        #     varepsilon = params["C_varepsilon"] * h * _l * _phi
        #
        #     # difference between the current modulus and the nominal modulus
        #     Edelta = C_E * h * _l * _phi
        #     Gdelta = C_G * h * _l * _phi
        #
        #     Sdelta = compute_stiffness_matrix_for_all_rods_fn(A, Ib, Edelta, Gdelta)
        #     S = Shat + Sdelta
        #
        #     actuation_strain = jnp.zeros_like(pxi)
        #     # consider axial strain generated by twisting rod
        #     actuation_strain = actuation_strain.at[..., 2].set(varepsilon)
        #     tau_xi = J_beta.T @ (-Sdelta @ (pxi - pxi_eq) + S @ actuation_strain)
        #     print("tau_xi before", tau_xi.shape)
        #     tau_xi = tau_xi.sum(axis=1).flatten()  # sum over all the rods and then flatten over all the segments
        #     print("tau_xi", tau_xi.shape)
        #
        #     tau_q = B_xi.T @ tau_xi @ B_xi
        #     return tau_q

        return B, C, G, K, D, tau_q

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
        xi = xi_eq + B_xi @ q
        xi_d = B_xi @ q_d
        # add a small number to the bending strain to avoid singularities
        xi_epsed = apply_eps_to_bend_strains(xi, eps)

        # convert the dictionary of parameters to a list, which we can pass to the lambda function
        params_for_lambdify = select_params_for_lambdify(params)

        # end-effector Jacobian and its time-derivative
        Jee = Jee_lambda(*params_for_lambdify, *xi_epsed)
        Jee_d = Jee_d_lambda(*params_for_lambdify, *xi_epsed, *xi_d)

        # inverse of the inertia matrix in the configuration space
        B_inv = jnp.linalg.inv(B)

        Lambda = jnp.linalg.inv(Jee @ B_inv @ Jee.T)
        nu = Lambda @ (Jee @ B_inv @ C - Jee_d) @ q_d

        JB_pinv = B_inv @ Jee.T @ Lambda

        return Lambda, nu, JB_pinv

    return (
        B_xi,
        forward_kinematics_virtual_backbone_fn,
        forward_kinematics_rod_fn,
        forward_kinematics_platform_fn,
        forward_kinematics_end_effector_fn,
        jacobian_end_effector_fn,
        inverse_kinematics_end_effector_fn,
        dynamical_matrices_fn,
        operational_space_dynamical_matrices_fn,
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
    def ode_fn(t: float, x: Array, *args, u: Array) -> Array:
        """
        ODE of the dynamical Lagrangian system.
        Args:
            t: time
            x: state vector of shape (2 * n_q, )
            args: additional arguments
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
