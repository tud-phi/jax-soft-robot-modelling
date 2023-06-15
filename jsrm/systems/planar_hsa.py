import dill
import jax
from jax import Array, debug, jit, lax, vmap
from jax import numpy as jnp
import sympy as sp
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple, Union

from .utils import compute_strain_basis, compute_planar_stiffness_matrix
from jsrm.math_utils import blk_diag


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
    Callable[
        [Dict[str, Array], Array, Array],
        Tuple[Array, Array, Array, Array, Array, Array],
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
        dynamical_matrices_fn: function that returns the B, C, G, K, D, and alpha matrices
    """
    # load saved symbolic data
    sym_exps = dill.load(open(str(filepath), "rb"))

    # symbols for robot parameters
    params_syms = sym_exps["params_syms"]

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
            params_syms_cat
            + sym_exps["state_syms"]["xi"],
            chip_exp,
            "jax",
        )
        chip_lambda_sms.append(chip_lambda)

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
        vmap(compute_planar_stiffness_matrix, in_axes=(0, 0, 0, 0), out_axes=0), in_axes=(0, 0, 0, 0), out_axes=0
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
    def forward_kinematics_virtual_backbone_fn(params: Dict[str, Array], q: Array, s: Array) -> Array:
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
    def forward_kinematics_rod_fn(params: Dict[str, Array], q: Array, s: Array, rod_idx: Array) -> Array:
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
        num_segments, num_rods_per_segment = params["rout"].shape[0], params["rout"].shape[1]

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

        chir_lambda_sms_idx = segment_idx*num_rods_per_segment + rod_idx
        chir = lax.switch(
            chir_lambda_sms_idx, chir_lambda_sms, *params_for_lambdify, *xi_epsed, s_segment
        ).squeeze()

        return chir

    @jit
    def forward_kinematics_platform_fn(params: Dict[str, Array], q: Array, segment_idx: Array) -> Array:
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
    def beta_fn(params: Dict[str, Array], vxi: Array) -> Array:
        """
        Map the generalized coordinates to the strains in the physical rods
        Args:
            params: Dictionary of robot parameters
            vxi: strains of the virtual backbone of shape (n_xi, )
        Returns:
            pxi: strains in the physical rods of shape (num_segments, num_rods_per_segment, 3)
        """
        num_segments, num_rods_per_segment = params["rout"].shape[0], params["rout"].shape[1]

        # strains of the virtual rod
        vxi = vxi.reshape((num_segments, 1, -1))

        pxi = jnp.repeat(vxi, num_rods_per_segment, axis=1)
        psigma_a = pxi[:, :, 2] + params["roff"] * jnp.repeat(vxi, num_rods_per_segment, axis=1)[..., 0]
        pxi = pxi.at[:, :, 2].set(psigma_a)

        return pxi

    @jit
    def dynamical_matrices_fn(
        params: Dict[str, Array], q: Array, q_d: Array
    ) -> Tuple[Array, Array, Array, Array, Array, Array]:
        """
        Compute the dynamical matrices of the system.
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (n_q, )
            q_d: generalized velocities of shape (n_q, )
        Returns:
            B: mass / inertia matrix of shape (n_q, n_q)
            C: coriolis / centrifugal matrix of shape (n_q, n_q)
            G: gravity vector of shape (n_q, )
            K: elastic vector of shape (n_q, )
            D: dissipative matrix of shape (n_q, n_q)
            alpha: actuation matrix of shape (n_q, n_tau)
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

        # offset of rods from the centerline
        roff = params["roff"]  # shape (num_segments, num_rods_per_segment)
        # handedness of the rods
        h = params["h"]  # shape (num_segments, num_rods_per_segment)

        # cross-sectional area and second moment of area for bending
        A = jnp.pi * (params["rout"] ** 2 - params["rin"] ** 2)
        Ib = jnp.pi / 4 * (params["rout"] ** 4 - params["rin"] ** 4)

        # volumetric mass density
        # nominal elastic and shear modulus
        Ehat, Ghat = params["E"], params["G"]
        # scale factors for the elastic and shear modulus
        C_E, C_G = params["C_E"], params["C_G"]
        # compute the elastic and shear modulus for each rod
        # will be arrays of shape (num_segments, num_rods_per_segment)
        # E = Ehat + C_E * h * (pxi[..., ] - pxi_eq)  # TODO fix this for current convention
        # G = Ghat + C_G * h * (xi_epsed[1::3] - xi_eq[1::3])

        # stiffness matrix of shape (num_segments, 3)
        Shat = compute_stiffness_matrix_for_all_rods_fn(A, Ib, Ehat, Ghat)

        # Jacobian of the strain of the physical HSA rods with respect to the configuration variables
        J_beta = jnp.zeros((num_segments, num_rods_per_segment, 3, 3))
        J_beta = J_beta.at[..., 0, 0].set(1.0)
        J_beta = J_beta.at[..., 1, 1].set(1.0)
        J_beta = J_beta.at[..., 2, 2].set(1.0)
        J_beta = J_beta.at[..., 2, 0].set(roff)

        # we define the elastic matrix of the physical rods of shape (n_xi, n_xi) as K(xi) = K @ xi where K is equal to
        vK = vmap(  # vmap over the segments
            vmap(  # vmap over the rods of each segment
                lambda _J_beta, _S, _pxi, _pxi_eq: _J_beta.T @ _S @ (_pxi - _pxi_eq),
                in_axes=(0, 0, 0, 0),
                out_axes=0,
            ),
            in_axes=(0, 0, 0, 0),
            out_axes=0,
        )(J_beta, Shat, pxi, pxi_eq)  # shape (num_segments, num_rods_per_segment, 3)
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
        )(J_beta, zeta)  # shape (num_segments, num_rods_per_segment, 3, 3)
        # dissipative matrix
        D = blk_diag(jnp.sum(vD, axis=1))  # shape (n_xi, n_xi)

        params_for_lambdify = select_params_for_lambdify(params)

        B = B_xi.T @ B_lambda(*params_for_lambdify, *xi_epsed) @ B_xi
        C_xi = C_lambda(*params_for_lambdify, *xi_epsed, *xi_d)
        C = B_xi.T @ C_xi @ B_xi
        G = B_xi.T @ G_lambda(*params_for_lambdify, *xi_epsed).squeeze()

        # apply the strain basis to the elastic and dissipative matrices
        K = B_xi.T @ K
        D = B_xi.T @ D @ B_xi

        # apply the strain basis to the actuation matrix
        alpha = B_xi.T @ jnp.identity(n_xi) @ B_xi

        return B, C, G, K, D, alpha

    return (
        B_xi,
        forward_kinematics_virtual_backbone_fn, forward_kinematics_rod_fn, forward_kinematics_platform_fn,
        dynamical_matrices_fn
    )
