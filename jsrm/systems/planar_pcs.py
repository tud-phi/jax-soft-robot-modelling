import dill
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
import jax
from jax import Array, debug, jit, lax, vmap
from jax import numpy as jnp
import sympy as sp
from pathlib import Path
from typing import Callable, Dict, Iterable, Sequence, Tuple, Union

from .utils import params_dict_to_list, compute_strain_basis


def make_jax_functions(
        filepath: Union[str, Path],
        strain_selector: Array = None,
        xi0: Array = None,
        eps: float = 1e-6,
) -> Tuple[Callable, Callable]:
    """
    Create jax functions from file containing symbolic expressions.
    Args:
        filepath: path to file containing symbolic expressions
        strain_selector: array of shape (3, ) with boolean values indicating which components of the
                strain are active / non-zero
        xi0: array of shape (3 * num_segments) with the rest strains of the rod
        eps: small number to avoid division by zero
    Returns:
        forward_kinematics_fn: function that returns the p vector of shape (3, n_q) with the positions
        dynamical_matrices_fn: function that returns the B, C, G, K, D, and A matrices
    """
    # load saved symbolic data
    sym_exps = dill.load(open(str(filepath), "rb"))

    # symbols for robot parameters
    params_syms = sym_exps["params_syms"]
    # symbols of state variables
    state_syms = sym_exps["state_syms"]
    # symbolic expressions
    exps = sym_exps["exps"]

    # concatenate the robot params symbols
    params_syms_cat = []
    for params_key, params_sym in params_syms.items():
        params_syms_cat += params_sym

    # number of degrees of freedom
    n_xi = len(sym_exps["state_syms"]["xi"])

    # compute the strain basis
    if strain_selector is None:
        strain_selector = jnp.ones((n_xi,), dtype=bool)
    else:
        assert strain_selector.shape == (n_xi,)
    B_xi = compute_strain_basis(strain_selector)

    # initialize the rest strain
    if xi0 is None:
        xi0 = jnp.zeros((n_xi,))
        # by default, set the axial rest strain (local y-axis) along the entire rod to 1.0
        rest_strain_reshaped = xi0.reshape((-1, 3))
        rest_strain_reshaped = rest_strain_reshaped.at[:, -1].set(1.0)
        xi0 = rest_strain_reshaped.flatten()
    else:
        assert xi0.shape == (n_xi,)

    # concatenate the list of state symbols
    state_syms_cat = sym_exps["state_syms"]["xi"] + sym_exps["state_syms"]["xi_d"]

    # lambdify symbolic expressions
    chi_lambda_sms = []
    # iterate through symbolic expressions for each segment
    for chi_exp in sym_exps["exps"]["chi_sms"]:
        chi_lambda = sp.lambdify(
            params_syms_cat + sym_exps["state_syms"]["xi"] + [sym_exps["state_syms"]["s"]],
            chi_exp,
            "jax"
        )
        chi_lambda_sms.append(chi_lambda)

    B_lambda = sp.lambdify(params_syms_cat + sym_exps["state_syms"]["xi"], sym_exps["exps"]["B"], "jax")
    C_lambda = sp.lambdify(params_syms_cat + state_syms_cat, sym_exps["exps"]["C"], "jax")
    G_lambda = sp.lambdify(params_syms_cat + sym_exps["state_syms"]["xi"], sym_exps["exps"]["G"], "jax")

    @jit
    def forward_kinematics_fn(params: Dict[str, Array], q: Array, s: float) -> Array:
        """
        Evaluate the forward kinematics the tip of the links
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (n_q, )
            s: point coordinate along the rod in the interval [0, L].
        Returns:
            chi_sms: poses of tip of links of shape (3, n_q) consisting of [p_x, p_y, theta]
                where p_x is the x-position, p_y is the y-position,
                and theta is the planar orientation with respect to the x-axis
        """
        # map the configuration to the strains
        xi = xi0 + B_xi @ q

        # make sure that we prevent singularities
        xi = xi + jnp.sign(xi + eps) * eps

        # cumsum of the segment lengths
        l_cum = jnp.cumsum(params["l"])
        # determine in which segment the point is located
        # use argmax to find the last index where the condition is true
        segment_idx = jnp.argmax((s >= l_cum)[::-1]).astype(int)
        # add zero to the beginning of the array
        l_cum_padded = jnp.concatenate([jnp.array([0.0]), l_cum], axis=0)
        # point coordinate along the segment in the interval [0, l_segment]
        s_segment = s - l_cum_padded[segment_idx]

        # convert the dictionary of parameters to a list, which we can pass to the lambda function
        params_ls = params_dict_to_list(params)

        chi = lax.switch(
            segment_idx,
            chi_lambda_sms,
            *params_ls, *xi, s_segment
        ).squeeze()

        return chi

    # actuation matrix
    A = jnp.identity(n_xi)

    @jit
    def dynamical_matrices_fn(
            params: Dict[str, Array],
            q: Array,
            q_d: Array
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
            A: actuation matrix of shape (n_q, n_tau)
        """
        # map the configuration to the strains
        xi = xi0 + B_xi @ q
        xi_d = B_xi @ q_d

        # make sure that we prevent singularities
        xi = xi + jnp.sign(xi + eps) * eps

        # elastic and dissipative matrices
        K = params.get("K", jnp.zeros((n_xi, n_xi)))
        D = params.get("D", jnp.zeros((n_xi, n_xi)))

        params_ls = params_dict_to_list(params)

        B = B_lambda(*params_ls, *xi)
        C = C_lambda(*params_ls, *xi, *xi_d)
        G = G_lambda(*params_ls, *xi).squeeze()

        _K = K @ xi

        return B, C, G, _K, D, A

    return forward_kinematics_fn, dynamical_matrices_fn
