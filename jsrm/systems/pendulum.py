import dill
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
import jax
from jax import Array, jit, vmap
from jax import numpy as jnp
import sympy as sp
from pathlib import Path
from typing import Callable, Dict, Iterable, Sequence, Tuple, Union

from .utils import substitute_symbolic_expressions, params_dict_to_list


def factory(filepath: Union[str, Path]) -> Tuple[Callable, Callable]:
    """
    Create jax functions from file containing symbolic expressions.
    Args:
        filepath: path to file containing symbolic expressions
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
    n_q = len(sym_exps["state_syms"]["q"])

    # concatenate the list of state symbols
    state_syms_cat = sym_exps["state_syms"]["q"] + sym_exps["state_syms"]["q_d"]

    # lambdify symbolic expressions
    chi_lambda = sp.lambdify(params_syms_cat + sym_exps["state_syms"]["q"], sym_exps["exps"]["chi_ls"], "jax")
    B_lambda = sp.lambdify(params_syms_cat + sym_exps["state_syms"]["q"], sym_exps["exps"]["B"], "jax")
    C_lambda = sp.lambdify(params_syms_cat + state_syms_cat, sym_exps["exps"]["C"], "jax")
    G_lambda = sp.lambdify(params_syms_cat + sym_exps["state_syms"]["q"], sym_exps["exps"]["G"], "jax")

    @jit
    def forward_kinematics_fn(params: Dict[str, Array], q: Array) -> Array:
        """
        Evaluate the forward kinematics the tip of the links
        Args:
            params: Dictionary of robot parameters
            q: generalized coordinates of shape (n_q, )
        Returns:
            chi_ls: poses of tip of links of shape (3, n_q) consisting of [p_x, p_y, theta]
                where p_x is the x-position, p_y is the y-position,
                and theta is the planar orientation with respect to the x-axis
        """
        params_list = params_dict_to_list(params)
        chi_ls = chi_lambda(*params_list, *q)
        return chi_ls

    # actuation matrix
    A = jnp.identity(n_q)

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
        # elastic and dissipative matrices
        K = params.get("K", jnp.zeros((n_q, n_q)))
        D = params.get("D", jnp.zeros((n_q, n_q)))

        params_list = params_dict_to_list(params)

        B = B_lambda(*params_list, *q)
        C = C_lambda(*params_list, *q, *q_d)
        G = G_lambda(*params_list, *q).squeeze()

        # K(q) = K @ q
        _K = K @ q

        return B, C, G, _K, D, A

    return forward_kinematics_fn, dynamical_matrices_fn
