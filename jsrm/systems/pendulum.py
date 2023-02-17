from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
import jax
from jax import Array, jit, vmap
from jax import numpy as jnp
import sympy as sp
from pathlib import Path
from typing import Callable, Dict, Iterable, Sequence, Tuple, Union

from .utils import load_and_substitute_symbolic_expressions


def make_jax_functions(filepath: Union[str, Path], params: Dict[str, Array]) -> Tuple[Callable, Callable]:
    """
    Create jax functions from file containing symbolic expressions.
    Args:
        filepath: path to file containing symbolic expressions
        params: dictionary of robot parameters
    Returns:
        forward_kinematics_fn: function that returns the p vector of shape (2, n_q) with the positions
        dynamical_matrices_fn: function that returns the B, C, G, K, D, and A matrices
    """
    sym_exps = load_and_substitute_symbolic_expressions(filepath, params)

    # number of degrees of freedom
    n_q = len(sym_exps["state_syms"]["q"])

    # concatenate the list of state symbols
    state_syms_cat = sym_exps["state_syms"]["q"] + sym_exps["state_syms"]["q_d"]

    # lambdify symbolic expressions
    p_lambda = sp.lambdify(sym_exps["state_syms"]["q"], sym_exps["exps"]["p"], "jax")
    B_lambda = sp.lambdify(sym_exps["state_syms"]["q"], sym_exps["exps"]["B"], "jax")
    C_lambda = sp.lambdify(state_syms_cat, sym_exps["exps"]["C"], "jax")
    G_lambda = sp.lambdify(sym_exps["state_syms"]["q"], sym_exps["exps"]["G"], "jax")

    @jit
    def forward_kinematics_fn(q: Array) -> Array:
        """
        Evaluate the forward kinematics the tip of the links
        Args:
            q: generalized coordinates of shape (n_q, )
        Returns:
            p: positions of tip of links of shape (2, n_q)
        """
        p = p_lambda(*q)
        return p

    # elastic and dissipative matrices
    K = params.get("K", jnp.zeros((n_q, n_q)))
    D = params.get("D", jnp.zeros((n_q, n_q)))

    # actuation matrix
    A = jnp.identity(n_q)

    @jit
    def dynamical_matrices_fn(
            q: Array,
            q_d: Array
    ) -> Tuple[Array, Array, Array, Array, Array, Array]:
        """
        Compute the dynamical matrices of the system.
        Args:
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
        B = B_lambda(*q)
        C = C_lambda(*q, *q_d)
        G = G_lambda(*q).squeeze()

        # K(q) = K @ q
        _K = K @ q

        return B, C, G, _K, D, A

    return forward_kinematics_fn, dynamical_matrices_fn
