import dill
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
import jax
from jax import jit, vmap
from jax import numpy as jnp
import sympy as sp
from pathlib import Path
from typing import Callable, Dict, Iterable, Sequence, Tuple, Union


def make_jax_functions(filepath: Union[str, Path], params: Dict[str, jnp.array]) -> Callable:
    """
    Create jax functions from file containing symbolic expressions.
    Args:
        filepath: path to file containing symbolic expressions
        params: dictionary of robot parameters
    Returns:
        dynamical_matrices_fn: function that returns the B, C, G, K, D, and A matrices
    """

    # load saved symbolic data
    sym_pendulum = dill.load(open(str(filepath), "rb"))

    # symbols for robot parameters
    params_syms = sym_pendulum["params_syms"]
    # symbols of state variables
    state_syms = sym_pendulum["state_syms"]
    # symbolic expressions
    exps = sym_pendulum["exps"]

    # number of degrees of freedom
    n_q = len(state_syms["q"])

    for exp_key in exps.keys():
        for param_key, param_sym in params_syms.items():
            if issubclass(type(param_sym), Iterable):
                for idx, param_sym_item in enumerate(param_sym):
                    exps[exp_key] = exps[exp_key].subs(param_sym_item, params[param_key][idx])
            else:
                exps[exp_key] = exps[exp_key].subs(param_sym, params[param_key])
            exps[exp_key] = exps[exp_key].subs(params_syms[param_key], params[param_key])

    # concatenate the list of state symbols
    state_syms_cat = state_syms["q"] + state_syms["q_d"]

    # lambdify symbolic expressions
    B_lambda = sp.lambdify(state_syms["q"], exps["B"], "jax")
    C_lambda = sp.lambdify(state_syms_cat, exps["C"], "jax")
    G_lambda = sp.lambdify(state_syms["q"], exps["G"], "jax")

    # elastic and dissipative matrices
    K = params.get("K", jnp.zeros((n_q, n_q)))
    D = params.get("D", jnp.zeros((n_q, n_q)))

    # actuation matrix
    A = jnp.identity(n_q)

    @jit
    def dynamical_matrices_fn(
            q: jnp.array,
            q_d: jnp.array
    ) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array, jnp.array, jnp.array]:
        """
        Compute the dynamical matrices of the system.
        Args:
            q: generalized coordinates
            q_d: generalized velocities
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

    return dynamical_matrices_fn
