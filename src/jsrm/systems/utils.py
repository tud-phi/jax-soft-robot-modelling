from copy import deepcopy
import jax
from jax import Array, jit, vmap
from jax import numpy as jnp
import sympy as sp
from typing import Callable, Dict, Iterable, List, Sequence, Tuple, Union


def substitute_params_into_all_symbolic_expressions(
    sym_exps: Dict, params: Dict[str, jnp.array]
) -> Dict:
    """
    Substitute robot parameters into symbolic expressions.
    Args:
        sym_exps: dictionary with entries
            params_syms: dictionary of list with symbols for parameters
            state_syms: dictionary of lists with symbols for state variables
            exps: dictionary of symbolic expressions
        params: dictionary of robot parameters
    Returns:
        sym_exps: dictionary with entries
            params_syms: dictionary of robot parameters
            state_syms: dictionary of state variables
            exps: dictionary of symbolic expressions
    """
    # symbols for robot parameters
    params_syms = sym_exps["params_syms"]
    # symbolic expressions
    exps = deepcopy(sym_exps["exps"])

    for exp_key, exp_val in exps.items():
        if issubclass(type(exp_val), list):
            for exp_item_idx, exp_item_val in enumerate(exp_val):
                exps[exp_key][
                    exp_item_idx
                ] = substitute_params_into_single_symbolic_expression(
                    exp_item_val, params_syms, params
                )
        else:
            exps[exp_key] = substitute_params_into_single_symbolic_expression(
                exp_val, params_syms, params
            )

    return exps


def substitute_params_into_single_symbolic_expression(
    sym_exp: sp.Expr,
    params_syms: Dict[str, List[sp.Symbol]],
    params: Dict[str, jnp.array],
) -> sp.Expr:
    """
    Substitute robot parameters into a single symbolic expression.
    Args:
        sym_exp: symbolic expression
        params_syms: Dictionary of list with symbols for parameters
        params: Dictionary of jax arrays with numerical values for parameters

    Returns:
        sym_exp: symbolic expression with parameters substituted
    """
    for param_key, param_sym in params_syms.items():
        if issubclass(type(param_sym), list):
            for idx, param_sym_item in enumerate(param_sym):
                if param_sym_item in sym_exp.free_symbols:
                    sym_exp = sym_exp.subs(
                        param_sym_item, params[param_key].flatten()[idx]
                    )
        else:
            if param_sym in sym_exp.free_symbols:
                sym_exp = sym_exp.subs(param_sym, params[param_key])

    return sym_exp


def concatenate_params_syms(
    params_syms: Dict[str, Union[sp.Symbol, List[sp.Symbol]]]
) -> List[sp.Symbol]:
    # concatenate the robot params symbols
    params_syms_cat = []
    for params_key, params_sym in sorted(params_syms.items()):
        if type(params_sym) in [list, tuple]:
            params_syms_cat += params_sym
        else:
            params_syms_cat.append(params_sym)
    return params_syms_cat


def compute_strain_basis(
    strain_selector: Array,
) -> jnp.ndarray:
    """
    Compute strain basis based on boolean strain selector.
    Args:
        strain_selector: boolean array of shape (n_xi, ) specifying which strain components are active
    Returns:
        strain_basis: strain basis matrix of shape (n_xi, n_q) where n_q is the number of configuration variables
            and n_xi is the number of strains
    """
    n_q = strain_selector.sum().item()
    strain_basis = jnp.zeros((strain_selector.shape[0], n_q))
    strain_basis_cumsum = jnp.cumsum(strain_selector)
    for i in range(strain_selector.shape[0]):
        j = int(strain_basis_cumsum[i].item()) - 1
        if strain_selector[i].item() is True:
            strain_basis = strain_basis.at[i, j].set(1.0)
    return strain_basis


@jit
def compute_planar_stiffness_matrix(A: Array, Ib: Array, E: Array, G: Array) -> Array:
    """
    Compute the stiffness matrix of the system.
    Args:
        A: cross-sectional area of shape ()
        Ib: second moment of area of shape ()
        E: Elastic modulus of shape ()
        G: Shear modulus of shape ()

    Returns:
        S: stiffness matrix of shape (3, 3)
    """
    S = jnp.diag(jnp.stack([Ib * E, 4 / 3 * A * G, A * E], axis=0))

    return S
