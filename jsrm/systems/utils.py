import dill
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
import jax
from jax import jit, vmap
from jax import numpy as jnp
from typing import Callable, Dict, Iterable, Sequence, Tuple, Union


def substitute_symbolic_expressions(
        sym_exps: Dict, params: Dict[str, jnp.array]
) -> Dict:
    """
    Substitute robot parameters into symbolic expressions.
    Args:
        sym_exps: dictionary with entries
            params_syms: dictionary of robot parameters
            state_syms: dictionary of state variables
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
    # symbols of state variables
    state_syms = sym_exps["state_syms"]
    # symbolic expressions
    exps = sym_exps["exps"]

    for exp_key in exps.keys():
        for param_key, param_sym in params_syms.items():
            if issubclass(type(param_sym), Iterable):
                for idx, param_sym_item in enumerate(param_sym):
                    exps[exp_key] = exps[exp_key].subs(param_sym_item, params[param_key][idx])
            else:
                exps[exp_key] = exps[exp_key].subs(param_sym, params[param_key])
            exps[exp_key] = exps[exp_key].subs(params_syms[param_key], params[param_key])

    return sym_exps


@jit
def params_dict_to_list(params_dict: Dict[str, jnp.array]) -> jnp.array:
    """
    Convert dictionary of robot parameters to unrolled list.
    Args:
        params_dict: dictionary of robot parameters
    Returns:
        params_list: list of robot parameters
    """
    params_list = []
    for param_key, param_val in params_dict.items():
        params_list.extend(param_val)
    return params_list
