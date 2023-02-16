import dill
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
import jax
from jax import jit, vmap
from jax import numpy as jnp
import sympy as sp
from pathlib import Path
from typing import Callable, Dict, Iterable, Sequence, Tuple, Union


def load_and_substitute_symbolic_expressions(
        filepath: Union[str, Path], params: Dict[str, jnp.array]
) -> Dict:
    """
    Load symbolic expressions and substitute in robot parameters.
    Args:
        filepath: path to file containing symbolic expressions
        params: dictionary of robot parameters
    Returns:
        sym_exps: dictionary with entries
            params_syms: dictionary of robot parameters
            state_syms: dictionary of state variables
            exps: dictionary of symbolic expressions
    """
    # load saved symbolic data
    sym_exps = dill.load(open(str(filepath), "rb"))

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
