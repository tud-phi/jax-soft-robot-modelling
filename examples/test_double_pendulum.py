import dill
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
import jax
from jax import jit, vmap
from jax import numpy as jnp
from functools import partial
import sympy as sp
from pathlib import Path
from typing import Callable, Dict, Tuple, Union

from jax_soft_robot_modelling.systems import euler_lagrangian
from jax_soft_robot_modelling.systems import pendulum

sym_exp_filepath = Path(__file__).parent.parent / "symbolic_expressions" / "double_pendulum.dill"
params = {
    "m": jnp.array([10.0, 6.0]),
    "I": jnp.array([3.0, 2.0]),
    "l": jnp.array([2.0, 1.0]),
    "lc": jnp.array([1.0, 0.5]),
    "g": jnp.array([0.0, -9.81]),
}

if __name__ == "__main__":
    dynamical_matrices_fn = pendulum.make_jax_functions(sym_exp_filepath, params)
    nonlinear_state_space_fn = partial(
        euler_lagrangian.nonlinear_state_space,
        dynamical_matrices_fn
    )

    x = jnp.zeros((4, ))
    tau = jnp.zeros((2, ))
    x_d = nonlinear_state_space_fn(x, tau)
    print("x_d =\n", x_d)
