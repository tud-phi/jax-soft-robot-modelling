from functools import partial
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
import jax
from jax import Array, jit, vmap
from jax import numpy as jnp
from typing import Callable, Dict

from jsrm.systems import euler_lagrangian


@jit
def make_ode(
        dynamical_matrices_fn: Callable,
        params: Dict[str, Array],
        tau: Array
) -> Callable[[float, Array], Array]:
    nonlinear_state_space_fn = partial(
        euler_lagrangian.nonlinear_state_space,
        dynamical_matrices_fn,
        params,
        tau=tau
    )

    def ode_fn(t: float, x: Array):
        return nonlinear_state_space_fn(x)

    return ode_fn
