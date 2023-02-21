import dill
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
import jax
from jax import jit, vmap
from jax import numpy as jnp
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Tuple, Union

from jsrm.systems import euler_lagrangian
from jsrm.systems import planar_pcs

sym_exp_filepath = Path(__file__).parent.parent / "symbolic_expressions" / "planar_pcs_two_segments.dill"
params = {
    "rho": 22.3 * jnp.ones((2, )),  # Surface density of Dragon Skin 20 [kg/m^2]
    "l": jnp.array([1e-1, 1e-1]),
    "r": jnp.array([2e-2, 2e-2]),
    "g": jnp.array([0.0, -9.81]),
}
# activate all strains (i.e. bending, shear, and axial)
strain_selector = jnp.ones((6, ), dtype=bool)

if __name__ == "__main__":
    forward_kinematics_fn, dynamical_matrices_fn = planar_pcs.make_jax_functions(
        sym_exp_filepath,
        strain_selector
    )

    q, q_d = jnp.zeros((6, )), jnp.zeros((6, ))
    chi_sms = forward_kinematics_fn(params, q, 0.05)
    print("chi_sms =\n", chi_sms)

    nonlinear_state_space_fn = partial(
        euler_lagrangian.nonlinear_state_space,
        dynamical_matrices_fn
    )

    x = jnp.concatenate((q, q_d), axis=0)
    print("x =\n", x)
    tau = jnp.zeros_like(q)

    x_d = nonlinear_state_space_fn(params, x, tau)
    print("x_d =\n", x_d)
