from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
import jax
from jax import jit, vmap
from jax import numpy as jnp
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Tuple, Union

from jsrm.integration import ode_factory
from jsrm.systems import euler_lagrangian, pendulum

sym_exp_filepath = Path(__file__).parent.parent / "symbolic_expressions" / "double_pendulum.dill"
params = {
    "m": jnp.array([10.0, 6.0]),
    "I": jnp.array([3.0, 2.0]),
    "l": jnp.array([2.0, 1.0]),
    "lc": jnp.array([1.0, 0.5]),
    "g": jnp.array([0.0, -9.81]),
}

if __name__ == "__main__":
    forward_kinematics_fn, dynamical_matrices_fn = pendulum.factory(sym_exp_filepath)
    nonlinear_state_space_fn = partial(
        euler_lagrangian.nonlinear_state_space,
        dynamical_matrices_fn
    )

    q, q_d = jnp.zeros((2, )), jnp.zeros((2, ))
    # compute the pose of the end-effector
    chiee = forward_kinematics_fn(params, q, -1)
    print("chiee =\n", chiee)

    x = jnp.concatenate((q, q_d), axis=0)
    tau = jnp.zeros(q.shape)

    x_d = nonlinear_state_space_fn(params, x, tau)
    print("x_d =\n", x_d)

    ode_fn = ode_factory(dynamical_matrices_fn, params, tau)
    term = ODETerm(ode_fn)

    x0 = jnp.zeros((4,))  # initial condition
    dt = 1e-2  # time step
    ts = jnp.arange(0.0, 1.0, dt)  # time steps
    sol = diffeqsolve(
        term,
        solver=Dopri5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=dt,
        y0=x0,
        saveat=SaveAt(ts=ts)
    )
    print("sol =\n", sol)
    print("sol.ts =\n", sol.ts)
    print("sol.ys =\n", sol.ys)
