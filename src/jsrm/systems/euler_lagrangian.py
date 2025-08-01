from jax import Array, debug, jit, vmap
from jax import numpy as jnp
from functools import partial
from typing import Callable, Dict, Tuple, Union


@partial(jit, static_argnums=0, static_argnames="dynamical_matrices_fn")
def forward_dynamics(
    dynamical_matrices_fn: Callable,
    params: Dict[str, Array],
    q: Array,
    qd: Array,
    tau: Array,
):
    """
    Compute the forward dynamics of a Lagrangian system.
    Args:
        dynamical_matrices_fn: Callable that returns the B, C, G, K, D, and alpha matrices. Needs to conform to the signature:
            dynamical_matrices_fn(params, q, qd) -> Tuple[B, C, G, K, D, alpha]
            where q and qd are the configuration and velocity vectors, respectively,
            B is the inertia matrix of shape (num_dofs, num_dofs),
            C is the Coriolis matrix of shape (num_dofs, num_dofs),
            G is the gravity vector of shape (num_dofs, ),
            K is the stiffness vector of shape (num_dofs, ),
            D is the damping matrix of shape (num_dofs, num_dofs),
            and alpha is the actuation matrix of shape (num_dofs, n_tau).
        params: Dictionary with robot parameters
        q: configuration vector of shape (num_dofs, )
        qd: configuration velocity vector of shape (num_dofs, )
        tau: generalized torque vector of shape (n_tau, )
    Returns:
        qdd: configuration acceleration vector of shape (num_dofs, )
    """
    B, C, G, K, D, alpha = dynamical_matrices_fn(params, q, qd)

    # inverse of B
    B_inv = jnp.linalg.inv(B)

    # compute the acceleration
    qdd = B_inv @ (alpha @ tau - C @ qd - G - K - D @ qd)

    return qdd


@partial(jit, static_argnums=0, static_argnames="dynamical_matrices_fn")
def nonlinear_state_space(
    dynamical_matrices_fn: Callable,
    params: Dict[str, Array],
    x: Array,
    tau: Array,
) -> jnp.array:
    """
    Compute the nonlinear state space dynamics of a Lagrangian system (i.e. the ODE function).
    Args:
        dynamical_matrices_fn: Callable that returns the B, C, G, K, D, and A matrices. Needs to conform to the signature:
            dynamical_matrices_fn(params, q, qd) -> Tuple[B, C, G, K, D, A]
            where q and qd are the configuration and velocity vectors, respectively,
            B is the inertia matrix of shape (num_dofs, num_dofs),
            C is the Coriolis matrix of shape (num_dofs, num_dofs),
            G is the gravity vector of shape (num_dofs, ),
            K is the stiffness vector of shape (num_dofs, ),
            D is the damping matrix of shape (num_dofs, num_dofs),
            and alpha is the actuation matrix of shape (num_dofs, n_tau).
        params: Dictionary with robot parameters
        x: state vector of shape (2 * num_dofs, ) containing the configuration and velocity vectors
        tau: generalized torque vector of shape (n_tau, )
    Returns:
        xd: state derivative vector of shape (2 * num_dofs, ) containing the velocity and acceleration vectors
    """
    q, qd = jnp.split(x, 2)
    qdd = forward_dynamics(dynamical_matrices_fn, params, q, qd, tau)
    xd = jnp.concatenate([qd, qdd])
    return xd
