from jax import jit, vmap
from jax import numpy as jnp
from functools import partial
from typing import Callable, Dict, Tuple, Union


@partial(jit, static_argnums=0, static_argnames="dynamical_matrices_fn")
def forward_dynamics(
        dynamical_matrices_fn: Callable,
        q: jnp.array,
        q_d: jnp.array,
        tau: jnp.array,
):
    """
    Compute the forward dynamics of a Lagrangian system.
    Args:
        dynamical_matrices_fn: Callable that returns the B, C, G, K, D, and A matrices. Needs to conform to the signature:
            dynamical_matrices_fn(q, q_d) -> Tuple[B, C, G, K, D, A]
            where q and q_d are the configuration and velocity vectors, respectively,
            B is the inertia matrix of shape (n_q, n_q),
            C is the Coriolis matrix of shape (n_q, n_q),
            G is the gravity vector of shape (n_q, ),
            K is the stiffness vector of shape (n_q, ),
            D is the damping matrix of shape (n_q, n_q),
            and A is the actuation matrix of shape (n_q, n_tau).
        q: configuration vector of shape (n_q, )
        q_d: configuration velocity vector of shape (n_q, )
        tau: generalized torque vector of shape (n_tau, )
    Returns:
        q_dd: configuration acceleration vector of shape (n_q, )
    """
    B, C, G, K, D, A = dynamical_matrices_fn(q, q_d)

    # inverse of B
    B_inv = jnp.linalg.inv(B)

    # compute the acceleration
    q_dd = B_inv @ (A @ tau - C @ q_d - G - K - D @ q_d)

    return q_dd


@partial(jit, static_argnums=0, static_argnames="dynamical_matrices_fn")
def nonlinear_state_space(
        dynamical_matrices_fn: Callable,
        x: jnp.array,
        tau: jnp.array,
) -> jnp.array:
    """
    Compute the nonlinear state space dynamics of a Lagrangian system (i.e. the ODE function).
    Args:
        dynamical_matrices_fn: Callable that returns the B, C, G, K, D, and A matrices. Needs to conform to the signature:
            dynamical_matrices_fn(q, q_d) -> Tuple[B, C, G, K, D, A]
            where q and q_d are the configuration and velocity vectors, respectively,
            B is the inertia matrix of shape (n_q, n_q),
            C is the Coriolis matrix of shape (n_q, n_q),
            G is the gravity vector of shape (n_q, ),
            K is the stiffness vector of shape (n_q, ),
            D is the damping matrix of shape (n_q, n_q),
            and A is the actuation matrix of shape (n_q, n_tau).
        x: state vector of shape (2 * n_q, ) containing the configuration and velocity vectors
        tau: generalized torque vector of shape (n_tau, )
    Returns:
        x_d: state derivative vector of shape (2 * n_q, ) containing the velocity and acceleration vectors
    """
    q_dd = forward_dynamics(dynamical_matrices_fn, x[:2], x[2:], tau)
    x_d = jnp.concatenate([x[2:], q_dd])
    return x_d
