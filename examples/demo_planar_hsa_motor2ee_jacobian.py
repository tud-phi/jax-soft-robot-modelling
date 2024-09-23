import jax

jax.config.update("jax_enable_x64", True)  # double precision
from jax import Array, jacrev, jit, random, vmap
from jax import numpy as jnp
from jaxopt import GaussNewton, LevenbergMarquardt
from functools import partial
import numpy as onp
from pathlib import Path
from typing import Callable, Dict

import jsrm
from jsrm.parameters.hsa_params import PARAMS_FPU_CONTROL
from jsrm.systems import planar_hsa

num_segments = 1
num_rods_per_segment = 2

# filepath to symbolic expressions
sym_exp_filepath = (
    Path(jsrm.__file__).parent
    / "symbolic_expressions"
    / f"planar_hsa_ns-{num_segments}_nrs-{num_rods_per_segment}.dill"
)

# activate all strains (i.e. bending, shear, and axial)
strain_selector = jnp.ones((3 * num_segments,), dtype=bool)
params = PARAMS_FPU_CONTROL
phi_max = params["phi_max"].flatten()

# define initial configuration
q0 = jnp.array([jnp.pi, 0.0, 0.0])

# increase damping for simulation stability
params["zetab"] = 5 * params["zetab"]
params["zetash"] = 5 * params["zetash"]
params["zetaa"] = 5 * params["zetaa"]


(
    forward_kinematics_virtual_backbone_fn,
    forward_kinematics_end_effector_fn,
    jacobian_end_effector_fn,
    inverse_kinematics_end_effector_fn,
    dynamical_matrices_fn,
    sys_helpers,
) = planar_hsa.factory(sym_exp_filepath, strain_selector)
jitted_dynamical_matrics_fn = jit(partial(dynamical_matrices_fn, params))


def phi2q_static_model(phi: Array, q0: Array = jnp.zeros((3, ))) -> Array:
    """
    A static model mapping the motor angles to the planar HSA configuration.
    Arguments:
        u: motor angles
    Returns:
        q: planar HSA configuration consisting of (k_be, sigma_sh, sigma_ax)
    """
    q_d = jnp.zeros((3,))

    def residual_fn(_q: Array) -> Array:
        _, _, _G, _K, _, _alpha = jitted_dynamical_matrics_fn(_q, q_d, phi=phi)
        res = _alpha - _G - _K
        return res

    # solve the nonlinear least squares problem
    lm = LevenbergMarquardt(residual_fun=residual_fn, jit=True, verbose=True)
    sol = lm.run(q0)

    # configuration that minimizes the residual
    q = sol.params

    # compute the L2 optimality
    optimality_error = lm.l2_optimality_error(sol.params)

    return q

def u2chi_static_model(u: Array) -> Array:
    """
    A static model mapping the motor angles to the planar end-effector pose.
    Arguments:
        u: motor angles
    Returns:
        chi: end-effector pose
    """
    q = phi2q_static_model(u)
    chi = forward_kinematics_end_effector_fn(params, q)
    return chi

def jac_u2chi_static_model(u: Array) -> Array:
    # evaluate the static model to convert motor angles into a configuration
    q = phi2q_static_model(u)
    # take the Jacobian between actuation and configuration space
    J_u2q = jacrev(phi2q_static_model)(u)

    # evaluate the closed-form, analytical jacobian of the forward kinematics
    J_q2chi = jacobian_end_effector_fn(params, q)

    # evaluate the Jacobian between the actuation and the task-space
    J_u2chi = J_q2chi @ J_u2q

    return J_u2chi


if __name__ == "__main__":
    jitted_phi2q_static_model_fn = jit(phi2q_static_model)
    J_u2chi_autodiff_fn = jacrev(u2chi_static_model)
    J_u2chi_fn = jac_u2chi_static_model

    rng = random.key(seed=0)
    for i in range(10):
        match i:
            case 0:
                u = jnp.array([0.0, 0.0])
            case 1:
                u = jnp.array([1.0, 1.0])
            case _:
                rng, subkey = random.split(rng)
                u = random.uniform(
                    subkey,
                    phi_max.shape,
                    minval=0.0,
                    maxval=phi_max
                )

        q = jitted_phi2q_static_model_fn(u)
        print("u", u, "q", q)
        # J_u2chi_autodiff = J_u2chi_autodiff_fn(u)
        # J_u2chi = J_u2chi_fn(u)
        # print("J_u2chi:\n", J_u2chi, "\nJ_u2chi_autodiff:\n", J_u2chi_autodiff)
        # print(J_u2chi.shape)