import jax

jax.config.update("jax_enable_x64", True)  # double precision
from jax import Array, jacfwd, jacrev, jit, random, vmap
from jax import numpy as jnp
from jaxopt import GaussNewton, LevenbergMarquardt
from functools import partial
import numpy as onp
from pathlib import Path
from typing import Callable, Dict, Tuple

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
q0 = jnp.array([0.0, 0.0, 0.0])

# increase damping for simulation stability
params["zetab"] = 5 * params["zetab"]
params["zetash"] = 5 * params["zetash"]
params["zetaa"] = 5 * params["zetaa"]

# nonlinear least squares solver settings
nlq_tol = 1e-5  # tolerance for the nonlinear least squares solver

(
    forward_kinematics_virtual_backbone_fn,
    forward_kinematics_end_effector_fn,
    jacobian_end_effector_fn,
    inverse_kinematics_end_effector_fn,
    dynamical_matrices_fn,
    sys_helpers,
) = planar_hsa.factory(sym_exp_filepath, strain_selector)
jitted_dynamical_matrics_fn = jit(partial(dynamical_matrices_fn, params))


def phi2q_static_model(phi: Array, q0: Array = jnp.zeros((3, ))) -> Tuple[Array, Dict[str, Array]]:
    """
    A static model mapping the motor angles to the planar HSA configuration.
    Arguments:
        phi: motor angles
    Returns:
        q: planar HSA configuration consisting of (k_be, sigma_sh, sigma_ax)
        aux: dictionary with auxiliary data
    """
    q_d = jnp.zeros((3,))

    def residual_fn(_q: Array) -> Array:
        _, _, _G, _K, _, _alpha = jitted_dynamical_matrics_fn(_q, q_d, phi=phi)
        res = _alpha - _G - _K
        return res

    # solve the nonlinear least squares problem
    lm = LevenbergMarquardt(residual_fun=residual_fn, tol=nlq_tol, jit=True, unroll=True, verbose=True)
    sol = lm.run(q0)

    # configuration that minimizes the residual
    q = sol.params

    # compute the L2 optimality
    optimality_error = lm.l2_optimality_error(sol.params)

    aux = dict(
        phi=phi,
        q=q,
        optimality_error=optimality_error,
    )

    return q, aux

def phi2chi_static_model(phi: Array) -> Tuple[Array, Dict[str, Array]]:
    """
    A static model mapping the motor angles to the planar end-effector pose.
    Arguments:
        phi: motor angles
    Returns:
        chi: end-effector pose
        aux: dictionary with auxiliary data
    """
    q, aux = phi2q_static_model(phi)
    chi = forward_kinematics_end_effector_fn(params, q)
    aux["chi"] = chi
    return chi, aux

def jac_phi2chi_static_model(phi: Array) -> Tuple[Array, Dict[str, Array]]:
    """
    Compute the Jacobian between the actuation space and the task-space.
    Arguments:
        phi: motor angles
    """
    # evaluate the static model to convert motor angles into a configuration
    q = phi2q_static_model(phi)
    # take the Jacobian between actuation and configuration space
    J_phi2q, aux = jacfwd(phi2q_static_model, has_aux=True)(phi)

    # evaluate the closed-form, analytical jacobian of the forward kinematics
    J_q2chi = jacobian_end_effector_fn(params, q)

    # evaluate the Jacobian between the actuation and the task-space
    J_phi2chi = J_q2chi @ J_phi2q

    return J_phi2chi, aux


if __name__ == "__main__":
    jitted_phi2q_static_model_fn = jit(phi2q_static_model)
    J_phi2chi_autodiff_fn = jit(jacfwd(phi2chi_static_model, has_aux=True))
    J_phi2chi_fn = jit(jac_phi2chi_static_model)

    rng = random.key(seed=0)
    for i in range(10):
        match i:
            case 0:
                phi = jnp.array([0.0, 0.0])
            case 1:
                phi = jnp.array([1.0, 1.0])
            case _:
                rng, subkey = random.split(rng)
                phi = random.uniform(
                    subkey,
                    phi_max.shape,
                    minval=0.0,
                    maxval=phi_max
                )

        print("i", i, "phi", phi)

        q, aux = jitted_phi2q_static_model_fn(phi)
        print("phi", phi, "q", q)

        # J_phi2chi_autodiff, aux = J_phi2chi_autodiff_fn(phi)
        # J_phi2chi, aux = J_phi2chi_fn(phi)
        # print("J_phi2chi:\n", J_phi2chi, "\nJ_phi2chi_autodiff:\n", J_phi2chi_autodiff)