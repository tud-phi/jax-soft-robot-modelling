from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
from jax import numpy as jnp
import jsrm
from functools import partial
from numpy.testing import assert_allclose
from pathlib import Path

from jsrm.systems import planar_pcs, euler_lagrangian
from jsrm import Tolerance


def test_planar_pcs_one_segment():
    sym_exp_filepath = (
        Path(jsrm.__file__).parent / "symbolic_expressions" / "planar_pcs_ns-1.dill"
    )
    params = {
        "th0": jnp.array(0.0),  # initial orientation angle [rad]
        "l": jnp.array([1e-1]),
        "r": jnp.array([2e-2]),
        "rho": 1000 * jnp.ones((1,)),
        "g": jnp.array([0.0, -9.81]),
        "E": 1e7 * jnp.ones((1,)),  # Elastic modulus [Pa]
        "G": 1e6 * jnp.ones((1,)),  # Shear modulus [Pa]
    }
    # activate all strains (i.e. bending, shear, and axial)
    strain_selector = jnp.ones((3,), dtype=bool)

    strain_basis, forward_kinematics_fn, dynamical_matrices_fn = planar_pcs.factory(
        sym_exp_filepath, strain_selector
    )
    forward_dynamics_fn = partial(
        euler_lagrangian.forward_dynamics, dynamical_matrices_fn
    )
    nonlinear_state_space_fn = partial(
        euler_lagrangian.nonlinear_state_space, dynamical_matrices_fn
    )

    # test forward kinematics
    assert_allclose(
        forward_kinematics_fn(params, q=jnp.zeros((3,)), s=params["l"][0] / 2),
        jnp.array([0.0, params["l"][0] / 2, 0.0]),
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )
    assert_allclose(
        forward_kinematics_fn(params, q=jnp.zeros((3,)), s=params["l"][0]),
        jnp.array([0.0, params["l"][0], 0.0]),
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )
    assert_allclose(
        forward_kinematics_fn(params, q=jnp.array([0.0, 0.0, 1.0]), s=params["l"][0]),
        jnp.array([0.0, 2 * params["l"][0], 0.0]),
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )
    assert_allclose(
        forward_kinematics_fn(params, q=jnp.array([0.0, 1.0, 0.0]), s=params["l"][0]),
        params["l"][0] * jnp.array([1.0, 1.0, 0.0]),
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )

    # test dynamical matrices
    q, q_d = jnp.zeros((3,)), jnp.zeros((3,))
    B, C, G, K, D, A = dynamical_matrices_fn(params, q, q_d)
    assert_allclose(K, jnp.zeros((3,)))
    assert_allclose(
        A,
        jnp.eye(3),
    )

    q = jnp.array([jnp.pi / (2 * params["l"][0]), 0.0, 0.0])
    q_d = jnp.zeros((3,))
    B, C, G, K, D, alpha = dynamical_matrices_fn(params, q, q_d)

    print("B =\n", B)
    print("C =\n", C)
    print("G =\n", G)
    print("K =\n", K)
    print("D =\n", D)
    print("alpha =\n", alpha)


if __name__ == "__main__":
    test_planar_pcs_one_segment()
