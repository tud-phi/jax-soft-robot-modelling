from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
from jax import numpy as jnp
from functools import partial
from numpy.testing import assert_allclose
from pathlib import Path
import pytest

from jsrm.systems import euler_lagrangian
from jsrm.systems import planar_pcs
from jsrm.utils import Tolerance


def test_planar_pcs_one_segment():
    sym_exp_filepath = Path(__file__).parent.parent / "symbolic_expressions" / "planar_pcs_one_segment.dill"
    params = {
        "rho": 1 * jnp.ones((1,)),  # Surface density of Dragon Skin 20 [kg/m^2]
        "l": jnp.array([1e-1]),
        "r": jnp.array([2e-2]),
        "g": jnp.array([0.0, -9.81]),
    }
    # activate all strains (i.e. bending, shear, and axial)
    strain_selector = jnp.ones((3,), dtype=bool)

    forward_kinematics_fn, dynamical_matrices_fn = planar_pcs.make_jax_functions(
        sym_exp_filepath,
        strain_selector
    )
    forward_dynamics_fn = partial(
        euler_lagrangian.forward_dynamics,
        dynamical_matrices_fn
    )
    nonlinear_state_space_fn = partial(
        euler_lagrangian.nonlinear_state_space,
        dynamical_matrices_fn
    )

    # test forward kinematics
    assert_allclose(
        forward_kinematics_fn(
            params,
            q=jnp.zeros((3,)),
            s=params["l"][0] / 2
        ),
        jnp.array([0.0, params["l"][0] / 2, 0.0]),
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )
    assert_allclose(
        forward_kinematics_fn(
            params,
            q=jnp.zeros((3,)),
            s=params["l"][0]
        ),
        jnp.array([0.0, params["l"][0], 0.0]),
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )
    assert_allclose(
        forward_kinematics_fn(
            params,
            q=jnp.array([0.0, 0.0, 1.0]),
            s=params["l"][0]
        ),
        jnp.array([0.0, 2 * params["l"][0], 0.0]),
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )
    assert_allclose(
        forward_kinematics_fn(
            params,
            q=jnp.array([0.0, 1.0, 0.0]),
            s=params["l"][0]
        ),
        params["l"][0] * jnp.array([1.0, 1.0, 0.0]),
        rtol=Tolerance.rtol(),
        atol=Tolerance.atol(),
    )


if __name__ == "__main__":
    test_planar_pcs_one_segment()
