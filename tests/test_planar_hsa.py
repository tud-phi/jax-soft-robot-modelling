import dill
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
from jax import Array, jit, random
from jax import numpy as jnp
import jsrm
from pathlib import Path
import sympy as sp
from typing import Tuple

from jsrm.parameters.hsa_params import PARAMS_FPU_CONTROL as params
from jsrm.systems import planar_hsa
from jsrm.systems.utils import substitute_params_into_all_symbolic_expressions

num_segments = 1
num_rods_per_segment = 2

# filepath to symbolic expressions
sym_exp_filepath = (
    Path(jsrm.__file__).parent
    / "symbolic_expressions"
    / f"planar_hsa_ns-{num_segments}_nrs-{num_rods_per_segment}.dill"
)


def test_end_effector_kinematics(seed: int = 0):
    print("Testing end effector kinematics...")
    (
        _,
        forward_kinematics_end_effector_fn,
        jacobian_end_effector_fn,
        inverse_kinematics_end_effector_fn,
        _,
        sys_helpers,
    ) = planar_hsa.factory(sym_exp_filepath)

    # load saved symbolic data
    sym_exps = dill.load(open(str(sym_exp_filepath), "rb"))

    rng = random.PRNGKey(seed)
    for idx in range(10):
        rng, subrng1, subrng2, subrng3, subrng4, subrng5 = random.split(rng, 6)
        kappa_b = random.uniform(
            subrng1,
            (num_segments,),
            minval=-jnp.pi / jnp.mean(params["l"]),
            maxval=jnp.pi / jnp.mean(params["l"]),
        )
        sigma_sh = random.uniform(subrng2, (num_segments,), minval=-0.2, maxval=0.2)
        sigma_a = random.uniform(subrng3, (num_segments,), minval=0.0, maxval=0.5)
        q = jnp.concatenate((kappa_b, sigma_sh, sigma_a), axis=0)

        print("q = ", q)

        # forward kinematics
        chiee = forward_kinematics_end_effector_fn(params, q)
        # inverse kinematics
        q_rec = inverse_kinematics_end_effector_fn(params, chiee)

        if not jnp.allclose(q, q_rec, atol=1e-6):
            print("q = ", q)
            print("q_rec = ", q_rec)
            raise ValueError("q != q_rec")


if __name__ == "__main__":
    test_end_effector_kinematics()
