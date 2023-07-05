import dill
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
from jax import Array, debug, jit, random, vmap
from jax import numpy as jnp
import jsrm
from pathlib import Path
import sympy as sp
from typing import Callable, Dict, Tuple

from jsrm.parameters.hsa_params import PARAMS_CONTROL as params
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


def test_symbolic_and_numeric_implementation(seed: int = 0):
    print("Testing symbolic and numeric implementation...")
    (
        forward_kinematics_virtual_backbone_fn,
        forward_kinematics_end_effector_fn,
        jacobian_end_effector_fn,
        inverse_kinematics_end_effector_fn,
        dynamical_matrices_fn,
        sys_helpers,
    ) = planar_hsa.factory(sym_exp_filepath)

    # load saved symbolic data
    sym_exps = dill.load(open(str(sym_exp_filepath), "rb"))

    exps_subs = substitute_params_into_all_symbolic_expressions(sym_exps, params)
    state_syms = sym_exps["state_syms"]

    K_lambda = sp.lambdify(
        state_syms["xi"],
        exps_subs["K"],
        "jax",
    )
    alpha_lambda = sp.lambdify(
        state_syms["xi"] + state_syms["phi"],
        exps_subs["alpha"],
        "jax",
    )
    D = jnp.array(exps_subs["D"], dtype=jnp.float64)

    @jit
    def matrices_sym(q: Array, phi: Array) -> Tuple[Array, Array, Array]:
        # map the configuration to the strains
        xi = sys_helpers["configuration_to_strains_fn"](params, q)

        # add a small number to the bending strain to avoid singularities
        xi_epsed = sys_helpers["apply_eps_to_bend_strains_fn"](xi, 1e-2)

        K = K_lambda(*xi_epsed).squeeze()
        alpha = alpha_lambda(*xi_epsed, *phi).squeeze()

        return K, D, alpha

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
        q_d = random.uniform(subrng4, (3 * num_segments,), minval=-1.0, maxval=1.0)
        phi = params["h"].flatten() * random.uniform(
            subrng5, params["h"].flatten().shape, minval=0.0, maxval=jnp.pi
        )

        print(f"q: {q}, q_d: {q_d}, phi: {phi}")

        _, _, _, K_num, D_num, alpha_num = dynamical_matrices_fn(params, q, q_d, phi)
        K_sym, D_sym, alpha_sym = matrices_sym(q, phi)

        if not jnp.allclose(K_num, K_sym, atol=1e-6):
            print("K_num = ", K_num)
            print("K_sym = ", K_sym)
            raise ValueError("K_num != K_sym")
        if not jnp.allclose(D_num, D_sym, atol=1e-6):
            print("D_num = ", D_num)
            print("D_sym = ", D_sym)
            raise ValueError("D_num != D_sym")
        if not jnp.allclose(alpha_num, alpha_sym, atol=1e-6):
            print("alpha_num = ", alpha_num)
            print("alpha_sym = ", alpha_sym)
            raise ValueError("alpha_num != alpha_sym")


if __name__ == "__main__":
    test_end_effector_kinematics()
    test_symbolic_and_numeric_implementation()
