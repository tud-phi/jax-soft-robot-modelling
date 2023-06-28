import dill
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
from jax import Array, debug, jit, random, vmap
from jax import numpy as jnp
import jsrm
from pathlib import Path
import sympy as sp
from typing import Callable, Dict, Tuple

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

# set parameters
ones_rod = jnp.ones((num_segments, num_rods_per_segment))
params = {
    "th0": jnp.array(0.0),  # initial orientation angle [rad]
    "l": 59e-3 * jnp.ones((num_segments,)),  # length of each rod [m]
    # length of the rigid proximal caps of the rods connecting to the base [m]
    "lpc": 25e-3 * jnp.ones((num_segments,)),
    # length of the rigid distal caps of the rods connecting to the platform [m]
    "ldc": 14e-3 * jnp.ones((num_segments,)),
    # scale factor for the rest length as a function of the twist strain [1/(rad/m) = m / rad]
    "C_varepsilon": 9.1e-3 * ones_rod,  # Average: 0.009118994, Std: 0.000696435
    # outside radius of each rod [m]. The rows correspond to the segments.
    "rout": 25.4e-3 / 2 * ones_rod,  # this is for FPU rods
    # inside radius of each rod [m]. The rows correspond to the segments.
    "rin": (25.4e-3 / 2 - 2.43e-3) * ones_rod,  # this is for FPU rods
    # handedness of each rod. The rows correspond to the segments.
    "h": ones_rod,
    # offset [m] of each rod from the centerline. The rows correspond to the segments.
    "roff": jnp.array([[-24e-3, 24e-3]]),
    "pcudim": jnp.array(
        [[80e-3, 12e-3, 80e-3]]
    ),  # width, height, depth of the platform [m]
    # mass of FPU rod: 14 g, mass of EPU rod: 26 g
    # For FPU, this corresponds to a measure volume of 0000175355 m^3 --> rho = 798.38 kg/m^3
    "rhor": 798.38 * ones_rod,  # Volumetric density of rods [kg/m^3],
    # Volumetric density of platform [kg/m^3],
    # weight of platform + marker holder + cylinder top piece: 0.107 kg
    # subtracting 4 x 9g for distal cap: 0.071 kg
    # volume of platform (excluding proximal and distal caps): 0.0000768 m^3
    # --> rho = 925 kg/m^3
    "rhop": 925 * jnp.ones((num_segments,)),
    # volumetric density of the rigid end pieces [kg/m^3]
    # mass of 3D printed rod (between rin and rout): 8.5g
    # mass of the rigid end piece (up to rin): 9g
    # volume: pi*lpc*rout^2 = 0.0000126677 m^3
    # --> rho = 710.4 kg/m^3
    "rhoec": 710.4 * jnp.ones((num_segments,)),
    "g": jnp.array([0.0, -9.81]),
    "Ehat": 1e4 * ones_rod,  # Elastic modulus of each rod [Pa]
    "Ghat": 1e3 * ones_rod,  # Shear modulus of each rod [Pa]
    # Constant to scale the Elastic modulus linearly with the twist strain [Pa/(rad/m)]
    "C_E": 1e3 * ones_rod,
    # Constant to scale the Shear modulus linearly with the twist strain [Pa/(rad/m)]
    "C_G": 1e2 * ones_rod,
    # damping coefficient for bending of shape (num_segments, rods_per_segment)
    "zetab": 1e-4 * ones_rod,
    # damping coefficient for shear of shape (num_segments, rods_per_segment)
    "zetash": 1e-2 * ones_rod,
    # damping coefficient for axial elongation of shape (num_segments, rods_per_segment)
    "zetaa": 1e-2 * ones_rod,
}


def test_symbolic_and_numeric_implementation(seed: int = 0):
    (
        strain_basis,
        forward_kinematics_virtual_backbone_fn,
        forward_kinematics_rod_fn,
        forward_kinematics_platform_fn,
        forward_kinematics_end_effector_fn,
        jacobian_end_effector_fn,
        inverse_kinematics_end_effector_fn,
        dynamical_matrices_fn,
        sys_helpers,
    ) = planar_hsa.factory(sym_exp_filepath)

    # load saved symbolic data
    sym_exps = dill.load(open(str(sym_exp_filepath), "rb"))

    B_xi = sys_helpers["B_xi"]
    xi_eq = sys_helpers["xi_eq"]

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
        xi = xi_eq + B_xi @ q

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
    test_symbolic_and_numeric_implementation()
