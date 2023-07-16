__all__ = ["PARAMS_CONTROL", "PARAMS_SYSTEM_ID", "generate_base_params"]

import jax.numpy as jnp
from typing import Dict


def generate_base_params(num_segments: int = 1, num_rods_per_segment: int = 2) -> Dict:
    assert num_rods_per_segment % 2 == 0, "num_rods_per_segment must be even"

    ones_rod = jnp.ones((num_segments, num_rods_per_segment))
    params = {
        "th0": jnp.array(0.0),  # initial orientation angle [rad]
        "l": 59e-3 * jnp.ones((num_segments,)),  # length of each rod [m]
        # length of the rigid proximal caps of the rods connecting to the base [m]
        "lpc": 25e-3 * jnp.ones((num_segments,)),
        # length of the rigid distal caps of the rods connecting to the platform [m]
        "ldc": 14e-3 * jnp.ones((num_segments,)),
        "sigma_a_eq": 9.98051512e-01 * ones_rod,  # axial rest strains of each rod
        # scale factor for the rest length as a function of the twist strain [1/(rad/m) = m / rad]
        "C_varepsilon": 1.12848472e-02 * ones_rod,  # Average: 0.009118994, Std: 0.000696435
        # outside radius of each rod [m]. The rows correspond to the segments.
        "rout": 25.4e-3 / 2 * ones_rod,  # this is for FPU rods
        # inside radius of each rod [m]. The rows correspond to the segments.
        "rin": (25.4e-3 / 2 - 2.43e-3) * ones_rod,  # this is for FPU rods
        # handedness of each rod. The rows correspond to the segments.
        "h": ones_rod,
        # offset [m] of each rod from the centerline. The rows correspond to the segments.
        "roff": jnp.repeat(
            jnp.repeat(
                jnp.array([[-24e-3, 24e-3]]),
                num_rods_per_segment // 2,
                axis=1
            ),
            num_segments,
            axis=0
        ),
        "pcudim": jnp.repeat(jnp.array(
            [[80e-3, 12e-3, 80e-3]]
        ), num_segments, axis=0),  # width, height, depth of the platform [m]
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
        "g": jnp.array([0.0, 9.81]),
        "Ehat": 4.67725384e03 * ones_rod,  # Elastic modulus of each rod [Pa]
        "Ghat": 4.11478872e03 * ones_rod,  # Shear modulus of each rod [Pa]
        # Constant to scale the Elastic modulus linearly with the twist strain [Pa/(rad/m)]
        # 53 rad / m is roughly the twist strain at 180 deg twist angle
        "C_E": 67.74549753 * ones_rod,
        # Constant to scale the Shear modulus linearly with the twist strain [Pa/(rad/m)]
        # 53 rad / m is roughly the twist strain at 180 deg twist angle
        "C_G": -1.64919053e01 * ones_rod,
        "S_b_sh": 7.56629739e-03 * ones_rod,  # Elastic coupling between bending and shear
        # damping coefficient for bending of shape (num_segments, rods_per_segment)
        "zetab": 8e-6 * ones_rod,
        # damping coefficient for shear of shape (num_segments, rods_per_segment)
        "zetash": 2e-4 * ones_rod,
        # damping coefficient for axial elongation of shape (num_segments, rods_per_segment)
        "zetaa": 2e-3 * ones_rod,
        "phi_max": 210 / 180 * jnp.pi * ones_rod,  # maximum twist angles (positive) [rad]
    }

    return params


PARAMS_SYSTEM_ID = generate_base_params(num_segments=1, num_rods_per_segment=4)
PARAMS_SYSTEM_ID.update({
    "h": jnp.array([[1.0, -1.0, 1.0, -1.0]]),
    "roff": 24e-3 * jnp.array([[1.0, 1.0, -1.0, -1.0]]),
})

PARAMS_CONTROL = generate_base_params(num_segments=1, num_rods_per_segment=2)
PARAMS_CONTROL.update({
    "rhor": 2 * PARAMS_CONTROL["rhor"],
    "rhoec": 2 * PARAMS_CONTROL["rhoec"],
    "Ehat": 2 * PARAMS_CONTROL["Ehat"],
    "Ghat": 2 * PARAMS_CONTROL["Ghat"],
    "C_E": 2 * PARAMS_CONTROL["C_E"],
    "C_G": 2 * PARAMS_CONTROL["C_G"],
    "zetab": 2 * PARAMS_CONTROL["zetab"],
    "zetash": 2 * PARAMS_CONTROL["zetash"],
    "zetaa": 2 * PARAMS_CONTROL["zetaa"],
})
