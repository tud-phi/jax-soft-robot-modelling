import cv2  # importing cv2
import jax

jax.config.update("jax_enable_x64", True)  # double precision
from diffrax import diffeqsolve, Euler, ODETerm, SaveAt, Tsit5
from jax import Array, jit, vmap
from jax import numpy as jnp
from functools import partial
import numpy as onp
from pathlib import Path
from typing import Callable, Dict

import jsrm
from jsrm.parameters.hsa_params import PARAMS_FPU_CONTROL, PARAMS_FPU_HYSTERESIS_CONTROL
from jsrm.rendering.planar_hsa.opencv_renderer import draw_robot, animate_robot
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
consider_hysteresis = True

params = PARAMS_FPU_HYSTERESIS_CONTROL if consider_hysteresis else PARAMS_FPU_CONTROL

# define initial configuration
q0 = jnp.array([jnp.pi, 0.0, 0.0])
phi = jnp.array([jnp.pi, jnp.pi / 2])  # motor actuation angles

# set simulation parameters
dt = 5e-5  # time step
ts = jnp.arange(0.0, 5, dt)  # time steps
skip_step = 100  # how many time steps to skip in between video frames
video_ts = ts[::skip_step]  # time steps for video

# increase damping for simulation stability
params["zetab"] = 5 * params["zetab"]
params["zetash"] = 5 * params["zetash"]
params["zetaa"] = 5 * params["zetaa"]

# video settings
video_width, video_height = 700, 700  # img height and width
video_path = Path(__file__).parent / "videos" / "planar_hsa.mp4"


if __name__ == "__main__":
    (
        forward_kinematics_virtual_backbone_fn,
        forward_kinematics_end_effector_fn,
        jacobian_end_effector_fn,
        inverse_kinematics_end_effector_fn,
        dynamical_matrices_fn,
        sys_helpers,
    ) = planar_hsa.factory(
        sym_exp_filepath, strain_selector, consider_hysteresis=consider_hysteresis
    )

    # import matplotlib.pyplot as plt
    # plt.plot(chi_ps[0, :], chi_ps[1, :])
    # plt.axis("equal")
    # plt.grid(True)
    # plt.xlabel("x [m]")
    # plt.ylabel("y [m]")
    # plt.show()

    # Displaying the image
    window_name = f"Planar HSA with {num_segments} segments"
    img = draw_robot(
        forward_kinematics_virtual_backbone_fn,
        sys_helpers["forward_kinematics_rod_fn"],
        sys_helpers["forward_kinematics_platform_fn"],
        params,
        q0,
        video_width,
        video_height,
    )
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, img)
    cv2.waitKey()
    cv2.destroyWindow(window_name)

    if consider_hysteresis:
        x0 = jnp.zeros((2 * q0.shape[0] + 1,))  # initial condition
    else:
        x0 = jnp.zeros((2 * q0.shape[0],))  # initial condition
    x0 = x0.at[: q0.shape[0]].set(q0)  # set initial configuration

    ode_fn = planar_hsa.ode_factory(
        dynamical_matrices_fn, params, consider_hysteresis=consider_hysteresis
    )
    ode_term = ODETerm(ode_fn)

    sol = diffeqsolve(
        ode_term,
        solver=Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=dt,
        y0=x0,
        args=phi,
        max_steps=None,
        saveat=SaveAt(ts=video_ts),
    )

    print("sol.ys =\n", sol.ys)

    # create video
    video_path.parent.mkdir(parents=True, exist_ok=True)
    animate_robot(
        forward_kinematics_virtual_backbone_fn,
        sys_helpers["forward_kinematics_rod_fn"],
        sys_helpers["forward_kinematics_platform_fn"],
        params,
        video_path,
        video_ts=video_ts,
        q_ts=sol.ys[:, :3],
        video_width=video_width,
        video_height=video_height,
    )
    print(f"Video saved at {video_path}")
