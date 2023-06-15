import cv2  # importing cv2
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
from diffrax import diffeqsolve, Dopri5, Euler, ODETerm, SaveAt
from jax import Array, jit, vmap
from jax import numpy as jnp
from functools import partial
import numpy as onp
from pathlib import Path
from typing import Callable, Dict

from jsrm.integration import ode_factory
from jsrm.systems import euler_lagrangian
from jsrm.systems import planar_hsa

num_segments = 1

# filepath to symbolic expressions
sym_exp_filepath = (
    Path(__file__).parent.parent
    / "symbolic_expressions"
    / "planar_hsa_ns-1_nrs-2.dill"
)

# set parameters
rods_per_segment = 2
# Damping coefficient
zeta = 1e-5 * jnp.repeat(
    jnp.repeat(
        jnp.diag(jnp.array([1e0, 1e3, 1e3])).reshape((1, 1, 3, 3)),
        axis=1, repeats=rods_per_segment
    ),
    axis=0, repeats=num_segments
)
params = {
    "l": jnp.array([1e-1, 1e-1]),  # length of each rod [m]
    # outside radius of each rod [m]. The rows correspond to the segments.
    "rout": 25.4e-3 / 2 * jnp.ones((num_segments, rods_per_segment)),
    # inside radius of each rod [m]. The rows correspond to the segments.
    "rin": (25.4e-3 / 2-2.43e-3) * jnp.ones((num_segments, rods_per_segment)),
    # handedness of each rod. The rows correspond to the segments.
    "h": jnp.ones((num_segments, rods_per_segment)),
    # offset [m] of each rod from the centerline. The rows correspond to the segments.
    "roff": jnp.array([[-24e-3, 24e-3]]),
    "pcudim": jnp.array([95e-3, 3e-3, 95e-3]),  # width, height, depth of the platform [m]
    "rhor": 1.05e3 * jnp.ones((num_segments, rods_per_segment)),  # Volumetric density of rods [kg/m^3],
    "rhop": 0.7e3 * jnp.ones((num_segments, )),  # Volumetric density of platform [kg/m^3],
    "g": jnp.array([0.0, -9.81]),
    "E": 1e4 * jnp.ones((num_segments, rods_per_segment)),  # Elastic modulus of each rod [Pa]
    "G": 1e3 * jnp.ones((num_segments, rods_per_segment)),  # Shear modulus of each rod [Pa]
    # Constant to scale the Elastic modulus linearly with the twist strain [Pa/(rad/m)]
    "C_E": 0e0 * jnp.ones((num_segments, rods_per_segment)),
    # Constant to scale the Shear modulus linearly with the twist strain [Pa/(rad/m)]
    "C_G": 0e0 * jnp.ones((num_segments, rods_per_segment)),
    "zeta": zeta,  # damping coefficient of shape (num_segments, rods_per_segment, 3, 3)
}

# activate all strains (i.e. bending, shear, and axial)
strain_selector = jnp.ones((3 * num_segments,), dtype=bool)

# define initial configuration
q0 = jnp.array([0.0, 0.0, 0.0])

# set simulation parameters
dt = 1e-4  # time step
ts = jnp.arange(0.0, 5, dt)  # time steps
skip_step = 100  # how many time steps to skip in between video frames
video_ts = ts[::skip_step]  # time steps for video

# video settings
video_width, video_height = 700, 700  # img height and width
video_path = Path(__file__).parent / "videos" / "planar_hsa.mp4"


def draw_robot(
    batched_forward_kinematics_fn: Callable,
    params: Dict[str, Array],
    q: Array,
    width: int,
    height: int,
    num_points: int = 50,
) -> onp.ndarray:
    # plotting in OpenCV
    h, w = height, width  # img height and width
    ppm = h / (2.0 * jnp.sum(params["l"]))  # pixel per meter
    base_color = (0, 0, 0)  # black robot_color in BGR
    robot_color = (255, 0, 0)  # black robot_color in BGR

    # we use for plotting N points along the length of the robot
    s_ps = jnp.linspace(0, jnp.sum(params["l"]), num_points)

    # poses along the robot of shape (3, N)
    chi_ps = batched_forward_kinematics_fn(params, q, s_ps)

    img = 255 * onp.ones((w, h, 3), dtype=jnp.uint8)  # initialize background to white
    curve_origin = onp.array(
        [w // 2, 0.1 * h], dtype=onp.int32
    )  # in x-y pixel coordinates
    # draw base
    cv2.rectangle(img, (0, h - curve_origin[1]), (w, h), color=base_color, thickness=-1)
    # transform robot poses to pixel coordinates
    # should be of shape (N, 2)
    curve = onp.array((curve_origin + chi_ps[:2, :].T * ppm), dtype=onp.int32)
    # invert the v pixel coordinate
    curve[:, 1] = h - curve[:, 1]
    cv2.polylines(img, [curve], isClosed=False, color=robot_color, thickness=10)

    return img


if __name__ == "__main__":
    (
        strain_basis,
        forward_kinematics_virtual_backbone_fn, forward_kinematics_rod_fn, forward_kinematics_platform_fn,
        dynamical_matrices_fn
    ) = planar_hsa.factory(
        sym_exp_filepath, strain_selector
    )
    batched_forward_kinematics_virtual_backbone_fn = vmap(
        forward_kinematics_virtual_backbone_fn, in_axes=(None, None, 0), out_axes=-1
    )

    s_ps = jnp.linspace(0, jnp.sum(params["l"]), 100)
    chi_ps = batched_forward_kinematics_virtual_backbone_fn(params, q0, s_ps)

    import matplotlib.pyplot as plt
    plt.plot(chi_ps[0, :], chi_ps[1, :])
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.show()

    # Displaying the image
    window_name = f"Planar HSA with {num_segments} segments"
    img = draw_robot(batched_forward_kinematics_virtual_backbone_fn, params, q0, video_width, video_height)
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, img)
    cv2.waitKey()
    cv2.destroyWindow(window_name)

    exit()

    x0 = jnp.zeros((2 * q0.shape[0],))  # initial condition
    x0 = x0.at[: q0.shape[0]].set(q0)  # set initial configuration
    tau = jnp.zeros_like(q0)  # torques

    ode_fn = ode_factory(dynamical_matrices_fn, params, tau)
    term = ODETerm(ode_fn)

    sol = diffeqsolve(
        term, solver=Euler(), t0=ts[0], t1=ts[-1], dt0=dt, y0=x0, max_steps=None, saveat=SaveAt(ts=video_ts)
    )

    print("sol.ys =\n", sol.ys)

    # create video
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(
        str(video_path),
        fourcc,
        1 / (skip_step * dt),  # fps
        (video_width, video_height),
    )

    for time_idx, t in enumerate(video_ts):
        x = sol.ys[time_idx]
        img = draw_robot(
            batched_forward_kinematics,
            params,
            x[: (x.shape[0] // 2)],
            video_width,
            video_height,
        )
        video.write(img)

    video.release()
