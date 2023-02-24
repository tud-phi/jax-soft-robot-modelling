import cv2  # importing cv2
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
import jax
from jax import Array, jit, vmap
from jax import numpy as jnp
from functools import partial
import numpy as onp
from pathlib import Path
from typing import Callable, Dict, Tuple, Union

from jsrm.integration import ode_factory
from jsrm.systems import euler_lagrangian, pendulum

num_links = 2

num_links_to_sym_exp_filepath_map = {
    1: "single_pendulum.dill",
    2: "double_pendulum.dill",
    3: "triple_pendulum.dill",
}

sym_exp_filepath = (
    Path(__file__).parent.parent / "symbolic_expressions" / num_links_to_sym_exp_filepath_map[num_links]
)
params = {
    "m": jnp.array([10.0, 6.0]),
    "I": jnp.array([3.0, 2.0]),
    "l": jnp.array([2.0, 1.0]),
    "lc": jnp.array([1.0, 0.5]),
    "g": jnp.array([0.0, -9.81]),
}

# define initial configuration
q0 = jnp.zeros((num_links, ))

# set simulation parameters
dt = 1e-4  # time step
ts = jnp.arange(0.0, 5, dt)  # time steps
skip_step = 100  # how many time steps to skip in between video frames
video_ts = ts[::skip_step]  # time steps for video

# video settings
video_width, video_height = 700, 700  # img height and width
video_path = Path(__file__).parent / "videos" / f"{sym_exp_filepath.stem}.mp4"


def draw_robot(
    batched_forward_kinematics_fn: Callable,
    params: Dict[str, Array],
    q: Array,
    width: int,
    height: int,
) -> onp.ndarray:
    # plotting in OpenCV
    h, w = height, width  # img height and width
    ppm = h / (2.5 * jnp.sum(params["l"]))  # pixel per meter
    robot_color = (0, 0, 0)  # black robot_color in BGR

    # poses along the robot of shape (3, N)
    link_indices = jnp.arange(params["l"].shape[0], dtype=jnp.int32)
    chi_ls = jnp.zeros((link_indices.shape[0] + 1, 3))
    chi_ls = chi_ls.at[:, 1:].set(batched_forward_kinematics_fn(params, q, link_indices))

    img = 255 * onp.ones((w, h, 3), dtype=jnp.uint8)  # initialize background to white
    curve_origin = onp.array(
        [w // 2, h // 2], dtype=onp.int32
    )  # in x-y pixel coordinates
    # transform robot poses to pixel coordinates
    # should be of shape (N, 2)
    curve = onp.array((curve_origin + chi_ls[:2, :].T * ppm), dtype=onp.int32)
    # invert the v pixel coordinate
    curve[:, 1] = h - curve[:, 1]
    cv2.polylines(img, [curve], isClosed=False, color=robot_color, thickness=10)

    return img


if __name__ == "__main__":
    forward_kinematics_fn, dynamical_matrices_fn = pendulum.factory(sym_exp_filepath)
    batched_forward_kinematics = vmap(
        forward_kinematics_fn, in_axes=(None, None, 0), out_axes=-1
    )

    x0 = jnp.zeros((2 * q0.shape[0],))  # initial condition
    x0 = x0.at[: q0.shape[0]].set(q0)  # set initial configuration
    tau = jnp.zeros_like(q0)  # torques

    ode_fn = ode_factory(dynamical_matrices_fn, params, tau)
    term = ODETerm(ode_fn)

    sol = diffeqsolve(
        term, solver=Dopri5(), t0=ts[0], t1=ts[-1], dt0=dt, y0=x0, max_steps=100000, saveat=SaveAt(ts=video_ts)
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
