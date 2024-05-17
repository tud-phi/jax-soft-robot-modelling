import cv2  # importing cv2
from functools import partial
import jax

jax.config.update("jax_enable_x64", True)  # double precision
from diffrax import diffeqsolve, Dopri5, Euler, ODETerm, SaveAt
from jax import Array, vmap
from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
from pathlib import Path
from typing import Callable, Dict

import jsrm
from jsrm import ode_factory
from jsrm.systems import planar_pcs

num_segments = 1

# filepath to symbolic expressions
sym_exp_filepath = (
    Path(jsrm.__file__).parent
    / "symbolic_expressions"
    / f"planar_pcs_ns-{num_segments}.dill"
)

# set parameters
rho = 1070 * jnp.ones((num_segments,))  # Volumetric density of Dragon Skin 20 [kg/m^3]
D = 5e-6 * jnp.diag(jnp.array([1e0, 1e3, 1e3]))  # Damping coefficient
params = {
    "th0": jnp.array(0.0),  # initial orientation angle [rad]
    "l": 1e-1 * jnp.ones((num_segments,)),
    "r": 2e-2 * jnp.ones((num_segments,)),
    "rho": rho,
    "g": jnp.array([0.0, 9.81]),
    "E": 2e2 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
    "G": 1e2 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
    "D": D,
}

# activate all strains (i.e. bending, shear, and axial)
strain_selector = jnp.ones((3 * num_segments,), dtype=bool)
strain_selector = jnp.array([True, False, False])

# define initial configuration
q0 = jnp.array([10 * jnp.pi])
# number of generalized coordinates
n_q = q0.shape[0]

# set simulation parameters
dt = 1e-3  # time step
ts = jnp.arange(0.0, 2, dt)  # time steps
skip_step = 10  # how many time steps to skip in between video frames
video_ts = ts[::skip_step]  # time steps for video

# video settings
video_width, video_height = 700, 700  # img height and width
video_path = Path(__file__).parent / "videos" / f"planar_pcs_ns-{num_segments}.mp4"


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
    strain_basis, forward_kinematics_fn, dynamical_matrices_fn, auxiliary_fns = planar_pcs.factory(
        sym_exp_filepath, strain_selector
    )
    batched_forward_kinematics = vmap(
        forward_kinematics_fn, in_axes=(None, None, 0), out_axes=-1
    )

    # import matplotlib.pyplot as plt
    # plt.plot(chi_ps[0, :], chi_ps[1, :])
    # plt.axis("equal")
    # plt.grid(True)
    # plt.xlabel("x [m]")
    # plt.ylabel("y [m]")
    # plt.show()

    # Displaying the image
    # window_name = f"Planar PCS with {num_segments} segments"
    # img = draw_robot(batched_forward_kinematics, params, q0, video_width, video_height)
    # cv2.namedWindow(window_name)
    # cv2.imshow(window_name, img)
    # cv2.waitKey()
    # cv2.destroyWindow(window_name)

    x0 = jnp.zeros((2 * q0.shape[0],))  # initial condition
    x0 = x0.at[: q0.shape[0]].set(q0)  # set initial configuration
    tau = jnp.zeros_like(q0)  # torques

    ode_fn = ode_factory(dynamical_matrices_fn, params, tau)
    term = ODETerm(ode_fn)

    sol = diffeqsolve(
        term,
        solver=Dopri5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=dt,
        y0=x0,
        max_steps=None,
        saveat=SaveAt(ts=video_ts),
    )

    print("sol.ys =\n", sol.ys)
    # the evolution of the generalized coordinates
    q_ts = sol.ys[:, :n_q]
    # the evolution of the generalized velocities
    q_d_ts = sol.ys[:, n_q:]

    # plot the energy along the trajectory
    kinetic_energy_fn_vmapped = vmap(partial(auxiliary_fns["kinetic_energy_fn"], params))
    potential_energy_fn_vmapped = vmap(partial(auxiliary_fns["potential_energy_fn"], params))
    U_ts = potential_energy_fn_vmapped(q_ts)
    T_ts = kinetic_energy_fn_vmapped(q_ts, q_d_ts)
    plt.figure()
    plt.plot(video_ts, U_ts, label="Potential energy")
    plt.plot(video_ts, T_ts, label="Kinetic energy")
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [J]")
    plt.legend()
    plt.grid(True)
    plt.box(True)
    plt.show()

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
