import cv2  # importing cv2
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
from diffrax import diffeqsolve, Euler, ODETerm, SaveAt
from jax import Array, jit, vmap
from jax import numpy as jnp
from functools import partial
import numpy as onp
from pathlib import Path
from typing import Callable, Dict

import jsrm
from jsrm.parameters.hsa_params import PARAMS_CONTROL as params
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

# define initial configuration
q0 = jnp.array([jnp.pi, 0.0, 0.0])
phi = jnp.array([jnp.pi, jnp.pi / 2])  # motor actuation angles

# set simulation parameters
dt = 1e-4  # time step
ts = jnp.arange(0.0, 5, dt)  # time steps
skip_step = 100  # how many time steps to skip in between video frames
video_ts = ts[::skip_step]  # time steps for video

# video settings
video_width, video_height = 700, 700  # img height and width
video_path = Path(__file__).parent / "videos" / "planar_hsa.mp4"


def draw_robot(
    batched_forward_kinematics_virtual_backbone_fn: Callable,
    batched_forward_kinematics_rod_fn: Callable,
    batched_forward_kinematics_platform_fn: Callable,
    params: Dict[str, Array],
    q: Array,
    width: int,
    height: int,
    num_points: int = 50,
) -> onp.ndarray:
    # plotting in OpenCV
    h, w = height, width  # img height and width
    ppm = h / (
        2.0 * jnp.sum(params["lpc"] + params["l"] + params["ldc"])
    )  # pixel per meter
    base_color = (0, 0, 0)  # black base color in BGR
    backbone_color = (255, 0, 0)  # blue robot color in BGR
    rod_color = (0, 255, 0)  # green rod color in BGR
    platform_color = (0, 0, 255)  # red platform color in BGR

    # we use for plotting N points along the length of the robot
    s_ps = jnp.linspace(0, jnp.sum(params["l"]), num_points)

    # poses along the robot of shape (3, N)
    chiv_ps = batched_forward_kinematics_virtual_backbone_fn(
        params, q, s_ps
    )  # poses of virtual backbone
    chiL_ps = batched_forward_kinematics_rod_fn(params, q, s_ps, 0)  # poses of left rod
    chiR_ps = batched_forward_kinematics_rod_fn(params, q, s_ps, 1)  # poses of left rod
    # poses of the platforms
    chip_ps = batched_forward_kinematics_platform_fn(
        params, q, jnp.arange(0, num_segments)
    )

    img = 255 * onp.ones((w, h, 3), dtype=jnp.uint8)  # initialize background to white
    uv_robot_origin = onp.array(
        [w // 2, h * (1 - 0.1)], dtype=jnp.int32
    )  # in x-y pixel coordinates

    @jit
    def chi2u(chi: Array) -> Array:
        """
        Map Cartesian coordinates to pixel coordinates.
        Args:
            chi: Cartesian poses of shape (3)

        Returns:
            uv: pixel coordinates of shape (2)
        """
        uv_off = jnp.array((chi[:2] * ppm), dtype=jnp.int32)
        # invert the v pixel coordinate
        uv_off = uv_off.at[1].set(-uv_off[1])
        # invert the v pixel coordinate
        uv = uv_robot_origin + uv_off
        return uv

    batched_chi2u = vmap(chi2u, in_axes=-1, out_axes=0)

    # draw base
    cv2.rectangle(img, (0, uv_robot_origin[1]), (w, h), color=base_color, thickness=-1)

    # draw the virtual backbone
    # add the first point of the proximal cap and the last point of the distal cap
    chiv_ps = jnp.concatenate(
        [
            (chiv_ps[:, 0] - jnp.array([0.0, params["lpc"][0], 0.0])).reshape(3, 1),
            chiv_ps,
            (
                chiv_ps[:, -1]
                + jnp.array(
                    [
                        -jnp.sin(chiv_ps[2, -1]) * params["ldc"][-1],
                        jnp.cos(chiv_ps[2, -1]) * params["ldc"][-1],
                        chiv_ps[2, -1],
                    ]
                )
            ).reshape(3, 1),
        ],
        axis=1,
    )
    curve_virtual_backbone = onp.array(batched_chi2u(chiv_ps))
    cv2.polylines(
        img, [curve_virtual_backbone], isClosed=False, color=backbone_color, thickness=5
    )

    # draw the rods
    # add the first point of the proximal cap and the last point of the distal cap
    chiL_ps = jnp.concatenate(
        [
            (chiL_ps[:, 0] - jnp.array([0.0, params["lpc"][0], 0.0])).reshape(3, 1),
            chiL_ps,
            (
                chiL_ps[:, -1]
                + jnp.array(
                    [
                        -jnp.sin(chiL_ps[2, -1]) * params["ldc"][-1],
                        jnp.cos(chiL_ps[2, -1]) * params["ldc"][-1],
                        chiL_ps[2, -1],
                    ]
                )
            ).reshape(3, 1),
        ],
        axis=1,
    )
    curve_rod_left = onp.array(batched_chi2u(chiL_ps))
    cv2.polylines(
        img,
        [curve_rod_left],
        isClosed=False,
        color=rod_color,
        thickness=10,
        # thickness=2*int(ppm * params["rout"].mean(axis=0)[0])
    )
    # add the first point of the proximal cap and the last point of the distal cap
    chiR_ps = jnp.concatenate(
        [
            (chiR_ps[:, 0] - jnp.array([0.0, params["lpc"][0], 0.0])).reshape(3, 1),
            chiR_ps,
            (
                chiR_ps[:, -1]
                + jnp.array(
                    [
                        -jnp.sin(chiR_ps[2, -1]) * params["ldc"][-1],
                        jnp.cos(chiR_ps[2, -1]) * params["ldc"][-1],
                        chiR_ps[2, -1],
                    ]
                )
            ).reshape(3, 1),
        ],
        axis=1,
    )
    curve_rod_right = onp.array(batched_chi2u(chiR_ps))
    cv2.polylines(img, [curve_rod_right], isClosed=False, color=rod_color, thickness=10)

    # draw the platform
    for i in range(chip_ps.shape[0]):
        # iterate over the platforms
        platform_R = jnp.array(
            [
                [jnp.cos(chip_ps[i, 2]), -jnp.sin(chip_ps[i, 2])],
                [jnp.sin(chip_ps[i, 2]), jnp.cos(chip_ps[i, 2])],
            ]
        )  # rotation matrix for the platform
        platform_llc = chip_ps[i, :2] + platform_R @ jnp.array(
            [
                -params["pcudim"][i, 0] / 2,  # go half the width to the left
                -params["pcudim"][i, 1] / 2,  # go half the height down
            ]
        )  # lower left corner of the platform
        platform_ulc = chip_ps[i, :2] + platform_R @ jnp.array(
            [
                -params["pcudim"][i, 0] / 2,  # go half the width to the left
                +params["pcudim"][i, 1] / 2,  # go half the height down
            ]
        )  # upper left corner of the platform
        platform_urc = chip_ps[i, :2] + platform_R @ jnp.array(
            [
                +params["pcudim"][i, 0] / 2,  # go half the width to the left
                +params["pcudim"][i, 1] / 2,  # go half the height down
            ]
        )  # upper right corner of the platform
        platform_lrc = chip_ps[i, :2] + platform_R @ jnp.array(
            [
                +params["pcudim"][i, 0] / 2,  # go half the width to the left
                -params["pcudim"][i, 1] / 2,  # go half the height down
            ]
        )  # lower right corner of the platform
        platform_curve = jnp.stack(
            [platform_llc, platform_ulc, platform_urc, platform_lrc, platform_llc],
            axis=1,
        )
        # cv2.polylines(img, [onp.array(batched_chi2u(platform_curve))], isClosed=True, color=platform_color, thickness=5)
        cv2.fillPoly(
            img, [onp.array(batched_chi2u(platform_curve))], color=platform_color
        )

        # # plot the CoG of the platform
        # CoG_color = (255, 255, 255)  # coloring for plotting center of gravities. White in BGR
        # cv2.circle(
        #     img,
        #     center=onp.array(chi2u(chip_ps[i, :2])),
        #     radius=10,
        #     thickness=2,
        #     color=CoG_color,
        # )

    return img


if __name__ == "__main__":
    (
        forward_kinematics_virtual_backbone_fn,
        forward_kinematics_end_effector_fn,
        jacobian_end_effector_fn,
        inverse_kinematics_end_effector_fn,
        dynamical_matrices_fn,
        sys_helpers,
    ) = planar_hsa.factory(sym_exp_filepath, strain_selector)

    batched_forward_kinematics_virtual_backbone_fn = vmap(
        forward_kinematics_virtual_backbone_fn, in_axes=(None, None, 0), out_axes=-1
    )
    batched_forward_kinematics_rod_fn = vmap(
        sys_helpers["forward_kinematics_rod_fn"], in_axes=(None, None, 0, None), out_axes=-1
    )
    batched_forward_kinematics_platform_fn = vmap(
        sys_helpers["forward_kinematics_platform_fn"], in_axes=(None, None, 0), out_axes=0
    )

    s_ps = jnp.linspace(0, jnp.sum(params["l"]), 100)
    chi_ps = batched_forward_kinematics_virtual_backbone_fn(params, q0, s_ps)

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
        batched_forward_kinematics_virtual_backbone_fn,
        batched_forward_kinematics_rod_fn,
        batched_forward_kinematics_platform_fn,
        params,
        q0,
        video_width,
        video_height,
    )
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, img)
    cv2.waitKey()
    cv2.destroyWindow(window_name)

    x0 = jnp.zeros((2 * q0.shape[0],))  # initial condition
    x0 = x0.at[: q0.shape[0]].set(q0)  # set initial configuration

    ode_fn = planar_hsa.ode_factory(dynamical_matrices_fn, params)
    ode_term = ODETerm(partial(ode_fn, u=phi))

    sol = diffeqsolve(
        ode_term,
        solver=Euler(),
        t0=ts[0],
        t1=ts[-1],
        dt0=dt,
        y0=x0,
        max_steps=None,
        saveat=SaveAt(ts=video_ts),
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
            batched_forward_kinematics_virtual_backbone_fn,
            batched_forward_kinematics_rod_fn,
            batched_forward_kinematics_platform_fn,
            params,
            x[: (x.shape[0] // 2)],
            video_width,
            video_height,
        )
        video.write(img)

    video.release()
