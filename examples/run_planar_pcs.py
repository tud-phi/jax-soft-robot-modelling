# importing cv2
import cv2
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
from jax import vmap
from jax import numpy as jnp
from functools import partial
import numpy as onp
from pathlib import Path

from jsrm.systems import euler_lagrangian
from jsrm.systems import planar_pcs

num_segments = 2

sym_exp_filepath = Path(__file__).parent.parent / "symbolic_expressions" / "planar_pcs_two_segments.dill"
params = {
    "rho": 1070 * jnp.ones((num_segments, )),  # Volumetric density of Dragon Skin 20 [kg/m^3]
    "l": jnp.array([1e-1, 1e-1]),
    "r": jnp.array([2e-2, 2e-2]),
    "g": jnp.array([0.0, -9.81]),
    "E": 1e7 * jnp.ones((num_segments, )),  # Elastic modulus [Pa]
    "G": 1e6 * jnp.ones((num_segments, )),  # Shear modulus [Pa]
}
# activate all strains (i.e. bending, shear, and axial)
strain_selector = jnp.ones((num_segments * 3, ), dtype=bool)

if __name__ == "__main__":
    strain_basis, forward_kinematics_fn, dynamical_matrices_fn = planar_pcs.factory(
        sym_exp_filepath,
        strain_selector
    )

    batched_forward_kinematics_fn = vmap(
        forward_kinematics_fn,
        in_axes=(None, None, 0),
        out_axes=-1
    )
    # we use for plotting N points along the length of the robot
    s_ps = jnp.linspace(0, jnp.sum(params["l"]), 50)

    # define q
    q = jnp.array([jnp.pi, 0.0, 0.5, -2*jnp.pi, 0.0, 0.0])
    # poses along the robot of shape (3, N)
    chi_ps = batched_forward_kinematics_fn(params, q, s_ps)[:, :]

    # plotting in OpenCV
    h, w = 500, 500  # img height and width
    ppm = h / (2.0 * jnp.sum(params["l"]))  # pixel per meter
    base_color = (0, 0, 0)  # black robot_color in BGR
    robot_color = (255, 0, 0)  # black robot_color in BGR

    import matplotlib.pyplot as plt
    plt.plot(chi_ps[0, :], chi_ps[1, :])
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.show()

    img = 255 * onp.ones((w, h, 3), dtype=jnp.uint8)  # initialize background to white
    curve_origin = onp.array([w // 2, 0.1 * h], dtype=onp.int32)  # in x-y pixel coordinates
    # draw base
    cv2.rectangle(img, (0, h - curve_origin[1]), (w, h), color=base_color, thickness=-1)
    # transform robot poses to pixel coordinates
    # should be of shape (N, 2)
    curve = onp.array((curve_origin + chi_ps[:2, :].T * ppm), dtype=onp.int32)
    # invert the v pixel coordinate
    curve[:, 1] = h - curve[:, 1]
    cv2.polylines(img, [curve], isClosed=False, color=robot_color, thickness=4)

    # Displaying the image
    winname = f"Planar PCS with {num_segments} segments"
    cv2.namedWindow(winname)
    cv2.imshow(winname, img)
    cv2.waitKey()
    cv2.destroyWindow(winname)
