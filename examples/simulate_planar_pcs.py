import cv2  # importing cv2
from functools import partial
import jax

jax.config.update("jax_enable_x64", True)  # double precision
from diffrax import diffeqsolve, Euler, ODETerm, SaveAt, Tsit5
from jax import Array, vmap
from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
from pathlib import Path
from typing import Callable, Dict

import jsrm
from jsrm import ode_factory
from jsrm.systems import planar_pcs, planar_pcs_num

import time

# ==================================================
# Simulation parameters
# ==================================================
num_segments = 2

type_of_derivation = "numeric" #"symbolic" #
type_of_integration = "trapezoid" #"gauss" #

if type_of_derivation == "numeric":
    if type_of_integration == "gauss":
        param_integration = 30
    elif type_of_integration == "trapezoid":
        param_integration = 100

if type_of_derivation == "symbolic":
    # filepath to symbolic expressions
    sym_exp_filepath = (
        Path(jsrm.__file__).parent
        / "symbolic_expressions"
        / f"planar_pcs_ns-{num_segments}.dill"
    )

# set parameters
rho = 1070 * jnp.ones((num_segments,))  # Volumetric density of Dragon Skin 20 [kg/m^3]
params = {
    "th0": jnp.array(0.0),  # initial orientation angle [rad]
    "l": 1e-1 * jnp.ones((num_segments,)),
    "r": 2e-2 * jnp.ones((num_segments,)),
    "rho": rho,
    "g": jnp.array([0.0, 9.81]),
    "E": 2e3 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
    "G": 1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
}
params["D"] = 1e-3 * jnp.diag(
    (jnp.repeat(
        jnp.array([[1e0, 1e3, 1e3]]), num_segments, axis=0
    ) * params["l"][:, None]).flatten()
)

# activate all strains (i.e. bending, shear, and axial)
strain_selector = jnp.ones((3 * num_segments,), dtype=bool)

# define initial configuration
q0 = jnp.repeat(jnp.array([5.0 * jnp.pi, 0.1, 0.2])[None, :], num_segments, axis=0).flatten()
# number of generalized coordinates
n_q = q0.shape[0]

# ===================================================
# For video generation
# ======================
# set simulation parameters
dt = 1e-4  # time step
ts = jnp.arange(0.0, 1, dt)  # time steps
skip_step = 10  # how many time steps to skip in between video frames
video_ts = ts[::skip_step]  # time steps for video

# video settings
video_width, video_height = 700, 700  # img height and width"
video_path_parent = Path(__file__).parent / "videos" 
video_path_parent.mkdir(parents=True, exist_ok=True)
extension = f"planar_pcs_ns-{num_segments}-{('symb' if type_of_derivation == 'symbolic' else 'num')}"
if type_of_derivation == "numeric":
    extension += f"-{type_of_integration}-{param_integration}"
elif type_of_derivation == "symbolic":
    extension += "-symb"
video_path = video_path_parent / f"{extension}.mp4"

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

# ===================================================
# For figure saving
# ======================
figures_path_parent = Path(__file__).parent / "figures" 
extension = f"planar_pcs_ns-{num_segments}-{('symb' if type_of_derivation == 'symbolic' else 'num')}"
if type_of_derivation == "numeric":
    extension += f"-{type_of_integration}-{param_integration}"
figures_path_parent.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print("Type of derivation:", type_of_derivation)
    if type_of_derivation == "numeric":
        print("Type of integration:", type_of_integration)
        print("Parameter for integration:", param_integration)
    print("Number of segments:", num_segments, "\n")
    
    print("Importing the planar PCS model...")
    timer_start = time.time()
    if type_of_derivation == "symbolic":
        strain_basis, forward_kinematics_fn, dynamical_matrices_fn, auxiliary_fns = (
            planar_pcs.factory(sym_exp_filepath, strain_selector)
        )
        
    elif type_of_derivation == "numeric":
        strain_basis, forward_kinematics_fn, dynamical_matrices_fn, auxiliary_fns = (
            planar_pcs_num.factory(num_segments, strain_selector, integration_type=type_of_integration, param_integration=param_integration)
        )
    else:
        raise ValueError("type_of_derivation must be 'symbolic' or 'numeric'")
        
    # jit the functions
    print("JIT-compiling the dynamical matrices function ...")
    dynamical_matrices_fn = jax.jit(partial(dynamical_matrices_fn))
    batched_forward_kinematics = vmap(
        forward_kinematics_fn, in_axes=(None, None, 0), out_axes=-1
    )
    timer_end = time.time()
    print(f"Importing the planar PCS model took {timer_end - timer_start:.2f} seconds. \n")

    print("Evaluating the dynamical matrices...")
    timer_start = time.time()
    B, C, G, K, D, A = dynamical_matrices_fn(params, q0, jnp.zeros_like(q0))
    print("B =\n", B)
    print("C =\n", C)
    print("G =\n", G)
    print("K =\n", K)
    print("D =\n", D)
    print("A =\n", A)
    timer_end = time.time()
    print(f"Evaluating the dynamical matrices took {timer_end - timer_start:.2f} seconds. \n")

    # Parameter for the simulation
    x0 = jnp.concatenate([q0, jnp.zeros_like(q0)])  # initial condition
    tau = jnp.zeros_like(q0)  # torques

    ode_fn = ode_factory(dynamical_matrices_fn, params, tau)
    term = ODETerm(ode_fn)

    # jit the functions
    print("JIT-compiling the ODE function...")
    diffeqsolve_fn = jax.jit(
        partial(diffeqsolve, 
                term, 
                solver=Tsit5(), 
                t0=ts[0], 
                t1=ts[-1], 
                dt0=dt, 
                y0=x0, 
                max_steps=None, 
                saveat=SaveAt(ts=video_ts)))
    
    print("Solving the ODE for the first time (JIT-compilation)...")
    timer_start = time.time()
    sol = diffeqsolve_fn()
    print("sol.ys =\n", sol.ys)
    timer_end = time.time()
    # the evolution of the generalized coordinates
    q_ts = sol.ys[:, :n_q]
    # the evolution of the generalized velocities
    q_d_ts = sol.ys[:, n_q:]
    print(f"Solving the ODE took {timer_end - timer_start:.2f} seconds. \n")
    
    print("Solving the ODE for the second time (after JIT-compilation)...")
    timer_start = time.time()
    sol = diffeqsolve_fn()
    print("sol.ys =\n", sol.ys)
    timer_end = time.time()
    print(f"Solving the ODE for a second time took {timer_end - timer_start:.2f} seconds. \n")

    print("Evaluating the forward kinematics...")
    timer_start = time.time()
    # evaluate the forward kinematics along the trajectory
    chi_ee_ts = vmap(forward_kinematics_fn, in_axes=(None, 0, None))(
        params, q_ts, jnp.array([jnp.sum(params["l"])])
    )
    print("chi_ee_ts =\n", chi_ee_ts)
    timer_end = time.time()
    print(f"Evaluating the forward kinematics took {timer_end - timer_start:.2f} seconds. ")
    
    # plot the configuration vs time
    plt.figure()
    plt.title("Configuration vs Time")
    for segment_idx in range(num_segments):
        plt.plot(
            video_ts, q_ts[:, 3 * segment_idx + 0],
            label=r"$\kappa_\mathrm{be," + str(segment_idx + 1) + "}$ [rad/m]"
        )
        plt.plot(
            video_ts, q_ts[:, 3 * segment_idx + 1],
            label=r"$\sigma_\mathrm{sh," + str(segment_idx + 1) + "}$ [-]"
        )
        plt.plot(
            video_ts, q_ts[:, 3 * segment_idx + 2],
            label=r"$\sigma_\mathrm{ax," + str(segment_idx + 1) + "}$ [-]"
        )
    plt.xlabel("Time [s]")
    plt.ylabel("Configuration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        figures_path_parent / f"config_vs_time_{extension}.png", bbox_inches="tight", dpi=300
    )
    print("Figures saved at", figures_path_parent / f"config_vs_time_{extension}.png")
    plt.show()
    
    # plot end-effector position vs time
    plt.figure()
    plt.title("End-effector position vs Time")
    plt.plot(video_ts, chi_ee_ts[:, 0], label="x")
    plt.plot(video_ts, chi_ee_ts[:, 1], label="y")
    plt.xlabel("Time [s]")
    plt.ylabel("End-effector Position [m]")
    plt.legend()
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.savefig(
        figures_path_parent / f"end_effector_position_vs_time_{extension}.png", bbox_inches="tight", dpi=300
    )
    print("Figures saved at", figures_path_parent / f"end_effector_position_vs_time_{extension}.png ")
    plt.show()
    
    # plot the end-effector position in the x-y plane as a scatter plot with the time as the color
    plt.figure()
    plt.title("End-effector position in the x-y plane")
    plt.scatter(chi_ee_ts[:, 0], chi_ee_ts[:, 1], c=video_ts, cmap="viridis")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("End-effector x [m]")
    plt.ylabel("End-effector y [m]")
    plt.colorbar(label="Time [s]")
    plt.tight_layout()
    plt.savefig(
        figures_path_parent / f"end_effector_position_xy_{extension}.png", bbox_inches="tight", dpi=300
    )
    print("Figures saved at", figures_path_parent / f"end_effector_position_xy_{extension}.png \n")
    plt.show()
    
    print("Evaluating the energy...")
    timer_start = time.time()
    # plot the energy along the trajectory
    kinetic_energy_fn_vmapped = vmap(
        partial(auxiliary_fns["kinetic_energy_fn"], params)
    )
    potential_energy_fn_vmapped = vmap(
        partial(auxiliary_fns["potential_energy_fn"], params)
    )
    U_ts = potential_energy_fn_vmapped(q_ts)
    T_ts = kinetic_energy_fn_vmapped(q_ts, q_d_ts)
    timer_end = time.time()
    print(f"Evaluating the energy took {timer_end - timer_start:.2f} seconds.")
    
    # plot the energy vs time
    plt.figure()
    plt.title("Energy vs Time")
    plt.plot(video_ts, U_ts, label="Potential energy")
    plt.plot(video_ts, T_ts, label="Kinetic energy")
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [J]")
    plt.legend()
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.savefig(
        figures_path_parent / f"energy_vs_time_{extension}.png", bbox_inches="tight", dpi=300
    )
    print("Figures saved at", figures_path_parent / f"energy_vs_time_{extension}.png \n")
    plt.show()

    print("Drawing the robot...")
    timer_start = time.time()
    # create video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
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
    timer_end = time.time()
    print(f"Drawing the robot took {timer_end - timer_start:.2f} seconds.")
    print(f"Video saved at {video_path}. \n")
    
    
