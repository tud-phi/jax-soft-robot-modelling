import jax

from jsrm.systems.planar_pcs import PlanarPCSNum
import jax.numpy as jnp

from typing import Callable, Dict
from jax import Array

import numpy as onp

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

from diffrax import Tsit5

from functools import partial

jax.config.update("jax_enable_x64", True)  # double precision
jnp.set_printoptions(
    threshold   =jnp.inf, 
    linewidth   =jnp.inf,
    formatter   ={'float_kind': lambda x: '0' if x==0 else f'{x:.2e}'}
)

def draw_robot_curve_class(
    batched_forward_kinematics_fn: Callable,
    params: Dict[str, Array],
    q: Array,
    width: int,
    height: int,
    num_points: int = 50,
):
    h, w = height, width
    ppm = h / (2.0 * jnp.sum(params["l"]))
    s_ps = jnp.linspace(0, jnp.sum(params["l"]), num_points)
    chi_ps = batched_forward_kinematics_fn(q, s_ps)
    
    # Position du robot dans les coordonnées pixel
    curve_origin = onp.array([w // 2, 0.1 * h])
    curve = onp.array((curve_origin[:, None] + chi_ps[1:, :] * ppm), dtype=onp.float32).T
    curve[:, 1] = h - curve[:, 1]
    
    return curve  # (N, 2)

def plot_robot_matplotlib(
    batched_forward_kinematics_fn: Callable,
    params: Dict[str, Array],
    q: Array,
    width: int = 500,
    height: int = 500,
    num_points: int = 50,
    show: bool = False,
):
    fig, ax = plt.subplots()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.invert_yaxis()
    line, = ax.plot([], [], lw=4, color="blue")
    curve = draw_robot_curve_class(batched_forward_kinematics_fn, params, q, width, height, num_points)
    line.set_data(curve[:, 0], curve[:, 1])

    if show:
        plt.show(fig)
    
    return fig

def animate_robot_matplotlib(
    batched_forward_kinematics_fn: Callable,
    params: Dict[str, Array],
    t_list: Array,  # shape (T,)
    q_list: Array,  # shape (T, DOF)
    width: int = 500,
    height: int = 500,
    num_points: int = 50,
    interval: int = 50,
    boolshow: bool = True,
):
    fig, ax = plt.subplots()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.invert_yaxis()
    line, = ax.plot([], [], lw=4, color="blue")
    title_text = ax.set_title("t = 0.00 s")

    def init():
        line.set_data([], [])
        title_text.set_text("t = 0.00 s")
        return line, title_text

    def update(frame_idx):
        q = q_list[frame_idx]
        t = t_list[frame_idx]
        curve = draw_robot_curve_class(batched_forward_kinematics_fn, params, q, width, height, num_points)
        line.set_data(curve[:, 0], curve[:, 1])
        title_text.set_text(f"t = {t:.2f} s")
        return line, title_text

    ani = FuncAnimation(
        fig, 
        update, 
        frames=len(q_list), 
        init_func=init, 
        blit=False,
        interval=interval)
    
    if boolshow:
        plt.show()
    plt.close(fig)
    return HTML(ani.to_jshtml())

if __name__ == "__main__":
    num_segments = 2
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

    # ======================================================
    # Robot initialization
    # ======================================================
    robot = PlanarPCSNum(
        num_segments=num_segments,
        params=params,
        order_gauss=5,
    )
    
    # =====================================================
    # Simulation upon time
    # =====================================================
    # Initial configuration
    q0 = jnp.repeat(jnp.array([5.0 * jnp.pi, 0.1, 0.2])[None, :], num_segments, axis=0).flatten()
    # Initial velocities
    qd0 = jnp.zeros_like(q0)
    
    # Actuation parameters
    tau = jnp.zeros_like(q0)
    # WARNING: actuation_args need to be a tuple, even if it contains only one element
    actuation_args = (tau,)
    
    # Simulation time parameters
    t0 = 0.0
    t1 = 2.0
    dt = 1e-4
    skip_step = 100  # how many time steps to skip in between video frames
    
    # Solver
    solver = Tsit5() # Runge-Kutta 5(4) method
    
    ts, q_ts, q_d_ts = robot.resolve_upon_time(
        q0=q0,
        qd0=qd0,
        actuation_args=actuation_args,
        t0=t0,
        t1=t1,
        dt=dt,
        skip_steps=skip_step,
        max_steps=None
    )
    
    # =====================================================
    # End-effector position upon time
    # =====================================================
    forward_kinematics_end_effector = jax.jit(partial(
        robot.forward_kinematics_fn,
        s=jnp.sum(robot.l)  # end-effector position
        ))
    chi_ee_ts = jax.vmap(forward_kinematics_end_effector)(q_ts)
    
    plt.figure()
    plt.plot(ts, chi_ee_ts[:, 1], label="End-effector x [m]")
    plt.plot(ts, chi_ee_ts[:, 2], label="End-effector y [m]")
    plt.xlabel("Time [s]")
    plt.ylabel("End-effector position [m]")
    plt.legend()
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.scatter(chi_ee_ts[:, 1], chi_ee_ts[:, 2], c=ts, cmap="viridis")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("End-effector x [m]")
    plt.ylabel("End-effector y [m]")
    plt.colorbar(label="Time [s]")
    plt.tight_layout()
    plt.show()
    
    # =====================================================
    # Energy computation upon time
    # =====================================================
    U_ts = jax.vmap(jax.jit(partial(robot.potential_energy)))(q_ts)
    T_ts = jax.vmap(jax.jit(partial(robot.kinetic_energy)))(q_ts, q_d_ts)
    
    plt.figure()
    plt.plot(ts, U_ts, label="Potential Energy")
    plt.plot(ts, T_ts, label="Kinetic Energy")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (J)")
    plt.legend()
    plt.title("Energy over Time")
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.show()
    
    # =====================================================
    # Plot the robot configuration upon time
    # =====================================================
    animate_robot_matplotlib(
        batched_forward_kinematics_fn=jax.vmap(robot.forward_kinematics_fn, in_axes=(None, 0), out_axes=-1),
        params=params,
        t_list=ts,  # shape (T,)
        q_list=q_ts,  # shape (T, DOF)
        width=700,
        height=700,
        num_points=50,
        interval=100, #ms
    )