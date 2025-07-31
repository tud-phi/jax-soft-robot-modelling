import jax
import jax.numpy as jnp
from jsrm.systems.gvs import (
    GVS,
    Joint,
    Basis
)
from jsrm.utils.gvs.custom_types import (
    LinkAttributes, 
    JointAttributes, 
    BasisAttributes,
)

from diffrax import Tsit5

from typing import List

import matplotlib.pyplot as plt

from functools import partial

jax.config.update("jax_enable_x64", True)  # double precision
jnp.set_printoptions(
    threshold=jnp.inf,
    linewidth=jnp.inf,
    formatter={"float_kind": lambda x: "0" if x == 0 else f"{x:.2e}"},
)

if __name__ == "__main__":
    # Define model inputs
    List_links  : List[LinkAttributes] = []
    List_joints : List[JointAttributes] = []
    List_basis  : List[BasisAttributes] = []
    List_nGauss : List[int] = []

    link1 = LinkAttributes(
        section='Circular',
        E=1e6,
        nu=0.5,
        rho=1000,
        eta=1e4,
        l=0.3,
        r_i=0.03,
        r_f=0.03
    )
    List_links.append(link1)
    joint1 = JointAttributes(jointtype='Fixed')
    List_joints.append(joint1)
    basis1 = BasisAttributes(
        basistype='Legendre',
        Bdof=[0, 1, 1, 0, 0, 0],
        Bodr=[0, 0, 0, 0, 0, 0],
        xi_star=[0, 0, 0, 1, 0, 0]
    )
    List_basis.append(basis1)
    List_nGauss.append(5)  # Number of Gauss points for the first link

    link2 = LinkAttributes(
        section='Circular',
        E=1e6,
        nu=0.5,
        rho=1000,
        eta=1e4,
        l=0.3,
        r_i=0.03,
        r_f=0.03
    )
    List_links.append(link2)
    joint2 = JointAttributes(jointtype='Fixed')
    # joint2 = JointAttributes(jointtype='Revolute', axis='z')
    List_joints.append(joint2)
    basis2 = BasisAttributes(
        basistype='Monomial',
        Bdof=[1, 1, 0, 0, 0, 0],
        Bodr=[0, 0, 0, 0, 0, 0],
        xi_star=[0, 0, 0, 1, 0, 0]
    )
    List_basis.append(basis2)
    List_nGauss.append(6)  # Number of Gauss points for the second link

    link3 = LinkAttributes(
        section='Elliptical',  # Section type
        E=1e7,                # Young's modulus in Pascals
        nu=0.4,               # Poisson's ratio [-1, 0.5]
        rho=1050,             # Density [kg/m^3]
        eta=1e4,              # Damping coefficient
        l=0.3,                # Length in meters
        a_i=0.04,             # Initial semi-major axis in meters
        a_f=0.04,             # Final semi-major axis in meters
        b_i=0.02,             # Initial semi-minor axis in meters
        b_f=0.02              # Final semi-minor axis in meters
    )
    List_links.append(link3)
    joint3 = JointAttributes(
        jointtype='Revolute',  # Prismatic joint
        axis='z',               # Axis of translation
    )
    List_joints.append(joint3)    
    basis3 = BasisAttributes(
        basistype='Chebychev',       # Type of basis
        Bdof=[0, 1, 0, 1, 0, 0],    # Degrees of freedom for each deformation type
        Bodr=[0, 0, 0, 0, 0, 0],    # Order of basis functions for each deformation type
        xi_star=[0, 0, 0, 1, 0, 0], # Reference strain values as vector
    )
    List_basis.append(basis3)
    List_nGauss.append(5)  # Number of Gauss points for the third link
    
    
    # Create the GVS model
    robot = GVS(
        links_list=List_links,
        joints_list=List_joints,
        basis_list=List_basis,
        n_gauss_list=List_nGauss,
        gravity_vector=[0, 0, 9.81]
    )

    # Initial configuration
    q0 = jnp.ones(robot.dof_tot_system)
    # Initial velocities
    qd0 = jnp.ones(robot.dof_tot_system)
    
    # Actuation parameters
    tau = jnp.zeros(robot.dof_tot_system)
    # WARNING: actuation_args need to be a tuple, even if it contains only one element
    # so (tau, ) is necessary NOT (tau) or tau
    actuation_args = (tau,)

    # Simulation time parameters
    t0 = 0.0
    t1 = 2.0
    dt = 1e-4
    skip_step = 100  # how many time steps to skip in between video frames

    # Solver
    solver = Tsit5()  # Runge-Kutta 5(4) method

    ts, q_ts, q_d_ts = robot.resolve_upon_time(
        q0=q0,
        qd0=qd0,
        actuation_args=actuation_args,
        t0=t0,
        t1=t1,
        dt=dt,
        skip_steps=skip_step,
        max_steps=None,
    )
    
    # =====================================================
    # End-effector position upon time
    # =====================================================
    forward_kinematics = jax.jit(
        partial(
            robot.forward_kinematics,
        )
    )
    g_ee_ts = jax.vmap(lambda q: forward_kinematics(q)[-1])(q_ts)

    plt.figure()
    plt.plot(ts, g_ee_ts[:, 0, 3], label="End-effector x [m]")
    plt.plot(ts, g_ee_ts[:, 1, 3], label="End-effector y [m]")
    plt.plot(ts, g_ee_ts[:, 2, 3], label="End-effector z [m]")
    plt.xlabel("Time [s]")
    plt.ylabel("End-effector position [m]")
    plt.legend()
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(
        g_ee_ts[:, 0, 3], g_ee_ts[:, 1, 3], g_ee_ts[:, 2, 3], c=ts, cmap="viridis"
    )
    ax.axis("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("End-effector trajectory (3D)")
    fig.colorbar(p, ax=ax, label="Time [s]")
    plt.show()

    # # =====================================================
    # # Plot the robot configuration upon time
    # # =====================================================
    # animate_robot_matplotlib(
    #     robot,
    #     t_list=ts,  # shape (T,)
    #     q_list=q_ts,  # shape (T, DOF)
    #     num_points=50,
    #     interval=100,  # ms
    #     slider=True,
    # )