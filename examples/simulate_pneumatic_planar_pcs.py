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
from jsrm.systems import pneumatic_planar_pcs

num_segments = 1

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
    "r_cham_in": 5e-3 * jnp.ones((num_segments,)),
    "r_cham_out": 2e-2 - 2e-3 * jnp.ones((num_segments,)),
    "varphi_cham": jnp.pi/2 * jnp.ones((num_segments,)),
}
params["D"] = 1e-3 * jnp.diag(
    (jnp.repeat(
        jnp.array([[1e0, 1e3, 1e3]]), num_segments, axis=0
    ) * params["l"][:, None]).flatten()
)

# activate all strains (i.e. bending, shear, and axial)
# strain_selector = jnp.ones((3 * num_segments,), dtype=bool)
strain_selector = jnp.array([True, False, True])[None, :].repeat(num_segments, axis=0).flatten()


def simulate_robot():
    # define initial configuration
    q0 = jnp.repeat(jnp.array([5.0 * jnp.pi, 0.2])[None, :], num_segments, axis=0).flatten()
    # number of generalized coordinates
    n_q = q0.shape[0]

    # set simulation parameters
    dt = 1e-3  # time step
    sim_dt = 5e-5  # simulation time step
    ts = jnp.arange(0.0, 2, dt)  # time steps

    strain_basis, forward_kinematics_fn, dynamical_matrices_fn, auxiliary_fns = (
        pneumatic_planar_pcs.factory(sym_exp_filepath, strain_selector)
    )
    # jit the functions
    dynamical_matrices_fn = jax.jit(partial(dynamical_matrices_fn))

    x0 = jnp.concatenate([q0, jnp.zeros_like(q0)])  # initial condition
    tau = jnp.zeros_like(q0)  # torques

    ode_fn = ode_factory(dynamical_matrices_fn, params, tau)
    term = ODETerm(ode_fn)

    sol = diffeqsolve(
        term,
        solver=Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=sim_dt,
        y0=x0,
        max_steps=None,
        saveat=SaveAt(ts=ts),
    )

    print("sol.ys =\n", sol.ys)
    # the evolution of the generalized coordinates
    q_ts = sol.ys[:, :n_q]
    # the evolution of the generalized velocities
    q_d_ts = sol.ys[:, n_q:]

    # evaluate the forward kinematics along the trajectory
    chi_ee_ts = vmap(forward_kinematics_fn, in_axes=(None, 0, None))(
        params, q_ts, jnp.array([jnp.sum(params["l"])])
    )
    # plot the configuration vs time
    plt.figure()
    for segment_idx in range(num_segments):
        plt.plot(
            ts, q_ts[:, 3 * segment_idx + 0],
            label=r"$\kappa_\mathrm{be," + str(segment_idx + 1) + "}$ [rad/m]"
        )
        plt.plot(
            ts, q_ts[:, 3 * segment_idx + 1],
            label=r"$\sigma_\mathrm{ax," + str(segment_idx + 1) + "}$ [-]"
        )
    plt.xlabel("Time [s]")
    plt.ylabel("Configuration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # plot end-effector position vs time
    plt.figure()
    plt.plot(ts, chi_ee_ts[:, 0], label="x")
    plt.plot(ts, chi_ee_ts[:, 1], label="y")
    plt.xlabel("Time [s]")
    plt.ylabel("End-effector Position [m]")
    plt.legend()
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.show()
    # plot the end-effector position in the x-y plane as a scatter plot with the time as the color
    plt.figure()
    plt.scatter(chi_ee_ts[:, 0], chi_ee_ts[:, 1], c=ts, cmap="viridis")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("End-effector x [m]")
    plt.ylabel("End-effector y [m]")
    plt.colorbar(label="Time [s]")
    plt.tight_layout()
    plt.show()
    # plt.figure()
    # plt.plot(chi_ee_ts[:, 0], chi_ee_ts[:, 1])
    # plt.axis("equal")
    # plt.grid(True)
    # plt.xlabel("End-effector x [m]")
    # plt.ylabel("End-effector y [m]")
    # plt.tight_layout()
    # plt.show()

    # plot the energy along the trajectory
    kinetic_energy_fn_vmapped = vmap(
        partial(auxiliary_fns["kinetic_energy_fn"], params)
    )
    potential_energy_fn_vmapped = vmap(
        partial(auxiliary_fns["potential_energy_fn"], params)
    )
    U_ts = potential_energy_fn_vmapped(q_ts)
    T_ts = kinetic_energy_fn_vmapped(q_ts, q_d_ts)
    plt.figure()
    plt.plot(ts, U_ts, label="Potential energy")
    plt.plot(ts, T_ts, label="Kinetic energy")
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [J]")
    plt.legend()
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_robot()
