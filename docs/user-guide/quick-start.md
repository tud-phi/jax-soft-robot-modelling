# Quick Start

This guide will help you get started with JSRM by walking through a simple example.

## Basic Example: N-link Pendulum

Let's start with the simplest example - simulating a pendulum:

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jsrm.systems import pendulum

# Define parameters for a double pendulum
num_links = 2
params = {
    "l": jnp.array([0.5, 0.3]),  # Link lengths
    "lc": jnp.array([0.25, 0.15]),  # Center of mass positions
    "m": jnp.array([1.0, 0.5]),  # Masses
    "I": jnp.array([0.1, 0.05]),  # Moments of inertia
    "g": 9.81,  # Gravity
}

# Initialize the system
ode_fn, forward_kinematics, jacobian_end_effector, _ = pendulum.factory(num_links)

# Initial conditions
q0 = jnp.array([jnp.pi/4, jnp.pi/6])  # Initial angles
q_dot0 = jnp.array([0.0, 0.0])  # Initial velocities
state0 = jnp.concatenate([q0, q_dot0])

# Simulate
import diffrax

# Time parameters
t_span = (0.0, 5.0)
dt = 0.01
saveat = diffrax.SaveAt(ts=jnp.arange(0, 5.0, dt))

# Solve ODE
solution = diffrax.diffeqsolve(
    diffrax.ODETerm(lambda t, state, args: ode_fn(params, state)),
    diffrax.Dopri5(),
    t0=t_span[0],
    t1=t_span[1],
    dt0=dt,
    y0=state0,
    saveat=saveat,
)

# Extract results
t = solution.ts
states = solution.ys
q = states[:, :num_links]
q_dot = states[:, num_links:]

# Compute end-effector trajectory
end_effector_positions = jnp.array([
    forward_kinematics(params, q_i) for q_i in q
])

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

# Joint angles
ax1.plot(t, q[:, 0], label='Joint 1')
ax1.plot(t, q[:, 1], label='Joint 2')
ax1.set_ylabel('Joint Angles [rad]')
ax1.legend()
ax1.grid(True)

# Joint velocities
ax2.plot(t, q_dot[:, 0], label='Joint 1')
ax2.plot(t, q_dot[:, 1], label='Joint 2')
ax2.set_ylabel('Joint Velocities [rad/s]')
ax2.legend()
ax2.grid(True)

# End-effector trajectory
ax3.plot(end_effector_positions[:, 0], end_effector_positions[:, 1])
ax3.set_xlabel('X [m]')
ax3.set_ylabel('Y [m]')
ax3.set_title('End-Effector Trajectory')
ax3.grid(True)
ax3.axis('equal')

plt.tight_layout()
plt.show()
```

## Key Concepts

### 1. System Factory Pattern

JSRM uses a factory pattern to create system functions:

```python
ode_fn, forward_kinematics, jacobian_fn, _ = system.factory(parameters)
```

This returns:
- `ode_fn`: Function for numerical integration
- `forward_kinematics`: Function to compute positions
- `jacobian_fn`: Function to compute Jacobians
- Additional system-specific functions

### 2. Parameter Dictionary

Each system expects a parameter dictionary with physical properties:

```python
params = {
    "l": link_lengths,
    "m": masses,
    "I": inertias,
    # ... other parameters
}
```

### 3. State Representation

States are typically organized as `[positions, velocities]`:

```python
state = jnp.concatenate([q, q_dot])
```

## Next Steps

- Explore more [Examples](examples.md)
- Check out the [API Reference](../api/systems.md)
- Learn about [Contributing](../development/contributing.md)
