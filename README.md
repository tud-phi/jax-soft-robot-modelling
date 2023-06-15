# JAX Soft Robot Modelling

## Installation

The plugin can be installed from PyPI:

```bash
pip install jax-soft-robot-modelling
```

or locally from the source code:

```bash
pip install .
```

If you want to run the examples, you will also need to install the following dependencies:

```bash
pip install ".[examples]"
```

## Usage
Always, first source all necessary environment variables when opening a new terminal:

```bash
source 01-configure-env-vars.sh
```

Then, we can symbolically derive the pendulum kinematics and dynamics

```bash
    python examples/derive_pendulum.py
```

Finally, we can simulate the pendulum
```bash
    python examples/simulate_pendulum.py
```
