# Installation

## Requirements

- Python >= 3.10
- JAX
- NumPy

## Install from PyPI

The easiest way to install JSRM is from PyPI:

```bash
pip install jsrm
```

## Install from Source

For development or to get the latest features:

```bash
git clone https://github.com/tud-cor-sr/jax-soft-robot-modelling.git
cd jax-soft-robot-modelling
pip install -e .
```

## Optional Dependencies

### Examples Dependencies

To run the examples, install the additional dependencies:

```bash
pip install -e ".[examples]"
```

This includes:
- `diffrax` - For differential equation solving
- `jaxopt` - For optimization
- `matplotlib` - For plotting
- `opencv-python` - For computer vision
- `scipy` - For scientific computing

### Development Dependencies

For development work:

```bash
pip install -e ".[dev]"
```

This includes testing, linting, and formatting tools.

### Documentation Dependencies

To build the documentation locally:

```bash
pip install -e ".[docs]"
```

## Environment Setup

After installation, always source the environment variables when opening a new terminal:

```bash
source 01-configure-env-vars.sh
```

## Verification

To verify your installation, try running a simple example:

```python
import jax.numpy as jnp
from jsrm.systems import planar_pcs_num

# Create a simple 1-segment PCS robot
num_segments = 1
strain_selector = jnp.ones((3 * num_segments,), dtype=bool)

# Initialize the system
_, forward_kinematics, _, _ = planar_pcs_num.factory(
    num_segments, 
    strain_selector
)

print("JSRM installation successful!")
```
