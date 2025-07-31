# JAX Soft Robot Modelling

Welcome to the documentation for JAX Soft Robot Modelling (JSRM)!

## Overview

This repository contains symbolic derivations of the kinematics and dynamics of various soft robots using Sympy.
The symbolic expressions are then implemented in JAX and can be used for fast, parallelizable, and differentiable simulations.

## Supported Robot Types

So far, we have focused on planar settings and implemented the following soft robots:

- **N-link pendulum** - Classical articulated robot for benchmarking
- **Planar Piecewise Constant Strain (PCS) continuum soft robot** - Continuum robot with constant strain segments
- **Planar Handed Shearing Auxetics (HSA) robot** - Soft robot with auxetic properties

## Key Features

- **JAX-based**: Fast, parallelizable, and differentiable computations
- **Symbolic derivations**: Mathematically rigorous kinematic and dynamic models
- **Multiple robot types**: Support for various soft robot architectures
- **Extensible**: Easy to add new robot types and configurations

## Quick Links

- [Installation Guide](installation.md)
- [Quick Start](user-guide/quick-start.md)
- [Examples](user-guide/examples.md)
- [API Reference](api/systems.md)

## Citation

This simulator is part of the publication **An Experimental Study of Model-based Control
for Planar Handed Shearing Auxetics Robots** presented at the _18th International Symposium on Experimental Robotics_. 

If you use our software in your research, please cite:

```bibtex
@inproceedings{stolzle2023experimental,
  title={An experimental study of model-based control for planar handed shearing auxetics robots},
  author={St{\"o}lzle, Maximilian and Rus, Daniela and Della Santina, Cosimo},
  booktitle={International Symposium on Experimental Robotics},
  pages={153--167},
  year={2023},
  organization={Springer}
}
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](development/contributing.md) for details on how to get started.

## License

This project is licensed under the MIT License - see the [LICENSE.txt](https://github.com/tud-cor-sr/jax-soft-robot-modelling/blob/main/LICENSE.txt) file for details.
