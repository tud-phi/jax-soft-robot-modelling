# Planar PCS Symbolic Systems

The symbolic implementation of Planar PCS systems provides pre-computed symbolic expressions for kinematics and dynamics, offering improved computational performance. This implementation is based on the discrete Cosserat approach by Renda et al. (2018).

## Overview

This module contains symbolic derivations of the PCS model equations, based on the discrete Cosserat approach for multisection soft manipulator dynamics (Renda et al., 2018), which are pre-computed using SymPy and then implemented in JAX for fast execution. This approach eliminates the need for numerical differentiation during runtime.

## API Reference

::: jsrm.systems.planar_pcs_sym
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source

## References

The PCS (Piecewise Constant Strain) model was originally proposed in:

Renda, F., Boyer, F., Dias, J., & Seneviratne, L. (2018). Discrete cosserat approach for multisection soft manipulator dynamics. *IEEE Transactions on Robotics*, 34(6), 1518-1533.
