# Planar PCS Symbolic Systems

The symbolic implementation of Planar PCS systems provides pre-computed symbolic expressions for kinematics and dynamics, offering improved computational performance.

## Overview

This module contains symbolic derivations of the PCS model equations, which are pre-computed using SymPy and then implemented in JAX for fast execution. This approach eliminates the need for numerical differentiation during runtime.

## API Reference

::: jsrm.systems.planar_pcs_sym
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source
