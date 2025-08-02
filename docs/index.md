# 🤖 JAX Soft Robot Modelling

<div class="doc-summary">
  <strong>Welcome to JAX Soft Robot Modelling (JSRM)!</strong> A cutting-edge library for fast, parallelizable, and differentiable simulations of soft robots using JAX and symbolic mathematics.
</div>

---

## 🎯 Overview

This repository contains symbolic derivations of the kinematics and dynamics of various soft robots using **SymPy**.
The symbolic expressions are then implemented in **JAX** and can be used for fast, parallelizable, and differentiable simulations.

<div class="feature-grid">
  <div class="feature-card">
    <h3><span class="icon">⚡</span> JAX-Powered</h3>
    <p>Leverage JAX for lightning-fast computations with automatic differentiation and JIT compilation</p>
  </div>
  
  <div class="feature-card">
    <h3><span class="icon">🧮</span> Symbolic Foundation</h3>
    <p>Mathematically rigorous models derived from first principles using symbolic computation</p>
  </div>
  
  <div class="feature-card">
    <h3><span class="icon">🔧</span> Modular Design</h3>
    <p>Extensible architecture that makes it easy to add new robot types and configurations</p>
  </div>
  
  <div class="feature-card">
    <h3><span class="icon">📊</span> Differentiable</h3>
    <p>Full gradient support for optimization, control, and machine learning applications</p>
  </div>
  
  <div class="feature-card">
    <h3><span class="icon">🚀</span> High Performance</h3>
    <p>Parallelizable computations that scale efficiently across multiple devices</p>
  </div>
  
  <div class="feature-card">
    <h3><span class="icon">🔬</span> Research-Ready</h3>
    <p>Validated implementations used in peer-reviewed robotics research</p>
  </div>
</div>

---

## 🤖 Supported Robot Types

We focus on planar settings and have implemented the following soft robot architectures:

!!! note "🦾 **N-link Pendulum**"
    Classical articulated robot perfect for benchmarking and comparison studies

!!! tip "🌊 **Planar Piecewise Constant Strain (PCS)**"
    Advanced continuum soft robot with constant strain segments for precise modeling

!!! info "🔗 **Planar Handed Shearing Auxetics (HSA)**"
    Novel soft robot with auxetic properties for unique deformation characteristics

---

## ✨ Key Features

=== "🚀 Performance"

    - **JAX Backend**: Ultra-fast computations with automatic vectorization
    - **JIT Compilation**: Optimized machine code generation for maximum speed  
    - **GPU/TPU Support**: Seamless acceleration on modern hardware
    - **Parallel Processing**: Scale across multiple devices effortlessly

=== "🧠 Intelligence"

    - **Automatic Differentiation**: Full gradient support for all operations
    - **Symbolic Derivation**: Mathematically rigorous kinematic and dynamic models
    - **Optimization Ready**: Perfect for control and learning applications
    - **Numerical Stability**: Robust implementations that handle edge cases

=== "🔧 Usability"

    - **Clean API**: Intuitive interfaces that are easy to learn and use
    - **Comprehensive Documentation**: Detailed guides, examples, and API reference
    - **Extensible Design**: Add new robot types and configurations with ease
    - **Research Proven**: Validated in real-world robotics applications

---

## 🚀 Quick Start

Get up and running in minutes:

=== "Installation"

    ```bash
    pip install jsrm
    ```

=== "Basic Usage"

    ```python
    import jax.numpy as jnp
    from jsrm.systems import PlanarPCS
    
    # Create a planar PCS robot
    robot = PlanarPCS(num_segments=3, params=params)
    
    # Compute forward kinematics
    q = jnp.array([0.1, 0.2, 0.3])  # Configuration
    chi = robot.forward_kinematics(q, s=1.0)  # End-effector pose
    ```

=== "Advanced Features"

    ```python
    # Compute Jacobians for control
    J = robot.jacobian(q, s=1.0)
    
    # Dynamic simulation
    B, C, G, K, D, A = robot.dynamical_matrices(q, qd)
    
    # Differentiable operations
    loss_fn = lambda q: jnp.sum(robot.forward_kinematics(q, s=1.0)**2)
    grad_fn = jax.grad(loss_fn)
    ```

---

## 📚 Quick Links

<div class="feature-grid">
  <div class="feature-card">
    <h3><span class="icon">📦</span> [Installation Guide](installation.md)</h3>
    <p>Get JSRM installed and configured on your system</p>
  </div>
  
  <div class="feature-card">
    <h3><span class="icon">🚀</span> [Quick Start](user-guide/quick-start.md)</h3>
    <p>Jump right in with hands-on tutorials and examples</p>
  </div>
  
  <div class="feature-card">
    <h3><span class="icon">📖</span> [Examples](user-guide/examples.md)</h3>
    <p>Explore comprehensive examples and use cases</p>
  </div>
  
  <div class="feature-card">
    <h3><span class="icon">📋</span> [API Reference](api/systems.md)</h3>
    <p>Complete documentation of all classes and functions</p>
  </div>
</div>

---

## 📄 Citation

This simulator is part of the publication **"An Experimental Study of Model-based Control for Planar Handed Shearing Auxetics Robots"** presented at the *18th International Symposium on Experimental Robotics*.

!!! quote "If you use our software in your research, please cite:"

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

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](development/contributing.md) for details on how to get started.

## License

This project is licensed under the MIT License - see the [LICENSE.txt](https://github.com/tud-phi/jax-soft-robot-modelling/blob/main/LICENSE.txt) file for details.
