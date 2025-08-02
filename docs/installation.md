# 📦 Installation

<div class="doc-summary">
  <strong>Get started with JSRM in minutes!</strong> Choose from multiple installation methods to get JAX Soft Robot Modelling running on your system.
</div>

---

## 🔧 Requirements

!!! note "System Requirements"
    - **Python** >= 3.10
    - **JAX** >= 0.4.0
    - **NumPy** >= 1.21.0

---

## 🚀 Quick Install

=== "🐍 PyPI (Recommended)"

    The easiest way to install JSRM is from PyPI:

    ```bash
    pip install jsrm
    ```

    !!! success "Ready to go!"
        This installs the core JSRM package with all essential dependencies.

=== "🔨 From Source"

    For development or to get the latest features:

    ```bash
    git clone https://github.com/tud-phi/jax-soft-robot-modelling.git
    cd jax-soft-robot-modelling
    pip install -e .
    ```

    !!! tip "Development Mode"
        The `-e` flag installs in "editable" mode, so changes to the source code are immediately available.

=== "🐋 Docker"

    Use our pre-built Docker container:

    ```bash
    docker pull ghcr.io/tud-phi/jax-soft-robot-modelling:latest
    docker run -it --rm ghcr.io/tud-phi/jax-soft-robot-modelling:latest
    ```

---

## 🎯 Installation Options

### 📚 Examples Dependencies

To run all examples and tutorials:

```bash
pip install jsrm[examples]
```

**Includes:**

- `diffrax` - Advanced differential equation solving
- `jaxopt` - High-performance optimization algorithms  
- `matplotlib` - Publication-ready plotting and visualization
- `opencv-python` - Computer vision and image processing
- `scipy` - Scientific computing utilities

### 🛠️ Development Dependencies

For contributing to JSRM:

```bash
pip install jsrm[dev]
```

**Includes:**

- `pytest` - Testing framework
- `black` - Code formatting
- `flake8` - Code linting
- `mypy` - Type checking
- `pre-commit` - Git hooks

### 📖 Documentation Dependencies

To build documentation locally:

```bash
pip install jsrm[docs]
```

**Includes:**

- `mkdocs-material` - Modern documentation theme
- `mkdocstrings` - API documentation generation
- `mkdocs-jupyter` - Jupyter notebook integration

### 🎉 Complete Installation

Get everything at once:

```bash
pip install jsrm[all]
```

---

## ⚙️ Environment Setup

!!! warning "Important Step"
    After installation, always source the environment variables when opening a new terminal:

    ```bash
    source 01-configure-env-vars.sh
    ```

    This ensures JSRM can find all necessary configuration files and paths.

---

## ✅ Verification

Test your installation with this quick verification script:

=== "🧪 Basic Test"

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

    print("🎉 JSRM installation successful!")
    ```

=== "🚀 Advanced Test"

    ```python
    import jax
    import jax.numpy as jnp
    from jsrm.systems.planar_pcs import PlanarPCS
    from jsrm.parameters import Params

    # Test JAX compilation and differentiation
    @jax.jit
    def test_function(q):
        robot = PlanarPCS(num_segments=2, params=Params.default())
        return robot.forward_kinematics(q, s=1.0)

    # Test with sample configuration
    q = jnp.array([0.1, 0.0, 0.0, 0.1, 0.0, 0.0])
    result = test_function(q)
    
    # Test differentiation
    grad_fn = jax.grad(lambda q: jnp.sum(test_function(q)**2))
    gradient = grad_fn(q)
    
    print(f"✅ Forward kinematics: {result}")
    print(f"✅ Gradient computation: {gradient}")
    print("🎉 Advanced features working perfectly!")
    ```

---

## 🔧 Troubleshooting

!!! question "Common Issues"

    === "🐍 Python Version"
        
        **Problem:** `ImportError` or compatibility issues
        
        **Solution:** Ensure you're using Python 3.10 or later:
        ```bash
        python --version  # Should be >= 3.10
        ```

    === "📦 JAX Installation"
        
        **Problem:** JAX import errors or GPU issues
        
        **Solution:** Install JAX with proper hardware support:
        ```bash
        # For CPU only
        pip install jax[cpu]
        
        # For NVIDIA GPU
        pip install jax[cuda12_pip]
        
        # For Apple Silicon
        pip install jax[metal]
        ```

    === "🔧 Environment Variables"
        
        **Problem:** Module not found errors
        
        **Solution:** Ensure environment is properly configured:
        ```bash
        # Check if variables are set
        echo $PYTHONPATH
        
        # Re-source the configuration
        source 01-configure-env-vars.sh
        ```

---

## 🆘 Getting Help

!!! info "Need assistance?"

    - 📚 **Documentation**: Check our [API Reference](api/systems.md)
    - 💬 **Discussions**: Join our [GitHub Discussions](https://github.com/tud-phi/jax-soft-robot-modelling/discussions)  
    - 🐛 **Issues**: Report bugs on [GitHub Issues](https://github.com/tud-phi/jax-soft-robot-modelling/issues)
    - 📧 **Email**: Contact us at `m.stolzle@tudelft.nl`
