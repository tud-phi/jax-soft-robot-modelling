from pathlib import Path

from jax_soft_robot_modelling.symbolic_derivation.pendulum import symbolically_derive_pendulum_model

sym_exp_filepath = Path(__file__).parent.parent / "symbolic_expressions" / "double_pendulum.dill"

if __name__ == "__main__":
    symbolically_derive_pendulum_model(num_links=2, filepath=sym_exp_filepath)
