from pathlib import Path

from jsrm.symbolic_derivation.pendulum import symbolically_derive_pendulum_model

NUM_LINKS = 2

num_links_to_filename_map = {
    1: "single_pendulum.dill",
    2: "double_pendulum.dill",
    3: "triple_pendulum.dill",
}

if __name__ == "__main__":
    sym_exp_filepath = (
            Path(__file__).parent.parent / "symbolic_expressions" / num_links_to_filename_map[NUM_LINKS]
    )
    symbolically_derive_pendulum_model(num_links=NUM_LINKS, filepath=sym_exp_filepath)
