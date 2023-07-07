from pathlib import Path

import jsrm
from jsrm.symbolic_derivation.pendulum import symbolically_derive_pendulum_model

NUM_LINKS = 2

if __name__ == "__main__":
    sym_exp_filepath = (
        Path(jsrm.__file__).parent
        / "symbolic_expressions"
        / f"pendulum_nl-{NUM_LINKS}.dill"
    )
    symbolically_derive_pendulum_model(num_links=NUM_LINKS, filepath=sym_exp_filepath)
