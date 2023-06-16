from pathlib import Path

from jsrm.symbolic_derivation.planar_pcs import symbolically_derive_planar_pcs_model

NUM_SEGMENTS = 1

if __name__ == "__main__":
    sym_exp_filepath = (
        Path(__file__).parent.parent
        / "symbolic_expressions"
        / f"planar_pcs_ns-{NUM_SEGMENTS}.dill"
    )
    symbolically_derive_planar_pcs_model(
        num_segments=NUM_SEGMENTS, filepath=sym_exp_filepath
    )
