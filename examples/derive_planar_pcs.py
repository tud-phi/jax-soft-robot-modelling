from pathlib import Path

from jsrm.symbolic_derivation.planar_pcs import symbolically_derive_planar_pcs_model

sym_exp_filepath = Path(__file__).parent.parent / "symbolic_expressions" / "planar_pcs_two_segments.dill"

if __name__ == "__main__":
    symbolically_derive_planar_pcs_model(num_segments=2, filepath=sym_exp_filepath)
