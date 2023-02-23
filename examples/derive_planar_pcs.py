from pathlib import Path

from jsrm.symbolic_derivation.planar_pcs import symbolically_derive_planar_pcs_model

NUM_SEGMENTS = 2

num_segments_to_filename_map = {
    1: "planar_pcs_one_segment.dill",
    2: "planar_pcs_two_segments.dill",
    3: "planar_pcs_three_segments.dill",
}

if __name__ == "__main__":
    sym_exp_filepath = (
        Path(__file__).parent.parent
        / "symbolic_expressions"
        / num_segments_to_filename_map[NUM_SEGMENTS]
    )
    symbolically_derive_planar_pcs_model(
        num_segments=NUM_SEGMENTS, filepath=sym_exp_filepath
    )
