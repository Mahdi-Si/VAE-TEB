r"""S1-T07: the raw gap/sentinel constants are importable and in-range.

Live prod confirmation is deferred; this test only pins the value-range contract the masking code
(Sprint 3) relies on. The inspection tool ``tools/inspect_raw_gaps.py`` refines these on a real fold.
"""
from __future__ import annotations

from model.vae_teb_prediction.model.model_raw import raw_mask_constants as rmc


def test_gap_encoding_in_allowed_set() -> None:
    assert rmc.GAP_ENCODING in {"nan", "zero", "held"}


def test_sentinel_type() -> None:
    assert rmc.SENTINEL is None or isinstance(rmc.SENTINEL, float)


def test_seeded_convention() -> None:
    # Seeded from the write pipeline (create_new_pipeline.py): gaps are 0.0 bpm, not NaN.
    assert rmc.GAP_ENCODING == "zero"
    assert rmc.SENTINEL == 0.0


def test_inspection_tool_importable() -> None:
    # Importing must not require h5py (it is imported lazily inside inspect()).
    import model.vae_teb_prediction.model.model_raw.tools.inspect_raw_gaps as inspect_raw_gaps

    assert hasattr(inspect_raw_gaps, "inspect")
    assert hasattr(inspect_raw_gaps, "main")
    # Pure-logic helper works without any HDF5/h5py.
    assert inspect_raw_gaps._infer_encoding({"present": True, "frac_nan": 0.0, "frac_inf": 0.0,
                                             "frac_zero": 0.01, "longest_flat_run": 3}) == "zero"


def test_infer_encoding_is_magnitude_based() -> None:
    import model.vae_teb_prediction.model.model_raw.tools.inspect_raw_gaps as inspect_raw_gaps

    # A mostly-zero fold with a few stray non-finite samples must infer 'zero', not 'nan'
    # (mere presence of a NaN must not flip the masking convention).
    dominant_zero = {"present": True, "frac_nan": 1e-5, "frac_inf": 0.0,
                     "frac_zero": 0.02, "longest_flat_run": 5}
    assert inspect_raw_gaps._infer_encoding(dominant_zero) == "zero"
    # A genuinely NaN-dominant fold still infers 'nan'.
    dominant_nan = {"present": True, "frac_nan": 0.03, "frac_inf": 0.0,
                    "frac_zero": 1e-5, "longest_flat_run": 0}
    assert inspect_raw_gaps._infer_encoding(dominant_nan) == "nan"
