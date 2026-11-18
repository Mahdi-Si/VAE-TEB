"""The two numeric rules of ``minimal_to_split_hdf5.py``: the gap measure and the float16 rounding."""
import numpy as np

from hdf5_dataset.minimal_to_split_hdf5 import gap_fraction, round_mantissa


def test_gap_fraction_counts_only_long_straight_runs():
    rng = np.random.default_rng(0)
    x = 140 + np.cumsum(rng.standard_normal((3, 5280)), axis=1)
    x[1, 1000:1100] = x[1, 999]                       # a 25 s hold: a filled gap
    x[2, 2000:2030] = np.linspace(x[2, 1999], 150, 30)  # a 7.5 s line: too short to count
    frac = gap_fraction(x)
    assert frac[0] == 0.0 and frac[2] == 0.0
    assert 0.018 < frac[1] < 0.020                    # about 100 of 5278 second differences


def test_round_mantissa_bounds_error_and_keeps_specials():
    x = np.array([1.0, -1.0, 65504.0, -65504.0, np.inf, np.nan, 0.0, 3.14159, 1e-3], np.float16)
    assert np.array_equal(round_mantissa(x, 10), x, equal_nan=True)
    y = round_mantissa(x, 6)
    assert np.isfinite(y[:4]).all() and np.isinf(y[4]) and np.isnan(y[5]) and y[6] == 0
    finite = [0, 1, 7, 8]
    err = np.abs(y[finite].astype(float) - x[finite].astype(float))
    assert (err <= np.abs(x[finite].astype(float)) * 2.0 ** -6).all()


def test_stored_fhr_repaired_follows_the_segments_the_build_asks_for(tmp_path):
    import h5py
    from hdf5_dataset import minimal_to_full_hdf5 as m2f

    path = str(tmp_path / "m.hdf5")
    mask = np.zeros((4, 8), np.uint8)
    mask[np.arange(4), np.arange(4)] = 1          # row r is marked at sample r
    with h5py.File(path, "w") as f:
        f["epoch"] = np.array([-900.0, -600.0, -300.0, -100.0], np.float32)
        f["fhr_repaired"] = mask
        f["fhr_repair_exact"] = np.array([1, 1, 1, 0], np.uint8)
        f["fhr_raw_match"] = np.full(4, 0.9, np.float32)
    m2f.MinimalAdaptor.index = {"g1": (path, np.arange(4))}
    got, info = m2f.stored_fhr_repaired("/x/g1.mat", np.array([-600.0, -300.0]), None, None)
    assert got.dtype == bool and np.flatnonzero(got[0]) == [1] and np.flatnonzero(got[1]) == [2]
    assert info["raw_match"] == np.float32(0.9)
    # A record without an exact mask falls back to the rules, as a build from the .mat does.
    assert m2f.stored_fhr_repaired("/x/g1.mat", np.array([-100.0]), None, None)[0] is None
