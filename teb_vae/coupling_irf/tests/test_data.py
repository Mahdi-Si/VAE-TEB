import h5py
import numpy as np

from teb_vae.coupling_irf.data import build_step_cache, guid_folds, load_steps


def test_step_cache_and_folds(tmp_path):
    n = 5280
    fhr = np.tile(np.arange(n, dtype=np.float32), (2, 1))
    fq = np.zeros((2, n), np.uint8)
    uq = np.zeros((2, n), np.uint16)
    # row 0 (exact repair: REPAIRED=2 invalid, HOLD=32 not): step 1 = raw 241..256
    fq[0, 241:245] = 2        # 4 invalid -> 12/16 valid, mean over 245..256
    fq[0, 257:273] = 32       # step 2 stays fully valid under the exact set
    # row 1 (rules: HOLD invalid): step 2 has no valid sample -> gap-filled mean, fraction 0
    fq[1, 257:273] = 32
    uq[0, 225:233] = 8 | 4096  # step 0 = raw 225..240: TRANSIENT invalid on the first 8
    uq[1, 225:241] = 4096      # BOX alone is not in the invalid set
    (tmp_path / "split" / "train").mkdir(parents=True)
    with h5py.File(tmp_path / "split" / "train" / "a.hdf5", "w") as f:
        f["fhr"], f["up"] = fhr, fhr / 100
        f["fhr_quality"], f["up_quality"] = fq, uq
        f["fhr_quality"].attrs.update(invalid_bits_exact=11, invalid_bits_rules=107)
        f["up_quality"].attrs["invalid_bits"] = 2299
        f["fhr_repair_exact"] = np.array([1, 0], np.uint8)
        f["guid"] = np.array([b"g0", b"g1"], dtype=object)
        f["epoch"] = np.array([-1200.0, -2400.0], np.float32)
        f["cs_label"], f["bg_label"] = np.array([1, 0], np.uint8), np.array([0, 1], np.uint8)
        f["second_stage_onset"] = np.array([0.0, np.nan], np.float32)

    (path,) = build_step_cache(tmp_path / "split", tmp_path / "out")
    s = load_steps(path)
    assert path.name == "steps_train.npz" and s["fhr"].shape == (2, 300)
    t = np.arange(300)
    np.testing.assert_allclose(s["fhr"][1, [0, 1, 299]], 240 + 16 * t[[0, 1, 299]] - 7.5)  # ends at 240+16t
    np.testing.assert_allclose(s["fhr"][0, [1, 2]], [250.5, 264.5])
    np.testing.assert_allclose(s["fhr_ok"][0, :3], [1, 0.75, 1])
    np.testing.assert_allclose(s["fhr"][1, 2], 264.5)
    assert s["fhr_ok"][1, 2] == 0 and s["fhr_sd"][1, 2] == 0
    np.testing.assert_allclose(s["fhr_sd"][1, 0], np.arange(16).std(ddof=1), rtol=1e-3)
    np.testing.assert_allclose(s["ua"][:, 0], [2.365, 2.325], rtol=1e-6)
    np.testing.assert_allclose(s["ua_ok"][:, 0], [0.5, 1])
    assert list(s["guid"]) == ["g0", "g1"] and list(s["row"]) == [0, 1] and list(s["subgroup"]) == ["a", "a"]
    assert list(s["cs"]) == [1, 0] and list(s["bg"]) == [0, 1]

    guids = np.array(["a", "b", "a", "c", "b"] + [f"g{i}" for i in range(50)])
    folds = guid_folds(guids)
    assert folds[0] == folds[2] and folds[1] == folds[4]
    assert set(folds) == set(range(5)) and (folds == guid_folds(guids)).all()
    assert (guid_folds(guids, seed=1) != folds).any()
