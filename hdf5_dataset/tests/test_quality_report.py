r"""The signal-quality report's ``usable_ua_steps``: the conjunction the model's source mask takes.

A step $t$ counts when it is at or after the largest warm-up of the UA channels, and it is usable
when $\rho_{c,t} \le \varepsilon$ for every UA channel $c$ of ``up_st`` and ``up_ph``. The column is
the reference for the model's ``source_mask_fraction``, so it must take the conjunction across
both blocks, compare with $\le$ (a coefficient at $\varepsilon$ is usable, as in the loader), and
cut at the largest warm-up, not at each channel's own.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import h5py
import numpy as np
import pytest

from hdf5_dataset.new_pipeline import quality_report
from hdf5_dataset.tests.test_quality_loader import LEN_SEQUENCE, WIDTHS, _write_shard

#: $\varepsilon = 0.1$ in stored units at ``quality_rho_scale = 255``: codes $\le 25$ are usable.
THRESHOLD = 0.1 * 255
OK, BAD = 25, 26


def _write_rho(path: Path, rho: Dict[str, np.ndarray], warmup: Dict[str, Sequence[int]]) -> Path:
    """Write the UA $\\rho$ blocks and the warm-up attributes ``_usable_steps`` reads.

    Args:
        path: Where to write.
        rho: ``{block: (n, C, T) uint8}`` for ``up_st`` and ``up_ph``.
        warmup: ``{block: per-channel warm-up steps}``.

    Returns:
        ``path``.
    """
    with h5py.File(path, "w") as f:
        for block, values in rho.items():
            f[f"{block}_rho"] = values.astype(np.uint8)
            f[block] = np.zeros(values.shape, dtype=np.float32)
            f[block].attrs["causal_warmup_steps"] = np.asarray(warmup[block], dtype=np.int64)
    return path


@pytest.fixture()
def rho_file(tmp_path: Path) -> Path:
    r"""Two segments, $T = 6$, warm-ups $(0, 2)$ in ``up_st`` and $(1)$ in ``up_ph``: the cut is $2$.

    Segment 0 after the cut: step 2 usable, step 3 bad in one ``up_st`` channel, step 4 bad in
    ``up_ph``, step 5 usable. Segment 1 is usable after the cut and bad before it, so a cut at a
    smaller warm-up would count its bad steps.
    """
    up_st = np.full((2, 2, 6), OK)
    up_ph = np.full((2, 1, 6), OK)
    up_st[0, 1, 3] = BAD
    up_ph[0, 0, 4] = BAD
    up_st[1, 0, :2] = BAD
    up_ph[1, 0, :2] = BAD
    return _write_rho(tmp_path / "rho.hdf5", {"up_st": up_st, "up_ph": up_ph},
                      {"up_st": (0, 2), "up_ph": (1,)})


def test_usable_steps_is_the_conjunction_after_the_largest_warmup(rho_file: Path) -> None:
    """Segment 0 keeps 2 of its 4 steps, segment 1 all 4: $6 / 8$."""
    with h5py.File(rho_file, "r") as f:
        share = quality_report._usable_steps(f, np.arange(2), ("up_st", "up_ph"), THRESHOLD)
    assert share == pytest.approx(0.75)


def test_usable_steps_reads_only_the_sampled_rows(rho_file: Path) -> None:
    with h5py.File(rho_file, "r") as f:
        assert quality_report._usable_steps(f, np.array([1]), ("up_st", "up_ph"), THRESHOLD) == 1.0
        assert quality_report._usable_steps(f, np.array([0]), ("up_st", "up_ph"), THRESHOLD) == 0.5


def test_a_coefficient_at_epsilon_is_usable(tmp_path: Path) -> None:
    """Code $25 \\le 25.5$ is usable and code $26$ is not, as in the loader's $\\rho > \\varepsilon$."""
    rho = {"up_st": np.full((1, 1, 4), OK), "up_ph": np.full((1, 1, 4), OK)}
    rho["up_ph"][0, 0, 3] = BAD
    path = _write_rho(tmp_path / "edge.hdf5", rho, {"up_st": (0,), "up_ph": (0,)})
    with h5py.File(path, "r") as f:
        assert quality_report._usable_steps(f, np.arange(1), ("up_st", "up_ph"), THRESHOLD) == 0.75


def test_a_warmup_past_the_segment_gives_nan(tmp_path: Path) -> None:
    rho = {"up_st": np.full((1, 1, 4), OK), "up_ph": np.full((1, 1, 4), OK)}
    path = _write_rho(tmp_path / "warm.hdf5", rho, {"up_st": (0,), "up_ph": (4,)})
    with h5py.File(path, "r") as f:
        assert np.isnan(quality_report._usable_steps(f, np.arange(1), ("up_st", "up_ph"), THRESHOLD))


def test_summarise_file_reports_usable_ua_steps(tmp_path: Path) -> None:
    """On the loader's synthetic shard, with its UA $\\rho$ rewritten to a known pattern.

    Every UA code is usable except one ``up_st`` step in segment 0 and ten ``up_ph`` steps in
    segment 3, so $11$ of the $4 T$ steps are bad. The shard has no warm-up, and its
    ``fhr_raw_match`` is NaN in the last segment, which the summary skips.
    """
    shard = _write_shard(tmp_path / "quality.hdf5")
    n = 4
    with h5py.File(shard, "r+") as f:
        up_st = np.full((n, WIDTHS["up_st"], LEN_SEQUENCE), OK, dtype=np.uint8)
        up_ph = np.full((n, WIDTHS["up_ph"], LEN_SEQUENCE), OK, dtype=np.uint8)
        up_st[0, 2, 10] = BAD
        up_ph[3, 0, 20:30] = BAD
        f["up_st_rho"][:] = up_st
        f["up_ph_rho"][:] = up_ph
    summary, _ = quality_report.summarise_file(str(shard), "quality", 0.1, np.random.default_rng(0))

    assert summary["usable_ua_steps"] == pytest.approx(1.0 - 11 / (n * LEN_SEQUENCE))
    assert summary["fhr_raw_match"] == pytest.approx(0.875)
    assert summary["quality_version"] == 4
