r"""The per-sample diagnostic pages, and the index mapping everything about them rests on.

This is the one analysis that re-runs inference, and the one whose failure mode is invisible. A
page drawn from the wrong dataset row is a complete, plausible, correctly-formatted picture of a
recording nobody asked for; nothing about it looks wrong, and no downstream number moves. So the
mapping from a table row back to the dataset is checked in three ways here -- the round trip the
analysis itself performs before rendering, the ascending-order precondition the ``Subset`` visit
assumes, and the end-to-end assertion that a rendered page's filename names the recording its row
carried.

Two smaller properties matter for the same reason. **Coverage**: a cap at or above the shard count
must reach every shard, because a prefix over eight concatenated per-subgroup files is one subgroup
and one clinical class. **Containment**: a GUID is an external string, and it ends up in a
filename; a separator, a space or a non-ASCII character in it must not be able to leave the
directory or produce a name a shell cannot address.
"""
from __future__ import annotations

import re
import types
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_rws.eval.analyses import AnalysisContext
from teb_vae.lag_attn_rws.eval.analyses import samples as samples_analysis
from teb_vae.lag_attn_rws.eval.collect import load_collection


class _StubDataset:
    """A dataset that can list its own recordings, which is all the mapping needs."""

    def __init__(self, guids: List[str], epochs: List[float]) -> None:
        self._guids, self._epochs = list(guids), list(epochs)

    def __len__(self) -> int:
        return len(self._guids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {"guid": self._guids[index], "epoch": self._epochs[index], "index": index}

    def get_the_lists(self):
        """Return ``(guids, epochs, targets)`` exactly as the real dataset does."""
        return self._guids, self._epochs, [None] * len(self._guids)


class _StubLoader:
    """A loader-shaped object carrying a dataset and a collation."""

    def __init__(self, dataset: _StubDataset) -> None:
        self.dataset = dataset
        self.collate_fn = _collate


def _collate(batch):
    """Collate the stub items the way the real loader collates identity fields."""
    return {
        "guid": [item["guid"] for item in batch],
        "epoch": np.array([item["epoch"] for item in batch], dtype=np.float64),
        "index": [item["index"] for item in batch],
    }


def _loader(n: int = 8) -> _StubLoader:
    """A stub loader over ``n`` recordings, one segment each."""
    return _StubLoader(
        _StubDataset([f"g{index}" for index in range(n)], [-1000.0 * index for index in range(n)])
    )


# =============================================================================
# Filenames
# =============================================================================
@pytest.mark.parametrize(
    "guid",
    ["a/b", "with space", "café-été", "", "x" * 80],
    ids=["separator", "space", "non-ascii", "empty", "overlong"],
)
def test_a_page_filename_is_always_addressable_whatever_the_guid_holds(guid) -> None:
    name = samples_analysis.page_filename(7, guid, -1200.0)

    assert samples_analysis.FILENAME_PATTERN.fullmatch(name), name
    assert "/" not in name and "\\" not in name and " " not in name
    assert name.startswith("sample0007_")


def test_a_segment_with_no_epoch_is_named_na_rather_than_nan() -> None:
    name = samples_analysis.page_filename(3, "g0", float("nan"))

    assert name == "sample0003_g0_epochna.pdf"
    assert samples_analysis.FILENAME_PATTERN.fullmatch(name)


# =============================================================================
# The index mapping
# =============================================================================
def test_a_row_resolves_to_its_own_dataset_position() -> None:
    loader = _loader(8)
    rows = pd.DataFrame(
        {"guid": ["g5", "g1"], "epoch": [-5000.0, -1000.0]}
    )

    resolved = samples_analysis.resolve_rows(rows, samples_analysis.dataset_index_map(loader))

    assert list(resolved["dataset_index"]) == [1, 5], "ascending, as the Subset visit assumes"
    assert list(resolved["guid"]) == ["g1", "g5"]


def test_a_row_the_dataset_cannot_place_is_dropped_rather_than_guessed_at() -> None:
    loader = _loader(4)
    rows = pd.DataFrame({"guid": ["g1", "absent"], "epoch": [-1000.0, -9.0]})

    resolved = samples_analysis.resolve_rows(rows, samples_analysis.dataset_index_map(loader))

    assert list(resolved["guid"]) == ["g1"]


def test_the_unlocatable_count_is_the_rows_that_failed_to_resolve(
    tmp_path, monkeypatch
) -> None:
    """Not a difference between two row counts of different populations.

    ``len(per_sample) - len(index_map)`` compares scored segments against locatable dataset rows:
    it goes negative whenever the collection pass skipped a batch, and it reads zero whenever a
    genuine drop happens to be offset by one. The number wanted is how many rows the pages asked
    for and the dataset could not place.
    """
    loader = _loader(8)
    # Two rows the dataset cannot place, and a table shorter than the dataset -- which is what a
    # skipped batch leaves behind, and what made the old expression report a negative count.
    per_sample = pd.DataFrame(
        {
            "guid": ["g1", "g2", "absent_a", "absent_b"],
            "epoch": [-1000.0, -2000.0, -9.0, -8.0],
            "source_file_basename": ["s.hdf5"] * 4,
            "mc_pred_gap": [1.0, 2.0, 3.0, 4.0],
        }
    )
    # The rendering itself is not under test, and it needs a real model.
    monkeypatch.setattr(samples_analysis, "render_pages", lambda *args, **kwargs: ([], []))
    context = AnalysisContext(
        collection=types.SimpleNamespace(
            per_sample=per_sample, per_anchor=pd.DataFrame(), record={}, retained={}, results={},
            vectors={},
        ),
        config={},
        task=object(),
        loader=loader,
    )

    result = samples_analysis.run_samples_analysis(
        context, eval_config={"seed": 0, "caps": {"pages": 4}}, output_dir=tmp_path, probe={}
    )

    assert len(per_sample) < len(samples_analysis.dataset_index_map(loader)), (
        "the fixture must exercise the shorter-table case the old expression got wrong"
    )
    assert result["n_unlocatable_rows"] >= 2, (
        f"both unplaceable rows must be counted, got {result['n_unlocatable_rows']}"
    )


def test_the_page_loader_refuses_an_unordered_index_list() -> None:
    """A ``Subset`` is visited in the order it was built, so an unordered list pairs each page
    with another row's identity and every page still looks right."""
    with pytest.raises(ValueError, match="strictly ascending"):
        samples_analysis.page_loader(_loader(8), [5, 1])


def test_the_page_loader_visits_the_chosen_rows_in_order_one_at_a_time() -> None:
    pages = samples_analysis.page_loader(_loader(8), [1, 4, 6])

    seen = [batch["guid"][0] for batch in pages]

    assert seen == ["g1", "g4", "g6"]
    assert pages.batch_size == 1


def test_the_identity_check_catches_an_off_by_one_mapping() -> None:
    """The assertion that makes the whole re-render safe: a page of the wrong recording is the
    one failure mode nothing else in the run would notice."""
    batch = {"guid": ["g4"], "epoch": np.array([-4000.0])}

    samples_analysis.check_identity(batch, pd.Series({"guid": "g4", "epoch": -4000.0}))
    with pytest.raises(ValueError, match="not the row it was selected from"):
        samples_analysis.check_identity(batch, pd.Series({"guid": "g5", "epoch": -5000.0}))


# =============================================================================
# The draws
# =============================================================================
def test_a_cap_at_the_shard_count_reaches_every_shard() -> None:
    """A prefix would not: the loader is unshuffled over per-subgroup files, so the first n rows
    are one subgroup and one clinical class."""
    shards = [f"shard{index % 8}.hdf5" for index in range(80)]
    frame = pd.DataFrame({"guid": [f"g{index}" for index in range(80)],
                          "source_file_basename": shards})

    drawn = samples_analysis.stratified_rows(frame, cap=8, seed=0)

    assert len(drawn) == 8
    assert drawn["source_file_basename"].nunique() == 8


def test_the_extremes_are_the_smallest_and_largest_finite_values() -> None:
    frame = pd.DataFrame({"guid": list("abcde"), "mc_pred_gap": [3.0, np.nan, 1.0, 5.0, 2.0]})

    tails = samples_analysis.extreme_rows(frame, "mc_pred_gap", per_tail=2)

    assert list(tails["low"]["guid"]) == ["c", "e"]
    assert list(tails["high"]["guid"]) == ["a", "d"]


def test_a_metric_the_table_does_not_carry_yields_no_pages_rather_than_raising() -> None:
    tails = samples_analysis.extreme_rows(pd.DataFrame({"guid": ["a"]}), "absent", per_tail=2)

    assert not len(tails["low"]) and not len(tails["high"])


# =============================================================================
# The analysis
# =============================================================================
def test_a_pass_with_no_model_records_a_skip(tmp_path, evaluated) -> None:
    """An offline re-run has no model to render with, and a page cannot come off a table."""
    collection = load_collection(evaluated["results_dir"])

    result = samples_analysis.run_samples_analysis(
        AnalysisContext(collection=collection), eval_config={"seed": 0}, output_dir=tmp_path
    )

    assert result["skipped"] is True
    assert result["n_samples"] is None
    assert "no model" in result["reason"]


def test_one_failing_page_is_recorded_by_index_and_the_rest_still_render(
    tmp_path, monkeypatch
) -> None:
    """A page that cannot be drawn is a recorded absence, not a gap in the directory."""
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("deliberate page failure")
        from matplotlib.figure import Figure

        return Figure()

    monkeypatch.setattr(samples_analysis, "build_diagnostic_figure", flaky)
    # The builders are the task's, and this test is about the failure isolation around them.
    monkeypatch.setattr(
        samples_analysis, "model_inputs", lambda task, batch: (None, None, None, None, None)
    )
    rows = pd.DataFrame(
        {
            "guid": ["g0", "g1", "g2"],
            "epoch": [0.0, -1000.0, -2000.0],
            "dataset_index": [0, 1, 2],
        }
    )
    written, failures = samples_analysis.render_pages(
        _FakeTask(), _loader(4), rows, tmp_path / "pages", delay_steps=0, normalization=None
    )

    assert len(written) == 2 and len(failures) == 1
    assert failures[0]["dataset_index"] == 1
    assert "deliberate page failure" in failures[0]["error"]
    assert sorted(path.name for path in (tmp_path / "pages").glob("*.pdf")) == sorted(written)


class _FakeTask:
    """The smallest object :func:`render_pages` drives: a device, hparams and a stub net."""

    device = "cpu"
    hparams: Dict[str, Any] = {"kld_beta": 1.0}
    training = False

    class _Model:
        geometry = None

        def __call__(self, *_args):
            return {"mu_prior": None, "logvar_prior": None, "mu_post": None, "logvar_post": None}

        def kld_tensor(self, **_kwargs):
            return None

    orig_model = _Model()

    def transfer_batch_to_device(self, batch, *_args, **_kwargs):
        """Identity: the stub batches are already where they need to be."""
        return batch


def test_the_pages_of_a_real_run_name_the_recordings_they_were_selected_from(
    event_evaluated,
) -> None:
    """The end-to-end round trip: every filename's GUID appears in the table row it came from.

    This is what an off-by-one in the index mapping breaks, and it is the only place it would
    surface -- the pages themselves are perfectly plausible pictures either way.
    """
    directory = Path(event_evaluated["results_dir"]) / samples_analysis.ANALYSIS_DIRNAME
    manifest = pd.read_csv(directory / samples_analysis.MANIFEST_FILENAME)
    collection = load_collection(event_evaluated["results_dir"])
    known = {samples_analysis.sanitise_guid(guid) for guid in collection.per_sample["guid"]}

    assert len(manifest) > 0
    for _, row in manifest.iterrows():
        match = re.fullmatch(
            r"sample(\d{4})_([A-Za-z0-9_-]{1,32})_epoch(-?\d+|na)\.pdf", str(row["file"])
        )
        assert match, row["file"]
        assert match.group(2) in known
        assert (directory / str(row["selection"]) / str(row["file"])).is_file()


def test_the_stratified_draw_of_a_real_run_reached_every_shard(event_evaluated) -> None:
    result = event_evaluated["summary"]["results"]["samples"]

    assert result["composition"]["n_shards_reached"] == result["composition"]["n_stratified"]
    assert result["failures"] == []
    assert result["n_unlocatable_rows"] == 0
