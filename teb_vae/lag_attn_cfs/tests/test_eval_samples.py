r"""The per-sample diagnostic pages, and the index mapping everything about them rests on.

This is one of the two analyses that re-run inference, and the one whose failure mode is
invisible. A page drawn from the wrong dataset row is a complete, plausible, correctly-formatted
picture of a recording nobody asked for; nothing about it looks wrong, and no downstream number
moves. So the mapping from a table row back to the dataset is checked in three ways here -- the
round trip the analysis itself performs before rendering, the ascending-order precondition the
``Subset`` visit assumes, and the end-to-end assertion that a rendered page's filename names the
recording its row carried.

Two smaller properties matter for the same reason. **Coverage**: a cap at or above the shard count
must reach every shard, because a prefix over eight concatenated per-subgroup files is one subgroup
and one clinical class. **Containment**: a GUID is an external string, and it ends up in a
filename; a separator, a space or a non-ASCII character in it must not be able to leave the
directory or produce a name a shell cannot address.

**And the property this cell adds: the page has fifteen rows, and the check is on the absence of a
warning rather than on the presence of the rows.** The shipped input-row builder is welded to the
production two-sided Morlet bank -- it refuses these channel widths -- and the *training* callback
catches that, warns, and continues with two rows missing. An evaluation that did the same would
write a silently shortened page into a results directory somebody reads months later, so the
builder here is the task's own and its failure is not swallowed. Both halves are asserted: the page
draws the fifteen rows, and nothing warns while it does.

The forward is dense, at the geometry the collection pass scored at. That is not a preference: this
cell's forecast tensors are indexed by *position in the decoded set*, so a page drawn at the
training tiling would place every window at the wrong time with no shape error anywhere in it.
"""
from __future__ import annotations

import logging
import re
import types
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_cfs.eval.analyses import AnalysisContext
from teb_vae.lag_attn_cfs.eval.analyses import samples as samples_analysis
from teb_vae.lag_attn_cfs.eval.collect import load_collection
from teb_vae.lag_attn_cfs.eval.metrics import DENSE_ANCHOR_GEOMETRY

from .conftest import make_stub_batch, make_task


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


class _FakeTask:
    """The smallest object :func:`render_pages` drives: a device, hparams and a stub net."""

    device = "cpu"
    hparams: Dict[str, Any] = {"kld_beta": 1.0}
    training = False
    # No page seams: this stub exists to test the failure isolation around the builder, and the
    # seams are resolved by ``page_seams`` off a real task in the tests that need them.
    forecast_extra_rows = ()

    class _Model:
        geometry = None

        def __call__(self, *_args, **_kwargs):
            return {"mu_prior": None, "logvar_prior": None, "mu_post": None, "logvar_post": None}

        def kld_tensor(self, **_kwargs):
            return None

    orig_model = _Model()

    def transfer_batch_to_device(self, batch, *_args, **_kwargs):
        """Identity: the stub batches are already where they need to be."""
        return batch


def _no_seams() -> Dict[str, Any]:
    """The resolved-seam mapping a task declaring nothing produces."""
    return {"forecast_rows": None, "forecast_extra_rows": (), "input_stream_panels": None}


# =================================================================================================
# Filenames
# =================================================================================================
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


# =================================================================================================
# The index mapping
# =================================================================================================
def test_a_row_resolves_to_its_own_dataset_position() -> None:
    loader = _loader(8)
    rows = pd.DataFrame({"guid": ["g5", "g1"], "epoch": [-5000.0, -1000.0]})

    resolved = samples_analysis.resolve_rows(rows, samples_analysis.dataset_index_map(loader))

    assert list(resolved["dataset_index"]) == [1, 5], "ascending, as the Subset visit assumes"
    assert list(resolved["guid"]) == ["g1", "g5"]


def test_a_row_the_dataset_cannot_place_is_dropped_rather_than_guessed_at() -> None:
    loader = _loader(4)
    rows = pd.DataFrame({"guid": ["g1", "absent"], "epoch": [-1000.0, -9.0]})

    resolved = samples_analysis.resolve_rows(rows, samples_analysis.dataset_index_map(loader))

    assert list(resolved["guid"]) == ["g1"]


def test_the_unlocatable_count_is_the_rows_that_failed_to_resolve(tmp_path, monkeypatch) -> None:
    """Not a difference between two row counts of different populations.

    ``len(per_sample) - len(index_map)`` compares scored segments against locatable dataset rows:
    it goes negative whenever the collection pass skipped a batch, and it reads zero whenever a
    genuine drop happens to be offset by one. The number wanted is how many rows the pages asked
    for and the dataset could not place.
    """
    loader = _loader(8)
    # Two rows the dataset cannot place, and a table shorter than the dataset -- which is what a
    # skipped batch leaves behind.
    per_sample = pd.DataFrame(
        {
            "guid": ["g1", "g2", "absent_a", "absent_b"],
            "epoch": [-1000.0, -2000.0, -9.0, -8.0],
            "source_file_basename": ["s.hdf5"] * 4,
            "mc_pred_gap": [1.0, 2.0, 3.0, 4.0],
        }
    )
    # The rendering itself is not under test, and it needs a real model.
    monkeypatch.setattr(samples_analysis, "render_pages", lambda *args, **kwargs: ([], [], None))
    context = AnalysisContext(
        collection=types.SimpleNamespace(
            per_sample=per_sample, per_anchor=pd.DataFrame(), record={}, retained={}, results={},
        ),
        config={},
        task=object(),
        loader=loader,
    )

    result = samples_analysis.run_samples_analysis(
        context, eval_config={"seed": 0, "caps": {"pages": 4}}, output_dir=tmp_path, probe={}
    )

    assert len(per_sample) < len(samples_analysis.dataset_index_map(loader)), (
        "the fixture must exercise the shorter-table case"
    )
    assert result["n_unlocatable_rows"] >= 2, (
        f"both unplaceable rows must be counted, got {result['n_unlocatable_rows']}"
    )
    # No page was rendered, so the row count is unmeasured rather than a guess from the seams.
    assert result["plan"]["page_rows"] is None


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


# =================================================================================================
# The draws
# =================================================================================================
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


def test_the_two_tails_never_share_a_segment() -> None:
    """Asking for more per tail than the split can fill lowers the count rather than overlapping.
    A segment rendered into both ``_low/`` and ``_high/`` reads as simultaneously the best and the
    worst case, and the ones that would double up sit nearest the median -- extreme in neither
    direction."""
    frame = pd.DataFrame({"guid": list("abcdefg"), "mc_pred_gap": [float(i) for i in range(7)]})

    tails = samples_analysis.extreme_rows(frame, "mc_pred_gap", per_tail=10)

    assert len(tails["low"]) == 3 and len(tails["high"]) == 3
    assert set(tails["low"]["guid"]).isdisjoint(set(tails["high"]["guid"]))
    # Still the actual extremes, taken from each end.
    assert list(tails["low"]["guid"]) == ["a", "b", "c"]
    assert list(tails["high"]["guid"]) == ["e", "f", "g"]


def test_a_single_finite_value_is_not_reported_as_both_extremes() -> None:
    """One value is one observation, not a low and a high."""
    frame = pd.DataFrame({"guid": ["a", "b"], "mc_pred_gap": [1.0, np.nan]})

    tails = samples_analysis.extreme_rows(frame, "mc_pred_gap", per_tail=10)

    assert not len(tails["low"]) and not len(tails["high"])


def test_the_extreme_metrics_are_columns_the_collection_pass_writes() -> None:
    """A stem naming a column no table carries renders nothing and is recorded as missing, which
    is a silent loss of a third of the pages. Checked against the readouts a real batch produces
    rather than against a list."""
    from teb_vae.lag_attn_cfs.eval.metrics import evaluate_batch

    module = make_task()
    module.eval()
    columns = set(evaluate_batch(module, make_stub_batch(seed=1), num_samples=1).columns)

    assert {column for _stem, column in samples_analysis.EXTREME_METRICS} <= columns


# =================================================================================================
# The task's page seams
# =================================================================================================
def test_the_three_seams_are_resolved_off_the_task_rather_than_imported() -> None:
    """The seams are the *task's* to declare -- it is the task that decides what the target is and
    what the encoders are fed -- and resolving them by name is what keeps one page definition
    behind both the fit's diagnostics and the evaluation's. An analysis that imported this
    package's ``sample_page`` would be a second builder that could drift from the callback's."""
    module = make_task()

    seams = samples_analysis.page_seams(module)

    assert set(seams) == set(samples_analysis.PAGE_SEAMS)
    assert callable(seams["forecast_rows"])
    assert callable(seams["input_stream_panels"])
    assert len(seams["forecast_extra_rows"]) == 6
    # The layering test forbids the analyses from importing the task; this is the behavioural half
    # of the same rule, and it is what makes the page the one the training callback draws.
    assert seams["forecast_extra_rows"] == module.forecast_extra_rows


def test_a_task_declaring_no_seams_costs_rows_rather_than_raising() -> None:
    """A model over another representation is a shorter page, not a failed run -- and the row
    count says which it was."""
    seams = samples_analysis.page_seams(object())

    assert seams["forecast_rows"] is None
    assert seams["forecast_extra_rows"] == ()
    assert samples_analysis.input_stream_rows(None, seams["input_stream_panels"], (), 0) == ()


def test_the_input_row_builder_is_this_cells_own_and_draws_two_streams() -> None:
    """The shipped builder refuses these channel widths, so a page that used it would lose two
    rows to a warning. Two panels -- target and source -- is what makes the page fifteen rows."""
    module = make_task()
    batch = make_stub_batch(seed=2)
    seams = samples_analysis.page_seams(module)
    streams = (batch.fhr_st, batch.fhr_ph, module._build_source_stream(batch))

    panels = samples_analysis.input_stream_rows(
        module.orig_model, seams["input_stream_panels"], streams, 0
    )

    assert [panel.name for panel in panels] == ["target", "source"]


def test_a_failing_input_builder_is_not_swallowed_into_a_shorter_page() -> None:
    """The deliberate divergence from the training callback's wrapper, which warns and continues.
    In an evaluation that writes a silently shortened page into a directory read months later, so
    the exception travels and the per-page handler records it instead."""

    def broken(_model, _inputs, *, sample_index=0):
        raise ValueError("deliberate builder failure")

    with pytest.raises(ValueError, match="deliberate builder failure"):
        samples_analysis.input_stream_rows(object(), broken, (), 0)


def test_the_raw_trace_statistics_come_from_the_loader_not_the_narrowed_record() -> None:
    """``collection.record['normalization']`` is deliberately the four stored feature blocks. Row
    one of the page draws the raw FHR and UP traces, which are different fields entirely, and the
    row labels its axis ``normalised`` unless it finds *those* constants -- so passing the narrowed
    record would mislabel a bpm trace rather than merely lose a conversion."""
    wanted = {"fhr": {"mean": 140.0, "std": 10.0}}
    loader = types.SimpleNamespace(
        dataset=types.SimpleNamespace(get_normalization_stats=lambda: wanted)
    )

    assert samples_analysis.raw_trace_normalization(loader) == wanted
    # A dataset that reports none is not an error: the row falls back to loader units and says so.
    assert samples_analysis.raw_trace_normalization(_loader(2)) is None


# =================================================================================================
# The page itself
# =================================================================================================
def test_a_rendered_page_has_fifteen_rows_and_logs_no_warning(tmp_path, caplog) -> None:
    """The demo, and the assertion the acceptance criterion is stated as: the shipped input-row
    builder fails inside a handler that warns and continues, so a page that lost its two input rows
    would still be a page. The absence of the warning is therefore the property, and the row count
    is read off the rendered figure rather than off a constant."""
    module = make_task()
    module.eval()
    batch = make_stub_batch(seed=4)

    class _OneRowLoader:
        """A loader over exactly the stub batch, so the page renders without a shard."""

        dataset = [0]
        collate_fn = staticmethod(lambda items: batch)

    rows = pd.DataFrame(
        {"guid": [batch.guid[0]], "epoch": [float(batch.epoch[0])], "dataset_index": [0]}
    )

    with caplog.at_level(logging.WARNING):
        written, failures, n_input_rows = samples_analysis.render_pages(
            module, _OneRowLoader(), rows, tmp_path / "pages",
            delay_steps=0, normalization=None, seams=samples_analysis.page_seams(module),
        )

    assert failures == [], failures
    assert len(written) == 1
    assert n_input_rows == 2
    assert 2 + 6 + n_input_rows + 5 == samples_analysis.EXPECTED_PAGE_ROWS
    assert "warning" not in caplog.text.lower(), caplog.text


def test_the_page_is_drawn_at_the_geometry_the_collection_pass_scored(tmp_path) -> None:
    r"""The forecast tensors are indexed by *position in the decoded set*, so a page drawn at the
    training tiling places every window at the wrong time with no shape error anywhere in it. The
    forward's keyword arguments are recorded and compared against
    :data:`DENSE_ANCHOR_GEOMETRY` -- which is what the collection pass used."""
    module = make_task()
    module.eval()
    batch = make_stub_batch(seed=5)
    seen: List[Dict[str, Any]] = []
    original = type(module.orig_model).forward

    def _record(self, *args, **kwargs):
        seen.append(dict(kwargs))
        return original(self, *args, **kwargs)

    class _OneRowLoader:
        dataset = [0]
        collate_fn = staticmethod(lambda items: batch)

    rows = pd.DataFrame(
        {"guid": [batch.guid[0]], "epoch": [float(batch.epoch[0])], "dataset_index": [0]}
    )
    model = module.orig_model
    model.forward = _record.__get__(model, type(model))
    try:
        _written, failures, _rows = samples_analysis.render_pages(
            module, _OneRowLoader(), rows, tmp_path / "pages",
            delay_steps=0, normalization=None, seams=samples_analysis.page_seams(module),
        )
    finally:
        del model.forward

    assert failures == [], failures
    assert int(module.orig_model.anchor_stride) != DENSE_ANCHOR_GEOMETRY[1], (
        "the task tiles in training, or this proves nothing"
    )
    assert seen and (seen[0]["anchor_phase"], seen[0]["anchor_stride"]) == DENSE_ANCHOR_GEOMETRY


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
    written, failures, _n_input_rows = samples_analysis.render_pages(
        _FakeTask(), _loader(4), rows, tmp_path / "pages",
        delay_steps=0, normalization=None, seams=_no_seams(),
    )

    assert len(written) == 2 and len(failures) == 1
    assert failures[0]["dataset_index"] == 1
    assert "deliberate page failure" in failures[0]["error"]
    assert sorted(path.name for path in (tmp_path / "pages").glob("*.pdf")) == sorted(written)


# =================================================================================================
# The analysis
# =================================================================================================
def test_a_pass_with_no_model_records_a_skip(tmp_path) -> None:
    """An offline re-run has no model to render with, and a page cannot come off a table."""
    per_sample = pd.DataFrame({"guid": ["g0"], "epoch": [-1000.0]})
    context = AnalysisContext(
        collection=types.SimpleNamespace(
            per_sample=per_sample, per_anchor=pd.DataFrame(), record={}, retained={}, results={},
        ),
        config={},
    )

    result = samples_analysis.run_samples_analysis(
        context, eval_config={"seed": 0}, output_dir=tmp_path
    )

    assert result["skipped"] is True
    assert result["n_samples"] is None
    assert "no model" in result["reason"]


# =================================================================================================
# Against the real run
# =================================================================================================
@pytest.mark.slow
def test_the_pages_of_a_real_run_name_the_recordings_they_were_selected_from(
    collected_run,
) -> None:
    """The end-to-end round trip: every filename's GUID appears in the table row it came from.

    This is what an off-by-one in the index mapping breaks, and it is the only place it would
    surface -- the pages themselves are perfectly plausible pictures either way.
    """
    directory = Path(collected_run["results_dir"]) / samples_analysis.ANALYSIS_DIRNAME
    manifest = pd.read_csv(directory / samples_analysis.MANIFEST_FILENAME)
    collection = load_collection(collected_run["results_dir"])
    known = {samples_analysis.sanitise_guid(guid) for guid in collection.per_sample["guid"]}

    assert len(manifest) > 0
    for _, row in manifest.iterrows():
        match = re.fullmatch(
            r"sample(\d{4})_([A-Za-z0-9_-]{1,32})_epoch(-?\d+|na)\.pdf", str(row["file"])
        )
        assert match, row["file"]
        assert match.group(2) in known
        assert (directory / str(row["selection"]) / str(row["file"])).is_file()


@pytest.mark.slow
def test_a_real_run_renders_every_page_it_asked_for_at_fifteen_rows(collected_run) -> None:
    """The two failure modes that leave a green run: a page that could not be drawn, and a page
    drawn without its input rows. The first is a recorded failure; the second is only visible as
    a row count, which is why the run records one."""
    result = collected_run["summary"]["results"]["samples"]

    assert result["failures"] == []
    assert result["n_unlocatable_rows"] == 0
    assert result["plan"]["page_rows"] == samples_analysis.EXPECTED_PAGE_ROWS
    assert all(result["plan"]["page_seams"].values())


@pytest.mark.slow
def test_the_stratified_draw_of_a_real_run_reached_every_shard(collected_run) -> None:
    """A cap at or above the shard count must reach every shard. Compared against the shards the
    table actually holds rather than against the drawn count: the shipped cap is above the shard
    count, so the two are equal only by coincidence on a smaller split -- and the property being
    tested is coverage, not the cap."""
    result = collected_run["summary"]["results"]["samples"]
    per_sample = load_collection(collected_run["results_dir"]).per_sample
    n_shards = int(per_sample["source_file_basename"].nunique())

    assert n_shards > 1, "one shard would make the stratification vacuous"
    assert result["composition"]["n_stratified"] >= n_shards
    assert result["composition"]["n_shards_reached"] == n_shards
    assert result["n_samples"] >= result["composition"]["n_stratified"]
