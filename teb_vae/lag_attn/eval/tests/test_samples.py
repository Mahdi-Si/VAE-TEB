"""Per-sample diagnostic pages: the cap is honoured, and one bad page does not lose the rest.

The failure-isolation test is the one worth having. These pages are the last artifact a
multi-hour run produces, so a single recording with a degenerate field taking down the step
would discard every page already written -- and the loss would be invisible, because a run that
emitted seven of eight pages and one that emitted none both look like "samples/ exists".

Every test here caps the draw to the fewest pages that can still fail. A page of the committed
shard is a full production-geometry render -- $T = 300$, $c_y = 109$ -- and costs seconds, so an
uncapped test would put minutes into the fast gate to prove what a two-page draw proves just as
well. The two tests that genuinely need the whole shard say why.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from teb_vae.lag_attn.eval import sample_figure
from teb_vae.lag_attn.eval.analyses import samples as samples_analysis
from teb_vae.lag_attn.eval.analyses import probe as probe_analysis


@pytest.fixture
def runner(make_eval_runner):
    """A runner whose full pathway is perturbed, so the pages show a live residual."""
    return make_eval_runner()


@pytest.fixture
def probe_record(runner, tiny_loader):
    """The loader probe's record, supplying the totals the stratified draw needs."""
    return probe_analysis.run_probe(runner, tiny_loader, configured_files=None)


def _config(**overrides):
    """A minimal validated-shape ``eval_config`` for the analysis."""
    config = {"seed": 7, "max_samples": None, "caps": {}}
    config.update(overrides)
    return config


def _run(runner, loader, tmp_path, probe=None, **config_overrides):
    return samples_analysis.run_samples_analysis(
        runner, loader, eval_config=_config(**config_overrides),
        output_dir=tmp_path, probe=probe,
    )


# ---------------------------------------------------------------------------
# Filenames
# ---------------------------------------------------------------------------
def test_the_filename_carries_the_index_the_guid_and_the_epoch():
    name = samples_analysis.sample_filename("abc-123", 4.5, 12)
    assert name == "sample0012_abc-123_epoch4.50"


def test_a_path_unsafe_guid_is_sanitised_rather_than_written_through():
    """A GUID is an opaque record identifier with no path-safety guarantee."""
    name = samples_analysis.sample_filename("a/b\\c:d", None, 0)
    assert name == "sample0000_abcd_epochna"
    assert "/" not in name and "\\" not in name and ":" not in name


def test_an_empty_guid_still_yields_a_usable_filename():
    assert samples_analysis.sample_filename("///", None, 3).startswith("sample0003_unknown")


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------
def test_a_page_is_written_for_every_selected_sample(runner, tiny_loader, tmp_path, probe_record):
    """Uncapped on purpose: this is the test that proves the default draws the whole split."""
    summary = _run(runner, tiny_loader, tmp_path, probe=probe_record)
    directory = Path(tmp_path) / samples_analysis.ANALYSIS_DIRNAME

    written = sorted(directory.glob("*.pdf"))
    assert len(written) == summary["n_figures"] == 4  # the whole tiny shard
    assert summary["failures"] == {}
    for path in written:
        assert path.stat().st_size > 0


def test_the_cap_bounds_the_page_count(runner, tiny_loader, tmp_path, probe_record):
    """Uncapped, this analysis would emit one full-size PDF per recording in the split."""
    summary = _run(runner, tiny_loader, tmp_path, probe=probe_record, caps={"samples": 2})
    directory = Path(tmp_path) / samples_analysis.ANALYSIS_DIRNAME

    assert summary["n_figures"] == 2
    assert len(list(directory.glob("*.pdf"))) == 2
    assert summary["plan"]["cap"] == 2 and summary["plan"]["capped"] is True


def test_the_per_sample_csv_has_one_row_per_page_and_names_it(
    runner, tiny_loader, tmp_path, probe_record
):
    summary = _run(runner, tiny_loader, tmp_path, probe=probe_record, caps={"samples": 2})
    directory = Path(tmp_path) / samples_analysis.ANALYSIS_DIRNAME

    frame = pd.read_csv(directory / "per_sample.csv")
    assert len(frame) == 2 == summary["n_samples"]
    for column in ("sample_index", "guid", "source_file", "epoch", "figure"):
        assert column in frame.columns
    # The CSV is an index of the pages: every named file must be on disk.
    for name in frame["figure"]:
        assert (directory / str(name)).is_file()


def test_the_csv_carries_the_metrics_the_pages_draw(runner, tiny_loader, tmp_path, probe_record):
    summary = _run(runner, tiny_loader, tmp_path, probe=probe_record, caps={"samples": 1})
    frame = pd.read_csv(Path(summary["figures"][0]).parent / "per_sample.csv")
    for column in ("feat_mse_total", "kld_mean", "mean_argmax_lag"):
        assert column in frame.columns, f"{column} missing from the per-sample table"
    assert frame["feat_mse_total"].notna().all()


def test_the_draw_is_stratified_over_the_source_files(runner, tiny_loader, tmp_path, probe_record):
    """A prefix cap over concatenated per-subgroup shards is one subgroup and one class."""
    summary = _run(runner, tiny_loader, tmp_path, probe=probe_record, caps={"samples": 2})
    assert sum(summary["composition"].values()) == 2
    assert summary["plan"]["n_total"] == probe_record["n_samples"]


def test_the_te_row_label_is_recorded_alongside_the_pages(
    runner, tiny_loader, tmp_path, probe_record
):
    """A reader cannot tell an attribution from a diagnostic by looking at the picture."""
    summary = _run(runner, tiny_loader, tmp_path, probe=probe_record, caps={"samples": 1})
    assert summary["te_lag_map_label"] in {"attribution", "diagnostic"}


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------
def test_one_failing_page_leaves_the_others_written(
    runner, tiny_loader, tmp_path, probe_record, monkeypatch
):
    """The point of the per-sample guard: a bad recording costs its own page and nothing else."""
    real_builder = sample_figure.build_sample_figure
    state = {"calls": 0}

    def flaky(**kwargs):
        state["calls"] += 1
        if state["calls"] == 2:
            raise ValueError("synthetic per-sample failure")
        return real_builder(**kwargs)

    monkeypatch.setattr(sample_figure, "build_sample_figure", flaky)

    summary = _run(runner, tiny_loader, tmp_path, probe=probe_record, caps={"samples": 3})
    directory = Path(tmp_path) / samples_analysis.ANALYSIS_DIRNAME

    assert summary["n_samples"] == 3, "the failing sample must still get a CSV row"
    assert summary["n_figures"] == 2
    assert len(summary["failures"]) == 1
    assert "synthetic per-sample failure" in next(iter(summary["failures"].values()))
    assert len(list(directory.glob("*.pdf"))) == 2

    # The row is there, with its figure recorded as absent rather than pointing at nothing.
    frame = pd.read_csv(directory / "per_sample.csv")
    assert frame["figure"].isna().sum() == 1


def test_the_analysis_runs_without_a_probe_record(runner, tiny_loader, tmp_path):
    """``--only samples`` after a failed run still has to work; the draw is then unstratified.

    Uncapped for the same reason as the draw test: with no probe there is no total to plan
    against, so no cap can apply and every sample is rendered.
    """
    summary = _run(runner, tiny_loader, tmp_path, probe=None)
    assert summary["n_figures"] == 4
    assert summary["plan"] is None
