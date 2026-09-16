r"""The figures, drawn from a hand-written summary and from a hand-written acceptance record.

Every builder takes the parsed artifact and nothing else, which is what makes it testable here
without a fit, a shard or a checkpoint: a summary written by hand carries the same blocks a run
writes, and a builder that raises on one of them would raise at the last step of a real pass.

Two failure modes are the ones worth naming. A builder that raises on a block a legitimate run does
not produce -- a target-only arm has no bands and no lag profile, a normalised fusion has no latent
profile -- would lose the summary of exactly the arms the comparison needs. And a figure written
under a name the manifest does not list, or a manifest naming a figure that was not written, would
leave a reader of the directory unable to tell a missing figure from a renamed one.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from teb_vae.lag_attn.eval.figures import active_figure_format  # noqa: E402
from teb_vae.lag_slot_transformer_cfs.eval import acceptance, figures  # noqa: E402

#: Candidate lags of the hand-written summary. Small, and more than the bands need.
N_LAGS = 6

#: Horizon steps of the hand-written summary.
HORIZON = 4


def _interval(point: float, half: float = 0.5, n: int = 5) -> Dict[str, Any]:
    """A scalar bootstrap record.

    Args:
        point: The point estimate.
        half: Half the interval's width.
        n: Recordings behind it.

    Returns:
        The record.
    """
    return {"point": point, "lo": point - half, "hi": point + half, "n": n}


def _curve(values: List[float], half: float = 0.3, n: int = 5) -> Dict[str, Any]:
    """A curve bootstrap record.

    Args:
        values: The point estimates.
        half: Half the interval's width at every position.
        n: Recordings behind it.

    Returns:
        The record.
    """
    return {
        "point": list(values),
        "lo": [value - half for value in values],
        "hi": [value + half for value in values],
        "n": n,
    }


def candidate_summary() -> Dict[str, Any]:
    """A summary with every block a source-conditioned run of the recommended arm writes.

    Returns:
        The summary.
    """
    arms = ["base", "full", "suppress:none", "suppress:near", "suppress:far", "suppress:all",
            "silence", "replace:zeros", "replace:constant", "permute"]
    steps = list(range(1, HORIZON + 1))
    return {
        "arm_scores": {
            **{f"nll_{arm}": _interval(100.0 + index) for index, arm in enumerate(arms)},
            "pred_gap": _interval(-1.0),
            "kld_per_anchor": _interval(0.2),
        },
        "anchor_weighted": {"pred_gap": -1.1},
        "arm": {"source_disabled": False, "lag_fusion": "local", "source_stem": "pointwise"},
        "lag_readouts": {
            "band_suppression": {
                "none": {"margin_nats": 0.0, "margin_interval": _interval(0.0, 0.0),
                         "band_anchors": 0.0, "band_channels": 0.0},
                "near": {"margin_nats": 0.4, "margin_interval": _interval(0.4),
                         "band_anchors": 40.0, "band_channels": 400.0},
                "far": {"margin_nats": None, "margin_interval": None,
                        "band_anchors": 40.0, "band_channels": 0.0},
                "all": {"margin_nats": -1.0, "margin_interval": _interval(-1.0),
                        "band_anchors": 80.0, "band_channels": 400.0},
            },
            "band_edges": {"near": [0, 2], "far": [3, N_LAGS - 1]},
            "cancellation": {"mean": {"ratio": 0.5, "numerator": 1.0, "denominator": 2.0}},
            "exposure": {
                "per_band": {},
                "per_lag_anchors": [40.0] * N_LAGS,
                "per_lag_channels": [200.0] * 3 + [0.0] * 3,
                "per_source_channel": [40.0] * 5,
            },
            "lag_profile": {
                "latent": {
                    "anchors_per_lag": [40.0] * N_LAGS,
                    "proposal_norm": [0.1] * N_LAGS,
                    "update_shift": [0.05] * N_LAGS,
                    "divergence_drop": [0.01, 0.02, -0.01, 0.0, 0.0, 0.0],
                    "scale_proposal_norm": [0.02] * N_LAGS,
                },
                "predictive": {
                    "status": "READ", "n_segments": 8, "n_recordings": 5, "cap": 8,
                    "margin_nats": _curve([0.1, 0.2, 0.0, -0.1, 0.0, 0.0]),
                },
            },
            "lag_axis": {"n_lags": N_LAGS, "seconds_per_step": 4.0, "delay_steps": 0},
            "qualification": "fitted computation",
        },
        "horizon_resolved": {
            "positions": steps,
            "unit": "nats per anchor per horizon step",
            "nll": {arm: _curve([10.0 + step for step in steps]) for arm in arms},
            "pred_gap": _curve([-0.2] * HORIZON),
            "band_margins": {name: _curve([0.1] * HORIZON) for name in ("none", "near", "far", "all")},
            "control_margins": {name: _curve([0.05] * HORIZON)
                                for name in ("silence", "replace_zeros", "replace_constant", "permute")},
        },
        "block_resolved": {
            "positions": ["st", "ph"],
            "unit": "nats per anchor",
            "nll": {arm: _curve([60.0, 40.0]) for arm in arms},
            "pred_gap": _curve([-0.5, -0.5]),
            "band_margins": {name: _curve([0.2, 0.2]) for name in ("none", "near", "far", "all")},
            "control_margins": {name: _curve([0.1, 0.1])
                                for name in ("silence", "replace_zeros", "replace_constant", "permute")},
            "channels_per_block": {"st": 3, "ph": 4},
        },
        "source_controls": {
            "silence_margin_nats": -1.0, "silence_margin_interval": _interval(-1.0),
            "replace_zeros_margin_nats": 0.3, "replace_zeros_margin_interval": _interval(0.3),
            "replace_constant_margin_nats": 0.2, "replace_constant_margin_interval": _interval(0.2),
            "permute_margin_nats": 0.1, "permute_margin_interval": _interval(0.1),
        },
        "mixture_calibration": {
            branch: {
                "n_coefficients": 100.0, "pit_mean": 0.5, "pit_var": 0.08,
                "coverage": {"0.5": 0.48, "0.9": 0.88, "0.99": 0.97},
                "uniform_pit_mean": 0.5, "uniform_pit_var": 1.0 / 12.0,
            }
            for branch in ("base", "full")
        },
        "draws": {"num_mc_samples": 8},
    }


def target_only_summary() -> Dict[str, Any]:
    """A summary with only the blocks a target-only run writes: no bands, no profile, no controls.

    Returns:
        The summary.
    """
    return {
        "arm_scores": {"nll_base": _interval(100.0), "nll_full": _interval(100.0),
                     "pred_gap": _interval(0.0, 0.0)},
        "anchor_weighted": {"pred_gap": 0.0},
        "arm": {"source_disabled": True},
        "lag_readouts": {"band_suppression": {}, "cancellation": {},
                         "exposure": {"per_band": {}, "per_lag_anchors": [],
                                      "per_lag_channels": [], "per_source_channel": []},
                         "lag_profile": {"latent": {}, "predictive": {"status": "SKIPPED"}},
                         "lag_axis": {}, "qualification": "fitted computation"},
        "horizon_resolved": {"positions": [1, 2], "nll": {"base": _curve([1.0, 2.0]),
                                                           "full": _curve([1.0, 2.0])},
                             "pred_gap": _curve([0.0, 0.0]), "band_margins": {},
                             "control_margins": {}},
        "block_resolved": {},
        "source_controls": {"silence_margin_nats": None, "skipped": {"suppress": "no source"}},
        "mixture_calibration": {"base": {"coverage": {}}, "full": {"coverage": {}}},
        "draws": {"num_mc_samples": 8},
    }


def per_recording_rows() -> Dict[str, Dict[str, float]]:
    """A per-recording table with the two columns the distribution figure reads.

    Returns:
        ``{recording: {column: value}}``.
    """
    return {
        f"rec{index}": {"pred_gap": -1.0 + 0.1 * index, "draw_concentration_full": 3.0 + index}
        for index in range(5)
    }


BUILDERS = {
    "headline": lambda summary: figures.build_headline_figure(summary),
    "distribution": lambda summary: figures.build_gap_distribution_figure(
        summary, per_recording_rows()
    ),
    "bands": lambda summary: figures.build_band_figure(summary),
    "lag_profile": lambda summary: figures.build_lag_profile_figure(summary),
    "horizon": lambda summary: figures.build_horizon_figure(summary),
    "block": lambda summary: figures.build_block_figure(summary),
    "calibration": lambda summary: figures.build_calibration_figure(summary),
}


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_every_builder_draws_the_candidate_summary(name: str) -> None:
    """Every block present, every panel drawn, and the figure handed back rather than saved.

    Args:
        name: The builder.
    """
    figure = BUILDERS[name](candidate_summary())
    try:
        assert figure.axes, name
        # Something was drawn on at least one panel: a line, a marker, a bar or a patch.
        assert any(ax.lines or ax.patches or ax.collections or ax.texts for ax in figure.axes)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_every_builder_survives_a_target_only_summary(name: str) -> None:
    """The arms with the fewest blocks are the ones a comparison cannot do without.

    Args:
        name: The builder.
    """
    figure = BUILDERS[name](target_only_summary())
    try:
        assert figure.axes, name
    finally:
        plt.close(figure)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_every_builder_survives_an_empty_summary(name: str) -> None:
    """An empty artifact draws empty panels rather than raising.

    Args:
        name: The builder.
    """
    figure = BUILDERS[name]({})
    try:
        assert figure.axes, name
    finally:
        plt.close(figure)


def test_the_three_readout_analyses_write_their_figures_and_tables(tmp_path: Path) -> None:
    """The run's figures are written by this cell's own analyses, each into its own subdirectory,
    from the results block alone: no model, no loader, no tensor."""
    from types import SimpleNamespace

    import pandas as pd

    from teb_vae.lag_slot_transformer_cfs.eval.analyses import (
        arms, lag_suppression, resolved_axes,
    )

    figures.configure_figure_style()
    results = candidate_summary()
    results["per_recording"] = {
        guid: {**row, "n_segments": 1.0, "n_scored_anchors": 4.0}
        for guid, row in per_recording_rows().items()
    }
    per_sample = pd.DataFrame(
        {"guid": list(results["per_recording"]), "mc_nll_full_block": [1.0] * 5}
    )
    context = SimpleNamespace(
        collection=SimpleNamespace(results=results, per_sample=per_sample, record={}),
        config={"dataset_config": {"vae_test_datasets": ["a/x.hdf5"], "stat_path": "a/s.hdf5"}},
        task=None,
        loader=None,
    )
    eval_config = {"seed": 0, "bootstrap_resamples": 100, "caps": {"lag_profile": 8}}

    blocks = {
        arms.ANALYSIS_DIRNAME: arms.run_arms_analysis(
            context, eval_config=eval_config, output_dir=tmp_path
        ),
        lag_suppression.ANALYSIS_DIRNAME: lag_suppression.run_lag_suppression_analysis(
            context, eval_config=eval_config, output_dir=tmp_path
        ),
        resolved_axes.ANALYSIS_DIRNAME: resolved_axes.run_resolved_axes_analysis(
            context, eval_config=eval_config, output_dir=tmp_path
        ),
    }

    extension = active_figure_format()
    for directory, block in blocks.items():
        assert {"n_samples", "composition", "plan"} <= set(block), directory
        for name in block["files"]:
            assert (tmp_path / directory / name).is_file(), (directory, name)
    assert {
        path.name for path in (tmp_path / arms.ANALYSIS_DIRNAME).iterdir()
    } == {
        arms.PER_RECORDING_FILENAME,
        f"{arms.HEADLINE_FIGURE}.{extension}",
        f"{arms.GAP_FIGURE}.{extension}",
        f"{arms.CALIBRATION_FIGURE}.{extension}",
    }
    assert (tmp_path / lag_suppression.ANALYSIS_DIRNAME / lag_suppression.LAG_PROFILE_FILENAME).is_file()
    assert (tmp_path / resolved_axes.ANALYSIS_DIRNAME / resolved_axes.HORIZON_FILENAME).is_file()
    # The scored-split record names the table the acceptance pass reads.
    assert blocks[arms.ANALYSIS_DIRNAME]["scored_split"]["per_recording_table"] == arms.PER_RECORDING_TABLE
    assert blocks[arms.ANALYSIS_DIRNAME]["scored_split"]["n_recordings"] == 5


def test_the_lag_axis_is_labelled_in_stored_coefficient_time() -> None:
    """A lag figure that did not say what its axis is time IN would read as physiological time."""
    figure = figures.build_lag_profile_figure(candidate_summary())
    try:
        # The seconds axis is a child of the top panel rather than a panel of its own.
        labels = [ax.get_xlabel() for ax in figure.axes]
        labels += [child.get_xlabel() for ax in figure.axes for child in ax.child_axes]
        assert any("stored-coefficient time" in label for label in labels)
        assert any("stored steps back from the anchor" in label for label in labels)
        # And the qualification travels on the figure itself.
        assert any("fitted computation" in text.get_text() for text in figure.texts)
    finally:
        plt.close(figure)


def test_the_lag_axis_and_the_band_labels_follow_the_summary_under_another_window() -> None:
    """No builder holds a band name or a lag count: a summary over a longer window with six
    declared bands draws that many lag positions and names those six bands."""
    n_lags = 25
    names = ["instant", "recent", "intermediate", "tail", "common_head", "common_tail"]
    edges = {"instant": [0, 0], "recent": [1, 4], "intermediate": [5, 12], "tail": [13, 24],
             "common_head": [0, 14], "common_tail": [15, 24]}
    results = candidate_summary()
    readouts = results["lag_readouts"]
    readouts["band_suppression"] = {
        "none": readouts["band_suppression"]["none"],
        **{name: {"margin_nats": 0.1, "margin_interval": _interval(0.1),
                  "band_anchors": 40.0, "band_channels": 400.0} for name in names},
        "all": readouts["band_suppression"]["all"],
    }
    readouts["band_edges"] = edges
    readouts["exposure"]["per_lag_anchors"] = [40.0] * n_lags
    readouts["exposure"]["per_lag_channels"] = [200.0] * n_lags
    for key, values in readouts["lag_profile"]["latent"].items():
        readouts["lag_profile"]["latent"][key] = list(values)[:1] * n_lags
    readouts["lag_profile"]["predictive"]["margin_nats"] = _curve([0.1] * n_lags)
    readouts["lag_axis"]["n_lags"] = n_lags

    assert figures._declared_bands(results) == names
    figure = figures.build_lag_profile_figure(results)
    try:
        lengths = {len(line.get_xdata()) for ax in figure.axes for line in ax.get_lines()}
        assert n_lags in lengths
        assert not any(length > n_lags for length in lengths)
    finally:
        plt.close(figure)
    axes = plt.figure().add_subplot(111)
    try:
        figures._shade_bands(axes, results, label=True)
        assert [text.get_text() for text in axes.texts] == names
    finally:
        plt.close(axes.figure)


def test_a_missing_band_margin_is_drawn_as_not_measured() -> None:
    """A band with nothing to remove stays on the figure, marked, rather than vanishing."""
    figure = figures.build_band_figure(candidate_summary())
    try:
        texts = [text.get_text() for ax in figure.axes for text in ax.texts]
        assert "not measured" in texts
    finally:
        plt.close(figure)


# =============================================================================
# The acceptance record
# =============================================================================
def acceptance_record() -> Dict[str, Any]:
    """A record with the blocks the protocol assembles, at the shape it writes them.

    Returns:
        The record.
    """
    analysis = {
        "arms": {
            "candidate": {"training_seeds": ["1", "2", "3"], "n_training_seeds": 3,
                          "meets_minimum": True, "runs": []},
            "target_only": {"training_seeds": ["1"], "n_training_seeds": 1,
                            "meets_minimum": False, "runs": []},
        },
        "primary_comparisons": {
            "the_pathway_entire": {"left_arm": "candidate", "right_arm": "target_only",
                                   "column": "nll_full", "status": "READ",
                                   "difference_nats": _interval(-0.8),
                                   "isolates": "the source pathway and its budget."},
            "the_variance_update": {"left_arm": "candidate", "right_arm": "mean_only",
                                    "status": "NO_EVIDENCE", "detail": "no run"},
        },
        "per_arm": {
            "candidate": {"n_training_seeds": 3, "internal_gap_nats": _interval(-1.0),
                          "divergence_per_anchor": _interval(0.2),
                          "against_reference": {"status": "READ",
                                                "full_minus_reference_nats": _interval(-0.5),
                                                "base_minus_reference_nats": _interval(0.1)}},
            "target_only": {"n_training_seeds": 1, "internal_gap_nats": _interval(0.0, 0.0),
                            "divergence_per_anchor": _interval(0.0, 0.0),
                            "against_reference": {"status": "NO_REFERENCE"}},
        },
        "source_controls": {
            "candidate": {"controls": {
                "silence": {"status": "READ", "margin_nats": _interval(-1.0)},
                "replace_zeros": {"status": "READ", "margin_nats": _interval(0.3)},
                "replace_constant": {"status": "NOT_RUN"},
                "permute": {"status": "READ", "margin_nats": _interval(0.1)},
            }},
        },
        "exploratory_bands": {
            "candidate": {"status": "READ", "searched_bands": ["near", "far"], "peak_band": "near",
                          "family_size": 2,
                          "bands": {"near": {"status": "READ", "margin_nats": _interval(0.4),
                                             "margin_nats_family_adjusted": _interval(0.4, 0.8)},
                                    "far": {"status": "READ", "margin_nats": _interval(0.1),
                                            "margin_nats_family_adjusted": _interval(0.1, 0.8)}}},
            "target_only": {"status": "NOT_SEARCHED", "detail": "no source"},
        },
        "monte_carlo_stability": {"candidate": {"gap_by_draw_count": {"8": -1.0, "32": -1.1},
                                                "declared_draw_counts": [8, 32, 128],
                                                "missing_draw_counts": [128], "range_nats": 0.1}},
        "mixture_calibration": {"candidate": {"full": {"pit_mean": 0.5, "coverage": {"0.5": 0.49}}}},
        "latent_probes": {"candidate": {"status": "READ", "r2": {"prior_mean": 0.2}}},
        "verdicts": [{"name": "training_seeds", "status": "PASS", "detail": "ok"}],
        "n_runs": 4,
    }
    return {
        "plan": {"path": "plan.yaml", "digest": "abc", "declared_on": "2026-09-09", "revision": 1,
                 "protocol": {"minimum_training_seeds": 3}},
        "runs": [{"directory": "/runs/a", "arm": "candidate", "training_seed": 1, "eval_seed": 42,
                  "num_mc_samples": 32, "split_label": "/data/test", "n_recordings": 5}],
        "reference": {"directory": "/runs/ref"},
        "selection": analysis,
        "verdicts": [{"name": "confirmation_partition", "status": "INCONCLUSIVE",
                      "detail": "no confirmation runs"}],
        "failed": [],
        "passed": True,
    }


@pytest.mark.parametrize(
    "builder",
    [figures.build_acceptance_comparisons_figure, figures.build_acceptance_arms_figure,
     figures.build_acceptance_bands_figure],
    ids=lambda fn: fn.__name__,
)
def test_every_acceptance_builder_draws_the_record_and_an_empty_one(builder) -> None:
    """The record as the protocol writes it, and an empty one, both draw.

    Args:
        builder: The builder.
    """
    for record in (acceptance_record(), {}):
        figure = builder(record)
        try:
            assert figure.axes
        finally:
            plt.close(figure)


def test_the_acceptance_figures_are_written_under_the_listed_names(tmp_path: Path) -> None:
    """The names are the record's manifest, and a reader finds each under it."""
    figures.configure_figure_style()
    written = figures.render_acceptance_figures(acceptance_record(), tmp_path)

    assert set(written) == set(figures.ACCEPTANCE_FIGURES)
    for path in written.values():
        assert Path(path).is_file()


def test_the_markdown_report_reads_the_record_rather_than_recomputing_it() -> None:
    """Every number in the document is the record's; the comparison and the verdicts appear."""
    record = acceptance_record()
    text = acceptance.report_markdown(record)

    assert "# Acceptance record" in text
    assert "the_pathway_entire" in text
    assert "-0.8 [-1.3, -0.3]" in text
    assert "NO_EVIDENCE" in text
    assert "confirmation_partition" in text
    assert "peak" in text
    # A control that did not run is reported as such rather than as a blank cell.
    assert "NOT_RUN" in text
    # The document is valid text for a file, round-tripping through JSON like the record does.
    assert acceptance.report_markdown(json.loads(json.dumps(record))) == text
