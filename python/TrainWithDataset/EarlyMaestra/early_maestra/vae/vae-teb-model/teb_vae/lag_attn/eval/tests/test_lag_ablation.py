r"""Tests for the lag-band ablation.

Every meaningful test here runs on a ``perturb_full_pathway`` model, and that is load-bearing
rather than cautious. ``_zero_init_delta_heads`` zeroes ``residual_decoder.mean_head``, so on an
untouched model ``mu_full`` equals ``mu_base`` for every band and the ablation is *bit-identical*
across all of them -- every delta is exactly zero, every bar has zero height, and a test written
without the fixture passes on an ablation that detects nothing whatsoever.

Deliberately not re-asserted: that ``lag_band_mask=None`` is a bit-exact no-op. That is a model
contract and ``tests/test_lag_band_mask.py`` already pins it with ``torch.equal``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn.eval import figures, masks
from teb_vae.lag_attn.eval.analyses import lag_ablation as lag_ablation_analysis


def _run(runner, loader, eval_config, output_dir):
    """Run the analysis and return ``(summary, per-band frame)``."""
    torch.manual_seed(11)
    summary = lag_ablation_analysis.run_lag_ablation_analysis(
        runner, loader, eval_config=eval_config, output_dir=output_dir, probe=None
    )
    directory = Path(output_dir) / lag_ablation_analysis.ANALYSIS_DIRNAME
    return summary, pd.read_csv(directory / "per_band.csv")


# ---------------------------------------------------------------------------
# The common support
# ---------------------------------------------------------------------------
def test_every_band_and_the_baseline_are_scored_on_one_identical_support(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, perturb_full_pathway
) -> None:
    """A per-band support would confound the ablation's effect with the anchors it was scored on."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_full_pathway(runner.model)
    summary, frame = _run(
        runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "support"
    )

    assert frame["anchors_scored"].nunique() == 1
    assert int(frame["anchors_scored"].iloc[0]) == summary["anchors_scored"]
    # The baseline is in the table and shares the support, or its deltas compare two anchor sets.
    assert lag_ablation_analysis.BASELINE_NAME in set(frame["band"])

    bands = tiny_eval_config["eval_config"]["bands"]
    expected = masks.common_scoring_start(runner.model, bands, int(runner.model.sequence_length))
    assert summary["common_scoring_start"] == expected


def test_the_excluded_anchor_counts_are_recorded_per_band(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, perturb_full_pathway
) -> None:
    """A band that forces every other band off the front of the recording must say so."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_full_pathway(runner.model)
    summary, frame = _run(
        runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "excl"
    )

    assert {"dead_before", "anchors_excluded", "anchors_scored"} <= set(frame.columns)
    ablated = frame[frame["band"] != lag_ablation_analysis.BASELINE_NAME]
    assert bool((ablated["anchors_excluded"] >= 0).all())
    # The fixture ships lag_0_2 and lag_3_8, so lag_3_8 sets the common start at 3 and lag_0_2
    # gives up the anchors between the warm-up and there.
    for name, record in summary["per_band"].items():
        assert record["dead_before"] == int(
            tiny_eval_config["eval_config"]["bands"][name][0]
        )


def test_a_band_beyond_the_anchor_range_raises_rather_than_scoring_nothing(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """An empty support would emit a table of NaN that reads as a broken analysis."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    # T = 300 and H_d = 4 at the tiny geometry, but max_lag = 8, so no configurable band can
    # exceed the anchor range -- shrink the model's own sequence length to force the condition.
    runner.model.sequence_length = 6
    config = dict(tiny_eval_config["eval_config"], bands={"far": (5, 8)})
    with pytest.raises(ValueError, match="common scoring support is empty"):
        _run(runner, tiny_loader, config, tmp_path / "empty")


def test_no_bands_configured_raises(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    config = dict(tiny_eval_config["eval_config"], bands={})
    with pytest.raises(ValueError, match="no lag bands are configured"):
        _run(runner, tiny_loader, config, tmp_path / "none")


def test_too_many_bands_raises_rather_than_running_for_a_day(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """One forward per band per batch, on top of the attention window's dense clone."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    bands = {f"b{index}": (index, index) for index in range(lag_ablation_analysis.MAX_BANDS + 1)}
    config = dict(tiny_eval_config["eval_config"], bands=bands)
    with pytest.raises(ValueError, match="above the bound"):
        _run(runner, tiny_loader, config, tmp_path / "many")


# ---------------------------------------------------------------------------
# The property that matters: the bands actually differ
# ---------------------------------------------------------------------------
def test_two_bands_produce_different_numbers(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, perturb_full_pathway
) -> None:
    """The whole point. On an untouched model every band is bit-identical and this cannot fail.

    ``mu_full == mu_base`` at initialisation whatever the latent, so removing lags changes the
    forecast by exactly nothing and a test without the full-pathway fixture would pass on an
    ablation that detects nothing.
    """
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_full_pathway(runner.model)
    _, frame = _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "differ")

    ablated = frame[frame["band"] != lag_ablation_analysis.BASELINE_NAME]
    assert len(ablated) == 2
    mses = ablated["feat_mse"].to_numpy(dtype=np.float64)
    assert not np.allclose(mses[0], mses[1]), (
        "the two bands produced identical numbers, so the ablation is not discriminating"
    )
    klds = ablated["kld"].to_numpy(dtype=np.float64)
    assert not np.allclose(klds[0], klds[1])


def test_an_untouched_model_is_bit_identical_across_bands(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """The negative control that makes the previous test's fixture requirement explicit."""
    runner = make_eval_runner(perturb=False, output_dir=tmp_path / "runner")
    _, frame = _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "flat")

    ablated = frame[frame["band"] != lag_ablation_analysis.BASELINE_NAME]
    assert ablated["feat_mse_delta"].to_numpy() == pytest.approx(0.0, abs=1e-9)


def test_the_per_band_kl_is_recomputed_not_read_from_kld_raw(
    make_eval_runner, tiny_loader, tmp_path, perturb_full_pathway
) -> None:
    r"""``kld_raw`` reduces over the model's band-unaware support, which includes dead anchors.

    At a dead anchor the ablation drives the attended source to zero, which under a
    head-structured posterior still produces a non-zero delta against the prior -- so a long-lag
    band's ``kld_raw`` folds in anchors the ablation never actually ablated. The two numbers must
    therefore differ, and the analysis must report the recomputed one.
    """
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_full_pathway(runner.model)

    num_lags = int(runner.num_lags)
    band = masks.lag_band_keep_mask((3, num_lags - 1), num_lags)
    batch = runner.to_device(next(iter(tiny_loader)))

    with runner.inference_mode():
        outputs = runner.forward(batch, lag_band_mask=band)
        losses = runner.compute_loss(batch, outputs)
    scored = lag_ablation_analysis._score_one(runner, batch, band, scoring_start=3)

    assert float(losses["kld_raw"]) > 0.0, "the perturbation did not make the KL nonzero"
    assert scored["kld"][0] != pytest.approx(float(losses["kld_raw"]), rel=1e-6), (
        "the recomputed KL equals kld_raw, so the common-support restriction is not applied"
    )


# ---------------------------------------------------------------------------
# Micro-batching
# ---------------------------------------------------------------------------
def test_bands_are_compared_under_common_random_numbers(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, perturb_full_pathway
) -> None:
    r"""Two bands with identical keep-masks must produce *bit-identical* numbers.

    ``forward`` samples $z$, so without a shared RNG state each band would be scored under its
    own independent draw and the band-to-band difference -- the entire measurement -- would carry
    that sampling noise on top of the ablation's effect. On a band whose real effect is small the
    noise is the larger of the two, which is how an ablation comes to rank bands by their draws.

    Two names for one mask is the cleanest possible probe: the ablation is identical, so any
    difference at all is sampling.
    """
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_full_pathway(runner.model)
    config = dict(
        tiny_eval_config["eval_config"],
        bands={"twin_a": (2, 6), "twin_b": (2, 6), "other": (0, 3)},
    )
    _, frame = _run(runner, tiny_loader, config, tmp_path / "crn")

    indexed = frame.set_index("band")
    for column in ("feat_mse", "kld"):
        assert float(indexed.loc["twin_a", column]) == float(
            indexed.loc["twin_b", column]
        ), f"{column} differs between two identical bands, so the z draw is not shared"
    # And the probe is not vacuous: a genuinely different band does move.
    assert float(indexed.loc["other", "feat_mse"]) != float(indexed.loc["twin_a", "feat_mse"])


def test_the_ablation_batch_size_changes_nothing_material(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, perturb_full_pathway
) -> None:
    r"""It bounds peak memory; it is not a knob that changes what is measured.

    Not asserted bit-exactly, and the reason is worth stating: ``forward`` samples $z$ and the
    noise tensor's shape follows the batch, so splitting a batch changes which draws are made.
    The model has no BatchNorm, so nothing else about it is batch-size dependent -- what remains
    is sampling noise, which is small and which a pooled accumulation carrying numerator and
    denominator separately does not amplify. A per-batch mean-of-means would drift much further.
    """
    results = []
    for label, size in (("whole", None), ("micro", 1)):
        runner = make_eval_runner(output_dir=tmp_path / f"runner_{label}")
        perturb_full_pathway(runner.model)
        config = dict(tiny_eval_config["eval_config"], ablation_batch_size=size)
        _, frame = _run(runner, tiny_loader, config, tmp_path / label)
        results.append(frame.sort_values("band").reset_index(drop=True))

    assert list(results[0]["band"]) == list(results[1]["band"])
    for column in ("feat_mse", "kld"):
        assert results[0][column].to_numpy() == pytest.approx(
            results[1][column].to_numpy(), rel=5e-3
        ), f"{column} moved further than the z-sampling noise explains"
    # The anchor accounting is exact regardless -- it is arithmetic on the geometry, not a mean.
    for column in ("anchors_scored", "anchors_excluded", "dead_before"):
        assert list(results[0][column]) == list(results[1][column])


def test_micro_batches_partition_the_batch(make_eval_runner, tiny_loader, tmp_path) -> None:
    """A slice that dropped or duplicated a sample would silently reweight every pooled mean."""
    batch = next(iter(tiny_loader))
    pieces = lag_ablation_analysis._micro_batches(batch, 1)
    assert len(pieces) == int(batch.fhr_st.shape[0])
    rebuilt = torch.cat([piece.fhr_st for piece in pieces], dim=0)
    assert torch.equal(rebuilt, batch.fhr_st)
    # ``guid`` is a list[str] and must survive slicing as one.
    assert [piece.guid[0] for piece in pieces] == list(batch.guid)


# ---------------------------------------------------------------------------
# S6-T06: the figures
# ---------------------------------------------------------------------------
def test_two_figures_are_written_with_dual_labels_and_exclusion_counts(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, monkeypatch, perturb_full_pathway
) -> None:
    captured: dict = {}
    original = figures.render_to_pdf

    def _capture(fig, path, **kwargs):
        ax = fig.axes[0]
        captured[Path(path).name] = {
            "labels": [label.get_text() for label in ax.get_xticklabels()],
            "title": ax.get_title(),
            "has_data": ax.has_data(),
            "texts": [text.get_text() for text in ax.texts],
        }
        return original(fig, path, **kwargs)

    monkeypatch.setattr(figures, "render_to_pdf", _capture)
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_full_pathway(runner.model)
    summary, _ = _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "figs")

    assert set(captured) == {"forecast_degradation.pdf", "kl_change.pdf"}
    for record in captured.values():
        assert record["has_data"]
        labels = " ".join(record["labels"])
        assert "$\\ell$" in labels, "no model-lag labelling"
        assert " s" in labels, "no physical-second labelling"
        assert "anchors given up" in labels, "exclusion counts are not visible on the figure"
    assert len(summary["figures"]) == 2

    # The mask KEEPS the named band (nets/model.py combines it as ``validity & band``), so a
    # caption phrased as removal inverts the ranking: it would read the band that alone
    # reproduces the forecast *worst* as the one that mattered *most*. The title is the only
    # thing carrying that convention onto the page, and it goes into papers.
    for name, record in captured.items():
        title = record["title"].lower()
        assert "kept" in title, f"{name} title does not state that the band is kept: {title!r}"
        assert "remov" not in title, (
            f"{name} title describes removal, but lag_band_mask keeps: {title!r}"
        )


def test_the_summary_names_both_ends_under_keep_semantics(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, perturb_full_pathway
) -> None:
    """A keep-mask has no "most damaging" band, so the summary must not claim one.

    Under ``validity & band`` a large ``feat_mse_delta`` means the band alone was *insufficient*.
    Reporting that as the most damaging band inverts the pipeline's central causal claim -- which
    lag range carries the UP influence -- so both ends are named explicitly instead.
    """
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_full_pathway(runner.model)
    summary, _ = _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "ends")

    assert "most_damaging_band" not in summary, "the inverted key is back"
    assert "most_sufficient_band" in summary and "least_sufficient_band" in summary
    assert "keeps only the named band" in summary["semantics"].lower()

    per_band = summary["per_band"]
    finite = {
        name: row["feat_mse_delta"]
        for name, row in per_band.items()
        if np.isfinite(row["feat_mse_delta"])
    }
    if finite:
        assert summary["most_sufficient_band"] == min(finite, key=lambda name: finite[name])
        assert summary["least_sufficient_band"] == max(finite, key=lambda name: finite[name])


def test_suppressing_one_bands_contribution_moves_that_bands_bar(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, monkeypatch, perturb_full_pathway
) -> None:
    """The sabotage case, required because a bare structural assertion cannot discriminate.

    One band's forward is degraded and no other's, so exactly one bar must respond. Without this
    the figure test passes on a chart that plots the same number four times.
    """
    captured: dict = {}
    original = figures.render_to_pdf

    def _capture(fig, path, **kwargs):
        if Path(path).name == "forecast_degradation.pdf":
            captured["heights"] = [patch.get_height() for patch in fig.axes[0].patches]
        return original(fig, path, **kwargs)

    monkeypatch.setattr(figures, "render_to_pdf", _capture)
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_full_pathway(runner.model)

    num_lags = int(runner.num_lags)
    sabotaged = masks.lag_band_keep_mask((3, 8), num_lags)
    inner = runner.forward

    def _forward(batch, *, lag_band_mask=None):
        outputs = inner(batch, lag_band_mask=lag_band_mask)
        if lag_band_mask is not None and torch.equal(lag_band_mask.cpu(), sabotaged.cpu()):
            # A large forecast error for this band alone.
            outputs = dict(outputs)
            outputs["mu_base"] = outputs["mu_base"] + 50.0
            outputs["mu_full"] = outputs["mu_full"] + 50.0
        return outputs

    runner.forward = _forward
    _, frame = _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "sabotage")

    ablated = frame[frame["band"] != lag_ablation_analysis.BASELINE_NAME].sort_values("lag_lo")
    deltas = dict(zip(ablated["band"], ablated["feat_mse_delta"]))
    assert deltas["lag_3_8"] > 100.0, "the sabotaged band's bar did not respond"
    assert abs(deltas["lag_0_2"]) < 1.0, "an untouched band's bar moved"

    heights = captured["heights"]
    assert max(heights) == pytest.approx(deltas["lag_3_8"], rel=1e-6)
