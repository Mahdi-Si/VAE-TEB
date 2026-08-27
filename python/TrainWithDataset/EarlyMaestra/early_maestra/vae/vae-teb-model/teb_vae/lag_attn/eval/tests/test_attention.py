r"""Tests for the attention analysis.

The structural assertions -- files written, columns present, panels drawn -- are necessary and
not sufficient. On a real checkpoint the attention is whatever it is, so a test that only checked
"``argmax_lag`` is an integer in $[0, L)$" would pass on an analysis that reported a constant.
The sabotage case is what makes the suite non-vacuous: attention is forced to a known lag and the
reported argmax is required to be that lag.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn.eval import figures, metrics
from teb_vae.lag_attn.eval.analyses import attention as attention_analysis

#: The tiny fixture's shard holds four samples from one file.
PROBE = {"n_samples": 4, "source_files": ["tiny_shard.hdf5"] * 4}


def _run(runner, loader, eval_config, output_dir):
    """Run the analysis and return ``(summary, per-sample frame)``."""
    torch.manual_seed(11)
    summary = attention_analysis.run_attention_analysis(
        runner, loader, eval_config=eval_config, output_dir=output_dir, probe=PROBE
    )
    directory = Path(output_dir) / attention_analysis.ANALYSIS_DIRNAME
    return summary, pd.read_csv(directory / "per_sample.csv")


def _concentrate_attention_at(runner, lag: int) -> None:
    r"""Force every attention row to a one-hot at ``lag``, in place on the runner.

    Wraps ``forward`` rather than reaching into the attention module: the analysis consumes the
    forward dict, so this substitutes exactly what it reads and nothing else. Anchors at
    $t < \ell$ are left alone -- lag $\ell$ is not causally available there, and forcing mass
    onto it would fabricate a row the model could never produce.
    """
    original = runner.forward

    def _forward(batch, **kwargs):
        outputs = original(batch, **kwargs)
        alpha = torch.zeros_like(outputs["attn_weights"])
        alpha[:, lag:, :, lag] = 1.0
        for step in range(min(lag, alpha.shape[1])):
            alpha[:, step, :, 0] = 1.0
        outputs["attn_weights"] = alpha
        return outputs

    runner.forward = _forward


# ---------------------------------------------------------------------------
# Outputs and schema
# ---------------------------------------------------------------------------
def test_the_three_tables_are_written(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    summary, frame = _run(
        runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "tables"
    )
    directory = tmp_path / "tables" / attention_analysis.ANALYSIS_DIRNAME

    for name in ("per_sample.csv", "mass_by_lag.csv", "head_entropy.csv"):
        assert (directory / name).is_file(), f"{name} missing"
    assert summary["n_samples"] == 4
    assert {"argmax_lag", "lag_seconds_physical", "entropy_mean", "head_diversity"} <= set(
        frame.columns
    )


def test_the_lag_column_pair_holds_its_arithmetic_relationship(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    r"""$\mathrm{seconds} = s\ell - \Delta_{UP}$, with $\Delta_{UP}$ from the config.

    Emitted as a pair rather than derived downstream, so this is the assertion that keeps the two
    columns describing the same lag.
    """
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    config = tiny_eval_config["eval_config"]
    _, frame = _run(runner, tiny_loader, config, tmp_path / "pair")

    # `+ up_shift_secs`, not `-`: the dataset ADVANCED the UP trace by 20 s, so a peak at lag l is
    # a lead of 4l - 20 s in the original recording. This asserted the subtraction until the sign
    # was corrected -- the same arithmetic the function had, which is why it could not catch it.
    expected = metrics.STEP_SECONDS * frame["argmax_lag"].to_numpy() + float(
        config["up_shift_secs"]
    )
    assert frame["lag_seconds_physical"].to_numpy() == pytest.approx(expected)
    # The fixture ships up_shift_secs=-20, so the pair must not be a plain 4*lag axis.
    assert not np.allclose(
        frame["lag_seconds_physical"].to_numpy(),
        metrics.STEP_SECONDS * frame["argmax_lag"].to_numpy(),
    )


def test_the_mass_table_carries_one_row_per_sample_and_lag(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "mass")

    mass = pd.read_csv(
        tmp_path / "mass" / attention_analysis.ANALYSIS_DIRNAME / "mass_by_lag.csv"
    )
    num_lags = int(runner.model.lag_attn.L)
    assert set(mass["lag"]) == set(range(num_lags))
    assert len(mass) == 4 * num_lags
    # Each sample's profile is a distribution over the support, so it sums to 1.
    assert mass.groupby("sample_index")["mass"].sum().to_numpy() == pytest.approx(1.0, abs=1e-4)


def test_head_entropy_has_one_row_per_head(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "heads")

    heads = pd.read_csv(
        tmp_path / "heads" / attention_analysis.ANALYSIS_DIRNAME / "head_entropy.csv"
    )
    assert set(heads["head"]) == set(range(int(runner.model.lag_attn.num_heads)))


# ---------------------------------------------------------------------------
# The sabotage case
# ---------------------------------------------------------------------------
def test_attention_concentrated_at_a_known_lag_is_reported_at_that_lag(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """The assertion that makes every other one here mean something.

    Without it the suite passes on an analysis that reports a constant, or one that reads the lag
    axis reversed -- the attention module stores its window oldest-first internally and flips it
    back, so an off-by-a-flip is a live failure mode rather than a hypothetical one.
    """
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    target_lag = 5
    _concentrate_attention_at(runner, target_lag)

    config = tiny_eval_config["eval_config"]
    summary, frame = _run(runner, tiny_loader, config, tmp_path / "sabotage")

    assert set(frame["argmax_lag"]) == {target_lag}
    assert frame["lag_seconds_physical"].to_numpy() == pytest.approx(
        metrics.STEP_SECONDS * target_lag + float(config["up_shift_secs"])
    )
    # A one-hot row has zero entropy: the peak is real, not the argmax of a flat profile.
    assert summary["mean_entropy_nats"] == pytest.approx(0.0, abs=1e-6)
    # All heads forced to the same lag, so there is nothing to distinguish them.
    assert summary["mean_head_diversity"] == pytest.approx(0.0, abs=1e-6)


def _flatten_attention_over_valid_lags(runner) -> None:
    r"""Force every row uniform over its $\min(t+1, L)$ causally available lags, in place.

    The degenerate case with no lag structure at all -- and the only row shape that attains the
    entropy ceiling exactly, which is what makes it the probe for whether the reported ceiling is
    the right one. Mass is never placed on $\ell > t$: that lag does not exist at anchor $t$, and
    a row that used it would be one the model can never produce.
    """
    original = runner.forward

    def _flat(batch, **kwargs):
        outputs = original(batch, **kwargs)
        alpha = torch.zeros_like(outputs["attn_weights"])
        num_lags = alpha.shape[-1]
        for step in range(alpha.shape[1]):
            available = min(step + 1, num_lags)
            alpha[:, step, :, :available] = 1.0 / available
        outputs["attn_weights"] = alpha
        return outputs

    runner.forward = _flat


def test_a_flat_attention_reports_entropy_at_its_attainable_ceiling(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    r"""The other half of the sabotage: an argmax that must not be read as a finding.

    Uniform-over-available attention is the case the ceiling exists to catch, so it must report a
    ratio of $1$ -- to floating-point slack, not to a percent. Asserting only against $\log L$
    would pass on a ceiling nothing can reach: at this fixture's $L = 9$ the two bounds differ by
    just $0.5\%$, which any loose tolerance hides, and at the production $L = 91$ the same gap is
    $2.5\%$ and turns the downstream uniformity check into dead code.
    """
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    _flatten_attention_over_valid_lags(runner)
    summary, frame = _run(
        runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "flat"
    )

    attainable = summary["mean_attainable_entropy_nats"]
    assert summary["mean_entropy_nats"] == pytest.approx(attainable, rel=1e-6)
    # And the two ceilings are genuinely different numbers, or the assertion above proves nothing.
    assert attainable < summary["max_possible_entropy_nats"]
    assert frame["attainable_entropy"].to_numpy() == pytest.approx(
        frame["entropy_mean"].to_numpy(), rel=1e-6
    )


def test_the_attainable_ceiling_is_reachable_at_the_production_lag_geometry() -> None:
    r"""$L = 91$ with a warm-up of $30$: the geometry the shipped config runs, checked directly.

    The fixture's $L = 9$ cannot exercise this -- only $6$ of its $\sim 294$ anchors are lag-
    starved, so both ceilings agree to $0.5\%$. At the production geometry $60$ of the $240$
    supported anchors are, the flat case sits at $0.975$ of $\log L$, and ``check_argmax_lag``'s
    ``entropy >= 0.99 * ceiling`` branch can never fire against $\log L$: the systematic gap is
    $24\times$ the margin that was meant to absorb floating-point slack.

    Driven through :func:`~teb_vae.lag_attn.eval.metrics.attention_diagnostics` rather than the
    analysis, because the committed shard fixes $L$ and no eval config can move it.
    """
    seq_len, num_lags, warmup, horizon = 300, 91, 30, 30
    alpha = torch.zeros(1, seq_len, 2, num_lags)
    for step in range(seq_len):
        available = min(step + 1, num_lags)
        alpha[:, step, :, :available] = 1.0 / available

    # The support the model's ``kld_support='anchor'`` leaves: warm-up dropped at the head, the
    # forecast horizon at the tail. 240 anchors, of which t < 90 are lag-starved.
    support = torch.zeros(1, seq_len)
    support[:, warmup:seq_len - horizon] = 1.0

    diagnostics = metrics.attention_diagnostics(alpha, support)
    entropy = float(diagnostics["entropy_mean"].nanmean())
    attainable = float(diagnostics["attainable_entropy"][0])

    assert entropy == pytest.approx(attainable, rel=1e-6)
    assert entropy == pytest.approx(4.3977, abs=1e-3)
    # The shortfall against log L that the old ceiling reported as lag concentration.
    assert entropy / float(np.log(num_lags)) == pytest.approx(0.9749, abs=1e-3)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def test_the_summary_figure_stacks_three_panels_and_carries_a_second_lag_axis(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, monkeypatch
) -> None:
    """The dual axis is the figure's whole contract: model lag and physical seconds together."""
    captured: dict = {}
    original = figures.render_figure

    def _capture(fig, path, **kwargs):
        if Path(path).name == "attention":
            captured["titles"] = [ax.get_title() for ax in fig.axes if ax.get_title()]
            captured["has_data"] = [ax.has_data() for ax in fig.axes if ax.get_title()]
            # A secondary axis is a *child* of the axes that created it, not a figure-level
            # axes, so it never appears in ``fig.axes``.
            captured["secondary_x"] = [
                child.get_xlabel()
                for ax in fig.axes
                for child in ax.child_axes
                if child.get_xlabel()
            ]
            captured["suptitle"] = fig._suptitle.get_text() if fig._suptitle else ""
        return original(fig, path, **kwargs)

    monkeypatch.setattr(figures, "render_figure", _capture)
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "figure")

    assert len(captured["titles"]) == 3
    assert captured["titles"][0].startswith("Per-sample argmax lag")
    assert captured["titles"][1].startswith("Attention mass by lag")
    assert captured["titles"][2].startswith("Head diversity")
    assert all(captured["has_data"])
    assert captured["secondary_x"], "the lag panel has no physical-second axis"
    # The offset has never been applied anywhere in this repository, so the figure states it.
    assert "-20" in captured["suptitle"]


def test_the_heatmap_figure_draws_one_row_per_retained_sample(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, monkeypatch
) -> None:
    """The heatmap cap is the `samples` one, not the much larger `attention` retention cap."""
    captured: dict = {}
    original = figures.render_figure

    def _capture(fig, path, **kwargs):
        if Path(path).name == "attention_heatmaps":
            captured["titles"] = [ax.get_title() for ax in fig.axes if ax.get_title()]
            captured["secondary_y"] = [
                child.get_ylabel()
                for ax in fig.axes
                for child in ax.child_axes
                if child.get_ylabel()
            ]
        return original(fig, path, **kwargs)

    monkeypatch.setattr(figures, "render_figure", _capture)
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    summary, _ = _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "heat")

    expected_rows = int(tiny_eval_config["eval_config"]["caps"]["samples"])
    assert len(captured["titles"]) == expected_rows
    assert all(title.startswith("Head-averaged attention") for title in captured["titles"])
    assert captured["secondary_y"] == ["Lag (s)"] * expected_rows, (
        "every heatmap row must carry the physical-second axis"
    )
    assert len(summary["figures"]) == 2


def test_both_figures_are_written_as_pdfs(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    summary, _ = _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "pdfs")
    for path in summary["figures"]:
        assert Path(path).suffix == ".pdf" and Path(path).stat().st_size > 0
