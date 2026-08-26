r"""Tests for the latent-health analysis.

Almost everything here runs on a perturbed model, and that is not incidental. At initialisation
the posterior *equals* the prior, so every KL is exactly zero, every bar in the per-dimension
figure has zero height, and every assertion of the form "the KL is a number in a plausible range"
passes on a model that is entirely wrong. ``ax.has_data()`` passes on that figure too.

The masking test is the load-bearing one for S5-T02: the two variants of each diagnostic must
*differ* on a batch with a partially zero ``weight``, or the masking is not being applied and the
second column is a copy of the first under a different name.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn.eval import figures, masks, metrics
from teb_vae.lag_attn.eval.analyses import latent as latent_analysis
from teb_vae.lag_attn.eval.tests.conftest import SHIPPED_KWARGS
from teb_vae.lag_attn.nets.model import SeqVaeLagAttn
from teb_vae.lag_attn.tests.conftest import SEQ_LEN, make_stub_batch

PROBE = {"n_samples": 4, "source_files": ["tiny_shard.hdf5"] * 4}


def _run(runner, loader, eval_config, output_dir):
    """Run the analysis and return ``(summary, per-sample frame, per-dim frame)``."""
    torch.manual_seed(11)
    summary = latent_analysis.run_latent_analysis(
        runner, loader, eval_config=eval_config, output_dir=output_dir, probe=PROBE
    )
    directory = Path(output_dir) / latent_analysis.ANALYSIS_DIRNAME
    return (
        summary,
        pd.read_csv(directory / "per_sample.csv"),
        pd.read_csv(directory / "per_dim.csv"),
    )


# ---------------------------------------------------------------------------
# S5-T01: the tables
# ---------------------------------------------------------------------------
def test_per_dim_has_one_row_per_latent_dimension(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, perturb_posterior
) -> None:
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_posterior(runner.model)
    summary, _, per_dim = _run(
        runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "dims"
    )

    d_z = int(runner.model.d_z)
    assert len(per_dim) == d_z == summary["d_z"]
    assert list(per_dim["dim"]) == list(range(d_z))
    assert {"kld_mean", "active", "kld_p25", "kld_median", "kld_p75"} <= set(per_dim.columns)


def test_the_active_fraction_matches_the_forward_dicts_own_scalar(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, perturb_posterior
) -> None:
    """The raw reading is a passthrough, so it must equal what the model itself computed.

    On a perturbed model, because on an untouched one both are $0$ and the comparison is vacuous.
    """
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_posterior(runner.model)

    # Driven over a *single* batch so the pass-through can be asserted as an equality. Over the
    # tiny loader's two batches the summary averages them, and the previous form of this test
    # bound the single-batch scalar and then never compared it -- all three of its assertions
    # were range checks on $[0, 1]$, so it passed even when the raw reading was silently sourced
    # from the masked recomputation instead of the model's own forward scalar.
    batch = runner.to_device(next(iter(tiny_loader)))
    summary, _, _ = _run(
        runner, [batch], tiny_eval_config["eval_config"], tmp_path / "frac"
    )

    with runner.inference_mode():
        outputs = runner.forward(batch)
    expected = float(metrics.latent_health(outputs)["kld_active_frac"])

    assert summary["diagnostics"]["kld_active_frac_raw"] == pytest.approx(expected, rel=1e-6), (
        "the raw active fraction is not the model's own forward scalar"
    )
    # The perturbation must actually open the bottleneck, or the equality above is 0 == 0.
    assert expected > 0.0, "the posterior perturbation did not activate any dimension"


def test_per_sample_carries_the_aggregates_and_not_the_flattened_columns(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, perturb_posterior
) -> None:
    """A batch-level diagnostic is constant within a batch and has no per-sample meaning."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_posterior(runner.model)
    _, frame, _ = _run(
        runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "cols"
    )

    assert {"kld_mean", "kld_sum", "kld_dim_l2", "posterior_drift"} <= set(frame.columns)
    assert not [name for name in frame.columns if name.startswith("d") and name[1:].isdigit()]
    assert not [name for name in frame.columns if name.startswith("t") and name[1:].isdigit()]
    assert not [name for name in frame.columns if name.endswith("_raw")]


def test_a_saturating_bound_is_flagged(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    r"""A binding $\tanh$ bound is a mis-set hyperparameter whose gradient has vanished.

    On ``delta_mu_scale`` it also caps every transfer-entropy number the run reports, so it must
    be surfaced rather than left in a table.
    """
    # A tiny delta_mu_scale against a perturbed posterior drives the tanh straight into its bound.
    runner = make_eval_runner(
        dict(SHIPPED_KWARGS, delta_mu_scale=1e-3), output_dir=tmp_path / "runner"
    )
    summary, _, _ = _run(
        runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "saturated"
    )
    assert summary["saturation_flagged"]["delta_mu_sat_frac_masked"] is True

    healthy = make_eval_runner(perturb=False, output_dir=tmp_path / "runner2")
    clean, _, _ = _run(
        healthy, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "clean"
    )
    assert clean["saturation_flagged"]["delta_mu_sat_frac_masked"] is False


# ---------------------------------------------------------------------------
# S5-T02: the masked variants
# ---------------------------------------------------------------------------
def _half_saturated_outputs(model):
    r"""Synthesise a forward dict whose saturation and KL live in the second half of the sequence.

    Hand-built rather than produced by a real forward, because the property under test is a
    property of the *masking* and a real forward gives no control over where the saturation
    falls. A perturbation large enough to saturate anything saturates everything, so both
    fractions land on $0$ or $1$ and no masking rule can be distinguished from any other.

    With unit variances the KL reduces to $\tfrac{1}{2}(\mu^q - \mu^p)^2$, so every quantity here
    has an answer that can be worked out on paper.

    Args:
        model: The model whose ``mu_scale`` and ``delta_mu_scale`` set the saturation bounds.

    Returns:
        A dict carrying the four tensors ``masked_latent_diagnostics`` reads, $(4, T, d_z)$.
    """
    d_z = int(model.d_z)
    mu_prior = torch.zeros(4, SEQ_LEN, d_z)
    delta = torch.zeros(4, SEQ_LEN, d_z)
    mu_prior[:, SEQ_LEN // 2:] = 0.999 * float(model.mu_scale)
    delta[:, SEQ_LEN // 2:] = 0.999 * float(model.delta_mu_scale)
    return {
        "mu_prior": mu_prior,
        "logvar_prior": torch.zeros(4, SEQ_LEN, d_z),
        "mu_post": mu_prior + delta,
        "logvar_post": torch.zeros(4, SEQ_LEN, d_z),
    }


def test_the_masked_diagnostics_honour_the_mask_they_are_given() -> None:
    """Three masks, three answers that can be worked out on paper.

    This is the assertion that the masked variant is a genuinely different computation from the
    model's own and not a copy of it under a second name.
    """
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**SHIPPED_KWARGS)
    outputs = _half_saturated_outputs(model)

    everywhere = torch.ones(4, SEQ_LEN)
    first_half = torch.zeros(4, SEQ_LEN)
    first_half[:, : SEQ_LEN // 2] = 1.0
    second_half = torch.zeros(4, SEQ_LEN)
    second_half[:, SEQ_LEN // 2:] = 1.0

    full = metrics.masked_latent_diagnostics(outputs, model, everywhere)
    early = metrics.masked_latent_diagnostics(outputs, model, first_half)
    late = metrics.masked_latent_diagnostics(outputs, model, second_half)

    # Saturation occupies exactly half the steps.
    assert full["mu_prior_sat_frac"] == pytest.approx(0.5)
    assert early["mu_prior_sat_frac"] == pytest.approx(0.0)
    assert late["mu_prior_sat_frac"] == pytest.approx(1.0)
    assert full["delta_mu_sat_frac"] == pytest.approx(0.5)
    assert early["delta_mu_sat_frac"] == pytest.approx(0.0)
    assert late["delta_mu_sat_frac"] == pytest.approx(1.0)

    # KL is 0.5 * (0.999 * delta_mu_scale)^2 on the late steps and 0 on the early ones, so the
    # per-dimension mean clears the threshold in every case but the early-only one.
    assert early["kld_active_frac"] == pytest.approx(0.0)
    assert late["kld_active_frac"] == pytest.approx(1.0)
    assert full["kld_active_frac"] == pytest.approx(1.0)


def test_the_masked_reading_differs_from_the_models_unmasked_formula() -> None:
    r"""The model's saturation fractions are a flat ``.mean()`` over every $(B, T, d_z)$ element.

    Zeroing the weight over the saturated half must move this pipeline's reading and leave the
    model's own untouched -- which is the entire reason both are reported.
    """
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**SHIPPED_KWARGS)
    outputs = _half_saturated_outputs(model)

    # The model's own formula, applied verbatim: no masking of any kind.
    unmasked = float(
        (outputs["mu_prior"].abs() >= 0.99 * float(model.mu_scale)).float().mean()
    )
    assert unmasked == pytest.approx(0.5)

    weight = torch.ones(4, SEQ_LEN)
    weight[:, SEQ_LEN // 2:] = 0.0
    masked = metrics.masked_latent_diagnostics(outputs, model, weight)

    assert masked["mu_prior_sat_frac"] == pytest.approx(0.0)
    assert masked["mu_prior_sat_frac"] != pytest.approx(unmasked, abs=1e-9)


def test_the_active_fraction_honours_the_weight_the_model_ignores() -> None:
    r"""``kld_active_frac`` respects the KL support but not ``weight``; this variant does both.

    Two of four recordings are marked entirely invalid, and they are the only two carrying any
    KL -- so a reading that ignored ``weight`` would report every dimension active.
    """
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**SHIPPED_KWARGS)
    d_z = int(model.d_z)
    mu_prior = torch.zeros(4, SEQ_LEN, d_z)
    mu_post = mu_prior.clone()
    mu_post[2:] += 2.0  # only recordings 2 and 3 have a posterior away from the prior
    outputs = {
        "mu_prior": mu_prior,
        "logvar_prior": torch.zeros(4, SEQ_LEN, d_z),
        "mu_post": mu_post,
        "logvar_post": torch.zeros(4, SEQ_LEN, d_z),
    }

    weight = torch.ones(4, SEQ_LEN)
    weight[2:] = 0.0

    without = metrics.masked_latent_diagnostics(
        outputs, model, masks.kld_mask(model, None, 4, SEQ_LEN)
    )
    with_weight = metrics.masked_latent_diagnostics(
        outputs, model, masks.kld_mask(model, weight, 4, SEQ_LEN)
    )
    assert without["kld_active_frac"] == pytest.approx(1.0)
    assert with_weight["kld_active_frac"] == pytest.approx(0.0)


def test_an_empty_mask_yields_nan_rather_than_zero() -> None:
    """Zero is a legitimate fraction, so it cannot double as "nothing was measured"."""
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**SHIPPED_KWARGS)
    batch = make_stub_batch(batch_size=2)
    with torch.no_grad():
        outputs = model(
            batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], dim=-1)
        )
    result = metrics.masked_latent_diagnostics(outputs, model, torch.zeros(2, SEQ_LEN))
    assert all(np.isnan(value) for value in result.values())


def test_the_summary_reports_both_readings_and_says_why_they_differ(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, perturb_posterior
) -> None:
    """A reader must not have to know the model's source to interpret two similar numbers."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_posterior(runner.model)
    summary, _, _ = _run(
        runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "both"
    )

    for stem in ("kld_active_frac", "mu_prior_sat_frac", "delta_mu_sat_frac"):
        assert f"{stem}_raw" in summary["diagnostics"]
        assert f"{stem}_masked" in summary["diagnostics"]
    assert "weight" in summary["diagnostics_note"]


# ---------------------------------------------------------------------------
# S5-T03: the figures
# ---------------------------------------------------------------------------
def test_three_figures_are_written(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, perturb_posterior
) -> None:
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_posterior(runner.model)
    summary, _, _ = _run(
        runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "figs"
    )

    assert len(summary["figures"]) == 3
    assert {Path(path).name for path in summary["figures"]} == {
        "per_dim_kl.pdf", "per_dim_violin.pdf", "kt_curve.pdf"
    }
    for path in summary["figures"]:
        assert Path(path).stat().st_size > 0


def test_the_threshold_line_sits_at_exactly_the_models_own_epsilon(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, monkeypatch, perturb_posterior
) -> None:
    r"""$10^{-2}$, imported from the model rather than restated, and the axis is symlog about it."""
    captured: dict = {}
    original = figures.render_figure

    def _capture(fig, path, **kwargs):
        if Path(path).name == "per_dim_kl":
            ax = fig.axes[0]
            captured["lines"] = [line.get_ydata()[0] for line in ax.get_lines()]
            captured["scale"] = ax.get_yscale()
            captured["title"] = ax.get_title()
            captured["has_data"] = ax.has_data()
        return original(fig, path, **kwargs)

    monkeypatch.setattr(figures, "render_figure", _capture)
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_posterior(runner.model)
    _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "line")

    assert metrics.KLD_ACTIVE_EPS == 1e-2
    assert captured["lines"] == pytest.approx([1e-2])
    assert captured["scale"] == "symlog"
    assert captured["has_data"]


def test_only_the_dimension_driven_above_the_threshold_is_coloured_active(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, monkeypatch
) -> None:
    """The sabotage case. On an untouched model every bar is zero height and this cannot fail.

    One dimension is driven far above the threshold and every other left at its collapsed value,
    so the bar colours are forced to a known pattern that a hardcoded palette would not produce.
    """
    captured: dict = {}
    original = figures.render_figure

    def _capture(fig, path, **kwargs):
        if Path(path).name == "per_dim_kl":
            bars = [patch for patch in fig.axes[0].patches]
            captured["colors"] = [patch.get_facecolor() for patch in bars]
            captured["heights"] = [patch.get_height() for patch in bars]
        return original(fig, path, **kwargs)

    monkeypatch.setattr(figures, "render_figure", _capture)
    runner = make_eval_runner(perturb=False, output_dir=tmp_path / "runner")

    # An untouched model has mu_post == mu_prior exactly, so shifting one posterior dimension is
    # the entire KL: dimension 3 becomes active and every other stays at exactly zero.
    live_dim = 3
    inner = runner.forward

    def _forward(batch, **kwargs):
        outputs = inner(batch, **kwargs)
        shifted = outputs["mu_post"].clone()
        shifted[:, :, live_dim] += 3.0
        outputs["mu_post"] = shifted
        return outputs

    runner.forward = _forward
    _, _, per_dim = _run(
        runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "sabotage"
    )

    assert bool(per_dim.loc[live_dim, "active"])
    assert not bool(per_dim.drop(index=live_dim)["active"].any())

    active_color = captured["colors"][live_dim]
    others = [captured["colors"][i] for i in range(len(captured["colors"])) if i != live_dim]
    assert all(color != active_color for color in others)
    assert captured["heights"][live_dim] > metrics.KLD_ACTIVE_EPS


def test_the_kt_curve_shades_both_ends_under_anchor_support(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, monkeypatch, perturb_posterior
) -> None:
    r"""The trailing $H_d$ steps are outside the support too, and that is the forgotten half.

    Left unshaded, the KL falling to zero there reads as the model losing interest late in the
    recording rather than as a window with no supervised target.
    """
    captured: dict = {}
    original = figures.render_figure

    def _capture(fig, path, **kwargs):
        if Path(path).name == "kt_curve":
            captured["spans"] = [
                patch.get_x() for patch in fig.axes[0].patches if patch.get_width() > 0
            ]
            captured["notes"] = [text.get_text() for text in fig.axes[0].texts]
        return original(fig, path, **kwargs)

    monkeypatch.setattr(figures, "render_figure", _capture)
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    assert str(runner.model.kld_support) == "anchor", "the shipped kwargs must use anchor support"
    perturb_posterior(runner.model)
    _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "kt")

    assert "warm-up" in captured["notes"]
    assert "no forecast window" in captured["notes"]
    assert len(captured["spans"]) == 2
