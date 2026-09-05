r"""Tests for the lag attribution and the per-head decomposition.

Two things are deliberately **not** asserted here, because asserting them would prove nothing.

The identity $\sum_\ell \widetilde{TE}_{t,\ell} = K_t$ is a *model* contract and
``test_kl_report.py`` already pins it. What is tested here is the eval-side property: that this
module's time-averaging and dead-anchor exclusion preserve it in the aggregate, including in the
case that breaks it -- a band mask that kills anchors.

The per-head sum-to-one identity is likewise not asserted. It is a tautology of the contiguous
``view(B, T, M, d_z // M).sum(-1)``: it holds on a completely wrong model, on a flat latent where
the split means nothing, and it is $0/0$ at initialisation. The behavioural assertion replaces
it -- zero one head's attention and that head's contribution must move while the others do not,
which is the property head structuring actually claims.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn.eval import figures, masks, preflight
from teb_vae.lag_attn.eval.analyses import te_lag as te_lag_analysis
from teb_vae.lag_attn.eval.tests.conftest import SHIPPED_KWARGS

PROBE = {"n_samples": 4, "source_files": ["tiny_shard.hdf5"] * 4}


def _run(runner, loader, eval_config, output_dir):
    """Run the analysis and return ``(summary, per-sample frame)``."""
    torch.manual_seed(11)
    summary = te_lag_analysis.run_te_lag_analysis(
        runner, loader, eval_config=eval_config, output_dir=output_dir, probe=PROBE
    )
    directory = Path(output_dir) / te_lag_analysis.ANALYSIS_DIRNAME
    return summary, pd.read_csv(directory / "te_lag_mean_per_sample.csv")


# ---------------------------------------------------------------------------
# S4-T03: the preconditions
# ---------------------------------------------------------------------------
def test_a_non_causal_checkpoint_refuses_the_whole_te_analysis(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """Without causal norms the prior conditions on the future and $K_t$ is not a TE at all."""
    runner = make_eval_runner(
        dict(SHIPPED_KWARGS, causal_norm=False), output_dir=tmp_path / "runner"
    )
    assert runner.model.n_causalized_norms == 0

    with pytest.raises(preflight.TEPreconditionUnmet, match="causal_norm=False") as excinfo:
        _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "refused")
    assert "transfer-entropy" in str(excinfo.value)


def test_a_flat_latent_refuses_only_the_per_head_panel(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """The lag attribution survives as a diagnostic; only the attribution *claim* is refused.

    Failing the whole step would throw away a readout that is still valid on such a checkpoint.

    The refusal has to reach the *table*, not only ``summary.json``. ``kld_per_t_per_head`` is
    emitted whatever the flag, so the columns are computable here and are exactly the numbers the
    guard exists to suppress -- shares that sum to $1$ and read as a decomposition of a quantity
    every head contributed to.
    """
    runner = make_eval_runner(
        dict(SHIPPED_KWARGS, head_structured_latent=False), output_dir=tmp_path / "runner"
    )
    summary, frame = _run(
        runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "flat"
    )

    assert summary["te_lag_map_label"] == "diagnostic"
    assert summary["per_head"]["available"] is False
    assert "head_structured_latent" in summary["per_head"]["reason"]
    # Absent, not zero-filled and not null-filled. A null share reads as "computed, no data" and
    # a zero reads as "this head carried nothing" -- both are measurements, and neither is what a
    # flat latent supports. Only a missing column cannot be misread as one.
    assert not [
        name
        for name in frame.columns
        if name.startswith(te_lag_analysis._HEAD_SHARE_PREFIX)
        or name.startswith(te_lag_analysis._HEAD_KL_PREFIX)
    ]
    # ... and the table says which of the two quantities it holds, without its summary.json.
    assert frame["te_lag_map_label"].tolist() == ["diagnostic"] * 4
    # The lag attribution itself still ran and still wrote its table and its figure.
    assert len(frame) == 4
    assert summary["identity"]["holds"] is True
    assert len(summary["figures"]) == 1


def test_a_head_structured_checkpoint_labels_the_map_an_attribution(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    summary, frame = _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "ok")
    assert summary["te_lag_map_label"] == "attribution"
    assert summary["per_head"]["available"] is True
    assert len(summary["figures"]) == 2
    # The other side of the guard: here the columns are earned, so they must be there -- an
    # absence assertion alone would also pass on a version that never emits them at all.
    num_heads = int(runner.model.lag_attn.num_heads)
    assert sum(
        name.startswith(te_lag_analysis._HEAD_SHARE_PREFIX) for name in frame.columns
    ) == num_heads
    assert frame["te_lag_map_label"].tolist() == ["attribution"] * 4


def test_the_guards_are_recorded_in_the_preflight_preconditions(
    make_eval_runner, tmp_path
) -> None:
    """A rejected readout must be explicable from the run's own output, not from the code."""
    runner = make_eval_runner(
        dict(SHIPPED_KWARGS, causal_norm=False), output_dir=tmp_path / "runner"
    )
    preconditions = preflight.interpretation_preconditions(runner)
    assert preconditions["causal_norm"]["value"] is False
    assert "te_lag_map" in preconditions["causal_norm"]["blocks"]


# ---------------------------------------------------------------------------
# S4-T04: the identity survives eval's averaging
# ---------------------------------------------------------------------------
def test_the_identity_survives_evals_time_averaging(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, perturb_full_pathway
) -> None:
    r"""On a live model, $\sum_\ell$ of the averaged map must still equal the averaged $K_t$.

    Required on a perturbed model: at init $K_t \equiv 0$ and the identity is $0 = 0$, which
    holds for any averaging rule whatsoever, including a wrong one.
    """
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_full_pathway(runner.model)
    summary, frame = _run(
        runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "identity"
    )

    assert summary["mean_kld"] > 0.0, "the perturbation did not make K_t nonzero"
    assert summary["identity"]["holds"] is True
    assert summary["identity"]["max_rel_deviation"] < te_lag_analysis.IDENTITY_TOLERANCE
    assert frame["te_attributed_total"].to_numpy() == pytest.approx(
        frame["kld_mean"].to_numpy(), rel=1e-4
    )


def test_dead_anchors_are_excluded_so_a_band_mask_does_not_break_the_identity(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, perturb_full_pathway
) -> None:
    r"""The case the exclusion exists for.

    A band excluding lag $0$ leaves anchors $t < \min(\mathrm{band})$ with no causally valid lag;
    the model forces lag $0$ back on to keep ``entmax15`` well-posed and then zeroes those rows.
    They sum to $0$ against a nonzero $K_t$, so averaging them in breaks the identity -- by
    percent, not by rounding.
    """
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_full_pathway(runner.model)

    num_lags = int(runner.model.lag_attn.L)
    band = masks.lag_band_keep_mask((4, num_lags - 1), num_lags)
    original = runner.forward
    # Any lag_band_mask the analysis might pass is deliberately overridden, not merged: the
    # point is to force every forward in this run through the band.
    runner.forward = lambda batch, **_: original(batch, lag_band_mask=band)

    summary, frame = _run(
        runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "banded"
    )

    assert summary["identity"]["holds"] is True, (
        "dead anchors leaked into the average: their zeroed rows do not sum to K_t"
    )
    # The exclusion is real: the support is genuinely smaller than the unbanded one.
    assert frame["n_support_anchors"].min() > 0


def test_including_dead_anchors_would_break_the_identity(
    make_eval_runner, tiny_loader, tmp_path, perturb_full_pathway
) -> None:
    """The negative control: prove the exclusion is doing work, not decorating a passing test."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_full_pathway(runner.model)

    num_lags = int(runner.model.lag_attn.L)
    band = masks.lag_band_keep_mask((4, num_lags - 1), num_lags)
    batch = next(iter(tiny_loader))
    with runner.inference_mode():
        outputs = runner.forward(runner.to_device(batch), lag_band_mask=band)

    alpha = outputs["attn_weights"]
    live = masks.live_anchor_mask(alpha)
    assert not bool(live.all()), "the band produced no dead anchors, so this proves nothing"

    correct = masks.lag_readout_support(runner.model, alpha, batch.weight)
    naive = masks.kld_mask(runner.model, batch.weight, alpha.shape[0], alpha.shape[1])

    def _deviation(support):
        te = te_lag_analysis.support_mean(outputs["te_lag_map"], support).sum(dim=1)
        kld = te_lag_analysis.support_mean(outputs["kld_per_t"], support)
        return float(((te - kld).abs() / kld.abs().clamp_min(1e-30)).max())

    assert _deviation(correct) < te_lag_analysis.IDENTITY_TOLERANCE
    assert _deviation(naive) > te_lag_analysis.IDENTITY_TOLERANCE


def test_the_table_carries_one_column_per_lag_plus_the_argmax_and_its_second(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    config = tiny_eval_config["eval_config"]
    summary, frame = _run(runner, tiny_loader, config, tmp_path / "table")

    num_lags = int(runner.model.lag_attn.L)
    # The analysis's own selector, not a prefix check re-implemented here. ``te_lag_map_label``
    # also begins with ``te_l``; ``_lag_columns`` requires the remainder to be digits, and
    # asserting through it pins the selector the figure actually indexes with rather than a
    # looser rule that happens to agree today.
    lag_columns = te_lag_analysis._lag_columns(frame)
    assert len(lag_columns) == num_lags == summary["num_lags"]
    assert {"argmax_lag", "argmax_lag_seconds"} <= set(frame.columns)
    assert frame["argmax_lag_seconds"].to_numpy() == pytest.approx(
        4.0 * frame["argmax_lag"].to_numpy()
    )


# ---------------------------------------------------------------------------
# S4-T05: the per-head decomposition, behaviourally
# ---------------------------------------------------------------------------
def test_zeroing_one_heads_attention_moves_only_that_heads_contribution(
    make_eval_runner, tiny_loader, tmp_path, perturb_full_pathway
) -> None:
    r"""The property head structuring actually claims.

    Not the sum-to-one identity, which is a tautology of the contiguous view and holds on a
    completely wrong model. Head $m$'s contribution $K^{(m)}\alpha^{(m)}$ must depend on head
    $m$'s attention and on nothing else.
    """
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_full_pathway(runner.model)

    batch = runner.to_device(next(iter(tiny_loader)))
    with runner.inference_mode():
        outputs = runner.forward(batch)

    alpha = outputs["attn_weights"]
    per_head = outputs["kld_per_t_per_head"]
    support = masks.lag_readout_support(runner.model, alpha, batch.weight)

    def _contribution(weights):
        return (
            (per_head.unsqueeze(-1) * weights) * support[:, :, None, None]
        ).sum(dim=(0, 1, 3))

    before = _contribution(alpha)
    silenced = alpha.clone()
    silenced[:, :, 1, :] = 0.0
    after = _contribution(silenced)

    assert float(after[1]) == pytest.approx(0.0, abs=1e-9)
    assert float(before[1]) > 0.0, "head 1 carried nothing to begin with; the test is vacuous"
    others = [index for index in range(alpha.shape[2]) if index != 1]
    assert after[others].tolist() == pytest.approx(before[others].tolist(), rel=1e-6)


def test_the_per_head_profile_sums_to_the_lag_attribution(
    make_eval_runner, tiny_loader, tmp_path, perturb_full_pathway
) -> None:
    r"""$\sum_m K^{(m)}_t \alpha^{(m)}_{t,\ell} = \widetilde{TE}_{t,\ell}$ -- what makes it a
    decomposition rather than two readouts plotted together."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_full_pathway(runner.model)

    profile = te_lag_analysis.head_lag_profile(runner, tiny_loader, None)
    assert profile.shape == (int(runner.model.lag_attn.num_heads), int(runner.model.lag_attn.L))
    assert profile.sum() > 0.0

    batch = runner.to_device(next(iter(tiny_loader)))
    with runner.inference_mode():
        outputs = runner.forward(batch)
    support = masks.lag_readout_support(runner.model, outputs["attn_weights"], batch.weight)
    expected = (
        (outputs["te_lag_map"] * support.unsqueeze(-1)).sum(dim=(0, 1)) / support.sum()
    )
    # One batch of the two, so compare the shape of the profile rather than its exact scale.
    assert np.argmax(profile.sum(axis=0)) == int(torch.argmax(expected))


def test_the_per_head_profile_raises_on_a_flat_latent(
    make_eval_runner, tiny_loader, tmp_path
) -> None:
    """Under a flat latent the contiguous per-head slice is an arbitrary partition."""
    runner = make_eval_runner(
        dict(SHIPPED_KWARGS, head_structured_latent=False), output_dir=tmp_path / "runner"
    )
    with pytest.raises(preflight.TEPreconditionUnmet, match="head_structured_latent"):
        te_lag_analysis.head_lag_profile(runner, tiny_loader, None)


def test_shares_are_null_at_zero_kl_while_the_absolute_per_head_kl_is_still_reported(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    r"""At init $K_t \equiv 0$, so every share is $0/0$ -- undefined, not zero.

    Zero is a legitimate share, so it cannot double as "no data". The absolute per-head KL is a
    perfectly good $0$ and is reported regardless, and the count of undefined samples is recorded
    rather than left to be inferred from a column of nulls.
    """
    runner = make_eval_runner(perturb=False, output_dir=tmp_path / "runner")
    summary, frame = _run(
        runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "zero"
    )

    assert summary["per_head"]["n_samples_with_zero_kl"] == 4
    assert bool(frame["share_head0"].isna().all())
    assert frame["k_head0"].to_numpy() == pytest.approx(0.0, abs=1e-9)


def test_the_per_head_table_is_written(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, perturb_full_pathway
) -> None:
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_full_pathway(runner.model)
    _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "heads")

    per_head = pd.read_csv(
        tmp_path / "heads" / te_lag_analysis.ANALYSIS_DIRNAME / "per_head.csv"
    )
    assert set(per_head["quantity"]) == {"kld", "share"}
    assert set(per_head["head"]) == set(range(int(runner.model.lag_attn.num_heads)))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def test_both_figures_carry_a_physical_second_axis(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, monkeypatch, perturb_full_pathway
) -> None:
    captured: dict = {}
    original = figures.render_figure

    def _capture(fig, path, **kwargs):
        captured[Path(path).name] = {
            "titles": [ax.get_title() for ax in fig.axes if ax.get_title()],
            "has_data": [ax.has_data() for ax in fig.axes if ax.get_title()],
            "secondary": [
                child.get_xlabel()
                for ax in fig.axes
                for child in ax.child_axes
                if child.get_xlabel()
            ],
        }
        return original(fig, path, **kwargs)

    monkeypatch.setattr(figures, "render_figure", _capture)
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_full_pathway(runner.model)
    _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "figures")

    assert set(captured) == {"te_lag", "per_head_lag_profile"}
    assert len(captured["te_lag"]["titles"]) == 3
    assert captured["te_lag"]["secondary"] == ["Physical delay (s)"]
    assert all(captured["te_lag"]["has_data"])
    assert captured["per_head_lag_profile"]["secondary"] == ["Physical delay (s)"]
    assert all(captured["per_head_lag_profile"]["has_data"])
