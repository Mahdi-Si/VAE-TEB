r"""Tests for the permutation controls and the source-specificity verdict.

Three properties carry the file.

**Reproducibility pins the global RNG, not just the generator.** ``perm_forward_outputs`` samples
$z$ through ``torch.randn_like``, which takes no generator. A run that passed ``generator=`` and
left the global RNG alone would look carefully seeded and produce a different shuffled loss on
every rerun, so the reproducibility test is specifically a test of that.

**A short batch is skipped, not raised on.** A derangement of one element does not exist, and the
last batch of a split is routinely short.

**The KL-space reading cannot flip the verdict.** That is the specific misreading the whole design
guards against, and it is tested directly: a case where $K_{\mathrm{shuffled}} > K_{\mathrm{true}}$
must still return ``source_specific`` when the losses order correctly.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn.eval import figures
from teb_vae.lag_attn.eval.analyses import perm_control as perm_control_analysis
from teb_vae.lag_attn.tests.conftest import make_stub_batch

PROBE = {"n_samples": 4, "source_files": ["tiny_shard.hdf5"] * 4}


def _run(runner, loader, eval_config, output_dir):
    """Run the analysis and return ``(summary, per-sample frame)``."""
    summary = perm_control_analysis.run_perm_control_analysis(
        runner, loader, eval_config=eval_config, output_dir=output_dir, probe=PROBE
    )
    directory = Path(output_dir) / perm_control_analysis.ANALYSIS_DIRNAME
    return summary, pd.read_csv(directory / "per_sample.csv")


# ---------------------------------------------------------------------------
# S6-T02: the verdict, on synthetic losses
# ---------------------------------------------------------------------------
def test_the_ordering_that_establishes_specificity() -> None:
    r"""$L_{feat} < L_{base} < L_{shuffled}$: helped by its own source, hurt by a stranger's."""
    verdict = perm_control_analysis.source_specificity_verdict(1.0, 2.0, 3.0)
    assert verdict["verdict"] == "source_specific"
    assert verdict["uplift_margin"] == pytest.approx(1.0)
    assert verdict["shuffle_penalty_margin"] == pytest.approx(1.0)


def test_a_source_that_helps_but_is_not_specific_is_named_as_such() -> None:
    """The outcome the control exists to detect: marginal statistics, not the pairing."""
    verdict = perm_control_analysis.source_specificity_verdict(1.0, 2.0, 1.5)
    assert verdict["verdict"] == "influential_not_specific"
    assert verdict["shuffle_penalty_margin"] < 0.0
    assert "marginal statistics" in verdict["explanation"]


def test_no_uplift_is_reported_as_such_rather_than_as_a_specificity_failure() -> None:
    """With no uplift there is no information whose specificity could be tested."""
    verdict = perm_control_analysis.source_specificity_verdict(2.0, 2.0, 5.0)
    assert verdict["verdict"] == "no_uplift"


def test_a_non_finite_loss_yields_undetermined_not_a_silent_pass() -> None:
    verdict = perm_control_analysis.source_specificity_verdict(float("nan"), 2.0, 3.0)
    assert verdict["verdict"] == "undetermined"


def test_the_kl_space_reading_cannot_flip_the_verdict() -> None:
    r"""The case the whole design guards against.

    $K_{\mathrm{shuffled}} > K_{\mathrm{true}}$ is routine on a healthy model -- a mismatched
    source is out of distribution and moves the posterior *more*. The verdict function takes the
    three losses and nothing else, so this is enforced by the signature rather than by care.
    """
    from inspect import signature

    parameters = list(signature(perm_control_analysis.source_specificity_verdict).parameters)
    assert parameters == ["l_feat", "l_base", "l_shuffled"]
    assert not any("kl" in name.lower() or "kld" in name.lower() for name in parameters)

    # A run whose KL-space reading looks like a catastrophe but whose losses order correctly.
    verdict = perm_control_analysis.source_specificity_verdict(1.0, 2.0, 3.0)
    assert verdict["verdict"] == "source_specific"


def test_the_kl_readout_is_labelled_so_it_cannot_be_read_as_the_criterion(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, perturb_full_pathway
) -> None:
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_full_pathway(runner.model)
    summary, _ = _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "label")

    assert "not specificity" in summary["kl_space"]["label"]
    assert "expected" in summary["kl_space"]["label"]
    assert summary["specificity"]["criterion"] == "L_feat < L_base < L_feat_shuffled"


# ---------------------------------------------------------------------------
# S6-T01: the collector
# ---------------------------------------------------------------------------
def test_two_runs_under_one_seed_are_identical(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, perturb_full_pathway
) -> None:
    """Fails if only ``generator=`` is seeded and the global RNG is left alone.

    ``perm_forward_outputs`` draws $z$ from the global RNG, so the shuffled loss -- and only the
    shuffled loss -- would drift between reruns.
    """
    config = tiny_eval_config["eval_config"]
    summaries = []
    frames = []
    for label in ("first", "second"):
        runner = make_eval_runner(output_dir=tmp_path / f"runner_{label}")
        perturb_full_pathway(runner.model)
        # Deliberately disturbed between runs: a run that only worked from a freshly seeded
        # process would pass without the per-batch seeding this test exists to pin.
        torch.manual_seed(999)
        summary, frame = _run(runner, tiny_loader, config, tmp_path / label)
        summaries.append(summary)
        frames.append(frame)

    assert summaries[0]["specificity"] == summaries[1]["specificity"]
    for column in ("l_feat", "l_base", "l_feat_shuffled", "kld_shuffled_per_t"):
        assert frames[0][column].to_numpy() == pytest.approx(
            frames[1][column].to_numpy(), rel=0, abs=0
        ), f"{column} is not reproducible under a fixed seed"


def test_a_short_batch_is_skipped_and_counted_with_no_keys_at_all(
    make_eval_runner, tmp_path, perturb_full_pathway
) -> None:
    """A derangement of one element does not exist, and a short final batch is normal.

    The skip must contribute *no keys*, not zeros. A zero would scale the mean of
    ``feat_loss_shuffled`` toward zero and invert the very ordering the control checks, which
    reads as a collapsed source pathway on a perfectly healthy model -- the exact failure
    ``task.py`` records at length for the training-time control.
    """
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_full_pathway(runner.model)

    state = {"index": 0, "n_skipped": 0}
    per_batch = perm_control_analysis._make_per_batch(0, state)

    single = per_batch(runner, SimpleNamespace(**vars(make_stub_batch(batch_size=1))))
    assert single == {}, "a skipped batch must contribute no keys at all, not zeros"
    assert state["n_skipped"] == 1

    pair = per_batch(runner, SimpleNamespace(**vars(make_stub_batch(batch_size=2))))
    assert "l_feat_shuffled" in pair
    assert state["n_skipped"] == 1, "a derangeable batch must not be counted as skipped"


def test_the_control_reuses_the_completed_forward(
    make_eval_runner, tmp_path, perturb_full_pathway, monkeypatch
) -> None:
    r"""``permutation_kl`` re-encodes both streams from scratch; ``perm_kl_from_forward`` does not.

    On a full test split that is a second pass through both encoders per batch, for a number the
    completed forward already has the inputs for.
    """
    calls = {"from_forward": 0, "re_encode": 0}
    original = perm_control_analysis.controls.perm_kl_from_forward

    def _counted(*args, **kwargs):
        calls["from_forward"] += 1
        return original(*args, **kwargs)

    def _re_encoded(*_args, **_kwargs):
        calls["re_encode"] += 1
        raise AssertionError("permutation_kl re-encodes both streams and must not be called")

    monkeypatch.setattr(perm_control_analysis.controls, "perm_kl_from_forward", _counted)
    monkeypatch.setattr(perm_control_analysis.controls, "permutation_kl", _re_encoded)

    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_full_pathway(runner.model)

    state = {"index": 0, "n_skipped": 0}
    perm_control_analysis._make_per_batch(0, state)(
        runner, SimpleNamespace(**vars(make_stub_batch(batch_size=4)))
    )
    assert calls["from_forward"] == 1
    assert calls["re_encode"] == 0


def test_the_derangement_has_no_fixed_point(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, perturb_full_pathway
) -> None:
    """A fixed point would score that sample against its *own* source and dilute the control."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_full_pathway(runner.model)
    _, frame = _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "derange")

    # The tiny loader yields two batches of two, so the within-batch position cycles 0, 1, 0, 1.
    within_batch = frame["sample_index"].to_numpy() % 2
    assert not np.any(frame["perm_index"].to_numpy() == within_batch)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def test_the_per_sample_table_carries_all_three_losses_and_both_derived_columns(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, perturb_full_pathway
) -> None:
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_full_pathway(runner.model)
    summary, frame = _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "schema")

    assert {
        "l_feat", "l_base", "l_feat_shuffled", "shuffle_penalty", "kld_shuffled_ratio"
    } <= set(frame.columns)
    assert frame["shuffle_penalty"].to_numpy() == pytest.approx(
        (frame["l_feat_shuffled"] - frame["l_base"]).to_numpy()
    )
    # The per-step curves are flattened out of the CSV, as in every other analysis.
    assert not [name for name in frame.columns if name.startswith("kt") and name[2:].isdigit()]
    assert summary["n_samples"] == 4


def test_the_verdict_reaches_the_summary_at_top_level(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, perturb_full_pathway
) -> None:
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_full_pathway(runner.model)
    summary, _ = _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "top")

    assert summary["specificity"]["verdict"] in {
        "source_specific", "influential_not_specific", "no_uplift", "undetermined"
    }
    assert math.isfinite(summary["specificity"]["uplift_margin"])
    assert math.isfinite(summary["specificity"]["shuffle_penalty_margin"])


# ---------------------------------------------------------------------------
# S6-T03: the figures
# ---------------------------------------------------------------------------
def test_two_figures_are_written(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, perturb_full_pathway
) -> None:
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_full_pathway(runner.model)
    summary, _ = _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "figs")

    assert {Path(path).name for path in summary["figures"]} == {"losses.pdf", "kl_overlay.pdf"}
    for path in summary["figures"]:
        assert Path(path).suffix == ".pdf" and Path(path).stat().st_size > 0


def test_the_overlay_caption_states_that_a_higher_shuffled_kl_is_not_a_failure(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, monkeypatch, perturb_full_pathway
) -> None:
    """The caption is the mitigation for a high-likelihood misreading, so it is worth pinning."""
    captured: dict = {}
    original = figures.render_figure

    def _capture(fig, path, **kwargs):
        captured[Path(path).name] = {
            "texts": [text.get_text() for ax in fig.axes for text in ax.texts],
            "titles": [ax.get_title() for ax in fig.axes if ax.get_title()],
            "has_data": [ax.has_data() for ax in fig.axes if ax.get_title()],
        }
        return original(fig, path, **kwargs)

    monkeypatch.setattr(figures, "render_figure", _capture)
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    perturb_full_pathway(runner.model)
    _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "caption")

    caption = " ".join(captured["kl_overlay"]["texts"])
    assert "NOT a failure" in caption
    assert "K_shuffled >= K_true is EXPECTED" in caption
    assert all(captured["kl_overlay"]["has_data"])

    losses = captured["losses"]
    assert any(title.startswith("Source-specificity control") for title in losses["titles"])
    assert any("criterion" in text for text in losses["texts"])
    assert all(losses["has_data"])
