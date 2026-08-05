r"""The specificity criterion, and the two ways of getting it wrong.

**The verdict must take three losses and nothing else.** That is asserted by *signature*, not by
reading the implementation: a criterion that could see the KL would fail exactly the healthy models
it should pass, because a stranger's source is out of distribution for a posterior trained on
matched pairs and therefore moves it **more**. The case that would fail under the abandoned
KL-space criterion -- $K_{\mathrm{shuffled}} > K_{\mathrm{true}}$ with the loss ordering intact --
is written out here and must PASS.

**A stale key from the permuted dict must be caught.** ``perm_forward_outputs`` returns a *shallow
copy*: only :data:`~teb_vae.lag_attn_rws.nets.controls.RECOMPUTED_KEYS` describe the permuted
pairing, and every other key is the matched forward's own tensor -- the same object. So
``permuted['kld_per_t']`` is the **true** KL, and an evaluation that read it would report the
matched coupling under the control's name with nothing failing. Two tests close that: one pins
which keys the function actually replaces, by identity; the other shows that the shuffled KL the
collection pass reports is *not* the matched one, which it would be if that key had been read.
"""
from __future__ import annotations

import inspect
import types
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn_rws.eval import metrics as metrics_module
from teb_vae.lag_attn_rws.eval.analyses import perm_control as perm_control_analysis
from teb_vae.lag_attn_rws.eval.metrics import (
    FAIL,
    INCONCLUSIVE,
    PASS,
    evaluate_batch,
    source_specificity_verdict,
)
from teb_vae.lag_attn_rws.nets import controls

#: Bootstrap settings: instant, and seeded so every interval is reproducible.
EVAL_CONFIG = {"bootstrap_resamples": 200, "seed": 0}


# =============================================================================
# The verdict takes three losses
# =============================================================================
def test_the_specificity_verdict_accepts_only_the_three_losses() -> None:
    """By signature. An implementation that also read the KL would be a different criterion, and
    reviewing for it is what this assertion replaces."""
    parameters = list(inspect.signature(source_specificity_verdict).parameters)

    assert parameters == ["d_base", "d_full", "d_shuffled"]


def test_the_ordering_passes_and_carries_its_numbers() -> None:
    verdict = source_specificity_verdict(10.0, 8.0, 14.0)

    assert verdict.status == PASS
    assert verdict.values["shuffle_penalty"] == pytest.approx(4.0)
    assert verdict.criterion == "D_full < D_base < D_shuffled"


def test_a_healthy_model_whose_shuffled_kl_exceeds_its_true_one_still_passes() -> None:
    """The case the abandoned KL-space criterion would have failed. The KL cannot reach this
    function, so the case is expressed as: the losses order correctly, and the verdict passes
    whatever the KL did."""
    verdict = source_specificity_verdict(10.0, 8.0, 14.0)

    assert verdict.status == PASS
    assert "kl" not in " ".join(verdict.values).lower()


def test_a_broken_ordering_fails_rather_than_being_reported_as_inconclusive() -> None:
    assert source_specificity_verdict(10.0, 8.0, 9.0).status == FAIL


def test_a_control_that_did_not_run_is_inconclusive_rather_than_failed() -> None:
    """A control that could not run and a control that failed are different facts; reporting the
    first as FAIL makes a small last batch look like a broken model."""
    verdict = source_specificity_verdict(10.0, 8.0, None)

    assert verdict.status == INCONCLUSIVE


@pytest.mark.parametrize(
    "scores,expected",
    [
        ((10.0, 8.0, 14.0), "specific"),
        ((10.0, 8.0, 9.0), "influential_not_specific"),
        ((10.0, 10.5, 14.0), "no_improvement"),
        ((10.0, 8.0, None), "inconclusive"),
    ],
)
def test_the_outcome_classification_names_what_the_three_losses_did(scores, expected) -> None:
    """``influential_not_specific`` is a real finding: a model whose forecast improves under any
    source it is handed has learned that the source stream exists, not how to read this one."""
    assert perm_control_analysis.classify_outcome(*scores) == expected
    assert expected in perm_control_analysis.OUTCOMES


# =============================================================================
# The permuted dict's stale keys
# =============================================================================
def test_only_the_declared_keys_are_recomputed_under_the_permutation(task, perturb_posterior):
    """Pinned by identity. Everything not on the list is the matched forward's own tensor -- the
    prior, both encoder states and the base forecast are source-free, so a derangement cannot move
    them -- and a reader that treats one of them as the control's gets the true value."""
    from .conftest import make_stub_batch

    module = task()
    perturb_posterior(module.orig_model)
    module.eval()
    model = module.orig_model
    batch = make_stub_batch(seed=5)
    y_st, y_ph, u_stream, _fhr, _weight = metrics_module.model_inputs(module, batch)
    with torch.no_grad():
        outputs = model(y_st, y_ph, u_stream)
        permuted = controls.perm_forward_outputs(
            model, outputs, generator=torch.Generator().manual_seed(0)
        )

    replaced = {
        name for name, value in permuted.items()
        if name not in outputs or value is not outputs[name]
    }

    assert replaced == set(controls.RECOMPUTED_KEYS)
    # Named explicitly because it is the trap: the KL analysis keys keep their matched values.
    assert permuted["kld_per_t"] is outputs["kld_per_t"]
    assert permuted["source_kl_lag_map"] is outputs["source_kl_lag_map"]
    assert permuted["mu_base"] is outputs["mu_base"]


def test_the_shuffled_kl_readout_is_not_the_matched_one(task, perturb_posterior):
    """The behavioural half of the guard above. Reading ``permuted['kld_per_t']`` would make this
    column bit-identical to ``source_conditioned_kl_raw`` on every sample -- which is exactly what
    a stale read looks like from the outside, and nothing else would show it."""
    from .conftest import make_stub_batch

    module = task()
    perturb_posterior(module.orig_model)
    module.eval()
    torch.manual_seed(0)
    readout = evaluate_batch(module, make_stub_batch(seed=7), num_samples=1)

    true_kl = readout.columns["source_conditioned_kl_raw"]
    shuffled_kl = readout.columns["source_conditioned_kl_shuffled_raw"]

    assert shuffled_kl.shape == true_kl.shape
    assert not torch.allclose(shuffled_kl, true_kl), (
        "the shuffled KL is bit-identical to the matched one, which is what reading the permuted "
        "dict's stale kld_per_t produces"
    )
    assert float(shuffled_kl.min()) > 0.0


# =============================================================================
# The analysis
# =============================================================================
def _per_sample(**columns: List[float]) -> pd.DataFrame:
    """A per-sample frame carrying the named columns over three recordings of two segments."""
    frame = pd.DataFrame({"guid": ["a", "a", "b", "b", "c", "c"], **columns})
    for _branch, column in perm_control_analysis.BRANCH_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame


def _context(per_sample: pd.DataFrame, results: Optional[Dict[str, Any]] = None) -> Any:
    from teb_vae.lag_attn_rws.eval.analyses import AnalysisContext

    collection = types.SimpleNamespace(
        per_sample=per_sample, per_anchor=pd.DataFrame(), record={}, retained={},
        results=results or {},
    )
    return AnalysisContext(collection=collection, config={})


def test_the_analysis_reports_the_ordering_the_outcome_and_the_pairing(tmp_path) -> None:
    per_sample = _per_sample(
        mc_nll_base_block=[10.0] * 6,
        mc_nll_full_block=[8.0] * 6,
        mc_nll_shuffled_block=[14.0] * 6,
        mc_nll_base_shuffled_mu_block=[12.0] * 6,
        source_conditioned_kl_raw=[2.0] * 6,
        source_conditioned_kl_shuffled_raw=[3.0] * 6,
    )
    pairing = {"same_recording_pairing_rate": 0.0, "n_control_pairs": 6}

    result = perm_control_analysis.run_perm_control_analysis(
        _context(per_sample, {"controls": pairing}),
        eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None,
    )

    assert result["outcome"] == "specific"
    assert result["specificity_verdict"]["status"] == PASS
    assert result["pairing"] == pairing
    penalties = {row["penalty"]: row for row in result["penalties"]}
    assert penalties["shuffle_penalty"]["mean"] == pytest.approx(4.0)
    assert penalties["prior_shuffle_penalty"]["mean"] == pytest.approx(2.0)
    assert penalties["shuffle_penalty"]["positive_fraction"] == pytest.approx(1.0)
    assert penalties["shuffle_penalty"]["n_recordings_scored"] == 3


def test_the_source_margin_is_referenced_against_full_and_signs_like_its_neighbours(
    tmp_path,
) -> None:
    """The third paired control, and the two properties that make it readable beside the others.

    It is $D_{\\rm shuffled} - D_{\\rm full}$, so a positive value says the matched source beat the
    stranger -- the same "positive means the control is worse" convention the two penalties above
    it use, which is what lets all three be read down one column without a sign table.
    """
    per_sample = _per_sample(
        mc_nll_base_block=[10.0] * 6,
        mc_nll_full_block=[8.0] * 6,
        mc_nll_shuffled_block=[14.0] * 6,
        mc_nll_base_shuffled_mu_block=[12.0] * 6,
    )

    result = perm_control_analysis.run_perm_control_analysis(
        _context(per_sample), eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None
    )

    penalties = {row["penalty"]: row for row in result["penalties"]}
    margin = penalties["source_margin"]
    assert margin["mean"] == pytest.approx(6.0)  # 14 - 8, not 14 - 10
    assert margin["positive_fraction"] == pytest.approx(1.0)
    assert margin["n_recordings_scored"] == 3
    # The same statistical furniture the other two carry, so it is quotable the same way.
    shuffle = penalties["shuffle_penalty"]
    assert set(margin) == set(shuffle), "the margin must be quotable exactly as the penalties are"
    assert "D_shuffled - D_full" in margin["meaning"]


def test_the_source_margin_is_positive_where_the_base_referenced_penalty_is_not(
    tmp_path,
) -> None:
    """The state that motivates a third control at all.

    The source pathway costs more than it delivers, so the forecast is worse than the target-only
    one -- and a stranger's source is worse still. Referenced against base, the shuffle penalty is
    *negative* and reads as a failed control; referenced against full, the margin is positive and
    says the model is reading this recording. Both are true and they are different questions.
    """
    per_sample = _per_sample(
        mc_nll_base_block=[10.0] * 6,
        mc_nll_full_block=[12.0] * 6,
        mc_nll_shuffled_block=[13.0] * 6,
        mc_nll_base_shuffled_mu_block=[11.0] * 6,
    )

    result = perm_control_analysis.run_perm_control_analysis(
        _context(per_sample), eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None
    )

    penalties = {row["penalty"]: row for row in result["penalties"]}
    assert penalties["shuffle_penalty"]["mean"] == pytest.approx(3.0)
    assert penalties["source_margin"]["mean"] == pytest.approx(1.0)
    assert result["outcome"] == "no_improvement"
    assert result["specificity_verdict"]["status"] == FAIL


def test_the_margin_is_also_emitted_as_a_keyed_scalar(tmp_path) -> None:
    """The headline block is assembled by walking key paths, and ``penalties`` is a list -- which
    is why the shuffle penalty has never reached it. The margin is promoted, so it is emitted
    under its own key as well as in the list, and the two must be the same number."""
    result = perm_control_analysis.run_perm_control_analysis(
        _context(
            _per_sample(
                mc_nll_base_block=[10.0] * 6,
                mc_nll_full_block=[8.0] * 6,
                mc_nll_shuffled_block=[14.0] * 6,
                mc_nll_base_shuffled_mu_block=[12.0] * 6,
            )
        ),
        eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None,
    )

    keyed = result[perm_control_analysis.SOURCE_MARGIN_SCALAR]
    row = next(
        row for row in result["penalties"]
        if row["penalty"] == perm_control_analysis.SOURCE_MARGIN_PENALTY
    )
    assert keyed == pytest.approx(row["mean"])


def test_the_summary_csv_still_carries_branch_rows_only(tmp_path) -> None:
    """Stated as an assertion rather than left implicit: the margin is a *penalty* row, and
    ``perm_control_summary.csv`` has only ever held the branch table. A reader looking for the
    margin finds it in ``summary.json``'s ``penalties`` and in the headline, not here."""
    perm_control_analysis.run_perm_control_analysis(
        _context(
            _per_sample(
                mc_nll_base_block=[10.0] * 6,
                mc_nll_full_block=[8.0] * 6,
                mc_nll_shuffled_block=[14.0] * 6,
                mc_nll_base_shuffled_mu_block=[12.0] * 6,
            )
        ),
        eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None,
    )

    written = pd.read_csv(
        tmp_path / perm_control_analysis.ANALYSIS_DIRNAME
        / perm_control_analysis.SUMMARY_FILENAME
    )
    assert "penalty" not in written.columns
    assert list(written["branch"]) == [
        name for name, _ in perm_control_analysis.BRANCH_COLUMNS
    ]


def test_the_kl_reading_is_a_description_that_nothing_consumes(tmp_path) -> None:
    """``shuffled_exceeds_true`` sits true on a healthy model, so it is reported *and* labelled --
    and the verdict beside it is decided without it."""
    per_sample = _per_sample(
        mc_nll_base_block=[10.0] * 6,
        mc_nll_full_block=[8.0] * 6,
        mc_nll_shuffled_block=[14.0] * 6,
        mc_nll_base_shuffled_mu_block=[12.0] * 6,
        source_conditioned_kl_raw=[2.0] * 6,
        source_conditioned_kl_shuffled_raw=[5.0] * 6,
    )

    result = perm_control_analysis.run_perm_control_analysis(
        _context(per_sample), eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None
    )

    assert result["kl_space"]["shuffled_exceeds_true"] is True
    assert result["kl_space"]["difference"] == pytest.approx(3.0)
    assert "descriptive only" in result["kl_space"]["note"]
    # The KL says "the control moved the posterior more", and the verdict still passes.
    assert result["specificity_verdict"]["status"] == PASS


def test_the_analysis_writes_its_tables(tmp_path) -> None:
    result = perm_control_analysis.run_perm_control_analysis(
        _context(
            _per_sample(
                mc_nll_base_block=[10.0] * 6,
                mc_nll_full_block=[8.0] * 6,
                mc_nll_shuffled_block=[14.0] * 6,
                mc_nll_base_shuffled_mu_block=[12.0] * 6,
            )
        ),
        eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None,
    )

    directory = tmp_path / perm_control_analysis.ANALYSIS_DIRNAME
    assert (directory / perm_control_analysis.PER_RECORDING_FILENAME).is_file()
    assert (directory / perm_control_analysis.SUMMARY_FILENAME).is_file()
    branches = pd.read_csv(directory / perm_control_analysis.SUMMARY_FILENAME)
    assert list(branches["branch"]) == [
        name for name, _ in perm_control_analysis.BRANCH_COLUMNS
    ]
    assert result["files"]


def test_on_a_real_run_the_analysis_and_the_summary_agree_on_the_verdict(evaluated) -> None:
    """The analysis applies the run's own criterion to the run's own per-recording means, so a
    disagreement here would mean two different populations were reduced under one name."""
    results = evaluated["summary"]["results"]
    reported = {
        verdict["name"]: verdict["status"] for verdict in results["verdicts"]
    }

    assert results["perm_control"]["specificity_verdict"]["status"] == (
        reported["source_specificity"]
    )
    assert results["perm_control"]["outcome"] in perm_control_analysis.OUTCOMES


@pytest.mark.slow
def test_the_margin_is_produced_offline_with_no_model_loaded(
    trained_run, repointed_overrides, tmp_path
) -> None:
    """The readout must survive the path it will actually be read on.

    ``--only perm_control`` against a finished directory builds no model and touches no GPU: it
    reads ``per_sample.csv`` and recomputes. Every branch score the margin needs is already in
    that table, so a re-run must produce the row, the keyed scalar and the headline entry without
    a checkpoint -- which is what makes an already-finished production run reportable under the
    new criterion without paying for the forward pass again.
    """
    import json

    from teb_vae.lag_attn_rws.eval import run as run_module

    output_dir = tmp_path / "run"
    assert run_module.main(
        trained_run, output_dir, overrides=repointed_overrides, device="cpu", num_samples=1,
    ) == 0

    # No checkpoint this time: the tables stand in for the model.
    assert run_module.main(
        None, output_dir, overrides=repointed_overrides, only="perm_control",
    ) == 0

    results_dir = output_dir / run_module.RESULTS_DIRNAME
    summary = json.loads(
        (results_dir / run_module.SUMMARY_FILENAME).read_text(encoding="utf-8")
    )
    analysis = summary["results"]["perm_control"]

    row = next(
        row for row in analysis["penalties"]
        if row["penalty"] == perm_control_analysis.SOURCE_MARGIN_PENALTY
    )
    assert row["n_recordings_scored"] > 0
    assert analysis[perm_control_analysis.SOURCE_MARGIN_SCALAR] == pytest.approx(row["mean"])
    assert summary["results"]["headline"]["source_margin_nats"] == pytest.approx(row["mean"])
