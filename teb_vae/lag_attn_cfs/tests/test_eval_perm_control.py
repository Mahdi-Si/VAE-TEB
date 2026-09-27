r"""The specificity criterion, the two ways of getting it wrong, and the analysis that reports it.

**The verdict must take three losses and nothing else.** A criterion that could see the KL would
fail exactly the healthy models it should pass, because a stranger's source is out of distribution
for a posterior trained on matched pairs and therefore moves it **more**. The case that would fail
under the abandoned KL-space criterion -- $K_{\mathrm{shuffled}} > K_{\mathrm{true}}$ with the loss
ordering intact -- is written out here and must PASS.

**A stale key from the permuted dict must be caught.** ``perm_forward_outputs`` returns a *shallow
copy*: only :data:`~teb_vae.lag_attn_rws.nets.controls.RECOMPUTED_KEYS` describe the permuted
pairing, and every other key is the matched forward's own tensor -- the same object. So
``permuted['kld_per_t']`` is the **true** KL, and an evaluation that read it would report the
matched coupling under the control's name with nothing failing. The test here shows that the
shuffled KL the collection pass reports is *not* the matched one; which keys the control replaces
is pinned by identity in ``test_perm_control.py``.
"""
from __future__ import annotations

import types
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn_cfs.eval import metrics as metrics_module
from teb_vae.lag_attn_cfs.eval.analyses import AnalysisContext
from teb_vae.lag_attn_cfs.eval.analyses import perm_control as perm_control_analysis
from teb_vae.lag_attn_cfs.eval.metrics import (
    FAIL,
    INCONCLUSIVE,
    PASS,
    evaluate_batch,
    source_specificity_verdict,
)

#: Bootstrap settings: instant, and seeded so every interval is reproducible.
EVAL_CONFIG = {"bootstrap_resamples": 200, "seed": 0}


# =================================================================================================
# The verdict takes three losses
# =================================================================================================
def test_the_ordering_passes_and_carries_its_numbers() -> None:
    verdict = source_specificity_verdict(10.0, 8.0, 14.0)

    assert verdict.status == PASS
    assert verdict.values["shuffle_penalty"] == pytest.approx(4.0)


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


# =================================================================================================
# The permuted dict's stale keys
# =================================================================================================
def test_the_shuffled_kl_readout_is_not_the_matched_one(task, perturb_posterior):
    """The behavioural guard on the stale key. Reading ``permuted['kld_per_t']`` would make this
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


# =================================================================================================
# The analysis
# =================================================================================================
def _per_sample(**columns: List[float]) -> pd.DataFrame:
    """A per-sample frame carrying the named columns over three recordings of two segments."""
    frame = pd.DataFrame({"guid": ["a", "a", "b", "b", "c", "c"], **columns})
    for _branch, column in perm_control_analysis.BRANCH_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame


def _context(
    per_sample: pd.DataFrame, results: Optional[Dict[str, Any]] = None
) -> AnalysisContext:
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
    """The summary CSV holds the branch table only: the source margin is a *penalty* row, and it
    travels in ``summary.json``'s ``penalties`` and the headline, not here."""
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
    assert "penalty" not in branches.columns
    # The three paired controls -- the source margin among them -- reach a table of their own
    # rather than only the summary record.
    penalties = pd.read_csv(directory / perm_control_analysis.PENALTIES_FILENAME)
    assert set(penalties["penalty"]) == {
        "shuffle_penalty", "prior_shuffle_penalty", perm_control_analysis.SOURCE_MARGIN_PENALTY,
    }
    assert perm_control_analysis.PENALTIES_FILENAME in result["files"]


class _Loader:
    """A dataloader-shaped iterable over a fixed list of batches."""

    def __init__(self, batches) -> None:
        self._batches = list(batches)

    def __iter__(self):
        return iter(self._batches)


def test_a_batch_with_no_cross_recording_partner_is_scored_without_its_control_and_counted(
    task, perturb_posterior
) -> None:
    """The derangement is GUID-aware, and a batch too concentrated to pair across recordings keeps
    every readout but its control.

    Two halves are asserted: the per-batch entry point never falls back to a within-recording
    pairing -- its control columns are NaN instead -- and the loop above it counts the batch, since
    the samples missing from the control are the longest recordings' and the control's average
    leans away from them.
    """
    from teb_vae.lag_attn_cfs.eval.metrics import CONTROL_COLUMNS

    from .conftest import make_stub_batch

    module = task()
    perturb_posterior(module.orig_model)
    module.eval()
    torch.manual_seed(0)
    # Every segment of one recording: a derangement cannot put any row against another recording.
    concentrated = make_stub_batch(seed=11)
    concentrated.guid = ["ONE"] * len(concentrated.guid)

    readout = evaluate_batch(module, concentrated, num_samples=1)
    assert readout.n_control_pairs == 0
    for name in CONTROL_COLUMNS:
        assert bool(torch.isnan(readout.columns[name]).all()), name
    assert bool(torch.isfinite(readout.columns["mc_pred_gap"]).all())

    torch.manual_seed(0)
    results = metrics_module.evaluate(
        module, _Loader([concentrated, make_stub_batch(seed=12)]), num_samples=1
    )
    record = results["controls"]

    assert record["n_batches_excluded_no_cross_recording_partner"] == 1
    assert record["n_samples_excluded_no_cross_recording_partner"] == len(concentrated.guid)
    # And the batch that could be paired still was, so the exclusion is selective rather than total.
    assert record["n_control_pairs"] > 0
    assert record["same_recording_pairing_rate"] == pytest.approx(0.0)


@pytest.mark.slow
def test_the_pairing_record_reaches_the_summary_even_at_zero(collected_run) -> None:
    """Both the rate and the excluded counts, whatever they are: this number is the only evidence
    that the control is still a control, and a key that appeared only when non-zero would be
    indistinguishable from a run that never checked."""
    controls_record = collected_run["summary"]["results"]["controls"]

    for key in (
        "same_recording_pairing_rate",
        "n_control_pairs",
        "n_same_recording_pairs",
        "n_batches_excluded_no_cross_recording_partner",
        "n_samples_excluded_no_cross_recording_partner",
    ):
        assert key in controls_record, key
    assert collected_run["summary"]["results"]["perm_control"]["pairing"] == controls_record


@pytest.mark.slow
def test_on_a_real_run_the_analysis_and_the_summary_agree_on_the_verdict(collected_run) -> None:
    """The analysis applies the run's own criterion to the run's own per-recording means, so a
    disagreement here would mean two different populations were reduced under one name."""
    results = collected_run["summary"]["results"]
    reported = {verdict["name"]: verdict["status"] for verdict in results["verdicts"]}

    assert results["perm_control"]["specificity_verdict"]["status"] == (
        reported["source_specificity"]
    )
    assert results["perm_control"]["outcome"] in perm_control_analysis.OUTCOMES
