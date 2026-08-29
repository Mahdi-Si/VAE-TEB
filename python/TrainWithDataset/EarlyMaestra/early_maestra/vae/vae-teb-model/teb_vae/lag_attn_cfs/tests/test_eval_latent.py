r"""The failure this analysis exists for: a prior variance pinned on its clamp.

The KL carries $(\mu^q - \mu^p)^2/\sigma_p^2$. Pin $\sigma_p^2$ on its lower bound and every
coupling number in the run is multiplied by an arbitrary factor -- while ``mean_logvar_full`` and
``mean_logvar_base``, which are *decoder* variances and are what a reader actually looks at, stay
exactly where they were. That is the construction at the centre of this file: a model whose prior
log-variance is pinned at the floor, run through the real readout path, must fail the verdict
**while every decoder-side diagnostic beside it looks healthy**. A test that only checked the
verdict on a hand-built aggregate would not show that the two are separable, which is the entire
reason the detector is worth having.

Three smaller properties travel with it:

* **The margin is the model's own.** The bound is a sigmoid, so an exact-equality test against the
  asymptote reads $0.0$ forever while the variance sits pinned against it. Everything is measured
  against :data:`~teb_vae.lag_attn_rws.nets.model.LOGVAR_FLOOR_MARGIN_FRAC` of the clamp's range,
  imported rather than restated, and the test asserts the import rather than the number.
* **Only the two saturation fractions need a masked recomputation.** In this model the
  log-variance fractions are *already* masked and it is ``mu_prior_sat_frac`` and
  ``delta_mu_sat_frac`` that are flat means over every element. Both are emitted and they may
  legitimately disagree.
* **The spectrum sums to the KL it decomposes**, and is drawn sorted with the activity threshold
  marked -- the count of active dimensions is a count against a line, and a bar chart without that
  line invites a reader to pick their own.

**Nothing here is target-domain specific, and that is the assertion.** Of the analyses that read
only the durable tables, this is the one the divergence manifest classifies ``equivalent``: it
reads the latent, and a latent is $d_z$ numbers per step whether the decoder emits raw samples or
wavelet coefficients. The agreement is pinned against the sibling's implementation in
``test_eval_sibling_agreement.py``; what is pinned here is that the readouts it reduces exist on
*this* cell's per-sample table and carry this cell's own anchor support.
"""
from __future__ import annotations

import types
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn_cfs.eval.figures_seam import figure_filename
from teb_vae.lag_attn_cfs.eval import metrics as metrics_module
from teb_vae.lag_attn_cfs.eval.analyses import latent as latent_analysis
from teb_vae.lag_attn_cfs.eval.metrics import (
    FAIL,
    INCONCLUSIVE,
    PASS,
    Aggregate,
    build_verdicts,
    evaluate_batch,
)
from teb_vae.lag_attn_rws.nets.losses import KLD_ACTIVE_EPS
from teb_vae.lag_attn_rws.nets.model import LOGVAR_FLOOR_MARGIN_FRAC

from .conftest import make_stub_batch, shipped_warmup_kwargs

#: Bootstrap settings: instant, and seeded.
EVAL_CONFIG = {"bootstrap_resamples": 200, "seed": 0}


def _by_name(verdicts) -> Dict[str, Any]:
    """Index verdicts by name."""
    return {verdict.name: verdict for verdict in verdicts}


def _aggregate(**overall) -> Aggregate:
    """An aggregate carrying the named readouts and a healthy latent spectrum."""
    return Aggregate(overall=dict(overall), kld_per_dim=[0.4, 0.3, 0.2, 0.001])


# =============================================================================
# The margin is the model's, not a second copy of 0.05
# =============================================================================
def test_the_evaluation_reuses_the_models_own_floor_margin() -> None:
    """Identity, not equality: two constants that happen to agree today are two constants."""
    from teb_vae.lag_attn_rws.nets import model as model_module

    assert metrics_module.LOGVAR_FLOOR_MARGIN_FRAC is model_module.LOGVAR_FLOOR_MARGIN_FRAC
    assert metrics_module.SATURATION_FRAC is model_module.SATURATION_FRAC


def test_the_margin_on_the_shipped_clamp_is_four_tenths_of_a_nat() -> None:
    """The number a reader will see on the log-variance figure, derived rather than asserted from
    memory: $0.05 \\times (3 - (-5)) = 0.4$."""
    lo, hi = shipped_warmup_kwargs()["logvar_clamp"]

    assert LOGVAR_FLOOR_MARGIN_FRAC * (hi - lo) == pytest.approx(0.4)


# =============================================================================
# The verdict, on a model whose prior variance is pinned
# =============================================================================
@pytest.fixture
def pinned_prior_task(task, perturb_posterior):
    """A task whose prior log-variance is pinned at its floor, and whose decoder is untouched.

    The prior head's own forward is wrapped rather than its weights being driven to an extreme:
    what is under test is the *detector*, and a construction that reached through the residual MLP
    to find the layer that happens to produce the bound would break the day that head is
    restructured. The decoder is left alone precisely so the test can show the decoder-side
    diagnostics staying healthy while this fails.
    """
    module = task()
    perturb_posterior(module.orig_model)
    module.eval()
    head = module.orig_model.prior_head
    lo, _hi = module.orig_model.logvar_clamp
    original = head.forward

    # ``clock`` is forwarded rather than dropped: the prior head takes it whenever the model was
    # built with `prior_availability_input`, and a stub that swallowed it would make this fixture
    # silently untestable on exactly the arm the family ships.
    def _pinned(h_y, clock=None):
        mu_prior, _logvar_prior, raw_logvar_prior = original(h_y, clock)
        # Just above the asymptote, which is where a saturated sigmoid actually lands -- the bound
        # is never reached exactly, and a test that placed it *at* lo would be testing a value the
        # model cannot produce.
        return mu_prior, torch.full_like(mu_prior, float(lo) + 1e-3), raw_logvar_prior

    head.forward = _pinned  # type: ignore[method-assign]
    return module


def test_a_pinned_prior_variance_is_detected_while_the_decoder_looks_healthy(
    pinned_prior_task,
) -> None:
    """The failure this analysis exists for, through the real readout path."""
    torch.manual_seed(0)
    readout = evaluate_batch(pinned_prior_task, make_stub_batch(seed=3), num_samples=1)

    assert float(readout.columns["logvar_prior_floor_frac"].mean()) == pytest.approx(1.0)
    # And the decoder-side diagnostics -- the ones a reader looks at -- are untouched. Not
    # asserted at exactly zero: a handful of coefficients land inside the margin at random init,
    # which is a distribution with a tail rather than a variance pinned against a bound, and the
    # verdict's own threshold is what separates the two.
    threshold = metrics_module.DEFAULT_PINNED_VARIANCE_MAX_FRAC
    assert float(readout.columns["logvar_full_floor_frac"].mean()) < 0.1 * threshold
    assert float(readout.columns["logvar_full_ceil_frac"].mean()) < 0.1 * threshold
    lo, hi = pinned_prior_task.orig_model.logvar_clamp
    assert lo < float(readout.columns["mean_logvar_full"].mean()) < hi


def test_a_healthy_prior_variance_is_not_flagged(task, perturb_posterior) -> None:
    """Non-vacuity for the case above: a detector that returned 1.0 always would pass it."""
    module = task()
    perturb_posterior(module.orig_model)
    module.eval()
    torch.manual_seed(0)

    readout = evaluate_batch(module, make_stub_batch(seed=3), num_samples=1)

    assert float(readout.columns["logvar_prior_floor_frac"].mean()) < 0.5


def test_the_verdict_fails_on_a_pinned_prior_and_carries_its_fraction_and_margin() -> None:
    verdict = _by_name(
        build_verdicts(
            _aggregate(
                logvar_prior_floor_frac=1.0,
                mean_logvar_prior=-4.99,
                # Healthy, and unable to save the verdict: these are decoder variances.
                mean_logvar_full=0.1,
                logvar_full_floor_frac=0.0,
                logvar_full_ceil_frac=0.0,
            )
        )
    )["prior_variance_not_pinned"]

    assert verdict.status == FAIL
    assert verdict.values["floor_frac"] == pytest.approx(1.0)
    assert verdict.values["max_frac"] == pytest.approx(
        metrics_module.DEFAULT_PINNED_VARIANCE_MAX_FRAC
    )
    assert "sigma_p" in verdict.detail or "coupling" in verdict.detail


def test_the_verdict_passes_on_a_healthy_prior() -> None:
    verdict = _by_name(
        build_verdicts(_aggregate(logvar_prior_floor_frac=0.01, mean_logvar_prior=-1.0))
    )["prior_variance_not_pinned"]

    assert verdict.status == PASS
    assert verdict.values["floor_frac"] == pytest.approx(0.01)


def test_an_unmeasured_fraction_is_inconclusive_rather_than_a_pass() -> None:
    """Tables collected before these columns existed report nothing here, and "unmeasured" must
    not read as "fine"."""
    verdict = _by_name(build_verdicts(_aggregate()))["prior_variance_not_pinned"]

    assert verdict.status == INCONCLUSIVE
    assert verdict.values, "a status with nothing behind it is a claim a reader cannot check"


def test_the_decoder_verdict_watches_both_ends() -> None:
    """The two clamps fail differently: on the floor the decoder is over-confident and the NLL's
    squared term explodes; on the ceiling it has given up and is predicting noise, which reads as
    a healthy falling NLL while pred_gap goes to zero."""
    ceiling = _by_name(
        build_verdicts(
            _aggregate(
                logvar_prior_floor_frac=0.0,
                logvar_full_floor_frac=0.0,
                logvar_full_ceil_frac=0.9,
                mean_logvar_full=2.6,
            )
        )
    )["decoder_variance_not_pinned"]

    assert ceiling.status == FAIL
    assert ceiling.values["ceil_frac"] == pytest.approx(0.9)


# =============================================================================
# The prior's scale rate
# =============================================================================
def test_the_prior_rate_is_emitted_whatever_the_objective_weighted_it_at(
    task, perturb_posterior
) -> None:
    r"""``prior_rate`` is on the per-sample table unconditionally, and this is the case that would
    otherwise drop it: a run whose ``beta_prior`` is zero.

    A prior collapsing onto its clamp is visible in *any* checkpoint, so a column that appeared
    only for anchored runs would be missing from exactly the runs it diagnoses. The recombination
    of this column into the objective's own term is pinned separately, in ``test_eval_parity.py``.
    """
    module = task(hparams={"beta_prior": 0.0})
    perturb_posterior(module.orig_model)
    module.eval()
    torch.manual_seed(0)

    readout = evaluate_batch(module, make_stub_batch(seed=3), num_samples=1)

    assert "prior_rate" in readout.columns
    assert readout.columns["prior_rate"].shape == readout.n_anchors.shape
    assert torch.isfinite(readout.columns["prior_rate"]).all()


# =============================================================================
# The two saturation framings
# =============================================================================
def test_both_saturation_framings_are_emitted(task, perturb_posterior) -> None:
    """Masked and unmasked, because in this model it is these two -- and not the log-variance
    fractions -- that the model computes as flat means over every element."""
    module = task()
    perturb_posterior(module.orig_model)
    module.eval()
    torch.manual_seed(0)

    readout = evaluate_batch(module, make_stub_batch(seed=3), num_samples=1)

    for name in (
        "mu_prior_sat_frac_raw", "mu_prior_sat_frac_masked",
        "delta_mu_sat_frac_raw", "delta_mu_sat_frac_masked",
    ):
        assert name in readout.columns
        assert readout.columns[name].shape == readout.n_anchors.shape


def test_the_masked_saturation_framing_can_disagree_with_the_raw_one(
    task, perturb_posterior
) -> None:
    r"""Constructed rather than hoped for: the prior mean is driven to its bound on the warm-up
    prefix only, which the KL support excludes -- so the raw framing sees saturation and the
    masked one, correctly, does not.

    The prefix this cell excludes is the causal one and it is most of the sequence: the anchor
    floor is $133$ of $300$ steps rather than the raw cells' $30$, so the two framings differ here
    by far more than they do there.
    """
    module = task()
    perturb_posterior(module.orig_model)
    module.eval()
    model = module.orig_model
    warmup = model.geometry.warmup
    head = model.prior_head
    original = head.forward

    # ``clock`` is forwarded rather than dropped; see `pinned_prior_task` for why.
    def _saturated_prefix(h_y, clock=None):
        mu_prior, logvar_prior, raw_logvar_prior = original(h_y, clock)
        mu_prior = mu_prior.clone()
        mu_prior[:, :warmup] = model.mu_scale
        return mu_prior, logvar_prior, raw_logvar_prior

    head.forward = _saturated_prefix  # type: ignore[method-assign]
    torch.manual_seed(0)
    readout = evaluate_batch(module, make_stub_batch(seed=3), num_samples=1)

    raw = float(readout.columns["mu_prior_sat_frac_raw"].mean())
    masked = float(readout.columns["mu_prior_sat_frac_masked"].mean())
    assert raw > masked, "the warm-up prefix is in the raw framing and out of the masked one"
    assert masked == pytest.approx(0.0, abs=1e-6)


# =============================================================================
# The spectrum and its figure
# =============================================================================
def test_the_spectrum_is_sorted_by_what_each_dimension_carries() -> None:
    frame = latent_analysis.spectrum_frame([0.1, 2.0, 0.001, 0.5])

    assert list(frame["dimension"]) == [1, 3, 0, 2]
    assert list(frame["kl_nats"]) == pytest.approx([2.0, 0.5, 0.1, 0.001])
    assert list(frame["active"]) == [True, True, True, False]
    assert float(frame["share"].sum()) == pytest.approx(1.0)


def test_an_empty_spectrum_yields_an_empty_frame_rather_than_raising() -> None:
    assert latent_analysis.spectrum_frame([]).empty


def test_the_spectrum_figure_marks_the_activity_threshold_and_sorts_its_bars() -> None:
    """Both assertions are on the drawn artists: a figure plotting the unsorted spectrum, or
    marking a threshold of its own choosing, passes every table assertion above."""
    from teb_vae.lag_attn.eval import figures as shared_figures

    frame = latent_analysis.spectrum_frame([0.1, 2.0, 0.001, 0.5])

    figure = latent_analysis.build_spectrum_figure(frame)
    try:
        axis = figure.axes[0]
        heights = [float(patch.get_height()) for patch in axis.patches]
        # ``axhline`` draws under a blended transform: its y data are in data coordinates.
        horizontals = [
            float(line.get_ydata()[0]) for line in axis.lines
            if len(set(np.asarray(line.get_ydata(), dtype=np.float64))) == 1
        ]
    finally:
        shared_figures.plt.close(figure)

    assert heights == pytest.approx([2.0, 0.5, 0.1, 0.001])
    assert heights == sorted(heights, reverse=True)
    assert KLD_ACTIVE_EPS in horizontals


# =============================================================================
# The analysis
# =============================================================================
def _context(per_sample: pd.DataFrame, results: Dict[str, Any], record: Optional[Dict] = None):
    from teb_vae.lag_attn_cfs.eval.analyses import AnalysisContext

    collection = types.SimpleNamespace(
        per_sample=per_sample, per_anchor=pd.DataFrame(), record=record or {}, retained={},
        results=results,
    )
    return AnalysisContext(collection=collection, config={})


def test_the_analysis_writes_the_spectrum_the_diagnostics_and_the_figure(tmp_path) -> None:
    per_sample = pd.DataFrame({"guid": ["a", "a", "b", "b"]})
    for name, _meaning in latent_analysis.DIAGNOSTIC_COLUMNS:
        per_sample[name] = [0.1, 0.2, 0.3, 0.4]
    per_sample["source_conditioned_kl_raw"] = [2.0, 2.0, 3.0, 3.0]
    results = {
        "latent_health": {
            "d_z": 4, "active_dims": 3, "kl_total_nats": 2.601,
            "kld_per_dimension": [0.1, 2.0, 0.001, 0.5],
        },
        "verdicts": [{"name": "prior_variance_not_pinned", "status": "PASS"}],
    }
    record = {"bounds": {"logvar_clamp": [-5.0, 3.0], "logvar_margin": 0.4}}

    result = latent_analysis.run_latent_analysis(
        _context(per_sample, results, record),
        eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None,
    )

    directory = tmp_path / latent_analysis.ANALYSIS_DIRNAME
    for name in (
        latent_analysis.SPECTRUM_FILENAME,
        latent_analysis.DIAGNOSTICS_FILENAME,
        latent_analysis.PER_RECORDING_FILENAME,
        figure_filename(latent_analysis.SPECTRUM_FIGURE),
    ):
        assert (directory / name).is_file(), name
    reported = {row["metric"] for row in result["diagnostics"]}
    assert reported == {name for name, _ in latent_analysis.DIAGNOSTIC_COLUMNS}
    assert result["prior_variance_verdict"]["status"] == "PASS"
    assert result["bounds"]["logvar_margin"] == pytest.approx(0.4)
    assert result["activity_threshold_nats"] == pytest.approx(KLD_ACTIVE_EPS)


def test_every_diagnostic_the_analysis_reduces_is_a_column_this_cell_actually_produces(
    task, perturb_posterior
) -> None:
    """The join between a copied analysis and a forked ``metrics``: a column name that survived the
    copy but is not on *this* cell's per-sample table would be reported as an all-``NaN`` row with
    a confident meaning beside it, and nothing in the summary would say it was never measured."""
    module = task()
    perturb_posterior(module.orig_model)
    module.eval()
    torch.manual_seed(0)

    readout = evaluate_batch(module, make_stub_batch(seed=3), num_samples=1)

    produced = set(readout.columns)
    reduced = {name for name, _ in latent_analysis.DIAGNOSTIC_COLUMNS}
    assert reduced <= produced, sorted(reduced - produced)
    assert "source_conditioned_kl_raw" in produced


@pytest.mark.slow
def test_on_a_real_run_the_spectrum_sums_to_the_kl_it_decomposes(collected_run) -> None:
    """The identity the aggregation chain exists to preserve. Also checked in the sanity block;
    asserted here because this analysis is what publishes the spectrum."""
    results = collected_run["summary"]["results"]
    block = results["latent"]

    total = sum(block["health"]["kld_per_dimension"])
    assert total == pytest.approx(results["readouts"]["source_conditioned_kl_raw"], rel=1e-5)
    assert block["health"]["active_dims"] <= block["health"]["d_z"]


@pytest.mark.slow
def test_the_real_run_reports_every_diagnostic_with_its_meaning(collected_run) -> None:
    rows = collected_run["summary"]["results"]["latent"]["diagnostics"]

    assert {row["metric"] for row in rows} == {
        name for name, _ in latent_analysis.DIAGNOSTIC_COLUMNS
    }
    for row in rows:
        assert row["meaning"], f"{row['metric']} reports a number with no statement of what it is"
        assert row["n"] == collected_run["summary"]["results"]["n_recordings"]


@pytest.mark.slow
def test_the_verdict_carries_the_margin_that_defines_pinned(collected_run) -> None:
    r"""A fraction of "within the margin" is meaningless without the margin: the bound is a
    sigmoid, so nothing ever reaches a clamp exactly and the number is *entirely* a statement
    about how close counts. Read off a real run, where the margin comes from the checkpoint."""
    verdicts = {
        verdict["name"]: verdict
        for verdict in collected_run["summary"]["results"]["verdicts"]
    }
    values = verdicts["prior_variance_not_pinned"]["values"]
    bounds = collected_run["summary"]["collection"]["bounds"]

    assert values["clamp_margin_nats"] == pytest.approx(bounds["logvar_margin"])
    assert bounds["logvar_margin"] == pytest.approx(
        LOGVAR_FLOOR_MARGIN_FRAC * (bounds["logvar_clamp"][1] - bounds["logvar_clamp"][0])
    )
    assert "floor_frac" in values and "max_frac" in values
