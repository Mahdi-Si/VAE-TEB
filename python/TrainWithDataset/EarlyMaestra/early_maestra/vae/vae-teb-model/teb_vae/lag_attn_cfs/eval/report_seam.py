r"""The run's reporting surface: fail-soft steps, ``summary.json``, and the three summary blocks.

The mechanism is the shared one, bound rather than forked. What an evaluation run needs from a
reporting layer -- a step wrapper that captures a failure and continues, a serialiser that turns a
NaN into ``null`` instead of into a token no other JSON parser accepts, and a by-group emitter
that never raises -- says nothing about which model produced the numbers, and two copies of that
arithmetic would be two chances for a summary written by one package to disagree with a summary
written by the other.

Three properties of that mechanism are load-bearing and are stated here because a future edit
could take any of them away without anything obviously breaking:

* :meth:`Report.step` catches ``Exception`` and nothing wider. ``KeyboardInterrupt`` and
  ``SystemExit`` derive from ``BaseException``, so Ctrl-C still stops a multi-hour run rather
  than being recorded as a failed analysis the run then continues past.
* The **full traceback** is captured, not ``str(exc)``. On an unattended run the traceback is the
  entire debugging surface, and ``KeyError: 'mu_full'`` alone names none of the dozen call sites
  that could have produced it.
* :func:`json_safe` runs *before* serialisation rather than as ``json.dump(default=...)``, which
  is consulted only for types the encoder does not already recognise -- and ``float('nan')`` is
  recognised.

The grouped variants are bound here rather than written per analysis for the reason the shared
implementation gives: a single-group split must be a recorded skip rather than a one-violin
figure inviting a comparison there is nothing to compare against, an unlabelled group column is
the ordinary case on the healthy-only pretraining split and must not raise, and the two grouping
axes differ only in which column they read. The pooled output is never touched.

**What is written here is the content, not the mechanism.** Three blocks:

* The **headline** is a flat registry of ``(name, path into results)``. Its purpose is exactly
  its constraint: a number that is not registered here is invisible to the acceptance gate and to
  the arm tables, which read this block and nothing else. It carries **two** ``pred_gap`` columns
  under names that say which is which, because the Monte Carlo marginalised score and the
  training-path single-draw score are different estimators of the same quantity and a bare
  ``pred_gap`` would leave a reader to guess.
* The **sanity** block is the run's self-consistency checks, three-valued, with a ``warning``
  flag. It deliberately does **not** change the exit code: the exit code says a *step* raised,
  and a run whose every step succeeded can still be one nobody should quote a number from. That
  asymmetry is why an offline acceptance gate exists separately.
* The **verdicts** are the model's own acceptance criteria and come from the readout module; what
  happens here is only their promotion into the headline, driven by the same registry that
  decides their order.

**Forked from** ``teb_vae/lag_attn_rws/eval/report_seam.py``. The mechanism above is identical --
every binding in the first half of this module is the same shared object the sibling binds -- and
the content diverges in exactly three places, each stated where it is defined: the headline
registry carries no frequency-domain entries and a per-*coefficient* calibration gain rather than
a per-raw-sample one, the verdict registry carries two more, and the sanity block carries no
cross-spectral checks. All three follow from one fact about this cell -- the forecast is
$H \cdot C_{\mathrm{keep}}$ wavelet coefficients rather than a raw window -- and each is entered in
``divergences.json`` beside the code.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_cfs.causal_warmup import ALIGNMENT_DELAY_FACTOR
from teb_vae.lag_attn_cfs.eval import lag_shape
from teb_vae.lag_attn_cfs.eval.lag_axis import compensated_seconds_axis
from teb_vae.lag_attn_cfs.eval._reuse import report

#: The written summary's filename. Bound rather than restated so the runner, the acceptance gate
#: and the arm comparison cannot disagree about which file a run leaves behind.
SUMMARY_FILENAME = report.SUMMARY_FILENAME

#: The per-step heartbeat, rewritten as each analysis finishes. A run killed outright -- rather
#: than failing -- leaves no summary at all, and this is then the only record of how far it got.
STEPS_FILENAME = "steps.json"

#: Outcome of one analysis step, and the accumulator that owns the step wrapper and the write.
StepRecord = report.StepRecord
Report = report.Report

#: Serialisation. See the module docstring for why it is applied before ``json.dump`` rather than
#: handed to it.
json_safe = report.json_safe

#: Grouped variants: the long-form ``(group, metric, n, mean, q25, median, q75)`` table.
summarise_by_group = report.summarise_by_group


def emit_grouped_variants(frame: Any, directory: Any, **kwargs: Any) -> Dict[str, Any]:
    """Write this package's by-group tables and figures, in its own cohort order and palette.

    The mechanism is the shared emitter's; what is added here is the two presentation decisions
    this package makes and the sibling does not, supplied together because a figure whose violins
    are ordered one way and coloured by another convention is worse than either alone:

    * cohorts run **HIE, acidosis, healthy** and the eight subgroups in the reverse of their
      canonical order,
      from :func:`~teb_vae.lag_attn_cfs.eval.cohort.ordered_groups`, rather than alphabetically --
      which would put ``acidosis`` first on every class figure and interleave the classes on every
      subgroup one;
    * they are coloured green / amber / red by severity, from
      :func:`~teb_vae.lag_attn_cfs.eval.figures_seam.group_colors`.

    Both are passed rather than reimplemented, so the skip rules, the counts and the record's
    shape stay the shared ones and cannot drift from the sibling's.

    Args:
        frame: The analysis's per-recording frame.
        directory: The analysis's output directory.
        **kwargs: Forwarded to :func:`~teb_vae.lag_attn.eval.report.emit_grouped_variants`.

    Returns:
        Its record, unchanged in shape.
    """
    # Imported here rather than at module scope: this module is the runner's reporting surface and
    # is imported for ``Report`` alone on paths that draw nothing, and ``figures_seam`` costs
    # matplotlib. The shared emitter takes its own figure import the same way and for the reason.
    from teb_vae.lag_attn_cfs.eval import figures_seam
    from teb_vae.lag_attn_cfs.eval.cohort import ordered_groups

    return report.emit_grouped_variants(
        frame,
        directory,
        order_groups=lambda groups, axis: ordered_groups(groups, axis),
        group_palette=figures_seam.group_colors,
        **kwargs,
    )

#: Derived blocks whose content is model-agnostic: what each analysis saw, which caps never fired,
#: and every file the run emitted.
build_coverage = report.build_coverage
build_manifest = report.build_manifest
check_inert_caps = report.check_inert_caps

#: Path resolution for the headline registry: ``None`` when a path does not resolve, rather than a
#: raise. Bound rather than restated so one walker serves both packages' headline blocks -- a path
#: that resolves in one must resolve in the other.
_dig = report._dig

#: Verdict when a check cannot be evaluated from what the run produced. Distinct from a pass: "the
#: run did not carry what this needs" and "this held" are different statements.
INCONCLUSIVE = report.INCONCLUSIVE


# =============================================================================
# The headline block
# =============================================================================
#: Headline scalars, as ``(name, path into results)``. Flattened out of the per-analysis blocks so
#: a reader -- or a ``pandas`` merge across two arms -- does not need to know which analysis
#: produced which number. A path that does not resolve yields ``None``.
#:
#: Both ``pred_gap`` estimators are here under names that say which is which. The **headline** one
#: is the Monte Carlo marginalised score, $D = -[\operatorname{logsumexp}_r(-D_r) - \log K]$ --
#: the log of the average likelihood. The training-path column is one latent draw scored through
#: the objective's own functions, and it sits beside the headline as the parity check rather than
#: as a second answer.
#:
#: Only the **unfloored** KL appears. ``source_conditioned_kl_train`` has free bits applied per
#: dimension per step before summing, so it exceeds the raw value by construction and hides a
#: collapsed source pathway; the shipped ``free_bits: 0.0`` makes the two coincide today, which is
#: exactly why the distinction lives in code rather than in an observation.
HEADLINE_SCALARS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("pred_gap_mc_nats", ("readouts", "mc_pred_gap")),
    ("pred_gap_train_path_nats", ("readouts", "pred_gap")),
    # The same answer as a proportion. Registered because nats do not compare across checkpoints
    # whose block scores differ in scale, which is exactly what an arm table asks them to do -- and
    # because "the source removed 4% of the forecast error" is a sentence a reader can check
    # against a trace, where "3.1 nats per anchor" is not. Both spaces are here because they fail
    # differently: the error-space pair is blind to the forecast's variance, and the
    # likelihood-space one is the headline score itself and so inherits its every property.
    ("pred_gap_rmse_pct", ("coupling", "pred_gap_percent", "headline", "pred_gap_rmse_pct")),
    ("pred_gap_mse_pct", ("coupling", "pred_gap_percent", "headline", "pred_gap_mse_pct")),
    (
        "pred_gap_mc_likelihood_pct",
        ("coupling", "pred_gap_percent", "headline", "pred_gap_mc_likelihood_pct"),
    ),
    ("d_base_mc_nats", ("readouts", "mc_nll_base_block")),
    ("d_full_mc_nats", ("readouts", "mc_nll_full_block")),
    ("d_shuffled_mc_nats", ("readouts", "mc_nll_shuffled_block")),
    # D_shuffled - D_base, per recording and then averaged, from the permutation control. The one
    # predictive comparison here that is not referenced against the base branch, and therefore the
    # one that still carries signal when the predictive gain is negative: it changes only the
    # source, holding prior, decoder and latent geometry fixed. Read from the analysis rather than
    # differenced from the two block scalars above, because the paired per-recording construction
    # is not the difference of two pooled means and only the paired one has an interval behind it.
    #
    # It resolves through a *keyed* path rather than out of the analysis's ``penalties`` list:
    # this block is assembled by walking key paths, which is exactly why the two penalties beside
    # it have never appeared here.
    ("source_margin_nats", ("perm_control", "source_margin_nats")),
    ("source_conditioned_kl_raw_nats", ("readouts", "source_conditioned_kl_raw")),
    # The prior scale rate, beside the divergence it shares a support and a unit with. Registered
    # because it is the quantity the anchor weight is calibrated against and an arm table
    # comparing weights reads this block and nothing else -- and because it is reported under
    # every objective, so a run that never weighted it still says what its prior was doing.
    ("prior_rate_nats", ("readouts", "prior_rate")),
    ("kl_total_nats", ("latent_health", "kl_total_nats")),
    ("kl_active_dims", ("latent_health", "active_dims")),
    ("kl_top_dimension_share", ("latent_health", "top_dimension_share")),
    ("kl_argmax_lag_step", ("lag", "kl_argmax_lag_step")),
    ("kl_argmax_lag_step_support_corrected", ("lag", "kl_argmax_lag_step_support_corrected")),
    ("kl_lag_compensated_seconds", ("lag", "kl_lag_compensated_seconds")),
    ("attention_argmax_lag_step", ("lag", "attention_argmax_lag_step")),
    # The attention argmax that survives the truncation bias, and the entropy beside the ceiling
    # it can actually reach. Registered because an arm comparison asking "did the lag structure
    # move" reads this block and nothing else -- and because an entropy quoted against $\log L$
    # rather than against the attainable ceiling is the misreading these two exist to foreclose.
    (
        "attention_argmax_lag_step_untruncated",
        ("lag", "attention_argmax_lag_step_untruncated"),
    ),
    ("attention_entropy_nats", ("readouts", "attention_entropy_nats")),
    ("attention_entropy_attainable_nats", ("readouts", "attention_entropy_attainable_nats")),
    ("delta_mu_rms", ("readouts", "delta_mu_rms")),
    # The bound-variance detectors. The prior's floor fraction is the one that decides whether the
    # KL above it is a rate or a number divided by a clamp, and it is invisible to every other
    # readout in this block -- the decoder variances beside it stay healthy while it fails.
    ("mean_logvar_prior", ("readouts", "mean_logvar_prior")),
    ("logvar_prior_floor_frac", ("readouts", "logvar_prior_floor_frac")),
    ("mean_logvar_full", ("readouts", "mean_logvar_full")),
    ("logvar_full_floor_frac", ("readouts", "logvar_full_floor_frac")),
    ("logvar_full_ceil_frac", ("readouts", "logvar_full_ceil_frac")),
    # The observation model's calibration, pooled over scored coefficients rather than chained per
    # recording -- see the calibration analysis for why. ``None`` under a likelihood with no
    # predictive distribution, which is what an 'mse' checkpoint produces.
    ("calibration_mean_standardised_sq", ("calibration", "mean_standardised_sq")),
    ("calibration_pit_max_cdf_deviation", ("calibration", "pit", "max_cdf_deviation")),
    # Per **coefficient**, not per raw sample: the scored unit here is one of the
    # $H \cdot C_{\mathrm{keep}}$ wavelet coefficients of a forecast block, and there is no raw
    # sample anywhere in this pipeline for a gain to be per. The rename is not cosmetic -- the two
    # denominators differ by a factor of three at the shipped geometry, so a column carried across
    # under the sibling's name would be silently non-comparable with the sibling's number.
    ("calibration_nll_gain_per_coefficient", ("calibration", "nll", "gain_per_coefficient")),
    # No frequency-domain entries. ``coherence`` is not ported at all: the stored coefficients are
    # moduli, so phase agreement, group delay and the residual's three-way split have no analogue
    # here at any window length. The frequency-resolved readout this pipeline *does* have is
    # ``spectral_skill``, which resolves the forecast gap by the band of the target coefficient --
    # and it registers its own scalars through the binding's ``headline_scalars`` rather than
    # here, because this tuple's every path must resolve on a run of every model that uses this
    # pipeline.
    ("same_recording_pairing_rate", ("controls", "same_recording_pairing_rate")),
    ("n_samples", ("n_samples",)),
    ("n_recordings", ("n_recordings",)),
)

#: Verdict names promoted into the headline, resolved out of ``results['verdicts']`` by name.
#: Restated rather than imported from the readout module, which pulls in ``torch`` and the whole
#: network; ``tests/test_eval_report.py`` pins this equal to ``metrics.PROMOTED_VERDICTS``, so a
#: registry change fails a test instead of silently dropping a criterion from the headline.
#:
#: **Ten here against the sibling's eight**, and the two additions are the two verdicts only this
#: cell can have:
#:
#: ``coupling_exceeds_availability_clock``
#:     The source availability pattern is a deterministic function of $t$, identical in every row
#:     of a batch, and it enters $q(z \mid Y, U)$ but not $p(z \mid Y)$ -- so the posterior can be
#:     pushed off the prior by the availability *clock* alone. No permutation of rows can remove
#:     something every row shares, which is why ``source_specificity`` cannot see this hazard and
#:     why this is a separate verdict rather than a tightening of that one.
#: ``anchor_geometry_intact``
#:     The dense anchor set and the fully warm target axis are what every number in a run is
#:     computed over. A count off by one anchor means the population moved, and nothing else in
#:     the summary would say so.
HEADLINE_VERDICTS: Tuple[str, ...] = (
    "predictive_improvement",
    "source_margin_positive",
    "source_specificity",
    "prior_carries_target_state",
    "latent_not_collapsed",
    "prior_variance_not_pinned",
    "decoder_variance_not_pinned",
    "calibration_near_nominal",
    "coupling_exceeds_availability_clock",
    "anchor_geometry_intact",
)

#: Written into the headline verbatim. The block carries two ``pred_gap`` columns and this is the
#: sentence that says which one an acceptance criterion means, inside the artifact rather than in
#: a document beside it.
PRED_GAP_CONVENTION = (
    "pred_gap_mc_nats is the headline: the Monte Carlo marginalised block score difference "
    "D_base - D_full in nats per anchor, where D = -[logsumexp_r(-D_r) - log K] is the log of "
    "the average likelihood over K latent draws. pred_gap_train_path_nats is the training-path "
    "difference the objective itself sees, reported beside it for objective parity only: under "
    "base_decode: mean the training path decodes the base branch at the prior MEAN and the full "
    "branch at one sampled latent, so that column mixes a source comparison with a "
    "decoding-policy difference and is not a second estimate of pred_gap_mc_nats. "
    "A block here is H*C_keep target coefficients -- H horizon steps by the C_keep channels the "
    "warm-up budget kept; preflight.json records this run's own horizon, target_kept_width and "
    "block_width -- and not an H*R-sample raw window: the "
    "forecast is over wavelet-modulus and phase-harmonic coefficients in the loader's z units, "
    "and there is no bpm anywhere in this pipeline. "
    "Three percentage columns restate the same finding as a proportion, each computed per "
    "recording and then averaged: pred_gap_rmse_pct and pred_gap_mse_pct are the percentage of "
    "the target-only branch's point-forecast error the source removed, in root-mean-square and "
    "mean-square respectively, and are scale-free; pred_gap_mc_likelihood_pct is "
    "100*(exp(pred_gap_mc_nats / (H*C_keep)) - 1), the extra probability density the "
    "source-conditioned forecast puts on each observed coefficient, where H*C_keep is the fixed "
    "block size rather than a per-anchor scored-coefficient count and so understates the "
    "improvement wherever forecast steps are masked. "
    "That third column is additionally BUDGET-LOCAL and is not comparable across arms: C_keep is "
    "whatever the warm-up budget decided, so two arms of this model at two budgets divide by two "
    "different numbers -- as well as being mutually unloadable checkpoints. The nats columns "
    "carry the same caveat one step removed, since a block score is a sum over the kept channels. "
    "That third column also exists only under gaussian_nll: under "
    "mse a block score is a sum of squared errors rather than a log density, so it is null there "
    "rather than exponentiated into a density ratio that would not be one. None of the three is "
    "pred_gap divided by a block score: a "
    "log score has no natural zero, D_base is legitimately negative for a sharp forecast, and "
    "that ratio would change sign with its own denominator."
)


def build_headline(
    results: Dict[str, Any],
    extra: Sequence[Tuple[str, Tuple[str, ...]]] = (),
) -> Dict[str, Any]:
    """Flatten the run's headline scalars and verdict statuses out of the per-analysis blocks.

    Args:
        results: The accumulated results.
        extra: Additional ``(name, path)`` entries, from the binding of the model being evaluated,
            resolved after the shared ones and in declaration order. Empty for a model whose every
            analysis is shared. They are appended rather than added to :data:`HEADLINE_SCALARS`
            because every path in *that* tuple must resolve on every run of every model, and a
            model-specific entry there would read as a number the others failed to produce.

    Returns:
        Name to value, with ``None`` wherever the producing analysis did not report, plus the
        ``pred_gap_convention`` sentence.

    Raises:
        ValueError: If an extra entry reuses a shared name. The extras resolve last, so a reused
            name would replace a shared reading with a model-specific one under the shared
            name -- and every arm table, every acceptance gate and every cross-model row reads
            this block by name, so the substitution would be invisible in the artifact.
    """
    collisions = sorted({name for name, _ in extra} & {name for name, _ in HEADLINE_SCALARS})
    if collisions:
        raise ValueError(
            f"a binding registers headline scalars whose names are already shared: {collisions}. "
            f"An extra scalar is an addition, never an override: name it for the analysis that "
            f"produces it, as the shared entries are named for theirs."
        )
    headline: Dict[str, Any] = {
        name: _dig(results, path) for name, path in tuple(HEADLINE_SCALARS) + tuple(extra)
    }
    by_name = {
        str(verdict.get("name")): verdict
        for verdict in (results.get("verdicts") or [])
        if isinstance(verdict, dict)
    }
    for name in HEADLINE_VERDICTS:
        headline[f"verdict_{name}"] = (by_name.get(name) or {}).get("status")
    headline["pred_gap_convention"] = PRED_GAP_CONVENTION
    return headline


# =============================================================================
# The sanity block
# =============================================================================
#: Relative tolerance for the two cross-table identities. Both sides are float64 accumulations of
#: float32 model output taken over different groupings, so exact equality is unreachable while
#: anything looser than this would stop distinguishing a rounding difference from a real one.
IDENTITY_RTOL = 1e-5

#: Absolute floor under the relative tolerance, so a quantity that is legitimately near zero -- an
#: untrained source pathway's KL -- is not compared against a tolerance of nothing.
IDENTITY_ATOL = 1e-8

#: How far, in nats, a per-anchor structural identity may miss before the run says so. Applied to
#: the **worst** anchor of the whole pass, not to a mean -- see
#: ``metrics.identity_residual_per_sample`` for why a mean is precisely the statistic that cannot
#: see the failure these guard against.
#:
#: A floor rather than the whole rule: both identities are exact in real arithmetic, so the only
#: thing they can differ by is float32 accumulation over $L = 91$ or $M = 4$ terms -- which grows
#: with the *magnitude of the KL being decomposed*, not with a fixed number of nats. At the tiny
#: geometry the measured residual is $2 \times 10^{-5}$ against a KL of $11$ nats; at a
#: production KL an order of magnitude larger it would cross a flat $10^{-4}$ while saying
#: nothing except that the KL is large. So the floor is lifted proportionally, by the same
#: relative tolerance the cross-table check uses and for the same reason.
#:
#: Nothing is lost in detection power: the mechanism these guard against -- dropout on the
#: attention probabilities -- misses by a *fraction* of the KL, which is four orders of magnitude
#: above either bound.
IDENTITY_TOLERANCE = 1e-4

#: The structural identities checked every run, as ``name -> (residual key, what it means)``. Both
#: are properties of the model layer that the *evaluation* re-measures on the run's own numbers:
#: a model test proves them on a fixture, and this proves them on the checkpoint and the data that
#: produced the summary a reader is holding.
LAG_IDENTITIES: Dict[str, Tuple[str, str]] = {
    "lag_map_sums_to_kl": (
        "lag_map_sums_to_kl_max_abs_nats",
        "the per-lag KL attribution sums over lags to the per-step KL it attributes",
    ),
    "per_head_kl_sums_to_kl": (
        "per_head_kl_sums_to_kl_max_abs_nats",
        "the per-head KL sums over heads to the per-step KL it decomposes",
    ),
    "null_lag_map_sums_to_kl": (
        "null_lag_map_sums_to_kl_max_abs_nats",
        "the source-null arm's per-lag attribution sums over lags to the null per-step KL, "
        "which is what makes the clock-excess profile the per-lag decomposition of "
        "coupling_minus_clock rather than a difference of two differently normalised readings",
    ),
}

#: Named in the failure message of both checks above. Attention dropout is the one mechanism that
#: breaks either identity -- it scales the attention probabilities by $1/(1-p)$ and zeroes a
#: random subset, so each head's weights no longer sum to one and the attribution holds only in
#: expectation. The model builds its attention at zero dropout for exactly this reason, so a
#: violation here points at that line before it points anywhere else.
_IDENTITY_LIKELY_CAUSE = (
    "the most likely cause is dropout on the attention probabilities: it rescales them so each "
    "head's weights no longer sum to one, which makes the attribution hold only in expectation. "
    "The model builds LagCrossAttention at dropout=0.0 for that reason, and an evaluation that "
    "forgot to put the model in eval() would produce the same symptom"
)

#: Per-anchor column -> the per-sample column it must average to, over the anchors the per-anchor
#: table holds. Every pair is a plain mean because the per-sample reduction weights each
#: contributing anchor equally and the anchor table holds exactly the contributing anchors; a
#: weighted reduction would not recombine this way and does not belong here.
#:
#: ``kld_per_t`` is on the list because ``kl_mask`` is derived *from* the forecast mask rather than
#: restated, so the KL support and the reconstruction support are the same anchor set by
#: construction -- and this check is what would notice if a later edit made them two rules again.
#:
#: The three warm-up tertiles are on the list for a second reason beside recombination, and it is
#: this cell's own: they are a **decomposition** of ``pred_gap`` over the $C_{\mathrm{keep}}$ kept channels, so
#: they must both average back per anchor (checked here) and sum to ``pred_gap`` over the same
#: denominator (checked by the ``warmup`` analysis). Only the second makes them three
#: parts of one number rather than three unrelated readouts; only the first makes each of them a
#: quantity the per-recording chain may be run on. A column the pass has not produced yet is
#: skipped rather than raising, so listing them before the collection pass produces them costs
#: nothing.
RECOMBINED_COLUMNS: Dict[str, str] = {
    "nll_base_block": "nll_base_block",
    "nll_full_block": "nll_full_block",
    "pred_gap": "pred_gap",
    "pred_gap_warm_lo": "pred_gap_warm_lo",
    "pred_gap_warm_mid": "pred_gap_warm_mid",
    "pred_gap_warm_hi": "pred_gap_warm_hi",
    "mc_nll_base_block": "mc_nll_base_block",
    "mc_nll_full_block": "mc_nll_full_block",
    "mc_pred_gap": "mc_pred_gap",
    "kld_per_t": "source_conditioned_kl_raw",
}


def _verdict(passed: bool, detail: str, **numbers: Any) -> Dict[str, Any]:
    """Build one check's record."""
    return {"verdict": "pass" if passed else "fail", "detail": detail, **numbers}


def _inconclusive(detail: str, **numbers: Any) -> Dict[str, Any]:
    """Build the record for a check the run could not evaluate."""
    return {"verdict": INCONCLUSIVE, "detail": detail, **numbers}


def _agrees(left: float, right: float) -> bool:
    """Whether two numbers agree within the identity tolerance."""
    return abs(left - right) <= max(IDENTITY_ATOL, IDENTITY_RTOL * max(abs(left), abs(right)))


def _reported(value: float) -> float:
    """Round a measured floating-point residual to what it can actually claim.

    A group mean's last bits depend on the frame's internal layout, so the same table reduced in
    memory and after a parquet round trip disagree around the twelfth digit. That is noise, but it
    lands in ``results`` -- the block two runs of one checkpoint must compare **equal** -- so the
    residual is reported to six significant digits, which is far more than it can distinguish and
    exactly as much as a reader asking "how far off was it?" needs.
    """
    return float(f"{float(value):.6g}")


def check_kl_identity(results: Dict[str, Any]) -> Dict[str, Any]:
    r"""The per-dimension KL must sum to the KL it decomposes.

    ``latent_health.kl_total_nats`` is $\sum_d \bar K_d$ over the per-dimension spectrum and
    ``readouts.source_conditioned_kl_raw`` is $\bar K$ itself. They are equal only if both travel
    the identical aggregation chain -- per-sample support-weighted, then per recording, then
    across recordings. Reduced per *batch* instead, as a stack-and-mean over batches does, the
    spectrum weights every batch equally however many anchors or recordings it held, and the two
    numbers then disagree by an amount that depends on the batch composition and on nothing else.
    """
    total = (results.get("latent_health") or {}).get("kl_total_nats")
    raw = (results.get("readouts") or {}).get("source_conditioned_kl_raw")
    if total is None or raw is None:
        return _inconclusive("the run reported no latent spectrum or no raw KL")
    total, raw = float(total), float(raw)
    if not (math.isfinite(total) and math.isfinite(raw)):
        return _inconclusive("the KL readouts are not finite", kl_total_nats=total, kl_raw=raw)
    agrees = _agrees(total, raw)
    return _verdict(
        agrees,
        "the per-dimension KL sums to the raw source-conditioned KL" if agrees
        else "the per-dimension KL does not sum to the raw source-conditioned KL, so one of the "
             "two is not on the per-recording aggregation chain",
        kl_total_nats=total, source_conditioned_kl_raw=raw,
        abs_difference=abs(total - raw), rtol=IDENTITY_RTOL,
    )


def check_per_anchor_recombines(
    per_sample: Optional[Any] = None, per_anchor: Optional[Any] = None
) -> Dict[str, Any]:
    """The per-anchor table must average back into the per-sample table, column by column.

    The two durable tables are written from one forward pass, so a disagreement here is not a
    numerical curiosity: it means the anchor rows are not the rows the per-sample columns were
    reduced over, and every later analysis that reads one table while quoting a headline from the
    other is describing a different population.

    Args:
        per_sample: The per-sample table, keyed by ``sample_index``.
        per_anchor: The per-anchor table, carrying the same key.

    Returns:
        The check record, naming the worst-disagreeing column and by how much.
    """
    if per_sample is None or per_anchor is None or len(per_anchor) == 0:
        return _inconclusive("the run carried no per-anchor table to recombine")
    if "sample_index" not in per_sample.columns or "sample_index" not in per_anchor.columns:
        return _inconclusive("the tables carry no sample_index to join on")

    # In float64, and explicitly. The anchor columns arrive as float32 -- the model's own dtype --
    # and a float32 group mean over thousands of anchors is exactly the rounding this check exists
    # to distinguish from a real disagreement. It is also not path-independent: the same table
    # reduced in memory and after a parquet round trip differed in the twelfth digit, which put a
    # non-reproducible number into a block two runs of one checkpoint must compare equal.
    shared = [name for name in RECOMBINED_COLUMNS if name in per_anchor.columns]
    if not shared:
        return _inconclusive(
            "no column appears on both tables", n_anchor_rows=int(len(per_anchor))
        )
    means = (
        per_anchor[["sample_index", *shared]]
        .astype({name: np.float64 for name in shared})
        .groupby("sample_index")
        .mean()
    )
    indexed = per_sample.set_index("sample_index")
    differences: Dict[str, float] = {}
    scale = 0.0
    for anchor_column, sample_column in RECOMBINED_COLUMNS.items():
        if anchor_column not in means.columns or sample_column not in indexed.columns:
            continue
        left = np.asarray(means[anchor_column].reindex(indexed.index), dtype=np.float64)
        right = np.asarray(indexed[sample_column], dtype=np.float64)
        # A segment that scored no anchors is NaN on the sample table and absent from the anchor
        # table, which is the same exclusion seen from both sides rather than a disagreement.
        both = np.isfinite(left) & np.isfinite(right)
        if not bool(both.any()):
            continue
        differences[sample_column] = float(np.max(np.abs(left[both] - right[both])))
        scale = max(scale, float(np.max(np.abs(right[both]))))
    if not differences:
        return _inconclusive(
            "no column appears on both tables", n_anchor_rows=int(len(per_anchor))
        )

    # One tolerance for every column, set by the largest quantity on the table rather than by each
    # column's own magnitude. The two sides reduce the same float32 values in different orders and
    # different dtypes, so the disagreement is bounded by the rounding of the *block scores* --
    # and ``pred_gap`` is their difference, hundreds of nats cancelling into ones. Scaled to its
    # own size it could never pass, and the check would fail on every healthy run.
    tolerance = max(IDENTITY_ATOL, IDENTITY_RTOL * scale)
    offending = sorted(name for name, gap in differences.items() if gap > tolerance)
    return _verdict(
        not offending,
        f"all {len(differences)} shared column(s) recombine within {tolerance:.3g}"
        if not offending
        else f"column(s) {offending} do not recombine: the anchor rows are not the rows the "
             f"per-sample columns were reduced over",
        columns_checked=sorted(differences),
        max_abs_difference={
            name: _reported(differences[name]) for name in sorted(differences)
        },
        tolerance=_reported(tolerance),
        tolerance_scale=_reported(scale),
        n_anchor_rows=int(len(per_anchor)),
    )


def identity_tolerance_for(scale: Optional[float]) -> float:
    """The tolerance a structural identity is judged at, given the quantity it decomposes.

    Args:
        scale: The magnitude the identity is over -- the headline KL. ``None`` or non-finite
            falls back to the bare floor, which is the conservative direction.

    Returns:
        :data:`IDENTITY_TOLERANCE`, lifted to :data:`IDENTITY_RTOL` of ``scale`` where that is
        larger. See :data:`IDENTITY_TOLERANCE` for why a flat bound is wrong here.
    """
    if scale is None:
        return IDENTITY_TOLERANCE
    value = float(scale)
    if not math.isfinite(value):
        return IDENTITY_TOLERANCE
    return max(IDENTITY_TOLERANCE, IDENTITY_RTOL * abs(value))


def check_lag_identity(results: Dict[str, Any], name: str) -> Dict[str, Any]:
    r"""One of the structural lag identities, measured on this run's own worst anchor.

    All of them hold exactly in real arithmetic -- $\sum_\ell \widetilde K_{t,\ell} = K_t$ because
    each head's attention sums to one, $\sum_m K^{(m)}_t = K_t$ because the latent groups are
    head-aligned, and the same pair on the source-null arm because that arm re-poses the posterior
    through the identical head-structured attribution -- so the residual reported here is float32
    accumulation and nothing else. A residual above :data:`IDENTITY_TOLERANCE` means one of those
    structural facts stopped being true, and the message names the mechanism that would do it.

    The null arm's identity is checked separately from the matched one rather than assumed to
    follow from it: it is a **different** attention, computed against a zeroed source encode, so
    its rows summing to one is its own property. The clock-excess profile is the difference of the
    two attributions, and a violation on either side would leave that difference something other
    than the per-lag decomposition of ``coupling_minus_clock``.

    Args:
        results: The accumulated results, read for the lag block's residuals.
        name: Which identity, a key of :data:`LAG_IDENTITIES`.

    Returns:
        The check record, carrying the measured residual and the tolerance it was judged against.
    """
    key, meaning = LAG_IDENTITIES[name]
    residual = ((results.get("lag") or {}).get("identity_residuals") or {}).get(key)
    if residual is None:
        return _inconclusive(f"the run reported no residual for {key}")
    residual = float(residual)
    if not math.isfinite(residual):
        return _inconclusive(
            f"the measured residual for {key} is not finite", max_abs_residual_nats=residual
        )
    scale = (results.get("readouts") or {}).get("source_conditioned_kl_raw")
    tolerance = identity_tolerance_for(scale)
    holds = residual <= tolerance
    return _verdict(
        holds,
        f"{meaning}, to {tolerance:.3g} nats on the worst anchor of the pass"
        if holds
        else f"{meaning} -- but not on this run: the worst anchor is off by "
             f"{residual:.3g} nats against a tolerance of {tolerance:.3g}. "
             f"{_IDENTITY_LIKELY_CAUSE}",
        max_abs_residual_nats=_reported(residual),
        tolerance_nats=_reported(tolerance),
        tolerance_floor_nats=IDENTITY_TOLERANCE,
        tolerance_scale_nats=None if scale is None else _reported(float(scale)),
        # A maximum over every scored anchor of every segment, so one bad recording cannot be
        # averaged away by the rest.
        statistic="max over anchors and samples",
    )


#: The identity a lag on this axis is read under, written with its symbols. Interpolated into the
#: near-edge message rather than stated as prose, because what makes a pin at the floor
#: *censoring* rather than inertness is arithmetic a reader has to be able to redo: the reported
#: lag is the smallest one the window carries, and every physical delay shorter than what that
#: lag encodes lands below it and is reported there whatever the model found.
#:
#: Written on the canonical stored timeline: there is no dataset-shift term, and ``tau_y_ref`` is
#: the clock of the SCORED target (``tau_min`` under a physical forecast clock), not the target
#: encoder's input reference.
PHYSICAL_LAG_IDENTITY = (
    f"tau_phys(l, h) = {SECONDS_PER_STEP:g}*(l + 1 + h) "
    f"+ {ALIGNMENT_DELAY_FACTOR:g}*(tau_u_ref - tau_y_ref)"
)


def check_argmax_lag(results: Dict[str, Any]) -> Dict[str, Any]:
    r"""The KL attribution must have a readable shape, and its peak must not sit on a censoring edge.

    **Both ends of the window censor, and the check is symmetric in them.** An argmax at the
    largest attainable lag means the true maximum may lie beyond $L$. An argmax at the *smallest*
    attainable lag means exactly the mirror image: by the identity
    $\tau^{\mathrm{phys}}_{\ell,h} = \Delta(\ell + 1 + h) + \kappa(\tau^u_{\mathrm{ref}}
    - \tau^y_{\mathrm{ref}})$ on the canonical stored timeline, every physical delay shorter than
    what that lag encodes is reported *at* it, because the window carries no bin below it. On this
    cell's geometry a $20$-$60$ s physiological delay sits below lag $0$ at most horizon steps
    under the single reference and grazes lag $0$ at the far horizon step under the shipped dual
    reference ($\ell = -0.1$ at $h = 29$), so a pin
    at the floor is the readout hitting a wall rather than the machinery failing to look back --
    and reading it as inertness is a conclusion the geometry does not support.

    Both edges are read from the per-lag anchor counts rather than taken as $0$ and $L - 1$: a lag
    no anchor contributed to is not attainable at all, which at short sequence lengths removes the
    top of the window entirely and can lift the floor off zero.

    **What "the machinery is alive" is judged on instead.** The shape vocabulary the pipeline
    already computes: a profile whose peak is not distinguishable from its bulk, or one that is
    almost entirely exact zeros, has an argmax that names a bin rather than a lag -- and that is a
    FAIL at either edge or in the middle, because there is no measurement to censor. The per-head
    entropies travel beside it as the evidence a reader needs to tell a confident wrong answer from
    a flat one.

    So the three outcomes are: FAIL on a degenerate profile or a far-edge pin, INCONCLUSIVE on a
    near-edge pin with the arithmetic named, PASS on a profile that peaks strictly inside the
    attainable window. An ideal model at this geometry can reach the last of those; today's cannot.
    """
    lag = results.get("lag")
    if not isinstance(lag, dict) or not lag:
        return _inconclusive("the run reported no lag summary")

    argmax = lag.get("kl_argmax_lag_step")
    counts = list(lag.get("kl_lag_anchor_counts") or [])
    attainable = [index for index, count in enumerate(counts) if float(count) > 0.0]
    if argmax is None or not attainable:
        return _inconclusive("the lag summary carries no argmax or no per-lag support")

    argmax = int(argmax)
    floor, ceiling = int(min(attainable)), int(max(attainable))
    # The support-corrected profile where the run carries one, because that is the profile whose
    # argmax the rest of this pipeline quotes; the raw one is the fallback rather than a second
    # opinion, and which was used is emitted so the two cannot be confused.
    corrected = list(lag.get("kl_lag_profile_support_corrected") or [])
    profile = corrected or list(lag.get("kl_lag_profile") or [])
    shape = lag_shape.degeneracy(profile)
    above_half = lag_shape.mass_above(profile)
    entropies = [
        float(value) for value in (lag.get("attention_entropy_per_head_nats") or [])
    ]
    facts: Dict[str, Any] = {
        "kl_argmax_lag_step": argmax,
        "attainable_lag_floor": floor,
        "attainable_lag_ceiling": ceiling,
        "n_lags": len(counts),
        # Reported rather than judged: where the two argmaxes disagree, the difference *is* the
        # short-lag bias the support correction exists to remove, which is a finding about the
        # profile and not a failure of the run.
        "kl_argmax_lag_step_support_corrected": lag.get("kl_argmax_lag_step_support_corrected"),
        "profile_judged": "support_corrected" if corrected else "raw",
        "peak_degenerate": bool(shape["degenerate"]),
        "peak_to_median": shape["peak_to_median"],
        "zero_fraction": shape["zero_fraction"],
        "mass_above_half_peak": above_half.get("share"),
        "n_bins_above_half_peak": above_half.get("n_bins"),
        "attention_entropy_per_head_nats": entropies,
    }

    # First, because it is the only branch that says there is nothing to censor: a profile with no
    # shape has an argmax that names a bin, and a censoring verdict on it would report a geometry
    # limit where the measurement itself is absent.
    if shape["degenerate"]:
        return _verdict(
            False,
            "the KL attribution has no readable shape, so its argmax names a bin rather than a "
            "lag: " + "; ".join(shape["reasons"]),
            **facts,
        )
    if ceiling > floor and argmax >= ceiling:
        return _verdict(
            False,
            f"the argmax lag sits at the largest attainable lag ({ceiling}), so the peak is "
            f"against the window edge and the true maximum may lie beyond it",
            **facts,
        )
    if argmax <= floor:
        return _inconclusive(
            f"the argmax lag sits at the smallest attainable lag ({floor}), which is a CENSORING "
            f"edge and not inertness: under {PHYSICAL_LAG_IDENTITY}, every physical delay shorter "
            f"than the one lag {floor} encodes is reported at lag {floor}, because the window "
            f"carries no bin below it. The profile is not degenerate (peak-to-median "
            f"{shape['peak_to_median']:.3g}, {float(above_half['share']):.3f} "
            f"of its mass above half the peak), so the machinery is alive and the lag it would "
            f"report is outside what this geometry can express. Read the per-head profiles and "
            f"the shape vocabulary, and move the window with the alignment references rather than "
            f"with the model.",
            **facts,
        )
    return _verdict(
        True,
        f"the KL attribution peaks strictly inside the attainable lag window "
        f"[{floor}, {ceiling}]",
        **facts,
    )


def build_lag_heads(results: Dict[str, Any]) -> Dict[str, Any]:
    r"""Describe each attention head's own lag profile, beside the pooled one.

    **The pooled profile is a mean over $M$ distributions that need not agree**, and on this
    geometry it is also the statistic that cannot move: its argmax sits at the window's near
    censoring edge on every arm this family has measured, because the three heads that peak there
    outweigh the one that does not. A comparison between two alignment arms, or between two K/V
    memories, is expressible only per head -- a head peaking inside the readable band while the
    pooled argmax stays at the floor is a real finding the pooled row reports as nothing.

    Everything here is *already computed*: the per-head profiles and the attention entropies come
    off the collection pass, and the shape statistics are the same
    :mod:`~teb_vae.lag_attn_cfs.eval.lag_shape` vocabulary the pooled check reads. What this adds
    is that they reach the summary and the console, which is the surface an operator holds.

    The entropies are the attention's own and the KL split is the posterior's, so both travel: a
    head can be sharp in where it attends and carry almost none of the divergence, and the pair is
    what separates "this head looks at lag 33" from "this head is what moved the belief".

    Args:
        results: The accumulated results, read for the ``lag`` block.

    Returns:
        One record per head with its argmax, peak width, concentration, near and far mass,
        degeneracy flag, attention entropy and share of the KL -- plus the pooled row under the
        same keys, so the two are read in one table. Empty when the run carried no per-head
        profiles, which is a run whose collection pass did not reach the attention rather than a
        model without heads.
    """
    lag = results.get("lag")
    if not isinstance(lag, dict) or not lag:
        return {}
    per_head = [list(profile) for profile in (lag.get("attention_lag_profile_per_head") or [])]
    if not per_head:
        return {}

    entropies = list(lag.get("attention_entropy_per_head_nats") or [])
    kld_per_head = [float(value) for value in (lag.get("kld_per_head") or [])]
    kld_total = sum(kld_per_head)

    # The compensated axis every shape statistic is stated on. Built once from the run's own delay
    # so the near and far windows are the same offsets the lag analyses use; a per-row axis would
    # make two heads' ``near_mass`` mean two different things.
    seconds = compensated_seconds_axis(len(per_head[0]), int(lag.get("delay_steps") or 0))

    def _row(name: str, profile: List[float], head: Optional[int]) -> Dict[str, Any]:
        statistics, _counts = lag_shape.profile_statistics(
            np.asarray([profile], dtype=np.float64), seconds
        )
        above_half = lag_shape.mass_above(profile)
        peak = lag_shape.peak_width(profile)
        return {
            "profile": name,
            "argmax_lag_step": None if peak.get("argmax") is None else int(peak["argmax"]),
            "peak_width_bins": peak.get("width_bins"),
            "mass_above_half_peak": above_half.get("share"),
            "near_mass": float(statistics["near_mass"][0]),
            "far_mass": float(statistics["far_mass"][0]),
            "entropy_nats": float(statistics["entropy"][0]),
            "degenerate": bool(statistics["peak_degenerate"][0]),
            # The attention's own entropy, from the collection pass, rather than the entropy of
            # this profile: the first is measured per anchor and averaged, the second is the
            # entropy of the average, and Jensen puts them on opposite sides of each other.
            "attention_entropy_nats": (
                None if head is None or head >= len(entropies) else float(entropies[head])
            ),
            "kld_nats": None if head is None or head >= len(kld_per_head) else kld_per_head[head],
            "kld_share": (
                None
                if head is None or head >= len(kld_per_head) or kld_total == 0.0
                else kld_per_head[head] / kld_total
            ),
        }

    pooled = list(
        lag.get("attention_lag_profile_support_corrected")
        or lag.get("attention_lag_profile")
        or []
    )
    rows = [_row(f"head_{head}", profile, head) for head, profile in enumerate(per_head)]
    return {
        "n_heads": len(per_head),
        "pooled": _row("pooled", pooled, None) if pooled else {},
        "heads": rows,
        "kld_per_head_total_nats": kld_total,
    }


def build_sanity(
    results: Dict[str, Any],
    headline: Dict[str, Any],
    *,
    per_sample: Optional[Any] = None,
    per_anchor: Optional[Any] = None,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run every self-consistency check and summarise the outcome.

    Args:
        results: The accumulated results.
        headline: The flattened headline block.
        per_sample: The per-sample table, for the cross-table recombination.
        per_anchor: The per-anchor table, for the same.
        probe: The loader probe's record, for the population checks. ``None`` on a run that
            reused a finished directory's tables and found no probe record beside them.

    Returns:
        Per-check verdicts plus ``n_failed``, ``n_inconclusive`` and an overall ``warning`` flag.
        The flag is **not** the exit code: see the module docstring.
    """
    checks = {
        "kl_identity": check_kl_identity(results),
        "per_anchor_recombines": check_per_anchor_recombines(per_sample, per_anchor),
        "argmax_lag": check_argmax_lag(results),
        # The two structural identities of the lag readout, re-measured on this run rather than
        # inherited from the model tests: the checkpoint and the data are what a summary's reader
        # is holding, and neither is what a fixture proved.
        **{name: check_lag_identity(results, name) for name in LAG_IDENTITIES},
        # No cross-spectral checks. The sibling's two -- an exact Parseval identity against the
        # time domain, and a bound on how much of the forecast error the spectrum can see -- are
        # properties of an estimator this package does not have: ``coherence`` is not ported,
        # because a stored coefficient is a modulus and the phase the estimator needs was
        # discarded before the value was written. Two checks that could only ever be INCONCLUSIVE
        # would read as an analysis that failed rather than one that does not exist.
        # The population checks are the shared ones: what they read is the probe's record, which
        # says nothing about which model produced it.
        "per_file_counts": report.check_per_file_counts(probe),
        "classes_present": report.check_classes_present(probe),
        "target_not_truncated": report.check_target_not_truncated(probe),
        "headline_finite": report.check_headline_finite(headline),
    }
    failed = sorted(name for name, record in checks.items() if record["verdict"] == "fail")
    return {
        "checks": checks,
        "failed": failed,
        "n_failed": len(failed),
        "n_inconclusive": sum(
            1 for record in checks.values() if record["verdict"] == INCONCLUSIVE
        ),
        # Deliberately distinct from the process exit code, which reflects whether a step raised.
        "warning": bool(failed),
    }


# =============================================================================
# Assembly
# =============================================================================
def finalise(
    report_state: Report,
    *,
    output_dir: Any,
    analyses: Sequence[str],
    eval_config: Dict[str, Any],
    started_at: Optional[float] = None,
    per_sample: Optional[Any] = None,
    per_anchor: Optional[Any] = None,
    probe: Optional[Dict[str, Any]] = None,
    headline_scalars: Sequence[Tuple[str, Tuple[str, ...]]] = (),
) -> Dict[str, Any]:
    """Assemble the derived blocks immediately before the summary is written.

    Everything here is computed *from* what the analyses already reported rather than alongside
    them, so an analysis that failed costs its own block and nothing else. **Each block is built
    under its own guard**: this runs after every analysis has completed, so anything raising here
    would lose the entire run -- every result and every captured traceback -- to a failure in the
    bookkeeping, which is precisely what the step wrapper exists to prevent.

    Args:
        report_state: The run's accumulated report.
        output_dir: The results directory, scanned for the artifact manifest.
        analyses: The analyses that were selected, for the coverage record.
        eval_config: The validated block, for the inert-cap warnings.
        started_at: Run start as a POSIX timestamp, so the manifest can exclude a previous run's
            files when the output directory is reused.
        per_sample: The per-sample table, for the cross-table sanity check.
        per_anchor: The per-anchor table, for the same.
        probe: The loader probe's record, for the population checks.
        headline_scalars: The evaluated model's own headline entries, appended to the shared
            registry. Empty for a model that registers no analysis of its own.

    Returns:
        The artifact manifest, which belongs beside the summary rather than inside its results.
    """

    def _safe(name: str, builder: Any, fallback: Any) -> Any:
        try:
            return builder()
        except Exception as exc:  # noqa: BLE001 - see the docstring
            logger.error(
                f"could not assemble the {name!r} summary block: {type(exc).__name__}: {exc}. "
                f"The run's results and step records are unaffected."
            )
            return {**fallback, "error": f"{type(exc).__name__}: {exc}"}

    results = report_state.results
    # Before the headline and the sanity block, because both are read beside it: the pooled lag
    # check reports a censoring edge the per-head rows are the only evidence against, and a reader
    # who has the verdict without the heads has the half that cannot move.
    report_state.set("lag_heads", _safe("lag_heads", lambda: build_lag_heads(results), {}))
    report_state.set(
        "headline",
        _safe("headline", lambda: build_headline(results, headline_scalars), {}),
    )
    report_state.set(
        "coverage",
        _safe(
            "coverage", lambda: build_coverage(results, list(analyses)),
            {"per_analysis": {}, "warnings": []},
        ),
    )
    report_state.set(
        "sanity",
        _safe(
            "sanity",
            lambda: build_sanity(
                results, results["headline"],
                per_sample=per_sample, per_anchor=per_anchor, probe=probe,
            ),
            {"checks": {}, "failed": [], "n_failed": 0, "n_inconclusive": 0, "warning": True},
        ),
    )
    report_state.set(
        "config_warnings",
        _safe("config_warnings", lambda: check_inert_caps(eval_config), {}) or [],
    )

    warnings = list(results["coverage"].get("warnings") or [])
    if isinstance(results["config_warnings"], list):
        warnings += results["config_warnings"]
    for warning in warnings:
        logger.warning(warning)
    for name in results["sanity"].get("failed") or []:
        logger.error(
            f"sanity check FAILED [{name}]: {results['sanity']['checks'][name]['detail']}"
        )

    # Last, so it sees every file the analyses wrote. Outside ``results`` because it is a
    # description of the directory rather than a finding, and because two runs into one directory
    # legitimately produce different manifests from identical results.
    return _safe(
        "artifacts", lambda: build_manifest(output_dir, since=started_at),
        {"files": {}, "figures": [], "n_files": 0, "n_figures": 0},
    )


def step_records(steps: Sequence[StepRecord]) -> List[Dict[str, Any]]:
    """Return the per-step records for the summary, each with an explicit status.

    ``ok`` is a bool and reads as one; ``status`` is what an operator greps for and what the run
    log prints, so both travel rather than a reader having to map one onto the other.

    Args:
        steps: The report's step records.

    Returns:
        One JSON-shaped record per step, in the order they ran.
    """
    return [
        {**record.as_dict(), "status": "ok" if record.ok else "failed"} for record in steps
    ]


def write_steps(steps: Sequence[StepRecord], output_dir: Any) -> Path:
    """Write the per-step heartbeat, overwriting the previous one.

    Called as each analysis finishes rather than once at the end: a run killed outright leaves no
    summary, and on a multi-hour pass the question afterwards is which step it was inside.

    Args:
        steps: The report's step records so far.
        output_dir: The results directory.

    Returns:
        The path written.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / STEPS_FILENAME
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(step_records(steps)), handle, indent=2, allow_nan=False)
    return path


def _cell(value: Any, spec: str) -> str:
    """Render one table cell, or ``'-'`` where the run measured nothing.

    A dash rather than ``nan`` or ``0``: a head whose entropy was never collected and a head whose
    entropy is zero are different states, and a table that renders both as a number invites the
    second reading of the first.

    Args:
        value: The number, or ``None``.
        spec: A ``str.format`` spec applied to a finite number.

    Returns:
        The rendered cell.
    """
    if value is None:
        return "-"
    number = float(value)
    return "-" if not math.isfinite(number) else spec.format(
        int(number) if spec.endswith("d}") else number
    )


def console_summary(results: Dict[str, Any], steps: Sequence[StepRecord]) -> str:
    """Render the end-of-run console table from the assembled state.

    Guarded for the same reason :func:`finalise` is: this is called between the last analysis and
    the summary write, and a formatting error here must not be what loses the run.

    Args:
        results: The accumulated results, after :func:`finalise`.
        steps: The per-step records.

    Returns:
        The table as a multi-line string.
    """
    try:
        lines = ["", "=" * 78, "eval summary", "=" * 78]
        headline = results.get("headline") or {}
        # The block's own keys rather than :data:`HEADLINE_SCALARS`, so a binding's registered
        # scalars reach the operator's table as well as ``summary.json``. Insertion order is the
        # shared entries then the binding's, so a model that registers none prints what it always
        # printed. The two trailing kinds are rendered below and as prose respectively.
        for name in headline:
            if name.startswith("verdict_") or name == "pred_gap_convention":
                continue
            value = headline.get(name)
            rendered = (
                f"{float(value):.6g}"
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                else ("-" if value is None else str(value))
            )
            lines.append(f"  {name:<38s} {rendered}")
        for verdict in results.get("verdicts") or []:
            lines.append(f"  [{str(verdict.get('status')):>12s}] {verdict.get('name')}")

        # The per-head lag table, printed between the headline and the sanity block because that
        # is where it is read: the argmax check below reports the pooled profile pinned at a
        # censoring edge, and these rows are what say whether any head escaped it.
        heads = results.get("lag_heads") or {}
        if heads.get("heads"):
            lines.append("-" * 78)
            lines.append(
                f"  lag profile per head ({heads.get('n_heads')} heads; "
                f"near/far mass over the compensated axis, entropy in nats)"
            )
            lines.append(
                f"  {'profile':<10s}{'argmax':>8s}{'width':>7s}{'>half':>8s}"
                f"{'near':>8s}{'far':>8s}{'attn H':>8s}{'KL share':>10s}  degenerate"
            )
            for row in [heads.get("pooled") or {}, *heads["heads"]]:
                if not row:
                    continue
                lines.append(
                    f"  {str(row['profile']):<10s}{_cell(row['argmax_lag_step'], '{:d}'):>8s}"
                    f"{_cell(row['peak_width_bins'], '{:d}'):>7s}"
                    f"{_cell(row['mass_above_half_peak'], '{:.3f}'):>8s}"
                    f"{_cell(row['near_mass'], '{:.3f}'):>8s}"
                    f"{_cell(row['far_mass'], '{:.3f}'):>8s}"
                    f"{_cell(row['attention_entropy_nats'], '{:.3f}'):>8s}"
                    f"{_cell(row['kld_share'], '{:.3f}'):>10s}"
                    f"  {'yes' if row['degenerate'] else 'no'}"
                )

        lines.append("-" * 78)
        for name, record in ((results.get("sanity") or {}).get("checks") or {}).items():
            lines.append(f"  [{record['verdict']:>12s}] {name}: {record['detail']}")
        if (results.get("sanity") or {}).get("warning"):
            lines.append(
                f"  !! {results['sanity']['n_failed']} sanity check(s) FAILED -- read them "
                f"before quoting any number above"
            )
        for warning in (results.get("coverage") or {}).get("warnings", []):
            lines.append(f"  !! {warning}")
        for warning in results.get("config_warnings") or []:
            lines.append(f"  !! {warning}")

        lines.append("-" * 78)
        for record in steps:
            lines.append(
                f"  {'ok  ' if record.ok else 'FAIL'} {record.name:<28s} {record.elapsed_s:8.1f}s"
            )
        peak = results.get("max_memory_allocated_gb")
        if peak is not None:
            lines.append(f"  peak CUDA memory {float(peak):.2f} GB")
        lines.append("=" * 78)
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001 - a formatting error must not cost the run
        return f"(could not render the summary table: {type(exc).__name__}: {exc})"


__all__ = [
    "HEADLINE_SCALARS",
    "HEADLINE_VERDICTS",
    "IDENTITY_RTOL",
    "INCONCLUSIVE",
    "PHYSICAL_LAG_IDENTITY",
    "PRED_GAP_CONVENTION",
    "RECOMBINED_COLUMNS",
    "Report",
    "StepRecord",
    "STEPS_FILENAME",
    "SUMMARY_FILENAME",
    "build_coverage",
    "build_headline",
    "build_lag_heads",
    "build_manifest",
    "build_sanity",
    "check_argmax_lag",
    "check_inert_caps",
    "check_kl_identity",
    "check_per_anchor_recombines",
    "console_summary",
    "emit_grouped_variants",
    "finalise",
    "json_safe",
    "step_records",
    "summarise_by_group",
    "write_steps",
]
