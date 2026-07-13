"""Decision rules for the causal-TE validation suite.

Translates per-test ``evidence`` dicts (produced by each
``test_NN_*.py`` module) into a single ``verdict`` per test plus a
manuscript-level ``headline_claim`` of strength ``"strong"``,
``"moderate"``, or ``"weak"``.

The rules follow the spec in
``model/vae_teb_prediction/testing/causal_te_validation/causal_te.md``, adapted to the raw target
(S8-T03): the feature-domain **band-uplift** test (Test 3) has no raw analogue -- the raw future
FHR has no frequency-band partition -- so it is replaced by a **raw uplift-specificity** test
(``test_03_raw_uplift``): the true UP must produce a positive forecast uplift that *collapses* when
UP is time-shuffled (specificity to the temporal coupling, not generic UP energy).

* **Strong** — every in-scope test passes, including lag-event
  alignment.
* **Moderate** — Tests 1, 2, 3 pass (UP corruption hurts forecast,
  KLD predicts uplift, the uplift is specific to the true temporal UP)
  but Test 4 (event alignment) is inconclusive.
* **Weak** — Test 1 passes but Test 2 or Test 3 do not.
* **Inconclusive** — none of the above patterns are met.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

# Test ids in stable display order. ``test_06_*`` and ``test_05/07/08`` are
# explicitly out of scope for this submodule.
TEST_IDS: Tuple[str, ...] = (
    "test_01_up_ablation",
    "test_02_kld_uplift",
    "test_03_raw_uplift",
    "test_04_lag_event",
    "test_09_dim_spec",
    "test_10_event_kld",
)


# ---------------------------------------------------------------------------
# Per-test verdict helpers
# ---------------------------------------------------------------------------


def verdict_test_01_up_ablation(evidence: Mapping[str, Any]) -> str:
    """UP causal utility test verdict.

    ``evidence`` must include:

    * ``deltaE_pass`` — True iff $\\Delta E > 0$ for all 3 perturbations
      with Holm-adjusted $p < 0.05$.
    * ``deltaK_pass`` — same for $\\Delta K$.
    * ``deltaR_pass`` — same for $\\Delta R$.
    """
    e = dict(evidence or {})
    if not (e.get("deltaE_pass") and e.get("deltaK_pass") and e.get("deltaR_pass")):
        # Fail-mode classification per spec section "Test 1 acceptance".
        if e.get("deltaK_pass") and not e.get("deltaE_pass"):
            return "fail_mode_a"  # KLD reflects sensitivity, not useful TE
        if e.get("deltaE_pass") and not e.get("deltaK_pass"):
            return "fail_mode_b"  # Forecast helped but bottleneck silent
        if (
            e.get("normal_vs_shuffle_similar") is True
        ):
            return "fail_mode_c"  # Generic UP statistics, not temporal coupling
        return "inconclusive"
    return "pass"


def verdict_test_02_kld_uplift(evidence: Mapping[str, Any]) -> str:
    """KLD-uplift regression verdict (random-intercept on GUID)."""
    e = dict(evidence or {})
    beta1_positive = bool(e.get("beta1_positive"))
    gamma1_positive = bool(e.get("gamma1_positive"))
    if beta1_positive:
        return "pass"
    if gamma1_positive and not beta1_positive:
        return "fail_mode_difficulty"
    return "inconclusive"


def verdict_test_03_raw_uplift(evidence: Mapping[str, Any]) -> str:
    r"""Raw uplift-specificity verdict (the raw replacement for the pruned band-uplift test).

    The raw future FHR has no frequency-band partition, so "KLD predicts *band-specific* uplift" has
    no analogue. The raw specificity claim is temporal instead: the true UP must produce a positive
    forecast uplift ($l_{\mathrm{base}} > l_{\mathrm{full}}$) **and** that uplift must **collapse when
    UP is time-shuffled** -- a scrambled source that keeps UP's marginal statistics but destroys its
    temporal coupling must not help. Both are read (with significance) from the ``up_effect`` deltas.

    ``evidence`` must include:

    * ``uplift_positive`` — True iff the median ``normal`` uplift is $> 0$ (Wilcoxon $p < 0.05$).
    * ``shuffle_breaks_uplift`` — True iff the median uplift *drop* under ``time_shuffle`` is $> 0$
      (Wilcoxon $p < 0.05$): the true UP helps more than a temporally-scrambled one.
    """
    e = dict(evidence or {})
    uplift_positive = bool(e.get("uplift_positive"))
    shuffle_breaks = bool(e.get("shuffle_breaks_uplift"))
    if uplift_positive and shuffle_breaks:
        return "pass"
    if uplift_positive and not shuffle_breaks:
        # The source helps, but a temporally-scrambled UP helps as much -> not coupling-specific.
        return "fail_not_specific"
    return "inconclusive"


def verdict_test_04_lag_event(evidence: Mapping[str, Any]) -> str:
    """Lag-event alignment verdict (event-aligned KLD enrichment)."""
    e = dict(evidence or {})
    n_pairs = int(e.get("n_pairs", 0) or 0)
    if n_pairs < 5:
        return "inconclusive"
    median_err = float(e.get("median_abs_error_s", float("inf")))
    kld_enriched = bool(e.get("kld_enriched_at_event"))
    if median_err < 30.0 and kld_enriched:
        return "pass"
    if e.get("attention_at_lag_zero_only") is True:
        return "fail_mode_lag_zero_shortcut"
    return "inconclusive"


def verdict_test_09_dim_spec(evidence: Mapping[str, Any]) -> str:
    """Latent-dimension specificity verdict."""
    e = dict(evidence or {})
    n_contrastive = int(e.get("n_contrastive_dims", 0) or 0)
    n_stable_top3 = int(e.get("n_stable_in_top3", 0) or 0)
    if n_contrastive >= 3 and n_stable_top3 >= 2:
        return "pass"
    if n_contrastive >= 1:
        return "fail_unstable"
    return "inconclusive"


def verdict_test_10_event_kld(evidence: Mapping[str, Any]) -> str:
    """Event-triggered KLD / TE-lag verdict."""
    e = dict(evidence or {})
    deltaK = bool(e.get("delta_K_positive"))
    deltaC = bool(e.get("delta_C_positive"))
    deltaTE = bool(e.get("delta_TE_positive"))
    if deltaK and deltaC and deltaTE:
        return "pass"
    if not (deltaK or deltaC or deltaTE):
        return "inconclusive"
    return "fail_partial"


# ---------------------------------------------------------------------------
# Headline claim
# ---------------------------------------------------------------------------


_VERDICT_FN = {
    "test_01_up_ablation": verdict_test_01_up_ablation,
    "test_02_kld_uplift":  verdict_test_02_kld_uplift,
    "test_03_raw_uplift":  verdict_test_03_raw_uplift,
    "test_04_lag_event":   verdict_test_04_lag_event,
    "test_09_dim_spec":    verdict_test_09_dim_spec,
    "test_10_event_kld":   verdict_test_10_event_kld,
}


def aggregate_verdicts(
    tests: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Run the per-test verdict functions on a ``tests`` dict.

    Args:
        tests: Mapping ``{test_id: {"evidence": {...}, ...}}``.

    Returns:
        Mapping ``{test_id: {"verdict": str, "evidence": {...}}}``.
        Tests missing from input get ``verdict = "missing"``.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for test_id in TEST_IDS:
        entry: Optional[Mapping[str, Any]] = tests.get(test_id)
        if entry is None:
            out[test_id] = {"verdict": "missing", "evidence": {}}
            continue
        evidence = dict(entry.get("evidence", {}))
        verdict_fn = _VERDICT_FN[test_id]
        try:
            verdict = verdict_fn(evidence)
        except Exception as exc:  # noqa: BLE001
            verdict = "error"
            evidence = {**evidence, "verdict_error": str(exc)}
        out[test_id] = {"verdict": verdict, "evidence": evidence}
    return out


def headline_claim(verdicts: Mapping[str, Mapping[str, Any]]) -> str:
    """Translate per-test verdicts into a single manuscript claim.

    Decision tree (Test 3 is the raw uplift-specificity test, S8-T03):

    * **strong**: tests 1, 2, 3, 4 *and* (9 *or* 10) all pass.
    * **moderate**: tests 1, 2, 3 pass (test 4 may be inconclusive).
    * **weak**: test 1 passes but test 2 or test 3 fail.
    * **inconclusive**: anything else.

    Args:
        verdicts: Output of :func:`aggregate_verdicts`.

    Returns:
        ``"strong"``, ``"moderate"``, ``"weak"``, or ``"inconclusive"``.
    """
    def passed(tid: str) -> bool:
        v = verdicts.get(tid, {}).get("verdict", "missing")
        return v == "pass"

    test1, test2, test3, test4 = (
        passed("test_01_up_ablation"),
        passed("test_02_kld_uplift"),
        passed("test_03_raw_uplift"),
        passed("test_04_lag_event"),
    )
    test9, test10 = passed("test_09_dim_spec"), passed("test_10_event_kld")

    if test1 and test2 and test3 and test4 and (test9 or test10):
        return "strong"
    if test1 and test2 and test3:
        return "moderate"
    if test1 and not (test2 and test3):
        return "weak"
    return "inconclusive"
