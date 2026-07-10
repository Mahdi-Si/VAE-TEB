r"""S4-T01 / S4-T02: out-of-support KLD summaries, and the null-cell intercept gate.

**S4-T01.** ``kbar_full`` averages $K_t$ over *all* $T$ steps. Under ``kld_support: anchor``
that span includes a warm-up prefix and an untrained final-$H$ tail -- regions the model was
never asked to shape -- so it is not admissible as a TE surrogate, however well it happens to
correlate. It stays in the table (it is evidence about what the untrained region does) but is
stamped ``out_of_support`` and can never top the ranking. ``kbar_postwarm`` is the exact anchor
support and is the correct comparator. Under ``kld_support: full`` (the ``parity`` arm, and
every v1 run) nothing is flagged.

**S4-T02.** v3 initialises with $K \equiv 0$, so the calibration intercept $\alpha$ no longer
absorbs a random log-variance-head floor. That makes
$\bar K \big|_{\mathrm{TE}_{\mathrm{inj}} = 0} \to 0$ a claim the model can be held to. The
bootstrap CI over the (few) null cells is percentile-based and seeded; a Monte-Carlo test pins
its coverage rather than trusting the implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (  # noqa: E402
    eval_v2,
    final_report_v2 as fr,
    visualize_v2 as viz,
)

_CELLS = [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.5), (4, 1.0), (5, 2.0), (6, 3.0)]
_PER_CELL = 8


def _arrs(*, null_kbar: float = 0.01, gamma: float = 0.5, seed: int = 0) -> dict:
    r"""Per-sample arrays over 7 cells (3 null + 4 signal) with $\bar K = \gamma\,TE$ + noise.

    ``kbar_full`` is deliberately built to correlate with TE *better* than ``kbar`` does, so a
    ranking that ignores ``out_of_support`` would crown it. That is the trap S4-T01 closes.
    """
    rng = np.random.default_rng(seed)
    cid, te, kbar = [], [], []
    for c, t in _CELLS:
        for _ in range(_PER_CELL):
            cid.append(c)
            te.append(t)
            kbar.append((null_kbar if t == 0.0 else gamma * t)
                        + rng.normal(0.0, 0.02))
    te_a = np.asarray(te, dtype=np.float64)
    out = {
        "cell_id": np.asarray(cid, dtype=np.int64),
        "te_inj": te_a,
        "te_scat": te_a * 0.9,
        "frac_phi": np.full(te_a.shape, 1.1),
        "delay": np.full(te_a.shape, 8, dtype=np.int64),
        "kbar": np.asarray(kbar, dtype=np.float64),
        "n": len(cid),
    }
    # A noiseless (hence perfectly correlated) full-sequence mean, and a realistic postwarm one.
    out["kbar_full"] = gamma * te_a
    out["kbar_postwarm"] = out["kbar"] + rng.normal(0.0, 0.01, size=te_a.shape)
    for extra in ("kbar_sum", "kbar_max", "kbar_median", "kbar_p90",
                  "kbar_inband", "kbar_outband"):
        out[extra] = out["kbar"] + rng.normal(0.0, 0.05, size=te_a.shape)
    return out


# ---------------------------------------------------------------------------
# S4-T01: out_of_support
# ---------------------------------------------------------------------------
def test_kbar_full_flagged_under_anchor_support() -> None:
    cal = eval_v2.fit_calibration(_arrs(), kld_support="anchor")
    assert cal["kld_support"] == "anchor"
    assert cal["kld_variants"]["kbar_full"]["out_of_support"] is True
    for v in ("kbar", "kbar_postwarm", "kbar_inband", "kbar_outband"):
        assert cal["kld_variants"][v]["out_of_support"] is False


def test_nothing_flagged_under_full_support() -> None:
    """``parity`` (and every v1 run) trains on the whole sequence; nothing is out of support."""
    cal = eval_v2.fit_calibration(_arrs(), kld_support="full")
    for entry in cal["kld_variants"].values():
        assert entry["out_of_support"] is False


def test_default_kld_support_is_full_so_a_v1_alias_never_flags() -> None:
    """``getattr(model, 'kld_support', 'full')`` on a v1 model must not raise or flag."""
    cal = eval_v2.fit_calibration(_arrs())
    assert cal["kld_support"] == "full"
    assert cal["kld_variants"]["kbar_full"]["out_of_support"] is False


def test_kbar_full_is_still_collected_and_fitted() -> None:
    """Flagged is not dropped: it is evidence about the untrained region."""
    cal = eval_v2.fit_calibration(_arrs(), kld_support="anchor")
    entry = cal["kld_variants"]["kbar_full"]
    assert entry["n"] == len(_CELLS) * _PER_CELL
    assert entry["gamma_inj"] is not None and np.isfinite(entry["gamma_inj"])
    assert "kbar_full" in eval_v2.KLD_SCALAR_VARIANTS


def test_the_ranking_never_crowns_a_flagged_variant() -> None:
    r"""``kbar_full`` correlates *perfectly* here, yet must not head the report table."""
    cal = eval_v2.fit_calibration(_arrs(), kld_support="anchor")
    kv = cal["kld_variants"]
    # Precondition: the trap is armed -- the flagged variant genuinely tracks TE best.
    assert abs(kv["kbar_full"]["spearman_inj"]) >= abs(kv["kbar"]["spearman_inj"])

    lines = fr._kld_variants_section(cal)
    body = [ln for ln in lines if ln.startswith("| ") and not ln.startswith("| KLD summary")]
    assert "†" not in body[0], f"a flagged variant tops the ranking: {body[0]}"
    assert "†" in body[-1], "the flagged variant is not sorted to the bottom"
    assert any("Out of support" in ln for ln in lines)


def test_the_report_section_has_no_dagger_under_full_support() -> None:
    cal = eval_v2.fit_calibration(_arrs(), kld_support="full")
    text = "\n".join(fr._kld_variants_section(cal))
    assert "†" not in text and "Out of support" not in text


def test_is_out_of_support_helper_tolerates_legacy_metrics() -> None:
    """A pre-S4 ``metrics.json`` has no ``out_of_support`` key; nothing is greyed."""
    assert viz._is_out_of_support({"kbar_full": {"spearman_inj": 0.9}}, "kbar_full") is False
    assert viz._is_out_of_support({}, "kbar_full") is False
    assert viz._is_out_of_support({"kbar_full": {"out_of_support": True}}, "kbar_full") is True


def test_ranking_figures_render_with_a_flagged_variant(tmp_path) -> None:
    cal = eval_v2.fit_calibration(_arrs(), kld_support="anchor")
    arrs = _arrs()
    metrics = {"run_tag": "unit", "split": "test", "calibration": cal, "per_cell": []}
    assert viz.plot_kld_te_correlation(metrics, tmp_path / "corr")
    assert viz.plot_kld_variants_vs_te(arrs, metrics, tmp_path / "variants")


# ---------------------------------------------------------------------------
# S4-T02: null-cell intercept gate
# ---------------------------------------------------------------------------
def test_null_cell_gate_passes_on_a_near_zero_kbar() -> None:
    cal = eval_v2.fit_calibration(_arrs(null_kbar=0.01), kbar_null_threshold=0.05)
    knc = cal["kbar_at_null_cells"]
    assert knc["n_cells"] == 3
    assert knc["pass"] is True
    assert knc["ci_lo"] <= knc["mean"] <= knc["ci_hi"]
    assert knc["threshold"] == 0.05
    assert knc["n_boot"] == eval_v2._N_BOOT


def test_null_cell_gate_fails_on_a_floored_kbar() -> None:
    r"""The ``parity`` arm's random independent-logvar head leaves $\bar K \gg 0$ at TE = 0."""
    cal = eval_v2.fit_calibration(_arrs(null_kbar=0.9), kbar_null_threshold=0.05)
    assert cal["kbar_at_null_cells"]["pass"] is False


def test_null_cell_gate_is_none_without_a_null_cell() -> None:
    arrs = _arrs()
    keep = arrs["te_inj"] > 0.0
    trimmed = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape == keep.shape else v)
               for k, v in arrs.items()}
    cal = eval_v2.fit_calibration(trimmed)
    assert cal["kbar_at_null_cells"] is None


def test_null_cell_gate_is_reproducible_under_a_fixed_seed() -> None:
    a = eval_v2.fit_calibration(_arrs(), boot_seed=7)["kbar_at_null_cells"]
    b = eval_v2.fit_calibration(_arrs(), boot_seed=7)["kbar_at_null_cells"]
    assert (a["ci_lo"], a["ci_hi"]) == (b["ci_lo"], b["ci_hi"])


def test_alpha_inj_is_close_to_kbar_at_null_cells() -> None:
    r"""A calibrated surrogate has $\alpha \approx \bar K \big|_{\mathrm{TE} = 0}$."""
    cal = eval_v2.fit_calibration(_arrs(null_kbar=0.01, gamma=0.5))
    assert abs(cal["alpha_inj"] - cal["kbar_at_null_cells"]["mean"]) < 0.05


def test_headline_rows_render_the_null_cell_gate() -> None:
    cal = eval_v2.fit_calibration(_arrs(null_kbar=0.9), kbar_null_threshold=0.05)
    rows = "\n".join(fr._null_cell_rows(cal))
    assert "**FAIL**" in rows
    assert "alpha" in rows or r"\alpha" in rows
    # A legacy metrics.json degrades to n/a rather than raising.
    assert "n/a" in "\n".join(fr._null_cell_rows({}))


# ---------------------------------------------------------------------------
# S4-T02: bootstrap CI coverage (Monte Carlo)
# ---------------------------------------------------------------------------
def test_bootstrap_ci_edge_cases() -> None:
    lo_hi = eval_v2._bootstrap_ci(np.asarray([]))
    assert all(np.isnan(x) for x in lo_hi)
    assert eval_v2._bootstrap_ci(np.asarray([0.3])) == (0.3, 0.3, 0.3)
    # NaNs are dropped, not propagated.
    mean, lo, hi = eval_v2._bootstrap_ci(np.asarray([1.0, np.nan, 1.0]), n_boot=50,
                                         rng=np.random.default_rng(0))
    assert mean == pytest.approx(1.0) and lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


def test_bootstrap_ci_never_exceeds_the_data_range() -> None:
    x = np.asarray([0.1, 0.2, 0.9])
    _, lo, hi = eval_v2._bootstrap_ci(x, n_boot=500, rng=np.random.default_rng(1))
    assert x.min() <= lo <= hi <= x.max()


@pytest.mark.slow
def test_bootstrap_ci_covers_the_true_mean_at_the_nominal_rate() -> None:
    r"""Monte Carlo over 200 synthetic datasets: the 95% CI must cover in $[0.90, 0.99]$.

    The percentile bootstrap under-covers on tiny samples, so the acceptance band is stated
    rather than assumed to be exactly 0.95. With ``n_cells = 8`` draws per dataset (a small but
    not degenerate null-cell count) empirical coverage lands near 0.93.
    """
    rng = np.random.default_rng(20260709)
    true_mean, n_cells, n_trials = 0.03, 8, 200
    covered = 0
    for _ in range(n_trials):
        sample = rng.normal(true_mean, 0.01, size=n_cells)
        _, lo, hi = eval_v2._bootstrap_ci(sample, n_boot=500, rng=rng)
        covered += int(lo <= true_mean <= hi)
    coverage = covered / n_trials
    assert 0.90 <= coverage <= 0.99, f"empirical 95% CI coverage = {coverage:.3f}"
