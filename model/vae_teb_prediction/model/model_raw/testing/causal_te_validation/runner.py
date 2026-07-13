"""Orchestrator for the causal-TE validation suite.

Runs all six in-scope tests in a single pipeline step:

* Tests 1, 2, 3, 9 are CSV post-processors and consume artifacts
  produced by earlier pipeline steps (``up_effect``, ``histogram``).
  Test 3 is the raw uplift-specificity test (S8-T03), reading the
  ``up_effect`` deltas -- the scattering band-uplift Test 3 is pruned.
* Tests 4 and 10 share **one** ``collect_predictions`` inference pass
  (capped at the caller-supplied ``max_samples``, typically
  ``HEAVY_PRED_CAP=2000``). The shared sample list is pruned in place
  to keep only the fields the two tests actually need
  (``fhr, up, kld_t, te_lag, attn`` plus metadata) — saves ~12 MB per
  sample compared with the full ``collect_predictions`` payload.

Outputs a single ``summary.json`` under
``<output>/causal_te_validation/`` containing each test's verdict
(``pass`` / ``fail_mode_*`` / ``inconclusive`` / ``missing``) and the
manuscript-level headline claim from
:func:`decision_rules.headline_claim`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from model.vae_teb_prediction.model.model_raw.testing.base import TestRunner
from model.vae_teb_prediction.model.model_raw.testing.collectors import collect_predictions

from . import decision_rules
from . import plots
from . import test_01_up_ablation_stats
from . import test_02_kld_uplift_regression
# S6-T01c: test_03_band_uplift_regression (scattering-band UP->FHR transfer regression) is pruned in
# the raw port -- the raw target has no frequency-band partition. S8-T03 replaces it with
# test_03_raw_uplift (temporal uplift-specificity), so the decision tree keeps its three-legged
# strong/moderate structure.
from . import test_03_raw_uplift
from . import test_04_lag_event_alignment
from . import test_09_dim_specificity
from . import test_10_event_triggered_kld


# Fields kept on each pre-collected sample for Tests 4 + 10. Everything
# else from ``collect_predictions`` (mu_full, mu_base, y_plus, z, ...)
# is dropped immediately to keep memory bounded.
_KEEP_FIELDS = (
    "guid", "epoch", "label",
    "fhr", "up", "kld_t", "te_lag", "attn",
)


def _slim_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Trim each sample dict to the fields needed by Tests 4 + 10.

    Modifies the list in place to release the heavy fields back to the
    garbage collector as soon as possible.
    """
    out: List[Dict[str, Any]] = []
    for s in samples:
        slim = {k: s.get(k) for k in _KEEP_FIELDS}
        out.append(slim)
    samples.clear()
    return out


def _select_representative_sample(
    samples: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Pick the sample with the largest mean $K_t$ as a headline panel exemplar.

    Args:
        samples: Slim sample list.

    Returns:
        The exemplar dict, or ``None`` when the list is empty / has no
        valid KLD traces.
    """
    if not samples:
        return None
    best = None
    best_score = -float("inf")
    for s in samples:
        kld_t = s.get("kld_t")
        if kld_t is None:
            continue
        try:
            import numpy as np
            score = float(np.nanmean(np.asarray(kld_t, dtype=float)))
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best = s
    return best


def _safe_run(name: str, fn, **kwargs) -> Dict[str, Any]:
    """Wrap a single ``test_NN.run(...)`` call in try/except.

    Mirrors the ``_safe`` helper used elsewhere in the testing pipeline.
    Any exception turns into ``{"verdict": "error", "evidence": {...}}``
    so the rest of the suite continues.
    """
    try:
        logger.info(f"causal_te_validation: running {name}")
        result = fn(**kwargs)
        if not isinstance(result, dict):
            result = {"verdict": "error", "evidence": {"return": "non_dict"}}
        return result
    except Exception as exc:  # noqa: BLE001
        logger.error(f"causal_te_validation: {name} failed: {exc}")
        return {
            "verdict": "error",
            "evidence": {"exception": str(exc)},
            "csv_paths": [],
            "figure_paths": [],
        }


def _strip_unserializable(obj: Any) -> Any:
    """Recursively convert numpy / DataFrame leaves into JSON-safe values.

    Handles ``np.ndarray`` (via ``.tolist()``), every ``np.generic``
    scalar (floats, ints, and ``np.bool_``), ``Path`` and drops
    ``pandas.DataFrame`` values from dicts. Non-finite floats
    (``nan``, ``inf``) are converted to ``None`` so the resulting
    structure is round-trippable through strict JSON parsers as well as
    Python's permissive ``json.dump`` defaults.
    """
    try:
        import math
        import numpy as np
        import pandas as pd
    except Exception:
        np = None  # type: ignore[assignment]
        pd = None  # type: ignore[assignment]
        math = None  # type: ignore[assignment]

    if isinstance(obj, dict):
        return {
            k: _strip_unserializable(v)
            for k, v in obj.items()
            if not (pd is not None and isinstance(v, pd.DataFrame))
        }
    if isinstance(obj, (list, tuple)):
        return [_strip_unserializable(v) for v in obj]
    if np is not None and isinstance(obj, np.ndarray):
        return _strip_unserializable(obj.tolist())
    if np is not None and isinstance(obj, np.generic):
        # Covers np.bool_, np.float*, np.int*, np.complex*, etc.
        return _strip_unserializable(obj.item())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and math is not None and not math.isfinite(obj):
        return None
    return obj


def run_causal_te_validation(
    runner: TestRunner,
    loader: Any,
    *,
    output_dir: Path,
    max_samples: int,
    histogram_csv: Path,
    up_effect_dir: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run the causal-TE validation suite end-to-end.

    Args:
        runner: Loaded :class:`TestRunner`.
        loader: Standard segment-level DataLoader.
        output_dir: ``<root>/causal_te_validation``.
        max_samples: Sample cap for the shared
            :func:`collect_predictions` pass (Tests 4 + 10).
        histogram_csv: Path to ``<root>/histograms/histogram_metrics.csv``.
        up_effect_dir: Path to ``<root>/up_effect``.
        seed: RNG seed shared by every bootstrap helper inside the suite.

    Returns:
        Dict with ``tests`` (per-test verdicts + evidence),
        ``headline_claim``, ``summary_path``, ``output_dir``,
        ``n_samples_collected``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --- Tests 1, 2, 3, 9: CSV post-processors -----------------------------
    test_01 = _safe_run(
        "test_01_up_ablation",
        test_01_up_ablation_stats.run,
        up_effect_dir=Path(up_effect_dir),
        output_dir=output_dir / "up_ablation_stats",
        seed=int(seed),
    )
    test_02 = _safe_run(
        "test_02_kld_uplift",
        test_02_kld_uplift_regression.run,
        histogram_csv=Path(histogram_csv),
        output_dir=output_dir / "kld_uplift_regression",
    )
    test_03 = _safe_run(
        "test_03_raw_uplift",
        test_03_raw_uplift.run,
        up_effect_dir=Path(up_effect_dir),
        output_dir=output_dir / "raw_uplift",
        seed=int(seed),
    )
    test_09 = _safe_run(
        "test_09_dim_spec",
        test_09_dim_specificity.run,
        histogram_csv=Path(histogram_csv),
        output_dir=output_dir / "dim_specificity",
        seed=int(seed),
    )

    # --- Shared inference pass for Tests 4 + 10 ----------------------------
    cap = int(max(1, int(max_samples)))
    logger.info(
        f"causal_te_validation: shared collect_predictions pass "
        f"(cap={cap})"
    )
    try:
        full_samples = collect_predictions(runner, loader, max_samples=cap)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"causal_te_validation: collect_predictions failed: {exc}")
        full_samples = []
    samples = _slim_samples(full_samples) if full_samples else []
    logger.info(
        f"causal_te_validation: collected {len(samples)} samples for "
        f"event-detection tests."
    )

    test_04 = _safe_run(
        "test_04_lag_event",
        test_04_lag_event_alignment.run,
        samples=samples,
        warmup=int(runner.warmup_steps),
        horizon=int(runner.horizon),
        output_dir=output_dir / "lag_event_alignment",
    )
    test_10 = _safe_run(
        "test_10_event_kld",
        test_10_event_triggered_kld.run,
        samples=samples,
        warmup=int(runner.warmup_steps),
        horizon=int(runner.horizon),
        output_dir=output_dir / "event_triggered_kld",
    )

    # --- Per-test plotters --------------------------------------------------
    fig_paths: List[Path] = []
    plot_calls = [
        ("test_01_forest",
         lambda: plots.plot_up_ablation_forest(
             output_dir / "up_ablation_stats" / "wilcoxon_results.csv",
             output_dir / "up_ablation_stats" / "forest_deltas.pdf",
         )),
        ("test_04_hist",
         lambda: plots.plot_alignment_error_hist(
             output_dir / "lag_event_alignment" / "event_pairs.csv",
             output_dir / "lag_event_alignment" / "alignment_error_hist.pdf",
         )),
        ("test_09_forest",
         lambda: plots.plot_dim_contrast_forest(
             output_dir / "dim_specificity" / "per_dim_class_contrast.csv",
             output_dir / "dim_specificity" / "contrast_forest.pdf",
         )),
        ("test_10_violins",
         lambda: plots.plot_event_vs_quiet_violins(
             output_dir / "event_triggered_kld" / "per_sample_event_quiet.csv",
             output_dir / "event_triggered_kld" / "violin_event_vs_quiet.pdf",
         )),
    ]
    for label, fn in plot_calls:
        try:
            p = fn()
            if p is not None:
                fig_paths.append(Path(p))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"causal_te_validation: {label} plot failed: {exc}")

    # Test 2 scatter requires the in-memory DataFrame from the regression
    # step. The regression dict carries it under key ``"df"``.
    try:
        df_for_scatter = test_02.pop("df", None)
        if df_for_scatter is not None:
            p = plots.plot_kld_vs_uplift_scatter(
                df_for_scatter,
                output_dir / "kld_uplift_regression" / "kld_vs_uplift_scatter.pdf",
            )
            if p is not None:
                fig_paths.append(Path(p))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"causal_te_validation: test_02 scatter failed: {exc}")

    # Headline 4-panel figure.
    try:
        rep = _select_representative_sample(samples)
        p = plots.plot_causal_te_summary(
            results_dir=output_dir,
            out_path=figures_dir / "causal_te_summary_4panel.pdf",
            representative_sample=rep,
            histogram_csv=Path(histogram_csv),
        )
        if p is not None:
            fig_paths.append(Path(p))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"causal_te_validation: headline figure failed: {exc}")

    # --- Aggregate + write summary.json ------------------------------------
    raw_tests = {
        "test_01_up_ablation": test_01,
        "test_02_kld_uplift":  test_02,
        "test_03_raw_uplift":  test_03,
        "test_04_lag_event":   test_04,
        "test_09_dim_spec":    test_09,
        "test_10_event_kld":   test_10,
    }
    verdicts = decision_rules.aggregate_verdicts({
        tid: {"evidence": entry.get("evidence", {})}
        for tid, entry in raw_tests.items()
    })
    headline = decision_rules.headline_claim(verdicts)

    summary = {
        "n_samples_collected": int(len(samples)),
        "tests": {
            tid: {
                "verdict": verdicts.get(tid, {}).get("verdict", "missing"),
                "evidence": _strip_unserializable(entry.get("evidence", {})),
                "csv_paths": [str(p) for p in entry.get("csv_paths", [])],
            }
            for tid, entry in raw_tests.items()
        },
        "headline_claim": headline,
        "figure_paths": [str(p) for p in fig_paths],
        "inference_method_per_test": {
            "test_02_kld_uplift":  test_02.get("evidence", {}).get("method", "unknown"),
        },
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(_strip_unserializable(summary), fh, indent=2)
    logger.info(
        f"causal_te_validation: headline_claim={headline!r}; "
        f"summary written to {summary_path}"
    )

    return {
        "tests": summary["tests"],
        "headline_claim": headline,
        "summary_path": str(summary_path),
        "output_dir": str(output_dir),
        "n_samples_collected": int(len(samples)),
    }


__all__ = ["run_causal_te_validation"]
