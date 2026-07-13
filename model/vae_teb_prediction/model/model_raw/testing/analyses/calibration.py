r"""Calibration analysis for the learned predictive distribution (G10, S5-T02b).

Under ``sigma_obs='learned'`` the decoder emits a per-element Gaussian
:math:`\mathcal{N}(\mu_{\mathrm{full}}, \sigma_{\mathrm{full}}^2)` rather than a bare point
forecast. This analysis scores that distribution as a distribution: negative log-likelihood
and CRPS (both proper scoring rules), central-interval coverage, and probability-integral-
transform reliability resolved by lead time.

Why it matters here specifically. The v3 model reports a transfer-entropy surrogate
:math:`K_{\mathrm{raw}}` derived from a *latent* KL. Nothing in that quantity checks whether
the model's uncertainty is honest -- a variance-collapsed decoder can post an excellent MSE
and a confident-looking :math:`K`, while its 95% intervals cover 60% of the truth. Coverage
and reliability are the diagnostics that catch it.

Every score is accompanied by the homoscedastic reference :math:`\hat{\sigma}` fitted to the
same residuals. It answers "does the learned variance head buy anything?", and it keeps the
report interpretable on a checkpoint trained with a fixed ``sigma_obs``, whose ``logvar_full``
head never received a gradient. The model does not record which likelihood it was trained
under, so that case cannot be detected automatically -- read ``nll_gain_over_constant``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from loguru import logger

from model.vae_teb_prediction.model.model_raw.testing.base import TestRunner
from model.vae_teb_prediction.model.model_raw.testing.collectors import collect_calibration

try:  # pragma: no cover - plotting is optional and must never fail the analysis
    from model.vae_teb_prediction.model.model_raw.testing.visualizers import (
        plot_coverage_vs_nominal,
        plot_reliability_curve,
        plot_sharpness_by_horizon,
    )

    _PLOTTING = True
except Exception as exc:  # noqa: BLE001
    logger.warning(f"calibration: plotting unavailable ({exc}); CSVs will still be written")
    _PLOTTING = False


def run_calibration_analysis(
    runner: TestRunner,
    loader: Any,
    max_samples: Optional[int] = 1000,
    output_dir: Optional[Union[str, Path]] = None,
    *,
    levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95),
    n_bins: int = 20,
) -> Dict[str, Any]:
    r"""Score the model's predictive distribution and write the calibration report.

    Args:
        runner: The configured :class:`TestRunner`.
        loader: Dataloader over the evaluation subset.
        max_samples: Cap on samples consumed; ``<= 0`` skips the analysis entirely.
        output_dir: Destination directory; defaults to ``runner.ensure_dir("calibration")``.
        levels: Nominal central-interval levels for coverage.
        n_bins: Quantile resolution of the reliability curves.

    Returns:
        The scalar summary dict, extended with the paths of the artefacts written. On a model
        with no ``logvar_full`` key the dict carries an ``error`` entry instead of raising, so
        the pipeline's step harness records the skip and continues.
    """
    if max_samples is not None and max_samples <= 0:
        logger.info("calibration: skipped (max_samples <= 0)")
        return {}

    try:
        collected = collect_calibration(
            runner, loader, max_samples, levels=levels, n_bins=n_bins
        )
    except RuntimeError as exc:
        logger.warning(f"calibration: {exc}")
        return {"error": str(exc)}

    per_sample = collected["per_sample"]
    per_horizon = collected["per_horizon"]
    reliability = collected["reliability"]
    summary: Dict[str, Any] = dict(collected["summary"])

    if per_sample.empty:
        logger.warning("calibration: no samples collected")
        return summary

    out = Path(output_dir) if output_dir is not None else runner.ensure_dir("calibration")
    out.mkdir(parents=True, exist_ok=True)

    per_sample.to_csv(out / "per_sample.csv", index=False)
    per_horizon.to_csv(out / "per_horizon.csv", index=False)
    if not reliability.empty:
        reliability.to_csv(out / "reliability.csv", index=False)

    if _PLOTTING:
        try:
            plot_reliability_curve(reliability, out / "reliability.pdf")
            plot_coverage_vs_nominal(per_sample, out / "coverage.pdf")
            plot_sharpness_by_horizon(
                per_horizon,
                out / "sharpness_by_horizon.pdf",
                constant_sigma=summary.get("constant_sigma"),
            )
        except Exception as exc:  # noqa: BLE001 - a bad figure must not lose the CSVs
            logger.warning(f"calibration: plotting failed ({exc})")

    summary["per_sample_csv"] = str(out / "per_sample.csv")
    summary["per_horizon_csv"] = str(out / "per_horizon.csv")
    with (out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    gain = summary.get("nll_gain_over_constant")
    if gain is not None and gain <= 0.0:
        logger.warning(
            "calibration: the learned variance head does not beat a single global sigma "
            "(nll_gain_over_constant={:.4f}). Either the checkpoint was trained with a fixed "
            "sigma_obs, or the variance head has not learned anything.",
            gain,
        )
    logger.info(f"calibration: wrote report to {out}")
    return summary
