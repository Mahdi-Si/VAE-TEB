"""Test 4 — Lag-attention validity via UP-event $\\to$ FHR-deceleration alignment.

For each sample with at least one matched (UP_contraction, FHR_deceleration)
pair within the lag window, compare:

* Empirical delay
  $d^{\\mathrm{event}} = 4 (t^{\\mathrm{FHR}}_{\\mathrm{dec}} - t^{\\mathrm{UP}}_{\\mathrm{dec}})$ seconds
  (raw 4 Hz indices mapped to the decimated grid).
* Model lag-TE peak per anchor
  $\\widetilde{\\mathrm{TE}}_{i,t,\\ell} = K_{i,t} \\bar\\alpha_{i,t,\\ell}$,
  with $d^{\\mathrm{model}}_{i,t} = 4 \\arg\\max_\\ell \\widetilde{\\mathrm{TE}}_{i,t,\\ell} - 20$ s
  (the $-20\\,$s accounts for the dataset's UP pre-shift).

Decision rule: median absolute alignment error
$|d^{\\mathrm{model}} - d^{\\mathrm{event}}| < 30\\,$s **and**
$K_{t}$ at event-aligned anchors exceeds quiet-window baseline KLD
(Wilcoxon).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from model.vae_teb_prediction.testing.causal_te_validation.events import (
    detect_contractions,
    detect_decelerations,
    pair_events_by_window,
    raw_to_decim_index,
)
from model.vae_teb_prediction.testing.causal_te_validation.statistics import (
    paired_wilcoxon,
)


# Per the dataset preprocessing, UP is shifted earlier by 20 s before the
# model sees it; this is the constant we subtract from the model's lag
# peak (in seconds) to put it on the same time grid as the empirical
# UP$\\to$FHR delay.
_UP_PRESHIFT_SECONDS: float = 20.0
_DECIM_FACTOR: int = 16
_FS_RAW: float = 4.0
_DECIM_STEP_SECONDS: float = float(_DECIM_FACTOR) / _FS_RAW  # 4 s/step


def _alpha_bar(attn: np.ndarray) -> np.ndarray:
    """Head-average ``attn (T, M, L)`` -> ``(T, L)``."""
    if attn.ndim == 3:
        return attn.mean(axis=1).astype(np.float32)
    return attn.astype(np.float32)


def _attention_concentration_per_t(attn: np.ndarray) -> np.ndarray:
    """Compute $C_{t} = 1 - H(\\bar\\alpha_{t,\\cdot}) / \\log(L)$.

    Entropy of a head-averaged distribution as a per-time-step scalar.
    Uses natural log; concentration is dimensionless in $[0, 1]$.

    Args:
        attn: ``(T, M, L)`` or ``(T, L)`` attention array.

    Returns:
        ``(T,)`` concentration trace.
    """
    a = _alpha_bar(attn)
    eps = 1e-12
    p = np.clip(a, eps, 1.0)
    H = -np.sum(p * np.log(p), axis=-1)  # (T,)
    L = a.shape[-1]
    norm = float(np.log(max(L, 2)))
    return np.asarray(1.0 - H / norm, dtype=np.float32)


def _model_lag_seconds(
    te_lag: np.ndarray,
    *,
    kld_t: np.ndarray,
    attn: np.ndarray,
    t_dec: int,
    pad_steps: int = 4,
) -> float:
    """Compute $d^{\\mathrm{model}}_{i,t}$ in seconds at decimated step ``t_dec``.

    Aggregates the lag-TE map within $[t_{\\mathrm{dec}} - \\mathrm{pad}, t_{\\mathrm{dec}} + \\mathrm{pad}]$,
    averages, and reports
    $4 \\cdot \\arg\\max_\\ell \\widetilde{\\mathrm{TE}}_{i,t,\\ell} - 20$ s.

    The function prefers the model's own pre-computed ``te_lag_map``
    (which is $K_t \\cdot \\bar\\alpha$) but falls back to the
    on-the-fly product when the te-lag map is missing.
    """
    T_dec = int(te_lag.shape[0]) if te_lag is not None else int(attn.shape[0])
    lo = int(max(0, t_dec - pad_steps))
    hi = int(min(T_dec, t_dec + pad_steps + 1))
    if hi <= lo:
        return float("nan")
    if te_lag is not None and te_lag.size:
        win = te_lag[lo:hi]
    else:
        a = _alpha_bar(attn)[lo:hi]
        k = kld_t[lo:hi].reshape(-1, 1)
        win = a * k
    if not np.isfinite(win).any():
        return float("nan")
    mean_win = np.nanmean(win, axis=0)
    if not np.isfinite(mean_win).any():
        return float("nan")
    lag_idx = int(np.nanargmax(mean_win))
    return float(_DECIM_STEP_SECONDS * lag_idx - _UP_PRESHIFT_SECONDS)


def _process_sample(
    sample: Dict[str, Any],
    *,
    warmup: int,
    horizon: int,
    fs: float = _FS_RAW,
    max_lag_seconds: float = 360.0,
) -> List[Dict[str, Any]]:
    """Detect events, pair them, and compute alignment metrics for one sample.

    Returns a list of paired-event row dicts. Empty list when the sample
    has no detectable events or no valid pairs.
    """
    fhr_raw = sample.get("fhr")
    up_raw = sample.get("up")
    te_lag_raw = sample.get("te_lag")
    attn_raw = sample.get("attn")
    kld_t_raw = sample.get("kld_t")
    if any(x is None for x in (fhr_raw, up_raw, te_lag_raw, attn_raw, kld_t_raw)):
        return []
    te_lag = np.asarray(te_lag_raw, dtype=np.float32)
    attn = np.asarray(attn_raw, dtype=np.float32)
    kld_t = np.asarray(kld_t_raw, dtype=np.float32)
    if te_lag.ndim != 2 or attn.ndim != 3 or kld_t.ndim != 1:
        return []
    T_dec = int(te_lag.shape[0])

    up_events = detect_contractions(np.asarray(up_raw, dtype=np.float32), fs=fs)
    fhr_events = detect_decelerations(np.asarray(fhr_raw, dtype=np.float32), fs=fs)
    if up_events["peak_raw"].size == 0 or fhr_events["nadir_raw"].size == 0:
        return []

    up_peak_dec = raw_to_decim_index(up_events["peak_raw"], decim=_DECIM_FACTOR, T_dec=T_dec)
    fhr_nadir_dec = raw_to_decim_index(fhr_events["nadir_raw"], decim=_DECIM_FACTOR, T_dec=T_dec)
    max_lag_steps = int(round(max_lag_seconds / _DECIM_STEP_SECONDS))
    pairs = pair_events_by_window(
        up_peak_dec, fhr_nadir_dec, max_lag_steps=max_lag_steps,
    )
    if pairs is None or pairs.size == 0:
        return []

    # Drop pairs that fall in the warmup or trailing horizon-exclusion zone.
    valid_lo, valid_hi = int(warmup), int(T_dec - horizon)
    rows: List[Dict[str, Any]] = []
    conc = _attention_concentration_per_t(attn)
    label = sample.get("label")
    guid = sample.get("guid")
    epoch = sample.get("epoch")
    for u_dec, f_dec in pairs:
        if not (valid_lo <= u_dec < valid_hi):
            continue
        d_event_s = float(_DECIM_STEP_SECONDS * (int(f_dec) - int(u_dec)))
        d_model_s = _model_lag_seconds(
            te_lag, kld_t=kld_t, attn=attn, t_dec=int(u_dec),
        )
        K_at_event = float(np.nanmean(kld_t[max(0, u_dec - 2): u_dec + 3]))
        te_max_at_event = float(np.nanmax(te_lag[u_dec])) if u_dec < T_dec else float("nan")
        c_at_event = float(np.nanmean(conc[max(0, u_dec - 2): u_dec + 3]))
        rows.append({
            "guid": guid,
            "epoch": epoch,
            "label": label,
            "t_up_dec": int(u_dec),
            "t_decel_dec": int(f_dec),
            "d_event_s": d_event_s,
            "d_model_s": float(d_model_s),
            "abs_error_s": float(abs(d_model_s - d_event_s)) if np.isfinite(d_model_s) else float("nan"),
            "K_at_event": K_at_event,
            "te_max_at_event": te_max_at_event,
            "C_at_event": c_at_event,
        })
    return rows


def _baseline_kld_per_segment(
    samples: List[Dict[str, Any]], *, warmup: int, horizon: int,
) -> Dict[Any, float]:
    """Per-`(guid, epoch)` mean of $K_t$ over valid (non-warmup) anchors.

    Returns a lookup ``{(guid, epoch): mean_K_t}``. Each segment in the
    test set has a unique ``(guid, epoch)`` key, so this lets Test 4
    pair every event with the *same segment*'s baseline rather than
    collapsing every segment from a patient onto one value.
    """
    out: Dict[Any, float] = {}
    for s in samples:
        kld_t_raw = s.get("kld_t")
        if kld_t_raw is None:
            continue
        kld_t = np.asarray(kld_t_raw, dtype=np.float32)
        if kld_t.ndim != 1 or kld_t.size == 0:
            continue
        T_dec = kld_t.size
        lo = int(max(0, warmup))
        hi = int(min(T_dec, T_dec - horizon))
        if hi <= lo:
            continue
        m = float(np.nanmean(kld_t[lo:hi]))
        if not np.isfinite(m):
            continue
        key = (s.get("guid"), s.get("epoch"))
        out[key] = m
    return out


def run(
    *,
    samples: List[Dict[str, Any]],
    warmup: int,
    horizon: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run Test 4 against a list of pre-collected samples.

    Args:
        samples: List of sample dicts as returned by
            :func:`testing.collectors.collect_predictions` (must include
            ``fhr``, ``up``, ``te_lag``, ``attn``, ``kld_t``, ``guid``,
            ``epoch``, ``label``).
        warmup: Model warmup steps.
        horizon: Model forecast horizon steps.
        output_dir: ``<output>/causal_te_validation/lag_event_alignment``.

    Returns:
        Dict with ``verdict``, ``evidence``, ``csv_paths``,
        ``figure_paths``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not samples:
        return {
            "verdict": "missing", "evidence": {},
            "error": "no samples provided",
            "csv_paths": [], "figure_paths": [],
        }

    rows: List[Dict[str, Any]] = []
    n_samples_with_events = 0
    for s in samples:
        sample_rows = _process_sample(s, warmup=int(warmup), horizon=int(horizon))
        if sample_rows:
            n_samples_with_events += 1
        rows.extend(sample_rows)

    csv_path = output_dir / "event_pairs.csv"
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    if df.empty:
        evidence = {
            "n_pairs": 0,
            "n_samples_with_events": int(n_samples_with_events),
            "median_abs_error_s": float("nan"),
            "kld_enriched_at_event": False,
            "attention_at_lag_zero_only": False,
        }
        from model.vae_teb_prediction.testing.causal_te_validation.decision_rules import (
            verdict_test_04_lag_event,
        )
        return {
            "verdict": verdict_test_04_lag_event(evidence),
            "evidence": evidence,
            "csv_paths": [str(csv_path)],
            "figure_paths": [],
        }

    finite_err = df["abs_error_s"].dropna().to_numpy(dtype=np.float64)
    median_err = float(np.median(finite_err)) if finite_err.size else float("nan")

    # KLD enrichment test: for each *segment* (guid, epoch), compare the
    # mean K_t at event-aligned anchors against that segment's own
    # baseline mean K_t over the valid anchor range. Paired Wilcoxon on
    # the per-segment delta. Keying by ``(guid, epoch)`` keeps multiple
    # segments per patient distinct — collapsing on ``guid`` alone would
    # discard all but the last segment's baseline.
    base_lookup = _baseline_kld_per_segment(
        samples, warmup=int(warmup), horizon=int(horizon),
    )
    per_sample_event_means: List[float] = []
    per_sample_baseline_means: List[float] = []
    group_keys = ["guid", "epoch"] if "epoch" in df.columns else ["guid"]
    for key, sub in df.groupby(group_keys):
        ev = float(np.nanmean(sub["K_at_event"].to_numpy(dtype=np.float64)))
        if isinstance(key, tuple):
            base = base_lookup.get(key, float("nan"))
        else:
            base = base_lookup.get((key, None), float("nan"))
        if np.isfinite(ev) and np.isfinite(base):
            per_sample_event_means.append(ev)
            per_sample_baseline_means.append(base)
    deltas = np.asarray(per_sample_event_means) - np.asarray(per_sample_baseline_means)
    wilc = paired_wilcoxon(deltas, alternative="greater")
    kld_enriched = bool(
        np.isfinite(wilc.get("p_value", float("nan")))
        and wilc.get("p_value", 1.0) < 0.05
        and wilc.get("median_delta", 0.0) > 0.0
    )

    # Lag-zero shortcut detector: fraction of model lag peaks landing at
    # lag 0 (in raw seconds, $-20\\,$s after the pre-shift correction).
    n_at_lag_zero = int(np.sum(np.abs(df["d_model_s"].to_numpy() + _UP_PRESHIFT_SECONDS) <= _DECIM_STEP_SECONDS / 2))
    frac_lag_zero = float(n_at_lag_zero / max(int(len(df)), 1))

    evidence = {
        "n_pairs": int(len(df)),
        "n_samples_with_events": int(n_samples_with_events),
        "median_abs_error_s": median_err,
        "kld_enriched_at_event": kld_enriched,
        "kld_enrichment_p_value": float(wilc.get("p_value", float("nan"))),
        "kld_enrichment_median_delta": float(wilc.get("median_delta", float("nan"))),
        "attention_at_lag_zero_only": bool(frac_lag_zero > 0.85),
        "frac_model_lag_zero": frac_lag_zero,
    }

    from model.vae_teb_prediction.testing.causal_te_validation.decision_rules import (
        verdict_test_04_lag_event,
    )
    verdict = verdict_test_04_lag_event(evidence)

    return {
        "verdict": verdict,
        "evidence": evidence,
        "csv_paths": [str(csv_path)],
        "figure_paths": [],
    }


__all__ = ["run"]
