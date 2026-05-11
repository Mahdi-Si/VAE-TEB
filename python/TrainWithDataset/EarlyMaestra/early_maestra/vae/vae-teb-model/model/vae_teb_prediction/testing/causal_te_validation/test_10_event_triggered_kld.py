"""Test 10 — Event-triggered KLD and TE-lag response.

For each sample, compute the per-time bottleneck information $K_t$,
attention concentration $C_t$, and the lag-TE peak
$\\max_\\ell \\widetilde{\\mathrm{TE}}_{t,\\ell}$ inside event windows
$W^{+}$ (within $\\pm 1$ decim step of any UP-onset / UP-peak /
FHR-deceleration nadir) and quiet windows $W^{-}$ (everything else
outside warmup and trailing horizon-exclusion).

Decision rule: $\\Delta K, \\Delta C, \\Delta \\widetilde{\\mathrm{TE}}$
are each significantly $> 0$ (paired one-sample Wilcoxon $p < 0.05$).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from model.vae_teb_prediction.testing.causal_te_validation.events import (
    detect_contractions,
    detect_decelerations,
    label_event_windows,
    quiet_mask,
    raw_to_decim_index,
)
from model.vae_teb_prediction.testing.causal_te_validation.statistics import (
    paired_wilcoxon,
)


_FS_RAW: float = 4.0
_DECIM_FACTOR: int = 16


def _attention_concentration(attn: np.ndarray) -> np.ndarray:
    """Per-time concentration $C_t = 1 - H(\\bar\\alpha_{t,\\cdot})/\\log(L)$.

    Args:
        attn: ``(T, M, L)`` or ``(T, L)`` head-resolved attention.

    Returns:
        ``(T,)`` float array in $[0, 1]$.
    """
    if attn.ndim == 3:
        a = attn.mean(axis=1)
    else:
        a = attn
    eps = 1e-12
    p = np.clip(a, eps, 1.0)
    H = -np.sum(p * np.log(p), axis=-1)
    L = a.shape[-1]
    norm = float(np.log(max(int(L), 2)))
    return np.asarray(1.0 - H / norm, dtype=np.float32)


def _process_sample(
    sample: Dict[str, Any],
    *,
    warmup: int,
    horizon: int,
    onset_pad: int = 1,
    peak_pad: int = 1,
) -> Dict[str, Any]:
    """Compute event/quiet means for a single sample.

    Returns ``None``-equivalent NaN row when the sample has no events.
    """
    fhr_raw = sample.get("fhr")
    up_raw = sample.get("up")
    te_lag_raw = sample.get("te_lag")
    attn_raw = sample.get("attn")
    kld_t_raw = sample.get("kld_t")
    if any(x is None for x in (fhr_raw, up_raw, te_lag_raw, attn_raw, kld_t_raw)):
        return {"valid": False}
    te_lag = np.asarray(te_lag_raw, dtype=np.float32)
    attn = np.asarray(attn_raw, dtype=np.float32)
    kld_t = np.asarray(kld_t_raw, dtype=np.float32)
    if te_lag.ndim != 2 or attn.ndim != 3 or kld_t.ndim != 1:
        return {"valid": False}

    T_dec = int(te_lag.shape[0])
    up_events = detect_contractions(np.asarray(up_raw, dtype=np.float32), fs=_FS_RAW)
    fhr_events = detect_decelerations(np.asarray(fhr_raw, dtype=np.float32), fs=_FS_RAW)

    event_indices_raw = np.concatenate([
        up_events["onset_raw"], up_events["peak_raw"], fhr_events["nadir_raw"],
    ])
    if event_indices_raw.size == 0:
        return {"valid": False, "n_events": 0}

    events_dec = raw_to_decim_index(
        event_indices_raw, decim=_DECIM_FACTOR, T_dec=T_dec,
    )
    event_mask = label_event_windows(
        events_dec, T_dec=T_dec,
        onset_pad=int(onset_pad), peak_pad=int(peak_pad),
        warmup=int(warmup), horizon_exclude=int(horizon),
    )
    quiet = quiet_mask(
        event_mask, warmup=int(warmup), horizon_exclude=int(horizon),
    )

    if not event_mask.any() or not quiet.any():
        return {"valid": False, "n_events": int(events_dec.size)}

    conc = _attention_concentration(attn)
    te_max_t = np.nanmax(te_lag, axis=-1).astype(np.float32)

    K_event = float(np.nanmean(kld_t[event_mask]))
    K_quiet = float(np.nanmean(kld_t[quiet]))
    C_event = float(np.nanmean(conc[event_mask]))
    C_quiet = float(np.nanmean(conc[quiet]))
    te_event = float(np.nanmean(te_max_t[event_mask]))
    te_quiet = float(np.nanmean(te_max_t[quiet]))

    return {
        "valid": True,
        "guid": sample.get("guid"),
        "epoch": sample.get("epoch"),
        "label": sample.get("label"),
        "n_events": int(events_dec.size),
        "K_event": K_event, "K_quiet": K_quiet,
        "C_event": C_event, "C_quiet": C_quiet,
        "te_max_event": te_event, "te_max_quiet": te_quiet,
    }


def run(
    *,
    samples: List[Dict[str, Any]],
    warmup: int,
    horizon: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """Compute event-triggered KLD/TE deltas per sample and aggregate.

    Args:
        samples: As in :func:`test_04_lag_event_alignment.run`.
        warmup: Model warmup steps.
        horizon: Model forecast horizon steps.
        output_dir: ``<output>/causal_te_validation/event_triggered_kld``.

    Returns:
        Dict with ``verdict``, ``evidence``, ``csv_paths``, ``figure_paths``.
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
    for s in samples:
        res = _process_sample(s, warmup=int(warmup), horizon=int(horizon))
        if res.get("valid"):
            rows.append(res)

    csv_path = output_dir / "per_sample_event_quiet.csv"
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    if df.empty:
        evidence = {
            "delta_K_positive": False,
            "delta_C_positive": False,
            "delta_TE_positive": False,
            "n_samples_with_events": 0,
        }
        from model.vae_teb_prediction.testing.causal_te_validation.decision_rules import (
            verdict_test_10_event_kld,
        )
        return {
            "verdict": verdict_test_10_event_kld(evidence),
            "evidence": evidence,
            "csv_paths": [str(csv_path)],
            "figure_paths": [],
        }

    deltaK = (df["K_event"] - df["K_quiet"]).to_numpy(dtype=np.float64)
    deltaC = (df["C_event"] - df["C_quiet"]).to_numpy(dtype=np.float64)
    deltaTE = (df["te_max_event"] - df["te_max_quiet"]).to_numpy(dtype=np.float64)
    wK = paired_wilcoxon(deltaK, alternative="greater")
    wC = paired_wilcoxon(deltaC, alternative="greater")
    wTE = paired_wilcoxon(deltaTE, alternative="greater")

    summary_csv = output_dir / "summary_deltas.csv"
    pd.DataFrame([
        {"metric": "K", **wK},
        {"metric": "C", **wC},
        {"metric": "te_max", **wTE},
    ]).to_csv(summary_csv, index=False)

    def _passes(w: Dict[str, float]) -> bool:
        p = float(w.get("p_value", float("nan")))
        m = float(w.get("median_delta", float("nan")))
        return bool(np.isfinite(p) and p < 0.05 and m > 0.0)

    evidence = {
        "delta_K_positive": _passes(wK),
        "delta_C_positive": _passes(wC),
        "delta_TE_positive": _passes(wTE),
        "median_delta_K": float(wK.get("median_delta", float("nan"))),
        "median_delta_C": float(wC.get("median_delta", float("nan"))),
        "median_delta_TE": float(wTE.get("median_delta", float("nan"))),
        "p_K": float(wK.get("p_value", float("nan"))),
        "p_C": float(wC.get("p_value", float("nan"))),
        "p_TE": float(wTE.get("p_value", float("nan"))),
        "n_samples_with_events": int(len(df)),
    }

    from model.vae_teb_prediction.testing.causal_te_validation.decision_rules import (
        verdict_test_10_event_kld,
    )
    verdict = verdict_test_10_event_kld(evidence)

    return {
        "verdict": verdict,
        "evidence": evidence,
        "csv_paths": [str(csv_path), str(summary_csv)],
        "figure_paths": [],
    }


__all__ = ["run"]
