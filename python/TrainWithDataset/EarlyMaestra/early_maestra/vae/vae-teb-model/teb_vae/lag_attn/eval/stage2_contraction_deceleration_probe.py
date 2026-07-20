r"""Stage 2: a strictly-causal, raw-signal, event-locked test of contraction$\to$deceleration.

The Stage-0 probes answered "does UP predict future FHR *features*?" across every representation and
found nothing. This is a different, more physiologically direct question, at the level the mechanism
actually lives: **does a uterine contraction causally precede a fetal-heart-rate deceleration?**
Contractions transiently reduce placental perfusion and the fetus responds with a deceleration
delayed by roughly $20$-$120\,$s. That response is a *transient event* in the raw FHR trace, which the
scattering / phase-harmonic feature representation may smear -- so a feature-level null does not
exclude it. This probe works on the raw $4\,$Hz signals directly.

It is model-free (no VAE, no checkpoint) and strictly causal (it only ever looks at FHR *after* a
contraction). Two complementary readouts, both event-locked to detected contractions:

* **Contraction-triggered FHR average (CTA).** Average the baseline-corrected raw FHR in a window
  around each contraction onset. If contractions cause decelerations, the average dips in the
  post-onset lag window. The control is a **random-trigger null**: the same average around random
  times (matched per recording), which preserves each signal's own statistics but breaks the
  contraction$\leftrightarrow$FHR pairing. A dip significantly deeper than the null is the signature.
* **Coincidence lift.** The fraction of contractions followed by a *detected* deceleration within the
  lag window, over the same fraction at random times. Lift $> 1$ beyond the null confirms it.

Events are detected with the repo's own detectors
(``model_raw/testing/causal_te_validation/events.py``), so the definitions match the causal-TE
validation suite.

**The $-20\,$s UP pre-shift.** The pipeline advances the UP trace $20\,$s earlier than FHR before
storage, so a contraction detected in the raw UP is $\approx 20\,$s "early": the *detected*
onset-to-deceleration lag is the true physiological delay plus $\approx 20\,$s. The lag window is
widened accordingly and the report states the peak-response lag so the offset can be read back.

Reading the result:

* **NO_COUPLING** -- the CTA dip is inside the null band and the coincidence lift is $\approx 1$: no
  time-locked deceleration response follows contractions. Combined with the Stage-0 feature nulls,
  this is a maximally-defensible negative -- accept it.
* **COUPLING_PRESENT** -- a dip significantly deeper than the null at a plausible lag: the coupling is
  real at the raw-signal level even though it is not extractable from the model's features. That is a
  *representational* finding -- the model's scattering/phase inputs are the wrong basis for it.

Run (from the repo root):

.. code-block:: bash

    python -m teb_vae.lag_attn.eval.stage2_contraction_deceleration_probe \
        --config teb_vae/lag_attn/configs/default.yaml

    # Correctness self-test (fabricated raw signals with a known coupling and a null; no dataset):
    python -m teb_vae.lag_attn.eval.stage2_contraction_deceleration_probe --self-test

From a PyCharm/IDE Run button (no command line): hit Run. With no arguments it uses the ``RUN_*``
constants near the bottom. Any ``--flag`` on the command line overrides them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model.vae_teb_prediction.model.model_raw.testing.causal_te_validation.events import (  # noqa: E402
    detect_contractions,
    detect_decelerations,
)

# The physiological contraction->deceleration delay is 20-120 s; the -20 s UP pre-shift adds ~20 s to
# the *detected* lag, so the response window is widened to 20-160 s post-onset. The pre window is the
# quiet baseline each snippet is corrected against.
_FS = 4.0                     # raw sampling rate (Hz)
_PRE_SECONDS = 30.0           # baseline window before the contraction onset
_POST_SECONDS = 170.0         # extent after onset (covers the widened response window plus margin)
# Where the deceleration response is searched, post detected onset. The physiological delay is
# 20-120 s from the contraction; the detector's onset is the rising edge (a few tens of seconds
# before the peak) and the -20 s UP pre-shift adds ~20 s, so the *detected* onset-to-dip lag spans
# roughly 10-150 s. The low end is kept small so a short effective lag is not missed.
_LAG_WINDOW_SECONDS = (10.0, 150.0)
_N_SHUFFLE = 300              # null draws
# A CTA dip clears the null only if it is beyond this z (deeper) AND at least this many bpm deep, so a
# statistically-significant but physiologically-trivial dip is not called coupling.
_Z_THRESHOLD = -3.0
_MIN_DIP_BPM = 1.0


@dataclass
class ProbeConfig:
    """Knobs for one run.

    Attributes:
        fs: Raw sampling rate (Hz).
        pre_seconds: Baseline window before onset each snippet is corrected against.
        post_seconds: Extent of each snippet after onset.
        lag_window_seconds: ``(lo, hi)`` post-onset window the deceleration response is searched in.
        n_shuffle: Number of random-trigger null draws.
        max_recordings: Cap on recordings loaded.
        seed: RNG seed for the null.
    """

    fs: float = _FS
    pre_seconds: float = _PRE_SECONDS
    post_seconds: float = _POST_SECONDS
    lag_window_seconds: Tuple[float, float] = _LAG_WINDOW_SECONDS
    n_shuffle: int = _N_SHUFFLE
    max_recordings: int = 1200
    seed: int = 0


@dataclass
class RawSample:
    """One recording's raw traces.

    Attributes:
        fhr: Raw FHR $(R,)$ at ``fs`` Hz (bpm when loaded un-normalised).
        up: Raw UP $(R,)$ at ``fs`` Hz.
        guid: Recording identifier.
    """

    fhr: np.ndarray
    up: np.ndarray
    guid: str


# --- Data loading -----------------------------------------------------------------------------
def load_raw_samples_from_config(
    config_path: str, *, split: str, max_samples: Optional[int]
) -> List[RawSample]:
    r"""Load the raw ``fhr``/``up`` traces, **un-normalised**, through the training loader.

    The raw signals are read without normalisation so FHR stays in bpm -- the deceleration detector's
    bpm path applies and dip depths are physiologically interpretable. ``trim_minutes`` still applies
    (matching the model's geometry), so both traces span the same $1200\,$s the decimated model sees.

    Args:
        config_path: Path to a leaf YAML config; its ``base:`` chain is resolved.
        split: ``'train'``, ``'test'`` or ``'both'``.
        max_samples: Cap on recordings loaded (``None`` for all).

    Returns:
        The loaded raw samples.

    Raises:
        ValueError: If the split names no shards, or a recording lacks ``fhr``/``up``.
    """
    from teb_vae.lag_attn.config import load_config
    from hdf5_dataset.hdf5_dataset import CombinedHDF5Dataset

    config = load_config(config_path)
    dataset_config = config.get("dataset_config", {}) or {}
    dataloader_config = dataset_config.get("dataloader_config", {}) or {}

    train_shards = list(dataset_config.get("vae_train_datasets") or [])
    test_shards = list(dataset_config.get("vae_test_datasets") or [])
    shards = {"train": train_shards, "test": test_shards, "both": train_shards + test_shards}.get(split)
    if not shards:
        raise ValueError(f"split={split!r} names no shards in {config_path}")

    accepted = {
        "load_fields", "allowed_guids", "cs_label", "bg_label", "epoch_min", "epoch_max",
        "label", "pin_memory", "trim_minutes",
    }
    raw_kwargs = dict(dataloader_config.get("dataset_kwargs") or {})
    dataset_kwargs = {name: value for name, value in raw_kwargs.items() if name in accepted}
    # Load only the raw traces (plus guid), and DO NOT normalise them: the detectors want bpm.
    dataset_kwargs["load_fields"] = ["fhr", "up", "guid"]

    dataset = CombinedHDF5Dataset(
        paths=shards,
        stats_path=dataset_config.get("stat_path"),
        normalize_fields=[],  # empty -> raw fhr/up in native units
        cache_size=0,
        **dataset_kwargs,
    )

    count = len(dataset)
    if max_samples is not None:
        count = min(count, int(max_samples))

    samples: List[RawSample] = []
    for index in range(count):
        item = dataset[index]
        fhr = getattr(item, "fhr", None)
        up = getattr(item, "up", None)
        if fhr is None or up is None:
            raise ValueError("a recording is missing 'fhr' or 'up'; both are required for Stage 2")
        fhr_np = np.asarray(fhr.numpy() if hasattr(fhr, "numpy") else fhr, dtype=np.float32).reshape(-1)
        up_np = np.asarray(up.numpy() if hasattr(up, "numpy") else up, dtype=np.float32).reshape(-1)
        guid = str(getattr(item, "guid", f"_row{index}"))
        samples.append(RawSample(fhr=fhr_np, up=up_np, guid=guid))
    return samples


# --- Event detection --------------------------------------------------------------------------
def detect_events(
    samples: Sequence[RawSample], config: ProbeConfig
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    r"""Detect contraction onsets (UP) and deceleration onsets (FHR) per recording.

    Uses the repo's own detectors so the event definitions match the causal-TE validation suite.

    Args:
        samples: The raw recordings.
        config: The probe configuration.

    Returns:
        ``(contraction_onsets, deceleration_onsets)``: two parallel lists (one array of raw-sample
        onset indices per recording).
    """
    contraction_onsets: List[np.ndarray] = []
    deceleration_onsets: List[np.ndarray] = []
    for sample in samples:
        contractions = detect_contractions(sample.up, fs=config.fs)
        decelerations = detect_decelerations(sample.fhr, fs=config.fs)
        contraction_onsets.append(np.asarray(contractions["onset_raw"], dtype=np.int64))
        deceleration_onsets.append(np.asarray(decelerations["onset_raw"], dtype=np.int64))
    return contraction_onsets, deceleration_onsets


# --- Contraction-triggered average ------------------------------------------------------------
def _gather_snippets(
    fhr: np.ndarray, triggers: np.ndarray, pre: int, post: int
) -> np.ndarray:
    r"""Return baseline-corrected FHR snippets around each trigger.

    Only triggers whose full ``[t-pre, t+post)`` window fits inside the trace are kept; each snippet
    is corrected by subtracting its own pre-onset mean, so the average measures the FHR change
    *relative to the pre-contraction level* rather than the absolute bpm.

    Args:
        fhr: Raw FHR trace $(R,)$.
        triggers: Raw-sample trigger indices.
        pre: Baseline samples before the trigger.
        post: Samples after the trigger.

    Returns:
        A $(k, \mathrm{pre}+\mathrm{post})$ array of baseline-corrected snippets ($k$ may be $0$).
    """
    n = fhr.shape[0]
    valid = triggers[(triggers >= pre) & (triggers < n - post)]
    if valid.size == 0:
        return np.empty((0, pre + post), dtype=np.float32)
    offsets = np.arange(-pre, post, dtype=np.int64)
    snippets = fhr[valid[:, None] + offsets[None, :]]
    baseline = snippets[:, :pre].mean(axis=1, keepdims=True)
    return (snippets - baseline).astype(np.float32)


def contraction_triggered_average(
    samples: Sequence[RawSample],
    contraction_onsets: Sequence[np.ndarray],
    config: ProbeConfig,
) -> Tuple[np.ndarray, int]:
    """Average baseline-corrected FHR over all contraction onsets.

    Args:
        samples: The raw recordings.
        contraction_onsets: Per-recording contraction onset indices.
        config: The probe configuration.

    Returns:
        ``(cta, n_events)``: the mean snippet $(\\mathrm{pre}+\\mathrm{post},)$ and the number of
        contributing contractions.
    """
    pre = int(round(config.pre_seconds * config.fs))
    post = int(round(config.post_seconds * config.fs))
    total = np.zeros(pre + post, dtype=np.float64)
    count = 0
    for sample, onsets in zip(samples, contraction_onsets):
        snippets = _gather_snippets(sample.fhr, onsets, pre, post)
        if snippets.shape[0]:
            total += snippets.sum(axis=0)
            count += snippets.shape[0]
    cta = (total / count).astype(np.float32) if count else total.astype(np.float32)
    return cta, count


def _dip_statistic(cta: np.ndarray, config: ProbeConfig) -> Tuple[float, float]:
    """Return the deepest FHR dip in the response window and the lag (s) at which it occurs."""
    pre = int(round(config.pre_seconds * config.fs))
    lag_lo = pre + int(round(config.lag_window_seconds[0] * config.fs))
    lag_hi = pre + int(round(config.lag_window_seconds[1] * config.fs))
    lag_hi = min(lag_hi, cta.shape[0])
    if lag_hi <= lag_lo:
        return 0.0, 0.0
    window = cta[lag_lo:lag_hi]
    rel_idx = int(np.argmin(window))
    dip = float(window[rel_idx])
    lag_seconds = (lag_lo + rel_idx - pre) / config.fs
    return dip, lag_seconds


def _null_dip_distribution(
    samples: Sequence[RawSample],
    contraction_onsets: Sequence[np.ndarray],
    config: ProbeConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Random-trigger null: the dip statistic under contractions replaced by random times.

    For each shuffle, every recording contributes the *same number* of triggers as it has real
    contractions, drawn uniformly from its valid range -- so the null matches the per-recording event
    count and each recording's own FHR statistics, and differs from the observed average only in that
    the triggers are decoupled from the contractions.

    Args:
        samples: The raw recordings.
        contraction_onsets: Per-recording contraction onset indices.
        config: The probe configuration.
        rng: RNG for the random triggers.

    Returns:
        A $(\mathrm{n\_shuffle},)$ array of null dip statistics.
    """
    pre = int(round(config.pre_seconds * config.fs))
    post = int(round(config.post_seconds * config.fs))
    counts = [int((o[(o >= pre) & (o < s.fhr.shape[0] - post)]).size)
              for s, o in zip(samples, contraction_onsets)]

    null_stats = np.empty(config.n_shuffle, dtype=np.float32)
    for shuffle in range(config.n_shuffle):
        total = np.zeros(pre + post, dtype=np.float64)
        count = 0
        for sample, k in zip(samples, counts):
            if k <= 0:
                continue
            n = sample.fhr.shape[0]
            if n - post <= pre:
                continue
            triggers = rng.integers(pre, n - post, size=k)
            snippets = _gather_snippets(sample.fhr, triggers, pre, post)
            if snippets.shape[0]:
                total += snippets.sum(axis=0)
                count += snippets.shape[0]
        cta_null = (total / count).astype(np.float32) if count else total.astype(np.float32)
        null_stats[shuffle], _ = _dip_statistic(cta_null, config)
    return null_stats


# --- Coincidence lift -------------------------------------------------------------------------
def _coincidence_rate(
    contraction_onsets: Sequence[np.ndarray],
    deceleration_onsets: Sequence[np.ndarray],
    config: ProbeConfig,
) -> Tuple[float, int, int]:
    """Fraction of contractions followed by a deceleration onset within the lag window.

    Args:
        contraction_onsets: Per-recording contraction onsets.
        deceleration_onsets: Per-recording deceleration onsets.
        config: The probe configuration.

    Returns:
        ``(rate, hits, total)``: the coincidence fraction and its numerator / denominator.
    """
    lag_lo = int(round(config.lag_window_seconds[0] * config.fs))
    lag_hi = int(round(config.lag_window_seconds[1] * config.fs))
    hits = 0
    total = 0
    for con, dec in zip(contraction_onsets, deceleration_onsets):
        if con.size == 0:
            continue
        total += int(con.size)
        if dec.size == 0:
            continue
        for c in con:
            lags = dec - c
            if np.any((lags >= lag_lo) & (lags <= lag_hi)):
                hits += 1
    rate = hits / total if total else 0.0
    return rate, hits, total


def _null_coincidence_distribution(
    samples: Sequence[RawSample],
    contraction_onsets: Sequence[np.ndarray],
    deceleration_onsets: Sequence[np.ndarray],
    config: ProbeConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Null coincidence rate with contraction times replaced by random times per recording."""
    lag_lo = int(round(config.lag_window_seconds[0] * config.fs))
    lag_hi = int(round(config.lag_window_seconds[1] * config.fs))
    counts = [int(o.size) for o in contraction_onsets]

    null_rates = np.empty(config.n_shuffle, dtype=np.float32)
    for shuffle in range(config.n_shuffle):
        hits = 0
        total = 0
        for sample, dec, k in zip(samples, deceleration_onsets, counts):
            if k <= 0:
                continue
            n = sample.fhr.shape[0]
            total += k
            if dec.size == 0:
                continue
            triggers = rng.integers(0, n, size=k)
            for c in triggers:
                lags = dec - c
                if np.any((lags >= lag_lo) & (lags <= lag_hi)):
                    hits += 1
        null_rates[shuffle] = hits / total if total else 0.0
    return null_rates


# --- Orchestration ----------------------------------------------------------------------------
def run_probe(samples: Sequence[RawSample], config: ProbeConfig) -> Dict[str, object]:
    r"""Detect events, compute the CTA and coincidence lift with their nulls, and return a verdict.

    Args:
        samples: The raw recordings.
        config: The probe configuration.

    Returns:
        A JSON-serialisable results dict.
    """
    rng = np.random.default_rng(config.seed)
    contraction_onsets, deceleration_onsets = detect_events(samples, config)

    n_contractions = int(sum(int(o.size) for o in contraction_onsets))
    n_decelerations = int(sum(int(o.size) for o in deceleration_onsets))

    cta, n_events = contraction_triggered_average(samples, contraction_onsets, config)
    dip_obs, dip_lag = _dip_statistic(cta, config)
    null_dips = _null_dip_distribution(samples, contraction_onsets, config, rng)
    null_mean = float(null_dips.mean()) if null_dips.size else 0.0
    null_std = float(null_dips.std()) or 1e-6
    dip_z = (dip_obs - null_mean) / null_std
    dip_p = float((null_dips <= dip_obs).mean()) if null_dips.size else 1.0

    rate_obs, hits, total = _coincidence_rate(contraction_onsets, deceleration_onsets, config)
    null_rates = _null_coincidence_distribution(
        samples, contraction_onsets, deceleration_onsets, config, rng
    )
    rate_null_mean = float(null_rates.mean()) if null_rates.size else 0.0
    lift = rate_obs / rate_null_mean if rate_null_mean > 0 else 0.0
    rate_p = float((null_rates >= rate_obs).mean()) if null_rates.size else 1.0

    # Coupling requires a dip that is both statistically beyond the null and physiologically real.
    dip_significant = dip_z <= _Z_THRESHOLD and dip_obs <= -_MIN_DIP_BPM
    verdict = "coupling_present" if dip_significant else "no_coupling"

    return {
        "verdict": verdict,
        "n_recordings": int(len(samples)),
        "n_contractions": n_contractions,
        "n_decelerations": n_decelerations,
        "cta_n_events": int(n_events),
        "dip_bpm": dip_obs,
        "dip_lag_seconds": dip_lag,
        "dip_null_mean_bpm": null_mean,
        "dip_null_std_bpm": null_std,
        "dip_z": float(dip_z),
        "dip_p_value": dip_p,
        "coincidence_rate": rate_obs,
        "coincidence_null_rate": rate_null_mean,
        "coincidence_lift": float(lift),
        "coincidence_p_value": rate_p,
        "coincidence_hits": int(hits),
        "coincidence_total": int(total),
        "lag_window_seconds": list(config.lag_window_seconds),
        "up_preshift_note_seconds": 20.0,
        "z_threshold": _Z_THRESHOLD,
        "min_dip_bpm": _MIN_DIP_BPM,
    }


# --- Reporting --------------------------------------------------------------------------------
def format_report(results: Dict[str, object]) -> str:
    """Render a human-readable report of a probe result.

    Args:
        results: The dict from :func:`run_probe`.

    Returns:
        The formatted multi-line report.
    """
    def num(key: str) -> float:
        return float(results[key])  # type: ignore[arg-type]

    lines = ["", "=" * 74, "Stage 2 - raw contraction->deceleration event-locked probe", "=" * 74]
    lines.append(
        f"recordings={results['n_recordings']}  contractions={results['n_contractions']}  "
        f"decelerations={results['n_decelerations']}"
    )
    lines.append(f"lag window = {results['lag_window_seconds']} s post-onset "
                 f"(includes the ~{int(num('up_preshift_note_seconds'))} s UP pre-shift)")
    lines.append("")
    lines.append("  Contraction-triggered FHR average (deepest dip in the lag window):")
    lines.append(f"     observed dip        = {num('dip_bpm'):+.3f} bpm  at lag {num('dip_lag_seconds'):.0f} s")
    lines.append(f"     random-trigger null = {num('dip_null_mean_bpm'):+.3f} +/- {num('dip_null_std_bpm'):.3f} bpm")
    lines.append(f"     z vs null           = {num('dip_z'):+.2f}   (p = {num('dip_p_value'):.4f})")
    lines.append("")
    lines.append("  Coincidence (deceleration within the lag window after a contraction):")
    lines.append(f"     observed rate       = {num('coincidence_rate')*100:.1f} %  "
                 f"({results['coincidence_hits']}/{results['coincidence_total']})")
    lines.append(f"     null rate           = {num('coincidence_null_rate')*100:.1f} %")
    lines.append(f"     lift                = {num('coincidence_lift'):.2f}x   (p = {num('coincidence_p_value'):.4f})")
    lines.append("")
    lines.append(f"  VERDICT: {str(results['verdict']).upper()}")
    lines.append(_verdict_gloss(str(results["verdict"])))
    lines.append("=" * 74)
    return "\n".join(lines)


def _verdict_gloss(verdict: str) -> str:
    """One-line interpretation for each verdict slug."""
    return {
        "coupling_present": (
            "  -> FHR dips significantly after contractions, beyond the random-trigger null, at a\n"
            "     plausible lag. The contraction->deceleration coupling is real at the raw-signal\n"
            "     level even though it is not extractable from the model's features -- a\n"
            "     REPRESENTATIONAL finding: scattering/phase is the wrong basis for it."
        ),
        "no_coupling": (
            "  -> No time-locked deceleration response follows contractions (dip inside the null\n"
            "     band). With the Stage-0 feature nulls, this is a maximally-defensible negative:\n"
            "     no detectable contraction->deceleration coupling in these recordings. Accept it."
        ),
    }.get(verdict, "")


# --- Self-test (no dataset required) ----------------------------------------------------------
def _fabricate_raw_samples(
    *, coupled: bool, n_records: int, rng: np.random.Generator
) -> List[RawSample]:
    r"""Fabricate raw recordings with or without a contraction$\to$deceleration coupling.

    Each recording is $1200\,$s at $4\,$Hz. Contractions are broad UP bumps ($\sim 50\,$s wide,
    spaced $\sim 120\,$s). FHR is a wandering $\sim 140\,$bpm baseline; when ``coupled``, a
    deceleration ($\sim -18\,$bpm, $\sim 30\,$s wide) is added at a fixed lag after each contraction,
    so the probe must find a CTA dip beyond the null at that lag. When not, FHR carries the same
    number of decelerations at *random* times, so the dip must stay inside the null band.

    Args:
        coupled: Whether decelerations follow contractions.
        n_records: Number of recordings.
        rng: RNG.

    Returns:
        The fabricated raw samples.
    """
    fs = 4.0
    r = int(1200 * fs)                       # 1200 s
    steps = np.arange(r, dtype=np.float32)
    # 90 s from the *fabricated* onset t0. The detector's onset lands ~30 s later than t0 (it walks
    # back only to the rising edge), so the detected onset-to-dip lag is ~60 s -- inside the window.
    inject_lag = int(90 * fs)
    con_width = int(50 * fs)
    dec_width = int(30 * fs)
    lo1, hi1 = int(40 * fs), int(90 * fs)
    lo2, hi2 = int(100 * fs), int(140 * fs)
    edge = int(60 * fs)

    samples: List[RawSample] = []
    for record in range(n_records):
        up = (0.2 * rng.standard_normal(r)).astype(np.float32)
        # A flat 140 bpm baseline plus fast noise -- no slow drift, so the average isolates the
        # contraction-locked response rather than a trend that could alias with contraction timing.
        fhr = (140.0 + 1.5 * rng.standard_normal(r)).astype(np.float32)

        # Contraction onsets ~ every 120 s with jitter.
        onsets: List[int] = []
        t = int(rng.integers(lo1, hi1))
        while t < r - int(200 * fs):
            onsets.append(t)
            t += int(rng.integers(lo2, hi2))
        for t0 in onsets:
            bump = np.exp(-0.5 * ((steps - (t0 + con_width)) / (con_width / 2.0)) ** 2)
            up += (2.5 * bump).astype(np.float32)

        # Decelerations: at a fixed lag after each contraction (coupled) or at random times (null).
        if coupled:
            dec_centers = [t0 + inject_lag for t0 in onsets]
        else:
            dec_centers = [int(rng.integers(edge, r - edge)) for _ in onsets]
        for c in dec_centers:
            dip = np.exp(-0.5 * ((steps - c) / (dec_width / 2.0)) ** 2)
            fhr -= (18.0 * dip).astype(np.float32)

        samples.append(RawSample(fhr=fhr, up=up, guid=f"rec{record}"))
    return samples


def self_test() -> int:
    """Validate the probe against a known coupling and a null; return a process exit code.

    Returns:
        ``0`` if both assertions pass, ``1`` otherwise.
    """
    rng = np.random.default_rng(0)
    config = ProbeConfig(n_shuffle=200, seed=0)

    coupled = run_probe(_fabricate_raw_samples(coupled=True, n_records=60, rng=rng), config)
    null = run_probe(_fabricate_raw_samples(coupled=False, n_records=60, rng=rng), config)

    print(format_report(coupled))
    print(format_report(null))

    ok = True
    if coupled["verdict"] != "coupling_present":
        print(f"FAIL: coupled data -> {coupled['verdict']} (expected coupling_present)")
        ok = False
    # The injected lag is 45 s; the detected onset-to-dip lag should land in a plausible band.
    if not (20.0 <= float(coupled["dip_lag_seconds"]) <= 120.0):  # type: ignore[arg-type]
        print(f"FAIL: coupled dip lag {coupled['dip_lag_seconds']} s is implausible")
        ok = False
    if null["verdict"] != "no_coupling":
        print(f"FAIL: null data -> {null['verdict']} (expected no_coupling)")
        ok = False
    print("\nSELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --- IDE Run-button configuration -------------------------------------------------------------
#: Used only when the module is launched with no command-line arguments (a PyCharm/IDE Run button).
#: Any ``--flag`` on the command line overrides these. See stage0's note on repo-root resolution.
RUN_CONFIG: Optional[str] = "teb_vae/lag_attn/configs/default.yaml"
RUN_SELF_TEST: bool = False       # True -> run the no-dataset self-test instead of a real probe
RUN_SPLIT: str = "test"           # 'train' | 'test' | 'both'
RUN_MAX_SAMPLES: int = 1200       # cap on recordings loaded
RUN_SEED: int = 0
RUN_JSON_OUT: Optional[str] = None


def _run_button_argv() -> List[str]:
    """Build a CLI ``argv`` from the ``RUN_*`` constants for a no-argument (IDE Run button) launch."""
    if RUN_SELF_TEST:
        return ["--self-test"]
    if RUN_CONFIG is None:
        raise SystemExit(
            "RUN_CONFIG is None: set it near the bottom of this file (or RUN_SELF_TEST=True), "
            "or launch with --config on the command line."
        )
    argv = ["--config", RUN_CONFIG, "--split", RUN_SPLIT, "--max-samples", str(RUN_MAX_SAMPLES),
            "--seed", str(RUN_SEED)]
    if RUN_JSON_OUT:
        argv += ["--json-out", RUN_JSON_OUT]
    return argv


# --- CLI --------------------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments and run either the self-test or a real probe.

    Args:
        argv: Argument vector (defaults to ``sys.argv``).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to the YAML config (resolves its base: chain).")
    parser.add_argument("--split", default="test", choices=("train", "test", "both"),
                        help="Which shard list to read (default: the held-out test shards).")
    parser.add_argument("--max-samples", type=int, default=1200, help="Cap on recordings loaded.")
    parser.add_argument("--n-shuffle", type=int, default=_N_SHUFFLE, help="Random-trigger null draws.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the null.")
    parser.add_argument("--json-out", default=None, help="Optional path to write the results dict as JSON.")
    parser.add_argument("--self-test", action="store_true", help="Run the no-dataset self-test and exit.")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.config is None:
        parser.error("--config is required unless --self-test is given")

    config = ProbeConfig(max_recordings=args.max_samples, n_shuffle=args.n_shuffle, seed=args.seed)
    samples = load_raw_samples_from_config(args.config, split=args.split, max_samples=args.max_samples)
    print(f"loaded {len(samples)} raw recordings from the {args.split} split of {args.config}")
    results = run_probe(samples, config)
    print(format_report(results))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _argv = _run_button_argv()
        if os.path.abspath(os.getcwd()) != _REPO_ROOT:
            print(f"no CLI args (IDE Run button); using RUN_* constants and chdir to {_REPO_ROOT}")
            os.chdir(_REPO_ROOT)
        raise SystemExit(main(_argv))
    raise SystemExit(main())
