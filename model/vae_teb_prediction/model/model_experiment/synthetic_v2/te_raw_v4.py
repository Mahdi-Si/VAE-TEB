r"""S1-T02: the re-targeted model-free raw-TE realizability probe for ``synthetic_v4``.

The v2/v3 probe (:func:`eval_v2.measure_te_raw`) assumes the coupling is **amplitude-modulated**
onto a carrier at $f_{\mathrm{pulse}}$, so it band-passes to the carrier band and Hilbert-
demodulates before the block-TE estimator. In ``synthetic_v4``'s primary ``direct`` render there is
**no carrier**: the $c\to d$ coupling lives at *low frequency*, so that same band-pass would filter
it away. :func:`measure_te_raw_v4` therefore dispatches by render mode:

* ``direct`` / ``pulse_train`` -- **no** band-pass: Fourier-decimate $5280\to330$, per-channel
  z-score, shape $(n,330,1)$, and run the identical held-out ridge block-TE estimator
  (:func:`_r0_gain_over_anchors` -> :func:`realizable_te_block_from_arrays`) with explicit
  $K=$``data.K_history`` and $H=$``data.horizon`` and source-lag scope $\le D$;
* ``am_carrier`` -- delegate to :func:`eval_v2.measure_te_raw` (the band-pass + Hilbert path).

Both branches feed the *same* estimator, so on a ``direct``-rendered cell the direct branch (which
keeps the low-frequency coupling) recovers a larger $\mathrm{TE}_{\mathrm{raw}}$ than the band-pass
branch (which removes it) -- the discriminating check the realizability pre-flight relies on.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.analytic_te import (
    snr_per_step_for_te_block,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_v2 import (
    _fourier_decimate,
    _probe_knobs,
    _r0_gain_over_anchors,
    _zscore_channel,
    measure_te_raw,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.raw_generators import DECIMATION

#: Render modes whose coupling lives at low frequency (no carrier) -> no band-pass in the probe.
_LOWFREQ_RENDER_MODES = ("direct", "pulse_train")


def measure_te_raw_v4(
    fhr_raw: np.ndarray,
    up_raw: np.ndarray,
    *,
    D: int,
    render_mode: str,
    config: Dict[str, Any],
    benchmark: str = "G1_raw_v4",
    trim: int = 0,
) -> Dict[str, Any]:
    r"""Estimate the model-free raw block TE $\mathrm{TE}_{\mathrm{raw}}$, dispatched by render mode.

    Args:
        fhr_raw: FHR waveform(s) $(n, N)$ at $4$ Hz (untrimmed $N=5280$).
        up_raw: UP waveform(s) $(n, N)$.
        D: The cell's fixed lag $D$ (the probe's upper source-lag scope).
        render_mode: ``direct`` / ``pulse_train`` (low-frequency, no band-pass) or ``am_carrier``
            (delegates to :func:`eval_v2.measure_te_raw`).
        config: The parsed ``config_synth_v4.yaml`` tree.
        benchmark: Active benchmark key under ``benchmarks``.
        trim: Symmetric edge trim (in decimated steps) for the ``direct`` path; ``0`` keeps the full
            $(n,330,1)$ grid (the spec default).

    Returns:
        ``{te_raw, snr_per_step, n_used, ill_fraction, render_mode}``.

    Raises:
        ValueError: On a non-2-D input or an unknown ``render_mode``.
    """
    fhr_raw = np.asarray(fhr_raw, dtype=float)
    up_raw = np.asarray(up_raw, dtype=float)
    if fhr_raw.ndim != 2 or up_raw.ndim != 2:
        raise ValueError("measure_te_raw_v4: fhr_raw and up_raw must be (n, N).")

    rm = str(render_mode)

    if rm == "am_carrier":
        res = measure_te_raw(fhr_raw, up_raw, D=D, config=config, benchmark=benchmark)
        return {**res, "render_mode": "am_carrier"}

    if rm not in _LOWFREQ_RENDER_MODES:
        raise ValueError(
            f"measure_te_raw_v4: unknown render_mode {rm!r} "
            f"(expected 'direct', 'pulse_train', or 'am_carrier')."
        )

    # Low-frequency (direct/pulse_train) path: NO band-pass, NO demodulation. Decimate straight to
    # the analysis grid and run the held-out ridge block-TE estimator with explicit K / H.
    knobs = _probe_knobs(config, benchmark)
    q = int(DECIMATION)
    n_dec = int(fhr_raw.shape[1] // q)
    fhr_dec = _fourier_decimate(fhr_raw, n_dec)
    up_dec = _fourier_decimate(up_raw, n_dec)
    if trim > 0:
        fhr_dec = fhr_dec[:, trim : n_dec - trim]
        up_dec = up_dec[:, trim : n_dec - trim]

    # Per-channel z-score before the probe (its Gram-scaled ridge must not be inflated by DC).
    Y = np.ascontiguousarray(_zscore_channel(fhr_dec)[:, :, None])   # (n, 330, 1)
    U = np.ascontiguousarray(_zscore_channel(up_dec)[:, :, None])    # (n, 330, 1)

    r0 = _r0_gain_over_anchors(
        Y, U, K=int(knobs["K"]), H=int(knobs["H"]), delay_max=int(D), ridge=float(knobs["ridge"]),
        n_anchors=int(knobs["n_anchors"]), n_seeds=int(knobs["n_seeds"]),
    )
    te_raw = r0["gain"]
    snr = (float(snr_per_step_for_te_block(te_raw, int(knobs["H"]), 1))
           if np.isfinite(te_raw) and te_raw > 0.0 else 0.0)
    return {
        "te_raw": float(te_raw) if np.isfinite(te_raw) else float("nan"),
        "snr_per_step": snr,
        "n_used": int(r0["n_used"]),
        "ill_fraction": float(r0["ill_fraction"]),
        "render_mode": rm,
    }
