r"""Sprint 6 (goal G-F): interventional lag attribution against a *known* true lag.

The model publishes ``te_lag_map``, a per-lag attribution of the KL surrogate. Whether that
attribution is *faithful* -- whether the forecast actually depends on the lags the attention
points at -- is the "Attention is not Explanation" question, and it is normally unanswerable
because the true lag is unknown. Here it is not: ``build_dataset_v2`` injects a single block
transfer entropy at a fixed per-cell lag :math:`D`.

So the question becomes an intervention. Mask a band of lags :math:`G` inside the attention,
re-run the forward, and record how much worse the forecast gets:

$$\Delta L_G \;=\; \mathcal{L}_{\mathrm{feat}}^{\text{masked }G} \;-\; \mathcal{L}_{\mathrm{feat}}.$$

Two partitions are scored. The **fixed physiologic bands** :math:`\{0\text{-}5, 6\text{-}15,
16\text{-}30, 31\text{-}60, 61\text{-}90\}` are comparable with the production plan. The
**per-cell true band** :math:`\mathcal{L}^\star = \{\max(0, D-H), \dots, D-1\}` gives a two-sided
gate that only ground truth makes possible: masking :math:`\mathcal{L}^\star` must degrade the
forecast, masking :math:`\{\ell \ge D\}` must not.

$$\Delta L_{\mathrm{rel}}(\mathcal{L}^\star) \;-\; \Delta L_{\mathrm{rel}}(\{\ell \ge D\})
  \;>\; \texttt{margin\_delta\_L}.$$

Three implementation facts drive the shape of this module.

* **The mask API is not per-sample.** ``SeqVaeLagAttnV3.forward(lag_band_mask=...)`` takes
  ``(L,)`` or ``(T, L)``; a ``(B, T, L)`` mask would be silently collapsed to sample ``0``'s row
  by ``LagCrossAttention``. The per-cell true band depends on :math:`D`, which varies within a
  batch, so the batch is **bucketed by its distinct** ``delay`` **values** (at most three on this
  grid) and one forward is run per bucket, keeping only the matching rows.
* **The forecast loss must be the same number ``eval`` prints.** It is:
  :func:`eval_v2._masked_forecast_loss` is the single source of truth for both, so
  :math:`\Delta L` is a difference of two identically-normalised quantities.
* **Every forward in a batch must share its posterior noise.** ``SeqVaeLagAttnV3.forward``
  samples :math:`z \sim q` unconditionally -- there is no ``self.training`` guard -- and
  :math:`\mu_{\mathrm{full}} = \mu_{\mathrm{base}} + \Delta\mu_{\mathrm{src}}(z)`. Two
  independent forwards of the *same* batch therefore differ by reparameterisation noise alone.
  Measured on the ``v3_prod`` pilot that noise moves the forecast loss by
  :math:`2.9\times10^{-4}`, roughly :math:`10^{2}` times the effect being measured, so an
  unpaired :math:`\Delta L` is pure noise. Every forward for a batch is thus seeded identically
  (**common random numbers**): :math:`\varepsilon` cancels in the difference and
  :math:`\Delta L` isolates the mask. The no-op control -- which returns exactly :math:`0` only
  under this pairing -- is what catches a regression here.
* **This stage is opt-in** (``default_on=False``). It costs
  :math:`1 + 1 + |G| + 2\,|\{D\}|` forwards per batch and must never be folded into ``eval``.

Reading the numbers on a pilot: at 400 steps the source pathway has not switched on (Sprint 3
measured ``shuffle_penalty`` :math:`\approx 4\times10^{-6}`), so every :math:`\Delta L` sits at
zero and the gate fails everywhere. That is a training-progress signal, not a verdict.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from loguru import logger

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.dataset_v2 import (
    build_u_stream,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_v2 import (
    _bootstrap_ci,
    _clean_window_mean,
    _jsonable,
    _masked_forecast_loss,
    _spearman_finite,
    _true_lag_band,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.final_report_v2 import (
    SectionContext,
    SectionSpec,
    _fmt,
    register_section,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.run_pipeline_v2 import (
    StageContext,
    StageSpec,
    _build_runner_and_loader,
    _split_dir,
    register_stage,
)

_STAGE_ORDER = 13
_SECTION_ORDER = 30

#: Inclusive lag-index ranges. The production physiologic partition.
_DEFAULT_BANDS: Dict[str, Tuple[int, int]] = {
    "0-5": (0, 5), "6-15": (6, 15), "16-30": (16, 30),
    "31-60": (31, 60), "61-90": (61, 90),
}
#: Documented cap; ``--max-samples`` overrides. The stage logs what it consumed and skipped.
_DEFAULT_MAX_SAMPLES = 512
#: Strict inequality in *relative* units, per Section 11. Retuned in S8-T04.
_DEFAULT_MARGIN_DELTA_L = 0.0
#: The no-op control must reproduce the clean forward to this absolute tolerance.
_DEFAULT_NOOP_ATOL = 1.0e-6
#: Below this many cells a rank correlation over cells is reported, never gated.
_MIN_CELLS_FOR_RHO = 8
_N_BOOT_RHO = 2000

_INBAND, _OUTBAND = "inband", "outband"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _lag_cfg(config: Dict[str, Any], benchmark: str) -> Dict[str, Any]:
    r"""Read ``benchmarks.<benchmark>.eval.lag_intervention``, falling back to the defaults."""
    bench = (config.get("benchmarks") or {}).get(benchmark) or {}
    cfg = ((bench.get("eval") or {}).get("lag_intervention")) or {}
    raw_bands = cfg.get("bands") or _DEFAULT_BANDS
    bands = {str(k): (int(v[0]), int(v[1])) for k, v in raw_bands.items()}
    return {
        "bands": bands,
        "max_samples": cfg.get("max_samples", _DEFAULT_MAX_SAMPLES),
        "margin_delta_L": float(cfg.get("margin_delta_L", _DEFAULT_MARGIN_DELTA_L)),
        "noop_atol": float(cfg.get("noop_atol", _DEFAULT_NOOP_ATOL)),
    }


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------
def band_keep_mask(n_lags: int, lags: Sequence[int]) -> torch.Tensor:
    r"""A ``(L,)`` keep-mask that **excludes** ``lags`` (the ablation), keeping everything else.

    Args:
        n_lags: The lag-axis extent :math:`L = \texttt{max\_lag} + 1`.
        lags: The lag indices to ablate.

    Returns:
        A boolean ``(L,)`` tensor, ``True`` = keep.
    """
    keep = torch.ones(int(n_lags), dtype=torch.bool)
    idx = [int(l) for l in lags if 0 <= int(l) < int(n_lags)]
    if idx:
        keep[torch.tensor(idx, dtype=torch.long)] = False
    return keep


def _band_lags(n_lags: int, lo: int, hi: int) -> List[int]:
    r"""The inclusive lag range ``[lo, hi]``, clipped to ``[0, L)``."""
    return [l for l in range(int(lo), int(hi) + 1) if 0 <= l < int(n_lags)]


def _true_band_masks(n_lags: int, delay: int, horizon: int) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Keep-masks that ablate :math:`\mathcal{L}^\star` and its complement :math:`\{\ell \ge D\}`.

    :math:`\mathcal{L}^\star = \{\max(0, D-H), \dots, D-1\}` comes from
    :func:`eval_v2._true_lag_band`, the same definition ``kbar_inband`` / ``kbar_outband`` split
    ``te_lag_map`` over -- so :math:`\Delta L` and the attention mass refer to identical lag sets.

    Args:
        n_lags: Lag-axis extent :math:`L`.
        delay: The cell's true lag :math:`D`.
        horizon: The forecast horizon :math:`H`.

    Returns:
        ``(mask_ablating_Lstar, mask_ablating_complement)``.
    """
    star = _true_lag_band(int(delay), int(horizon))
    complement = [l for l in range(int(n_lags)) if l >= int(delay)]
    return band_keep_mask(n_lags, star), band_keep_mask(n_lags, complement)


# ---------------------------------------------------------------------------
# Per-batch scoring
# ---------------------------------------------------------------------------
def _attention_mass(te_lag_map: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    r"""Clean-window mean of ``te_lag_map`` per lag, :math:`(B, T, L) \to (B, L)`.

    Because :math:`\sum_\ell \texttt{te\_lag\_map}[b,t,\ell] = \texttt{kld\_per\_t}[b,t]`, the
    row total is exactly :math:`\bar K_b` and a band's share is the fraction of the KL surrogate
    that the attention attributes to that band.
    """
    w = valid.to(te_lag_map.dtype).unsqueeze(-1)              # (B, T, 1)
    denom = w.sum(dim=1).clamp(min=1.0)                       # (B, 1)
    return (te_lag_map * w).sum(dim=1) / denom                # (B, L)


def _rel(delta: np.ndarray, feat: np.ndarray) -> np.ndarray:
    r"""$\Delta L / \mathcal{L}_{\mathrm{feat}}$, guarding a zero denominator."""
    return delta / np.clip(feat, 1e-12, None)


@contextlib.contextmanager
def _isolated_rng():
    r"""Save and restore the global torch RNG state around a block that reseeds it."""
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _seeded_forward(
    model: Any,
    y_st: torch.Tensor,
    y_ph: torch.Tensor,
    u_stream: torch.Tensor,
    *,
    seed: int,
    lag_band_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    r"""Forward under a fixed RNG seed, so every mask in a batch shares one $\varepsilon$ draw.

    ``forward`` samples :math:`z` on every call. Without this, :math:`\Delta L` would be the
    difference of two *independent* noise realisations and would swamp the mask's effect. Common
    random numbers turn it into a paired comparison, and ``lag_band_mask=None`` stays the
    bit-exact default path.

    Args:
        model: The ``SeqVaeLagAttnV3`` under test.
        y_st: Target scattering features.
        y_ph: Target phase features.
        u_stream: Source stream.
        seed: The per-batch seed shared by every forward of that batch.
        lag_band_mask: The keep-mask, or ``None`` for the clean forward.

    Returns:
        The model's forward dict.
    """
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return model(y_st, y_ph, u_stream, lag_band_mask=lag_band_mask)


def collect_lag_intervention(
    model: Any,
    runner: Any,
    loader: Any,
    *,
    bands: Dict[str, Tuple[int, int]],
    max_samples: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    r"""Mask each lag band, re-forward, and record the forecast degradation per sample.

    Every forward belonging to one batch is seeded identically (see :func:`_seeded_forward`), so
    :math:`\Delta L` is a paired comparison and the reparameterisation noise cancels exactly.

    Args:
        model: The rebuilt ``SeqVaeLagAttnV3`` (must accept ``lag_band_mask``).
        runner: The configured ``TestRunner`` (device, batching, future target).
        loader: Dataloader over the evaluation subset.
        bands: Named inclusive lag ranges to ablate.
        max_samples: Cap on samples consumed; ``None`` consumes all.
        seed: Base seed for the per-batch common random numbers.

    Returns:
        A dict of length-:math:`N` per-sample arrays plus ``n_lags``, ``noop_max_abs_delta``
        and ``n_samples``.

    Raises:
        AttributeError: If ``model`` predates the ``lag_band_mask`` keyword (a v1 alias).
    """
    if not hasattr(model, "_combined_lag_mask"):
        raise AttributeError(
            f"{type(model).__name__} has no lag_band_mask support; the lag_intervention stage "
            "requires SeqVaeLagAttnV3 (S6-T01)."
        )

    warmup, horizon = int(runner.warmup_steps), int(runner.horizon)
    n_lags = int(model.lag_attn.L)
    band_masks = {
        name: band_keep_mask(n_lags, _band_lags(n_lags, lo, hi))
        for name, (lo, hi) in bands.items()
    }
    keep_all = torch.ones(n_lags, dtype=torch.bool)

    cols: Dict[str, List[np.ndarray]] = {}
    noop_max = 0.0
    n_seen = 0

    def _push(key: str, arr: np.ndarray) -> None:
        cols.setdefault(key, []).append(arr)

    with runner.inference_mode(), _isolated_rng():
        for batch_idx, batch in enumerate(runner.iter_batches(loader, max_samples)):
            y_st, y_ph = batch.fhr_st, batch.fhr_ph
            u_stream = build_u_stream(batch)
            delay_t = torch.as_tensor(batch.delay)
            bsz = int(y_st.shape[0])
            # One seed for the whole batch: the clean forward and every masked forward below
            # then share a single epsilon draw, and delta_L isolates the mask.
            batch_seed = int(seed) * 1_000_003 + batch_idx

            out = _seeded_forward(model, y_st, y_ph, u_stream, seed=batch_seed)
            _, valid = _clean_window_mean(
                out["kld_per_t"], delay_t, warmup=warmup, horizon=horizon
            )
            y_plus = runner.build_future_target(batch)
            feat = _masked_forecast_loss(out["mu_full"], y_plus, valid)      # (B,)
            feat_np = feat.detach().cpu().numpy().astype(np.float64)

            def _delta(mask: torch.Tensor, _seed: int = batch_seed) -> np.ndarray:
                r"""$\Delta L$ for one keep-mask, over the shared clean window and target."""
                masked = _seeded_forward(
                    model, y_st, y_ph, u_stream, seed=_seed, lag_band_mask=mask
                )
                loss = _masked_forecast_loss(masked["mu_full"], y_plus, valid)
                return (loss - feat).detach().cpu().numpy().astype(np.float64)

            # No-op control: an all-keep mask reduces to the causal validity mask, so it must
            # reproduce the clean forward. This is the stage's own correctness check.
            noop = _delta(keep_all)
            noop_max = max(noop_max, float(np.abs(noop).max()) if noop.size else 0.0)

            for name, mask in band_masks.items():
                d = _delta(mask)
                _push(f"delta_L_{name}", d)
                _push(f"delta_L_rel_{name}", _rel(d, feat_np))

            # Per-cell true band. The mask API is (L,) / (T, L) only, so bucket by delay.
            d_in = np.full(bsz, np.nan, dtype=np.float64)
            d_out = np.full(bsz, np.nan, dtype=np.float64)
            for d_val in sorted({int(v) for v in delay_t.tolist()}):
                sel = (delay_t == d_val).numpy()
                m_star, m_comp = _true_band_masks(n_lags, d_val, horizon)
                d_in[sel] = _delta(m_star)[sel]
                d_out[sel] = _delta(m_comp)[sel]
            _push(f"delta_L_{_INBAND}", d_in)
            _push(f"delta_L_rel_{_INBAND}", _rel(d_in, feat_np))
            _push(f"delta_L_{_OUTBAND}", d_out)
            _push(f"delta_L_rel_{_OUTBAND}", _rel(d_out, feat_np))

            # Attention mass, from the CLEAN forward, over the same lag sets.
            lag_map = out.get("te_lag_map")
            if lag_map is None:
                lag_map = out["attn_weights"].mean(dim=2)
            mass = _attention_mass(lag_map, valid)                           # (B, L)
            total = mass.sum(dim=-1).clamp(min=1e-12)
            mass_np = mass.detach().cpu().numpy().astype(np.float64)
            total_np = total.detach().cpu().numpy().astype(np.float64)
            for name, (lo, hi) in bands.items():
                idx = _band_lags(n_lags, lo, hi)
                _push(f"mass_{name}", mass_np[:, idx].sum(axis=1) / total_np)

            m_in = np.zeros(bsz, dtype=np.float64)
            for d_val in sorted({int(v) for v in delay_t.tolist()}):
                sel = (delay_t == d_val).numpy()
                star = _true_lag_band(d_val, horizon)
                m_in[sel] = mass_np[np.ix_(sel, star)].sum(axis=1) if star else 0.0
            _push(f"mass_{_INBAND}", m_in / total_np)
            _push(f"mass_{_OUTBAND}", 1.0 - m_in / total_np)

            _push("feat_loss", feat_np)
            _push("delay", delay_t.numpy().astype(np.int64))
            for field, dtype in (("cell_id", np.int64), ("te_true", np.float64),
                                 ("te_scat", np.float64), ("held_out", np.int64)):
                value = getattr(batch, field, None)
                if value is not None:
                    _push(field, torch.as_tensor(value).cpu().numpy().astype(dtype))
            n_seen += bsz

    result: Dict[str, Any] = {k: np.concatenate(v) for k, v in cols.items()}
    result["n_lags"] = n_lags
    result["n_samples"] = n_seen
    result["noop_max_abs_delta"] = noop_max
    return result


# ---------------------------------------------------------------------------
# S6-T03: per-cell aggregation, the true-band gate, and attention-vs-intervention
# ---------------------------------------------------------------------------
def _bootstrap_rho_ci(
    x: np.ndarray, y: np.ndarray, *, n_boot: int = _N_BOOT_RHO, rng=None
) -> Tuple[Optional[float], Optional[float]]:
    r"""Percentile bootstrap CI of Spearman :math:`\rho`, resampling the **cells**.

    ``eval_v2._bootstrap_ci`` bootstraps a *mean*; a correlation needs the pairs resampled
    jointly, so the statistic is recomputed on each resampled cell set.

    Args:
        x: Per-cell values of the first variable.
        y: Per-cell values of the second, paired with ``x``.
        n_boot: Number of bootstrap resamples.
        rng: Seeded generator.

    Returns:
        ``(ci_lo, ci_hi)``, or ``(None, None)`` when :math:`\rho` is undefined.
    """
    gen = rng if rng is not None else np.random.default_rng()
    n = int(x.size)
    if n < 3:
        return None, None
    idx = gen.integers(0, n, size=(int(n_boot), n))
    rhos = [_spearman_finite(x[row], y[row]) for row in idx]
    rhos = np.asarray([r for r in rhos if r is not None], dtype=np.float64)
    if rhos.size < 2:
        return None, None
    lo, hi = np.quantile(rhos, [0.025, 0.975])
    return float(lo), float(hi)


def summarise_lag_intervention(
    per_sample: Dict[str, Any],
    *,
    bands: Dict[str, Tuple[int, int]],
    margin: float = _DEFAULT_MARGIN_DELTA_L,
    seed: int = 0,
) -> Dict[str, Any]:
    r"""Aggregate per cell, evaluate the true-band gate, correlate :math:`\Delta L` vs attention.

    The gate is asserted only on **signal** cells: a :math:`\mathrm{TE}_{\mathrm{inj}} = 0` cell
    has no true lag, so ``inband_gate_pass`` is ``None`` there.

    The rank correlation is taken **across cells at a fixed band**. A :math:`\rho` over the five
    bands of a single cell would be computed from five points and is near-meaningless; it is not
    reported. When fewer than :math:`8` cells are present the correlation is reported and
    explicitly **not** gated.

    Args:
        per_sample: The output of :func:`collect_lag_intervention`.
        bands: The named lag ranges scored.
        margin: ``margin_delta_L``, in relative units.
        seed: Bootstrap seed.

    Returns:
        A JSON-able summary with ``per_cell``, ``overall`` and ``rho_by_band``.
    """
    rng = np.random.default_rng(int(seed))
    cell_id = per_sample.get("cell_id")
    if cell_id is None:
        return {"error": "per-sample cell_id missing; cannot aggregate per cell"}
    te = per_sample.get("te_true", np.full(cell_id.shape, np.nan))
    delay = per_sample.get("delay", np.full(cell_id.shape, -1))

    band_names = list(bands) + [_INBAND, _OUTBAND]
    per_cell: Dict[str, Dict[str, Any]] = {}
    for cid in sorted({int(c) for c in cell_id}):
        sel = cell_id == cid
        entry: Dict[str, Any] = {
            "cell_id": cid,
            "n": int(sel.sum()),
            "te_inj": float(np.nanmean(te[sel])),
            "delay": int(np.median(delay[sel])),
            "feat_loss": float(np.nanmean(per_sample["feat_loss"][sel])),
        }
        for name in band_names:
            for prefix in ("delta_L", "delta_L_rel", "mass"):
                key = f"{prefix}_{name}"
                if key in per_sample:
                    entry[key] = float(np.nanmean(per_sample[key][sel]))

        is_signal = np.isfinite(entry["te_inj"]) and entry["te_inj"] > 0.0
        entry["is_signal"] = bool(is_signal)
        if is_signal:
            lift = entry.get(f"delta_L_rel_{_INBAND}", np.nan) - entry.get(
                f"delta_L_rel_{_OUTBAND}", np.nan
            )
            entry["inband_lift"] = float(lift)
            entry["inband_gate_pass"] = bool(lift > margin) if np.isfinite(lift) else False
        else:
            entry["inband_lift"] = None
            entry["inband_gate_pass"] = None       # no true lag on a null cell
        per_cell[str(cid)] = entry

    signal = [e for e in per_cell.values() if e["is_signal"]]
    n_pass = sum(1 for e in signal if e["inband_gate_pass"])
    overall = {
        "n_cells": len(per_cell),
        "n_signal_cells": len(signal),
        "inband_gate_pass_count": n_pass,
        "inband_gate_pass_frac": (n_pass / len(signal)) if signal else None,
        "inband_gate_pass": bool(signal) and n_pass == len(signal),
        "margin_delta_L": float(margin),
        "mean_inband_lift": float(np.mean([e["inband_lift"] for e in signal])) if signal else None,
    }
    if signal:
        lifts = np.asarray([e["inband_lift"] for e in signal], dtype=np.float64)
        mean, lo, hi = _bootstrap_ci(lifts, rng=rng)
        overall["inband_lift_ci"] = [lo, hi]
        overall["mean_inband_lift"] = mean

    # rho(delta_L, attention mass) across cells, one correlation per band. The per-cell true band
    # (``inband`` / ``outband``) is included alongside the fixed physiologic partition: it is the
    # only split ground truth makes possible, and ``arms_report`` reads ``rho_by_band["inband"]``
    # as its ``rho_deltaL_attn`` column.
    cells = list(per_cell.values())
    gated = len(cells) >= _MIN_CELLS_FOR_RHO
    rho_by_band: Dict[str, Any] = {}
    for name in band_names:
        x = np.asarray([c.get(f"delta_L_rel_{name}", np.nan) for c in cells], dtype=np.float64)
        y = np.asarray([c.get(f"mass_{name}", np.nan) for c in cells], dtype=np.float64)
        rho = _spearman_finite(x, y)
        lo, hi = _bootstrap_rho_ci(x, y, rng=rng) if rho is not None else (None, None)
        rho_by_band[name] = {
            "rho": rho, "ci": [lo, hi], "n_cells": len(cells), "gated": bool(gated),
        }

    return {
        "bands": {k: list(v) for k, v in bands.items()},
        "n_samples": int(per_sample.get("n_samples", len(cell_id))),
        "n_lags": int(per_sample.get("n_lags", 0)),
        "noop_max_abs_delta": float(per_sample.get("noop_max_abs_delta", float("nan"))),
        "per_cell": per_cell,
        "overall": overall,
        "rho_by_band": rho_by_band,
        "rho_reported_not_gated": not gated,
    }


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------
def run_lag_intervention_stage(ctx: StageContext) -> int:
    r"""Run the lag-band intervention for one arm, on every requested split.

    Args:
        ctx: The arm-resolved stage context. ``ctx.max_samples`` overrides the configured cap.

    Returns:
        ``0`` on success. Registered ``fatal=False``, so a raise here is logged and the run
        continues.
    """
    cfg = _lag_cfg(ctx.config, ctx.benchmark)
    max_samples = ctx.max_samples if ctx.max_samples is not None else cfg["max_samples"]
    seed = int(((ctx.config.get("seeds") or {}).get("base_seed", 0)))
    run_dir = ctx.run_dir()

    for split in ctx.splits():
        runner, loader, used_split, ckpt_path = _build_runner_and_loader(
            ctx.config, benchmark=ctx.benchmark, arm=ctx.arm, ckpt=ctx.ckpt, split=split,
        )
        split_dir = _split_dir(run_dir, used_split)
        model = runner.model
        if not hasattr(model, "_combined_lag_mask"):
            logger.warning(
                "lag_intervention[{}/{}]: {} has no lag_band_mask support; skipping.",
                ctx.arm or "-", used_split, type(model).__name__,
            )
            continue

        n_total = len(getattr(loader, "dataset", []) or [])
        logger.info(
            "lag_intervention[{}/{}]: {} bands x {} lags from {} (max_samples={})",
            ctx.arm or "-", used_split, len(cfg["bands"]), int(model.lag_attn.L),
            ckpt_path.name, max_samples,
        )

        per_sample = collect_lag_intervention(
            model, runner, loader, bands=cfg["bands"], max_samples=max_samples, seed=seed
        )
        consumed = int(per_sample["n_samples"])
        skipped = max(n_total - consumed, 0)
        logger.info(
            "lag_intervention[{}/{}]: consumed {} samples, skipped {} of {} "
            "(cap={}); no-op |delta_L| max = {:.3e}",
            ctx.arm or "-", used_split, consumed, skipped, n_total, max_samples,
            per_sample["noop_max_abs_delta"],
        )
        if per_sample["noop_max_abs_delta"] > cfg["noop_atol"]:
            logger.error(
                "lag_intervention[{}/{}]: the no-op control moved the forecast by {:.3e} "
                "(> {:.1e}). An all-keep band mask must reduce to the causal validity mask; "
                "every delta_L below is suspect.",
                ctx.arm or "-", used_split, per_sample["noop_max_abs_delta"], cfg["noop_atol"],
            )

        summary = summarise_lag_intervention(
            per_sample, bands=cfg["bands"], margin=cfg["margin_delta_L"], seed=seed
        )
        summary["arm"] = ctx.arm
        summary["model_class"] = type(model).__name__
        summary["split"] = used_split
        summary["n_samples_skipped"] = skipped
        summary["noop_atol"] = cfg["noop_atol"]

        with (split_dir / "lag_intervention.json").open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(summary), handle, indent=2)
        np.savez_compressed(
            split_dir / "per_sample_lag_intervention.npz",
            **{k: v for k, v in per_sample.items() if isinstance(v, np.ndarray)},
        )
        _render_figure(summary, split_dir / "figures")

        gate = summary["overall"]
        logger.info(
            "lag_intervention[{}/{}]: in-band gate {}/{} signal cells "
            "(mean lift {})",
            ctx.arm or "-", used_split, gate["inband_gate_pass_count"],
            gate["n_signal_cells"], gate["mean_inband_lift"],
        )
    return 0


def _render_figure(summary: Dict[str, Any], figures_dir: Path) -> None:
    r"""Draw the per-cell intervention-vs-attention overlay; never fatal."""
    try:
        from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (
            visualize_v2 as viz,
        )

        figures_dir.mkdir(parents=True, exist_ok=True)
        viz.plot_lag_intervention(summary, figures_dir / "lag_intervention")
    except Exception as exc:  # noqa: BLE001 - a bad figure must not lose the JSON
        logger.warning("lag_intervention: figure failed ({})", exc)


# ---------------------------------------------------------------------------
# Report section
# ---------------------------------------------------------------------------
def _render_lag_intervention_section(ctx: SectionContext) -> List[str]:
    r"""Render the "Interventional lag attribution" section, or ``n/a`` when absent."""
    lines = ["## Interventional lag attribution (G-F)", ""]
    path = Path(ctx.results_dir) / "lag_intervention.json"
    if not path.is_file():
        lines += ["> n/a — `--stage lag_intervention` has not been run for this split.", ""]
        return lines
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "error" in payload:
        lines += [f"> n/a — {payload['error']}", ""]
        return lines

    overall = payload.get("overall") or {}
    noop = payload.get("noop_max_abs_delta")
    lines += [
        "Masks lag bands inside the attention and measures the forecast degradation "
        "$\\Delta L_G = \\mathcal{L}_{\\mathrm{feat}}^{\\text{masked }G} - "
        "\\mathcal{L}_{\\mathrm{feat}}$. The gate is two-sided: masking the true band "
        "$\\mathcal{L}^\\star = \\{\\max(0, D-H), \\dots, D-1\\}$ must hurt, masking "
        "$\\{\\ell \\ge D\\}$ must not.",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| samples | {payload.get('n_samples', 'n/a')} "
        f"(skipped {payload.get('n_samples_skipped', 0)}) |",
        f"| no-op control $\\max|\\Delta L|$ | {_fmt(noop, '.2e')} "
        f"(tol {_fmt(payload.get('noop_atol'), '.0e')}) |",
        f"| `margin_delta_L` (relative) | {_fmt(overall.get('margin_delta_L'))} |",
        f"| mean in-band lift | {_fmt(overall.get('mean_inband_lift'))} |",
    ]
    frac = overall.get("inband_gate_pass_frac")
    mark = "n/a" if frac is None else ("pass" if overall.get("inband_gate_pass") else "**FAIL**")
    lines += [
        f"| signal cells passing the in-band gate | {_fmt(frac, '.0%')} of "
        f"{overall.get('n_signal_cells', '?')} — {mark} |",
        "",
    ]

    rho = payload.get("rho_by_band") or {}
    if rho:
        note = (
            "" if not payload.get("rho_reported_not_gated")
            else " (fewer than 8 cells: reported, not gated)"
        )
        lines += [
            f"Rank correlation between interventional importance and attention mass, across "
            f"cells at a fixed band{note}. This is \"Attention is not Explanation\" against "
            f"ground truth.",
            "",
            "| band | $\\rho(\\Delta L_{\\mathrm{rel}}, \\text{mass})$ | 95% CI |",
            "|---|---|---|",
        ]
        for name, entry in rho.items():
            ci = entry.get("ci") or [None, None]
            lines.append(
                f"| {name} | {_fmt(entry.get('rho'), '.3f')} | "
                f"[{_fmt(ci[0], '.3f')}, {_fmt(ci[1], '.3f')}] |"
            )
        lines.append("")

    per_cell = payload.get("per_cell") or {}
    signal = [c for c in per_cell.values() if c.get("is_signal")]
    if signal:
        lines += [
            "| cell | $\\mathrm{TE}_{\\mathrm{inj}}$ | $D$ | "
            "$\\Delta L_{\\mathrm{rel}}(\\mathcal{L}^\\star)$ | "
            "$\\Delta L_{\\mathrm{rel}}(\\ell \\ge D)$ | lift | gate |",
            "|---|---|---|---|---|---|---|",
        ]
        for cell in sorted(signal, key=lambda c: (c.get("te_inj", 0), c.get("delay", 0))):
            passed = cell.get("inband_gate_pass")
            lines.append(
                f"| {cell.get('cell_id')} | {_fmt(cell.get('te_inj'), '.1f')} | "
                f"{cell.get('delay')} | {_fmt(cell.get('delta_L_rel_inband'))} | "
                f"{_fmt(cell.get('delta_L_rel_outband'))} | {_fmt(cell.get('inband_lift'))} | "
                f"{'pass' if passed else 'FAIL'} |"
            )
        lines.append("")
    return lines


register_stage(
    StageSpec(
        "lag_intervention", _STAGE_ORDER, False, True, run_lag_intervention_stage,
        fatal=False,
        help="interventional lag attribution: mask lag bands, measure delta_L (opt-in)",
    )
)
register_section(
    SectionSpec("Interventional lag attribution", _SECTION_ORDER, _render_lag_intervention_section)
)
