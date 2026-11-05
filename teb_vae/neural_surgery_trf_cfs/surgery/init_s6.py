r"""Build the warm-start checkpoint for ladder rung 6 from the epoch-524 checkpoint.

Rung 6 (``configs/sweep_next_s6_soft_ua_mask.yaml``) changes five model keywords against the
epoch-524 run: ``persistence_lags`` $1 \to 6$, ``forecast_ar_max`` $1 \to 0.95$,
``source_mask_mode`` ``hard`` $\to$ ``soft``, ``base_decode`` ``mean`` $\to$ ``sample`` and
``target_weight_ph`` $0.1 \to 1.0$. Every other tensor keeps its shape, so the run can start from
the trained weights instead of from random ones. The driver's warm start is a strict load, so the
new tensors must be present in the file. This script writes them, and it sets two of them from
data rather than leaving them at their inert values.

**What is copied.** Every parameter of the epoch-524 network, unchanged. Buffers are taken from the
rung-6 model built here, so that a buffer whose value depends on a changed keyword (for example
the block weights) describes rung 6.

**What is fitted, each on held-out recordings.** Recordings are split by
$\operatorname{crc32}(\mathrm{guid}) \bmod 4$: code $0$ fits, code $2$ selects, codes $1$ and $3$
test.

1. *The five extra AR lags* $w_{\tau,p,c}$, $p = 1, \dots, 5$. At epoch 524 an own-channel
   linear AR(6) beat the model at $\tau = 0$. The weights are a ridge regression of the
   epoch-524 base-branch residual $r = y - \hat\mu^{\mathrm{base}}$ on the channel's own values at
   lags $1, \dots, 5$, one regression per $(\tau, c)$:

   $$w_{\tau,\cdot,c} = (X^\top X + \lambda\,\bar d\,I)^{-1} X^\top r,
     \qquad \bar d = \operatorname{tr}(X^\top X)/5 .$$

   $\lambda$ is chosen on the select split by the closed-form squared error; the test split
   reports the gain.
2. *The AR(1) coefficient* $\phi_c = 0.95\tanh(a_c)$ *and the decoder's per-channel variance
   level*. At epoch 524 $\phi$ saturated at $1.000$ on 60 of 76 channels, where $\tanh$ has no
   gradient, and the decoder's variance was trained against those small innovations. For each
   channel, $\phi_c$ (from a grid capped at $0.92$, so $|\tanh(a_c)| \le 0.97$ and the gradient
   stays alive) and a shift $\Delta_c$ of the raw log-variance (added to the bias of
   ``decoder.logvar_head``) are chosen jointly by the model's own AR(1) innovation negative
   log-density on the select split, after the lag term of step 1.
3. *A verification pass* scores the built rung-6 network and the epoch-524 one on the test
   recordings, each under its own likelihood, and writes both numbers into the report.

**What stays at zero.** The soft UA mask's three tensors, the adapter's quality projection, the
per-head key bias and the gate's mass weight. At zero the soft model treats a bad UA step as a
valid one, and at epoch 524 marking every UA step valid raised the source gain from 5.0 to 6.3
nats per anchor.

**Units.** The local data is normalised with stand-in statistics (see ``common``), so the fitted
weights carry that approximation. The lag weights regress a channel on its own lags, which a
per-channel affine change of units leaves almost unchanged.

Run from an IDE's Run button by editing :data:`RUN_ARGS`, or from the command line::

    python -m teb_vae.neural_surgery_trf_cfs.surgery.init_s6 --limit 40
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from teb_vae.lag_attn_rws.nets.losses import ar_innovation  # noqa: E402
from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.common import (  # noqa: E402
    CHECKPOINT,
    N_ST,
    RUN_DIR,
    batches,
    checkpoint_variant,
    forward_parts,
    setup,
    split_codes,
    write_overrides,
)
from train.graph_models_utils import check_model_class, load_checkpoint_strict  # noqa: E402

#: The model keywords rung 6 sets differently from the epoch-524 run. Checked against the resolved
#: rung-6 config: no other constructor keyword differs.
S6_MODEL_OVERRIDES: Dict[str, Any] = dict(
    persistence_lags=6,
    forecast_ar_max=0.95,
    source_mask_mode="soft",
    base_decode="sample",
    target_weight_ph=1.0,
)

#: Extra lags beyond the anchor's own value.
EXTRA_LAGS = S6_MODEL_OVERRIDES["persistence_lags"] - 1
#: Split codes: fit, select, test.
FIT, SELECT, TEST = (0,), (2,), (1, 3)

#: The IDE entry: edit these and press Run. ``output`` ``None`` writes
#: ``lag-attn-trf-cfs-epoch=524-s6init.ckpt`` next to the source checkpoint. ``limit`` caps the
#: batches read per shard (``None`` reads every batch).
RUN_ARGS: Dict[str, Any] = dict(
    checkpoint=str(CHECKPOINT),
    output=None,
    shards=("healthy_no_bg_no_cs.hdf5", "acidosis_no_cs.hdf5"),
    ridge_grid=(1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0),
    phi_grid=(0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92),
    shift_grid=(-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0),
    limit=None,
    device="cuda:0",
)


def _split_of(codes: torch.Tensor, members: Sequence[int]) -> torch.Tensor:
    """Return a ``(B,)`` float mask of the samples whose split code is in ``members``.

    Args:
        codes: Split code per sample, ``(B,)``.
        members: The codes that belong to the split.

    Returns:
        ``(B,)`` floats in $\\{0, 1\\}$.
    """
    hit = torch.zeros_like(codes, dtype=torch.bool)
    for member in members:
        hit = hit | (codes == member)
    return hit.to(torch.float64)


def _passes(task: Any, shards: Sequence[str], device: str,
            limit: Optional[int]) -> Iterator[Dict[str, torch.Tensor]]:
    r"""Yield, per batch, the epoch-524 tensors the two fits read.

    Args:
        task: The loaded epoch-524 task; its network runs in eval mode.
        shards: Shard file names inside the run directory.
        device: Torch device string.
        limit: Batches per shard, or ``None`` for all.

    Yields:
        ``residual`` $(B, A, H, C)$ of the base branch, ``older`` $(B, A, P-1, C)$ lag values,
        ``cells`` $(B, A, H, C)$ scored cells, ``step_mask`` $(B, A, H)$, ``logvar``
        $(B, A, H, C)$ of the base branch and ``codes`` $(B,)$.
    """
    model = task.orig_model
    cell_mask = getattr(model, "target_cell_mask", None)
    for shard in shards:
        overrides = write_overrides(name=f"init_s6_{Path(shard).stem}.yaml", data=RUN_DIR / shard)
        dev, cfg, _ = setup(overrides=overrides, device=device)
        for batch in batches(task, cfg, dev, limit=limit):
            with torch.no_grad():
                _, outputs, target, step_mask, anchors, anchor_valid = forward_parts(
                    task, model, batch
                )
                target_features, _ = task._build_raw_target(batch)
                # The rung-6 gather of the six most recent stored values, read through the model's
                # own method so the warm-up masking is the one training will apply.
                model.persistence_lags = EXTRA_LAGS + 1
                try:
                    lagged = model._anchor_target_values(target_features, anchors)
                finally:
                    model.persistence_lags = 1
            channels = target.shape[-1]
            older = lagged[..., channels:].unflatten(-1, (EXTRA_LAGS, channels))
            step = step_mask * anchor_valid[..., None].to(step_mask.dtype)
            cells = step[..., None].expand_as(target)
            if cell_mask is not None:
                cells = cells * cell_mask.to(cells.dtype)
            yield dict(
                residual=(target - outputs["mu_base"]).double(),
                older=older.double(),
                cells=cells.double(),
                step_mask=step.double(),
                logvar=outputs["logvar_base"].double(),
                codes=split_codes(batch, target.device),
                logvar_prior=outputs["logvar_prior"].double().mean(),
            )


def _ridge(xtx: torch.Tensor, xtr: torch.Tensor, lam: float) -> torch.Tensor:
    r"""Solve $(X^\top X + \lambda \bar d I) w = X^\top r$ for every $(\tau, c)$ at once.

    Args:
        xtx: $(H, C, P, P)$ Gram matrices.
        xtr: $(H, C, P)$ cross moments.
        lam: Relative ridge $\lambda$.

    Returns:
        $(H, C, P)$ weights.
    """
    lags = xtx.shape[-1]
    scale = xtx.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)
    eye = torch.eye(lags, dtype=xtx.dtype, device=xtx.device)
    system = xtx + lam * scale[..., None, None] * eye
    return torch.linalg.solve(system, xtr[..., None])[..., 0]


def _sq_error(stats: Dict[str, torch.Tensor], w: torch.Tensor) -> torch.Tensor:
    r"""The squared error $\lVert r - X w\rVert^2$ per $(\tau, c)$, from sufficient statistics.

    Args:
        stats: ``rr`` $(H, C)$, ``xtr`` $(H, C, P)$ and ``xtx`` $(H, C, P, P)$ of one split.
        w: $(H, C, P)$ weights.

    Returns:
        $(H, C)$ squared errors.
    """
    quad = torch.einsum("hcp,hcpq,hcq->hc", w, stats["xtx"], w)
    return stats["rr"] - 2.0 * (w * stats["xtr"]).sum(-1) + quad


def fit_lags(task: Any, args: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    r"""Pass 1: fit the extra-lag weights by ridge regression and select $\lambda$.

    Args:
        task: The loaded epoch-524 task.
        args: :data:`RUN_ARGS`.

    Returns:
        ``(weights, report)``: the $(H, P-1, C)$ weights in the decoder's layout, and the numbers.
    """
    splits = {"fit": FIT, "select": SELECT, "test": TEST}
    stats: Dict[str, Dict[str, torch.Tensor]] = {}
    prior_logvar: List[float] = []
    for part in _passes(task, args["shards"], args["device"], args["limit"]):
        prior_logvar.append(float(part["logvar_prior"]))
        for name, members in splits.items():
            weight = _split_of(part["codes"], members)[:, None, None, None] * part["cells"]
            older, residual = part["older"], part["residual"]
            entry = stats.setdefault(name, {})
            for key, value in (
                ("xtx", torch.einsum("bahc,bapc,baqc->hcpq", weight, older, older)),
                ("xtr", torch.einsum("bahc,bapc,bahc->hcp", weight, older, residual)),
                ("rr", (weight * residual * residual).sum(dim=(0, 1))),
                ("n", weight.sum(dim=(0, 1))),
            ):
                entry[key] = entry[key] + value if key in entry else value

    zero = torch.zeros_like(stats["fit"]["xtr"])
    select_err = {
        lam: float(_sq_error(stats["select"], _ridge(stats["fit"]["xtx"], stats["fit"]["xtr"], lam)).sum())
        for lam in args["ridge_grid"]
    }
    best = min(select_err, key=select_err.get)
    weights = _ridge(stats["fit"]["xtx"], stats["fit"]["xtr"], best)

    def block_mse(split: str, w: torch.Tensor, tau: Optional[int] = None) -> Dict[str, float]:
        err, n = _sq_error(stats[split], w), stats[split]["n"]
        if tau is not None:
            err, n = err[tau : tau + 1], n[tau : tau + 1]
        return {
            "st": float(err[:, :N_ST].sum() / n[:, :N_ST].sum().clamp_min(1)),
            "ph": float(err[:, N_ST:].sum() / n[:, N_ST:].sum().clamp_min(1)),
        }

    report = {
        "ridge_select_sq_error": {str(k): v for k, v in select_err.items()},
        "ridge_lambda": best,
        "test_mse_without_lags": block_mse("test", zero),
        "test_mse_with_lags": block_mse("test", weights),
        "test_mse_tau0_without_lags": block_mse("test", zero, 0),
        "test_mse_tau0_with_lags": block_mse("test", weights, 0),
        "test_cells": float(stats["test"]["n"].sum()),
        "mean_logvar_prior_epoch524": sum(prior_logvar) / max(len(prior_logvar), 1),
    }
    # (H, C, P) -> the decoder's (H, P, C).
    return weights.permute(0, 2, 1).contiguous(), report


def _innovation_nll(part: Dict[str, torch.Tensor], residual: torch.Tensor, phi: torch.Tensor,
                    raw: torch.Tensor, shifts: Sequence[float],
                    bounds: Tuple[float, float]) -> torch.Tensor:
    r"""Per-channel AR(1) innovation NLL for each shift of the raw decoder log-variance.

    The innovation $e = r_\tau - \phi_c\,r_{\tau-1}$ (gaps bridged as the objective bridges them)
    does not depend on the variance, so it is computed once. Each shift $\Delta$ then rescores it
    under $\ell = \mathrm{smoothbound}(r^{\mathrm{raw}} + \Delta)$, which is what adding $\Delta$
    to the decoder's log-variance bias does, floor included. The variance carried across a masked
    step is left out of this search; the verification pass scores the exact objective.

    Args:
        part: One batch from :func:`_passes`.
        residual: $(B, A, H, C)$ residual to score.
        phi: $(C,)$ coefficient.
        raw: $(B, A, H, C)$ pre-bound decoder log-variance.
        shifts: The candidate shifts $\Delta$.
        bounds: The decoder's ``logvar_clamp``.

    Returns:
        $(D, B, C)$ sums over anchors and horizon steps, $D$ = ``len(shifts)``.
    """
    innovation, _ = ar_innovation(residual, phi, part["step_mask"], None)
    squared = innovation.pow(2)
    out = []
    for shift in shifts:
        log_v = bounds[0] + (bounds[1] - bounds[0]) * torch.sigmoid(raw + shift)
        nll = 0.5 * (math.log(2.0 * math.pi) + log_v + squared * torch.exp(-log_v))
        out.append((nll * part["cells"]).sum(dim=(1, 2)))
    return torch.stack(out)


def fit_phi(task: Any, args: Dict[str, Any], lag_weight: torch.Tensor,
            phi_now: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    r"""Pass 2: choose $\phi_c$ and a log-variance bias shift $\Delta_c$ per channel, jointly.

    The epoch-524 decoder variance was trained with $\phi \approx 1$, where the innovations are
    small, so any $\phi$ scored against that variance unchanged favours $\phi \to 1$. Each
    candidate $\phi$ is therefore scored at every candidate shift $\Delta$ of the decoder's
    raw log-variance, and the pair with the lowest NLL on the select split is kept. The
    candidates are ``args['phi_grid']`` (capped at $0.92$) and the epoch-524 value carried into
    the bound, $0.95\min(\tanh a_c, 0.97)$; the shifts are ``args['shift_grid']``. The residual is
    the base residual after the fitted lag term.

    Args:
        task: The loaded epoch-524 task.
        args: :data:`RUN_ARGS`.
        lag_weight: $(H, P-1, C)$ fitted lag weights.
        phi_now: $(C,)$ the epoch-524 $\tanh(a_c)$.

    Returns:
        ``(phi, shift, report)``: the chosen $(C,)$ coefficient, the $(C,)$ shift and the numbers.
    """
    lo, hi = (float(v) for v in task.orig_model.decoder.logvar_clamp)
    carried = 0.95 * phi_now.clamp(max=0.97)
    grid = [torch.full_like(phi_now, min(value, 0.92)) for value in args["phi_grid"]] + [carried]
    shifts = tuple(args["shift_grid"])
    zero_shift = shifts.index(0.0)
    channels = phi_now.numel()
    sums = {name: torch.zeros(len(grid), len(shifts), channels, dtype=torch.float64,
                              device=phi_now.device) for name in ("select", "test")}
    before = torch.zeros(channels, dtype=torch.float64, device=phi_now.device)
    w = lag_weight.to(torch.float64)
    for part in _passes(task, args["shards"], args["device"], args["limit"]):
        span = (part["logvar"] - lo) / (hi - lo)
        raw = torch.logit(span.clamp(1e-7, 1.0 - 1e-7))
        moved = part["residual"] - torch.einsum("bapc,hpc->bahc", part["older"], w)
        keeps = {name: _split_of(part["codes"], members)[None, :, None]
                 for name, members in (("select", SELECT), ("test", TEST))}
        for index, phi in enumerate(grid):
            scored = _innovation_nll(part, moved, phi, raw, shifts, (lo, hi))
            for name, keep in keeps.items():
                sums[name][index] += (keep * scored).sum(1)
        before += (keeps["test"][0] * _innovation_nll(
            part, part["residual"], phi_now, raw, (0.0,), (lo, hi))[0]).sum(0)

    flat = sums["select"].reshape(-1, channels)
    choice = flat.argmin(dim=0)
    phi_index, shift_index = choice // len(shifts), choice % len(shifts)
    phi = torch.stack(grid).gather(0, phi_index[None])[0]
    shift = torch.tensor(shifts, dtype=torch.float64, device=phi.device)[shift_index]
    chosen_test = sums["test"].reshape(-1, channels).gather(0, choice[None])[0]
    report = {
        "phi_epoch524_saturated": int((phi_now > 0.999).sum()),
        "phi_chosen": [round(float(v), 3) for v in phi],
        "logvar_shift_chosen": [round(float(v), 2) for v in shift],
        "phi_chosen_carried_count": int((phi_index == len(grid) - 1).sum()),
        "test_nll_epoch524": float(before.sum()),
        "test_nll_lags_carried_phi_no_shift": float(sums["test"][-1, zero_shift].sum()),
        "test_nll_lags_chosen_phi_and_shift": float(chosen_test.sum()),
    }
    return phi, shift, report


@torch.no_grad()
def verify(task: Any, args: Dict[str, Any], kwargs: Dict[str, Any],
           state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    r"""Pass 3: score the built rung-6 network against epoch 524 on the test recordings.

    Both networks run the same batches. The rung-6 network is switched to ``base_decode='mean'``
    for this pass only, so its base branch is deterministic as the epoch-524 one is. Each block is
    scored with its own model's ``forecast_likelihood_kwargs`` (cell mask and $\phi$), over the
    same step mask, as the objective scores it.

    Args:
        task: The loaded epoch-524 task.
        args: :data:`RUN_ARGS`.
        kwargs: Rung-6 model keywords.
        state: Rung-6 state.

    Returns:
        Per-anchor mean NLL and per-block MSE of the base branch, for both networks.
    """
    from teb_vae.lag_attn_rws.nets.losses import masked_raw_block_per_anchor

    old = task.orig_model
    new = SeqVaeLagAttnTrfCfs(**kwargs)
    new.load_state_dict(state, strict=True)
    new = new.to(next(old.parameters()).device).eval()
    new.base_decode = "mean"
    totals = {k: 0.0 for k in ("anchors", "nll_524", "nll_s6", "cells_st", "cells_ph",
                                "se_524_st", "se_524_ph", "se_s6_st", "se_s6_ph")}
    for shard in args["shards"]:
        overrides = write_overrides(name=f"init_s6_{Path(shard).stem}.yaml", data=RUN_DIR / shard)
        dev, cfg, _ = setup(overrides=overrides, device=args["device"])
        for batch in batches(task, cfg, dev, limit=args["limit"]):
            test = _split_of(split_codes(batch, dev), TEST)
            if float(test.sum()) == 0.0:
                continue
            for tag, model in (("524", old), ("s6", new)):
                _, outputs, target, step_mask, _, anchor_valid = forward_parts(task, model, batch)
                block, _ = masked_raw_block_per_anchor(
                    outputs["mu_base"], target, step_mask, likelihood="gaussian_nll",
                    logvar=outputs["logvar_base"], **model.forecast_likelihood_kwargs(),
                )
                live = (step_mask.amax(-1) > 0).double() * anchor_valid.double() * test[:, None]
                totals[f"nll_{tag}"] += float((block.double() * live).sum())
                cells = step_mask[..., None].double() * live[..., None, None]
                cell_mask = getattr(model, "target_cell_mask", None)
                if cell_mask is not None:
                    cells = cells * cell_mask.double()
                error = (outputs["mu_base"].double() - target.double()).pow(2) * cells
                totals[f"se_{tag}_st"] += float(error[..., :N_ST].sum())
                totals[f"se_{tag}_ph"] += float(error[..., N_ST:].sum())
                if tag == "524":
                    totals["anchors"] += float(live.sum())
                    totals["cells_st"] += float(cells[..., :N_ST].sum())
                    totals["cells_ph"] += float(cells[..., N_ST:].sum())
    anchors = max(totals["anchors"], 1.0)
    return {
        "test_anchors": totals["anchors"],
        "test_nll_per_anchor_epoch524": totals["nll_524"] / anchors,
        "test_nll_per_anchor_s6init": totals["nll_s6"] / anchors,
        "test_mse_st_epoch524": totals["se_524_st"] / max(totals["cells_st"], 1.0),
        "test_mse_st_s6init": totals["se_s6_st"] / max(totals["cells_st"], 1.0),
        "test_mse_ph_epoch524": totals["se_524_ph"] / max(totals["cells_ph"], 1.0),
        "test_mse_ph_s6init": totals["se_s6_ph"] / max(totals["cells_ph"], 1.0),
    }


def build_state(blob: Dict[str, Any], lag_weight: torch.Tensor, phi: torch.Tensor,
                shift: torch.Tensor) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], Dict[str, Any]]:
    r"""Build the rung-6 network and fill its state from the epoch-524 one and the fits.

    Args:
        blob: The epoch-524 checkpoint.
        lag_weight: $(H, P-1, C)$ fitted lag weights.
        phi: $(C,)$ chosen AR(1) coefficient.
        shift: $(C,)$ chosen shift of the decoder's raw log-variance, added to the bias of
            ``decoder.logvar_head``.

    Returns:
        ``(model_kwargs, state, report)``.

    Raises:
        RuntimeError: If a tensor other than the expected new ones is missing from the epoch-524
            state, or a shared one changed shape.
    """
    kwargs = dict(blob["model_kwargs"], **S6_MODEL_OVERRIDES)
    model = SeqVaeLagAttnTrfCfs(**kwargs)
    old = {key[len("model."):]: value for key, value in blob["state_dict"].items()
           if key.startswith("model.")}
    # Every name a parameter is reachable under: a module shared between two owners appears in
    # the state dict under both names, and the default listing reports it under one only.
    params = dict(model.named_parameters(remove_duplicate=False))
    expected_new = {
        "decoder.persistence_lag_weight",
        "source_adapter.quality_proj.weight",
        "source_invalid_key_bias",
        "source_invalid_gate.weight",
    }
    state: Dict[str, torch.Tensor] = {}
    buffers_changed: List[str] = []
    for key, value in model.state_dict().items():
        if key not in old:
            if key not in expected_new:
                raise RuntimeError(f"{key} is new in rung 6 but not one of {sorted(expected_new)}")
            state[key] = value.clone()
            continue
        if tuple(old[key].shape) != tuple(value.shape):
            raise RuntimeError(f"{key} changed shape {tuple(old[key].shape)} -> {tuple(value.shape)}")
        if key in params:
            state[key] = old[key].to(value.dtype).clone()
        else:
            if not torch.equal(old[key].to(value.dtype), value):
                buffers_changed.append(key)
            state[key] = value.clone()
    unused = sorted(set(old) - set(state))
    if unused:
        raise RuntimeError(f"epoch-524 tensors with no place in rung 6: {unused}")

    state["decoder.persistence_lag_weight"] = lag_weight.to(torch.float32).cpu()
    state["target_ar_logit"] = torch.atanh(phi / 0.95).to(torch.float32).cpu()
    state["decoder.logvar_head.bias"] = (
        state["decoder.logvar_head.bias"] + shift.to(torch.float32).cpu()
    )
    model.load_state_dict(state, strict=True)
    # Read back through the loaded network: every epoch-524 tensor arrives unchanged except the
    # two this script edits on purpose, and a buffer only where rung 6 defines it differently.
    edited = {"target_ar_logit", "decoder.logvar_head.bias"}
    loaded = model.state_dict()
    drifted = [key for key in old if key not in edited and key not in buffers_changed
               and not torch.equal(loaded[key], old[key].to(loaded[key].dtype))]
    if drifted:
        raise RuntimeError(f"tensors that did not arrive as epoch 524 left them: {drifted[:10]}")
    report = {"buffers_taken_from_rung6": buffers_changed, "new_tensors": sorted(expected_new)}
    return kwargs, state, report


def main(args: Dict[str, Any]) -> Path:
    """Fit, build and verify the rung-6 warm-start checkpoint.

    Args:
        args: :data:`RUN_ARGS` or the command-line equivalent.

    Returns:
        The written checkpoint path.
    """
    source = Path(args["checkpoint"])
    output = Path(args["output"]) if args["output"] else checkpoint_variant("s6init")
    blob = torch.load(str(source), map_location="cpu", weights_only=False)
    _, _, task = setup(checkpoint=source, device=args["device"])

    lag_weight, lag_report = fit_lags(task, args)
    print(json.dumps(lag_report, indent=2))
    phi_now = torch.tanh(task.orig_model.target_ar_logit.detach().double())
    phi, shift, phi_report = fit_phi(task, args, lag_weight, phi_now)
    print(json.dumps({k: v for k, v in phi_report.items()
                      if k not in ("phi_chosen", "logvar_shift_chosen")}, indent=2))

    kwargs, state, build_report = build_state(blob, lag_weight.cpu(), phi.cpu(), shift.cpu())
    verify_report = verify(task, args, kwargs, state)
    print(json.dumps(verify_report, indent=2))
    report = {"source": str(source), **lag_report, **phi_report, **build_report, **verify_report}
    out = {
        "state_dict": {f"model.{k}": v for k, v in state.items()},
        "model_class": blob.get("model_class"),
        "model_kwargs": kwargs,
        "causal_representation": blob.get("causal_representation"),
        "s6_init_report": report,
    }
    torch.save(out, str(output))
    output.with_suffix(".json").write_text(json.dumps(report, indent=2))

    # The driver's own two checks, on a network built from the stamped keywords.
    check_model_class(out, SeqVaeLagAttnTrfCfs.__name__)
    fresh = SeqVaeLagAttnTrfCfs(**kwargs)
    if load_checkpoint_strict(model=fresh, checkpoint=str(output)) is None:
        raise RuntimeError(f"{output} does not load strictly into rung 6")
    print(f"wrote {output}")
    return output


def _parse(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Read the command line into a copy of :data:`RUN_ARGS`.

    Args:
        argv: Arguments, or ``None`` for ``sys.argv``.

    Returns:
        The argument dictionary.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=RUN_ARGS["checkpoint"])
    parser.add_argument("--output", default=RUN_ARGS["output"])
    parser.add_argument("--limit", type=int, default=RUN_ARGS["limit"])
    parser.add_argument("--device", default=RUN_ARGS["device"])
    parsed = parser.parse_args(argv)
    return dict(RUN_ARGS, **vars(parsed))


if __name__ == "__main__":
    main(_parse() if len(sys.argv) > 1 else dict(RUN_ARGS))
