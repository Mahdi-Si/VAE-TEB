r"""Score a checkpoint's forecast against linear baselines on the same anchors and cells.

The question is whether the model forecasts better than a linear filter on the target's own
history. At the 2026-11-04 epoch-524 checkpoint it did not at $\tau = 0$: an own-channel AR(6)
had a scattering MSE of 0.009 against the model's 0.015, and a phase MSE of 0.016 against 0.081.
The model won on scattering only from about $\tau = 2$. A source-conditioned forecast that loses
to a linear filter at the first step is not yet evidence of coupling, so every rung of the
``sweep_next_s*`` ladder is read beside this table.

**This module is an exception to the package's rule that the evaluation defines no numeric
function of its own**, as ``compare_quality_arms.py`` is. The baselines are not readouts of the
model; they are reference forecasters fitted here. The model's own side is read through the shared
seams (:func:`~teb_vae.lag_attn_cfs.eval.probe.load_task`, the task's batch assembly, the model's
cell mask and forecast target), so the cells scored are the cells the objective scores.

**The forecasters.** Each predicts the $(H, C)$ block of kept target coefficients after an anchor
$t$, from values stored at or before $t$:

* ``model_full`` -- the decoder at the posterior mean $z = \mu^q_t$;
* ``model_base`` -- the decoder at the prior mean $z = \mu^p_t$;
* ``persistence`` -- $\hat y_{t+\tau+1} = y_t$;
* ``trend`` -- $\hat y_{t+\tau+1} = y_t + (\tau + 1)(y_t - y_{t-1})$;
* ``zero`` -- the normalised climatology, $\hat y = 0$;
* ``own_ar`` -- per channel $c$, a direct multi-step ridge fit from $y_{t}, \dots, y_{t-P+1}$ of
  channel $c$ alone to its $H$ future values;
* ``ridge_y`` -- one direct ridge fit from the last $P$ values of every kept target channel to every
  $(\tau, c)$ cell;
* ``ridge_yu`` -- ``ridge_y`` plus the last $P$ values of every kept source channel, with a source
  value set to zero where the source validity mask closes it.

``ridge_yu`` minus ``ridge_y`` is the linear analogue of the model's source gain. If it is near zero
while the model's gain is positive, the model reads something from the source that a linear filter
cannot; if both are near zero, the source carries little linear forecast information at all.

**The split.** The linear forecasters are fitted on one half of the evaluation recordings and
scored on the other half, split by the parity of the MD5 hash of the recording GUID, so no
recording contributes to both. The model is scored on the same half. The fit therefore uses
recordings from the evaluation distribution, which the model never saw; that favours the
baselines, so a model that still wins has won against a strong reference.

**The cells.** Only anchors whose whole horizon is valid are used, and only every
``anchor_subsample``-th anchor of a sample is kept, to bound the design matrices. The MSE is
averaged over the cells the model's cell mask scores, in the normalised units the model is trained
in. Squared error, not NLL, because the baselines have no variance model.

The output holds aggregates only (MSE per block and per horizon step, and the counts), no GUID and
no signal, so it can leave the production box.

Run it from the PyCharm Run button after filling in :data:`RUN_ARGS`, or:

.. code-block:: bash

    python -m teb_vae.lag_attn_transformer_cfs.eval.linear_baselines \
        --checkpoint <run>/model_checkpoints/<name>.ckpt --output-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

#: Repository root: ``teb_vae/lag_attn_transformer_cfs/eval/linear_baselines.py`` -> up four.
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Launched with the Run button, this file's own directory goes on sys.path instead of the
# repository root, and every absolute import below fails before __main__ is reached.
if not __package__ and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

from teb_vae.lag_attn.config import load_config  # noqa: E402
from teb_vae.lag_attn_cfs.eval.config_schema import (  # noqa: E402
    force_single_process_loader,
    merge_eval_overrides,
)
from teb_vae.lag_attn_cfs.eval.probe import (  # noqa: E402
    load_task,
    read_checkpoint,
    resolve_device,
    resolved_config_for,
)
from teb_vae.lag_attn_rws.eval.launch import missing_required, resolve_launch_args  # noqa: E402
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask  # noqa: E402
from teb_vae.lag_attn_transformer_cfs.eval.binding import TRF_CFS_BINDING  # noqa: E402

OUTPUT_JSON = "linear_baselines.json"
OUTPUT_MD = "linear_baselines.md"

#: The forecasters, in table order.
FORECASTERS = (
    "model_full", "model_base", "persistence", "trend", "zero", "own_ar", "ridge_y", "ridge_yu",
)

_FIT, _EVAL = "fit", "eval"


def is_fit_recording(guid: Any) -> bool:
    """Return whether a recording belongs to the fit half of the split.

    Args:
        guid: The recording identifier, as ``str`` or as bytes.

    Returns:
        ``True`` when the MD5 hash of the identifier is even.
    """
    text = guid if isinstance(guid, str) else bytes(guid).decode(errors="ignore")
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % 2 == 0


def _lagged(stream: torch.Tensor, anchors: torch.Tensor, lags: int) -> torch.Tensor:
    r"""Gather $x_{t}, x_{t-1}, \dots, x_{t-P+1}$ at every anchor.

    Args:
        stream: ``(B, T, C)``.
        anchors: ``(B, A)`` anchor step indices, each at least ``lags - 1``.
        lags: $P$.

    Returns:
        ``(B, A, P, C)``, lag-major with lag 0 first.
    """
    index = anchors[:, :, None] - torch.arange(lags, device=anchors.device)[None, None, :]
    batch = torch.arange(stream.shape[0], device=stream.device)[:, None, None]
    return stream[batch, index]


@torch.no_grad()
def collect(
    task: Any,
    loader: Any,
    device: torch.device,
    *,
    lags: int,
    anchor_subsample: int,
    max_batches: Optional[int],
) -> Dict[str, Any]:
    """Run the model over the loader and keep, per split half, the lagged inputs and the targets.

    Args:
        task: The rebuilt task; its ``orig_model`` is the network.
        loader: The evaluation dataloader.
        device: Where the forward runs.
        lags: $P$, the history length of the linear forecasters.
        anchor_subsample: Keep every this-many-th anchor of a sample.
        max_batches: Stop after this many batches; ``None`` reads them all.

    Returns:
        ``{"fit": {...}, "eval": {...}, "width_st": int}``. Each half holds ``y_lags``
        ``(N, P, C)``, ``u_lags`` ``(N, P, C_u)`` and ``target`` ``(N, H, C)``; ``eval`` also holds
        ``model_full`` and ``model_base`` ``(N, H, C)``. All in float64 on the CPU. ``width_st``
        is the declared width of the ``fhr_st`` block, the forward's first input, so a kept
        channel whose declared index is below it is a scattering channel.
    """
    model = task.orig_model
    target_keep = model.target_gate.keep_index
    source_gate = getattr(model, "source_gate", None)
    source_keep = None if source_gate is None else source_gate.keep_index
    store: Dict[str, Dict[str, List[torch.Tensor]]] = {
        _FIT: {"y_lags": [], "u_lags": [], "target": []},
        _EVAL: {"y_lags": [], "u_lags": [], "target": [], "model_full": [], "model_base": []},
    }
    width_st = None
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = task.transfer_batch_to_device(batch, device, 0)
        inputs = task._build_forward_inputs(batch)
        width_st = int(inputs[0].shape[-1])
        raw_target, weight = task._build_raw_target(batch)
        weight = task._target_quality_weight(batch, weight)
        out = model(*inputs)
        anchors = out["anchor_index"].to(torch.long)
        anchor_valid = out["anchor_valid"]
        target = model._build_forecast_target(raw_target, anchors)  # (B, A, H, C)
        step_mask, _ = forecast_mask(
            model.scored_weight(weight), model.geometry, coverage_floor=model.coverage_floor,
            anchors=anchors, anchor_valid=anchor_valid,
        )
        keep = (step_mask > 0).all(-1) & anchor_valid & (anchors >= lags - 1)
        subsample = torch.zeros_like(keep)
        subsample[:, ::anchor_subsample] = True
        keep &= subsample

        # Both branches at their latent means, so neither carries sampling noise.
        def _decode(mu: torch.Tensor) -> torch.Tensor:
            z = mu.gather(1, anchors[:, :, None].expand(-1, -1, mu.shape[-1]))
            return model.decoder(z, persistence=out.get("persistence"))[0]

        model_full, model_base = _decode(out["mu_post"]), _decode(out["mu_prior"])

        y = torch.index_select(raw_target, -1, target_keep)
        source = inputs[2]
        source_valid = inputs[5] if len(inputs) > 5 else torch.ones_like(source, dtype=torch.bool)
        if source_keep is not None:
            source = torch.index_select(source, -1, source_keep)
            source_valid = torch.index_select(source_valid, -1, source_keep)
        source = source * source_valid.to(source.dtype)
        y_lags, u_lags = _lagged(y, anchors, lags), _lagged(source, anchors, lags)

        for row, guid in enumerate(batch.guid):
            mask = keep[row]
            if not bool(mask.any()):
                continue
            part = store[_FIT if is_fit_recording(guid) else _EVAL]
            part["y_lags"].append(y_lags[row][mask].double().cpu())
            part["u_lags"].append(u_lags[row][mask].double().cpu())
            part["target"].append(target[row][mask].double().cpu())
            if "model_full" in part:
                part["model_full"].append(model_full[row][mask].double().cpu())
                part["model_base"].append(model_base[row][mask].double().cpu())
        if batch_index % 20 == 0:
            logger.info(f"batch {batch_index}")
    collected: Dict[str, Any] = {"width_st": width_st}
    for name, part in store.items():
        if not part["target"]:
            raise ValueError(f"no anchor fell in the {name!r} half: read more batches")
        collected[name] = {key: torch.cat(values) for key, values in part.items()}
    return collected


def _ridge(features_fit: torch.Tensor, targets_fit: torch.Tensor, features_eval: torch.Tensor,
           ridge: float) -> torch.Tensor:
    r"""Fit $W = (X^\top X + \lambda N I)^{-1} X^\top Y$ with an intercept and predict.

    Args:
        features_fit: ``(N, F)``.
        targets_fit: ``(N, K)``.
        features_eval: ``(M, F)``.
        ridge: $\lambda$, scaled by the fit count $N$ so it does not depend on the sample size.

    Returns:
        ``(M, K)`` predictions.
    """
    ones_fit = torch.ones(features_fit.shape[0], 1, dtype=features_fit.dtype)
    ones_eval = torch.ones(features_eval.shape[0], 1, dtype=features_eval.dtype)
    x_fit = torch.cat([features_fit, ones_fit], 1)
    x_eval = torch.cat([features_eval, ones_eval], 1)
    gram = x_fit.T @ x_fit
    gram = gram + ridge * x_fit.shape[0] * torch.eye(gram.shape[0], dtype=gram.dtype)
    return x_eval @ torch.linalg.solve(gram, x_fit.T @ targets_fit)


def forecasts(data: Dict[str, Any], ridge: float) -> Dict[str, torch.Tensor]:
    """Build every forecaster's prediction on the eval half.

    Args:
        data: The output of :func:`collect`.
        ridge: The ridge strength of the three fitted forecasters.

    Returns:
        ``{forecaster: (N_eval, H, C)}`` in :data:`FORECASTERS` order.
    """
    fit, ev = data[_FIT], data[_EVAL]
    target = ev["target"]
    horizon, channels = target.shape[1], target.shape[2]
    last, previous = ev["y_lags"][:, 0, None, :], ev["y_lags"][:, 1, None, :]
    steps = torch.arange(1, horizon + 1, dtype=target.dtype)[None, :, None]

    own = torch.empty_like(target)
    for channel in range(channels):
        own[:, :, channel] = _ridge(
            fit["y_lags"][:, :, channel], fit["target"][:, :, channel],
            ev["y_lags"][:, :, channel], ridge,
        )
    y_fit, y_eval = fit["y_lags"].flatten(1), ev["y_lags"].flatten(1)
    u_fit, u_eval = fit["u_lags"].flatten(1), ev["u_lags"].flatten(1)
    flat_target = fit["target"].flatten(1)
    return {
        "model_full": ev["model_full"],
        "model_base": ev["model_base"],
        "persistence": last.expand_as(target),
        "trend": last + steps * (last - previous),
        "zero": torch.zeros_like(target),
        "own_ar": own,
        "ridge_y": _ridge(y_fit, flat_target, y_eval, ridge).reshape(target.shape),
        "ridge_yu": _ridge(
            torch.cat([y_fit, u_fit], 1), flat_target, torch.cat([y_eval, u_eval], 1), ridge
        ).reshape(target.shape),
    }


def score(predictions: Dict[str, torch.Tensor], target: torch.Tensor, cell_mask: np.ndarray,
          is_scattering: np.ndarray) -> Dict[str, Dict[str, Any]]:
    r"""Mean squared error over the scored cells, per block and per horizon step.

    Args:
        predictions: ``{forecaster: (N, H, C)}``.
        target: ``(N, H, C)``.
        cell_mask: ``(H, C)`` boolean, the model's scored cells.
        is_scattering: ``(C,)`` boolean, ``True`` for a kept ``fhr_st`` channel.

    Returns:
        Per forecaster: ``mse_all``, ``mse_st``, ``mse_ph`` over the scored cells; ``mse_tau0_st``
        and ``mse_tau0_ph`` at the first step; and ``mse_by_tau_st`` and ``mse_by_tau_ph``, the
        per-step means over scored cells (``None`` at a step with none).
    """
    def _mean(cells: np.ndarray, mask: np.ndarray) -> Optional[float]:
        return float(cells[mask].mean()) if mask.any() else None

    table: Dict[str, Dict[str, Any]] = {}
    for name, prediction in predictions.items():
        cells = ((prediction - target) ** 2).mean(0).numpy()  # (H, C)
        st_mask = cell_mask & is_scattering[None, :]
        ph_mask = cell_mask & ~is_scattering[None, :]
        table[name] = {
            "mse_all": _mean(cells, cell_mask),
            "mse_st": _mean(cells, st_mask),
            "mse_ph": _mean(cells, ph_mask),
            "mse_tau0_st": _mean(cells[0], st_mask[0]),
            "mse_tau0_ph": _mean(cells[0], ph_mask[0]),
            "mse_by_tau_st": [_mean(cells[tau], st_mask[tau]) for tau in range(cells.shape[0])],
            "mse_by_tau_ph": [_mean(cells[tau], ph_mask[tau]) for tau in range(cells.shape[0])],
        }
    return table


def write_markdown(path: Path, report: Dict[str, Any]) -> None:
    """Write the summary table as markdown.

    Args:
        path: Destination file.
        report: The report :func:`main` builds.
    """
    def _cell(value: Optional[float]) -> str:
        return "--" if value is None else f"{value:.4f}"

    columns = ("mse_all", "mse_st", "mse_ph", "mse_tau0_st", "mse_tau0_ph")
    lines = [
        "# Linear baselines",
        "",
        f"Checkpoint `{report['checkpoint_name']}`. {report['n_fit']} fit anchors and "
        f"{report['n_eval']} scored anchors, split by recording. History length P = "
        f"{report['lags']}, ridge {report['ridge']}. MSE over the model's scored cells, in "
        "normalised units.",
        "",
        "| Forecaster | " + " | ".join(f"`{c}`" for c in columns) + " |",
        "|" + "---|" * (len(columns) + 1),
    ]
    for name in FORECASTERS:
        row = report["table"][name]
        lines.append(f"| `{name}` | " + " | ".join(_cell(row[c]) for c in columns) + " |")
    gain = report["source_gain"]
    lines += [
        "",
        f"Linear source gain (`ridge_y` - `ridge_yu`, MSE): {_cell(gain['linear'])}. "
        f"Model source gain (`model_base` - `model_full`, MSE): {_cell(gain['model'])}.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(checkpoint: str, output_dir: str, config: Optional[str] = None,
         overrides: Optional[str] = None, device: Optional[str] = None, lags: Optional[int] = None,
         anchor_subsample: Optional[int] = None, ridge: Optional[float] = None,
         max_batches: Optional[int] = None) -> int:
    """Collect, fit, score and write the report.

    Args:
        checkpoint: Path to the checkpoint.
        output_dir: Where the JSON and markdown reports go.
        config: The run's resolved config. ``None`` finds it beside the checkpoint, as the
            evaluation runner does.
        overrides: The evaluation override delta. ``None`` uses this package's committed delta,
            which points the loader at the holdout split.
        device: Torch device. ``None`` picks ``cuda:0`` when available.
        lags: $P$. ``None`` means 6.
        anchor_subsample: Keep every this-many-th anchor. ``None`` means 10.
        ridge: Ridge strength, per fit sample. ``None`` means ``1e-4``.
        max_batches: Stop after this many batches. ``None`` reads them all.

    Returns:
        The process exit code.
    """
    # Imported here so the parser and RUN_ARGS can be read without the training stack.
    from train.data_module import GraphDataModule

    lags = 6 if lags is None else int(lags)
    anchor_subsample = 10 if anchor_subsample is None else int(anchor_subsample)
    ridge = 1e-4 if ridge is None else float(ridge)
    if lags < 2:
        raise ValueError(f"lags must be at least 2 for the trend forecaster, got {lags}")

    checkpoint_path = Path(checkpoint)
    config_path = resolved_config_for(checkpoint_path) if config is None else Path(config)
    run_config = merge_eval_overrides(
        load_config(str(config_path)),
        TRF_CFS_BINDING.overrides_path if overrides is None else overrides,
    )
    force_single_process_loader(run_config)
    torch_device = resolve_device(device)
    blob = read_checkpoint(checkpoint_path)
    task = load_task(checkpoint_path, torch_device, blob=blob, binding=TRF_CFS_BINDING)
    task.eval()
    model = task.orig_model

    data = collect(
        task, GraphDataModule(run_config).test_dataloader(), torch_device,
        lags=lags, anchor_subsample=anchor_subsample, max_batches=max_batches,
    )
    cell_mask = model.forecast_likelihood_kwargs()["cell_mask"].bool().cpu().numpy()
    is_scattering = model.target_gate.keep_index.cpu().numpy() < data["width_st"]

    table = score(forecasts(data, ridge), data[_EVAL]["target"], cell_mask, is_scattering)
    linear_gain = (
        None if table["ridge_y"]["mse_all"] is None
        else table["ridge_y"]["mse_all"] - table["ridge_yu"]["mse_all"]
    )
    model_gain = (
        None if table["model_base"]["mse_all"] is None
        else table["model_base"]["mse_all"] - table["model_full"]["mse_all"]
    )
    report = {
        "checkpoint_name": checkpoint_path.name,
        "lags": lags,
        "anchor_subsample": anchor_subsample,
        "ridge": ridge,
        "n_fit": int(data[_FIT]["target"].shape[0]),
        "n_eval": int(data[_EVAL]["target"].shape[0]),
        "n_scored_cells": int(cell_mask.sum()),
        "source_gain": {"linear": linear_gain, "model": model_gain},
        "table": table,
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / OUTPUT_JSON).write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(out / OUTPUT_MD, report)
    logger.info(f"wrote {out / OUTPUT_JSON} and {out / OUTPUT_MD}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser; every ``dest`` is also a :data:`RUN_ARGS` key."""
    p = argparse.ArgumentParser(
        prog="python -m teb_vae.lag_attn_transformer_cfs.eval.linear_baselines",
        description="Score a checkpoint against linear forecast baselines.")
    p.add_argument("--checkpoint")
    p.add_argument("--output-dir", dest="output_dir")
    p.add_argument("--config")
    p.add_argument("--overrides")
    p.add_argument("--device")
    p.add_argument("--lags", type=int)
    p.add_argument("--anchor-subsample", dest="anchor_subsample", type=int)
    p.add_argument("--ridge", type=float)
    p.add_argument("--max-batches", dest="max_batches", type=int)
    return p


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    """Merge the command line over :data:`RUN_ARGS` and run."""
    values, _ = resolve_launch_args(build_parser(), RUN_ARGS, argv)
    refusal = missing_required(values, ("checkpoint", "output_dir"))
    if refusal:
        raise SystemExit(refusal)
    # The shard paths inside a resolved config may be repo-root-relative.
    os.chdir(_REPO_ROOT)
    return main(**values)


#: **Edit the values below and press Run.** A flag on the command line overrides its entry.
RUN_ARGS: Dict[str, Any] = {
    # ================================ FILL IN: the paths ================================
    # str -- the checkpoint to score.
    "checkpoint": None,
    # str -- where linear_baselines.json and .md go. Aggregates only: safe to send.
    "output_dir": None,
    # ========================= Everything below has a default =========================
    # str | None -- the run's resolved config.  (None -> found beside the checkpoint)
    "config": None,
    # str | None -- the evaluation override delta.  (None -> this package's committed delta)
    "overrides": None,
    # str | None -- torch device.  (None -> cuda:0 when available, else cpu)
    "device": None,
    # int | None -- history length P of the linear forecasters.  (None -> 6)
    "lags": None,
    # int | None -- keep every this-many-th anchor of a sample.  (None -> 10)
    "anchor_subsample": None,
    # float | None -- ridge strength per fit sample.  (None -> 1e-4)
    "ridge": None,
    # int | None -- stop after this many batches; for a smoke run.  (None -> all)
    "max_batches": None,
}


if __name__ == "__main__":
    sys.exit(_cli())
