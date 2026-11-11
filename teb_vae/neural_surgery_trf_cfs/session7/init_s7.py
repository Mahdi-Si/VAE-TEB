r"""Build the rung-7 warm-start checkpoint from the epoch-424 checkpoint, and check it by digest.

Rung 7 changes how the model reads its inputs and how it scores its forecast. Only one of those
changes touches a trained tensor:

* ``source_running_gap_keys: remove`` and ``source_valid_tolerance_s`` add no parameter. They
  change which UA keys an anchor may read and which validity flags the UA adapter sees.
* ``forecast_student_nu`` adds no parameter. It changes the density that scores an innovation.
* ``decoder_logvar_clamp`` gives the decoder its own log-variance bound. The bound is
  $\ell(r) = lo + (hi - lo)\,\sigma(r)$ on the raw head output $r$, so a new $lo$ changes the
  log-variance of every cell. The warm start therefore refits the decoder's log-variance head.

**The refit.** For each target channel $c$ the script fits, by least squares over the decoder
outputs on the fit recordings (split code 0, both branches, mean decoding),

$$r' \approx a_c\, r + b_c, \qquad r' = \operatorname{logit}\frac{\ell_{\text{old}}(r) - lo'}{hi - lo'},$$

and writes $W_c \leftarrow a_c W_c$, $\beta_c \leftarrow a_c \beta_c + b_c$ into
``decoder.logvar_head``. The head is linear in the decoder features, so the affine map of its
output is exact in the weights. Every other tensor is the epoch-424 tensor, unchanged.

**The refit is approximate, on purpose.** One affine map per channel cannot keep both the cells
that saturate on the old floor and the free cells of the same channel, so the decoder
log-variance moves by about 0.5 on average at the start (the report holds the value measured on
the select recordings). A fit that minimises the log-variance error directly was tried and was
worse (0.67). This construction is the one the pilot fine-tunes started from, so the pilot
evidence describes this file.

**Two modes.**

* ``fit`` needs the local shards. It fits the head, builds the rung-7 network, checks the result
  on the select recordings (split code 2), writes the checkpoint with its ``.json`` report, and
  stores the two refitted tensors and two SHA-256 digests in ``assets/s7init_fitted.npz``.
* ``build`` needs no data. On a machine that holds only the epoch-424 checkpoint it rebuilds the
  same file from the asset. It refuses an input whose network digest differs from the one the
  fit started from, and refuses to write an output whose digest differs from the fit's output.

Both modes finish with a strict load into the rung-7 network.

Edit :data:`RUN_ARGS` and run this file from the IDE, or::

    python -m teb_vae.neural_surgery_trf_cfs.session7.init_s7 --mode build \
        --checkpoint <epoch-424 file> --output <warm-start file>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs  # noqa: E402
from train.graph_models_utils import check_model_class, load_checkpoint_strict  # noqa: E402

#: The committed fit: the two refitted tensors, the two digests and the fit's settings.
ASSET = _REPO / "teb_vae" / "neural_surgery_trf_cfs" / "assets" / "s7init_fitted.npz"
#: Checkpoint folder of the epoch-424 run on this machine.
RUN_DIR = _REPO / "tmp" / "checkpoint-2026-11-09"
#: The tensors the warm start does not take from epoch 424.
FITTED_KEYS: Tuple[str, ...] = ("decoder.logvar_head.weight", "decoder.logvar_head.bias")

#: Model keywords of rung 7 on top of the epoch-424 ones. Edit here and re-run ``fit`` when the
#: rung changes; ``build`` reads them back from the asset, not from this dictionary.
S7_MODEL_OVERRIDES: Dict[str, Any] = dict(
    source_running_gap_keys="remove",
    decoder_logvar_clamp=(-9.0, 3.0),
    forecast_student_nu=8.0,
)
#: Task hyperparameters of rung 7, stamped into the file so a local evaluation rebuilds the
#: task the rung trains. A training run takes these from its config, not from this file.
S7_TASK_OVERRIDES: Dict[str, Any] = dict(
    source_valid_tolerance_s=16.0,
    ua_bridge_guard_s=4.0,
)

#: Run-button inputs.
RUN_ARGS: Dict[str, Any] = dict(
    mode="fit",                                               # "fit" (needs data) or "build" (no data)
    checkpoint=str(RUN_DIR / "lag-attn-trf-cfs-epoch=424.ckpt"),
    output=str(RUN_DIR / "lag-attn-trf-cfs-epoch=424-s7init.ckpt"),
    device="cuda:0",
    anchor_stride=5,
    fit_codes=[0],
    check_codes=[2],
)


def state_digest(state: Mapping[str, torch.Tensor]) -> str:
    """SHA-256 over every tensor's name, dtype, shape and bytes, in sorted name order.

    Args:
        state: A network state dictionary without a prefix.

    Returns:
        The hex digest.
    """
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def network_state(blob: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    """The network's tensors of a checkpoint, without the task prefix.

    A Lightning checkpoint holds each tensor under ``model.`` and under ``_orig_model.``; a
    warm-start file holds it under ``model.`` only. Both give the same dictionary here.

    Args:
        blob: The checkpoint dictionary.

    Returns:
        ``{name: tensor}``.

    Raises:
        RuntimeError: If the checkpoint holds no ``model.`` tensor.
    """
    state = {key[len("model."):]: value for key, value in blob["state_dict"].items()
             if key.startswith("model.")}
    if not state:
        raise RuntimeError("the checkpoint holds no tensor under the 'model.' prefix")
    return state


def build_state(blob: Mapping[str, Any], head_weight: torch.Tensor, head_bias: torch.Tensor,
                model_overrides: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """Build the rung-7 network and fill it from the epoch-424 state and the refitted head.

    Args:
        blob: The epoch-424 checkpoint.
        head_weight: The refitted ``decoder.logvar_head.weight``.
        head_bias: The refitted ``decoder.logvar_head.bias``.
        model_overrides: The rung-7 model keywords.

    Returns:
        ``(model_kwargs, state)``.

    Raises:
        RuntimeError: If the two networks do not hold the same tensor names and shapes, or if a
            tensor other than the two refitted ones does not arrive unchanged.
    """
    kwargs = dict(blob["model_kwargs"], **model_overrides)
    model = SeqVaeLagAttnTrfCfs(**kwargs)
    old = network_state(blob)
    # Every name a parameter is reachable under: a module shared between two owners appears in
    # the state dict under both names, and the default listing reports it under one only.
    params = dict(model.named_parameters(remove_duplicate=False))
    fresh = model.state_dict()
    if set(fresh) != set(old):
        raise RuntimeError(f"tensor names differ: only in rung 7 {sorted(set(fresh) - set(old))[:5]}, "
                           f"only in epoch 424 {sorted(set(old) - set(fresh))[:5]}")
    state: Dict[str, torch.Tensor] = {}
    for key, value in fresh.items():
        if tuple(old[key].shape) != tuple(value.shape):
            raise RuntimeError(f"{key} changed shape {tuple(old[key].shape)} -> {tuple(value.shape)}")
        # A parameter takes the epoch-424 value; a buffer takes the rung-7 network's own value,
        # because buffers are resolved from the keywords.
        state[key] = (old[key] if key in params else value).to(value.dtype).clone()
    state["decoder.logvar_head.weight"] = head_weight.to(torch.float32).cpu().clone()
    state["decoder.logvar_head.bias"] = head_bias.to(torch.float32).cpu().clone()
    model.load_state_dict(state, strict=True)
    loaded = model.state_dict()
    drifted = [key for key in old if key in params and key not in FITTED_KEYS
               and not torch.equal(loaded[key], old[key].to(loaded[key].dtype))]
    if drifted:
        raise RuntimeError(f"tensors that did not arrive as epoch 424 left them: {drifted[:10]}")
    return kwargs, state


def write_checkpoint(blob: Mapping[str, Any], kwargs: Dict[str, Any], state: Dict[str, torch.Tensor],
                     task_overrides: Mapping[str, Any], report: Dict[str, Any], output: Path) -> None:
    """Write the warm-start file and check that it loads strictly into the rung-7 network.

    Args:
        blob: The epoch-424 checkpoint (class name, representation stamp, hyperparameters).
        kwargs: The rung-7 model keywords.
        state: The rung-7 network state.
        task_overrides: The rung-7 task hyperparameters.
        report: The build report, stored in the file and beside it as ``.json``.
        output: Where to write.

    Raises:
        RuntimeError: If the written file does not load strictly.
    """
    hparams = dict(blob.get("hyper_parameters") or {})
    hparams.update(task_overrides)
    hparams["model_kwargs"] = kwargs
    out = {
        "state_dict": {f"model.{key}": value for key, value in state.items()},
        "model_class": blob.get("model_class"),
        "model_kwargs": kwargs,
        "hyper_parameters": hparams,
        "causal_representation": blob.get("causal_representation"),
        "s7_init_report": report,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(output))
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str))
    check_model_class(out, SeqVaeLagAttnTrfCfs.__name__)
    fresh = SeqVaeLagAttnTrfCfs(**kwargs)
    if load_checkpoint_strict(model=fresh, checkpoint=str(output)) is None:
        raise RuntimeError(f"{output} does not load strictly into rung 7")


@torch.no_grad()
def fit_head(task: Any, cfg: dict, dev: torch.device, new_floor: float, codes: Sequence[int],
             anchor_stride: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    r"""Fit the per-channel affine map of the raw log-variance to the new bound.

    The raw value is read back from the bounded output, $r = \operatorname{logit}\frac{\ell - lo}{hi - lo}$,
    clipped to $\pm\operatorname{logit}(10^{-6})$, which is how the pilot fine-tunes built their
    start. A cell on the old floor therefore enters at the clip value and lands near the old
    floor under the new bound, not on the new floor.

    Args:
        task: The epoch-424 task.
        cfg: The merged evaluation configuration.
        dev: Device.
        new_floor: The decoder's new lower bound $lo'$.
        codes: Split codes of the recordings to fit on.
        anchor_stride: Spacing between decoded anchors.

    Returns:
        ``(slope, intercept, report)`` with ``slope`` and ``intercept`` of shape $(C,)$.
    """
    from teb_vae.neural_surgery_trf_cfs.session7 import s7_interventions as I
    from teb_vae.neural_surgery_trf_cfs.session7 import s7_pilot as P

    model = task.orig_model
    lo, hi = (float(v) for v in model.decoder.logvar_clamp)
    store = P.SegmentStore(cfg, P.guids_with_code(codes))
    width = int(model.decoder.logvar_head.out_features)
    sums = torch.zeros(5, width, dtype=torch.float64, device=dev)          # n, x, y, xx, xy
    for raw in store.ordered(32):
        m = I.matched_forward(task, model, task.transfer_batch_to_device(raw, dev, 0), anchor_stride=anchor_stride)
        keep = m.anchor_valid
        for latent in (m.out["mu_prior"], m.out["mu_post"]):
            _, logvar = model.decoder(I.at_anchors(latent, m.anchors), persistence=m.out.get("persistence"))
            logvar = logvar[keep].reshape(-1, width).double()
            x = torch.logit(((logvar - lo) / (hi - lo)).clamp(1e-6, 1 - 1e-6))
            y = torch.logit(((logvar - new_floor) / (hi - new_floor)).clamp(1e-6, 1 - 1e-6))
            sums += torch.stack([torch.ones_like(x).sum(0), x.sum(0), y.sum(0), (x * x).sum(0), (x * y).sum(0)])
    n, sx, sy, sxx, sxy = sums
    slope = (sxy - sx * sy / n) / (sxx - sx * sx / n).clamp_min(1e-9)
    intercept = (sy - slope * sx) / n
    report = {"fit_codes": list(codes), "n_fit_segments": len(store), "cells_per_channel": float(n[0]),
              "old_bound": [lo, hi], "new_floor": float(new_floor),
              "slope_min_max": [float(slope.min()), float(slope.max())],
              "intercept_min_max": [float(intercept.min()), float(intercept.max())]}
    return slope.cpu(), intercept.cpu(), report


@torch.no_grad()
def check_heldout(task: Any, new_model: Any, cfg: dict, dev: torch.device, codes: Sequence[int],
                  anchor_stride: int) -> Dict[str, Any]:
    """Compare the decoder log-variance of the epoch-424 and the rung-7 network on held-out data.

    Args:
        task: The epoch-424 task.
        new_model: The rung-7 network with the refitted head, on ``dev``.
        cfg: The merged evaluation configuration.
        dev: Device.
        codes: Split codes of the recordings to check on.
        anchor_stride: Spacing between decoded anchors.

    Returns:
        Mean and maximum absolute change of the log-variance, the share of cells that moved by
        more than 0.25, and the share of cells on each network's own floor.
    """
    from teb_vae.neural_surgery_trf_cfs.session7 import s7_interventions as I
    from teb_vae.neural_surgery_trf_cfs.session7 import s7_pilot as P

    model = task.orig_model
    store = P.SegmentStore(cfg, P.guids_with_code(codes))
    old_floor = float(model.decoder.logvar_clamp[0])
    new_floor = float(new_model.decoder.logvar_clamp[0])
    total = moved = 0.0
    abs_sum = abs_max = 0.0
    on_old = on_new = 0.0
    for raw in store.ordered(32):
        m = I.matched_forward(task, model, task.transfer_batch_to_device(raw, dev, 0), anchor_stride=anchor_stride)
        z = I.at_anchors(m.out["mu_post"], m.anchors)
        _, before = model.decoder(z, persistence=m.out.get("persistence"))
        _, after = new_model.decoder(z, persistence=m.out.get("persistence"))
        before, after = before[m.anchor_valid], after[m.anchor_valid]
        change = (after - before).abs()
        total += float(change.numel())
        moved += float((change > 0.25).sum())
        abs_sum += float(change.sum())
        abs_max = max(abs_max, float(change.max()))
        on_old += float((before <= old_floor + 1e-3).sum())
        on_new += float((after <= new_floor + 1e-3).sum())
    return {"check_codes": list(codes), "n_check_segments": len(store), "mean_abs_logvar_change": abs_sum / total,
            "max_abs_logvar_change": abs_max, "share_moved_over_0p25": moved / total,
            "share_on_old_floor_before": on_old / total, "share_on_new_floor_after": on_new / total}


def fit(args: Dict[str, Any]) -> Path:
    """Fit the head on local data, build, check and write the warm start and the asset.

    Args:
        args: :data:`RUN_ARGS` or the command-line equivalent.

    Returns:
        The written checkpoint path.
    """
    from teb_vae.lag_attn_cfs.ua_tolerance import declared_source_windows
    from teb_vae.neural_surgery_trf_cfs.session7 import s7_pilot as P
    from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S

    source, output = Path(args["checkpoint"]), Path(args["output"])
    blob = torch.load(str(source), map_location="cpu", weights_only=False)
    dev, _, task = S.load_model(checkpoint=source, device=args["device"])
    cfg = P.load_eval_config()
    model = task.orig_model
    new_floor = float(S7_MODEL_OVERRIDES["decoder_logvar_clamp"][0])
    slope, intercept, fit_report = fit_head(task, cfg, dev, new_floor, args["fit_codes"], args["anchor_stride"])
    head = model.decoder.logvar_head
    weight = (head.weight.detach().double().cpu() * slope[:, None]).float()
    bias = (head.bias.detach().double().cpu() * slope + intercept).float()
    kwargs, state = build_state(blob, weight, bias, S7_MODEL_OVERRIDES)

    new_model = SeqVaeLagAttnTrfCfs(**kwargs)
    new_model.load_state_dict(state, strict=True)
    new_model.to(dev).eval()
    check = check_heldout(task, new_model, cfg, dev, args["check_codes"], args["anchor_stride"])

    windows = declared_source_windows([str(path) for path in S.SHARDS.values()],
                                      use_up_st=bool(blob["model_kwargs"].get("use_up_st", True)))
    task_overrides = dict(S7_TASK_OVERRIDES, source_valid_windows=[int(w) for w in windows])
    old_state = network_state(blob)
    report = {"source": str(source), "source_digest": state_digest(old_state), "output_digest": state_digest(state),
              "model_overrides": {k: (list(v) if isinstance(v, tuple) else v) for k, v in S7_MODEL_OVERRIDES.items()},
              "task_overrides": task_overrides, "fitted_tensors": list(FITTED_KEYS), **fit_report, **check}
    write_checkpoint(blob, kwargs, state, task_overrides, report, output)
    np.savez(ASSET, head_weight=weight.numpy(), head_bias=bias.numpy(),
             source_digest=np.array(report["source_digest"]), output_digest=np.array(report["output_digest"]),
             model_overrides=np.array(json.dumps(report["model_overrides"])),
             task_overrides=np.array(json.dumps(task_overrides)))
    print(json.dumps(report, indent=2, default=str))
    print(f"wrote {output} and {ASSET}")
    return output


def build(args: Dict[str, Any]) -> Path:
    """Rebuild the warm start from the epoch-424 checkpoint and the asset, with no data.

    Args:
        args: :data:`RUN_ARGS` or the command-line equivalent.

    Returns:
        The written checkpoint path.

    Raises:
        RuntimeError: If the input's network digest is not the one the fit started from, or the
            rebuilt network's digest is not the one the fit wrote.
    """
    source, output = Path(args["checkpoint"]), Path(args["output"])
    blob = torch.load(str(source), map_location="cpu", weights_only=False)
    with np.load(ASSET, allow_pickle=False) as stored:
        weight = torch.from_numpy(stored["head_weight"])
        bias = torch.from_numpy(stored["head_bias"])
        source_digest, output_digest = str(stored["source_digest"]), str(stored["output_digest"])
        model_overrides = json.loads(str(stored["model_overrides"]))
        task_overrides = json.loads(str(stored["task_overrides"]))
    if "decoder_logvar_clamp" in model_overrides:
        model_overrides["decoder_logvar_clamp"] = tuple(model_overrides["decoder_logvar_clamp"])
    if state_digest(network_state(blob)) != source_digest:
        raise RuntimeError(f"{source} is not the epoch-424 checkpoint the fit started from (its network "
                           f"digest differs from the one in {ASSET.name}); refusing to build")
    kwargs, state = build_state(blob, weight, bias, model_overrides)
    if state_digest(state) != output_digest:
        raise RuntimeError("the rebuilt network's digest differs from the fit's; refusing to write")
    report = {"source": str(source), "source_digest": source_digest, "output_digest": output_digest,
              "model_overrides": {k: (list(v) if isinstance(v, tuple) else v) for k, v in model_overrides.items()},
              "task_overrides": task_overrides, "fitted_tensors": list(FITTED_KEYS), "built_from_asset": ASSET.name}
    write_checkpoint(blob, kwargs, state, task_overrides, report, output)
    written = torch.load(str(output), map_location="cpu", weights_only=False)
    if state_digest(network_state(written)) != output_digest:
        raise RuntimeError(f"{output} read back with a different digest")
    print(f"wrote {output}; digest verified against {ASSET.name}; strict load into rung 7 passed")
    return output


def main(args: Dict[str, Any]) -> Path:
    """Run the selected mode.

    Args:
        args: :data:`RUN_ARGS` or the command-line equivalent.

    Returns:
        The written checkpoint path.
    """
    if args["mode"] == "fit":
        return fit(args)
    if args["mode"] == "build":
        return build(args)
    raise ValueError(f"mode must be 'fit' or 'build', got {args['mode']!r}")


def _parse(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Read the command line into a copy of :data:`RUN_ARGS`.

    Args:
        argv: Arguments, or ``None`` for ``sys.argv``.

    Returns:
        The run arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("fit", "build"), default=RUN_ARGS["mode"])
    parser.add_argument("--checkpoint", default=RUN_ARGS["checkpoint"])
    parser.add_argument("--output", default=RUN_ARGS["output"])
    parser.add_argument("--device", default=RUN_ARGS["device"])
    parsed = parser.parse_args(argv)
    return dict(RUN_ARGS, mode=parsed.mode, checkpoint=parsed.checkpoint, output=parsed.output, device=parsed.device)


if __name__ == "__main__":
    main(_parse())
