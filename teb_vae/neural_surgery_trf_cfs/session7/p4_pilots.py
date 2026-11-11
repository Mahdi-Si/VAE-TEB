r"""Phase 4 pilot fine-tunes: one arm, one seed, one process.

Each call loads the epoch-424 checkpoint, applies the patches of one arm, fine-tunes on the fit
recordings (split code 0) for the fixed budget of :mod:`s7_pilot` and scores the result on the
scored recordings (split code 2 by default) with mean decoding. A process runs one arm and one
seed, because two arms patch module-level functions and must not see each other.

**Arms.** An arm is a ``+``-joined list of patches, so a stack is written as its parts:

* ``control``: no patch. It absorbs the effect of fine-tuning itself.
* ``epoch424``: no patch and no training; the starting point.
* ``tolflags<T>``: tolerant validity flags with $T_{\mathrm{tol}} = T$ seconds (:mod:`s7_tolerance`),
  step-level rule with the running-gap guard (kept for the record; killed in the surgery phase).
* ``hard``: per-anchor key removal. A key whose value rests on a still-running invalid run is
  taken out of the attention for that anchor, and an anchor with no key left closes its gate.
* ``tolhard<T>``: ``hard`` plus tolerant flags without the step-level guard: the flags of every
  step that sees no long run turn valid, also inside a short run.
* ``tolreward<T>``: tolerant contraction reward with the same rule, closed at running gaps.
* ``rewardfree``: the reward ignores UA validity (``event_weight_ua_valid: false``).
* ``null<k>``: ``lambda_source_null`` $= k$.
* ``decfloor<f>``: the decoder gets its own log-variance floor $-f$; the prior and posterior keep
  theirs. A per-channel affine refit of the log-variance head keeps the function close.
* ``studentt<nu>``: a Student-$t$ innovation density with $\nu$ degrees of freedom.
* ``phweight<w>``: phase block weight $w$ against scattering weight 1.
* ``floor<w>``: ``event_weight_floor`` $= w/100$ (reward floor between contractions).
* ``detv``: the reward comes from the session-7 contraction detector (absolute thresholds in
  toco units and its own tolerant validity rule) instead of the shipped detector.

Outputs go to ``out/phase4/pilots/<arm>_s<seed>.parquet`` (per-anchor scores) and ``.json``
(budget, last training metrics, calibration). An existing output is not recomputed.

Edit :data:`RUN_ARGS` and run this file from the IDE, or pass ``--arm`` and ``--seed``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.lag_attn_rws.nets import losses  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import s7_interventions as I  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import s7_pilot as P  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import s7_tolerance as T  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S  # noqa: E402

#: Run-button inputs.
RUN_ARGS: Dict[str, Any] = {
    "arm": "control",
    "seed": 0,
    "device": "cuda:0",
    "fit_codes": [0],
    "score_codes": [2],
    "tag": "",                 # suffix of the output folder, for example "_test" for the final pass
    "steps": None,             # None: the budget of s7_pilot.BUDGET
    "save_checkpoint": False,  # write the fine-tuned checkpoint beside the scores
    "checkpoint": None,        # None: the epoch-424 checkpoint; a path: another start, e.g. the s7 warm start
}
OUT = S.OUT / "phase4"


# --------------------------------------------------------------------------------------------
# Patches
# --------------------------------------------------------------------------------------------
def declared_windows(model: Any) -> torch.Tensor:
    """Causal window per DECLARED UA channel in steps, from the builder's stored warm-up.

    Args:
        model: The network (its declared source width is checked against the shard).

    Returns:
        ``(c_u,)`` long tensor.
    """
    windows = torch.as_tensor(T.declared_windows(), dtype=torch.long)
    if int(windows.numel()) != int(model.c_u):
        raise ValueError(f"the shard declares {int(windows.numel())} UA channels, the model {int(model.c_u)}")
    return windows


def patch_tolerant_flags(task: Any, model: Any, tolerance_s: float) -> None:
    """Make the task build tolerant per-channel validity flags.

    Args:
        task: The task to patch.
        model: The network.
        tolerance_s: The tolerance in seconds.
    """
    windows = declared_windows(model)
    original = task._build_source_valid

    def build(batch: Any, source: torch.Tensor) -> torch.Tensor:
        flags = original(batch, source)
        return T.tolerant_flags(flags, batch["up_valid"].to(torch.bool), windows, tolerance_s)

    task._build_source_valid = build


def patch_key_removal(task: Any, model: Any) -> None:
    r"""Remove, per anchor, every key whose value rests on a still-running invalid run.

    The removal mask $R_{t\ell}$ (:func:`s7_tolerance.removed_keys`) is built from the batch's raw
    UA validity when the task assembles the forward inputs, and the model's attention method is
    replaced by a copy of its soft-mask branch that takes $R$ out of the lag mask. The source
    controls of the objective call the same method, so the source-null penalty sees the same keys.

    Args:
        task: The task to patch.
        model: The network to patch.
    """
    from teb_vae.lag_attn_cfs.nets.causal_inputs import LaggedSourceAttention

    build_inputs = task._build_forward_inputs

    def build(batch: Any):
        inputs = build_inputs(batch)
        model._s7_removed = T.removed_keys(batch["up_valid"].to(torch.bool), int(model.lag_attn.L)).to(inputs[0].device)
        return inputs

    def attend(query: torch.Tensor, h_u: torch.Tensor, key_mask: Optional[torch.Tensor] = None, *_extra: Any, **_more: Any):
        if h_u.shape[0] == 1 and query.shape[0] > 1:
            h_u = h_u.expand(query.shape[0], -1, -1)
        lag_mask = model.build_lag_mask(h_u.shape[1], h_u.device).to(torch.bool)
        removed = getattr(model, "_s7_removed", None)
        kept = lag_mask[None].expand(h_u.shape[0], -1, -1)
        if removed is not None and removed.shape[0] == h_u.shape[0]:
            kept = kept & ~removed
        invalid = (lag_mask[None] & ~key_mask.to(torch.bool)).to(h_u.dtype)
        bias = invalid[:, :, None, :] * model.source_invalid_key_bias[None, None, :, None]
        _, alpha, heads = model.lag_attn(query, h_u, kept, score_bias=bias)
        invalid_mass = (alpha * invalid[:, :, None, :]).sum(dim=-1)
        logits = model.posterior_head.event_gate_logits(heads)
        gate = torch.sigmoid(logits + model.source_invalid_gate(invalid_mass))
        gate = gate * kept.any(dim=-1, keepdim=True).to(gate.dtype)
        return LaggedSourceAttention(alpha, heads, gate, gate, None, None, invalid_mass)

    task._build_forward_inputs = build
    model.attend_lagged_source = attend


def patch_tolerant_flags_unguarded(task: Any, model: Any, tolerance_s: float) -> None:
    """Tolerant flags without the step-level guard; to be used with :func:`patch_key_removal`.

    Args:
        task: The task to patch.
        model: The network.
        tolerance_s: The tolerance in seconds.
    """
    windows = declared_windows(model)
    original = task._build_source_valid

    def build(batch: Any, source: torch.Tensor) -> torch.Tensor:
        flags = original(batch, source)
        return T.tolerant_flags(flags, batch["up_valid"].to(torch.bool), windows, tolerance_s,
                                guard_running=False)

    task._build_source_valid = build


def patch_tolerant_reward(task: Any, model: Any, tolerance_s: float) -> None:
    """Make the contraction reward ignore short invalid runs, and close it at running gaps.

    Args:
        task: The task to patch.
        model: The network (unused).
        tolerance_s: The tolerance in seconds.
    """
    original = task._event_anchor_weight
    task._event_ua_valid = lambda batch: T.tolerant_reward_validity(batch["up_valid"].to(torch.bool), tolerance_s)

    def weight(batch: Any, step_weight: torch.Tensor) -> Optional[torch.Tensor]:
        omega = original(batch, step_weight)
        running = T.running_steps(batch["up_valid"].to(torch.bool), tolerance_s)
        return omega * (~running).to(device=omega.device, dtype=omega.dtype)

    task._event_anchor_weight = weight


def patch_detector_v7(task: Any) -> None:
    r"""Build the contraction reward with the session-7 detector.

    The detector thresholds are in toco units, while the loader hands the task a z-scored trace.
    The patch passes the loader's UA standard deviation $\sigma_u$ as the detector's scale, read
    from the statistics file the evaluation normalises with.

    Args:
        task: The task to patch.
    """
    import functools

    import h5py

    from teb_vae.neural_surgery_trf_cfs.session7.detectors7 import detect_contractions_v7

    with h5py.File(S.STATS, "r") as handle:
        sigma = float(handle["up"]["std"][()])
    detector = functools.partial(detect_contractions_v7, scale=sigma)
    task._event_detector = lambda: detector


def patch_student_t(nu: float) -> None:
    r"""Replace the Gaussian innovation density of the objective by a Student-$t$ density.

    The patch wraps the module-level ``raw_sample_score`` of the losses module, which every
    reconstruction term calls. Residual, AR(1) innovation, weights and cell mask are unchanged;
    only the per-cell term changes (:func:`s7_pilot.innovation_nll`).

    Args:
        nu: Degrees of freedom $\nu$.
    """

    def score(mu, target, *, likelihood, logvar=None, channel_weight=None, horizon_weight=None,
              cell_mask=None, ar_coef=None, step_mask=None, **_ignored):
        residual = target - mu
        if ar_coef is not None:
            residual, logvar = losses.ar_innovation(residual, ar_coef, step_mask, logvar)
        if likelihood == "mse" or logvar is None:
            value = residual ** 2
        else:
            value = P.innovation_nll(residual ** 2 * torch.exp(-logvar), logvar, nu)
        if channel_weight is not None:
            value = value * channel_weight
        if horizon_weight is not None:
            value = value * horizon_weight[:, None]
        if cell_mask is not None:
            value = value * cell_mask
        return value

    losses.raw_sample_score = score


@torch.no_grad()
def patch_decoder_floor(task: Any, model: Any, store: P.SegmentStore, dev: torch.device,
                        new_floor: float, batches: int = 6) -> Dict[str, float]:
    r"""Give the decoder its own log-variance floor and refit the head so the function stays close.

    The bound is $\ell(r) = lo + (hi - lo)\,\sigma(r)$. Lowering $lo$ changes $\ell$ for every raw
    value $r$. For each channel the patch fits $r' \approx a_c r + b_c$ by least squares, where
    $r'$ is the raw value that gives the old log-variance under the new bound, and writes
    $(a_c, b_c)$ into the log-variance head. Cells that sat on the old floor start just above it
    and can then move down.

    Args:
        task: The task (batch assembly).
        model: The network.
        store: Segments to fit the affine map on.
        dev: Device.
        new_floor: The decoder's new lower bound.
        batches: Number of batches of 32 segments used for the fit.

    Returns:
        ``{'max_abs_logvar_change', 'mean_abs_logvar_change'}`` on the fitted cells.
    """
    lo, hi = (float(v) for v in model.decoder.logvar_clamp)
    raw_old: List[torch.Tensor] = []
    raw_new: List[torch.Tensor] = []
    for index, raw in enumerate(store.ordered(32)):
        if index >= batches:
            break
        m = I.matched_forward(task, model, task.transfer_batch_to_device(raw, dev, 0), anchor_stride=5)
        for latent in (m.out["mu_prior"], m.out["mu_post"]):
            _, logvar = model.decoder(I.at_anchors(latent, m.anchors), persistence=m.out.get("persistence"))
            logvar = logvar.reshape(-1, logvar.shape[-1]).double()
            raw_old.append(torch.logit(((logvar - lo) / (hi - lo)).clamp(1e-6, 1 - 1e-6)))
            raw_new.append(torch.logit(((logvar - new_floor) / (hi - new_floor)).clamp(1e-6, 1 - 1e-6)))
    x, y = torch.cat(raw_old), torch.cat(raw_new)
    x_mean, y_mean = x.mean(0), y.mean(0)
    slope = ((x - x_mean) * (y - y_mean)).sum(0) / ((x - x_mean) ** 2).sum(0).clamp_min(1e-9)
    intercept = y_mean - slope * x_mean
    head = model.decoder.logvar_head
    head.weight.mul_(slope.to(head.weight.dtype)[:, None])
    head.bias.copy_(slope.to(head.bias.dtype) * head.bias + intercept.to(head.bias.dtype))
    model.decoder.logvar_clamp = (float(new_floor), hi)
    fitted = new_floor + (hi - new_floor) * torch.sigmoid(slope * x + intercept)
    old = lo + (hi - lo) * torch.sigmoid(x)
    change = (fitted - old).abs()
    return {"max_abs_logvar_change": float(change.max()), "mean_abs_logvar_change": float(change.mean())}


def apply_arm(arm: str, task: Any, model: Any, store: P.SegmentStore, dev: torch.device) -> Dict[str, Any]:
    """Apply every patch named in ``arm``.

    Args:
        arm: ``+``-joined patch names; see the module docstring.
        task: The task.
        model: The network.
        store: The fit segments (for data-driven patches).
        dev: Device.

    Returns:
        ``{'nu': degrees of freedom or None, 'train': bool, 'notes': {...}}``.
    """
    info: Dict[str, Any] = {"nu": None, "train": True, "notes": {}}
    for part in arm.split("+"):
        match = re.fullmatch(r"([a-z_]+?)(\d+(?:\.\d+)?)?", part)
        if match is None:
            raise ValueError(f"unknown arm part {part!r}")
        name, value = match.group(1), match.group(2)
        number = None if value is None else float(value)
        if name == "control":
            continue
        if name == "epoch":            # "epoch424": the untrained starting point
            info["train"] = False
        elif name == "tolflags":
            patch_tolerant_flags(task, model, number)
        elif name == "hard":
            patch_key_removal(task, model)
        elif name == "tolhard":
            patch_tolerant_flags_unguarded(task, model, number)
            patch_key_removal(task, model)
        elif name == "tolreward":
            patch_tolerant_reward(task, model, number)
        elif name == "rewardfree":
            task.hparams["event_weight_ua_valid"] = False
        elif name == "null":
            task.hparams["lambda_source_null"] = number
        elif name == "decfloor":
            info["notes"]["decfloor"] = patch_decoder_floor(task, model, store, dev, -number)
        elif name == "studentt":
            patch_student_t(number)
            info["nu"] = number
        elif name == "phweight":
            keep = model.target_gate.keep_index.tolist()
            weights = model._resolve_channel_weights(keep, weight_st=1.0, weight_ph=number)
            model.target_channel_weight.copy_(weights.to(model.target_channel_weight))
        elif name == "detv":
            patch_detector_v7(task)
        elif name == "floor":
            task.hparams["event_weight_floor"] = number / 100.0
        else:
            raise ValueError(f"unknown arm part {part!r}")
    return info


def main(arm: str, seed: int, device: str, fit_codes: List[int], score_codes: List[int], tag: str,
         steps: Optional[int], save_checkpoint: bool, checkpoint: Optional[str] = None,
         name: Optional[str] = None) -> Dict[str, Any]:
    """Fine-tune one arm with one seed and score it.

    Args:
        arm: The arm name.
        seed: The seed.
        device: Torch device string.
        fit_codes: Split codes of the fine-tune recordings.
        score_codes: Split codes of the scored recordings.
        tag: Suffix of the output folder.
        steps: Optimizer steps; ``None`` uses the fixed budget.
        save_checkpoint: Also write the fine-tuned network as a checkpoint that the evaluation
            pipeline loads (the start checkpoint's keywords and hyperparameters, new weights).
        checkpoint: The start checkpoint; ``None`` is the epoch-424 file. A rung-7 warm start
            carries its own model keywords and task hyperparameters, so it needs no patch.
        name: Output name; ``None`` uses the arm name.

    Returns:
        The run record.
    """
    out_dir = OUT / f"pilots{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{name or arm}_s{seed}"
    if stem.with_suffix(".json").exists():
        print(f"{stem.name}: exists, skipped")
        return json.loads(stem.with_suffix(".json").read_text())
    started = time.time()
    start = S.CHECKPOINT if checkpoint is None else Path(checkpoint)
    dev, cfg, task = S.load_model(checkpoint=start, device=device)
    model = task.orig_model
    fit = P.SegmentStore(cfg, P.guids_with_code(fit_codes))
    scored = P.SegmentStore(cfg, P.guids_with_code(score_codes))
    torch.manual_seed(seed)
    info = apply_arm(arm, task, model, fit, dev)
    budget = dict(P.BUDGET)
    if steps is not None:
        budget["steps"] = int(steps)
    history: List[Dict[str, float]] = []
    if info["train"]:
        history = P.fine_tune(task, fit, dev, seed, budget,
                              on_step=lambda step, row: print(
                                  f"step {step}: loss {row.get('total_loss', float('nan')):.1f} "
                                  f"kl {row.get('source_conditioned_kl_raw', float('nan')):.3f} "
                                  f"gap {row.get('pred_gap', float('nan')):.2f}", flush=True) if step % 25 == 0 else None)
    nu = info["nu"] if info["nu"] is not None else getattr(model, "forecast_student_nu", None)
    table = P.evaluate(task, scored, dev, nu=nu)
    table.to_parquet(stem.with_suffix(".parquet"))
    tail = history[-20:]
    record = {
        "arm": arm, "seed": seed, "budget": budget, "fit_codes": fit_codes, "score_codes": score_codes,
        "n_fit_segments": len(fit), "n_scored_segments": len(scored), "seconds": time.time() - started,
        "notes": info["notes"], "nu": nu, "start_checkpoint": str(start),
        "train_tail": {key: float(np.mean([row[key] for row in tail])) for key in (tail[0] if tail else {})},
        "calibration": P.calibration(table),
        "means": {key: float(table[key].mean()) for key in ("nll_full", "nll_base", "gain", "kl", "kl_null", "g")},
    }
    if save_checkpoint:
        source = torch.load(str(start), map_location="cpu", weights_only=False)
        torch.save({"state_dict": {f"model.{key}": value.detach().cpu() for key, value in model.state_dict().items()},
                    "model_class": source.get("model_class"), "model_kwargs": source.get("model_kwargs"),
                    "hyper_parameters": source.get("hyper_parameters"),
                    "causal_representation": source.get("causal_representation"),
                    "s7_pilot": {"arm": arm, "seed": seed, "budget": budget, "fit_codes": fit_codes}},
                   stem.with_suffix(".ckpt"))
    stem.with_suffix(".json").write_text(json.dumps(record, indent=1))
    print(f"{stem.name}: {record['means']} | {record['calibration']} | {record['seconds']:.0f} s")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default=RUN_ARGS["arm"])
    parser.add_argument("--seed", type=int, default=RUN_ARGS["seed"])
    parser.add_argument("--device", default=RUN_ARGS["device"])
    parser.add_argument("--tag", default=RUN_ARGS["tag"])
    parser.add_argument("--steps", type=int, default=RUN_ARGS["steps"])
    parser.add_argument("--score-codes", default=",".join(str(c) for c in RUN_ARGS["score_codes"]))
    parser.add_argument("--fit-codes", default=",".join(str(c) for c in RUN_ARGS["fit_codes"]))
    parser.add_argument("--save-checkpoint", action="store_true", default=RUN_ARGS["save_checkpoint"])
    parser.add_argument("--checkpoint", default=RUN_ARGS["checkpoint"])
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    main(args.arm, args.seed, args.device, [int(c) for c in args.fit_codes.split(",")],
         [int(c) for c in args.score_codes.split(",")], args.tag, args.steps, args.save_checkpoint,
         args.checkpoint, args.name)
