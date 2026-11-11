r"""Phase 5: check the production port of the four session-7 mechanisms against the prototypes.

The four mechanisms live in production code behind keys that are inert by default:

* M1 ``source_running_gap_keys`` (model): per-anchor removal of the keys whose value can rest on
  bridged UA after the anchor;
* M2 ``source_valid_tolerance_s`` (task): tolerant per-channel validity flags;
* M3 ``event_weight_ua_tolerance_s`` (task): tolerant contraction reward;
* M4 ``decoder_logvar_clamp`` (model): the decoder's own log-variance bound.

The script runs on the CPU, on 8 rows of the first evaluation batch: the first 8 for the bitwise
reference, and the 8 with the most invalid UA samples for the other checks, so long runs, removed
keys and tolerant flags are all exercised (check b is also repeated on the first 8). It checks:

a. **Inert.** With every new key at its default, the forward outputs and the validation loss are
   bitwise the reference outputs that the unchanged code wrote (``--mode reference``, run before
   the port).
b. **Equivalence.** With M1 to M3 on, the production forward gives the prototype's ``mu_post``,
   ``logvar_post``, ``event_gate`` and ``attn_weights`` (prototype arm ``tolhard16`` of
   :mod:`p4_pilots`), and the production contraction weight equals the prototype reward without
   its closing at running steps. The tolerance tensors (flags, removal mask, windows) are compared
   too. The bound is a maximum absolute difference of $10^{-5}$.
c. **M4.** A model with ``decoder_logvar_clamp = (-9, 3)`` loads the checkpoint strictly, its
   decoder log-variances lie in $(-9, 3)$, its prior and posterior log-variances equal the default
   model's, and ``decoder_logvar_clamp = None`` gives the default decoder output bitwise.
d. **Training smoke.** With every key on, one training step computes the loss and back-propagates;
   one validation step runs the permutation and source-null controls under the removal.

Results go to ``out/phase5/check_production.json``. Edit :data:`RUN_ARGS` and run this file from
the IDE, or pass ``--mode``.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S  # noqa: E402

#: Run-button inputs. ``mode`` is ``'reference'`` (write the reference outputs of the unchanged
#: code) or ``'check'`` (run checks a to d against them).
RUN_ARGS: Dict[str, Any] = {"mode": "check", "rows": 8, "seed": 0, "tolerance_s": 16.0}
OUT = S.OUT / "phase5"
REFERENCE = OUT / "reference_outputs.pt"
COMPARED = ("mu_post", "logvar_post", "event_gate", "attn_weights")
BOUND = 1e-5


# --------------------------------------------------------------------------------------------
# Batches and forwards
# --------------------------------------------------------------------------------------------
def select(batch: Any, index: Sequence[int]) -> Any:
    """Keep the samples ``index`` of every per-sample field, in that order, in a shallow copy.

    Args:
        batch: A batch dictionary.
        index: Row indices to keep.

    Returns:
        A new batch of the same type.
    """
    size = int(batch["up_valid"].shape[0])
    picked = type(batch)((key, value) for key, value in batch.items())
    rows = torch.as_tensor(list(index), dtype=torch.long)
    for key in list(picked.keys()):
        value = picked[key]
        if torch.is_tensor(value) and value.dim() > 0 and int(value.shape[0]) == size:
            picked[key] = value[rows.to(value.device)]
        elif isinstance(value, (list, tuple)) and len(value) == size:
            picked[key] = type(value)(value[i] for i in rows.tolist())
    return picked


def first_batch(task: Any, cfg: dict, dev: Any) -> Any:
    """The first evaluation batch, whole.

    Args:
        task: The loaded task.
        cfg: The merged configuration.
        dev: Device.

    Returns:
        The batch.
    """
    return next(iter(S.iter_batches(task, cfg, dev, limit=1)))


def most_invalid_rows(batch: Any, rows: int) -> list:
    """The ``rows`` samples with the most invalid raw UA samples, so long runs are exercised.

    Args:
        batch: The batch.
        rows: How many rows.

    Returns:
        Row indices, most invalid first.
    """
    invalid = (~batch["up_valid"].to(torch.bool)).sum(dim=1)
    return torch.argsort(invalid, descending=True, stable=True)[:rows].tolist()


def present_epoch(task: Any, epoch: int = 424) -> None:
    """Make the task read ``epoch`` as the current training epoch (KL warm-up, anchor phase).

    Args:
        task: The task.
        epoch: The epoch to present.
    """
    task._trainer = types.SimpleNamespace(current_epoch=int(epoch))


def run_outputs(task: Any, batch: Any, seed: int) -> Dict[str, Any]:
    """Forward outputs and the validation loss on one batch, with fixed seeds.

    Args:
        task: The task.
        batch: The batch.
        seed: The seed set before the forward and before the loss.

    Returns:
        ``{'outputs', 'loss', 'metrics'}`` on the CPU.
    """
    model = task.orig_model
    present_epoch(task)
    with torch.no_grad():
        torch.manual_seed(seed)
        inputs = task._build_forward_inputs(batch)
        out = model(*inputs)
        torch.manual_seed(seed)
        loss, metrics = task.compute_loss_and_metrics(batch, 0, "val")
    return {"outputs": {k: v.detach().cpu() for k, v in out.items() if torch.is_tensor(v)},
            "loss": loss.detach().cpu(),
            "metrics": {k: v.detach().cpu() for k, v in metrics.items() if torch.is_tensor(v)}}


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    """Maximum absolute difference of two tensors of one shape.

    Args:
        left: First tensor.
        right: Second tensor.

    Returns:
        The maximum absolute difference, ``inf`` on a shape mismatch.
    """
    if tuple(left.shape) != tuple(right.shape):
        return float("inf")
    if left.dtype == torch.bool:
        return float((left != right.to(torch.bool)).sum())
    return float((left.double() - right.double()).abs().max()) if left.numel() else 0.0


def build_task(model_extra: Optional[Dict[str, Any]] = None,
               hparam_extra: Optional[Dict[str, Any]] = None) -> Any:
    """Build a task from the epoch-424 checkpoint with extra model kwargs and hyperparameters.

    The checkpoint's ``model_kwargs`` and ``hyper_parameters`` are edited in memory and the task
    is rebuilt through the evaluation loader, which loads the state dict strictly. This is the
    path an s7 checkpoint takes.

    Args:
        model_extra: Model keywords to add.
        hparam_extra: Task hyperparameters to add.

    Returns:
        The task on the CPU, in evaluation mode.
    """
    from teb_vae.lag_attn_cfs.eval.probe import load_task, read_checkpoint
    from teb_vae.lag_attn_transformer_cfs.eval.binding import TRF_CFS_BINDING

    blob = read_checkpoint(str(S.CHECKPOINT))
    blob["model_kwargs"] = {**blob["model_kwargs"], **(model_extra or {})}
    blob["hyper_parameters"] = {**blob["hyper_parameters"], **(hparam_extra or {})}
    task = load_task(str(S.CHECKPOINT), torch.device("cpu"), blob=blob, binding=TRF_CFS_BINDING)
    task.orig_model.eval()
    return task


# --------------------------------------------------------------------------------------------
# The four checks
# --------------------------------------------------------------------------------------------
def check_inert(task: Any, batch: Any, seed: int) -> Dict[str, Any]:
    """Check a: the defaults reproduce the reference outputs bitwise.

    Args:
        task: The default task.
        batch: The batch.
        seed: The seed.

    Returns:
        The record.
    """
    ref = torch.load(REFERENCE)
    rec = run_outputs(task, batch, seed)
    out_keys = sorted(set(ref["outputs"]) | set(rec["outputs"]))
    mismatched = [k for k in out_keys if k not in ref["outputs"] or k not in rec["outputs"]
                  or not torch.equal(ref["outputs"][k], rec["outputs"][k])]
    metric_keys = sorted(set(ref["metrics"]) & set(rec["metrics"]))
    metric_mismatch = [k for k in metric_keys if not torch.equal(ref["metrics"][k], rec["metrics"][k])]
    return {
        "output_keys_equal": sorted(ref["outputs"]) == sorted(rec["outputs"]),
        "outputs_mismatched": mismatched,
        "loss_bitwise_equal": bool(torch.equal(ref["loss"], rec["loss"])),
        "loss": float(rec["loss"]),
        "metrics_compared": len(metric_keys),
        "metrics_mismatched": metric_mismatch,
        "metric_keys_only_now": sorted(set(rec["metrics"]) - set(ref["metrics"])),
        "metric_keys_only_before": sorted(set(ref["metrics"]) - set(rec["metrics"])),
        "state_dict_keys": len(task.orig_model.state_dict()),
        "pass": (not mismatched and not metric_mismatch and bool(torch.equal(ref["loss"], rec["loss"]))
                 and sorted(ref["outputs"]) == sorted(rec["outputs"])),
    }


def check_equivalence(cfg: dict, dev: Any, batch: Any, seed: int, tolerance_s: float,
                      windows: Any) -> Dict[str, Any]:
    """Check b: production M1 to M3 against the prototype patches.

    Args:
        cfg: The merged configuration.
        dev: Device.
        batch: The batch.
        seed: The seed.
        tolerance_s: The tolerance of both tolerant rules.
        windows: The untrimmed window vector the production task is given.

    Returns:
        The record, with the maximum absolute differences.
    """
    from teb_vae.lag_attn_rws.event_weight import contraction_anchor_weight
    from teb_vae.neural_surgery_trf_cfs.session7 import p4_pilots
    from teb_vae.neural_surgery_trf_cfs.session7 import s7_tolerance as T

    prod = build_task({"source_running_gap_keys": "remove"},
                      {"source_valid_tolerance_s": tolerance_s, "event_weight_ua_tolerance_s": tolerance_s,
                       "source_valid_windows": list(windows)})
    _, _, proto = S.load_model(device="cpu")
    p4_pilots.apply_arm(f"tolhard{int(tolerance_s)}", proto, proto.orig_model, None, dev)
    proto_windows = p4_pilots.declared_windows(proto.orig_model)

    record: Dict[str, Any] = {"windows_equal": [int(w) for w in proto_windows.tolist()] == [int(w) for w in windows]}
    with torch.no_grad():
        torch.manual_seed(seed)
        prod_inputs = prod._build_forward_inputs(batch)
        prod_out = prod.orig_model(*prod_inputs)
        torch.manual_seed(seed)
        proto_inputs = proto._build_forward_inputs(batch)
        proto_out = proto.orig_model(*proto_inputs)
    record["n_inputs"] = len(prod_inputs)
    record["flags_mismatch"] = max_abs(prod_inputs[5], proto_inputs[5])
    dataset_flags = torch.cat([batch["up_st_step_valid"], batch["up_ph_step_valid"]], dim=-1).to(torch.bool)
    record["flags_turned_valid"] = int((prod_inputs[5] & ~dataset_flags).sum())
    record["flags_invalid_before"] = int((~dataset_flags).sum())
    record["removed_mismatch"] = max_abs(prod_inputs[6], proto.orig_model._s7_removed)
    record["removed_fraction"] = float(prod_inputs[6].float().mean())
    record["forward_max_abs"] = {key: max_abs(prod_out[key], proto_out[key]) for key in COMPARED}

    # The reward: production weight against the prototype reward without its closing at running steps.
    _, weight = prod._build_raw_target(batch)
    prod_weight = prod._event_anchor_weight(batch, weight)
    model = prod.orig_model
    reach = prod.hparams.get("event_weight_reach_steps")
    proto_weight = contraction_anchor_weight(
        batch["up"], weight, decimation=int(model.geometry.decimation),
        reach_steps=int(model.max_lag) if reach is None else int(reach),
        floor=float(prod.hparams.get("event_weight_floor")), keys=None, cache={},
        detector=prod._event_detector(),
        raw_valid=T.tolerant_reward_validity(batch["up_valid"].to(torch.bool), tolerance_s))
    record["reward_max_abs"] = max_abs(prod_weight, proto_weight)
    # The tolerance must have moved the reward somewhere for the comparison to mean something.
    strict = contraction_anchor_weight(
        batch["up"], weight, decimation=int(model.geometry.decimation),
        reach_steps=int(model.max_lag) if reach is None else int(reach),
        floor=float(prod.hparams.get("event_weight_floor")), keys=None, cache={},
        detector=prod._event_detector(), raw_valid=batch["up_valid"].to(torch.bool))
    record["reward_steps_changed_by_tolerance"] = int((prod_weight != strict).sum())
    worst = max([record["flags_mismatch"], record["removed_mismatch"], record["reward_max_abs"],
                 *record["forward_max_abs"].values()])
    record["pass"] = bool(worst <= BOUND and record["windows_equal"] and record["n_inputs"] == 7)
    return record


def check_decoder_clamp(batch: Any, seed: int) -> Dict[str, Any]:
    """Check c: the decoder's own log-variance bound.

    Args:
        batch: The batch.
        seed: The seed.

    Returns:
        The record.
    """
    default = build_task()
    wide = build_task({"decoder_logvar_clamp": [-9.0, 3.0]})
    none = build_task({"decoder_logvar_clamp": None})
    outs = {}
    for name, task in (("default", default), ("wide", wide), ("none", none)):
        with torch.no_grad():
            torch.manual_seed(seed)
            outs[name] = task.orig_model(*task._build_forward_inputs(batch))
    floor_hits = float((outs["wide"]["logvar_full"] == -9.0).float().mean())
    lo_full = float(outs["wide"]["logvar_full"].min())
    hi_full = float(outs["wide"]["logvar_full"].max())
    lo_base = float(outs["wide"]["logvar_base"].min())
    hi_base = float(outs["wide"]["logvar_base"].max())
    record = {
        "strict_load": True,
        "decoder_clamp": list(wide.orig_model.decoder.logvar_clamp),
        "shared_clamp": list(wide.orig_model.logvar_clamp),
        "decoder_logvar_range": [min(lo_full, lo_base), max(hi_full, hi_base)],
        "default_decoder_logvar_range": [float(outs["default"]["logvar_full"].min()),
                                         float(outs["default"]["logvar_full"].max())],
        "at_new_floor_exactly_frac": floor_hits,
        "default_at_old_floor_exactly_frac": float(
            (outs["default"]["logvar_full"] == float(default.orig_model.logvar_clamp[0])).float().mean()),
        "below_old_floor_frac": float((outs["wide"]["logvar_full"] < float(default.orig_model.logvar_clamp[0])).float().mean()),
        "prior_post_bitwise_equal": all(torch.equal(outs["wide"][k], outs["default"][k])
                                        for k in ("logvar_prior", "logvar_post", "mu_prior", "mu_post")),
        "none_decoder_bitwise_equal": all(torch.equal(outs["none"][k], outs["default"][k])
                                          for k in ("mu_full", "logvar_full", "mu_base", "logvar_base")),
    }
    # The loss diagnostics read the decoder's own bound.
    present_epoch(wide)
    with torch.no_grad():
        torch.manual_seed(seed)
        _, metrics = wide.compute_loss_and_metrics(batch, 0, "val")
    record["wide_logvar_full_floor_frac"] = float(metrics["logvar_full_floor_frac"])
    # Closed interval: a cell whose raw value saturates the sigmoid in float32 lands on the bound
    # itself, as it does on -6 under the default model.
    record["pass"] = bool(-9.0 <= record["decoder_logvar_range"][0] and record["decoder_logvar_range"][1] <= 3.0
                          and record["prior_post_bitwise_equal"] and record["none_decoder_bitwise_equal"])
    return record


def check_training(batch: Any, seed: int, tolerance_s: float, windows: Any) -> Dict[str, Any]:
    """Check d: one training step and one validation step with every key on.

    Args:
        batch: The batch.
        seed: The seed.
        tolerance_s: The tolerance of both tolerant rules.
        windows: The untrimmed window vector.

    Returns:
        The record.
    """
    from teb_vae.lag_attn_rws.nets import controls

    task = build_task({"source_running_gap_keys": "remove", "decoder_logvar_clamp": [-9.0, 3.0]},
                      {"source_valid_tolerance_s": tolerance_s, "event_weight_ua_tolerance_s": tolerance_s,
                       "source_valid_windows": list(windows)})
    present_epoch(task)
    model = task.orig_model
    model.train()
    task.train()
    torch.manual_seed(seed)
    loss, metrics = task.compute_loss_and_metrics(batch, 0, "train")
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    with_grad = [g for g in grads if g is not None]
    record: Dict[str, Any] = {
        "train_loss": float(loss.detach()),
        "loss_finite": bool(torch.isfinite(loss.detach())),
        "params_with_grad": len(with_grad),
        "params_total": len(grads),
        "grads_finite": all(bool(torch.isfinite(g).all()) for g in with_grad),
        "lambda_source_null": task.hparams.get("lambda_source_null"),
    }
    model.zero_grad(set_to_none=True)
    model.eval()
    task.eval()
    with torch.no_grad():
        torch.manual_seed(seed)
        val_loss, val_metrics = task.compute_loss_and_metrics(batch, 0, "val")
        inputs = task._build_forward_inputs(batch)
        out = model(*inputs)
        perm = controls.perm_forward_outputs(model, out, perm_index=torch.roll(torch.arange(out["mu_post"].shape[0]), 1),
                                             anchors=out["anchor_index"])
        null = controls.source_null_forward_outputs(model, out, inputs[2])
    record["val_loss"] = float(val_loss)
    record["val_kld_source_null"] = float(val_metrics.get("kld_source_null", float("nan")))
    record["perm_carries_donor_removal"] = bool(torch.equal(
        perm["source_key_removed"], out["source_key_removed"][torch.roll(torch.arange(out["mu_post"].shape[0]), 1)]))
    # An anchor with every key removed must have its posterior equal its prior in every arm.
    closed = ~(model.build_lag_mask(out["mu_post"].shape[1], out["mu_post"].device)[None]
               & ~out["source_key_removed"]).any(dim=-1)
    record["anchors_with_no_key_left"] = int(closed.sum())
    if bool(closed.any()):
        record["closed_gate_max"] = float(out["event_gate"][..., 0][closed].abs().max())
        record["closed_null_gate_max"] = float(null["event_gate"][..., 0][closed].abs().max())
    record["pass"] = bool(record["loss_finite"] and record["grads_finite"] and record["params_with_grad"] > 0
                          and record["perm_carries_donor_removal"] and torch.isfinite(val_loss))
    return record


def main(mode: str, rows: int, seed: int, tolerance_s: float) -> Dict[str, Any]:
    """Write the reference outputs, or run checks a to d.

    Args:
        mode: ``'reference'`` or ``'check'``.
        rows: Rows of the batch.
        seed: The seed.
        tolerance_s: The tolerance of both tolerant rules.

    Returns:
        The record.
    """
    dev, cfg, task = S.load_model(device="cpu")
    full = first_batch(task, cfg, dev)
    # The first rows for the bitwise reference; the rows with the most invalid UA for the rest,
    # so the long runs, the removed keys and the closed anchors are exercised.
    batch = select(full, range(rows))
    hard = select(full, most_invalid_rows(full, rows))
    OUT.mkdir(parents=True, exist_ok=True)
    if mode == "reference":
        record = run_outputs(task, batch, seed)
        torch.save(record, REFERENCE)
        print("saved", REFERENCE, "loss", float(record["loss"]))
        return {"reference": str(REFERENCE)}

    from teb_vae.lag_attn_cfs.ua_tolerance import declared_source_windows

    # The evaluation shards only: the run's training shards are on the server. The trainer reads
    # both lists, and read_causal_warmup refuses shards that disagree.
    paths = list(cfg["dataset_config"].get("vae_test_datasets") or [])
    windows = declared_source_windows(paths, use_up_st=bool(task.orig_model.use_up_st))
    report: Dict[str, Any] = {"rows": rows, "seed": seed, "tolerance_s": tolerance_s,
                              "hard_rows": most_invalid_rows(full, rows),
                              "up_valid_shape": list(batch["up_valid"].shape), "windows": list(windows)}
    for name, run in (("a_inert", lambda: check_inert(task, batch, seed)),
                      ("b_equivalence", lambda: check_equivalence(cfg, dev, hard, seed, tolerance_s, windows)),
                      ("b_equivalence_first_rows",
                       lambda: check_equivalence(cfg, dev, batch, seed, tolerance_s, windows)),
                      ("c_decoder_clamp", lambda: check_decoder_clamp(hard, seed)),
                      ("d_training", lambda: check_training(hard, seed, tolerance_s, windows))):
        report[name] = run()
        print(name, json.dumps(report[name], indent=1, default=str))
    (OUT / "check_production.json").write_text(json.dumps(report, indent=1, default=str))
    print("ALL PASS" if all(report[k]["pass"] for k in report if k[:2] in ("a_", "b_", "c_", "d_"))
          else "SOME CHECK FAILED")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default=RUN_ARGS["mode"], choices=("reference", "check"))
    parser.add_argument("--rows", type=int, default=RUN_ARGS["rows"])
    parser.add_argument("--seed", type=int, default=RUN_ARGS["seed"])
    parser.add_argument("--tolerance-s", type=float, default=RUN_ARGS["tolerance_s"])
    args = parser.parse_args()
    main(args.mode, args.rows, args.seed, args.tolerance_s)
