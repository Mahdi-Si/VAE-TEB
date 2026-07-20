r"""Everything that must hold before a single number is computed.

Preflight exists because the expensive failures in this pipeline are the silent ones. A
checkpoint that did not load, an objective taken from the config instead of the run, a shard
whose widths moved -- each produces a full set of plausible-looking numbers and no error. The
guards here run before the loader is built and before any analysis writes anything, so a
rejected run costs a checkpoint load and two HDF5 shape reads.

**Load is verified in weight space, not only in behaviour space.** A behavioural probe on
``residual_ratio`` cannot distinguish "the checkpoint never loaded" from "a real model whose
source pathway collapsed": both read near zero, and hard-failing on the second would destroy a
genuine finding. So the check that *raises* is that the zero-initialised heads are no longer at
their zero initialisation, which only a real load can produce, and the behavioural reading is
recorded and warned on.

The obvious behavioural test has a trap worth knowing about. ``_zero_init_delta_heads`` zeroes
``residual_decoder.mean_head`` as well as the posterior deltas, so ``delta_mu_src`` is
identically zero at initialisation **regardless of** $z$ -- a model perturbed only through its
posterior head still reads as collapsed.

**Preconditions are enforced, not annotated.** ``causal_norm=False`` means the encoders'
GroupNorm pools across time, the prior conditions on the future, and $K_t$ is not a
transfer-entropy surrogate at all. That is recorded here with its consequence and blocks the
TE-labelled readouts where they are computed; it does not fail the run, because the forecast
and calibration analyses remain perfectly valid. A number that looks like a transfer entropy
and is not one is worse than no number.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import torch
from loguru import logger

from teb_vae.lag_attn.eval.runner import EvalRunner
from teb_vae.lag_attn.trainer import _check_declared_widths_against_shard, _check_stat_path

#: The placeholder ``configs/default.yaml`` deliberately ships instead of a real path, so a
#: launch fails on a missing file rather than on a width mismatch someone might "fix" by
#: reverting the channel counts. Caught by name here so the failure says what to do.
REPOINT_MARKER = "REPOINT_ME"

#: File written into the run directory recording every check and its verdict.
PREFLIGHT_FILENAME = "preflight.json"


def _dataset_paths(config: Dict[str, Any]) -> List[str]:
    """Return every filesystem path the eval loader will open.

    Args:
        config: The merged run config.

    Returns:
        The test shard paths plus the normalization statistics path.
    """
    dataset_config = config.get("dataset_config") or {}
    paths = [str(path) for path in (dataset_config.get("vae_test_datasets") or [])]
    stat_path = dataset_config.get("stat_path")
    if stat_path is not None:
        paths.append(str(stat_path))
    return paths


def _check_repointed(config: Dict[str, Any]) -> None:
    """Refuse a config still carrying the ``REPOINT_ME`` placeholder.

    Checked before the existence guards, so the message names the real cause instead of
    reporting a missing file the operator would then go looking for.

    Args:
        config: The merged run config.

    Raises:
        ValueError: If any resolved dataset path contains the placeholder.
    """
    offenders = [path for path in _dataset_paths(config) if REPOINT_MARKER in path]
    if offenders:
        raise ValueError(
            "dataset_config still carries the REPOINT_ME placeholder:\n  "
            + "\n  ".join(offenders)
            + f"\nThese are deliberate non-paths, not typos: set them to the real k-fold test "
            f"directory and the matching stats file (regenerated from the same dataset at "
            f"trim_minutes=1.0). See {REPOINT_MARKER} in configs/default.yaml for why the "
            f"placeholder is not simply pointed at the previous dataset."
        )


def _config_for_shard_guards(config: Dict[str, Any], runner: EvalRunner) -> Dict[str, Any]:
    """Build the config view the two reused trainer guards expect.

    Two remappings, each of which the guards would otherwise silently no-op on.

    ``_check_declared_widths_against_shard`` reads ``vae_train_datasets``; an eval config sets
    only ``vae_test_datasets``, so without the first remapping it returns early having checked
    nothing at all.

    The widths it compares must be the **model's**, not the config's. Eval rebuilds from the
    checkpoint's ``model_kwargs``, so a checkpoint whose geometry differs from the config's
    would pass a config-versus-shard check and then fail inside the forward, with a channel
    error naming neither the checkpoint nor the config.

    Args:
        config: The merged run config. Not mutated.
        runner: The runner holding the rebuilt model.

    Returns:
        A deep copy with the test shards under the training key and the model's own widths.
    """
    remapped = copy.deepcopy(config)
    dataset_config = remapped.setdefault("dataset_config", {})
    dataset_config["vae_train_datasets"] = list(dataset_config.get("vae_test_datasets") or [])
    vae_config = remapped.setdefault("model_config", {}).setdefault("VAE_model", {})
    vae_config["c_y"] = int(runner.model.c_y)
    vae_config["c_u"] = int(runner.model.c_u)
    vae_config["use_up_st"] = bool(runner.model.use_up_st)
    return remapped


def _zero_initialised_heads(model) -> Dict[str, torch.Tensor]:
    r"""Return the weight tensors ``_zero_init_delta_heads`` zeroes at construction.

    These three are the load witnesses: each starts at exactly $0$ and each receives gradient
    during training, so a nonzero value in any of them can only come from a real load (or from
    a deliberate perturbation, which is how the test suite builds a loadable fixture).

    ``delta_mu_head`` and ``delta_logvar_head`` are ``ModuleList``s under
    ``head_structured_latent``, so each is flattened rather than assumed to be one layer.

    Args:
        model: The rebuilt model.

    Returns:
        ``{name: weight tensor}``.
    """
    weights: Dict[str, torch.Tensor] = {}
    for head_name in ("delta_mu_head", "delta_logvar_head"):
        module = getattr(model.posterior_head, head_name)
        layers = list(module) if isinstance(module, torch.nn.ModuleList) else [module]
        for index, layer in enumerate(layers):
            weights[f"posterior_head.{head_name}[{index}]"] = cast(torch.Tensor, layer.weight)
    weights["residual_decoder.mean_head"] = cast(
        torch.Tensor, model.residual_decoder.mean_head.weight
    )
    return weights


def verify_weights_loaded(model) -> Dict[str, Any]:
    """Verify in weight space that a checkpoint actually reached the model.

    Args:
        model: The rebuilt model, after the checkpoint load.

    Returns:
        A record of each witness weight's maximum absolute value and the verdict.

    Raises:
        RuntimeError: If every zero-initialised head is still exactly zero, which no trained
            model produces and every failed load does.
    """
    magnitudes = {
        name: float(weight.detach().abs().max()) for name, weight in _zero_initialised_heads(model).items()
    }
    loaded = any(value > 0.0 for value in magnitudes.values())
    if not loaded:
        raise RuntimeError(
            "every zero-initialised delta head is still exactly zero, so no checkpoint weights "
            "reached this model. The likeliest causes are a checkpoint whose keys did not align "
            "(load_checkpoint_strict returns None rather than raising), or a path pointing at a "
            "freshly written but untrained checkpoint. This is a weight-space check, not a "
            "behavioural one: a genuinely trained model whose source pathway collapsed still "
            "has nonzero weights here and passes.\n  "
            + "\n  ".join(f"{name}: max|w| = {value:.3e}" for name, value in magnitudes.items())
        )
    return {"passed": True, "max_abs_weight": magnitudes}


def probe_load_health(
    runner: EvalRunner, batch: Any, *, floor: float
) -> Dict[str, Any]:
    r"""Run one batch and record whether the source pathway does anything.

    Warns, never raises. A collapsed source pathway on a genuinely loaded model is a finding to
    report -- possibly the most important one a run can produce -- not a reason to abort.

    ``residual_ratio`` is the primary signal: the RMS of ``delta_mu_src`` relative to the RMS of
    the full forecast, over the supervised anchor range. ``uplift_rel`` is recorded but not
    gated on, because under ``gaussian_nll`` with ``sigma_obs='learned'`` the full and baseline
    losses consume *different* log-variance heads and therefore differ even when
    ``delta_mu_src`` is identically zero -- so a nonzero uplift is not evidence the mean
    pathway is alive.

    Args:
        runner: The loaded runner.
        batch: One batch, already on the compute device.
        floor: ``residual_ratio`` below this warns. From ``eval_config.health_probe_floor``.

    Returns:
        The readings and the verdict.
    """
    model = runner.model
    with runner.inference_mode():
        outputs = runner.forward(batch)
        losses = runner.compute_loss(batch, outputs)

    horizon = int(model.horizon)
    seq_len = int(outputs["mu_full"].shape[1])
    warmup = int(model._warmup_steps(seq_len))
    # The supervised anchor range. Coarse on purpose: this is a probe, not a metric, and the
    # weight-aware mask the analyses share is not needed to tell "alive" from "identically zero".
    low, high = warmup, max(seq_len - horizon, warmup + 1)

    delta_rms = float(outputs["delta_mu_src"][:, low:high].pow(2).mean().sqrt())
    full_rms = float(outputs["mu_full"][:, low:high].pow(2).mean().sqrt())
    residual_ratio = delta_rms / full_rms if full_rms > 0.0 else 0.0

    feat_loss, base_loss = float(losses["feat_loss"]), float(losses["base_loss"])
    uplift_rel = (base_loss - feat_loss) / abs(base_loss) if base_loss != 0.0 else 0.0

    record = {
        "residual_ratio": residual_ratio,
        "residual_ratio_floor": float(floor),
        "delta_mu_src_rms": delta_rms,
        "mu_full_rms": full_rms,
        "feat_loss": feat_loss,
        "base_loss": base_loss,
        "uplift_rel": uplift_rel,
        "kld_raw": float(losses["kld_raw"]),
        "kld_active_frac": float(losses["kld_active_frac"]),
        "anchor_range": [low, high],
        "raised": False,
    }
    if residual_ratio < float(floor):
        record["warning"] = (
            f"residual_ratio {residual_ratio:.3e} is below the configured floor {floor:.3e}: "
            f"the source pathway is contributing almost nothing to the forecast. The weights "
            f"loaded (that is checked separately and separately reported), so this is a "
            f"reading about the model, not about the run. Expect a near-zero uplift and a "
            f"near-zero kld_raw to corroborate it."
        )
        logger.warning(record["warning"])
    return record


def interpretation_preconditions(runner: EvalRunner) -> Dict[str, Any]:
    """Record which readouts this checkpoint's configuration permits, and why.

    Recorded rather than enforced here: neither flag invalidates the forecast, uplift or
    calibration analyses, and a run that produces those is worth having. What each flag blocks
    is blocked where the blocked quantity is computed.

    Args:
        runner: The loaded runner.

    Returns:
        One entry per precondition, each with its verdict and its consequence.
    """
    model = runner.model
    causal_norm = bool(model.causal_norm)
    head_structured = bool(model.head_structured_latent)
    return {
        "causal_norm": {
            "value": causal_norm,
            "blocks": [] if causal_norm else ["te_lag_map", "kld_raw_as_te", "per_head_te"],
            "consequence": (
                "the encoders' GroupNorm pools statistics across time, so the prior "
                "p(z_t | Y_<=t) conditions on the future and K_t is NOT a transfer-entropy "
                "surrogate. Every TE-labelled readout is refused."
                if not causal_norm
                else "K_t is a valid transfer-entropy surrogate: the history states are "
                "strictly causal."
            ),
            "n_causalized_norms": int(model.n_causalized_norms),
        },
        "head_structured_latent": {
            "value": head_structured,
            "blocks": [] if head_structured else ["per_head_kl_decomposition"],
            "consequence": (
                "the posterior consumes the fused attention projection rather than the "
                "per-head summaries, so the per-head KL is an arbitrary slice, not an "
                "additive decomposition. te_lag_map remains available as a diagnostic only."
                if not head_structured
                else "the per-head KL is an additive decomposition, K_t = sum_m K_t^(m), and "
                "te_lag_map is a rigorous lag attribution."
            ),
        },
        "kld_support": {
            "value": str(model.kld_support),
            "blocks": [],
            "consequence": (
                "the KL is reduced over [warmup, T - H_d): the tail anchors have no fully "
                "observed forecast window and receive no supervised gradient."
                if model.kld_support == "anchor"
                else "the KL is reduced over [warmup, T), including tail anchors whose "
                "forecast targets run off the end of the sequence."
            ),
        },
    }


class TEPreconditionUnmet(RuntimeError):
    """A TE-labelled readout was requested on a checkpoint whose configuration invalidates it.

    A distinct type rather than a bare ``RuntimeError`` because ``report.step`` records the
    exception's class name into ``summary.json``: a failed step reading
    ``TEPreconditionUnmet: causal_norm=False ...`` is legibly a *refusal*, while a bare
    ``RuntimeError`` sits in the same list as a genuine crash and reads as one.
    """


def require_causal_norm(runner: EvalRunner, readout: str) -> None:
    """Refuse a transfer-entropy-labelled readout when the encoders are not causal.

    Args:
        runner: The loaded runner.
        readout: What was being computed, named in the message.

    Raises:
        TEPreconditionUnmet: If the checkpoint was built with ``causal_norm=False``.
    """
    if bool(runner.model.causal_norm):
        return
    raise TEPreconditionUnmet(
        f"{readout} is a transfer-entropy readout, and this checkpoint was built with "
        f"causal_norm=False (n_causalized_norms=0). GroupNorm then pools its statistics across "
        f"the whole time axis, so every history state carries a low-bandwidth image of its own "
        f"future and the prior p(z_t | Y_<=t) conditions on Y_>t as well. K_t is therefore not a "
        f"transfer-entropy surrogate at all, and te_lag_map is not an attribution of one. The "
        f"leak is small, invisible in a loss curve, and corrupts precisely the quantity this "
        f"model exists to measure -- so the number is refused rather than reported with a "
        f"caveat. The forecast, uplift, residual and calibration analyses are unaffected and "
        f"still ran."
    )


def require_head_structured_latent(runner: EvalRunner, readout: str) -> None:
    """Refuse the per-head decomposition when the posterior does not consume per-head summaries.

    ``kld_per_t_per_head`` is emitted either way -- it is always the contiguous
    ``view(B, T, M, d_z // M).sum(-1)`` -- which is exactly why this guard is needed. Under a
    flat latent every dimension depends on every head, so that slice is an arbitrary partition
    of a shared quantity and its per-head "shares" sum to $K_t$ as a tautology of the view
    rather than as a decomposition of anything.

    Args:
        runner: The loaded runner.
        readout: What was being computed, named in the message.

    Raises:
        TEPreconditionUnmet: If the checkpoint was built with ``head_structured_latent=False``.
    """
    if bool(runner.model.head_structured_latent):
        return
    raise TEPreconditionUnmet(
        f"{readout} requires head_structured_latent=True and this checkpoint was built with it "
        f"False. The posterior then consumes the fused attention projection rather than the "
        f"per-head summaries, so every latent dimension depends on every head and the "
        f"contiguous per-head slice of kld_per_t_per_head is an arbitrary partition, not an "
        f"additive decomposition. Its shares would still sum to K_t -- that is a property of "
        f"the view, not evidence of attribution. te_lag_map remains available on this "
        f"checkpoint as a diagnostic, computed from the head-averaged attention."
    )


def te_lag_map_label(runner: EvalRunner) -> str:
    """Return how ``te_lag_map`` may be labelled on this checkpoint.

    Under ``head_structured_latent=False`` the model falls back to
    ``kld_per_t * attn_weights.mean(dim=-2)``, which redistributes $K_t$ by an average over
    heads rather than attributing it head by head. That is still worth plotting and still sums
    to $K_t$; it is simply not the rigorous attribution the other branch produces, and the
    output says so rather than leaving a reader to infer it from a flag.

    Args:
        runner: The loaded runner.

    Returns:
        ``'attribution'`` or ``'diagnostic'``.
    """
    return "attribution" if bool(runner.model.head_structured_latent) else "diagnostic"


def run_preflight(*, config: Dict[str, Any], runner: EvalRunner) -> Dict[str, Any]:
    """Run every hard guard, then record the preconditions. No data is read beyond two shapes.

    Order is deliberate: the placeholder check first so its message is not pre-empted by a
    missing-file error, then the reused trainer guards, then the objective reconciliation, then
    the weight-space load verification.

    Args:
        config: The merged run config.
        runner: The runner holding the rebuilt model.

    Returns:
        The preflight record, ready for :func:`write_preflight`.

    Raises:
        ValueError: On a ``REPOINT_ME`` path, a missing ``stat_path``, a width mismatch, or an
            objective disagreement between config and checkpoint.
        RuntimeError: If the checkpoint's weights never reached the model.
    """
    _check_repointed(config)

    guard_config = _config_for_shard_guards(config, runner)
    # Reused rather than copied, so their long actionable messages can never drift from the
    # trainer's.
    _check_stat_path(guard_config)
    _check_declared_widths_against_shard(guard_config)

    runner.objective.reconcile_with_config(config)
    load_check = verify_weights_loaded(runner.model)

    record = {
        "checkpoint": str(runner.checkpoint_path),
        "device": str(runner.device),
        "geometry": runner.geometry(),
        "model_kwargs": dict(runner.model_kwargs),
        "objective": runner.objective.as_dict(),
        "dataset_paths": _dataset_paths(config),
        "checks": {
            "repoint_placeholder": {"passed": True},
            "stat_path": {"passed": True, "path": (config.get("dataset_config") or {}).get("stat_path")},
            "declared_widths": {
                "passed": True,
                "compared_against": "model.c_y / model.c_u from the checkpoint's model_kwargs",
            },
            "objective_matches_config": {"passed": True},
            "weights_loaded": load_check,
        },
        "preconditions": interpretation_preconditions(runner),
    }
    logger.info(
        "preflight passed: widths, stats path and objective all agree with the checkpoint; "
        "delta heads carry loaded weights"
    )
    return record


def write_preflight(record: Dict[str, Any], output_dir: Any) -> Path:
    """Write the preflight record into the run directory.

    Args:
        record: The record from :func:`run_preflight`, optionally extended with the health
            probe.
        output_dir: The run directory.

    Returns:
        The path written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / PREFLIGHT_FILENAME
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, default=str)
    logger.info(f"wrote {path}")
    return path


def first_batch(loader: Any) -> Optional[Any]:
    """Return the loader's first batch, or ``None`` when the split is empty.

    Args:
        loader: The eval dataloader.

    Returns:
        The first batch, or ``None``.
    """
    for batch in loader:
        return batch
    return None
