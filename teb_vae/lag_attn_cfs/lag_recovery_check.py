r"""Can this architecture recover a delay it is known to be looking at?

Every lag readout in this family answers *where in the past did the source inform the future*, and
on real recordings that answer has nothing to be scored against: there is no ground-truth delay
written down anywhere, so a profile pinned at lag $0$ is unattributable. It could be an
architecture that cannot express a lag, or a domain in which none exists.

This module removes the ambiguity by measuring the readout on data where the answer is known. The
committed planted-delay shard carries a source-to-target coupling at a delay $\delta$ stamped on the
file as a root attribute; ``scripts/make_tiny_shard.py`` verified by direct cross-correlation, before
any model existed, that the stored coefficients actually carry it. What this script does is fit the
configured model on that shard for a few epochs and read its lag profile back through the evaluation
pipeline's own code.

**The band the peak must fall in, and where it comes from.** Target content at stored step $s$ is a
function of source content at $s - \delta$. A model at anchor $t$ forecasts target step $t + 1 + h$,
whose source content therefore sits at $t + 1 + h - \delta$, i.e. at attention lag

$$\ell = \delta - 1 - h, \qquad h \in [0, H),$$

so across the forecast block the informative lags are exactly $[\delta - H,\ \delta - 1]$. The shard
is built with $\delta \in (H, L-1)$ precisely so that band is non-empty, strictly inside the searched
window, and clear of lag $0$ -- which is what makes a profile pinned at the near edge a **failure**
rather than an ambiguity. The band is derived here from the shard's stamped $\delta$ and the run's
own $H$, never written down: a fixture rebuilt at another delay moves it.

**Nothing about the profile is recomputed here.** The lag block comes from
:func:`~teb_vae.lag_attn_cfs.eval.metrics.evaluate` and its peaks from
:func:`~teb_vae.lag_attn_cfs.eval.analyses.lag_kl.build_summary_rows`, which is the same
:mod:`~teb_vae.lag_attn_cfs.eval.lag_shape` vocabulary a production evaluation reports -- so
"degenerate", "the peak's width" and "the mass above half the peak" mean here exactly what they mean
in a ``summary.json``. A second implementation of any of it would let this check and the evaluation
disagree about one profile.

**What it prints, and why each line is there.**

* A **switch header**: the configured value and the built model's own value of every architecture
  switch, and both alignment references. Two readings rather than one, because a config key that
  never reaches a constructor is dropped silently by the driver's ``inspect.signature`` sweep -- so
  a key reading ``configured=conv_stem, model=absent`` is the failure mode this line exists to
  catch. A switch a model does not have yet reads ``absent`` on both, which is the honest statement
  and becomes a real reading the moment the key lands.
* The **KL at initialisation**, measured on a freshly constructed model in train mode. The whole
  coupling readout rests on the source starting at exactly zero nats, and every addition to the
  prior path is a chance to break it.
* The **profile** and its pass/fail line against the band, the **per-head** peaks and band shares,
  and ``kld_source_null`` -- the part of the KL that survives zeroing the source, which is the
  availability clock rather than the coupling.
* A **manifest** mode: state-dict keys, shapes and the parameter total, which is what a bitwise
  off-state claim is checked by. It builds the model and stops, so it costs no fit.

**Overrides are the comparison mechanism.** ``--override model_config.VAE_model.<key>=<value>``
applies a delta over the planted config before anything is built, and the merged document is dumped
into the run directory as its ``resolved_config.yaml``. That is what makes a one-switch contrast
possible -- the same tree, the same seed, the same shard, one key different -- and what keeps the
contrast recoverable from the run's own artifacts afterwards rather than from a shell history.

.. code-block:: bash

    python teb_vae/lag_attn_cfs/lag_recovery_check.py
    python teb_vae/lag_attn_cfs/lag_recovery_check.py --mode manifest
    python teb_vae/lag_attn_cfs/lag_recovery_check.py \
        --config teb_vae/lag_attn_transformer_cfs/configs/planted.yaml

From an IDE's Run button, with no command line: everything has a default, and ``RUN_ARGS`` near the
bottom of this file is where they are edited.

**Why a module here rather than beside ``check_run.py``'s tier scoring.** That module is torch-free
by contract -- it scores a run *while it is still in flight*, from a CSV, with no checkpoint and no
shard. This one builds a network, fits it and runs a decoder pass, so the two cannot share a file
without the cheap one acquiring a numeric stack.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Repository root: ``teb_vae/lag_attn_cfs/lag_recovery_check.py`` -> up three.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# An IDE's Run button executes this file as a script, which puts *this directory* on ``sys.path``
# rather than the repository root -- so every ``teb_vae.`` import below would fail with
# ModuleNotFoundError before ``__main__`` is ever reached.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yaml  # noqa: E402

from teb_vae.lag_attn.config import load_config  # noqa: E402
from teb_vae.lag_attn_rws.eval.launch import (  # noqa: E402
    missing_required,
    resolve_launch_args,
)

# =================================================================================================
# What the header states
# =================================================================================================
#: Every architecture switch this check reports, as ``(config key, dotted path on the built net)``.
#:
#: Written out rather than discovered, and reported even where the model has no such attribute: the
#: point of the line is the **pair** of readings. A config key that names no constructor parameter
#: is dropped in silence by the driver's ``inspect.signature`` sweep, so an arm can train as the
#: baseline with nothing in its log saying so -- and the only thing that catches it is the
#: configured value and the built value printed side by side.
SWITCH_KEYS: Tuple[Tuple[str, str], ...] = (
    ("lag_kv_source", "lag_kv_source"),
    ("prior_availability_input", "prior_availability_input"),
    ("persistence_residual", "persistence_residual"),
    ("horizon_weight_halflife_steps", "horizon_weight_halflife_steps"),
    # Not a model attribute: the scale is consumed where the bias is built, so the attention module
    # is where a reader has to look for what a run actually seeded.
    ("alibi_slope_scale", "lag_attn.alibi_slope_scale"),
)

#: The alignment references, which are resolution-time keys rather than constructor arguments -- so
#: their model-side reading is the resolved budget's rather than an attribute's, and it is reported
#: separately below.
REFERENCE_KEYS: Tuple[str, ...] = ("causal_align_reference", "causal_align_reference_source")

#: What a switch reads when neither the config nor the model has it. Written out because it is the
#: value a reader has to be able to tell apart from ``None``: a key absent from the tree and a key
#: present and set to null are different states, and the second is a configured off-state.
ABSENT = "absent"


@dataclass(frozen=True)
class Cell:
    """Which cell of the causal grid a planted config belongs to, and what rebuilds its model.

    Attributes:
        package: The package the config lives in, which is how a config path names its cell.
        trainer: ``(module, attribute)`` of the experiment driver's entry point, called with the
            merged config path.
        driver_cls: ``(module, attribute)`` of the driver **class**, which builds a model without
            fitting one -- the manifest and the initialisation reading both need that, and the
            entry point above always fits.
        binding: ``(module, attribute)`` of the evaluation binding, which is what rebuilds a
            checkpoint into the right class and scores it through the right task.
    """

    package: str
    trainer: Tuple[str, str]
    driver_cls: Tuple[str, str]
    binding: Tuple[str, str]


#: The two feature-target causal cells, keyed by the package directory a planted config sits in.
#:
#: A registry rather than an import derived from the path, because the two facts a config cannot
#: carry -- which driver fits it and which binding rebuilds its checkpoint -- are exactly what the
#: evaluation pipeline's own ``ModelBinding`` exists to state. Guessing them from a module name
#: would rebuild one architecture under another's name wherever two constructors happened to accept
#: the same keys.
CELLS: Dict[str, Cell] = {
    "lag_attn_cfs": Cell(
        package="lag_attn_cfs",
        trainer=("teb_vae.lag_attn_cfs.trainer", "main"),
        driver_cls=("teb_vae.lag_attn_cfs.trainer", "LagAttnCfsTrainer"),
        binding=("teb_vae.lag_attn_cfs.eval.binding", "CFS_BINDING"),
    ),
    "lag_attn_transformer_cfs": Cell(
        package="lag_attn_transformer_cfs",
        trainer=("teb_vae.lag_attn_transformer_cfs.trainer", "main"),
        driver_cls=("teb_vae.lag_attn_transformer_cfs.trainer", "LagAttnTrfCfsTrainer"),
        binding=("teb_vae.lag_attn_transformer_cfs.eval.binding", "TRF_CFS_BINDING"),
    ),
}

#: Where this check's own artifacts land inside the run directory the fit created.
RECORD_STEM = "lag_recovery_check"


def resolve_cell(config_path: str) -> Cell:
    """Name the cell a planted config belongs to, from where the config lives.

    Args:
        config_path: Path to the planted config, absolute or repository-root-relative.

    Returns:
        The cell.

    Raises:
        ValueError: If the config does not sit inside a known cell's ``configs/`` directory. A
            guessed cell would fit one architecture and score it under another's name.
    """
    package = Path(config_path).resolve().parent.parent.name
    if package not in CELLS:
        raise ValueError(
            f"{config_path} sits in package {package!r}, which is not a cell this check knows: "
            f"{', '.join(sorted(CELLS))}. The driver and the evaluation binding are facts a config "
            f"cannot carry, so they are looked up by the package the config lives in."
        )
    return CELLS[package]


def _import(target: Tuple[str, str]) -> Any:
    """Import one ``(module, attribute)`` pair, deferring the numeric stack until it is needed."""
    import importlib

    return getattr(importlib.import_module(target[0]), target[1])


# =================================================================================================
# The override delta
# =================================================================================================
def override_tree(overrides: Sequence[str]) -> Dict[str, Any]:
    """Turn ``dotted.key=value`` strings into the nested mapping a config delta is.

    Values are parsed as YAML, so ``true``, ``null``, ``0.0`` and ``conv_stem`` all arrive as the
    types a config file would give them -- which matters because a switch read as the **string**
    ``"false"`` is truthy and would turn a feature on while the header printed that it was off.

    Args:
        overrides: Zero or more ``dotted.key=value`` strings.

    Returns:
        The delta, nested by the dots.

    Raises:
        ValueError: If an entry carries no ``=``, or names an empty key path.
    """
    tree: Dict[str, Any] = {}
    for entry in overrides:
        if "=" not in entry:
            raise ValueError(
                f"--override {entry!r} carries no '='. Each override is one "
                f"dotted.key=value pair, e.g. model_config.VAE_model.max_lag=90."
            )
        path, _, raw = entry.partition("=")
        keys = [part for part in path.split(".") if part]
        if not keys:
            raise ValueError(f"--override {entry!r} names an empty key path.")
        cursor = tree
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = yaml.safe_load(raw)
    return tree


def write_override_config(config_path: str, overrides: Sequence[str], directory: Path) -> Path:
    """Write a child config that inherits the planted one and applies the overrides.

    A child with a ``base:`` rather than a pre-merged document, so the merge is the loader's own
    deep merge and an override of one ``VAE_model`` key cannot drop the rest of the block. The
    driver resolves the chain again and dumps the result beside the checkpoints, which is what puts
    the exact configuration that ran into the run's own artifacts.

    Args:
        config_path: The planted config, absolute.
        overrides: The ``dotted.key=value`` deltas.
        directory: Where the child is written.

    Returns:
        The child's path, or the planted config itself when there is nothing to override.
    """
    if not overrides:
        return Path(config_path)
    directory.mkdir(parents=True, exist_ok=True)
    child = directory / "planted_override.yaml"
    document: Dict[str, Any] = {"base": os.path.abspath(config_path)}
    document.update(override_tree(overrides))
    with open(child, "w", encoding="utf-8") as handle:
        yaml.safe_dump(document, handle, sort_keys=False)
    return child


# =================================================================================================
# The header
# =================================================================================================
def _attribute(model: Any, path: str) -> Any:
    """Follow a dotted attribute path, returning :data:`ABSENT` where it breaks."""
    cursor: Any = model
    for name in path.split("."):
        if not hasattr(cursor, name):
            return ABSENT
        cursor = getattr(cursor, name)
    return cursor


def switch_lines(config: Dict[str, Any], model: Optional[Any], budget: Any) -> List[str]:
    """The header: every switch as configured and as built, and both alignment references.

    Args:
        config: The merged run config.
        model: The built network, or ``None`` when no model exists yet.
        budget: The resolved warm-up budget, for the reference the run actually got.

    Returns:
        The header lines.
    """
    vae_config = (config.get("model_config") or {}).get("VAE_model") or {}
    lines = ["switches (configured | as built):"]
    for key, path in SWITCH_KEYS:
        configured = vae_config[key] if key in vae_config else ABSENT
        built = ABSENT if model is None else _attribute(model, path)
        lines.append(f"  {key:>32} = {configured!r:>16} | {built!r}")
    lines.append("references:")
    for key in REFERENCE_KEYS:
        lines.append(
            f"  {key:>32} = {(vae_config[key] if key in vae_config else ABSENT)!r}"
        )
    lines.append(
        f"  {'resolved target reference (s)':>32} = "
        f"{None if budget is None else budget.reference_delay_s!r}"
    )
    # The clock the SOURCE stream was actually shifted onto, which is the target's unless the dual
    # key names another one -- printed resolved rather than as configured, because the configured
    # value is null on the arm where the two coincide and a reader of this header needs the number
    # the lag axis was built from either way. The offset beside it is what a physical lag carries.
    lines.append(
        f"  {'resolved source reference (s)':>32} = "
        f"{None if budget is None else budget.source_clock_delay_s!r}"
    )
    lines.append(
        f"  {'inter-stream offset (s)':>32} = "
        f"{None if budget is None else budget.inter_stream_offset_s!r}"
    )
    lines.append(
        f"  {'source delay steps':>32} = "
        f"{ABSENT if model is None else _attribute(model, 'source_delay_steps')!r}"
    )
    if model is not None:
        lines.extend(clock_lines(model))
    return lines


def clock_lines(model: Any) -> List[str]:
    """Report whether the prior's clock can carry anything at all over the scored anchors.

    This line exists because the first form of this mechanism could not, and nothing said so. The
    intuitive clock is the availability staircase $\\mathbb 1[t \\ge W'_c + d_c]$; the constructor
    refuses any anchor floor below $\\max_c(W'_c + d_c)$, which is exactly the last step at which
    that staircase changes, so **every scored anchor sees the same constant vector** -- an offset
    the prior head's biases already span. The mechanism was fully built, the flag read on, the
    projection was reached by gradient, and the quantity it existed to move did not move.

    So the diagnostic is the count that decides it: how many distinct rows the clock takes over
    $[F, T_{\\mathrm{valid}})$. One is inert whatever else is true. It is reported beside the
    train/eval difference, because a clock that is not identical in the two modes is not a
    deterministic function of $t$ and the prior's input distribution would differ between the mode
    the objective runs in and the mode every readout is measured in.

    Args:
        model: The built network.

    Returns:
        The clock's shape, its distinct-row count over the scored region, and the two modes'
        agreement -- or a line saying the flag is off.
    """
    import torch

    if not getattr(model, "prior_availability_input", False):
        return [f"  {'prior clock':>32} = {ABSENT} (prior_availability_input is off)"]

    # The clock reads its argument for shape, dtype and device only -- it encodes a stream of
    # zeros -- so no batch is needed and this line costs no loader.
    reference = next(model.parameters())
    u_stream = torch.zeros(
        1, int(model.sequence_length), int(model.c_u),
        device=reference.device, dtype=reference.dtype,
    )
    with torch.no_grad():
        was_training = model.training
        model.train()
        try:
            train_clock = model._prior_clock(u_stream)
        finally:
            model.train(was_training)
        model.eval()
        eval_clock = model._prior_clock(u_stream)
        model.train(was_training)

    floor, t_valid = int(model.warmup_period), int(model.geometry.t_valid)
    scored = eval_clock[0, floor:t_valid]
    # Rounded before the distinct count: these are float activations, so two rows that differ in
    # the last bit are "distinct" to `unique` and would report an inert clock as a live one.
    distinct = int(torch.unique(scored.round(decimals=4), dim=0).shape[0])
    return [
        f"  {'prior clock':>32} = shape {tuple(eval_clock.shape)}, "
        f"{distinct} distinct row(s) over the {t_valid - floor} scored steps",
        f"  {'clock train vs eval':>32} = "
        f"{float((train_clock - eval_clock).abs().max()):.3e} max-abs "
        f"(must be 0: the clock is a function of t, not of the dropout switch)",
    ]


# =================================================================================================
# Construction-only readings: the manifest, and the KL at initialisation
# =================================================================================================
def build_model(cell: Cell, config_path: Path) -> Tuple[Any, Dict[str, Any]]:
    """Build the net through the driver's own config-to-constructor path, without fitting.

    The driver rather than a direct constructor call, because the translation is not trivial: the
    warm-up budget names no constructor argument at all and is resolved against the configured
    shards into four channel tuples, and a second implementation of that would be a second
    description of the geometry.

    **The seeding is the driver's own** rather than a `seed_everything` call here, and the
    distinction is not cosmetic. `configure_determinism` reads `general_config.seed` *and*
    `advanced_config.trainer.deterministic` and reconciles the cuDNN and TF32 backend flags with
    them; a bare seed sets the streams and leaves the backends wherever the process found them, so
    two runs of this check could seed identically and still diverge in a convolution. It runs
    before `create_model`, because the weights this check compares are drawn there -- which is the
    same order the full run uses, where it is `setup_config` that calls it.

    Args:
        cell: Which cell's driver to use.
        config_path: The merged config.

    Returns:
        ``(driver, config)``. The driver carries ``pytorch_model``, ``pl_model`` and the resolved
        warm-up budget.
    """
    trainer_cls = _import(cell.driver_cls)
    config = load_config(str(config_path))

    # The driver reads a path and takes no dict, so the merged document reaches it as a file. The
    # directory is temporary because nothing is fitted here and no run exists to own it.
    with tempfile.TemporaryDirectory(prefix="lag_recovery_") as directory:
        resolved = Path(directory) / "resolved_config.yaml"
        with open(resolved, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        driver = trainer_cls(config_file_path=str(resolved))
        driver.configure_determinism()
        driver.create_model()
    return driver, config


def manifest(model: Any) -> Dict[str, Any]:
    """Describe a built model by its state dict, which is what an off-state claim is checked by.

    A parameter total alone cannot say that two constructions are the same model: a key renamed, a
    buffer that became persistent or a tensor that changed shape while another compensated would all
    leave the total standing. The keys and their shapes cannot.

    Args:
        model: The built network.

    Returns:
        The parameter total, the trainable total, the buffer count and one ``key: shape`` entry per
        state-dict tensor, in the state dict's own order.
    """
    state = model.state_dict()
    return {
        "n_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "n_trainable": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "n_state_dict_entries": int(len(state)),
        "state_dict": {name: list(tensor.shape) for name, tensor in state.items()},
    }


def kl_at_initialisation(driver: Any, loader: Any) -> Dict[str, Any]:
    """Measure the source-conditioned KL of a freshly built model on one batch.

    In **train** mode, with dropout live, because that is where the objective runs: the posterior is
    a zero-initialised residual on the prior under one shared $\\epsilon$, so the KL is exactly zero
    at step $0$, and every addition to the prior path is a chance to break that. It is reported as a
    worst case over the batch rather than a mean -- a KL that is zero on average and non-zero
    somewhere is not zero.

    Args:
        driver: The driver holding the freshly built net.
        loader: A dataloader to draw one batch from.

    Returns:
        The two worst-case magnitudes, and the batch they were measured over.
    """
    import torch

    model, task = driver.pytorch_model, driver.pl_model
    batch = next(iter(loader))
    was_training = model.training
    model.train()
    try:
        with torch.no_grad():
            inputs = task._build_forward_inputs(batch)
            outputs = model(*inputs)
    finally:
        model.train(was_training)
    return {
        "kld_per_t_max_abs": float(outputs["kld_per_t"].abs().max()),
        "source_kl_lag_map_max_abs": float(outputs["source_kl_lag_map"].abs().max()),
        "batch_size": int(outputs["kld_per_t"].shape[0]),
    }


# =================================================================================================
# The measurement
# =================================================================================================
def planted_band(delay_steps: int, horizon: int) -> Tuple[int, int]:
    r"""The lag band a planted delay of $\delta$ occupies across the forecast block.

    $\ell = \delta - 1 - h$ over $h \in [0, H)$, so the band is $[\delta - H,\ \delta - 1]$. Derived
    from the shard's stamped delay and the run's own horizon rather than written down, because a
    fixture rebuilt at another delay or a run at another horizon moves it.

    Args:
        delay_steps: $\delta$, the planted delay in decimated steps.
        horizon: $H$, the run's forecast length in decimated steps.

    Returns:
        The inclusive band ``(lo, hi)``.

    Raises:
        ValueError: If the band would reach at or below lag $0$ -- which happens exactly when
            $\delta \le H$, and means the instrument cannot distinguish recovery from a profile
            pinned at the near censoring edge.
    """
    low, high = int(delay_steps) - int(horizon), int(delay_steps) - 1
    if low <= 0:
        raise ValueError(
            f"a planted delay of {delay_steps} steps at horizon {horizon} puts the readable band "
            f"at [{low}, {high}], which reaches lag 0. The whole point of the instrument is that a "
            f"peak in the band and a peak at the near edge are different findings, so the shard "
            f"must be built with a delay strictly above the horizon."
        )
    return low, high


def band_share(profile: Sequence[float], band: Tuple[int, int]) -> float:
    """What fraction of a profile's total mass sits inside the band.

    A plain sum over an interval rather than anything from the shape vocabulary, and deliberately:
    the band is a property of the **plant**, which no production profile has, while ``peak_width``,
    ``mass_above`` and ``degeneracy`` describe a profile against itself and are taken from the
    evaluation's own module so that they mean one thing everywhere.

    Args:
        profile: One value per lag.
        band: The inclusive lag band.

    Returns:
        The share in $[0, 1]$, or ``NaN`` when the profile carries no positive mass.
    """
    import numpy as np

    values = np.asarray(list(profile), dtype=np.float64)
    finite = np.where(np.isfinite(values), values, 0.0)
    total = float(finite.sum())
    if not finite.size or total <= 0.0:
        return float("nan")
    low, high = band
    return float(finite[max(low, 0) : min(high + 1, finite.size)].sum() / total)


def recovery_record(
    results: Dict[str, Any], *, band: Tuple[int, int], delay_steps: int
) -> Dict[str, Any]:
    """Score the evaluated lag block against the planted band.

    Args:
        results: What :func:`~teb_vae.lag_attn_cfs.eval.metrics.evaluate` returned.
        band: The inclusive lag band the plant occupies.
        delay_steps: The planted delay, carried into the record so a reader of the JSON alone has
            the geometry the verdict was reached at.

    Returns:
        The verdict, the argmax of every profile the evaluation reports with the peak description
        the evaluation's own vocabulary gives it, the per-head peaks and band shares, and the two
        control readouts the fixture is also a check on.
    """
    from teb_vae.lag_attn_cfs.eval import lag_shape
    from teb_vae.lag_attn_cfs.eval.analyses import lag_kl

    lag = dict(results.get("lag") or {})
    readouts = dict(results.get("readouts") or {})
    delay = int(lag.get("delay_steps") or 0)
    corrected = list(lag.get("kl_lag_profile_support_corrected") or [])
    argmax = lag.get("kl_argmax_lag_step_support_corrected")

    heads: List[Dict[str, Any]] = []
    entropies = list(lag.get("attention_entropy_per_head_nats") or [])
    for index, profile in enumerate(lag.get("attention_lag_profile_per_head") or []):
        peak = lag_shape.peak_width(profile)
        heads.append(
            {
                "head": index,
                "argmax_lag_step": peak["argmax"],
                "peak_width_bins": peak["width_bins"],
                "band_share": band_share(profile, band),
                "degenerate": lag_shape.degeneracy(profile)["degenerate"],
                "entropy_nats": (
                    float(entropies[index]) if index < len(entropies) else float("nan")
                ),
            }
        )

    # The pass criterion, stated once: inside the band AND not at the near edge. The second clause
    # is not implied by the first only because a band that reached lag 0 would make it so -- which
    # ``planted_band`` refuses, so the two agree by construction and are both written down anyway.
    recovered = (
        argmax is not None and band[0] <= int(argmax) <= band[1] and int(argmax) != 0
    )
    return {
        "planted_delay_steps": int(delay_steps),
        "band": [int(band[0]), int(band[1])],
        "recovered": bool(recovered),
        "argmax_support_corrected": None if argmax is None else int(argmax),
        "band_share_support_corrected": band_share(corrected, band),
        # The evaluation's own peak vocabulary, over all three profiles it reports.
        "peaks": lag_kl.build_summary_rows(lag, delay),
        "heads": heads,
        "n_lags": lag.get("n_lags"),
        "num_heads": lag.get("num_heads"),
        "delay_steps": delay,
        "kld_source_null": readouts.get("kld_source_null"),
        "source_conditioned_kl_raw": readouts.get("source_conditioned_kl_raw"),
        "coupling_minus_clock": readouts.get("coupling_minus_clock"),
        "n_samples": results.get("n_samples"),
        "n_recordings": results.get("n_recordings"),
    }


def format_recovery(record: Dict[str, Any]) -> str:
    """Render the scored record as the lines an operator reads.

    Args:
        record: What :func:`recovery_record` returned.

    Returns:
        The pass/fail line, the three profiles' peaks, the per-head table and the control readouts.
    """
    band = record["band"]
    lines = [
        f"planted delay {record['planted_delay_steps']} steps -> readable lag band "
        f"[{band[0]}, {band[1]}] of [0, {int(record['n_lags'] or 0) - 1}]",
        f"VERDICT: {'PASS' if record['recovered'] else 'FAIL'} -- support-corrected argmax at lag "
        f"{record['argmax_support_corrected']}, {record['band_share_support_corrected']:.3f} of the "
        f"profile's mass inside the band",
        "profiles (the evaluation's own peak vocabulary):",
        f"  {'profile':>18}  {'argmax':>6}  {'width':>6}  {'mass>half':>9}  degenerate",
    ]
    for row in record["peaks"]:
        width = row["peak_width_bins"]
        mass = row["mass_above_half_peak"]
        lines.append(
            f"  {row['profile']:>18}  {str(row['argmax_lag_step']):>6}  {str(width):>6}  "
            f"{'nan' if mass is None else format(float(mass), '.3f'):>9}  {row['degenerate']}"
        )
    lines.append("attention per head:")
    lines.append(f"  {'head':>6}  {'argmax':>6}  {'width':>6}  {'in band':>8}  {'entropy':>8}")
    for head in record["heads"]:
        lines.append(
            f"  {head['head']:>6}  {str(head['argmax_lag_step']):>6}  "
            f"{str(head['peak_width_bins']):>6}  {head['band_share']:>8.3f}  "
            f"{head['entropy_nats']:>8.3f}"
        )
    lines.append(
        f"controls: kld_source_null = {record['kld_source_null']!r}, "
        f"source_conditioned_kl_raw = {record['source_conditioned_kl_raw']!r}, "
        f"coupling_minus_clock = {record['coupling_minus_clock']!r}"
    )
    return "\n".join(lines)


# =================================================================================================
# Entry point
# =================================================================================================
def build_parser() -> argparse.ArgumentParser:
    """The command line, with every default left at ``None``.

    A non-``None`` argparse default would be indistinguishable from a value the operator typed,
    which would make the matching :data:`RUN_ARGS` entry unreachable: the dict would be edited,
    nothing would change, and nothing would say why.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description="Fit the planted-delay fixture and read the lag profile back."
    )
    parser.add_argument(
        "--config",
        help=(
            "A planted config, from either feature-target causal cell. A relative path is "
            "resolved against the repository root."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("check", "manifest"),
        help=(
            "'check' fits and scores the lag profile; 'manifest' builds the model, prints its "
            "state-dict manifest and stops, which is what a bitwise off-state claim is diffed on."
        ),
    )
    parser.add_argument(
        "--override",
        dest="override",
        nargs="+",
        help=(
            "dotted.key=value deltas applied over the config before anything is built, e.g. "
            "model_config.VAE_model.alibi_slope_scale=1.0. The merged document is dumped into the "
            "run directory, so the contrast is recoverable from the run's own artifacts."
        ),
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help=(
            "Where the manifest is written in 'manifest' mode. In 'check' mode the record goes "
            "into the run directory the fit created, and this is ignored."
        ),
    )
    parser.add_argument("--device", help="Device the evaluation pass runs on; CUDA when present.")
    return parser


def main(
    *,
    config: str,
    mode: str = "check",
    override: Optional[Sequence[str]] = None,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> int:
    """Build, optionally fit, and report.

    Args:
        config: The planted config, absolute or repository-root-relative.
        mode: ``'check'`` or ``'manifest'``.
        override: ``dotted.key=value`` deltas.
        output_dir: Where a manifest is written; ignored in ``'check'`` mode.
        device: Device for the evaluation pass.

    Returns:
        The process exit code: $0$ when the mode completed and, in ``'check'`` mode, the planted
        band was recovered; $1$ when it was not. A failure here is a **measurement**, not an error,
        and the non-zero code is what lets a caller act on it without parsing the report.
    """
    from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget

    config_path = config if os.path.isabs(config) else os.path.join(_REPO_ROOT, config)
    cell = resolve_cell(config_path)

    with tempfile.TemporaryDirectory(prefix="lag_recovery_cfg_") as staging:
        merged_path = write_override_config(config_path, override or (), Path(staging))
        merged = load_config(str(merged_path))
        budget = resolve_warmup_budget(merged)

        if mode == "manifest":
            driver, _ = build_model(cell, merged_path)
            record = manifest(driver.pytorch_model)
            print("\n".join(switch_lines(merged, driver.pytorch_model, budget)))
            print(
                f"manifest: {record['n_parameters']} parameters "
                f"({record['n_trainable']} trainable) over {record['n_state_dict_entries']} "
                f"state-dict entries"
            )
            for name, shape in record["state_dict"].items():
                print(f"  {name}  {shape}")
            if output_dir is not None:
                directory = Path(output_dir)
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"{RECORD_STEM}_manifest.json"
                path.write_text(json.dumps(record, indent=2), encoding="utf-8")
                print(f"wrote {path}")
            return 0

        # The fit. Everything below needs a run directory, and the driver is the only handle on
        # where it went -- the name carries a timestamp resolved inside ``setup_config``.
        fit = _import(cell.trainer)(str(merged_path))
        checkpoint = getattr(fit.checkpoint_callback, "best_model_path", "") or ""
        if not checkpoint:
            print(
                "the fit produced no checkpoint, so there is nothing to score. Check that the "
                "config's model_checkpoint monitor names a metric the run actually emits."
            )
            return 1
        run_dir = Path(fit.model_checkpoint_dir).parent
        print(f"fitted {cell.package}: checkpoint {checkpoint}")

    return _score(
        cell,
        checkpoint=checkpoint,
        config=merged,
        budget=budget,
        run_dir=run_dir,
        device=device,
    )


def _score(
    cell: Cell,
    *,
    checkpoint: str,
    config: Dict[str, Any],
    budget: Any,
    run_dir: Path,
    device: Optional[str],
) -> int:
    """Load the fitted checkpoint, evaluate it on the planted shard and write the record.

    Args:
        cell: Which cell's binding rebuilds the checkpoint.
        checkpoint: The checkpoint to score.
        config: The merged run config.
        budget: The resolved warm-up budget, for the header.
        run_dir: The run directory this check's artifacts are written into.
        device: Device the pass runs on.

    Returns:
        The process exit code.
    """
    import h5py

    from teb_vae.lag_attn_cfs.eval import metrics, probe
    from train.data_module import GraphDataModule

    binding = _import(cell.binding)
    resolved_device = probe.resolve_device(device)
    task = probe.load_task(checkpoint, resolved_device, binding=binding)
    model = task.orig_model

    header = switch_lines(config, model, budget)

    # The plant's geometry comes off the shard rather than out of this file: a fixture rebuilt at
    # another delay must move the band it is scored against, and a constant here would keep scoring
    # against the delay the fixture used to have.
    shard = ((config.get("dataset_config") or {}).get("vae_test_datasets") or [None])[0]
    with h5py.File(shard, "r") as handle:
        if "planted_delay_steps" not in handle.attrs:
            print(
                f"{shard} carries no 'planted_delay_steps' attribute, so no delay was planted in "
                f"it and there is nothing to score a lag profile against. Point the config at the "
                f"planted shard."
            )
            return 1
        delay_steps = int(handle.attrs["planted_delay_steps"])

    band = planted_band(delay_steps, int(model.horizon))
    loader = GraphDataModule(config).test_dataloader()

    # Before the trained numbers, because it is a property of the construction rather than of the
    # fit, and because a broken zero-KL start makes every KL below un-comparable to any other run.
    fresh, _ = build_model(cell, Path(run_dir) / "model_checkpoints" / "resolved_config.yaml")
    initialisation = kl_at_initialisation(fresh, loader)

    results = metrics.evaluate(
        task, loader, delay_steps=int(model.source_delay_steps)
    )
    record = recovery_record(results, band=band, delay_steps=delay_steps)
    record["switches"] = header
    record["kl_at_initialisation"] = initialisation
    record["checkpoint"] = str(checkpoint)

    report = "\n".join(
        header
        + [
            f"KL at initialisation: max|kld_per_t| = "
            f"{initialisation['kld_per_t_max_abs']:.3e}, max|source_kl_lag_map| = "
            f"{initialisation['source_kl_lag_map_max_abs']:.3e}",
            format_recovery(record),
        ]
    )
    print(report)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{RECORD_STEM}.txt").write_text(report + "\n", encoding="utf-8")
    (run_dir / f"{RECORD_STEM}.json").write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {run_dir / RECORD_STEM}.txt and .json")
    return 0 if record["recovered"] else 1


#: Values used for arguments absent from the command line -- i.e. an IDE's Run button. Keyed by
#: argparse ``dest``, and merged per key, so a flag overrides one value and leaves the rest of the
#: dict standing.
#:
#: Nothing here has to be filled in: the file runs as it stands and checks the conv-LSTM cell.
#: Point ``config`` at the transformer cell's planted config to check that parent instead, and use
#: ``override`` for a one-switch contrast at the same tree.
RUN_ARGS: Dict[str, Any] = {
    # A planted config from either feature-target causal cell.
    "config": "teb_vae/lag_attn_cfs/configs/planted.yaml",
    # 'check' fits and scores; 'manifest' builds and prints the state dict, and costs no fit.
    "mode": None,
    # dotted.key=value deltas, e.g. ['model_config.VAE_model.alibi_slope_scale=1.0'].
    "override": None,
    # Where a manifest is written; ignored by 'check', whose record goes into the run directory.
    "output_dir": None,
    # Device for the evaluation pass; None picks CUDA when it is there.
    "device": None,
}

#: Applied after the merge, so the key above stays reachable from the dict.
_DEFAULT_MODE = "check"


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    """Merge the command line over :data:`RUN_ARGS` and run the check.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        The process exit code.
    """
    values, sources = resolve_launch_args(build_parser(), RUN_ARGS, argv)
    refusal = missing_required(values, ("config",))
    if refusal is not None:
        raise SystemExit(refusal)
    # The shard paths inside a config are repo-root-relative, and under an IDE Run button the
    # working directory is whatever the IDE chose -- where a relative path resolves to nothing and
    # the loader dies as "No samples match the specified filters" with no mention of the real cause.
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        os.chdir(_REPO_ROOT)
    print(
        "resolved arguments: "
        + ", ".join(f"{key}={values[key]!r} (from {sources[key]})" for key in sorted(values))
    )
    return main(
        config=values["config"],
        mode=values["mode"] or _DEFAULT_MODE,
        override=values["override"],
        output_dir=values["output_dir"],
        device=values["device"],
    )


if __name__ == "__main__":
    sys.exit(_cli())
