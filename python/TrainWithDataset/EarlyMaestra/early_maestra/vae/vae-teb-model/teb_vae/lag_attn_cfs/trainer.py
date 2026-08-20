r"""The experiment driver: build the model from config, build the callbacks, run the fit.

Run from the repository root, which is what puts ``teb_vae``, ``train`` and ``utils`` on
``sys.path``:

.. code-block:: bash

    # Single GPU / local smoke
    python -m teb_vae.lag_attn_cfs.trainer --config teb_vae/lag_attn_cfs/configs/tiny.yaml

    # Prod box. The rank count must equal len(general_config.cuda_devices) -- default.yaml ships
    # 7 -- or Lightning rejects the device/world-size mismatch at Trainer construction. Export
    # the stamp so ranks 1..N-1 share rank 0's run directory.
    TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
        -m teb_vae.lag_attn_cfs.trainer --config teb_vae/lag_attn_cfs/configs/default.yaml

From an IDE's Run button, with no command line: ``RUN_CONFIG`` at the bottom of this file names
the config to use. Note a Run-button launch of ``default.yaml`` is a *single* process whose seven
``cuda_devices`` make Lightning spawn DDP workers underneath it; for a single-device smoke run
point ``RUN_CONFIG`` at ``configs/tiny.yaml``.

Almost everything is inherited. The config-to-constructor sweep, the four pre-flight guards, the
callback assembly, the resolved-config persistence, the DDP strategy selection and the diagnostic-
plot wiring are all model-independent and are reused rather than copied -- copies are how two models
that must stay comparable stop being comparable.

What this module supplies, beyond the five class attributes every cell of the grid re-points, is the
**warm-up budget**. ``causal_warmup_budget_steps`` names no constructor argument: it is a threshold,
and what the network takes is the four concrete channel tuples it resolves to against the configured
shards. Resolved in :meth:`LagAttnCfsTrainer._build_model_kwargs` rather than in ``create_model``, so
those tuples land in the ``model_kwargs`` written into every checkpoint -- the input adapters' widths
depend on them, so a checkpoint recording only the threshold could not be rebuilt without re-reading
the shards.

The pre-flight refusals are this driver's other half, and every one of them guards a failure that is
silent rather than loud: a two-sided shard whose coefficients contain their own future, a floor that
does not pair with the budget and therefore scores assumed pre-recording history as signal, a
``load_fields`` missing the two keys the tile phase is derived from, and a boundary shape term whose
formula is a slicing identity over *adjacent* anchors and has no meaning over a tiled set.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional, Tuple

#: Repository root: ``teb_vae/lag_attn_cfs/trainer.py`` -> up three.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# An IDE's Run button executes this file as a script, which puts *this directory* on sys.path
# rather than the repository root -- so the `teb_vae.` and `train.` imports below would fail with
# ModuleNotFoundError before __main__ is ever reached. Launching as
# `python -m teb_vae.lag_attn_cfs.trainer` from the repo root sets __package__ and needs none of
# this, which is why the insert is guarded rather than unconditional.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from loguru import logger  # noqa: E402

from teb_vae.lag_attn_cfs.causal_warmup import (  # noqa: E402
    BUDGET_KEY,
    SOURCE_BLOCKS,
    WarmupBudget,
    resolve_warmup_budget,
)
from teb_vae.lag_attn_cfs.model_kwargs import warmup_model_kwargs  # noqa: E402
from teb_vae.lag_attn_cfs.nets.causal_feature_target import (  # noqa: E402
    CausalFeatureForecastTarget,
)
from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs  # noqa: E402
from teb_vae.lag_attn_cfs.task import SeqVaeLagAttnCfsTask  # noqa: E402
from teb_vae.lag_attn_fs.trainer import _FORECAST_GAP_SUFFIXES  # noqa: E402
from teb_vae.lag_attn_rws.trainer import _TRACKED_METRICS, LagAttnRwsTrainer  # noqa: E402
from teb_vae.lag_attn_rws.trainer import main as run_training  # noqa: E402

#: The cross-signal phase-harmonic block. Absent from the causal variant by design -- a coefficient
#: mixing both signals would destroy the target-only / source-conditioned separation the whole
#: family rests on -- so a config naming it describes a dataset that does not exist.
FORBIDDEN_FIELD = "fhr_up_ph"

#: Loader fields the anchor tiling's phase is keyed on. ``load_fields`` is honoured literally, with
#: no forced additions, so dropping either leaves every segment of every recording on one tile grid
#: forever -- with no shape, no count and no metric differing.
PHASE_KEY_FIELDS: Tuple[str, ...] = ("guid", "epoch")

#: Metric suffixes this model emits and the two-sided feature sibling does not, on **both** stages.
#:
#: Two of the ten are geometry *guards* rather than results. ``target_warm_frac`` is a stamped
#: provenance column that must read exactly $1.0$: it is resolved at construction and the
#: constructor already refuses a violating budget-and-floor pairing, so any other value means the
#: checkpoint was built by code predating that refusal. ``anchors_per_sample`` must sit at its
#: geometry-derived value -- $[4, 5]$ at the shipped training tiling and exactly
#: $T_{\mathrm{valid}} - F$ at the dense evaluation stride -- and a value off that band means the
#: geometry broke rather than that the model learned something.
#:
#: The other eight are results. The two ``source_lag_warmth_frac`` columns size the compromise the
#: design makes on the source: lag attention searches back into a region where much of the source is
#: still inside its own warm-up, and the design keeps every source channel rather than gating them,
#: so the residual is measured instead of resolved. The three warm-up tertiles recompose to
#: ``pred_gap`` over the same denominator exactly as the block split does, cutting the channel axis
#: by filter speed rather than by stored block -- which the block split cannot, since both stored
#: blocks span nearly the same rebased warm-up range.
#:
#: The three novelty tertiles recompose to the same number over the same denominator, and cut the
#: same axis a third way: by $\nu_c$, the share of a scored coefficient drawn from raw samples the
#: anchor has not seen. That split is what makes "forecast" readable per channel rather than assumed
#: uniform -- the block score sums $H \cdot C_{\mathrm{keep}}$ coefficients whose novelty runs from
#: $1.000$ to $0.026$, and a good score on the low end is the model inverting its own delayed
#: history rather than predicting anything. Not a restatement of the warm-up split: a channel warm
#: across the whole window can still be almost entirely history at the horizon.
_CAUSAL_SUFFIXES: Tuple[str, ...] = (
    "target_warm_frac",
    "anchors_per_sample",
    "source_lag_warmth_frac_st",
    "source_lag_warmth_frac_ph",
    "pred_gap_warm_lo",
    "pred_gap_warm_mid",
    "pred_gap_warm_hi",
    "pred_gap_novel_lo",
    "pred_gap_novel_mid",
    "pred_gap_novel_hi",
)

#: The one readout that runs on validation batches only, beside the permutation control's three.
#:
#: It costs a source encode per step, and -- more importantly -- it is a *readout* that never enters
#: the objective. Tracking a ``train/`` variant would produce a column that is NaN in every row of
#: every run.
_CAUSAL_VAL_ONLY_SUFFIXES: Tuple[str, ...] = ("kld_source_null",)


class LagAttnCfsTrainer(LagAttnRwsTrainer):
    """Experiment driver for :class:`~teb_vae.lag_attn_cfs.nets.model.SeqVaeLagAttnCfs`."""

    MODEL_CLS = SeqVaeLagAttnCfs
    TASK_CLS = SeqVaeLagAttnCfsTask
    CHECKPOINT_STEM = "lag-attn-cfs"

    #: The loader fields this model's reconstruction target is built from, checked by the shared
    #: entry point's normalisation guard. Both blocks, because the target is their concatenation
    #: and a config carrying one of them is a target with a hole in it.
    TARGET_FIELDS: Tuple[str, ...] = ("fhr_st", "fhr_ph")

    #: The inherited metric surface, plus the two-sided sibling's four forecast-gap columns, plus
    #: this model's ten, plus the source-null KL on validation. An attribute rather than an
    #: override of ``train_model``, which is 75 lines of callback assembly whose copy would be free
    #: to drift from the one the comparison model runs under.
    TRACKED_METRICS: Tuple[str, ...] = (
        _TRACKED_METRICS
        + tuple(
            f"{stage}/{name}"
            for stage in ("train", "val")
            for name in _FORECAST_GAP_SUFFIXES + _CAUSAL_SUFFIXES
        )
        + tuple(f"val/{name}" for name in _CAUSAL_VAL_ONLY_SUFFIXES)
    )

    #: The resolved warm-up budget this run got, populated by :meth:`_build_model_kwargs` and read
    #: by :meth:`causal_standing_message` for the startup log. ``None`` means no budget is
    #: configured, which for this architecture is an ungated run rather than a default.
    resolved_warmup: Optional[WarmupBudget] = None

    # ------------------------------------------------------------------
    # Config -> constructor
    # ------------------------------------------------------------------
    def _build_model_kwargs(self) -> Dict[str, Any]:
        """Translate ``model_config.VAE_model`` into constructor kwargs, warm-up budget included.

        The inherited sweep forwards every key naming a real constructor argument and resolves
        ``causal_reach_budget_s``, which this family always ships as ``null`` -- the reach quantile
        is measured on the two-sided Morlet bank, which did not produce these coefficients, and the
        resolver refuses the two guards together. What is added here is the *warm-up* budget, whose
        four channel tuples the network takes and whose threshold names no constructor argument at
        all, so the inherited ``name in valid_parameters`` filter drops it for free.

        Returns:
            Constructor kwargs for the net.
        """
        model_kwargs = super()._build_model_kwargs()
        # Held on the instance so the startup log can state this run's causal standing without
        # resolving it a second time from a second read of the same config.
        self.resolved_warmup = resolve_warmup_budget(self.config)
        model_kwargs.update(warmup_model_kwargs(self.resolved_warmup, self.MODEL_CLS))
        return model_kwargs

    def causal_standing_message(self) -> str:
        """Return the one-line statement of this run's causal standing, for the startup log.

        The inherited sentence is **false** for this architecture and would be the most misleading
        line in the log: it says the stored two-sided features let step $t$ read up to $974$ s into
        its own future, which is exactly the property the causal variant removes. What is true here
        is the pair -- one-sidedness, which is a property of the transform, and the warm-up budget,
        which is what this run did about the region where one-sidedness costs something.

        Called from ``create_model`` *before* the network exists, so it is derived from the resolved
        budget rather than from ``self.pytorch_model``.

        Returns:
            The message, already formatted for a single ``logger.info`` call.
        """
        if self.resolved_warmup is None:
            return (
                "causal warm-up budget: none -- every stored channel is used at every step, "
                "including the leading region where a one-sided filter's output is a function of "
                "assumed pre-recording history rather than of the recording, on coefficients whose "
                "normalisation constants excluded exactly that region."
            )
        return (
            "one-sided features: a coefficient at step t is a function of {x(s) : s <= t} alone, "
            "so a forecast of step t + 1 + tau is a genuine forecast. " + self.resolved_warmup.summary()
        )

    def create_model(self) -> None:
        """Build the net through the inherited path, then log its geometry and hand it the seed.

        Three additions, all about things a run cannot recover afterwards.

        The **seed** is read once by the framework's own determinism setup and reaches no task in
        the family; the tile phase is derived from it, so a resumed run that did not know it would
        silently re-tile every segment. Applied through ``apply_config_hyperparameters``, which is
        the same route the learning rate takes and puts it in the checkpoint's hyperparameters.

        The **resolved geometry** is logged because nothing in the shipped code ties the decoder's
        horizon receptive field to the horizon, and nothing ties the anchor stride to it either: a
        config that shortened the horizon and left either behind would train a different model with
        every shape correct. Written into the run's own first lines rather than left to be
        re-derived from the config months later.

        The **resolved budget** is handed to the task, which is what lets the diagnostic callback
        draw the run-level warm-up figure. Its subject is the channels the budget *dropped* beside
        the ones it kept, and a dropped channel's own $W'_c$ is precisely what the checkpoint does
        not carry -- ``model_kwargs`` stamps the survivors' vector, because that is what the
        constructor needs -- so the figure cannot be built from the network alone.
        """
        super().create_model()
        model = self.pytorch_model
        stride = int(model.anchor_stride)
        floor = int(model.warmup_period)
        t_valid, horizon = int(model.geometry.t_valid), int(model.horizon)
        # ceil((T_valid - F) / S) is both the geometry constant every rank agrees on and the tile
        # count at phase 0, which is the most a sample can get; the last phase gets the fewest.
        a_max = -(-(t_valid - floor) // stride)
        fewest = -(-(t_valid - floor - (stride - 1)) // stride)
        receptive_field = _horizon_receptive_field(model)
        logger.info(
            f"resolved anchor geometry: horizon H={horizon}, stride S={stride}, "
            f"floor F={floor}, T_valid={t_valid}, A_max={a_max}, "
            f"tiles per sample {fewest}-{a_max} (phase 0 gets the most), "
            f"block width H*C_keep={horizon * int(model.decoder_out_channels)}, "
            f"horizon receptive field={receptive_field} tokens against H+1={horizon + 1}"
        )
        self.apply_config_hyperparameters(
            {"seed": (self.config.get("general_config") or {}).get("seed")}, self.pl_model
        )
        # A plain attribute rather than a hyperparameter: the four channel tuples the network needs
        # are already in the checkpoint's ``model_kwargs``, and a second copy of them under another
        # name would be a second thing to keep true across a resume.
        self.pl_model.warmup_budget = self.resolved_warmup

    # ------------------------------------------------------------------
    # Pre-flight
    # ------------------------------------------------------------------
    @classmethod
    def preflight(cls, config: Dict[str, Any]) -> None:
        """Refuse a launch this architecture cannot serve, before any run directory exists.

        Called by the shared entry point after its own four guards and before ``setup_config``, so
        a refusal here leaves no run directory, no log sink and no MLflow run behind.

        Every refusal below guards a failure whose symptom is a *number*, never an exception. The
        shared guards cover the loud ones; these cover a two-sided shard the objective would happily
        score, a floor that admits anchors whose target is pre-recording history, a boundary term
        whose formula is a slicing identity over adjacent anchors, a source stream missing the block
        that decides whether the start indicator is built at all, and the two loader fields the tile
        phase is keyed on.

        Args:
            config: The already-loaded resolved config dict, not a path.

        Raises:
            ValueError: Naming the offending key and, where there is one, the value to change it to.
        """
        _check_no_cross_channel_block(config)
        _check_boundary_term_is_off(config)
        _check_phase_key_fields(config)
        _check_source_block_kept(config)
        # Last, because it is the expensive one -- it opens every configured shard -- and because
        # its own refusals (a two-sided shard, disagreeing shards, a trim that does not produce the
        # declared window, a reach budget set alongside this one) are the ones a cheap check cannot
        # reach. The pairing is checked against what it returns.
        _check_floor_pairs_with_budget(config, resolve_warmup_budget(config))


def _horizon_receptive_field(model: Any) -> int:
    r"""How many horizon tokens one output token of the refine stack can see.

    Read off the **built** stack rather than off the config, because that is what the log line is
    for: ``horizon_depth`` and ``horizon_kernel`` are constructor arguments the net does not keep,
    and a config that left either at its default would otherwise be reported as whatever the config
    happened to say. The stack's dilations are $2^i$, so a depth-$d$ stack of kernel-$k$ convolutions
    spans

    $$\mathrm{RF} = 1 + (k - 1)\,(2^{d} - 1)$$

    tokens. It is logged beside the horizon because nothing in the shipped code ties the two
    together, and the family's own recorded criterion is $\mathrm{RF} \ge H + 1$.

    Args:
        model: The built network.

    Returns:
        The receptive field in horizon tokens, or $1$ if the stack cannot be read.
    """
    blocks = getattr(getattr(model.decoder.core, "refine", None), "blocks", None)
    if not blocks:
        return 1
    import torch.nn as nn

    kernels = [
        int(module.kernel_size[0])
        for module in blocks[0].modules()
        if isinstance(module, nn.Conv1d)
    ]
    return 1 + (max(kernels, default=1) - 1) * (2 ** len(blocks) - 1)


def _vae_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the ``model_config.VAE_model`` block, or an empty mapping."""
    return (config.get("model_config") or {}).get("VAE_model") or {}


def _check_no_cross_channel_block(config: Dict[str, Any]) -> None:
    """Refuse a config naming the cross-signal phase-harmonic block.

    A ``fhr_up_ph`` coefficient mixes both signals, which would put the source's own signal into the
    forecast target and destroy the target-only / source-conditioned separation the coupling readout
    rests on. The causal variant does not store it at all, so the loader would raise -- but only
    after every rank had initialised, and only for ``load_fields``; in ``normalize_fields`` it is
    silently ignored and reads as though the block were being handled.

    Args:
        config: The resolved run config.

    Raises:
        ValueError: Naming the list it appears in.
    """
    dataloader = (config.get("dataset_config") or {}).get("dataloader_config") or {}
    lists = {
        "dataset_config.dataloader_config.dataset_kwargs.load_fields": (
            dataloader.get("dataset_kwargs") or {}
        ).get("load_fields")
        or [],
        "dataset_config.dataloader_config.normalize_fields": dataloader.get("normalize_fields")
        or [],
    }
    offenders = [path for path, fields in lists.items() if FORBIDDEN_FIELD in fields]
    if offenders:
        raise ValueError(
            f"{FORBIDDEN_FIELD!r} appears in {' and '.join(offenders)}. That block mixes both "
            f"signals in one coefficient, so it would put the source's own signal into the "
            f"forecast target -- and the causal dataset variant does not store it at all. Remove "
            f"it from every list."
        )


def _check_boundary_term_is_off(config: Dict[str, Any]) -> None:
    r"""Refuse a non-zero ``lambda_boundary``, unconditionally.

    The boundary-continuity term is a slicing identity over **adjacent** anchors: it joins anchor
    $t$'s first forecast sample to anchor $t-1$'s last. This family always supplies an anchor set,
    whose entries are $S$ apart, so the term would join two windows separated by a whole horizon and
    call the discontinuity an error. The shared objective already raises on that combination -- this
    moves the failure to before the run directory exists rather than into the first training step.

    Args:
        config: The resolved run config.

    Raises:
        ValueError: Naming the key and the value.
    """
    weight = float(_vae_config(config).get("lambda_boundary", 0.0) or 0.0)
    if weight != 0.0:
        raise ValueError(
            f"model_config.VAE_model.lambda_boundary={weight} but this architecture always decodes "
            f"a tiled anchor set, and the boundary term is a slicing identity over *adjacent* "
            f"anchors -- at stride S it would join two windows a whole horizon apart and score the "
            f"gap between them as an error. Set lambda_boundary: 0.0."
        )


def _check_phase_key_fields(config: Dict[str, Any]) -> None:
    """Refuse a ``load_fields`` missing either key the tile phase is derived from.

    ``load_fields`` is honoured literally, with no forced additions. Without ``guid`` and ``epoch``
    the phase has nothing per-segment to key on, so every segment of every recording would be
    decoded at one tile grid forever -- and $A_{\\max}$ is a geometry constant either way, so no
    shape, no count and no metric would differ.

    Args:
        config: The resolved run config.

    Raises:
        ValueError: Naming the missing fields and the list.
    """
    dataloader = (config.get("dataset_config") or {}).get("dataloader_config") or {}
    load_fields = (dataloader.get("dataset_kwargs") or {}).get("load_fields") or []
    missing = [field for field in PHASE_KEY_FIELDS if field not in load_fields]
    if missing:
        raise ValueError(
            f"{missing} missing from dataset_config.dataloader_config.dataset_kwargs.load_fields. "
            f"The anchor tiling's per-segment phase is a stable hash of the recording identifier, "
            f"the segment's own start time, the epoch and the seed; without both keys every "
            f"segment is decoded at one tile grid forever and nothing about the run says so."
        )


def _check_source_block_kept(config: Dict[str, Any]) -> None:
    """Refuse ``use_up_st: false`` together with a warm-up budget.

    Dropping the first stored source block changes a *construction-time* decision rather than a
    width: the availability adapter builds its start indicator only when every channel of a stream
    waits at least one step, and ``up_st`` is the block that reaches zero. Without it the source's
    minimum warm-up is $41$ and the indicator comes into existence -- a parameter reached only by
    the leading steps of a segment, which under ``find_unused_parameters=False`` is a DDP hazard
    rather than a size change.

    Args:
        config: The resolved run config.

    Raises:
        ValueError: Naming both keys.
    """
    vae_config = _vae_config(config)
    if vae_config.get("causal_warmup_budget_steps") is None:
        return
    if not bool(vae_config.get("use_up_st", True)):
        raise ValueError(
            f"model_config.VAE_model.use_up_st=false with {BUDGET_KEY} set. Dropping "
            f"{SOURCE_BLOCKS[0]!r} leaves the source stream with a minimum warm-up above zero, "
            f"which flips the availability adapter's start embedding into existence -- a "
            f"construction-time change, not a width change, and one whose parameter is reached "
            f"only by the leading steps of a segment. Keep use_up_st: true, or run this arm "
            f"without a warm-up budget."
        )


def _check_floor_pairs_with_budget(
    config: Dict[str, Any], budget: Optional[WarmupBudget]
) -> None:
    r"""Refuse an anchor floor the kept target channels do not admit.

    The same two inequalities the constructor enforces, over the *survivors*, checked here so a
    mis-paired configuration fails before a run directory exists. The distinction between the
    configured threshold and the survivors' own maximum is load-bearing: a threshold of $151$ keeps
    the identical $98$ channels whose slowest still waits $134$ steps, so a floor derived from the
    threshold would sit $17$ steps too high and cost two tiles for nothing.

    The resolved **shifts** are passed as well as the warm-up, because the two requirements do not
    move together: the target tile is never shifted, so its validity half stays at $F \\ge B - 1$,
    while a shifted input stream must additionally have every kept channel warm at the anchor. An
    unaligned budget carries ``None`` there, which is the inert value.

    Args:
        config: The resolved run config.
        budget: The resolved warm-up budget, or ``None`` when none is configured.

    Raises:
        ValueError: Naming which requirement binds and both numbers, from the constructor's own
            check.
    """
    if budget is None:
        return
    CausalFeatureForecastTarget._check_anchor_floor(
        int(_vae_config(config)["warmup_period"]),
        budget.target.warmup_steps,
        budget.target.align_delays or (),
    )


def main(config_path: str) -> None:
    """Resolve the config, build everything, and run the fit.

    Delegates to the shared entry point with this package's driver. The pre-flight guards, the
    temporary resolved-config file, ``setup_config``'s ordering constraints and the resolved-config
    persistence are all model-independent, and a copy of them here would be free to drift from the
    ones the comparison model actually runs under.

    Args:
        config_path: Path to the YAML config. Its ``base:`` chain is resolved first.
    """
    run_training(config_path, trainer_cls=LagAttnCfsTrainer)


def _resolve_cli_config_path(config_path: str) -> str:
    """Resolve a command-line config path against the repository root.

    Every documented invocation runs from the repo root and uses repo-root-relative paths; an
    IDE's working directory is not something this module can rely on. Absolute paths pass through
    untouched.

    Args:
        config_path: The path as supplied on the command line or via ``RUN_CONFIG``.

    Returns:
        An absolute path.
    """
    if os.path.isabs(config_path):
        return config_path
    return os.path.join(_REPO_ROOT, config_path)


#: Config used when the module is launched with no ``--config`` -- i.e. an IDE's Run button.
#: ``--config`` on the command line always wins over this value. A relative path is resolved
#: against the repository root, not the working directory.
RUN_CONFIG: str | None = "teb_vae/lag_attn_cfs/configs/default.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the YAML config, e.g. teb_vae/lag_attn_cfs/configs/default.yaml. Run from "
        "the repo root. Optional only if RUN_CONFIG is set in this file (for an IDE Run button).",
    )
    _args = parser.parse_args()

    _config_path = _args.config or RUN_CONFIG
    if _config_path is None:
        parser.error(
            "--config is required. To launch from an IDE Run button instead, set RUN_CONFIG "
            "near the bottom of this file to a config path."
        )

    _config_path = _resolve_cli_config_path(_config_path)

    # The paths *inside* a config are repo-root-relative too (see configs/tiny.yaml), and under an
    # IDE Run button the working directory is whatever the IDE chose -- a relative shard path then
    # resolves to nothing and the loader dies as "No samples match the specified filters" with no
    # mention of the real cause.
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        logger.info(f"changing working directory to the repo root: {_REPO_ROOT}")
        os.chdir(_REPO_ROOT)

    if _args.config is None:
        logger.info(f"no --config given; using RUN_CONFIG={_config_path}")

    main(_config_path)
