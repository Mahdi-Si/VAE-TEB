r"""The experiment driver: build the model from config, build the callbacks, run the fit.

Run from the repository root, which is what puts ``teb_vae``, ``train`` and ``utils`` on
``sys.path``:

.. code-block:: bash

    # Single GPU / local smoke
    python -m teb_vae.lag_attn_transformer_e2e.trainer \
        --config teb_vae/lag_attn_transformer_e2e/configs/tiny.yaml

    # Prod box. The rank count must equal len(general_config.cuda_devices) -- default.yaml ships
    # 7 -- or Lightning rejects the device/world-size mismatch at Trainer construction. Export
    # the stamp so ranks 1..N-1 share rank 0's run directory.
    TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
        -m teb_vae.lag_attn_transformer_e2e.trainer \
        --config teb_vae/lag_attn_transformer_e2e/configs/default.yaml

From an IDE's Run button, with no command line: ``RUN_CONFIG`` at the bottom of this file names
the config to use. Note a Run-button launch of ``default.yaml`` is a *single* process whose seven
``cuda_devices`` make Lightning spawn DDP workers underneath it; for a single-device smoke run
point ``RUN_CONFIG`` at ``configs/tiny.yaml``.

Almost everything here is inherited, two levels deep: the config-to-constructor sweep, the callback
assembly, the resolved-config persistence, the DDP strategy selection and the four pre-flight
guards come from the raw-signal driver, and the step-granular learning-rate monitor from the
conv-Transformer one. What this module supplies is the three class attributes naming this
architecture, the two guards that are specific to reading raw signals, and the entry point.

The startup log is the fourth thing, and it is not decoration. The inherited driver states that
"input features at step t read up to 974 s into their own future, so the source-conditioned KL is
not a transfer entropy" -- which is the negation of this package's central claim, and would
otherwise appear in every one of its production runs.
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys
from typing import Any, Dict, Optional

#: Repository root: ``teb_vae/lag_attn_transformer_e2e/trainer.py`` -> up three.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# An IDE's Run button executes this file as a script, which puts *this directory* on sys.path
# rather than the repository root -- so the `teb_vae.` and `train.` imports below would fail
# with ModuleNotFoundError before __main__ is ever reached. Launching as
# `python -m teb_vae.lag_attn_transformer_e2e.trainer` from the repo root sets __package__ and
# needs none of this, which is why the insert is guarded rather than unconditional.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from loguru import logger  # noqa: E402

from teb_vae.lag_attn_rws.nets.lag_report import SECONDS_PER_STEP  # noqa: E402
from teb_vae.lag_attn_rws.trainer import main as run_training  # noqa: E402
from teb_vae.lag_attn_transformer_e2e.nets.frontend import CausalRawFrontend  # noqa: E402
from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E  # noqa: E402
from teb_vae.lag_attn_transformer_e2e.task import SeqVaeLagAttnTrfE2ETask  # noqa: E402
from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer  # noqa: E402

#: Raw samples per minute of recording. The loader trims ``int(4 * 60 * trim_minutes)`` samples
#: from **each** end of every raw field (see ``hdf5_dataset/hdf5_dataset.py``), and the shard-side
#: pre-flight below has to reproduce that arithmetic to compare a *stored* length against the
#: *trimmed* one the model is built for.
RAW_SAMPLES_PER_MINUTE = 4 * 60

#: ``model_config.VAE_model`` keys that describe the input representation this model replaces,
#: mapped to what took each one's place. Every one of them names no constructor argument here, so
#: the inherited signature sweep would drop it without a word and the run would be a different one
#: from the one the operator believes they launched -- most damagingly ``causal_reach_budget_s``,
#: whose whole purpose was to bound a leak that no longer exists.
INERT_MODEL_KEYS: Dict[str, str] = {
    "c_y": "there is no stored target feature block to declare a width for; the target front end "
           "consumes the raw `fhr` directly",
    "c_u": "there is no stored source feature block to declare a width for; the source front end "
           "consumes the raw `up` directly",
    "use_up_st": "there is no source scattering block to include or exclude",
    "causal_reach_budget_s": "the front ends are strictly one-sided by construction, so there is "
                             "no forward reach to prune channels against; the reach that IS "
                             "budgeted is the front end's backward reach, which is "
                             "warmup_period * raw_per_step and is derived rather than configured",
    "target_keep_index": "a channel-selection tuple of the pruned-feature guard, which this model "
                         "has no channels to select from",
    "target_delays": "a per-channel delay tuple of the pruned-feature guard; a raw front end has "
                     "nothing to compensate for and reads the stream at delay 0",
    "source_keep_index": "a channel-selection tuple of the pruned-feature guard, which this model "
                         "has no channels to select from",
    "source_delays": "a per-channel delay tuple of the pruned-feature guard; a raw front end has "
                     "nothing to compensate for and reads the stream at delay 0",
}


class LagAttnTrfE2ETrainer(LagAttnTrfRwsTrainer):
    """Experiment driver for
    :class:`~teb_vae.lag_attn_transformer_e2e.nets.model.SeqVaeLagAttnTrfE2E`."""

    MODEL_CLS = SeqVaeLagAttnTrfE2E
    TASK_CLS = SeqVaeLagAttnTrfE2ETask
    CHECKPOINT_STEM = "lag-attn-trf-e2e"

    @classmethod
    def preflight(cls, config: Dict[str, Any]) -> None:
        """Refuse a launch this architecture cannot serve, before anything is built.

        Three checks, in the order a wrong config is most likely to be wrong. Every one of them
        guards a failure that is otherwise **silent**: a key the signature sweep drops, a field the
        loader hands over unnormalized, or a shard whose raw length disagrees with the geometry
        only inside the first forward.

        The disposition of the four inherited module-level guards, which ``main`` calls by name
        before this hook and which are unchanged:

        * ``_check_stat_path`` applies unchanged -- more forcefully here than anywhere, since both
          model inputs *and* the reconstruction target are normalized fields.
        * ``_check_raw_target_normalized`` applies unchanged and covers ``fhr`` in both lists, so
          only ``up`` is new below.
        * ``_check_declared_widths_against_shard`` self-disables: it returns early unless the
          config carries ``c_y`` and ``c_u``, and a config carrying either is refused here anyway.
          A copy-pasted sibling config whose widths are *correct* therefore passes that guard in
          silence and gets the inert-key message from this one.
        * ``_check_causal_budget_resolves`` is a no-op: it resolves ``causal_reach_budget_s``,
          which is absent from this package's configs and refused here when present.

        Args:
            config: The already-loaded resolved config dict, not a path.

        Raises:
            ValueError: Naming the offending key or field, and what replaced it.
        """
        _check_no_inert_model_keys(config)
        _check_raw_source_normalized(config)
        _check_raw_length_against_shard(config)

    def causal_standing_message(self) -> str:
        r"""Return this run's causal standing, measured on the front end it is about to build.

        The inherited sentence is false here and is the negation of what this package exists to
        claim, so it is replaced rather than supplemented. What goes in its place is a *number*:
        the front end's own accumulated reach, in raw samples and seconds, against the budget
        ``warmup_period * raw_per_step`` that keeps a trained anchor off the zero-padded
        convolution transient at the segment's start.

        The base calls this **before** ``self.pytorch_model`` is assigned -- deliberately, so a
        launch that dies in the constructor has still said what it was about to build -- so the
        reach cannot be read off the built model. It is measured on a throwaway front end built
        from the same resolved kwargs instead, which is the route that recorded decision names. A
        test pins the logged number against ``pytorch_model.target_frontend.reach_samples`` after a
        real ``create_model``, so the two cannot drift into disagreement.

        Returns:
            The message, already formatted for a single ``logger.info`` call.
        """
        frontend = self._reference_frontend()
        seconds_per_sample = SECONDS_PER_STEP / frontend.total_stride
        return (
            f"causal standing: raw-signal inputs through strictly one-sided front ends -- the "
            f"history state at step t is a function of raw samples at index <= "
            f"{frontend.total_stride}t + {frontend.total_stride - 1} and no further, so the "
            f"source-conditioned KL reads no source content recorded after its own anchor. "
            f"Front-end reach {frontend.reach_samples} raw samples "
            f"({frontend.reach_samples * seconds_per_sample:.1f} s) against a budget of "
            f"{frontend.reach_budget} "
            f"({frontend.reach_budget * seconds_per_sample:.1f} s)."
        )

    def _reference_frontend(self) -> CausalRawFrontend:
        """Build one front end at the settings this launch's model will build its two at.

        The four settings are derived from the constructor kwargs exactly as the net derives them,
        with anything the config leaves unset taken from the constructor's own defaults rather than
        from a second table here -- a defaults table is the thing that goes stale when a default
        moves.

        Returns:
            A front end at this run's geometry. Discarded by the caller; it exists to be measured.
        """
        signature = inspect.signature(self.MODEL_CLS.__init__).parameters
        settings: Dict[str, Any] = {
            name: parameter.default
            for name, parameter in signature.items()
            if parameter.default is not inspect.Parameter.empty
        }
        settings.update(self._build_model_kwargs())

        raw_per_step = int(settings["raw_per_step"])
        return CausalRawFrontend(
            d_model=int(settings["d_model"]),
            raw_per_step=raw_per_step,
            # The same derivation the constructor makes, and the reason it is a derivation rather
            # than a key: the budget is a fact about the geometry, not a preference.
            reach_budget=int(settings["warmup_period"]) * raw_per_step,
            kernels=tuple(settings["frontend_kernels"]),
        )


def _check_no_inert_model_keys(config: Dict[str, Any]) -> None:
    """Refuse a ``VAE_model`` block carrying a key of the input representation this model replaces.

    The inherited sweep forwards ``VAE_model`` keys by name against the real constructor signature,
    so a key that names nothing cannot crash a launch -- it is dropped in silence, and the run is
    not the one the operator configured. Every key below is one an operator copying the sibling's
    config would carry across.

    Args:
        config: The resolved run config.

    Raises:
        ValueError: Naming each offending key and what replaced it.
    """
    vae_config = (config.get("model_config", {}) or {}).get("VAE_model", {}) or {}
    offenders = [name for name in INERT_MODEL_KEYS if name in vae_config]
    if not offenders:
        return
    detail = "; ".join(f"{name}: {INERT_MODEL_KEYS[name]}" for name in offenders)
    raise ValueError(
        f"model_config.VAE_model carries {len(offenders)} key(s) that name nothing in this "
        f"architecture and would be dropped in silence by the constructor sweep -- {detail}. "
        f"Remove them: this model reads the raw `fhr` and `up` signals through two learned causal "
        f"front ends, and has no stored feature blocks to declare, select or delay."
    )


def _check_raw_source_normalized(config: Dict[str, Any]) -> None:
    """Refuse to start unless the raw UP source is loaded *and* normalized.

    The mirror of the inherited ``_check_raw_target_normalized``, for the field that is new here.
    Without ``'up'`` in ``load_fields`` the task fails on the first batch -- late, after every rank
    has initialised. Without it in ``normalize_fields`` nothing fails at all: the source arrives in
    raw contraction units, the front end (which owns no statistics of its own, by design) feeds
    that scale straight into the source encoder, and every source-side readout the model produces
    is measured at an operating point nobody chose.

    Args:
        config: The resolved run config.

    Raises:
        ValueError: Naming the offending config key.
    """
    dataloader_config = (config.get("dataset_config", {}) or {}).get(
        "dataloader_config", {}
    ) or {}
    load_fields = (dataloader_config.get("dataset_kwargs", {}) or {}).get("load_fields") or []
    normalize_fields = dataloader_config.get("normalize_fields") or []

    missing = []
    if "up" not in load_fields:
        missing.append("dataset_config.dataloader_config.dataset_kwargs.load_fields")
    if "up" not in normalize_fields:
        missing.append("dataset_config.dataloader_config.normalize_fields")
    if missing:
        raise ValueError(
            "'up' must be listed in " + " and in ".join(missing) + ": the raw UP is this model's "
            "source stream, and the front end that reads it consumes what the loader produced "
            "rather than standardising anything itself -- so an unnormalized source shifts every "
            "coupling number the run reports, with nothing else raising."
        )


def _check_raw_length_against_shard(config: Dict[str, Any]) -> None:
    r"""Compare the first training shard's *trimmed* raw length against the model's geometry.

    Shards store the untrimmed geometry -- ``fhr`` and ``up`` at $5280$ samples, ``weight`` at
    $330$ steps -- while the model is built for the trimmed one, $4800$ and $300$. So the
    comparison has to apply the configured ``trim_minutes`` first; comparing the stored length
    directly would fail on every real shard.

    The task already checks this against every real batch, and the net's own forward guard is the
    authoritative one. This exists only to move the failure earlier: without it a ``trim_minutes``
    that disagrees with ``sequence_length`` surfaces inside the first ``training_step``, by which
    point every rank has initialised and the run directory and MLflow run exist.

    Deliberately not fatal on anything but a genuine mismatch: a missing file, a missing field or
    an unreadable shard is left to the data module, which reports those far better than a
    pre-flight peek can.

    Args:
        config: The resolved run config.

    Raises:
        ValueError: Naming the stored length, the trim and the trimmed length that was expected.
    """
    dataset_config = config.get("dataset_config", {}) or {}
    shards = dataset_config.get("vae_train_datasets") or []
    dataloader_config = dataset_config.get("dataloader_config", {}) or {}
    dataset_kwargs = dataloader_config.get("dataset_kwargs", {}) or {}
    vae_config = (config.get("model_config", {}) or {}).get("VAE_model", {}) or {}
    if not shards or "sequence_length" not in vae_config or "raw_per_step" not in vae_config:
        return

    try:
        import h5py

        with h5py.File(str(shards[0]), "r") as handle:
            # Stored layout is (N, L) for the raw fields; the loader takes one sample and trims it.
            stored_raw = int(handle["fhr"].shape[1])
    except Exception:  # noqa: BLE001 - the data module reports these properly; see docstring.
        return

    trim_minutes = dataset_kwargs.get("trim_minutes")
    # The loader's own arithmetic, reproduced: int(4 * 60 * trim_minutes) samples off EACH end.
    trim_samples = 0 if trim_minutes is None else int(RAW_SAMPLES_PER_MINUTE * float(trim_minutes))
    trimmed_raw = stored_raw - 2 * trim_samples
    expected = int(vae_config["sequence_length"]) * int(vae_config["raw_per_step"])
    if trimmed_raw != expected:
        raise ValueError(
            f"{shards[0]} stores {stored_raw} raw samples per segment; at "
            f"trim_minutes={trim_minutes} the loader hands the model "
            f"{stored_raw} - 2*{trim_samples} = {trimmed_raw}, but the model is built for "
            f"sequence_length {vae_config['sequence_length']} * raw_per_step "
            f"{vae_config['raw_per_step']} = {expected}. Either fix "
            f"dataset_config.dataloader_config.dataset_kwargs.trim_minutes (it must also match the "
            f"stats file's), or point model_config.VAE_model at the geometry these shards carry."
        )


def main(config_path: str) -> None:
    """Resolve the config, build everything, and run the fit.

    Delegates to the shared entry point with this package's driver. The pre-flight guards, the
    temporary resolved-config file, ``setup_config``'s ordering constraints and the
    resolved-config persistence are all model-independent, and a copy of them here would be free
    to drift from the ones the comparison model actually runs under. What this package adds of its
    own goes in :meth:`LagAttnTrfE2ETrainer.preflight`, which that entry point calls after the four
    inherited guards and before ``setup_config``.

    Args:
        config_path: Path to the YAML config. Its ``base:`` chain is resolved first.
    """
    run_training(config_path, trainer_cls=LagAttnTrfE2ETrainer)


def _resolve_cli_config_path(config_path: str) -> str:
    """Resolve a command-line config path against the repository root.

    Every documented invocation runs from the repo root and uses repo-root-relative paths; an
    IDE's working directory is not something this module can rely on. Absolute paths pass
    through untouched.

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
RUN_CONFIG: Optional[str] = "teb_vae/lag_attn_transformer_e2e/configs/default.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the YAML config, e.g. "
        "teb_vae/lag_attn_transformer_e2e/configs/default.yaml. Run from the repo root. "
        "Optional only if RUN_CONFIG is set in this file (for an IDE Run button).",
    )
    _args = parser.parse_args()

    _config_path = _args.config or RUN_CONFIG
    if _config_path is None:
        parser.error(
            "--config is required. To launch from an IDE Run button instead, set RUN_CONFIG "
            "near the bottom of this file to a config path."
        )

    _config_path = _resolve_cli_config_path(_config_path)

    # The paths *inside* a config are repo-root-relative too (see configs/tiny.yaml), and under
    # an IDE Run button the working directory is whatever the IDE chose -- a relative shard path
    # then resolves to nothing and the loader dies as "No samples match the specified filters"
    # with no mention of the real cause.
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        logger.info(f"changing working directory to the repo root: {_REPO_ROOT}")
        os.chdir(_REPO_ROOT)

    if _args.config is None:
        logger.info(f"no --config given; using RUN_CONFIG={_config_path}")

    main(_config_path)
