r"""Run the lag-attn-v1 testing pipeline on a synthetic-TE checkpoint.

This runner is the bridge between the v2 synthetic-TE harness in this
package and the broad model-diagnostic pipeline at
``model/vae_teb_prediction/testing/``. It loads a ``SeqVaeLagAttnV1``
checkpoint trained on a cached synthetic split, builds a
:class:`SyntheticTEDataset` DataLoader, and invokes
:func:`testing.run_tests.run_full_test_pipeline` via the new
``loader_override`` kwarg -- so every per-analysis output (histograms,
forecast quality, attention diagnostics, KL-PCA, frequency-band
forecast, residual usage, lag attribution, ...) is produced without any
HDF5 plumbing.

Per V2-D8 (``model_validation_v2_plan.md``) this file exposes **both** a
CLI and an edit-and-run ``RUN_CONFIG`` dict. The dispatch is automatic:
if any ``sys.argv[1:]`` is present the CLI parser owns the run;
otherwise the ``RUN_CONFIG`` dict at the bottom is consumed.

Selection model
---------------

Checkpoint -- pass exactly one of:

* ``--checkpoint-path /abs/path/to/final.ckpt`` -- recommended for any
  checkpoint that does not sit at the canonical
  ``<results_root>/<benchmark>/<run_tag>/<filename>`` location (sweep
  cells, beta-sweep outputs, calibration cells, custom backup paths,
  ...).
* ``--run-tag <tag> [--checkpoint final.ckpt]`` -- resolves to
  ``<results_root>/<benchmark>/<run_tag>/<checkpoint>``.

Dataset -- in priority order:

1. ``--data-npz /abs/path/to/test.npz`` -- explicit, no lookup.
2. ``--data-tag <tag>`` -- resolves to
   ``<data_root>/<benchmark>/<data_tag>/test.npz``.
3. Auto-resolved from the checkpoint's embedded ``data_meta``
   (``train_minimal.py`` writes the training split's ``meta.json`` into
   every checkpoint, so the matching cache is reachable without any
   user input). This branch only fires when ``--checkpoint-path`` is
   used; the tag-based branch always requires an explicit ``--data-tag``.

The checkpoint and dataset are **decoupled by design**: a single
checkpoint can be evaluated against any compatible cache (e.g. a
G1_baseline model on the G1_twoband test split for cross-evaluation).

Output layout -- in priority order:

1. ``--output-dir <dir>`` (explicit; final path is
   ``<dir>/testing_pipeline/<output_tag>/`` -- wait, see below).

   Specifically: when you pass ``--output-dir``, the runner uses
   ``<output_dir>/<output_tag>/`` as the actual output location, so the
   ``<output_tag>`` always appears as a leaf so multiple runs against
   the same checkpoint stay siblings.
2. ``--checkpoint-path`` is set ->
   ``<ckpt.parent>/testing_pipeline/<output_tag>/`` (outputs land next
   to the checkpoint -- exactly the right behaviour for sweep cells).
3. Tag-based ->
   ``<results_root>/<benchmark>/<run_tag>/testing_pipeline/<output_tag>/``.

The ``<output_tag>`` is **required** so multiple test runs on the same
checkpoint (e.g. ``final.ckpt`` vs ``best.ckpt``, or different
``max_samples`` caps) coexist as siblings rather than overwriting.

Analyses skipped on synthetic data (per the synthetic-data contract -- no
raw ``fhr``/``up`` signals, no clinical labels, no per-epoch metadata):

* ``trajectory`` -- requires ``guid`` + ``epoch`` for hours-before-birth axis.
* ``causal_te_validation`` -- requires raw ``fhr``/``up`` for event detection.
* ``class_separation`` / ``per_class_breakdown`` -- auto-skipped by
  ``single_class_mode=True`` because synthetic data carries no class label.

Examples:
    Edit ``RUN_CONFIG`` then::

        python -m model.vae_teb_prediction.model.model_experiment.synthetic.run_pipeline_tests

    CLI -- tag-based (canonical layout)::

        python -m ...synthetic.run_pipeline_tests \
            --benchmark G1 --run-tag G1_baseline --checkpoint final.ckpt \
            --data-tag G1_baseline --output-tag smoke --max-samples 200

    CLI -- direct paths (sweep cells, deep trees)::

        python -m ...synthetic.run_pipeline_tests \
            --checkpoint-path results/G1/beta_sweep/beta_1e-3/cell_a/final.ckpt \
            --output-tag beta_1e3_cellA
        # data_tag + benchmark auto-resolved from the checkpoint's data_meta;
        # outputs land in results/G1/beta_sweep/beta_1e-3/cell_a/testing_pipeline/beta_1e3_cellA/.

    CLI -- cross-evaluate (train on G1_baseline, test on G1_twoband)::

        python -m ...synthetic.run_pipeline_tests \
            --checkpoint-path results/G1/G1_baseline/best.ckpt \
            --data-tag G1_twoband --output-tag cross_eval_twoband
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from loguru import logger

# Project root on sys.path so absolute imports work when this file is
# invoked as a module from any working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model.vae_teb_prediction.model.model_experiment.synthetic import (
    train_minimal as tm,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.dataset import (
    SyntheticTEDataset,
    make_dataloader,
)
from model.vae_teb_prediction.testing.run_tests import run_full_test_pipeline


_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _PKG_DIR / "config_synth.yaml"


# =============================================================================
# Config / path helpers
# =============================================================================


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Parse a YAML file into a dict (empty dict if the file is empty)."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _data_root(synth_config: Dict[str, Any]) -> Path:
    """Resolve the data cache root from ``paths.data_dir`` (mirrors evaluate_te.py).
    """
    return tm.resolve_user_path(synth_config["paths"]["data_dir"])


def _results_root(synth_config: Dict[str, Any]) -> Path:
    """Resolve the results root from ``paths.results_dir``."""
    return tm.resolve_user_path(synth_config["paths"]["results_dir"])


def _locate_checkpoint(
    results_root: Path, benchmark: str, run_tag: str, checkpoint_filename: str
) -> Path:
    """Find a checkpoint at ``<results_root>/<benchmark>/<run_tag>/<filename>``.

    Args:
        results_root: Resolved synthetic results root.
        benchmark: Benchmark identifier (e.g. ``G1``).
        run_tag: Per-run subdirectory (e.g. ``G1_baseline``).
        checkpoint_filename: Either ``final.ckpt`` or ``best.ckpt`` (or a
            custom filename).

    Returns:
        Absolute path of the checkpoint.

    Raises:
        FileNotFoundError: If the checkpoint does not exist on disk.
    """
    ckpt_path = results_root / str(benchmark) / str(run_tag) / str(checkpoint_filename)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {ckpt_path}\n"
            f"Train one first, e.g.:\n"
            f"  python -m model.vae_teb_prediction.model.model_experiment."
            f"synthetic.train_minimal --tag {run_tag}"
        )
    return ckpt_path


def _locate_test_split(
    data_root: Path, benchmark: str, data_tag: str
) -> Path:
    """Locate ``<data_root>/<benchmark>/<data_tag>/test.npz``.

    Args:
        data_root: Resolved data cache root.
        benchmark: Benchmark identifier.
        data_tag: Cache subdirectory tag.

    Returns:
        Absolute path of ``test.npz``.

    Raises:
        FileNotFoundError: If the cached split is missing.
    """
    test_npz = data_root / str(benchmark) / str(data_tag) / "test.npz"
    if not test_npz.is_file():
        raise FileNotFoundError(
            f"cached test split not found: {test_npz}\n"
            f"Build it first, e.g.:\n"
            f"  python -m model.vae_teb_prediction.model.model_experiment."
            f"synthetic.build_dataset --tag {data_tag}"
        )
    return test_npz


def _load_checkpoint_meta(ckpt_path: Path) -> Dict[str, Any]:
    r"""Load the full ``train_minimal`` checkpoint dict (architecture + provenance).

    ``train_minimal.py`` saves checkpoints as a plain ``torch.save({...})``
    dict containing:

    * ``model_state_dict`` -- the weights;
    * ``model_kwargs``     -- exact constructor args for
      :class:`SeqVaeLagAttnV1`;
    * ``data_meta``        -- a copy of the training split's ``meta.json``
      including ``tag``, ``benchmark``, ``te_true``, ``true_lag_band``;
    * ``config``           -- the full effective post-override config;
    * ``loss_settings``, ``epoch``, ``val_total_loss``, ``train_metrics``,
      ``latent_stats_fitted``, ``torch_version``, ``created``.

    The downstream auto-resolution code reads ``data_meta`` to pick the
    matching dataset when the caller does not supply ``--data-tag`` /
    ``--benchmark`` explicitly.

    Args:
        ckpt_path: Path to ``final.ckpt`` / ``best.ckpt`` (or any other
            file written by :func:`train_minimal.train`).

    Returns:
        The full checkpoint dict.

    Raises:
        KeyError: If the checkpoint does not carry ``model_kwargs``.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_kwargs" not in ckpt:
        raise KeyError(
            f"{ckpt_path} is not a train_minimal checkpoint -- "
            f"`model_kwargs` is missing. Available keys: {sorted(ckpt)}"
        )
    return ckpt


def _checkpoint_model_kwargs(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """Return the ``model_kwargs`` constructor args from a loaded checkpoint."""
    return dict(ckpt["model_kwargs"])


def _synth_to_testing_config(
    model_kwargs: Dict[str, Any],
    *,
    tag: str,
    out_dir_base: Path,
    batch_size: int,
) -> Dict[str, Any]:
    r"""Translate synthetic ``model_kwargs`` into a testing-pipeline YAML config.

    :func:`testing.base._lag_attn_kwargs_from_config` reads model
    constructor arguments from ``model_config.VAE_model.*``. The synthetic
    package keeps them flat under ``model:`` (and ``train_minimal``
    persists them flat under ``model_kwargs`` in the checkpoint). This
    helper builds an equivalent nested dict so the testing pipeline can
    consume it without modification.

    Args:
        model_kwargs: The ``model_kwargs`` dict pulled from the
            checkpoint (single source of truth for architecture).
        tag: Experiment tag stamped under ``general_config.tag``.
        out_dir_base: Base output directory used only as a fallback when
            ``output_dir`` is not passed to ``run_full_test_pipeline``.
        batch_size: Test-time batch size to record under
            ``general_config.batch_size.test``.

    Returns:
        Dict ready to be ``yaml.safe_dump`` -- shape matches the standard
        v1 testing YAML schema for the keys the pipeline reads.
    """
    return {
        "general_config": {
            "tag": tag,
            "folders_config": {"out_dir_base": str(out_dir_base)},
            "batch_size": {"test": int(batch_size)},
        },
        "model_config": {"VAE_model": dict(model_kwargs)},
        "dataset_config": {
            "stat_path": None,
            "vae_test_datasets": [],
            "dataloader_config": {"num_workers": 0},
        },
    }


# =============================================================================
# Main runner
# =============================================================================


def run_synthetic_pipeline_tests(
    *,
    output_tag: str,
    # ----- Checkpoint selection: pass *either* checkpoint_path *or* the
    # (run_tag, checkpoint_filename) pair (with benchmark either explicit
    # or read from the config). -----
    checkpoint_path: Optional[Path] = None,
    run_tag: Optional[str] = None,
    checkpoint_filename: str = "final.ckpt",
    # ----- Dataset selection: pass *either* data_npz *or* data_tag (the
    # latter resolves to data/<benchmark>/<data_tag>/test.npz). When both
    # are omitted *and* checkpoint_path is supplied, the dataset is
    # auto-resolved from the checkpoint's embedded `data_meta`. -----
    data_npz: Optional[Path] = None,
    data_tag: Optional[str] = None,
    benchmark: Optional[str] = None,
    # ----- Where outputs land: explicit > <checkpoint_dir>/testing_pipeline/<output_tag>/ > <results_root>/<benchmark>/<run_tag>/testing_pipeline/<output_tag>/.
    output_dir: Optional[Path] = None,
    # ----- Misc -----
    config_path: Optional[Path] = None,
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
    analysis_samples: int = 10,
    batch_size: Optional[int] = None,
    skip_up_effect: bool = False,
    skip_frequency_band: bool = False,
    skip_attention: bool = False,
    skip_forecast_heatmaps: bool = False,
    skip_kld_pca: bool = False,
    skip_interactive: bool = False,
) -> Dict[str, Any]:
    r"""Top-level driver: run the testing pipeline on a synthetic checkpoint.

    Two equivalent ways to point at a checkpoint:

    1. **Direct path** -- pass ``checkpoint_path=<abs/path/to/final.ckpt>``.
       This is the right choice for deeply-nested sweep cells like
       ``results/G1/beta_sweep/beta_1e-3/cell_a/final.ckpt`` that the
       ``<run_tag>/<filename>`` convention cannot address.
    2. **Tag-based** -- pass ``run_tag`` (with ``benchmark`` either
       explicit or read from ``experiment.benchmark`` in the config) and
       ``checkpoint_filename`` (defaults to ``final.ckpt``). The
       checkpoint lands at
       ``<results_root>/<benchmark>/<run_tag>/<checkpoint_filename>``.

    Dataset resolution mirrors the same flexibility:

    * If ``data_npz`` is supplied, that ``.npz`` is loaded directly.
    * Otherwise if ``data_tag`` is supplied, the dataset is
      ``data/<benchmark>/<data_tag>/test.npz``.
    * Otherwise, when ``checkpoint_path`` is supplied, the dataset is
      auto-resolved from the checkpoint's embedded ``data_meta`` --
      ``train_minimal`` saves the training split's ``meta.json`` inside
      every checkpoint, so ``data_meta.benchmark`` and ``data_meta.tag``
      reconstruct the matching cache path.

    Note that the checkpoint and dataset are **decoupled by design**:
    you can cross-evaluate a G1_baseline checkpoint against the
    G1_twoband test split just by passing ``data_tag="G1_twoband"``.

    Builds a :class:`SyntheticTEDataset`, translates the checkpoint's
    ``model_kwargs`` into a testing-pipeline YAML, then delegates to
    :func:`run_full_test_pipeline` via ``loader_override``. After the
    pipeline returns it writes a ``synthetic_ground_truth.json`` sidecar
    pairing the run's outputs with the dataset's analytic
    ``te_true`` / ``true_lag_band``.

    Args:
        output_tag: Folder name under ``testing_pipeline/`` (required so
            multiple runs against the same checkpoint coexist as siblings).
        checkpoint_path: Absolute path to the checkpoint. Mutually
            exclusive with ``run_tag``.
        run_tag: Training-run subdirectory under
            ``<results_root>/<benchmark>/``. Mutually exclusive with
            ``checkpoint_path``.
        checkpoint_filename: Filename inside the run directory
            (only consulted when ``run_tag`` is used).
        data_npz: Absolute path to a ``test.npz`` cache split.
        data_tag: Cache tag (alternative to ``data_npz``).
        benchmark: Benchmark identifier (e.g. ``G1`` / ``G2`` / ``G3``).
            When omitted, derived from the checkpoint's ``data_meta``
            (if ``checkpoint_path`` is used) or from
            ``experiment.benchmark`` in ``config_synth.yaml`` (if
            ``run_tag`` is used).
        output_dir: Absolute output directory. Defaults to:
            (a) ``<ckpt.parent>/testing_pipeline/<output_tag>/`` when
            ``checkpoint_path`` is used, or
            (b) ``<results_root>/<benchmark>/<run_tag>/testing_pipeline/<output_tag>/``
            when ``run_tag`` is used.
        config_path: Path to ``config_synth.yaml`` (defaults to the
            packaged one). Used only when ``run_tag`` / ``data_tag``
            need ``paths.data_dir`` / ``paths.results_dir`` resolution.
        device: Torch device override (``cuda:0``, ``cpu``, ``auto``).
        max_samples: Cap for aggregate analyses. ``None`` iterates the
            whole test split.
        analysis_samples: Number of per-sample diagnostic PDFs to emit.
        batch_size: Test-time batch size. ``None`` reuses
            ``optim.batch_size`` from the synthetic config (default 32).
        skip_up_effect: Skip the UP-ablation analysis.
        skip_frequency_band: Skip the frequency-band-stratified forecast
            analysis.
        skip_attention: Skip attention + te_lag analyses.
        skip_forecast_heatmaps: Skip per-sample diagnostic PDFs.
        skip_kld_pca: Skip the per-dim KL PCA analysis.
        skip_interactive: Skip Plotly interactive plots.

    Returns:
        The result dict from :func:`run_full_test_pipeline`.

    Raises:
        ValueError: If neither ``checkpoint_path`` nor ``run_tag`` is
            supplied, or if dataset resolution fails after exhausting
            ``data_npz`` / ``data_tag`` / ``data_meta`` fallbacks.
        FileNotFoundError: If the checkpoint or test split is missing.
    """
    if checkpoint_path is None and run_tag is None:
        raise ValueError(
            "Must supply either `checkpoint_path` (absolute path) or "
            "`run_tag` (subdirectory under results/<benchmark>/)."
        )

    cfg_path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    synth_config = _load_yaml(cfg_path)

    data_root = _data_root(synth_config)
    results_root = _results_root(synth_config)

    # --- Resolve checkpoint ----------------------------------------------
    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path).resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    else:
        # `benchmark` may still be None here -- fill from the config
        # default before locating the checkpoint.
        resolved_benchmark = benchmark or str(
            (synth_config.get("experiment") or {}).get("benchmark", "G1")
        )
        ckpt_path = _locate_checkpoint(
            results_root, resolved_benchmark, str(run_tag), checkpoint_filename
        )
        benchmark = resolved_benchmark

    # --- Load checkpoint dict ONCE (architecture + provenance) -----------
    ckpt = _load_checkpoint_meta(ckpt_path)
    model_kwargs = _checkpoint_model_kwargs(ckpt)
    ckpt_data_meta = ckpt.get("data_meta") or {}

    # --- Resolve dataset -------------------------------------------------
    if data_npz is not None:
        test_npz = Path(data_npz).resolve()
        if not test_npz.is_file():
            raise FileNotFoundError(f"data_npz not found: {test_npz}")
        # Pull benchmark / data_tag from npz parent if not given.
        if benchmark is None:
            benchmark = test_npz.parent.parent.name
        if data_tag is None:
            data_tag = test_npz.parent.name
    else:
        # Need a benchmark + data_tag pair. Auto-derive from data_meta
        # if either is missing and the checkpoint carries that info.
        if benchmark is None and ckpt_data_meta.get("benchmark"):
            benchmark = str(ckpt_data_meta["benchmark"])
            logger.info(
                f"benchmark auto-resolved from checkpoint data_meta: {benchmark}"
            )
        if data_tag is None and ckpt_data_meta.get("tag"):
            data_tag = str(ckpt_data_meta["tag"])
            logger.info(
                f"data_tag auto-resolved from checkpoint data_meta: {data_tag}"
            )
        if benchmark is None:
            raise ValueError(
                "Could not resolve `benchmark`: pass --benchmark, --data-npz, "
                "or use a checkpoint that carries data_meta.benchmark."
            )
        if data_tag is None:
            raise ValueError(
                "Could not resolve `data_tag`: pass --data-tag, --data-npz, "
                "or use a checkpoint that carries data_meta.tag."
            )
        test_npz = _locate_test_split(data_root, benchmark, data_tag)

    # --- Resolve output dir ----------------------------------------------
    if output_dir is not None:
        resolved_output = Path(output_dir).resolve() / str(output_tag)
    elif checkpoint_path is not None:
        # Sweep-cell convention: put outputs next to the checkpoint.
        resolved_output = ckpt_path.parent / "testing_pipeline" / str(output_tag)
    else:
        # Tag-based convention: <results_root>/<benchmark>/<run_tag>/testing_pipeline/<output_tag>/.
        resolved_output = (
            results_root
            / str(benchmark)
            / str(run_tag)
            / "testing_pipeline"
            / str(output_tag)
        )
    resolved_output.mkdir(parents=True, exist_ok=True)
    output_dir = resolved_output

    # Resolve batch size: explicit > synth config > default 32.
    effective_batch_size = (
        int(batch_size)
        if batch_size is not None
        else int(synth_config.get("optim", {}).get("batch_size", 32))
    )

    # Translate synthetic config -> testing-pipeline YAML on disk so
    # TestRunner.from_checkpoint(config_path=...) can read it.
    run_tag_str: str = str(run_tag) if run_tag else ckpt_path.parent.name
    testing_cfg = _synth_to_testing_config(
        model_kwargs,
        tag=f"{benchmark}_{run_tag_str}_{output_tag}",
        out_dir_base=output_dir.parent,
        batch_size=effective_batch_size,
    )
    testing_cfg_path = output_dir / "effective_testing_config.yaml"
    with open(testing_cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(testing_cfg, fh, sort_keys=False)
    logger.info(f"Wrote translated testing config to {testing_cfg_path}")

    # Build the synthetic loader once (it preloads the split into RAM and
    # closes its file handle so subsequent iterations are pure in-memory).
    logger.info(f"Loading synthetic test split: {test_npz}")
    dataset = SyntheticTEDataset(test_npz)
    loader = make_dataloader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    logger.info(
        f"Built synthetic loader: n={len(dataset)} samples, "
        f"batch_size={effective_batch_size}, te_true={dataset.te_true:.4f} nats"
    )

    # Resolve device.
    if device is None or device == "auto":
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device_str = str(device)

    logger.info("=" * 70)
    logger.info("Synthetic testing-pipeline run")
    logger.info(f"  benchmark        : {benchmark}")
    logger.info(f"  data_tag         : {data_tag}")
    logger.info(f"  run_tag          : {run_tag_str}")
    logger.info(f"  checkpoint       : {ckpt_path}")
    logger.info(f"  test_split       : {test_npz}")
    logger.info(f"  output_tag       : {output_tag}")
    logger.info(f"  output_dir       : {output_dir}")
    logger.info(f"  device           : {device_str}")
    logger.info(f"  max_samples      : {max_samples}")
    logger.info(f"  analysis_samples : {analysis_samples}")
    logger.info(f"  batch_size       : {effective_batch_size}")
    logger.info("=" * 70)

    # Drive the standard pipeline. ``single_class_mode=True`` is the
    # critical knob: synthetic batches have no clinical label, so the
    # Step-0 probe would otherwise log ERROR on "only 0 canonical
    # classes found"; in single-class mode that's an INFO. Trajectory
    # needs an `epoch` field synthetic data lacks; causal-TE needs raw
    # `fhr`/`up`. Both are skipped explicitly here.
    results = run_full_test_pipeline(
        checkpoint_path=str(ckpt_path),
        data_path=None,
        output_dir=str(output_dir),
        config_path=str(testing_cfg_path),
        device=device_str,
        max_samples=max_samples,
        batch_size=effective_batch_size,
        skip_trajectory=True,
        skip_causal_te=True,
        skip_per_class_breakdown=True,
        skip_attention=skip_attention,
        skip_forecast_heatmaps=skip_forecast_heatmaps,
        skip_kld_pca=skip_kld_pca,
        skip_up_effect=skip_up_effect,
        skip_frequency_band=skip_frequency_band,
        skip_interactive=skip_interactive,
        analysis_samples=analysis_samples,
        single_class_mode=True,
        loader_override=loader,
        guid_loader_override=None,
    )

    # Sidecar JSON pairing this run's outputs with the dataset's analytic
    # ground truth, the checkpoint provenance, and the headline KL surrogate.
    _write_synthetic_ground_truth(
        output_dir=output_dir,
        benchmark=str(benchmark),
        data_tag=str(data_tag),
        run_tag=run_tag_str,
        output_tag=output_tag,
        ckpt_path=ckpt_path,
        dataset=dataset,
        results=results,
    )

    logger.info(f"Synthetic testing run complete. Outputs under {output_dir}")
    return results


def _write_synthetic_ground_truth(
    *,
    output_dir: Path,
    benchmark: str,
    data_tag: str,
    run_tag: str,
    output_tag: str,
    ckpt_path: Path,
    dataset: SyntheticTEDataset,
    results: Dict[str, Any],
) -> None:
    r"""Persist a ``synthetic_ground_truth.json`` sidecar.

    Records benchmark / checkpoint provenance, the analytic block TE in
    nats and the true lag band, plus -- when available -- the
    pipeline's mean $\bar K$ surrogate read straight off
    ``histograms/histogram_metrics.csv``. The sidecar is robust to
    partial pipeline failures: missing keys just fall through to
    ``None``.

    Args:
        output_dir: Directory the JSON is written into.
        benchmark, data_tag, run_tag, output_tag: Provenance fields.
        ckpt_path: Resolved checkpoint path.
        dataset: The synthetic dataset (carries ``te_true`` /
            ``true_lag_band`` from ``meta.json``).
        results: Result dict returned by ``run_full_test_pipeline``.
    """
    import pandas as pd

    # Pull $\bar K$ from the histograms artifact if it exists -- the
    # value will sit at hist[`kld_mean`].mean(). This is the same KL
    # surrogate `evaluate_te.py` reports.
    k_bar: Optional[float] = None
    try:
        hist = results.get("histogram")
        if isinstance(hist, pd.DataFrame) and not hist.empty and "kld_mean" in hist.columns:
            k_bar = float(hist["kld_mean"].mean())
    except Exception:  # noqa: BLE001
        pass

    n_seen = int((results.get("loader_probe") or {}).get("n_samples_seen", 0))

    payload: Dict[str, Any] = {
        "benchmark": benchmark,
        "data_tag": data_tag,
        "run_tag": run_tag,
        "output_tag": output_tag,
        "checkpoint": str(ckpt_path),
        "test_split": str(dataset.npz_path),
        "n_test_samples": int(len(dataset)),
        "n_samples_seen": n_seen,
        "te_true_nats": float(dataset.te_true),
        "true_lag_band": [int(x) for x in dataset.true_lag_band],
        "k_bar_mean_from_histograms": k_bar,
        "written_at": datetime.now().isoformat(timespec="seconds"),
    }

    sidecar = output_dir / "synthetic_ground_truth.json"
    try:
        with open(sidecar, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        logger.info(f"Wrote synthetic ground-truth sidecar to {sidecar}")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to write synthetic_ground_truth.json: {exc}")


# =============================================================================
# CLI entry point
# =============================================================================


def _parse_cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the lag-attn-v1 testing pipeline on a synthetic checkpoint.\n\n"
            "Two ways to point at a checkpoint:\n"
            "  (1) --checkpoint-path <abs/path/to/final.ckpt>  -- recommended\n"
            "      for deep sweep cells.\n"
            "  (2) --run-tag <tag> [--checkpoint final.ckpt] -- resolves to\n"
            "      <results_root>/<benchmark>/<run_tag>/<checkpoint>.\n\n"
            "Dataset resolution (in priority order):\n"
            "  --data-npz <abs/path/to/test.npz>\n"
            "    > --data-tag <tag> (-> data/<benchmark>/<data_tag>/test.npz)\n"
            "    > auto-resolve from checkpoint's embedded data_meta\n"
            "      (only when --checkpoint-path is used)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG,
        help="Path to config_synth.yaml (uses paths.data_dir / paths.results_dir).",
    )
    # ---- checkpoint selection ----
    parser.add_argument(
        "--checkpoint-path", type=Path, default=None,
        help="Absolute path to the .ckpt file. Mutually exclusive with --run-tag.",
    )
    parser.add_argument(
        "--run-tag", type=str, required=False, default=None,
        help="Run subdirectory under results/<benchmark>/ holding the checkpoint "
             "(used together with --checkpoint).",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="final.ckpt",
        help="Checkpoint filename inside the run directory (default final.ckpt). "
             "Only consulted when --run-tag is used.",
    )
    # ---- dataset selection ----
    parser.add_argument(
        "--data-npz", type=Path, default=None,
        help="Absolute path to a test.npz cache file (overrides --data-tag).",
    )
    parser.add_argument(
        "--data-tag", type=str, required=False, default=None,
        help="Cache tag selecting data/<benchmark>/<data_tag>/test.npz. "
             "When omitted *and* --checkpoint-path is used, the dataset is "
             "auto-resolved from the checkpoint's data_meta.",
    )
    parser.add_argument(
        "--benchmark", type=str, required=False, default=None,
        help="Benchmark identifier (G1/G2/G3). When omitted, derived from "
             "data_meta or experiment.benchmark in the config.",
    )
    # ---- output layout ----
    parser.add_argument(
        "--output-tag", type=str, required=True,
        help="Required folder name appended under <output_dir>/testing_pipeline/.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Explicit output directory. Defaults: <ckpt.parent>/testing_pipeline/<output_tag>/ "
             "(when --checkpoint-path is used) or "
             "<results_root>/<benchmark>/<run_tag>/testing_pipeline/<output_tag>/.",
    )
    # ---- misc ----
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Torch device (auto / cpu / cuda:N).",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap on aggregate-analysis samples. Omit / None for the whole split.",
    )
    parser.add_argument(
        "--analysis-samples", type=int, default=10,
        help="Per-sample diagnostic PDFs to emit (default 10).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Test-time batch size; defaults to optim.batch_size from the config.",
    )
    parser.add_argument("--skip-up-effect", action="store_true")
    parser.add_argument("--skip-frequency-band", action="store_true")
    parser.add_argument("--skip-attention", action="store_true")
    parser.add_argument("--skip-forecast-heatmaps", action="store_true")
    parser.add_argument("--skip-kld-pca", action="store_true")
    parser.add_argument("--skip-interactive", action="store_true")
    return parser.parse_args(argv)


def _normalise_run_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a ``RUN_CONFIG`` dict into :func:`run_synthetic_pipeline_tests` kwargs.

    Pure helper -- accepts the user's edit-and-run dict, casts the
    path-like fields to :class:`pathlib.Path`, applies defaults, and
    returns the keyword-argument dict the runner expects.

    Args:
        cfg: A dict shaped like ``RUN_CONFIG`` (see ``__main__``).

    Returns:
        Kwargs ready to splat into :func:`run_synthetic_pipeline_tests`.
    """
    cfg_path = cfg.get("config_path")
    cfg_path = Path(cfg_path) if cfg_path else _DEFAULT_CONFIG
    return dict(
        checkpoint_path=(
            Path(cfg["checkpoint_path"]) if cfg.get("checkpoint_path") else None
        ),
        run_tag=cfg.get("run_tag"),
        checkpoint_filename=str(cfg.get("checkpoint", "final.ckpt")),
        data_npz=(Path(cfg["data_npz"]) if cfg.get("data_npz") else None),
        data_tag=cfg.get("data_tag"),
        benchmark=cfg.get("benchmark"),
        output_tag=str(cfg["output_tag"]),
        output_dir=(Path(cfg["output_dir"]) if cfg.get("output_dir") else None),
        config_path=cfg_path,
        device=cfg.get("device", "auto"),
        max_samples=cfg.get("max_samples"),
        analysis_samples=int(cfg.get("analysis_samples", 10)),
        batch_size=cfg.get("batch_size"),
        skip_up_effect=bool(cfg.get("skip_up_effect", False)),
        skip_frequency_band=bool(cfg.get("skip_frequency_band", False)),
        skip_attention=bool(cfg.get("skip_attention", False)),
        skip_forecast_heatmaps=bool(cfg.get("skip_forecast_heatmaps", False)),
        skip_kld_pca=bool(cfg.get("skip_kld_pca", False)),
        skip_interactive=bool(cfg.get("skip_interactive", False)),
    )


def _from_cli_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Build :func:`run_synthetic_pipeline_tests` kwargs from parsed CLI args."""
    return dict(
        checkpoint_path=args.checkpoint_path,
        run_tag=args.run_tag,
        checkpoint_filename=args.checkpoint,
        data_npz=args.data_npz,
        data_tag=args.data_tag,
        benchmark=args.benchmark,
        output_tag=args.output_tag,
        output_dir=args.output_dir,
        config_path=args.config,
        device=args.device,
        max_samples=args.max_samples,
        analysis_samples=args.analysis_samples,
        batch_size=args.batch_size,
        skip_up_effect=args.skip_up_effect,
        skip_frequency_band=args.skip_frequency_band,
        skip_attention=args.skip_attention,
        skip_forecast_heatmaps=args.skip_forecast_heatmaps,
        skip_kld_pca=args.skip_kld_pca,
        skip_interactive=args.skip_interactive,
    )


if __name__ == "__main__":
    # =========================================================================
    # >>>>>>>>>>>>>>>>>>>>  EDIT BELOW TO RUN FROM YOUR IDE  <<<<<<<<<<<<<<<<<<<
    # =========================================================================
    #
    # This block is the edit-and-run entry point (V2-D8 convention -- mirrors
    # `evaluate_te.py`, `lag_recovery.py`, etc.). If you launch the file with
    # NO command-line args (e.g. via "Run File" in your IDE, or
    # ``python -m ...synthetic.run_pipeline_tests``), the ``RUN_CONFIG`` dict
    # below drives the run.
    #
    # If you DO pass any CLI flag, argparse takes over and ``RUN_CONFIG`` is
    # ignored. So you can keep one tested configuration here for IDE runs and
    # still drive sweeps from the shell.
    #
    # Selection priorities (same as the CLI flags):
    #   checkpoint:  checkpoint_path > (run_tag + checkpoint)
    #   dataset:     data_npz > data_tag > auto-from ckpt["data_meta"]
    #                (the auto branch fires only when checkpoint_path is set)
    #   output dir:  output_dir > <ckpt.parent>/testing_pipeline/<output_tag>/
    #                (when checkpoint_path is set) >
    #                <results_root>/<benchmark>/<run_tag>/testing_pipeline/<output_tag>/
    # =========================================================================

    RUN_CONFIG: Dict[str, Any] = {
        # ---- CHECKPOINT --- pick ONE pattern (path OR tag) ------------------
        # Pattern A: deep path (recommended for sweep cells / calibration cells).
        "checkpoint_path": None,
        # e.g. "model/vae_teb_prediction/model/model_experiment/results/G1/G1_baseline/final.ckpt"

        # Pattern B: tag-based. Used only when checkpoint_path is None.
        "run_tag":    None,           # subdir under results/<benchmark>/
        "checkpoint": "final.ckpt",   # filename inside that subdir

        # ---- DATASET --- highest non-None wins. Auto-resolve only fires
        # when checkpoint_path is set AND both fields below are None. ---------
        "data_npz":  None,            # absolute path to a test.npz
        "data_tag":  None,            # tag under data/<benchmark>/<data_tag>/
        "benchmark": None,            # derived from data_meta when omitted

        # ---- OUTPUT --- output_tag is REQUIRED ------------------------------
        "output_tag": "default",      # leaf folder under testing_pipeline/
        "output_dir": None,           # full override; otherwise auto-resolved

        # ---- RUN KNOBS ------------------------------------------------------
        "config_path":      None,     # None -> packaged synthetic/config_synth.yaml
        "device":           "auto",   # "auto" | "cpu" | "cuda:0" | ...
        "max_samples":      None,     # None -> whole test split
        "analysis_samples": 10,       # per-sample diagnostic PDFs
        "batch_size":       None,     # None -> synth optim.batch_size

        # ---- ANALYSIS GATES -------------------------------------------------
        "skip_up_effect":         False,
        "skip_frequency_band":    False,
        "skip_attention":         False,
        "skip_forecast_heatmaps": False,
        "skip_kld_pca":           False,
        "skip_interactive":       False,
    }

    if len(sys.argv) > 1:
        kwargs = _from_cli_args(_parse_cli(sys.argv[1:]))
        logger.info(f"CLI mode: {kwargs}")
    else:
        kwargs = _normalise_run_config(RUN_CONFIG)
        logger.info(f"RUN_CONFIG mode: {kwargs}")

    run_synthetic_pipeline_tests(**kwargs)
