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

Output layout: ``<results_dir>/<benchmark>/<run_tag>/testing_pipeline/<output_tag>/``.

The ``<output_tag>`` is **required** so multiple test runs on the same
checkpoint (e.g. ``final.ckpt`` vs ``best.ckpt``, or different
``max_samples`` caps) coexist as siblings rather than overwriting.

Analyses skipped on synthetic data (per the synthetic-data contract -- no
raw ``fhr``/``up`` signals, no clinical labels, no per-epoch metadata):

* ``trajectory`` -- requires ``guid`` + ``epoch`` for hours-before-birth axis.
* ``causal_te_validation`` -- requires raw ``fhr``/``up`` for event detection.
* ``class_separation`` / ``per_class_breakdown`` -- auto-skipped by
  ``single_class_mode=True`` because synthetic data carries no class label.

Example:
    >>> # Edit the RUN_CONFIG dict below, then:
    >>> python -m model.vae_teb_prediction.model.model_experiment.synthetic.run_pipeline_tests
    >>> # Or as a CLI:
    >>> python -m model.vae_teb_prediction.model.model_experiment.synthetic.run_pipeline_tests \
    ...     --benchmark G1 --data-tag G1_baseline --run-tag G1_baseline \
    ...     --checkpoint final.ckpt --output-tag smoke --max-samples 200
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


def _checkpoint_model_kwargs(ckpt_path: Path) -> Dict[str, Any]:
    r"""Extract the ``model_kwargs`` block from a synthetic checkpoint.

    ``train_minimal.py`` saves checkpoints as a plain ``torch.save({...})``
    dict containing ``model_state_dict``, ``model_kwargs``, ``config``,
    ``data_meta`` and a few diagnostics. We need only ``model_kwargs`` to
    reconstruct the model under :class:`SeqVaeLagAttnV1`. Reading the
    architecture from the checkpoint (rather than from a paired YAML)
    guarantees the testing pipeline builds the exact shape the weights
    were trained for.

    Args:
        ckpt_path: Path to ``final.ckpt`` or ``best.ckpt``.

    Returns:
        The ``model_kwargs`` dict ready to feed
        ``SeqVaeLagAttnV1(**kwargs)``.

    Raises:
        KeyError: If the checkpoint does not carry ``model_kwargs``.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_kwargs" not in ckpt:
        raise KeyError(
            f"{ckpt_path} is not a train_minimal checkpoint -- "
            f"`model_kwargs` is missing. Available keys: {sorted(ckpt)}"
        )
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
    benchmark: str,
    data_tag: str,
    run_tag: str,
    output_tag: str,
    checkpoint_filename: str = "final.ckpt",
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

    Builds a :class:`SyntheticTEDataset` over the test split paired with
    the checkpoint, translates the architecture kwargs into a
    testing-pipeline YAML, then delegates to
    :func:`run_full_test_pipeline` via ``loader_override``. After the
    pipeline returns it writes a ``synthetic_ground_truth.json`` sidecar
    pairing the run's outputs with the dataset's analytic
    ``te_true`` / ``true_lag_band`` so cross-referencing with
    :mod:`evaluate_te` is trivial.

    Args:
        benchmark: Benchmark identifier (e.g. ``G1`` / ``G2`` / ``G3``).
        data_tag: Cache tag selecting
            ``data/<benchmark>/<data_tag>/test.npz``.
        run_tag: Training-run subdirectory under
            ``results/<benchmark>/`` holding the checkpoint.
        output_tag: Folder name under
            ``results/<benchmark>/<run_tag>/testing_pipeline/`` for this
            run's outputs. Required.
        checkpoint_filename: Either ``final.ckpt`` or ``best.ckpt``
            (or any custom filename in the run directory).
        config_path: Path to ``config_synth.yaml`` (defaults to the
            packaged one). Only ``paths.data_dir`` and
            ``paths.results_dir`` are consumed.
        device: Torch device override (``cuda:0``, ``cpu``, ``auto``).
            Defaults to ``cuda:0`` if available, else CPU.
        max_samples: Cap for aggregate analyses. ``None`` means iterate
            the whole test split.
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
        FileNotFoundError: If the checkpoint or test split is missing.
    """
    cfg_path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    synth_config = _load_yaml(cfg_path)

    data_root = _data_root(synth_config)
    results_root = _results_root(synth_config)

    ckpt_path = _locate_checkpoint(
        results_root, benchmark, run_tag, checkpoint_filename
    )
    test_npz = _locate_test_split(data_root, benchmark, data_tag)

    # Output dir -- one sibling folder per output_tag.
    output_dir = (
        results_root
        / str(benchmark)
        / str(run_tag)
        / "testing_pipeline"
        / str(output_tag)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Architecture from the checkpoint -- single source of truth.
    model_kwargs = _checkpoint_model_kwargs(ckpt_path)

    # Resolve batch size: explicit > synth config > default 32.
    effective_batch_size = (
        int(batch_size)
        if batch_size is not None
        else int(synth_config.get("optim", {}).get("batch_size", 32))
    )

    # Translate synthetic config -> testing-pipeline YAML on disk so
    # TestRunner.from_checkpoint(config_path=...) can read it.
    testing_cfg = _synth_to_testing_config(
        model_kwargs,
        tag=f"{benchmark}_{run_tag}_{output_tag}",
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
    logger.info(f"  run_tag          : {run_tag}")
    logger.info(f"  checkpoint       : {ckpt_path}")
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
        benchmark=benchmark,
        data_tag=data_tag,
        run_tag=run_tag,
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
        description="Run the lag-attn-v1 testing pipeline on a synthetic checkpoint.",
    )
    parser.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG,
        help="Path to config_synth.yaml (uses paths.data_dir / paths.results_dir).",
    )
    parser.add_argument(
        "--benchmark", type=str, required=False, default=None,
        help="Benchmark identifier (defaults to experiment.benchmark from the config).",
    )
    parser.add_argument(
        "--data-tag", type=str, required=False, default=None,
        help="Cache tag selecting data/<benchmark>/<data_tag>/test.npz "
             "(defaults to experiment.tag from the config).",
    )
    parser.add_argument(
        "--run-tag", type=str, required=False, default=None,
        help="Run subdirectory under results/<benchmark>/ holding the checkpoint "
             "(defaults to --data-tag).",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="final.ckpt",
        help="Checkpoint filename inside the run directory (default final.ckpt).",
    )
    parser.add_argument(
        "--output-tag", type=str, required=True,
        help="Required folder name under "
             "results/<benchmark>/<run_tag>/testing_pipeline/.",
    )
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


def _resolve_run_keys(
    *,
    config_path: Path,
    benchmark: Optional[str],
    data_tag: Optional[str],
    run_tag: Optional[str],
) -> tuple[str, str, str]:
    r"""Fall back to ``experiment.benchmark`` / ``experiment.tag`` from the YAML.

    Mirrors the project convention used by every other ``synthetic/``
    runner: if the caller omits the keys, read them from the active
    benchmark in ``config_synth.yaml``.
    """
    synth = _load_yaml(config_path)
    experiment = synth.get("experiment", {}) or {}
    bm = benchmark or str(experiment.get("benchmark", "G1"))
    dt = data_tag or str(experiment.get("tag", bm))
    rt = run_tag or dt
    return bm, dt, rt


# =============================================================================
# Edit-and-run config (V2-D8)
# =============================================================================


RUN_CONFIG: Dict[str, Any] = {
    "config_path": None,        # None -> _DEFAULT_CONFIG
    "benchmark": None,          # None -> config_synth.yaml: experiment.benchmark
    "data_tag": None,           # None -> config_synth.yaml: experiment.tag
    "run_tag": None,            # None -> defaults to data_tag
    "checkpoint": "final.ckpt", # filename inside results/<benchmark>/<run_tag>/
    "output_tag": "default",    # required; folder name under testing_pipeline/
    "device": "auto",
    "max_samples": None,        # None -> whole test split
    "analysis_samples": 10,
    "batch_size": None,         # None -> synth optim.batch_size (default 32)
    "skip_up_effect": False,
    "skip_frequency_band": False,
    "skip_attention": False,
    "skip_forecast_heatmaps": False,
    "skip_kld_pca": False,
    "skip_interactive": False,
}


def _from_run_config() -> Dict[str, Any]:
    """Build :func:`run_synthetic_pipeline_tests` kwargs from ``RUN_CONFIG``."""
    cfg_path = RUN_CONFIG.get("config_path")
    cfg_path = Path(cfg_path) if cfg_path else _DEFAULT_CONFIG
    bm, dt, rt = _resolve_run_keys(
        config_path=cfg_path,
        benchmark=RUN_CONFIG.get("benchmark"),
        data_tag=RUN_CONFIG.get("data_tag"),
        run_tag=RUN_CONFIG.get("run_tag"),
    )
    return dict(
        benchmark=bm,
        data_tag=dt,
        run_tag=rt,
        output_tag=str(RUN_CONFIG["output_tag"]),
        checkpoint_filename=str(RUN_CONFIG.get("checkpoint", "final.ckpt")),
        config_path=cfg_path,
        device=RUN_CONFIG.get("device", "auto"),
        max_samples=RUN_CONFIG.get("max_samples"),
        analysis_samples=int(RUN_CONFIG.get("analysis_samples", 10)),
        batch_size=RUN_CONFIG.get("batch_size"),
        skip_up_effect=bool(RUN_CONFIG.get("skip_up_effect", False)),
        skip_frequency_band=bool(RUN_CONFIG.get("skip_frequency_band", False)),
        skip_attention=bool(RUN_CONFIG.get("skip_attention", False)),
        skip_forecast_heatmaps=bool(RUN_CONFIG.get("skip_forecast_heatmaps", False)),
        skip_kld_pca=bool(RUN_CONFIG.get("skip_kld_pca", False)),
        skip_interactive=bool(RUN_CONFIG.get("skip_interactive", False)),
    )


def _from_cli_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Build :func:`run_synthetic_pipeline_tests` kwargs from parsed CLI args."""
    bm, dt, rt = _resolve_run_keys(
        config_path=args.config,
        benchmark=args.benchmark,
        data_tag=args.data_tag,
        run_tag=args.run_tag,
    )
    return dict(
        benchmark=bm,
        data_tag=dt,
        run_tag=rt,
        output_tag=args.output_tag,
        checkpoint_filename=args.checkpoint,
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
    if len(sys.argv) > 1:
        kwargs = _from_cli_args(_parse_cli(sys.argv[1:]))
        logger.info(f"CLI mode: {kwargs}")
    else:
        kwargs = _from_run_config()
        logger.info(f"RUN_CONFIG mode: {kwargs}")

    run_synthetic_pipeline_tests(**kwargs)
