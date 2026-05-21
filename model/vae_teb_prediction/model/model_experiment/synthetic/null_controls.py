r"""Null controls -- re-evaluate a trained checkpoint on broken-source caches.

A faithful transfer-entropy surrogate must collapse $\bar K$ when the
source-to-target link is destroyed in ways that leave the marginal data
distributions intact. This module re-evaluates an already-trained source
benchmark checkpoint on two such control caches:

    * **wrong-delay** -- same DGP family as the source benchmark but with a
      delay $D \gg L_{\max} + H$, so the model's lag-attention window cannot
      reach the true source-to-target alignment. **Caveat**: the
      unidirectional LSTM in the source encoder can still propagate signal
      across more than $L_{\max}$ steps (see ``model_validation_v2_plan.md``
      sec. 4.2), so $\bar K$ may not fully collapse -- this row is
      **informational**, not pass/fail.
    * **zero-coupling** -- same DGP with the source-to-target coupling
      coefficient $c = 0$, so $\mathrm{TE}_{\rm true} = 0$ by construction.
      A faithful surrogate must collapse $\bar K$ towards zero.

The runner is read-only on the model side: no training is performed. The
source checkpoint must already exist on disk (calibration runs or a
baseline ``train_minimal`` invocation produce it).

This module **reuses** :mod:`build_dataset`, :mod:`evaluate_te` and
:mod:`train_minimal` wholesale -- it reimplements no generation, training or
scoring code. The key reuse is :func:`evaluate_te.evaluate_checkpoint`'s
``benchmark_override`` kwarg, which lets a G2 checkpoint be scored against
a ``G2_wrong_delay`` / ``G2_zero_coupling`` test cache without rewriting
any harness internals.

Run modes (project convention -- Decision V2-D8 in
``model_validation_v2_plan.md``): like every ``synthetic/`` runner this file
supports **both** a CLI and an edit-and-run ``__main__``, auto-detected from
whether any command-line argument is present.

    * CLI mode (any ``--flag`` passed)::

        python -m ...synthetic.null_controls [--build-missing]
            [--config PATH] [--source-benchmark B] [--source-run-tag T]
            [--device DEV] [--seed S]

    * Edit-and-run mode (no arguments) -- edit the ``RUN_CONFIG`` dict in the
      ``__main__`` block, then run the file directly (IDE / notebook)::

        python -m ...synthetic.null_controls
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

import torch

from model.vae_teb_prediction.model.model_experiment.synthetic import (
    build_dataset as bd,
)
from model.vae_teb_prediction.model.model_experiment.synthetic import (
    evaluate_te as ev,
)
from model.vae_teb_prediction.model.model_experiment.synthetic import (
    train_minimal as tm,
)

_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _PKG_DIR / "config_synth.yaml"

# Default source benchmark when neither the call argument nor the YAML
# ``null_controls.source_benchmark`` overrides it. Kept here so the CLI
# help text and the runtime default cannot drift apart.
_DEFAULT_SOURCE_BENCHMARK = "G2"


class ControlSpec(NamedTuple):
    """A named triple describing one null-control re-evaluation cell.

    Attributes:
        label: Human-readable identifier (used as dict key in
            ``metrics.json["controls"]`` and in the CSV ``control`` column).
        benchmark: Benchmark id whose DGP produces the control cache. Must be
            a key of :data:`build_dataset._GENERATORS`.
        data_tag: Cache subdirectory name under
            ``data/<benchmark>/<data_tag>/``.
    """

    label: str
    benchmark: str
    data_tag: str


# Default controls. Overridable via ``config["null_controls"]["controls"]``
# as a list of three-element lists (label, benchmark, data_tag).
_DEFAULT_CONTROLS: Sequence[ControlSpec] = (
    ControlSpec("wrong_delay",   "G2_wrong_delay",   "G2_wd_test"),
    ControlSpec("zero_coupling", "G2_zero_coupling", "G2_zc_test"),
)

# Columns of the null-controls summary CSV (one row per control).
_SUMMARY_FIELDS = [
    "control", "control_benchmark", "data_tag",
    "source_benchmark", "source_run_tag", "source_ckpt",
    "te_true",
    "k_bar", "k_bar_shuffled", "k_bar_reversed", "pred_gap",
    "feat_loss", "base_loss", "kld_loss",
    "epoch",
]


# =============================================================================
# Source-checkpoint discovery
# =============================================================================

def _find_source_ckpt(
    results_root: Path, source_benchmark: str,
    source_run_tag: Optional[str],
) -> Optional[Path]:
    """Resolve the source-benchmark checkpoint to re-evaluate.

    The runner does not train. Either ``source_run_tag`` points at a specific
    subdirectory under ``results/<source_benchmark>/`` whose ``final.ckpt``
    must exist, or it is ``None`` and we pick the most-recently modified
    ``final.ckpt`` under any subdir.

    Args:
        results_root: Resolved ``paths.results_dir``.
        source_benchmark: Benchmark whose trained checkpoint is re-evaluated
            (e.g. ``"G2"``).
        source_run_tag: Specific run subdirectory, or ``None`` for "newest".

    Returns:
        Absolute path of the chosen ``final.ckpt``, or ``None`` if no
        candidate exists.
    """
    source_root = results_root / source_benchmark
    if source_run_tag is not None:
        candidate = source_root / source_run_tag / "final.ckpt"
        return candidate if candidate.is_file() else None
    if not source_root.is_dir():
        return None
    candidates = sorted(
        source_root.glob("*/final.ckpt"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return candidates[0] if candidates else None


# =============================================================================
# Per-control runner
# =============================================================================

def _run_control(
    config: Dict[str, Any], *,
    label: str,
    control_benchmark: str,
    data_tag: str,
    source_benchmark: str,
    source_run_tag: Optional[str],
    source_ckpt: Path,
    build_missing: bool,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    """Build the control cache (opt-in) and re-evaluate the source checkpoint.

    Args:
        config: The parsed config.
        label: Human-readable control name (``"wrong_delay"`` /
            ``"zero_coupling"``).
        control_benchmark: Benchmark id whose DGP is used for the control
            cache (must be in :data:`build_dataset._GENERATORS`).
        data_tag: Cache subdirectory name under
            ``data/<control_benchmark>/``.
        source_benchmark: Benchmark whose trained checkpoint is re-evaluated.
        source_run_tag: Run subdir of the source checkpoint (purely
            informational; used in the row provenance).
        source_ckpt: Absolute path to the source ``final.ckpt``.
        build_missing: If ``True``, generate the control cache when it is
            not present. If ``False`` and the cache is missing, return ``None``.
        device: Compute device.

    Returns:
        A flat row dict matching :data:`_SUMMARY_FIELDS`, or ``None`` when
        the control cache is missing and ``build_missing`` is ``False``.
    """
    # Build the control cache under its OWN benchmark dir using a copy of
    # ``config`` where ``experiment.benchmark`` selects the control block.
    cfg = deepcopy(config)
    cfg["experiment"]["benchmark"] = control_benchmark
    tm.resolve_active_benchmark(cfg)

    cache_dir = ev._data_root(cfg) / control_benchmark / data_tag
    if not (cache_dir / "test.npz").is_file():
        if build_missing:
            print(f"  [build] {label}: cache "
                  f"'{control_benchmark}/{data_tag}'")
            bcfg = deepcopy(cfg)
            bd._apply_overrides(bcfg, {"tag": data_tag})
            bd.build_dataset(bcfg, force=False)
        else:
            print(f"  [skip ] {label}: cache "
                  f"'{control_benchmark}/{data_tag}' not built "
                  f"(pass --build-missing to generate it)")
            return None

    # Re-evaluate the SOURCE checkpoint on the CONTROL cache. The
    # ``benchmark_override`` kwarg makes ``evaluate_checkpoint`` resolve the
    # test loader from ``data/<control_benchmark>/<data_tag>/`` even though
    # the checkpoint itself was trained under ``source_benchmark``. We pass a
    # fresh deepcopy because ``evaluate_checkpoint`` -> ``make_test_loader``
    # may mutate the top-level config in some code paths; the orchestrator's
    # downstream JSON / CSV writers rely on the original being intact.
    eval_cfg = deepcopy(config)
    row = ev.evaluate_checkpoint(
        source_ckpt, eval_cfg,
        device=device,
        data_tag=data_tag,
        benchmark_override=control_benchmark,
    )

    source_ckpt_path = Path(source_ckpt)
    return {
        "control": label,
        "control_benchmark": control_benchmark,
        "data_tag": data_tag,
        "source_benchmark": source_benchmark,
        "source_run_tag": (source_run_tag or source_ckpt_path.parent.name),
        "source_ckpt": str(source_ckpt_path),
        "te_true": float(row.get("te_true", float("nan"))),
        "k_bar": float(row.get("k_bar", float("nan"))),
        "k_bar_shuffled": float(row.get("k_bar_shuffled", float("nan"))),
        "k_bar_reversed": float(row.get("k_bar_reversed", float("nan"))),
        "pred_gap": float(row.get("pred_gap", float("nan"))),
        "feat_loss": float(row.get("feat_loss", float("nan"))),
        "base_loss": float(row.get("base_loss", float("nan"))),
        "kld_loss": float(row.get("kld_loss", float("nan"))),
        "epoch": row.get("epoch"),
    }


# =============================================================================
# Orchestrator
# =============================================================================

_UNSET = object()


def run_null_controls(
    config: Dict[str, Any], *,
    source_benchmark: Optional[str] = None,
    source_run_tag: Any = _UNSET,
    controls: Optional[Sequence[ControlSpec]] = None,
    device: Optional[torch.device] = None,
    build_missing: bool = False,
) -> Dict[str, Any]:
    r"""Re-evaluate a trained source-benchmark checkpoint on the control caches.

    Writes ``results/<source_benchmark>/<out_dir>/{summary.csv, metrics.json}``
    where ``out_dir`` defaults to ``"null_controls"`` (overridable via
    ``config["null_controls"]["out_dir"]``).

    Args:
        config: The parsed ``config_synth.yaml`` (must carry blocks for every
            control benchmark and the source benchmark).
        source_benchmark: Override for ``config["null_controls"][
            "source_benchmark"]``. ``None`` falls back to the config value or
            :data:`_DEFAULT_SOURCE_BENCHMARK`.
        source_run_tag: Override for ``config["null_controls"][
            "source_run_tag"]``. Sentinel default ``_UNSET`` falls back to the
            config value; passing the literal ``None`` forces "pick the newest
            ``final.ckpt`` under ``results/<source_benchmark>/``" regardless
            of the config.
        controls: Override for ``config["null_controls"]["controls"]``. A
            sequence of :class:`ControlSpec` triples (label, benchmark, tag).
        device: Compute device. Defaults to :func:`train_minimal.resolve_device`.
        build_missing: If ``True``, generate any missing control cache.

    Returns:
        A results dict: ``rows`` (per-control), ``out_dir``, ``skipped``,
        ``source_benchmark`` and ``source_ckpt``.
    """
    device = device or tm.resolve_device(config["runtime"])
    cfg_nc: Dict[str, Any] = config.get("null_controls", {}) or {}
    source_benchmark = str(
        source_benchmark
        or cfg_nc.get("source_benchmark", _DEFAULT_SOURCE_BENCHMARK)
    )
    resolved_run_tag: Optional[str] = (
        cfg_nc.get("source_run_tag") if source_run_tag is _UNSET
        else source_run_tag  # explicit None forces newest-pick
    )
    if controls is None:
        raw_controls = cfg_nc.get("controls")
        if raw_controls:
            controls = [
                ControlSpec(str(label), str(bench), str(tag))
                for label, bench, tag in raw_controls
            ]
        else:
            controls = list(_DEFAULT_CONTROLS)

    results_root = ev._results_root(config)
    out_dir = (
        results_root / source_benchmark
        / str(cfg_nc.get("out_dir", "null_controls"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    source_ckpt = _find_source_ckpt(
        results_root, source_benchmark, resolved_run_tag,
    )
    tag_segment = f"/{resolved_run_tag}" if resolved_run_tag else ""
    print(
        f"[null-controls] source={source_benchmark}{tag_segment}  "
        f"controls={[c.label for c in controls]}  "
        f"device={device}  build_missing={build_missing}"
    )
    if source_ckpt is None:
        path_pattern = f"{resolved_run_tag}/" if resolved_run_tag else "*/"
        msg = (
            f"  [error] no source checkpoint found under "
            f"results/{source_benchmark}/{path_pattern}"
            f"final.ckpt -- train the source benchmark first."
        )
        print(msg)
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "created": datetime.now(timezone.utc).isoformat(),
                    "source_benchmark": source_benchmark,
                    "source_run_tag": resolved_run_tag,
                    "source_ckpt": None,
                    "controls": {},
                    "skipped": [c.label for c in controls],
                    "error": msg.strip(),
                },
                fh, indent=2,
            )
        return {
            "rows": [], "out_dir": str(out_dir),
            "skipped": [c.label for c in controls],
            "source_benchmark": source_benchmark,
            "source_ckpt": None,
        }

    # Resolve once so the per-control loop and the final metrics dict share
    # the same canonical path string.
    source_ckpt_resolved = source_ckpt.resolve()
    print(f"  source checkpoint: {source_ckpt_resolved}")

    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for spec in controls:
        try:
            row = _run_control(
                config, label=spec.label,
                control_benchmark=spec.benchmark, data_tag=spec.data_tag,
                source_benchmark=source_benchmark,
                source_run_tag=resolved_run_tag,
                source_ckpt=source_ckpt_resolved,
                build_missing=build_missing, device=device,
            )
        # Narrow: any of these means a control cache is missing / a forward
        # blew up / a tensor mismatched. Programming errors (KeyError,
        # AttributeError) propagate as before. The traceback is logged so the
        # one-line summary is debuggable.
        except (RuntimeError, FileNotFoundError, ValueError,
                torch.cuda.OutOfMemoryError) as exc:
            print(f"  [error] {spec.label}: {type(exc).__name__}: {exc}")
            print(traceback.format_exc())
            skipped.append(spec.label)
            continue
        if row is None:
            skipped.append(spec.label)
            continue
        rows.append(row)

    _write_summary_csv(rows, out_dir / "summary.csv")
    by_label: Dict[str, Dict[str, Any]] = {r["control"]: r for r in rows}
    source_ckpt_str = str(source_ckpt_resolved)
    metrics: Dict[str, Any] = {
        "created": datetime.now(timezone.utc).isoformat(),
        "source_benchmark": source_benchmark,
        "source_run_tag": (resolved_run_tag
                           or source_ckpt_resolved.parent.name),
        "source_ckpt": source_ckpt_str,
        "controls": {
            label: {k: r.get(k) for k in _SUMMARY_FIELDS}
            for label, r in by_label.items()
        },
        "skipped": skipped,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    for r in rows:
        print(
            f"  [{r['control']:<14s}] K_bar={r['k_bar']:.5f}  "
            f"K_shuffled={r['k_bar_shuffled']:.5f}  "
            f"K_reversed={r['k_bar_reversed']:.5f}  "
            f"pred_gap={r['pred_gap']:+.5f}  "
            f"te_true={r['te_true']:.4f} nats"
        )
    print(
        f"[done] {len(rows)} control(s) evaluated, "
        f"{len(skipped)} skipped\n       artifacts -> {out_dir}"
    )
    return {
        "rows": rows, "out_dir": str(out_dir), "skipped": skipped,
        "source_benchmark": source_benchmark,
        "source_ckpt": source_ckpt_str,
    }


def _write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write the per-control summary CSV.

    Args:
        rows: The per-control evaluation rows.
        path: Destination CSV path (overwritten).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in _SUMMARY_FIELDS})


# =============================================================================
# Overrides + dispatch
# =============================================================================

def _apply_overrides(
    config: Dict[str, Any], overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply the config-level overrides onto ``config`` in place.

    Only ``device`` / ``seed`` are config fields; ``build_missing`` /
    ``source_benchmark`` / ``source_run_tag`` are call arguments handled by
    :func:`_dispatch`.

    Args:
        config: The config dict (mutated in place).
        overrides: Flat ``{key: value}`` overrides; ``None`` values ignored.

    Returns:
        The same ``config`` dict.
    """
    if overrides.get("device") is not None:
        config["runtime"]["device"] = overrides["device"]
    if overrides.get("seed") is not None:
        config["experiment"]["seed"] = overrides["seed"]
    return config


def _dispatch(
    config: Dict[str, Any], overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve overrides, seed, device and run the null-controls.

    Args:
        config: The parsed ``config_synth.yaml``.
        overrides: Flat overrides (from ``vars(args)`` or ``RUN_CONFIG``).

    Returns:
        The :func:`run_null_controls` result dict.
    """
    config = deepcopy(config)
    _apply_overrides(config, overrides)
    tm.set_seed(int(config["experiment"].get("seed", 0)))
    device = tm.resolve_device(config["runtime"])
    # ``source_run_tag``: passing the literal ``None`` to ``run_null_controls``
    # forces the newest-pick path, which is what we want when the CLI flag is
    # unset (argparse leaves it as ``None``). The ``_UNSET`` sentinel is only
    # for callers that want the YAML to win.
    return run_null_controls(
        config,
        source_benchmark=overrides.get("source_benchmark"),
        source_run_tag=overrides.get("source_run_tag"),
        device=device,
        build_missing=bool(overrides.get("build_missing")),
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        The parsed :class:`argparse.Namespace`.
    """
    p = argparse.ArgumentParser(
        description="Null controls (wrong-delay + zero-coupling) for "
                    "SeqVaeLagAttnV1 on synthetic data."
    )
    p.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG,
        help="path to config_synth.yaml",
    )
    p.add_argument(
        "--source-benchmark", type=str, default=None, dest="source_benchmark",
        help=("override null_controls.source_benchmark (default "
              f"{_DEFAULT_SOURCE_BENCHMARK})"),
    )
    p.add_argument(
        "--source-run-tag", type=str, default=None, dest="source_run_tag",
        help="override null_controls.source_run_tag (default: newest "
             "final.ckpt under results/<source_benchmark>/)",
    )
    p.add_argument(
        "--build-missing", action=argparse.BooleanOptionalAction, default=False,
        dest="build_missing",
        help="generate any missing control cache (opt-in)",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="override runtime.device (auto / cpu / cuda / cuda:N)",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="override experiment.seed",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point: parse args, load config, dispatch.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).
    """
    args = parse_args(argv)
    config = tm.load_config(args.config)
    _dispatch(config, vars(args))


if __name__ == "__main__":
    # =========================================================================
    # How to run this script  (project convention -- Decision V2-D8)
    # -------------------------------------------------------------------------
    # Two equivalent modes, auto-detected from the command line:
    #
    #   * CLI mode      -- launched with any --flag -> argparse `main()`.
    #   * EDIT-AND-RUN  -- launched with NO arguments -> the `RUN_CONFIG` dict
    #                      below is used. Edit it and run the file directly;
    #                      no terminal flags required.
    #
    # Every key in RUN_CONFIG mirrors a CLI flag and is forwarded to
    # `_dispatch`; `None` means "fall back to config_synth.yaml".
    # =========================================================================

    CONFIG_PATH = _DEFAULT_CONFIG

    RUN_CONFIG = {
        "build_missing": False,        # build the wrong-delay / zero-coupling caches
        "source_benchmark": None,      # None -> config null_controls.source_benchmark
        "source_run_tag": None,        # None -> newest final.ckpt under results/<bench>/
        "device": None,                # None -> config runtime.device
        "seed": None,                  # None -> config experiment.seed
    }

    if len(sys.argv) > 1:
        main()                              # CLI mode -- argparse
    else:
        config = tm.load_config(CONFIG_PATH)
        # --- optional: tweak any config value not covered by RUN_CONFIG ------
        # e.g.  config["null_controls"]["controls"] = [["wrong_delay", ...]]
        # ---------------------------------------------------------------------
        _dispatch(config, RUN_CONFIG)
