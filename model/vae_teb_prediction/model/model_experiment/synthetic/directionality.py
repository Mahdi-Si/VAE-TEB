r"""Directionality test -- $\bar K_{X\to Y}$ versus $\bar K_{Y\to X}$.

A faithful transfer-entropy surrogate must be **direction-specific**: for a
process whose causal arrow is $X \to Y$, the model should report a large
$\bar K$ when the source slot holds $X$ and the target slot holds $Y$, and a
near-zero $\bar K$ when the streams are swapped.

The model :class:`SeqVaeLagAttnV1` is directional *by construction* -- it always
predicts the target (``fhr``) stream from the source (``up``) stream -- so the
two directions need two separately trained models:

    * **forward** -- v2 Benchmark G1: the AR(2)-oscillator state $X$ occupies
      the 101-channel source slot, the dependent stream $Y$ the 87-channel
      target. The model measures $\mathrm{TE}_{X\to Y} > 0$.
    * **reverse** -- v2 Benchmark G1-rev
      (``gen_state_space_oscillator(reverse_roles=True)``): the oscillator
      state is placed in the 87-channel *target* slot and the dependent
      stream in the 101-channel *source* slot. The model then measures
      $\mathrm{TE}_{Y\to X}$, whose true value is $0$ (the slot-wise direction
      is anti-causal).

A strong directionality result is $\bar K_{\rm forward} \gg \bar K_{\rm
reverse}$. This module builds (opt-in) the two datasets, trains (opt-in) a
model on each, evaluates both with :func:`evaluate_te.evaluate_checkpoint`, and
writes the comparison to ``results/directionality/``.

This module **reuses** :mod:`build_dataset`, :mod:`train_minimal` and
:mod:`evaluate_te` wholesale -- it reimplements no generation, training or
scoring code.

Run modes (project convention -- Decision D9 in
``synthetic_te_validation_plan.md``): like every ``synthetic/`` runner this file
supports **both** a CLI and an edit-and-run ``__main__``, auto-detected from
whether any command-line argument is present.

    * CLI mode (any ``--flag`` passed)::

        python -m ...synthetic.directionality [--build-missing]
            [--train-missing] [--config PATH] [--device DEV] [--seed S]

    * Edit-and-run mode (no arguments) -- edit the ``RUN_CONFIG`` dict in the
      ``__main__`` block, then run the file directly (IDE / notebook)::

        python -m ...synthetic.directionality
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
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

# ``synthetic/`` package dir and its parent ``model_experiment/`` -- the
# ``paths.*`` config values are resolved relative to ``model_experiment/``.
_PKG_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _PKG_DIR.parent
_DEFAULT_CONFIG = _PKG_DIR / "config_synth.yaml"

# The two directionality runs: (label, benchmark, data_tag, run_tag).
# The reverse cell uses the dedicated G1-rev benchmark config, which sets
# reverse_roles=True so the slot-wise TE is 0 by construction.
_DIRECTIONS = (
    ("forward", "G1",     "G1_dir_forward",     "directionality/forward"),
    ("reverse", "G1-rev", "G1-rev_dir_reverse", "directionality/reverse"),
)

# Columns of the directionality summary CSV (one row per direction).
_SUMMARY_FIELDS = [
    "direction", "benchmark", "data_tag", "run_tag", "te_true",
    "k_bar", "k_bar_shuffled", "pred_gap", "feat_loss", "epoch", "ckpt_path",
]


def _run_direction(
    config: Dict[str, Any],
    *,
    label: str,
    benchmark: str,
    data_tag: str,
    run_tag: str,
    build_missing: bool,
    train_missing: bool,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    """Build (opt-in), train (opt-in) and evaluate one directionality run.

    Args:
        config: The parsed config.
        label: ``"forward"`` or ``"reverse"``.
        benchmark: The benchmark block to use (``"G1"`` forward, ``"G1-rev"`` reverse).
        data_tag: Cache tag for this direction's dataset.
        run_tag: Results subdirectory name for this direction's checkpoint.
        build_missing: If True, generate the dataset when it is not cached.
        train_missing: If True, train the model when its checkpoint is missing.
        device: Compute device.

    Returns:
        The :func:`evaluate_te.evaluate_checkpoint` row augmented with
        ``direction``, or ``None`` when the dataset / checkpoint is missing and
        the corresponding opt-in flag is off.
    """
    cfg = deepcopy(config)
    cfg["experiment"]["benchmark"] = benchmark
    tm.resolve_active_benchmark(cfg)

    cache_dir = ev._data_root(cfg) / benchmark / data_tag
    ckpt_path = ev._results_root(cfg) / benchmark / run_tag / "final.ckpt"

    if not (cache_dir / "test.npz").is_file():
        if build_missing:
            print(f"  [build] {label}: dataset '{data_tag}'")
            bcfg = deepcopy(cfg)
            bd._apply_overrides(bcfg, {"tag": data_tag})
            bd.build_dataset(bcfg, force=False)
        else:
            print(f"  [skip ] {label}: dataset '{data_tag}' not cached "
                  f"(pass --build-missing to generate it)")
            return None

    if not ckpt_path.is_file():
        if train_missing:
            print(f"  [train] {label}: {run_tag}")
            tcfg = deepcopy(cfg)
            tm.train(tcfg, overrides={"data_tag": data_tag, "run_tag": run_tag})
        else:
            print(f"  [skip ] {label}: checkpoint not found ({ckpt_path}) "
                  f"(pass --train-missing to train it)")
            return None

    row = ev.evaluate_checkpoint(
        ckpt_path, cfg, device=device, data_tag=data_tag
    )
    row["direction"] = label
    row["run_tag"] = run_tag
    return row


def run_directionality(
    config: Dict[str, Any],
    *,
    device: Optional[torch.device] = None,
    build_missing: bool = False,
    train_missing: bool = False,
) -> Dict[str, Any]:
    r"""Run the forward-vs-reverse directionality test (task 7.4).

    Args:
        config: The parsed ``config_synth.yaml`` (must carry the ``G1`` and
            ``G1-rev`` benchmark blocks).
        device: Compute device. Defaults to :func:`train_minimal.resolve_device`.
        build_missing: If True, generate any missing directionality dataset.
        train_missing: If True, train any missing directionality model.

    Returns:
        A results dict: ``rows`` (per-direction), ``comparison`` (the
        $\bar K$ comparison + verdict), ``out_dir`` and ``skipped``.
    """
    device = device or tm.resolve_device(config["runtime"])
    out_dir = ev._results_root(config) / "directionality"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[directionality] forward (Benchmark G1) vs reverse (Benchmark G1-rev)  "
        f"device={device}  build_missing={build_missing}  "
        f"train_missing={train_missing}"
    )

    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for label, benchmark, data_tag, run_tag in _DIRECTIONS:
        try:
            row = _run_direction(
                config, label=label, benchmark=benchmark,
                data_tag=data_tag, run_tag=run_tag,
                build_missing=build_missing, train_missing=train_missing,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001 - one bad run must not abort
            print(f"  [error] {label}: {type(exc).__name__}: {exc}")
            skipped.append(label)
            continue
        if row is None:
            skipped.append(label)
            continue
        rows.append(row)

    by_dir = {r["direction"]: r for r in rows}
    k_fwd = float(by_dir.get("forward", {}).get("k_bar", float("nan")))
    k_rev = float(by_dir.get("reverse", {}).get("k_bar", float("nan")))
    ratio = (
        k_fwd / k_rev
        if (np.isfinite(k_fwd) and np.isfinite(k_rev) and abs(k_rev) > 1e-9)
        else float("nan")
    )
    # A strong directionality result is K_fwd >> K_rev. The ratio > 5 gate is
    # a heuristic (see model_validation.md success criterion 5); the verdict is
    # None until both directions have been evaluated.
    verdict = (
        bool(np.isfinite(ratio) and ratio > 5.0)
        if len(rows) == 2 else None
    )
    comparison = {
        "k_bar_forward": k_fwd,
        "k_bar_reverse": k_rev,
        "te_true_forward": float(
            by_dir.get("forward", {}).get("te_true", float("nan"))
        ),
        "te_true_reverse": float(
            by_dir.get("reverse", {}).get("te_true", float("nan"))
        ),
        "directionality_ratio": ratio,
        "verdict_direction_specific": verdict,
        "note": (
            "verdict_direction_specific is K_bar_forward / K_bar_reverse > 5 "
            "(a heuristic gate -- model_validation.md success criterion 5). "
            "It is None until both directions have a converged checkpoint; "
            "the training runs are deferred to the multi-GPU box."
        ),
    }

    _write_summary_csv(rows, out_dir / "summary.csv")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "created": datetime.now(timezone.utc).isoformat(),
                "comparison": comparison,
                "rows": [{k: r.get(k) for k in _SUMMARY_FIELDS} for r in rows],
                "skipped": skipped,
            },
            fh, indent=2,
        )
    if len(rows) == 2:
        _make_plot(by_dir, comparison, out_dir)
    else:
        print("[plot] both directions are needed -- skipping the bar plot "
              "(the directionality training runs are deferred; see the plan).")

    print(
        f"\n[directionality] forward K_bar={k_fwd:.5f}  "
        f"reverse K_bar={k_rev:.5f}  ratio={ratio:.3f}  "
        f"verdict={verdict}\n"
        f"[done] {len(rows)} direction(s) evaluated, {len(skipped)} skipped\n"
        f"       artifacts -> {out_dir}"
    )
    return {
        "rows": rows,
        "comparison": comparison,
        "out_dir": str(out_dir),
        "skipped": skipped,
    }


def _write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write the per-direction summary CSV.

    Args:
        rows: The per-direction evaluation rows.
        path: Destination CSV path (overwritten).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in _SUMMARY_FIELDS})


def _make_plot(
    by_dir: Dict[str, Dict[str, Any]],
    comparison: Dict[str, Any],
    out_dir: Path,
) -> None:
    r"""Render the two-bar $\bar K_{\rm forward}$ vs $\bar K_{\rm reverse}$ plot.

    Writes ``kbar_forward_vs_reverse.{pdf,png}`` using the shared publication
    style in :mod:`plot_style`.

    Args:
        by_dir: Per-direction rows keyed by ``direction``.
        comparison: The comparison dict from :func:`run_directionality`.
        out_dir: Destination directory.
    """
    import matplotlib.pyplot as plt

    from model.vae_teb_prediction.model.model_experiment.synthetic import (
        plot_style as ps,
    )

    ps.apply_style()

    labels = ["forward\n$X\\to Y$", "reverse\n$Y\\to X$"]
    kbars = [comparison["k_bar_forward"], comparison["k_bar_reverse"]]
    te = [comparison["te_true_forward"], comparison["te_true_reverse"]]

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    bars = ax.bar(labels, kbars, color=[ps.COLOR_BLUE, ps.COLOR_VERMILLION],
                  width=0.6, edgecolor=ps.COLOR_BLACK, linewidth=0.6)
    for bar, t in zip(bars, te):
        ax.annotate(
            f"te_true={t:.3g}", (bar.get_x() + bar.get_width() / 2,
                                 bar.get_height()),
            ha="center", va="bottom", fontsize=ps.FONT_LEGEND,
            color=ps.COLOR_BLACK,
            textcoords="offset points", xytext=(0, 3),
        )
    ratio = comparison["directionality_ratio"]
    ax.set_title(
        r"directionality: $\bar K$ forward vs reverse"
        + (f"   (ratio={ratio:.2f})" if np.isfinite(ratio) else "")
    )
    ax.set_ylabel(r"$\bar K$ (nats)")
    ax.margins(y=0.12)
    ps.style_axes(ax)
    fig.tight_layout()
    ps.save_figure(fig, out_dir / "kbar_forward_vs_reverse")


# =============================================================================
# Overrides + dispatch
# =============================================================================

def _apply_overrides(
    config: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply the config-level overrides onto ``config`` in place.

    Only ``device`` / ``seed`` are config fields; ``build_missing`` /
    ``train_missing`` are call arguments handled by :func:`_dispatch`.

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
    # data_dir / results_dir overrides -> config["paths"] (None -> YAML default).
    tm.apply_path_overrides(config, overrides)
    return config


def _dispatch(
    config: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Resolve overrides, seed, device and run the directionality test.

    Args:
        config: The parsed ``config_synth.yaml``.
        overrides: Flat overrides (from ``vars(args)`` or ``RUN_CONFIG``).

    Returns:
        The :func:`run_directionality` result dict.
    """
    config = deepcopy(config)
    _apply_overrides(config, overrides)
    tm.set_seed(int(config["experiment"].get("seed", 0)))
    device = tm.resolve_device(config["runtime"])
    return run_directionality(
        config,
        device=device,
        build_missing=bool(overrides.get("build_missing")),
        train_missing=bool(overrides.get("train_missing")),
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
        description="Directionality test (forward vs reverse-roles K_bar) for "
                    "SeqVaeLagAttnV1 on synthetic data (task 7.4)."
    )
    p.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG,
        help="path to config_synth.yaml",
    )
    p.add_argument(
        "--build-missing", action=argparse.BooleanOptionalAction, default=False,
        dest="build_missing",
        help="generate any missing directionality dataset (opt-in)",
    )
    p.add_argument(
        "--train-missing", action=argparse.BooleanOptionalAction, default=False,
        dest="train_missing",
        help="train any missing directionality model (opt-in, multi-hour)",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="override runtime.device (auto / cpu / cuda / cuda:N)",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="override experiment.seed",
    )
    p.add_argument(
        "--data-dir", type=str, default=None, dest="data_dir",
        help="override paths.data_dir (absolute/relative path, ~, or $VAR); "
             "None -> config paths.data_dir",
    )
    p.add_argument(
        "--results-dir", type=str, default=None, dest="results_dir",
        help="override paths.results_dir (same format as --data-dir); "
             "None -> config paths.results_dir",
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
    # How to run this script  (project convention -- Decision D9)
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
        "build_missing": False,    # generate the forward / reverse datasets
        "train_missing": False,    # train the forward / reverse models
        "device": None,            # None -> config runtime.device
        "seed": None,              # None -> config experiment.seed
        "data_dir": None,          # None -> config paths.data_dir
        "results_dir": None,       # None -> config paths.results_dir
    }

    if len(sys.argv) > 1:
        main()                              # CLI mode -- argparse
    else:
        config = tm.load_config(CONFIG_PATH)
        # --- optional: tweak any config value not covered by RUN_CONFIG ------
        # e.g.  config["optim"]["epochs"] = 50
        # ---------------------------------------------------------------------
        _dispatch(config, RUN_CONFIG)
