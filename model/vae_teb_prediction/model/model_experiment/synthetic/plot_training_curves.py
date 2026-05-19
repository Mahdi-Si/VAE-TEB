r"""Train/validation loss-curve figure for the synthetic-TE training loop.

A pure **CSV $\to$ figure** module: it reads the per-epoch ``metrics.csv`` that
:mod:`train_minimal` writes and renders a single multi-panel figure overlaying
the training and validation curves for every loss component, so convergence and
instability are visible at a glance.

The figure has six panels (a $3\times2$ grid):

    * total loss $\mathcal{L}_{\mathrm{total}}$
    * feature loss $\mathcal{L}_{\mathrm{feat}}$
    * baseline loss $\mathcal{L}_{\mathrm{base}}$
    * KL loss $\mathcal{L}_{\mathrm{KL}}$ (log $y$ -- it spans decades)
    * predictive gap ``pred_gap`` $= \mathcal{L}_{\mathrm{base}} -
      \mathcal{L}_{\mathrm{feat}}$ (linear -- it is negative early in training)
    * optimisation diagnostics: learning rate + gradient norm (twin $y$-axes)

This module pulls in **no torch** -- it is a fast, standalone plotting utility.
:mod:`train_minimal` calls :func:`plot_training_curves` periodically during the
epoch loop (lazily imported, behind a ``try/except`` so a plotting failure can
never abort training).

Public API:
    plot_training_curves: Render a run's ``metrics.csv`` into
        ``training_curves.{pdf,png}``. Reusable -- safe to call repeatedly; each
        call overwrites the previous figure.

Run modes (project convention -- see Decision D9 in
``synthetic_te_validation_plan.md``): like every ``synthetic/`` runner this file
supports **both** a CLI and an edit-and-run ``__main__``, auto-detected from
whether any command-line argument is present.

    * CLI mode (any ``--flag`` passed)::

        python -m ...synthetic.plot_training_curves --csv PATH/metrics.csv
        python -m ...synthetic.plot_training_curves --run-tag TAG [--benchmark B]
            [--config PATH] [--formats pdf png]

    * Edit-and-run mode (no arguments) -- edit the ``RUN_CONFIG`` dict in the
      ``__main__`` block, then run the file directly (IDE / notebook)::

        python -m ...synthetic.plot_training_curves
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from model.vae_teb_prediction.model.model_experiment.synthetic import (
    plot_style as ps,
)

# ``synthetic/`` package dir and its parent ``model_experiment/`` -- the
# ``paths.results_dir`` config value is resolved relative to
# ``model_experiment/`` (identical convention to train_minimal.py).
_PKG_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _PKG_DIR.parent
_DEFAULT_CONFIG = _PKG_DIR / "config_synth.yaml"

# Loss-component panels, in grid order. Each tuple is
# ``(title, ylabel, train_col, val_col, log_scale)``; ``log_scale`` is a hint
# honoured only when every plotted value is strictly positive.
_LOSS_PANELS = [
    ("Total loss", r"$\mathcal{L}_{\mathrm{total}}$",
     "train_total_loss", "val_total_loss", False),
    ("Feature loss", r"$\mathcal{L}_{\mathrm{feat}}$",
     "train_feat_loss", "val_feat_loss", False),
    ("Baseline loss", r"$\mathcal{L}_{\mathrm{base}}$",
     "train_base_loss", "val_base_loss", False),
    ("KL divergence", r"$\mathcal{L}_{\mathrm{KL}}$",
     "train_kld_loss", "val_kld_loss", True),
]

# Below this many epochs, draw point markers so a short run stays legible.
_MARKER_MAX_EPOCHS = 30


# =============================================================================
# CSV reading
# =============================================================================

def _to_float(value: Any) -> float:
    """Parse one CSV cell to ``float``, mapping blanks/garbage to ``NaN``.

    Args:
        value: A raw cell value from :class:`csv.DictReader`.

    Returns:
        The parsed float, or ``float("nan")`` when ``value`` is empty,
        ``None``, or not numeric.
    """
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _read_metrics_csv(csv_path: Path) -> Dict[str, np.ndarray]:
    """Read a per-epoch ``metrics.csv`` into per-column float arrays.

    No fixed-schema assumption: every column present in the file becomes a key.
    A missing file or a file with no data rows yields an empty dict, which the
    caller treats as "nothing to plot".

    Args:
        csv_path: Path to the ``metrics.csv`` written by :mod:`train_minimal`.

    Returns:
        A ``{column_name: np.ndarray(dtype=float)}`` mapping, or ``{}`` when the
        file is missing or empty.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return {}
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    columns = list(rows[0].keys())
    return {
        col: np.array([_to_float(r.get(col)) for r in rows], dtype=float)
        for col in columns
    }


def _has_finite(arr: Optional[np.ndarray]) -> bool:
    """Return whether ``arr`` holds at least one finite value.

    Args:
        arr: A column array, or ``None`` when the column is absent.

    Returns:
        ``True`` if ``arr`` is non-``None`` and contains a finite entry.
    """
    return arr is not None and bool(np.isfinite(arr).any())


def _use_log(*arrays: Optional[np.ndarray]) -> bool:
    """Decide whether a log $y$-axis is safe for the given series.

    A log scale is used only when every finite value across all series is
    strictly positive -- otherwise matplotlib would silently drop the
    non-positive points.

    Args:
        *arrays: One or more column arrays (``None`` entries are ignored).

    Returns:
        ``True`` when a log scale would not hide any data.
    """
    finite: List[np.ndarray] = []
    for arr in arrays:
        if arr is None:
            continue
        finite.append(arr[np.isfinite(arr)])
    if not finite:
        return False
    stacked = np.concatenate(finite)
    return stacked.size > 0 and bool(np.all(stacked > 0.0))


# =============================================================================
# Plotting
# =============================================================================

def _plot_curve(
    ax,
    epochs: np.ndarray,
    values: Optional[np.ndarray],
    *,
    color: str,
    label: str,
    marker: Optional[str],
) -> bool:
    """Draw one epoch-indexed curve on ``ax`` if it carries finite data.

    Args:
        ax: The destination matplotlib axes.
        epochs: The shared epoch index ($x$-values).
        values: The series to plot, or ``None`` when its column is absent.
        color: Line colour.
        label: Legend label.
        marker: Point marker, or ``None`` for a plain line.

    Returns:
        ``True`` when a curve was drawn (so the caller can decide on a legend).
    """
    if not _has_finite(values):
        return False
    ax.plot(epochs, values, color=color, label=label, marker=marker)
    return True


def plot_training_curves(
    csv_path: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
    *,
    run_tag: Optional[str] = None,
    formats: Sequence[str] = ("pdf", "png"),
) -> Optional[List[Path]]:
    r"""Render a run's ``metrics.csv`` into a multi-panel loss-curve figure.

    Reads every per-epoch metric and draws a $3\times2$ grid: the four loss
    components, the predictive gap, and an optimisation-diagnostics panel. Each
    loss panel overlays the training and validation curves. The figure is
    written next to the CSV (or to ``out_path``) and overwrites any previous
    render, so it can be refreshed safely during training.

    Args:
        csv_path: Path to the ``metrics.csv`` to plot.
        out_path: Output path *without* extension. Defaults to
            ``<csv parent>/training_curves``.
        run_tag: Run label for the figure title. Defaults to the name of the
            directory containing the CSV.
        formats: Image formats to write (passed to
            :func:`plot_style.save_figure`).

    Returns:
        The list of written file paths, or ``None`` when the CSV is missing or
        has no epochs yet (nothing is rendered in that case).
    """
    import matplotlib.pyplot as plt

    csv_path = Path(csv_path)
    data = _read_metrics_csv(csv_path)
    if not data or "epoch" not in data or not _has_finite(data["epoch"]):
        print(f"[plot] nothing to plot -- no epoch rows in {csv_path}")
        return None

    epochs = data["epoch"]
    n_epochs = int(epochs.size)
    last_epoch = int(np.nanmax(epochs))
    marker = "o" if n_epochs <= _MARKER_MAX_EPOCHS else None
    val_marker = "s" if n_epochs <= _MARKER_MAX_EPOCHS else None
    label = run_tag or csv_path.parent.name

    ps.apply_style()
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 9.5))
    axes = axes.ravel()

    def _finalise(ax, *, drew: bool) -> None:
        """Apply shared axis cosmetics after a panel's curves are drawn."""
        ax.set_xlabel("epoch")
        if drew:
            ax.legend(loc="best")
        ps.style_axes(ax)
        if n_epochs >= 2:
            ps.tighten_xaxis(ax, epochs)
        else:
            ax.set_xlim(epochs[0] - 0.5, epochs[0] + 0.5)

    # --- Panels 1-4: loss components (train vs val) -----------------------
    for ax, (title, ylabel, train_col, val_col, log_hint) in zip(
        axes[:4], _LOSS_PANELS
    ):
        train = data.get(train_col)
        val = data.get(val_col)
        drew_t = _plot_curve(
            ax, epochs, train, color=ps.COLOR_BLUE, label="train", marker=marker
        )
        drew_v = _plot_curve(
            ax, epochs, val, color=ps.COLOR_ORANGE, label="val",
            marker=val_marker,
        )
        if log_hint and _use_log(train, val):
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        _finalise(ax, drew=drew_t or drew_v)

    # --- Panel 5: predictive gap (linear -- it is negative early) ---------
    ax_gap = axes[4]
    ax_gap.axhline(0.0, color=ps.COLOR_GRAY, ls="--", lw=0.8, zorder=1)
    train_gap = data.get("train_pred_gap")
    val_gap = data.get("val_pred_gap")
    drew_t = _plot_curve(
        ax_gap, epochs, train_gap, color=ps.COLOR_BLUE, label="train",
        marker=marker,
    )
    drew_v = _plot_curve(
        ax_gap, epochs, val_gap, color=ps.COLOR_ORANGE, label="val",
        marker=val_marker,
    )
    ax_gap.set_title(r"Predictive gap")
    ax_gap.set_ylabel(r"$\mathcal{L}_{\mathrm{base}} - \mathcal{L}_{\mathrm{feat}}$")
    _finalise(ax_gap, drew=drew_t or drew_v)

    # --- Panel 6: optimisation diagnostics (lr + grad norm, twin axes) ----
    ax_lr = axes[5]
    ax_gn = ax_lr.twinx()
    lr = data.get("lr")
    grad_norm = data.get("train_grad_norm")
    drew_lr = _plot_curve(
        ax_lr, epochs, lr, color=ps.COLOR_GREEN, label="learning rate",
        marker=marker,
    )
    drew_gn = _plot_curve(
        ax_gn, epochs, grad_norm, color=ps.COLOR_PURPLE, label="grad norm",
        marker=marker,
    )
    if drew_lr and _use_log(lr):
        ax_lr.set_yscale("log")
    ax_lr.set_title("Optimisation diagnostics")
    ax_lr.set_xlabel("epoch")
    ax_lr.set_ylabel("learning rate", color=ps.COLOR_GREEN)
    ax_lr.tick_params(axis="y", colors=ps.COLOR_GREEN)
    ax_gn.set_ylabel("gradient norm", color=ps.COLOR_PURPLE)
    ax_gn.tick_params(axis="y", colors=ps.COLOR_PURPLE)
    ax_gn.grid(False)
    ps.style_axes(ax_lr)
    if n_epochs >= 2:
        ps.tighten_xaxis(ax_lr, epochs)
    else:
        ax_lr.set_xlim(epochs[0] - 0.5, epochs[0] + 0.5)
    handles_lr, labels_lr = ax_lr.get_legend_handles_labels()
    handles_gn, labels_gn = ax_gn.get_legend_handles_labels()
    if handles_lr or handles_gn:
        ax_lr.legend(handles_lr + handles_gn, labels_lr + labels_gn, loc="best")

    fig.suptitle(
        f"Training curves -- {label}  (epoch {last_epoch})",
        fontsize=ps.FONT_TITLE,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    out = Path(out_path) if out_path is not None else csv_path.parent / "training_curves"
    return ps.save_figure(fig, out, formats=tuple(formats))


# =============================================================================
# CLI
# =============================================================================

def _resolve_csv_path(
    config_path: Path,
    run_tag: Optional[str],
    benchmark: Optional[str],
) -> Path:
    """Resolve a run's ``metrics.csv`` path from its tag and benchmark.

    Mirrors the ``<results_root>/<benchmark>/<run_tag>/`` layout that
    :mod:`train_minimal` writes (``results_root`` is ``paths.results_dir``
    resolved relative to ``model_experiment/``).

    Args:
        config_path: Path to ``config_synth.yaml`` -- supplies ``results_dir``
            and the default benchmark.
        run_tag: The run subdirectory name (required).
        benchmark: The benchmark letter; defaults to
            ``experiment.benchmark`` from the config.

    Returns:
        The resolved ``metrics.csv`` path.

    Raises:
        ValueError: If ``run_tag`` is not given.
    """
    if not run_tag:
        raise ValueError(
            "pass --csv PATH, or --run-tag TAG to resolve it from the config."
        )
    # Lazy import: only the tag-resolution path needs the (torch-importing)
    # train_minimal config loader -- the --csv path stays torch-free.
    from model.vae_teb_prediction.model.model_experiment.synthetic.train_minimal import (
        load_config,
    )

    config = load_config(config_path)
    bench = benchmark or str(config["experiment"]["benchmark"])
    results_root = (
        _EXPERIMENT_DIR / str(config["paths"]["results_dir"])
    ).resolve()
    return results_root / bench / run_tag / "metrics.csv"


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        The parsed :class:`argparse.Namespace`.
    """
    p = argparse.ArgumentParser(
        description="Render train/validation loss curves from a synthetic-TE "
                    "training run's metrics.csv."
    )
    p.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG,
        help="path to config_synth.yaml (used to resolve --run-tag)",
    )
    p.add_argument(
        "--csv", type=str, default=None,
        help="direct path to a metrics.csv (takes priority over --run-tag)",
    )
    p.add_argument(
        "--run-tag", type=str, default=None, dest="run_tag",
        help="run subdirectory name -- resolves "
             "<results_dir>/<benchmark>/<run-tag>/metrics.csv",
    )
    p.add_argument(
        "--benchmark", type=str, default=None,
        help="benchmark letter for --run-tag (defaults to experiment.benchmark)",
    )
    p.add_argument(
        "--formats", type=str, nargs="+", default=["pdf", "png"],
        help="output image formats (default: pdf png)",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    """CLI entry point: resolve the CSV path and render the figure.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).
    """
    args = parse_args(argv)
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = _resolve_csv_path(args.config, args.run_tag, args.benchmark)

    written = plot_training_curves(
        csv_path, run_tag=args.run_tag, formats=args.formats
    )
    if written:
        print(f"[done] training curves -> {', '.join(str(p) for p in written)}")


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
    # `csv` wins when set; otherwise `run_tag` (+ `benchmark`) resolves the path
    # via config_synth.yaml. `None` means "fall back to the config".
    # =========================================================================

    CONFIG_PATH = _DEFAULT_CONFIG

    RUN_CONFIG = {
        "csv": None,                 # direct metrics.csv path (wins if set)
        "run_tag": "pol_easy_a1",    # else resolve results/<benchmark>/<tag>/
        "benchmark": None,           # None -> config experiment.benchmark
        "formats": ["pdf", "png"],   # output image formats
    }

    if len(sys.argv) > 1:
        main()                       # CLI mode -- argparse
    else:
        if RUN_CONFIG["csv"]:
            csv_path = Path(RUN_CONFIG["csv"])
        else:
            csv_path = _resolve_csv_path(
                CONFIG_PATH, RUN_CONFIG["run_tag"], RUN_CONFIG["benchmark"]
            )
        written = plot_training_curves(
            csv_path, run_tag=RUN_CONFIG["run_tag"],
            formats=RUN_CONFIG["formats"],
        )
        if written:
            print(
                f"[done] training curves -> "
                f"{', '.join(str(p) for p in written)}"
            )
