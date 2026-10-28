r"""Summaries and example pages of the signal-quality fields of finished HDF5 files.

``build_fold_datasets.py`` calls :func:`write_quality_report` at the end of a build with the
signal-quality step. The report reads the files back, so every number and every page shows what
a training loader reads. The bit names and the invalid bits come from the attributes of
``up_quality`` and ``fhr_quality``, so this module does not import the rules.

Outputs in ``output_dir``:

* ``quality_summary.csv``: one row per file: segments, valid fractions, bit shares, and the share
  of usable coefficients ($\rho \le \varepsilon$ after the warm-up) per block;
* ``REPORT.md``: the same table, and the list of pages;
* ``examples/``: one page per example segment, and ``examples.pdf`` with all of them.

The pages hold clinical traces. Keep the folder on the production box.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Rows read per file for the bit shares, the coefficient shares and the example pool.
SAMPLE_ROWS = 1000
#: The four coefficient blocks and their contamination datasets.
BLOCKS = ("up_st", "up_ph", "fhr_st", "fhr_ph")
#: Example categories: name -> (title, column of the sampled rows ranked high to low).
EXAMPLE_CATEGORIES = {
    "oscillation": ("UA regular oscillation (kept valid)", "ua_oscillation"),
    "step": ("UA baseline step", "ua_step"),
    "transient": ("UA transient", "ua_transient"),
    "censored": ("UA clamped crest (censored)", "ua_censored"),
    "island": ("UA island in an artifact region", "ua_island"),
    "code": ("UA invalid code (above 127)", "ua_code"),
    "hold_floor": ("UA hold or floor", "ua_hold_floor"),
    "fhr_repaired": ("FHR repair", "fhr_repaired"),
    "low_valid": ("Least valid UA", "ua_invalid_share"),
    "clean": ("Clean UA (control)", "clean"),
}

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")


def _bits(dataset: Any) -> Dict[str, int]:
    """Bit name -> value, from the ``bits`` attribute of a quality dataset."""
    return {name: int(value) for name, value in json.loads(dataset.attrs["bits"]).items()}


def _sample(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sorted row indices: every row, or :data:`SAMPLE_ROWS` drawn without replacement."""
    if n <= SAMPLE_ROWS:
        return np.arange(n)
    return np.sort(rng.choice(n, size=SAMPLE_ROWS, replace=False))


def summarise_file(path: str, label: str, epsilon: float,
                   rng: np.random.Generator) -> Tuple[Dict[str, Any], pd.DataFrame]:
    r"""Summarise one file and return its sampled rows for the example pool.

    Args:
        path: The HDF5 file.
        label: Its name in the report, e.g. its path below the build directory.
        epsilon: A coefficient is usable when $\rho \le \varepsilon$.
        rng: Generator of the row sample.

    Returns:
        ``(summary, rows)``: one summary row, and one row per sampled segment with its bit
        shares, ready for :func:`choose_examples`.
    """
    import h5py

    with h5py.File(path, "r") as f:
        n = int(f["fhr"].shape[0])
        summary: Dict[str, Any] = {"file": label, "segments": n,
                                   "quality_version": int(f.attrs.get("quality_version", -1))}
        if n == 0:
            return summary, pd.DataFrame()
        up_valid = f["up_valid_frac"][:]
        summary.update(
            guids=len(set(f["guid"].asstr()[:])),
            up_valid=float(up_valid.mean()),
            segments_up_valid_below_half=float(np.mean(up_valid < 0.5)),
            up_censored=float(f["up_censored_frac"][:].mean()),
            fhr_valid=float(f["fhr_valid_frac"][:].mean()),
            fhr_repair_exact=float(f["fhr_repair_exact"][:].mean()),
        )
        rows = _sample(n, rng)
        ua_bits, fhr_bits = _bits(f["up_quality"]), _bits(f["fhr_quality"])
        ua = f["up_quality"][rows].astype(np.int64)
        fh = f["fhr_quality"][rows].astype(np.int64)
        table = pd.DataFrame({"row": rows, "guid": f["guid"].asstr()[rows],
                              "epoch": f["epoch"][rows], "up_valid_frac": up_valid[rows]})
        for name, value in ua_bits.items():
            table[f"ua_{name.lower()}"] = np.mean((ua & value) != 0, axis=1)
        for name, value in fhr_bits.items():
            table[f"fhr_{name.lower()}"] = np.mean((fh & value) != 0, axis=1)
        for column in table.columns:
            if column.startswith(("ua_", "fhr_")):
                summary[column] = float(table[column].mean())
        scale = float(f.attrs.get("quality_rho_scale", 255))
        for block in BLOCKS:
            rho = f[f"{block}_rho"][rows]
            warmup = np.asarray(f[block].attrs["causal_warmup_steps"])
            after = np.arange(rho.shape[-1])[None, :] >= warmup[:, None]
            usable = (rho <= epsilon * scale) & after[None]
            summary[f"usable_{block}"] = float(usable.sum() / max(after.sum() * len(rows), 1))
    table["file"] = label
    table["path"] = path
    table["ua_hold_floor"] = table.get("ua_hold", 0.0) + table.get("ua_floor", 0.0)
    table["ua_invalid_share"] = 1.0 - table["up_valid_frac"]
    table["clean"] = (table["up_valid_frac"] >= 0.999).astype(float)
    return summary, table


def choose_examples(rows: pd.DataFrame, n_examples: int,
                    rng: np.random.Generator) -> List[Tuple[str, Dict[str, Any]]]:
    """Pick example segments per category: half the strongest, half at random, one per GUID.

    Args:
        rows: The sampled rows of every file.
        n_examples: Pages per category.
        rng: Generator of the random half.

    Returns:
        ``[(category, row), ...]``.
    """
    picks: List[Tuple[str, Dict[str, Any]]] = []
    for category, (_, column) in EXAMPLE_CATEGORIES.items():
        if column not in rows.columns:
            continue
        pool = rows[rows[column] > 0]
        if pool.empty:
            continue
        if category == "clean":
            pool = pool.iloc[rng.permutation(len(pool))].drop_duplicates("guid")
            top = pool.head(n_examples)
        else:
            pool = pool.sort_values(column, ascending=False).drop_duplicates("guid")
            top = pool.head((n_examples + 1) // 2)
            rest = pool.drop(top.index)
            if len(rest):
                top = pd.concat([top, rest.iloc[rng.permutation(len(rest))[: n_examples - len(top)]]])
        picks.extend((category, row.to_dict()) for _, row in top.iterrows())
    return picks


def _style() -> Any:
    """Import matplotlib headless and apply the report style."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 8.5,
        "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "grid.color": GRID, "grid.linewidth": 0.6,
    })
    return plt


def _bit_strip(ax: Any, bits: np.ndarray, names: Mapping[str, int], t: np.ndarray,
               title: str) -> None:
    """One lane per bit, filled where the bit is set."""
    lanes = list(names.items())
    for lane, (_, value) in enumerate(lanes):
        on = np.concatenate([[0], ((bits & value) != 0).astype(np.int8), [0]])
        edges = np.diff(on)
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            ax.axvspan(t[start], t[min(stop, t.size - 1)], ymin=(len(lanes) - lane - 1) / len(lanes),
                       ymax=(len(lanes) - lane) / len(lanes),
                       color=CATEGORICAL[lane % len(CATEGORICAL)], linewidth=0)
    ax.set_yticks((len(lanes) - np.arange(len(lanes)) - 0.5) / len(lanes),
                  [name.lower() for name, _ in lanes], fontsize=6.5)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=8.5)
    ax.grid(axis="x")


def _display(block: np.ndarray) -> np.ndarray:
    """Per-channel robust z-score of ``asinh`` values."""
    a = np.arcsinh(block)
    centre = np.median(a, axis=1, keepdims=True)
    spread = np.subtract(*np.percentile(a, [75, 25], axis=1))[:, None]
    return np.clip((a - centre) / np.where(spread > 1e-9, spread, 1.0), -3, 3)


def plot_example(plt: Any, category: str, row: Mapping[str, Any], epsilon: float,
                 title: Optional[str] = None) -> Any:
    """Draw one example page from the file: signals, bits, coefficients and contamination.

    Args:
        plt: ``matplotlib.pyplot`` with the report style (:func:`_style`).
        category: A key of :data:`EXAMPLE_CATEGORIES`, or any name when *title* is given.
        row: The segment: ``path``, ``row``, ``file``, ``guid``, ``epoch`` and
            ``up_valid_frac``.
        epsilon: A coefficient is usable when $\\rho \\le \\varepsilon$.
        title: The page title; ``None`` takes it from :data:`EXAMPLE_CATEGORIES`.

    Returns:
        The figure.
    """
    import h5py
    from matplotlib.colors import LinearSegmentedColormap

    i = int(row["row"])
    with h5py.File(row["path"], "r") as f:
        up, fhr = f["up"][i], f["fhr"][i]
        up_raw, fhr_raw = f["up_raw"][i].astype(np.float64), f["fhr_raw"][i].astype(np.float64)
        ua_q, fhr_q = f["up_quality"][i].astype(np.int64), f["fhr_quality"][i].astype(np.int64)
        ua_names, fhr_names = _bits(f["up_quality"]), _bits(f["fhr_quality"])
        ua_invalid = (ua_q & int(f["up_quality"].attrs["invalid_bits"])) != 0
        exact = bool(f["fhr_repair_exact"][i])
        fhr_mask = int(f["fhr_quality"].attrs["invalid_bits_exact" if exact else "invalid_bits_rules"])
        fhr_invalid = (fhr_q & fhr_mask) != 0
        coeffs = {block: f[block][i] for block in BLOCKS}
        rho = {block: f[f"{block}_rho"][i] / float(f.attrs.get("quality_rho_scale", 255))
               for block in BLOCKS}
        warm = {block: np.asarray(f[block].attrs["causal_warmup_steps"]) for block in BLOCKS}
    n = up.size
    t = np.arange(n) / 4.0
    decimation = n // coeffs["up_st"].shape[-1]
    fig = plt.figure(figsize=(12, 15), layout="constrained")
    grid = fig.add_gridspec(7, 1, height_ratios=[1.3, 0.8, 1.2, 0.6, 1.4, 1.4, 1.4])
    fig.suptitle(f"{title or EXAMPLE_CATEGORIES[category][0]} · {row['file']} · {row['guid']} · "
                 f"epoch {row['epoch']:.0f} s · UA valid {100 * row['up_valid_frac']:.1f}%",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)

    ax = fig.add_subplot(grid[0])
    ax.fill_between(t, 0, 1, where=ua_invalid, transform=ax.get_xaxis_transform(), color=GRID,
                    linewidth=0, label="invalid (bridged)")
    ax.plot(t, up_raw, color=CATEGORICAL[0], linewidth=0.7, label="up_raw (before the fill)")
    ax.plot(t, up, color=CATEGORICAL[2], linewidth=0.9, label="up (stored, bridged)")
    for level in (100, 128):
        ax.axhline(level, color=AXIS, linewidth=0.8, linestyle="--")
    ax.set_ylim(min(-5.0, float(up_raw.min()) - 5), max(130.0, float(up_raw.max()) + 5))
    ax.set_ylabel("UA (units)")
    ax.set_title("UA (dashed: 100 and 128)")
    ax.legend(loc="upper right", fontsize=7.5, ncols=3)
    ax.grid(True)
    _bit_strip(fig.add_subplot(grid[1], sharex=ax), ua_q, ua_names, t, "UA quality bits")

    ax2 = fig.add_subplot(grid[2], sharex=ax)
    ax2.fill_between(t, 0, 1, where=fhr_invalid, transform=ax2.get_xaxis_transform(), color=GRID,
                     linewidth=0, label="invalid (bridged)")
    ax2.plot(t, np.where(fhr_raw > 0, fhr_raw, np.nan), color=CATEGORICAL[0], linewidth=0.7,
             label="fhr_raw")
    ax2.plot(t, fhr, color=CATEGORICAL[2], linewidth=0.8, label="fhr (stored)")
    lo, hi = np.percentile(fhr, [0.5, 99.5])
    ax2.set_ylim(max(lo - 15, 30), min(hi + 15, 240))
    ax2.set_ylabel("FHR (bpm)")
    ax2.set_title(f"FHR · repair mask {'from the raw channel' if exact else 'from the rules'}")
    ax2.legend(loc="upper right", fontsize=7.5, ncols=3)
    ax2.grid(True)
    _bit_strip(fig.add_subplot(grid[3], sharex=ax), fhr_q, fhr_names, t, "FHR quality bits")

    rho_cmap = LinearSegmentedColormap.from_list("rho", ["#ffffff", "#ffffff", "#ee8f8e", "#b52d2d"])
    end = n / 4.0
    panels = (
        (np.concatenate([coeffs["up_st"], coeffs["up_ph"]]), None, coeffs["up_st"].shape[0],
         "UA coefficients as stored (up_st above the dashed line, up_ph below)"),
        (np.concatenate([rho["up_st"], rho["up_ph"]]),
         np.concatenate([warm["up_st"], warm["up_ph"]]), coeffs["up_st"].shape[0],
         f"UA contamination ρ (white: ρ ≤ {epsilon:g}; yellow: warm-up end)"),
        (np.concatenate([rho["fhr_st"], rho["fhr_ph"]]),
         np.concatenate([warm["fhr_st"], warm["fhr_ph"]]), coeffs["fhr_st"].shape[0],
         "FHR contamination ρ (fhr_st above the dashed line, fhr_ph below)"),
    )
    for k, (block, warmup, split, title) in enumerate(panels):
        axk = fig.add_subplot(grid[4 + k], sharex=ax)
        extent = (0.0, end, block.shape[0], 0)
        if warmup is None:
            image = axk.imshow(_display(block), aspect="auto", cmap="viridis", vmin=-3, vmax=3,
                               extent=extent, interpolation="nearest")
        else:
            image = axk.imshow(block, aspect="auto", cmap=rho_cmap, vmin=0, vmax=1, extent=extent,
                               interpolation="nearest")
            axk.plot(warmup * decimation / 4.0, np.arange(block.shape[0]) + 0.5,
                     color=CATEGORICAL[3], linewidth=1.0)
        axk.axhline(split, color=SURFACE if warmup is None else INK, linewidth=0.8, linestyle="--")
        axk.set_ylabel("channel")
        axk.set_title(title)
        fig.colorbar(image, ax=axk, shrink=0.9, pad=0.01)
    axk.set_xlabel("time in segment (s)")
    return fig


def _markdown(df: pd.DataFrame) -> str:
    """A pipe table of *df*, without ``tabulate``."""
    def cell(value: Any) -> str:
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in values) + " |" for values in df.itertuples(index=False)]
    return "\n".join(lines)


def write_quality_report(files: Sequence[Tuple[str, str]], output_dir: str, n_examples: int = 4,
                         epsilon: float = 0.1, seed: int = 0) -> Dict[str, Any]:
    r"""Summarise the quality fields of *files* and draw example pages.

    Args:
        files: ``(path, label)`` per file. Pass each segment of the build once, e.g. the
            pretraining shards and the partition files of one fold.
        output_dir: A directory for the report. Existing files in it are overwritten.
        n_examples: Pages per category; 0 draws none.
        epsilon: A coefficient is usable when $\rho \le \varepsilon$.
        seed: Seed of the row sample and of the random half of the examples.

    Returns:
        ``{"files": n, "pages": n, "output_dir": str}``.
    """
    out = Path(output_dir)
    (out / "examples").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    summaries, pools = [], []
    for path, label in files:
        summary, rows = summarise_file(path, label, epsilon, rng)
        summaries.append(summary)
        if len(rows):
            pools.append(rows)
    table = pd.DataFrame(summaries)
    table.to_csv(out / "quality_summary.csv", index=False)

    pages: List[str] = []
    if n_examples > 0 and pools:
        from matplotlib.backends.backend_pdf import PdfPages

        plt = _style()
        picks = choose_examples(pd.concat(pools, ignore_index=True), n_examples, rng)
        with PdfPages(out / "examples" / "examples.pdf") as pdf:
            for number, (category, row) in enumerate(picks, start=1):
                try:
                    fig = plot_example(plt, category, row, epsilon)
                except Exception:  # noqa: BLE001 -- one bad page must not cost the others
                    logger.exception(f"quality page {category} {row['guid']} failed")
                    continue
                name = f"{number:02d}_{category}_{row['guid']}_{int(row['row'])}.png"
                fig.savefig(out / "examples" / name, dpi=110)
                pdf.savefig(fig)
                plt.close(fig)
                pages.append(name)

    columns = [c for c in ("file", "segments", "guids", "up_valid", "segments_up_valid_below_half",
                           "up_censored", "ua_oscillation", "ua_transient", "ua_step", "ua_island",
                           "fhr_valid", "fhr_repair_exact", "usable_up_st", "usable_up_ph",
                           "usable_fhr_st", "usable_fhr_ph") if c in table.columns]
    lines = [
        "# Signal quality of the built files",
        "",
        f"Created {time.strftime('%Y-%m-%d %H:%M')}. Files: {len(table)}. A coefficient is usable "
        f"when $\\rho \\le$ {epsilon:g} after its warm-up. Bit shares and usable shares use up to "
        f"{SAMPLE_ROWS} segments per file; the valid fractions use every segment.",
        "",
        "Every column of `quality_summary.csv` is a mean over segments; `ua_<bit>` is the share of",
        "samples with that bit.",
        "",
        _markdown(table[columns]) if len(table) else "No file.",
        "",
        "## Example pages",
        "",
        "One page per segment in `examples/`, all of them in `examples/examples.pdf`.",
        "",
    ]
    lines += [f"- `examples/{name}`" for name in pages]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"quality report: {len(table)} files, {len(pages)} pages in {out}")
    return {"files": int(len(table)), "pages": len(pages), "output_dir": str(out)}
