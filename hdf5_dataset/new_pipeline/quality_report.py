r"""Summaries and example pages of the signal-quality fields of finished HDF5 files.

``build_fold_datasets.py`` calls :func:`write_quality_report` at the end of a build with the
signal-quality step. The report reads the files back, so every number and every page shows what
a training loader reads. The bit names and the invalid bits come from the attributes of
``up_quality`` and ``fhr_quality``, so this module does not import the rules, and a bit that a
later rule version adds (``REST``, ``PUSH`` and ``FLAT`` in version 4) gets its column and its lane
without a change here.

Outputs in ``output_dir``:

* ``quality_summary.csv``: one row per file: segments, valid fractions, bit shares, the share
  of usable coefficients ($\rho \le \varepsilon$ after the warm-up) per block, and
  ``usable_ua_steps``, the share of steps where every UA coefficient is usable;
* ``REPORT.md``: the same table, and the list of pages;
* ``examples/``: one page per example segment, and ``examples.pdf`` with all of them. A page
  shows the raw and the bridged signals, the bits (red: invalid; green: valid, recorded only),
  the stored UA coefficients and the contamination $\rho$ of every block; :func:`plot_example`
  lists the rows. Given the build's transform, it also shows the UA coefficients of ``up_raw``,
  before the mask and the bridge, on the colour scale of the stored ones.

The pages hold clinical traces. Keep the folder on the production box.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

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
    "rest": ("UA clamped rest (kept valid)", "ua_rest"),
    "push": ("UA second-stage push (kept valid)", "ua_push"),
    "flat": ("UA flat line past the rest limit", "ua_flat"),
    "island": ("UA island in an artifact region", "ua_island"),
    "code": ("UA invalid code (above 127)", "ua_code"),
    "hold_floor": ("UA hold or floor", "ua_hold_floor"),
    "fhr_repaired": ("FHR repair", "fhr_repaired"),
    "low_valid": ("Least valid UA", "ua_invalid_share"),
    "clean": ("Clean UA (control)", "clean"),
}

#: Largest difference between the recomputed and the stored coefficients, as a share of the
#: block's largest value, at which the raw panel still counts as drawn with the build's transform.
RECOMPUTE_TOLERANCE = 1e-3

#: ``transform(fhr, up)`` with ``(B, N)`` ``float32`` signals returns one ``{block: (C, T)}`` per
#: row, or ``None`` for a row that failed.
Transform = Callable[[np.ndarray, np.ndarray], Sequence[Optional[Mapping[str, np.ndarray]]]]

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
#: Lane colours of the bit strips: a bit that invalidates the sample, and a bit that does not.
INVALID_COLOR, VALID_COLOR = "#d0473c", "#1baf7a"


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
        if "fhr_raw_match" in f:
            match = f["fhr_raw_match"][:]
            summary["fhr_raw_match"] = (float(np.nanmean(match)) if np.isfinite(match).any()
                                        else float("nan"))
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
        summary["usable_ua_steps"] = _usable_steps(f, rows, ("up_st", "up_ph"), epsilon * scale)
    table["file"] = label
    table["path"] = path
    table["ua_hold_floor"] = table.get("ua_hold", 0.0) + table.get("ua_floor", 0.0)
    table["ua_invalid_share"] = 1.0 - table["up_valid_frac"]
    table["clean"] = (table["up_valid_frac"] >= 0.999).astype(float)
    return summary, table


def _usable_steps(f: Any, rows: np.ndarray, blocks: Sequence[str], threshold: float) -> float:
    r"""Return the share of steps where every channel of ``blocks`` is usable.

    A step $t$ counts when it is at or after the largest warm-up of the channels, and it is usable
    when $\rho_{c,t} \le \varepsilon$ for every channel $c$. This is the conjunction that the
    model's source mask takes, so the share is the reference for its ``source_mask_fraction``. The
    model keeps a subset of the channels (42 of the UA channels), so its share can be higher.

    Args:
        f: The open HDF5 file.
        rows: The sampled rows.
        blocks: The coefficient blocks, e.g. ``("up_st", "up_ph")``.
        threshold: $\varepsilon$ in stored units, i.e. multiplied by ``quality_rho_scale``.

    Returns:
        The share, or NaN when no step is after the warm-up.
    """
    rho = np.concatenate([f[f"{block}_rho"][rows] for block in blocks], axis=1)
    warmup = max(int(np.max(f[block].attrs["causal_warmup_steps"])) for block in blocks)
    ok = np.all(rho[:, :, warmup:] <= threshold, axis=1)
    return float(ok.mean()) if ok.size else float("nan")


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
               title: str, invalid: int) -> None:
    """One lane per bit, filled where the bit is set.

    A bit in *invalid* removes the sample from the mask and is drawn in :data:`INVALID_COLOR`;
    any other bit only records what the rules saw (``CENSORED``, ``REST``, ``PUSH``,
    ``OSCILLATION``), is drawn in :data:`VALID_COLOR`, and its lane name ends in a check mark.
    """
    lanes = list(names.items())
    for lane, (_, value) in enumerate(lanes):
        on = np.concatenate([[0], ((bits & value) != 0).astype(np.int8), [0]])
        edges = np.diff(on)
        color = INVALID_COLOR if value & invalid else VALID_COLOR
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            ax.axvspan(t[start], t[min(stop, t.size - 1)], ymin=(len(lanes) - lane - 1) / len(lanes),
                       ymax=(len(lanes) - lane) / len(lanes), color=color, linewidth=0)
    labels = [name.lower() + ("" if value & invalid else " ✓") for name, value in lanes]
    ax.set_yticks((len(lanes) - np.arange(len(lanes)) - 0.5) / len(lanes), labels, fontsize=6.5)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=8.5)
    ax.grid(axis="x")


def _display(block: np.ndarray, reference: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-channel robust z-score of ``asinh`` values.

    Args:
        block: ``(C, T)`` coefficients.
        reference: ``(C, T)`` coefficients that set each channel's median and interquartile
            range; ``None`` uses *block*. Two panels drawn against one reference share one
            colour scale, so their colours compare.
    """
    a = np.arcsinh(block)
    r = a if reference is None else np.arcsinh(reference)
    centre = np.median(r, axis=1, keepdims=True)
    spread = np.subtract(*np.percentile(r, [75, 25], axis=1))[:, None]
    return np.clip((a - centre) / np.where(spread > 1e-9, spread, 1.0), -3, 3)


def _recompute(transform: Optional[Transform], fhr: Sequence[np.ndarray], up: Sequence[np.ndarray],
               stored: Mapping[str, np.ndarray]) -> Tuple[Optional[Dict[str, np.ndarray]], str]:
    """Transform the raw signals, and check the transform against the stored coefficients.

    The batch holds the raw signals and the stored (bridged) ones. The stored ones must give the
    stored ``up_st`` and ``up_ph`` back, within :data:`RECOMPUTE_TOLERANCE` of each block's
    largest value; otherwise the raw panel is not drawn with the build's transform, and the page
    says so.

    Args:
        transform: See :func:`write_quality_report`; ``None`` skips the panel.
        fhr: ``(fhr_raw, fhr)`` of the segment.
        up: ``(up_raw, up)`` of the segment.
        stored: The stored coefficient blocks of the segment.

    Returns:
        ``(raw_blocks, note)``: the blocks of the raw signals, or ``None``; and a note for the
        panel title, empty when the check passed.
    """
    if transform is None:
        return None, ""
    try:
        out = transform(np.stack(fhr).astype(np.float32), np.stack(up).astype(np.float32))
    except Exception:  # noqa: BLE001 -- the page draws without the panel
        logger.exception("quality page: the raw transform failed")
        return None, ""
    if out is None or len(out) != 2 or out[0] is None or out[1] is None:
        return None, ""
    raw, again = out
    note = ""
    for block in ("up_st", "up_ph"):
        scale = max(float(np.max(np.abs(stored[block]))), 1e-12)
        if float(np.max(np.abs(np.asarray(again[block]) - stored[block]))) > RECOMPUTE_TOLERANCE * scale:
            note = " · WARNING: recomputed stored blocks differ, see the log"
            logger.warning(f"quality page: the recomputed {block} differs from the stored one")
    return {block: np.asarray(raw[block]) for block in ("up_st", "up_ph")}, note


def plot_example(plt: Any, category: str, row: Mapping[str, Any], epsilon: float,
                 title: Optional[str] = None, transform: Optional[Transform] = None) -> Any:
    r"""Draw one example page from the file: signals, bits, coefficients and contamination.

    The rows, top to bottom:

    1. UA: ``up_raw`` (before the fill), the stored bridged ``up``, and the invalid samples shaded.
    2. The UA bits, one lane each: red lanes invalidate the sample, green lanes (marked with a check)
       only record what the rules saw.
    3. FHR, as row 1. The title names the repair mask and ``fhr_raw_match``.
    4. The FHR bits, as row 2, against the mask the segment uses.
    5. With *transform* only: the UA coefficients of ``up_raw``, before the mask and the bridge.
    6. The UA coefficients as stored, computed from the bridged ``up``.
    7. The UA contamination $\rho$: invalid samples and level shifts.
    8. The FHR contamination $\rho$.

    Rows 5 and 6 share one colour scale per channel, set by the stored coefficients, so a
    colour means the same value in both. A value of the raw panel beyond that scale saturates.

    Args:
        plt: ``matplotlib.pyplot`` with the report style (:func:`_style`).
        category: A key of :data:`EXAMPLE_CATEGORIES`, or any name when *title* is given.
        row: The segment: ``path``, ``row``, ``file``, ``guid``, ``epoch`` and
            ``up_valid_frac``.
        epsilon: A coefficient is usable when $\rho \le \varepsilon$.
        title: The page title; ``None`` takes it from :data:`EXAMPLE_CATEGORIES`.
        transform: The build's causal transform (see :func:`write_quality_report`); ``None``
            leaves out row 5.

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
        ua_mask = int(f["up_quality"].attrs["invalid_bits"])
        ua_invalid = (ua_q & ua_mask) != 0
        exact = bool(f["fhr_repair_exact"][i])
        fhr_mask = int(f["fhr_quality"].attrs["invalid_bits_exact" if exact else "invalid_bits_rules"])
        fhr_invalid = (fhr_q & fhr_mask) != 0
        raw_match = float(f["fhr_raw_match"][i]) if "fhr_raw_match" in f else float("nan")
        coeffs = {block: f[block][i] for block in BLOCKS}
        rho = {block: f[f"{block}_rho"][i] / float(f.attrs.get("quality_rho_scale", 255))
               for block in BLOCKS}
        warm = {block: np.asarray(f[block].attrs["causal_warmup_steps"]) for block in BLOCKS}
    raw_blocks, note = _recompute(transform, (fhr_raw, fhr), (up_raw, up), coeffs)
    n = up.size
    t = np.arange(n) / 4.0
    decimation = n // coeffs["up_st"].shape[-1]
    heights = [1.3, 1.1, 1.2, 0.6] + ([1.4] if raw_blocks is not None else []) + [1.4, 1.4, 1.4]
    fig = plt.figure(figsize=(12, 2.0 + 1.75 * sum(heights)), layout="constrained")
    grid = fig.add_gridspec(len(heights), 1, height_ratios=heights)
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
    _bit_strip(fig.add_subplot(grid[1], sharex=ax), ua_q, ua_names, t,
               "UA quality bits (red: invalid; green ✓: valid, recorded only)", ua_mask)

    ax2 = fig.add_subplot(grid[2], sharex=ax)
    ax2.fill_between(t, 0, 1, where=fhr_invalid, transform=ax2.get_xaxis_transform(), color=GRID,
                     linewidth=0, label="invalid (bridged)")
    ax2.plot(t, np.where(fhr_raw > 0, fhr_raw, np.nan), color=CATEGORICAL[0], linewidth=0.7,
             label="fhr_raw")
    ax2.plot(t, fhr, color=CATEGORICAL[2], linewidth=0.8, label="fhr (stored)")
    lo, hi = np.percentile(fhr, [0.5, 99.5])
    ax2.set_ylim(max(lo - 15, 30), min(hi + 15, 240))
    ax2.set_ylabel("FHR (bpm)")
    match = "" if not np.isfinite(raw_match) else f" · raw channel match {100 * raw_match:.1f}%"
    ax2.set_title(f"FHR · repair mask {'from the raw channel' if exact else 'from the rules'}{match}")
    ax2.legend(loc="upper right", fontsize=7.5, ncols=3)
    ax2.grid(True)
    _bit_strip(fig.add_subplot(grid[3], sharex=ax), fhr_q, fhr_names, t,
               "FHR quality bits (red: invalid; green ✓: valid)", fhr_mask)

    rho_cmap = LinearSegmentedColormap.from_list("rho", ["#ffffff", "#ffffff", "#ee8f8e", "#b52d2d"])
    end = n / 4.0
    ua_stored = np.concatenate([coeffs["up_st"], coeffs["up_ph"]])
    split_up, split_fhr = coeffs["up_st"].shape[0], coeffs["fhr_st"].shape[0]
    panels: List[Tuple[np.ndarray, Optional[np.ndarray], int, str]] = []
    if raw_blocks is not None:
        panels.append((_display(np.concatenate([raw_blocks["up_st"], raw_blocks["up_ph"]]),
                                ua_stored), None, split_up,
                       "UA coefficients of up_raw, before the mask and the bridge "
                       f"(scale of the stored panel below){note}"))
    panels += [
        (_display(ua_stored), None, split_up,
         "UA coefficients as stored, from the bridged up (up_st above the dashed line, up_ph "
         "below)"),
        (np.concatenate([rho["up_st"], rho["up_ph"]]),
         np.concatenate([warm["up_st"], warm["up_ph"]]), split_up,
         f"UA contamination ρ: invalid samples and level shifts (white: ρ ≤ {epsilon:g}; "
         "yellow: warm-up end)"),
        (np.concatenate([rho["fhr_st"], rho["fhr_ph"]]),
         np.concatenate([warm["fhr_st"], warm["fhr_ph"]]), split_fhr,
         "FHR contamination ρ (fhr_st above the dashed line, fhr_ph below)"),
    ]
    first = len(heights) - len(panels)
    for k, (block, warmup, split, label) in enumerate(panels):
        axk = fig.add_subplot(grid[first + k], sharex=ax)
        extent = (0.0, end, block.shape[0], 0)
        if warmup is None:
            image = axk.imshow(block, aspect="auto", cmap="viridis", vmin=-3, vmax=3,
                               extent=extent, interpolation="nearest")
        else:
            image = axk.imshow(block, aspect="auto", cmap=rho_cmap, vmin=0, vmax=1, extent=extent,
                               interpolation="nearest")
            axk.plot(warmup * decimation / 4.0, np.arange(block.shape[0]) + 0.5,
                     color=CATEGORICAL[3], linewidth=1.0)
        axk.axhline(split, color=SURFACE if warmup is None else INK, linewidth=0.8, linestyle="--")
        axk.set_ylabel("channel")
        axk.set_title(label)
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
                         epsilon: float = 0.1, seed: int = 0,
                         transform: Optional[Transform] = None) -> Dict[str, Any]:
    r"""Summarise the quality fields of *files* and draw example pages.

    Args:
        files: ``(path, label)`` per file. Pass each segment of the build once, e.g. the
            pretraining shards and the partition files of one fold.
        output_dir: A directory for the report. Existing files in it are overwritten.
        n_examples: Pages per category; 0 draws none.
        epsilon: A coefficient is usable when $\rho \le \varepsilon$.
        seed: Seed of the row sample and of the random half of the examples.
        transform: The build's causal transform, ``transform(fhr, up)`` on ``(B, N)``
            ``float32`` signals, returning one ``{block: (C, T)}`` per row (:data:`Transform`).
            Each page then adds the UA coefficients of ``up_raw``; the page checks the transform
            against the stored coefficients first (:data:`RECOMPUTE_TOLERANCE`). ``None`` leaves
            the panel out.

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
                    fig = plot_example(plt, category, row, epsilon, transform=transform)
                except Exception:  # noqa: BLE001 -- one bad page must not cost the others
                    logger.exception(f"quality page {category} {row['guid']} failed")
                    continue
                name = f"{number:02d}_{category}_{row['guid']}_{int(row['row'])}.png"
                fig.savefig(out / "examples" / name, dpi=110)
                pdf.savefig(fig)
                plt.close(fig)
                pages.append(name)

    columns = [c for c in ("file", "segments", "guids", "up_valid", "segments_up_valid_below_half",
                           "up_censored", "ua_rest", "ua_push", "ua_flat", "ua_oscillation",
                           "ua_transient", "ua_step", "ua_island", "fhr_valid", "fhr_repair_exact",
                           "fhr_raw_match", "usable_up_st", "usable_up_ph", "usable_ua_steps",
                           "usable_fhr_st", "usable_fhr_ph") if c in table.columns]
    lines = [
        "# Signal quality of the built files",
        "",
        f"Created {time.strftime('%Y-%m-%d %H:%M')}. Files: {len(table)}. A coefficient is usable "
        f"when $\\rho \\le$ {epsilon:g} after its warm-up. Bit shares and usable shares use up to "
        f"{SAMPLE_ROWS} segments per file; the valid fractions use every segment.",
        "",
        "Every column of `quality_summary.csv` is a mean over segments; `ua_<bit>` is the share of",
        "samples with that bit. `ua_censored`, `ua_rest`, `ua_push` and `ua_oscillation` are valid",
        "bits: they record what the rules saw and keep the sample. `fhr_raw_match` is the share of",
        "present samples where the raw FHR channel equals the stored FHR. `usable_ua_steps` is the",
        "share of steps after the warm-up where every UA coefficient is usable: the conjunction the",
        "model's source mask takes, and the reference for its `source_mask_fraction`.",
        "",
        _markdown(table[columns]) if len(table) else "No file.",
        "",
        "## Example pages",
        "",
        "One page per segment in `examples/`, all of them in `examples/examples.pdf`. "
        + ("Each page shows the UA coefficients of `up_raw` before the mask and the bridge."
           if transform is not None else "The raw-coefficient panel is off: no transform."),
        "",
    ]
    lines += [f"- `examples/{name}`" for name in pages]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"quality report: {len(table)} files, {len(pages)} pages in {out}")
    return {"files": int(len(table)), "pages": len(pages), "output_dir": str(out)}
