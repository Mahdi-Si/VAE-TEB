r"""The parts every panel of the run dashboard is built from: the files, the traces and the panels.

:mod:`~teb_vae.lag_attn_cfs.eval.run_dashboard` assembles one HTML page from a finished run
directory. This module holds what its panel builders share:

* :class:`RunFiles` reads the tables and the summaries of a run directory, and raises
  :class:`MissingInput` for a file the directory does not hold. The assembler turns that exception
  into a card that names the missing file, so a run with a few analyses still gets a page.
* The trace helpers (:func:`line`, :func:`band`, :func:`bars`, :func:`heatmap`) return Plotly trace
  dictionaries in one style. Lines are solid and $2.6$ px wide unless a builder asks otherwise.
* The panel constructors (:func:`plot_panel`, :func:`table_panel`, :func:`tiles_panel`,
  :func:`html_panel`) return the dictionaries the page script draws.

**Every plot carries three explanations**, and :func:`plot_panel` requires them: how the plot was
made, what it shows, and how to read it. A fourth text, ``run``, states what this run's numbers
say and is computed by the builder from the data.

The module imports ``numpy``, ``pandas`` and the ``numpy``-only gate thresholds, and no ``torch``,
so the page builds with no model and no GPU.
"""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from teb_vae.lag_attn_cfs.gate_states import GATE_OPEN_FROM, GATE_SHUT_BELOW

#: The summary a run writes, and the pattern of the copies an earlier pass left beside it.
SUMMARY_FILENAME = "summary.json"
SUMMARY_BACKUP_GLOB = "summary.bak.*.json"

#: The result blocks a pass derives from the analyses it ran, rather than from one analysis.
#: A partial pass writes a weaker copy of each: its consistency checks cannot test an analysis
#: that did not run. The merged summary therefore takes these from the pass that ran the most
#: steps, and not from the newest pass.
DERIVED_RESULT_BLOCKS = ("sanity", "coverage")

#: Seconds per stored step. Restated from ``teb_vae.lag_attn.nets.lag_report`` because that
#: module imports ``torch`` and this one must not; ``tests/test_eval_run_dashboard.py`` pins the
#: two values equal.
SECONDS_PER_STEP = 4.0

#: The palette of the evaluation figures, so a quantity keeps one colour on the page and in the
#: PDF beside it.
BLUE = "#2A78D6"
ORANGE = "#EDA100"
GREEN = "#1BAF7A"
PURPLE = "#6250D6"
VERMILLION = "#E4572E"
MAGENTA = "#E87BA4"
GRAY = "#7A808A"
TEAL = "#1A9BB0"

#: A sequence of distinct colours for series with no fixed meaning, such as attention heads.
SERIES = (BLUE, VERMILLION, GREEN, ORANGE, PURPLE, MAGENTA, TEAL, GRAY)

#: Clinical class to colour, the evaluation figures' own table.
CLASS_COLORS: Dict[str, str] = {"healthy": "#1F9D6B", "acidosis": "#E8A33D", "hie": "#C0392B"}

#: The gate populations of the ``gate_open`` analysis: key, label on the page, rule and colour.
#: The two thresholds are the ones the analysis itself uses, so a rule on the page cannot differ
#: from the rule that made the tables.
POPULATIONS: Sequence[Dict[str, str]] = (
    {"key": "all", "label": "All", "rule": "Every scored anchor.", "color": GRAY},
    {"key": "gate_on", "label": "Gate on",
     "rule": f"Anchors with a gate of {GATE_SHUT_BELOW:g} or more.", "color": BLUE},
    {"key": "fully_open", "label": "Fully open",
     "rule": f"Anchors with a gate of {GATE_OPEN_FROM:g} or more.", "color": GREEN},
    {"key": "gate_off", "label": "Gate off",
     "rule": f"Anchors with a gate below {GATE_SHUT_BELOW:g}.", "color": VERMILLION},
)
POPULATION_COLORS: Dict[str, str] = {entry["key"]: entry["color"] for entry in POPULATIONS}
POPULATION_NAMES: Dict[str, str] = {
    "all": "all anchors", "gate_on": "gate on", "fully_open": "fully open", "gate_off": "gate off",
}

#: Default width of a line trace, in pixels.
LINE_WIDTH = 2.6

#: Sentinel of the 16-bit array encoding for a value that does not exist.
_INT16_NAN = -32768


class MissingInput(Exception):
    """A panel needs a file the run directory does not hold.

    Args:
        path: The path relative to the results directory.
        analysis: The analysis that writes the file, when the builder knows it.
    """

    def __init__(self, path: str, analysis: Optional[str] = None) -> None:
        super().__init__(path)
        self.path = str(path)
        self.analysis = analysis


# =================================================================================================
# The run directory
# =================================================================================================
class RunFiles:
    """Read access to one finished run's results directory.

    Args:
        results_dir: The ``eval_results`` directory of a run.

    Attributes:
        root: The results directory.
    """

    def __init__(self, results_dir: Any) -> None:
        self.root = Path(results_dir)
        self._tables: Dict[str, pd.DataFrame] = {}
        self._frames: Dict[Any, pd.DataFrame] = {}
        self._arrays: Dict[str, Dict[str, np.ndarray]] = {}
        self._summary: Optional[Dict[str, Any]] = None

    def exists(self, relative: str) -> bool:
        """Say whether the directory holds a file.

        Args:
            relative: The path relative to the results directory.

        Returns:
            ``True`` when the file exists.
        """
        return (self.root / relative).is_file()

    def require(self, relative: str, analysis: Optional[str] = None) -> Path:
        """Return the path of a file, or raise :class:`MissingInput`.

        Args:
            relative: The path relative to the results directory.
            analysis: The analysis that writes it. Defaults to the first folder of the path.

        Returns:
            The absolute path.

        Raises:
            MissingInput: If the file is absent.
        """
        path = self.root / relative
        if not path.is_file():
            parts = Path(relative).parts
            raise MissingInput(relative, analysis or (parts[0] if len(parts) > 1 else None))
        return path

    def csv(self, relative: str, analysis: Optional[str] = None) -> pd.DataFrame:
        """Read one CSV table of the run, once.

        Args:
            relative: The path relative to the results directory.
            analysis: The analysis that writes it.

        Returns:
            The table. An empty file gives an empty frame.

        Raises:
            MissingInput: If the file is absent.
        """
        if relative not in self._tables:
            path = self.require(relative, analysis)
            try:
                self._tables[relative] = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                self._tables[relative] = pd.DataFrame()
        return self._tables[relative]

    def parquet(self, relative: str, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Read a parquet table of the run, once per set of columns.

        Args:
            relative: The path relative to the results directory.
            columns: The columns to read. Names the file lacks are left out.

        Returns:
            The table. The same frame is returned to every caller, so a caller must not change it.

        Raises:
            MissingInput: If the file is absent.
        """
        key = (relative, None if columns is None else tuple(columns))
        if key not in self._frames:
            path = self.require(relative, "the collection pass")
            if columns is None:
                self._frames[key] = pd.read_parquet(path)
            else:
                import pyarrow.parquet as parquet

                present = set(parquet.read_schema(path).names)
                self._frames[key] = pd.read_parquet(
                    path, columns=[name for name in columns if name in present]
                )
        return self._frames[key]

    def npz(self, relative: str) -> Dict[str, np.ndarray]:
        """Read every array of an ``.npz`` file of the run, once.

        Args:
            relative: The path relative to the results directory.

        Returns:
            The arrays by name.

        Raises:
            MissingInput: If the file is absent.
        """
        if relative not in self._arrays:
            path = self.require(relative, "the collection pass")
            with np.load(path, allow_pickle=False) as handle:
                self._arrays[relative] = {name: handle[name] for name in handle.files}
        return self._arrays[relative]

    def json(self, relative: str) -> Dict[str, Any]:
        """Read one JSON file of the run.

        Args:
            relative: The path relative to the results directory.

        Returns:
            The decoded content.

        Raises:
            MissingInput: If the file is absent.
        """
        with open(self.require(relative), encoding="utf-8") as handle:
            return json.load(handle)

    def glob(self, pattern: str) -> List[str]:
        """List the files that match a pattern.

        Args:
            pattern: A glob pattern relative to the results directory.

        Returns:
            The matching files, as sorted paths relative to the results directory.
        """
        return sorted(
            path.relative_to(self.root).as_posix() for path in self.root.glob(pattern)
            if path.is_file()
        )

    @property
    def summary(self) -> Dict[str, Any]:
        r"""The run's summary, completed from the summaries earlier passes left behind.

        A pass that runs a subset of the analyses writes a summary of that subset and renames the
        earlier summary aside. A directory filled by several passes therefore holds each analysis's
        result block in a different file. This property reads ``summary.json`` and every backup,
        newest first, and keeps the newest copy of each result block and of each headline value
        that is not ``None``.

        **The derived blocks are the exception.** A pass computes its consistency checks and its
        coverage record from the analyses it ran, so a one-analysis pass reports "inconclusive"
        for a check an earlier full pass had decided. The blocks of
        :data:`DERIVED_RESULT_BLOCKS` are therefore taken from the pass that ran the most steps,
        the newest one on a tie. ``derived_blocks_from`` names that file.

        Returns:
            The merged summary. Empty when the directory holds no summary at all.
        """
        if self._summary is None:
            paths = [self.root / SUMMARY_FILENAME] if self.exists(SUMMARY_FILENAME) else []
            paths += sorted(
                self.root.glob(SUMMARY_BACKUP_GLOB), key=lambda p: p.stat().st_mtime, reverse=True
            )
            merged: Dict[str, Any] = {}
            results: Dict[str, Any] = {}
            headline: Dict[str, Any] = {}
            steps: Dict[str, Any] = {}
            widest: Dict[str, Any] = {}
            widest_steps, widest_name = -1, None
            for path in paths:
                try:
                    with open(path, encoding="utf-8") as handle:
                        record = json.load(handle)
                except (OSError, ValueError):
                    continue
                for key, value in record.items():
                    merged.setdefault(key, value)
                block = dict(record.get("results") or {})
                n_steps = len(record.get("steps") or [])
                # Strictly more, so the newest pass keeps a tie: the files are read newest first.
                if n_steps > widest_steps and any(name in block for name in DERIVED_RESULT_BLOCKS):
                    widest, widest_steps, widest_name = block, n_steps, path.name
                for key, value in dict(block.get("headline") or {}).items():
                    if headline.get(key) is None:
                        headline[key] = value
                for key, value in block.items():
                    results.setdefault(key, value)
                for step in record.get("steps") or []:
                    steps.setdefault(str(step.get("name")), step)
            if merged:
                for name in DERIVED_RESULT_BLOCKS:
                    if name in widest:
                        results[name] = widest[name]
                results["headline"] = headline
                merged["results"] = results
                merged["steps_merged"] = list(steps.values())
                merged["derived_blocks_from"] = widest_name
            self._summary = merged
        return self._summary

    def result(self, name: str) -> Dict[str, Any]:
        """Return one analysis's result block from the merged summary.

        Args:
            name: The analysis name.

        Returns:
            The block, empty when the analysis never ran into this directory.
        """
        return dict(dict(self.summary.get("results") or {}).get(name) or {})

    def headline(self, name: str) -> Optional[float]:
        """Return one headline scalar of the merged summary.

        Args:
            name: The headline key.

        Returns:
            The value, or ``None`` when no pass resolved it.
        """
        value = dict(dict(self.summary.get("results") or {}).get("headline") or {}).get(name)
        return value if isinstance(value, (int, float, str, bool)) else None


# =================================================================================================
# The run's geometry, for the page's texts and axes
# =================================================================================================
class Geometry:
    """The lag window, the horizon and the segment length of one run, as the page's texts need them.

    A text that states "38 lags" or "a 20-minute segment" is true for one configuration. This
    class reads the numbers from the run's own summary, so a text is true for the run it
    describes. A number the summary does not hold reads as a general phrase with no number.

    Args:
        run: The run's files.

    Attributes:
        n_lags: The lag window width $L$, or ``None``.
        delay_steps: The causal input delay $\\delta$ in stored steps.
        horizon_steps: The forecast horizon $H$ in stored steps, or ``None``.
        segment_steps: The stored steps of one segment, or ``None``.
        anchors_per_segment: The scored anchors of one full segment, or ``None``.
    """

    def __init__(self, run: "RunFiles") -> None:
        lag = dict(dict(run.summary.get("results") or {}).get("lag") or {})
        geometry = dict(dict(run.summary.get("collection") or {}).get("geometry") or {})

        def whole(value: Any) -> Optional[int]:
            """A positive whole number, or ``None`` when the record does not hold one."""
            try:
                number = int(value)
            except (TypeError, ValueError):
                return None
            return number if number > 0 else None

        self.n_lags = whole(lag.get("n_lags"))
        self.delay_steps = int(lag.get("delay_steps") or 0)
        self.horizon_steps = whole(geometry.get("horizon"))
        self.segment_steps = whole(geometry.get("t"))
        self.anchors_per_segment = whole(geometry.get("anchors_per_sample"))

    def lag_seconds(self, n_lags: Optional[int] = None) -> List[float]:
        r"""The lag axis in seconds, $\tau_\ell = \Delta(\ell + \delta)$.

        The same compensated axis the lag tables of the analyses carry, so a lag on a page built
        here sits where the tables put it.

        Args:
            n_lags: The number of lags. ``None`` uses the run's own lag window.

        Returns:
            One value per lag. Empty when neither count is known.
        """
        count = n_lags if n_lags is not None else (self.n_lags or 0)
        return [SECONDS_PER_STEP * (lag + self.delay_steps) for lag in range(int(count))]

    def band_seconds(self, lag_lo: Any, lag_hi: Any) -> str:
        """A band of lag indices as a span of seconds on the compensated lag axis.

        Args:
            lag_lo: The first lag index of the band.
            lag_hi: The last lag index of the band.

        Returns:
            Text such as ``'0 to 16 s'``.
        """
        low = SECONDS_PER_STEP * (int(lag_lo) + self.delay_steps)
        high = SECONDS_PER_STEP * (int(lag_hi) + self.delay_steps)
        return f"{low:g} to {high:g} s"

    @property
    def lags_text(self) -> str:
        """The lag window in words, such as ``'38 lags of the UA history, from 0 to 148 s back'``."""
        seconds = self.lag_seconds()
        if not seconds:
            return "the lags of its lag window"
        return f"{len(seconds)} lags of the UA history, from {seconds[0]:g} to {seconds[-1]:g} s back"

    @property
    def horizon_text(self) -> str:
        """The forecast horizon in words, such as ``'the next 30 steps (120 s)'``."""
        if self.horizon_steps is None:
            return "the next steps"
        return f"the next {self.horizon_steps} steps ({self.horizon_steps * SECONDS_PER_STEP:g} s)"

    @property
    def segment_text(self) -> str:
        """The segment length in words, such as ``'20-minute segment'``."""
        if self.segment_steps is None:
            return "segment"
        return f"{self.segment_steps * SECONDS_PER_STEP / 60.0:g}-minute segment"

    @property
    def anchors_text(self) -> str:
        """The scored anchors of a segment in words, such as ``'136 scored anchors'``."""
        if self.anchors_per_segment is None:
            return "a bounded number of scored anchors"
        return f"{self.anchors_per_segment} scored anchors"


# =================================================================================================
# Values
# =================================================================================================
def clean(values: Any) -> List[Any]:
    """Return a sequence as a JSON-safe list.

    Args:
        values: A sequence or an array, of one or two dimensions.

    Returns:
        A list, nested for two dimensions, with ``None`` in place of a value that is not finite.
    """
    array = np.asarray(values)
    if array.dtype.kind in "fc":
        return [None if not math.isfinite(v) else float(v) for v in array.astype(float).ravel()] \
            if array.ndim == 1 else [clean(row) for row in array]
    if array.dtype.kind in "iub":
        return array.tolist()
    return [None if (isinstance(v, float) and not math.isfinite(v)) else v for v in array.tolist()]


def json_safe(value: Any) -> Any:
    """Convert a nested payload to plain JSON types.

    Args:
        value: A mapping, a sequence, an array or a scalar, nested to any depth.

    Returns:
        The same structure in built-in types, with ``None`` in place of a non-finite number.
    """
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def encode_array(values: Any) -> Dict[str, Any]:
    r"""Pack a numeric array as base64 little-endian 16-bit integers with an offset and a scale.

    The page decodes $x = o + s\,q$. A non-finite value is stored as a sentinel and comes back as
    a missing value.

    Args:
        values: The array, any shape.

    Returns:
        ``{'d': shape, 'o': offset, 's': scale, 'b': base64 text}``.
    """
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    low = float(array[finite].min()) if finite.any() else 0.0
    high = float(array[finite].max()) if finite.any() else 0.0
    scale = (high - low) / 65534.0 if high > low else 1.0
    offset = 0.5 * (high + low)
    quantised = np.full(array.shape, _INT16_NAN, dtype=np.int16)
    quantised[finite] = np.clip(np.rint((array[finite] - offset) / scale), -32767, 32767).astype(np.int16)
    return {
        "d": list(array.shape), "o": offset, "s": scale,
        "b": base64.b64encode(quantised.astype("<i2").tobytes()).decode("ascii"),
    }


def fmt(value: Any, digits: int = 3, unit: str = "") -> str:
    """Format a number for a sentence: ``digits`` significant figures, or ``n/a``.

    Args:
        value: The number, possibly ``None`` or not finite.
        digits: Significant figures.
        unit: Text appended after a space, such as ``'nats'``.

    Returns:
        The formatted text.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    if number == 0:
        text = "0"
    elif abs(number) >= 10 ** digits:
        text = f"{number:,.0f}"
    else:
        text = f"{number:.{digits}g}"
    return f"{text} {unit}".strip()


def count(value: Any) -> str:
    """Format a whole number with thousands separators.

    Args:
        value: The number, possibly ``None`` or not finite.

    Returns:
        Text such as ``'808,927'``, or ``'n/a'``.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{int(round(number)):,}" if math.isfinite(number) else "n/a"


def pct(value: Any, digits: int = 0) -> str:
    """Format a share in $[0, 1]$ as a percentage.

    Args:
        value: The share, possibly ``None`` or not finite.
        digits: Decimal places of the percentage.

    Returns:
        Text such as ``'43%'``, or ``'n/a'``.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{100.0 * number:.{digits}f}%" if math.isfinite(number) else "n/a"


def rgba(color: str, alpha: float) -> str:
    """Give a ``#rrggbb`` colour an opacity.

    Args:
        color: The colour as ``'#rrggbb'``.
        alpha: The opacity in $[0, 1]$.

    Returns:
        The colour as an ``rgba()`` string.
    """
    red, green, blue = (int(color[index:index + 2], 16) for index in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha:g})"


def class_color(name: Any, index: int = 0) -> str:
    """Return the colour of a clinical class.

    Args:
        name: The class label.
        index: The position of the label in its series, used for a label with no fixed colour.

    Returns:
        The class colour, or a series colour for an unknown label.
    """
    return CLASS_COLORS.get(str(name), SERIES[index % len(SERIES)])


# =================================================================================================
# Traces and layouts
# =================================================================================================
def line(
    x: Any, y: Any, name: str, color: str, *, width: float = LINE_WIDTH, dash: Optional[str] = None,
    markers: bool = False, axis: Optional[str] = None, hover: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """A solid line trace.

    Args:
        x: The x values.
        y: The y values.
        name: The legend entry.
        color: The line colour.
        width: The line width in pixels.
        dash: A Plotly dash style. ``None`` draws a solid line.
        markers: Draw a marker at each point as well.
        axis: The y axis, such as ``'y2'``. ``None`` uses the first.
        hover: A Plotly hover template. ``None`` shows the value to four significant figures.
        **extra: Further trace attributes.

    Returns:
        The trace.
    """
    trace: Dict[str, Any] = {
        "type": "scatter", "mode": "lines+markers" if markers else "lines", "x": clean(x),
        "y": clean(y), "name": name, "line": {"color": color, "width": width},
        "hovertemplate": hover or "%{y:.4g}<extra>" + name + "</extra>",
    }
    if dash:
        trace["line"]["dash"] = dash
    if markers:
        trace["marker"] = {"size": 6, "color": color}
    if axis:
        trace["yaxis"] = axis
    trace.update(extra)
    return trace


def band(x: Any, lo: Any, hi: Any, color: str, *, axis: Optional[str] = None) -> List[Dict[str, Any]]:
    """A shaded band between two curves, as the two traces Plotly needs for it.

    Args:
        x: The x values.
        lo: The lower curve.
        hi: The upper curve.
        color: The colour of the band, drawn at low opacity.
        axis: The y axis, such as ``'y2'``.

    Returns:
        Two traces, to be placed before the line the band belongs to.
    """
    base: Dict[str, Any] = {
        "type": "scatter", "mode": "lines", "x": clean(x), "line": {"width": 0},
        "hoverinfo": "skip", "showlegend": False,
    }
    if axis:
        base["yaxis"] = axis
    return [
        {**base, "y": clean(lo)},
        {**base, "y": clean(hi), "fill": "tonexty", "fillcolor": rgba(color, 0.18)},
    ]


def bars(
    x: Any, y: Any, name: str, color: Any, *, lo: Any = None, hi: Any = None,
    axis: Optional[str] = None, **extra: Any,
) -> Dict[str, Any]:
    """A bar trace, with an asymmetric error bar when both bounds are given.

    Args:
        x: The categories or positions.
        y: The bar heights.
        name: The legend entry.
        color: One colour, or one per bar.
        lo: The lower bound of each bar's interval.
        hi: The upper bound of each bar's interval.
        axis: The y axis, such as ``'y2'``.
        **extra: Further trace attributes.

    Returns:
        The trace.
    """
    heights = np.asarray(y, dtype=float)
    trace: Dict[str, Any] = {
        "type": "bar", "x": clean(x) if not isinstance(x, list) else x, "y": clean(heights),
        "name": name, "marker": {"color": color},
        "hovertemplate": "%{y:.4g}<extra>" + name + "</extra>",
    }
    if lo is not None and hi is not None:
        low, high = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        trace["error_y"] = {
            "type": "data", "symmetric": False, "thickness": 1.4, "width": 3,
            "array": clean(np.clip(high - heights, 0, None)),
            "arrayminus": clean(np.clip(heights - low, 0, None)),
        }
    if axis:
        trace["yaxis"] = axis
    trace.update(extra)
    return trace


def heatmap(
    x: Any, y: Any, z: Any, *, label: str, log: bool = False, floor: float = 1e-3,
    diverging: bool = False, hover: Optional[str] = None, **extra: Any,
) -> Dict[str, Any]:
    r"""A heatmap trace.

    Args:
        x: The column coordinates.
        y: The row coordinates.
        z: The field, rows by columns.
        label: The colour bar title.
        log: Colour $\log_{10} z$ rather than $z$, with values below ``floor`` drawn at the floor.
            The hover text still shows $z$.
        floor: The smallest value a logarithmic scale draws.
        diverging: Use a diverging scale centred on zero, for a signed field.
        hover: A Plotly hover template.
        **extra: Further trace attributes.

    Returns:
        The trace.
    """
    field = np.asarray(z, dtype=float)
    trace: Dict[str, Any] = {
        "type": "heatmap", "x": clean(x), "y": clean(y), "customdata": clean(field),
        "colorbar": {"title": {"text": label, "side": "right"}, "thickness": 12, "len": 0.9},
        "hovertemplate": hover or "x %{x:.4g}, y %{y:.4g}: %{customdata:.4g}<extra></extra>",
    }
    if log:
        with np.errstate(invalid="ignore", divide="ignore"):
            logged = np.log10(np.clip(field, floor, None))
        trace["z"] = clean(logged)
        top = float(np.nanmax(logged)) if np.isfinite(logged).any() else 0.0
        ticks = [value for value in range(int(math.floor(math.log10(floor))), int(math.ceil(top)) + 1)]
        trace["colorbar"].update({
            "tickvals": ticks, "ticktext": [f"{10.0 ** value:g}" for value in ticks],
        })
        trace["colorscale"] = "Viridis"
    elif diverging:
        limit = float(np.nanmax(np.abs(field))) if np.isfinite(field).any() else 1.0
        trace.update({"z": clean(field), "colorscale": "RdBu", "reversescale": True,
                      "zmin": -limit, "zmax": limit})
    else:
        trace.update({"z": clean(field), "colorscale": "Viridis"})
    trace.update(extra)
    return trace


def layout(xtitle: str = "", ytitle: str = "", **extra: Any) -> Dict[str, Any]:
    """A layout with the two axis titles set.

    Args:
        xtitle: The x axis title.
        ytitle: The y axis title.
        **extra: Further layout attributes. ``xaxis`` and ``yaxis`` dictionaries are merged over
            the titles.

    Returns:
        The layout.
    """
    result: Dict[str, Any] = {
        "xaxis": {"title": xtitle, **dict(extra.pop("xaxis", {}))},
        "yaxis": {"title": ytitle, **dict(extra.pop("yaxis", {}))},
    }
    result.update(extra)
    return result


def second_axis(title: str, **extra: Any) -> Dict[str, Any]:
    """Build a right-hand y axis that overlays the first, for a second quantity on one plot.

    Args:
        title: The axis title.
        **extra: Further axis attributes.

    Returns:
        The axis, for a layout key such as ``yaxis2``.
    """
    return {"title": title, "overlaying": "y", "side": "right", "showgrid": False, **extra}


def hline(y: float, *, color: str = GRAY, dash: str = "dot", axis: str = "y") -> Dict[str, Any]:
    """Build a horizontal reference line across the plot.

    Args:
        y: The y value of the line.
        color: The line colour.
        dash: The Plotly dash style.
        axis: The y axis the value is read on, such as ``'y2'``.

    Returns:
        The line as a layout shape.
    """
    return {"type": "line", "xref": "paper", "yref": axis, "x0": 0, "x1": 1, "y0": y, "y1": y,
            "line": {"color": color, "width": 1.2, "dash": dash}}


def vline(x: float, *, color: str = GRAY, dash: str = "dot") -> Dict[str, Any]:
    """Build a vertical reference line across the plot.

    Args:
        x: The x value of the line.
        color: The line colour.
        dash: The Plotly dash style.

    Returns:
        The line as a layout shape.
    """
    return {"type": "line", "xref": "x", "yref": "paper", "x0": x, "x1": x, "y0": 0, "y1": 1,
            "line": {"color": color, "width": 1.2, "dash": dash}}


def variant(
    data: Sequence[Any], figure_layout: Mapping[str, Any], *, pop: Optional[str] = None,
    opt: Optional[str] = None,
) -> Dict[str, Any]:
    """One drawable state of a plot.

    Args:
        data: The traces. A nested list, as :func:`band` returns, is flattened.
        figure_layout: The layout.
        pop: The gate population this state belongs to, when the plot follows the page's
            ``Anchors`` filter.
        opt: The entry of the plot's own selector this state belongs to.

    Returns:
        The variant.
    """
    flat: List[Any] = []
    for trace in data:
        flat.extend(trace) if isinstance(trace, list) else flat.append(trace)
    return {"pop": pop, "opt": opt, "data": flat, "layout": dict(figure_layout)}


# =================================================================================================
# Panels
# =================================================================================================
def plot_panel(
    identifier: str,
    title: str,
    variants: Sequence[Dict[str, Any]],
    *,
    how: str,
    shows: str,
    read: str,
    run: Optional[str] = None,
    sources: Iterable[str] = (),
    pdfs: Iterable[str] = (),
    wide: bool = False,
    height: int = 380,
    opt_label: Optional[str] = None,
    scope: Optional[str] = None,
) -> Dict[str, Any]:
    """One plot card of the page.

    Args:
        identifier: A stable name, unique on the page.
        title: The card title.
        variants: From :func:`variant`. More than one makes the plot follow a selector.
        how: How the plot was made: the input table, the selection and the averaging.
        shows: What the axes and the traces are.
        read: How to read it, and what a good or a bad result looks like.
        run: What this run's numbers say. ``None`` leaves the note out.
        sources: The data files behind the plot, relative to the results directory.
        pdfs: The PDF figures that show the same result.
        wide: Give the plot the full card width and put the explanation below it.
        height: The plot height in pixels.
        opt_label: The label of the plot's own selector.
        scope: The badge text of a plot that does not follow the ``Anchors`` filter. ``None``
            reads "all anchors only".

    Returns:
        The panel.
    """
    return {
        "kind": "plot", "id": identifier, "title": title, "variants": list(variants),
        "explain": {"how": how, "shows": shows, "read": read, "run": run},
        "sources": list(sources), "pdfs": list(pdfs), "wide": wide, "height": int(height),
        "opt_label": opt_label, "scope": scope,
    }


def table_panel(
    identifier: str, title: str, columns: Sequence[str], rows: Sequence[Sequence[Any]], *,
    how: str = "", shows: str = "", read: str = "", run: Optional[str] = None,
    sources: Iterable[str] = (), pdfs: Iterable[str] = (),
) -> Dict[str, Any]:
    """One table card of the page.

    Args:
        identifier: A stable name, unique on the page.
        title: The card title.
        columns: The column headings.
        rows: The rows. A number is right-aligned and formatted by the page.
        how: How the table was made.
        shows: What the columns are.
        read: How to read it.
        run: What this run's numbers say.
        sources: The data files behind the table.
        pdfs: The PDF figures that show the same result.

    Returns:
        The panel.
    """
    explain = {"how": how, "shows": shows, "read": read, "run": run}
    return {
        "kind": "table", "id": identifier, "title": title, "columns": list(columns),
        "rows": [list(row) for row in rows],
        "explain": explain if any(explain.values()) else None,
        "sources": list(sources), "pdfs": list(pdfs),
    }


def tiles_panel(identifier: str, tiles: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Build a row of headline tiles.

    Args:
        identifier: A stable name, unique on the page.
        tiles: From :func:`tile`.

    Returns:
        The panel.
    """
    return {"kind": "tiles", "id": identifier, "title": "Headline numbers", "tiles": list(tiles),
            "explain": {}}


def html_panel(identifier: str, title: str, body: str) -> Dict[str, Any]:
    """Build a card of prose.

    Args:
        identifier: A stable name, unique on the page.
        title: The card title.
        body: The card content as HTML. It is written by a builder and inserted as it is, so any
            text that comes from data must be escaped before it is put in.

    Returns:
        The panel.
    """
    return {"kind": "html", "id": identifier, "title": title, "html": body, "explain": {}}


def missing_panel(identifier: str, title: str, reason: str) -> Dict[str, Any]:
    """Build the card that stands in for a plot whose input the run directory does not hold.

    Args:
        identifier: A stable name, unique on the page.
        title: The title the plot would have had.
        reason: Which file is missing and what writes it, as HTML.

    Returns:
        The panel.
    """
    return {"kind": "missing", "id": identifier, "title": title, "reason": reason, "explain": {}}


def tile(name: str, value: str, note: str = "", tone: str = "") -> Dict[str, str]:
    """Build one headline tile.

    Args:
        name: The quantity the tile shows.
        value: The formatted value.
        note: A short text under the value.
        tone: ``'good'``, ``'warn'``, ``'bad'`` or empty, for the tile's accent.

    Returns:
        The tile.
    """
    return {"name": name, "value": value, "note": note, "tone": tone}


__all__ = [
    "BLUE", "CLASS_COLORS", "DERIVED_RESULT_BLOCKS", "GATE_OPEN_FROM", "GATE_SHUT_BELOW", "GRAY",
    "GREEN", "LINE_WIDTH", "MAGENTA", "ORANGE", "POPULATIONS", "POPULATION_COLORS",
    "POPULATION_NAMES", "PURPLE", "SECONDS_PER_STEP", "SERIES", "TEAL", "VERMILLION", "Geometry",
    "MissingInput", "RunFiles", "band", "bars", "class_color", "clean", "count", "encode_array",
    "fmt",
    "heatmap", "hline", "html_panel", "json_safe", "layout", "line", "missing_panel", "pct",
    "plot_panel", "rgba", "second_axis", "table_panel", "tile", "tiles_panel", "variant", "vline",
]
