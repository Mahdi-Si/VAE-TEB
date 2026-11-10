r"""The gate-aware panels of the run dashboard: the gate, the gate-on results, the lags, the signal.

Each function here builds one card from the tables of a finished run and returns it. The tables
come from the ``event_gate`` and ``gate_open`` analyses and from the collection pass. A function
raises :class:`~teb_vae.lag_attn_cfs.eval.run_dashboard_core.MissingInput` when its table is
absent, and the assembler then draws a card that names the missing file.

**The populations.** A plot that follows the page's ``Anchors`` filter carries one variant per
gate population of the ``gate_open`` analysis: every scored anchor, the anchors with the gate on
($g \ge g_{\mathrm{shut}}$), the anchors with the gate fully open ($g \ge g_{\mathrm{open}}$), and
the anchors with the gate off ($g < g_{\mathrm{shut}}$). The two thresholds are those of
:mod:`teb_vae.lag_attn_cfs.gate_states`. The filter then swaps the variant, so one plot answers
the same question on each population.

**A lag axis here is stored-coefficient time**, as everywhere in this package. The stored timeline
is canonical, and no panel applies a correction to it.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from teb_vae.lag_attn_cfs.eval.run_dashboard_core import (
    BLUE,
    GATE_OPEN_FROM,
    GATE_SHUT_BELOW,
    GRAY,
    GREEN,
    ORANGE,
    POPULATION_COLORS,
    POPULATION_NAMES,
    POPULATIONS,
    PURPLE,
    SECONDS_PER_STEP,
    VERMILLION,
    Geometry,
    MissingInput,
    RunFiles,
    band,
    bars,
    class_color,
    count,
    encode_array,
    fmt,
    heatmap,
    hline,
    layout,
    line,
    pct,
    plot_panel,
    second_axis,
    variant,
    vline,
)

#: The folders of the two gate analyses, and the collection pass's tables.
GATE = "gate_open/"
EVENT = "event_gate/"
PER_ANCHOR = "per_anchor.parquet"
PER_ANCHOR_VECTORS = "per_anchor_vectors.npz"
RETAINED = "retained_arrays.npz"

#: How many segments the signal view offers, at most. The retained segments are a seeded draw, so
#: a cap keeps the page small without choosing by result.
MAX_EXPLORER_SEGMENTS = 48

#: The raw sampling the signal view keeps, in samples per stored step.
EXPLORER_RAW_PER_STEP = 4

POPULATION_KEYS: Sequence[str] = tuple(entry["key"] for entry in POPULATIONS)

#: The smallest difference between the highest and the lowest gate-on share over the contraction
#: age that the page calls a response to the contraction. Below it the profile is called flat.
CONTRACTION_SWING_MIN = 0.2

#: The sentence a lag-resolved card carries about its axis.
LAG_NOTE = (
    "The lag axis is stored-coefficient time. It says how many seconds back in the stored UA "
    "features the model reads. It is not a physiological delay."
)

#: The colours of the four gate states of the ``event_gate`` analysis, in table order.
STATE_COLORS = {"forced_shut": GRAY, "model_shut": VERMILLION, "partly_open": ORANGE, "open": BLUE}
STATE_NAMES = {
    "forced_shut": "forced shut (no usable UA key)", "model_shut": "model shut (g < 0.2)",
    "partly_open": "partly open (0.2 to 0.8)", "open": "open (g of 0.8 or more)",
}


def _pooled(table: pd.DataFrame, population: Optional[str] = None, cohort: str = "all") -> pd.DataFrame:
    """Select the rows of one population and cohort of a ``gate_open`` table.

    Args:
        table: A table of the ``gate_open`` analysis.
        population: The population key, or ``None`` to keep every population.
        cohort: The cohort key. ``'all'`` is the pooled cohort.

    Returns:
        The selected rows. A column the table lacks is not filtered on.
    """
    rows = table
    if "cohort" in rows.columns:
        rows = rows[rows["cohort"] == cohort]
    if population is not None and "population" in rows.columns:
        rows = rows[rows["population"] == population]
    return rows


def _cohorts(table: pd.DataFrame) -> List[str]:
    """The clinical classes a ``gate_open`` table carries, without the pooled cohort.

    The order is fixed (healthy, acidosis, HIE, then any other label), so a class keeps its place
    in every legend.
    """
    present = [str(name) for name in pd.unique(table["cohort"]) if name != "all"]
    known = [name for name in ("healthy", "acidosis", "hie") if name in present]
    return known + sorted(set(present) - set(known))


def _summary_value(summary: pd.DataFrame, population: str, quantity: str) -> float:
    """Read one pooled value of the ``gate_open`` summary table.

    Args:
        summary: The summary table.
        population: The population key.
        quantity: The column to read.

    Returns:
        The value, ``NaN`` when the population or the column is absent.
    """
    rows = _pooled(summary, population)
    return float(rows[quantity].iloc[0]) if len(rows) and quantity in rows.columns else float("nan")


def gate_headline(run: RunFiles) -> Dict[str, Any]:
    """The headline numbers of the ``gate_open`` analysis.

    Read from the analysis's result block when a summary carries it, and otherwise rebuilt from
    the analysis's own tables, so a page built before the next full pass shows the same numbers.

    Args:
        run: The run's files.

    Returns:
        The headline values, plus ``gate_off_median_run_s`` and ``gate_on_q90_run_s``. Empty when
        the run holds neither the block nor the tables.
    """
    block = run.result("gate_open")
    headline = dict(block.get("headline") or {})
    runs = dict(block.get("runs") or {})
    if headline:
        headline["gate_off_median_run_s"] = dict(runs.get("gate_off") or {}).get("median_s")
        headline["gate_on_q90_run_s"] = dict(runs.get("gate_on") or {}).get("q90_s")
        return headline
    if not run.exists(GATE + "gate_open_summary.csv"):
        return {}
    summary = run.csv(GATE + "gate_open_summary.csv")
    all_kl = _summary_value(summary, "all", "kl_nats")
    headline = {
        "gate_on_frac": _summary_value(summary, "gate_on", "frac"),
        "fully_open_frac": _summary_value(summary, "fully_open", "frac"),
        "gate_on_kl_nats": _summary_value(summary, "gate_on", "kl_nats"),
        "gate_off_kl_nats": _summary_value(summary, "gate_off", "kl_nats"),
        "gate_on_gain_mc_nats": _summary_value(summary, "gate_on", "gain_mc_nats"),
        "gate_off_gain_mc_nats": _summary_value(summary, "gate_off", "gain_mc_nats"),
        "fully_open_gain_mc_nats": _summary_value(summary, "fully_open", "gain_mc_nats"),
        "gate_on_kl_share_of_run": _summary_value(summary, "gate_on", "kl_contribution_nats") / all_kl,
        "gate_on_kl_centroid_s": _summary_value(summary, "gate_on", "kl_centroid_s"),
        "gate_on_kl_lag0_share": _summary_value(summary, "gate_on", "kl_lag0_share"),
    }
    if run.exists(GATE + "gate_open_run_durations.csv"):
        table = run.csv(GATE + "gate_open_run_durations.csv")
        for key in ("gate_on", "gate_off"):
            rows = table[table["state"] == key]
            if not len(rows):
                continue
            cumulative = rows["share_of_runs"].cumsum().to_numpy()
            edges = rows["duration_hi_s"].to_numpy()
            last = len(edges) - 1
            headline[f"{key}_median_run_s"] = float(edges[min(np.searchsorted(cumulative, 0.5), last)])
            if key == "gate_on":
                headline["gate_on_q90_run_s"] = float(edges[min(np.searchsorted(cumulative, 0.9), last)])
    return headline


def _per_population(build: Callable[[str], Optional[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Call a variant builder once per population.

    Args:
        build: A function from a population key to a variant, or to ``None`` when the run has
            no data for that population.

    Returns:
        The variants that were built, in population order.
    """
    variants = [build(key) for key in POPULATION_KEYS]
    return [item for item in variants if item is not None]


# =================================================================================================
# Section: the gate
# =================================================================================================
def gate_histogram(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the distribution of the gate value per clinical class.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    frame = run.parquet(PER_ANCHOR, ["event_gate", "clinical_class"])
    if "event_gate" not in frame.columns:
        raise MissingInput(f"{PER_ANCHOR} (column event_gate)", "the collection pass")
    edges = np.linspace(0.0, 1.0, 51)
    centres = 0.5 * (edges[:-1] + edges[1:])
    traces = []
    for index, (name, rows) in enumerate(frame.groupby("clinical_class")):
        counts, _ = np.histogram(rows["event_gate"].to_numpy(dtype=float), bins=edges)
        traces.append(line(centres, counts / max(counts.sum(), 1), str(name), class_color(name, index),
                           hover="%{y:.2%} of anchors<extra>" + str(name) + "</extra>"))
    gate = frame["event_gate"].to_numpy(dtype=float)
    figure = layout("gate value g", "share of anchors per bin",
                    shapes=[vline(GATE_SHUT_BELOW), vline(GATE_OPEN_FROM)],
                    yaxis={"type": "log", "tickformat": ".1%", "dtick": "D2"})
    return plot_panel(
        "gate-histogram", "How the gate value is distributed", [variant(traces, figure)],
        how="Each scored anchor has one gate value g between 0 and 1. The plot counts the anchors "
            "in 50 bins of g, per clinical class, and divides by the class total.",
        shows="The x axis is the gate value. The y axis is the share of anchors in each bin, on a "
              "logarithmic scale. The dotted lines mark the two thresholds, "
              f"{GATE_SHUT_BELOW:g} and {GATE_OPEN_FROM:g}.",
        read="A gate near 0 stops the UA from changing the model's belief. A gate near 1 lets it "
             f"through. Mass on the left of {GATE_SHUT_BELOW:g} means that the model ignores the UA "
             "at most anchors. "
             "If the classes overlap, the gate does not behave differently between classes.",
        run=f"The median gate is {fmt(np.nanmedian(gate), 2)}. {pct(np.mean(gate < GATE_SHUT_BELOW))} "
            f"of the anchors are below {GATE_SHUT_BELOW:g} and {pct(np.mean(gate >= GATE_OPEN_FROM), 1)} "
            f"are at {GATE_OPEN_FROM:g} or above.",
        sources=[PER_ANCHOR], pdfs=[EVENT + "event_gate.pdf"],
    )


def gate_states(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the share, the KL and the forecast gain of each gate state.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv(EVENT + "event_gate_states.csv")
    names = [STATE_NAMES.get(name, name) for name in table["state"]]
    colours = [STATE_COLORS.get(name, GRAY) for name in table["state"]]
    options = (
        ("Share of anchors", "frac", "share of anchors", ".0%"),
        ("KL per anchor", "kl_nats", "nats per anchor", None),
        ("Forecast gain per anchor", "gain_nats", "nats per anchor", None),
    )
    variants = []
    for label, column, ytitle, tick in options:
        figure = layout("", ytitle, showlegend=False, yaxis={"tickformat": tick} if tick else {})
        variants.append(variant(
            [bars(names, table[column], label, colours, lo=table[f"{column}_lo"], hi=table[f"{column}_hi"])],
            figure, opt=label,
        ))
    by_state = table.set_index("state")
    return plot_panel(
        "gate-states", "The four gate states", variants, opt_label="Quantity",
        how="The event_gate analysis puts each scored anchor in one of four states from its gate "
            "value and from the UA quality mask. Each bar is a mean within a recording, then a "
            "mean over recordings. The error bar is a 95% bootstrap interval over recordings.",
        shows="One bar per state. The selector changes the quantity: the share of anchors, the KL "
              "per anchor, or the forecast gain per anchor.",
        read="Forced shut is a property of the data: no usable UA key was left. Model shut is a "
             "decision of the model. A useful gate gives a high KL and a high forecast gain in the "
             "open state and values near zero in the shut states.",
        run=f"Open anchors are {pct(by_state.loc['open', 'frac'], 1)} of the run and gain "
            f"{fmt(by_state.loc['open', 'gain_nats'])} nats each. Model-shut anchors are "
            f"{pct(by_state.loc['model_shut', 'frac'])} and gain "
            f"{fmt(by_state.loc['model_shut', 'gain_nats'], 2)} nats. Forced-shut anchors are "
            f"{pct(by_state.loc['forced_shut', 'frac'], 1)}.",
        sources=[EVENT + "event_gate_states.csv"], pdfs=[EVENT + "event_gate.pdf"],
    )


def gate_states_over_time(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the share of each gate state against time before delivery.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv(EVENT + "event_gate_by_time_to_delivery.csv")
    traces = [
        {**line(table["bin_center"], table[f"{name}_frac"], STATE_NAMES[name], STATE_COLORS[name],
                width=0.6, hover="%{y:.1%}<extra>" + STATE_NAMES[name] + "</extra>"),
         "stackgroup": "states", "fillcolor": STATE_COLORS[name]}
        for name in STATE_COLORS if f"{name}_frac" in table.columns
    ]
    traces.append(line(table["bin_center"], table["applied_gate_mean"], "mean gate", "#111111",
                       dash="dot", hover="%{y:.3f}<extra>mean gate</extra>"))
    figure = layout("hours before delivery", "share of anchors",
                    xaxis={"autorange": "reversed"}, yaxis={"range": [0, 1], "tickformat": ".0%"})
    early, late = table.iloc[-1], table.iloc[0]
    return plot_panel(
        "gate-states-time", "Gate states against time before delivery", [variant(traces, figure)],
        how="The anchors are put in 30-minute bins of their time before delivery. In each bin the "
            "plot stacks the share of anchors in each gate state. These are anchor-level shares, "
            "because one recording rarely covers every bin.",
        shows="The x axis runs toward delivery, which is at the right edge. The coloured areas are "
              "the four states and sum to 100%. The dotted line is the mean gate value.",
        read="A growing blue or orange area toward delivery means that the model reads the UA "
             "more often in late labour. A growing gray area means that the data removes more UA "
             "keys there, which is not a decision of the model.",
        run=f"The share of open anchors is {pct(early['open_frac'], 1)} at {fmt(early['bin_center'], 2)} h "
            f"before delivery and {pct(late['open_frac'], 1)} in the last half hour. The mean gate goes "
            f"from {fmt(early['applied_gate_mean'], 2)} to {fmt(late['applied_gate_mean'], 2)}.",
        sources=[EVENT + "event_gate_by_time_to_delivery.csv"], pdfs=[EVENT + "event_gate.pdf"],
    )


def gate_vs_contraction(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the gate against the time since the last contraction onset.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = _pooled(run.csv(GATE + "gate_open_by_contraction_age.csv"))
    traces = []
    for key in ("gate_on", "fully_open", "gate_off"):
        rows = table[table["population"] == key]
        traces.append(line(rows["bin_center"], rows["share_of_anchors"], f"share {POPULATION_NAMES[key]}",
                           POPULATION_COLORS[key], markers=True,
                           hover="%{y:.1%}<extra>" + POPULATION_NAMES[key] + "</extra>"))
    everyone = table[table["population"] == "all"]
    traces.append(line(everyone["bin_center"], everyone["gate_mean"], "mean gate", PURPLE, dash="dot",
                       hover="%{y:.3f}<extra>mean gate</extra>"))
    on = table[table["population"] == "gate_on"]
    peak = on.loc[on["share_of_anchors"].idxmax()]
    trough = on.loc[on["share_of_anchors"].idxmin()]
    swing = float(peak["share_of_anchors"] - trough["share_of_anchors"])
    # The conclusion is stated only when this run's numbers carry it: a clear swing, with the
    # peak before the trough. A flat profile, or a late peak, gets its own sentence.
    if swing < CONTRACTION_SWING_MIN:
        conclusion = "The gate-on share changes little with the age of the contraction."
    elif float(peak["bin_center"]) < float(trough["bin_center"]):
        conclusion = "The gate therefore opens early in a contraction and closes later."
    else:
        conclusion = "The gate-on share is highest late after an onset, and not during the contraction."
    figure = layout("seconds since the last contraction onset", "share of anchors, or gate value",
                    yaxis={"range": [0, 1]})
    return plot_panel(
        "gate-contraction", "Does the gate follow the contractions?", [variant(traces, figure)],
        how="For each anchor the collection pass measures the time since the last contraction "
            "onset in the UA signal. The anchors are put in 15-second bins of that time. Each "
            "point is the share of the bin's anchors that are in one gate population.",
        shows="The x axis is the age of the current contraction in seconds. The solid lines are "
              "the shares of anchors with the gate on, fully open and off. The dotted line is the "
              "mean gate value.",
        read="A gate that reacts to contractions rises shortly after an onset and falls later. A "
             "flat line means that the gate ignores the contraction timing.",
        run=f"The gate-on share is highest at {pct(peak['share_of_anchors'])}, "
            f"{fmt(peak['bin_center'], 3)} s after an onset, and lowest at "
            f"{pct(trough['share_of_anchors'])}, at {fmt(trough['bin_center'], 3)} s. {conclusion}",
        sources=[GATE + "gate_open_by_contraction_age.csv"],
        pdfs=[GATE + "gate_open.pdf", EVENT + "event_gate.pdf"],
    )


def gate_value_payoff(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the KL and the forecast gain against the gate value.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv(GATE + "gate_open_by_gate.csv")
    traces = [
        bars(table["bin_center"], table["share_of_anchors"], "share of anchors", "rgba(122,128,138,.35)",
             axis="y3", hovertemplate="%{y:.1%}<extra>share of anchors</extra>"),
        line(table["bin_center"], table["kl_nats"], "KL (nats)", VERMILLION, markers=True),
        line(table["bin_center"], table["gain_mc_nats"], "forecast gain (nats)", ORANGE, markers=True, axis="y2"),
    ]
    figure = layout(
        "gate value g (bins of 0.1)", "KL per anchor (nats)",
        yaxis2=second_axis("forecast gain per anchor (nats)"),
        yaxis3={"overlaying": "y", "side": "right", "showgrid": False, "showticklabels": False,
                "range": [0, max(float(table["share_of_anchors"].max()) * 3.0, 1e-6)]},
        shapes=[vline(GATE_SHUT_BELOW), vline(GATE_OPEN_FROM)], bargap=0.05,
    )
    first, last = table.iloc[0], table.iloc[-1]
    return plot_panel(
        "gate-payoff", "What a larger gate value buys", [variant(traces, figure)],
        how="The anchors are put in ten bins of the gate value. In each bin the plot takes the "
            "anchor-level mean of the KL and of the Monte Carlo forecast gain.",
        shows="The x axis is the gate value. The red line is the KL, on the left axis. The orange "
              "line is the forecast gain, on the right axis. The gray bars show how many anchors "
              "each bin holds.",
        read="Both lines should rise with the gate. The KL is the size of the change that the UA "
             "makes in the model's belief. The gain says whether that change improved the forecast. "
             "A KL that rises while the gain stays flat means that the model pays for information "
             "it does not use.",
        run=f"From the lowest gate bin to the highest, the KL goes from {fmt(first['kl_nats'], 2)} to "
            f"{fmt(last['kl_nats'])} nats and the forecast gain from {fmt(first['gain_mc_nats'], 2)} "
            f"to {fmt(last['gain_mc_nats'])} nats per anchor.",
        sources=[GATE + "gate_open_by_gate.csv"], pdfs=[EVENT + "event_gate.pdf"],
    )


def gate_runs(run: RunFiles) -> Dict[str, Any]:
    """Build the card: how long the gate stays on, and how long it stays off.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv(GATE + "gate_open_run_durations.csv")
    block = dict(run.result("gate_open").get("runs") or {})
    headline = gate_headline(run)
    geometry = Geometry(run)
    bin_width = (
        float((table["duration_hi_s"] - table["duration_lo_s"]).iloc[0]) if len(table) else float("nan")
    )
    variants = []
    for label, column in (("Share of time", "share_of_time"), ("Share of runs", "share_of_runs")):
        traces = []
        for key in ("gate_on", "gate_off"):
            rows = table[table["state"] == key]
            traces.append(line(rows["duration_hi_s"], rows[column], POPULATION_NAMES[key],
                               POPULATION_COLORS[key],
                               hover="%{x:.0f} s: %{y:.1%}<extra>" + POPULATION_NAMES[key] + "</extra>",
                               **{"line": {"color": POPULATION_COLORS[key], "width": 2.6, "shape": "hvh"}}))
        variants.append(variant(traces, layout("run duration (s)", label.lower(),
                                               yaxis={"tickformat": ".0%"}), opt=label))
    on = dict(block.get("gate_on") or {})
    return plot_panel(
        "gate-runs", "How long the gate stays on", variants, opt_label="Weight",
        how=f"Inside each {geometry.segment_text} the gate_open analysis finds every run of "
            "consecutive anchors with the gate on, and every run with the gate off. One anchor is "
            f"{SECONDS_PER_STEP:g} s. The plot is a histogram of the run durations in bins of "
            f"{fmt(bin_width, 3)} s.",
        shows="The x axis is the duration of a run. 'Share of time' weights each run by its "
              "length, so it answers where a gate-on anchor usually sits. 'Share of runs' counts "
              "each run once.",
        read="Runs of 30 to 90 s have the length of one contraction. Runs of one or two anchors "
             f"mean that the gate flickers. A segment holds {geometry.anchors_text}, so a run that "
             "touches a segment edge is cut short.",
        run=f"The median gate-on run lasts {fmt(headline.get('gate_on_median_run_s'), 3)} s and 90% "
            f"last {fmt(headline.get('gate_on_q90_run_s'), 3)} s or less. The median gate-off run lasts "
            f"{fmt(headline.get('gate_off_median_run_s'), 3)} s."
            + (f" {count(on.get('n_touching_segment_edge'))} of {count(on.get('n'))} gate-on runs touch "
               f"a segment edge." if on.get("n") else ""),
        sources=[GATE + "gate_open_run_durations.csv"], pdfs=[GATE + "gate_open.pdf"],
    )


def gate_share_by_class(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the gate-on share against time before delivery, per clinical class.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv(GATE + "gate_open_by_time_to_delivery.csv")
    variants = []
    for key in ("gate_on", "fully_open", "gate_off"):
        traces = []
        for index, name in enumerate(_cohorts(table)):
            rows = _pooled(table, key, name)
            traces.append(line(rows["bin_center"], rows["share_of_anchors"], name, class_color(name, index),
                               markers=True, hover="%{y:.1%}<extra>" + name + "</extra>"))
        figure = layout("hours before delivery", f"share of anchors with the {POPULATION_NAMES[key]}",
                        xaxis={"autorange": "reversed"}, yaxis={"tickformat": ".0%", "rangemode": "tozero"})
        variants.append(variant(traces, figure, opt=POPULATION_NAMES[key]))
    return plot_panel(
        "gate-share-class", "Gate share over labour, per clinical class", variants,
        opt_label="Population",
        how="The anchors are put in 30-minute bins of their time before delivery, per clinical "
            "class. Each point is the share of the bin's anchors in the selected gate population.",
        shows="One line per clinical class. The x axis runs toward delivery at the right edge.",
        read="Lines that lie on each other mean that the gate opens equally often in every class. "
             "A class line that separates near delivery is a possible class signal. Treat the HIE "
             "line with care, because that class has few recordings and its bins are noisy.",
        sources=[GATE + "gate_open_by_time_to_delivery.csv"],
        pdfs=[GATE + "gate_open_per_recording_by_clinical_class.pdf"],
    )


def gate_share_per_recording(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the per-recording gate-on share, per clinical class.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv(GATE + "gate_open_per_recording.csv")
    variants = []
    for key in ("gate_on", "fully_open"):
        traces = []
        for index, (name, rows) in enumerate(table.groupby("clinical_class")):
            traces.append({
                "type": "box", "y": rows[f"{key}_frac"].round(4).tolist(), "name": str(name),
                "marker": {"color": class_color(name, index), "size": 4}, "boxpoints": "all",
                "jitter": 0.5, "pointpos": 0, "line": {"width": 2},
                "hovertemplate": "%{y:.1%}<extra>" + str(name) + "</extra>",
            })
        variants.append(variant(
            traces, layout("", f"share of the recording's anchors with the {POPULATION_NAMES[key]}",
                           showlegend=False, yaxis={"tickformat": ".0%"}),
            opt=POPULATION_NAMES[key],
        ))
    share = table["gate_on_frac"].to_numpy(dtype=float)
    return plot_panel(
        "gate-share-recording", "Gate share per recording", variants, opt_label="Population",
        how="For each recording the gate_open analysis divides the number of anchors in the "
            "selected population by the number of scored anchors of that recording.",
        shows="One dot per recording, grouped by clinical class. The box marks the median and the "
              "quartiles.",
        read="A narrow box means that every recording uses the gate about equally often. A long "
             "tail of low values points to recordings where the model almost never reads the UA.",
        run=f"Across {len(table)} recordings the gate-on share has a median of {pct(np.nanmedian(share))} "
            f"and ranges from {pct(np.nanmin(share))} to {pct(np.nanmax(share))}.",
        sources=[GATE + "gate_open_per_recording.csv"],
        pdfs=[GATE + "gate_open_per_recording_by_clinical_class.pdf",
              GATE + "gate_open_per_recording_by_subgroup.pdf"],
    )


# =================================================================================================
# Section: results on the gate-on anchors
# =================================================================================================
def population_summary(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the share, the KL and the forecast gain of each gate population, per class.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv(GATE + "gate_open_summary.csv")
    labels = [POPULATION_NAMES[key] for key in POPULATION_KEYS]
    options = (
        ("Forecast gain, Monte Carlo", "gain_mc_nats", "nats per anchor"),
        ("Forecast gain, posterior mean", "gain_mean_nats", "nats per anchor"),
        ("KL", "kl_nats", "nats per anchor"),
        ("Share of anchors", "frac", "share of scored anchors"),
        ("NLL of the full forecast", "nll_full_nats", "nats per anchor"),
    )
    variants = []
    for label, column, ytitle in options:
        traces = []
        for index, cohort in enumerate(pd.unique(table["cohort"])):
            rows = table[table["cohort"] == cohort].set_index("population").reindex(POPULATION_KEYS)
            traces.append(bars(
                labels, rows[column], "all classes" if cohort == "all" else cohort,
                GRAY if cohort == "all" else class_color(cohort, index),
                lo=rows[f"{column}_lo"], hi=rows[f"{column}_hi"],
            ))
        variants.append(variant(traces, layout("", ytitle, barmode="group"), opt=label))
    gain_on = _summary_value(table, "gate_on", "gain_mc_nats")
    return plot_panel(
        "population-summary", "The run on each gate population", variants, opt_label="Quantity",
        how="The gate_open analysis splits the scored anchors by their gate value: all, gate on "
            "(g of 0.2 or more), fully open (g of 0.8 or more) and gate off (g below 0.2). For each "
            "group it takes the mean within a recording, then the mean over recordings. The error "
            "bar is a 95% bootstrap interval over recordings.",
        shows="One group of bars per population and one bar per clinical class. The selector "
              "changes the quantity. The forecast gain is the base forecast's error score minus "
              "the full forecast's, in nats, so a positive value means that the UA helped.",
        read="The 'all' bars are what the standard evaluation reports. They mix the two regimes. "
             "The 'gate on' bars show what the model does when it reads the UA. If the gate-on gain "
             "is clearly positive and the gate-off gain is near zero, the gate does its job.",
        run=f"With the gate on, the UA improves the forecast by {fmt(gain_on)} nats per anchor "
            f"(95% interval {fmt(_summary_value(table, 'gate_on', 'gain_mc_nats_lo'))} to "
            f"{fmt(_summary_value(table, 'gate_on', 'gain_mc_nats_hi'))}). Over all anchors the gain is "
            f"{fmt(_summary_value(table, 'all', 'gain_mc_nats'))} nats, and with the gate off it is "
            f"{fmt(_summary_value(table, 'gate_off', 'gain_mc_nats'), 2)} nats.",
        sources=[GATE + "gate_open_summary.csv"], pdfs=[GATE + "gate_open.pdf"],
    )


def population_contributions(run: RunFiles) -> Dict[str, Any]:
    """Build the card: how much of the run's mean KL and gain each gate regime supplies.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = _pooled(run.csv(GATE + "gate_open_summary.csv"))
    rows = table.set_index("population")
    quantities = (
        ("KL", "kl_contribution_nats", "kl_nats"),
        ("gain, Monte Carlo", "gain_mc_contribution_nats", "gain_mc_nats"),
        ("gain, posterior mean", "gain_mean_contribution_nats", "gain_mean_nats"),
    )
    names = [name for name, _, _ in quantities]
    traces = [
        bars(names, [rows.loc[key, column] for _, column, _ in quantities],
             f"from {POPULATION_NAMES[key]} anchors", POPULATION_COLORS[key])
        for key in ("gate_on", "gate_off")
    ]
    traces.append({
        "type": "scatter", "mode": "markers", "x": names, "name": "mean over all anchors",
        "y": [float(rows.loc["all", column]) for _, _, column in quantities],
        "marker": {"symbol": "diamond", "size": 11, "color": "#111111"},
        "hovertemplate": "%{y:.4g}<extra>all anchors</extra>",
    })
    share = float(rows.loc["gate_on", "kl_contribution_nats"]) / float(rows.loc["all", "kl_nats"])
    gain_share = float(rows.loc["gate_on", "gain_mc_contribution_nats"]) / float(rows.loc["all", "gain_mc_nats"])
    return plot_panel(
        "population-contributions", "Where the run's KL and gain come from",
        [variant(traces, layout("", "nats per scored anchor", barmode="relative"))],
        how="For each recording the analysis sums a quantity over the gate-on anchors and divides "
            "by the number of all scored anchors. It does the same for the gate-off anchors. The "
            "two parts add up to the recording's mean over all anchors.",
        shows="One stacked bar per quantity. The blue part comes from the gate-on anchors and the "
              "red part from the gate-off anchors. The diamond is the mean over all anchors.",
        read="If the blue part reaches the diamond, the run's result comes from the gate-on "
             "anchors alone. A red part below zero means that the gate-off anchors make the "
             "forecast slightly worse.",
        run=f"The gate-on anchors are {pct(rows.loc['gate_on', 'frac'])} of the anchors. They carry "
            f"{pct(share, 1)} of the run's KL and {pct(gain_share)} of its Monte Carlo forecast gain.",
        sources=[GATE + "gate_open_summary.csv"],
    )


def population_distributions(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the per-anchor distributions of the KL and the gain, per population.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv(GATE + "gate_open_distributions.csv")
    variants = []
    for label, quantity, xtitle in (
        ("Forecast gain", "gain_mc_nats", "Monte Carlo forecast gain per anchor (nats)"),
        ("KL", "kl_nats", "KL per anchor (nats)"),
    ):
        traces = []
        for key in POPULATION_KEYS:
            rows = table[(table["quantity"] == quantity) & (table["population"] == key)]
            if len(rows):
                traces.append(line(0.5 * (rows["bin_lo"] + rows["bin_hi"]), rows["density"],
                                   POPULATION_NAMES[key], POPULATION_COLORS[key]))
        figure = layout(xtitle, "density (log scale)", yaxis={"type": "log"},
                        shapes=[vline(0.0)] if quantity == "gain_mc_nats" else [])
        variants.append(variant(traces, figure, opt=label))
    return plot_panel(
        "population-distributions", "Per-anchor distributions, per population", variants,
        opt_label="Quantity",
        how="The analysis makes a histogram of the per-anchor values in each population and "
            "divides by the number of anchors and the bin width. The first and the last bin also "
            "hold the values outside the axis range.",
        shows="One curve per population. The y axis is logarithmic, so the tails stay visible.",
        read="For the gain, mass to the right of zero means anchors where the UA helped, and mass "
             "to the left means anchors where it hurt. A gate-off curve that is a narrow spike at "
             "zero confirms that nothing happens there. A long left tail on the gate-on curve "
             "means that reading the UA sometimes costs accuracy.",
        sources=[GATE + "gate_open_distributions.csv"],
    )


#: The metrics of the class trajectory: table key, selector label, axis title, zero line.
_TRAJECTORY_METRICS = (
    ("kl_nats", "KL", "KL per anchor (nats)", False),
    ("gain_mc_nats", "Forecast gain, Monte Carlo", "forecast gain per anchor (nats)", True),
    ("gain_mean_nats", "Forecast gain, mean forecast", "forecast gain per anchor (nats)", True),
    ("share_of_anchors", "Share of the anchors", "share of the window's anchors", False),
    ("kl_centroid_s", "Mean lag of the KL", "mean lag of the KL (s, stored-coefficient time)", False),
    ("kl_lag0_share", "Share of the KL at lag 0", "share of the KL at lag 0", False),
)


def class_trajectory(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the trajectory of each clinical class over labour, per recording, per population.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv(GATE + "gate_open_trajectory.csv")
    tests = run.csv(GATE + "gate_open_trajectory_significance.csv") if run.exists(
        GATE + "gate_open_trajectory_significance.csv") else pd.DataFrame()
    variants = []
    for key in POPULATION_KEYS:
        for metric, label, ytitle, zero in _TRAJECTORY_METRICS:
            rows = table[(table["population"] == key) & (table["metric"] == metric)]
            if rows.empty:
                continue
            traces: List[Any] = []
            for index, name in enumerate(_cohorts(rows)):
                part = rows[rows["cohort"] == name].sort_values("bin_center_h")
                colour = class_color(name, index)
                traces.append(band(part["bin_center_h"], part["q25"], part["q75"], colour))
                traces.append(line(part["bin_center_h"], part["median"], name, colour, markers=True,
                                   customdata=part["n_recordings"].tolist(),
                                   hover="%{y:.4g} (%{customdata} recordings)<extra>" + name + "</extra>"))
            shapes = [hline(0.0)] if zero else []
            if len(tests):
                marked = tests[(tests["population"] == key) & (tests["metric"] == metric)
                               & tests["significant"].astype(str).str.lower().eq("true")]
                shapes += [vline(float(centre), color=PURPLE) for centre in marked["bin_center_h"]]
            figure = layout("hours before delivery", f"{ytitle}, {POPULATION_NAMES[key]}",
                            xaxis={"autorange": "reversed"}, shapes=shapes)
            variants.append(variant(traces, figure, pop=key, opt=label))
    on = table[(table["population"] == "gate_on") & (table["metric"] == "kl_nats") & (table["cohort"] == "all")]
    on = on.sort_values("bin_center_h")
    note = None
    if len(on) >= 2:
        early, late = on.iloc[-1], on.iloc[0]
        count = 0
        if len(tests):
            count = int((tests[(tests["population"] == "gate_on") & (tests["metric"] == "kl_nats")]
                         ["significant"].astype(str).str.lower() == "true").sum())
        note = (f"With the gate on, the median KL over recordings goes from {fmt(early['median'])} nats at "
                f"{fmt(early['bin_center_h'], 3)} h before delivery to {fmt(late['median'])} nats in the last "
                f"half hour. The classes differ in {count} of the tested half hours.")
    return plot_panel(
        "class-trajectory", "Class trajectories over labour, per recording", variants,
        opt_label="Quantity",
        how="The gate_open analysis cuts labour into half hours. In each half hour a recording "
            "gives one value: the mean over its own anchors of the selected population. A "
            "recording counts only if it has at least 10 such anchors in the half hour. The plot "
            "shows the median and the quartiles over the recordings of each class.",
        shows="One line per clinical class, with a band between the 25th and the 75th percentile. "
              "Delivery is at the right edge. The Anchors filter at the top changes the "
              "population, and the selector changes the quantity. A purple dotted line marks a "
              "half hour in which the classes differ. Hover a point for the number of recordings.",
        read="Select 'Gate on' and 'KL' to see the KL trajectory on the gate-on anchors alone. "
             "Over all anchors, a change in the KL can come from the gate opening more often or "
             "from a larger KL per open anchor. With the gate on, only the second cause is left. "
             "Class lines that stay inside each other's bands do not differ. The HIE class has "
             "few recordings, so its line is noisy.",
        run=note,
        sources=[GATE + "gate_open_trajectory.csv", GATE + "gate_open_trajectory_significance.csv"],
        pdfs=[GATE + "gate_open_trajectory.pdf"],
    )


def class_trajectory_tests(run: RunFiles) -> Dict[str, Any]:
    """Build the card: whether the classes differ in each half hour of labour, per population.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv(GATE + "gate_open_trajectory_significance.csv")
    if table.empty:
        raise MissingInput(GATE + "gate_open_trajectory_significance.csv (no rows)", "gate_open")
    alpha = float(table["alpha"].iloc[0]) if "alpha" in table.columns else 0.05
    labels = {metric: label for metric, label, _, _ in _TRAJECTORY_METRICS}

    def build(key: str) -> Optional[Dict[str, Any]]:
        rows = table[table["population"] == key]
        if rows.empty:
            return None
        traces = []
        for index, metric in enumerate(labels):
            part = rows[rows["metric"] == metric].sort_values("bin_center_h")
            if len(part):
                traces.append(line(part["bin_center_h"], part["p_value"], labels[metric],
                                   [BLUE, ORANGE, GREEN, PURPLE, VERMILLION, GRAY][index % 6], markers=True,
                                   customdata=part["p_holm"].tolist(),
                                   hover="raw p %{y:.3g}, corrected p %{customdata:.3g}<extra>"
                                         + labels[metric] + "</extra>"))
        figure = layout("hours before delivery", f"raw p-value of the class difference, {POPULATION_NAMES[key]}",
                        xaxis={"autorange": "reversed"}, yaxis={"type": "log"},
                        shapes=[hline(alpha, color=VERMILLION, dash="dash")])
        return variant(traces, figure, pop=key)

    flagged = table[table["significant"].astype(str).str.lower() == "true"]
    listed = "; ".join(
        f"'{labels.get(row['metric'], row['metric'])}' on {POPULATION_NAMES[row['population']]} anchors "
        f"at {fmt(row['bin_center_h'], 3)} h"
        for _, row in flagged.head(4).iterrows()
    )
    return plot_panel(
        "class-trajectory-tests", "Do the classes differ over labour? (per population)",
        _per_population(build),
        how="In each half hour, the analysis tests whether the per-recording values of the "
            "classes come from the same distribution (Kruskal-Wallis test). A class enters a test "
            "with at least three recordings. The p-values of one population and one quantity are "
            "then corrected for the number of half hours tested (Holm correction).",
        shows="One line per quantity, for the selected population. The y axis is the raw p-value "
              "on a logarithmic scale. The dashed red line is the significance level. Hover a "
              "point for the corrected p-value.",
        read="A raw p-value below the red line is only a hint, because 24 half hours are tested "
             "for each quantity and some fall below the line by chance. The decision uses the "
             "corrected p-value. Few or no significant half hours mean that this readout does "
             "not separate the classes at the level of recordings.",
        run=f"{len(flagged)} of {len(table)} tests are significant after correction"
            + (f": {listed}." if len(flagged) else "."),
        sources=[GATE + "gate_open_trajectory_significance.csv"], pdfs=[GATE + "gate_open_trajectory.pdf"],
    )


def gain_vs_contraction(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the forecast gain and the KL against contraction age, per population.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = _pooled(run.csv(GATE + "gate_open_by_contraction_age.csv"))

    def build(key: str) -> Optional[Dict[str, Any]]:
        rows = table[table["population"] == key]
        if rows.empty:
            return None
        traces = [
            line(rows["bin_center"], rows["kl_nats"], "KL (nats)", VERMILLION, markers=True),
            line(rows["bin_center"], rows["gain_mc_nats"], "forecast gain (nats)", ORANGE,
                 markers=True, axis="y2"),
        ]
        figure = layout("seconds since the last contraction onset",
                        f"KL per anchor (nats), {POPULATION_NAMES[key]}",
                        yaxis2=second_axis("forecast gain per anchor (nats)"))
        return variant(traces, figure, pop=key)

    on = table[table["population"] == "gate_on"]
    best = on.loc[on["gain_mc_nats"].idxmax()]
    return plot_panel(
        "gain-contraction", "KL and forecast gain through a contraction", _per_population(build),
        how="The anchors of the selected population are put in 15-second bins of the time since "
            "the last contraction onset. Each point is the anchor-level mean in the bin.",
        shows="The red line is the KL, on the left axis. The orange line is the Monte Carlo "
              "forecast gain, on the right axis.",
        read="A peak shortly after the onset means that the UA carries the most useful "
             "information while the contraction develops. With 'Gate on' selected the plot shows "
             "the size of the effect at the anchors where the model reads the UA.",
        run=f"With the gate on, the forecast gain is largest {fmt(best['bin_center'], 3)} s after a "
            f"contraction onset, at {fmt(best['gain_mc_nats'])} nats per anchor.",
        sources=[GATE + "gate_open_by_contraction_age.csv"], pdfs=[GATE + "gate_open.pdf"],
    )


def gain_per_recording(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the per-recording forecast gain per class, per population.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv(GATE + "gate_open_per_recording.csv")

    def build(key: str) -> Optional[Dict[str, Any]]:
        column = f"{key}_gain_mc_nats"
        if column not in table.columns:
            return None
        traces = []
        for index, (name, rows) in enumerate(table.groupby("clinical_class")):
            traces.append({
                "type": "box", "y": rows[column].round(3).tolist(), "name": str(name),
                "marker": {"color": class_color(name, index), "size": 4}, "boxpoints": "all",
                "jitter": 0.5, "pointpos": 0, "line": {"width": 2},
                "hovertemplate": "%{y:.3g} nats<extra>" + str(name) + "</extra>",
            })
        figure = layout("", f"forecast gain per anchor (nats), {POPULATION_NAMES[key]}",
                        showlegend=False, shapes=[hline(0.0)])
        return variant(traces, figure, pop=key)

    positive = float(np.mean(table["gate_on_gain_mc_nats"].to_numpy(dtype=float) > 0))
    return plot_panel(
        "gain-recording", "Forecast gain per recording", _per_population(build),
        how="For each recording the analysis takes the mean Monte Carlo forecast gain over the "
            "anchors of the selected population.",
        shows="One dot per recording, grouped by clinical class. The box marks the median and the "
              "quartiles. The dotted line is zero.",
        read="Dots above zero are recordings where the UA improved the forecast. A box that sits "
             "above zero in every class means that the gain is general. A few very high dots can "
             "lift the mean, so compare the median with the mean of the summary plot.",
        run=f"With the gate on, {pct(positive)} of the recordings have a positive forecast gain.",
        sources=[GATE + "gate_open_per_recording.csv"],
        pdfs=[GATE + "gate_open_per_recording_by_clinical_class.pdf"],
    )


def gain_vs_kl_scatter(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the per-recording forecast gain against the per-recording KL, per population.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv(GATE + "gate_open_per_recording.csv")

    def build(key: str) -> Optional[Dict[str, Any]]:
        traces = []
        for index, (name, rows) in enumerate(table.groupby("clinical_class")):
            traces.append({
                "type": "scatter", "mode": "markers", "name": str(name),
                "x": rows[f"{key}_kl_nats"].round(4).tolist(),
                "y": rows[f"{key}_gain_mc_nats"].round(3).tolist(),
                "marker": {"color": class_color(name, index), "size": 7, "opacity": 0.75},
                "hovertemplate": "KL %{x:.3g}, gain %{y:.3g}<extra>" + str(name) + "</extra>",
            })
        figure = layout(f"KL per anchor (nats), {POPULATION_NAMES[key]}",
                        f"forecast gain per anchor (nats), {POPULATION_NAMES[key]}", shapes=[hline(0.0)])
        return variant(traces, figure, pop=key)

    pair = table[["gate_on_kl_nats", "gate_on_gain_mc_nats"]].dropna()
    rho = float(pair.corr(method="spearman").iloc[0, 1]) if len(pair) > 2 else float("nan")
    return plot_panel(
        "gain-kl-scatter", "Does more KL give more gain?", _per_population(build),
        how="Each recording gives one point: its mean KL and its mean Monte Carlo forecast gain "
            "over the anchors of the selected population.",
        shows="The x axis is the KL and the y axis is the forecast gain. The colour is the "
              "clinical class.",
        read="Points that rise to the right mean that recordings where the UA changes the belief "
             "more also forecast better. Points far to the right but near zero gain are recordings "
             "where the model uses UA information that does not help.",
        run=f"With the gate on, the rank correlation between KL and gain over recordings is "
            f"{fmt(rho, 2)}.",
        sources=[GATE + "gate_open_per_recording.csv"],
    )


# =================================================================================================
# Section: lags
# =================================================================================================
def lag_profile_populations(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the KL attribution per lag, one curve per population.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv(GATE + "gate_open_lag_profile.csv")
    variants = []
    for cohort in pd.unique(table["cohort"]):
        traces: List[Any] = []
        for key in POPULATION_KEYS:
            rows = _pooled(table, key, cohort)
            if rows.empty:
                continue
            traces.append(band(rows["lag_s"], rows["kl_nats_lo"], rows["kl_nats_hi"], POPULATION_COLORS[key]))
            traces.append(line(rows["lag_s"], rows["kl_nats"], POPULATION_NAMES[key], POPULATION_COLORS[key]))
        figure = layout("lag (s, stored-coefficient time)", "KL attributed to the lag (nats per anchor)",
                        yaxis={"type": "log"})
        variants.append(variant(traces, figure, opt="all classes" if cohort == "all" else cohort))
    on = _pooled(table, "gate_on")
    within = float(on.loc[on["lag_s"] <= 30.0, "kl_share"].sum())
    return plot_panel(
        "lag-profile-populations", "Where on the lag axis the KL comes from", variants,
        opt_label="Cohort",
        how=f"At each anchor the model attends over {Geometry(run).lags_text}. "
            "The collection pass splits the anchor's KL over those lags. The analysis averages that "
            "split within a recording and then over recordings, for each gate population. The band "
            "is a 95% bootstrap interval over recordings.",
        shows="The x axis is the lag. The y axis is the mean KL attributed to that lag, on a "
              "logarithmic scale. There is one curve per gate population.",
        read="A curve that falls fast means that the model uses only the most recent UA. A bump "
             "at a later lag would mean that the model looks back to a specific earlier time. The "
             "gate-on and gate-off curves differ in height by design, so compare their shapes. "
             + LAG_NOTE,
        run=f"With the gate on, lag 0 carries {pct(float(on['kl_share'].iloc[0]))} of the KL and the "
            f"lags up to 30 s carry {pct(within, 1)}.",
        sources=[GATE + "gate_open_lag_profile.csv"], pdfs=[GATE + "gate_open_lags.pdf"],
    )


def lag_shares(run: RunFiles) -> Dict[str, Any]:
    """Build the card: three readings of the lag structure of one population.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = _pooled(run.csv(GATE + "gate_open_lag_profile.csv"))

    def build(key: str) -> Optional[Dict[str, Any]]:
        rows = table[table["population"] == key]
        if rows.empty:
            return None
        traces = [
            line(rows["lag_s"], rows["kl_share"], "share of the KL", VERMILLION, markers=True,
                 hover="%{y:.2%}<extra>share of the KL</extra>"),
            line(rows["lag_s"], rows["attention"], "attention weight", PURPLE, markers=True,
                 hover="%{y:.2%}<extra>attention weight</extra>"),
            line(rows["lag_s"], rows["argmax_share"], "share of anchors that peak here", GREEN,
                 markers=True, hover="%{y:.2%}<extra>anchors that peak here</extra>"),
        ]
        figure = layout("lag (s, stored-coefficient time)", f"share, {POPULATION_NAMES[key]}",
                        yaxis={"type": "log", "range": [-3.3, 0.02], "tickformat": ".1%"})
        return variant(traces, figure, pop=key)

    on = table[table["population"] == "gate_on"]
    return plot_panel(
        "lag-shares", "Three readings of the lag", _per_population(build),
        how="For the selected population, the plot draws three profiles over the lags. Each "
            "profile sums to 100%. The KL share is the KL attribution divided by its total. The "
            "attention weight is the mean attention the model puts on the lag. The third profile "
            "counts, for each lag, the anchors whose largest KL attribution sits there.",
        shows="The x axis is the lag. The y axis is a share on a logarithmic scale.",
        read="The attention says where the model looks. The KL share says which lags change the "
             "belief. If the attention is spread out but the KL share is concentrated, the model "
             "looks at many lags and uses few of them. " + LAG_NOTE,
        run=f"With the gate on, lag 0 has {pct(float(on['attention'].iloc[0]))} of the attention, "
            f"{pct(float(on['kl_share'].iloc[0]))} of the KL, and is the peak lag at "
            f"{pct(float(on['argmax_share'].iloc[0]))} of the anchors.",
        sources=[GATE + "gate_open_lag_profile.csv"], pdfs=[GATE + "gate_open_lags.pdf"],
    )


#: The fields a lag heatmap can colour: selector label, table column, colour-bar label, log scale.
_LAG_FIELDS = (
    ("KL share (each row sums to 100%)", "kl_share", "share of KL", True),
    ("KL attribution (nats)", "kl_nats", "nats", True),
    ("Attention weight", "attention", "attention", True),
)


def _lag_heatmap(
    run: RunFiles, identifier: str, title: str, filename: str, *, ytitle: str,
    how: str, shows: str, read: str,
    run_note: Optional[Callable[[pd.DataFrame], Optional[str]]] = None,
    by_population: bool = True,
) -> Dict[str, Any]:
    """Build a heatmap card of the lag structure against one clock.

    Args:
        run: The run's files.
        identifier: A stable name, unique on the page.
        title: The card title.
        filename: The lag table of the ``gate_open`` analysis to draw.
        ytitle: The title of the clock axis.
        how: How the plot was made.
        shows: What the axes and the colour are.
        read: How to read it.
        run_note: A function from the table to this run's note, or ``None``.
        by_population: Give the card one variant per population. ``False`` draws a table that
            has no population column.

    Returns:
        The card, with one variant per population and per colour field.

    Raises:
        MissingInput: If the table is absent.
    """
    table = run.csv(GATE + filename)
    variants = []
    for key in (POPULATION_KEYS if by_population else (None,)):
        rows = table if key is None else table[table["population"] == key]
        if rows.empty:
            continue
        for label, column, bar_label, log in _LAG_FIELDS:
            field = rows.pivot_table(index="bin_center", columns="lag_s", values=column, aggfunc="mean")
            trace = heatmap(
                field.columns.to_numpy(), field.index.to_numpy(), field.to_numpy(), label=bar_label,
                log=log, floor=1e-4 if column == "kl_nats" else 1e-3,
                hover="lag %{x:.0f} s, " + ytitle + " %{y:.4g}: %{customdata:.3g}<extra></extra>",
            )
            name = "" if key is None else f", {POPULATION_NAMES[key]}"
            figure = layout("lag (s, stored-coefficient time)", f"{ytitle}{name}")
            variants.append(variant([trace], figure, pop=key, opt=label))
    return plot_panel(
        identifier, title, variants, opt_label="Colour", how=how, shows=shows, read=read,
        run=run_note(table) if run_note else None, wide=True, height=430,
        sources=[GATE + filename], pdfs=[GATE + "gate_open_lags.pdf"],
        scope=None if by_population else "all anchors, binned by the gate",
    )


def lag_vs_contraction(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the lag structure against the age of the contraction.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    def note(table: pd.DataFrame) -> Optional[str]:
        rows = table[table["population"] == "gate_on"]
        far = rows[rows["lag_s"] > 30.0].groupby("bin_center")["kl_share"].sum()
        if far.empty or not np.isfinite(far).any():
            # A lag window of 30 s or less has no lag beyond it to report on.
            return None
        return (f"With the gate on, the share of the KL beyond a lag of 30 s is largest "
                f"{fmt(far.idxmax(), 3)} s after a contraction onset ({pct(far.max(), 1)}) and smallest "
                f"at {fmt(far.idxmin(), 3)} s ({pct(far.min(), 1)}).")

    return _lag_heatmap(
        run, "lag-contraction", "Lag structure through a contraction",
        "gate_open_lag_by_contraction_age.csv", ytitle="seconds since contraction onset",
        how="The anchors of the selected population are put in 15-second bins of the time since "
            "the last contraction onset. For each bin the analysis takes the anchor-level mean of "
            "the KL attribution per lag. The 'KL share' view divides each row by its own total.",
        shows="Each row is one contraction age and each column is one lag. The colour is "
              "logarithmic, so small values at long lags stay visible.",
        read="If the model tracked the contraction onset, the bright area would move to longer "
             "lags as the contraction ages, and form a diagonal. A bright column at lag 0 with no "
             "diagonal means that the model reads the current UA level and not a past event. "
             + LAG_NOTE,
        run_note=note,
    )


def lag_vs_time(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the lag structure against time before delivery.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    return _lag_heatmap(
        run, "lag-time", "Lag structure over labour", "gate_open_lag_by_time_to_delivery.csv",
        ytitle="hours before delivery",
        how="The anchors of the selected population are put in 30-minute bins of their time "
            "before delivery. For each bin the analysis takes the anchor-level mean of the KL "
            "attribution per lag.",
        shows="Each row is one half hour of labour and each column is one lag. Delivery is the "
              "bottom row. The colour is logarithmic.",
        read="Rows that look the same mean that the model uses the same lags through labour. A "
             "bright area that widens toward the bottom would mean that the model looks further "
             "back near delivery. " + LAG_NOTE,
    )


def lag_vs_gate(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the lag structure against the gate value.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    return _lag_heatmap(
        run, "lag-gate", "Lag structure against the gate value", "gate_open_lag_by_gate.csv",
        ytitle="gate value g", by_population=False,
        how="All scored anchors are put in ten bins of the gate value. For each bin the analysis "
            "takes the anchor-level mean of the KL attribution per lag.",
        shows="Each row is one gate bin and each column is one lag. The colour is logarithmic.",
        read="This plot shows whether the model reads different lags when the gate opens further. "
             "Rows that look the same in the 'KL share' view mean that the gate only scales the "
             "amount of information and does not change where it comes from. " + LAG_NOTE,
    )


def lag_centroid_clocks(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the lag centroid and the lag-0 share against the two clocks, per population.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    variants = []
    for clock, filename, xtitle, reverse in (
        ("Time before delivery", "gate_open_by_time_to_delivery.csv", "hours before delivery", True),
        ("Contraction age", "gate_open_by_contraction_age.csv", "seconds since contraction onset", False),
    ):
        table = _pooled(run.csv(GATE + filename))
        for quantity, column, ytitle in (
            ("KL centroid", "kl_centroid_s", "mean lag of the KL (s)"),
            ("lag-0 share of the KL", "kl_lag0_share", "share of the KL at lag 0"),
            ("attention centroid", "attention_centroid_s", "mean lag of the attention (s)"),
        ):
            traces = []
            for key in POPULATION_KEYS:
                rows = table[table["population"] == key]
                if len(rows):
                    traces.append(line(rows["bin_center"], rows[column], POPULATION_NAMES[key],
                                       POPULATION_COLORS[key], markers=True))
            figure = layout(xtitle, ytitle, xaxis={"autorange": "reversed"} if reverse else {})
            variants.append(variant(traces, figure, opt=f"{quantity} vs {clock.lower()}"))
    return plot_panel(
        "lag-centroid-clocks", "Does the lag move over labour or through a contraction?", variants,
        opt_label="Reading",
        how="For each clock bin and each population, the analysis takes the mean KL attribution "
            "per lag. The centroid is the mean lag weighted by that attribution. The lag-0 share "
            "is the part of the attribution at lag 0.",
        shows="One line per gate population. The selector changes the reading and the clock.",
        read="A centroid of 0 s means that all the information comes from the current step. A "
             "larger centroid means that the model looks further back. A flat line means that the "
             "lag does not depend on the clock. " + LAG_NOTE,
        sources=[GATE + "gate_open_by_time_to_delivery.csv", GATE + "gate_open_by_contraction_age.csv"],
    )


def lag_centroid_per_recording(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the per-recording lag centroid per class, per population.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv(GATE + "gate_open_per_recording.csv")

    def build(key: str) -> Optional[Dict[str, Any]]:
        traces = []
        for index, (name, rows) in enumerate(table.groupby("clinical_class")):
            traces.append({
                "type": "box", "y": rows[f"{key}_kl_centroid_s"].round(3).tolist(), "name": str(name),
                "marker": {"color": class_color(name, index), "size": 4}, "boxpoints": "all",
                "jitter": 0.5, "pointpos": 0, "line": {"width": 2},
                "hovertemplate": "%{y:.3g} s<extra>" + str(name) + "</extra>",
            })
        return variant(traces, layout("", f"mean lag of the KL (s), {POPULATION_NAMES[key]}",
                                      showlegend=False, yaxis={"rangemode": "tozero"}), pop=key)

    centroid = table["gate_on_kl_centroid_s"].to_numpy(dtype=float)
    return plot_panel(
        "lag-centroid-recording", "Mean lag per recording", _per_population(build),
        how="For each recording the analysis averages the KL attribution per lag over the anchors "
            "of the selected population. The centroid is the mean lag weighted by that profile.",
        shows="One dot per recording, grouped by clinical class. The box marks the median and the "
              "quartiles.",
        read="Boxes at the same height mean that the classes use the same lags. A class with a "
             "higher box would use older UA information. " + LAG_NOTE,
        run=f"With the gate on, the median recording has a KL centroid of "
            f"{fmt(np.nanmedian(centroid), 3)} s, and the middle half of the recordings lie between "
            f"{fmt(np.nanquantile(centroid, 0.25), 3)} and {fmt(np.nanquantile(centroid, 0.75), 3)} s.",
        sources=[GATE + "gate_open_per_recording.csv"],
        pdfs=[GATE + "gate_open_per_recording_by_clinical_class.pdf"],
    )


# =================================================================================================
# Section: the gate on the signal
# =================================================================================================
def _physical_signals(run: RunFiles) -> Callable[[Any, str, Any], Any]:
    """Return the conversion of a raw signal from loader units to its physical unit.

    The conversion is the sample page's own
    (:func:`teb_vae.lag_attn_cfs.eval.traces.raw_signal_scales` and
    :func:`~teb_vae.lag_attn_cfs.eval.traces.physical_raw`), so this page and the sample dashboard
    draw one signal in one unit, with the same gap rule. That module needs ``torch``, so it is
    imported here and not at the top of this ``torch``-free module. When it cannot be imported, or
    the run's configuration cannot be read, the signal stays in loader units.

    Args:
        run: The run's files, read for the merged configuration.

    Returns:
        A function ``(values, name, valid) -> (array, unit)`` for ``name`` in ``'fhr'`` or ``'up'``.
    """
    try:
        import yaml

        from teb_vae.lag_attn_cfs.eval import traces

        with open(run.require("resolved_config.yaml"), encoding="utf-8") as handle:
            scales = traces.raw_signal_scales(yaml.safe_load(handle))
        return lambda values, name, valid=None: traces.physical_raw(values, name, scales, valid)
    except Exception:  # noqa: BLE001 - no conversion available: the units stay the loader's
        def unconverted(values: Any, name: str, valid: Any = None) -> Any:
            """The signal in loader units, ``NaN`` where it holds no reading."""
            array = np.asarray(values, dtype=np.float64).reshape(-1).copy()
            if valid is not None:
                array[~np.asarray(valid, dtype=bool).reshape(-1)[: array.size]] = np.nan
            return array, "normalised"

        return unconverted


def signal_explorer(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the interactive view of single segments: the signals, the gate, the KL, the gain, the lags.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    retained = run.npz(RETAINED)
    for name in ("fhr_raw", "up_raw", "anchor_index", "waveforms_sample_index"):
        if name not in retained:
            raise MissingInput(f"{RETAINED} (array {name})", "the collection pass")
    anchors = run.parquet(PER_ANCHOR, [
        "sample_index", "anchor", "event_gate", "kld_per_t", "mc_pred_gap", "guid", "epoch",
        "clinical_class",
    ])
    if "event_gate" not in anchors.columns:
        raise MissingInput(f"{PER_ANCHOR} (column event_gate)", "the collection pass")
    kept = np.asarray(retained["waveforms_sample_index"], dtype=np.int64)
    wanted = anchors["sample_index"].isin(kept).to_numpy()
    positions = np.flatnonzero(wanted)
    subset = anchors.iloc[positions].assign(_row=positions)
    vectors = run.npz(PER_ANCHOR_VECTORS) if run.exists(PER_ANCHOR_VECTORS) else {}
    to_physical = _physical_signals(run)
    units = {"fhr": "normalised", "up": "normalised"}
    raw_per_step = int(retained["fhr_raw"].shape[1] // retained["weight"].shape[1]) if "weight" in retained else 16
    stride = max(raw_per_step // EXPLORER_RAW_PER_STEP, 1)

    contractions: Dict[int, List[List[float]]] = {}
    if run.exists("events/contraction_detections.csv"):
        detections = run.csv("events/contraction_detections.csv")
        if {"sample_index", "onset_s", "end_s"} <= set(detections.columns):
            for sample, rows in detections.groupby("sample_index"):
                contractions[int(sample)] = rows[["onset_s", "end_s"]].round(1).values.tolist()
    # A sample page is found by its segment, (recording, segment start), and not by an index:
    # the page manifest numbers the dataset rows and the collection pass numbers its own rows,
    # and the two numberings differ.
    pages: Dict[Any, str] = {}
    if run.exists("samples/sample_pages.csv"):
        listed = run.csv("samples/sample_pages.csv")
        if {"guid", "epoch", "dashboard_file", "selection"} <= set(listed.columns):
            for _, row in listed.dropna(subset=["dashboard_file"]).iterrows():
                pages.setdefault((str(row["guid"]), round(float(row["epoch"]), 3)),
                                 f"samples/{row['selection']}/{row['dashboard_file']}")

    by_sample = {int(sample): rows.sort_values("anchor") for sample, rows in subset.groupby("sample_index")}

    def gate_on_share(index: int) -> float:
        """The share of one kept segment's scored anchors with the gate on."""
        gate_values = by_sample[int(kept[index])]["event_gate"].to_numpy(dtype=float)
        return float(np.mean(gate_values >= GATE_SHUT_BELOW))

    # The segments with the most gate-on time come first, so the view opens on a segment that
    # shows the gate at work. Every label states its gate-on share.
    order = sorted((index for index, sample in enumerate(kept) if int(sample) in by_sample),
                   key=gate_on_share, reverse=True)
    # A fixed stride through the sorted draw keeps the whole range of gate-on shares in a capped list.
    if len(order) > MAX_EXPLORER_SEGMENTS:
        order = [order[int(i)] for i in np.linspace(0, len(order) - 1, MAX_EXPLORER_SEGMENTS).round()]

    segments: List[Dict[str, Any]] = []
    lag_width = 0
    for index in order:
        sample = int(kept[index])
        rows = by_sample[sample]
        fhr, units["fhr"] = to_physical(retained["fhr_raw"][index], "fhr", None)
        ua, units["up"] = to_physical(
            retained["up_raw"][index], "up", retained["up_valid"][index] if "up_valid" in retained else None
        )
        fhr, ua = fhr[::stride], ua[::stride]
        # The scored anchors of a segment can have gaps. Each readout is laid on the dense anchor
        # grid from the first to the last scored anchor, so a gap is drawn as a gap.
        steps = rows["anchor"].to_numpy(dtype=np.int64)
        first, width = int(steps.min()), int(steps.max() - steps.min() + 1)

        def dense(values: np.ndarray) -> np.ndarray:
            """One readout on the dense anchor grid, ``NaN`` at an anchor that was not scored."""
            grid = np.full((width, *values.shape[1:]), np.nan)
            grid[steps - first] = values
            return grid

        epoch = float(rows["epoch"].iloc[0])
        hours = -(epoch + float(first) * SECONDS_PER_STEP) / 3600.0
        gate = rows["event_gate"].to_numpy(dtype=float)
        label = (f"gate on {100 * np.mean(gate >= GATE_SHUT_BELOW):.0f}% · {rows['clinical_class'].iloc[0]} · "
                 f"{hours:.1f} h before delivery · sample {sample}")
        entry: Dict[str, Any] = {
            "label": label, "class": str(rows["clinical_class"].iloc[0]), "hours": hours,
            "sample": sample, "guid": str(rows["guid"].iloc[0]),
            "anchors": list(range(first, first + width)), "n_scored": int(len(steps)),
            "contractions": contractions.get(sample, []),
            "page": pages.get((str(rows["guid"].iloc[0]), round(epoch, 3))),
            "fhr": encode_array(fhr), "ua": encode_array(ua), "gate": encode_array(dense(gate)),
            "kl": encode_array(dense(rows["kld_per_t"].to_numpy(dtype=float))),
            "gain": encode_array(dense(rows["mc_pred_gap"].to_numpy(dtype=float))),
        }
        for key, name in (("kl_map", "kl_lag_map"), ("attention_map", "attention_lag_map")):
            if name in vectors:
                field = np.asarray(vectors[name][rows["_row"].to_numpy()], dtype=np.float64)
                lag_width = field.shape[1]
                entry[key] = encode_array(dense(field))
        segments.append(entry)
    if not segments:
        raise MissingInput(f"{RETAINED} (no retained segment has scored anchors)", "the collection pass")
    physical = units["fhr"] != "normalised"
    return {
        "kind": "explorer", "id": "signal-explorer", "title": "The gate on the signal",
        "segments": segments, "dt": SECONDS_PER_STEP, "raw_dt": SECONDS_PER_STEP * stride / raw_per_step,
        # The compensated lag axis of the run, the one every lag table of the analyses carries.
        "lag_s": Geometry(run).lag_seconds(lag_width),
        "fhr_unit": units["fhr"], "ua_unit": units["up"],
        "gate_shut_below": GATE_SHUT_BELOW, "gate_open_from": GATE_OPEN_FROM,
        "explain": {
            "how": "The collection pass keeps the raw signals of a seeded draw of segments. This "
                   "view joins each kept segment with its per-anchor readouts: the gate, the KL, "
                   "the Monte Carlo forecast gain and the two lag maps. "
                   + ("The signals are converted back to physical units with the run's statistics file."
                      if physical else
                      "The run's statistics file is not readable here, so the signals are in normalised units."),
            "shows": "From top to bottom: the FHR, the UA, the gate value, the KL with the forecast "
                     "gain, and the lag map. The blue shading marks the time where the gate is on "
                     f"(g of {GATE_SHUT_BELOW:g} or more). The green shading on the UA row marks detected "
                     "contractions. The dotted vertical line is the first scored anchor. Before it "
                     "the model only warms up, so there is no gate value.",
            "read": "Tick 'Hide the gate-off parts' to keep only the signal that the model read "
                    "with the gate on. Check whether the shaded spans sit on the contractions, "
                    "and whether the KL and the gain rise inside them. In the lag map, a bright "
                    "bottom row means that the model uses the current UA value. " + LAG_NOTE,
            "run": None,
        },
        "sources": [RETAINED, PER_ANCHOR, PER_ANCHOR_VECTORS], "pdfs": [],
    }


__all__ = [
    "class_trajectory", "class_trajectory_tests", "gain_per_recording", "gain_vs_contraction",
    "gain_vs_kl_scatter",
    "gate_headline", "gate_histogram", "gate_runs", "gate_share_by_class", "gate_share_per_recording",
    "gate_states", "gate_states_over_time", "gate_value_payoff", "gate_vs_contraction",
    "lag_centroid_clocks", "lag_centroid_per_recording", "lag_profile_populations",
    "lag_shares", "lag_vs_contraction", "lag_vs_gate", "lag_vs_time", "population_contributions",
    "population_distributions", "population_summary", "signal_explorer",
]
