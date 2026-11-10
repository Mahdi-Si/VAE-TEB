r"""The run dashboard's panels for the registered analyses, over all scored anchors.

Each function here builds one card from the tables one analysis wrote, and returns it. The cards
of the gate populations live in :mod:`~teb_vae.lag_attn_cfs.eval.run_dashboard_gate`; these cards
are the standard evaluation, and none of them follows the page's ``Anchors`` filter.

A function raises :class:`~teb_vae.lag_attn_cfs.eval.run_dashboard_core.MissingInput` when its
table is absent, and the assembler then draws a card that names the missing file and the analysis
that writes it. The four tuples at the bottom of the module give each section's cards in reading
order.

**A lag axis here is stored-coefficient time**, as everywhere in this package. The stored timeline
is canonical, and no panel applies a correction to it.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from teb_vae.lag_attn_cfs.eval.run_dashboard_core import (
    BLUE,
    GRAY,
    GREEN,
    ORANGE,
    PURPLE,
    SECONDS_PER_STEP,
    SERIES,
    TEAL,
    VERMILLION,
    Geometry,
    MissingInput,
    RunFiles,
    band,
    bars,
    class_color,
    fmt,
    heatmap,
    hline,
    layout,
    line,
    pct,
    plot_panel,
    second_axis,
    table_panel,
    variant,
)

#: The sentence a lag-resolved card carries about its axis.
LAG_NOTE = (
    "The lag axis is stored-coefficient time. It says how many seconds back in the stored UA "
    "features the model reads. It is not a physiological delay."
)

#: The fixed styles of the two forecast branches, the evaluation figures' own.
BASE_COLOR = "#64748B"
FULL_COLOR = VERMILLION

#: How the two headline metrics read on an axis.
METRIC_TITLES = {
    "pred_gap_mc_nats": "forecast gain per anchor (nats)",
    "mc_pred_gap": "forecast gain per anchor (nats)",
    "source_conditioned_kl_raw_nats": "KL per anchor (nats)",
    "source_conditioned_kl_raw": "KL per anchor (nats)",
    "kld_per_t": "KL per anchor (nats)",
    "pred_gap_mean_nats": "forecast gain of the mean forecast (nats)",
}
METRIC_NAMES = {
    "pred_gap_mc_nats": "Forecast gain", "mc_pred_gap": "Forecast gain",
    "source_conditioned_kl_raw_nats": "KL", "source_conditioned_kl_raw": "KL", "kld_per_t": "KL",
    "pred_gap_mean_nats": "Forecast gain, mean forecast",
}


def _classes(values: Sequence[Any]) -> List[str]:
    """List the distinct labels of a column in a fixed order.

    Args:
        values: The labels, possibly with missing entries.

    Returns:
        Healthy, acidosis and HIE where present, then every other label, sorted.
    """
    known = ["healthy", "acidosis", "hie"]
    present = [str(value) for value in pd.unique(pd.Series(values).dropna())]
    return [name for name in known if name in present] + sorted(set(present) - set(known))


def _box(name: str, color: str, values: Any, hover_unit: str = "") -> Dict[str, Any]:
    """Build a box trace with every point drawn, one point per recording.

    Args:
        name: The category the box stands for.
        color: The colour of the box and its points.
        values: One value per recording. A value that is not finite is left out.
        hover_unit: Text appended to the value in the hover label.

    Returns:
        The trace.
    """
    return {
        "type": "box", "y": [round(float(v), 4) for v in np.asarray(values, dtype=float) if np.isfinite(v)],
        "name": name, "marker": {"color": color, "size": 4}, "boxpoints": "all", "jitter": 0.5,
        "pointpos": 0, "line": {"width": 2},
        "hovertemplate": "%{y:.4g}" + hover_unit + "<extra>" + name + "</extra>",
    }


# =================================================================================================
# Lags
# =================================================================================================
def lag_kl_profile(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the KL attribution per lag over all anchors, under three accounting rules.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("lag_kl/lag_kl_profile.csv")
    seconds = table["compensated_seconds"]
    traces = [
        line(seconds, table["kl_nats"], "raw (sums to the KL)", VERMILLION, markers=True),
        line(seconds, table["kl_nats_support_corrected"], "support-corrected", BLUE, dash="dash"),
        line(seconds, table["kl_nats_untruncated"], "anchors where every lag exists", GREEN, dash="dot"),
    ]
    figure = layout("lag (s, stored-coefficient time)", "KL attributed to the lag (nats per anchor)",
                    yaxis={"type": "log"})
    peak = table.loc[table["kl_nats"].idxmax()]
    same = bool(np.allclose(table["kl_nats"], table["kl_nats_untruncated"], rtol=1e-6, equal_nan=True))
    return plot_panel(
        "lag-kl-profile", "KL per lag, all anchors", [variant(traces, figure)],
        how="The lag_kl analysis averages the per-lag split of the KL over segments and "
            "recordings, for every scored anchor. It computes the profile under three rules. The "
            "raw rule divides every lag by the same anchor count. The support-corrected rule "
            "divides each lag by the anchors at which that lag exists. The third rule keeps only "
            "the anchors at which every lag exists.",
        shows="The x axis is the lag. The y axis is the mean KL attributed to that lag, on a "
              "logarithmic scale. Each line is one accounting rule.",
        read="If the three lines lie on each other, every lag exists at every scored anchor and "
             "the profile needs no correction. A line that separates at long lags would mean that "
             "the raw profile under-counts those lags. " + LAG_NOTE,
        run=f"The profile peaks at a lag of {fmt(peak['compensated_seconds'], 3)} s, which carries "
            f"{pct(peak['share'])} of the KL. "
            + ("The three accounting rules agree, so no lag is truncated in this run."
               if same else "The accounting rules differ, so read the support-corrected line."),
        sources=["lag_kl/lag_kl_profile.csv", "lag_kl/lag_kl_summary.csv"],
        pdfs=["lag_kl/lag_kl_profile.pdf"],
    )


#: The profiles of ``lag_kl_stratified_profile.csv``: selector label, profile key, colour-bar label.
_STRATIFIED_PROFILES = (
    ("KL attribution", "kl_support_corrected", "nats"),
    ("Attention, mean of heads", "attention_support_corrected", "attention"),
    ("Attention, head 0", "attention_head_0", "attention"),
    ("Attention, head 1", "attention_head_1", "attention"),
    ("Attention, head 2", "attention_head_2", "attention"),
    ("Attention, head 3", "attention_head_3", "attention"),
)


def lag_profile_over_labour(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the lag profiles of the segments in each half hour of labour, over all anchors.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("lag_kl/lag_kl_stratified_profile.csv")
    rows = table[table["group_column"] == "time_window"]
    variants = []
    for label, key, bar_label in _STRATIFIED_PROFILES:
        selected = rows[rows["profile"] == key]
        if selected.empty:
            continue
        field = selected.pivot_table(index="bin_center_h", columns="compensated_seconds", values="value")
        trace = heatmap(field.columns.to_numpy(), field.index.to_numpy(), field.to_numpy(),
                        label=bar_label, log=True, floor=1e-4,
                        hover="lag %{x:.0f} s, %{y:.2f} h before delivery: %{customdata:.3g}<extra></extra>")
        variants.append(variant([trace], layout("lag (s, stored-coefficient time)", "hours before delivery"),
                                opt=label))
    if not variants:
        raise MissingInput("lag_kl/lag_kl_stratified_profile.csv (time_window rows)", "lag_kl")
    return plot_panel(
        "lag-profile-labour", "Lag profiles over labour, all anchors", variants, opt_label="Profile",
        wide=True, height=430,
        how="The lag_kl analysis groups the segments by the half hour of labour they start in. "
            "For each group it averages the per-lag KL attribution and the attention weights, "
            "first within a recording and then over recordings.",
        shows="Each row is one half hour before delivery and each column is one lag. The selector "
              "changes between the KL attribution and the attention of each head. The colour is "
              "logarithmic.",
        read="Compare this plot with 'Lag structure over labour' above, which uses the gate "
             "populations. This one uses every anchor, so its KL view also changes when the gate "
             "opens more or less often. The attention views show where each head looks, whether "
             "or not the gate lets the result through. " + LAG_NOTE,
        sources=["lag_kl/lag_kl_stratified_profile.csv"], pdfs=["attention/lag_heatmap.pdf"],
    )


def lag_profile_by_class(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the lag profiles per clinical class, over all anchors.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("lag_kl/lag_kl_stratified_profile.csv")
    rows = table[table["group_column"] == "clinical_class"]
    variants = []
    for label, key, unit in _STRATIFIED_PROFILES[:2]:
        selected = rows[rows["profile"] == key]
        traces = []
        for index, name in enumerate(_classes(selected["group"])):
            group = selected[selected["group"] == name]
            count = int(group["n_recordings"].iloc[0])
            traces.append(line(group["compensated_seconds"], group["value"], f"{name} (n={count})",
                               class_color(name, index)))
        if traces:
            variants.append(variant(
                traces, layout("lag (s, stored-coefficient time)", f"{label.lower()} ({unit})",
                               yaxis={"type": "log"}), opt=label))
    if not variants:
        raise MissingInput("lag_kl/lag_kl_stratified_profile.csv (clinical_class rows)", "lag_kl")
    return plot_panel(
        "lag-profile-class", "Lag profiles per clinical class, all anchors", variants,
        opt_label="Profile",
        how="The lag_kl analysis averages the per-lag KL attribution and the attention weights "
            "within each recording, then over the recordings of each clinical class.",
        shows="One line per class. The y axis is logarithmic.",
        read="Lines that lie on each other mean that the classes use the same lags. A class line "
             "that is higher at long lags would mean that the model looks further back in that "
             "class. " + LAG_NOTE,
        sources=["lag_kl/lag_kl_stratified_profile.csv"],
        pdfs=["lag_kl/lag_kl_per_recording_by_clinical_class.pdf"],
    )


def attention_heads(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the attention over lags of each head, with the KL each head carries.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("attention/attention_per_head.csv")
    traces = []
    notes = []
    total = float(table.groupby("head")["head_kl_nats"].first().sum())
    for index, (head, rows) in enumerate(table.groupby("head")):
        share = float(rows["head_kl_nats"].iloc[0]) / total if total > 0 else float("nan")
        name = f"head {int(head)} ({pct(share)} of the KL)"
        traces.append(line(rows["compensated_seconds"], rows["attention"], name, SERIES[index % len(SERIES)],
                           markers=True, hover="%{y:.2%}<extra>head " + str(int(head)) + "</extra>"))
        notes.append((share, int(head), float(rows.loc[rows["attention"].idxmax(), "compensated_seconds"])))
    notes.sort(reverse=True)
    figure = layout("lag (s, stored-coefficient time)", "mean attention weight",
                    yaxis={"type": "log", "range": [-3.3, 0.02], "tickformat": ".1%"})
    return plot_panel(
        "attention-heads", "Where each attention head looks", [variant(traces, figure)],
        how="The lag attention has several heads. Each head puts weights on the lags, and the "
            "weights of a head sum to 100%. The attention analysis averages the weights over "
            "anchors, segments and recordings. The legend also gives the share of the KL that "
            "each head carries.",
        shows="One line per head. The x axis is the lag and the y axis is the mean weight, on a "
              "logarithmic scale. A weight below 0.05% is not drawn.",
        read="A head with a sharp peak reads one lag. A head with a flat line spreads its "
             "attention and reads an average of the UA history. A head with a large share of the "
             "KL is one the model relies on. " + LAG_NOTE,
        run="; ".join(
            f"head {head} carries {pct(share)} of the KL and looks most at a lag of {fmt(peak, 3)} s"
            for share, head, peak in notes[:2]
        ) + ".",
        sources=["attention/attention_per_head.csv"],
        pdfs=["attention/attention_profile.pdf", "attention/lag_heatmap.pdf"],
    )


def attention_entropy(run: RunFiles) -> Dict[str, Any]:
    """Build the card: how spread out each head's attention is.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("attention/attention_entropy.csv")
    heads = table[table["scope"] == "head"]
    names = [f"head {int(head)}" for head in heads["head"]] + ["mean of heads"]
    pooled = table[table["scope"] == "head_averaged"]
    values = list(heads["normalised_against_attainable"]) + list(pooled["normalised_against_attainable"])
    colours = [SERIES[index % len(SERIES)] for index in range(len(heads))] + [GRAY]
    figure = layout("", "entropy as a share of the maximum", showlegend=False,
                    yaxis={"range": [0, 1], "tickformat": ".0%"})
    return plot_panel(
        "attention-entropy", "How spread out the attention is",
        [variant([bars(names, values, "entropy", colours)], figure)],
        how="The entropy of a head's attention weights measures how evenly the head spreads its "
            "attention over the lags. The attention analysis averages it over anchors and "
            "recordings and divides it by the largest value the lag window allows.",
        shows="One bar per head and one for the average of the heads. 0% means that all the "
              "attention is on one lag. 100% means that every lag gets the same weight.",
        read="A low bar is a head that selects a lag. A bar near 100% is a head that does not "
             "select, so its lag profile carries no timing information.",
        run=f"The most selective head is at {pct(float(np.nanmin(values[:-1])))} of the maximum entropy "
            f"and the least selective at {pct(float(np.nanmax(values[:-1])))}.",
        sources=["attention/attention_entropy.csv"],
        pdfs=["attention/attention_per_recording_by_clinical_class.pdf"],
    )


# =================================================================================================
# Forecast quality
# =================================================================================================
def forecast_horizon(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the forecast score and error at each step of the forecast horizon.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("forecast/forecast_horizon.csv")
    lead = table["lead_seconds"]
    views = (
        ("Forecast gain", [line(lead, table["gap_nats"], "gain, base minus full", ORANGE, markers=True)],
         "forecast gain per anchor at this step (nats)", [hline(0.0)]),
        ("Error (RMSE)", [
            line(lead, table["rmse_base_normalised"], "base forecast", BASE_COLOR, dash="dash"),
            line(lead, table["rmse_full_normalised"], "full forecast", FULL_COLOR),
        ], "RMSE (normalised units)", []),
        ("Score per coefficient (NLL)", [
            line(lead, table["d_base_nats_per_coefficient"], "base forecast", BASE_COLOR, dash="dash"),
            line(lead, table["d_full_nats_per_coefficient"], "full forecast", FULL_COLOR),
        ], "NLL per scored coefficient (nats)", []),
        ("Squared error over predicted variance", [
            line(lead, table["standardised_sq_base"], "base forecast", BASE_COLOR, dash="dash"),
            line(lead, table["standardised_sq_full"], "full forecast", FULL_COLOR),
        ], "mean squared error divided by predicted variance", [hline(1.0)]),
        ("Scored channels per step", [
            line(lead, table["scored_channels_per_anchor"], "scored channels", GRAY, markers=True),
        ], "feature channels scored at this step", []),
    )
    variants = [
        variant(traces, layout("seconds ahead of the anchor", ytitle, shapes=shapes), opt=label)
        for label, traces, ytitle, shapes in views
    ]
    best = table.loc[table["gap_nats"].idxmax()]
    return plot_panel(
        "forecast-horizon", "Forecast quality over the horizon", variants, opt_label="Quantity",
        how=f"At each anchor the model forecasts {Geometry(run).horizon_text} of the FHR features. "
            "The forecast analysis scores each step of that horizon on its own, summed over the "
            "feature channels and averaged over the anchors. These values come from one random "
            "draw of the latent variable, as in training.",
        shows="The x axis is how far ahead the forecast step is. The dashed gray line is the base "
              "forecast (FHR history only). The red line is the full forecast (with the UA). The "
              "selector changes the quantity.",
        read="Error grows with the lead time for every forecast. The interesting part is the "
             "distance between the two lines, which the 'Forecast gain' view draws directly. The "
             "variance view should be near 1, and a value below 1 means that the predicted "
             "variance is too large. Each channel is scored up to its own horizon, so the number "
             "of scored channels changes along the axis. The last view shows that number. A jump "
             "in a summed quantity at the same lead time comes from that change.",
        run=f"The UA helps most {fmt(best['lead_seconds'], 3)} s ahead, where it gains "
            f"{fmt(best['gap_nats'])} nats per anchor. At the first step the gain is "
            f"{fmt(table['gap_nats'].iloc[0], 2)} nats.",
        sources=["forecast/forecast_horizon.csv"], pdfs=["forecast/horizon_skill.pdf"],
    )


def forecast_skill(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the forecast error against three simple reference forecasts.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("forecast/forecast_skill.csv")
    names = {"persistence": "persistence (last value)", "climatology": "climatology (training mean)",
             "segment_mean": "segment mean"}
    traces = []
    for branch, colour in (("base", BASE_COLOR), ("full", FULL_COLOR)):
        rows = table[table["branch"] == branch]
        traces.append(bars([names.get(name, name) for name in rows["baseline"]], rows["mse_skill"],
                           f"{branch} forecast", colour, lo=rows["mse_skill_lo"], hi=rows["mse_skill_hi"]))
    figure = layout("", "share of the reference's squared error removed", barmode="group",
                    yaxis={"tickformat": ".0%", "rangemode": "tozero"})
    full = table[table["branch"] == "full"].set_index("baseline")["mse_skill"]
    return plot_panel(
        "forecast-skill", "Skill against simple reference forecasts", [variant(traces, figure)],
        how="The forecast analysis compares the model's squared error with three reference "
            "forecasts. Persistence repeats the last observed value. Climatology predicts the "
            "training mean. The segment mean predicts the mean of the segment. The skill is 1 "
            "minus the ratio of the model's error to the reference's error, per recording. The "
            "error bar is a 95% bootstrap interval over recordings.",
        shows="One group of bars per reference forecast. The gray bar is the base forecast and "
              "the red bar is the full forecast.",
        read="A skill of 0% means that the model is no better than the reference. A skill of 100% "
             "would be a perfect forecast. The two bars of a group are close, because most of the "
             "forecast comes from the FHR history and the UA adds a small part.",
        run="The full forecast removes " + ", ".join(
            f"{pct(value)} of the {name.replace('_', ' ')} error" for name, value in full.items()) + ".",
        sources=["forecast/forecast_skill.csv"], pdfs=["forecast/baseline_comparison.pdf"],
    )


def forecast_anchor_profile(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the forecast gain against the position of the anchor in its segment.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("forecast/forecast_anchor_profile.csv")
    seconds = table["anchor"] * SECONDS_PER_STEP
    traces = [
        line(seconds, table["pred_gap"], "forecast gain", ORANGE),
        line(seconds, table["nll_full_block"], "NLL of the full forecast", FULL_COLOR, axis="y2", width=1.8),
    ]
    figure = layout("position of the anchor in the segment (s)", "forecast gain per anchor (nats)",
                    yaxis2=second_axis("NLL per anchor (nats)"))
    return plot_panel(
        "forecast-anchor-profile", "Does the position in the segment matter?", [variant(traces, figure)],
        how="A segment is encoded on its own, so an early anchor has less history than a late "
            "one. The forecast analysis averages the gain and the score over all segments at each "
            "anchor position.",
        shows="The x axis is the time of the anchor from the start of the segment. The orange "
              "line is the forecast gain, on the left axis. The red line is the score of the full "
              "forecast, on the right axis.",
        read="Flat lines mean that the model has enough history at every scored anchor. A slope "
             "means that the result depends on how much history the anchor has, which is a "
             "property of the segment length and not of labour.",
        run=f"The gain is {fmt(table['pred_gap'].iloc[0])} nats at the first scored anchor and "
            f"{fmt(table['pred_gap'].iloc[-1])} nats at the last one.",
        sources=["forecast/forecast_anchor_profile.csv"], pdfs=["forecast/anchor_profile.pdf"],
    )


def forecast_scores_by_class(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the per-recording forecast scores per clinical class.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("forecast/forecast_scores.csv")
    options = (
        ("NLL, full forecast", "nll_full_block", "NLL per anchor (nats)"),
        ("NLL, base forecast", "nll_base_block", "NLL per anchor (nats)"),
        ("Squared error, full forecast", "sq_error_full", "mean squared error (normalised)"),
        ("Squared error, persistence", "sq_error_persistence", "mean squared error (normalised)"),
    )
    variants = []
    for label, column, ytitle in options:
        if column not in table.columns:
            continue
        traces = [_box(name, class_color(name, index), table.loc[table["clinical_class"] == name, column])
                  for index, name in enumerate(_classes(table["clinical_class"]))]
        variants.append(variant(traces, layout("", ytitle, showlegend=False), opt=label))
    return plot_panel(
        "forecast-class", "Forecast scores per clinical class", variants, opt_label="Quantity",
        how="The forecast analysis averages each score over the anchors and the segments of a "
            "recording.",
        shows="One dot per recording, grouped by clinical class. The box marks the median and the "
              "quartiles. A lower NLL and a lower squared error are better.",
        read="A class whose box is higher is harder to forecast. That can be a property of the "
             "signal, for example more variability, and not a fault of the model.",
        sources=["forecast/forecast_scores.csv"],
        pdfs=["forecast/forecast_scores_by_clinical_class.pdf", "forecast/forecast_scores_by_subgroup.pdf"],
    )


def calibration_pit(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the probability integral transform histogram of the full forecast.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("calibration/calibration_pit.csv")
    traces = [bars(table["bin_center"], table["density"], "observed", BLUE,
                   hovertemplate="%{y:.3f}<extra>density</extra>")]
    figure = layout("probability the forecast gave to a value below the observed one", "density",
                    shapes=[hline(1.0, color=VERMILLION, dash="dash")], showlegend=False, bargap=0.04)
    centre = float(table.loc[(table["bin_center"] > 0.4) & (table["bin_center"] < 0.6), "density"].mean())
    edge = float(table["density"].iloc[[0, -1]].mean())
    return plot_panel(
        "calibration-pit", "Is the predicted uncertainty right? (PIT)", [variant(traces, figure)],
        how="For every scored forecast value, the calibration analysis computes the probability "
            "that the forecast distribution gave to values below the observed one. It makes a "
            "histogram of those probabilities over every scored coefficient.",
        shows="The bars are the histogram, in 20 bins from 0 to 1. The dashed red line at 1 is "
              "what a perfectly calibrated forecast gives.",
        read="Flat bars at the red line mean that the predicted uncertainty matches the errors. "
             "A hump in the middle means that the predicted variance is too large. Bars that rise "
             "at both edges mean that it is too small. A slope means that the forecast is biased.",
        run=f"The density is {fmt(centre, 3)} in the middle and {fmt(edge, 3)} at the edges. "
            + ("The predicted variance is too large for most values." if centre > 1.05 and edge < 0.95
               else "The predicted variance is too small for most values." if centre < 0.95 and edge > 1.05
               else "The histogram is close to flat."),
        sources=["calibration/calibration_pit.csv"], pdfs=["calibration/pit_reliability.pdf"],
    )


def calibration_coverage(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the observed coverage of the 1, 2 and 3 sigma forecast intervals.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("calibration/calibration_coverage.csv")
    names = [f"{int(level)} sigma" for level in table["level_sigma"]]
    variants = [
        variant([bars(names, table["nominal_tail"], "expected", GRAY),
                 bars(names, table["observed_tail"], "observed", BLUE)],
                layout("", "share of values outside the interval", barmode="group",
                       yaxis={"type": "log", "tickformat": ".1%", "dtick": 1}), opt="Outside the interval"),
        variant([bars(names, table["nominal"], "expected", GRAY),
                 bars(names, table["observed"], "observed", BLUE)],
                layout("", "share of values inside the interval", barmode="group",
                       yaxis={"tickformat": ".1%", "range": [0.6, 1.0]}), opt="Inside the interval"),
    ]
    worst = table.loc[table["relative_tail_error"].abs().idxmax()]
    return plot_panel(
        "calibration-coverage", "Coverage of the forecast intervals", variants, opt_label="Count",
        how="For each scored forecast value, the calibration analysis checks whether the observed "
            "value lies within 1, 2 and 3 predicted standard deviations of the forecast mean.",
        shows="The gray bars are the shares a normal distribution gives: 68.3%, 95.4% and 99.7% "
              "inside. The blue bars are the observed shares. The first view shows the share "
              "outside the interval on a logarithmic scale, which makes the tails visible.",
        read="A blue bar below the gray bar in the first view means fewer errors outside the "
             "interval than expected, so the interval is too wide. A blue bar above the gray bar "
             "means more large errors than a normal distribution has.",
        run=f"The largest relative error is at {int(worst['level_sigma'])} sigma: "
            f"{pct(worst['observed_tail'], 2)} of the values are outside the interval, against "
            f"{pct(worst['nominal_tail'], 2)} expected.",
        sources=["calibration/calibration_coverage.csv"], pdfs=["calibration/pit_reliability.pdf"],
    )


def calibration_logvar(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the distribution of the predicted log-variance of the full forecast.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("calibration/calibration_logvar.csv")
    traces = [bars(table["bin_center"], table["fraction"], "share", PURPLE,
                   hovertemplate="%{y:.2%}<extra>share</extra>")]
    figure = layout("predicted log-variance", "share of scored coefficients", showlegend=False,
                    bargap=0.02, yaxis={"tickformat": ".0%"})
    floor = float(table["fraction"].iloc[0])
    return plot_panel(
        "calibration-logvar", "Predicted variance of the forecast", [variant(traces, figure)],
        how="The decoder predicts a log-variance for every forecast value. The model limits it to "
            "a fixed range. The calibration analysis makes a histogram over every scored "
            "coefficient.",
        shows="The x axis is the predicted log-variance and the y axis is the share of "
              "coefficients in each bin. The left edge of the axis is the lower limit.",
        read="A tall bar at the left edge means that the decoder wants a smaller variance than "
             "the limit allows. There the reported variance is the limit and not a fitted value, "
             "and the calibration plots above are affected.",
        run=f"{pct(floor)} of the scored coefficients are in the lowest bin, at the lower limit.",
        sources=["calibration/calibration_logvar.csv"], pdfs=["calibration/logvar_distribution.pdf"],
    )


# =================================================================================================
# Coupling and controls
# =================================================================================================
def coupling_summary(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the forecast gain under three estimators, and as a percentage.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("coupling/coupling_summary.csv").set_index("metric")
    nats = [("pred_gap_mc_nats", "Monte Carlo"), ("pred_gap_mean_nats", "mean forecast"),
            ("pred_gap_train_path_nats", "single draw (training path)")]
    percent = [("pred_gap_rmse_pct", "RMSE removed"), ("pred_gap_mse_pct", "squared error removed"),
               ("pred_gap_mc_likelihood_pct", "likelihood per coefficient, MC"),
               ("pred_gap_mean_likelihood_pct", "likelihood per coefficient, mean")]
    variants = []
    for label, keys, ytitle, colour in (("Nats per anchor", nats, "forecast gain per anchor (nats)", ORANGE),
                                        ("Percent", percent, "improvement (%)", TEAL)):
        present = [(key, name) for key, name in keys if key in table.index]
        if not present:
            continue
        rows = table.loc[[key for key, _ in present]]
        variants.append(variant(
            [bars([name for _, name in present], rows["mean"], "mean over recordings", colour,
                  lo=rows["ci_lo"], hi=rows["ci_hi"])],
            layout("", ytitle, showlegend=False, shapes=[hline(0.0)]), opt=label))
    headline = table.loc["pred_gap_mc_nats"]
    cells = dict(run.summary.get("likelihood_structure") or {}).get("scored_cells")
    spread = f"{int(cells):,} scored coefficients per anchor" if cells else "many scored coefficients per anchor"
    return plot_panel(
        "coupling-summary", "How much the UA improves the forecast", variants, opt_label="Unit",
        how="The coupling analysis takes the forecast gain per recording and averages over "
            "recordings. It reports three estimators. Monte Carlo averages the likelihood over "
            "several random draws of the latent variable and is the headline. The mean forecast "
            "decodes both branches at the latent mean. The single draw is the path that training "
            "uses. The error bar is a 95% bootstrap interval over recordings.",
        shows="One bar per estimator. The 'Percent' view gives the same gain as a percentage of "
              "the forecast error and of the likelihood per coefficient.",
        read="A bar whose interval is above zero means that the UA improves the forecast across "
             f"recordings. The percentages are small, because the gain is spread over {spread} "
             "and most of the forecast comes from the FHR history.",
        run=f"The Monte Carlo gain is {fmt(headline['mean'])} nats per anchor (95% interval "
            f"{fmt(headline['ci_lo'])} to {fmt(headline['ci_hi'])}). It is positive in "
            f"{pct(headline['positive_fraction'])} of the recordings.",
        sources=["coupling/coupling_summary.csv"],
        pdfs=["coupling/pred_gap_distribution.pdf", "coupling/pred_gap_percent.pdf"],
    )


def permutation_control(run: RunFiles) -> Dict[str, Any]:
    """Build the card: what the UA of another recording does to the forecast.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("perm_control/perm_control_penalties.csv").set_index("penalty")
    keys = [("source_margin", "own UA vs another recording's UA"),
            ("shuffle_penalty", "another recording's UA vs no UA")]
    present = [(key, name) for key, name in keys if key in table.index]
    rows = table.loc[[key for key, _ in present]]
    traces = [bars([name for _, name in present], rows["mean"], "mean over recordings",
                   [GREEN, VERMILLION][:len(present)], lo=rows["ci_lo"], hi=rows["ci_hi"])]
    figure = layout("", "forecast score difference (nats per anchor)", showlegend=False, shapes=[hline(0.0)])
    prior = table.loc["prior_shuffle_penalty", "mean"] if "prior_shuffle_penalty" in table.index else float("nan")
    prior_low = (
        table.loc["prior_shuffle_penalty", "ci_lo"] if "prior_shuffle_penalty" in table.index else float("nan")
    )
    if np.isfinite(prior_low) and prior_low > 0:
        prior_note = (f"Shuffling the prior latent between recordings costs {fmt(prior, 4)} nats, with "
                      f"an interval above zero, so the prior carries recording-specific FHR state.")
    elif np.isfinite(prior):
        prior_note = (f"Shuffling the prior latent between recordings costs {fmt(prior, 4)} nats, with "
                      f"an interval that includes zero.")
    else:
        prior_note = ""
    return plot_panel(
        "permutation-control", "Control: the UA of another recording", [variant(traces, figure)],
        how="The collection pass also runs the model with the UA of a different recording. The "
            "perm_control analysis compares the forecast scores per recording. The error bar is a "
            "95% bootstrap interval over recordings.",
        shows="The green bar is how much worse the forecast is with another recording's UA than "
              "with the recording's own UA. The red bar is how much worse it is with another "
              "recording's UA than with no UA.",
        read="A green bar above zero means that the model uses this recording's UA, and does not "
             "only react to the presence of any UA. A red bar above zero means that a wrong UA "
             "misleads the model, which is the expected result for a model that reads the UA.",
        run=f"The own UA beats another recording's UA by {fmt(table.loc['source_margin', 'mean'])} nats "
            f"per anchor, in {pct(table.loc['source_margin', 'positive_fraction'])} of the recordings. "
            + prior_note,
        sources=["perm_control/perm_control_penalties.csv", "perm_control/perm_control_summary.csv"],
        pdfs=["perm_control/perm_control_per_recording_by_clinical_class.pdf"],
    )


def latent_spectrum(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the KL carried by each latent dimension.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("latent/latent_spectrum.csv").sort_values("rank")
    heads = sorted(table["head"].dropna().unique())
    traces = []
    for index, head in enumerate(heads):
        rows = table[table["head"] == head]
        traces.append(bars(rows["rank"], rows["kl_nats"], f"head {int(head)}", SERIES[index % len(SERIES)],
                           customdata=rows["dimension"].tolist(),
                           hovertemplate="dimension %{customdata}: %{y:.3g} nats<extra>head "
                                         + str(int(head)) + "</extra>"))
    figure = layout("latent dimensions, sorted from the largest KL", "KL (nats per anchor)",
                    yaxis={"type": "log"}, barmode="stack", bargap=0.1)
    active = int(table["active"].astype(str).str.lower().eq("true").sum())
    return plot_panel(
        "latent-spectrum", "Which latent dimensions carry the UA information", [variant(traces, figure)],
        how="The KL is a sum over the latent dimensions. The latent analysis averages each "
            "dimension's part over anchors and recordings and sorts the dimensions. Each "
            "dimension belongs to one attention head, which gives the colour.",
        shows="One bar per latent dimension, from the largest KL on the left. The y axis is "
              "logarithmic.",
        read="A few tall bars mean that the UA information is concentrated in a few dimensions. "
             "That is normal for this kind of model. It becomes a problem only when one dimension "
             "carries everything, because the latent then has no room for different kinds of UA "
             "information.",
        run=f"{active} of {len(table)} dimensions are active. The largest carries "
            f"{pct(table['share'].iloc[0])} of the KL and the two largest together carry "
            f"{pct(table['share'].iloc[:2].sum())}.",
        sources=["latent/latent_spectrum.csv"], pdfs=["latent/kl_spectrum.pdf"],
    )


def latent_diagnostics(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the health checks of the latent variable and of the decoder variance.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("latent/latent_diagnostics.csv")
    rows = [[row["metric"], row["meaning"], row["mean"], row["q25"], row["q75"]]
            for _, row in table.iterrows()]
    return table_panel(
        "latent-diagnostics", "Latent and variance diagnostics",
        ["Metric", "Meaning", "Mean over recordings", "25th percentile", "75th percentile"], rows,
        how="The latent analysis averages each diagnostic within a recording and then over "
            "recordings.",
        shows="One row per diagnostic, with its meaning as the analysis states it.",
        read="The 'floor' and 'sat' fractions should be small. A large floor fraction of the "
             "decoder log-variance means that the decoder is at its lower variance limit for many "
             "coefficients.",
        sources=["latent/latent_diagnostics.csv"],
    )


def residual_summary(run: RunFiles) -> Dict[str, Any]:
    """Build the card: how far the UA moves the forecast and the latent mean.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("residual/residual_summary.csv")
    rows = [[row["metric"], row["meaning"], row["rms_normalised"], row["rms_lo_normalised"],
             row["rms_hi_normalised"], row["unit"]] for _, row in table.iterrows()]
    return table_panel(
        "residual-summary", "How far the UA moves the forecast",
        ["Metric", "Meaning", "RMS", "Interval low", "Interval high", "Unit"], rows,
        how="The residual analysis takes the root mean square of the difference between the full "
            "and the base forecast, and between the posterior and the prior latent mean.",
        shows="One row per difference, with a bootstrap interval over recordings.",
        read="These values say how much the UA moves the model's output. They do not say whether "
             "the move is an improvement. Read them together with the forecast gain.",
        sources=["residual/residual_summary.csv"],
    )


def events_conditioned(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the gain and the KL at contraction anchors against matched control anchors.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("events/conditioned_coupling.csv")
    variants = []
    for metric in pd.unique(table["metric"]):
        rows = table[table["metric"] == metric]
        cohorts = ["pooled", *_classes(rows.loc[rows["cohort"] != "pooled", "cohort"])]
        rows = rows.set_index("cohort").reindex(cohorts)
        traces = [
            bars(cohorts, rows["event_mean"], "anchors near a contraction", GREEN),
            bars(cohorts, rows["control_mean"], "matched control anchors", GRAY),
        ]
        title = METRIC_TITLES.get(metric, metric)
        variants.append(variant(traces, layout("", title, barmode="group"),
                                opt=METRIC_NAMES.get(metric, metric)))
    pooled = table[(table["cohort"] == "pooled")].set_index("metric")
    gain = pooled.loc["pred_gap_mc_nats"] if "pred_gap_mc_nats" in pooled.index else None
    return plot_panel(
        "events-conditioned", "Contractions against matched controls", variants, opt_label="Quantity",
        how="The events analysis marks the anchors near a contraction. In each recording it draws "
            "the same number of control anchors away from contractions. It averages the gain and "
            "the KL over each set per recording, then over recordings.",
        shows="The green bars are the anchors near a contraction and the gray bars are the "
              "controls. There is one group for all recordings and one per clinical class.",
        read="Green bars above the gray bars mean that the UA matters more around contractions, "
             "which is what the physiology suggests. Equal bars would mean that the model uses "
             "the UA in the same way at any time.",
        run=None if gain is None else
        f"Near contractions the forecast gain is {fmt(gain['event_mean'])} nats per anchor, against "
        f"{fmt(gain['control_mean'])} nats at the control anchors.",
        sources=["events/conditioned_coupling.csv"], pdfs=["events/conditioned_coupling.pdf"],
    )


# =================================================================================================
# Labour time and cohorts
# =================================================================================================
def _trajectory(
    run: RunFiles, identifier: str, title: str, filename: str, *, xtitle: str, reverse: bool,
    how: str, shows: str, read: str, pdfs: Sequence[str],
) -> Dict[str, Any]:
    """Build a card of the per-class trajectory of the gain and the KL against a labour clock.

    Args:
        run: The run's files.
        identifier: A stable name, unique on the page.
        title: The card title.
        filename: The trajectory table to draw.
        xtitle: The title of the clock axis.
        reverse: Reverse the x axis, for a clock that counts down to delivery.
        how: How the plot was made.
        shows: What the axes and the traces are.
        read: How to read it.
        pdfs: The PDF figures that show the same result.

    Returns:
        The card, with one variant per metric and statistic.

    Raises:
        MissingInput: If the table is absent.
    """
    table = run.csv(filename)
    rows = table[table["group_column"] == "clinical_class"]
    variants = []
    for metric in pd.unique(rows["metric"]):
        for statistic, label in (("median", "median"), ("mean", "mean")):
            traces: List[Any] = []
            for index, name in enumerate(_classes(rows["group"])):
                group = rows[(rows["metric"] == metric) & (rows["group"] == name)].sort_values("bin_center_h")
                colour = class_color(name, index)
                if statistic == "median":
                    traces.append(band(group["bin_center_h"], group["q25"], group["q75"], colour))
                traces.append(line(group["bin_center_h"], group[statistic], name, colour, markers=True,
                                   customdata=group["n_recordings"].tolist(),
                                   hover="%{y:.4g} (%{customdata} recordings)<extra>" + name + "</extra>"))
            figure = layout(xtitle, METRIC_TITLES.get(metric, metric),
                            xaxis={"autorange": "reversed"} if reverse else {}, shapes=[hline(0.0)])
            variants.append(variant(traces, figure, opt=f"{METRIC_NAMES.get(metric, metric)}, {label}"))
    return plot_panel(identifier, title, variants, opt_label="Quantity", how=how, shows=shows, read=read,
                      sources=[filename], pdfs=pdfs)


def time_to_delivery_trajectory(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the gain and the KL of each class against time before delivery.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    return _trajectory(
        run, "time-to-delivery", "Gain and KL over labour, per class",
        "time_to_delivery/time_to_delivery_trajectory.csv", xtitle="hours before delivery", reverse=True,
        how="The time_to_delivery analysis groups the segments by the half hour of labour they "
            "start in. In each half hour it averages a recording's segments, so each recording "
            "gives one value. It then takes the median, the quartiles and the mean over the "
            "recordings of each class.",
        shows="One line per clinical class. In the median view, the band is the range between the "
              "25th and the 75th percentile. Delivery is at the right edge. Hover a point to see "
              "how many recordings it holds.",
        read="This is the standard reading over all anchors, at the level of segments. Lines "
             "that separate near delivery would be a class signal. A wide band means that the "
             "recordings differ a lot, so a difference between the class lines may not be real. "
             "The next plot tests it.",
        pdfs=["time_to_delivery/time_to_delivery_trajectory_pred_gap.pdf",
              "time_to_delivery/time_to_delivery_trajectory_kl.pdf",
              "time_to_delivery/time_to_delivery_windows_pred_gap.pdf",
              "time_to_delivery/time_to_delivery_windows_kl.pdf"],
    )


def time_to_delivery_significance(run: RunFiles) -> Dict[str, Any]:
    """Build the card: whether the classes differ in each half hour of labour.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("time_to_delivery/time_to_delivery_significance.csv")
    traces = []
    for index, (metric, rows) in enumerate(table.groupby("metric_column")):
        rows = rows.sort_values("bin_center_h")
        name = METRIC_NAMES.get(metric, metric)
        traces.append(line(rows["bin_center_h"], rows["p_value"], f"{name}, raw p-value",
                           SERIES[index % len(SERIES)], dash="dot", width=1.8))
        traces.append(line(rows["bin_center_h"], rows["p_holm"], f"{name}, corrected p-value",
                           SERIES[index % len(SERIES)], markers=True))
    alpha = float(table["alpha"].iloc[0]) if "alpha" in table.columns and len(table) else 0.05
    figure = layout("hours before delivery", "p-value of the class difference",
                    xaxis={"autorange": "reversed"}, yaxis={"type": "log"},
                    shapes=[hline(alpha, color=VERMILLION, dash="dash")])
    significant = int(table["significant"].astype(str).str.lower().eq("true").sum())
    return plot_panel(
        "time-to-delivery-significance", "Do the classes differ over labour?", [variant(traces, figure)],
        how="In each half hour, the time_to_delivery analysis tests whether the per-recording "
            "values of the classes come from the same distribution (Kruskal-Wallis test). It then "
            "corrects the p-values for the number of half hours tested (Holm correction).",
        shows="The solid lines are the corrected p-values and the dotted lines are the raw ones. "
              "The dashed red line is the significance level.",
        read="A solid point below the red line is a half hour in which the classes differ. Points "
             "above the line mean that the data shows no class difference there.",
        run=f"{significant} of {len(table)} tests are significant after correction.",
        sources=["time_to_delivery/time_to_delivery_significance.csv",
                 "time_to_delivery/time_to_delivery_pairwise.csv"],
        pdfs=["time_to_delivery/time_to_delivery_windows_pred_gap.pdf"],
    )


def second_stage_trajectory(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the gain and the KL of each class against the onset of the second stage.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    return _trajectory(
        run, "second-stage", "Gain and KL around the second stage of labour",
        "second_stage/second_stage_trajectory.csv", xtitle="hours from the second-stage onset",
        reverse=False,
        how="The second_stage analysis places each segment on a clock whose zero is the onset of "
            "the second stage of labour (the pushing stage). It uses only the recordings that "
            "have a recorded onset. The averaging is the same as for the time before delivery.",
        shows="One line per clinical class. Negative hours are before the second-stage onset and "
              "positive hours are after it.",
        read="A step at zero would mean that the model uses the UA differently once the second "
             "stage starts. Fewer recordings reach the positive side, so the lines are noisier "
             "there.",
        pdfs=["second_stage/second_stage_trajectory_pred_gap.pdf",
              "second_stage/second_stage_trajectory_kl.pdf"],
    )


def within_segment_profile(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the gain and the KL against the position in the segment.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("trajectory/within_segment_profile.csv")
    variants = []
    for metric in pd.unique(table["metric"]):
        rows = table[table["metric"] == metric].sort_values("seconds_in_segment")
        colour = VERMILLION if "kld" in metric else ORANGE
        traces = [band(rows["seconds_in_segment"], rows["q25"], rows["q75"], colour),
                  line(rows["seconds_in_segment"], rows["median"], "median over recordings", colour),
                  line(rows["seconds_in_segment"], rows["mean"], "mean over recordings", colour, dash="dot", width=1.8)]
        variants.append(variant(traces, layout("position in the segment (s)", METRIC_TITLES.get(metric, metric)),
                                opt=METRIC_NAMES.get(metric, metric)))
    return plot_panel(
        "within-segment", "Readouts along the segment", variants, opt_label="Quantity",
        how="The trajectory analysis averages each per-anchor readout at every anchor position "
            "within a recording, then takes the median, the quartiles and the mean over "
            "recordings.",
        shows="The x axis is the time from the start of the segment. The solid line is the median "
              "and the band is the range between the quartiles. The dotted line is the mean.",
        read="A flat profile means that the readout does not depend on where the anchor sits in "
             "the segment. A mean far above the median means that a few recordings with "
             "large values pull the mean up.",
        sources=["trajectory/within_segment_profile.csv"],
        pdfs=["trajectory/trajectory_profile_pred_gap.pdf", "trajectory/trajectory_profile_kl.pdf"],
    )


def class_distributions(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the per-recording distribution of each readout per clinical class.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("distributions/distribution_summary.csv")
    rows = table[(table["group_column"] == "clinical_class") & (table["level"] == "recording")]
    if rows.empty:
        raise MissingInput("distributions/distribution_summary.csv (clinical_class rows)", "distributions")
    variants = []
    for metric in pd.unique(rows["metric"]):
        selected = rows[rows["metric"] == metric].set_index("group")
        names = _classes(selected.index)
        selected = selected.reindex(names)
        trace = {
            "type": "box", "x": names, "q1": selected["q25"].tolist(), "median": selected["q50"].tolist(),
            "q3": selected["q75"].tolist(), "lowerfence": selected["min"].tolist(),
            "upperfence": selected["max"].tolist(), "mean": selected["mean"].tolist(),
            "marker": {"color": BLUE}, "line": {"width": 2}, "name": metric,
        }
        unit = str(selected["unit"].iloc[0])
        variants.append(variant([trace], layout("", f"{metric} ({unit})", showlegend=False),
                                opt=str(metric).replace("_", " ")))
    return plot_panel(
        "class-distributions", "Readouts per clinical class", variants, opt_label="Readout",
        how="The distributions analysis averages each readout over the segments of a recording "
            "and summarises the recordings of each class by their quartiles, their range and "
            "their mean.",
        shows="One box per clinical class. The box is the range between the quartiles, the line "
              "in it is the median, the dashed line is the mean, and the whiskers are the smallest "
              "and the largest recording.",
        read="Boxes that overlap mean that the readout does not separate the classes at the "
             "level of whole recordings.",
        sources=["distributions/distribution_summary.csv", "distributions/per_segment_metrics.csv"],
        pdfs=["distributions/class_histograms.pdf", "distributions/subgroup_histograms.pdf"],
    )


# =================================================================================================
# Analyses that intervene on the source, or select anchors
# =================================================================================================
def source_null_summary(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the KL with the real UA against the KL with the UA values removed.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("source_null/source_null_summary.csv").set_index("metric")
    keys = [("source_conditioned_kl_raw", "KL with the real UA", VERMILLION),
            ("kld_source_null", "KL with the UA values removed", GRAY),
            ("coupling_minus_clock", "difference", GREEN)]
    present = [(key, name, colour) for key, name, colour in keys if key in table.index]
    rows = table.loc[[key for key, _, _ in present]]
    traces = [bars([name for _, name, _ in present], rows["mean"], "mean over recordings",
                   [colour for _, _, colour in present], lo=rows["ci_lo"], hi=rows["ci_hi"])]
    difference = table.loc["coupling_minus_clock"]
    total = float(table.loc["source_conditioned_kl_raw", "mean"])
    return plot_panel(
        "source-null", "How much of the KL comes from the UA values",
        [variant(traces, layout("", "KL per anchor (nats)", showlegend=False))],
        how="The collection pass runs the model a second time with the UA values set to zero. It "
            "keeps the flags that say where the UA is valid. The KL of that run is what the model "
            "produces from the timing and the availability of the UA alone. The source_null "
            "analysis subtracts it from the KL of the real run, per recording. The error bar is a "
            "95% bootstrap interval over recordings.",
        shows="The red bar is the KL with the real UA. The gray bar is the KL with the UA values "
              "removed. The green bar is the difference.",
        read="Only the green bar is information from the UA values. A gray bar almost as tall as "
             "the red one means that most of the KL is a fixed pattern that the model produces "
             "whatever the UA does. The KL alone therefore over-states the coupling.",
        run=f"The KL is {fmt(total)} nats per anchor. {fmt(table.loc['kld_source_null', 'mean'])} nats "
            f"remain without the UA values, so {fmt(difference['mean'])} nats "
            f"({pct(float(difference['mean']) / total)}) come from the UA values (95% interval "
            f"{fmt(difference['ci_lo'])} to {fmt(difference['ci_hi'])}).",
        sources=["source_null/source_null_summary.csv"], pdfs=["source_null/source_null_difference.pdf"],
    )


def source_null_lag_profile(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the KL per lag with the real UA and with the UA values removed.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("source_null/source_null_lag_profile.csv")
    seconds = table["compensated_seconds"]
    traces = [
        line(seconds, table["kl_nats"], "real UA", VERMILLION),
        line(seconds, table["kl_null_nats"], "UA values removed", GRAY, dash="dash"),
        line(seconds, table["clock_excess_nats"], "difference", GREEN, markers=True),
    ]
    variants = [
        variant(traces, layout("lag (s, stored-coefficient time)", "KL attributed to the lag (nats per anchor)",
                               shapes=[hline(0.0)]), opt="Linear scale"),
        variant(traces[:2] + [line(seconds, table["clock_excess_positive_nats"], "difference, positive part",
                                    GREEN, markers=True)],
                layout("lag (s, stored-coefficient time)", "KL attributed to the lag (nats per anchor)",
                       yaxis={"type": "log"}), opt="Logarithmic scale"),
    ]
    excess = table["clock_excess_nats"]
    peak = table.loc[excess.idxmax()]
    positive = float(table["clock_excess_positive_nats"].sum())
    return plot_panel(
        "source-null-lag", "At which lags the UA values matter", variants, opt_label="Scale",
        how="The source_null analysis takes the per-lag split of the KL from the real run and "
            "from the run with the UA values removed, and subtracts the second from the first at "
            "each lag.",
        shows="The red line is the real run and the dashed gray line is the run without UA "
              "values. The green line is the difference.",
        read="The green line shows the lags at which the UA values add information. It is the "
             "lag profile to trust, because the fixed pattern is removed. A green line below zero "
             "at a lag means that the run without UA values puts more KL there than the real run. "
             + LAG_NOTE,
        run=f"The difference is largest at a lag of {fmt(peak['compensated_seconds'], 3)} s, with "
            f"{fmt(peak['clock_excess_nats'])} nats, which is "
            f"{pct(float(peak['clock_excess_positive_nats']) / positive if positive > 0 else float('nan'))} "
            f"of its positive total.",
        sources=["source_null/source_null_lag_profile.csv"], pdfs=["source_null/source_null_lag_profile.pdf"],
    )


def time_shift_control(run: RunFiles) -> Dict[str, Any]:
    """Build the card: what happens when the UA is taken from another time of the same recording.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("time_shift/time_shift_summary.csv").set_index("quantity")
    views = (
        ("Forecast gain", [("gap_aligned", "UA from the same time", GREEN),
                           ("gap_shifted", "UA from another time", VERMILLION)],
         "forecast gain per anchor (nats)"),
        ("KL", [("kld_aligned", "UA from the same time", GREEN),
                ("kld_shifted", "UA from another time", VERMILLION),
                ("kld_source_null", "UA values removed", GRAY)], "KL per step (nats)"),
    )
    variants = []
    for label, keys, ytitle in views:
        present = [(key, name, colour) for key, name, colour in keys if key in table.index]
        rows = table.loc[[key for key, _, _ in present]]
        variants.append(variant(
            [bars([name for _, name, _ in present], rows["value"], label,
                  [colour for _, _, colour in present], lo=rows["ci_lo"], hi=rows["ci_hi"])],
            layout("", ytitle, showlegend=False, shapes=[hline(0.0)]), opt=label))
    count = int(table["n"].iloc[0])
    return plot_panel(
        "time-shift", "Control: the UA from another time of the same recording", variants,
        opt_label="Quantity",
        how="The time_shift analysis runs the model on a segment's FHR with the UA of a different "
            "segment of the same recording, at least several minutes away. The UA then has the "
            "right statistics for the patient but the wrong timing. The analysis uses a capped "
            "draw of segments, and the error bar is a 95% bootstrap interval over recordings.",
        shows="The green bar is the normal run and the red bar is the run with the UA from "
              "another time. The selector changes between the forecast gain and the KL.",
        read="If the model uses the timing of the UA, the forecast gain falls or turns negative "
             "when the UA comes from another time. If the gain stayed the same, the model would "
             "use only slow properties of the UA, such as its average level.",
        run=f"Over {count} recordings, the forecast gain is {fmt(table.loc['gap_aligned', 'value'])} nats "
            f"with the UA from the same time and {fmt(table.loc['gap_shifted', 'value'])} nats with the UA "
            f"from another time.",
        sources=["time_shift/time_shift_summary.csv", "time_shift/time_shift_per_segment.csv"],
        pdfs=["time_shift/time_shift_control.pdf"],
    )


def occlusion_bands(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the forecast cost of removing the UA values in one band of lags.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    summary = run.csv("occlusion/occlusion_summary.csv")
    horizon = run.csv("occlusion/occlusion_per_horizon.csv")
    geometry = Geometry(run)
    names = [f"{row['band']} ({geometry.band_seconds(row['lag_lo'], row['lag_hi'])})"
             for _, row in summary.iterrows()]
    colours = [SERIES[index % len(SERIES)] for index in range(len(summary))]
    total = variant(
        [bars(names, summary["delta_total_recording_mean_nats"], "mean over recordings", colours,
              lo=summary["delta_total_ci_lo"], hi=summary["delta_total_ci_hi"])],
        layout("band of lags removed", "forecast score lost (nats per anchor)", showlegend=False,
               shapes=[hline(0.0)]), opt="Total cost per band")
    traces = []
    for index, (band_name, rows) in enumerate(horizon.groupby("band", sort=False)):
        traces.append(line(SECONDS_PER_STEP * (rows["horizon_step"] + 1), rows["delta_nats"], str(band_name),
                           colours[index % len(colours)]))
    per_step = variant(traces, layout("seconds ahead of the anchor", "forecast score lost at this step (nats)",
                                      shapes=[hline(0.0)]), opt="Cost along the forecast horizon")
    best = summary.loc[summary["delta_total_recording_mean_nats"].idxmax()]
    return plot_panel(
        "occlusion-bands", "Which lags the forecast needs (occlusion test)", [total, per_step],
        opt_label="View",
        how="The occlusion analysis removes the UA values inside one band of lags, leaves the "
            "rest of the UA in place, and runs the model again. The cost is the full forecast's "
            "score with the band removed minus its score with the band present. It uses a capped "
            "draw of segments. The error bar is a 95% interval over recordings.",
        shows="The first view has one bar per band of lags. The second view shows where along "
              "the forecast horizon the cost falls, with one line per band.",
        read="This is an intervention, so it shows which lags the forecast needs. The attention "
             "and KL profiles show only where the model looks. A band with a high cost is one the "
             "forecast depends on. A cost near zero means that the model can do without those "
             "lags. " + LAG_NOTE,
        run=f"The '{best['band']}' band (lags {geometry.band_seconds(best['lag_lo'], best['lag_hi'])}) costs "
            f"the most: {fmt(best['delta_total_recording_mean_nats'])} nats per anchor (95% interval "
            f"{fmt(best['delta_total_ci_lo'])} to {fmt(best['delta_total_ci_hi'])}).",
        sources=["occlusion/occlusion_summary.csv", "occlusion/occlusion_per_horizon.csv"],
        pdfs=["occlusion/occlusion_horizon_delta.pdf", "occlusion/occlusion_clock_delta.pdf"],
    )


#: The lag features of ``lag_clocks_trajectory.csv`` the page offers: metric key, name, axis title.
_LAG_FEATURES = (
    ("lag_centroid_kl_s", "mean lag of the KL", "mean lag of the KL (s)"),
    ("lag_centroid_attn_s", "mean lag of the attention", "mean lag of the attention (s)"),
    ("lag_spread_kl_s", "spread of the KL over lags", "spread of the KL over lags (s)"),
    ("lag_near_mass_kl", "share of the KL at near lags", "share of the KL at near lags"),
    ("lag_far_mass_kl", "share of the KL at far lags", "share of the KL at far lags"),
    ("lag_entropy_attn_nats", "entropy of the attention", "entropy of the attention (nats)"),
)


def lag_features_over_labour(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the lag features of each class against the two labour clocks, over all anchors.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("lag_clocks/lag_clocks_trajectory.csv")
    rows = table[table["group_column"] == "clinical_class"]
    variants = []
    for clock, clock_name, xtitle, reverse in (
        ("time_to_delivery", "time before delivery", "hours before delivery", True),
        ("second_stage", "second stage", "hours from the second-stage onset", False),
    ):
        for metric, name, ytitle in _LAG_FEATURES:
            selected = rows[(rows["clock"] == clock) & (rows["metric"] == metric)]
            if selected.empty:
                continue
            traces: List[Any] = []
            for index, group in enumerate(_classes(selected["group"])):
                part = selected[selected["group"] == group].sort_values("bin_center_h")
                colour = class_color(group, index)
                traces.append(band(part["bin_center_h"], part["q25"], part["q75"], colour))
                traces.append(line(part["bin_center_h"], part["median"], group, colour, markers=True,
                                   customdata=part["n_recordings"].tolist(),
                                   hover="%{y:.4g} (%{customdata} recordings)<extra>" + group + "</extra>"))
            figure = layout(xtitle, ytitle, xaxis={"autorange": "reversed"} if reverse else {})
            variants.append(variant(traces, figure, opt=f"{name}, {clock_name}"))
    if not variants:
        raise MissingInput("lag_clocks/lag_clocks_trajectory.csv (clinical_class rows)", "lag_clocks")
    return plot_panel(
        "lag-features-labour", "Lag features over labour, per class, all anchors", variants,
        opt_label="Feature and clock",
        how="The lag_clocks analysis reduces each segment's lag profile to a few numbers, such as "
            "its mean lag and its spread. It averages them per recording in each half hour of "
            "labour, then takes the median and the quartiles over the recordings of each class.",
        shows="One line per clinical class, with a band between the quartiles. The selector "
              "changes the feature and the clock.",
        read="A flat line means that the lag structure does not change over labour. Lines that "
             "separate between classes would be a class difference in how far back the model "
             "looks. " + LAG_NOTE,
        sources=["lag_clocks/lag_clocks_trajectory.csv", "lag_clocks/lag_clocks_significance.csv"],
        pdfs=["lag_clocks/lag_time_to_delivery.pdf", "lag_clocks/lag_time_to_delivery_features.pdf",
              "lag_clocks/lag_second_stage.pdf", "lag_clocks/lag_second_stage_features.pdf"],
    )


def high_kl_gain(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the forecast gain of the anchors in each tenth of the KL distribution.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("lag_high_kl/lag_high_kl_gain_by_kl_quantile.csv")
    table = table[table["n_anchors"] > 0]
    variants = []
    for label, column, ytitle, tick in (
        ("Mean forecast gain", "mean_gain_nats", "forecast gain per anchor (nats)", None),
        ("Share of anchors with a positive gain", "positive_anchor_share", "share of anchors", ".0%"),
    ):
        traces = []
        for index, group in enumerate(["all", *_classes(table.loc[table["group"] != "all", "group"])]):
            rows = table[table["group"] == group]
            colour = "#111111" if group == "all" else class_color(group, index)
            traces.append(line(rows["kl_bin"], rows[column], "all classes" if group == "all" else group,
                               colour, markers=True, width=3.2 if group == "all" else 2.2,
                               customdata=rows["kl_hi_nats"].round(3).tolist(),
                               hover="%{y:.4g} (KL up to %{customdata} nats)<extra>" + group + "</extra>"))
        figure = layout("tenth of the KL distribution (1 is the lowest KL)", ytitle,
                        yaxis={"tickformat": tick} if tick else {}, shapes=[] if tick else [hline(0.0)])
        variants.append(variant(traces, figure, opt=label))
    pooled = table[table["group"] == "all"]
    return plot_panel(
        "high-kl-gain", "Is a large KL a useful KL?", variants, opt_label="Quantity",
        how="The lag_high_kl analysis sorts all scored anchors by their KL and cuts them into ten "
            "groups of equal size. For each group it takes the mean forecast gain per recording, "
            "then over recordings.",
        shows="The x axis is the KL group, from the lowest tenth to the highest. The y axis is "
              "the forecast gain of the anchors in the group. Hover a point to see the upper KL "
              "limit of the group.",
        read="A line that rises means that the anchors where the UA changes the belief most are "
             "also the anchors where the forecast improves most. That is the sign of a useful "
             "coupling. A flat line would mean that the KL does not predict the gain.",
        run=f"The lowest KL group gains {fmt(pooled['mean_gain_nats'].iloc[0], 2)} nats per anchor and the "
            f"highest gains {fmt(pooled['mean_gain_nats'].iloc[-1])} nats.",
        sources=["lag_high_kl/lag_high_kl_gain_by_kl_quantile.csv", "lag_high_kl/lag_high_kl_thresholds.csv"],
        pdfs=["lag_high_kl/lag_high_kl_usefulness.pdf", "lag_high_kl/lag_high_kl_selection.pdf"],
    )


def high_kl_lag_profiles(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the lag profile of the high-KL anchors against the rest.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("lag_high_kl/lag_high_kl_selection.csv")
    seconds = table["compensated_seconds"]
    series = [("pooled_high_kl_nats", "high KL (top 30%)", VERMILLION),
              ("pooled_top_kl_nats", "highest KL (top 10%)", PURPLE),
              ("pooled_gain_kl_nats", "high forecast gain (top 30%)", ORANGE),
              ("pooled_rest_kl_nats", "the other anchors", GRAY)]
    variants = []
    for label, normalise in (("Share of each group's KL", True), ("Nats per anchor", False)):
        traces = []
        for column, name, colour in series:
            if column not in table.columns:
                continue
            values = table[column].to_numpy(dtype=float)
            if normalise:
                total = np.nansum(np.clip(values, 0, None))
                values = np.clip(values, 0, None) / total if total > 0 else values
            traces.append(line(seconds, values, name, colour))
        figure = layout("lag (s, stored-coefficient time)",
                        "share of the group's KL" if normalise else "KL attributed to the lag (nats per anchor)",
                        yaxis={"type": "log", "tickformat": ".1%", "dtick": 1} if normalise else {"type": "log"})
        variants.append(variant(traces, figure, opt=label))
    hot = table[table["hot"].astype(str).str.lower() == "true"] if "hot" in table.columns else table.iloc[:0]
    return plot_panel(
        "high-kl-lags", "Lag profile of the anchors that carry the coupling", variants, opt_label="Unit",
        how="The lag_high_kl analysis selects anchors by their KL: the top 30%, the top 10%, and "
            "the remaining 70%. It also selects the top 30% by forecast gain. For each group it "
            "averages the per-lag split of the KL.",
        shows="One line per group of anchors. The first view divides each line by its own total, "
              "so the shapes can be compared. The y axis is logarithmic.",
        read="This selection is close to the gate populations, because the gate-on anchors carry "
             "almost all the KL. If the high-KL line has the same shape as the line of the other "
             "anchors, the coupling does not use special lags when it is strong. " + LAG_NOTE,
        run=(f"The analysis marks {len(hot)} lags as carrying the coupling, up to a lag of "
             f"{fmt(hot['compensated_seconds'].max(), 3)} s." if len(hot) else None),
        sources=["lag_high_kl/lag_high_kl_selection.csv"],
        pdfs=["lag_high_kl/lag_high_kl_selection.pdf", "lag_high_kl/lag_high_kl_time_to_delivery.pdf",
              "lag_high_kl/lag_high_kl_time_to_delivery_histogram.pdf"],
    )


def more_lag_figures(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the lag figures that exist only as PDF pages.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    groups = (
        ("lag_kld_scaled", "The lag statistics after the per-recording KL scale is divided out, on "
                           "the lags that carry the coupling, against both labour clocks."),
        ("lag_high_kl", "The lag structure of the high-KL anchors against both labour clocks, with "
                        "the lag histograms per window and their drift over labour."),
        ("lag_clocks", "The lag profile and its features against both labour clocks, with the "
                       "per-window class tests."),
        ("attention", "The mean attention profile and the attention heatmaps of single segments."),
    )
    rows = []
    for folder, text in groups:
        files = run.glob(f"{folder}/*.pdf")
        if files:
            links = ", ".join(f"<a href='{name}' target='_blank' rel='noopener'>{name.split('/')[-1]}</a>"
                              for name in files)
            rows.append([folder, text, links])
    if not rows:
        raise MissingInput("lag_clocks/*.pdf", "lag_clocks, lag_kld_scaled or lag_high_kl")
    return table_panel(
        "more-lag-figures", "More lag figures (PDF)", ["Analysis", "What its pages show", "Files"], rows,
        how="Three lag analyses write many pages whose tables are too large to redraw here. This "
            "card links their PDF pages.",
        read="Open a page from the link. The evaluation's figure guide explains each panel.",
    )


# =================================================================================================
# More of the forecast
# =================================================================================================
#: How the frequency bands of the FHR features read on the page, in physiological order.
_BAND_NAMES = {
    "slow_baseline": "baseline (below 0.008 Hz)", "deceleration": "decelerations (0.008 to 0.04 Hz)",
    "variability": "variability (0.04 to 0.25 Hz)", "beat_to_beat": "fast (above 0.25 Hz)",
    "unknown": "no band",
}


def spectral_skill(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the forecast gain per frequency band of the FHR features.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("spectral_skill/spectral_skill_bands.csv")
    names = [f"{_BAND_NAMES.get(band_name, band_name)}, {int(count)} ch."
             for band_name, count in zip(table["band"], table["n_channels"])]
    variants = [
        variant([bars(names, table["pred_gap_nats"], "forecast gain", ORANGE,
                      lo=table["pred_gap_ci_lo"], hi=table["pred_gap_ci_hi"])],
                layout("", "forecast gain per anchor (nats)", showlegend=False, shapes=[hline(0.0)]),
                opt="Forecast gain (nats)"),
        variant([bars(names, table["mse_skill"], "squared error removed", TEAL,
                      lo=table["mse_skill_ci_lo"], hi=table["mse_skill_ci_hi"])],
                layout("", "share of the base forecast's squared error removed", showlegend=False,
                       yaxis={"tickformat": ".2%"}, shapes=[hline(0.0)]), opt="Squared error removed"),
    ]
    best = table.loc[table["pred_gap_nats"].idxmax()]
    worst = table.loc[table["pred_gap_nats"].idxmin()]
    return plot_panel(
        "spectral-skill", "Which part of the FHR the UA helps to forecast", variants, opt_label="Quantity",
        how="Each FHR feature channel belongs to a frequency band. The spectral_skill analysis "
            "sums the forecast gain over the channels of each band per recording and averages "
            "over recordings. The error bar is a 95% bootstrap interval over recordings. These "
            "values come from one random draw of the latent variable, as in training.",
        shows="One bar per frequency band, with the number of channels in the band. The "
              "deceleration band holds the slow FHR changes that follow contractions.",
        read="A positive bar means that the UA improves the forecast of that band. The "
             "deceleration band is the clinically relevant one: a gain there means that the model "
             "uses the UA to forecast decelerations. Bands with more channels can show larger "
             "sums, so also read the 'Squared error removed' view.",
        run=f"The largest gain is in the band '{_BAND_NAMES.get(best['band'], best['band'])}' with "
            f"{fmt(best['pred_gap_nats'])} nats per anchor. The smallest is in "
            f"'{_BAND_NAMES.get(worst['band'], worst['band'])}' with {fmt(worst['pred_gap_nats'])} nats.",
        sources=["spectral_skill/spectral_skill_bands.csv", "spectral_skill/spectral_skill_channels.csv"],
        pdfs=["spectral_skill/spectral_skill_bands.pdf",
              "spectral_skill/spectral_skill_per_recording_by_clinical_class.pdf"],
    )


def spectral_channels(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the forecast gain of every FHR feature channel.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("spectral_skill/spectral_skill_channels.csv")
    traces = []
    for index, (band_name, rows) in enumerate(table.groupby("band", sort=False)):
        traces.append(bars(rows["kept_channel"], rows["pred_gap"], _BAND_NAMES.get(band_name, band_name),
                           SERIES[index % len(SERIES)],
                           customdata=rows[["kind", "freq_hz_primary"]].astype(str).values.tolist(),
                           hovertemplate="channel %{x} (%{customdata[0]}, %{customdata[1]} Hz): %{y:.3g} nats<extra></extra>"))
    figure = layout("FHR feature channel", "forecast gain per anchor (nats)", shapes=[hline(0.0)], bargap=0.1)
    positive = int((table["pred_gap"] > 0).sum())
    return plot_panel(
        "spectral-channels", "Forecast gain per feature channel", [variant(traces, figure)],
        how="The spectral_skill analysis reports the forecast gain of each of the kept FHR "
            "feature channels, averaged over anchors and recordings. The colour is the frequency "
            "band of the channel.",
        shows="One bar per channel. The first channels are the scattering features and the later "
              "ones are the phase features. Hover a bar for the channel type and its frequency.",
        read="Bars above zero are channels that the UA helps to forecast. A group of negative "
             "bars shows a block of channels where the full forecast is worse than the base "
             "forecast, which can come from the predicted variance and not from the forecast mean.",
        run=f"{positive} of {len(table)} channels have a positive gain.",
        sources=["spectral_skill/spectral_skill_channels.csv"], pdfs=["spectral_skill/spectral_skill_bands.pdf"],
    )


def warmup_tertiles(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the forecast gain split by how long each feature channel needs to settle.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("warmup/warmup_summary.csv").set_index("metric")
    keys = [("pred_gap_warm_lo", "fast channels"), ("pred_gap_warm_mid", "middle channels"),
            ("pred_gap_warm_hi", "slow channels")]
    present = [(key, name) for key, name in keys if key in table.index]
    rows = table.loc[[key for key, _ in present]]
    traces = [bars([name for _, name in present], rows["mean"], "mean over recordings",
                   [GREEN, ORANGE, VERMILLION][:len(present)], lo=rows["ci_lo"], hi=rows["ci_hi"])]
    return plot_panel(
        "warmup-tertiles", "Forecast gain by filter settling time",
        [variant(traces, layout("", "forecast gain per anchor (nats)", showlegend=False, shapes=[hline(0.0)]))],
        how="Each FHR feature channel comes from a one-sided filter that needs some time to "
            "settle. The warmup analysis sorts the channels by that time and cuts them into three "
            "groups. It sums the single-draw forecast gain over each group.",
        shows="One bar per group of channels: the fastest third, the middle third and the slowest "
              "third. The three bars add up to the single-draw forecast gain.",
        read="Similar bars mean that the gain does not depend on the settling time. A gain only "
             "in the slow channels would need care, because those channels are the closest to the "
             "limit of what the segment length allows.",
        run="The gain is " + ", ".join(
            f"{fmt(rows.loc[key, 'mean'])} nats in the {name}" for key, name in present) + ".",
        sources=["warmup/warmup_summary.csv"], pdfs=["warmup/warmup_tertiles.pdf"],
    )


def sufficiency(run: RunFiles) -> Dict[str, Any]:
    """Build the card: what the latent bottleneck costs the forecast, against a decoder that reads the FHR state directly.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("sufficiency/sufficiency_summary.csv").set_index("metric")
    scores = [("nll_oracle_block", "oracle decoder", PURPLE), ("mean_nll_base_block", "base, mean forecast", BASE_COLOR),
              ("mc_nll_base_block", "base, Monte Carlo", BASE_COLOR), ("mc_nll_full_block", "full, Monte Carlo", FULL_COLOR)]
    present = [(key, name, colour) for key, name, colour in scores if key in table.index]
    rows = table.loc[[key for key, _, _ in present]]
    variants = [variant(
        [bars([name for _, name, _ in present], rows["value"], "score", [c for _, _, c in present],
              lo=rows["lo"], hi=rows["hi"])],
        layout("", "forecast score (nats per anchor, lower is better)", showlegend=False),
        opt="Scores")]
    gaps = [("delta_suff_nats", "base (MC) minus oracle"), ("delta_suff_mean_nats", "base (mean) minus oracle"),
            ("pred_gap_mc_nats", "base minus full (the UA gain)")]
    present_gaps = [(key, name) for key, name in gaps if key in table.index]
    gap_rows = table.loc[[key for key, _ in present_gaps]]
    variants.append(variant(
        [bars([name for _, name in present_gaps], gap_rows["value"], "difference",
              [PURPLE, PURPLE, ORANGE][:len(present_gaps)], lo=gap_rows["lo"], hi=gap_rows["hi"])],
        layout("", "score difference (nats per anchor)", showlegend=False, shapes=[hline(0.0)]),
        opt="Differences"))
    if run.exists("sufficiency/oracle_training_curve.csv"):
        curve = run.csv("sufficiency/oracle_training_curve.csv")
        traces = []
        for index, (width, part) in enumerate(curve.groupby("width_multiplier")):
            colour = SERIES[index % len(SERIES)]
            traces.append(line(part["step"], part["held_out_nats"], f"held-out, width x{width:g}", colour))
            traces.append(line(part["step"], part["fit_nats"], f"training, width x{width:g}", colour, dash="dot", width=1.8))
        variants.append(variant(traces, layout("training step of the oracle", "score (nats per anchor)"),
                                opt="Oracle training curve"))
    gap = table.loc["delta_suff_nats"] if "delta_suff_nats" in table.index else None
    return plot_panel(
        "sufficiency", "What the latent bottleneck costs", variants, opt_label="View",
        how="The model forecasts through a small latent variable. The sufficiency analysis trains "
            "a second decoder of the same size, the oracle, that reads the encoder's FHR state "
            "directly, with no latent. It trains the oracle on half of the recordings and scores "
            "it on the other half.",
        shows="'Scores' compares the oracle with the model's forecasts on the held-out "
              "recordings. 'Differences' shows the base forecast minus the oracle, beside the UA "
              "gain for scale. 'Oracle training curve' shows whether the oracle finished training.",
        read="A base score far above the oracle's score would mean that the latent variable "
             "loses FHR information that the encoder has. A base score below the oracle's score "
             "means that the latent path loses nothing that this oracle can recover. The oracle "
             "is a lower bound of what is possible only if its training curve is flat at the end.",
        run=None if gap is None else
        f"The base forecast minus the oracle is {fmt(gap['value'])} nats per anchor on {int(gap['n'])} "
        f"held-out recordings. A negative value means that the model's own forecast is better than "
        f"the oracle's.",
        sources=["sufficiency/sufficiency_summary.csv", "sufficiency/oracle_training_curve.csv"],
        pdfs=["sufficiency/sufficiency.pdf"],
    )


# =================================================================================================
# Cohort tests, attribution and pages
# =================================================================================================
def cross_subgroup(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the test of each readout for a difference between the cohorts.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("cross_subgroup/cross_subgroup_significance.csv")
    if table.empty:
        raise MissingInput("cross_subgroup/cross_subgroup_significance.csv (no rows)", "cross_subgroup")
    table = table.sort_values("p_holm")
    rows = [
        [str(row.get("metric", "")), str(row.get("analysis", "")), int(row.get("n_recordings", 0) or 0),
         float(row.get("statistic", float("nan"))), float(row.get("p_value", float("nan"))),
         float(row.get("p_holm", float("nan"))),
         "<span class='pill %s'>%s</span>" % (
             ("fail", "differs") if str(row.get("significant")).lower() == "true" else ("skip", "no difference"))]
        for _, row in table.iterrows()
    ]
    significant = int(table["significant"].astype(str).str.lower().eq("true").sum())
    return table_panel(
        "cross-subgroup", "Do the cohorts differ? One test per readout",
        ["Readout", "Analysis", "Recordings", "Test statistic", "p-value", "Corrected p-value", "Result"], rows,
        how="The cross_subgroup analysis takes the per-recording value of each readout from the "
            "analyses above. It tests whether the cohorts have the same distribution "
            "(Kruskal-Wallis test) and corrects the p-values for the number of readouts tested "
            "(Holm correction).",
        shows="One row per readout, sorted from the smallest corrected p-value.",
        read="'differs' marks a readout whose corrected p-value is below the significance level. "
             "A difference between cohorts is a statistical statement about whole recordings. It "
             "does not say that the readout can classify a single recording.",
        run=f"{significant} of {len(table)} readouts differ between the cohorts after correction.",
        sources=["cross_subgroup/cross_subgroup_significance.csv", "cross_subgroup/cross_subgroup_pairwise.csv"],
    )


def attribution_class_tests(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the class tests of the attribution cohort.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("attribution/attribution_class_stats.csv", "attribution")
    if table.empty:
        raise MissingInput("attribution/attribution_class_stats.csv (no rows)", "attribution")
    wanted = [name for name in ("metric", "n_classes", "statistic", "p_value", "p_holm", "significant")
              if name in table.columns]
    for name in table.columns:
        if name.startswith(("mean_", "median_")) and len(wanted) < 10:
            wanted.append(name)
    shown = table.sort_values("p_holm") if "p_holm" in table.columns else table
    rows = [[row[name] if isinstance(row[name], str) else (None if pd.isna(row[name]) else
             (bool(row[name]) if isinstance(row[name], (bool, np.bool_)) else float(row[name])))
             for name in wanted] for _, row in shown.iterrows()]
    rows = [[str(cell) if isinstance(cell, bool) else cell for cell in row] for row in rows]
    return table_panel(
        "attribution-class-tests", "Attribution: do the classes differ?",
        [name.replace("_", " ") for name in wanted], rows,
        how="The attribution analysis computes integrated gradients and Grad-CAM maps on a "
            "class-balanced draw of segments, one per recording. It reduces each map to a few "
            "numbers and tests each number for a difference between the clinical classes, with a "
            "Holm correction.",
        shows="One row per attribution metric, sorted from the smallest corrected p-value.",
        read="A small corrected p-value means that the model's readout depends on different "
             "inputs in different classes. The attribution PDF pages show where the difference "
             "lies.",
        sources=["attribution/attribution_class_stats.csv", "attribution/attribution_class_pairwise.csv"],
        pdfs=["attribution/attribution_classes.pdf", "attribution/attribution_class_maps.pdf",
              "attribution/gradcam_classes.pdf"],
    )


#: What each attribution page shows, for the index card. A page the run did not write is left out.
_ATTRIBUTION_PAGES = (
    ("attribution_maps.pdf", "Example attribution maps of single anchors: which input cells the readout depends on."),
    ("attribution_blocks.pdf", "The share of the attribution on each input block: FHR scattering, FHR phase, UA scattering, UA phase."),
    ("attribution_channels.pdf", "The attribution per input channel."),
    ("attribution_bands.pdf", "The attribution per frequency band, beside the occlusion cost and the band skill."),
    ("attribution_lag_profile.pdf", "The attribution of the UA input against the lag before the anchor."),
    ("attribution_lag_channel.pdf", "The attribution of the UA input per channel and per lag."),
    ("attribution_time_profile.pdf", "The attribution against the time before the anchor."),
    ("attribution_time_to_delivery.pdf", "The attribution summaries against the time before delivery."),
    ("attribution_horizon.pdf", "The attribution of the forecast at each step of the horizon."),
    ("attribution_layer.pdf", "The attribution at the model's internal layers."),
    ("attribution_checks.pdf", "The completeness and sensitivity checks of the attribution method."),
    ("attribution_null.pdf", "The attribution against the baseline with the UA values removed."),
    ("attribution_classes.pdf", "The class comparison of the attribution summaries."),
    ("attribution_class_maps.pdf", "The mean attribution map of each clinical class."),
    ("gradcam_classes.pdf", "The Grad-CAM maps of each clinical class."),
    ("late_behaviour.pdf", "The attribution in the last part of labour."),
)


def attribution_pages(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the pages of the attribution analysis.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    files = run.glob("attribution/*.pdf")
    if not files:
        raise MissingInput("attribution/*.pdf", "attribution")
    known = dict(_ATTRIBUTION_PAGES)
    ordered = [name for name, _ in _ATTRIBUTION_PAGES if f"attribution/{name}" in files]
    ordered += sorted(name.split("/")[-1] for name in files if name.split("/")[-1] not in known)
    rows = [[f"<a href='attribution/{name}' target='_blank' rel='noopener'>{name}</a>",
             known.get(name, "A further page of the attribution analysis.")] for name in ordered]
    return table_panel(
        "attribution-pages", "Attribution pages (PDF)", ["Page", "What it shows"], rows,
        how="The attribution analysis asks which input values a readout of the model depends on. "
            "It uses integrated gradients, which follow the readout from a baseline input to the "
            "real input, and Grad-CAM, which reads the model's internal activations. The maps are "
            "images, so this card links the PDF pages.",
        read="Start with the block and band pages for the overall picture, then open the maps of "
             "single anchors. The attribution guide of the evaluation explains each page.",
    )


def sample_pages(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the per-segment pages and dashboards of the run.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("samples/sample_pages.csv", "samples")
    draws = table.groupby("selection")
    names = {
        "stratified": "A draw that covers every shard and class.",
        "by_class": "Segments of each clinical class.",
        "mean_pred_gap_high": "The segments with the largest forecast gain.",
        "mean_pred_gap_low": "The segments with the smallest forecast gain.",
        "nll_full_block_high": "The segments with the worst forecast score.",
        "nll_full_block_low": "The segments with the best forecast score.",
        "source_conditioned_kl_raw_high": "The segments with the largest KL.",
        "source_conditioned_kl_raw_low": "The segments with the smallest KL.",
    }
    rows = []
    for selection, part in draws:
        gallery = f"samples/{selection}/index.html"
        link = (f"<a href='{gallery}' target='_blank' rel='noopener'>open the gallery</a>"
                if run.exists(gallery) else "no gallery")
        rows.append([str(selection), names.get(str(selection), "A draw of segments."),
                     int(part["dataset_index"].nunique()), link])
    return table_panel(
        "sample-pages", "Full pages of single segments", ["Draw", "What it holds", "Segments", "Interactive pages"], rows,
        how="The samples analysis draws segments in several ways and writes, for each one, a full "
            "PDF page, a compact PDF page and an interactive dashboard with every input, output "
            "and readout of the model's forward.",
        shows="One row per draw, with a link to its gallery of interactive pages.",
        read="Use the view above for a quick look at the gate on a signal. Open a sample "
             "dashboard when you need every quantity of one segment, such as the forecast of a "
             "single channel or the attention of one head.",
        sources=["samples/sample_pages.csv"],
    )


#: The cards of each section, in reading order. The assembler puts the gate-aware lag cards before
#: :data:`LAG_PANELS`.
LAG_PANELS: Tuple[Callable[[RunFiles], Dict[str, Any]], ...] = (
    lag_kl_profile, source_null_lag_profile, occlusion_bands, attention_heads, attention_entropy,
    high_kl_lag_profiles, lag_profile_over_labour, lag_profile_by_class, lag_features_over_labour,
    more_lag_figures,
)
FORECAST_PANELS: Tuple[Callable[[RunFiles], Dict[str, Any]], ...] = (
    forecast_horizon, forecast_skill, spectral_skill, spectral_channels, forecast_anchor_profile,
    warmup_tertiles, forecast_scores_by_class, calibration_pit, calibration_coverage,
    calibration_logvar, sufficiency,
)
COUPLING_PANELS: Tuple[Callable[[RunFiles], Dict[str, Any]], ...] = (
    coupling_summary, source_null_summary, permutation_control, time_shift_control,
    events_conditioned, high_kl_gain, latent_spectrum, latent_diagnostics, residual_summary,
)
TIME_PANELS: Tuple[Callable[[RunFiles], Dict[str, Any]], ...] = (
    time_to_delivery_trajectory, time_to_delivery_significance, second_stage_trajectory,
    within_segment_profile, class_distributions, cross_subgroup,
)
ATTRIBUTION_PANELS: Tuple[Callable[[RunFiles], Dict[str, Any]], ...] = (
    attribution_class_tests, attribution_pages,
)
SIGNAL_PANELS: Tuple[Callable[[RunFiles], Dict[str, Any]], ...] = (sample_pages,)
