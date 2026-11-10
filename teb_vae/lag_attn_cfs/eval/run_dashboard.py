r"""The run dashboard: every result of one evaluation run on one interactive page.

.. code-block:: bash

    python -m teb_vae.lag_attn_cfs.eval.run_dashboard --output-dir <run directory>

A run of the evaluation leaves one folder per analysis, each with tables and PDF figures. This
module reads those tables and writes ``dashboard.html`` into the results directory, beside
``summary.json``. The page draws every result with Plotly, and each plot has a text beside it
that says how the plot was made, what it shows and how to read it, in plain words. The PDF figures
stay the printed record, and each card links the PDFs and the tables it was drawn from.

**The page has three controls.** The section list on the left changes the group of plots. The
``Anchors`` filter at the top chooses which anchors the gate-aware plots use: every scored anchor,
the anchors with the posterior event gate on, the anchors with the gate fully open, or the anchors
with the gate off. A plot that follows the filter says so on its card. The search box finds a plot
by a word of its title or its explanation.

**It reads files and nothing else.** No model is built and no forward runs, so the page can be
rebuilt at any time from a finished run directory. A plot whose table is absent becomes a card
that names the missing file and the analysis that writes it, so a directory that holds a few
analyses still gets a page.

**A directory filled by several passes is read as one run.** A pass that runs a subset of the
analyses writes a summary of that subset and renames the earlier summary aside.
:class:`~teb_vae.lag_attn_cfs.eval.run_dashboard_core.RunFiles` merges those files, newest first.

The evaluation runner calls :func:`write_dashboard` at the end of every pass, so the page is always
as new as the tables. This module is also a runner of its own: edit :data:`RUN_ARGS` at the bottom
of the file and press the IDE's Run button.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: Repository root: ``teb_vae/lag_attn_cfs/eval/run_dashboard.py`` -> up four.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Launched as a script (an IDE's Run button) this file's own directory goes on sys.path instead
# of the repository root, and every absolute import below fails before __main__ is reached.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

from teb_vae.lag_attn_cfs.eval import run_dashboard_analyses as analyses  # noqa: E402
from teb_vae.lag_attn_cfs.eval import run_dashboard_gate as gate  # noqa: E402
from teb_vae.lag_attn_cfs.eval.run_dashboard_core import (  # noqa: E402
    GATE_OPEN_FROM,
    GATE_SHUT_BELOW,
    POPULATIONS,
    SECONDS_PER_STEP,
    Geometry,
    MissingInput,
    RunFiles,
    count,
    fmt,
    html_panel,
    json_safe,
    missing_panel,
    pct,
    table_panel,
    tile,
    tiles_panel,
)

#: The results directory of a run, inside its run directory.
RESULTS_DIRNAME = "eval_results"

#: The page this module writes, and the plotting library it writes beside the page.
DASHBOARD_FILENAME = "dashboard.html"
PLOTLY_FILENAME = "plotly.min.js"

#: The page's own template files, and the sample dashboard's stylesheet the page loads first so
#: both dashboards share one set of colour tokens.
_ASSET_DIR = Path(__file__).resolve().parent / "run_dashboard_assets"
_BASE_STYLESHEET = Path(__file__).resolve().parent / "sample_dashboard_assets" / "dashboard.css"

#: The population the page shows first. The gate-on anchors, because they are the question the
#: page exists to answer; the filter's first entry returns the standard all-anchor reading.
DEFAULT_POPULATION = "gate_on"

#: One builder of one card: it takes the run's files and returns a panel dictionary.
Builder = Callable[[RunFiles], Dict[str, Any]]


# =================================================================================================
# Section: overview
# =================================================================================================
def _verdict_tone(status: str) -> str:
    """Map a verdict or a check result to the class of its pill.

    Args:
        status: The status text, such as ``'PASS'`` or ``'inconclusive'``.

    Returns:
        ``'pass'``, ``'fail'``, ``'inc'`` or ``'skip'``.
    """
    text = str(status).lower()
    if text == "pass":
        return "pass"
    if text == "fail":
        return "fail"
    return "inc" if text.startswith("inc") else "skip"


def overview_tiles(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the headline numbers of the run.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    results = dict(run.summary.get("results") or {})
    readouts = dict(results.get("readouts") or {})
    if not readouts:
        raise MissingInput("summary.json (results.readouts)", "the collection pass")
    collection = dict(run.summary.get("collection") or {})
    gate_block = gate.gate_headline(run)
    coupling = dict(run.result("source_null").get("difference") or {})
    verdicts = list(results.get("verdicts") or [])
    passed = sum(str(entry.get("status")).upper() == "PASS" for entry in verdicts)
    failed = sum(str(entry.get("status")).upper() == "FAIL" for entry in verdicts)
    gain = readouts.get("mc_pred_gap")
    clock = coupling.get("coupling_minus_clock_nats", readouts.get("coupling_minus_clock"))
    tiles = [
        tile("Recordings", count(results.get("n_recordings")),
             f"{count(collection.get('n_per_sample_rows'))} segments, "
             f"{count(collection.get('n_per_anchor_rows'))} scored anchors"),
        tile("Forecast gain, all anchors", f"{fmt(gain)} nats",
             "Base forecast score minus full forecast score, per anchor. Positive means that the "
             "UA helped.", "good" if (gain or 0) > 0 else "bad"),
        tile("Forecast gain, gate on", f"{fmt(gate_block.get('gate_on_gain_mc_nats'))} nats",
             f"On the {pct(gate_block.get('gate_on_frac'))} of anchors with a gate of "
             f"{GATE_SHUT_BELOW:g} or more.",
             "good" if (gate_block.get("gate_on_gain_mc_nats") or 0) > 0 else ""),
        tile("Forecast gain, gate off", f"{fmt(gate_block.get('gate_off_gain_mc_nats'), 2)} nats",
             f"On the anchors with a gate below {GATE_SHUT_BELOW:g}. A value near zero is the "
             "expected result."),
        tile("KL, all anchors", f"{fmt(readouts.get('source_conditioned_kl_raw'))} nats",
             f"The size of the change the UA makes in the model's belief. "
             f"{pct(gate_block.get('gate_on_kl_share_of_run'), 1)} of it comes from gate-on anchors."),
        tile("KL above the clock", f"{fmt(clock)} nats",
             "The KL minus the KL that remains when the UA values are removed. It is the part that "
             "comes from the UA values."),
        tile("Lag of the KL, gate on", f"{fmt(gate_block.get('gate_on_kl_centroid_s'))} s",
             f"Mean lag of the KL attribution. Lag 0 carries {pct(gate_block.get('gate_on_kl_lag0_share'))}."),
        tile("Verdicts", f"{passed} pass / {failed} fail",
             f"Inconclusive: {len(verdicts) - passed - failed} of {len(verdicts)}.",
             "bad" if failed else "good"),
    ]
    return tiles_panel("overview-tiles", tiles)


def overview_findings(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the main results of the run as a short list, written from the numbers.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    results = dict(run.summary.get("results") or {})
    readouts = dict(results.get("readouts") or {})
    gate_block = gate.gate_headline(run)
    items: List[str] = []
    if readouts:
        items.append(
            f"<b>The UA improves the forecast by {fmt(readouts.get('mc_pred_gap'))} nats per anchor "
            f"over all anchors.</b> The base forecast scores {fmt(readouts.get('mc_nll_base_block'), 5)} "
            f"nats and the full forecast {fmt(readouts.get('mc_nll_full_block'), 5)} nats. A lower "
            f"score is a better forecast."
        )
    if gate_block:
        items.append(
            f"<b>The gate is on at {pct(gate_block.get('gate_on_frac'))} of the anchors, and those "
            f"anchors carry {pct(gate_block.get('gate_on_kl_share_of_run'), 1)} of the KL.</b> With the "
            f"gate on, the gain is {fmt(gate_block.get('gate_on_gain_mc_nats'))} nats per anchor. With "
            f"the gate fully open it is {fmt(gate_block.get('fully_open_gain_mc_nats'))} nats. With the "
            f"gate off it is {fmt(gate_block.get('gate_off_gain_mc_nats'), 2)} nats."
        )
        items.append(
            f"<b>A gate-on episode lasts {fmt(gate_block.get('gate_on_median_run_s'), 3)} s in "
            f"the median.</b> The Gate section shows when the gate opens relative to the "
            f"contractions."
        )
        items.append(
            f"<b>With the gate on, the mean lag of the KL is "
            f"{fmt(gate_block.get('gate_on_kl_centroid_s'))} s.</b> Lag 0 carries "
            f"{pct(gate_block.get('gate_on_kl_lag0_share'))} of the KL. The lag axis is "
            f"stored-coefficient time and is not a physiological delay."
        )
    failed = [str(entry.get("name")) for entry in results.get("verdicts") or []
              if str(entry.get("status")).upper() == "FAIL"]
    if failed:
        items.append(
            f"<b>Failed verdicts: {', '.join(failed)}.</b> The Overview table below gives the "
            f"criterion and the reason for each one."
        )
    if not items:
        raise MissingInput("summary.json", "the collection pass")
    return html_panel("overview-findings", "Main results of this run",
                      "<ul class='findings'>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>")


def _stat_path(run: RunFiles) -> str:
    """Read the normalisation statistics file the run's merged configuration names.

    Args:
        run: The run's files.

    Returns:
        The path, or ``'unknown'`` when the configuration cannot be read.
    """
    try:
        import yaml

        with open(run.require("resolved_config.yaml"), encoding="utf-8") as handle:
            return str(yaml.safe_load(handle)["dataset_config"]["stat_path"])
    except Exception:  # noqa: BLE001 - the path is a courtesy line, not a result
        return "unknown"


def overview_glossary(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the terms the page uses, each defined once.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    context = dict(run.summary.get("run_context") or {})
    geometry = Geometry(run)
    steps = "" if geometry.segment_steps is None else (
        f" It has {geometry.segment_steps} steps of {SECONDS_PER_STEP:g} s."
    )
    rows = [
        ("Recording", "The FHR and UA signals of one delivery."),
        ("Segment", f"One piece of a recording, a {geometry.segment_text}.{steps}"),
        ("Anchor", f"A step at which the model makes a forecast of {geometry.horizon_text} of the "
                   f"FHR features. A segment has {geometry.anchors_text}."),
        ("Base forecast", "The forecast that uses the FHR history only."),
        ("Full forecast", "The forecast that uses the FHR history and the UA history."),
        ("NLL", "Negative log-likelihood in nats. It scores a forecast against what happened. "
                "A lower value is a better forecast."),
        ("Forecast gain", "The NLL of the base forecast minus the NLL of the full forecast. A "
                          "positive value means that the UA improved the forecast."),
        ("KL", "The Kullback-Leibler divergence between the model's belief with the UA and its "
               "belief without it, in nats. It measures how much the UA changes the belief. It "
               "does not say whether the change is useful."),
        ("Gate (g)", "A number between 0 and 1 that the model computes at each anchor. It scales "
                     "how much the UA can change the belief. At 0 the UA has no effect."),
        ("Gate on / fully open / gate off",
         f"Anchors with g of {GATE_SHUT_BELOW:g} or more / g of {GATE_OPEN_FROM:g} or more / "
         f"g below {GATE_SHUT_BELOW:g}."),
        ("Lag", f"How far back in the UA history the model reads. The model attends over "
                f"{geometry.lags_text}. The lag axis is stored-coefficient time and is not a "
                f"physiological delay."),
        ("Monte Carlo (MC) gain", "The forecast gain averaged over several random draws of the "
                                  "model's latent variable. It is the headline version of the gain."),
        ("Clinical class", "Healthy, acidosis or HIE (hypoxic-ischaemic encephalopathy), from the "
                           "outcome of the delivery."),
    ]
    body = "<table class='data'><tbody>" + "".join(
        f"<tr><td><b>{html.escape(name)}</b></td><td>{html.escape(text)}</td></tr>" for name, text in rows
    ) + "</tbody></table>"
    identity = (
        f"<p>Model <b>{html.escape(str(context.get('model_class', 'unknown')))}</b>, training epoch "
        f"{html.escape(str(context.get('train_epoch', 'unknown')))}, "
        f"{count(context.get('n_parameters'))} parameters. Checkpoint: "
        f"<code>{html.escape(str(run.summary.get('checkpoint', 'unknown')))}</code>.</p>"
        f"<p>Normalisation statistics: <code>{html.escape(_stat_path(run))}</code>. Evaluation "
        f"overrides: <code>{html.escape(str(dict(run.summary.get('config_overrides') or {}).get('path', 'unknown')))}"
        f"</code>. Scores in nats depend on the statistics file, so compare two runs only when "
        f"they use the same one.</p>"
    )
    return html_panel("overview-glossary", "Terms used on this page", "<div class='prose'>" + identity + "</div>" + body)


def overview_verdicts(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the run's verdicts: each criterion, its status and the reason.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    verdicts = list(dict(run.summary.get("results") or {}).get("verdicts") or [])
    if not verdicts:
        raise MissingInput("summary.json (results.verdicts)", "the collection pass")
    rows = [
        [str(entry.get("name", "")).replace("_", " "),
         f"<span class='pill {_verdict_tone(entry.get('status'))}'>{html.escape(str(entry.get('status')))}</span>",
         html.escape(str(entry.get("criterion", ""))), html.escape(str(entry.get("detail", "")))]
        for entry in verdicts
    ]
    return table_panel(
        "overview-verdicts", "Verdicts", ["Verdict", "Status", "Criterion", "What the run found"], rows,
        how="The collection pass tests each criterion on the run's own readouts. The thresholds "
            "are in the evaluation config.",
        shows="D is a forecast score in nats per anchor, and a lower D is better. D_base is the "
              "base forecast, D_full the full forecast, and D_shuffled the forecast with the UA "
              "of another recording.",
        read="PASS means that the criterion holds. FAIL means that it does not. INCONCLUSIVE "
             "means that the run cannot decide, for example because a threshold is not set.",
        sources=["summary.json"],
    )


def overview_sanity(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the run's internal consistency checks.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    checks = dict(dict(dict(run.summary.get("results") or {}).get("sanity") or {}).get("checks") or {})
    if not checks:
        raise MissingInput("summary.json (results.sanity)", "a full evaluation pass")
    rows = [
        [name.replace("_", " "),
         f"<span class='pill {_verdict_tone(entry.get('verdict'))}'>{html.escape(str(entry.get('verdict')))}</span>",
         html.escape(str(entry.get("detail", "")))]
        for name, entry in checks.items()
    ]
    return table_panel(
        "overview-sanity", "Consistency checks", ["Check", "Result", "Detail"], rows,
        how="After the analyses, the runner tests identities that must hold if the tables are "
            "correct. For example, the KL per latent dimension must sum to the total KL. A pass "
            "can test only the analyses it ran, so this table shows the checks of the pass that "
            "ran the most analyses into this directory"
            + (f" (<code>{html.escape(str(run.summary.get('derived_blocks_from')))}</code>)."
               if run.summary.get("derived_blocks_from") else "."),
        shows="One row per check, with its result and the measured numbers.",
        read="A failed check means that a table or a readout is not trustworthy, and you must "
             "resolve it before you read the plots. An inconclusive check states why it could "
             "not decide.",
        sources=["summary.json"],
    )


def overview_population(run: RunFiles) -> Dict[str, Any]:
    """Build the card: the evaluated population per clinical class.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    table = run.csv("per_sample.csv", "the collection pass")
    rows = []
    for name, group in table.groupby("clinical_class"):
        hours = -group["epoch"].to_numpy(dtype=float) / 3600.0
        rows.append([str(name), int(group["guid"].nunique()), int(len(group)),
                     int(np.nansum(group["n_anchors"])), float(np.nanmin(hours)), float(np.nanmax(hours))])
    rows.append(["all", int(table["guid"].nunique()), int(len(table)), int(np.nansum(table["n_anchors"])),
                 float(np.nanmin(-table["epoch"] / 3600.0)), float(np.nanmax(-table["epoch"] / 3600.0))])
    return table_panel(
        "overview-population", "Evaluated population",
        ["Class", "Recordings", "Segments", "Scored anchors", "Closest segment start (h before delivery)",
         "Earliest segment start (h before delivery)"], rows,
        how="The collection pass writes one row per segment. This table counts the rows per "
            "clinical class.",
        shows="The number of recordings, segments and scored anchors in each class, and the span "
              "of labour that the segments cover.",
        read="A class with few recordings gives wide intervals and noisy lines on every per-class "
             "plot. Read that class with care.",
        sources=["per_sample.csv"],
    )


def overview_steps(run: RunFiles) -> Dict[str, Any]:
    """Build the card: which analyses ran, and how long each took.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.

    Raises:
        MissingInput: If a table the card needs is absent.
    """
    steps = list(run.summary.get("steps_merged") or [])
    if not steps:
        raise MissingInput("summary.json (steps)", "any evaluation pass")
    rows = [
        [str(step.get("name")),
         f"<span class='pill {'pass' if step.get('ok') else 'fail'}'>{html.escape(str(step.get('status', 'ok' if step.get('ok') else 'failed')))}</span>",
         float(step.get("elapsed_s") or 0.0)]
        for step in steps
    ]
    return table_panel(
        "overview-steps", "Analyses in this run directory", ["Analysis", "Status", "Seconds"], rows,
        how="Each evaluation pass records its steps in summary.json. This table merges the "
            "records of every pass into this directory and keeps the newest record of each step.",
        shows="One row per analysis, its status, and its run time.",
        read="An analysis that is not listed did not run, and its plots show a card that names "
             "the missing file.",
        sources=["summary.json"],
    )


# =================================================================================================
# Section: files
# =================================================================================================
def files_index(run: RunFiles) -> Dict[str, Any]:
    """Build the card: every figure and table of the run directory, as links.

    Args:
        run: The run's files.

    Returns:
        The card, as the page script reads it.
    """
    folders: Dict[str, Dict[str, List[str]]] = {}
    for relative in run.glob("**/*"):
        suffix = Path(relative).suffix.lower()
        parts = Path(relative).parts
        if suffix not in (".pdf", ".csv", ".html", ".json", ".parquet", ".npz") or ".bak." in relative:
            continue
        folder = parts[0] if len(parts) > 1 else "(run directory)"
        if folder == "samples" and len(parts) > 2:
            # The sample pages have a gallery of their own, linked from the side bar.
            continue
        folders.setdefault(folder, {}).setdefault(suffix, []).append(relative)

    def links(items: Sequence[str]) -> str:
        """Render file paths as a comma-separated list of links named by file."""
        return ", ".join(
            f"<a href='{html.escape(item)}' target='_blank' rel='noopener'>{html.escape(Path(item).name)}</a>"
            for item in items
        )

    rows = [
        [folder, links(kinds.get(".pdf", [])) or "none",
         links([*kinds.get(".csv", []), *kinds.get(".parquet", []), *kinds.get(".npz", []),
                *kinds.get(".json", []), *kinds.get(".html", [])]) or "none"]
        for folder, kinds in sorted(folders.items())
    ]
    if not rows:
        raise MissingInput("(no files)", "any evaluation pass")
    return table_panel(
        "files-index", "All figures and tables", ["Folder", "PDF figures", "Tables and records"], rows,
        how="This table lists the files of the results directory, grouped by the folder of the "
            "analysis that wrote them.",
        shows="Each name is a link that opens the file from this page's own folder.",
        read="The PDF figures are the printed version of the results. The CSV tables hold the "
             "numbers behind the plots of this page.",
    )


# =================================================================================================
# The page
# =================================================================================================
#: The sections of the page, in reading order: identifier, title, introduction, builders.
SECTIONS: Tuple[Tuple[str, str, str, Tuple[Builder, ...]], ...] = (
    ("overview", "Overview",
     "The main numbers of the run, its verdicts and its population. Start here, then use the "
     "sections on the left.",
     (overview_tiles, overview_findings, overview_verdicts, overview_sanity, overview_population,
      overview_glossary, overview_steps)),
    ("gate", "The gate",
     "The posterior event gate decides, at each anchor, how much the UA can change the model's "
     "belief. This section shows how often the gate is on, when it opens, and what an open gate "
     "gives.",
     (gate.gate_histogram, gate.gate_states, gate.gate_vs_contraction, gate.gate_value_payoff,
      gate.gate_states_over_time, gate.gate_runs, gate.gate_share_by_class,
      gate.gate_share_per_recording)),
    ("gate-on", "Results with the gate on",
     "The same readouts as the standard evaluation, restricted to a gate population. Use the "
     "Anchors filter at the top of the page to change between all anchors, gate on, fully open "
     "and gate off.",
     (gate.population_summary, gate.population_contributions, gate.class_trajectory,
      gate.class_trajectory_tests, gate.gain_vs_contraction, gate.gain_per_recording,
      gate.gain_vs_kl_scatter, gate.population_distributions)),
    ("signal", "The gate on the signal",
     "Single segments with their FHR and UA signals, the gate, the KL, the forecast gain and the "
     "lag map on one time axis. The view can hide the parts of the signal where the gate is off.",
     (gate.signal_explorer, *analyses.SIGNAL_PANELS)),
    ("lags", "Lags",
     "Where in the UA history the model reads. The first plots follow the Anchors filter. The "
     "later plots are the standard lag analyses over all anchors.",
     (gate.lag_profile_populations, gate.lag_shares, gate.lag_vs_contraction, gate.lag_vs_time,
      gate.lag_vs_gate, gate.lag_centroid_clocks, gate.lag_centroid_per_recording,
      *analyses.LAG_PANELS)),
    ("forecast", "Forecast quality",
     "How good the forecasts are over the horizon, against simple baselines, and how well the "
     "predicted uncertainty matches the errors.",
     analyses.FORECAST_PANELS),
    ("coupling", "Coupling and controls",
     "How much the UA changes the model's belief and its forecast, and the controls that test "
     "whether that change comes from this recording's UA.",
     analyses.COUPLING_PANELS),
    ("time", "Labour time and cohorts",
     "The gain and the KL against time before delivery and against the second stage, and the "
     "comparison of the clinical classes.",
     analyses.TIME_PANELS),
    ("attribution", "Attribution",
     "Which inputs the model's readouts depend on, from the integrated-gradients and Grad-CAM "
     "pass.",
     analyses.ATTRIBUTION_PANELS),
    ("files", "Files",
     "Every PDF figure and every table of this run, in one list.",
     (files_index,)),
)


def _guarded(builder: Builder, run: RunFiles) -> Dict[str, Any]:
    """Run one builder, and turn a missing input or an error into a card that says so.

    Args:
        builder: The card's builder.
        run: The run's files.

    Returns:
        The builder's panel, or a ``missing`` panel that names the cause.
    """
    name = builder.__name__.replace("_", "-")
    title = (builder.__doc__ or name).strip().splitlines()[0].rstrip(".")
    try:
        return builder(run)
    except MissingInput as error:
        source = f" It is written by <b>{html.escape(error.analysis)}</b>." if error.analysis else ""
        return missing_panel(
            name, title,
            f"This run directory does not hold <code>{html.escape(error.path)}</code>.{source} "
            f"Write it into this directory, then rebuild the dashboard.",
        )
    except Exception as error:  # noqa: BLE001 - one card is not worth the rest of the page
        logger.warning(f"run dashboard: {builder.__name__} failed: {type(error).__name__}: {error}")
        return missing_panel(
            name, title,
            f"This plot could not be built from the tables: "
            f"<code>{html.escape(type(error).__name__)}: {html.escape(str(error))}</code>.",
        )


def build_payload(results_dir: Any) -> Dict[str, Any]:
    """Build the page's data from a results directory.

    Args:
        results_dir: The ``eval_results`` directory of a run.

    Returns:
        The payload the page script reads: the run's identity, the populations and the sections
        with their panels.
    """
    run = RunFiles(results_dir)
    context = dict(run.summary.get("run_context") or {})
    sections = []
    for identifier, title, intro, builders in SECTIONS:
        panels = [_guarded(builder, run) for builder in builders]
        sections.append({"id": identifier, "title": title, "intro": intro, "panels": panels})
    links = []
    if run.exists("samples/index.html"):
        links.append({"label": "Sample dashboards", "href": "samples/index.html"})
    if run.exists("summary.json"):
        links.append({"label": "summary.json", "href": "summary.json"})
    model = str(context.get("model_class") or "evaluation run")
    epoch = context.get("train_epoch")
    return {
        "meta": {
            "title": f"Run dashboard: {model}" + (f", epoch {epoch}" if epoch is not None else ""),
            "subtitle": str(context.get("training_tag") or Path(results_dir).resolve().parent.name),
            "links": links,
        },
        "populations": list(POPULATIONS),
        "default_population": DEFAULT_POPULATION,
        "sections": sections,
    }


def render_page(payload: Dict[str, Any]) -> str:
    """Fill the page template with a payload.

    Args:
        payload: From :func:`build_payload`.

    Returns:
        The page's HTML text.
    """
    # A closing tag inside a JSON string would end the script block early.
    data = json.dumps(json_safe(payload), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    return (
        (_ASSET_DIR / "dashboard.html").read_text(encoding="utf-8")
        .replace("__TITLE__", html.escape(payload["meta"]["title"]))
        .replace("__PLOTLY__", PLOTLY_FILENAME)
        .replace("__BASE_CSS__", _BASE_STYLESHEET.read_text(encoding="utf-8"))
        .replace("__CSS__", (_ASSET_DIR / "dashboard.css").read_text(encoding="utf-8"))
        .replace("__JS__", (_ASSET_DIR / "dashboard.js").read_text(encoding="utf-8"))
        .replace("__PAYLOAD__", data)
    )


def write_dashboard(results_dir: Any) -> Dict[str, Any]:
    """Write ``dashboard.html`` and the plotting library into a results directory.

    Args:
        results_dir: The ``eval_results`` directory of a run.

    Returns:
        A record of what was written: the file, the number of cards, and the cards that stand in
        for a missing input.
    """
    results_dir = Path(results_dir)
    payload = build_payload(results_dir)
    # The library first: a page written without it cannot draw anything, so a failure to write
    # the library must leave no new page behind.
    library = results_dir / PLOTLY_FILENAME
    if not library.is_file():
        from plotly.offline import get_plotlyjs

        library.write_text(get_plotlyjs(), encoding="utf-8")
    page = results_dir / DASHBOARD_FILENAME
    page.write_text(render_page(payload), encoding="utf-8")
    panels = [panel for section in payload["sections"] for panel in section["panels"]]
    missing = [panel["id"] for panel in panels if panel["kind"] == "missing"]
    logger.info(
        f"run dashboard: wrote {page} with {len(panels)} card(s), "
        f"{len(missing)} without their input"
    )
    return {"file": DASHBOARD_FILENAME, "n_panels": len(panels), "missing": missing}


def resolve_results_dir(output_dir: Any) -> Path:
    """Find the results directory a path names.

    Args:
        output_dir: A run directory that holds ``eval_results``, or that results directory itself.

    Returns:
        The results directory.

    Raises:
        FileNotFoundError: If the path holds no evaluation tables.
    """
    path = Path(output_dir)
    for candidate in (path / RESULTS_DIRNAME, path):
        if (candidate / "per_sample.csv").is_file() or (candidate / "summary.json").is_file():
            return candidate
    raise FileNotFoundError(
        f"{path} holds no evaluation results: neither it nor its {RESULTS_DIRNAME} folder has a "
        f"per_sample.csv or a summary.json."
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The parser. Every ``dest`` is also a valid :data:`RUN_ARGS` key.
    """
    parser = argparse.ArgumentParser(
        prog="python -m teb_vae.lag_attn_cfs.eval.run_dashboard",
        description="Write the interactive dashboard of a finished evaluation run.",
    )
    parser.add_argument(
        "--output-dir", dest="output_dir", default=None,
        help="The run directory given to the evaluation as --output-dir, or its eval_results "
             "folder.",
    )
    return parser


def _cli(argv: Optional[List[str]] = None) -> int:
    """Parse arguments, fall back to :data:`RUN_ARGS`, and write the page.

    Args:
        argv: The command line, or ``None`` for the process's own.

    Returns:
        The process exit code: zero when the page holds at least one card.
    """
    values = vars(build_parser().parse_args(argv))
    unknown = sorted(set(RUN_ARGS) - set(values))
    if unknown:
        raise SystemExit(f"RUN_ARGS has keys that are not arguments: {unknown}")
    for key, value in RUN_ARGS.items():
        if values.get(key) is None:
            values[key] = value
    if values["output_dir"] is None:
        raise SystemExit("--output-dir is required: name a finished evaluation run directory.")
    if not os.path.isabs(str(values["output_dir"])) and os.path.abspath(os.getcwd()) != _REPO_ROOT:
        os.chdir(_REPO_ROOT)
    record = write_dashboard(resolve_results_dir(values["output_dir"]))
    return 0 if record["n_panels"] else 1


#: Values used for arguments absent from the command line, which is the IDE's Run button.
#:
#: ``output_dir`` is the run directory the evaluation wrote into, the same value as the
#: ``output_dir`` of ``eval/run.py``. A relative path is read from the repository root.
RUN_ARGS: Dict[str, Any] = {
    "output_dir": "tmp/checkpoint-2026-11-09/eval",
}


if __name__ == "__main__":
    sys.exit(_cli())
