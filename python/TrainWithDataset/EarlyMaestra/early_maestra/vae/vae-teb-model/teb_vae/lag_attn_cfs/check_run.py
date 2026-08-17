r"""Score a finished run against the criteria its record registered, by code rather than by eye.

``RESULTS.md`` registers its criteria **before** the run, in two tiers, and the distinction is the
whole point of the document: tier 1 asks whether the machinery did what it was built to do and a
failure there voids the run; tier 2 is the science, has no prior against which a threshold could
have been calibrated, and is therefore *reported and interpreted* rather than passed or failed. This
module is that reading, executed.

.. code-block:: bash

    python -m teb_vae.lag_attn_cfs.check_run --run-dir <run>              # one run
    python -m teb_vae.lag_attn_cfs.check_run --run-dir <a> --second-run-dir <b>

From an IDE's Run button, with no command line: fill in ``RUN_ARGS`` near the bottom of this file.

**Why a module and not an eye.** Three of the five tier-1 criteria are statements about *every
logged row* -- one of them a $10^{-6}$ relative recomposition of two channel-axis splits against
each other -- and a multi-thousand-row CSV cannot honestly be checked by inspection. The two that
could be read off a chart are here anyway, because a checker that covered four criteria would leave
the reader to remember which one it did not.

**Every tier-1 criterion is a geometry statement, not a quality one.** ``target_warm_frac`` is a
stamped provenance column that must read exactly $1.0$; ``anchors_per_sample`` must sit at the value
this run's own resolved geometry implies; the loss must be finite and the spike breaker must never
have latched; and the two channel-axis splits of the forecast gap must recompose to each other. A
value outside any of them means the run measured something other than what its configuration says,
so no tier-2 number from it is worth reading -- which is why the exit code follows tier 1 alone and
never a tier-2 value.

**The anchor band is derived, never assumed.** ``anchors_per_sample`` is compared against
$\lceil (T_{\mathrm{valid}} - F)/S \rceil$ and its phase-dependent floor, computed from the run's
own ``resolved_config.yaml``. Hard-coding the shipped $[4, 5]$ and $137$ would make the checker
pass a run at another horizon, floor or stride for the wrong reason -- and the arms that move all
three ship in ``configs/``.

**This module is not the evaluation, and the two answer different questions.**
:mod:`teb_vae.lag_attn_cfs.eval.verify` reads a finished run's ``eval_results/summary.json`` and
answers *is this checkpoint acceptable* -- on the held-out causal split, per recording, with
bootstrap intervals and ten pre-registered verdicts. This module reads
``train_results/metrics_history.csv`` and answers *did the fit behave* -- in-sample, over one
configured dataset, as per-epoch means with no denominator and no interval on anything. The
difference that matters operationally is when each can be run: this one needs no checkpoint, no
shard and no ``torch``, so it works **while a run is still in flight**, which is what it was built
for; the gate needs a finished checkpoint and a completed evaluation pass. Two green checks that
answer two questions are only confusing if nothing says so, so both ``DESIGN.md`` §16 and
``eval/EVAL.md`` carry the same pairing in the other direction.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

#: Repository root: ``teb_vae/lag_attn_cfs/check_run.py`` -> up three.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# An IDE's Run button executes this file as a script, which puts *this directory* on sys.path
# rather than the repository root -- so the `teb_vae.` import below would fail with
# ModuleNotFoundError before __main__ is ever reached.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from teb_vae.lag_attn_rws.eval.launch import (  # noqa: E402
    missing_required,
    resolve_launch_args,
)

#: Where a run's metric history is looked for, in order, relative to the run directory. The training
#: entry point writes ``train_results/metrics_history.csv``; the other two admit a directory that has
#: been copied or narrowed by hand, which is what a run retained off the production box usually is.
METRICS_HISTORY_CANDIDATES: Tuple[str, ...] = (
    "train_results/metrics_history.csv",
    "metrics_history.csv",
    "model_checkpoints/metrics_history.csv",
)

#: Where the resolved configuration is looked for, in order. The driver writes it beside the
#: checkpoints, and its write is deliberately non-fatal on failure -- so a run can legitimately lack
#: it, which is why ``--config`` exists rather than this being the only route.
RESOLVED_CONFIG_CANDIDATES: Tuple[str, ...] = (
    "model_checkpoints/resolved_config.yaml",
    "resolved_config.yaml",
    "train_results/resolved_config.yaml",
)

#: The stages the per-stage criteria are read on. ``test`` is deliberately absent: no shipped
#: configuration runs it, so requiring it would fail every real run for a column nothing writes.
STAGES: Tuple[str, ...] = ("train", "val")

#: Relative tolerance of the recomposition criterion. The two channel-axis splits difference the
#: same per-element scores under the same mask, so they agree to float32 noise; neither agrees with
#: ``pred_gap`` itself to anything like this, which is why the criterion compares them against each
#: **other** rather than against the number they both decompose.
RECOMPOSITION_TOLERANCE = 1e-6

#: How many trailing rows the coupling readout's tail is read over. Not a criterion -- criterion 6
#: carries no threshold and no epoch -- but "still rising at the end" needs a window to be a
#: statement at all, and one stated here is one a reader can check.
TAIL_ROWS = 10

#: Tier-2 readouts, by the suffix the task emits them under, with the stages each appears on. The
#: source-null KL is validation-only: it is a readout that never enters the objective, so a
#: ``train/`` column would be NaN in every row of every run.
TIER_TWO_METRICS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("source_conditioned_kl_raw", STAGES),
    ("kld_source_null", ("val",)),
    ("kld_active_frac", STAGES),
    ("logvar_prior_floor_frac", STAGES),
    ("shuffle_penalty", ("val",)),
    ("source_lag_warmth_frac_st", STAGES),
    ("source_lag_warmth_frac_ph", STAGES),
)

#: The three warm-up tertile columns, whose *spread* is what tier-2 criterion 9 asks about: they
#: recompose to the gap by construction whether or not they differ from each other, so the
#: recomposition says nothing about whether the split distinguishes anything.
TERTILE_SUFFIXES: Tuple[str, ...] = (
    "pred_gap_warm_lo",
    "pred_gap_warm_mid",
    "pred_gap_warm_hi",
)

#: Verdict strings. ``NOT_EVALUATED`` is not a pass: criterion 5 needs two run directories, and a
#: checker that reported an unevaluated criterion as satisfied would be worse than one that did not
#: evaluate it at all.
PASS, FAIL, NOT_EVALUATED = "PASS", "FAIL", "n/a"


class Verdict(NamedTuple):
    """One criterion's outcome.

    Attributes:
        number: The criterion's number in the record, so the output cites what a reader cites.
        title: What the criterion says, in one line.
        status: :data:`PASS`, :data:`FAIL` or :data:`NOT_EVALUATED`.
        detail: The number behind the verdict, or what stopped it being reached.
    """

    number: int
    title: str
    status: str
    detail: str


class AnchorBand(NamedTuple):
    r"""What ``anchors_per_sample`` must read, derived from one run's own geometry.

    Attributes:
        train_low: $\lceil (T_{\mathrm{valid}} - F - (S - 1))/S \rceil$, the tile count at the
            least favourable phase.
        train_high: $A_{\max} = \lceil (T_{\mathrm{valid}} - F)/S \rceil$, the count at $\varphi=0$.
        dense: $T_{\mathrm{valid}} - F$, every valid anchor -- what both evaluation stages decode.
    """

    train_low: int
    train_high: int
    dense: int


def _first_existing(root: Path, candidates: Sequence[str]) -> Optional[Path]:
    """Return the first candidate that exists under ``root``, or ``None``.

    Args:
        root: The run directory.
        candidates: Relative paths, in preference order.

    Returns:
        The resolved path, or ``None`` when none of them is a file.
    """
    for relative in candidates:
        path = root / relative
        if path.is_file():
            return path
    return None


def read_metric_rows(path: Path) -> List[Dict[str, str]]:
    """Read a run's ``metrics_history.csv`` as raw text rows.

    The stdlib reader and **no float conversion**: criterion 5 compares two runs' rows for
    identity, and a parse-then-compare would let two differently-written numbers that round to one
    float read as identical. Each per-criterion check converts the columns it needs itself.

    Args:
        path: The CSV.

    Returns:
        One dict per row, values as written.
    """
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _floats(rows: Sequence[Dict[str, str]], column: str) -> List[float]:
    """Every parsable value of one column, in row order.

    Blank cells are skipped rather than read as zero: a metric logged on one stage alone leaves the
    other stage's column empty on every row, and a zero there would be a measurement nobody made.

    Args:
        rows: The parsed rows.
        column: The column name, with its stage prefix.

    Returns:
        The values.
    """
    values: List[float] = []
    for row in rows:
        text = (row.get(column) or "").strip()
        if not text:
            continue
        try:
            values.append(float(text))
        except ValueError:
            continue
    return values


def resolve_anchor_band(config: Dict[str, Any]) -> AnchorBand:
    r"""Derive the anchor counts a run's configuration implies.

    Anchors live in $[F, T_{\mathrm{valid}})$ with $T_{\mathrm{valid}} = T - H$, and a training step
    decodes the tile set $\mathcal{A}(\varphi) = \{F + \varphi + kS\}$ whose size is
    $\lceil (T_{\mathrm{valid}} - F - \varphi)/S \rceil$ -- a **ceiling**, so it varies by one across
    phases. Both evaluation stages decode the dense set instead.

    Args:
        config: The resolved run configuration.

    Returns:
        The band.

    Raises:
        KeyError: If the configuration carries no ``model_config.VAE_model`` geometry.
    """
    vae = (config.get("model_config") or {}).get("VAE_model") or {}
    sequence_length = int(vae["sequence_length"])
    horizon = int(vae["horizon"])
    floor = int(vae["warmup_period"])
    # A model constructed without an opinion decodes the dense range, which is stride 1 -- the
    # inert value the whole family shares, and the one an arm config restores deliberately.
    stride = int(vae.get("anchor_stride") or 1)

    dense = max(sequence_length - horizon - floor, 0)
    return AnchorBand(
        train_low=-(-max(dense - (stride - 1), 0) // stride),
        train_high=-(-dense // stride),
        dense=dense,
    )


# =================================================================================================
# Tier 1: the machinery did what it was built to do, or the run is void
# =================================================================================================
def check_warm_fraction(rows: Sequence[Dict[str, str]]) -> Verdict:
    """Criterion 1: ``target_warm_frac`` is exactly $1.0$ on every logged row of both stages.

    A **stamped provenance column** rather than a runtime measurement: it is resolved at
    construction and the constructor already refuses a budget-and-floor pairing that would violate
    it, so any other value means the checkpoint was built by code predating that refusal. Exactness
    is the point -- the constant is written into the metric dict as a literal, so a nearly-one value
    is not a rounding but a different constant.

    Args:
        rows: The parsed rows.

    Returns:
        The verdict.
    """
    offenders: List[str] = []
    seen, empty = 0, []
    for stage in STAGES:
        values = _floats(rows, f"{stage}/target_warm_frac")
        seen += len(values)
        if not values:
            empty.append(stage)
        offenders += [f"{stage} row {index}: {value!r}"
                      for index, value in enumerate(values) if value != 1.0]
    if empty:
        # Both stages, because a per-row test over an absent column passes over nothing: the
        # criterion is a statement about every logged row of *both*.
        return Verdict(1, "target_warm_frac is exactly 1.0", FAIL,
                       f"no column on {', '.join(empty)}, so the pairing cannot be read there")
    if offenders:
        return Verdict(1, "target_warm_frac is exactly 1.0", FAIL,
                       f"{len(offenders)} of {seen} rows differ: {offenders[:3]}")
    return Verdict(1, "target_warm_frac is exactly 1.0", PASS, f"{seen} rows, all exactly 1.0")


def check_anchor_count(rows: Sequence[Dict[str, str]], band: Optional[AnchorBand]) -> Verdict:
    """Criterion 2: ``anchors_per_sample`` sits at the value this run's geometry implies.

    Not a fixed band. The training stages decode a tile set whose size varies by one with the
    per-segment phase, and both evaluation stages decode every valid anchor; both numbers follow
    from the horizon, the floor and the stride, and three shipped arms move one of those.

    Args:
        rows: The parsed rows.
        band: The derived band, or ``None`` when no configuration could be read.

    Returns:
        The verdict.
    """
    title = "anchors_per_sample is the geometry-derived count"
    if band is None:
        return Verdict(2, title, FAIL,
                       "no resolved configuration was found, so the geometry this must equal is "
                       "unknown; pass --config")

    train = _floats(rows, "train/anchors_per_sample")
    dense = _floats(rows, "val/anchors_per_sample")
    if not train or not dense:
        return Verdict(2, title, FAIL, "one of the two stages' columns is absent")

    # A step's value is the mean over batch elements, whose phases differ, so a training row is a
    # real number strictly inside the band rather than one of its two endpoints.
    train_bad = [value for value in train if not band.train_low <= value <= band.train_high]
    dense_bad = [value for value in dense if value != float(band.dense)]
    if train_bad or dense_bad:
        return Verdict(2, title, FAIL,
                       f"expected train in [{band.train_low}, {band.train_high}] and val exactly "
                       f"{band.dense}; {len(train_bad)} train and {len(dense_bad)} val rows differ "
                       f"(e.g. {(train_bad + dense_bad)[:3]})")
    return Verdict(2, title, PASS,
                   f"train in [{band.train_low}, {band.train_high}] on {len(train)} rows, val "
                   f"exactly {band.dense} on {len(dense)}")


def check_loss_and_breaker(rows: Sequence[Dict[str, str]]) -> Verdict:
    """Criterion 3: the loss is finite on every row and the spike breaker never latched.

    Both halves, because they fail differently. A non-finite loss should have been caught by the
    breaker's own non-finite guard first, so seeing one here says the guard did not fire; a
    ``spike_skipped`` above zero says the additive margin is mis-tuned for this objective, which is
    a statement about the configuration rather than about the model.

    Args:
        rows: The parsed rows.

    Returns:
        The verdict.
    """
    title = "the loss is finite and the breaker never latched"
    losses = _floats(rows, "train/total_loss")
    skipped = _floats(rows, "train/spike_skipped")
    if not losses:
        return Verdict(3, title, FAIL, "train/total_loss is absent")

    non_finite = [value for value in losses if not math.isfinite(value)]
    latched = [value for value in skipped if value != 0.0]
    if non_finite or latched:
        return Verdict(3, title, FAIL,
                       f"{len(non_finite)} non-finite loss row(s) and {len(latched)} row(s) with "
                       f"a skipped batch (max skip fraction {max(latched, default=0.0):g})")
    if not skipped:
        return Verdict(3, title, FAIL,
                       "train/spike_skipped is absent, so the breaker cannot be shown not to have "
                       "latched")
    return Verdict(3, title, PASS, f"{len(losses)} finite rows, no batch skipped")


def check_gap_recomposition(rows: Sequence[Dict[str, str]]) -> Verdict:
    r"""Criterion 4: the two channel-axis splits of the forecast gap agree with each other.

    Compared **split against split** rather than either against ``pred_gap``, deliberately.
    ``pred_gap`` is ``nll_base_block - nll_full_block``, a difference of two order-$10^{3}$ sums over
    $2940$ coefficients, so it loses several decimal digits to cancellation *before* any split is
    formed; the two splits difference the same per-element scores elementwise and agree to float32
    noise, while neither agrees with ``pred_gap`` to $10^{-6}$.

    Args:
        rows: The parsed rows.

    Returns:
        The verdict.
    """
    title = "the two gap splits recompose to each other"
    worst, compared = 0.0, 0
    for stage in STAGES:
        tertiles = [_floats(rows, f"{stage}/{name}") for name in TERTILE_SUFFIXES]
        blocks = [_floats(rows, f"{stage}/pred_gap_{name}") for name in ("st", "ph")]
        lengths = {len(series) for series in tertiles + blocks}
        # One length, and not zero: a missing column reads as an empty series, and comparing the
        # rows the two splits happen to share would let a half-absent split pass.
        if lengths != {len(tertiles[0])} or lengths == {0}:
            return Verdict(4, title, FAIL,
                           f"{stage}: the five split columns have lengths "
                           f"{[len(series) for series in tertiles + blocks]}")
        for index in range(len(tertiles[0])):
            warm = sum(series[index] for series in tertiles)
            block = sum(series[index] for series in blocks)
            scale = max(abs(warm), abs(block))
            relative = abs(warm - block) if scale == 0.0 else abs(warm - block) / scale
            worst = max(worst, relative)
            compared += 1
    if not compared:
        return Verdict(4, title, FAIL, "no stage carries all five split columns")
    if worst > RECOMPOSITION_TOLERANCE:
        return Verdict(4, title, FAIL,
                       f"worst relative disagreement {worst:.3e} over {compared} rows, against a "
                       f"tolerance of {RECOMPOSITION_TOLERANCE:g}")
    return Verdict(4, title, PASS, f"worst relative disagreement {worst:.3e} over {compared} rows")


def check_two_evaluations(
    rows: Sequence[Dict[str, str]], second: Optional[Sequence[Dict[str, str]]]
) -> Verdict:
    """Criterion 5: two evaluations of the final checkpoint produce an identical metric row set.

    Not derivable from one directory, so it is reported as not evaluated rather than assumed when
    no second one is given. Compared as **text**: identical anchor indices are necessary and not
    sufficient, since the reparameterisation draw and the permutation generator also move, and a
    comparison that parsed first would hide a difference below the last printed digit.

    Args:
        rows: The first run's rows.
        second: The second run's rows, or ``None``.

    Returns:
        The verdict.
    """
    title = "two evaluations produce an identical metric row set"
    if second is None:
        return Verdict(5, title, NOT_EVALUATED,
                       "no second run directory given; pass --second-run-dir to evaluate it")
    if len(rows) != len(second):
        return Verdict(5, title, FAIL, f"{len(rows)} rows against {len(second)}")

    columns, other_columns = set(rows[0] if rows else {}), set(second[0] if second else {})
    if columns != other_columns:
        return Verdict(5, title, FAIL,
                       f"column sets differ: only in the first {sorted(columns - other_columns)}, "
                       f"only in the second {sorted(other_columns - columns)}")
    differing = [index for index, (left, right) in enumerate(zip(rows, second)) if left != right]
    if differing:
        return Verdict(5, title, FAIL,
                       f"{len(differing)} of {len(rows)} rows differ, first at row "
                       f"{differing[0]}")
    return Verdict(5, title, PASS, f"{len(rows)} rows identical")


# =================================================================================================
# Tier 2: reported and interpreted, never gated
# =================================================================================================
def _tail_direction(values: Sequence[float]) -> str:
    """Whether a series is still rising at its end, over :data:`TAIL_ROWS` rows.

    Args:
        values: The series, in epoch order.

    Returns:
        A short phrase naming the direction and the change behind it.
    """
    if len(values) < 2:
        return "too few rows to say"
    tail = values[-min(TAIL_ROWS, len(values)):]
    change = tail[-1] - tail[0]
    direction = "rising" if change > 0 else ("falling" if change < 0 else "flat")
    return f"{direction} by {change:+.4g} over the last {len(tail)} rows"


def tier_two_readings(rows: Sequence[Dict[str, str]]) -> List[Tuple[str, str]]:
    """Every tier-2 number, beside its name.

    Reported and never gated. This is the first causal-feature model in the tree: there is no prior
    against which a threshold on any of these could have been calibrated, and the decoded anchor
    count per step fell by roughly $15\\times$ against the two-sided sibling, which changes the
    optimisation regime -- so a fixed threshold here would be a guess dressed as a gate.

    Args:
        rows: The parsed rows.

    Returns:
        ``(name, reading)`` pairs, in report order.
    """
    readings: List[Tuple[str, str]] = []
    for suffix, stages in TIER_TWO_METRICS:
        for stage in stages:
            values = _floats(rows, f"{stage}/{suffix}")
            if not values:
                readings.append((f"{stage}/{suffix}", "absent"))
                continue
            reading = f"final {values[-1]:.6g}"
            if suffix == "source_conditioned_kl_raw":
                # The trajectory, not only the endpoint: criterion 6 asks whether the coupling
                # readout is still rising at the end, which an endpoint alone cannot answer.
                reading += f"; first {values[0]:.6g}; {_tail_direction(values)}"
            readings.append((f"{stage}/{suffix}", reading))

    # The coupling readout beside its own floor. If the two are close, the readout is measuring the
    # availability clock rather than source content -- the single most important number on the page,
    # and one no permutation control can produce, since every row of a batch shares the clock.
    kl_raw = _floats(rows, "val/source_conditioned_kl_raw")
    null = _floats(rows, "val/kld_source_null")
    if kl_raw and null:
        readings.append(
            ("val/source_conditioned_kl_raw - val/kld_source_null",
             f"{kl_raw[-1] - null[-1]:+.6g} (the part attributable to source variation)")
        )

    for stage in STAGES:
        finals = [_floats(rows, f"{stage}/{name}") for name in TERTILE_SUFFIXES]
        if all(finals):
            last = [series[-1] for series in finals]
            readings.append(
                (f"{stage}/pred_gap_warm spread",
                 f"lo {last[0]:.6g}, mid {last[1]:.6g}, hi {last[2]:.6g}; "
                 f"max-min {max(last) - min(last):.6g}")
            )
    return readings


# =================================================================================================
# The report
# =================================================================================================
def _load_config(path: Path) -> Dict[str, Any]:
    """Read a resolved configuration.

    Args:
        path: The YAML file.

    Returns:
        The parsed mapping, or an empty one when the file holds nothing.

    Raises:
        ValueError: If the file does not parse as YAML. Translated rather than left as
            ``yaml.YAMLError``, which derives from ``Exception`` and not from any of the three the
            caller catches -- so a truncated ``resolved_config.yaml``, which is what a killed run
            leaves behind and exactly when this checker gets pointed at a directory, would abort
            the whole report instead of failing the one criterion that needs the geometry.
    """
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        try:
            return yaml.safe_load(handle) or {}
        except yaml.YAMLError as error:
            raise ValueError(f"{path} does not parse as YAML: {error}") from error


def main(
    run_dir: Optional[str] = None,
    second_run_dir: Optional[str] = None,
    config: Optional[str] = None,
) -> int:
    """Score one run directory and print the report.

    Args:
        run_dir: The run directory to score.
        second_run_dir: A second one, for the two-evaluations criterion alone.
        config: The resolved configuration, when it is not inside the run directory. The driver's
            write of that file is deliberately non-fatal, so a run can legitimately lack it.

    Returns:
        The exit code: non-zero when any **tier 1** criterion failed, and never because of a
        tier-2 value.
    """
    if run_dir is None:
        print(missing_required({}, ["run_dir"]))
        return 2

    root = Path(run_dir)
    history = _first_existing(root, METRICS_HISTORY_CANDIDATES)
    if history is None:
        print(f"no metric history under {root}; looked for "
              f"{', '.join(METRICS_HISTORY_CANDIDATES)}")
        return 2
    rows = read_metric_rows(history)
    if not rows:
        print(f"{history} carries no rows")
        return 2

    config_path = Path(config) if config else _first_existing(root, RESOLVED_CONFIG_CANDIDATES)
    band: Optional[AnchorBand] = None
    if config_path is not None and config_path.is_file():
        try:
            band = resolve_anchor_band(_load_config(config_path))
        except (KeyError, TypeError, ValueError) as error:
            # Reported through criterion 2's own FAIL rather than raised: the other four criteria
            # are readable without it, and a checker that died here would report nothing at all.
            print(f"could not derive the anchor geometry from {config_path}: {error}")

    second_rows: Optional[List[Dict[str, str]]] = None
    if second_run_dir is not None:
        second_history = _first_existing(Path(second_run_dir), METRICS_HISTORY_CANDIDATES)
        if second_history is None:
            print(f"no metric history under {second_run_dir}; criterion 5 cannot be evaluated")
        else:
            second_rows = read_metric_rows(second_history)

    verdicts = [
        check_warm_fraction(rows),
        check_anchor_count(rows, band),
        check_loss_and_breaker(rows),
        check_gap_recomposition(rows),
        check_two_evaluations(rows, second_rows),
    ]

    print(f"run directory: {root}")
    print(f"metric history: {history} ({len(rows)} rows)")
    print(f"resolved configuration: {config_path if config_path else 'not found'}")
    print()
    print("Tier 1 -- must hold, or the run is void")
    for verdict in verdicts:
        print(f"  [{verdict.status:>4}] {verdict.number}. {verdict.title}")
        print(f"         {verdict.detail}")
    print()
    print("Tier 2 -- reported and interpreted, not passed or failed")
    for name, reading in tier_two_readings(rows):
        print(f"  {name}: {reading}")

    tally = {status: sum(1 for verdict in verdicts if verdict.status == status)
             for status in (PASS, FAIL, NOT_EVALUATED)}
    print()
    print(f"tier 1: {tally[PASS]} passed, {tally[FAIL]} failed, "
          f"{tally[NOT_EVALUATED]} not evaluated")
    return 1 if tally[FAIL] else 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    No argument is ``required=True`` and none carries a non-``None`` default, both deliberately:
    ``required=True`` fires before :data:`RUN_ARGS` is ever consulted, and the merge reads any
    non-``None`` parsed value as having come from the command line, so an argparse default would
    make that key's ``RUN_ARGS`` entry unreachable with nothing saying why.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description="Score a run against its registered criteria.")
    parser.add_argument("--run-dir", default=None, help="The run directory to score.")
    parser.add_argument(
        "--second-run-dir",
        default=None,
        help="A second run directory, for the two-evaluations criterion alone.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="The resolved configuration, when it is not inside the run directory.",
    )
    return parser


#: Values used when this module is launched with no command line -- i.e. an IDE's Run button. Keyed
#: by argparse ``dest``; a flag on the command line overrides its entry here, per key.
#:
#: ``run_dir`` must be filled in for the module to run at all.
RUN_ARGS: Dict[str, Any] = {
    # The run directory to score: the one holding train_results/metrics_history.csv.
    "run_dir": None,
    # A second run directory, to evaluate the two-evaluations criterion. None leaves that criterion
    # reported as not evaluated rather than as satisfied.
    "second_run_dir": None,
    # The resolved configuration, when the run directory does not carry one. None discovers it.
    "config": None,
}


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    """Resolve the launch arguments and run.

    Args:
        argv: Command-line arguments. ``None`` reads ``sys.argv[1:]``.

    Returns:
        The exit code, which ``__main__`` hands to ``sys.exit``.
    """
    values, sources = resolve_launch_args(build_parser(), RUN_ARGS, argv)
    refusal = missing_required(values, ["run_dir"])
    if refusal is not None:
        print(refusal)
        return 2
    print(f"launch arguments: "
          f"{', '.join(f'{key}={values[key]!r} ({sources[key]})' for key in sorted(values))}")
    return main(**values)


if __name__ == "__main__":
    sys.exit(_cli())
