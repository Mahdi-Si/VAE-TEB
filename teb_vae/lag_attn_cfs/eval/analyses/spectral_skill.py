r"""The forecast gap resolved by the frequency band of the target coefficient.

**The channel axis of this target domain is a frequency axis, and this is the analysis that uses
it.** A first-order scattering coefficient is $|x \star \psi_\lambda|$ -- the envelope of the
signal filtered at one centre frequency -- and a phase-harmonic channel pairs two such filters, so
the frequency resolution the raw cells build with a Welch periodogram is already present here, per
channel, for free. What is reported is therefore *how well the model forecasts the envelope in
each clinical band*, per recording and bootstrapped over recordings, in both the likelihood space
the objective is stated in and the error space that has a natural zero.

**This is band-resolved skill. It is not coherence, and the difference is not a technicality.** A
scattering coefficient is a **modulus**: the analysing filter's phase was discarded before the
value was stored. So the three things the raw pipeline's frequency-domain analysis exists to
separate -- phase agreement, group delay, and the residual's split into irreducible, timing and
amplitude terms -- have no analogue here at any window length. A forecast that is right in every
band but arrives a step late reads here as a forecast that is right. The name says so, and the
emitted record says so in a sentence, so that a reader who knows the raw pipeline cannot carry the
wrong contract across.

Two further limits ride in the record for the same reason. The band is the band of the **analysing
filter**, not a bin of the forecast's own spectrum -- two different objects a reader will conflate
if nothing says otherwise. And a phase-harmonic channel has a *pair* of frequencies; it is banded
by the shared channel map's ``freq_hz_primary`` convention, which is stated rather than assumed.

**One join is the whole correctness risk of this module.** The per-channel readouts are positional
against the $C_{\mathrm{keep}}$ channels the warm-up budget left standing; the channel-to-band map
is over the $c_y$ **declared** ones. Joining them positionally would shift band membership across
the axis -- and on the shipped dataset the dropped channels happen to be the trailing four, so a
positional join looks right here and is wrong on any dataset whose survivors are not a prefix. A
join that is accidentally correct on the fixture is worse than one that is wrong, because no test
catches it.

The join therefore goes through the **kept-axis map persisted to disk** by the channel-map step,
which carries each kept channel's position on the scored axis beside the declared index it came
from. Reading it off disk rather than asking the model which channels survived is what makes this
analysis work on the path it has to work on: ``--only spectral_skill --output-dir <a finished
run>``, with no checkpoint, no model and no GPU. It is the same file-on-disk dependency the
cross-cohort analysis already has on the per-recording tables above it, and a map whose length
disagrees with the vectors' width is a **raise** rather than a truncation, because a silently
shortened join gives every band statement a silently wrong denominator.

**Coverage is emitted as five counts rather than one ratio.** On the shipped dataset the declared
and scored numerators coincide at $95$ by arithmetic accident ($102 - 7 = 98 - 3$), and quoting
"95 of 102" would imply this analysis scored channels the decoder never emitted. The channels no
selected filter pair named are reported as their own ``unknown`` row with their count, never
bucketed into a neighbouring band whose skill they do not share and never dropped.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from teb_vae.lag_attn_cfs.eval._reuse import band_partition as shared
from teb_vae.lag_attn_cfs.eval._reuse import figures, stats as shared_stats
from teb_vae.lag_attn_cfs.eval.frames import (
    RECOMPOSITION_SCALE_COLUMN,
    describe,
    finite_column,
    grouped_frame_entry,
    per_recording_means,
    recomposition_check,
    scored_sample_count,
    skill_against,
)

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "spectral_skill"

#: What it writes. The per-recording frame's name and its ``pred_gap_<band>`` columns are what
#: ``cross_subgroup`` reads, so both are a contract rather than a filename and a label.
PER_RECORDING_FILENAME = "spectral_skill_per_recording.csv"
BAND_FILENAME = "spectral_skill_bands.csv"
CHANNEL_FILENAME = "spectral_skill_channels.csv"
BAND_FIGURE = "spectral_skill_bands"

#: The two channel maps this analysis joins through, both written into the results **root** by the
#: unskippable channel-map step. Named here rather than imported from the module that writes them,
#: because an analysis may not import another and because the dependency is genuinely on the files
#: being on disk -- which is what makes an offline re-run of this analysis alone work at all.
KEPT_CHANNEL_MAP_FILENAME = "band_channel_map_kept.csv"
DECLARED_CHANNEL_MAP_FILENAME = "band_channel_map.csv"

#: The stream the forecast is over. The declared-axis map carries both streams, and only the
#: gated one has anything to do with the target channel axis these vectors are indexed on.
TARGET_STREAM = "target"

#: The per-sample vector readouts this is computed from, all $C_{\mathrm{keep}}$ wide and in
#: ``per_sample.csv``'s row order.
GAP_VECTOR = "gap_per_channel"
ERROR_VECTORS: Tuple[Tuple[str, str], ...] = (
    ("base", "sq_error_per_channel_base"),
    ("full", "sq_error_per_channel_full"),
)

#: The scalar the per-band gaps must sum to, over the same denominator.
TOTAL_COLUMN = "pred_gap"

#: The band order every table, figure and column set is emitted in: the clinical bands in
#: ascending frequency as the shared table declares them, then the unbanded channels. Read off the
#: shared table rather than restated, so a band added there appears here rather than vanishing.
BAND_ORDER: Tuple[str, ...] = tuple(shared.CLINICAL_BANDS) + (shared.UNKNOWN_BAND,)

#: The three limits that travel in the emitted record. Written out rather than left to the module
#: docstring: a reader of ``summary.json`` has the record and not this file.
LIMITS: Tuple[str, ...] = (
    "the stored coefficients are moduli -- the analysing filter's phase was discarded before the "
    "value was stored -- so nothing here can say whether a forecast is mistimed rather than "
    "mis-scaled. Phase agreement, group delay and the residual's timing/amplitude split have no "
    "analogue in this target domain at any window length. This is band-resolved skill, not "
    "coherence.",
    "the band is the band of the ANALYSING FILTER that produced the target coefficient, not a bin "
    "of the forecast's own spectrum. The two are different objects and only the first is "
    "recoverable from what the shards store.",
    "a phase-harmonic channel has a pair of centre frequencies and is banded by the shared "
    "channel map's freq_hz_primary convention (the higher-frequency member of the pair); the "
    "secondary frequency travels in the channel map beside it.",
)


def read_channel_maps(output_dir: Any) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
    """Read both persisted channel maps off disk, or say why the join cannot be made.

    Args:
        output_dir: The run's results directory, where the channel-map step wrote both files.

    Returns:
        ``(kept, declared, reason)``. Either frame is ``None`` when its file is absent, and
        ``reason`` is empty exactly when the kept-axis map -- the one the join needs -- was read.
        A missing map is a skip rather than a raise: it means the shards carried no channel
        provenance, which is a property of the dataset and not a fault of this run.
    """
    root = Path(output_dir)
    kept_path = root / KEPT_CHANNEL_MAP_FILENAME
    declared_path = root / DECLARED_CHANNEL_MAP_FILENAME
    if not kept_path.is_file():
        return None, None, (
            f"{KEPT_CHANNEL_MAP_FILENAME} was not written, so no channel has a band and there is "
            f"nothing to resolve the forecast by. The channel map records its own skip reason."
        )
    kept = pd.read_csv(kept_path)
    declared = pd.read_csv(declared_path) if declared_path.is_file() else None
    return kept, declared, ""


def band_positions(kept: pd.DataFrame, width: int) -> Dict[str, np.ndarray]:
    r"""Group the **scored** channel axis by band, through the map's own kept-axis positions.

    The one place this module can be wrong in silence, and the reason it indexes by
    ``kept_channel`` rather than by the row's position in the file or by its declared index: the
    per-channel vectors are positional against the surviving channels, and on the shipped dataset
    the survivors happen to be a prefix -- so a positional join is accidentally correct here and
    wrong on any dataset whose dropped channels are not the trailing ones.

    Args:
        kept: The persisted kept-axis channel map.
        width: The per-channel vectors' width, $C_{\mathrm{keep}}$.

    Returns:
        Band name to the positions on the scored axis it covers, in :data:`BAND_ORDER` and with
        empty bands omitted. The positions tile $[0, C_{\mathrm{keep}})$ exactly.

    Raises:
        ValueError: If the map and the vectors describe different axes -- a length disagreement, a
            missing column, or positions that do not tile the axis. Every one of them would
            otherwise give each band a silently wrong denominator, which is the failure a
            truncating join produces and no downstream number would reveal.
    """
    for column in ("kept_channel", "band"):
        if column not in kept.columns:
            raise ValueError(
                f"the persisted kept-axis channel map carries no {column!r} column, so the "
                f"per-channel readouts cannot be placed on the frequency axis. Re-run the channel "
                f"map step against a shard carrying sel_* provenance."
            )
    if len(kept) != int(width):
        raise ValueError(
            f"the kept-axis channel map describes {len(kept)} channel(s) but the per-channel "
            f"readouts are {int(width)} wide. The map and the collected vectors describe "
            f"different channel axes, so no band-resolved statement over them is meaningful -- "
            f"and a join that silently used the shorter of the two would give every band a wrong "
            f"denominator rather than failing."
        )

    positions = np.asarray(kept["kept_channel"], dtype=np.int64)
    if sorted(positions.tolist()) != list(range(int(width))):
        raise ValueError(
            f"the kept-axis channel map's kept_channel column is not a permutation of "
            f"[0, {int(width)}): it holds {sorted(positions.tolist())[:8]}... . Those values are "
            f"the positions the per-channel readouts are indexed by, so anything else would "
            f"attribute one channel's skill to another."
        )

    bands = [str(value) for value in kept["band"]]
    grouped: Dict[str, np.ndarray] = {}
    for name in BAND_ORDER:
        selected = positions[np.asarray([band == name for band in bands], dtype=bool)]
        if selected.size:
            grouped[name] = np.sort(selected)
    # A band the shared table does not declare would otherwise vanish from every count below.
    unknown_names = sorted({band for band in bands} - set(BAND_ORDER))
    if unknown_names:
        raise ValueError(
            f"the kept-axis channel map names band(s) {unknown_names} that the shared band table "
            f"does not declare. A band nothing here knows about would be dropped from every "
            f"per-band table silently."
        )
    return grouped


def coverage_counts(
    kept: pd.DataFrame, declared: Optional[pd.DataFrame], groups: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    r"""The five counts of the covered axis, never one ratio.

    Five rather than one because the declared and scored numerators can coincide by arithmetic
    accident -- on the shipped dataset both are $95$, since $102 - 7 = 98 - 3$ -- and "95 of 102"
    would imply this analysis scored channels the decoder never emitted.

    Args:
        kept: The persisted kept-axis map.
        declared: The declared-axis map, or ``None`` when it was not written. The two declared
            counts are then ``None`` rather than guessed from the kept map, which cannot see a
            channel the budget dropped.
        groups: What :func:`band_positions` produced.

    Returns:
        The five counts, the per-band channel counts, and the band breakdown of the channels the
        budget dropped -- which is what says *which* end of the frequency axis the budget removed.
    """
    unknown_kept = int(groups.get(shared.UNKNOWN_BAND, np.empty(0)).size)
    counts: Dict[str, Any] = {
        "declared_total": None,
        "dropped_declared": None,
        "kept_total": int(len(kept)),
        "known_kept": int(len(kept)) - unknown_kept,
        "unknown_kept": unknown_kept,
        "channels_per_band": {name: int(positions.size) for name, positions in groups.items()},
        "dropped_bands": {},
    }
    if declared is None or "stream" not in declared.columns:
        return counts

    target = pd.DataFrame(declared[declared["stream"].astype(str) == TARGET_STREAM])
    counts["declared_total"] = int(len(target))
    if "kept" not in target.columns:
        return counts
    dropped = pd.DataFrame(target[~target["kept"].astype(bool)])
    counts["dropped_declared"] = int(len(dropped))
    breakdown: Dict[str, int] = {}
    if "band" in dropped.columns:
        for band in list(dropped["band"]):
            name = str(band)
            breakdown[name] = breakdown.get(name, 0) + 1
    counts["dropped_bands"] = breakdown
    return counts


def _vector(collection: Any, name: str, n_rows: int) -> Optional[np.ndarray]:
    """Return one per-sample vector readout as a float array, or ``None`` when it is unusable.

    Args:
        collection: The collection, whose ``vectors`` are in the per-sample table's row order.
        name: The readout's name.
        n_rows: How many rows the per-sample table has, which the vector must match.

    Returns:
        The $(N, C)$ array, or ``None`` when the readout is absent or its row count disagrees with
        the table -- a mis-assembled sidecar rather than a short one, and reading it row for row
        would attribute one recording's channels to another.
    """
    vectors: Dict[str, Any] = dict(getattr(collection, "vectors", None) or {})
    values = vectors.get(name)
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != int(n_rows) or array.shape[1] == 0:
        return None
    return array


def band_frame(
    per_sample: pd.DataFrame,
    gap: np.ndarray,
    errors: Dict[str, np.ndarray],
    groups: Dict[str, np.ndarray],
) -> pd.DataFrame:
    r"""Reduce the per-channel vectors onto the band axis, per **sample**.

    Per sample rather than per recording, because everything in this pipeline reduces per recording
    only after it has reduced per anchor -- so the band columns join the per-sample table and then
    travel the same aggregation chain every other readout does.

    The gap is **summed** over a band's channels and the squared errors are **averaged**. That
    asymmetry is the arithmetic rather than a preference: the gap is a block score difference whose
    channel decomposition is additive, so the bands must sum back to ``pred_gap``; a squared error
    is a per-coefficient quantity, and summing it would make a band's error a statement about how
    many channels it holds.

    Args:
        per_sample: The collected per-sample table, for the key columns the chain resolves on.
        gap: The per-channel gap vectors, $(N, C_{\mathrm{keep}})$.
        errors: Branch name to its per-channel squared-error vectors, the same shape.
        groups: Band name to the positions it covers.

    Returns:
        A frame in ``per_sample``'s row order carrying ``guid``, the cohort columns, and one
        ``pred_gap_<band>`` plus one ``sq_error_<branch>_<band>`` per band.
    """
    carried = [
        name for name in ("guid", "epoch", "clinical_class", "subgroup", "sample_index")
        if name in per_sample.columns
    ]
    frame = pd.DataFrame(per_sample[carried]).copy()
    for band, positions in groups.items():
        frame[f"pred_gap_{band}"] = gap[:, positions].sum(axis=1)
        for branch, values in errors.items():
            frame[f"sq_error_{branch}_{band}"] = values[:, positions].mean(axis=1)
    return frame


def build_band_rows(
    per_guid: pd.DataFrame, groups: Dict[str, np.ndarray], *, resamples: int, seed: int
) -> List[Dict[str, Any]]:
    """Summarise every band over the recordings: the gap, both branches' error and the skill.

    The skill is computed per recording and then averaged rather than as a ratio of two averages,
    which is the form every acceptance criterion in this pipeline is stated in: a forecast equal to
    the truth scores exactly $1$ on every recording and one equal to the baseline exactly $0$, so
    the mean carries those answers unchanged and a bootstrap has a per-recording quantity to
    resample.

    Args:
        per_guid: Per-recording means of the band columns.
        groups: Band name to the positions it covers, for the channel count on each row.
        resamples: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        One row per band, in :data:`BAND_ORDER`, each carrying its channel count so a band
        statement is never read without the width it was measured over.
    """
    rows: List[Dict[str, Any]] = []
    for band, positions in groups.items():
        gap = finite_column(per_guid, f"pred_gap_{band}")
        interval = shared_stats.bootstrap_ci(gap, resamples=resamples, seed=seed)
        base = finite_column(per_guid, f"sq_error_base_{band}")
        full = finite_column(per_guid, f"sq_error_full_{band}")
        skill = skill_against(full, base)
        skill_interval = shared_stats.bootstrap_ci(skill, resamples=resamples, seed=seed)
        rows.append(
            {
                "band": band,
                "n_channels": int(positions.size),
                "band_hz": list(shared.CLINICAL_BANDS.get(band, (float("nan"), float("nan")))),
                "pred_gap_nats": interval["point"],
                "pred_gap_ci_lo": interval["lo"],
                "pred_gap_ci_hi": interval["hi"],
                "n_recordings": int(interval["n"]),
                "sq_error_base": float(np.nanmean(base)) if np.isfinite(base).any()
                else float("nan"),
                "sq_error_full": float(np.nanmean(full)) if np.isfinite(full).any()
                else float("nan"),
                # 1 - MSE_full / MSE_base, per recording then averaged: the error-space reading,
                # which has a natural zero the nats column does not.
                "mse_skill": skill_interval["point"],
                "mse_skill_ci_lo": skill_interval["lo"],
                "mse_skill_ci_hi": skill_interval["hi"],
                "bootstrap_resamples": int(interval["resamples"]),
                "unit": "nats per anchor",
            }
        )
    return rows


def channel_profile(
    per_sample: pd.DataFrame,
    kept: pd.DataFrame,
    gap: np.ndarray,
    errors: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """The full per-channel profile behind the band rows, on the same aggregation chain.

    Emitted beside the per-band table so a reader can go from a band statement to the channels it
    came from: a band whose gap is carried by one channel of thirty is a different finding from one
    spread evenly, and the band row alone cannot tell them apart.

    Args:
        per_sample: The collected per-sample table, for the recording each row belongs to.
        kept: The persisted kept-axis map, joined on its own ``kept_channel`` positions.
        gap: The per-channel gap vectors.
        errors: Branch name to its per-channel squared-error vectors.

    Returns:
        One row per kept channel carrying its band, its declared index, its centre frequencies and
        the three readouts, each averaged within a recording and then across recordings.
    """
    guids = (
        per_sample["guid"] if "guid" in per_sample.columns
        else pd.Series(["all"] * len(per_sample))
    )
    columns: Dict[str, np.ndarray] = {"pred_gap": gap}
    for branch, values in errors.items():
        columns[f"sq_error_{branch}"] = values

    means: Dict[str, np.ndarray] = {}
    per_recording: Dict[str, pd.DataFrame] = {}
    for name, values in columns.items():
        frame = pd.DataFrame(values)
        frame["guid"] = list(guids)
        per_guid = frame.groupby("guid").mean()
        per_recording[name] = per_guid
        means[name] = np.asarray(per_guid.mean(axis=0), dtype=np.float64)
    # The per-channel skill on the same chain as the band rows and the figure: skill PER
    # RECORDING, then averaged -- not one minus the ratio of two recording means, which is a
    # different statistic under the same name.
    channel_skill = np.full(len(means.get("sq_error_full", [])), np.nan)
    if "sq_error_full" in per_recording and "sq_error_base" in per_recording:
        full = per_recording["sq_error_full"].to_numpy(dtype=np.float64)
        base = per_recording["sq_error_base"].to_numpy(dtype=np.float64)
        per_recording_skill = np.column_stack(
            [skill_against(full[:, c], base[:, c]) for c in range(full.shape[1])]
        ) if full.size else np.zeros((0, 0))
        if per_recording_skill.size:
            channel_skill = np.nanmean(per_recording_skill, axis=0)

    ordered = kept.sort_values("kept_channel")
    rows: List[Dict[str, Any]] = []
    for _, record in ordered.iterrows():
        position = int(record["kept_channel"])
        row: Dict[str, Any] = {
            "kept_channel": position,
            # The declared index the position came from, so the two axes can be reconciled by a
            # reader rather than trusted.
            "declared_channel": record.get("channel"),
            "band": record.get("band"),
            "block": record.get("block"),
            "kind": record.get("kind"),
            "freq_hz_primary": record.get("freq_hz_primary"),
            "freq_hz_secondary": record.get("freq_hz_secondary"),
            "causal_warmup_steps": record.get("causal_warmup_steps"),
            "causal_delay_s": record.get("causal_delay_s"),
        }
        for name, values in means.items():
            row[name] = float(values[position]) if position < values.size else float("nan")
        row["mse_skill"] = (
            float(channel_skill[position]) if position < channel_skill.size else float("nan")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def recomposition(per_guid: pd.DataFrame, groups: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Whether the per-band gaps sum back to ``pred_gap``, on the worst recording.

    Args:
        per_guid: Per-recording means carrying the per-band gaps, ``pred_gap`` and the block score
            the tolerance is scaled by.
        groups: The bands that were emitted.

    Returns:
        What :func:`~teb_vae.lag_attn_cfs.eval.frames.recomposition_check` reports.
    """
    return recomposition_check(
        per_guid,
        [f"pred_gap_{band}" for band in groups],
        TOTAL_COLUMN,
        identity="sum over bands of pred_gap_<band> == pred_gap",
    )


def band_headline(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten the per-band gaps into the block the headline registry digs into.

    Every band the shared table declares gets a key whether or not this run had channels in it, so
    an arm table's column set does not change with the dataset -- and an absent band is ``None``
    rather than ``NaN``, because the headline's finiteness check reads a number that is not finite
    as a broken readout while ``None`` correctly means "this run did not report it".

    Args:
        rows: What :func:`build_band_rows` produced.

    Returns:
        ``{'pred_gap_<band>_nats': value}`` for every band in :data:`BAND_ORDER`.
    """
    by_band = {str(row["band"]): row for row in rows}
    headline: Dict[str, Any] = {}
    for band in BAND_ORDER:
        value = (by_band.get(band) or {}).get("pred_gap_nats")
        headline[f"pred_gap_{band}_nats"] = (
            None if value is None or not np.isfinite(float(value)) else float(value)
        )
    return headline


def build_band_figure(per_guid: pd.DataFrame, rows: Sequence[Dict[str, Any]]) -> Any:
    """Draw the gap and the error-space skill per band, on separate axes.

    Two panels because the two are in different units and one shared axis would flatten whichever
    is smaller into a line at zero. Each violin's label carries its channel count, so a band
    carried by three channels cannot be read as one carried by forty.

    Args:
        per_guid: Per-recording means of the band columns.
        rows: The summary rows, read for the channel counts.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(2)
    labelled = {
        f"{row['band']} ({int(row['n_channels'])} ch)": finite_column(
            per_guid, f"pred_gap_{row['band']}"
        )
        for row in rows
    }
    figures.violin_panel(
        axes[0, 0],
        labelled,
        title="forecast gap per recording, by the band of the target coefficient",
        ylabel="nats per anchor",
        reference=0.0,
        reference_label="no improvement",
    )
    skills = {
        f"{row['band']} ({int(row['n_channels'])} ch)": skill_against(
            finite_column(per_guid, f"sq_error_full_{row['band']}"),
            finite_column(per_guid, f"sq_error_base_{row['band']}"),
        )
        for row in rows
    }
    figures.violin_panel(
        axes[1, 0],
        skills,
        title="error-space skill of the source-conditioned branch against the target-only one",
        ylabel="1 - MSE_full / MSE_base",
        reference=0.0,
        reference_label="no improvement",
    )
    return figure


def skip_record(reason: str) -> Dict[str, Any]:
    """The protocol's keys for a run this analysis cannot resolve a band axis for.

    Args:
        reason: What was missing.

    Returns:
        The skip, with ``n_samples`` at ``None`` rather than zero so the coverage block reads it as
        "this analysis described nothing" rather than as a population disagreement.
    """
    return {
        "n_samples": None,
        "composition": {},
        "plan": {"capped": False},
        "skipped": True,
        "reason": reason,
        "limits": list(LIMITS),
    }


def run_spectral_skill_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve the forecast by the band of the target coefficient, per recording.

    Args:
        context: The analysis context, read for the per-sample table and the per-channel vectors.
        eval_config: The validated block, for the bootstrap settings.
        output_dir: The results directory. Both channel maps are read from its root and this
            analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys, the per-band rows, the coverage counts, the recomposition check, the
        three limits and the paths written -- or a skip when no band axis could be resolved.

    Raises:
        ValueError: If the persisted map and the collected vectors describe different channel axes.
            See :func:`band_positions` for why that is a raise rather than a truncation.
    """
    collection = context.collection
    per_sample = collection.per_sample
    kept, declared, reason = read_channel_maps(output_dir)
    if kept is None:
        return skip_record(reason)

    gap = _vector(collection, GAP_VECTOR, len(per_sample))
    if gap is None:
        return skip_record(
            f"the collection carries no usable {GAP_VECTOR!r} vector in the per-sample table's "
            f"row order, so no per-channel readout can be placed on the frequency axis"
        )
    errors: Dict[str, np.ndarray] = {}
    for branch, name in ERROR_VECTORS:
        values = _vector(collection, name, len(per_sample))
        if values is not None:
            errors[branch] = values

    groups = band_positions(kept, gap.shape[1])
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    bands = band_frame(per_sample, gap, errors, groups)
    value_columns = [
        name for name in bands.columns
        if name.startswith("pred_gap_") or name.startswith("sq_error_")
    ]
    # ``pred_gap`` itself comes along so the recomposition below compares two columns of one frame
    # rather than a band sum against a number reduced somewhere else; the block score comes along
    # because it is the magnitude that check's tolerance is scaled by.
    carried = {
        name: finite_column(per_sample, name)
        for name in (TOTAL_COLUMN, RECOMPOSITION_SCALE_COLUMN)
    }
    per_guid = per_recording_means(
        bands.assign(**carried), value_columns + list(carried),
    )
    per_guid.to_csv(directory / PER_RECORDING_FILENAME)

    resamples = int(eval_config.get("bootstrap_resamples", 2000))
    seed = int(eval_config.get("seed", 0))
    rows = build_band_rows(per_guid, groups, resamples=resamples, seed=seed)
    pd.DataFrame(rows).to_csv(directory / BAND_FILENAME, index=False)
    channels = channel_profile(per_sample, kept, gap, errors)
    channels.to_csv(directory / CHANNEL_FILENAME, index=False)

    figure_name = str(
        figures.render_figure(
            build_band_figure(per_guid, rows), directory / BAND_FIGURE
        ).name
    )
    return {
        "n_samples": scored_sample_count(per_sample, TOTAL_COLUMN),
        "composition": {"n_recordings": int(len(per_guid))},
        "plan": {"capped": False, "bootstrap_resamples": resamples, "seed": seed},
        "skipped": False,
        "bands": rows,
        # Flat, finite scalars only: this is what the binding's headline registry digs into.
        "headline": band_headline(rows),
        "coverage": coverage_counts(kept, declared, groups),
        "recomposition": recomposition(per_guid, groups),
        "branches_scored": sorted(errors),
        "band_hz_ranges": {
            name: list(edges) for name, edges in shared.CLINICAL_BANDS.items()
        },
        "limits": list(LIMITS),
        # The quartiles behind each band row's mean, because these distributions are routinely
        # skewed and a mean with an interval describes a symmetric one.
        "gap_distribution": [
            describe(finite_column(per_guid, f"pred_gap_{row['band']}"), name=str(row["band"]))
            for row in rows
        ],
        "grouped_frames": [
            grouped_frame_entry(
                ANALYSIS_DIRNAME,
                PER_RECORDING_FILENAME,
                [f"pred_gap_{band}" for band in groups],
            )
        ],
        "files": [PER_RECORDING_FILENAME, BAND_FILENAME, CHANNEL_FILENAME, figure_name],
    }
