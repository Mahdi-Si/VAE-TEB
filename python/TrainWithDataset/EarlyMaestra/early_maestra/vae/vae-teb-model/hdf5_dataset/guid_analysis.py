"""
GUID-level analysis pipeline for HDF5 dataset creation.

Provides visibility into per-GUID segment acceptance/rejection during dataset
creation, coverage/gap analysis, and diagnostic plots.

Usage:
    # Standalone (post-hoc from existing HDF5):
    from guid_analysis import run_guid_analysis
    results = run_guid_analysis("path/to/dataset.hdf5")

    # Integrated (during dataset creation with tracking data):
    results = run_guid_analysis("path/to/dataset.hdf5", guid_tracking=tracking_dict)
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Journal-quality matplotlib defaults
# ---------------------------------------------------------------------------
_JOURNAL_RC = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.4,
    'ytick.major.width': 0.4,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'grid.linewidth': 0.3,
    'grid.alpha': 0.4,
    'lines.linewidth': 0.8,
    'patch.linewidth': 0.4,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GuidTrackingEntry:
    """Per-GUID tracking data collected during dataset creation."""
    all_domain_starts: List[float] = field(default_factory=list)
    included_domain_starts: List[float] = field(default_factory=list)
    skipped_low_weight: List[float] = field(default_factory=list)
    skipped_flat_region: List[float] = field(default_factory=list)
    skipped_scatter_failed: List[float] = field(default_factory=list)
    skipped_duplicate: List[float] = field(default_factory=list)
    error: bool = False
    error_msg: Optional[str] = None


@dataclass
class GuidCoverageResult:
    """Coverage analysis result for a single GUID."""
    guid: str
    total_segments: int
    included_segments: int
    skipped_low_weight: int
    skipped_flat_region: int
    skipped_scatter_failed: int
    skipped_duplicate: int
    total_extent_sec: float
    covered_sec: float
    missing_pct: float
    n_gaps: int
    gap_durations_sec: List[float]
    longest_gap_sec: float
    merged_intervals: List[Tuple[float, float]]
    extent_start_sec: float = 0.0
    extent_end_sec: float = 0.0
    n_post_delivery_all: int = 0
    n_post_delivery_included: int = 0
    error: bool = False
    error_msg: Optional[str] = None


# ---------------------------------------------------------------------------
# HDF5 reader
# ---------------------------------------------------------------------------

def _load_guid_epoch_data_from_hdf5(
    hdf5_path: str,
) -> Dict[str, List[float]]:
    """Read guid and epoch arrays from an HDF5 file.

    Returns:
        Mapping guid -> sorted list of epoch (domain_start) values.
    """
    guid_epochs: Dict[str, List[float]] = {}
    with h5py.File(hdf5_path, 'r') as f:
        if 'guid' not in f or 'epoch' not in f:
            logger.warning("HDF5 file missing 'guid' or 'epoch' datasets.")
            return guid_epochs
        guids = f['guid'][()]
        epochs = f['epoch'][()]
        for g, e in zip(guids, epochs):
            g_str = g.decode('utf-8') if isinstance(g, bytes) else str(g)
            guid_epochs.setdefault(g_str, []).append(float(e))
    # Sort epochs per guid
    for g in guid_epochs:
        guid_epochs[g].sort()
    return guid_epochs


# ---------------------------------------------------------------------------
# Interval merging & coverage
# ---------------------------------------------------------------------------

def _merge_intervals(
    intervals: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Merge overlapping or touching intervals. Input need not be sorted."""
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_iv[0]]
    for start, end in sorted_iv[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _compute_coverage_for_guid(
    guid: str,
    included_starts: List[float],
    all_starts: Optional[List[float]],
    segment_duration_sec: float,
    skipped_low_weight: Optional[List[float]] = None,
    skipped_flat_region: Optional[List[float]] = None,
    skipped_scatter_failed: Optional[List[float]] = None,
    skipped_duplicate: Optional[List[float]] = None,
    error: bool = False,
    error_msg: Optional[str] = None,
) -> GuidCoverageResult:
    """Compute coverage statistics for a single GUID."""
    n_skipped_lw = len(skipped_low_weight) if skipped_low_weight else 0
    n_skipped_flat = len(skipped_flat_region) if skipped_flat_region else 0
    n_skipped_scatter = len(skipped_scatter_failed) if skipped_scatter_failed else 0
    n_skipped_dup = len(skipped_duplicate) if skipped_duplicate else 0

    # Count post-delivery segments (domain_start >= 0 means segment starts at or after birth)
    n_post_all = sum(1 for s in all_starts if s >= 0) if all_starts else 0
    n_post_included = sum(1 for s in included_starts if s >= 0)

    if error:
        return GuidCoverageResult(
            guid=guid,
            total_segments=len(all_starts) if all_starts else 0,
            included_segments=0,
            skipped_low_weight=n_skipped_lw,
            skipped_flat_region=n_skipped_flat,
            skipped_scatter_failed=n_skipped_scatter,
            skipped_duplicate=n_skipped_dup,
            total_extent_sec=0.0,
            covered_sec=0.0,
            missing_pct=100.0,
            n_gaps=0,
            gap_durations_sec=[],
            longest_gap_sec=0.0,
            merged_intervals=[],
            n_post_delivery_all=n_post_all,
            n_post_delivery_included=n_post_included,
            error=True,
            error_msg=error_msg,
        )

    # Determine the reference set for total extent
    ref_starts = all_starts if all_starts else included_starts
    if not ref_starts:
        return GuidCoverageResult(
            guid=guid,
            total_segments=0,
            included_segments=len(included_starts),
            skipped_low_weight=n_skipped_lw,
            skipped_flat_region=n_skipped_flat,
            skipped_scatter_failed=n_skipped_scatter,
            skipped_duplicate=n_skipped_dup,
            total_extent_sec=0.0,
            covered_sec=0.0,
            missing_pct=100.0,
            n_gaps=0,
            gap_durations_sec=[],
            longest_gap_sec=0.0,
            merged_intervals=[],
        )

    total_segments = len(all_starts) if all_starts else len(included_starts)
    min_start = min(ref_starts)
    max_start = max(ref_starts)
    total_extent = max_start + segment_duration_sec - min_start
    if total_extent <= 0:
        total_extent = segment_duration_sec  # single segment

    extent_start = min_start
    extent_end = max_start + segment_duration_sec

    # Build and merge included intervals
    if not included_starts:
        return GuidCoverageResult(
            guid=guid,
            total_segments=total_segments,
            included_segments=0,
            skipped_low_weight=n_skipped_lw,
            skipped_flat_region=n_skipped_flat,
            skipped_scatter_failed=n_skipped_scatter,
            skipped_duplicate=n_skipped_dup,
            total_extent_sec=total_extent,
            covered_sec=0.0,
            missing_pct=100.0,
            n_gaps=0,
            gap_durations_sec=[],
            longest_gap_sec=0.0,
            merged_intervals=[],
            extent_start_sec=extent_start,
            extent_end_sec=extent_end,
            n_post_delivery_all=n_post_all,
            n_post_delivery_included=n_post_included,
        )

    intervals = [(s, s + segment_duration_sec) for s in included_starts]
    merged = _merge_intervals(intervals)
    covered = sum(end - start for start, end in merged)

    # Compute gaps within total extent (extent_start/extent_end already set above)
    gaps: List[float] = []

    # Gap before first merged interval
    if merged[0][0] > extent_start:
        gaps.append(merged[0][0] - extent_start)
    # Gaps between merged intervals
    for i in range(1, len(merged)):
        gap = merged[i][0] - merged[i - 1][1]
        if gap > 0:
            gaps.append(gap)
    # Gap after last merged interval
    if merged[-1][1] < extent_end:
        gaps.append(extent_end - merged[-1][1])

    missing_pct = (total_extent - covered) / total_extent * 100.0 if total_extent > 0 else 0.0
    missing_pct = max(0.0, min(100.0, missing_pct))

    return GuidCoverageResult(
        guid=guid,
        total_segments=total_segments,
        included_segments=len(included_starts),
        skipped_low_weight=n_skipped_lw,
        skipped_flat_region=n_skipped_flat,
        skipped_scatter_failed=n_skipped_scatter,
        skipped_duplicate=n_skipped_dup,
        total_extent_sec=total_extent,
        covered_sec=covered,
        missing_pct=missing_pct,
        n_gaps=len(gaps),
        gap_durations_sec=gaps,
        longest_gap_sec=max(gaps, default=0.0),
        merged_intervals=merged,
        extent_start_sec=extent_start,
        extent_end_sec=extent_end,
        n_post_delivery_all=n_post_all,
        n_post_delivery_included=n_post_included,
    )


def _compute_all_coverage(
    hdf5_guid_epochs: Dict[str, List[float]],
    guid_tracking: Optional[Dict[str, GuidTrackingEntry]],
    segment_duration_sec: float,
) -> List[GuidCoverageResult]:
    """Compute coverage for every GUID, merging HDF5 data with tracking data."""
    results: List[GuidCoverageResult] = []

    # Collect all known GUIDs from both sources
    all_guids = set(hdf5_guid_epochs.keys())
    if guid_tracking:
        all_guids |= set(guid_tracking.keys())

    for guid in sorted(all_guids):
        hdf5_starts = hdf5_guid_epochs.get(guid, [])
        if guid_tracking and guid in guid_tracking:
            entry = guid_tracking[guid]
            result = _compute_coverage_for_guid(
                guid=guid,
                included_starts=hdf5_starts if hdf5_starts else entry.included_domain_starts,
                all_starts=entry.all_domain_starts or None,
                segment_duration_sec=segment_duration_sec,
                skipped_low_weight=entry.skipped_low_weight,
                skipped_flat_region=entry.skipped_flat_region,
                skipped_scatter_failed=entry.skipped_scatter_failed,
                skipped_duplicate=entry.skipped_duplicate,
                error=entry.error,
                error_msg=entry.error_msg,
            )
        else:
            result = _compute_coverage_for_guid(
                guid=guid,
                included_starts=hdf5_starts,
                all_starts=None,
                segment_duration_sec=segment_duration_sec,
            )
        results.append(result)

    # Sort by total extent descending
    results.sort(key=lambda r: r.total_extent_sec, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _generate_summary_stats(
    coverage_results: List[GuidCoverageResult],
    guid_tracking: Optional[Dict[str, GuidTrackingEntry]],
) -> Dict[str, Any]:
    """Compute aggregate summary statistics."""
    non_error = [r for r in coverage_results if not r.error]
    errored = [r for r in coverage_results if r.error]
    fully_rejected = [r for r in non_error if r.included_segments == 0]

    total_included = sum(r.included_segments for r in non_error)
    total_all_segments = sum(r.total_segments for r in non_error)
    total_skipped_lw = sum(r.skipped_low_weight for r in non_error)
    total_skipped_flat = sum(r.skipped_flat_region for r in non_error)
    total_skipped_scatter = sum(r.skipped_scatter_failed for r in non_error)
    total_skipped_dup = sum(r.skipped_duplicate for r in non_error)
    total_post_delivery_all = sum(r.n_post_delivery_all for r in non_error)
    total_post_delivery_included = sum(r.n_post_delivery_included for r in non_error)
    guids_with_post_delivery = [r.guid for r in non_error if r.n_post_delivery_included > 0]

    missing_pcts = [r.missing_pct for r in non_error if r.total_extent_sec > 0]
    all_gaps = []
    for r in non_error:
        all_gaps.extend(r.gap_durations_sec)

    gap_stats = {}
    if all_gaps:
        gaps_arr = np.array(all_gaps)
        gap_stats = {
            'count': len(all_gaps),
            'mean_sec': float(np.mean(gaps_arr)),
            'median_sec': float(np.median(gaps_arr)),
            'max_sec': float(np.max(gaps_arr)),
            'min_sec': float(np.min(gaps_arr)),
            'p25_sec': float(np.percentile(gaps_arr, 25)),
            'p75_sec': float(np.percentile(gaps_arr, 75)),
            'p95_sec': float(np.percentile(gaps_arr, 95)),
            'mean_min': float(np.mean(gaps_arr) / 60),
            'median_min': float(np.median(gaps_arr) / 60),
            'max_min': float(np.max(gaps_arr) / 60),
        }

    missing_stats = {}
    if missing_pcts:
        mp = np.array(missing_pcts)
        missing_stats = {
            'mean': float(np.mean(mp)),
            'median': float(np.median(mp)),
            'max': float(np.max(mp)),
            'min': float(np.min(mp)),
            'p25': float(np.percentile(mp, 25)),
            'p75': float(np.percentile(mp, 75)),
            'p95': float(np.percentile(mp, 95)),
        }

    # Segment accounting: total = included + skipped_lw + skipped_flat + skipped_scatter + skipped_dup + unaccounted
    total_accounted = total_included + total_skipped_lw + total_skipped_flat + total_skipped_scatter + total_skipped_dup
    unaccounted = max(0, total_all_segments - total_accounted)

    return {
        'total_guids': len(coverage_results),
        'guids_with_data': len(non_error) - len(fully_rejected),
        'fully_rejected_guids': len(fully_rejected),
        'errored_guids': len(errored),
        'total_segments_from_prepare_data': total_all_segments,
        'total_included_segments': total_included,
        'total_skipped_low_weight': total_skipped_lw,
        'total_skipped_flat_region': total_skipped_flat,
        'total_skipped_scatter_failed': total_skipped_scatter,
        'total_skipped_duplicate': total_skipped_dup,
        'total_unaccounted_segments': unaccounted,
        'has_tracking_data': guid_tracking is not None,
        'total_post_delivery_all': total_post_delivery_all,
        'total_post_delivery_included': total_post_delivery_included,
        'guids_with_post_delivery': guids_with_post_delivery,
        'gap_stats': gap_stats,
        'missing_pct_stats': missing_stats,
    }


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _write_json_report(
    output_dir: str,
    summary: Dict[str, Any],
    coverage_results: List[GuidCoverageResult],
) -> str:
    path = os.path.join(output_dir, 'guid_analysis_data.json')
    per_guid = []
    for r in coverage_results:
        entry = {
            'guid': r.guid,
            'total_segments': r.total_segments,
            'included_segments': r.included_segments,
            'skipped_low_weight': r.skipped_low_weight,
            'skipped_flat_region': r.skipped_flat_region,
            'total_extent_sec': round(r.total_extent_sec, 2),
            'covered_sec': round(r.covered_sec, 2),
            'missing_pct': round(r.missing_pct, 2),
            'n_gaps': r.n_gaps,
            'longest_gap_sec': round(r.longest_gap_sec, 2),
            'skipped_scatter_failed': r.skipped_scatter_failed,
            'skipped_duplicate': r.skipped_duplicate,
            'extent_start_sec': round(r.extent_start_sec, 2),
            'extent_end_sec': round(r.extent_end_sec, 2),
            'n_post_delivery_all': r.n_post_delivery_all,
            'n_post_delivery_included': r.n_post_delivery_included,
            'error': r.error,
        }
        if r.error_msg:
            entry['error_msg'] = r.error_msg
        per_guid.append(entry)

    report = {
        'generated_at': datetime.now().isoformat(),
        'summary': summary,
        'per_guid': per_guid,
    }
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    return path


def _write_markdown_report(
    output_dir: str,
    summary: Dict[str, Any],
    coverage_results: List[GuidCoverageResult],
) -> str:
    path = os.path.join(output_dir, 'guid_analysis_report.md')
    lines = []
    lines.append('# GUID Analysis Report')
    lines.append(f'\nGenerated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    # Summary table
    lines.append('\n## Summary')
    lines.append('')
    lines.append('| Metric | Value |')
    lines.append('|--------|-------|')
    lines.append(f'| Total GUIDs | {summary["total_guids"]} |')
    lines.append(f'| GUIDs with included data | {summary["guids_with_data"]} |')
    lines.append(f'| Fully rejected GUIDs | {summary["fully_rejected_guids"]} |')
    lines.append(f'| Errored GUIDs | {summary["errored_guids"]} |')
    lines.append(f'| Total segments (from prepare_data) | {summary["total_segments_from_prepare_data"]} |')
    lines.append(f'| Included segments | {summary["total_included_segments"]} |')
    if summary['has_tracking_data']:
        lines.append(f'| Skipped (low weight) | {summary["total_skipped_low_weight"]} |')
        lines.append(f'| Skipped (flat region) | {summary["total_skipped_flat_region"]} |')
        lines.append(f'| Skipped (scatter failed) | {summary["total_skipped_scatter_failed"]} |')
        lines.append(f'| Skipped (duplicate) | {summary["total_skipped_duplicate"]} |')
        if summary['total_unaccounted_segments'] > 0:
            lines.append(f'| **Unaccounted segments** | **{summary["total_unaccounted_segments"]}** |')

    # Post-delivery diagnostics (domain_start >= 0)
    n_post_all = summary.get('total_post_delivery_all', 0)
    n_post_inc = summary.get('total_post_delivery_included', 0)
    guids_post = summary.get('guids_with_post_delivery', [])
    if n_post_all > 0 or n_post_inc > 0:
        lines.append('\n## WARNING: Post-Delivery Segments Detected')
        lines.append('')
        lines.append('Segments with `domain_start >= 0` start at or after delivery.')
        lines.append('These come from equalization padding in `split_long()` pushing')
        lines.append('the last segment past t=0, or from recordings that extend past delivery.')
        lines.append('')
        lines.append('| Metric | Value |')
        lines.append('|--------|-------|')
        lines.append(f'| Post-delivery segments (from prepare_data) | {n_post_all} |')
        lines.append(f'| Post-delivery segments (included in HDF5) | {n_post_inc} |')
        lines.append(f'| GUIDs affected | {len(guids_post)} |')
        if guids_post:
            lines.append('')
            lines.append('Affected GUIDs:')
            for g in guids_post:
                lines.append(f'- `{g}`')

    # Coverage statistics
    if summary['missing_pct_stats']:
        ms = summary['missing_pct_stats']
        lines.append('\n## Coverage Statistics (Missing %)')
        lines.append('')
        lines.append('| Statistic | Value |')
        lines.append('|-----------|-------|')
        lines.append(f'| Mean | {ms["mean"]:.1f}% |')
        lines.append(f'| Median | {ms["median"]:.1f}% |')
        lines.append(f'| Min | {ms["min"]:.1f}% |')
        lines.append(f'| Max | {ms["max"]:.1f}% |')
        lines.append(f'| P25 | {ms["p25"]:.1f}% |')
        lines.append(f'| P75 | {ms["p75"]:.1f}% |')
        lines.append(f'| P95 | {ms["p95"]:.1f}% |')

    # Gap statistics
    if summary['gap_stats']:
        gs = summary['gap_stats']
        lines.append('\n## Gap Statistics')
        lines.append('')
        lines.append('| Statistic | Value |')
        lines.append('|-----------|-------|')
        lines.append(f'| Total gaps | {gs["count"]} |')
        lines.append(f'| Mean gap | {gs["mean_min"]:.1f} min ({gs["mean_sec"]:.0f} sec) |')
        lines.append(f'| Median gap | {gs["median_min"]:.1f} min ({gs["median_sec"]:.0f} sec) |')
        lines.append(f'| Max gap | {gs["max_min"]:.1f} min ({gs["max_sec"]:.0f} sec) |')
        lines.append(f'| P25 | {gs["p25_sec"]:.0f} sec |')
        lines.append(f'| P75 | {gs["p75_sec"]:.0f} sec |')
        lines.append(f'| P95 | {gs["p95_sec"]:.0f} sec |')

    # Fully rejected GUIDs
    rejected = [r for r in coverage_results if not r.error and r.included_segments == 0]
    if rejected:
        lines.append(f'\n## Fully Rejected GUIDs ({len(rejected)})')
        lines.append('')
        for r in rejected:
            lines.append(f'- `{r.guid}` -- {r.total_segments} total segments, '
                         f'{r.skipped_low_weight} low weight, {r.skipped_flat_region} flat region, '
                         f'{r.skipped_duplicate} duplicate')

    # Errored GUIDs
    errored = [r for r in coverage_results if r.error]
    if errored:
        lines.append(f'\n## Errored GUIDs ({len(errored)})')
        lines.append('')
        for r in errored:
            lines.append(f'- `{r.guid}` -- {r.error_msg or "unknown error"}')

    # Per-GUID table
    non_error = [r for r in coverage_results if not r.error]
    if non_error:
        lines.append('\n## Per-GUID Detail')
        lines.append('')
        lines.append('| GUID | Total | Included | Skip LW | Skip Flat | Skip Scatter | Skip Dup | Post-Deliv | Missing % | Longest Gap (min) |')
        lines.append('|------|-------|----------|---------|-----------|--------------|----------|------------|-----------|-------------------|')
        for r in non_error:
            longest_gap_min = r.longest_gap_sec / 60.0
            post_flag = f'**{r.n_post_delivery_included}**' if r.n_post_delivery_included > 0 else '0'
            lines.append(
                f'| `{r.guid}` | {r.total_segments} | {r.included_segments} '
                f'| {r.skipped_low_weight} | {r.skipped_flat_region} '
                f'| {r.skipped_scatter_failed} | {r.skipped_duplicate} | {post_flag} '
                f'| {r.missing_pct:.1f}% | {longest_gap_min:.1f} |'
            )

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_coverage_timeline(
    coverage_results: List[GuidCoverageResult],
    output_dir: str,
    max_guids: int = 100,
    dpi: int = 300,
) -> Optional[str]:
    """Horizontal broken-bar chart: blue=covered, red=gap per GUID."""
    non_error = [r for r in coverage_results if not r.error and r.total_extent_sec > 0]
    if not non_error:
        return None

    # Already sorted by total_extent desc from _compute_all_coverage
    plotted = non_error[:max_guids]
    n = len(plotted)

    fig_height = max(3, n * 0.22 + 0.8)
    with plt.rc_context(_JOURNAL_RC):
        fig, ax = plt.subplots(figsize=(7, fig_height))

        for idx, r in enumerate(reversed(plotted)):  # reverse so largest at top
            y = idx
            if r.total_extent_sec > 0:
                extent_start = r.extent_start_sec
                ax.barh(y, r.total_extent_sec / 60.0, left=extent_start / 60.0,
                        height=0.7, color='#f4cccc', edgecolor='#e06666', linewidth=0.3)
                for start, end in r.merged_intervals:
                    ax.barh(y, (end - start) / 60.0, left=start / 60.0,
                            height=0.7, color='#6fa8dc', edgecolor='#3d85c6', linewidth=0.3)

        labels = [r.guid[:20] for r in reversed(plotted)]
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=max(4.5, min(6.5, 120 / n)))
        ax.set_xlabel('Time (min, relative to delivery)')
        ax.set_title(f'Coverage Timeline ({n} GUIDs)')
        ax.axvline(0, color='#333333', linestyle='--', linewidth=0.6, alpha=0.7)
        ax.legend(handles=[
            Patch(facecolor='#6fa8dc', edgecolor='#3d85c6', label='Covered', linewidth=0.3),
            Patch(facecolor='#f4cccc', edgecolor='#e06666', label='Gap / Missing', linewidth=0.3),
        ], loc='lower right', frameon=True, edgecolor='#cccccc', framealpha=0.9)
        ax.grid(axis='x', linewidth=0.3, alpha=0.4)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

        path = os.path.join(output_dir, 'coverage_timeline.png')
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
    return path


def _plot_gap_distribution(
    coverage_results: List[GuidCoverageResult],
    output_dir: str,
    dpi: int = 300,
) -> Optional[str]:
    """Histogram of gap durations in minutes."""
    all_gaps = []
    for r in coverage_results:
        if not r.error:
            all_gaps.extend(r.gap_durations_sec)

    if not all_gaps:
        return None

    gaps_min = np.array(all_gaps) / 60.0

    with plt.rc_context(_JOURNAL_RC):
        fig, ax = plt.subplots(figsize=(3.5, 2.4))
        n_bins = min(50, max(10, len(gaps_min) // 5))
        ax.hist(gaps_min, bins=n_bins, color='#6fa8dc', edgecolor='#3d85c6',
                linewidth=0.3, alpha=0.85)

        mean_val = float(np.mean(gaps_min))
        median_val = float(np.median(gaps_min))
        max_val = float(np.max(gaps_min))

        ax.axvline(mean_val, color='#c0392b', linestyle='--', linewidth=0.7,
                   label=f'Mean: {mean_val:.1f} min')
        ax.axvline(median_val, color='#e67e22', linestyle='--', linewidth=0.7,
                   label=f'Median: {median_val:.1f} min')

        ax.text(0.97, 0.95,
                f'n = {len(gaps_min)}\nMean = {mean_val:.1f} min\n'
                f'Median = {median_val:.1f} min\nMax = {max_val:.1f} min',
                transform=ax.transAxes, fontsize=6, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fafafa',
                          edgecolor='#cccccc', linewidth=0.4, alpha=0.9))

        ax.set_xlabel('Gap Duration (min)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Gap Durations')
        ax.legend(frameon=True, edgecolor='#cccccc', framealpha=0.9)
        ax.grid(axis='y', linewidth=0.3, alpha=0.4)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

        path = os.path.join(output_dir, 'gap_distribution.png')
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
    return path


def _plot_missing_data_percentage(
    coverage_results: List[GuidCoverageResult],
    output_dir: str,
    max_guids: int = 50,
    dpi: int = 300,
) -> Optional[str]:
    """Sorted horizontal bar chart of missing data % per GUID."""
    non_error = [r for r in coverage_results if not r.error and r.total_extent_sec > 0]
    if not non_error:
        return None

    sorted_results = sorted(non_error, key=lambda r: r.missing_pct, reverse=True)
    plotted = sorted_results[:max_guids]
    n = len(plotted)

    fig_height = max(3, n * 0.22 + 0.8)
    with plt.rc_context(_JOURNAL_RC):
        fig, ax = plt.subplots(figsize=(4.5, fig_height))

        guids = [r.guid[:20] for r in plotted]
        pcts = [r.missing_pct for r in plotted]

        # Color gradient green->red (muted tones)
        colors = []
        for p in pcts:
            t = p / 100.0
            r_ch = min(1.0, 0.3 + 0.7 * t)
            g_ch = min(1.0, 0.3 + 0.7 * (1 - t))
            colors.append((r_ch, g_ch, 0.25, 0.75))

        y_pos = range(n)
        ax.barh(y_pos, pcts, color=colors, edgecolor='#888888', linewidth=0.3)

        mean_pct = float(np.mean([r.missing_pct for r in non_error]))
        ax.axvline(mean_pct, color='#2c3e50', linestyle='--', linewidth=0.7,
                   label=f'Mean: {mean_pct:.1f}%')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(guids, fontsize=max(4.5, min(6.5, 120 / n)))
        ax.set_xlabel('Missing Data (%)')
        ax.set_title(f'Missing Data Percentage by GUID (top {n})')
        ax.set_xlim(0, 105)
        ax.legend(frameon=True, edgecolor='#cccccc', framealpha=0.9)
        ax.grid(axis='x', linewidth=0.3, alpha=0.4)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

        path = os.path.join(output_dir, 'missing_data_percentage.png')
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
    return path


def _plot_segment_counts(
    coverage_results: List[GuidCoverageResult],
    output_dir: str,
    has_tracking: bool = False,
    max_guids: int = 50,
    dpi: int = 300,
) -> Optional[str]:
    """Stacked bar chart of segment counts per GUID."""
    non_error = [r for r in coverage_results if not r.error]
    if not non_error:
        return None

    sorted_results = sorted(non_error, key=lambda r: r.total_segments, reverse=True)
    plotted = sorted_results[:max_guids]
    n = len(plotted)

    # Muted palette for stacked categories
    _PAL = {
        'included': ('#6fa8dc', '#3d85c6'),
        'low_weight': ('#f6b26b', '#e69138'),
        'flat_region': ('#e06666', '#cc0000'),
        'scatter': ('#b4a7d6', '#8e7cc3'),
        'duplicate': ('#76d7c4', '#45b39d'),
        'unaccounted': ('#b7b7b7', '#888888'),
    }

    fig_height = max(3, n * 0.22 + 0.8)
    with plt.rc_context(_JOURNAL_RC):
        fig, ax = plt.subplots(figsize=(4.5, fig_height))

        guids = [r.guid[:20] for r in plotted]
        y_pos = np.arange(n)
        edge_lw = 0.3

        if has_tracking:
            included = [r.included_segments for r in plotted]
            skipped_lw = [r.skipped_low_weight for r in plotted]
            skipped_flat = [r.skipped_flat_region for r in plotted]
            skipped_scatter = [r.skipped_scatter_failed for r in plotted]
            skipped_dup = [r.skipped_duplicate for r in plotted]
            unaccounted = [max(0, r.total_segments - r.included_segments
                              - r.skipped_low_weight - r.skipped_flat_region
                              - r.skipped_scatter_failed - r.skipped_duplicate)
                          for r in plotted]

            ax.barh(y_pos, included, color=_PAL['included'][0],
                    edgecolor=_PAL['included'][1], linewidth=edge_lw, label='Included')
            left1 = list(included)
            ax.barh(y_pos, skipped_lw, left=left1, color=_PAL['low_weight'][0],
                    edgecolor=_PAL['low_weight'][1], linewidth=edge_lw, label='Low Weight')
            left2 = [a + b for a, b in zip(left1, skipped_lw)]
            ax.barh(y_pos, skipped_flat, left=left2, color=_PAL['flat_region'][0],
                    edgecolor=_PAL['flat_region'][1], linewidth=edge_lw, label='Flat Region')
            left3 = [a + b for a, b in zip(left2, skipped_flat)]
            ax.barh(y_pos, skipped_scatter, left=left3, color=_PAL['scatter'][0],
                    edgecolor=_PAL['scatter'][1], linewidth=edge_lw, label='Scatter Failed')
            left4 = [a + b for a, b in zip(left3, skipped_scatter)]
            ax.barh(y_pos, skipped_dup, left=left4, color=_PAL['duplicate'][0],
                    edgecolor=_PAL['duplicate'][1], linewidth=edge_lw, label='Duplicate')
            if any(u > 0 for u in unaccounted):
                left5 = [a + b for a, b in zip(left4, skipped_dup)]
                ax.barh(y_pos, unaccounted, left=left5, color=_PAL['unaccounted'][0],
                        edgecolor=_PAL['unaccounted'][1], linewidth=edge_lw,
                        label='Unaccounted')
        else:
            included = [r.included_segments for r in plotted]
            ax.barh(y_pos, included, color=_PAL['included'][0],
                    edgecolor=_PAL['included'][1], linewidth=edge_lw, label='Included')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(guids, fontsize=max(4.5, min(6.5, 120 / n)))
        ax.set_xlabel('Segment Count')
        ax.set_title(f'Segment Counts by GUID (top {n})')
        ax.legend(loc='lower right', frameon=True, edgecolor='#cccccc', framealpha=0.9)
        ax.grid(axis='x', linewidth=0.3, alpha=0.4)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

        path = os.path.join(output_dir, 'segment_counts.png')
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
    return path


def _plot_rejection_reasons(
    summary: Dict[str, Any],
    output_dir: str,
    dpi: int = 300,
) -> Optional[str]:
    """Pie chart of rejection reasons. Only generated with tracking data."""
    if not summary.get('has_tracking_data'):
        return None

    included = summary['total_included_segments']
    lw = summary['total_skipped_low_weight']
    flat = summary['total_skipped_flat_region']
    scatter = summary.get('total_skipped_scatter_failed', 0)
    dup = summary.get('total_skipped_duplicate', 0)
    errored = summary['errored_guids']

    labels = []
    sizes = []
    colors = []

    # Muted palette matching the segment counts chart
    if included > 0:
        labels.append(f'Included ({included})')
        sizes.append(included)
        colors.append('#6fa8dc')
    if lw > 0:
        labels.append(f'Low Weight ({lw})')
        sizes.append(lw)
        colors.append('#f6b26b')
    if flat > 0:
        labels.append(f'Flat Region ({flat})')
        sizes.append(flat)
        colors.append('#e06666')
    if scatter > 0:
        labels.append(f'Scatter Failed ({scatter})')
        sizes.append(scatter)
        colors.append('#b4a7d6')
    if dup > 0:
        labels.append(f'Duplicate ({dup})')
        sizes.append(dup)
        colors.append('#76d7c4')
    if errored > 0:
        labels.append(f'Errored GUIDs ({errored})')
        sizes.append(errored)
        colors.append('#b7b7b7')

    if not sizes:
        return None

    with plt.rc_context(_JOURNAL_RC):
        fig, ax = plt.subplots(figsize=(3.8, 3))
        _, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, pctdistance=0.82,
            wedgeprops=dict(edgecolor='white', linewidth=0.8))
        for t in texts:
            t.set_fontsize(6.5)
        for t in autotexts:
            t.set_fontsize(6)
        ax.set_title('Segment Rejection Reasons')

        path = os.path.join(output_dir, 'rejection_reasons.png')
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Interactive signal strip plot (plotly)
# ---------------------------------------------------------------------------

# Color scheme for segment status
_STRIP_COLORS = {
    'included':       {'line': 'rgb(30,30,30)',    'band': None},
    'low_weight':     {'line': 'rgb(200,0,0)',     'band': 'rgba(255,180,180,0.35)'},
    'flat_region':    {'line': 'rgb(200,100,0)',   'band': 'rgba(255,200,150,0.35)'},
    'scatter_failed': {'line': 'rgb(140,0,140)',   'band': 'rgba(220,180,255,0.35)'},
    'pending':        {'line': 'rgb(150,150,150)', 'band': 'rgba(200,200,200,0.2)'},
}

_STATUS_LABELS = {
    'included':       'Included',
    'low_weight':     'Rejected: Low Weight',
    'flat_region':    'Rejected: Flat Region',
    'scatter_failed': 'Rejected: Scatter Failed',
    'pending':        'Pending (unprocessed)',
}


def plot_guid_signal_strip(
    guid: str,
    fhr: np.ndarray,
    up: np.ndarray,
    domain_starts: List[float],
    segment_status: List[str],
    output_dir: str,
    sampling_rate: float = 4.0,
) -> Optional[str]:
    """Generate an interactive plotly signal strip for a single GUID.

    Creates a clinical-strip-style HTML plot with FHR (top) and UP (bottom),
    where rejected segments are colored by their filtering status.
    Overlapping segments are trimmed so each time point shows only once.

    Args:
        guid: GUID identifier string.
        fhr: Signal array of shape ``(n_segments, 5760)``.
        up: Signal array of shape ``(n_segments, 5760)``.
        domain_starts: Domain start (seconds relative to delivery) per segment.
        segment_status: Status string per segment — one of
            ``'included'``, ``'low_weight'``, ``'flat_region'``,
            ``'scatter_failed'``, ``'pending'``.
        output_dir: Directory where the HTML file will be saved.
        sampling_rate: Sampling rate in Hz (default 4.0).

    Returns:
        Path to the saved HTML file, or ``None`` on failure.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.warning("plotly not installed — skipping signal strip plot for %s", guid)
        return None

    n_segments = fhr.shape[0]
    if n_segments == 0:
        return None

    seg_samples = fhr.shape[1]
    segment_duration_sec = seg_samples / sampling_rate  # 1440 sec
    downsample = 4  # 5760 → 1440 points per segment

    # --- Compute per-segment display boundaries to eliminate overlap ---
    # Sort indices by domain_start; for each consecutive pair that overlaps,
    # place a boundary at the midpoint of the overlap region.  Each segment
    # then displays only the samples between its left and right boundaries.
    #
    # Example (50% overlap, dur=1440s):
    #   Seg A ds=-7200  covers [-7200, -5760]
    #   Seg B ds=-6480  covers [-6480, -5040]
    #   Overlap = [-6480, -5760] (720s).  Midpoint = -6120.
    #   Seg A shows [-7200, -6120] → samples 0..4320   (75%)
    #   Seg B shows [-6120, -5040] → samples 1440..5760 (75%)
    sorted_idx = sorted(range(n_segments), key=lambda i: domain_starts[i])

    # Boundary between each consecutive pair (None = no overlap, show fully)
    boundaries_sec = []  # length = n_segments - 1
    for pos in range(len(sorted_idx) - 1):
        si_cur = sorted_idx[pos]
        si_nxt = sorted_idx[pos + 1]
        ds_cur = domain_starts[si_cur]
        ds_nxt = domain_starts[si_nxt]
        overlap_sec = (ds_cur + segment_duration_sec) - ds_nxt
        if overlap_sec > 0:
            # Midpoint of the overlap region in absolute time
            boundary = ds_nxt + overlap_sec / 2.0
        else:
            boundary = None
        boundaries_sec.append(boundary)

    # Convert to per-segment (start_sample, end_sample)
    seg_display = {}  # seg index -> (start_sample, end_sample)
    for pos, si in enumerate(sorted_idx):
        ds = domain_starts[si]
        start_sample = 0
        end_sample = seg_samples

        # Left boundary: trim start if previous segment overlaps us
        if pos > 0 and boundaries_sec[pos - 1] is not None:
            left_sec = boundaries_sec[pos - 1] - ds
            start_sample = max(0, int(round(left_sec * sampling_rate)))

        # Right boundary: trim end if we overlap the next segment
        if pos < len(boundaries_sec) and boundaries_sec[pos] is not None:
            right_sec = boundaries_sec[pos] - ds
            end_sample = min(seg_samples, int(round(right_sec * sampling_rate)))

        # Safety: ensure at least 1 sample
        if end_sample <= start_sample:
            end_sample = min(start_sample + downsample, seg_samples)

        seg_display[si] = (start_sample, end_sample)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.6, 0.4],
    )

    legend_shown = set()

    # Plot rejected segments first, then accepted on top
    order = [i for i in range(n_segments) if segment_status[i] != 'included']
    order += [i for i in range(n_segments) if segment_status[i] == 'included']

    for seg_i in order:
        status = segment_status[seg_i]
        colors = _STRIP_COLORS.get(status, _STRIP_COLORS['pending'])
        ds = domain_starts[seg_i]

        # Trim to display boundary then downsample
        start_samp, end_samp = seg_display[seg_i]
        fhr_trimmed = fhr[seg_i, start_samp:end_samp:downsample]
        up_trimmed = up[seg_i, start_samp:end_samp:downsample]
        n_pts = len(fhr_trimmed)
        if n_pts == 0:
            continue

        # Time axis: offset by start_samp into the segment
        t_start_min = (ds + start_samp / sampling_rate) / 60.0
        t_end_min = (ds + end_samp / sampling_rate) / 60.0
        t = np.linspace(t_start_min, t_end_min, n_pts, endpoint=False)

        show_legend = status not in legend_shown
        legend_shown.add(status)

        hover_text = (f"Status: {_STATUS_LABELS.get(status, status)}<br>"
                      f"Epoch: {ds:.0f}s ({ds/60:.1f} min)")

        # FHR trace
        fig.add_trace(
            go.Scattergl(
                x=t, y=fhr_trimmed,
                mode='lines',
                line=dict(color=colors['line'], width=1),
                name=_STATUS_LABELS.get(status, status),
                legendgroup=status,
                showlegend=show_legend,
                hovertemplate=hover_text + '<br>FHR: %{y:.0f} bpm<extra></extra>',
            ),
            row=1, col=1,
        )

        # UP trace
        fig.add_trace(
            go.Scattergl(
                x=t, y=up_trimmed,
                mode='lines',
                line=dict(color=colors['line'], width=1),
                name=_STATUS_LABELS.get(status, status),
                legendgroup=status,
                showlegend=False,
                hovertemplate=hover_text + '<br>UP: %{y:.0f} mmHg<extra></extra>',
            ),
            row=2, col=1,
        )

    # Delivery marker (t=0)
    for row in (1, 2):
        fig.add_vline(
            x=0, line_dash='dash', line_color='red', line_width=1.5,
            annotation_text='Delivery' if row == 1 else None,
            annotation_position='top right' if row == 1 else None,
            row=row, col=1,
        )

    # Count summary
    n_included = sum(1 for s in segment_status if s == 'included')
    n_rejected = n_segments - n_included

    fig.update_layout(
        title=dict(
            text=(f'{guid} — {n_segments} segments '
                  f'({n_included} included, {n_rejected} rejected)'),
            font_size=13,
            x=0.5,
            xanchor='center',
            y=0.98,
            yanchor='top',
        ),
        width=3600,
        height=350,
        margin=dict(l=60, r=30, t=80, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation='h',
            yanchor='top',
            y=1.15,
            xanchor='right',
            x=1.0,
            font_size=10,
        ),
        hovermode='x unified',
    )

    # Axis styling
    fig.update_yaxes(
        range=[50, 210], title_text='FHR (bpm)',
        gridcolor='lightgray', gridwidth=0.5,
        row=1, col=1,
    )
    fig.update_yaxes(
        range=[0, 100], title_text='UP (mmHg)',
        gridcolor='lightgray', gridwidth=0.5,
        row=2, col=1,
    )
    fig.update_xaxes(
        title_text='Time (min, relative to delivery)',
        gridcolor='lightgray', gridwidth=0.5,
        row=2, col=1,
    )
    fig.update_xaxes(gridcolor='lightgray', gridwidth=0.5, row=1, col=1)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'{guid}_signal_strip.html')
    fig.write_html(out_path, include_plotlyjs=True)
    logger.info(f"Signal strip plot saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_guid_analysis(
    hdf5_path: str,
    guid_tracking: Optional[Dict[str, GuidTrackingEntry]] = None,
    segment_duration_sec: float = 1440.0,
    output_dir: Optional[str] = None,
    max_guids_in_timeline_plot: int = 100,
    dpi: int = 300,
) -> Dict[str, Any]:
    """Run full GUID-level analysis and generate reports + plots.

    Args:
        hdf5_path: Path to the HDF5 dataset file.
        guid_tracking: Optional per-GUID tracking data from dataset creation.
            If None, analysis is performed post-hoc using only HDF5 data.
        segment_duration_sec: Duration of each segment in seconds (default 1440 = 5760/4).
        output_dir: Directory for output files. Defaults to ``<hdf5_stem>_guid_analysis/``
            alongside the HDF5 file.
        max_guids_in_timeline_plot: Cap on GUIDs shown in the timeline plot.
        dpi: Resolution for saved plots (default 300).

    Returns:
        Summary statistics dictionary.
    """
    # Determine output directory
    if output_dir is None:
        hdf5_dir = os.path.dirname(os.path.abspath(hdf5_path))
        hdf5_stem = os.path.splitext(os.path.basename(hdf5_path))[0]
        output_dir = os.path.join(hdf5_dir, f'{hdf5_stem}_guid_analysis')
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Running GUID analysis for {hdf5_path}")
    logger.info(f"Output directory: {output_dir}")

    # Load guid/epoch data from HDF5
    hdf5_guid_epochs = _load_guid_epoch_data_from_hdf5(hdf5_path)
    n_hdf5_guids = len(hdf5_guid_epochs)
    n_hdf5_samples = sum(len(v) for v in hdf5_guid_epochs.values())
    logger.info(f"HDF5 contains {n_hdf5_samples} samples from {n_hdf5_guids} GUIDs")

    if n_hdf5_samples == 0 and (guid_tracking is None or len(guid_tracking) == 0):
        logger.warning("No data found. Writing empty report.")
        summary = {
            'total_guids': 0, 'guids_with_data': 0, 'fully_rejected_guids': 0,
            'errored_guids': 0, 'total_segments_from_prepare_data': 0,
            'total_included_segments': 0, 'total_skipped_low_weight': 0,
            'total_skipped_flat_region': 0, 'has_tracking_data': guid_tracking is not None,
            'gap_stats': {}, 'missing_pct_stats': {},
        }
        _write_json_report(output_dir, summary, [])
        _write_markdown_report(output_dir, summary, [])
        return summary

    # Compute coverage
    coverage_results = _compute_all_coverage(
        hdf5_guid_epochs, guid_tracking, segment_duration_sec)

    # Summary stats
    summary = _generate_summary_stats(coverage_results, guid_tracking)
    logger.info(f"Summary: {summary['total_guids']} GUIDs, "
                f"{summary['total_included_segments']} included segments, "
                f"{summary['fully_rejected_guids']} fully rejected, "
                f"{summary['errored_guids']} errored")

    # Write reports
    _write_json_report(output_dir, summary, coverage_results)
    logger.info("JSON report written")
    _write_markdown_report(output_dir, summary, coverage_results)
    logger.info("Markdown report written")

    # Generate plots
    _plot_coverage_timeline(coverage_results, output_dir,
                            max_guids=max_guids_in_timeline_plot, dpi=dpi)
    _plot_gap_distribution(coverage_results, output_dir, dpi=dpi)
    _plot_missing_data_percentage(coverage_results, output_dir, dpi=dpi)
    _plot_segment_counts(coverage_results, output_dir,
                         has_tracking=guid_tracking is not None, dpi=dpi)
    _plot_rejection_reasons(summary, output_dir, dpi=dpi)
    logger.info("All plots generated")

    logger.info(f"GUID analysis complete. Results in {output_dir}")
    return summary
