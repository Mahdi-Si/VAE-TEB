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
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GuidTrackingEntry:
    """Per-GUID tracking data collected during dataset creation."""
    all_domain_starts: List[float] = field(default_factory=list)
    included_domain_starts: List[float] = field(default_factory=list)
    skipped_low_weight: List[float] = field(default_factory=list)
    skipped_flat_region: List[float] = field(default_factory=list)
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
    total_extent_sec: float
    covered_sec: float
    missing_pct: float
    n_gaps: int
    gap_durations_sec: List[float]
    longest_gap_sec: float
    merged_intervals: List[Tuple[float, float]]
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
    error: bool = False,
    error_msg: Optional[str] = None,
) -> GuidCoverageResult:
    """Compute coverage statistics for a single GUID."""
    n_skipped_lw = len(skipped_low_weight) if skipped_low_weight else 0
    n_skipped_flat = len(skipped_flat_region) if skipped_flat_region else 0

    if error:
        return GuidCoverageResult(
            guid=guid,
            total_segments=len(all_starts) if all_starts else 0,
            included_segments=0,
            skipped_low_weight=n_skipped_lw,
            skipped_flat_region=n_skipped_flat,
            total_extent_sec=0.0,
            covered_sec=0.0,
            missing_pct=100.0,
            n_gaps=0,
            gap_durations_sec=[],
            longest_gap_sec=0.0,
            merged_intervals=[],
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

    # Build and merge included intervals
    if not included_starts:
        return GuidCoverageResult(
            guid=guid,
            total_segments=total_segments,
            included_segments=0,
            skipped_low_weight=n_skipped_lw,
            skipped_flat_region=n_skipped_flat,
            total_extent_sec=total_extent,
            covered_sec=0.0,
            missing_pct=100.0,
            n_gaps=0,
            gap_durations_sec=[],
            longest_gap_sec=0.0,
            merged_intervals=[],
        )

    intervals = [(s, s + segment_duration_sec) for s in included_starts]
    merged = _merge_intervals(intervals)
    covered = sum(end - start for start, end in merged)

    # Compute gaps within total extent
    extent_start = min_start
    extent_end = max_start + segment_duration_sec
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
        total_extent_sec=total_extent,
        covered_sec=covered,
        missing_pct=missing_pct,
        n_gaps=len(gaps),
        gap_durations_sec=gaps,
        longest_gap_sec=max(gaps, default=0.0),
        merged_intervals=merged,
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

    return {
        'total_guids': len(coverage_results),
        'guids_with_data': len(non_error) - len(fully_rejected),
        'fully_rejected_guids': len(fully_rejected),
        'errored_guids': len(errored),
        'total_segments_from_prepare_data': total_all_segments,
        'total_included_segments': total_included,
        'total_skipped_low_weight': total_skipped_lw,
        'total_skipped_flat_region': total_skipped_flat,
        'has_tracking_data': guid_tracking is not None,
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
            lines.append(f'- `{r.guid}` — {r.total_segments} total segments, '
                         f'{r.skipped_low_weight} low weight, {r.skipped_flat_region} flat region')

    # Errored GUIDs
    errored = [r for r in coverage_results if r.error]
    if errored:
        lines.append(f'\n## Errored GUIDs ({len(errored)})')
        lines.append('')
        for r in errored:
            lines.append(f'- `{r.guid}` — {r.error_msg or "unknown error"}')

    # Per-GUID table
    non_error = [r for r in coverage_results if not r.error]
    if non_error:
        lines.append('\n## Per-GUID Detail')
        lines.append('')
        lines.append('| GUID | Total | Included | Skipped LW | Skipped Flat | Missing % | Longest Gap (min) |')
        lines.append('|------|-------|----------|------------|--------------|-----------|-------------------|')
        for r in non_error:
            longest_gap_min = r.longest_gap_sec / 60.0
            lines.append(
                f'| `{r.guid}` | {r.total_segments} | {r.included_segments} '
                f'| {r.skipped_low_weight} | {r.skipped_flat_region} '
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
    dpi: int = 200,
) -> Optional[str]:
    """Horizontal broken-bar chart: blue=covered, red=gap per GUID."""
    non_error = [r for r in coverage_results if not r.error and r.total_extent_sec > 0]
    if not non_error:
        return None

    # Already sorted by total_extent desc from _compute_all_coverage
    plotted = non_error[:max_guids]
    n = len(plotted)

    fig_height = max(4, n * 0.3 + 1)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    for idx, r in enumerate(reversed(plotted)):  # reverse so largest at top
        y = idx
        # Draw total extent as red background
        if r.total_extent_sec > 0:
            # Use domain_start values (seconds before delivery, often negative)
            if r.merged_intervals:
                extent_start = min(iv[0] for iv in r.merged_intervals)
            else:
                extent_start = 0

            # Red background for total extent
            ax.barh(y, r.total_extent_sec / 60.0, left=extent_start / 60.0,
                    height=0.7, color='#ffcccc', edgecolor='#ff6666', linewidth=0.5)

            # Blue bars for covered intervals
            for start, end in r.merged_intervals:
                ax.barh(y, (end - start) / 60.0, left=start / 60.0,
                        height=0.7, color='#4488cc', edgecolor='#2266aa', linewidth=0.5)

    labels = [r.guid[:20] for r in reversed(plotted)]
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=max(5, min(8, 200 // n)))
    ax.set_xlabel('Time (minutes, relative to delivery)')
    ax.set_title(f'Coverage Timeline ({n} GUIDs)')
    ax.legend(handles=[
        Patch(facecolor='#4488cc', edgecolor='#2266aa', label='Covered'),
        Patch(facecolor='#ffcccc', edgecolor='#ff6666', label='Gap / Missing'),
    ], loc='lower right', fontsize=8)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, 'coverage_timeline.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_gap_distribution(
    coverage_results: List[GuidCoverageResult],
    output_dir: str,
    dpi: int = 200,
) -> Optional[str]:
    """Histogram of gap durations in minutes."""
    all_gaps = []
    for r in coverage_results:
        if not r.error:
            all_gaps.extend(r.gap_durations_sec)

    if not all_gaps:
        return None

    gaps_min = np.array(all_gaps) / 60.0

    fig, ax = plt.subplots(figsize=(10, 6))
    n_bins = min(50, max(10, len(gaps_min) // 5))
    ax.hist(gaps_min, bins=n_bins, color='#4488cc', edgecolor='#2266aa', alpha=0.8)

    mean_val = float(np.mean(gaps_min))
    median_val = float(np.median(gaps_min))
    max_val = float(np.max(gaps_min))

    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.1f} min')
    ax.axvline(median_val, color='orange', linestyle='--', linewidth=1.5, label=f'Median: {median_val:.1f} min')

    ax.text(0.98, 0.95,
            f'Count: {len(gaps_min)}\nMean: {mean_val:.1f} min\nMedian: {median_val:.1f} min\nMax: {max_val:.1f} min',
            transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax.set_xlabel('Gap Duration (minutes)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Gap Durations')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, 'gap_distribution.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_missing_data_percentage(
    coverage_results: List[GuidCoverageResult],
    output_dir: str,
    max_guids: int = 50,
    dpi: int = 200,
) -> Optional[str]:
    """Sorted horizontal bar chart of missing data % per GUID."""
    non_error = [r for r in coverage_results if not r.error and r.total_extent_sec > 0]
    if not non_error:
        return None

    # Sort by missing_pct descending
    sorted_results = sorted(non_error, key=lambda r: r.missing_pct, reverse=True)
    plotted = sorted_results[:max_guids]
    n = len(plotted)

    fig_height = max(4, n * 0.35 + 1)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    guids = [r.guid[:20] for r in plotted]
    pcts = [r.missing_pct for r in plotted]

    # Color gradient green->red
    colors = []
    for p in pcts:
        t = p / 100.0
        r = min(1.0, 2 * t)
        g = min(1.0, 2 * (1 - t))
        colors.append((r, g, 0.2, 0.8))

    y_pos = range(n)
    ax.barh(y_pos, pcts, color=colors, edgecolor='gray', linewidth=0.5)

    mean_pct = float(np.mean([r.missing_pct for r in non_error]))
    ax.axvline(mean_pct, color='blue', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_pct:.1f}%')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(guids, fontsize=max(5, min(8, 200 // n)))
    ax.set_xlabel('Missing Data (%)')
    ax.set_title(f'Missing Data Percentage by GUID (top {n})')
    ax.set_xlim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, 'missing_data_percentage.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_segment_counts(
    coverage_results: List[GuidCoverageResult],
    output_dir: str,
    has_tracking: bool = False,
    max_guids: int = 50,
    dpi: int = 200,
) -> Optional[str]:
    """Stacked bar chart of segment counts per GUID."""
    non_error = [r for r in coverage_results if not r.error]
    if not non_error:
        return None

    # Sort by total segments descending
    sorted_results = sorted(non_error, key=lambda r: r.total_segments, reverse=True)
    plotted = sorted_results[:max_guids]
    n = len(plotted)

    fig_height = max(4, n * 0.35 + 1)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    guids = [r.guid[:20] for r in plotted]
    y_pos = np.arange(n)

    if has_tracking:
        included = [r.included_segments for r in plotted]
        skipped_lw = [r.skipped_low_weight for r in plotted]
        skipped_flat = [r.skipped_flat_region for r in plotted]

        ax.barh(y_pos, included, color='#4488cc', edgecolor='#2266aa',
                linewidth=0.5, label='Included')
        ax.barh(y_pos, skipped_lw, left=included, color='#ff9933',
                edgecolor='#cc7722', linewidth=0.5, label='Skipped (Low Weight)')
        left2 = [a + b for a, b in zip(included, skipped_lw)]
        ax.barh(y_pos, skipped_flat, left=left2, color='#cc3333',
                edgecolor='#aa2222', linewidth=0.5, label='Skipped (Flat Region)')
    else:
        included = [r.included_segments for r in plotted]
        ax.barh(y_pos, included, color='#4488cc', edgecolor='#2266aa',
                linewidth=0.5, label='Included')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(guids, fontsize=max(5, min(8, 200 // n)))
    ax.set_xlabel('Segment Count')
    ax.set_title(f'Segment Counts by GUID (top {n})')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, 'segment_counts.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_rejection_reasons(
    summary: Dict[str, Any],
    output_dir: str,
    dpi: int = 200,
) -> Optional[str]:
    """Pie chart of rejection reasons. Only generated with tracking data."""
    if not summary.get('has_tracking_data'):
        return None

    included = summary['total_included_segments']
    lw = summary['total_skipped_low_weight']
    flat = summary['total_skipped_flat_region']
    errored = summary['errored_guids']

    labels = []
    sizes = []
    colors = []

    if included > 0:
        labels.append(f'Included ({included})')
        sizes.append(included)
        colors.append('#4488cc')
    if lw > 0:
        labels.append(f'Low Weight ({lw})')
        sizes.append(lw)
        colors.append('#ff9933')
    if flat > 0:
        labels.append(f'Flat Region ({flat})')
        sizes.append(flat)
        colors.append('#cc3333')
    if errored > 0:
        labels.append(f'Errored GUIDs ({errored})')
        sizes.append(errored)
        colors.append('#999999')

    if not sizes:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    _, _, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.85,
        wedgeprops=dict(edgecolor='white', linewidth=1.5))
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title('Segment Rejection Reasons')
    plt.tight_layout()

    path = os.path.join(output_dir, 'rejection_reasons.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_guid_analysis(
    hdf5_path: str,
    guid_tracking: Optional[Dict[str, GuidTrackingEntry]] = None,
    segment_duration_sec: float = 1440.0,
    output_dir: Optional[str] = None,
    max_guids_in_timeline_plot: int = 100,
    dpi: int = 200,
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
        dpi: Resolution for saved plots.

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
