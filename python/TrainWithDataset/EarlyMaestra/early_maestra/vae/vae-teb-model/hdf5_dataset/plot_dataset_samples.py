r"""Plot random samples of an HDF5 dataset: the two signals above one panel per coefficient block.

Each page shows one segment as the loader serves it: ``fhr`` and ``up`` as line plots, then one
image per stored FHR coefficient block (:data:`COEFFICIENT_PANELS`).

**A signal-quality file adds the quality display.** A file built with the signal-quality step
(root attribute ``quality_version``) stores the bridged signals in ``fhr``/``up`` and, beside
them, the bit fields ``up_quality``/``fhr_quality`` and the recorded signals ``up_raw``/``fhr_raw``.
On such a file the two signal panels change:

* they draw the stored signal and the ``<signal>_raw`` signal together, both in recorded units
  read from the file and trimmed exactly as the loader trims ``fhr``/``up``;
* they shade every invalid sample. A UA sample is invalid where
  ``up_quality & invalid_bits`` $\ne 0$. An FHR sample of segment $i$ is invalid where
  ``fhr_quality & B`` $\ne 0$, with $B$ = ``invalid_bits_exact`` if ``fhr_repair_exact[i]``,
  else ``invalid_bits_rules``. UA samples marked ``CENSORED`` (clamped by the device at a
  ceiling or at a rest floor, and valid) get a second, lighter shade;
* the UA axis reaches at least 130, because UA values from 100 to 127 are real sensor values.

A file without the quality fields gets the pages it always got.

Run from the command line::

    python hdf5_dataset/plot_dataset_samples.py --hdf5-file-path SHARD --output-dir DIR

or edit :data:`RUN_ARGS` at the bottom of this file and press Run.

The pages hold clinical traces. Keep the output folder on the machine that holds the data.
"""
import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional

#: Repository root: ``hdf5_dataset/plot_dataset_samples.py`` -> up two.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Launched as a script -- an IDE's Run button -- this file's own directory goes on sys.path
# instead of the repository root, and every absolute import below fails before __main__ is
# reached.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402

from hdf5_dataset.hdf5_dataset import CombinedHDF5Dataset  # noqa: E402
from teb_vae.lag_attn_rws.eval.launch import (  # noqa: E402
    missing_required,
    resolve_launch_args,
)

#: Coefficient panels, in display order: ``(field, y-axis label, panel title, colourbar label)``.
#: Which of them a figure actually gets is decided per sample, because a causal file stores no
#: ``fhr_up_ph`` — the cross-phase block is not produced causally — and reaching for it there
#: raises ``AttributeError`` from the middle of a plotting loop, naming nothing useful.
COEFFICIENT_PANELS = (
    ('fhr_st', 'Scattering Channels', 'Normalized FHR Scattering Transform',
     'Normalized Amplitude'),
    ('fhr_ph', 'Phase Channels', 'Normalized FHR Phase Harmonics', 'Normalized Phase'),
    ('fhr_up_ph', 'Cross-Phase Channels', 'Normalized FHR-UP Cross-Phase',
     'Normalized Cross-Phase'),
)

#: The datasets the quality display reads from a file with the root attribute
#: ``quality_version``.
QUALITY_DATASETS = ('up', 'fhr', 'up_raw', 'fhr_raw', 'up_quality', 'fhr_quality',
                    'fhr_repair_exact', 'up_valid_frac', 'fhr_valid_frac')

#: The lowest upper limit of the UA axis in the quality display. UA values from 100 to 127 are
#: real sensor values, and a value above 127 is an invalid code.
UA_AXIS_MIN_TOP = 130.0

#: Raw sampling rate of ``fhr``/``up`` in Hz.
FS = 4


def file_has_quality(path: str) -> bool:
    """Say whether a file was built with the signal-quality step.

    Args:
        path: The HDF5 file.

    Returns:
        ``True`` if the file carries the root attribute ``quality_version``.
    """
    with h5py.File(path, 'r') as handle:
        return 'quality_version' in handle.attrs


def read_quality_trace(handle: h5py.File, row: int, trim: int) -> Dict[str, Any]:
    r"""Read one segment's signals and quality fields, trimmed like the loader trims ``fhr``/``up``.

    Args:
        handle: An open file with the root attribute ``quality_version``.
        row: The segment's row in that file.
        trim: Raw samples removed from each end (``CombinedHDF5Dataset.trim_samples_raw``).

    Returns:
        A dict with the stored signals ``fhr`` and ``up`` and the recorded signals ``fhr_raw``
        and ``up_raw`` (float64, recorded units); the boolean masks ``up_invalid``,
        ``up_censored`` and ``fhr_invalid``; ``fhr_exact`` (the FHR repair mask came from the raw
        channel); the per-segment ``up_valid_frac`` and ``fhr_valid_frac``; and
        ``ua_ceilings``, the device ceilings of the root attribute ``quality_ceiling_values``
        (empty if absent).

    Raises:
        ValueError: If the file declares ``quality_version`` but lacks a dataset of
            :data:`QUALITY_DATASETS` or a bit-field attribute, naming them.
    """
    missing = [name for name in QUALITY_DATASETS if name not in handle]
    if not missing:
        needed = (('up_quality', ('bits', 'invalid_bits')),
                  ('fhr_quality', ('bits', 'invalid_bits_exact', 'invalid_bits_rules')))
        missing = [f'{name}.{attr}' for name, attrs in needed for attr in attrs
                   if attr not in handle[name].attrs]
    if missing:
        raise ValueError(
            f"{handle.filename} declares quality_version={handle.attrs['quality_version']} but "
            f"lacks {missing}. The quality display reads the bit fields, their attributes and "
            f"the recorded signals together; a partial quality file is a broken build."
        )
    stop = handle['fhr'].shape[1] - trim
    window = slice(trim, stop)
    up_bits = handle['up_quality'][row][window].astype(np.int64)
    fhr_bits = handle['fhr_quality'][row][window].astype(np.int64)
    up_names = json.loads(handle['up_quality'].attrs['bits'])
    exact = bool(handle['fhr_repair_exact'][row])
    fhr_mask = int(handle['fhr_quality'].attrs['invalid_bits_exact' if exact
                                               else 'invalid_bits_rules'])
    censored_bit = int(up_names.get('CENSORED', 0))
    return {
        'fhr': handle['fhr'][row][window].astype(np.float64),
        'up': handle['up'][row][window].astype(np.float64),
        'fhr_raw': handle['fhr_raw'][row][window].astype(np.float64),
        'up_raw': handle['up_raw'][row][window].astype(np.float64),
        'up_invalid': (up_bits & int(handle['up_quality'].attrs['invalid_bits'])) != 0,
        'up_censored': (up_bits & censored_bit) != 0,
        'fhr_invalid': (fhr_bits & fhr_mask) != 0,
        'fhr_exact': exact,
        'up_valid_frac': float(handle['up_valid_frac'][row]),
        'fhr_valid_frac': float(handle['fhr_valid_frac'][row]),
        'ua_ceilings': np.asarray(handle.attrs.get('quality_ceiling_values', []),
                                  dtype=np.float64).ravel(),
    }


def _shade(ax: Any, t: np.ndarray, where: np.ndarray, color: str, label: str) -> None:
    """Shade the full height of ``ax`` where ``where`` is set; no legend entry if never set."""
    if np.any(where):
        ax.fill_between(t, 0, 1, where=where, transform=ax.get_xaxis_transform(), color=color,
                        linewidth=0, step='mid', label=label, zorder=0)


def _plot_quality_signals(axes: List[Any], t: np.ndarray, trace: Dict[str, Any],
                          colors: Dict[str, str]) -> None:
    """Draw the quality display into the FHR panel ``axes[0]`` and the UA panel ``axes[1]``.

    Args:
        axes: The two signal axes.
        t: Time of each sample in seconds.
        trace: The output of :func:`read_quality_trace`.
        colors: The page palette.
    """
    fhr_axis, up_axis = axes
    _shade(fhr_axis, t, trace['fhr_invalid'], colors['invalid'], 'invalid (bridged)')
    # A recorded FHR of 0 is a missing sample, not a heart rate; drawing it would set the scale.
    fhr_axis.plot(t, np.where(trace['fhr_raw'] > 0, trace['fhr_raw'], np.nan), linewidth=0.8,
                  color=colors['raw'], alpha=0.9, label='fhr_raw (as recorded)')
    fhr_axis.plot(t, trace['fhr'], linewidth=1.2, color=colors['fhr'], alpha=0.85,
                  label='fhr (stored, bridged)')
    fhr_axis.set_ylabel('FHR (bpm)', fontweight='normal')
    repair = 'from the raw channel' if trace['fhr_exact'] else 'from the rules'
    fhr_axis.set_title(f"Fetal Heart Rate (FHR) Signal: valid {100 * trace['fhr_valid_frac']:.1f}%,"
                       f" repair mask {repair}", fontweight='normal', pad=12)
    fhr_axis.legend(loc='upper right', fontsize=9, ncols=3)

    _shade(up_axis, t, trace['up_invalid'], colors['invalid'], 'invalid (bridged)')
    _shade(up_axis, t, trace['up_censored'], colors['censored'], 'censored (valid)')
    up_axis.plot(t, trace['up_raw'], linewidth=0.8, color=colors['raw'], alpha=0.9,
                 label='up_raw (as recorded)')
    up_axis.plot(t, trace['up'], linewidth=1.2, color=colors['up'], alpha=0.85,
                 label='up (stored, bridged)')
    for position, level in enumerate(trace['ua_ceilings']):
        up_axis.axhline(level, color='#9E9D9D', linewidth=0.8, linestyle='--',
                        label='device ceilings' if position == 0 else None)
    finite = np.concatenate([trace['up_raw'], trace['up']])
    finite = finite[np.isfinite(finite)]
    low = min(-5.0, float(finite.min()) - 5.0) if finite.size else -5.0
    high = max(UA_AXIS_MIN_TOP, float(finite.max()) + 5.0) if finite.size else UA_AXIS_MIN_TOP
    up_axis.set_ylim(low, high)
    up_axis.set_ylabel('UP Amplitude', fontweight='normal')
    up_axis.set_title(f"Uterine Pressure (UP) Signal: valid {100 * trace['up_valid_frac']:.1f}%",
                      fontweight='normal', pad=12)
    up_axis.legend(loc='upper right', fontsize=9, ncols=4)


def plot_random_dataset_samples(
    hdf5_file_path: str,
    stats_file_path: Optional[str] = None,
    output_dir: str = "./dataset_sample_plots",
    n_samples: int = 10,
    trim_minutes: Optional[float] = None,
    seed: int = 42,
    quality: Optional[bool] = None,
) -> List[str]:
    """Plot randomly selected samples of an HDF5 dataset, one PDF page per sample.

    Each page shows:

    - the FHR and UP signals (line plots); on a signal-quality file, the quality display
      described in the module docstring;
    - one image per stored FHR coefficient block, from :data:`COEFFICIENT_PANELS` filtered by
      what the file carries — three on a two-sided file, two on a causal one, which stores no
      ``fhr_up_ph``.

    Args:
        hdf5_file_path: The HDF5 dataset file, or a list of files read as one dataset.
        stats_file_path: The statistics file the coefficient blocks are normalised with, or
            ``None`` to draw them unnormalised.
        output_dir: Directory to save the plots in.
        n_samples: Number of random samples to plot.
        trim_minutes: Symmetric trim in minutes, as the loader applies it, or ``None`` for none.
        seed: Seed of the sample selection.
        quality: ``None`` draws the quality display on every page whose file has the root
            attribute ``quality_version``; ``True`` requires it and refuses a file without it;
            ``False`` never draws it.

    Returns:
        The paths of the saved pages.

    Raises:
        ValueError: If ``quality`` is ``True`` and a file has no quality fields.
    """
    # Set random seed for reproducible results
    random.seed(seed)
    np.random.seed(seed)

    # Professional scientific paper color palette (from plotting callback)
    colors = {
        'fhr': "#055C9A",           # Deep blue-gray
        'up': "#0DD8A2",            # Sage green
        'gt': '#456882',            # Medium blue-gray
        'recon': '#BB3E00',         # Deep orange-red
        'uncertainty': '#F7AD45',    # Golden yellow
        'samples': "#4BD605",       # Muted green-gray
        'background': '#F9F3EF',    # Warm off-white
        'raw': '#7A7A7A',           # Neutral gray: the signal as recorded
        'invalid': '#F2C4BE',       # Light red: invalid (bridged) samples
        'censored': '#F8E2B0',      # Light amber: censored samples, still valid
    }

    # Set professional scientific paper styling
    plt.style.use('default')  # Reset to default first
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'axes.linewidth': 0.7,
        'axes.edgecolor': "#9E9D9D",
        'axes.facecolor': colors['background'],
        'grid.color': "#838383",
        'grid.linewidth': 0.4,
        'grid.alpha': 0.6,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.shadow': False,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '#A2B9A7',
        'legend.facecolor': colors['background'],
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 300
    })

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {hdf5_file_path}")
    dataset = CombinedHDF5Dataset(
        paths=hdf5_file_path,
        stats_path=stats_file_path,
        trim_minutes=trim_minutes,
        cache_size=0  # Disable caching for memory efficiency
    )

    print(f"Dataset loaded with {len(dataset)} samples")

    # Which files get the quality display: decided per file, once.
    with_quality = [quality is not False and file_has_quality(path) for path in dataset.paths]
    if quality is True and not all(with_quality):
        lacking = [path for path, on in zip(dataset.paths, with_quality) if not on]
        raise ValueError(f"quality=True but {lacking} carry no quality_version: they were built "
                         f"without the signal-quality step, so there is no quality to draw.")

    # Randomly select sample indices
    total_samples = len(dataset)
    if n_samples > total_samples:
        print(f"Warning: Requested {n_samples} samples but dataset only contains {total_samples}. Using all samples.")
        n_samples = total_samples

    sample_indices = random.sample(range(total_samples), n_samples)
    print(f"Selected sample indices: {sample_indices}")

    saved: List[str] = []
    for i, sample_idx in enumerate(sample_indices):
        print(f"Processing sample {i+1}/{n_samples} (index {sample_idx})")

        # Load sample
        sample = dataset[sample_idx]

        # Extract data
        fhr_signal = sample.fhr.cpu().numpy()  # Raw FHR signal
        up_signal = sample.up.cpu().numpy()    # Raw UP signal
        # Whatever coefficient blocks this file carries, in the documented order. A two-sided file
        # yields all three and the figure is what it always was; a causal file yields two.
        panels = [entry for entry in COEFFICIENT_PANELS if entry[0] in sample]

        trace = None
        file_idx, row = dataset.index_map[sample_idx]
        if with_quality[file_idx]:
            with h5py.File(dataset.paths[file_idx], 'r') as handle:
                trace = read_quality_trace(handle, row, dataset.trim_samples_raw)
            if trace['fhr'].shape != fhr_signal.shape:
                raise ValueError(
                    f"quality trace of sample {sample_idx} has {trace['fhr'].size} samples but the "
                    f"loader served {fhr_signal.size}: the trim here no longer matches the loader's."
                )

        # Create time axis for signals
        N = len(fhr_signal)
        t_signal = np.arange(0, N) / FS

        # Create figure: two signal plots above one imshow per coefficient block
        n_rows = 2 + len(panels)
        fig, axes = plt.subplots(n_rows, 1, figsize=(20, 4 * n_rows), constrained_layout=True)

        # Configure scientific paper grid style for all subplots
        for ax in axes[:2]:  # Only for signal plots
            ax.grid(True, linestyle='-', alpha=0.4, linewidth=0.4, color='#D2C1B6')
            ax.grid(True, which='minor', linestyle=':', alpha=0.25, linewidth=0.3, color='#D2C1B6')
            ax.minorticks_on()
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#A2B9A7')
            ax.spines['bottom'].set_color('#A2B9A7')
            ax.spines['left'].set_linewidth(0.7)
            ax.spines['bottom'].set_linewidth(0.7)

        if trace is None:
            # Plot 1: FHR Signal
            axes[0].plot(t_signal, fhr_signal, linewidth=1.2, color=colors['fhr'], alpha=0.85)
            axes[0].set_ylabel('FHR (bpm)', fontweight='normal')
            axes[0].set_title('Fetal Heart Rate (FHR) Signal', fontweight='normal', pad=12)

            # Plot 2: UP Signal
            axes[1].plot(t_signal, up_signal, linewidth=1.2, color=colors['up'], alpha=0.85)
            axes[1].set_ylabel('UP Amplitude', fontweight='normal')
            axes[1].set_title('Uterine Pressure (UP) Signal', fontweight='normal', pad=12)
        else:
            _plot_quality_signals(list(axes[:2]), t_signal, trace, colors)
        axes[0].autoscale(enable=True, axis='x', tight=True)
        axes[1].set_xlabel('Time (s)', fontweight='normal')
        axes[1].autoscale(enable=True, axis='x', tight=True)

        # Coefficient panels, one per stored block.
        # Note: Data comes from dataloader as (sequence, channels), transpose for correct display
        for panel_row, (field, ylabel, title, cbar_label) in enumerate(panels, start=2):
            if not dataset.is_normalization_enabled():
                # No statistics file: the block is drawn as stored, so the label must not say
                # otherwise.
                title = title.replace('Normalized ', 'Unnormalized ')
                cbar_label = cbar_label.replace('Normalized ', '')
            axis = axes[panel_row]
            image = axis.imshow(sample[field].cpu().numpy().T, aspect='auto', cmap='seismic',
                                origin='upper', interpolation='none')
            axis.set_ylabel(ylabel, fontweight='normal')
            axis.set_title(title, fontweight='normal', pad=12)
            if panel_row == n_rows - 1:
                axis.set_xlabel('Time Steps', fontweight='normal')
            cbar = plt.colorbar(image, ax=axis, shrink=0.8)
            cbar.ax.tick_params(labelsize=10, colors='#666666')
            cbar.set_label(cbar_label, fontweight='normal', fontsize=11, color='#666666')
            cbar.outline.set_color('#A2B9A7')
            cbar.outline.set_linewidth(0.7)

        # Set overall title
        sample_guid = sample.guid if hasattr(sample, 'guid') else f"Sample_{sample_idx}"
        fig.suptitle(f'Dataset Sample Analysis — {sample_guid}',
                     fontsize=14, fontweight='normal', y=0.97, color='#456882')

        # Save plot
        save_path = os.path.join(output_dir, f'sample_{i+1:02d}_idx_{sample_idx}.pdf')
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white', edgecolor='none')
        plt.close(fig)
        saved.append(save_path)

        print(f"Saved plot to {save_path}")

    print(f"\nCompleted plotting {n_samples} random samples")
    print(f"All plots saved to: {output_dir}")
    return saved


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    No argument is ``required`` and none carries a default: :data:`RUN_ARGS` fills every gap,
    and :func:`_cli` applies the real defaults after the merge.

    Returns:
        The parser.
    """
    p = argparse.ArgumentParser(
        prog="python hdf5_dataset/plot_dataset_samples.py",
        description="Plot random samples of an HDF5 dataset, with the quality display on a "
                    "signal-quality file.")
    p.add_argument("--hdf5-file-path", dest="hdf5_file_path", nargs="+")
    p.add_argument("--stats-file-path", dest="stats_file_path")
    p.add_argument("--output-dir", dest="output_dir")
    p.add_argument("--n-samples", dest="n_samples", type=int)
    p.add_argument("--trim-minutes", dest="trim_minutes", type=float)
    p.add_argument("--seed", dest="seed", type=int)
    p.add_argument("--quality", dest="quality", action=argparse.BooleanOptionalAction,
                   default=None)
    return p


def _cli(argv: Optional[List[str]] = None) -> int:
    """Merge the command line over :data:`RUN_ARGS` and run.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv[1:]``.

    Returns:
        The process exit code.

    Raises:
        SystemExit: If the dataset path was supplied by neither route.
    """
    values, sources = resolve_launch_args(build_parser(), RUN_ARGS, argv)
    refusal = missing_required(values, ("hdf5_file_path",))
    if refusal:
        raise SystemExit(refusal)
    print("resolved arguments: " + ", ".join(
        f"{key}={values[key]!r} (from {sources[key]})" for key in sorted(values)))
    paths = values["hdf5_file_path"]
    if isinstance(paths, (list, tuple)) and len(paths) == 1:
        paths = paths[0]
    plot_random_dataset_samples(
        hdf5_file_path=paths,
        stats_file_path=values["stats_file_path"],
        output_dir=values["output_dir"] or "./dataset_sample_plots",
        n_samples=10 if values["n_samples"] is None else values["n_samples"],
        trim_minutes=values["trim_minutes"],
        seed=42 if values["seed"] is None else values["seed"],
        quality=values["quality"],
    )
    return 0


#: **Edit the values below and press Run.** A flag on the command line overrides its entry.
RUN_ARGS: Dict[str, Any] = {
    # str | list[str] -- the HDF5 dataset file, or several read as one dataset. Required.
    "hdf5_file_path": r"C:\Users\mahdi\Desktop\McGill\data\acidosis_no_cs.hdf5",

    # str | None -- statistics file the coefficient blocks are normalised with.
    #        (None -> unnormalised blocks)
    #        A signal-quality file needs a statistics file built from a signal-quality dataset.
    "stats_file_path": r"C:\Users\mahdi\Desktop\McGill\data\stats.hdf5",

    # str | None -- directory for the PDF pages.  (None -> "./dataset_sample_plots")
    "output_dir": r"C:\Users\mahdi\Desktop\McGill\data\plots_samples",

    # int | None -- number of random samples.  (None -> 10)
    "n_samples": 10,

    # float | None -- symmetric trim in minutes; match the statistics file.  (None -> no trim)
    "trim_minutes": 2,

    # int | None -- seed of the sample selection.  (None -> 42)
    "seed": 42,

    # bool | None -- the quality display.  (None -> on for every file with quality_version)
    #   True  : required; a file built without the signal-quality step is refused.
    #   False : never drawn.
    "quality": None,
}


if __name__ == "__main__":
    sys.exit(_cli())
