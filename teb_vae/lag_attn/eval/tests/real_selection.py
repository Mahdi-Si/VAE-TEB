r"""The production channel selection, measured once and pinned here as test data.

Not synthesised. Every array below was read out of the real selector -- ``create_new_pipeline.py
::compute_scattering_masks`` at the production geometry ($J = 11$, $Q = 4$, $T = 16$,
``shape=5280``, $f_s = 4$ Hz) -- and is what ``_write_selection_attrs`` stamps onto a shard
written by the current pipeline.

It is recorded here rather than recomputed at test time for three reasons. Recomputing needs
``kymatio`` and the ``hdf5_dataset`` package, which would couple the eval suite to the
dataset-building tree that the whole package is meant to be independent of. It costs seconds of
filter-bank construction per run, in a suite that has to stay in the fast gate. And a *pinned*
measurement is a regression guard: if the pipeline's selection changes, the assertions built on
these numbers fail and say so, whereas a recomputed one would silently agree with whatever the
selector now does.

The measured facts these arrays encode, which are the acceptance criteria the band partition was
written against:

* ``fhr_ph`` is 66 channels, ``up_ph`` is 15.
* The ``fhr_ph`` harmonic-kind distribution is $k = 4 \to 24$, $k = 6 \to 22$, $k = 8 \to 20$.
* The filter centre frequencies are in **Hz** and run *descending* with index, from $1.49$ Hz at
  index $0$ to $5.15 \times 10^{-4}$ Hz at index $41$.
* Only filters $3$ to $30$ are referenced by any selected pair, because both endpoints of a pair
  must lie inside the $(0.008, 1.00)$ Hz band. The other 14 are the reason 14 scattering channels
  have no recoverable centre frequency.

Regenerate with ``compute_scattering_masks(signal_length=5280, scattering_T=16)``, reading
``.i`` / ``.j`` off each ``PhaseChannelSelection`` and ``model.center_freqs * fs`` off the
transform.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

#: Order-1 filter centre frequencies in Hz, index-aligned with the scattering channel map:
#: scattering channel $c \ge 1$ is filter $c - 1$, and channel $0$ is the order-0 lowpass.
#: Descending, as kymatio orders them.
FILTER_HZ: Tuple[float, ...] = (
    1.49153948, 1.25423026, 1.05467772, 0.886874735, 0.745769739, 0.62711513,
    0.527338862, 0.443437368, 0.37288487, 0.313557565, 0.263669431, 0.221718684,
    0.186442435, 0.156778783, 0.131834716, 0.110859342, 0.0932212174, 0.0783893913,
    0.0659173578, 0.055429671, 0.0466106087, 0.0391946957, 0.0329586789, 0.0277148355,
    0.0233053043, 0.0195973478, 0.0164793395, 0.0138574177, 0.0116526522, 0.00979867391,
    0.00823966973, 0.00692870887, 0.00582632609, 0.00489933696, 0.00411983486, 0.00346435443,
    0.00291316304, 0.00244966848, 0.00205991743, 0.00154493807, 0.00102995872, 0.000514979358,
)

#: Lower-frequency filter index per ``fhr_ph`` channel, in stored channel order.
FHR_I: Tuple[int, ...] = (
    7, 8, 9, 9, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16,
    17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20, 21, 21, 21, 22, 22, 22, 23, 23, 23, 24, 24,
    24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30,
)

#: Higher-frequency filter index per ``fhr_ph`` channel. Lower *index*, since ``FILTER_HZ``
#: descends -- which is exactly the trap that makes ``sel_i``/``sel_j`` worth reading rather than
#: re-deriving.
FHR_J: Tuple[int, ...] = (
    3, 4, 3, 5, 4, 6, 3, 5, 7, 4, 6, 8, 5, 7, 9, 6, 8, 10, 7, 9, 11, 8, 10, 12,
    9, 11, 13, 10, 12, 14, 11, 13, 15, 12, 14, 16, 13, 15, 17, 14, 16, 18, 15, 17, 19, 16, 18,
    20, 17, 19, 21, 18, 20, 22, 19, 21, 23, 20, 22, 24, 21, 23, 25, 22, 24, 26,
)

#: The same for ``up_ph``, whose band is the narrower $(0.008, 0.05)$ Hz.
UP_I: Tuple[int, ...] = (24, 25, 26, 26, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30)
UP_J: Tuple[int, ...] = (20, 21, 20, 22, 21, 23, 20, 22, 24, 21, 23, 25, 22, 24, 26)

#: Measured widths and distributions, asserted by the tests so a pipeline change is visible.
N_SCATTERING = 43
N_FHR_PHASE = 66
N_UP_PHASE = 15
FHR_KIND_COUNTS: Dict[str, int] = {"ph_k4": 24, "ph_k6": 22, "ph_k8": 20}

#: Selection parameters, as stored in the ``sel_band_hz`` / ``sel_k_steps`` attributes.
FHR_BAND_HZ = (0.008, 1.0)
UP_BAND_HZ = (0.008, 0.05)
K_STEPS = (4, 6, 8)

#: Order-1 filters no selected pair references, so the scattering channels above them
#: ($c = f + 1$) have no recoverable centre frequency. Both endpoints of a pair must sit inside
#: the phase band, which excludes the three fastest filters and the eleven slowest.
UNREFERENCED_FILTERS: Tuple[int, ...] = (0, 1, 2, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41)

#: Clinical band occupancy over the full 109-channel target, measured. ``unknown`` is the 14
#: scattering channels above :data:`UNREFERENCED_FILTERS`.
CLINICAL_BAND_COUNTS: Dict[str, int] = {
    "slow_baseline": 1,
    "deceleration": 22,
    "variability": 40,
    "beat_to_beat": 32,
    "unknown": 14,
}


def selection_attrs(which: str) -> Dict[str, np.ndarray]:
    """Return one block's ``sel_*`` attributes exactly as the pipeline writes them.

    Args:
        which: ``'fhr_ph'`` or ``'up_ph'``.

    Returns:
        The seven attributes, with the frequencies already in Hz and the arrays in stored
        channel order.

    Raises:
        ValueError: If ``which`` names neither block.
    """
    if which == "fhr_ph":
        index_i, index_j, band = FHR_I, FHR_J, FHR_BAND_HZ
    elif which == "up_ph":
        index_i, index_j, band = UP_I, UP_J, UP_BAND_HZ
    else:
        raise ValueError(f"unknown phase block {which!r}; expected 'fhr_ph' or 'up_ph'.")

    frequencies = np.asarray(FILTER_HZ, dtype=np.float64)
    xi_i = frequencies[list(index_i)]
    xi_j = frequencies[list(index_j)]
    return {
        "sel_i": np.asarray(index_i, dtype=np.int32),
        "sel_j": np.asarray(index_j, dtype=np.int32),
        "sel_xi_i_hz": xi_i.astype(np.float32),
        "sel_xi_j_hz": xi_j.astype(np.float32),
        # p = xi_j / xi_i >= 1, since j is the higher-frequency filter of the pair.
        "sel_power": (xi_j / xi_i).astype(np.float32),
        "sel_band_hz": np.asarray(band, dtype=np.float32),
        "sel_k_steps": np.asarray(K_STEPS, dtype=np.int32),
    }


def write_shard(path) -> str:
    """Write a shard carrying the real provenance at the real widths.

    Args:
        path: Destination file.

    Returns:
        The path written, as a string.
    """
    import h5py

    with h5py.File(str(path), "w") as handle:
        for name, width in (("fhr_ph", N_FHR_PHASE), ("up_ph", N_UP_PHASE)):
            node = handle.create_dataset(name, shape=(2, width, 330), dtype="f4")
            for key, value in selection_attrs(name).items():
                node.attrs[key] = value
        handle.create_dataset("fhr_st", shape=(2, N_SCATTERING, 330), dtype="f4")
    return str(path)
