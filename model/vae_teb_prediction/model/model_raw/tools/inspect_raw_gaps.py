r"""Inspect how gaps/dropouts appear in raw ``fhr``/``up`` of a real HDF5 fold (S1-T07).

The v4 raw mask (Sprint 3) derives per-sample validity from the decimated ``weight`` (upsampled by
$16$) refined with the raw gap convention. This script opens a real fold, reports how gaps present
in the raw signal (exact ``0.0``, ``NaN``/``Inf``, held/flat runs) and how they align with the
decimated ``weight``, and prints the inferred :data:`raw_mask_constants.GAP_ENCODING` /
:data:`raw_mask_constants.SENTINEL` so a mismatch with the seeded constants is caught.

Usage::

    python -m model.vae_teb_prediction.model.model_raw.tools.inspect_raw_gaps --fold <path.hdf5>

Prod fold paths are Linux-only (``/data1/...``); locally, point ``--fold`` at a synthetic HDF5
built by ``hdf5_dataset/guid_hdf5_dataset.py`` (identical schema).
"""
from __future__ import annotations

import argparse
from typing import Optional

import numpy as np

from model.vae_teb_prediction.model.model_raw import raw_mask_constants as rmc


def _longest_run(mask: np.ndarray) -> int:
    """Return the longest run of ``True`` in a 1-D boolean array."""
    if mask.size == 0:
        return 0
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def _load_field(h5, name: str, num_samples: int) -> Optional[np.ndarray]:
    """Best-effort load of the first ``num_samples`` rows of a raw field.

    Handles both the flat ``(N, L)`` dataset layout and a per-sample group layout.
    """
    if name not in h5:
        return None
    obj = h5[name]
    # Flat dataset (N, L).
    if hasattr(obj, "shape") and getattr(obj, "ndim", 0) >= 1:
        n = min(num_samples, obj.shape[0])
        return np.asarray(obj[:n])
    # Group of per-sample datasets.
    try:
        keys = list(obj.keys())[:num_samples]
        rows = [np.asarray(obj[k]) for k in keys]
        return np.stack(rows) if rows else None
    except Exception:  # noqa: BLE001 - best-effort inspection tool
        return None


def inspect(fold_path: str, *, num_samples: int = 64) -> dict:
    """Analyse raw ``fhr``/``up`` gap encoding in a fold and return a summary dict.

    Args:
        fold_path: Path to an HDF5 fold file.
        num_samples: Number of rows to sample.

    Returns:
        A summary dict per field with gap statistics and the inferred convention.

    Raises:
        ImportError: If ``h5py`` is unavailable.
        FileNotFoundError: If ``fold_path`` cannot be opened.
    """
    import h5py  # local import so importing this module never requires h5py

    summary: dict = {"fold": fold_path, "fields": {}}
    with h5py.File(fold_path, "r") as h5:
        weight = _load_field(h5, "weight", num_samples)
        for field in ("fhr", "up"):
            arr = _load_field(h5, field, num_samples)
            if arr is None:
                summary["fields"][field] = {"present": False}
                continue
            arr = arr.astype(np.float64)
            total = arr.size
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            n_zero = int((arr == 0.0).sum())
            # Held/flat runs (per row), excluding zeros already counted.
            flat_rows = []
            for row in arr:
                d = np.diff(row)
                flat_rows.append(_longest_run(d == 0))
            longest_flat = int(max(flat_rows)) if flat_rows else 0

            # Alignment: does a zero-fhr sample coincide with weight==0 (upsampled by 16)?
            # NOTE: this row-indexed comparison is only meaningful for the flat (N, L) dataset
            # layout (the production layout, where row i of fhr and weight are the same sample). For
            # the per-sample group fallback, the two groups may enumerate keys in different orders,
            # so treat this stat as diagnostic only.
            align = None
            if weight is not None and weight.shape[0] >= arr.shape[0]:
                w = weight[: arr.shape[0]].astype(np.float64)
                if w.shape[1] * 16 == arr.shape[1]:
                    w_up = np.repeat(w > 0, 16, axis=1)  # valid where weight>0
                    zero_mask = arr == 0.0
                    # Of the zero-valued raw samples, what fraction are also weight-invalid?
                    denom = int(zero_mask.sum())
                    align = (
                        float((zero_mask & ~w_up).sum() / denom) if denom else None
                    )

            summary["fields"][field] = {
                "present": True,
                "shape": tuple(arr.shape),
                "frac_nan": n_nan / total,
                "frac_inf": n_inf / total,
                "frac_zero": n_zero / total,
                "longest_flat_run": longest_flat,
                "frac_zero_also_weight_invalid": align,
            }
    return summary


def _infer_encoding(field_summary: dict) -> str:
    """Infer the **dominant** gap encoding from a field summary (``nan`` / ``zero`` / ``held``).

    Uses the larger of the non-finite and zero fractions rather than mere presence: a fold that is
    overwhelmingly zero-encoded plus a handful of stray non-finite samples must infer ``zero``, not
    ``nan`` (which would wrongly push the operator to flip the masking convention away from the
    verified ``SENTINEL=0.0``).
    """
    if not field_summary.get("present"):
        return "unknown"
    frac_nonfinite = field_summary["frac_nan"] + field_summary["frac_inf"]
    frac_zero = field_summary["frac_zero"]
    if frac_nonfinite == 0.0 and frac_zero == 0.0:
        return "held" if field_summary["longest_flat_run"] > 0 else "none"
    return "nan" if frac_nonfinite > frac_zero else "zero"


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Inspect raw fhr/up gap encoding in an HDF5 fold.")
    parser.add_argument("--fold", required=True, help="Path to an HDF5 fold file.")
    parser.add_argument("--num-samples", type=int, default=64, help="Rows to sample.")
    args = parser.parse_args()

    summary = inspect(args.fold, num_samples=args.num_samples)
    print(f"Fold: {summary['fold']}")
    for field, fs in summary["fields"].items():
        if not fs.get("present"):
            print(f"  {field}: (absent)")
            continue
        inferred = _infer_encoding(fs)
        print(f"  {field}: shape={fs['shape']}")
        print(
            f"    frac_nan={fs['frac_nan']:.3e} frac_inf={fs['frac_inf']:.3e} "
            f"frac_zero={fs['frac_zero']:.3e} longest_flat_run={fs['longest_flat_run']}"
        )
        if fs["frac_zero_also_weight_invalid"] is not None:
            print(
                f"    of zero-valued samples, {fs['frac_zero_also_weight_invalid']:.1%} "
                "are also weight-invalid"
            )
        print(f"    inferred GAP_ENCODING = {inferred!r}")
        if inferred not in ("unknown", "none") and inferred != rmc.GAP_ENCODING:
            print(
                f"    WARNING: inferred {inferred!r} != seeded raw_mask_constants.GAP_ENCODING "
                f"{rmc.GAP_ENCODING!r} -- update raw_mask_constants.py."
            )
    print(
        f"Seeded convention: GAP_ENCODING={rmc.GAP_ENCODING!r}, SENTINEL={rmc.SENTINEL!r} "
        "(confirm on a prod fold)."
    )


if __name__ == "__main__":
    main()
