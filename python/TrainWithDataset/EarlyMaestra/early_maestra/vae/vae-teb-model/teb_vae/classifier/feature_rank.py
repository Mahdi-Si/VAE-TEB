r"""Per-channel spread and effective rank of a frozen feature cache (SPEC §8.4, §8.5; the 2026-10-07 diagnosis).

    python -m teb_vae.classifier.feature_rank --cache-dir runs/classifier_cache/<hash> \
        [--sample-segments 4096] [--sample-steps 200000] [--seed 0] [--out report.json]

or edit :data:`RUN_ARGS` at the bottom and press Run. Reads ``features.h5`` (``values`` (N, T', C), ``attn``,
``step_mask``, the ``channels`` attribute) and reports, per key (the channel name before ``[``), over a random sample
of valid steps:

* the per-channel standard deviation (min, median, max) and how many channels sit under the scale floor the scaler
  would use, ``max(1e-3, 0.1 x median positive std over every channel)`` (§8.5): the 2026-10-07 cache had 60 of 64
  ``delta_mu`` channels there;
* the effective rank of the key's block, ``exp(entropy)`` of the normalised eigenvalues of its correlation matrix
  (channels z-scored) and of its raw covariance, and the components that carry 90 % and 99 % of the variance.

The sample is uniform over segments, then over their valid steps, so a long recording weighs by its segments. Torch-free.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if not __package__:  # run as a file (IDE Run button): the repo root first, never this directory
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _SCRIPT_DIR]
    if _REPO_ROOT in sys.path:
        sys.path.remove(_REPO_ROOT)
    sys.path.insert(0, _REPO_ROOT)

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from teb_vae.classifier.config import resolve_path  # noqa: E402
from teb_vae.lag_attn_cfs.eval.launch import resolve_launch_args  # noqa: E402


def sample_steps(cache_dir: Path, *, sample_segments: int, sample_steps: int, seed: int) -> tuple:
    """``(X (M, C) float64, channels)``: a uniform sample of valid steps of a uniform sample of cached segments."""
    rng = np.random.default_rng(seed)
    with h5py.File(cache_dir / "features.h5", "r") as h5:
        n = h5["values"].shape[0]
        rows = np.sort(rng.choice(n, min(sample_segments, n), replace=False))
        values, mask = h5["values"][rows].astype(np.float32), h5["step_mask"][rows]
        if "attn" in h5:
            values = np.concatenate([values, h5["attn"][rows].astype(np.float32)], -1)
        channels = list(json.loads(h5.attrs["channels"]))
    x = values[mask]
    if len(x) > sample_steps:
        x = x[np.sort(rng.choice(len(x), sample_steps, replace=False))]
    return x.astype(np.float64), channels


def spectrum(x: np.ndarray) -> Dict[str, Any]:
    """Effective rank and the 90 % / 99 % component counts of ``x``'s correlation and covariance spectra."""
    out: Dict[str, Any] = {}
    std = x.std(0)
    for name, z in (("corr", (x - x.mean(0)) / np.where(std > 0, std, 1.0)), ("cov", x - x.mean(0))):
        w = np.clip(np.linalg.eigvalsh(np.cov(z, rowvar=False).reshape(z.shape[1], z.shape[1])), 0.0, None)[::-1]
        p = w / w.sum() if w.sum() > 0 else np.full_like(w, 1.0 / len(w))
        cum = np.cumsum(p)
        out[name] = {"effective_rank": float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())),
                     "n_90": int(np.searchsorted(cum, 0.90) + 1), "n_99": int(np.searchsorted(cum, 0.99) + 1),
                     "top_share": float(p[0])}
    return out


def report(cache_dir: Any, *, sample_segments: int = 4096, sample_steps_: int = 200_000, seed: int = 0
           ) -> Dict[str, Any]:
    """The per-key table (:func:`spectrum` and the spread) of ``cache_dir``."""
    cache_dir = resolve_path(cache_dir)
    x, channels = sample_steps(cache_dir, sample_segments=sample_segments, sample_steps=sample_steps_, seed=seed)
    std_all = x.std(0)
    positive = std_all[std_all > 0]
    floor = max(1e-3, 0.1 * float(np.median(positive))) if positive.size else 1e-3
    keys: Dict[str, List[int]] = {}
    for i, name in enumerate(channels):
        keys.setdefault(name.split("[")[0], []).append(i)
    table = {}
    for key, idx in keys.items():
        s = std_all[idx]
        table[key] = {"n_channels": len(idx), "std_min": float(s.min()), "std_median": float(np.median(s)),
                      "std_max": float(s.max()), "n_under_floor": int((s < floor).sum()),
                      "n_under_10pct_of_key_median": int((s < 0.1 * np.median(s)).sum()),
                      **(spectrum(x[:, idx]) if len(idx) > 1 else {})}
    return {"cache_dir": str(cache_dir), "n_steps": int(len(x)), "n_channels": len(channels),
            "scaler_floor": floor, "keys": table}


def _print(rep: Dict[str, Any]) -> None:
    print(f"cache {rep['cache_dir']}: {rep['n_steps']} sampled valid steps, {rep['n_channels']} channels, "
          f"scaler floor {rep['scaler_floor']:.4g}")
    head = f"{'key':24s} {'C':>4s} {'std min':>9s} {'std med':>9s} {'std max':>9s} {'<floor':>7s} {'<10%med':>8s} " \
           f"{'erank corr':>11s} {'n90':>4s} {'n99':>4s} {'erank cov':>10s} {'n90':>4s} {'n99':>4s} {'top':>6s}"
    print(head)
    for key, t in rep["keys"].items():
        c, v = t.get("corr", {}), t.get("cov", {})
        print(f"{key:24s} {t['n_channels']:4d} {t['std_min']:9.4f} {t['std_median']:9.4f} {t['std_max']:9.4f} "
              f"{t['n_under_floor']:7d} {t['n_under_10pct_of_key_median']:8d} "
              f"{c.get('effective_rank', float('nan')):11.2f} {c.get('n_90', 0):4d} {c.get('n_99', 0):4d} "
              f"{v.get('effective_rank', float('nan')):10.2f} {v.get('n_90', 0):4d} {v.get('n_99', 0):4d} "
              f"{v.get('top_share', float('nan')):6.2f}")


def main(*, cache_dir: str, sample_segments: int = 4096, sample_steps: int = 200_000, seed: int = 0,
         out: Optional[str] = None) -> int:
    rep = report(cache_dir, sample_segments=int(sample_segments), sample_steps_=int(sample_steps), seed=int(seed))
    _print(rep)
    if out:
        Path(out).write_text(json.dumps(rep, indent=2))
        print(f"written {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Per-channel spread and effective rank of a feature cache (SPEC §8.5).")
    parser.add_argument("--cache-dir", dest="cache_dir", help="the <cache_root>/<fingerprint hash>/ directory")
    parser.add_argument("--sample-segments", dest="sample_segments", type=int)
    parser.add_argument("--sample-steps", dest="sample_steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", help="write the report as JSON here")
    return parser


def _cli(argv: Optional[List[str]] = None) -> int:
    values, _ = resolve_launch_args(build_parser(), RUN_ARGS, sys.argv[1:] if argv is None else list(argv))
    return main(**{key: value for key, value in values.items() if value is not None})


#: Arguments for an argument-less launch (the IDE Run button): edit the values below and press Run.
RUN_ARGS: Dict[str, Any] = {
    # The cache directory: <source.cache_root>/<fingerprint hash> (manifest.json["source"]["cache_dir"] of a run).
    "cache_dir": "runs/classifier_cache/5586c324d1c35b27",
    # Segments sampled uniformly, then valid steps sampled from them.
    "sample_segments": 4096,
    "sample_steps": 200000,
    "seed": 0,
    # None prints only; a path also writes the JSON report there.
    "out": None,
}


if __name__ == "__main__":
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        os.chdir(_REPO_ROOT)
    sys.exit(_cli())
