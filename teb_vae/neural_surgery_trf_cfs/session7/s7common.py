r"""Shared paths, cohorts, split codes and statistics of every session-7 probe.

Every session-7 script reads the same checkpoint, the same three local shards and the same
stand-in statistics. This module is the one place where those choices are written, so two probes
cannot disagree about the split, the cohort labels or the bootstrap.

**The split.** A recording has the code $h = \operatorname{crc32}(\mathrm{guid}) \bmod 4$.
Code 0 is the *fit* set, code 2 is the *select* set, and codes 1 and 3 are the *test* set. The
test set stays closed until the final s7 decision.

**Training membership.** The epoch-424 run trained on the server folders
``pre_training_dataset/train/healthy_*`` with no GUID filter. The dataset builder fills those
folders with healthy recordings only, drawn by a seeded 90/10 shuffle on the server. Two
consequences follow for the local shards:

* ``acidosis_no_cs`` and ``hie_no_cs`` hold unhealthy recordings, which the builder never puts in
  the pre-training set. They are **unseen** with certainty.
* ``healthy_no_bg_no_cs`` was cut from a server shard whose folder (``train`` or ``test``) is not
  recorded. Its membership cannot be proven here, so it carries the label **unknown** and every
  result on it is reported as "possibly seen".

**The statistics.** The training ``stats.hdf5`` is not local. The loader normalises with stand-in
statistics, so an absolute value in nats is approximate. A difference between two arms under the
same statistics is the quantity to report.

**The bootstrap.** :func:`paired_bootstrap` resamples recordings, not anchors, because anchors of
one recording are dependent. It returns the mean paired difference and a percentile interval.
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

PKG = Path(__file__).resolve().parent
REPO = PKG.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RUN_DIR = REPO / "tmp" / "checkpoint-2026-11-09"
CHECKPOINT = RUN_DIR / "lag-attn-trf-cfs-epoch=424.ckpt"
CONFIG = RUN_DIR / "resolved_config.yaml"
OVERRIDES = RUN_DIR / "eval_overrides_local.yaml"
EVAL_DIR = RUN_DIR / "eval" / "eval_results"
SHARDS: Dict[str, Path] = {
    "healthy": RUN_DIR / "healthy_no_bg_no_cs.hdf5",
    "acidosis": RUN_DIR / "acidosis_no_cs.hdf5",
    "hie": RUN_DIR / "hie_no_cs.hdf5",
}
STATS = PKG.parent / "assets" / "stats_standin_acidosis.hdf5"
OUT = PKG / "out"

#: Training membership per cohort; see the module docstring for the evidence.
MEMBERSHIP: Dict[str, str] = {"healthy": "unknown", "acidosis": "unseen", "hie": "unseen"}

#: Split code of each role.
FIT_CODE, SELECT_CODE, TEST_CODES = 0, 2, (1, 3)

#: Raw sampling rate (Hz), raw samples per feature step, and seconds per feature step.
FS, RAW_PER_STEP, STEP_S = 4, 16, 4.0

#: Bits of ``up_quality`` that the dataset builder counts as invalid (attribute ``invalid_bits``).
UP_BITS: Dict[str, int] = {
    "MISSING": 1, "CODE": 2, "CENSORED": 4, "TRANSIENT": 8, "STEP": 16, "HOLD": 32, "FLOOR": 64,
    "ISLAND": 128, "OSCILLATION": 256, "REST": 512, "PUSH": 1024, "FLAT": 2048,
}
UP_INVALID_BITS = 2299


def guid_text(guid: Any) -> str:
    """Decode a GUID as the loader or an HDF5 column yields it.

    Args:
        guid: A ``str`` or ``bytes`` GUID.

    Returns:
        The GUID as ``str``.
    """
    return guid.decode() if isinstance(guid, bytes) else str(guid)


def guid_hash(guid: Any) -> int:
    r"""Split code of one recording, $\operatorname{crc32}(\mathrm{guid}) \bmod 4$.

    Args:
        guid: The recording GUID.

    Returns:
        An integer in $\{0, 1, 2, 3\}$: 0 fit, 2 select, 1 and 3 test.
    """
    return zlib.crc32(guid_text(guid).encode()) % 4


def split_role(code: int) -> str:
    """Name the role of a split code.

    Args:
        code: A value of :func:`guid_hash`.

    Returns:
        ``'fit'``, ``'select'`` or ``'test'``.
    """
    return {FIT_CODE: "fit", SELECT_CODE: "select"}.get(int(code), "test")


def cohort_of_guids() -> Dict[str, str]:
    """Map every local GUID to its cohort name.

    Returns:
        ``{guid: 'healthy' | 'acidosis' | 'hie'}`` over the three local shards.
    """
    import h5py

    table: Dict[str, str] = {}
    for cohort, path in SHARDS.items():
        with h5py.File(path, "r") as handle:
            for guid in handle["guid"][:]:
                table[guid_text(guid)] = cohort
    return table


def paired_bootstrap(
    recording: Sequence[Any],
    delta: Sequence[float],
    weight: Optional[Sequence[float]] = None,
    n_resamples: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    r"""Recording-level paired bootstrap of a mean per-anchor difference.

    Each row is one anchor with a paired difference $\delta_i$ between two arms. The statistic
    is the pooled mean $\bar\delta = \sum_i w_i \delta_i / \sum_i w_i$. A resample draws
    recordings with replacement and keeps every anchor of a drawn recording, so the interval
    respects the dependence between anchors of one recording.

    Args:
        recording: Recording identifier per row.
        delta: Paired difference per row.
        weight: Optional row weight $w_i$; ``None`` weights every row equally.
        n_resamples: Number of bootstrap resamples.
        seed: Seed of the resampling generator.

    Returns:
        ``{'mean', 'lo', 'hi', 'n_rec', 'n_rows', 'p_le0'}`` where ``lo``/``hi`` bound the 95%
        percentile interval and ``p_le0`` is the share of resamples at or below zero.
    """
    delta = np.asarray(delta, dtype=np.float64)
    weight = np.ones_like(delta) if weight is None else np.asarray(weight, dtype=np.float64)
    keep = np.isfinite(delta) & np.isfinite(weight)
    delta, weight = delta[keep], weight[keep]
    _, inverse = np.unique(np.asarray(recording)[keep], return_inverse=True)
    n_rec = int(inverse.max()) + 1 if inverse.size else 0
    if n_rec == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_rec": 0,
                "n_rows": 0, "p_le0": float("nan")}
    num = np.bincount(inverse, weights=delta * weight, minlength=n_rec)
    den = np.bincount(inverse, weights=weight, minlength=n_rec)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_rec, size=(n_resamples, n_rec))
    means = num[draws].sum(axis=1) / np.maximum(den[draws].sum(axis=1), 1e-12)
    return {
        "mean": float(num.sum() / max(den.sum(), 1e-12)),
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
        "n_rec": n_rec,
        "n_rows": int(delta.size),
        "p_le0": float((means <= 0).mean()),
    }


def format_ci(result: Dict[str, float], digits: int = 3) -> str:
    """Format a :func:`paired_bootstrap` result as ``mean [lo, hi] (n recordings)``.

    Args:
        result: The dictionary :func:`paired_bootstrap` returns.
        digits: Decimal places.

    Returns:
        The formatted string.
    """
    return (f"{result['mean']:.{digits}f} [{result['lo']:.{digits}f}, {result['hi']:.{digits}f}] "
            f"(n={result['n_rec']} rec)")


def load_model(checkpoint: Path = CHECKPOINT, config: Path = CONFIG, overrides: Path = OVERRIDES,
               device: str = "cuda:0") -> Tuple[Any, dict, Any]:
    """Load a checkpoint through the evaluation pipeline's strict loader.

    Args:
        checkpoint: Checkpoint to load.
        config: The run's resolved configuration.
        overrides: Evaluation override file with the local shards and stand-in statistics.
        device: Torch device string.

    Returns:
        ``(device, config, task)``; ``task.orig_model`` is the network, in evaluation mode.
    """
    from teb_vae.neural_surgery_trf_cfs import common

    return common.setup(checkpoint=checkpoint, config=config, overrides=overrides, device=device)


def iter_batches(task: Any, cfg: dict, dev: Any, limit: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    """Iterate the evaluation batches on the device, in loader order.

    Args:
        task: The loaded task.
        cfg: The merged configuration.
        dev: Target device.
        limit: Stop after this many batches; ``None`` reads every shard.

    Returns:
        An iterator of batch dictionaries.
    """
    from teb_vae.neural_surgery_trf_cfs import common

    return common.batches(task, cfg, dev, limit=limit)
