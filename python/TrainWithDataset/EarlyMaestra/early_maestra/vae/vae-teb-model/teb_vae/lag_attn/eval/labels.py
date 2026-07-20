r"""Which clinical class and which canonical subgroup each sample belongs to.

Two identities, recovered from fields already on every batch, and both becoming **columns on the
per-sample DataFrame** rather than a second pass over the loader. `guid`, `target`, `weight` and
`source_file_basename` all arrive with the batch, so a by-class or by-subgroup number is a
``groupby`` after collection -- the same reasoning that ruled out a per-GUID DataLoader.

**The class is a ratio, not a value.** ``target`` is the class code *scaled by the per-step
validity* ``weight``, so a partially-valid step of an acidosis recording ($\mathrm{code} = 2$) at
$\mathrm{weight} = 0.5$ stores $1.0$ -- which is indistinguishable from a fully-valid healthy step.
Reading ``target`` directly therefore mislabels exactly the boundaries of every segment. Dividing
by ``weight`` where it is non-zero recovers the code on every step it covers.

**The dataset's own ``label`` filter is deliberately not used.** It tests exact float equality
against ``target``, so it matches only timesteps where ``weight`` is exactly $1.0$ and silently
drops every partially-masked segment. The eval config turns it off (``label: null``) and this
module does the work instead; :func:`clinical_class_code` is written so that the fractional case
the filter would have dropped is the one it handles correctly.

**Absent is not zero.** A sample whose ``weight`` is zero everywhere -- a pad-only window -- has no
class, and so does one whose ``target`` is uniformly zero, which is what the healthy-only
pretraining split writes. Both yield ``None``. There is no class $0$, and reporting one would
create a phantom cohort that every by-class table would then carry.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from loguru import logger

#: Clinical class code to name, as ``create_new_pipeline.py::SUBGROUP_META`` assigns them.
#: Restated rather than imported: ``teb_vae`` does not depend on ``hdf5_dataset``, and this is
#: three lines against a package dependency.
CLASS_NAMES: Dict[int, str] = {1: "healthy", 2: "acidosis", 3: "hie"}

#: The eight canonical subgroups the k-fold test split is written as, one HDF5 each. A basename
#: outside this set is not an error -- the pretraining split has its own names -- but it is worth
#: saying out loud once, because a typo in ``vae_test_datasets`` presents the same way.
CANONICAL_SUBGROUPS: Tuple[str, ...] = (
    "healthy_no_bg_no_cs",
    "healthy_no_bg_cs",
    "healthy_bg_no_cs",
    "healthy_bg_cs",
    "acidosis_no_cs",
    "acidosis_cs",
    "hie_no_cs",
    "hie_cs",
)

#: Column names this module contributes to a per-sample frame. Named here so the collectors, the
#: grouped emitter and the cross-subgroup analysis all agree without any of them hardcoding a
#: string that another could change.
CLASS_COLUMN = "clinical_class"
SUBGROUP_COLUMN = "subgroup"

#: Group columns a grouped variant is emitted over, in output order.
GROUP_COLUMNS: Tuple[str, ...] = (CLASS_COLUMN, SUBGROUP_COLUMN)

#: Basenames already warned about, so an unrecognised shard costs one log line rather than one
#: per sample -- twenty thousand identical warnings would bury the rest of the run's log.
_WARNED_UNKNOWN: Set[str] = set()


def clinical_class_code(target_row: Any, weight_row: Any) -> Optional[int]:
    r"""Recover one sample's clinical class code from its scaled target.

    $$\mathrm{code} = \mathrm{round}\!\left(\frac{\mathrm{target}_t}{\mathrm{weight}_t}\right),
    \qquad \mathrm{weight}_t > 0$$

    taken as the most common value over the steps where it is defined. Most common rather than
    first: the code is constant over the steps it covers, so any disagreement is numerical, and a
    single anomalous step should not decide a recording's cohort.

    Args:
        target_row: One sample's per-step target, $(T,)$.
        weight_row: One sample's per-step validity, $(T,)$.

    Returns:
        The class code, or ``None`` when the sample carries no class -- a pad-only window, or a
        uniformly zero target as the healthy-only pretraining split writes.
    """
    target = _as_array(target_row).ravel()
    weight = _as_array(weight_row).ravel()
    if target.size == 0 or weight.size == 0:
        return None
    length = min(target.size, weight.size)
    target, weight = target[:length], weight[:length]

    usable = np.isfinite(target) & np.isfinite(weight) & (weight > 0.0)
    if not usable.any():
        return None

    codes = np.round(target[usable] / weight[usable]).astype(np.int64)
    # Zero is "no class", not a class: there is no code 0, and the pretraining split's uniformly
    # zero target would otherwise create a phantom cohort in every by-class table.
    labelled = codes[codes != 0]
    if labelled.size == 0:
        return None
    return int(Counter(labelled.tolist()).most_common(1)[0][0])


def class_name(code: Optional[int]) -> Optional[str]:
    """Return the clinical name for a class code.

    Args:
        code: The class code, or ``None``.

    Returns:
        ``'healthy'`` / ``'acidosis'`` / ``'hie'``; ``None`` for ``None``; and ``'class_<n>'`` for
        a code outside the known table, which is reported rather than dropped -- an unknown code
        is a dataset question, and silently discarding it would hide it.
    """
    if code is None:
        return None
    return CLASS_NAMES.get(int(code), f"class_{int(code)}")


def subgroup_of(source_file: Optional[str]) -> Optional[str]:
    """Return the canonical subgroup a shard basename names.

    Args:
        source_file: The ``source_file_basename`` stamped on the sample, with or without its
            directory and its ``.hdf5`` suffix.

    Returns:
        The subgroup, or ``None`` when the name is not one of the canonical eight. An
        unrecognised name warns once per distinct name.
    """
    if not source_file:
        return None
    stem = Path(str(source_file)).name
    for suffix in (".hdf5", ".h5"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    if stem in CANONICAL_SUBGROUPS:
        return stem
    if stem not in _WARNED_UNKNOWN:
        _WARNED_UNKNOWN.add(stem)
        logger.warning(
            f"shard {stem!r} is not one of the canonical subgroups {list(CANONICAL_SUBGROUPS)}, "
            f"so its samples carry no subgroup and are excluded from every by-subgroup table. "
            f"Expected on the pretraining split; on the k-fold split it means "
            f"dataset_config.vae_test_datasets points somewhere else."
        )
    return None


def batch_labels(batch: Any, batch_size: int) -> Dict[str, List[Optional[str]]]:
    """Return the per-sample class and subgroup columns for one batch.

    Args:
        batch: A batch from the data module, already on the compute device.
        batch_size: The batch's sample count, taken from the batch rather than from any column.

    Returns:
        :data:`CLASS_COLUMN` and :data:`SUBGROUP_COLUMN`, each a list of ``batch_size`` entries.
        A batch carrying no ``target`` or no ``weight`` yields a column of ``None`` rather than
        raising: the class axis is optional, and a run over a split without labels should produce
        pooled output, not a failure.
    """
    from teb_vae.lag_attn.eval.runner import get_field

    target = get_field(batch, "target")
    weight = get_field(batch, "weight")
    basenames = get_field(batch, "source_file_basename")

    classes: List[Optional[str]] = []
    subgroups: List[Optional[str]] = []
    for index in range(int(batch_size)):
        if target is None or weight is None:
            classes.append(None)
        else:
            classes.append(class_name(clinical_class_code(target[index], weight[index])))
        subgroups.append(subgroup_of(_basename_at(basenames, index)))

    return {CLASS_COLUMN: classes, SUBGROUP_COLUMN: subgroups}


def distinct_groups(values: Sequence[Any]) -> List[str]:
    """Return the distinct non-null group values, sorted.

    Args:
        values: A group column, possibly holding ``None`` and ``NaN``.

    Returns:
        The distinct labels. ``None`` and ``NaN`` are not groups -- a sample with no class belongs
        to no cohort, and folding them together would create one named after the absence.
    """
    seen = {
        str(value)
        for value in values
        if value is not None and not (isinstance(value, float) and np.isnan(value))
    }
    return sorted(seen)


def _basename_at(basenames: Any, index: int) -> Optional[str]:
    """Read one sample's source basename out of whatever collation produced.

    Args:
        basenames: The batch's ``source_file_basename`` -- a ``list[str]`` after collation, or a
            bare string on a stub batch.
        index: Position within the batch.

    Returns:
        The basename, or ``None``.
    """
    if basenames is None:
        return None
    if isinstance(basenames, (list, tuple)):
        return str(basenames[index]) if index < len(basenames) else None
    return str(basenames)


def _as_array(values: Any) -> np.ndarray:
    """Return ``values`` as a float64 array, accepting a tensor, an array or a sequence."""
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(np.float64)
    return np.asarray(values, dtype=np.float64)
