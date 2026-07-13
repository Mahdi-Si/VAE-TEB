"""Subgroup discovery and identity helpers for the two-phase test pipeline.

The lag-attn v1 testing pipeline can run in *subgroup mode*, where each
subgroup is one of the 8 canonical CTG bg/cs cells (e.g. ``hie_no_cs``,
``healthy_bg_cs``) and the standard pipeline runs once per subgroup
(Phase 1) before a cross-subgroup post-processor produces overlays
(Phase 2).

This module owns the small amount of glue needed to:

- resolve a ``{name: [hdf5_paths]}`` mapping from any of (a) an explicit
  Python dict, (b) the YAML key ``dataset_config.vae_test_subgroups``,
  or (c) auto-glob ``*.hdf5`` from a fold/test directory using the
  filename stem as the subgroup name;
- ship a canonical iteration order so plots and CSVs are deterministic;
- ship a colour palette and label-id mapping derived from the existing
  ``CLASS_COLORS`` / ``CLASS_NAMES`` so cross-subgroup plots stay
  visually consistent with the per-class plots elsewhere in the pipeline.
"""

from __future__ import annotations

import glob
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import yaml

from model.vae_teb_prediction.model.model_raw.testing.visualizers import COLOR_GRAY

# ---------------------------------------------------------------------------
# Canonical subgroup identity
# ---------------------------------------------------------------------------

#: Iteration order for the 8 canonical subgroups. Healthy first (4
#: bg/cs cells), then acidosis (2), then HIE (2). The ordering matches
#: the clinical-severity gradient and keeps overlay legends readable.
CANONICAL_ORDER: tuple = (
    "healthy_no_bg_no_cs",
    "healthy_no_bg_cs",
    "healthy_bg_no_cs",
    "healthy_bg_cs",
    "acidosis_no_cs",
    "acidosis_cs",
    "hie_no_cs",
    "hie_cs",
)

#: Outcome-class id (1=HEALTHY, 2=ACIDOSIS, 3=HIE) for each canonical
#: subgroup. Phase 2 uses this to route stats / shading and to validate
#: the Phase 1 output (``loader_probe.json::per_label_counts``) against
#: the expected class.
SUBGROUP_TO_LABEL: Dict[str, int] = {
    "healthy_no_bg_no_cs": 1,
    "healthy_no_bg_cs":    1,
    "healthy_bg_no_cs":    1,
    "healthy_bg_cs":       1,
    "acidosis_no_cs":      2,
    "acidosis_cs":         2,
    "hie_no_cs":           3,
    "hie_cs":              3,
}

#: Per-subgroup display colour. The 8 values are the **Okabe-Ito**
#: colourblind-safe palette, ordered so outcome-family blocks stay
#: visually grouped (healthy = cool/yellow, acidosis = warm,
#: HIE = dark / red). Maximises pairwise separability across all 8
#: subgroups in cross-subgroup overlays.
SUBGROUP_COLORS: Dict[str, str] = {
    # HEALTHY family — blue / sky / bluish green / yellow
    "healthy_no_bg_no_cs": "#0072B2",   # blue
    "healthy_no_bg_cs":    "#56B4E9",   # sky blue
    "healthy_bg_no_cs":    "#009E73",   # bluish green
    "healthy_bg_cs":       "#F0E442",   # yellow
    # ACIDOSIS family — orange + reddish purple
    "acidosis_no_cs":      "#E69F00",   # orange
    "acidosis_cs":         "#CC79A7",   # reddish purple
    # HIE family — vermillion + black
    "hie_no_cs":           "#D55E00",   # vermillion
    "hie_cs":              "#000000",   # black
}

#: Fallback colour for non-canonical / user-supplied subgroup names.
SUBGROUP_FALLBACK_COLOR: str = COLOR_GRAY


def color_for_subgroup(name: str) -> str:
    """Return the display colour for ``name``, with a graceful fallback."""
    return SUBGROUP_COLORS.get(name, SUBGROUP_FALLBACK_COLOR)


def label_for_subgroup(name: str) -> Optional[int]:
    """Return the outcome class id ``{1, 2, 3}`` for a canonical subgroup,
    or ``None`` for non-canonical names."""
    return SUBGROUP_TO_LABEL.get(name)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _coerce_paths(value: Any) -> List[str]:
    """Normalise a YAML / dict value into a list of string paths."""
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, (list, tuple)):
        return [str(p) for p in value if p is not None]
    raise TypeError(
        f"Subgroup value must be str | Path | list[str|Path], got {type(value).__name__}"
    )


def _order_canonical(
    raw: Mapping[str, Sequence[Union[str, Path]]],
) -> "OrderedDict[str, List[str]]":
    """Reorder ``raw`` so canonical names appear in :data:`CANONICAL_ORDER`,
    followed by any non-canonical names in their original insertion order.
    Empty entries are dropped. Path lists are coerced to ``List[str]``."""
    out: "OrderedDict[str, List[str]]" = OrderedDict()
    for name in CANONICAL_ORDER:
        if name in raw:
            paths = _coerce_paths(raw[name])
            if paths:
                out[name] = paths
    for name, value in raw.items():
        if name in out:
            continue
        paths = _coerce_paths(value)
        if paths:
            out[name] = paths
    return out


def _resolve_from_explicit(
    explicit: Mapping[str, Sequence[Union[str, Path]]],
) -> "OrderedDict[str, List[str]]":
    return _order_canonical(explicit)


def _resolve_from_yaml(
    config_path: Union[str, Path],
) -> "OrderedDict[str, List[str]]":
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    dataset_cfg = cfg.get("dataset_config", {}) or {}

    # Preferred: explicit subgroup mapping.
    sg_map = dataset_cfg.get("vae_test_subgroups")
    if sg_map:
        return _order_canonical(sg_map)

    # Optional: auto-discovery from a fold/test directory listed in the
    # config under ``vae_test_fold_dir``.
    fold_dir = dataset_cfg.get("vae_test_fold_dir")
    if fold_dir:
        return _resolve_from_fold_dir(fold_dir)

    # Legacy fallback: a flat list of HDF5 paths. Group by the filename
    # stem so a list that already cleanly maps onto the 8 canonical
    # subgroups (the typical case) is handled identically to the
    # explicit dict form. Non-canonical filenames fall through to a
    # single ``"all"`` bucket.
    flat = dataset_cfg.get("vae_test_datasets") or []
    if flat:
        return _resolve_from_flat_list(flat)

    return OrderedDict()


def _resolve_from_fold_dir(
    fold_test_dir: Union[str, Path],
) -> "OrderedDict[str, List[str]]":
    fold_test_dir = Path(fold_test_dir)
    if not fold_test_dir.is_dir():
        raise FileNotFoundError(f"fold_test_dir is not a directory: {fold_test_dir}")
    paths = sorted(glob.glob(str(fold_test_dir / "*.hdf5")))
    if not paths:
        return OrderedDict()
    grouped: Dict[str, List[str]] = {}
    for p in paths:
        stem = Path(p).stem
        grouped.setdefault(stem, []).append(p)
    return _order_canonical(grouped)


def _resolve_from_flat_list(
    paths: Sequence[Union[str, Path]],
) -> "OrderedDict[str, List[str]]":
    """Group a flat list of HDF5 paths by filename stem.

    When every stem matches a canonical subgroup, we get a clean 8-cell
    map; otherwise unrecognised stems fall through to a single ``"all"``
    bucket — the same behaviour the legacy flat-list path provided.
    """
    grouped: Dict[str, List[str]] = {}
    misc: List[str] = []
    for p in paths:
        stem = Path(str(p)).stem
        if stem in SUBGROUP_TO_LABEL:
            grouped.setdefault(stem, []).append(str(p))
        else:
            misc.append(str(p))
    if not grouped:
        # Nothing matched — return a single anonymous subgroup so Phase 2
        # is auto-skipped.
        return OrderedDict([("all", [str(p) for p in paths])])
    if misc:
        grouped["misc"] = misc
    return _order_canonical(grouped)


def resolve_subgroups(
    *,
    explicit: Optional[Mapping[str, Sequence[Union[str, Path]]]] = None,
    config_path: Optional[Union[str, Path]] = None,
    fold_test_dir: Optional[Union[str, Path]] = None,
) -> "OrderedDict[str, List[str]]":
    """Build a ``{subgroup_name: [hdf5_paths]}`` mapping from one of three
    sources, in priority order.

    Args:
        explicit: When provided, used verbatim (after canonical reorder).
        config_path: YAML file. The first non-empty key wins among
            ``dataset_config.vae_test_subgroups``,
            ``dataset_config.vae_test_fold_dir`` (auto-glob), and
            ``dataset_config.vae_test_datasets`` (legacy flat list, grouped
            by filename stem).
        fold_test_dir: When provided, ``*.hdf5`` is auto-globbed from
            this directory and the filename stem becomes the subgroup
            name. Used when neither ``explicit`` nor ``config_path`` is
            given (or as a manual override).

    Returns:
        ``OrderedDict[str, List[str]]`` ordered by :data:`CANONICAL_ORDER`,
        with non-canonical names appended afterwards. Empty entries are
        dropped. Returns an empty mapping when nothing resolves —
        callers should treat that as an error condition.
    """
    if explicit:
        return _resolve_from_explicit(explicit)
    if config_path is not None:
        out = _resolve_from_yaml(config_path)
        if out:
            return out
    if fold_test_dir is not None:
        return _resolve_from_fold_dir(fold_test_dir)
    return OrderedDict()


__all__ = [
    "CANONICAL_ORDER",
    "SUBGROUP_TO_LABEL",
    "SUBGROUP_COLORS",
    "SUBGROUP_FALLBACK_COLOR",
    "color_for_subgroup",
    "label_for_subgroup",
    "resolve_subgroups",
]
