r"""Measure the UA source and FHR target quality masks on a built dataset, as training reads them.

The transformer cfs model (``teb_vae/lag_attn_transformer_cfs``) reads three masks from a
signal-quality build (``research_notes/UA_QUALITY_MASK_DESIGN.md`` sections 4.10 and 4.11):

* **The source key mask.** The loader's ``ua_step_mask`` gives a validity $v_{s,c}$ per UA
  coefficient. The model ANDs it over the kept source channels $\mathcal K$,
  $m_s = \bigwedge_{c \in \mathcal K} v_{s,c}$, and opens the key at lag $\ell$ of anchor $t$ when
  $t - \ell \ge F_u$ and $m_{t-\ell}$ holds. An anchor with no open key has $q = p$; the training
  metric ``source_mask_fraction`` is the share of KL anchors closed this way.
* **The target mask.** The loader's ``fhr_step_mask`` is ANDed over the kept target channels and
  multiplied into ``weight``. An anchor enters the loss and the KL only when at least
  ``coverage_floor`` of its $H$ forecast steps stay valid.
* **The detector mask** ``up_valid`` (``event_weight_ua_valid``). A contraction whose span touches an
  invalid UA sample leaves the full-branch weight.

For each candidate rule, this script measures what the rule leaves to learn from, per subgroup and
split, on a seeded sample of segments. It writes ``mask_coverage_audit.json`` and a short
``mask_coverage_audit.md`` into ``output_dir``.

**What is reused, and how.** The numbers must be the ones training sees, so no rule is written
again here:

1. One ``CombinedHDF5Dataset`` per file, built with the config's own ``dataset_kwargs`` (filters,
   trim, the shipped ``ua_step_mask`` and ``fhr_step_mask``) and no statistics file. The sampled
   population is therefore the training population.
2. The mask rules live inside that class, so each candidate goes through the loader's own private
   resolvers (``_resolve_ua_step_mask``, ``_resolve_fhr_step_mask``) and mask builders
   (``_ua_step_masks``, ``_fhr_step_masks``, ``_quality_masks``), with the candidate rule swapped in
   on the one open dataset. This is the smallest path that runs the loader's code without opening
   every file once per candidate. A self-check compares the shipped candidate with
   ``dataset[i]`` (the public path) on the first ``verify_segments`` segments of every group, and
   the run stops on any difference.
3. The kept channel sets come from ``resolve_warmup_budget`` run on the audited files, as the
   trainer resolves them.
4. The key mask is ``CausalWarmupInputs.source_key_mask`` itself, called on a stand-in with no
   channel gate and a reach of 1, which is the shipped ``lag_kv_source: adapter`` arm without
   channel alignment. The kept channels are gathered before the call. Other arms are refused.
5. The forecast mask, the coverage floor and the KL support are ``forecast_mask`` and ``kl_mask`` of
   ``teb_vae.lag_attn_rws.nets.raw_masks``. The contraction weight is ``contraction_onset_steps`` and
   ``contraction_anchor_weight`` with the causal-feature detector.

**Dense anchors.** A training step decodes the tile $\{F + \varphi + kS\}$ with a per-segment phase
$\varphi$. Over the phases, every anchor of $[F, T - H)$ enters equally often, so the dense shares
here are the expectation of the per-step training metrics.

**What the output holds.** Counts, shares and quantiles per group only: no GUID, epoch, signal,
per-segment row or path below the dataset root. The user can send both files on.

The stored UP/FHR timeline is canonical: this audit reads the stored grids as they are, and no lag
or index here carries any other term.

Run it from the IDE: edit :data:`RUN_ARGS` at the bottom and press Run. On the command line, a flag
overrides its ``RUN_ARGS`` entry::

    python hdf5_dataset/new_pipeline/mask_coverage_audit.py --root <build root> --output-dir <dir>
"""
from __future__ import annotations

import argparse
import contextlib
import inspect
import io
import json
import logging
import math
import sys
import time
import zlib
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

#: Repository root: ``hdf5_dataset/new_pipeline/mask_coverage_audit.py`` -> up three.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Launched with the Run button, this file's own directory goes on sys.path instead of the
# repository root, and neither the launch helper nor the loader can be found.
if not __package__ and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from teb_vae.lag_attn_rws.eval.launch import missing_required, resolve_launch_args  # noqa: E402

logger = logging.getLogger(__name__)

JSON_NAME = "mask_coverage_audit.json"
MD_NAME = "mask_coverage_audit.md"
PRETRAIN_DIR = "pre_training_dataset"
KFOLD_DIR = "k_fold_cross_validation_dataset"
DEFAULT_CONFIG = "teb_vae/lag_attn_transformer_cfs/configs/default.yaml"

#: The candidate key that stands for the config's own ``dataset_kwargs`` mask.
SHIPPED = "shipped"

#: Source (UA) candidates: name -> ``ua_step_mask`` mapping; ``None`` is the mask off. The
#: ``SHIPPED`` entry is replaced by the config's ``dataset_kwargs.ua_step_mask``. A $\rho$ rule
#: other than the statistics' $\varepsilon$ (0.1) needs its own statistics file in training; here it
#: only measures coverage. ``hold_steps`` 5 is 20 s (about a fast channel's memory), 30 is 120 s.
SOURCE_CANDIDATES: Dict[str, Optional[Dict[str, Any]]] = {
    SHIPPED: None,
    "rho_0.05": {"bits": None, "rho_epsilon": 0.05, "hold_steps": 0},
    "rho_0.20": {"bits": None, "rho_epsilon": 0.2, "hold_steps": 0},
    "rho_0.30": {"bits": None, "rho_epsilon": 0.3, "hold_steps": 0},
    "bits_invalid_hold0": {"bits": ["invalid"], "rho_epsilon": None, "hold_steps": 0},
    "bits_invalid_hold5": {"bits": ["invalid"], "rho_epsilon": None, "hold_steps": 5},
    "bits_invalid_hold30": {"bits": ["invalid"], "rho_epsilon": None, "hold_steps": 30},
    "rho_0.10": {"bits": None, "rho_epsilon": 0.1, "hold_steps": 0},
    "bits_invalid_or_rho_0.10": {"bits": ["invalid"], "rho_epsilon": 0.1, "hold_steps": 0},
    "off": None,
}

#: Target (FHR) candidates: name -> ``fhr_step_mask`` mapping; ``None`` is the mask off.
TARGET_CANDIDATES: Dict[str, Optional[Dict[str, Any]]] = {
    SHIPPED: None,
    "rho_0.10": {"bits": None, "rho_epsilon": 0.1, "hold_steps": 0},
    "bits_invalid_or_rho_0.10": {"bits": ["invalid"], "rho_epsilon": 0.1, "hold_steps": 0},
    "bits_invalid_hold5": {"bits": ["invalid"], "rho_epsilon": None, "hold_steps": 5},
    "off": None,
}

#: Views of the shipped source rule: the conjunction without the $k$ kept source channels that wait
#: longest (largest rebased warm-up $W'_c$, the slowest kernels). A drop is a config change
#: (``causal_source_channels``), not a model change.
DROP_SLOWEST = (2, 4, 8)

#: Quantiles of the per-segment distributions.
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90)

#: Decision thresholds that the report checks; section "Decision checks" of the Markdown file.
THRESHOLDS: Dict[str, float] = {
    # Shipped source_mask_fraction: keep at or below, act above the second value.
    "smf_keep": 0.10,
    "smf_act": 0.20,
    # Share of segments (with a KL anchor) whose every KL anchor the shipped source mask closes.
    "segments_zero_open_max": 0.05,
    # Max - min of the shipped source_mask_fraction over the subgroups of one dataset/split.
    "subgroup_spread_max": 0.05,
    # Drop the slowest 4 source channels when that lowers source_mask_fraction by at least this.
    "drop_slowest_gain_min": 0.05,
    # Target: shipped KL anchors over mask-off KL anchors.
    "target_kept_ratio_min": 0.85,
    # Target: rho 0.10 is affordable when it loses at most this share of anchors against shipped.
    "target_rho_loss_max": 0.03,
    # Detector: share of contractions that up_valid removes.
    "event_drop_max": 0.20,
}


# =================================================================================================
# Geometry and channels
# =================================================================================================
@dataclass(frozen=True)
class AuditGeometry:
    r"""The model geometry the audit measures against, read from the config.

    Attributes:
        sequence_length: $T$, trimmed steps per segment.
        raw_per_step: $D$, raw samples per step.
        horizon: $H$, forecast steps per anchor.
        warmup_period: $F$, the first anchor.
        max_lag: The largest lag; a query reads $L = $ ``max_lag`` $+ 1$ keys.
        lag_floor: $F_u$, the earliest source step a key may read.
        anchor_stride: $S$, the training tile stride (recorded only; the audit is dense).
        coverage_floor: The least valid share of an anchor's $H$ forecast steps.
        warmup_budget: ``causal_warmup_budget_steps``, which gates the target channels.
        use_up_st: Whether the source stream holds ``up_st`` before ``up_ph``.
        lag_kv_source: The K/V body of the lag attention.
        event_reach_steps: The contraction weight's reach $\ell$ after an onset.
        event_floor: ``event_weight_floor``; ``None`` means the weight is off in training.
        event_ua_valid: ``event_weight_ua_valid``.
    """

    sequence_length: int
    raw_per_step: int
    horizon: int
    warmup_period: int
    max_lag: int
    lag_floor: int
    anchor_stride: int
    coverage_floor: float
    warmup_budget: Optional[int]
    use_up_st: bool
    lag_kv_source: str
    event_reach_steps: int
    event_floor: Optional[float]
    event_ua_valid: bool

    @property
    def num_lags(self) -> int:
        """$L$, the keys per query."""
        return self.max_lag + 1

    @property
    def t_valid(self) -> int:
        """$T - H$, the end of the anchor range."""
        return self.sequence_length - self.horizon

    @property
    def anchors(self) -> slice:
        """The dense anchor range $[F, T - H)$."""
        return slice(self.warmup_period, self.t_valid)

    @property
    def source_region(self) -> slice:
        r"""The source steps the keys of the anchor range read: $[\max(F - L + 1, F_u, 0), T - H)$."""
        start = max(self.warmup_period - self.num_lags + 1, self.lag_floor, 0)
        return slice(start, self.t_valid)

    @property
    def source_blocks(self) -> Tuple[str, ...]:
        """The UA blocks of the source stream, in stream order."""
        return ("up_st", "up_ph") if self.use_up_st else ("up_ph",)


def read_geometry(config: Mapping[str, Any]) -> AuditGeometry:
    """Read the geometry from a resolved config and refuse an arm the audit does not model.

    Args:
        config: The config mapping after ``base:`` resolution.

    Returns:
        The geometry.

    Raises:
        ValueError: If a required key is missing, if ``lag_kv_source`` is not ``adapter`` (a K/V
            body widens the key mask by its receptive field, which needs the model), or if a
            channel alignment reference is set (the gate then delays the mask per channel).
    """
    vae = (config.get("model_config") or {}).get("VAE_model") or {}

    def need(key: str) -> Any:
        if vae.get(key) is None:
            raise ValueError(f"model_config.VAE_model.{key} is missing from the config.")
        return vae[key]

    kv = str(vae.get("lag_kv_source") or "adapter")
    if kv != "adapter":
        raise ValueError(
            f"lag_kv_source is {kv!r}. The audit models the per-step adapter only: a conv_stem or "
            f"encoder body widens each bad step by its receptive field, which needs the model."
        )
    for key in ("causal_align_reference", "causal_align_reference_source"):
        if vae.get(key) is not None:
            raise ValueError(
                f"{key} is {vae[key]!r}. The audit gathers the kept channels without a per-channel "
                f"delay, which is only right when no alignment reference is set."
            )
    reach = vae.get("event_weight_reach_steps")
    floor = vae.get("event_weight_floor")
    budget = vae.get("causal_warmup_budget_steps")
    return AuditGeometry(
        sequence_length=int(need("sequence_length")),
        raw_per_step=int(vae.get("raw_per_step") or 16),
        horizon=int(need("horizon")),
        warmup_period=int(need("warmup_period")),
        max_lag=int(need("max_lag")),
        lag_floor=int(vae.get("lag_floor") or 0),
        anchor_stride=int(vae.get("anchor_stride") or 1),
        coverage_floor=float(vae.get("coverage_floor") or 0.0),
        warmup_budget=None if budget is None else int(budget),
        use_up_st=bool(vae.get("use_up_st", True)),
        lag_kv_source=kv,
        event_reach_steps=int(reach if reach is not None else need("max_lag")),
        event_floor=None if floor is None else float(floor),
        event_ua_valid=bool(vae.get("event_weight_ua_valid", False)),
    )


@dataclass(frozen=True)
class ChannelSets:
    r"""The kept channels of both streams, as indices into the declared widths.

    Attributes:
        source_keep: Kept indices into the ``up_st ++ up_ph`` stream.
        source_warmup: $W'_c$ of each kept source channel, positional against ``source_keep``.
        source_names: ``block[channel]`` of each kept source channel.
        target_keep: Kept indices into the ``fhr_st ++ fhr_ph`` stream.
        resolved_by: ``"resolve_warmup_budget"``, or ``"fallback"`` when the resolver refused the
            audited files and the sets were read from the dataset's rebased warm-up instead.
    """

    source_keep: Tuple[int, ...]
    source_warmup: Tuple[int, ...]
    source_names: Tuple[str, ...]
    target_keep: Tuple[int, ...]
    resolved_by: str


def _stream_names(widths: Sequence[Tuple[str, int]]) -> List[str]:
    """Name every declared channel of a stream ``block[channel]``, in stream order."""
    return [f"{block}[{index}]" for block, width in widths for index in range(width)]


def resolve_channels(
    config: Mapping[str, Any], geometry: AuditGeometry, paths: Sequence[str], dataset: Any
) -> ChannelSets:
    """Resolve the kept channels the way the trainer does, on the audited files.

    Args:
        config: The resolved config.
        geometry: The audit geometry.
        paths: Files of the audited build; the resolver reads their attributes.
        dataset: An open dataset over one of them, for the fallback and the block widths.

    Returns:
        The channel sets.
    """
    from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget

    warmup = dataset.causal_warmup_steps or {}
    source_widths = [(block, int(len(warmup[block]))) for block in geometry.source_blocks]
    source_names = _stream_names(source_widths)
    source_all = np.concatenate([warmup[block] for block in geometry.source_blocks])
    probe = deepcopy(dict(config))
    probe.setdefault("dataset_config", {})
    probe["dataset_config"]["vae_train_datasets"] = list(paths)
    probe["dataset_config"]["vae_test_datasets"] = []
    try:
        budget = resolve_warmup_budget(probe)
    except Exception as exc:  # noqa: BLE001 -- the fallback is recorded in the output
        logger.warning("resolve_warmup_budget refused the audited files (%s); using the fallback.",
                       exc)
        budget = None
    if budget is not None:
        for stream in (budget.target, budget.source):
            if stream.align_delays is not None and any(int(d) != 0 for d in stream.align_delays):
                raise ValueError(f"the {stream.name} stream has channel delays; see read_geometry.")
        source_keep = tuple(int(i) for i in budget.source.keep_index)
        target_keep = tuple(int(i) for i in budget.target.keep_index)
        resolved_by = "resolve_warmup_budget"
    else:
        vae = (config.get("model_config") or {}).get("VAE_model") or {}
        declared = vae.get("causal_source_channels")
        source_keep = tuple(range(len(source_all))) if declared is None else tuple(
            int(i) for i in declared)
        target_all = np.concatenate([warmup["fhr_st"], warmup["fhr_ph"]])
        limit = geometry.warmup_budget
        target_keep = tuple(int(i) for i, w in enumerate(target_all)
                            if limit is None or int(w) <= limit)
        resolved_by = "fallback"
    return ChannelSets(
        source_keep=source_keep,
        source_warmup=tuple(int(source_all[i]) for i in source_keep),
        source_names=tuple(source_names[i] for i in source_keep),
        target_keep=target_keep,
        resolved_by=resolved_by,
    )


# =================================================================================================
# Files and groups
# =================================================================================================
@dataclass
class Group:
    """One audited group: the files of one dataset, split and subgroup.

    Attributes:
        dataset: ``"pretraining"``, ``"fold_<k>"`` or ``"kfold_holdout"``.
        split: ``"train"``, ``"val"`` or ``"test"``.
        subgroup: The subgroup directory or file stem.
        paths: The files, sorted.
    """

    dataset: str
    split: str
    subgroup: str
    paths: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """``dataset/split/subgroup``."""
        return f"{self.dataset}/{self.split}/{self.subgroup}"


def discover_groups(
    root: Path,
    datasets: Sequence[str],
    splits: Optional[Sequence[str]] = None,
    subgroups: Optional[Sequence[str]] = None,
) -> List[Group]:
    """Find the audited files under a build root.

    The layout is that of ``build_fold_datasets.py``: pretraining shards at
    ``pre_training_dataset/{train,test}/<subgroup>/shard_NNN.hdf5``, folds at
    ``k_fold_cross_validation_dataset/fold_<k>/{train,val,test}/<subgroup>.hdf5``, and in holdout
    mode the fixed test set at ``k_fold_cross_validation_dataset/test/<subgroup>.hdf5``.

    Args:
        root: The build root.
        datasets: Any of ``"pretraining"``, ``"fold_<k>"`` and ``"kfold_holdout"``.
        splits: Only these splits; ``None`` keeps all.
        subgroups: Only these subgroups; ``None`` keeps all.

    Returns:
        The groups that hold at least one file, sorted by name.
    """
    found: Dict[str, Group] = {}

    def add(dataset: str, split: str, subgroup: str, path: Path) -> None:
        if splits and split not in splits:
            return
        if subgroups and subgroup not in subgroups:
            return
        group = found.setdefault(f"{dataset}/{split}/{subgroup}", Group(dataset, split, subgroup))
        group.paths.append(str(path))

    for dataset in datasets:
        if dataset == "pretraining":
            for path in sorted((root / PRETRAIN_DIR).glob("*/*/*.hdf5")):
                add("pretraining", path.parent.parent.name, path.parent.name, path)
        elif dataset == "kfold_holdout":
            for path in sorted((root / KFOLD_DIR / "test").glob("*.hdf5")):
                add("kfold_holdout", "test", path.stem, path)
        elif dataset.startswith("fold_"):
            for path in sorted((root / KFOLD_DIR / dataset).glob("*/*.hdf5")):
                add(dataset, path.parent.name, path.stem, path)
        else:
            raise ValueError(f"unknown dataset {dataset!r}; use pretraining, fold_<k> or "
                             f"kfold_holdout.")
    for group in found.values():
        group.paths.sort()
    return [found[name] for name in sorted(found)]


# =================================================================================================
# The loader's own mask code, per candidate
# =================================================================================================
def _dataset_kwargs(config: Mapping[str, Any]) -> Dict[str, Any]:
    """The config's ``dataset_kwargs``, reduced to what the audit's loader needs.

    The filters, the trim and the shipped step masks pass through unchanged. The audit sets
    ``load_fields`` to the blocks the masks are built for, turns the cache and the statistics off
    (normalisation changes no mask), and turns ``emit_quality_mask`` on for ``up_valid``.

    Args:
        config: The resolved config.

    Returns:
        Keyword arguments for ``CombinedHDF5Dataset`` without ``paths``.
    """
    from hdf5_dataset.hdf5_dataset import CombinedHDF5Dataset

    loader = (config.get("dataset_config") or {}).get("dataloader_config") or {}
    kwargs = dict(loader.get("dataset_kwargs") or {})
    accepted = set(inspect.signature(CombinedHDF5Dataset.__init__).parameters) - {"self", "paths"}
    kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    for key in ("stats_path", "normalize_fields", "emit_validity_mask"):
        kwargs.pop(key, None)
    kwargs.update(
        load_fields=["up", "weight", "up_st", "up_ph", "fhr_st", "fhr_ph"],
        cache_size=0,
        pin_memory=False,
        emit_quality_mask=True,
    )
    return kwargs


def open_dataset(path: str, kwargs: Mapping[str, Any]) -> Any:
    """Open one file as the training loader would, with its console report captured.

    Args:
        path: The HDF5 file.
        kwargs: From :func:`_dataset_kwargs`.

    Returns:
        The ``CombinedHDF5Dataset``.
    """
    from hdf5_dataset.hdf5_dataset import CombinedHDF5Dataset

    with contextlib.redirect_stdout(io.StringIO()):
        return CombinedHDF5Dataset(paths=[path], **dict(kwargs))


def resolve_rule(dataset: Any, key: str, spec: Optional[Mapping[str, Any]]) -> Any:
    """Resolve one candidate mapping with the loader's own validation.

    Args:
        dataset: An open ``CombinedHDF5Dataset``.
        key: ``"ua_step_mask"`` or ``"fhr_step_mask"``.
        spec: The candidate mapping, or ``None`` for the mask off.

    Returns:
        The resolved rule tuple, or ``None`` when the candidate is off.
    """
    saved = getattr(dataset, key)
    setattr(dataset, key, None if spec is None else dict(spec))
    try:
        if key == "ua_step_mask":
            return dataset._resolve_ua_step_mask()
        return dataset._resolve_fhr_step_mask()
    finally:
        setattr(dataset, key, saved)


@contextlib.contextmanager
def _rule_swapped(dataset: Any, attribute: str, rule: Any) -> Iterator[None]:
    """Put a resolved rule on the dataset for the duration of one mask build."""
    saved = getattr(dataset, attribute)
    setattr(dataset, attribute, rule)
    try:
        yield
    finally:
        setattr(dataset, attribute, saved)


def ua_cells(dataset: Any, f: Any, row: int, blocks: Sequence[str], rule: Any,
             shape: Tuple[int, int]) -> np.ndarray:
    """The loader's UA step mask of one segment under one rule.

    Args:
        dataset: The open dataset.
        f: Its open file.
        row: The segment's row in ``f``.
        blocks: The source blocks, in stream order.
        rule: A rule from :func:`resolve_rule`, or ``None`` (every cell usable).
        shape: $(T, C_u)$, the shape of the all-usable mask.

    Returns:
        ``(T, C_u)`` bool in stream order, ``True`` = usable.
    """
    if rule is None:
        return np.ones(shape, dtype=bool)
    with _rule_swapped(dataset, "_ua_step_rule", rule):
        out = dataset._ua_step_masks(f, row, list(blocks))
    return np.concatenate([out[f"{block}_step_valid"].numpy() for block in blocks], axis=1)


def fhr_cells(dataset: Any, f: Any, row: int, rule: Any, shape: Tuple[int, int]) -> np.ndarray:
    """The loader's FHR step mask of one segment under one rule.

    Args:
        dataset: The open dataset.
        f: Its open file.
        row: The segment's row in ``f``.
        rule: A rule from :func:`resolve_rule`, or ``None`` (every cell usable).
        shape: $(T, C_y)$, the shape of the all-usable mask.

    Returns:
        ``(T, C_y)`` bool in ``fhr_st ++ fhr_ph`` order, ``True`` = usable.
    """
    if rule is None:
        return np.ones(shape, dtype=bool)
    with _rule_swapped(dataset, "_fhr_step_rule", rule):
        out = dataset._fhr_step_masks(f, row, ["fhr_st", "fhr_ph"])
    return np.concatenate([out["fhr_st_step_valid"].numpy(), out["fhr_ph_step_valid"].numpy()],
                          axis=1)


@dataclass
class SegmentData:
    """What the audit keeps of one segment. It never leaves the process.

    Attributes:
        weight: The trimmed decimated ``weight``, $(T,)$.
        up: The trimmed bridged ``up``, $(L,)$, for the detector only.
        up_valid: The loader's ``up_valid``, $(L,)$.
        source: Candidate -> kept-channel UA mask $(T, |\\mathcal K|)$.
        target: Candidate -> per-step target validity over the kept channels, $(T,)$.
    """

    weight: np.ndarray
    up: np.ndarray
    up_valid: np.ndarray
    source: Dict[str, np.ndarray]
    target: Dict[str, np.ndarray]


class FileReader:
    """One open file with every candidate resolved against it.

    Args:
        path: The HDF5 file.
        kwargs: From :func:`_dataset_kwargs`.
        source_specs: Source candidate mappings.
        target_specs: Target candidate mappings.
    """

    def __init__(self, path: str, kwargs: Mapping[str, Any],
                 source_specs: Mapping[str, Optional[Mapping[str, Any]]],
                 target_specs: Mapping[str, Optional[Mapping[str, Any]]]) -> None:
        self.path = path
        self.dataset = open_dataset(path, kwargs)
        self.source_rules = {name: resolve_rule(self.dataset, "ua_step_mask", spec)
                             for name, spec in source_specs.items()}
        self.target_rules = {name: resolve_rule(self.dataset, "fhr_step_mask", spec)
                             for name, spec in target_specs.items()}

    def __len__(self) -> int:
        return len(self.dataset)

    def read(self, index: int, geometry: AuditGeometry, channels: ChannelSets) -> SegmentData:
        """Build every candidate mask of one segment with the loader's code.

        Args:
            index: The segment's index in the dataset (after the config's filters).
            geometry: The audit geometry.
            channels: The kept channel sets.

        Returns:
            The segment's data.

        Raises:
            ValueError: If the trimmed length differs from ``sequence_length``.
        """
        ds = self.dataset
        file_idx, row = ds.index_map[index]
        f = ds._open_handle(file_idx)
        weight = np.asarray(f["weight"][row][ds._step_window()], dtype=np.float32)
        steps = int(weight.shape[0])
        if steps != geometry.sequence_length:
            raise ValueError(f"{self.path}: trimmed weight has {steps} steps but sequence_length "
                             f"is {geometry.sequence_length}; check trim_minutes.")
        up = np.asarray(f["up"][row][ds._raw_window()], dtype=np.float64)
        up_valid = ds._quality_masks(f, row)["up_valid"].numpy()
        c_u = sum(int(f[block].shape[1]) for block in geometry.source_blocks)
        c_y = int(f["fhr_st"].shape[1]) + int(f["fhr_ph"].shape[1])
        src_keep = list(channels.source_keep)
        tgt_keep = list(channels.target_keep)
        source = {
            name: ua_cells(ds, f, row, geometry.source_blocks, rule, (steps, c_u))[:, src_keep]
            for name, rule in self.source_rules.items()
        }
        target = {
            name: fhr_cells(ds, f, row, rule, (steps, c_y))[:, tgt_keep].all(axis=1)
            for name, rule in self.target_rules.items()
        }
        return SegmentData(weight=weight, up=up, up_valid=up_valid, source=source, target=target)

    def verify(self, index: int, data: SegmentData, geometry: AuditGeometry,
               channels: ChannelSets) -> List[str]:
        """Compare the shipped candidates with the public ``dataset[index]``.

        Args:
            index: The segment's index in the dataset.
            data: What :meth:`read` built for it.
            geometry: The audit geometry.
            channels: The kept channel sets.

        Returns:
            The names of the fields that differ; empty when the two paths agree.
        """
        sample = self.dataset[index]
        bad: List[str] = []
        if not np.array_equal(sample["weight"].numpy().astype(np.float32), data.weight):
            bad.append("weight")
        if not np.array_equal(sample["up_valid"].numpy(), data.up_valid):
            bad.append("up_valid")
        if self.dataset._ua_step_rule is not None:
            stream = np.concatenate([sample[f"{b}_step_valid"].numpy()
                                     for b in geometry.source_blocks], axis=1)
            if not np.array_equal(stream[:, list(channels.source_keep)], data.source[SHIPPED]):
                bad.append("ua_step_mask")
        if self.dataset._fhr_step_rule is not None:
            stream = np.concatenate([sample["fhr_st_step_valid"].numpy(),
                                     sample["fhr_ph_step_valid"].numpy()], axis=1)
            if not np.array_equal(stream[:, list(channels.target_keep)].all(axis=1),
                                  data.target[SHIPPED]):
                bad.append("fhr_step_mask")
        return bad

    def close(self) -> None:
        """Close the file handle."""
        with contextlib.suppress(Exception):
            self.dataset._cleanup_handles()


# =================================================================================================
# Measurement of one group
# =================================================================================================
def _key_stub() -> Any:
    """A stand-in for the model: no channel gate and a K/V reach of 1 (the adapter arm)."""
    return SimpleNamespace(source_gate=None, _source_kv_reach=lambda num_steps: 1)


def lag_mask(geometry: AuditGeometry) -> Any:
    r"""The model's lag mask $m_{t,\ell} = \mathbb 1[t - \ell \ge \max(0, F_u)]$, $(T, L)$.

    It is ``CausalWarmupInputs.build_lag_mask``: the base mask $t - \ell \ge 0$, ANDed with the
    floor $t - \ell \ge F_u$.

    Args:
        geometry: The audit geometry.

    Returns:
        A boolean tensor.
    """
    import torch

    steps = torch.arange(geometry.sequence_length)[:, None]
    lags = torch.arange(geometry.num_lags)[None, :]
    return (steps - lags >= 0) & (steps - lags >= geometry.lag_floor)


@dataclass
class GroupResult:
    """The sums and per-segment values of one group, before any ratio is taken.

    Attributes:
        sums: Metric numerators and denominators, summed over the sampled segments.
        values: Per-segment values, one array per metric; NaN where undefined.
        channel_bad: Source candidate -> bad cells per kept channel in the source region.
        channel_unique: Source candidate -> cells where that channel alone is bad.
        sampled: Segments measured.
        population: Segments in the group after the config's filters.
        files: Files in the group.
    """

    sums: Dict[str, float]
    values: Dict[str, np.ndarray]
    channel_bad: Dict[str, np.ndarray]
    channel_unique: Dict[str, np.ndarray]
    sampled: int
    population: int
    files: int


def _source_views(source: Dict[str, np.ndarray], channels: ChannelSets) -> Dict[str, np.ndarray]:
    """Add the drop-slowest views of the shipped source mask; see :data:`DROP_SLOWEST`."""
    order = np.argsort(-np.asarray(channels.source_warmup), kind="stable")
    out = dict(source)
    for k in DROP_SLOWEST:
        if k >= len(order):
            continue
        keep = np.sort(order[k:])
        out[f"{SHIPPED}_drop_slowest_{k}"] = source[SHIPPED][..., keep]
    return out


def measure_group(segments: Sequence[SegmentData], geometry: AuditGeometry,
                  channels: ChannelSets, events: bool) -> Tuple[Dict[str, float],
                                                                Dict[str, np.ndarray],
                                                                Dict[str, np.ndarray],
                                                                Dict[str, np.ndarray]]:
    r"""Measure every candidate on the segments of one group.

    Definitions, with $\mathcal A = [F, T - H)$ the anchors and $\mathcal R$ the source region
    (:attr:`AuditGeometry.source_region`):

    * source step usable: the share of $s \in \mathcal R$ with $m_s$;
    * source cell usable: the share of $(s, c) \in \mathcal R \times \mathcal K$ with $v_{s,c}$
      (the most a per-channel mask could keep);
    * anchor open: the share of anchors with at least one open key;
    * ``source_mask_fraction``: $1 -$ open KL anchors / KL anchors, with the KL support of the
      shipped target mask, as the training metric;
    * target KL share: KL anchors under a target candidate / $|\mathcal A|$.

    Args:
        segments: The group's segments.
        geometry: The audit geometry.
        channels: The kept channel sets.
        events: Whether to run the contraction detector.

    Returns:
        ``(sums, values, channel_bad, channel_unique)``; see :class:`GroupResult`.
    """
    import torch

    from teb_vae.lag_attn_cfs.eval.events import detect_contractions
    from teb_vae.lag_attn_cfs.nets.causal_inputs import CausalWarmupInputs
    from teb_vae.lag_attn_rws.event_weight import (contraction_anchor_weight,
                                                   contraction_onset_steps)
    from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry
    from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask, kl_mask

    g = geometry
    n = len(segments)
    geo = TrimmedRawGeometry(raw_len=g.sequence_length * g.raw_per_step,
                             decimation=g.raw_per_step, horizon=g.horizon,
                             warmup=g.warmup_period)
    anchors, region = g.anchors, g.source_region
    n_anchors = g.t_valid - g.warmup_period
    weight = torch.from_numpy(np.stack([s.weight for s in segments]))
    sums: Dict[str, float] = {"segments": float(n), "warm_anchors": float(n * n_anchors),
                              "scored_steps": float(n * (g.sequence_length - g.warmup_period - 1))}
    values: Dict[str, np.ndarray] = {}

    # --- Target candidates: forecast mask, coverage floor, KL support ---------------------------
    kl_by_target: Dict[str, Any] = {}
    for name in segments[0].target:
        valid = torch.from_numpy(np.stack([s.target[name] for s in segments]))
        w = weight * valid.to(weight.dtype)
        mask, coverage = forecast_mask(w, geo, coverage_floor=g.coverage_floor)
        kl = kl_mask(mask, geo).to(torch.bool)
        kl_by_target[name] = kl
        live = (w >= 1.0)[:, anchors]
        kl_count = kl[:, anchors].sum(dim=1).double()
        sums[f"tgt/{name}/kl"] = float(kl_count.sum())
        sums[f"tgt/{name}/anchor_valid"] = float(live.sum())
        sums[f"tgt/{name}/dropped_by_floor"] = float((live & ~kl[:, anchors]).sum())
        sums[f"tgt/{name}/coverage"] = float(coverage[:, anchors].sum())
        sums[f"tgt/{name}/scored_valid"] = float((w >= 1.0)[:, g.warmup_period + 1:].sum())
        sums[f"tgt/{name}/segments_zero_kl"] = float((kl_count == 0).sum())
        values[f"tgt/{name}/kl_share"] = (kl_count / n_anchors).numpy()

    kl_ship = kl_by_target[SHIPPED][:, anchors]
    kl_ship_count = kl_ship.sum(dim=1).double()
    has_kl = kl_ship_count > 0
    sums["kl_anchors"] = float(kl_ship_count.sum())
    sums["segments_with_kl"] = float(has_kl.sum())

    # --- Source candidates: the model's key mask ----------------------------------------------
    stub, lags = _key_stub(), lag_mask(g)
    source = _source_views({name: np.stack([s.source[name] for s in segments])
                            for name in segments[0].source}, channels)
    channel_bad: Dict[str, np.ndarray] = {}
    channel_unique: Dict[str, np.ndarray] = {}
    open_ship = None
    for name, cells_np in source.items():
        cells = torch.from_numpy(cells_np)
        keys = CausalWarmupInputs.source_key_mask(stub, cells, lags)  # (N, T, L)
        is_open = keys.any(dim=-1)
        if name == SHIPPED:
            open_ship = is_open
        step_ok = cells.all(dim=-1)
        open_kl = (is_open[:, anchors] & kl_ship)
        open_kl_count = open_kl.sum(dim=1).double()
        sums[f"src/{name}/region_steps"] = float(step_ok[:, region].numel())
        sums[f"src/{name}/usable_steps"] = float(step_ok[:, region].sum())
        sums[f"src/{name}/region_cells"] = float(cells[:, region].numel())
        sums[f"src/{name}/usable_cells"] = float(cells[:, region].sum())
        sums[f"src/{name}/open_warm"] = float(is_open[:, anchors].sum())
        sums[f"src/{name}/open_kl"] = float(open_kl_count.sum())
        sums[f"src/{name}/open_keys_kl"] = float(
            (keys[:, anchors].sum(dim=-1) * open_kl).sum())
        sums[f"src/{name}/segments_zero_open"] = float((has_kl & (open_kl_count == 0)).sum())
        values[f"src/{name}/step_usable"] = step_ok[:, region].double().mean(dim=1).numpy()
        share = torch.where(has_kl, open_kl_count / kl_ship_count.clamp_min(1.0),
                            torch.full_like(open_kl_count, float("nan")))
        values[f"src/{name}/open_kl_share"] = share.numpy()
        if name in segments[0].source:
            bad = ~cells_np[:, region, :]
            channel_bad[name] = bad.sum(axis=(0, 1)).astype(np.float64)
            alone = bad & (bad.sum(axis=-1, keepdims=True) == 1)
            channel_unique[name] = alone.sum(axis=(0, 1)).astype(np.float64)

    # --- Raw UA validity in the source region -------------------------------------------------
    d = g.raw_per_step
    raw = slice(max(region.start * d - (d - 1), 0), region.stop * d)
    up_valid = np.stack([s.up_valid for s in segments])
    sums["ua/raw_samples"] = float(up_valid[:, raw].size)
    sums["ua/raw_valid"] = float(up_valid[:, raw].sum())

    # --- Contraction detector, with and without up_valid --------------------------------------
    if events:
        active = {"off": [], "on": []}
        counts = {"off": [], "on": []}
        failures = 0
        for index, seg in enumerate(segments):
            for mode in ("off", "on"):
                valid = None if mode == "off" else seg.up_valid
                try:
                    onsets = contraction_onset_steps(seg.up, seg.weight, decimation=d,
                                                     detector=detect_contractions,
                                                     raw_valid=valid)
                except ValueError:
                    failures += 1
                    onsets = np.zeros(0, dtype=np.int16)
                w_event = contraction_anchor_weight(
                    torch.from_numpy(seg.up[None]), torch.from_numpy(seg.weight[None]),
                    decimation=d, reach_steps=g.event_reach_steps, floor=0.0,
                    keys=[index], cache={index: onsets}, detector=detect_contractions,
                    raw_valid=None if valid is None else torch.from_numpy(valid[None]))
                active[mode].append(w_event[0].numpy() >= 1.0)
                counts[mode].append(len(onsets))
        on = torch.from_numpy(np.stack(active["on"]))[:, anchors]
        off = torch.from_numpy(np.stack(active["off"]))[:, anchors]
        n_off, n_on = np.asarray(counts["off"], float), np.asarray(counts["on"], float)
        sums["evt/events_off"] = float(n_off.sum())
        sums["evt/events_on"] = float(n_on.sum())
        sums["evt/segments_losing_event"] = float((n_on < n_off).sum())
        sums["evt/active_off_kl"] = float((off & kl_ship).sum())
        sums["evt/active_on_kl"] = float((on & kl_ship).sum())
        sums["evt/effective_kl"] = float((on & kl_ship & open_ship[:, anchors]).sum())
        sums["evt/failures"] = float(failures)
        values["evt/events_on"] = n_on
    return sums, values, channel_bad, channel_unique


def sample_group(group: Group, kwargs: Mapping[str, Any], geometry: AuditGeometry,
                 channels: Optional[ChannelSets], config: Mapping[str, Any],
                 source_specs: Mapping[str, Optional[Mapping[str, Any]]],
                 target_specs: Mapping[str, Optional[Mapping[str, Any]]],
                 segments_per_group: int, seed: int, verify_segments: int,
                 events: bool) -> Tuple[GroupResult, ChannelSets, Dict[str, Any]]:
    """Open a group's files, draw its sample, build the masks and measure them.

    Args:
        group: The group.
        kwargs: From :func:`_dataset_kwargs`.
        geometry: The audit geometry.
        channels: The kept channel sets, or ``None`` to resolve them on this group's files.
        config: The resolved config, for the channel resolution.
        source_specs: Source candidate mappings.
        target_specs: Target candidate mappings.
        segments_per_group: The sample cap.
        seed: The base seed; the group's own seed adds a CRC of its name.
        verify_segments: Segments checked against the public loader path.
        events: Whether to run the contraction detector.

    Returns:
        ``(result, channels, info)``; ``info`` holds the verification count and the quality
        attributes.

    Raises:
        RuntimeError: If the shipped candidate differs from the public loader path.
    """
    readers = [FileReader(path, kwargs, source_specs, target_specs) for path in group.paths]
    try:
        if channels is None:
            channels = resolve_channels(config, geometry, group.paths[:1], readers[0].dataset)
        lengths = np.asarray([len(r) for r in readers], dtype=np.int64)
        population = int(lengths.sum())
        take = min(int(segments_per_group), population)
        rng = np.random.default_rng(int(seed) + zlib.crc32(group.name.encode()))
        picks = np.sort(rng.choice(population, size=take, replace=False))
        bounds = np.concatenate([[0], np.cumsum(lengths)])
        segments: List[SegmentData] = []
        checked, mismatches = 0, []
        for order, flat in enumerate(picks.tolist()):
            file_index = int(np.searchsorted(bounds, flat, side="right") - 1)
            local = flat - int(bounds[file_index])
            data = readers[file_index].read(local, geometry, channels)
            if order < verify_segments:
                bad = readers[file_index].verify(local, data, geometry, channels)
                checked += 1
                if bad:
                    mismatches.append(bad)
            segments.append(data)
        if mismatches:
            raise RuntimeError(
                f"{group.name}: the audit's mask path differs from dataset[i] on "
                f"{len(mismatches)} of {checked} checked segments ({mismatches[0]}). The audit "
                f"would not measure what training reads; stop and compare the two paths.")
        info = readers[0].dataset.quality_info
        quality = {} if info is None else {
            "quality_version": int(info.version),
            "quality_rho_scale": int(info.rho_scale),
            "quality_lookahead_s": float(info.lookahead_s),
        }
        sums, values, channel_bad, channel_unique = measure_group(
            segments, geometry, channels, events)
        result = GroupResult(sums=sums, values=values, channel_bad=channel_bad,
                             channel_unique=channel_unique, sampled=take, population=population,
                             files=len(group.paths))
        return result, channels, {"verified": checked, "quality": quality}
    finally:
        for reader in readers:
            reader.close()


# =================================================================================================
# Summaries
# =================================================================================================
def _ratio(num: float, den: float) -> Optional[float]:
    """``num / den``, or ``None`` when the denominator is 0."""
    return None if den <= 0 else float(num / den)


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray) -> Optional[Dict[str, float]]:
    """Weighted quantiles of the finite values, at :data:`QUANTILES`.

    Args:
        values: The per-segment values.
        weights: One weight per value.

    Returns:
        ``{"p05": ..., ...}``, or ``None`` when no value is finite.
    """
    keep = np.isfinite(values)
    if not keep.any():
        return None
    v, w = values[keep], weights[keep]
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    cum = (np.cumsum(w) - 0.5 * w) / w.sum()
    return {f"p{int(round(q * 100)):02d}": float(np.interp(q, cum, v)) for q in QUANTILES}


@dataclass
class Pool:
    """Group results combined with population weights, for one row of the report.

    Each sampled segment of a group stands for ``population / sampled`` segments, so a pooled ratio
    estimates the share over every segment of the pooled groups.
    """

    sums: Dict[str, float] = field(default_factory=dict)
    values: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    weights: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    channel_bad: Dict[str, np.ndarray] = field(default_factory=dict)
    channel_unique: Dict[str, np.ndarray] = field(default_factory=dict)
    sampled: int = 0
    population: int = 0
    files: int = 0
    groups: int = 0

    def add(self, result: GroupResult) -> None:
        """Add one group with its population weight."""
        scale = result.population / max(result.sampled, 1)
        for key, value in result.sums.items():
            self.sums[key] = self.sums.get(key, 0.0) + scale * value
        for key, value in result.values.items():
            self.values.setdefault(key, []).append(np.asarray(value, dtype=np.float64))
            self.weights.setdefault(key, []).append(np.full(len(value), scale))
        for target, source in ((self.channel_bad, result.channel_bad),
                               (self.channel_unique, result.channel_unique)):
            for key, value in source.items():
                target[key] = target.get(key, 0.0) + scale * value
        self.sampled += result.sampled
        self.population += result.population
        self.files += result.files
        self.groups += 1

    def quantiles(self, key: str) -> Optional[Dict[str, float]]:
        """Weighted quantiles of one per-segment metric."""
        if key not in self.values:
            return None
        return _weighted_quantiles(np.concatenate(self.values[key]),
                                   np.concatenate(self.weights[key]))

    def summary(self, source_names: Sequence[str], target_names: Sequence[str],
                num_lags: int) -> Dict[str, Any]:
        """The report row: every ratio and quantile of this pool.

        Args:
            source_names: Source candidates and views.
            target_names: Target candidates.
            num_lags: $L$, for the open-key share.

        Returns:
            A JSON-ready mapping.
        """
        s = self.sums
        out: Dict[str, Any] = {"groups": self.groups, "files": self.files,
                               "segments_sampled": self.sampled, "population": self.population,
                               "ua_raw_valid_in_source_region": _ratio(s.get("ua/raw_valid", 0),
                                                                       s.get("ua/raw_samples", 0))}
        source: Dict[str, Any] = {}
        for name in source_names:
            p = f"src/{name}/"
            open_kl = s.get(p + "open_kl", 0.0)
            smf = _ratio(open_kl, s.get("kl_anchors", 0.0))
            source[name] = {
                "step_usable": _ratio(s.get(p + "usable_steps", 0), s.get(p + "region_steps", 0)),
                "cell_usable": _ratio(s.get(p + "usable_cells", 0), s.get(p + "region_cells", 0)),
                "anchor_open_warm": _ratio(s.get(p + "open_warm", 0), s.get("warm_anchors", 0)),
                "source_mask_fraction": None if smf is None else 1.0 - smf,
                "open_key_frac": _ratio(s.get(p + "open_keys_kl", 0), num_lags * open_kl),
                "segments_zero_open": _ratio(s.get(p + "segments_zero_open", 0),
                                             s.get("segments_with_kl", 0)),
                "step_usable_per_segment": self.quantiles(p + "step_usable"),
                "open_kl_share_per_segment": self.quantiles(p + "open_kl_share"),
            }
        target: Dict[str, Any] = {}
        for name in target_names:
            p = f"tgt/{name}/"
            target[name] = {
                "kl_anchor_share": _ratio(s.get(p + "kl", 0), s.get("warm_anchors", 0)),
                "anchor_valid_share": _ratio(s.get(p + "anchor_valid", 0),
                                             s.get("warm_anchors", 0)),
                "dropped_by_coverage_floor": _ratio(s.get(p + "dropped_by_floor", 0),
                                                    s.get("warm_anchors", 0)),
                "coverage_mean": _ratio(s.get(p + "coverage", 0), s.get("warm_anchors", 0)),
                "scored_step_valid": _ratio(s.get(p + "scored_valid", 0),
                                            s.get("scored_steps", 0)),
                "segments_zero_kl": _ratio(s.get(p + "segments_zero_kl", 0),
                                           s.get("segments", 0)),
                "kl_share_per_segment": self.quantiles(p + "kl_share"),
            }
        out["source"], out["target"] = source, target
        if "evt/events_off" in s:
            kl = s.get("kl_anchors", 0.0)
            out["events"] = {
                "events_per_segment_off": _ratio(s["evt/events_off"], s["segments"]),
                "events_per_segment_on": _ratio(s["evt/events_on"], s["segments"]),
                "event_drop_share": (None if s["evt/events_off"] <= 0
                                     else 1.0 - s["evt/events_on"] / s["evt/events_off"]),
                "segments_losing_event": _ratio(s["evt/segments_losing_event"], s["segments"]),
                "active_kl_share_off": _ratio(s["evt/active_off_kl"], kl),
                "active_kl_share_on": _ratio(s["evt/active_on_kl"], kl),
                "effective_kl_share": _ratio(s["evt/effective_kl"], kl),
                "detector_failures": s["evt/failures"],
                "events_on_per_segment": self.quantiles("evt/events_on"),
            }
        return out

    def channel_table(self, names: Sequence[str], channels: ChannelSets) -> Dict[str, Any]:
        """Per kept source channel: the bad share and the unique-blocker share in the region.

        Args:
            names: Source candidates (not views).
            channels: The kept channel sets.

        Returns:
            Candidate -> list of rows, one per kept channel, in stream order.
        """
        out: Dict[str, Any] = {}
        for name in names:
            if name not in self.channel_bad:
                continue
            cells = self.sums.get(f"src/{name}/region_steps", 0.0)
            out[name] = [
                {"channel": channels.source_names[i], "warmup_steps": channels.source_warmup[i],
                 "bad_share": _ratio(float(self.channel_bad[name][i]), cells),
                 "unique_blocker_share": _ratio(float(self.channel_unique[name][i]), cells)}
                for i in range(len(channels.source_names))
            ]
        return out


# =================================================================================================
# Decision checks and the Markdown report
# =================================================================================================
def _fmt(value: Optional[float], digits: int = 3) -> str:
    """A share as text; ``-`` for a missing value."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{value:.{digits}f}"


def decision_checks(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Evaluate :data:`THRESHOLDS` on the overall pool and the subgroups.

    Args:
        report: The report mapping with ``aggregates`` and ``groups``.

    Returns:
        One row per check: name, value, threshold and verdict.
    """
    t = THRESHOLDS
    total = report["aggregates"]["ALL"]
    src, tgt = total["source"], total["target"]
    rows: List[Dict[str, Any]] = []

    def row(name: str, value: Optional[float], rule: str, verdict: str) -> None:
        rows.append({"check": name, "value": value, "threshold": rule, "verdict": verdict})

    smf = src[SHIPPED]["source_mask_fraction"]
    row("source_mask_fraction (shipped)", smf, f"<= {t['smf_keep']} keep; > {t['smf_act']} act",
        "-" if smf is None else ("keep" if smf <= t["smf_keep"]
                                 else "watch" if smf <= t["smf_act"] else "act"))
    zero = src[SHIPPED]["segments_zero_open"]
    row("segments with every KL anchor closed (shipped)", zero,
        f"<= {t['segments_zero_open_max']}",
        "-" if zero is None else ("ok" if zero <= t["segments_zero_open_max"] else "act"))
    spreads: List[float] = []
    by_split: Dict[str, List[float]] = {}
    for name, group in report["groups"].items():
        value = group["source"][SHIPPED]["source_mask_fraction"]
        if value is not None:
            by_split.setdefault(name.rsplit("/", 1)[0], []).append(value)
    for values in by_split.values():
        if len(values) > 1:
            spreads.append(max(values) - min(values))
    spread = max(spreads) if spreads else None
    row("subgroup spread of source_mask_fraction (shipped)", spread,
        f"<= {t['subgroup_spread_max']}",
        "-" if spread is None else ("ok" if spread <= t["subgroup_spread_max"]
                                    else "shortcut risk: stratify the evaluation"))
    view = src.get(f"{SHIPPED}_drop_slowest_4", {}).get("source_mask_fraction")
    gain = None if (smf is None or view is None) else smf - view
    row("source_mask_fraction gain from dropping the 4 slowest source channels", gain,
        f">= {t['drop_slowest_gain_min']} drop them",
        "-" if gain is None else ("drop them" if gain >= t["drop_slowest_gain_min"] else "keep"))
    ship_kl = tgt[SHIPPED]["kl_anchor_share"]
    off_kl = tgt.get("off", {}).get("kl_anchor_share")
    kept = None if (ship_kl is None or not off_kl) else ship_kl / off_kl
    row("target KL anchors, shipped / mask off", kept, f">= {t['target_kept_ratio_min']}",
        "-" if kept is None else ("ok" if kept >= t["target_kept_ratio_min"]
                                  else "coverage floor binds"))
    rho_kl = tgt.get("rho_0.10", {}).get("kl_anchor_share")
    loss = None if (ship_kl is None or rho_kl is None) else ship_kl - rho_kl
    row("target KL anchors lost by rho 0.10 against shipped", loss,
        f"<= {t['target_rho_loss_max']} rho affordable",
        "-" if loss is None else ("rho affordable" if loss <= t["target_rho_loss_max"]
                                  else "keep bits"))
    drop = (total.get("events") or {}).get("event_drop_share")
    row("contractions removed by up_valid", drop, f"<= {t['event_drop_max']}",
        "-" if drop is None else ("ok" if drop <= t["event_drop_max"]
                                  else "detector policy too strict"))
    mism = report["verification"]["mismatches"]
    row("loader path mismatches", float(mism), "== 0", "ok" if mism == 0 else "invalid run")
    return rows


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    """Write the short Markdown report.

    Args:
        report: The report mapping.
        path: Where to write.
    """
    g = report["geometry"]
    lines = [
        "# Mask coverage audit",
        "",
        f"Created {report['created']}. Build `{report['root_name']}`, config `{report['config']}`. "
        f"{report['totals']['segments_sampled']} segments sampled from "
        f"{report['totals']['population']} in {report['totals']['groups']} groups "
        f"(cap {report['sampling']['segments_per_group']} per group, seed "
        f"{report['sampling']['seed']}).",
        "",
        f"Geometry: T = {g['sequence_length']}, H = {g['horizon']}, F = {g['warmup_period']}, "
        f"L = {g['max_lag'] + 1}, S = {g['anchor_stride']}, coverage floor "
        f"{g['coverage_floor']}, source region steps {g['source_region']}, anchors "
        f"{g['anchors']}. Kept channels: {len(report['channels']['source_keep'])} source, "
        f"{report['channels']['target_keep_count']} target ({report['channels']['resolved_by']}).",
        "",
        "Shares are pooled over segments with population weights. `smf` is the expected "
        "`source_mask_fraction` (closed KL anchors over KL anchors, KL support of the shipped "
        "target mask). `cell` is the most a per-channel source mask could keep.",
        "",
        "## Decision checks",
        "",
        "| check | value | threshold | verdict |",
        "|---|---|---|---|",
    ]
    for item in report["decision_checks"]:
        lines.append(f"| {item['check']} | {_fmt(item['value'])} | {item['threshold']} | "
                     f"{item['verdict']} |")
    total = report["aggregates"]["ALL"]
    lines += ["", "## Source candidates (ALL)", "",
              "| candidate | step | cell | anchors open | smf | segs all closed | "
              "open share p10 | p50 | open keys |", "|---|---|---|---|---|---|---|---|---|"]
    for name, m in total["source"].items():
        q = m["open_kl_share_per_segment"] or {}
        lines.append(f"| {name} | {_fmt(m['step_usable'])} | {_fmt(m['cell_usable'])} | "
                     f"{_fmt(m['anchor_open_warm'])} | {_fmt(m['source_mask_fraction'])} | "
                     f"{_fmt(m['segments_zero_open'])} | {_fmt(q.get('p10'))} | "
                     f"{_fmt(q.get('p50'))} | {_fmt(m['open_key_frac'])} |")
    lines += ["", "## Target candidates (ALL)", "",
              "| candidate | KL anchors | anchor valid | dropped by floor | scored steps valid | "
              "segs no KL | KL share p10 | p50 |", "|---|---|---|---|---|---|---|---|"]
    for name, m in total["target"].items():
        q = m["kl_share_per_segment"] or {}
        lines.append(f"| {name} | {_fmt(m['kl_anchor_share'])} | {_fmt(m['anchor_valid_share'])} "
                     f"| {_fmt(m['dropped_by_coverage_floor'])} | {_fmt(m['scored_step_valid'])} "
                     f"| {_fmt(m['segments_zero_kl'])} | {_fmt(q.get('p10'))} | "
                     f"{_fmt(q.get('p50'))} |")
    lines += ["", "## Shipped masks per group", "",
              "| group | n | population | UA raw valid | src step | smf | segs all closed | "
              "tgt KL | events/seg off | on | effective KL |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = list(report["groups"].items()) + [(f"**{k}**", v) for k, v in
                                             report["aggregates"].items()]
    for name, m in rows:
        s, t = m["source"][SHIPPED], m["target"][SHIPPED]
        e = m.get("events") or {}
        lines.append(f"| {name} | {m['segments_sampled']} | {m['population']} | "
                     f"{_fmt(m['ua_raw_valid_in_source_region'])} | {_fmt(s['step_usable'])} | "
                     f"{_fmt(s['source_mask_fraction'])} | {_fmt(s['segments_zero_open'])} | "
                     f"{_fmt(t['kl_anchor_share'])} | {_fmt(e.get('events_per_segment_off'), 2)} | "
                     f"{_fmt(e.get('events_per_segment_on'), 2)} | "
                     f"{_fmt(e.get('effective_kl_share'))} |")
    table = (report.get("channels_pooled") or {}).get(SHIPPED) or []
    top = sorted(table, key=lambda r: -(r["unique_blocker_share"] or 0.0))[:10]
    lines += ["", "## Source channels that close steps alone (shipped, ALL)", "",
              "| channel | W' (steps) | bad share | alone |", "|---|---|---|---|"]
    for r in top:
        lines.append(f"| {r['channel']} | {r['warmup_steps']} | {_fmt(r['bad_share'])} | "
                     f"{_fmt(r['unique_blocker_share'])} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =================================================================================================
# Driver
# =================================================================================================
def _resolve_path(value: str) -> Path:
    """A path as given, or relative to the repository root when it is relative."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (_REPO_ROOT / path)


def candidate_specs(config: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The source and target candidates, with ``shipped`` taken from the config.

    Args:
        config: The resolved config.

    Returns:
        ``(source_specs, target_specs)``.
    """
    kwargs = ((config.get("dataset_config") or {}).get("dataloader_config") or {}).get(
        "dataset_kwargs") or {}
    source = dict(SOURCE_CANDIDATES)
    target = dict(TARGET_CANDIDATES)
    source[SHIPPED] = kwargs.get("ua_step_mask")
    target[SHIPPED] = kwargs.get("fhr_step_mask")
    return source, target


def main(root: Optional[str] = None, output_dir: Optional[str] = None,
         config_path: Optional[str] = None, datasets: Optional[List[str]] = None,
         splits: Optional[List[str]] = None, subgroups: Optional[List[str]] = None,
         segments_per_group: Optional[int] = None, seed: Optional[int] = None,
         verify_segments: Optional[int] = None, events: Optional[bool] = None) -> int:
    """Run the audit and write the JSON and Markdown reports.

    Args:
        root: The build root (the folder that holds ``pre_training_dataset`` and
            ``k_fold_cross_validation_dataset``).
        output_dir: Where the two reports go; created if absent.
        config_path: The model config. ``None`` -> :data:`DEFAULT_CONFIG`. Relative paths resolve
            against the repository root.
        datasets: Any of ``pretraining``, ``fold_<k>``, ``kfold_holdout``.
            ``None`` -> ``["pretraining", "fold_1"]``.
        splits: Only these splits. ``None`` -> all.
        subgroups: Only these subgroups. ``None`` -> all.
        segments_per_group: The sample cap per group. ``None`` -> 300.
        seed: The sampling seed. ``None`` -> 0.
        verify_segments: Segments per group checked against ``dataset[i]``. ``None`` -> 4.
        events: Run the contraction detector. ``None`` -> ``True``.

    Returns:
        The process exit code.

    Raises:
        SystemExit: If no audited file is found.
    """
    from teb_vae.lag_attn.config import load_config

    started = time.time()
    root_path = _resolve_path(str(root))
    out_path = _resolve_path(str(output_dir))
    cfg_path = _resolve_path(config_path or DEFAULT_CONFIG)
    datasets = list(datasets or ["pretraining", "fold_1"])
    segments_per_group = int(segments_per_group if segments_per_group is not None else 300)
    seed = int(seed if seed is not None else 0)
    verify_segments = int(verify_segments if verify_segments is not None else 4)
    events = True if events is None else bool(events)

    config = load_config(str(cfg_path))
    geometry = read_geometry(config)
    kwargs = _dataset_kwargs(config)
    source_specs, target_specs = candidate_specs(config)
    groups = discover_groups(root_path, datasets, splits, subgroups)
    if not groups:
        raise SystemExit(f"no HDF5 file found under {root_path} for datasets {datasets}.")
    logger.info("%d groups, %d files; cap %d segments per group.", len(groups),
                sum(len(g.paths) for g in groups), segments_per_group)

    channels: Optional[ChannelSets] = None
    results: Dict[str, GroupResult] = {}
    verified, quality = 0, {}
    for group in groups:
        tick = time.time()
        result, channels, info = sample_group(
            group, kwargs, geometry, channels, config, source_specs, target_specs,
            segments_per_group, seed, verify_segments, events)
        results[group.name] = result
        verified += info["verified"]
        quality = quality or info["quality"]
        logger.info("%s: %d of %d segments in %.1f s", group.name, result.sampled,
                    result.population, time.time() - tick)
    assert channels is not None

    source_names = list(source_specs) + [f"{SHIPPED}_drop_slowest_{k}" for k in DROP_SLOWEST
                                         if k < len(channels.source_keep)]
    target_names = list(target_specs)
    pools: Dict[str, Pool] = {}
    for name, result in results.items():
        dataset, split, _ = name.split("/")
        for key in (f"{dataset}/{split}/ALL", f"{dataset}/ALL", "ALL"):
            pools.setdefault(key, Pool()).add(result)
    group_rows = {}
    for name, result in results.items():
        pool = Pool()
        pool.add(result)
        group_rows[name] = pool.summary(source_names, target_names, geometry.num_lags)

    report: Dict[str, Any] = {
        "script": "hdf5_dataset/new_pipeline/mask_coverage_audit.py",
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "root_name": root_path.name,
        "config": cfg_path.name,
        "geometry": {
            "sequence_length": geometry.sequence_length, "horizon": geometry.horizon,
            "warmup_period": geometry.warmup_period, "max_lag": geometry.max_lag,
            "lag_floor": geometry.lag_floor, "anchor_stride": geometry.anchor_stride,
            "coverage_floor": geometry.coverage_floor, "warmup_budget": geometry.warmup_budget,
            "lag_kv_source": geometry.lag_kv_source,
            "event_reach_steps": geometry.event_reach_steps, "event_floor": geometry.event_floor,
            "event_ua_valid": geometry.event_ua_valid,
            "source_region": [geometry.source_region.start, geometry.source_region.stop],
            "anchors": [geometry.anchors.start, geometry.anchors.stop],
        },
        "channels": {
            "resolved_by": channels.resolved_by,
            "source_keep": list(channels.source_keep),
            "source_names": list(channels.source_names),
            "source_warmup": list(channels.source_warmup),
            "target_keep_count": len(channels.target_keep),
        },
        "candidates": {"source": source_specs, "target": target_specs,
                       "drop_slowest": list(DROP_SLOWEST)},
        "sampling": {"segments_per_group": segments_per_group, "seed": seed,
                     "verify_segments": verify_segments, "events": events,
                     "datasets": datasets, "splits": splits, "subgroups": subgroups},
        "quality": quality,
        "verification": {"segments_checked": verified, "mismatches": 0},
        "groups": group_rows,
        "aggregates": {key: pools[key].summary(source_names, target_names, geometry.num_lags)
                       for key in sorted(pools, key=lambda k: (k == "ALL", k))},
        "channels_pooled": pools["ALL"].channel_table(list(source_specs), channels),
        "thresholds": THRESHOLDS,
    }
    report["totals"] = {key: report["aggregates"]["ALL"][key]
                        for key in ("groups", "files", "segments_sampled", "population")}
    report["decision_checks"] = decision_checks(report)
    report["elapsed_s"] = round(time.time() - started, 1)

    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / JSON_NAME).write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, out_path / MD_NAME)
    logger.info("wrote %s and %s in %.1f s", out_path / JSON_NAME, out_path / MD_NAME,
                report["elapsed_s"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    No argument is ``required`` and none carries a default: :data:`RUN_ARGS` fills every gap, and
    :func:`main` applies the real defaults after the merge.

    Returns:
        The parser.
    """
    p = argparse.ArgumentParser(
        prog="python hdf5_dataset/new_pipeline/mask_coverage_audit.py",
        description="Measure the UA source and FHR target quality masks as training reads them.")
    p.add_argument("--root", dest="root")
    p.add_argument("--output-dir", dest="output_dir")
    p.add_argument("--config", dest="config_path")
    p.add_argument("--datasets", dest="datasets", nargs="+")
    p.add_argument("--splits", dest="splits", nargs="+")
    p.add_argument("--subgroups", dest="subgroups", nargs="+")
    p.add_argument("--segments-per-group", dest="segments_per_group", type=int)
    p.add_argument("--seed", dest="seed", type=int)
    p.add_argument("--verify-segments", dest="verify_segments", type=int)
    p.add_argument("--events", dest="events", action=argparse.BooleanOptionalAction,
                   default=None)
    return p


def _cli(argv: Optional[List[str]] = None) -> int:
    """Merge the command line over :data:`RUN_ARGS` and run.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv[1:]``.

    Returns:
        The process exit code.

    Raises:
        SystemExit: If a required path was supplied by neither route.
    """
    values, sources = resolve_launch_args(build_parser(), RUN_ARGS, argv)
    refusal = missing_required(values, ("root", "output_dir"))
    if refusal:
        raise SystemExit(refusal)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("resolved arguments: " + ", ".join(
        f"{key}={values[key]!r} (from {sources[key]})" for key in sorted(values)))
    return main(**values)


#: **Edit the values below and press Run.** A flag on the command line overrides its entry.
#:
#: The output holds aggregates only (no GUID, signal or clinical value), so both files can be sent.
RUN_ARGS: Dict[str, Any] = {
    # ================================ FILL IN: the paths ================================
    # str -- the build root, which holds pre_training_dataset/ and
    #        k_fold_cross_validation_dataset/. Required.
    "root": r"/data1/fetal-heart-tracing/HDF5_Datasets/last-12-hours-causal-p50-ua-masked-2026-10-27",

    # str -- a folder for mask_coverage_audit.json and mask_coverage_audit.md; created if absent.
    #        A relative path resolves against the repository root. Required.
    "output_dir": r"output/mask_coverage_audit",

    # str | None -- the model config whose geometry, channels and masks are audited.
    #        (None -> teb_vae/lag_attn_transformer_cfs/configs/default.yaml)
    "config_path": None,

    # ================================ What to audit ================================
    # list[str] | None -- any of "pretraining", "fold_<k>", "kfold_holdout".
    #        One fold is enough: the folds hold the same GUIDs in other roles.
    #        (None -> ["pretraining", "fold_1"])
    "datasets": None,
    # list[str] | None -- e.g. ["train", "val"].  (None -> every split)
    "splits": None,
    # list[str] | None -- e.g. ["hie_cs", "acidosis_no_cs"].  (None -> every subgroup)
    "subgroups": None,

    # ========================= Everything below has a default =========================
    # int | None -- segments drawn per dataset/split/subgroup. About 20 ms each on a local disk,
    #        detector included, so 40 groups x 300 take a few minutes.  (None -> 300)
    "segments_per_group": None,
    # int | None -- the sampling seed.  (None -> 0)
    "seed": None,
    # int | None -- segments per group compared with dataset[i]; a difference stops the run.
    #        (None -> 4)
    "verify_segments": None,
    # bool | None -- run the contraction detector with and without up_valid.  (None -> True)
    "events": None,
}


if __name__ == "__main__":
    sys.exit(_cli())
