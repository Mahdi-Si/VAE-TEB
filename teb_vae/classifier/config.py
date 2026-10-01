r"""Classifier configuration: strict schema of the ``classifier:`` block, loading, digest (SPEC §13).

The YAML is a plain ``base:`` chain resolved by :func:`teb_vae.lag_attn.config.load_config`; this
module only validates it. Every block is a pydantic model with ``extra="forbid"``, so an unknown or
misspelt key is an error rather than a silently ignored setting. The shipped ``configs/default.yaml``
is the single source of defaults: fields here carry none unless the YAML legitimately omits them.
``advanced_config`` (§13.1) rides along unvalidated; ``GraphModelBase.validate_config`` owns it.
:func:`unit_config` derives the framework-shaped config of one training unit (§13.2).
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

from pydantic import (
    BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt,
    model_validator,
)

from teb_vae.lag_attn.config import _deep_merge, load_config
from teb_vae.lag_attn_cfs.lag_recovery_check import override_tree

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "teb_vae" / "classifier" / "configs" / "default.yaml"

#: Task -> {class_code: target}. A code absent from the map is excluded from that task (§6.3).
#: ``cs_outcome`` has no code map: its target is the ``cs`` flag.
TASKS: Dict[str, Dict[int, int]] = {
    "adverse_vs_healthy": {1: 0, 2: 1, 3: 1},
    "hie_vs_rest": {1: 0, 2: 0, 3: 1},
    "hie_vs_healthy": {1: 0, 3: 1},
    "acidosis_vs_healthy": {1: 0, 2: 1},
    "three_class": {1: 0, 2: 1, 3: 2},
    "cs_outcome": {},
}
#: ``labels.head`` -> the ``train.loss.name`` values that fit it (§10.2); ``losses.loss_fn`` implements each.
HEAD_LOSSES: Dict[str, Tuple[str, ...]] = {
    "binary": ("bce", "weighted_bce", "focal", "logit_adjusted", "auc_margin", "pauc"),
    "multiclass": ("ce", "weighted_ce", "focal_ce"),
    "ordinal": ("coral", "cumulative_link"),
}

#: Execution knobs that never change a result, so they are left out of the digest (§14.1 resume).
_UNDIGESTED_RUN_KEYS = ("device", "devices", "num_workers", "report_workers")
#: Run-dir schema version, hashed into :func:`digest`, so a run dir written under an older schema is refused on
#: resume. Bump it whenever a run-dir schema changes (cohort/prediction columns, context width, file layout).
SCHEMA_VERSION = 5  # 3: raw covariate columns in cohort/segments.parquet, covariate_availability.parquet (P5);
#                     4: 3-class prediction columns (p_c*_cal, ord_score) and thresholds.json["ovr"] (P6);
#                     5: pooling-attention scalars (attn_late_mass, attn_centroid, seq_attn_final) in
#                        predictions/segments.parquet (P6 block E)

#: Regimes whose ``train.unfreeze`` allowlist trains VAE parameters (§10.1); ``frozen_online`` runs the VAE online
#: without gradients.
ONLINE_TRAINABLE = ("partial", "lpft", "cotrain")

Unit = Annotated[float, Field(ge=0.0, le=1.0)]
Prob = Annotated[float, Field(gt=0.0, lt=1.0)]


class _Block(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---- run / data / cohort -----------------------------------------------------------------------
class RunCfg(_Block):
    name: str
    out_root: str
    seeds: List[int] = Field(min_length=1)
    folds: List[int] = Field(min_length=1)
    device: str
    # §14.4 fold-parallel train and predict: one fold process per slot (a device may repeat); [] runs folds serially
    # on `device`. Execution only, like `device`: not digested, so a run can resume serially or on other GPUs.
    devices: List[Annotated[str, Field(pattern=r"^(cpu|cuda:\d+)$")]] = []
    num_workers: NonNegativeInt
    report_workers: NonNegativeInt  # figure processes of `report`; 0 renders serially
    fail_fast: bool
    plot_frequency: PositiveInt


class SplitDirs(_Block):
    train: str
    val: str
    test: str


class DataCfg(_Block):
    kfold_root: str
    fold_dir: str
    split_dirs: SplitDirs
    subgroups: Union[Literal["all"], List[str]]
    stride_s: Union[Literal["auto"], float]
    epoch_min_s: float  # evaluation window: val and test share it (L6); train too unless train_epoch_min_s
    train_epoch_min_s: Optional[float]  # training window; null = epoch_min_s
    min_valid_frac: Unit
    patient_map: Optional[str]
    shared_test_policy: Literal["first_fold", "exclude"]
    allow_pretrain_overlap: bool


class CohortCfg(_Block):
    include_healthy_no_bg: bool
    min_segments_per_guid: PositiveInt


class LabelsCfg(_Block):
    task: Literal[tuple(TASKS)]  # type: ignore[valid-type]
    head: Literal["binary", "multiclass", "ordinal"]
    aux_3class_weight: NonNegativeFloat
    strategy: Literal["propagate", "horizon", "horizon_decay", "final_only", "mil"]
    horizon_h: PositiveFloat
    decay_halflife_h: PositiveFloat
    k_warm: Union[Literal["auto"], NonNegativeInt]  # auto: 3 for propagate, 0 otherwise (cohort.warm_positions)
    eval_window: Literal["all", "horizon", "bins", "stage:first", "stage:second"]

    @model_validator(mode="after")
    def _head_fits_task(self) -> "LabelsCfg":
        k = 3 if self.task == "three_class" else 2
        if (self.head == "binary" and k != 2) or (self.head == "ordinal" and k < 3):
            raise ValueError(f"labels.head={self.head!r} does not fit task {self.task!r} (K={k})")
        return self


# ---- context -----------------------------------------------------------------------------------
class Toggle(_Block):
    enabled: bool


class TloCfg(Toggle):
    missing: Literal["indicator", "no_indicator"]
    pre_onset: Literal["clip", "signed"]  # clip: ψ(max(tlo, 0)); signed encodes time UNTIL onset (ablation only)


class Variable(_Block):
    name: str
    kind: Literal["numeric", "categorical"]
    available_at: Literal["prospective"]  # §7.3.6: only prospectively available variables


class CovariatesCfg(_Block):
    static_csv: Optional[str]
    timed_csv: Optional[str]
    variables: List[Variable]
    max_age_h: PositiveFloat
    age_feature: bool
    missing: Literal["indicator", "no_indicator"]
    dropout_p: Unit
    block_dropout_p: Unit


class ContextCfg(_Block):
    tlo: TloCfg
    stage: Toggle
    time_in_ss: Toggle
    delta_t: Toggle
    elapsed: Toggle
    valid_frac: Toggle
    covariates: CovariatesCfg
    fusion: Literal["concat", "film", "token", "late"]
    missing_confound_max: Unit
    auto_ablate_missing: bool


# ---- source ------------------------------------------------------------------------------------
class VaeKey(_Block):
    name: Literal[
        "mu_prior", "mu_post", "logvar_prior", "logvar_post", "target_state", "source_state",
        "kld_per_t", "kld_per_t_per_head", "source_kl_lag_map", "attn_weights", "delta_mu",
        "kld_per_dim", "attn_summary", "kld_excess",
    ]
    transform: Literal["none", "log1p", "asinh"] = "none"
    role: Literal["value", "attention"] = "value"


class VaeCfg(_Block):
    package: str
    checkpoint: str
    keys: List[VaeKey] = Field(min_length=1)
    step_support: Literal["causal_all", "supervised"]
    lag_bins: List[Tuple[int, int]]
    sample_z_train: bool


class Hdf5Cfg(_Block):
    fields: List[Literal["fhr_st", "fhr_ph", "up_st", "up_ph", "fhr_up_ph", "raw_fhr", "raw_up"]] = (
        Field(min_length=1)
    )
    stats_path: str
    trim_minutes: NonNegativeFloat
    min_step: Union[Literal["auto"], int, None]


class SourceCfg(_Block):
    kind: Literal["vae", "hdf5"]
    vae: VaeCfg
    hdf5: Hdf5Cfg
    time_pool: PositiveInt
    cache_root: str
    cache_dtype: Literal["float16", "float32"]


# ---- model -------------------------------------------------------------------------------------
class StepCfg(_Block):
    d: PositiveInt
    temporal: Literal["none", "causal_conv", "transformer"]
    dropout: Unit


class PoolingCfg(_Block):
    kind: Literal["mean", "mean_max", "gated_attention", "query", "conjunctive"]
    d_attn: PositiveInt


class TokenCfg(_Block):
    d: PositiveInt
    dropout: Unit


class SequenceCfg(_Block):
    kind: Literal["causal_transformer", "gru", "attention_mil"]
    causal: bool
    layers: PositiveInt
    heads: PositiveInt
    d_ff: PositiveInt
    dropout: Unit
    time_bias_buckets: PositiveInt
    time_bias_max_h: PositiveFloat


class HeadCfg(_Block):
    hidden: PositiveInt
    dropout: Unit
    bias_init: Literal["prior"]


class ModelCfg(_Block):
    scope: Literal["segment", "sequence"]
    step: StepCfg
    pooling: PoolingCfg
    token: TokenCfg
    sequence: SequenceCfg
    segment_head: bool
    head: HeadCfg
    segment_aggregators: List[Literal["max", "mean", "lse", "last", "topk_mean"]]
    lse_tau: PositiveFloat


# ---- train -------------------------------------------------------------------------------------
class LossCfg(_Block):
    name: Literal[
        "bce", "weighted_bce", "focal", "logit_adjusted", "auc_margin", "pauc", "ce",
        "weighted_ce", "focal_ce", "coral", "cumulative_link",
    ]
    weighting: Literal["none", "inverse", "sqrt_inverse", "effective_number"]
    beta_en: float = Field(ge=0.0, lt=1.0)  # 1.0 makes every effective-number weight 0/0
    focal_gamma: NonNegativeFloat
    focal_alpha: Optional[float]
    logit_adjust_tau: float
    label_smoothing: float = Field(ge=0.0, lt=1.0)


class LossWeightsCfg(_Block):
    final: NonNegativeFloat
    positions: NonNegativeFloat
    bag: NonNegativeFloat
    segment: NonNegativeFloat


class OptimizerCfg(_Block):
    lr: PositiveFloat
    weight_decay: NonNegativeFloat
    betas: Tuple[float, float]


class ScheduleCfg(_Block):
    warmup_steps: NonNegativeInt
    kind: Literal["cosine"]
    min_lr_frac: Unit


class CotrainCfg(_Block):
    vae_weight: NonNegativeFloat
    cls_weight: NonNegativeFloat
    l2sp: NonNegativeFloat
    detach_head_input: bool
    vae_chunk: PositiveInt
    grad_segments: str = Field(pattern=r"^(all|last_k:[1-9]\d*)$")  # last_k:0 would train no segment
    gates: Dict[Literal["forecast_mse_rel"], float]


class TrainCfg(_Block):
    regime: Literal["frozen_cached", "frozen_online", "partial", "lpft", "cotrain"]
    unfreeze: List[str]
    backbone_lr: PositiveFloat
    lpft_head_epochs: NonNegativeInt
    loss: LossCfg
    loss_weights: LossWeightsCfg
    sampler: Literal["natural", "class_balanced"]
    batch_guids: PositiveInt
    batch_segments: PositiveInt
    segment_dropout: Unit
    optimizer: OptimizerCfg
    schedule: ScheduleCfg
    max_epochs: PositiveInt
    ema: Optional[Prob]
    cotrain: CotrainCfg


class CalibrationCfg(_Block):
    method: Literal["temperature", "platt", "none"]


class BaselinesCfg(_Block):
    probe_last_n: PositiveInt
    shortcut_warn_auroc: Unit
    shuffled_control: bool


# ---- eval --------------------------------------------------------------------------------------
TimeAxis = Literal["to_delivery", "from_onset", "rel_second_stage", "position", "elapsed"]


class ThresholdCfg(_Block):
    id: str
    policy: Literal["fpr_cap", "youden", "sens_target", "fixed"]
    basis: Literal["guid_final", "instantaneous", "committed_cumulative", "committed_overall",
                   "segment"]
    axis: TimeAxis = "to_delivery"
    at: Union[Literal["end"], float] = "end"
    alpha: Optional[Prob] = None
    method: Optional[Literal["empirical", "np_umbrella"]] = None
    delta: Optional[Prob] = None
    allow_fallback: bool = False
    beta: Optional[Prob] = None  # sens_target: validation TPR >= beta
    value: Optional[float] = None  # fixed: a calibrated probability

    @model_validator(mode="after")
    def _policy_keys(self) -> "ThresholdCfg":
        need = {"fpr_cap": ("alpha", "method"), "sens_target": ("beta",), "fixed": ("value",)}.get(self.policy, ())
        need += ("delta",) if self.method == "np_umbrella" else ()
        missing = [key for key in need if getattr(self, key) is None]
        if missing:
            raise ValueError(f"threshold policy {self.id!r} ({self.policy}, method {self.method}) needs {missing}")
        if self.basis == "segment" and self.method not in (None, "empirical"):
            raise ValueError(f"threshold policy {self.id!r}: basis segment allows only method empirical (segments "
                             f"are correlated, so NP guarantees are void, SPEC §11.3)")
        return self


class AlarmRuleCfg(_Block):
    kind: Literal["latch", "k_of_n"]
    k: Optional[PositiveInt] = None
    n: Optional[PositiveInt] = None


class BootstrapCfg(_Block):
    resamples: PositiveInt
    refit_threshold: bool
    seed: int


class EvalCfg(_Block):
    levels: List[Literal["segment", "guid", "online"]]
    thresholds: List[ThresholdCfg] = Field(min_length=1)
    primary_policy: str
    report_tpr_at_fpr: List[float]
    metric_types: List[Literal["instantaneous", "committed_cumulative", "committed_overall"]]
    exclude_last_min: NonNegativeFloat
    alarm_rule: AlarmRuleCfg
    checkpoints_h: List[float]
    snapshot_max_staleness_h: PositiveFloat
    time_axes: List[TimeAxis]
    bin_h: PositiveFloat
    min_bin_class_n: NonNegativeInt
    subgroup_families: List[str]
    restricted_pairs: List[Tuple[str, str]]
    min_subgroup_n: NonNegativeInt
    ovr_thresholds: Union[Literal["auto"], bool]
    decision_horizons_h: List[float]
    per_fold_figures: bool
    fold_band: Literal["minmax", "none"]
    figure_formats: List[Literal["pdf", "png", "svg"]]
    error_analysis: Dict[Literal["top_k"], int]
    attribution: Dict[Literal["enabled", "n_steps"], Union[bool, int]]
    covariates_off: bool
    bootstrap: BootstrapCfg
    reference_prevalence: Optional[Prob]
    trajectory_pages: Dict[Literal["per_class", "top_errors"], int]

    @model_validator(mode="after")
    def _policies_consistent(self) -> "EvalCfg":
        ids = [policy.id for policy in self.thresholds]
        if len(set(ids)) != len(ids) or self.primary_policy not in ids:
            raise ValueError(f"threshold ids must be unique and contain primary_policy; got {ids}")
        return self


# ---- root --------------------------------------------------------------------------------------
class Classifier(_Block):
    run: RunCfg
    data: DataCfg
    cohort: CohortCfg
    labels: LabelsCfg
    context: ContextCfg
    source: SourceCfg
    model: ModelCfg
    train: TrainCfg
    calibration: CalibrationCfg
    baselines: BaselinesCfg
    eval: EvalCfg

    @model_validator(mode="after")
    def _cross_field(self) -> "Classifier":
        loss = self.train.loss
        if not self.model.sequence.causal and self.labels.strategy not in ("final_only", "mil"):
            raise ValueError("model.sequence.causal: false requires labels.strategy final_only|mil")
        if self.model.scope == "segment" and self.labels.strategy == "mil":
            raise ValueError("labels.strategy: mil needs model.scope: sequence (segment scope has no bag term, "
                             "so the loss would be identically 0)")
        if loss.name == "focal" and loss.focal_alpha is None:
            raise ValueError("train.loss.name: focal needs an explicit train.loss.focal_alpha (SPEC §10.2)")
        if loss.name in ("auc_margin", "pauc") and self.train.sampler != "class_balanced":
            raise ValueError(f"train.loss.name: {loss.name} needs train.sampler: class_balanced (a pairwise loss needs "
                             f"both classes in a batch, SPEC §10.2, §10.4)")
        if loss.name in ("auc_margin", "pauc") and self.calibration.method == "none":
            raise ValueError(f"train.loss.name: {loss.name} scores are not probabilities: calibration.method must be "
                             f"platt (recommended: it fits the intercept) or temperature, not none (SPEC §10.2)")
        return self

    @model_validator(mode="after")
    def _three_class_fits(self) -> "Classifier":
        """Head, loss, calibration and OvR thresholds agree (§6.3, §10.2, §10.8, §11.3)."""
        lab, loss = self.labels, self.train.loss.name
        if loss not in HEAD_LOSSES[lab.head]:
            raise ValueError(f"train.loss.name: {loss} does not fit labels.head: {lab.head}; expected one of "
                             f"{HEAD_LOSSES[lab.head]}")
        if lab.task == "three_class" and lab.aux_3class_weight > 0:
            raise ValueError("labels.aux_3class_weight must be 0 for labels.task: three_class (the main head is "
                             "already 3-class; λ3 adds a 3-class head to a binary task, §6.3)")
        if self.calibration.method == "platt" and lab.head != "binary":
            raise ValueError(f"calibration.method: platt is binary only; labels.head: {lab.head} takes temperature "
                             f"(multiclass T, ordinal scale + offsets) or none (§10.8)")
        if self.eval.ovr_thresholds is True and lab.task != "three_class":
            raise ValueError("eval.ovr_thresholds: true needs labels.task: three_class (calibrated class "
                             "probabilities, §11.3)")
        return self

    @model_validator(mode="after")
    def _context_fits(self) -> "Classifier":
        """P5 context checks (§6.5, §7.3, §9.4)."""
        cov, seg = self.context.covariates, self.model.scope == "segment"
        if seg and self.labels.k_warm not in ("auto", 0):
            raise ValueError("labels.k_warm drops the first positions of a sequence (§6.5); segment scope takes auto|0")
        names = [v.name for v in cov.variables]
        if len(set(names)) != len(names):
            raise ValueError(f"context.covariates.variables names must be unique; got {names}")
        if names and not (cov.static_csv or cov.timed_csv):
            raise ValueError("context.covariates.variables need a static_csv and/or timed_csv (§7.3)")
        if names and seg and self.context.fusion == "token":
            raise ValueError("context.fusion: token is sequence scope only (§9.4)")
        return self

    @model_validator(mode="after")
    def _regime_fits(self) -> "Classifier":
        """§10.1: online regimes run a VAE; the trainable ones name their allowlist, the frozen ones none."""
        t = self.train
        if t.regime != "frozen_cached" and self.source.kind != "vae":
            raise ValueError(f"train.regime: {t.regime} runs the encoder online and needs source.kind: vae")
        if (t.regime in ONLINE_TRAINABLE) != bool(t.unfreeze):
            raise ValueError(f"train.unfreeze {t.unfreeze} does not fit train.regime: {t.regime}: {ONLINE_TRAINABLE} "
                             f"need a non-empty allowlist of VAE module prefixes, the frozen regimes an empty one")
        if self.source.vae.sample_z_train and t.regime != "frozen_online":
            raise ValueError("source.vae.sample_z_train is a frozen_online augmentation (§10.1)")
        if t.regime == "cotrain" and t.cotrain.grad_segments != "all" and self.model.scope != "sequence":
            raise ValueError("train.cotrain.grad_segments: last_k counts positions of a GUID; segment scope takes all")
        return self


class Config(_Block):
    """The whole file: the validated ``classifier`` block plus the framework's ``advanced_config``."""

    classifier: Classifier
    advanced_config: Dict[str, Any] = {}

    @model_validator(mode="after")
    def _checkpoint_follows_early_stopping(self) -> "Config":
        ckpt = (self.advanced_config.get("callbacks") or {}).get("model_checkpoint") or {}
        want = dict(zip(("monitor", "mode"), selection_monitor(self.advanced_config, self.classifier.train.regime)))
        if any(key in ckpt and ckpt[key] != value for key, value in want.items()):
            raise ValueError(f"advanced_config.callbacks.model_checkpoint {ckpt} differs from early_stopping {want}: "
                             f"best.ckpt is selected on the early-stopping monitor (SPEC §10.10.2); drop monitor/mode")
        return self


def selection_monitor(advanced: Mapping[str, Any], regime: Optional[str] = None) -> Tuple[str, str]:
    """``(monitor, mode)`` of ``best.ckpt``: the early-stopping ones (SPEC §10.5, §10.10.2 #6); under ``cotrain`` the
    monitor's ``_gated`` twin (±∞ on an epoch the preservation gate fails, §10.10.2 #12). Early stopping keeps the
    ungated one: it stops on a non-finite monitor."""
    es = (advanced.get("callbacks") or {}).get("early_stopping") or {}
    monitor = es.get("monitor", "val/guid_logloss")
    return monitor + ("_gated" if regime == "cotrain" else ""), es.get("mode", "min")


def ovr_enabled(c: Classifier) -> bool:
    """``eval.ovr_thresholds`` resolved: ``auto`` is true for ``labels.task: three_class`` (§11.3 T5)."""
    return c.labels.task == "three_class" and c.eval.ovr_thresholds is not False


def load(path: Any = DEFAULT_CONFIG,
         overrides: Union[Sequence[str], Mapping[str, Any]] = ()) -> Config:
    """Resolve ``path``'s ``base:`` chain, apply overrides, validate.

    Args:
        path: The leaf YAML.
        overrides: ``dotted.key=value`` strings (values parsed as YAML, full path from the top,
            e.g. ``classifier.labels.task=hie_vs_rest``) or an already-nested delta mapping.

    Raises:
        pydantic.ValidationError: On any unknown key, wrong type or out-of-range value.
    """
    delta = dict(overrides) if isinstance(overrides, Mapping) else override_tree(list(overrides))
    return Config.model_validate(_deep_merge(load_config(str(path)), delta))


def digest(cfg: Config) -> str:
    """SHA-256 of the canonical JSON of ``cfg``, minus the execution-only ``run`` knobs, plus :data:`SCHEMA_VERSION`."""
    payload = cfg.model_dump(mode="json") | {"schema_version": SCHEMA_VERSION}
    for key in _UNDIGESTED_RUN_KEYS:
        payload["classifier"]["run"].pop(key)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def resolve_path(path: Any) -> Path:
    """A config path, relative ones resolved against the repository root (§13)."""
    return Path(path) if Path(path).is_absolute() else REPO_ROOT / path


def unit_dir(run_dir: Any, fold: int, seed: int, kind: str) -> Path:
    """``<run>/folds/fold_<k>/seed_<s>`` (kind ``model``) or ``.../seed_<s>/<kind>`` (§14.2): ``shuffled`` (§10.9.3),
    ``noind`` (the §7.3.4 ``no_indicator`` ablation), ``frozen`` (an online regime's frozen baseline, §10.1)."""
    if kind not in ("model", "shuffled", "noind", "frozen"):
        raise ValueError(f"unit kind must be model|shuffled|noind|frozen, got {kind!r}")
    base = Path(run_dir) / "folds" / f"fold_{fold}" / f"seed_{seed}"
    return base if kind == "model" else base / kind


def frozen_baseline(cfg: Config) -> Config:
    """``cfg`` under ``train.regime: frozen_cached`` with no allowlist: the ``frozen`` unit every online regime trains
    beside its own at the same seed, so the report can read the regime against frozen features (§10.1)."""
    c = cfg.classifier
    train = c.train.model_copy(update={"regime": "frozen_cached", "unfreeze": []})
    source = c.source.model_copy(update={"vae": c.source.vae.model_copy(update={"sample_z_train": False})})
    return cfg.model_copy(update={"classifier": c.model_copy(update={"train": train, "source": source})})


def unit_config(cfg: Config, *, run_dir: Any, fold: int, seed: int, kind: str,
                mlflow_parent_id: Optional[str] = None) -> Dict[str, Any]:
    """The framework-shaped config ``GraphModelBase`` loads for one unit (§13.2).

    ``model_config.classifier`` is the whole resolved block (flattened into MLflow params, §10.10.5);
    ``advanced_config`` is copied with the unit's MLflow run name and nesting tags, ``log_model`` and
    ``log_checkpoints`` forced off (F10). ``cuda_devices`` is ``[]`` for ``run.device: cpu`` (F5). An
    ``early_stopping`` block gets explicit ``monitor``/``mode`` from :func:`selection_monitor`, so it and ``best.ckpt``
    share one default (``GraphModelBase`` would otherwise fall back to ``val/total_loss``).
    """
    c = cfg.classifier
    advanced = copy.deepcopy(cfg.advanced_config)
    mlflow = advanced.setdefault("tracking", {}).setdefault("mlflow", {})
    parent = {"mlflow.parentRunId": mlflow_parent_id} if mlflow_parent_id else {}
    mlflow.update(run_name=f"fold{fold}-seed{seed}-{kind}", log_model=False, log_checkpoints=False,
                  tags={**(mlflow.get("tags") or {}), **parent, "fold": str(fold), "seed": str(seed), "kind": kind})
    es = (advanced.get("callbacks") or {}).get("early_stopping")
    if es is not None:  # one default for early stopping and best.ckpt: the base's own EarlyStopping default differs
        es["monitor"], es["mode"] = selection_monitor(advanced)
    device = c.run.device
    batch = c.train.batch_guids if c.model.scope == "sequence" else c.train.batch_segments
    return {
        "general_config": {
            "tag": f"{c.run.name}-fold{fold}-seed{seed}" + ("" if kind == "model" else f"-{kind}"),
            "seed": seed + 1000 * fold,
            "cuda_devices": [] if device == "cpu" else [int(device.partition(":")[2] or 0)],
            "epochs": c.train.max_epochs, "lr": c.train.optimizer.lr, "lr_milestone": [],
            "plot_frequency": c.run.plot_frequency, "batch_size": {"train": batch, "test": batch},
            "folders_config": {"out_dir_base": str(unit_dir(run_dir, fold, seed, kind))},
        },
        "model_config": {"classifier": c.model_dump(mode="json")},
        "advanced_config": advanced,
    }
