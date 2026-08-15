r"""The four facts this pipeline cannot derive about the model it is evaluating.

Everything else in this package reads a model through an interface two architectures already
share: the objective, the geometry, the data contract, the prior head, the lag cross-attention and
the head-structured posterior are one implementation, and the analyses reach only into ``nets``
and the tables the collection pass wrote. What is *not* derivable is which class to rebuild, which
constructor keys mean enough to reconcile against a checkpoint, what the model's own encoder has
to disclose about its causal standing, and which committed override delta belongs to it.

Those four facts are what a :class:`ModelBinding` carries, and passing one is what lets a second
architecture reuse this pipeline rather than fork it. A fork is how two models that must stay
comparable stop being comparable: an analysis fixed on one side keeps its bug on the other, and
the two ``summary.json`` files stop being readable side by side long before anyone notices.

The :class:`ModelBinding` dataclass itself names no model class and is stdlib-only, exactly as the
sibling's is: it is read by documentation tests and by anything that wants the *type* without a
numeric stack. What this module adds beside it is this cell's own concrete
:data:`CFS_BINDING`, which does name one -- so **this module is layer 1 here and layer 0 in the
sibling**, and the layering test says so with a named exemption rather than leaving the difference
to be discovered. The one rule that follows: nothing which must import without ``torch``, the
acceptance gate above all, may import this module. ``verify.py`` reads a finished ``summary.json``
and needs no binding at all, which is what makes that rule cost nothing.

The binding lives here rather than in ``run.py`` (where the sibling keeps its instance) because
the second cell's own binding is reconciled against ``preflight`` long before either package has
a runner: a declaration that can only be wrong in one interesting way -- a key that names nothing
-- is cheap to check as soon as the guards exist and expensive to check after everything else has
been validated against one cell alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Tuple

from teb_vae.lag_attn_cfs.eval.analyses import source_null as source_null_analysis
from teb_vae.lag_attn_cfs.eval.analyses import spectral_skill as spectral_skill_analysis
from teb_vae.lag_attn_cfs.eval.analyses import warmup as warmup_analysis
from teb_vae.lag_attn_cfs.eval.config_schema import DEFAULT_OVERRIDES_PATH
from teb_vae.lag_attn_cfs.eval.preflight import cfs_encoder_disclosure
from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs
from teb_vae.lag_attn_cfs.task import SeqVaeLagAttnCfsTask


@dataclass(frozen=True)
class ModelBinding:
    """Which model a run of this pipeline is evaluating, and what only that model knows.

    Frozen because a binding is a declaration rather than a run-time setting: every field decides
    what the numbers in a run *mean*, and a value mutated after the run started would leave a
    ``summary.json`` describing a contract that was not in force when the tables were collected.

    Attributes:
        model_cls: The network class rebuilt from a checkpoint's own ``model_kwargs``, and the
            name every class-mismatch refusal is built from. Wrong, and the run either refuses
            with a message naming the wrong class or -- worse, if the two constructors happen to
            accept the same keys -- evaluates one architecture under another's name.
        task_cls: The objective wrapper the model is scored through. Wrong, and the readouts are
            some other loss reported under this one's column headings.
        tag: The output-directory fallback used when ``general_config.tag`` is absent, as
            ``<tag>-eval``. Wrong, and two models' runs land in one directory and are told apart
            only by timestamp.
        geometry_keys: The constructor keys reconciled against the checkpoint's ``model_kwargs``.
            A key missing here is a key the config may contradict silently; a key here that the
            model does not accept can never match and refuses every run.
        encoder_disclosure: Called with the rebuilt net, returning the encoder-specific half of
            the causality record. What is true of one encoder is not true of the other, and a
            shared key that means nothing in one of them is worse than two honest blocks.
        overrides_path: The committed evaluation override delta merged over a checkpoint's own
            resolved config. Wrong, and the run evaluates the right checkpoint against another
            package's holdout split.
        extra_analyses: Analyses only this model can have, merged into the shared registry in
            declaration order -- after every shared analysis but the ones the runner keeps
            trailing, which read what the steps above them wrote. Empty for a model that adds
            none, which is the common case and says so by omission. A name already in the shared
            registry is a collision rather than an override: silently replacing a shared
            implementation would make two models report different things under one name.
        headline_scalars: Additional ``(name, path into results)`` entries appended to the shared
            headline registry, for what :attr:`extra_analyses` produces. Appended rather than
            merged into the shared tuple, and empty for a model that adds none: the shared
            registry's every path must resolve on a shared run, so a model-specific entry there
            would read as a number every other model failed to produce. A number that stays out of
            the headline stays out of every arm table too, which is why an extra analysis with a
            scalar worth comparing declares it here.
    """

    model_cls: type
    task_cls: type
    tag: str
    geometry_keys: Tuple[str, ...]
    encoder_disclosure: Callable[[Any], Dict[str, Any]]
    overrides_path: Path
    extra_analyses: Mapping[str, Any] = field(default_factory=dict)
    headline_scalars: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()


# =================================================================================================
# This cell's binding
# =================================================================================================
#: Constructor keys reconciled against the checkpoint's own ``model_kwargs``.
#:
#: **The rule a key must satisfy is narrower than "it is a constructor parameter", and the reason
#: is that ``preflight.reconcile`` silently skips a key absent from either side**
#: (``if key not in vae_config or key not in model_kwargs: continue``). So a key must be a
#: constructor parameter *and* a ``model_config.VAE_model`` key to be checked at all, and listing
#: one that is not is a silent no-op rather than a refusal. ``tests/test_eval_binding.py`` asserts
#: both halves against the class and against the shipped ``configs/default.yaml``.
#:
#: The sibling's fourteen, plus this cell's two. ``anchor_stride`` decides how many anchors a
#: forward scores and therefore what population every number is computed over; ``lag_floor`` is one
#: of the three quantities the lag-support margin is made of, and a non-zero one silently
#: reintroduces the truncation the shipped geometry does not have.
#:
#: **``causal_warmup_budget_steps`` is deliberately absent, and so are the four resolved tuples.**
#: The budget is a config key but **not** a constructor parameter: the driver resolves it against
#: the shards into ``target_keep_index``, ``target_warmup_steps``, ``source_keep_index`` and
#: ``source_warmup_steps``, and those four are what land in ``model_kwargs``. They are in turn not
#: config keys, so reconciling them here would compare a checkpoint value against nothing and pass
#: every run. They are checked instead by their own guard in ``preflight``, which re-resolves the
#: budget from the configured shards and compares the result against the checkpoint's stamped
#: tuples -- the only comparison that can actually fail. ``decoder_out_channels`` is absent for the
#: mirror-image reason: a parameter of this constructor but of no config, and recoverable from the
#: stamped ``target_keep_index`` anyway.
GEOMETRY_KEYS: Tuple[str, ...] = (
    "sequence_length",
    "d_model",
    "d_z",
    "horizon",
    "raw_per_step",
    "warmup_period",
    "c_y",
    "c_u",
    "use_up_st",
    "max_lag",
    "num_heads",
    "d_head",
    "horizon_attention_blocks",
    "causal_norm",
    "anchor_stride",
    "lag_floor",
)

#: Analyses only this cell can have, merged onto the shared registry in declaration order. They are
#: registered **here rather than on the shared registry** so that the fork's run order stays
#: readable as "the shared twelve, then this cell's three" -- and so that the second cfs cell picks
#: all three up by binding this pipeline rather than by editing it.
#:
#: The three are the questions only a causal cell can ask: where in the warm-up staircase the
#: forecast gap lives and whether the run decoded the population its configuration describes
#: (``warmup``); how much of the coupling readout survives zeroing the source, which is the
#: availability-clock hazard no permutation control can see (``source_null``); and the forecast
#: resolved by the frequency band of the target coefficient, which is the readout this target
#: domain has instead of the raw pipeline's phase-domain pair (``spectral_skill``).
EXTRA_ANALYSES: Dict[str, Any] = {
    "warmup": warmup_analysis.run_warmup_analysis,
    "source_null": source_null_analysis.run_source_null_analysis,
    # Last of the three: it is the only one whose input is a file another step wrote, the kept-axis
    # channel map, so it reads rather than produces.
    "spectral_skill": spectral_skill_analysis.run_spectral_skill_analysis,
}

#: What those analyses put in the headline block, which is the only block an arm table reads --
#: appended to the shared registry rather than added to it, because every path in *that* tuple must
#: resolve on a run of every model that uses this pipeline.
#:
#: ``coupling_minus_clock_nats`` is the load-bearing entry and it is registered unconditionally.
#: The verdict it belongs to ships with **no threshold**, so on the first production runs it is
#: INCONCLUSIVE -- and the whole point of shipping it that way is that the *measurement* still
#: reaches every arm table, so the threshold can eventually be set from the observed spread rather
#: than guessed. A number that stays out of the headline stays out of every arm table too.
#:
#: The two geometry guards are here for the mirror-image reason: they are exact structural numbers
#: rather than statistics, and an arm table whose rows disagree about the anchor count is an arm
#: table comparing two populations.
#:
#: The five per-band gaps come through here rather than through the shared tuple because no other
#: cell in the grid has a channel axis that *is* a frequency axis. Every band the shared table
#: declares gets a column whether or not a given dataset had channels in it, so an arm comparison's
#: column set does not change with the shards.
#:
#: **Every path is keyed all the way down.** Each of the three analyses assembles a flat block of
#: finite scalars for exactly this purpose, as the shared ``coupling`` analysis already does, rather
#: than being indexed into by position: a path whose last step is a list index resolves to the wrong
#: row the day a metric is added above it, and nothing in the artifact would say so.
HEADLINE_SCALARS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("kld_source_null_nats", ("source_null", "difference", "kld_source_null_nats")),
    ("coupling_minus_clock_nats", ("source_null", "difference", "coupling_minus_clock_nats")),
    ("coupling_minus_clock_ci_lo", ("source_null", "difference", "ci_lo")),
    ("coupling_minus_clock_ci_hi", ("source_null", "difference", "ci_hi")),
    ("pred_gap_warm_lo_nats", ("warmup", "headline", "pred_gap_warm_lo_nats")),
    ("pred_gap_warm_mid_nats", ("warmup", "headline", "pred_gap_warm_mid_nats")),
    ("pred_gap_warm_hi_nats", ("warmup", "headline", "pred_gap_warm_hi_nats")),
    ("source_lag_warmth_frac_st", ("warmup", "headline", "source_lag_warmth_frac_st")),
    ("source_lag_warmth_frac_ph", ("warmup", "headline", "source_lag_warmth_frac_ph")),
    ("anchors_per_sample", ("warmup", "geometry_guards", "anchors_per_sample")),
    ("target_warm_frac", ("warmup", "geometry_guards", "target_warm_frac")),
    (
        "spectral_gap_slow_baseline_nats",
        ("spectral_skill", "headline", "pred_gap_slow_baseline_nats"),
    ),
    (
        "spectral_gap_deceleration_nats",
        ("spectral_skill", "headline", "pred_gap_deceleration_nats"),
    ),
    (
        "spectral_gap_variability_nats",
        ("spectral_skill", "headline", "pred_gap_variability_nats"),
    ),
    (
        "spectral_gap_beat_to_beat_nats",
        ("spectral_skill", "headline", "pred_gap_beat_to_beat_nats"),
    ),
    ("spectral_gap_unknown_nats", ("spectral_skill", "headline", "pred_gap_unknown_nats")),
)


#: The model this pipeline evaluates by default.
#:
#: ``model_cls`` and ``task_cls`` are what a checkpoint is rebuilt through, so a wrong one either
#: refuses by name or -- if the two constructors happen to accept the same keys -- evaluates one
#: architecture under another's name. The tag decides where a run with no configured tag lands.
#:
#: ``encoder_disclosure`` is ``preflight.cfs_encoder_disclosure``, which reports ``causal_norm`` and
#: ``n_causalized_norms`` -- the recurrent encoder's own guard, and the only part of this cell's
#: causality record that is a property of the *encoder*. Everything else that record carries (the
#: one-sidedness, the group delays, the resolved warm-up budget, the anchor geometry and the measured
#: lag support) is a property of the target domain, is true of both cfs cells, and is therefore owned
#: by the shared half of ``preflight.causality_disclosure`` rather than by either binding.
CFS_BINDING = ModelBinding(
    model_cls=SeqVaeLagAttnCfs,
    task_cls=SeqVaeLagAttnCfsTask,
    tag="lag_attn_cfs",
    geometry_keys=GEOMETRY_KEYS,
    encoder_disclosure=cfs_encoder_disclosure,
    overrides_path=DEFAULT_OVERRIDES_PATH,
    extra_analyses=EXTRA_ANALYSES,
    headline_scalars=HEADLINE_SCALARS,
)
