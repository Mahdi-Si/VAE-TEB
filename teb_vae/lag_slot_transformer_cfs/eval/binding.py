r"""What this pipeline cannot derive about the model it scores, declared once.

The :class:`~teb_vae.lag_attn_cfs.eval.binding.ModelBinding` type is the shared one, bound rather
than forked: a second copy of it would be a second definition of what a run *means*, and the two
would drift. What is written here is one instance of it, and each field decides something no
amount of reading the checkpoint could recover.

Three of the fields differ from the lag-attentive cells' in ways worth stating outright, and
three more are what let the family's runner evaluate this model at all: ``collect`` names this
cell's own collection pass, because the shared one reads tensors only a lag-attention forward
emits; ``extra_analyses`` registers the readouts only this architecture has beside this cell's own
pages, traces and attributions; and ``headline_scalars`` puts this cell's paired gap interval and
source-control margins into the block every arm table reads.

``geometry_keys``
    The attention keys are gone, because this architecture refuses them at construction, and five
    of its own take their place. The rule a key must satisfy is unchanged and is narrower than "it
    is a constructor parameter": ``preflight.reconcile`` silently skips a key absent from either
    side, so a key must be a constructor parameter **and** a ``model_config.VAE_model`` key to be
    compared at all, and listing one that is not is a no-op that never says so.

``encoder_disclosure``
    The lag-attentive cells disclose their recurrent encoder's causalisation guard. There is no
    such guard here and no such key; what this architecture has to disclose instead is how far its
    **source** pathway reaches, beside how far its lag window searches. The two are different
    numbers and the gap between them is the point. On the recommended arm the reach is one stored
    sample; on the comparator arm that keeps the convolution stem it is the stem's own, read off
    the module, because that number is the resolution floor of every lag readout taken over that
    representation and the whole reason the two arms are compared.

``excluded_analyses``
    Seven analyses in the wider family read a tensor this architecture does not compute. Two of
    them are in the *shared* registry, so this binding removes them by name; the other five are the
    lag-attentive cell's own extras, which this binding simply never registers. All seven are
    recorded with the reason, because a reader comparing two runs sees seven columns missing and
    should not have to work out which mechanism left each one out.

    None of them is handed a substitute, which is the one outcome the design rules out explicitly:
    no proposal norm under an attention name, and no per-lag divergence allocation in any form,
    because none exists -- the cross terms in $\lVert c_L \sum_\ell r_\ell \rVert^2$ can reinforce
    or cancel, and the bound applied after the summation does not restore additivity.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from teb_vae.lag_attn.eval.band_partition import KIND_ORDER0
from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_cfs.eval.analyses import source_null as source_null_analysis
from teb_vae.lag_attn_cfs.eval.analyses import spectral_skill as spectral_skill_analysis
from teb_vae.lag_attn_cfs.eval.analyses import warmup as warmup_analysis
from teb_vae.lag_attn_cfs.eval.binding import ModelBinding
from teb_vae.lag_slot_transformer_cfs.eval.analyses import arms as arms_analysis
from teb_vae.lag_slot_transformer_cfs.eval.analyses import attribution as attribution_analysis
from teb_vae.lag_slot_transformer_cfs.eval.analyses import band_clocks as band_clocks_analysis
from teb_vae.lag_slot_transformer_cfs.eval.analyses import (
    high_kl_anchors as high_kl_anchors_analysis,
)
from teb_vae.lag_slot_transformer_cfs.eval.analyses import (
    lag_suppression as lag_suppression_analysis,
)
from teb_vae.lag_slot_transformer_cfs.eval.analyses import (
    proposal_clocks as proposal_clocks_analysis,
)
from teb_vae.lag_slot_transformer_cfs.eval.analyses import (
    proposal_profile as proposal_profile_analysis,
)
from teb_vae.lag_slot_transformer_cfs.eval.analyses import (
    recording_traces as recording_traces_analysis,
)
from teb_vae.lag_slot_transformer_cfs.eval.analyses import (
    resolved_axes as resolved_axes_analysis,
)
from teb_vae.lag_slot_transformer_cfs.eval.analyses import samples as samples_analysis
from teb_vae.lag_slot_transformer_cfs.nets.model import MODEL_KIND, SeqVaeLagResidualTrfCfs
from teb_vae.lag_slot_transformer_cfs.task import SeqVaeLagResidualTrfCfsTask

#: The committed override delta, deep-merged **over** a checkpoint's own resolved configuration.
DEFAULT_OVERRIDES_PATH = Path(__file__).resolve().parent / "configs" / "eval_overrides.yaml"

#: Constructor keys reconciled against the checkpoint's own ``model_kwargs``.
#:
#: Fourteen are the lag-attentive cell's, less the five it reconciles that this architecture
#: refuses outright -- ``num_heads`` and ``d_head`` (nothing here is head-structured),
#: ``causal_norm`` (no recurrent encoder), ``prior_availability_input`` (the metadata clock is
#: unconditional) and ``lag_kv_source`` (there are no keys and values). A key naming a mechanism
#: the constructor rejects could never match, and would refuse every run.
#:
#: ``coverage_floor`` is added here and the lag-attentive cell does not carry it. It decides which
#: anchors are scored at all, which is the same argument that puts ``anchor_stride`` on the list:
#: a checkpoint and a config disagreeing about it would report two populations under one heading.
#:
#: Five are this architecture's own, and every one of them changes what a reported number *means*
#: rather than how well the model fits:
#:
#: * ``residual_mu_scale`` and ``residual_logsigma_scale`` bound the update in prior-scale units,
#:   so they bound the divergence and every margin measured against it.
#: * ``lag_scale`` is $c_L$, which sets the magnitude of the summed update before the limiter.
#: * ``mean_only_residual`` decides whether a scale channel exists at all, so it changes the module
#:   tree, the checkpoint's key set, and whether a variance update is a thing the run can report.
#: * ``source_scalar_lift`` decides whether the source encoder holds parameters, which is the
#:   recommended arm's central structural claim and also what makes the mask-only control a
#:   distinct intervention.
#:
#: Four more name the **arm**, and they are the reason a mechanism-separating comparison reads as
#: one afterwards. ``source_disabled`` says the gap is exactly zero by construction rather than by
#: measurement. ``source_values_withheld`` says the fusion read lag identity and availability and
#: no coefficient, so a gain is evidence about capacity rather than about the source.
#: ``source_stem`` and ``lag_fusion`` say which of the four constructions produced every number in
#: the file -- and ``lag_fusion`` in particular decides what a band-suppression margin *is*, since
#: removing a lag from an explicit sum and removing it from a normalised distribution are different
#: interventions. A checkpoint and a configuration disagreeing about any of the four would report
#: one arm's numbers under another arm's name.
#:
#: Two more name the **input policy**: ``zero_fhr_scattering_s0`` and ``zero_up_scattering_s0``
#: decide which stored coefficients the model was permitted to read, in training and at scoring
#: alike. A checkpoint fitted under one policy and an override asking for another would score a
#: model on an input it never learned from, or read its labels against a stream it was never
#: denied, so the two are reconciled by name like the arm leaves.
#:
#: **Deliberately absent, and each for its own reason.** ``lag_embed_dim`` and ``proposal_hidden``
#: are capacity: they change the fit and no readout's meaning, and the rebuild takes the
#: checkpoint's value anyway. ``anchor_chunk`` and ``lag_chunk`` change floating-point summation
#: order and nothing else, and an evaluation legitimately runs at a different tiling from the fit
#: that produced the checkpoint -- reconciling them would refuse a correct run. The four resolved
#: warm-up tuples are constructor parameters of no config, so listing them would compare a
#: checkpoint value against nothing and pass every run; their guard re-resolves the budget from the
#: configured shards instead. ``horizon_weight_halflife_steps`` and the objective weights re-weight
#: the *training* criterion, and this pass scores every block unweighted.
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
    "horizon_attention_blocks",
    "anchor_stride",
    "lag_floor",
    "persistence_residual",
    "coverage_floor",
    "residual_mu_scale",
    "residual_logsigma_scale",
    "lag_scale",
    "mean_only_residual",
    "source_scalar_lift",
    "source_disabled",
    "source_values_withheld",
    "source_stem",
    "lag_fusion",
    "zero_fhr_scattering_s0",
    "zero_up_scattering_s0",
)

#: Every analysis of the lag-attentive family this architecture cannot produce under the
#: family's name, each with the tensor it would have needed and the analysis this cell asks the
#: same question through instead.
#:
#: The reasons are held beside the names rather than in a comment, because they are written into
#: the run's ``summary.json``: a reader comparing this cell's output against a lag-attentive one
#: finds different column names, and the correspondence has to be a declaration rather than
#: something to infer from an absence.
#:
#: **Two of the six are removals and four are absences, and the distinction is load-bearing.**
#: ``attention`` and ``lag_kl`` are in the *shared* registry, so a binding of this pipeline gets
#: them by default and has to take them off -- that is what :data:`EXCLUDED_ANALYSES` does, and an
#: exclusion naming an analysis the registry does not hold is refused rather than ignored. The
#: other four are the lag-attentive cell's own extras, registered on *its* binding; this one never
#: registers them. ``source_null``, ``warmup`` and ``spectral_skill`` are that cell's extras too,
#: and are **registered here as the family's own implementations**, because the columns they read
#: are the same quantities on this cell: the zeroed-source arm's divergence, the warm-up tertile
#: gaps and the per-channel gap vector.
#:
#: **None of the six is handed a substitute under its own name.** Each analogue below is
#: registered under a name that says what it reads -- a proposal norm, a signed divergence drop, a
#: suppression margin -- because a proposal norm under an attention name is a per-lag attribution
#: that does not exist, and :mod:`~teb_vae.lag_slot_transformer_cfs.eval.verify` refuses the
#: family's names for those tensors anywhere in the output.
ANALYSES_THIS_ARCHITECTURE_CANNOT_PRODUCE: Dict[str, str] = {
    "attention": (
        "reads the attention distribution over lags. This architecture poses no query and "
        "computes no distribution over lags; its per-lag quantity is a signed proposal, which is "
        "an update rather than an allocation and must not be reported under this name. The "
        "pooled and per-cohort shape of the proposal-norm profile is under proposal_profile."
    ),
    "lag_kl": (
        "reads a per-lag allocation of the divergence. None exists here: the divergence is a "
        "function of the summed update, its cross terms can reinforce or cancel, and two "
        "proposals of equal magnitude and opposite sign contribute zero to it while each has a "
        "nonzero isolated value. The signed per-lag divergence drop, which is not an "
        "allocation, is under proposal_profile."
    ),
    "occlusion": (
        "rebuilds the full branch through the lag attention after zeroing a band of the source "
        "stream. The fusion is rebuilt here instead, from the cached proposals, and reported as a "
        "band suppression margin under lag_suppression, by horizon step under resolved_axes and "
        "on both clinical clocks under band_clocks."
    ),
    "lag_clocks": (
        "resolves the per-lag divergence allocation against the two clinical clocks, and that "
        "allocation does not exist here. The shape of the proposal-norm and divergence-drop "
        "profiles on both clocks is under proposal_clocks."
    ),
    "lag_kld_scaled": (
        "rescales the same per-lag allocation over a geometry-fixed partition of the lag axis. "
        "Each declared band's proposal-norm mass and signed divergence drop on both clocks are "
        "under proposal_clocks."
    ),
    "lag_high_kl": (
        "selects anchors by their pooled divergence and reads the lag structure of the selection "
        "off the per-anchor KL lag map, which this architecture does not write. The same "
        "selection read off the per-anchor proposal map is under high_kl_anchors."
    ),
}

#: The subset of the above that the **shared** registry actually holds, which is what the binding
#: field may name. Written out rather than computed as an intersection: an intersection would
#: silently shrink to nothing the day the shared registry is reorganised, and the refusal that
#: exists to catch a misspelt exclusion would never fire.
EXCLUDED_ANALYSES: Tuple[str, ...] = ("attention", "lag_kl")

#: The four that are absent rather than removed, kept separately so the summary can say which is
#: which. Nothing registers them here, so nothing has to remove them.
UNREGISTERED_ANALYSES: Tuple[str, ...] = tuple(
    name
    for name in ANALYSES_THIS_ARCHITECTURE_CANNOT_PRODUCE
    if name not in EXCLUDED_ANALYSES
)

#: Which analysis of this cell asks each absent family analysis's question, written into the
#: summary beside the exclusions so a reader of two run directories has the correspondence on
#: the page rather than in this module.
ANALOGUE_ANALYSES: Dict[str, Tuple[str, ...]] = {
    "attention": ("proposal_profile",),
    "lag_kl": ("proposal_profile",),
    "occlusion": ("lag_suppression", "resolved_axes", "band_clocks"),
    "lag_clocks": ("proposal_clocks",),
    "lag_kld_scaled": ("proposal_clocks",),
    "lag_high_kl": ("high_kl_anchors",),
}


def effective_input_disclosure(model: Any) -> Dict[str, Any]:
    r"""Which stored coefficients the model was permitted to read, for the causality record.

    An input ablation changes what every number in a run means -- a forecast made without the
    anchor's own smoothed level is a different task from one made with it -- and it leaves no
    trace in any tensor shape, so it has to be stated. Each ablated coordinate is named by
    stream, stored field, declared channel index and the partition's own coefficient kind, so a
    reader can find it in the channel map without knowing the storage layout.

    Args:
        model: The rebuilt net.

    Returns:
        The two switches and the ablated coordinates, with a sentence on what the ablation is
        and is not.
    """
    record = dict(model.input_ablation_record())
    record["ablated_inputs"] = [
        {**entry, "kind": KIND_ORDER0} for entry in record["ablated_inputs"]
    ]
    record["policy"] = (
        "a VALUE ablation applied after loader normalisation and before every model path, the "
        "persistence shortcut included: each ablated coordinate is replaced by exactly zero, "
        "which is the channel's training mean on the standardised scale, while its availability "
        "announcement, the forecast labels and the stored batch are untouched. It is applied "
        "identically in training and at scoring, so the trivial baselines and the attributions "
        "read the same permitted view."
        if record["ablated_inputs"]
        else "no input ablation: every stored coefficient the warm-up budget kept is read."
    )
    return record


def residual_encoder_disclosure(model: Any) -> Dict[str, Any]:
    r"""What this architecture's encoders have to disclose about their causal standing.

    Three numbers and a sentence, and the first two are the ones a reader of a lag readout needs
    side by side. The **additional neural source receptive field is one stored sample**: a proposal
    reads the encoding at $t - \ell$ and no other stored time, so nothing in the source pathway
    mixes across the lag axis before the fusion. The **searched window** is far wider than that,
    and the gap between the two is exactly the resolution the readout has. The window's oldest
    centre is stated in seconds beside its length in steps, because a bank of $L$ entries that
    includes the anchor reaches $(L - 1)$ steps back, and "$L$ steps" is read as either.

    The **effective inputs** are disclosed on every arm, the target-only one included: an input
    ablation is a property of the task the model was fitted to, not of its source pathway.

    The sentence is the qualification neither number carries. One stored sample is the *neural*
    receptive field; the causal feature extraction upstream of the model still mixes raw history
    within each coefficient, over a span the feature geometry sets and no model-side change
    reaches. A disclosure reporting only the first number would read as a claim about the signal.

    On the **target-only arm** there is no source pathway at all, so the receptive field and the
    searched window are both absent rather than zero: nothing reads a source, so there is no reach
    to disclose. The record says which arm it describes, because a reader comparing two runs has to
    know whether an empty lag readout means "measured nothing" or "there was nothing to measure".

    Every value is read off the model rather than written as a literal, so a run at another
    geometry discloses its own.

    Args:
        model: The rebuilt net.

    Returns:
        The encoder half of the run's causality record.

    Raises:
        AttributeError: If the model carries none of these attributes, which means the binding is
            pointed at another architecture and every number in the run would be that model's.
    """
    if getattr(model, "source_disabled", False):
        return {
            "source_disabled": True,
            "source_stem": None,
            "source_receptive_field_steps": None,
            "searched_lag_steps": None,
            "furthest_searched_lag": None,
            "oldest_lag_seconds": None,
            "source_encoder_parameters": 0,
            "source_values_withheld": False,
            "lag_fusion": None,
            "effective_inputs": effective_input_disclosure(model),
            "qualification": (
                "This checkpoint has no source pathway: no pointwise encoder, no lag embeddings "
                "and no proposal head were built. Its full distribution is the prior, its "
                "divergence is exactly zero, and it is scored here as the target-only reference "
                "the source-conditioned arms are read against."
            ),
        }
    source_parameters = sum(
        parameter.numel() for parameter in model.source_encoder.parameters()
    )
    # The stem reports its own reach; the pointwise representation's is one stored sample by
    # construction. Read off the module rather than written down, so a comparator run at another
    # convolution schedule discloses the reach it actually has.
    receptive_field = int(getattr(model.source_encoder, "receptive_field", 1))
    fusion = str(getattr(model, "lag_fusion", "local"))
    return {
        "source_disabled": False,
        "source_stem": str(getattr(model, "source_stem", "pointwise")),
        "source_receptive_field_steps": receptive_field,
        "searched_lag_steps": int(model.n_lags),
        "furthest_searched_lag": int(model.n_lags) - 1,
        # The centre of the oldest gathered coefficient, in stored-coefficient seconds before
        # the anchor: the bank includes the anchor's own step, so it reaches one step less than
        # its length. Stated so a bank length in steps is not read as a depth in seconds.
        "oldest_lag_seconds": float(int(model.n_lags) - 1) * float(SECONDS_PER_STEP),
        "source_encoder_parameters": int(source_parameters),
        "source_values_withheld": bool(getattr(model, "source_values_withheld", False)),
        "lag_fusion": (
            "explicit sum over lags, bounded after summation"
            if fusion == "local"
            else "learned distribution over lags, read out through one projection and bounded"
        ),
        "effective_inputs": effective_input_disclosure(model),
        "qualification": (
            f"The additional NEURAL source receptive field is {receptive_field} stored "
            f"step(s): each lag's vector summarises that many stored source steps and no more. "
            f"The causal feature extraction upstream of the model still mixes raw history inside "
            f"every coefficient, over a span the feature geometry fixes and no model-side change "
            f"reaches, so this is not a claim that a readout resolves one stored step of the "
            f"signal. Two lags closer together than this reach are summaries of overlapping "
            f"windows, which is the resolution floor of any lag readout taken over this "
            f"representation."
        ),
    }


def collect_tables(task: Any, loader: Any, **kwargs: Any) -> Any:
    """This cell's own collection pass, as the family's runner calls it through the binding.

    Resolved at call time rather than imported above: the pass reads this module's declarations
    -- the excluded analyses and the model kind -- so importing it here would be a cycle. What
    the runner is handed is one callable with the shared pass's signature, and everything from
    the tables on is the family's.

    Args:
        task: The loaded task, in evaluation mode.
        loader: The evaluation dataloader.
        **kwargs: Exactly :func:`teb_vae.lag_attn_cfs.eval.collect.collect_tables`'s keywords.

    Returns:
        The collection this cell's pass assembled.
    """
    from teb_vae.lag_slot_transformer_cfs.eval.collect import collect_tables as run_pass

    return run_pass(task, loader, **kwargs)


#: This cell's own analyses, merged onto the shared registry in declaration order and run after
#: every shared analysis but the trailing cross-subgroup test, which reads what they write.
#:
#: Three draw the readouts only this architecture has, from the results block the pass assembled:
#: the scored arms and their paired margins, the lag suppression with the per-lag profile, and the
#: horizon- and block-resolved margins. Four read the lag structure off the two sidecars the pass
#: writes -- the pooled and per-cohort profile shape, the profiles and band masses on both
#: clinical clocks, the band and control margins on both clocks, and the anchors selected by
#: their own divergence. Three are the family's own implementations, registered here because the
#: columns they read are the same quantities on this cell: the warm-up tertile gaps and the
#: geometry guards (``warmup``), the zeroed-source arm's divergence against the matched one
#: (``source_null``) and the per-channel gap vector on the kept frequency bands
#: (``spectral_skill``). Three draw this model's own forward under the family's three names --
#: the pages, the traces and the attributions -- because each reads a tensor the lag-attentive
#: forward does not emit and the lag-attentive implementation reads one this forward does not;
#: the shared name is what keeps two run directories readable down one layout.
EXTRA_ANALYSES: Dict[str, Any] = {
    "arms": arms_analysis.run_arms_analysis,
    "lag_suppression": lag_suppression_analysis.run_lag_suppression_analysis,
    "resolved_axes": resolved_axes_analysis.run_resolved_axes_analysis,
    "proposal_profile": proposal_profile_analysis.run_proposal_profile_analysis,
    "proposal_clocks": proposal_clocks_analysis.run_proposal_clocks_analysis,
    "band_clocks": band_clocks_analysis.run_band_clocks_analysis,
    "high_kl_anchors": high_kl_anchors_analysis.run_high_kl_anchors_analysis,
    "warmup": warmup_analysis.run_warmup_analysis,
    "source_null": source_null_analysis.run_source_null_analysis,
    "spectral_skill": spectral_skill_analysis.run_spectral_skill_analysis,
    "samples": samples_analysis.run_samples_analysis,
    "recording_traces": recording_traces_analysis.run_recording_traces_analysis,
    "attribution": attribution_analysis.run_attribution_analysis,
}

#: What this cell's own pass puts in the headline block, appended to the family's registry.
#:
#: The family's ``pred_gap_mc_nats`` is this cell's marginalised gap already -- the same number
#: under the family's name -- so what is added is its paired interval over recordings and the four
#: source-control margins, which are the readouts an arm table of this cell is read down. Every
#: path is keyed all the way down, and none is keyed by a band name an operator chooses.
HEADLINE_SCALARS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("pred_gap_mc_ci_lo", ("arm_scores", "pred_gap", "lo")),
    ("pred_gap_mc_ci_hi", ("arm_scores", "pred_gap", "hi")),
    ("silence_margin_nats", ("source_controls", "silence_margin_nats")),
    ("replace_zeros_margin_nats", ("source_controls", "replace_zeros_margin_nats")),
    ("replace_constant_margin_nats", ("source_controls", "replace_constant_margin_nats")),
    ("permute_margin_nats", ("source_controls", "permute_margin_nats")),
    ("cancellation_ratio_mean", ("lag_readouts", "cancellation", "mean", "ratio")),
    ("draw_concentration_full", ("arm_scores", "draw_concentration_full", "point")),
    # The family's own scalars of the three family analyses registered here, under the
    # family's names, so an arm table of this cell and of a lag-attentive cell share the
    # columns for the quantities that are the same on both.
    ("kld_source_null_nats", ("source_null", "difference", "kld_source_null_nats")),
    ("coupling_minus_clock_nats", ("source_null", "difference", "coupling_minus_clock_nats")),
    ("coupling_minus_clock_ci_lo", ("source_null", "difference", "ci_lo")),
    ("coupling_minus_clock_ci_hi", ("source_null", "difference", "ci_hi")),
    ("pred_gap_warm_lo_nats", ("warmup", "headline", "pred_gap_warm_lo_nats")),
    ("pred_gap_warm_mid_nats", ("warmup", "headline", "pred_gap_warm_mid_nats")),
    ("pred_gap_warm_hi_nats", ("warmup", "headline", "pred_gap_warm_hi_nats")),
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
    # This cell's selection-by-divergence readout: the threshold every other number of that
    # analysis is conditional on, and the paired usefulness difference.
    ("high_kl_threshold_nats", ("high_kl_anchors", "thresholds", "high_nats")),
    (
        "high_minus_rest_gain_nats",
        ("high_kl_anchors", "usefulness", "high_minus_rest_mean_interval", "point"),
    ),
    (
        "high_in_gain_share",
        ("high_kl_anchors", "usefulness", "overlap", "share_of_high_in_gain"),
    ),
)

#: The model this package's evaluation scores.
#:
#: Everything the family's runner cannot derive about this model, in one frozen declaration: the
#: class to rebuild, the keys to reconcile, the disclosure to record, the override delta to merge,
#: the pass that produces the tables, the analyses only this cell has, the scalars they put in the
#: headline, and the two shared analyses this architecture cannot produce. ``excluded_analyses``
#: is a checkable statement rather than prose: the merged registry with this binding applied is
#: what a reader is told the architecture cannot produce.
LAG_RESIDUAL_BINDING = ModelBinding(
    model_cls=SeqVaeLagResidualTrfCfs,
    task_cls=SeqVaeLagResidualTrfCfsTask,
    tag="lag_slot_transformer_cfs",
    geometry_keys=GEOMETRY_KEYS,
    encoder_disclosure=residual_encoder_disclosure,
    overrides_path=DEFAULT_OVERRIDES_PATH,
    extra_analyses=EXTRA_ANALYSES,
    headline_scalars=HEADLINE_SCALARS,
    excluded_analyses=EXCLUDED_ANALYSES,
    collect=collect_tables,
)

__all__ = [
    "ANALOGUE_ANALYSES",
    "ANALYSES_THIS_ARCHITECTURE_CANNOT_PRODUCE",
    "DEFAULT_OVERRIDES_PATH",
    "EXCLUDED_ANALYSES",
    "EXTRA_ANALYSES",
    "GEOMETRY_KEYS",
    "HEADLINE_SCALARS",
    "LAG_RESIDUAL_BINDING",
    "MODEL_KIND",
    "UNREGISTERED_ANALYSES",
    "collect_tables",
    "effective_input_disclosure",
    "residual_encoder_disclosure",
]
