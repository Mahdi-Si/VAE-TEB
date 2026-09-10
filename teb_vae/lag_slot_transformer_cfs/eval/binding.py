r"""What this pipeline cannot derive about the model it scores, declared once.

The :class:`~teb_vae.lag_attn_cfs.eval.binding.ModelBinding` type is the shared one, bound rather
than forked: a second copy of it would be a second definition of what a run *means*, and the two
would drift. What is written here is one instance of it, and each field decides something no
amount of reading the checkpoint could recover.

Three of the fields differ from the lag-attentive cells' in ways worth stating outright.

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

from teb_vae.lag_attn_cfs.eval.binding import ModelBinding
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
)

#: Every analysis this architecture cannot produce, each with the tensor it would have needed.
#:
#: The reasons are held beside the names rather than in a comment, because they are written into
#: the run's ``summary.json``: a reader comparing this cell's output against a lag-attentive one
#: finds fewer columns, and the difference has to be a declaration rather than something to infer
#: from an absence.
#:
#: **Two of the seven are removals and five are absences, and the distinction is load-bearing.**
#: ``attention`` and ``lag_kl`` are in the *shared* registry, so a binding of this pipeline gets
#: them by default and has to take them off -- that is what :data:`EXCLUDED_ANALYSES` does, and an
#: exclusion naming an analysis the registry does not hold is refused rather than ignored. The
#: other five are the lag-attentive cell's own extras, registered on *its* binding; this one
#: registers no extras, so they were never in this registry to remove. Recording all seven together
#: is the honest statement: a reader comparing two runs sees seven columns missing, and the reason
#: is the same architectural fact in every case whichever mechanism left them out.
ANALYSES_THIS_ARCHITECTURE_CANNOT_PRODUCE: Dict[str, str] = {
    "attention": (
        "reads the attention distribution over lags. This architecture poses no query and "
        "computes no distribution over lags; its per-lag quantity is a signed proposal, which is "
        "an update rather than an allocation and must not be reported under this name."
    ),
    "lag_kl": (
        "reads a per-lag allocation of the divergence. None exists here: the divergence is a "
        "function of the summed update, its cross terms can reinforce or cancel, and two "
        "proposals of equal magnitude and opposite sign contribute zero to it while each has a "
        "nonzero isolated value."
    ),
    "source_null": (
        "encodes a stream of exact zeros through the source pathway. The recommended arm's source "
        "encoder holds no parameters, so there is nothing to encode; the analogous control here "
        "substitutes the source values and re-runs the fusion, and is reported as a source "
        "specificity margin instead."
    ),
    "occlusion": (
        "rebuilds the full branch through the lag attention after zeroing a band of the source "
        "stream. The fusion is rebuilt here instead, from the cached proposals, and reported as a "
        "band suppression margin."
    ),
    "lag_clocks": (
        "resolves the per-lag divergence allocation against the two clinical clocks, and that "
        "allocation does not exist here."
    ),
    "lag_kld_scaled": (
        "rescales the same per-lag allocation over a geometry-fixed partition of the lag axis."
    ),
    "lag_high_kl": (
        "selects anchors by their pooled divergence and reads the lag structure of the selection "
        "off the per-anchor lag map, which this architecture does not write."
    ),
}

#: The subset of the above that the **shared** registry actually holds, which is what the binding
#: field may name. Written out rather than computed as an intersection: an intersection would
#: silently shrink to nothing the day the shared registry is reorganised, and the refusal that
#: exists to catch a misspelt exclusion would never fire.
EXCLUDED_ANALYSES: Tuple[str, ...] = ("attention", "lag_kl")

#: The five that are absent rather than removed, kept separately so the summary can say which is
#: which. Nothing registers them here, so nothing has to remove them.
UNREGISTERED_ANALYSES: Tuple[str, ...] = tuple(
    name
    for name in ANALYSES_THIS_ARCHITECTURE_CANNOT_PRODUCE
    if name not in EXCLUDED_ANALYSES
)


def residual_encoder_disclosure(model: Any) -> Dict[str, Any]:
    r"""What this architecture's encoders have to disclose about their causal standing.

    Three numbers and a sentence, and the first two are the ones a reader of a lag readout needs
    side by side. The **additional neural source receptive field is one stored sample**: a proposal
    reads the encoding at $t - \ell$ and no other stored time, so nothing in the source pathway
    mixes across the lag axis before the fusion. The **searched window** is far wider than that,
    and the gap between the two is exactly the resolution the readout has.

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
            "source_encoder_parameters": 0,
            "source_values_withheld": False,
            "lag_fusion": None,
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
        "source_encoder_parameters": int(source_parameters),
        "source_values_withheld": bool(getattr(model, "source_values_withheld", False)),
        "lag_fusion": (
            "explicit sum over lags, bounded after summation"
            if fusion == "local"
            else "learned distribution over lags, read out through one projection and bounded"
        ),
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


#: The model this package's evaluation scores.
#:
#: ``extra_analyses`` is empty and ``headline_scalars`` with it, and that is not a gap to be filled
#: later: this package does not drive the shared collection pass at all, for the anchor-axis reason
#: its ``__init__`` records, so an analysis registered here would never be reached. The binding is
#: declared as a ``ModelBinding`` regardless, because the four facts it carries are exactly the
#: four this package's own pass needs -- the class to rebuild, the keys to reconcile, the
#: disclosure to record, and the override delta to merge -- and because ``excluded_analyses`` is
#: then a checkable statement rather than prose: the merged registry with this binding applied is
#: what a reader is told the architecture cannot produce.
LAG_RESIDUAL_BINDING = ModelBinding(
    model_cls=SeqVaeLagResidualTrfCfs,
    task_cls=SeqVaeLagResidualTrfCfsTask,
    tag="lag_slot_transformer_cfs",
    geometry_keys=GEOMETRY_KEYS,
    encoder_disclosure=residual_encoder_disclosure,
    overrides_path=DEFAULT_OVERRIDES_PATH,
    excluded_analyses=EXCLUDED_ANALYSES,
)

__all__ = [
    "ANALYSES_THIS_ARCHITECTURE_CANNOT_PRODUCE",
    "DEFAULT_OVERRIDES_PATH",
    "EXCLUDED_ANALYSES",
    "GEOMETRY_KEYS",
    "LAG_RESIDUAL_BINDING",
    "MODEL_KIND",
    "UNREGISTERED_ANALYSES",
    "residual_encoder_disclosure",
]
