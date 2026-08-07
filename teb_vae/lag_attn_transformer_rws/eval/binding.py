r"""What the shared evaluation pipeline cannot derive about this model.

Carried by one frozen :class:`~teb_vae.lag_attn_rws.eval.binding.ModelBinding`: the classes to
rebuild from a checkpoint, the constructor keys reconciled against it, this encoder's own causality
disclosure, this package's committed override delta, and the one analysis only this architecture
can have together with the headline scalars it registers. Everything else the pipeline needs it
computes from the tables, the config and the shared ``nets`` modules, all of which this model
imports unchanged.

**This module is the only place either addition is named.** ``encoder_attention`` appears in the
help text, in the selection and in ``summary.json`` by being registered here once, and the registry
parity test reads ``TRF_BINDING.extra_analyses`` rather than a list -- so registering it is what
makes that test pass and forgetting to is what makes it fail.

The disclosure is where the two models genuinely diverge and it is written that way. The sibling
records ``causal_norm`` and ``n_causalized_norms``, which describe a time-pooling ``GroupNorm``
this architecture does not have: the encoders here ban time-axis normalisers structurally, so the
key would always read the same value and would say nothing. What is true *here* -- the block
counts, the depthwise re-initialisation, the source window and the receptive field it produces --
is what this model discloses instead. A shared key that means nothing in one of them is worse
than two honest blocks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from teb_vae.lag_attn_rws.eval.binding import ModelBinding
from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_transformer_rws.eval.analyses import encoder_attention
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws
from teb_vae.lag_attn_transformer_rws.task import SeqVaeLagAttnTrfRwsTask

#: The committed override delta, merged over a checkpoint's own resolved config.
DEFAULT_OVERRIDES_PATH = Path(__file__).resolve().parent / "configs" / "eval_overrides.yaml"

#: Constructor keys reconciled against the checkpoint's own ``model_kwargs``.
#:
#: The sibling's thirteen, minus ``causal_norm`` and plus this architecture's seven. ``causal_norm``
#: is not merely irrelevant here: the constructor does not accept it, so a config carrying it is
#: already a ``TypeError`` at rebuild and reconciling it could only ever compare a key against
#: nothing.
#:
#: The seven are not documentation. Each one changes what the numbers mean: the stem schedule and
#: the block counts set how much history a state summarises, the head count and the feed-forward
#: width set the capacity behind it, and ``source_attention_window`` sets the source encoder's
#: reach -- which is the axis the whole locality sweep varies and the thing the lag attention's
#: ability to separate adjacent delays depends on.
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
    "encoder_conv_kernels",
    "encoder_conv_dilations",
    "encoder_num_heads",
    "encoder_d_ff",
    "target_attention_blocks",
    "source_attention_blocks",
    "source_attention_window",
)


def _attribute(model: Any, name: str) -> Any:
    """Read a public model attribute, or raise naming it.

    A ``getattr(model, name, default)`` would report a model that stopped exposing something as a
    model with nothing to report, and the disclosure would go quiet in exactly the case a reader
    most needs to be told.

    Args:
        model: The rebuilt net.
        name: The attribute to read.

    Returns:
        The attribute's value.

    Raises:
        AttributeError: Naming the attribute and the class that does not carry it.
    """
    if not hasattr(model, name):
        raise AttributeError(
            f"{type(model).__name__} carries no {name!r}, which the causality disclosure reports. "
            f"An evaluation cannot describe an encoder it cannot read; if the attribute was "
            f"renamed, this disclosure is what has to follow it."
        )
    return getattr(model, name)


def trf_encoder_disclosure(model: Any) -> Dict[str, Any]:
    r"""Return the encoder-specific half of the causality record for the conv-Transformer encoders.

    Reports what is structurally true of *these* encoders, in the record where the sibling reports
    ``causal_norm``:

    ``time_pooling_normalisers``
        Zero, and zero by construction rather than by configuration. Every normaliser on a history
        path is per-token -- ``RMSNorm`` inside the blocks, ``LayerNorm`` in the input adapter --
        so no statistic is pooled over the time axis and no history state can carry an image of
        its own future through one. There is no switch that turns this off, which is why the count
        is reported beside the test that enumerates the surviving modules rather than beside a
        config key.

    ``n_depthwise_init``
        How many depthwise convolutions the variance-preserving pass re-initialised. Recorded
        because it is the only evidence that pass was not a silent no-op: the generic
        initialisation reads a depthwise ``(C, 1, k)`` weight's fan wrongly and starts the stem an
        order of magnitude too quiet. A stem-free arm legitimately reports $0$.

    ``source_receptive_field_steps``
        The structural bound $R_U = R_{\mathrm{conv}} + N_U (W_U - 1)$, capped at $T$, or ``None``
        for the unbounded arm -- absent rather than $T$, because "no bound" and "a bound that
        happens to equal the sequence length" are different statements. Reported in steps and in
        seconds beside the lag search range in the same units, with which is larger stated: an
        encoder whose reach exceeded the lag range would already be doing the alignment the lag
        cross-attention exists to do, so the comparison is the point rather than the two numbers.

    Args:
        model: The rebuilt net, which is what was trained and is therefore the authority on the
            geometry it actually carries.

    Returns:
        This encoder's own keys, merged into the causality record in this order.

    Raises:
        AttributeError: If the model does not expose one of the attributes read, naming it.
    """
    target_encoder = _attribute(model, "target_encoder")
    source_encoder = _attribute(model, "source_encoder")
    window = _attribute(source_encoder, "attention_window")
    reach_steps = _attribute(source_encoder, "receptive_field")
    # The lag axis runs over indices 0..max_lag, so the *furthest* lag searched is max_lag steps
    # and there are max_lag + 1 of them. The reach is compared against the furthest lag, which is
    # the distance question; the count is recorded beside it because it is what an L-shaped array
    # in any table has on its lag axis.
    max_lag_steps = int(_attribute(model, "max_lag"))
    n_lags = max_lag_steps + 1

    record: Dict[str, Any] = {
        # Not "causal_norm: true" under another name. The sibling's key reports a *setting*; this
        # reports that the setting has no counterpart here because the modules it would govern do
        # not exist on a history path.
        "time_pooling_normalisers": 0,
        "time_pooling_normalisers_are_structural": True,
        "time_pooling_normalisers_proved_by": (
            "teb_vae/lag_attn_transformer_rws/tests/test_construct.py::"
            "test_no_time_pooling_normaliser_on_either_history_path"
        ),
        "n_depthwise_init": int(_attribute(model, "n_depthwise_init")),
        "target_attention_blocks": int(_attribute(target_encoder, "num_attention_blocks")),
        "source_attention_blocks": int(_attribute(source_encoder, "num_attention_blocks")),
        "source_attention_window": None if window is None else int(window),
        "source_receptive_field_steps": None if reach_steps is None else int(reach_steps),
        "source_receptive_field_seconds": (
            None if reach_steps is None else int(reach_steps) * SECONDS_PER_STEP
        ),
        # The lag search the reach above is meant to stay inside.
        "lag_range_max_steps": max_lag_steps,
        "lag_range_max_seconds": max_lag_steps * SECONDS_PER_STEP,
        "n_lags": n_lags,
    }
    if reach_steps is None:
        # The unbounded arm: there is no bound to compare, and reporting T as one would invent a
        # number the architecture does not claim.
        record["source_reach_vs_lag_range"] = (
            "the source encoder has no window, so its reach is the full causal prefix and is not "
            "bounded below the lag range; a source state is a whole-history summary rather than a "
            "local neighbourhood one, and lag attribution should be read with that in mind"
        )
    else:
        # Three-way, because the two are configured independently and a sweep arm can land them on
        # the same number: "the source reach is larger" printed beside two identical figures is a
        # sentence a reader would have to disbelieve to read the record correctly.
        if max_lag_steps > int(reach_steps):
            comparison = "the lag range is larger"
        elif max_lag_steps < int(reach_steps):
            comparison = "the source reach is larger"
        else:
            comparison = "they are equal"
        record["source_reach_vs_lag_range"] = (
            f"source reach {int(reach_steps)} steps "
            f"({int(reach_steps) * SECONDS_PER_STEP:g} s) against a furthest searched lag of "
            f"{max_lag_steps} steps ({max_lag_steps * SECONDS_PER_STEP:g} s); {comparison}"
        )
        record["source_reach_is_inside_the_lag_range"] = int(reach_steps) < max_lag_steps
    return record


#: This model's own analysis, and the only one: ``encoder_attention`` profiles the mechanism that
#: *is* the encoder replacement, and there is nothing for it to answer in a model whose history
#: encoders are recurrent. Registered here rather than on the shared registry, which is the shared
#: models' contract -- an entry there would reach the sibling too, where it could only ever record
#: a permanent skip. Merged after the shared registry, so the sibling's run order is untouched.
EXTRA_ANALYSES: Dict[str, Any] = {
    "encoder_attention": encoder_attention.run_encoder_attention_analysis,
}

#: What that analysis puts in the headline block, which is the only block an arm table reads. Six
#: scalars: the two per-stream entropy ratios, and the measured source reach's median and 95th
#: percentile in steps and in seconds. The reach pair is what gives the ``sweep_window_*`` family a
#: measured x-axis instead of a configured one -- a number that stayed in a CSV could not.
#:
#: No verdict is registered beside them. This analysis describes a mechanism rather than
#: adjudicating a difference, and there is no threshold here anyone has earned the right to set.
HEADLINE_SCALARS: Tuple[Tuple[str, Tuple[str, ...]], ...] = tuple(
    (f"encoder_attention_{name}", ("encoder_attention", "headline", name))
    for name in encoder_attention.HEADLINE_KEYS
)

#: The model this package evaluates.
TRF_BINDING = ModelBinding(
    model_cls=SeqVaeLagAttnTrfRws,
    task_cls=SeqVaeLagAttnTrfRwsTask,
    tag="lag_attn_trf_rws",
    geometry_keys=GEOMETRY_KEYS,
    encoder_disclosure=trf_encoder_disclosure,
    overrides_path=DEFAULT_OVERRIDES_PATH,
    extra_analyses=EXTRA_ANALYSES,
    headline_scalars=HEADLINE_SCALARS,
)
