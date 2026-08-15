r"""What the causal-feature evaluation pipeline cannot derive about this model.

Carried by one frozen :class:`~teb_vae.lag_attn_cfs.eval.binding.ModelBinding`: the classes to
rebuild from a checkpoint, the constructor keys reconciled against it, this encoder's own causality
disclosure, and this package's committed override delta. Everything else the pipeline needs it
computes from the tables, the config and the shared ``nets`` modules, all of which this model imports
unchanged.

**The disclosure is where the two cfs cells genuinely diverge, and it is written that way.** The
conv-LSTM cell records ``causal_norm`` and ``n_causalized_norms``, which describe a time-pooling
``GroupNorm`` this architecture does not have -- ``causal_norm`` is not even a keyword of this
constructor, so the key could only ever report a value nothing set. What is true *here* is that the
encoders ban time-axis normalisers **structurally**: every normaliser on a history path is per-token,
so no statistic is pooled over time and no history state can carry an image of its own future through
one. That, the block counts, the source window and the receptive field it produces are what this
model discloses instead.

**Everything about the target domain stays in the shared half**, and that is deliberate: the
one-sidedness of the coefficients, the per-block group delays, the resolved warm-up budget, the
anchor geometry and the measured lag support are properties of the data and of the target-domain
mixins both cells compose, not of either encoder. A reader comparing the two models'
``preflight.json`` files compares them down one set of key names, and
:data:`~teb_vae.lag_attn_cfs.eval.preflight.SHARED_CAUSALITY_KEYS` refuses a disclosure that tried
to restate one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_cfs.eval.binding import ModelBinding
from teb_vae.lag_attn_cfs.eval.binding import EXTRA_ANALYSES, HEADLINE_SCALARS
from teb_vae.lag_attn_cfs.eval.preflight import disclosed_attribute
from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs
from teb_vae.lag_attn_transformer_cfs.task import SeqVaeLagAttnTrfCfsTask

#: The committed override delta, merged over a checkpoint's own resolved config.
DEFAULT_OVERRIDES_PATH = Path(__file__).resolve().parent / "configs" / "eval_overrides.yaml"

#: The name of the test that enumerates every normaliser on a history path, quoted in the disclosure.
#: A structural claim is only as good as the check behind it, so the record names that check rather
#: than asserting the property on its own authority.
TIME_POOLING_PROOF = (
    "teb_vae/lag_attn_transformer_cfs/tests/test_causality.py::"
    "test_no_history_path_carries_a_time_pooling_normaliser"
)

#: Constructor keys reconciled against the checkpoint's own ``model_kwargs``.
#:
#: **The rule a key must satisfy is narrower than "it is a constructor parameter", and the reason is
#: that ``preflight.reconcile`` silently skips a key absent from either side**
#: (``if key not in vae_config or key not in model_kwargs: continue``). So a key must be a constructor
#: parameter *and* a ``model_config.VAE_model`` key to be checked at all, and listing one that is not
#: is a silent no-op rather than a refusal. ``tests/test_eval_binding.py`` asserts both halves
#: against the class and against this package's shipped ``configs/default.yaml``.
#:
#: The cfs cell's sixteen **minus** ``causal_norm`` **plus** this architecture's seven, giving
#: twenty-two. ``causal_norm`` is not merely irrelevant here: the constructor does not accept it, so
#: a config carrying it is already a ``TypeError`` at rebuild and reconciling it could only ever
#: compare a key against nothing.
#:
#: The seven are not documentation. Each one changes what the numbers mean: the stem schedule and the
#: block counts set how much history a state summarises, the head count and the feed-forward width
#: set the capacity behind it, and ``source_attention_window`` sets the source encoder's reach --
#: which is the thing the lag attention's ability to separate adjacent delays depends on.
#:
#: ``causal_warmup_budget_steps`` and the four resolved warm-up tuples are absent for the cfs cell's
#: reason and are checked by the same guard: the budget is a config key that names no constructor
#: parameter, and the tuples are constructor parameters that name no config key, so
#: ``preflight.check_warmup_budget_matches_checkpoint`` re-resolves the budget against the configured
#: shards instead -- the only comparison that can actually fail.
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
    "anchor_stride",
    "lag_floor",
    "encoder_conv_kernels",
    "encoder_conv_dilations",
    "encoder_num_heads",
    "encoder_d_ff",
    "target_attention_blocks",
    "source_attention_blocks",
    "source_attention_window",
)


def trf_cfs_encoder_disclosure(model: Any) -> Dict[str, Any]:
    r"""Return the encoder-specific half of the causality record for the conv-Transformer encoders.

    Reports what is structurally true of *these* encoders, in the record where the conv-LSTM cell
    reports ``causal_norm``:

    ``time_pooling_normalisers``
        Zero, and zero by construction rather than by configuration. Every normaliser on a history
        path is per-token -- ``RMSNorm`` inside the blocks, ``LayerNorm`` in the input adapter -- so
        no statistic is pooled over the time axis and no history state can carry an image of its own
        future through one. There is no switch that turns this off, which is why the count is
        reported beside :data:`TIME_POOLING_PROOF` rather than beside a config key.

        On this cell that claim carries further than it does on the raw one. The stored coefficients
        are already one-sided, so the *data* contains no future; an encoder that pooled over time
        would put one back, and the forecast claim in the shared statement would hold of the dataset
        and not of the model. Here it holds of both.

    ``n_depthwise_init``
        How many depthwise convolutions the variance-preserving pass re-initialised. Recorded because
        it is the only evidence that pass was not a silent no-op: the generic initialisation reads a
        depthwise $(C, 1, k)$ weight's fan wrongly and starts the stem an order of magnitude too
        quiet. A stem-free arm legitimately reports $0$.

    ``source_receptive_field_steps``
        The structural bound $R_U = R_{\mathrm{conv}} + N_U (W_U - 1)$, capped at $T$, or ``None`` for
        the unbounded arm -- absent rather than $T$, because "no bound" and "a bound that happens to
        equal the sequence length" are different statements. Compared against the furthest searched
        lag, which the shared record's ``lag_support`` block carries in full: an encoder whose reach
        exceeded the lag range would already be doing the alignment the lag cross-attention exists to
        do, so the comparison is the point rather than the two numbers.

    Args:
        model: The rebuilt net, which is what was trained and is therefore the authority on the
            geometry it actually carries.

    Returns:
        This encoder's own keys, merged into the causality record in this order.

    Raises:
        AttributeError: If the model does not expose one of the attributes read, naming it and the
            class -- through the same reader the conv-LSTM cell's disclosure uses, so the two cannot
            come to word the refusal differently.
    """
    target_encoder = disclosed_attribute(model, "target_encoder")
    source_encoder = disclosed_attribute(model, "source_encoder")
    window = disclosed_attribute(source_encoder, "attention_window")
    reach_steps = disclosed_attribute(source_encoder, "receptive_field")
    # The furthest lag searched, which is what a reach is compared against. The lag *count* and the
    # floor are not restated here: the shared record's ``lag_support`` block owns them, and a second
    # copy is a second thing to keep true.
    max_lag_steps = int(disclosed_attribute(model, "max_lag"))

    record: Dict[str, Any] = {
        # Not "causal_norm: true" under another name. The cfs cell's key reports a *setting*; this
        # reports that the setting has no counterpart here because the modules it would govern do not
        # exist on a history path.
        "time_pooling_normalisers": 0,
        "time_pooling_normalisers_are_structural": True,
        "time_pooling_normalisers_proved_by": TIME_POOLING_PROOF,
        "n_depthwise_init": int(disclosed_attribute(model, "n_depthwise_init")),
        "target_attention_blocks": int(
            disclosed_attribute(target_encoder, "num_attention_blocks")
        ),
        "source_attention_blocks": int(
            disclosed_attribute(source_encoder, "num_attention_blocks")
        ),
        "source_attention_window": None if window is None else int(window),
        "source_receptive_field_steps": None if reach_steps is None else int(reach_steps),
        "source_receptive_field_seconds": (
            None if reach_steps is None else int(reach_steps) * SECONDS_PER_STEP
        ),
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


#: The model this package evaluates.
#:
#: ``extra_analyses`` and ``headline_scalars`` are the cfs parent's objects rather than copies of
#: them, **by identity**. This cell adds no analysis of its own -- the encoder replacement changes
#: what produces the numbers, not which numbers there are -- so a second registry here could only
#: ever come to differ from the one the comparison model runs under, and the two ``summary.json``
#: files would stop being readable side by side long before anyone noticed.
#:
#: They carry the parent's three cfs-only analyses (``warmup``, ``source_null``,
#: ``spectral_skill``) and their headline scalars, so the second cell reports the same nineteen
#: analyses and the same headline surface as the first. That is what makes the cross-cell table
#: possible: a column present on one side and missing on the other is not a comparison.
#: ``tests/test_eval_binding.py`` asserts the identity, and that
#: ``merged_analysis_functions`` returns the same key set in the same order for both bindings.
TRF_CFS_BINDING = ModelBinding(
    model_cls=SeqVaeLagAttnTrfCfs,
    task_cls=SeqVaeLagAttnTrfCfsTask,
    tag="lag_attn_trf_cfs",
    geometry_keys=GEOMETRY_KEYS,
    encoder_disclosure=trf_cfs_encoder_disclosure,
    overrides_path=DEFAULT_OVERRIDES_PATH,
    extra_analyses=EXTRA_ANALYSES,
    headline_scalars=HEADLINE_SCALARS,
)
