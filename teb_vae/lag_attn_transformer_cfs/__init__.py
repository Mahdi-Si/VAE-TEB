r"""The causal-feature-domain conv-Transformer lag-attention sequential VAE.

The sixth cell of the encoder-by-target grid, and the one that closes it::

                            conv-LSTM encoders          conv-Transformer encoders
    raw FHR target          lag_attn_rws                lag_attn_transformer_rws
    two-sided feature       lag_attn_fs                 lag_attn_transformer_fs
    causal feature          lag_attn_cfs                lag_attn_transformer_cfs   <- this

Each half comes from a different parent and neither is re-implemented here:

* **the target domain and the one-sided input handling** from ``teb_vae.lag_attn_cfs.nets`` -- the
  input warm-up mask, the lag validity floor, the tiled anchor set and the forward that decodes at
  it (:class:`~teb_vae.lag_attn_cfs.nets.causal_inputs.CausalWarmupInputs`), plus the one-sided
  channel layout, the budget-and-floor refusal and the eleven resolved readouts
  (:class:`~teb_vae.lag_attn_cfs.nets.causal_feature_target.CausalFeatureForecastTarget`);
* **the architecture** from
  ``teb_vae.lag_attn_transformer_rws.nets.model.SeqVaeLagAttnTrfRws`` -- the two causal
  conv-Transformer encoders, the channel gates, the availability-aware input adapters, the prior and
  posterior heads, the lag cross-attention, the shared horizon decoder and the paired
  reparameterisation.

So :class:`~teb_vae.lag_attn_transformer_cfs.nets.model.SeqVaeLagAttnTrfCfs` is a constructor and
nothing else, and that is the point rather than an economy: with nothing else defined here, a
difference in results against ``lag_attn_cfs`` is attributable to the encoder alone and a difference
against ``lag_attn_transformer_fs`` to the transform alone. With all six cells present either axis
can be read at a fixed value of the other.

**Why the constructor is the one exception.** The experiment driver builds a run's kwargs by
sweeping ``inspect.signature(MODEL_CLS.__init__)``, so a cell whose architecture has a different
keyword schema has to write that schema out. This one has no ``lstm_layers``,
``encoder_extra_dilations``, ``encoder_extra_kernel``, ``conv_norm_groups`` or ``causal_norm``, and
it has seven encoder keys the conv-LSTM cell has never heard of. A ``**kwargs`` signature would
forward four keys and silently build an all-defaults model.

Layout::

    nets/       the network: two imported mixins over an imported architecture
    task.py     the Lightning task, a diamond of two parents with an empty body
    trainer.py  the experiment driver, three re-pointed class attributes
    configs/    YAML configs resolved through ``teb_vae.lag_attn.config``
    tests/      fast hermetic pytest
    RESULTS.md  the pre-registered criteria the headline run is scored against

**Causality is bought with a warm-up.** A one-sided filter's output depends on assumed pre-recording
history until $W_{0.95}$ has passed, and because the transform runs per $22$-minute segment that cost
is paid once per segment. The forecast therefore cannot begin at the model's own warm-up, and the set
of usable channels is a feasibility requirement rather than a tuning knob. Both resolutions are the
conv-LSTM cell's, reached through ``teb_vae.lag_attn_cfs.causal_warmup``.

**Step-wise causality holds unconditionally.** The conv-LSTM cell needs ``causal_norm: true`` to make
it true -- without it a time-pooling normaliser mixes the whole sequence. These encoders have no such
normaliser, and ``causal_norm`` is not a constructor keyword of this model at all, so there is no flag
to qualify the claim with.

**Stored causal time is not physical time.** One-sidedness and zero latency are different properties
and only the first is bought here: beyond its warm-up a causal channel still lags by its composed
group delay, $13.3$ to $791.0$ s depending on the channel, and nothing compensates for it. The
forecast claim survives that untouched -- a coefficient at $t$ is a function of the past, so
predicting step $t + 1 + \tau$ from history up to $t$ is a genuine forecast whatever the internal
latency -- but any lag-resolved reading is an attribution over *stored-coefficient* time, not over
physical delay.

lean-limit: no evaluation package, so this model is not comparable through the shared evaluation
pipeline and what a run emits -- ``metrics_history.csv``, the tracked metric surface,
``train/grad_norm`` and the per-epoch diagnostic figure -- is its only readout; replace with a model
binding and an evaluation package when the feature-domain evaluation contract exists.
"""
