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
    eval/       four files: a binding, an override delta, a runner and a gate -- and no numeric
                function, so both cfs cells are measured by one implementation
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

**This cell is evaluated by the causal parent's pipeline, not by a copy of it.**
``teb_vae.lag_attn_transformer_cfs.eval`` supplies a ``ModelBinding``, an override delta, a runner
and a gate, and delegates every readout, every analysis and every verdict to
``teb_vae.lag_attn_cfs.eval`` -- which is what makes a difference between the two cfs cells'
summaries attributable to the encoder rather than to two implementations. Since the final revision
(``DESIGN.md``, amendment of 2026-09-23) a loss level is comparable against neither neighbour.
``lag_attn_cfs`` forecasts $10$ steps and scores every cell; this cell forecasts $30$ steps over
$30 \times 76 = 2280$ cells, scores a fast phase channel over its first $5$ steps only, and scores
the residual under an AR(1) innovation. ``lag_attn_transformer_fs`` scores $30 \times 78 = 2340$
two-sided coefficients. Across either edge, read signs and orderings only.

lean-limit: the frequency-resolved readout is band-resolved skill and its timing half is unmeasured,
because a stored coefficient is a modulus and the analysing filter's phase was discarded before the
value was written; replace with a phase-carrying readout when the dataset stores a complex or
phase-preserving block, which is a dataset change rather than an evaluation one.

lean-limit: the lag axis is stored-coefficient time, uncorrected for a composed group delay reaching
$791$ s -- larger than the $148$ s lag search itself ($\ell \le 37$ steps); replace with a
per-channel-pair physical lag built from ``causal_delay_s`` when a lag result is to be reported as
a physiological delay rather than as a coefficient-time attribution.

lean-limit: ``eval_config.clock_margin_min_nats`` ships at $0.15$ nats, so the availability-clock
verdict is decided and the gate reads all ten verdicts. The value was calibrated on an earlier
unaligned run of this cell under the legacy representation, which measured $0.160$ nats
($[0.157, 0.164]$; ``LAG_READOUT_DIAGNOSIS.md``); replace it with a value derived from the spread
of the coupling-minus-clock difference across recordings once a production run of the shipped
geometry has written its ``source_null`` table.
"""
