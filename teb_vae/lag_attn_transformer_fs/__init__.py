r"""The conv-Transformer feature-domain lag-attention sequential VAE.

The fourth cell of the encoder-by-target grid, and the one that closes it::

                          conv-LSTM encoders          conv-Transformer encoders
    raw FHR target        lag_attn_rws                lag_attn_transformer_rws
    feature target        lag_attn_fs                 lag_attn_transformer_fs   <- this

Each half comes from a different parent, and neither is re-implemented here:

* **the target domain** from ``teb_vae.lag_attn_fs.nets.feature_target.FeatureForecastTarget`` --
  the decoder's width, the gathered-never-delayed target block, the four resolved forecast gaps and
  the delegation that hands the shared objective both;
* **everything else** from ``teb_vae.lag_attn_transformer_rws.nets.model.SeqVaeLagAttnTrfRws`` --
  the two causal conv-Transformer encoders, the channel gates, the availability-aware input
  adapters, the prior and posterior heads, the lag cross-attention, the shared horizon decoder and
  the paired reparameterisation.

So :class:`~teb_vae.lag_attn_transformer_fs.nets.model.SeqVaeLagAttnTrfFs` is an empty class body,
and that is the point rather than an economy: with nothing defined here, a difference in results
against ``lag_attn_fs`` is attributable to the encoder alone and a difference against
``lag_attn_transformer_rws`` to the target domain alone. With all four cells present either axis
can be read at a fixed value of the other, which is what neither of the two three-cell
configurations allowed.

**What it is not.** ``lag_attn_rws/PRIOR_SCALE_AND_PREDICTIVE_GAP.md`` section 5 establishes that
the raw models' held-out predictive gain is negative because the source pathway does not generalise
-- a failure living in the source encoder, the lag attention and the posterior fusion, none of
which an encoder swap of this kind touches. This model is expected to reproduce it. Its value is
that it removes a confound, and it must not be read as a fix for one.

Layout::

    nets/       the network, framework-free: torch + stdlib + entmax only
    tests/      fast hermetic pytest
    configs/    YAML configs resolved through ``teb_vae.lag_attn.config``

**The target is gathered, never delayed.** The input
:class:`~teb_vae.lag_attn.nets.delays.ChannelGate` does two things -- gather the channels surviving
the reach budget, then delay each survivor by $\delta_c = \lceil \rho_c / \Delta \rceil$ steps so
its forward reach falls behind the anchor's causal endpoint. The forecast target takes the gather
and **not** the delay: a delayed target would silently ask anchor $t$ to forecast the future of
anchor $t - \delta_c$, and no shape check anywhere would notice. The gate is built at *this*
architecture's own construction site, so that the mixin reaches an un-delayed keep-index here is a
fact this package tests rather than inherits.

**Step-wise causality holds unconditionally.** The conv-LSTM cell needs ``causal_norm: true`` to
make it true -- without it a time-pooling normaliser mixes the whole sequence. These encoders have
no such normaliser: ``RMSNorm`` reduces over channels only, convolutions pad left, and attention is
causal by kernel flag or explicit band mask. ``causal_norm`` is not a constructor keyword of this
model at all, so there is no flag to qualify the claim with.

**The nats are not comparable across target domains, nor across reach budgets within this model.**
The reconstruction is summed over $H_d \cdot C_{\mathrm{keep}} = 2340$ coefficients against the raw
models' $H_d \cdot R = 480$ samples, and $C_{\mathrm{keep}}$ moves with the budget, so every budget
sums a different block. The numbers rank arms of this model at one budget and nothing else.

lean-limit: no evaluation package, so this model is not comparable through the shared evaluation
pipeline and what a run emits -- ``metrics_history.csv``, the tracked metric surface,
``train/grad_norm`` and the per-epoch diagnostic figure -- is its only readout; replace with a model
binding and an evaluation package when the feature-domain evaluation contract exists. Deferred for
the reason the feature-domain sibling defers it: ``collect``, ``metrics``, ``spectra``, ``oracle``,
``events``, ``coherence`` and ``samples`` are structurally raw, and a feature-domain ``events`` or
``coherence`` is a new scientific construction rather than a port.
"""
