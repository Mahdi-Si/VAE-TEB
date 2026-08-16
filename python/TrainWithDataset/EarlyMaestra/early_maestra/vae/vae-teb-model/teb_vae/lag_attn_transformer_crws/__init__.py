r"""The causal-input raw-target conv-Transformer lag-attention sequential VAE.

The eighth cell of the encoder-by-target grid, and the conv-Transformer half of the one row whose
forecast claim and lag claim are both exact::

                            conv-LSTM encoders          conv-Transformer encoders
    raw FHR target          lag_attn_rws                lag_attn_transformer_rws
    two-sided feature       lag_attn_fs                 lag_attn_transformer_fs
    causal feature          lag_attn_cfs                lag_attn_transformer_cfs
    causal in / raw out     lag_attn_crws               lag_attn_transformer_crws   <- this

Each half comes from a different parent and neither is re-implemented here:

* **the input domain and the anchored raw objective** from
  ``teb_vae.lag_attn_crws.nets.causal_raw_inputs.CausalRawInputs`` -- the input warm-up mask, the
  lag validity floor, the tiled anchor set, the forward that decodes at it, the raw future window
  gathered at those anchors and the three kept readouts;
* **the architecture** from ``teb_vae.lag_attn_transformer_rws.nets.model.SeqVaeLagAttnTrfRws`` --
  the two causal conv-Transformer encoders, the channel gates, the availability-aware input
  adapters, the prior and posterior heads, the lag cross-attention, the shared horizon decoder and
  the paired reparameterisation.

So :class:`~teb_vae.lag_attn_transformer_crws.nets.model.SeqVaeLagAttnTrfCrws` is a constructor and
nothing else, and that is the point rather than an economy: with nothing else defined here, a
difference in results against ``lag_attn_crws`` is attributable to the encoder alone and a
difference against ``lag_attn_transformer_rws`` to the input representation alone. With all eight
cells present either axis can be read at a fixed value of the other.

**Why the constructor is the one exception.** The experiment driver builds a run's kwargs by
sweeping ``inspect.signature(MODEL_CLS.__init__)``, so a cell whose architecture has a different
keyword schema has to write that schema out. This one has no ``lstm_layers``,
``encoder_extra_dilations``, ``encoder_extra_kernel``, ``conv_norm_groups`` or ``causal_norm``, and
it has seven encoder keys the conv-LSTM cell has never heard of. A ``**kwargs`` signature would
forward four keys and silently build an all-defaults model.

Layout::

    DESIGN.md   the as-built record: which parent supplies which half, and the measured budget
    nets/       the network: one imported mixin over an imported architecture
    task.py     the Lightning task, a diamond of two parents with an empty body
    trainer.py  the experiment driver, three re-pointed class attributes
    configs/    YAML configs resolved through ``teb_vae.lag_attn.config``
    tests/      fast hermetic pytest
    RESULTS.md  the pre-registered criteria the headline run is scored against

**Neither side of the objective contains its own future.** An input coefficient at decimated step
$t$ is a function of $\{x(s) : s \le t\}$ and of nothing else, because the causal dataset variant
built it that way; the target is the raw FHR signal, which has no warm-up, no group delay and no
channel selection. ``lag_attn_transformer_rws`` is the direct control -- same architecture, same
target, same objective -- differing only in whether the inputs contain the answer.

**Step-wise causality holds unconditionally.** The conv-LSTM cell needs ``causal_norm: true`` to
make it true -- without it a time-pooling normaliser mixes the whole sequence. These encoders have
no such normaliser, and ``causal_norm`` is not a constructor keyword of this model at all, so there
is no flag to qualify the claim with.

**Stored causal time is still not physical time, on the input side.** One-sidedness and zero latency
are different properties and only the first is bought by the causal transform: beyond its warm-up a
causal input channel still lags by its composed group delay, $13.3$ to $791.0$ s, uncompensated. The
target carries no group delay at all, so the caveat that reaches a lag-resolved reading here is
one-sided rather than two-sided -- it attaches to the inputs alone.

**Nats from this cell are comparable to no sibling outside this row.** The raw block is
$H \cdot R = 15 \times 16 = 240$ samples against ``lag_attn_transformer_rws``'s $480$, and the tiled
anchor set decodes about $10.1$ anchors per step against a dense $240$. A loss level is comparable
against ``lag_attn_crws``, which ships the identical geometry, and against nothing else.

**There is no evaluation package.** Every number this cell reports is a scalar from one run's own
``train_results/metrics_history.csv``, so no reported difference carries an uncertainty. Stated here
so no row is read as though it had a confidence interval.
"""
