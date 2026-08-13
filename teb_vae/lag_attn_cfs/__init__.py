r"""The causal-feature-domain lag-attention sequential VAE.

The fifth cell of the encoder-by-target grid, and the first whose inputs do not contain their own
future::

                            conv-LSTM encoders          conv-Transformer encoders
    raw FHR target          lag_attn_rws                lag_attn_transformer_rws
    two-sided feature       lag_attn_fs                 lag_attn_transformer_fs
    causal feature          lag_attn_cfs   <- this      lag_attn_transformer_cfs

Structurally this is ``teb_vae.lag_attn_fs`` with one thing changed: which transform produced the
features it reads and forecasts. The two-sided cells read coefficients that at decimated step $t$
average raw samples on **both** sides of $t$ -- up to $965$ s into $t$'s own future on the slowest
channel -- which is why their coupling readout is called ``source_conditioned_kl_raw`` and
deliberately not called a transfer entropy. The causal dataset variant runs the same cascade
through a strictly one-sided gammatone bank, so a coefficient at $t$ is a function of
$\{x(s) : s \le t\}$ and of nothing else, and the forecast claim is exact.

**Causality is bought with a warm-up.** A one-sided filter's output depends on assumed
pre-recording history until $W_{0.95}$, the leading delay enclosing $95\%$ of its kernel energy,
has passed. Because the transform runs per $22$-minute segment, that cost is paid once per segment
rather than once per recording, and on the slowest surviving scattering channel it consumes $278$
of the $300$ steps a sample spans. Two consequences have no analogue in the shipped four cells:
the forecast cannot begin at the model's own $30$-step warm-up, and the set of usable channels is
a feasibility requirement rather than a tuning knob.

Layout::

    DESIGN.md           the as-built record: the geometry, the objective, the readouts, the budget
    causal_warmup.py    the warm-up budget, resolved from the shards themselves
    model_kwargs.py     that budget as the constructor keywords it stamps into every checkpoint
    nets/               the network: two encoder-agnostic mixins over the raw-signal architecture
                        (``causal_inputs`` -- the warm-up mask, the lag floor and the tiled forward;
                        ``causal_feature_target`` -- the one-sided layout and the added readouts)
    task.py             the Lightning task: the tiling phase, and the source-null readout
    trainer.py          the experiment driver and its pre-flight refusals
    configs/            the shipped configuration and its arms
    sample_page.py      the diagnostic page rows the tiling and the warm-up make different
    warmup_budget.py    the run-level warm-up figure, and the tradeoff curve that chose the budget
    check_run.py        scores a finished run directory against the criteria RESULTS.md registers
    tests/              fast hermetic pytest

**The stored warm-up region is not zeros and not NaN.** The dataset writer attaches the boundary
as a per-block attribute and leaves the coefficients untouched, and the normalisation constants
were accumulated *excluding* exactly that region -- so a consumer that ignores the attribute trains
on values that are on no defined scale, with nothing raising. Everything in this package exists to
make that boundary a resolved quantity rather than an assumption.

**Stored causal time is not physical time.** One-sidedness and zero latency are different
properties and only the first is bought here: beyond its warm-up a causal channel still lags by its
composed group delay, $13.3$ to $791.0$ s depending on the channel, and nothing compensates for it.
The forecast claim survives that untouched -- a coefficient at $t$ is a function of the past, so
predicting step $t + 1 + \tau$ from history up to $t$ is a genuine forecast whatever the internal
latency -- but any lag-resolved reading is an attribution over *stored-coefficient* time, not over
physical delay.

lean-limit: no evaluation package, so this model is not comparable through the shared evaluation
pipeline and what a run emits -- ``metrics_history.csv``, the tracked metric surface,
``train/grad_norm`` and the per-epoch diagnostic figures -- is its only readout, scored by
``check_run.py`` rather than by a verdict file; replace with a model binding and an evaluation
package when the feature-domain evaluation contract exists.
"""
