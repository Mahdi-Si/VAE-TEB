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
    eval/               the evaluation pipeline: one checkpoint in, one reviewable directory out
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

**A run has two readouts and they answer two questions.** ``check_run.py`` reads a run's own
``metrics_history.csv`` and says whether the fit behaved -- in-sample, per epoch, with no
denominator and no interval, and while the run is still going. ``eval/`` reads the finished
checkpoint against the held-out causal split and says whether it is acceptable -- per recording,
with bootstrap intervals, ten pre-registered verdicts and an offline gate that imports no ``torch``.
Neither substitutes for the other; ``DESIGN.md`` §14 and ``eval/EVAL.md`` both carry the pairing.

lean-limit: the frequency-resolved readout is band-resolved skill and its timing half is unmeasured,
because a stored coefficient is a modulus and the analysing filter's phase was discarded before the
value was written -- so a forecast that is right in every band but arrives a step late reads as a
forecast that is right; replace with a phase-carrying readout when the dataset stores a complex or
phase-preserving block, which is a dataset change rather than an evaluation one.

lean-limit: the lag axis is stored-coefficient time, uncorrected for a composed group delay reaching
$791$ s -- the same order as the $364$ s lag search itself; replace with a per-channel-pair physical
lag built from ``causal_delay_s`` when a lag result is to be reported as a physiological delay
rather than as a coefficient-time attribution.

lean-limit: ``eval_config.clock_margin_min_nats`` ships unset, so the availability-clock verdict
reports INCONCLUSIVE and the gate is nine criteria rather than ten; replace with a value derived
from the observed spread of the coupling-minus-clock difference across recordings once the first
production run on the causal holdout split has written its ``source_null`` table.
"""
