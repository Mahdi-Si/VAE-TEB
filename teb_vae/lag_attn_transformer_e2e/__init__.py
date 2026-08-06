r"""The end-to-end causal conv-Transformer lag-attention sequential VAE over raw signals.

Structurally this is ``teb_vae.lag_attn_transformer_rws`` with the two encoder *inputs* replaced,
and nothing else. Everything from the encoder inputs down -- the encoders themselves, the prior and
posterior heads, the lag cross-attention, the shared horizon decoder, the four-term objective, the
geometry, the masks and the metric surface -- is imported from the sibling packages unchanged, so a
difference in results between this model and that one is attributable to the input representation
and to nothing else.

What it replaces, and why. The sibling models condition on **stored wavelet features**: scattering
and phase-harmonic coefficients on the $4$ s decimated grid. Those coefficients are two-sided --
the value at decimated step $t$ is a weighted average over raw samples on both sides of $t$ -- so a
model conditioning on "the history up to $t$" is conditioning on part of the interval it is being
asked to forecast. The leak is largest in the deceleration and baseline bands, which are the
clinically load-bearing ones, and it is worst where it does the most damage: a source-stream leak
enters the posterior alone, so it inflates the predictive gain, inflates the source-conditioned KL
and shifts the lag map -- the three readouts the model exists to produce. Pruning channels by
analytic reach bounds the leak but cannot close it, and neither can a longer forecast horizon.

This package closes it by construction. Both streams are fed the raw $4$ Hz signals the batch
already carries, through a learned strictly causal front end, so the history state at anchor $t$ is
a function of raw samples at index $\le 16t + 15$ -- exactly the anchor's own causal endpoint --
and cannot read the future at all. What makes that affordable is the sibling's conv-Transformer
encoder: the features existed to hand the model long-range structure and handed it over acausally,
and $300$ tokens of causal attention supply the same structure from the strict past.

Layout::

    nets/       the network, framework-free: torch + stdlib + entmax only
    tests/      fast hermetic pytest
    configs/    YAML configs resolved through ``teb_vae.lag_attn.config``

The import rule: everything shared is imported from ``teb_vae.lag_attn.nets``,
``teb_vae.lag_attn_rws.nets`` and ``teb_vae.lag_attn_transformer_rws.nets``, and nothing here
re-implements any of it. A copy of the objective would make the comparison partly a comparison of
two losses; a copy of the encoder would let an encoder difference masquerade as a result about the
input representation. The single exception is the causal raw front end in :mod:`.nets.frontend`:
nothing under ``teb_vae`` performs causal anti-aliased decimation of a raw signal, and a strictly
one-sided map from the $4$ Hz grid to the $4$ s token grid cannot be assembled from what exists.

lean-limit: no evaluation package, so this model is not comparable through the shared evaluation
pipeline and what a run emits -- ``metrics_history.csv``, the tracked metric surface,
``train/grad_norm`` and the per-epoch diagnostic figure -- is its only readout; replace with a
model binding and an evaluation package when the evaluation contract for this architecture exists.
"""
