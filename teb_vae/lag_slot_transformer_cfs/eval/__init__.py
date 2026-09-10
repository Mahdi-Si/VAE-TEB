r"""Scoring a lag-residual checkpoint: the matched predictive gap, the lag readouts, the controls.

Seven modules, in the order a run uses them:

* :mod:`~teb_vae.lag_slot_transformer_cfs.eval.binding` -- which class a checkpoint is rebuilt
  through, which constructor keys are reconciled against it, what this encoder has to disclose
  about its causal standing, and which shared analyses this architecture cannot produce.
* :mod:`~teb_vae.lag_slot_transformer_cfs.eval.predictive` -- the marginalised Monte Carlo score,
  its concentration diagnostic, and the mixture calibration census.
* :mod:`~teb_vae.lag_slot_transformer_cfs.eval.lag_metrics` -- band suppression margins, the
  cancellation ratio with both of its parts, and per-lag and per-channel exposure.
* :mod:`~teb_vae.lag_slot_transformer_cfs.eval.run` -- the entry point that drives one pass and
  writes ``summary.json``.
* :mod:`~teb_vae.lag_slot_transformer_cfs.eval.verify` -- the single-run gate, which reads a
  finished summary and needs neither a checkpoint nor a numeric stack.
* :mod:`~teb_vae.lag_slot_transformer_cfs.eval.latent_probes` -- frozen probes from each latent
  readout onto the anchor-relative future block, fitted and scored on disjoint recordings.
* :mod:`~teb_vae.lag_slot_transformer_cfs.eval.acceptance` -- several runs together under one
  predeclaration: the seeds, the paired comparisons, the corrected band search and the partition
  no choice was made on.

**This package scores through its own pass rather than through the shared collection pipeline, and
the reason is the anchor axis.** Every latent tensor this architecture produces is indexed by
decoded anchor; the shared pass pairs a dense stored-step support with a latent produced at every
step, and its per-batch readout additionally requires eight attention-derived fields that no
attention-free model computes. Fabricating those fields to satisfy the readout is precisely what
this architecture must not do -- a proposal norm reported under an attention name is a per-lag
attribution that does not exist. What is reused instead is every piece that says nothing about an
architecture: the checkpoint and task loading, the override merge and its schema, the block score,
the log-mean-likelihood, the derangement, the recording-level bootstrap and the summary assembly.
"""
