"""Offline evaluation of a trained raw-signal lag-attention VAE checkpoint.

One command reads a checkpoint and writes a directory: :mod:`~teb_vae.lag_attn_rws.eval.run`.
What happens inside it is layered, and the layering is what keeps an evaluation from quietly
becoming a second training loop.

* :mod:`~teb_vae.lag_attn_rws.eval.config_schema` and
  :mod:`~teb_vae.lag_attn_rws.eval.preflight` decide whether the run may happen at all, before a
  model, a loader or an output directory exists.
* :mod:`~teb_vae.lag_attn_rws.eval.metrics` computes the readouts and turns them into explicit
  verdicts, through the same loss functions and the same task builders the training objective
  uses -- an evaluation that re-implements the objective it is evaluating measures its own
  re-implementation.
* :mod:`~teb_vae.lag_attn_rws.eval.collect` runs that pass **once** and writes down what it
  produced: two durable tables, a vector sidecar, and a provenance record.
* :mod:`~teb_vae.lag_attn_rws.eval.events` detects contractions and decelerations on the raw
  traces. It is model-free and sits below both consumers: the collection pass, which puts each
  anchor's distance from the last contraction on the per-anchor table, and the event analysis,
  which searches the forecast blocks themselves.
* :mod:`~teb_vae.lag_attn_rws.eval.oracle` fits the evaluation-only decoder that measures what the
  latent bottleneck costs the forecast. It is the one place this package *trains* anything, and
  what it trains has no path back into the checkpoint.
* ``analyses/`` reads those tables. No analysis touches the model, which is what makes re-running
  one against a finished directory cost seconds rather than hours -- with exactly two stated
  exceptions: the per-sample diagnostic pages, which are a *rendering* rather than a readout and
  are re-run over a handful of chosen segments, and the sufficiency gap, whose probe reads an
  encoder state that is on neither table.
* :mod:`~teb_vae.lag_attn_rws.eval.report_seam` assembles the headline and the sanity block, and
  isolates one analysis's failure from the rest of the run.

The model-free pieces -- the pinned numeric environment, the rank statistics, the clinical
labelling, the figure primitives, the channel-to-band map -- are the sibling package's, bound
through :mod:`~teb_vae.lag_attn_rws.eval._reuse` rather than forked.
"""
