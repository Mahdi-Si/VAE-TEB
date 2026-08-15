"""Offline evaluation of a trained causal-feature-domain lag-attention VAE checkpoint.

One command reads a checkpoint and writes a directory: :mod:`~teb_vae.lag_attn_cfs.eval.run`.
What happens inside it is layered, and the layering is what keeps an evaluation from quietly
becoming a second training loop.

* :mod:`~teb_vae.lag_attn_cfs.eval.config_schema` and
  :mod:`~teb_vae.lag_attn_cfs.eval.preflight` decide whether the run may happen at all, before a
  model, a loader or an output directory exists.
* :mod:`~teb_vae.lag_attn_cfs.eval.metrics` computes the readouts and turns them into explicit
  verdicts, through the same loss functions and the same task builders the training objective
  uses -- an evaluation that re-implements the objective it is evaluating measures its own
  re-implementation.
* :mod:`~teb_vae.lag_attn_cfs.eval.collect` runs that pass **once** and writes down what it
  produced: two durable tables, a vector sidecar, and a provenance record.
* ``analyses/`` reads those tables. No analysis touches the model, which is what makes re-running
  one against a finished directory cost seconds rather than hours.
* :mod:`~teb_vae.lag_attn_cfs.eval.report_seam` assembles the headline and the sanity block, and
  isolates one analysis's failure from the rest of the run.

**This package is a fork of** :mod:`teb_vae.lag_attn_rws.eval`, and the fork is deliberate rather
than incidental. The raw pipeline's four-field ``ModelBinding`` reaches the two history encoders
and nothing else, which is enough for a sibling that differs in its encoders -- and this cell
differs in its **target domain**. The raw grid is written into that pipeline in places a binding
does not reach: a three-argument forward against this cell's five, a dense
``build_future_target``/``forecast_mask`` pair against this cell's anchored gathers, baselines
built on the 4 Hz raw grid, and a bpm conversion applied to every error. Reaching all of that
through callables on a binding would have been six or seven new fields plus a bit-identity gate on
two shipped packages, so the pipeline is copied and edited instead.

A fork is how two things that must stay comparable stop being comparable, so four named measures
travel with it and each is a test rather than an intention:

1. The model-free primitives are **not** forked. :mod:`~teb_vae.lag_attn_cfs.eval._reuse` still
   binds ``teb_vae.lag_attn.eval``'s clinical labelling, rank statistics, bootstrap, figure
   primitives and summary assembly. Forking those would fork the definition of a cohort.
2. ``tests/test_eval_sibling_agreement.py`` re-derives the shared arithmetic through both
   packages on identical stub inputs and asserts equality.
3. ``divergences.json`` beside this file classifies every one of the sibling's modules as
   ``equivalent``, ``divergent`` or ``absent``, and the register in ``EVAL.md`` is rendered from
   it rather than hand-kept.
4. ``tests/test_eval_self_contained.py`` carries ``teb_vae.lag_attn_rws.eval`` on its **forbidden**
   list: this package must not reach sideways into the pipeline it was forked from, because a
   half-fork is worse than either whole.

Nothing is re-exported here that the sibling's ``__init__`` does not re-export, which is nothing:
every consumer names the module it wants.
"""
