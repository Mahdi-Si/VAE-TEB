r"""The evaluation pipeline for the conv-Transformer causal-feature lag-attention VAE.

**Almost none of it is here, and that is the design.** :class:`SeqVaeLagAttnTrfCfs` is
:class:`SeqVaeLagAttnCfs` with both history encoders replaced and nothing else changed -- the same
objective, the same target domain, the same anchor tiling, the same warm-up budget, the same prior
head, the same lag cross-attention, the same shared horizon decoder. Two architectures are only
comparable if they are measured by one implementation, so this package supplies what
:mod:`teb_vae.lag_attn_cfs.eval` cannot derive and imports the rest.

Local, because each is a fact about *this* model that no shared module can compute:

* :mod:`.binding` -- ``TRF_CFS_BINDING``: the classes to rebuild from a checkpoint, the constructor
  keys reconciled against it, this encoder's own causality disclosure, and this package's override
  delta;
* ``configs/eval_overrides.yaml`` -- the causal holdout split and the evaluation-only settings,
  deliberately the cfs cell's file key for key and value for value;
* :mod:`.run` -- the command line, which supplies the binding and a ``prog=`` string and enumerates
  its own flags for one reason: ``--only`` and ``--skip`` must name *this* model's registry;
* :mod:`.verify` -- the acceptance gate, delegated in full, beside the one sweep axis this cell
  ships an arm for (``anchor_stride``) and the cross-cell table the two cfs cells are read down.

Everything else -- the preflight guards, the probe, the collection pass, the nineteen analyses, the
readouts, the verdict registry and the gate's criteria -- comes from the cfs pipeline unchanged,
reached through the binding.

**Why the binding lands before the runner does.** ``SeqVaeLagAttnTrfCfs`` is a constructor over the
transformer encoders plus the same two target-domain mixins, so its binding is a set of
*declarations* that can be wrong in exactly one interesting way: a key that names nothing.
``preflight.reconcile`` silently skips any key absent from either the config or the checkpoint, so a
``geometry_keys`` entry that is not both a constructor parameter and a config key is a reconciliation
that never happens and never says so. That failure is cheap to find as soon as the guards exist and
expensive to find after the whole pipeline has been validated against one cell alone, which is why
this package exists before it can be run.

Launch from the repository root::

    python -m teb_vae.lag_attn_transformer_cfs.eval.run --checkpoint <run>/model_checkpoints/<name>.ckpt
    python -m teb_vae.lag_attn_transformer_cfs.eval.verify <run>/eval_results/summary.json
    python -m teb_vae.lag_attn_transformer_cfs.eval.verify --runs <dir> --out RESULTS_arms.md
"""
