r"""Evaluating a lag-residual checkpoint through the family's pipeline, with this cell's own pass.

**The runner, the tables, the analyses and the gate are the family's.**
:func:`teb_vae.lag_attn_cfs.eval.run.main` evaluates whichever model its
:class:`~teb_vae.lag_attn_cfs.eval.binding.ModelBinding` names, and this package supplies the
binding. A run of this cell therefore has everything a run of a lag-attentive cell has: the
preflight guards and their recovery table, the loader probe, the two durable tables and the
provenance sidecar that make an offline re-run against a finished directory possible, every
table-driven analysis of the family with its by-class and by-subgroup variants, the headline, the
sanity block, the coverage record and the artifact manifest -- and one directory layout, so two
cells' runs are read down one set of names.

**What is this cell's own, and why each has to be.**

* :mod:`.collect` -- the collection pass. The shared one is written against the lag-attention
  forward: it reads the attention weights, the per-lag divergence map and the source-null arm, and
  reduces a latent produced at every stored step. This architecture computes none of those and
  its latent is indexed by decoded anchor, so it produces the family's tables from its own forward
  -- the same identity columns, the same column names wherever the quantity is the same one, and
  no name at all where it is not -- and scores every arm the design defines in one draw loop.
* :mod:`.analyses` -- six analyses on the binding. Three draw the readouts only this architecture
  has (the arms, the lag suppression, the resolved axes); three draw this model's own forward
  under the family's names (``samples``, ``recording_traces``, ``attribution``), because each
  reads a tensor the other architecture does not emit.
* :mod:`.binding` -- which class a checkpoint is rebuilt through, which constructor keys are
  reconciled against it, what this encoder discloses, the override delta, the pass, the extras,
  and which shared analyses this architecture cannot produce.
* :mod:`.predictive`, :mod:`.lag_metrics`, :mod:`.figures` -- the marginalised score and its
  mixture calibration, the band and per-lag readouts with their limits, and the figures drawn from
  the results block alone.
* :mod:`.verify` -- this cell's own gate over a finished summary, stdlib-only, beside the
  family's sanity block and verdicts.
* :mod:`.latent_probes` and :mod:`.acceptance` -- frozen probes from each latent readout, and
  several runs together under one predeclaration.

**What is deliberately absent.** Seven analyses of the family read an attention distribution or a
per-lag allocation of the divergence. None is handed a substitute: a proposal norm under an
attention name is a per-lag attribution that does not exist, and every summary names the seven
with the tensor each would have needed.

Launch from the repository root::

    python -m teb_vae.lag_slot_transformer_cfs.eval.run --checkpoint <run>/model_checkpoints/<name>.ckpt
    python -m teb_vae.lag_slot_transformer_cfs.eval.verify --summary <run>/eval_results/summary.json
"""
