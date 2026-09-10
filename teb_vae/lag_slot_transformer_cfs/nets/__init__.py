r"""The network modules of the lag-residual causal-feature forecaster.

Framework-free by construction: nothing here imports Lightning or any training framework, so a
module in this package can be exercised from a plain script and a unit test without a trainer, a
config or a data module. The training half lives beside this package, not inside it.

Two modules carry the architecture's own computation:

* :mod:`~teb_vae.lag_slot_transformer_cfs.nets.pointwise_source` -- the value-and-mask
  representation of each available source coefficient, and the per-anchor per-lag gather that
  reads it.
* :mod:`~teb_vae.lag_slot_transformer_cfs.nets.lag_updates` -- the lag embeddings, the shared
  proposal head, the explicit sum with its bounds, the prior-relative full parameters and the
  residual-form divergence.

A third carries the interventions those two make answerable:

* :mod:`~teb_vae.lag_slot_transformer_cfs.nets.controls` -- proposal suppression over a lag band,
  source-value replacement, cross-recording pairing, and the selectors-off equality arm. It lives
  under ``nets`` rather than under the evaluation package because it is arithmetic on the
  architecture's own tensors: the training loop may reach for it, and it must import without a
  data module.
"""
