r"""The FHR-anchored lag-residual causal-feature forecaster.

One $d_z$-dimensional latent space, defined by a target-only Gaussian prior over the fetal-heart-rate
forecast state. Uterine activity makes a **bounded residual correction** to the parameters of a
second Gaussian over that same space, and both distributions feed one shared decoder. The source
stream has no latent partition of its own, no reconstruction objective and no route around the
latent.

What distinguishes this package from its lag-attentive sibling is the source pathway, and only the
source pathway:

* the source representation is **pointwise** -- one stored coefficient in, one value-and-mask pair
  out, no parameters and no temporal operator of any kind;
* each source lag produces one **local deterministic proposal**, conditioned on the anchor's own
  target state, reading exactly one stored source time;
* the proposals meet the latent through an **explicit sum** at the parameter boundary, bounded after
  summation, rather than through a learned competition across lags.

Everything else -- the causal target encoder, the availability adapters, the warm-up budget, the
anchor tiling, the target gather, the prior head and the shared horizon decoder -- is the family's,
reached by import rather than by copy, because two architectures are only comparable if they
optimise the same thing over the same data.
"""
