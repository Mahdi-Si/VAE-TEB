r"""The evaluation pipeline for the patch-token lag-attention VAE: a binding, not a fork (plan D1).

The pipeline is ``teb_vae/lag_attn_cfs/eval``. This package supplies what it cannot derive:

* :mod:`.view` -- the eval view of the model and task (CFS five-argument forward, patch streams);
* :mod:`.binding` -- ``TRF_PATCH_BINDING`` (classes, geometry keys, disclosure, registry, guards);
* :mod:`.raw` -- the raw-signal substrate every new analysis builds on;
* :mod:`.analyses` -- the four ports and ten new raw-signal analyses;
* :mod:`.run` / :mod:`.verify` -- the command lines; ``configs/`` -- the override deltas.

See ``EVAL.md`` and ``../notes/EVAL_PLAN.md``.
"""
