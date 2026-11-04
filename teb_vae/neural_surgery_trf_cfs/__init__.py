r"""Probing and gradient-free weight surgery of the ``lag_attn_transformer_cfs`` epoch-524 checkpoint.

The package holds every script that analysed the checkpoint and every script that edited its
weights without gradient descent, together with the results they produced. Sub-packages:

* ``probes``: layer, latent, attention, calibration and source probes, and linear baselines.
* ``surgery``: four rounds of weight edits, each a closed-form solve, a derivative-free search
  or an exact re-parameterisation, plus two small checkpoint utilities.
* ``evaluation``: the test-half override file and the official evaluation runner.
* ``data_tools``: a control shard of the acidosis shard's size, cut from a larger shard and
  matched recording by recording on the time before delivery.

Shared paths and loader helpers live in :mod:`teb_vae.neural_surgery_trf_cfs.common`.
"""
