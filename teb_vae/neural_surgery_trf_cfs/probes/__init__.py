r"""Probes of the epoch-524 checkpoint: layers, latent, attention, calibration, source and baselines.

Session 1 (``probe_layers``, ``linear_baselines``, ``analyze_probe``) measured static weights,
encoder norms and self-attention, latent statistics, lag attention, forward ablations and linear
forecast baselines. Session 2 (``probe_physiology``, ``analyze_physiology``,
``source_interventions``, ``training_history``, ``weight_norms``) added per-block ablations, linear
probes of the latent, innovation and marginal calibration, the source gain against contractions and
FHR drops, and the UA value-versus-mask intervention.
"""
