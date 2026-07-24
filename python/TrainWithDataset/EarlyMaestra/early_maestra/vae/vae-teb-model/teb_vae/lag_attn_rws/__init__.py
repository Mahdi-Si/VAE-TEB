"""The raw-signal lag-attention sequential VAE.

A single-latent VAE over decimated scattering/phase-harmonic features that forecasts the next
two minutes of *raw* normalized FHR ($480$ samples per anchor) twice from one shared decoder:
once from the target-only prior latent and once from the source-conditioned posterior latent.
The per-step KL between the two is the source-conditioned coupling readout, and because the
decoder receives nothing but $z$, the prior latent is forced to carry the FHR predictive state.

Layout::

    nets/       the network, framework-free: torch + stdlib + entmax only
    tests/      fast hermetic pytest, evaluation tests included
    configs/    YAML configs resolved through ``teb_vae.lag_attn.config``

Shared primitives (encoders, attention, decoder core, posterior head, blocks) are imported from
``teb_vae.lag_attn.nets``; config loading (``load_config`` / ``resolve_config_file``) comes from
``teb_vae.lag_attn.config``. Nothing in this package re-implements either. Variants are configs,
not subclasses.
"""
