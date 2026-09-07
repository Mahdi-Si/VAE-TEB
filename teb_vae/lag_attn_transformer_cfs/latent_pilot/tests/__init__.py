"""Tests for the latent-class fine-tuning pilot, split by what may run where.

``logic/``
    The small synthetic subset: timestamp and window boundaries, label and split consistency on
    artificial identifiers, bag-reduction weighting, training-only scaler moments on small arrays,
    and configuration precedence. Hand-checkable values, no model construction, no tensor
    optimization, no checkpoint or HDF5 I/O, no GPU, no plots, no subprocess launch. This subset is
    the only one runnable without production data, and it is deliberately import-isolated: nothing
    it imports may build a model or pull in the production pipeline.

``fixtures/``
    Generator code and definitions for the small, explicitly non-clinical fixtures the smoke
    scenario and the integration tests use. Identities are disjoint across splits and both binary
    classes appear in every evaluable split. Artificial data, times and labels are never presented
    as clinical.

Everything else in this directory -- checkpoint loading, gradient reach, mask and model contracts,
the end-to-end smoke scenario and the runner launch parity checks -- needs the real environment and
runs on the execution machine::

    python -m pytest teb_vae/lag_attn_transformer_cfs/latent_pilot/tests/logic -q
    python -m teb_vae.lag_attn_transformer_cfs.latent_pilot.tests.fixtures.generate
    python -m pytest teb_vae/lag_attn_transformer_cfs/latent_pilot/tests -q

The generator in the middle is what the fixture-backed tests need; without it, and only them, skip
with that command in the message. Everything that needs no fixture -- the model contract, the
extraction, the gates, the fits, the figures and the runner's launch parity -- runs on a bare
checkout with the repository environment.
"""
