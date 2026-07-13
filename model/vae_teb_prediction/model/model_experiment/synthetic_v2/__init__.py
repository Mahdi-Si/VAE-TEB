"""``synthetic_v2`` — raw-signal → scattering → block-transfer-entropy validation.

This package is the v2 of the synthetic validation pipeline for the VAE-TEB
lag-attention model. Unlike the v1 ``synthetic`` package (which synthesises the
scattering-*feature* channels directly from a linear-Gaussian state space), v2
synthesises **raw 4 Hz FHR/UP waveforms** with a controlled, analytically-known
block transfer entropy from the UP contraction band to the FHR deceleration band,
then passes them through the **real** ``KymatioPhaseScattering1D`` transform and
normalisation before feeding the model — so the experiment tests whether a known
transfer entropy survives the production feature encoder.

The full design and mathematics are documented in
``SYNTHETIC_V2_RAW_TE_PIPELINE_EXPLAINED.md`` in this folder.

The block-TE math (simulator, determinant-ratio estimator, inverter, SNR law,
realizability probe) is reused verbatim from ``..synthetic.analytic_te``; the
genuinely new code lives in ``raw_generators`` (raw composition + latent pair +
amplitude-modulation rendering) and ``scattering_adapter`` (transform + norm).
"""
