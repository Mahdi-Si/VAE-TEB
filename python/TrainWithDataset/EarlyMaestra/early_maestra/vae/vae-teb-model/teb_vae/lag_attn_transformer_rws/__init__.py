r"""The causal conv-Transformer lag-attention sequential VAE over raw fetal heart rate.

Structurally this is ``teb_vae.lag_attn_rws`` with both history encoders replaced. A single-latent
VAE over decimated scattering/phase-harmonic features forecasts the next two minutes of *raw*
normalized FHR ($480$ samples per anchor) twice from one shared decoder: once from the target-only
prior latent and once from the source-conditioned posterior latent. The per-step KL between the
two is the source-conditioned coupling readout.

What differs is how each stream's history state $H^s \in \mathbb R^{B \times T \times 128}$ is
computed. In place of a dilated causal convolution stack running beside a unidirectional LSTM,
each stream gets a lightweight causal conv-Transformer: a short gated depthwise convolutional stem
for local morphology, then pre-normalised causal self-attention blocks with rotary position
encoding -- full causal context for the target, a bounded causal window for the source, so a
source state stays a *local* neighbourhood summary and the late lag cross-attention keeps its
ability to tell adjacent delays apart.

Layout::

    nets/       the network, framework-free: torch + stdlib + entmax only
    tests/      fast hermetic pytest
    configs/    YAML configs resolved through ``teb_vae.lag_attn.config``

The import rule: everything shared is imported from ``teb_vae.lag_attn.nets`` and
``teb_vae.lag_attn_rws.nets`` -- the geometry, the raw targets and masks, the objective, the prior
and posterior heads, the lag cross-attention, the decoder, the channel gates, the lag report -- and
nothing here re-implements any of it. A change to one of those belongs in the package that owns it,
where both models get it.

The single exception is the Transformer primitive layer in :mod:`.nets.blocks`. Per-token RMSNorm,
rotary position encoding, LayerScale, SwiGLU, causal depthwise convolution and windowed causal
self-attention do not exist anywhere under ``teb_vae``, so they are written here rather than
imported from outside the package tree.
"""
