"""Network components for the causal conv-Transformer lag-attention VAE.

Every module here imports only ``torch``, the standard library, ``entmax`` and the framework-free
``teb_vae`` net layers. No Lightning, no config, no logging, no ``torch.distributed``, and no batch
field names: these are nets, and what feeds them is the caller's problem.
``tests/test_nets_are_framework_free.py`` enforces it by walking the import graph.

:mod:`.blocks` is the one place this package writes rather than imports. Its primitives -- per-token
RMSNorm, rotary position encoding, LayerScale, SwiGLU, causal depthwise convolution, the gated
causal convolution block and the windowed causal self-attention block -- have no equivalent
anywhere under ``teb_vae``, and reaching outside that tree for them would weaken the framework-free
guard for two sibling packages at once.

lean-limit: primitives imported from teb_vae/lag_attn/nets and teb_vae/lag_attn_rws/nets; promote
to a common package when this model outlives its comparison or when a fourth consumer appears.
"""
