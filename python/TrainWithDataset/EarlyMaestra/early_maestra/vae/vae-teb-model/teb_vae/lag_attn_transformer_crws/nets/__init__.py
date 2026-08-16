"""Network components for the causal-input raw-target conv-Transformer lag-attention VAE.

Every module here imports only ``torch``, the standard library, ``entmax`` and the framework-free
``teb_vae`` net layers. No Lightning, no config, no logging, no ``torch.distributed``, and no batch
field names: these are nets, and what feeds them is the caller's problem.
``tests/test_nets_are_framework_free.py`` enforces it by walking the import graph.

This package writes **no** network code. There is one module and it names two imports: the causal
input mixin from ``teb_vae.lag_attn_crws.nets.causal_raw_inputs`` and the architecture from
``teb_vae.lag_attn_transformer_rws.nets.model``. In particular the encoder primitives -- per-token
RMSNorm, rotary position encoding, LayerScale, SwiGLU, causal depthwise convolution and windowed
causal self-attention -- are imported through that parent and not copied, and the warm-up mask, the
lag floor, the anchor tiling and the anchored raw gather are imported through the conv-LSTM cell and
not copied either.
"""
