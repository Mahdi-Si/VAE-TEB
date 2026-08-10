"""Network components for the conv-Transformer feature-domain lag-attention VAE.

Every module here imports only ``torch``, the standard library, ``entmax`` and the framework-free
``teb_vae`` net layers. No Lightning, no config, no logging, no ``torch.distributed``, and no batch
field names: these are nets, and what feeds them is the caller's problem.
``tests/test_nets_are_framework_free.py`` enforces it by walking the import graph.

This package writes **no** network code. There is one module and it names two imports: the target
domain from ``teb_vae.lag_attn_fs.nets.feature_target`` and the architecture from
``teb_vae.lag_attn_transformer_rws.nets.model``. In particular the encoder primitives -- per-token
RMSNorm, rotary position encoding, LayerScale, SwiGLU, causal depthwise convolution and windowed
causal self-attention -- are imported through that parent and not copied: a search of
``lag_attn_transformer_rws/nets/blocks.py`` and ``nets/encoders.py`` for ``raw``, ``fhr`` or
``raw_per_step`` returns nothing, so there is no raw-domain assumption in either that a
feature-domain copy would have to edit.
"""
