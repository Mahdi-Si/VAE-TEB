"""Network components for the end-to-end causal conv-Transformer lag-attention VAE.

Every module here imports only ``torch``, the standard library, ``entmax`` and the framework-free
``teb_vae`` net layers. No Lightning, no config, no logging, no ``torch.distributed``, and no batch
field names: these are nets, and what feeds them is the caller's problem.
``tests/test_nets_are_framework_free.py`` enforces it by walking the import graph.

:mod:`.frontend` is the one place this package writes rather than imports. A strictly causal,
anti-aliased map from a raw $4$ Hz signal onto the $4$ s token grid has no equivalent anywhere
under ``teb_vae``, and it is the whole reason this package exists: every other component is the
sibling's, imported unchanged, so that the only difference between the two models is what their
encoders are shown.

lean-limit: primitives imported from teb_vae/lag_attn/nets, teb_vae/lag_attn_rws/nets and
teb_vae/lag_attn_transformer_rws/nets rather than promoted to a common package; this package is the
fourth consumer, so both siblings' promotion triggers have now fired and each is deferred again for
the reason recorded there. Promote to a common package when this model outlives its comparison, or
when a consumer needs to modify one of the shared primitives rather than only import it.
"""
