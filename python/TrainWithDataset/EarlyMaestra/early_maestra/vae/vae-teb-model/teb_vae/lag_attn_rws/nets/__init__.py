"""Network components for the raw-signal lag-attention VAE.

Every module here imports only ``torch``, the standard library, ``entmax`` and the framework-free
``teb_vae`` net layers. No Lightning, no config, no logging, no ``torch.distributed``, and no
batch field names: these are nets, and what feeds them is the caller's problem.
``tests/test_nets_are_framework_free.py`` enforces it by walking the import graph.

lean-limit: the shared primitives are imported from teb_vae/lag_attn/nets/ rather than promoted
to a common package; promote them when a third consumer appears or when this module needs to
modify one of them.
"""
