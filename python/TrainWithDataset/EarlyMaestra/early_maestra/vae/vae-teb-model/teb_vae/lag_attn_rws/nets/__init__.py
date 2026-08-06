"""Network components for the raw-signal lag-attention VAE.

Every module here imports only ``torch``, the standard library, ``entmax`` and the framework-free
``teb_vae`` net layers. No Lightning, no config, no logging, no ``torch.distributed``, and no
batch field names: these are nets, and what feeds them is the caller's problem.
``tests/test_nets_are_framework_free.py`` enforces it by walking the import graph.

lean-limit: the shared primitives are imported from teb_vae/lag_attn/nets/ rather than promoted
to a common package; there are now four consumers, so the trigger this note originally named --
a third -- has fired and the promotion is deferred again. Promote them when this module needs to
modify one of the shared primitives, or when the models sharing them stop being compared against
each other. Deferred because the packages downstream of these modules are training and under
active comparison, and a move would touch every test file of each of them at once for no change
in behaviour.
"""
