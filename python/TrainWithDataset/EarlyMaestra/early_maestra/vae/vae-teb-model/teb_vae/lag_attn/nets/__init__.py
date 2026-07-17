"""Network components for the lag-attention VAE.

Every module here imports only ``torch``, the standard library and ``entmax``. No Lightning, no
config, no logging, no ``torch.distributed``, and no batch field names: these are nets, and what
feeds them is the caller's problem. ``tests/test_nets_are_framework_free.py`` enforces it by
walking the import graph.

Keeping this layer free of the framework is what lets it be constructed, forwarded and asserted in
a test without a config file, a run directory, or a GPU.
"""
