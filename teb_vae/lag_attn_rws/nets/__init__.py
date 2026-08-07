"""Network components for the raw-signal lag-attention VAE.

Every module here imports only ``torch``, the standard library, ``entmax`` and the framework-free
``teb_vae`` net layers. No Lightning, no config, no logging, no ``torch.distributed``, and no
batch field names: these are nets, and what feeds them is the caller's problem.
``tests/test_nets_are_framework_free.py`` enforces it by walking the import graph.

The primitives more than one package builds now live in ``teb_vae/lag_attn/``, which is the
common layer every model already imports: ``nets/delays.py`` (``ChannelDelay``, ``ChannelGate``),
``nets/lag_report.py`` (``SECONDS_PER_STEP`` and the lag arithmetic), ``nets/encoders.py``
(``AvailabilityInputAdapter``, beside the ``InputAdapter`` whose stack it reproduces) and
``channel_reach.py`` (the reach table and the budget resolution). The trigger the deferral note
here used to carry -- "when this module needs to modify one of the shared primitives" -- fired
when the causal reach guard had to reach ``lag_attn`` as well: this package sits *downstream* of
``lag_attn``, so leaving the guard here would have made the dependency circular.

What stays local is what one package owns: the raw geometry, the raw masks and targets, the
objective, and the full-latent prior head.
"""
