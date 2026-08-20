"""Architecture visualisation of :class:`~teb_vae.lag_attn_transformer_cfs.nets.model.SeqVaeLagAttnTrfCfs`.

``README.md`` beside these files says how to repeat this for another model. Two files do the
work and one is their product:

* ``extract_arch.py`` -- builds the production model, traces one forward pass and writes
  ``arch.json``: every torch call with its real input/output shapes, every module with its real
  constructor arguments and parameter counts, and the dataflow edges between them.
* ``arch_viz.html`` -- a self-contained page with ``arch.json`` inlined; every number it renders
  comes from that file.

Not a net. This package is a tool that *reads* the model; it is deliberately outside the
``nets/*.py`` glob the framework-free test walks, because it imports ``yaml`` and walks Python
frames -- neither of which a network module may do.
"""
