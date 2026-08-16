"""Network components for the causal-input raw-target lag-attention VAE.

Every module here imports only ``torch``, the standard library, ``entmax`` and the framework-free
``teb_vae`` net layers. No Lightning, no config, no logging, no ``torch.distributed``, and no batch
field names: these are nets, and what feeds them is the caller's problem.

The warm-up resolution itself is deliberately **not** here. ``teb_vae.lag_attn_cfs.causal_warmup``
opens HDF5 files and ``teb_vae.lag_attn_cfs.model_kwargs`` reads a constructor signature, so both sit
above the net layer in the package that owns them; this package reaches them by reference and a net
that reached either would take ``h5py``, a filesystem and an introspected signature into a layer
whose whole contract is that it can be built from integers.
"""
