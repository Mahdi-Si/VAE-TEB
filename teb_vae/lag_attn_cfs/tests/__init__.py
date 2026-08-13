"""Fast hermetic pytest for the causal-feature-domain lag-attention VAE.

One flat test directory for the whole package. Anything that loads the committed shard or drives a
real fit is marked ``slow`` and excluded from the default run::

    .venv/Scripts/python.exe -m pytest teb_vae/lag_attn_cfs/tests -q -m "not slow"
    .venv/Scripts/python.exe -m pytest teb_vae/lag_attn_cfs/tests -q -m slow
"""
