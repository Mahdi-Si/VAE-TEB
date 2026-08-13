"""Fast hermetic pytest for the causal-feature-domain conv-Transformer lag-attention VAE.

One flat test directory for the whole package. Anything that loads the committed causal shard or
drives a real fit is marked ``slow`` and excluded from the default run::

    .venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_cfs/tests -q -m "not slow"
    .venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_cfs/tests -q -m slow
"""
