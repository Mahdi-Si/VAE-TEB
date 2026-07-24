"""Fast hermetic pytest for the raw-signal lag-attention VAE.

One test directory for the whole module, evaluation included. Slow validation (the empirical
causal-leak measurement) is marked ``slow`` and excluded from the default run::

    .venv/Scripts/python.exe -m pytest teb_vae/lag_attn_rws/tests -q -m "not slow"
    .venv/Scripts/python.exe -m pytest teb_vae/lag_attn_rws/tests -q -m slow
"""
