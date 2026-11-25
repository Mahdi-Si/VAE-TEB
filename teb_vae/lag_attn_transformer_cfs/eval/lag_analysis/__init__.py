"""Portable lag analysis of one checkpoint of the lag-attention / IRF model family.

    python -m teb_vae.lag_attn_transformer_cfs.eval.lag_analysis \
        --checkpoint <ckpt> --shards <glob or files ...> --stats <stats.hdf5> --out <dir>

It repeats the essential measurements of the lag study of checkpoint 369 on the shards it is given, all
of them, as one evaluation set, and writes a small results folder. See DESIGN.md, section "Lag analysis".
`core.py`: model, data, donors, arms. `steps.py`: the nine measurement steps. `report.py`: statistics,
tables and figures. `__main__.py`: the command line.
"""
