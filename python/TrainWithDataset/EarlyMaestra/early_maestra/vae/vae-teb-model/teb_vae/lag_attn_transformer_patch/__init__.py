"""The patch-token conv-Transformer lag-attention sequential VAE.

``teb_vae.lag_attn_transformer_cfs`` with the ST/PH coefficients replaced by non-overlapping
16-sample raw patches (one 4 s token per stream) and the forecast target replaced by two per-patch
summaries, level and variability. Everything else is imported from the sibling packages unchanged.
"""
