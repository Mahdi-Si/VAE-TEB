"""Deprecated compatibility stub for the removed experimental self-latent path."""

from __future__ import annotations

from loguru import logger
def run_self_latent_analysis(*args, **kwargs):
    """Return a skip marker for backward-compatible callers."""
    logger.warning(
        "Self-latent analysis is deprecated because the default model "
        "has no z_self path."
    )
    return {"skipped": True, "reason": "z_self removed from default model"}
