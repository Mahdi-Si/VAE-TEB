"""Transformer testing and analysis pipeline.

Provides comprehensive evaluation of the Causal Multimodal Forecasting
Transformer across forecast quality, TE coupling, representation structure,
temporal trajectories, and cross-class statistical comparisons.
"""

from .base import TransformerTestRunner
from .run_tests import quick_test, run_full_test_pipeline

__all__ = [
    "TransformerTestRunner",
    "run_full_test_pipeline",
    "quick_test",
]
