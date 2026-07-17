"""Training pipeline for the Causal Multimodal Forecasting Transformer.

Public API:
    - PlCausalTransformer: Lightning module with 3-stage training schedule.
    - CausalTransformerTrainer: Trainer orchestrator with MLflow and DDP.
    - TransformerPlotCallback: Diagnostic plotting callback.
    - main: Entry point for running the full training pipeline.
"""

from .lightning_module import PlCausalTransformer
from .plotting_callback import TransformerPlotCallback
from .trainer import CausalTransformerTrainer, main

__all__ = [
    "PlCausalTransformer",
    "CausalTransformerTrainer",
    "TransformerPlotCallback",
    "main",
]
