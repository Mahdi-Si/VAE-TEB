"""Back-compat re-export of the segment-level classifier classes.

The canonical implementation lives at
``model/vae_teb_prediction/new_classifier/prediction_classification_model.py``.
This shim keeps imports such as
``model.vae_teb_prediction.classifier.prediction_classification_model`` (used
by the legacy ``new_classifier.evaluate_classifier``) working.
"""

from model.vae_teb_prediction.new_classifier.prediction_classification_model import (  # noqa: F401
    BaseTimeSeriesClassifier,
    BiLSTMAttentionClassifier,
    CNN1DClassifier,
    CNNLSTMClassifier,
    CausalCNNLSTMClassifier,
    LSTMClassifier,
    MambaClassifier,
    MultiScaleConvAttentionClassifier,
    TransformerClassifier,
    VaeTebTimeSeriesClassifier,
)

__all__ = [
    "BaseTimeSeriesClassifier",
    "BiLSTMAttentionClassifier",
    "CNN1DClassifier",
    "CNNLSTMClassifier",
    "CausalCNNLSTMClassifier",
    "LSTMClassifier",
    "MambaClassifier",
    "MultiScaleConvAttentionClassifier",
    "TransformerClassifier",
    "VaeTebTimeSeriesClassifier",
]
