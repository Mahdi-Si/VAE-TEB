"""Back-compat re-export of the validation helpers.

The canonical implementation lives at
``model/vae_teb_prediction/old_classifier/validation_utils.py``. This shim
keeps imports such as ``model.vae_teb_prediction.classifier.validation_utils``
(used by the legacy ``new_classifier.evaluate_classifier``) working.
"""

from model.vae_teb_prediction.old_classifier.validation_utils import (  # noqa: F401
    ensure_epoch_hours,
    log_dataframe_stats,
    validate_guid_consistency,
    validate_predictions_df,
    verify_clinical_decision_rule,
)

__all__ = [
    "ensure_epoch_hours",
    "log_dataframe_stats",
    "validate_guid_consistency",
    "validate_predictions_df",
    "verify_clinical_decision_rule",
]
