"""Back-compat shim package.

Older modules under ``model.vae_teb_prediction.new_classifier`` import from
``model.vae_teb_prediction.classifier.*`` — a path that no longer exists in
the repository layout. This package provides minimal re-exports so those
imports continue to resolve, allowing the new ``guid_cls_v1`` pipeline to
reuse the existing CDR / threshold / metric / plotting utilities without
duplicating code.
"""
