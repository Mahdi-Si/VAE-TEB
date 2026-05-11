"""Causal-TE validation suite for VAE-TEB v1.

Public entry point :func:`run_causal_te_validation` is wired into the
testing pipeline as a Phase-1 step in ``run_full_test_pipeline`` (see
``run_tests.py``). Each individual test is also importable as a
post-processor for ad-hoc reruns against an existing Phase-1 output.

The suite implements 6 of the 10 tests defined in ``causal_te.md``;
Tests 5, 6, 7, and 8 are intentionally out of scope (deferred or
covered by other modules; see the spec for the rationale).
"""

from __future__ import annotations

from model.vae_teb_prediction.testing.causal_te_validation.runner import (
    run_causal_te_validation,
)

__all__ = ["run_causal_te_validation"]
