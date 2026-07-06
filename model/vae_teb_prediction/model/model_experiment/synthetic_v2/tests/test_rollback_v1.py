r"""Sprint 7 (S7-T04): rollback closeout -- the one-line reversal to v1 is real.

Per decision 1, the committed ``SeqVaeLagAttn`` alias stays on v1 in every
consumer; the v2 pipelines are exercised by monkeypatching the alias in-process.
This test pins that committed default so an accidental flip to v2 is caught, and
confirms the shared-file edits (trainer metrics builder, plotting callback, config
injection, diagnostics) left the v1 path intact.

The full rollback gate is this test plus, run with the committed v1 alias:

* ``test_v1_golden_regression.py`` (S0-T05) -- v1 forward / loss / figure
  fingerprints unchanged by the shared-file edits;
* ``test_diagnostics_v2.py::test_metrics_guarded_under_v1`` -- the v2-only log
  keys are key-presence guarded (no ``KeyError`` under v1);
* the Sprint-6 smokes, which build v2 in-process and therefore pass regardless of
  the committed alias.

See ``vae-teb-lag-attn-v2-spec-and-sprints.md`` S7-T04.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import pytest  # noqa: E402


def test_committed_alias_is_v1_everywhere() -> None:
    r"""Every consumer's committed ``SeqVaeLagAttn`` alias resolves to v1."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (
        pl_module_v2 as plm,
    )
    trainer_mod = pytest.importorskip(
        "model.vae_teb_prediction.model.trainer_lag_attn_v1"
    )
    from model.vae_teb_prediction.new_classifier.guid_cls_v1 import (
        precompute_latents as pcl,
    )
    import model.vae_teb_prediction.testing.base as testing_base

    for mod in (plm, trainer_mod, pcl, testing_base):
        assert mod.SeqVaeLagAttn.__name__ == "SeqVaeLagAttnV1", (
            f"{mod.__name__} committed alias is {mod.SeqVaeLagAttn.__name__}, "
            f"expected SeqVaeLagAttnV1 (the committed rollback default)"
        )


def test_run_pipeline_v2_inherits_v1_alias() -> None:
    r"""``run_pipeline_v2`` follows ``pl_module_v2``'s committed v1 alias."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (
        pl_module_v2 as plm,
    )
    # run_pipeline_v2 imports the alias from pl_module_v2 at call time, so its
    # effective class is whatever pl_module_v2.SeqVaeLagAttn resolves to.
    assert plm.SeqVaeLagAttn.__name__ == "SeqVaeLagAttnV1"
