r"""The two thin entry points, and the one property that makes them worth having.

``run.py`` and ``verify.py`` exist so that this cell can be launched and gated by name. What they
must **not** become is a second implementation: the two cfs cells exist to be compared, and a second
copy of the runner or of the gate's criteria is how two things that must stay comparable stop being
comparable -- the first fix to an analysis or a threshold lands on one side, and the two
``summary.json`` files quietly stop meaning the same thing.

So the assertions here are about delegation rather than about numbers: the runner hands the cfs
cell's runner this cell's binding and nothing else, the registry is re-derived on every call rather
than frozen at import, the gate imports no numeric stack, and the arm tables carry this cell's one
sweep axis beside the shared cross-cell comparison.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from teb_vae.lag_attn_cfs.eval import run as shared_run
from teb_vae.lag_attn_cfs.eval import verify as shared_verify
from teb_vae.lag_attn_cfs.eval.binding import CFS_BINDING
from teb_vae.lag_attn_transformer_cfs.eval import run as run_module
from teb_vae.lag_attn_transformer_cfs.eval import verify as verify_module
from teb_vae.lag_attn_transformer_cfs.eval.binding import TRF_CFS_BINDING

from .conftest import _REPO_ROOT


# =================================================================================================
# Delegation
# =================================================================================================
def test_the_runner_supplies_this_cells_binding_and_a_caller_may_override_it(monkeypatch) -> None:
    """``main`` adds one keyword and hands everything else on. Asserted by intercepting the shared
    runner rather than by running one, because what is being checked is which binding arrives.

    ``setdefault`` rather than an assignment: the offline re-run tests drive this entry point with
    another binding to prove no model is built, and an assignment would silently ignore them."""
    seen: Dict[str, Any] = {}

    def _capture(*args: Any, **kwargs: Any) -> int:
        seen.clear()
        seen.update(kwargs)
        seen["args"] = args
        return 0

    monkeypatch.setattr(shared_run, "main", _capture)

    assert run_module.main("ckpt.ckpt", "out", device="cpu") == 0
    assert seen["binding"] is TRF_CFS_BINDING
    assert seen["args"] == ("ckpt.ckpt", "out")
    assert seen["device"] == "cpu"

    run_module.main("ckpt.ckpt", binding=CFS_BINDING)
    assert seen["binding"] is CFS_BINDING


def test_the_registry_is_derived_on_every_call_rather_than_frozen_at_import(monkeypatch) -> None:
    """The help text, the selection ``main`` makes and the ``summary.json`` record must all read
    one mapping. Frozen at import, an analysis registered on the binding would reach the run and
    not the help text, or the reverse -- and nothing in the artifact would say which."""
    extended = dict(shared_run.ANALYSIS_FUNCTIONS)
    extended["a_new_shared_analysis"] = lambda *a, **k: None
    monkeypatch.setattr(shared_run, "ANALYSIS_FUNCTIONS", extended)

    assert "a_new_shared_analysis" in run_module.analysis_registry()


# =================================================================================================
# The gate's one non-negotiable property
# =================================================================================================
def test_importing_this_cells_gate_pulls_in_no_numeric_stack() -> None:
    """Run in a subprocess: this session has already imported ``torch``, so an in-process check
    would pass no matter what the module does. A summary produced on the production box has to be
    checkable on a machine that has never had a deep-learning stack on it, and that has to hold for
    the entry point an operator actually types."""
    source = (
        "import sys\n"
        "import teb_vae.lag_attn_transformer_cfs.eval.verify as gate\n"
        "leaked = sorted(name for name in sys.modules if name.split('.')[0] "
        "in {'torch', 'lightning', 'numpy', 'scipy', 'h5py', 'pandas', 'matplotlib'})\n"
        "assert gate.PRED_GAP_COLUMN\n"
        "print(','.join(leaked))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "", (
        f"importing this cell's gate pulled in {completed.stdout.strip()}"
    )


# =================================================================================================
# The tables
# =================================================================================================
def _write_arm(root: Path, name: str, *, model_class: str, anchor_stride: int = 15) -> None:
    """Write one finished-run shape under ``root``, through the cfs suite's own writer."""
    from teb_vae.lag_attn_cfs.tests.test_eval_verify import write_arm

    write_arm(root, name, model_class=model_class, anchor_stride=anchor_stride)


def test_the_tables_carry_this_cells_one_axis_and_the_cross_cell_comparison(tmp_path) -> None:
    """One sweep section rather than the cfs cell's four: this package ships one ``sweep_*.yaml``
    arm, and a section whose every row read ``(absent)`` would print a sweep nobody ran as though
    somebody had. The cross-cell table is the shared one, so the two cells' rows are assembled the
    same way."""
    _write_arm(tmp_path, "trf_dense", model_class="SeqVaeLagAttnTrfCfs", anchor_stride=1)
    _write_arm(tmp_path, "trf_tiled", model_class="SeqVaeLagAttnTrfCfs", anchor_stride=15)
    _write_arm(tmp_path, "cfs_tiled", model_class="SeqVaeLagAttnCfs", anchor_stride=15)
    out = tmp_path / "arms.md"

    assert verify_module.compare_arms(tmp_path, out) == 0
    document = out.read_text(encoding="utf-8")

    headings: List[str] = [line for line in document.splitlines() if line.startswith("## ")]
    assert headings == [
        "## Arm inventory",
        "## Anchor tiling sweep (`anchor_stride`)",
        "## Cross-cell comparison",
    ], headings
    # Both cells' rows in the cross-cell table, keyed on the class each run recorded.
    assert document.count("SeqVaeLagAttnTrfCfs") >= 2
    assert "SeqVaeLagAttnCfs" in document
    assert shared_verify.SELECTION_RULE in document


def test_the_gate_and_the_tables_dispatch_from_one_command_line(tmp_path) -> None:
    import json

    from teb_vae.lag_attn_cfs.tests.test_eval_verify import clean_summary

    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(clean_summary()), encoding="utf-8")

    assert verify_module._cli([str(summary)]) == 0
    assert verify_module._cli([str(summary), "--runs", str(tmp_path)]) == 2  # both is a usage error
    assert verify_module._cli([]) == 2  # neither is too
