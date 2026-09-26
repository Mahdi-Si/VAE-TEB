r"""The analysis protocol's return value, and the offline re-run the protocol exists to make possible.

An analysis reads the tables the shared collection pass wrote, never the model, so a run with
**no checkpoint at all** against a finished directory produces analysis output. That is what makes
re-running one analysis after a long pass cost seconds rather than hours, and it is proved with the
model's ``forward`` rigged to raise -- a spy is the only way to tell "did not need the model" from
"happened not to use it". That every shipped analysis runs cleanly under the protocol is asserted
on a real run in ``test_eval_run.py``, where each step must finish ``ok``.

The protocol definition itself stays importable without ``torch``, which is checked on its import
graph.
"""
from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from teb_vae.lag_attn_rws.eval import run as run_module
from teb_vae.lag_attn_rws.eval.analyses import REQUIRED_RESULT_KEYS, AnalysisContext

#: The directory holding the shipped analyses.
ANALYSES_ROOT = Path(run_module.__file__).resolve().parent / "analyses"


# =============================================================================
# The return value
# =============================================================================
def test_the_unskippable_analysis_returns_the_protocol_keys(
    multi_class_config, multi_class_shards, tmp_path
) -> None:
    context = AnalysisContext(collection=None, config=multi_class_config)

    result = run_module.UNSKIPPABLE_ANALYSES["band_partition"](
        context, eval_config={}, output_dir=tmp_path, probe=None
    )

    assert set(result) >= set(REQUIRED_RESULT_KEYS)
    # None rather than zero: this analysis scores no segments, and a zero would enter the coverage
    # block's population comparison as a disagreement with every analysis that does.
    assert result["n_samples"] is None
    assert result["plan"]["capped"] is False


# =============================================================================
# The offline re-run
# =============================================================================
def _table_only_analysis(context, *, eval_config, output_dir, probe):
    """Read the collected table and write a CSV, touching no model.

    Deliberately reads ``per_sample`` off the context rather than being handed numbers: the point
    of the offline path is that the table is enough.
    """
    frame = context.collection.per_sample
    path = Path(output_dir) / "table_only.csv"
    frame[["sample_index", "guid"]].to_csv(path, index=False)
    return {
        "n_samples": int(len(frame)),
        "composition": {"n_recordings": int(frame["guid"].nunique())},
        "plan": {"capped": False},
        "rows_written": int(len(frame)),
    }


def test_a_table_only_analysis_runs_against_a_finished_directory_with_no_model(
    evaluated, tmp_path, monkeypatch
) -> None:
    """The reason collection and emission are separate steps at all.

    The finished run is copied rather than re-entered so this pass cannot disturb the fixture
    every other file in the suite questions.
    """
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

    run_dir = tmp_path / "rerun"
    shutil.copytree(evaluated["results_dir"].parent, run_dir)

    def _explode(*args, **kwargs):
        raise AssertionError("the model was built and forwarded on an offline re-run")

    monkeypatch.setattr(SeqVaeLagAttnRws, "forward", _explode)
    monkeypatch.setattr(run_module, "ANALYSIS_FUNCTIONS", {"table_only": _table_only_analysis})

    exit_code = run_module.main(None, run_dir, only="table_only", device="cpu")

    results_dir = run_dir / run_module.RESULTS_DIRNAME
    summary = json.loads((results_dir / run_module.SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert exit_code == 0
    assert summary["checkpoint"] is None
    assert summary["analyses_selected"] == ["table_only"]
    assert summary["results"]["table_only"]["rows_written"] == len(
        pd.read_csv(results_dir / "table_only.csv")
    )
    # The readouts of the run being re-read are still there: an offline pass reports the same
    # findings as the pass that collected them, plus its own.
    assert summary["results"]["readouts"] == evaluated["summary"]["results"]["readouts"]


def test_an_offline_run_without_tables_says_what_is_missing(tmp_path) -> None:
    """A directory that is not a finished run must name the two ways out, not fail obscurely."""
    with pytest.raises(FileNotFoundError, match="--checkpoint is required"):
        run_module.main(None, tmp_path / "empty")


def test_the_protocol_module_imports_nothing_from_the_model() -> None:
    """The protocol definition itself must stay importable without ``torch``."""
    source = (ANALYSES_ROOT / "__init__.py").read_text(encoding="utf-8")
    imported = [
        node.module or ""
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    ] + [
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    ]

    assert all(
        not name.startswith(("torch", "lightning", "teb_vae.lag_attn_rws.nets"))
        for name in imported
    ), imported
