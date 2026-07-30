r"""The three names this pipeline is not allowed to use, enforced over a real run's artifacts.

**"Transfer entropy" appears nowhere except in the sentence refusing it.** Under the shipped
``causal_reach_budget_s: null`` the input features at step $t$ read far into their own future, the
reach guard is an energy quantile rather than a support, and no finite budget currently trains --
so ``pred_gap`` and ``source_conditioned_kl_raw`` are a coupling readout and not a transfer
entropy. The refusal is worth nothing if it holds in the source and then leaks into a CSV column,
a JSON key or a figure caption, which is where a reader actually meets it. So the scan is over the
**artifact tree a real run produced**, plus the two documents that explain it, rather than over
the package -- with the run's own disclosure sentence removed first, because that sentence is the
refusal and every run is required to carry it.

The sibling pipeline's ``te_lag`` directory name is scanned for too: it is the obvious thing to
copy when porting an analysis across, and it carries the same claim in three letters.

**No headline number may be a floored KL.** ``source_conditioned_kl_train`` has free bits applied
per dimension per step before summing, so it exceeds the raw value by construction and hides a
collapsed source pathway. The shipped ``free_bits: 0.0`` makes the two coincide today, which is
exactly why the distinction has to be enforced rather than observed.

**No ``nll_*_sample`` key without its caveat.** Such a key is a fixed $/480$ rescale of a block
score, not a mean over unmasked samples, so on any anchor with masked forecast steps it
under-reports -- and it reads exactly like a per-sample mean.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import List

import pytest

from teb_vae.lag_attn_rws.eval import preflight, report_seam
from teb_vae.lag_attn_rws.eval import run as run_module

#: Case-insensitive strings no artifact and no document may contain.
FORBIDDEN_STRINGS = ("transfer entropy", "te_lag")

#: Suffixes worth scanning. The parquet table is binary and its *columns* are covered by the
#: per-sample CSV, whose header carries the same names. ``.pdf`` is deliberately absent and adding
#: it would buy nothing: a figure's text is drawn into a Flate-compressed content stream and split
#: on its kerning, so the string is not contiguous in the file's bytes or in the inflated stream
#: either, and no PDF text extractor is a dependency here. Caption text is covered at its source
#: instead, by :func:`test_no_figure_caption_names_the_quantity_this_readout_is_not`.
TEXT_SUFFIXES = (".json", ".csv", ".md", ".yaml", ".yml", ".txt", ".log")

#: The package whose string literals become figure titles, axis labels and legend entries.
_EVAL_PACKAGE = Path(run_module.__file__).parent

#: The documents that explain the artifacts. Absent until the documentation sprint lands, and
#: scanned as soon as they exist rather than being added to this list then.
DOCUMENTS = ("EVAL.md", "FIGURE_GUIDE.md")

#: A key of this shape is a rescaled block score, and must ship beside a statement saying so.
_SAMPLE_KEY = re.compile(r"nll_\w+_sample")

#: The caveat such a key must be accompanied by, matched loosely on its load-bearing words.
_SAMPLE_CAVEAT = re.compile(r"not a mean over unmasked|fixed /?480 rescale", re.IGNORECASE)


def _scan(text: str) -> List[str]:
    """Return the forbidden strings present in ``text``, case-insensitively.

    The run's own causality disclosure is removed first. That sentence says the readout is *not* a
    transfer entropy and every run is required to carry it, so it is the one occurrence that is
    the opposite of a violation; anything left after it is removed is a claim.
    """
    normalised = re.sub(r"\s+", " ", text)
    disclosure = re.sub(r"\s+", " ", preflight.NOT_CAUSAL_STATEMENT)
    lowered = normalised.replace(disclosure, "").lower()
    return [name for name in FORBIDDEN_STRINGS if name in lowered]


def _artifact_files(results_dir: Path) -> List[Path]:
    """Every text artifact a run left behind."""
    return sorted(
        path for path in results_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
    )


# =============================================================================
# The artifact tree
# =============================================================================
def test_the_run_produced_artifacts_to_scan(evaluated) -> None:
    """A scan over an empty tree would pass every assertion below vacuously."""
    files = _artifact_files(Path(evaluated["results_dir"]))

    assert len(files) >= 4, [path.name for path in files]
    assert any(path.name == run_module.SUMMARY_FILENAME for path in files)


def source_string_literals(path: Path) -> List[str]:
    """Return every non-docstring string constant in one module.

    Docstrings are excluded because they *explain* the refusal, and a module is allowed to name
    what it is refusing. What is not allowed is a string the reader meets in an artifact -- a
    figure title, an axis label, a legend entry, a column name -- and those are ordinary constants.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # By node identity, not by text: ``ast.get_docstring`` returns the *cleaned* string, which
    # never equals the raw literal on any indented or multi-line docstring -- so a text comparison
    # excludes almost nothing and the walk reports every module docstring as a caption.
    docstring_nodes = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    return [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstring_nodes
    ]


def test_no_figure_caption_names_the_quantity_this_readout_is_not() -> None:
    """The one leak route the artifact scan cannot reach.

    A figure title is a hand-written literal, and the rendered PDF does not carry it as findable
    text -- so the caption is checked where it is written instead. This also covers axis labels,
    legend entries and any other string a reader meets on a page.
    """
    offenders = {
        f"{path.relative_to(_EVAL_PACKAGE)}: {literal}": found
        for path in sorted(_EVAL_PACKAGE.rglob("*.py"))
        for literal in source_string_literals(path)
        for found in [_scan(literal)]
        if found
    }

    assert offenders == {}, (
        f"a figure caption or label is where a reader actually meets the name this readout is "
        f"refused: {offenders}"
    )


def test_the_literal_scan_reads_the_captions_it_claims_to() -> None:
    """Non-vacuity: the walk must actually reach the modules that title the figures, and must
    return the titles rather than only their docstrings."""
    titles = source_string_literals(_EVAL_PACKAGE / "analyses" / "lag_kl.py")

    assert any("lag" in literal.lower() for literal in titles)
    assert _scan("Transfer entropy per lag") == ["transfer entropy"], (
        "the scan must flag a caption that names the refused quantity"
    )


def test_no_artifact_names_the_quantity_this_readout_is_not(evaluated) -> None:
    offenders = {
        path.name: found
        for path in _artifact_files(Path(evaluated["results_dir"]))
        for found in [_scan(path.read_text(encoding="utf-8", errors="replace"))]
        if found
    }

    assert offenders == {}, (
        f"the coupling readout is not a transfer entropy under the shipped configuration, and "
        f"no artifact may say otherwise: {offenders}"
    )


def test_the_documents_that_explain_the_artifacts_are_scanned_too() -> None:
    eval_root = Path(run_module.__file__).resolve().parent
    for name in DOCUMENTS:
        path = eval_root / name
        if not path.is_file():
            continue
        assert _scan(path.read_text(encoding="utf-8", errors="replace")) == [], name


def test_the_scan_would_catch_a_violation(tmp_path) -> None:
    """Non-vacuity: every assertion above passes on an empty file, and this is what says the
    scanner is looking at all."""
    planted = tmp_path / "planted.json"
    planted.write_text(json.dumps({"note": "the Transfer Entropy per lag"}), encoding="utf-8")

    assert _scan(planted.read_text(encoding="utf-8")) == ["transfer entropy"]
    assert _scan("te_lag_map") == ["te_lag"]


# =============================================================================
# The headline names which quantity it carries
# =============================================================================
def test_no_headline_path_resolves_to_a_floored_kl() -> None:
    floored = [
        (name, path) for name, path in report_seam.HEADLINE_SCALARS
        if any("kl_train" in part or part.endswith("_train") for part in path)
    ]

    assert floored == [], (
        "only the unfloored KL may be read as a rate: free bits are applied per dimension per "
        "step before summing, so the floored value exceeds it by construction"
    )


def test_the_headline_says_which_pred_gap_it_carries(evaluated) -> None:
    headline = evaluated["summary"]["results"]["headline"]

    assert "pred_gap_mc_nats" in headline and "pred_gap_train_path_nats" in headline
    assert "pred_gap" not in headline, "an unqualified name leaves a reader to guess"
    assert "marginalised" in headline["pred_gap_convention"]


def test_a_headline_pointed_at_the_floored_kl_would_be_caught(monkeypatch) -> None:
    """The guard above passes on an empty registry; this is what says it discriminates."""
    monkeypatch.setattr(
        report_seam,
        "HEADLINE_SCALARS",
        (("kl", ("readouts", "source_conditioned_kl_train")),),
    )

    with pytest.raises(AssertionError):
        test_no_headline_path_resolves_to_a_floored_kl()


# =============================================================================
# The rescaled per-sample score
# =============================================================================
def _keys_of(node, found=None):
    """Collect every dict key in a nested JSON structure."""
    found = [] if found is None else found
    if isinstance(node, dict):
        for key, value in node.items():
            found.append(str(key))
            _keys_of(value, found)
    elif isinstance(node, list):
        for item in node:
            _keys_of(item, found)
    return found


def test_a_rescaled_per_sample_score_ships_with_its_caveat_or_not_at_all(evaluated) -> None:
    summary = evaluated["summary"]
    offending = sorted({key for key in _keys_of(summary) if _SAMPLE_KEY.fullmatch(key)})

    if not offending:
        return
    text = evaluated["text"]
    assert _SAMPLE_CAVEAT.search(text), (
        f"{offending} is a fixed /480 rescale of a block score rather than a mean over unmasked "
        f"samples, and under-reports on any anchor with masked forecast steps"
    )
