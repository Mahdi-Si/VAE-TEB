r"""The evaluation contract of a package that delegates one, bound to the code that delegates it.

``eval/EVAL.md`` here is short on purpose: the analyses, the configuration reference, the guard
recovery table and the interpretation rules live in ``teb_vae/lag_attn_cfs/eval/EVAL.md`` and are
not restated. What this file checks is the handful of claims that are **this** package's and would
otherwise be prose nobody could falsify:

* every file the document says this package supplies exists, and the launch lines name modules that
  import;
* the document does not document an analysis -- a section here would be a second contract for a
  step the parent already owns, and the two would drift;
* the "defines no numeric function" claim is asserted **about the objects**: every analysis, every
  gate criterion and every headline scalar a run of this cell reports is the cfs cell's own object,
  compared by identity rather than by value, so a local copy fails here even when it is correct;
* and both edges of the comparability rule are stated, **asymmetrically**. That asymmetry is the
  one thing a reader is most likely to flatten -- a level comparison against the sibling cfs cell is
  legitimate and a level comparison against the two-sided transformer cell is not -- and a document
  that stated only one half would read as permission for both.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from teb_vae.lag_attn_cfs.eval import run as shared_run
from teb_vae.lag_attn_cfs.eval import verify as shared_verify
from teb_vae.lag_attn_cfs.eval.binding import CFS_BINDING
from teb_vae.lag_attn_transformer_cfs.eval import run as run_module
from teb_vae.lag_attn_transformer_cfs.eval import verify as verify_module
from teb_vae.lag_attn_transformer_cfs.eval.binding import TRF_CFS_BINDING

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_EVAL_ROOT = _PACKAGE_DIR / "eval"
_EVAL_DOC = _EVAL_ROOT / "EVAL.md"


@pytest.fixture(scope="module")
def eval_doc() -> str:
    return _EVAL_DOC.read_text(encoding="utf-8")


# =================================================================================================
# The document describes files that exist
# =================================================================================================
def test_the_document_exists_and_is_short_by_design(eval_doc) -> None:
    """Both directions. A stub would satisfy every "does it say X" assertion below by accident once
    the phrases were removed with it; and a document that grew into a second copy of the parent's
    contract is the drift this package's whole shape exists to avoid."""
    assert len(eval_doc) > 1500, "EVAL.md is a stub"
    assert len(eval_doc) < 12000, (
        "EVAL.md has grown into a second contract; the analyses, the configuration reference and "
        "the interpretation rules belong to teb_vae/lag_attn_cfs/eval/EVAL.md and are delegated"
    )


def test_the_document_defers_to_the_parents_contract_by_path(eval_doc) -> None:
    """The deferral is the document's first claim, so the path it names is load-bearing: a moved
    file turns it into a dead end and the reader who followed it cannot tell a missing contract from
    an unwritten one."""
    referenced = sorted(set(re.findall(r"teb_vae/[\w/]+\.md", eval_doc)))

    assert "teb_vae/lag_attn_cfs/eval/EVAL.md" in referenced
    repo_root = _PACKAGE_DIR.parents[1]
    missing = [path for path in referenced if not (repo_root / path).is_file()]
    assert missing == [], f"EVAL.md defers to documents that do not exist: {missing}"


def test_every_file_the_document_claims_this_package_supplies_exists(eval_doc) -> None:
    """The table is the answer to "what is local here", and it is the first thing read by someone
    deciding where to make a change. A row for a file that is not there sends them to the wrong
    package."""
    for name in ("binding.py", "configs/eval_overrides.yaml", "run.py", "verify.py"):
        assert f"`{name}`" in eval_doc, name
        assert (_EVAL_ROOT / name).is_file(), name


def test_every_launch_line_names_a_module_that_exists(eval_doc) -> None:
    """A launch line is copied and pasted; one naming a moved module fails at the shell with a
    message about an import rather than about a run."""
    modules = sorted(set(re.findall(r"python -m ([\w.]+)", eval_doc)))

    assert len(modules) >= 2, f"EVAL.md names only {modules}"
    repo_root = _PACKAGE_DIR.parents[1]
    for dotted in modules:
        assert (repo_root / Path(*dotted.split("."))).with_suffix(".py").is_file(), dotted
        assert dotted.startswith("teb_vae.lag_attn_transformer_cfs.eval"), dotted


def test_the_document_documents_no_analysis(eval_doc) -> None:
    """A ``### <analysis>`` section here would be a second contract for a step the parent owns, and
    the two would drift in the direction nobody checks -- the parent's is the one bound by test."""
    headings = set(re.findall(r"^###\s+(\S+)\s*$", eval_doc, flags=re.MULTILINE))
    registry = set(run_module.analysis_registry()) | set(run_module.UNSKIPPABLE_ANALYSES)

    assert headings & registry == set(), sorted(headings & registry)


# =================================================================================================
# "Defines no numeric function", asserted about the objects
# =================================================================================================
def test_every_analysis_this_cell_runs_is_the_parents_own_object(eval_doc) -> None:
    """By identity, not by value. A local re-implementation that happened to agree today would pass
    a value comparison and would be exactly the drift the delegation exists to prevent: the first
    fix lands on one side and the two summaries quietly stop meaning the same thing."""
    assert "defines no numeric function" in eval_doc

    here = run_module.analysis_registry()
    there = shared_run.merged_analysis_functions(CFS_BINDING)

    assert list(here) == list(there)
    for name, function in here.items():
        assert function is there[name], name


def test_the_gate_is_the_parents_gate_rather_than_a_second_criterion_set() -> None:
    """The gate decides whether a checkpoint is acceptable, so a criterion re-implemented here would
    let one cell pass under a rule the other was never held to. Asserted as an *absence* plus a
    delegation: this module defines no criteria of its own, and its ``main`` is the parent's."""
    assert not hasattr(verify_module, "CRITERIA"), (
        "verify.py has grown a criteria registry of its own; the two cfs cells are gated by one "
        "set or they are not comparable"
    )
    assert not hasattr(verify_module, "CFS_VERDICTS")
    assert len(shared_verify.CRITERIA) >= 10
    source = Path(verify_module.__file__).read_text(encoding="utf-8")
    assert "return shared.main(" in source


def test_the_headline_registry_and_the_extra_analyses_are_the_parents_objects() -> None:
    """The binding is a record of declarations, and these two fields are the ones that could
    silently become copies: a copied tuple keeps working while the parent's moves."""
    assert TRF_CFS_BINDING.extra_analyses is CFS_BINDING.extra_analyses
    assert TRF_CFS_BINDING.headline_scalars is CFS_BINDING.headline_scalars


def test_the_one_declaration_that_differs_is_stated_with_its_arithmetic(eval_doc) -> None:
    """``geometry_keys`` is the only field whose value is this package's own, and the document
    states how it is reached rather than only what it is -- because the rule it obeys (both a
    constructor parameter and a config key, or the reconciliation silently never happens) is what
    makes a wrong entry invisible instead of loud."""
    assert len(TRF_CFS_BINDING.geometry_keys) == 22
    assert "causal_norm" not in TRF_CFS_BINDING.geometry_keys
    assert "Twenty-two" in eval_doc
    assert "`causal_norm`" in eval_doc
    assert "silently skips" in eval_doc
    for key in (
        "encoder_conv_kernels",
        "encoder_conv_dilations",
        "encoder_num_heads",
        "encoder_d_ff",
        "target_attention_blocks",
        "source_attention_blocks",
        "source_attention_window",
    ):
        assert key in TRF_CFS_BINDING.geometry_keys, key
        assert f"`{key}`" in eval_doc, key


# =================================================================================================
# The comparability rule, stated in both directions and asymmetrically
# =================================================================================================
def test_both_edges_are_stated_and_they_are_not_the_same_statement(eval_doc) -> None:
    """The encoder edge is comparable on a loss level and the target edge is not. A document that
    stated one half would read as permission for both, which is the single most likely misuse of a
    cross-cell table."""
    assert "lag_attn_cfs" in eval_doc
    assert "lag_attn_transformer_fs" in eval_doc

    encoder_edge = eval_doc[eval_doc.index("Against `lag_attn_cfs`"):]
    encoder_edge = encoder_edge[:encoder_edge.index("Against `lag_attn_transformer_fs`")]
    target_edge = eval_doc[eval_doc.index("Against `lag_attn_transformer_fs`"):]

    assert "is* comparable" in encoder_edge
    assert "1470" in encoder_edge and "152" in encoder_edge
    assert "not* comparable" in target_edge
    assert "2340" in target_edge and "30" in target_edge


def test_the_asymmetry_is_given_its_reason_rather_than_asserted(eval_doc) -> None:
    """"Not comparable" without the reason is a rule that gets waived by whoever most wants the
    comparison. The reason is that a block score's scale is set by the block before the model is
    reached, and it is what makes the ban structural rather than cautious.

    Matched against the document with its line wrapping collapsed, because a phrase that happens to
    straddle a line break is a reflow rather than a lost claim.
    """
    flat = re.sub(r"\s+", " ", eval_doc)

    assert "the same objective" in flat
    assert "scale is set by the block" in flat
    assert "signs-and-orderings" in flat


def test_the_budget_local_rule_travels_here_too(eval_doc) -> None:
    """The same trap one step further out: two arms of *this* cell at two warm-up budgets are not
    comparable either, and a reader who took the encoder edge as licence would meet it next."""
    assert "budget-local" in eval_doc
    assert "C_{\\mathrm{keep}}" in eval_doc
