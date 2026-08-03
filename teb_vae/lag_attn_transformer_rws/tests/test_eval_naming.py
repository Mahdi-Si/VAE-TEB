r"""The names this pipeline is not allowed to use, enforced over a real run's artifacts.

**"Transfer entropy" appears nowhere except in the sentence refusing it.** Under the shipped
``causal_reach_budget_s: null`` the input features at step $t$ read far into their own future, the
reach guard is an energy quantile rather than a support, and no finite budget currently trains --
so ``pred_gap`` and ``source_conditioned_kl_raw`` are a coupling readout and not the quantity the
refusal names. **Replacing both history encoders changed none of that**: the leak is a property of
the shared two-sided feature bank, so this architecture inherits the refusal whole, and the one
thing that would make it easy to lose is a fresh set of documents written for a new package.

The refusal is worth nothing if it holds in the source and then leaks into a CSV column, a JSON key
or a figure caption, which is where a reader actually meets it. So the scan is over the **artifact
tree a real run produced**, plus the two documents that explain it, rather than over the package --
with the run's own disclosure sentence removed first, because that sentence is the refusal and every
run is required to carry it.

The sibling pipeline's ``te_lag`` directory name is scanned for too: it is the obvious thing to copy
when porting an analysis across, and it carries the same claim in three letters.

**The caption scan covers this package's own modules only.** The shared package's string literals
are walked by its own suite, at the same tolerance and by the same function, and a second walk here
would test one function twice. What is *not* covered there is everything local -- the
``encoder_attention`` analysis's panel titles, axis labels and column names -- which is exactly the
new surface a new readout adds.

**No headline number may be a floored KL**, and here that has to be checked on **two** registries:
the shared one, and the scalars this model's binding appends to it. An entry appended by a binding
reaches the same headline block and would be read the same way.

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

from teb_vae.lag_attn_rws.eval import preflight, report_seam
from teb_vae.lag_attn_transformer_rws.eval import run as trf_run
from teb_vae.lag_attn_transformer_rws.eval.binding import TRF_BINDING

#: Case-insensitive strings no artifact and no document may contain.
FORBIDDEN_STRINGS = ("transfer entropy", "te_lag")

#: Suffixes worth scanning. The parquet table is binary and its *columns* are covered by the
#: per-sample CSV, whose header carries the same names. ``.pdf`` is deliberately absent and adding
#: it would buy nothing: a figure's text is drawn into a Flate-compressed content stream and split
#: on its kerning, so the string is not contiguous in the file's bytes or in the inflated stream
#: either, and no PDF text extractor is a dependency here. Caption text is covered at its source
#: instead, by :func:`test_no_figure_caption_names_the_quantity_this_readout_is_not`.
TEXT_SUFFIXES = (".json", ".csv", ".md", ".yaml", ".yml", ".txt", ".log")

#: **This** package, whose string literals become figure titles, axis labels and legend entries.
_EVAL_PACKAGE = Path(trf_run.__file__).parent

#: The documents that explain the artifacts.
DOCUMENTS = ("EVAL.md", "FIGURE_GUIDE.md")

#: A key of this shape is a rescaled block score, and must ship beside a statement saying so.
_SAMPLE_KEY = re.compile(r"nll_\w+_sample")

#: The caveat such a key must be accompanied by, matched loosely on its load-bearing words.
_SAMPLE_CAVEAT = re.compile(r"not a mean over unmasked|fixed /?480 rescale", re.IGNORECASE)


def _scan(text: str) -> List[str]:
    """Return the forbidden strings present in ``text``, case-insensitively.

    The run's own causality disclosure is removed first. That sentence says the readout is *not*
    the quantity it names and every run is required to carry it, so it is the one occurrence that
    is the opposite of a violation; anything left after it is removed is a claim.
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
    assert any(path.name == trf_run.SUMMARY_FILENAME for path in files)


def test_the_scanned_tree_includes_this_models_own_analysis(evaluated) -> None:
    """The half of the tree the sibling's identical scan can never reach. If the cap that enables
    it were dropped from the fixture, ``encoder_attention`` would record a skip, write no tables,
    and this file would pass while scanning none of the surface it was written for."""
    names = {path.name for path in _artifact_files(Path(evaluated["results_dir"]))}

    assert "encoder_attention_entropy.csv" in names
    assert "encoder_attention_reach.csv" in names
    assert evaluated["summary"]["results"]["encoder_attention"].get("skipped") is not True


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
    """Non-vacuity: the walk must actually reach the module that titles this package's own figures,
    and must return the titles rather than only their docstrings."""
    titles = source_string_literals(_EVAL_PACKAGE / "analyses" / "encoder_attention.py")

    assert any("attention" in literal.lower() for literal in titles)
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
        f"the coupling readout is not the quantity the refusal names under the shipped "
        f"configuration, and no artifact may say otherwise: {offenders}"
    )


def test_every_run_carries_the_refusal_verbatim(evaluated) -> None:
    """The scan above removes the disclosure before looking, so it would pass on a run that never
    carried one. Both files an operator opens must state it, character for character."""
    results_dir = Path(evaluated["results_dir"])
    preflight_record = json.loads(
        (results_dir / preflight.PREFLIGHT_FILENAME).read_text(encoding="utf-8")
    )

    assert evaluated["summary"]["causality"]["statement"] == preflight.NOT_CAUSAL_STATEMENT
    assert preflight_record["causality"]["statement"] == preflight.NOT_CAUSAL_STATEMENT


def test_the_documents_that_explain_the_artifacts_are_scanned_too() -> None:
    eval_root = Path(trf_run.__file__).resolve().parent
    for name in DOCUMENTS:
        path = eval_root / name
        assert path.is_file(), f"{name} is part of the contract and must ship"
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
def floored_kl_entries(scalars) -> List:
    """Return the headline entries whose path resolves to a free-bits-floored KL."""
    return [
        (name, path) for name, path in scalars
        if any("kl_train" in part or part.endswith("_train") for part in path)
    ]


def test_no_headline_path_resolves_to_a_floored_kl() -> None:
    """Both registries this model's headline block is assembled from. The binding's additions land
    in the same block and are read the same way, so exempting them would leave the guard checking
    the half that never changes."""
    floored = floored_kl_entries(
        (*report_seam.HEADLINE_SCALARS, *TRF_BINDING.headline_scalars)
    )

    assert floored == [], (
        "only the unfloored KL may be read as a rate: free bits are applied per dimension per "
        "step before summing, so the floored value exceeds it by construction"
    )


def test_a_headline_pointed_at_the_floored_kl_would_be_caught() -> None:
    """The guard above passes on an empty registry; this is what says it discriminates. Exercised
    against a synthetic entry rather than by monkeypatching either registry, so the check is proved
    on exactly the tuple shape a binding contributes."""
    planted = (("kl", ("readouts", "source_conditioned_kl_train")),)

    assert floored_kl_entries(planted) == list(planted)


def test_the_headline_says_which_pred_gap_it_carries(evaluated) -> None:
    headline = evaluated["summary"]["results"]["headline"]

    assert "pred_gap_mc_nats" in headline and "pred_gap_train_path_nats" in headline
    assert "pred_gap" not in headline, "an unqualified name leaves a reader to guess"
    assert "marginalised" in headline["pred_gap_convention"]


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
