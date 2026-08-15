r"""The four names this pipeline is not allowed to use, enforced over a real run's artifacts.

The target here is **98 wavelet-modulus and phase-harmonic coefficients in the loader's $z$ units**,
and three of the words the raw-target pipeline uses freely describe something this domain does not
have. A refusal that holds in a design document and then leaks into a CSV column, a JSON key or a
figure caption is worth nothing, because the artifact is where a reader actually meets it. So the
scan is over the **artifact tree a real run produced**, plus the two documents that explain it, plus
the string literals that become titles and axis labels.

* **``bpm``** -- there is no clinical unit here. A scattering coefficient is an envelope and a
  phase-harmonic coefficient is a product of two of them; inverting the per-channel statistics would
  put the 98 scored channels on scales spanning orders of magnitude.
* **``raw sample`` / ``raw samples``** -- the block's elements are coefficients on a 4-second grid,
  not elements of a 4 Hz trace, and the two differ by a factor of sixteen in count and entirely in
  meaning.
* **``waveform``** -- the forecast is an $H \times C_{\mathrm{keep}}$ block over a channel axis with
  no order, not a trace that can be drawn as one line.
* **``transfer entropy``** -- the lag map is an attribution over stored-coefficient time, uncorrected
  for a composed one-sided group delay of up to 791 s, which is the same order as the lag search
  itself.

**Two allow-lists, and both are asserted explicitly rather than left implicit**, because the next
person to tighten this scan needs to see why it is loose where it is.

The first is the bare token ``raw``. It is inherited metric vocabulary meaning *unfloored* rather
than *raw-signal*: ``source_conditioned_kl_raw`` is the only KL that may be read as a rate,
``raw_logvar_prior`` is the pre-bound tensor the posterior residual is applied to, and
``mu_prior_sat_frac_raw`` is the unmasked framing. A scan that banned the substring would fail on the
three readouts the pipeline most depends on and would be "fixed" by renaming them, which would break
comparability with all five sibling cells for no gain. So every banned term is matched as a **phrase
with word boundaries**, and the three readouts are driven through the scanner as a negative control.

The second is the set of strings that exist **to refuse a term**. The causality disclosure says the
readout is not a transfer entropy; the ``pred_gap`` convention says there is no clinical unit here;
the removed-readout record says the two absent ``events`` readouts scored one. Each is a fixed
constant this package emits deliberately, so each is removed by exact match before the scan, and
anything left afterwards is a claim rather than a refusal. Removing them by name is what makes "the
one permitted occurrence" mechanical: a second one fails.

``caps.waveforms`` is the one carve-out that is neither. It is a retention **cap name**, shared with
all five sibling cells so a cap set stays portable, and it reaches ``resolved_config.yaml`` and
``collection.json`` as an ordinary key. It is stripped in exactly the two forms a serialiser writes
it in, so the word remains banned everywhere a caption or a column could use it.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Dict, List

import pytest

from teb_vae.lag_attn_cfs.eval import collect, lag_axis, preflight, report_seam
from teb_vae.lag_attn_cfs.eval import run as run_module
from teb_vae.lag_attn_cfs.eval.analyses import events as events_analysis

pytestmark = pytest.mark.slow

#: The banned phrases, each anchored on word boundaries so an identifier that merely contains one
#: -- ``to_bpm``, ``raw_sample_score``, ``BPM_UNIT`` -- is untouched. Keyed by the name a failure
#: message reports.
BANNED_PATTERNS: Dict[str, re.Pattern] = {
    "bpm": re.compile(r"\bbpm\b", re.IGNORECASE),
    "raw sample": re.compile(r"\braw samples?\b", re.IGNORECASE),
    "waveform": re.compile(r"\bwaveforms?\b", re.IGNORECASE),
    "transfer entropy": re.compile(r"\btransfer entropy\b", re.IGNORECASE),
}

#: The retention cap's own key, in the four forms a serialiser or a document writes it in: a JSON
#: key, a YAML key, a backticked name in prose, and the dotted path an operator sets. Stripped
#: before the scan so the *word* stays banned everywhere a caption, a column or a title could use
#: it -- "the forecast waveforms" matches none of these and still fails.
#:
#: The cap keeps its family-wide name on purpose: ``caps`` is one block across all six cells of the
#: grid, and renaming it here to satisfy a scan would make a cap set non-portable to buy nothing.
CAP_KEY_PATTERN = re.compile(r'"waveforms"|`waveforms`|caps\.waveforms|\bwaveforms(?=\s*:)')

#: Suffixes worth scanning. The parquet table is binary and its *columns* are covered by the
#: per-sample CSV, whose header carries the same names. ``.pdf`` is deliberately absent and adding
#: it would buy nothing: a figure's text is drawn into a Flate-compressed content stream and split
#: on its kerning, so the string is not contiguous in the file's bytes or in the inflated stream
#: either. Caption text is covered at its source instead.
TEXT_SUFFIXES = (".json", ".csv", ".md", ".yaml", ".yml", ".txt", ".log")

#: The package whose string literals become figure titles, axis labels and legend entries.
_EVAL_PACKAGE = Path(run_module.__file__).parent

#: The documents that explain the artifacts.
DOCUMENTS = ("EVAL.md", "FIGURE_GUIDE.md")

#: A key of this shape is a rescaled block score, and must ship beside a statement saying so.
_SAMPLE_KEY = re.compile(r"nll_\w+_sample")

#: The caveat such a key must be accompanied by, matched loosely on its load-bearing words.
_SAMPLE_CAVEAT = re.compile(r"not a mean over|fixed /?1470 rescale", re.IGNORECASE)


def permitted_statements() -> List[str]:
    """Return the fixed strings this package emits in order to *refuse* a banned term.

    Read off the constants rather than restated, so a reworded refusal stays permitted and a second
    occurrence of the term anywhere else still fails. Each is a deliberate emission: the causality
    disclosure and the ``pred_gap`` convention travel verbatim into ``summary.json``, and the
    removed-readout reasons travel into the ``events`` block so a reader who expects the raw
    pipeline's three meets the absence rather than inferring it from a missing key.
    """
    return [
        preflight.CAUSALITY_STATEMENT,
        lag_axis.GROUP_DELAY_CAVEAT,
        report_seam.PRED_GAP_CONVENTION,
        *(record["reason"] for record in events_analysis.REMOVED_READOUTS),
    ]


def _scan(text: str) -> List[str]:
    """Return the banned phrases present in ``text``, after the two allow-lists are applied.

    Whitespace is normalised on both sides before the refusal statements are removed, because a
    constant that reaches a document or a JSON string gets re-wrapped and would otherwise no longer
    match itself.
    """
    normalised = re.sub(r"\s+", " ", text)
    for statement in permitted_statements():
        normalised = normalised.replace(re.sub(r"\s+", " ", statement), "")
    normalised = CAP_KEY_PATTERN.sub("", normalised)
    return [name for name, pattern in BANNED_PATTERNS.items() if pattern.search(normalised)]


def _artifact_files(results_dir: Path) -> List[Path]:
    """Every text artifact a run left behind."""
    return sorted(
        path for path in results_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
    )


# =================================================================================================
# The scanner discriminates
# =================================================================================================
def test_the_scan_would_catch_each_banned_phrase(tmp_path) -> None:
    """Non-vacuity: every assertion below passes on an empty file, and this is what says the
    scanner is looking at all. One planted case per banned phrase."""
    planted = tmp_path / "planted.json"
    planted.write_text(
        json.dumps({"note": "the forecast error in BPM over a smoothed waveform"}), encoding="utf-8"
    )

    assert _scan(planted.read_text(encoding="utf-8")) == ["bpm", "waveform"]
    assert _scan("scored per raw sample") == ["raw sample"]
    assert _scan("scored over 480 raw samples") == ["raw sample"]
    assert _scan("this is the transfer entropy of the source") == ["transfer entropy"]


def test_the_inherited_raw_vocabulary_passes(tmp_path) -> None:
    """The negative control that keeps this scan from being tightened into a rename. All three are
    readouts the pipeline depends on, and the bare token ``raw`` in each means *unfloored* or
    *pre-bound*, not *raw-signal*."""
    for name in (
        "source_conditioned_kl_raw",
        "source_conditioned_kl_shuffled_raw",
        "source_conditioned_kl_train",
        "raw_logvar_prior",
        "mu_prior_sat_frac_raw",
        "raw_sample_score",
        "up_raw",
        "raw_per_step",
    ):
        assert _scan(name) == [], name


def test_the_retention_cap_keeps_its_family_wide_name(tmp_path) -> None:
    """``caps.waveforms`` is a cap *name* shared with every sibling cell, so a cap set stays
    portable across the grid; it reaches two artifacts as an ordinary serialised key. Stripped in
    exactly those two forms -- and the word is still banned in prose, which is the half that
    matters."""
    assert _scan('{"caps": {"waveforms": 64, "attention": 64}}') == []
    assert _scan("eval_config:\n  caps:\n    waveforms: 64\n") == []
    # ...and the carve-out does not launder the word itself.
    assert _scan("the forecast waveforms are drawn per channel") == ["waveform"]


def test_the_permitted_statements_are_the_ones_that_refuse(tmp_path) -> None:
    """Each is removed by exact match, so the term it refuses is permitted exactly once. A second
    occurrence anywhere -- including a second copy of the statement itself -- fails."""
    statements = permitted_statements()

    assert len(statements) >= 3
    for statement in statements:
        assert _scan(statement) == [], statement[:60]
    # The disclosure carries the refused name once; a second sentence naming it is a claim.
    assert _scan(preflight.CAUSALITY_STATEMENT) == []
    assert _scan(
        preflight.CAUSALITY_STATEMENT + " The lag map is a transfer entropy."
    ) == ["transfer entropy"]


# =================================================================================================
# The artifact tree
# =================================================================================================
def test_the_run_produced_artifacts_to_scan(collected_run) -> None:
    """A scan over an empty tree would pass every assertion below vacuously."""
    files = _artifact_files(Path(collected_run["results_dir"]))

    assert len(files) >= 4, [path.name for path in files]
    assert any(path.name == run_module.SUMMARY_FILENAME for path in files)


def test_no_artifact_makes_a_claim_this_target_domain_cannot(collected_run) -> None:
    """Every emitted CSV, JSON, YAML, markdown and log file, not a sampled subset."""
    offenders = {
        str(path.relative_to(Path(collected_run["results_dir"]))): found
        for path in _artifact_files(Path(collected_run["results_dir"]))
        for found in [_scan(path.read_text(encoding="utf-8", errors="replace"))]
        if found
    }

    assert offenders == {}, (
        f"this model forecasts wavelet-modulus and phase-harmonic coefficients in the loader's z "
        f"units, and no artifact may say otherwise: {offenders}"
    )


def test_the_documents_that_explain_the_artifacts_are_scanned_too() -> None:
    """A reader who meets a term in ``EVAL.md`` carries it to every number the document explains,
    so the two documents are held to the artifacts' own standard."""
    for name in DOCUMENTS:
        path = _EVAL_PACKAGE / name
        assert path.is_file(), name
        assert _scan(path.read_text(encoding="utf-8", errors="replace")) == [], name


# =================================================================================================
# Figure captions, which the artifact scan cannot reach
# =================================================================================================
def source_string_literals(path: Path) -> List[str]:
    """Return every non-docstring string constant in one module.

    Docstrings are excluded because they *explain* the refusal, and a module is allowed to name what
    it is refusing and why. What is not allowed is a string the reader meets in an artifact -- a
    figure title, an axis label, a legend entry, a column name -- and those are ordinary constants.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # By node identity, not by text: ``ast.get_docstring`` returns the *cleaned* string, which never
    # equals the raw literal on any indented or multi-line docstring -- so a text comparison
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


def _is_permitted_literal(literal: str) -> bool:
    """Whether a source literal is a piece of a permitted statement, or a retention cap's name.

    The refusal statements are assembled from adjacent and interpolated pieces, so the AST walk
    yields *fragments* rather than the finished sentence -- and the fragment carrying "not a
    transfer entropy" is exactly the clause that refuses it. A fragment of a permitted statement is
    therefore permitted, matched as a substring so that rewording the statement keeps its own pieces
    permitted while a new sentence elsewhere is not.

    The cap names are the other case: ``RETAINED_QUANTITIES``' keys are ``eval_config.caps`` keys,
    and one of them is the bare word. A literal is admitted only when it is *exactly* a key, so a
    caption that happens to contain the word is not.
    """
    if literal in collect.RETAINED_QUANTITIES:
        return True
    flat = re.sub(r"\s+", " ", literal).strip()
    if not flat:
        return True
    return any(flat in re.sub(r"\s+", " ", statement) for statement in permitted_statements())


def test_no_figure_caption_names_a_quantity_this_domain_does_not_have() -> None:
    """The one leak route the artifact scan cannot reach.

    A figure title is a hand-written literal and the rendered PDF does not carry it as findable
    text, so the caption is checked where it is written. This also covers axis labels, legend
    entries and any other string a reader meets on a page.
    """
    offenders = {
        f"{path.relative_to(_EVAL_PACKAGE)}: {literal[:80]}": found
        for path in sorted(_EVAL_PACKAGE.rglob("*.py"))
        for literal in source_string_literals(path)
        if not _is_permitted_literal(literal)
        for found in [_scan(literal)]
        if found
    }

    assert offenders == {}, (
        f"a figure caption or label is where a reader actually meets a name this domain does not "
        f"have: {offenders}"
    )


def test_the_literal_scan_reads_the_captions_it_claims_to() -> None:
    """Non-vacuity: the walk must actually reach the modules that title the figures, and must
    return the titles rather than only their docstrings."""
    titles = source_string_literals(_EVAL_PACKAGE / "analyses" / "lag_kl.py")

    assert any("lag" in literal.lower() for literal in titles)
    assert _scan("Transfer entropy per lag") == ["transfer entropy"]
    # And the two exemptions the caption walk applies do not swallow a caption: a cap name is
    # admitted only by exact equality, and a statement fragment only when it is one.
    assert _is_permitted_literal("waveforms")
    assert not _is_permitted_literal("the retained waveforms, per anchor")
    assert not _is_permitted_literal("the forecast error in bpm")


# =================================================================================================
# The headline names which quantity it carries
# =================================================================================================
def test_no_headline_path_resolves_to_a_floored_kl() -> None:
    floored = [
        (name, path) for name, path in report_seam.HEADLINE_SCALARS
        if any("kl_train" in part or part.endswith("_train") for part in path)
    ]

    assert floored == [], (
        "only the unfloored KL may be read as a rate: free bits are applied per dimension per step "
        "before summing, so the floored value exceeds it by construction"
    )


def test_a_headline_pointed_at_the_floored_kl_would_be_caught(monkeypatch) -> None:
    """The guard above passes on an empty registry; this is what says it discriminates."""
    monkeypatch.setattr(
        report_seam,
        "HEADLINE_SCALARS",
        (("kl", ("readouts", "source_conditioned_kl_train")),),
    )

    with pytest.raises(AssertionError):
        test_no_headline_path_resolves_to_a_floored_kl()


def test_the_headline_says_which_pred_gap_it_carries(collected_run) -> None:
    headline = collected_run["summary"]["results"]["headline"]

    assert "pred_gap_mc_nats" in headline and "pred_gap_train_path_nats" in headline
    assert "pred_gap" not in headline, "an unqualified name leaves a reader to guess"
    assert "marginalised" in headline["pred_gap_convention"]
    # And the caveat this cell adds, in the artifact rather than in a document beside it.
    assert "BUDGET-LOCAL" in headline["pred_gap_convention"]


# =================================================================================================
# The rescaled per-anchor score
# =================================================================================================
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


def test_a_rescaled_per_sample_score_ships_with_its_caveat_or_not_at_all(collected_run) -> None:
    summary = collected_run["summary"]
    offending = sorted({key for key in _keys_of(summary) if _SAMPLE_KEY.fullmatch(key)})

    if not offending:
        return
    assert _SAMPLE_CAVEAT.search(collected_run["text"]), (
        f"{offending} is a fixed /1470 rescale of a block score rather than a mean over the "
        f"coefficients actually scored, and under-reports on any anchor with masked forecast steps"
    )
