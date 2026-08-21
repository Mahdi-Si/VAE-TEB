r"""The documentation contract: `EVAL.md` and `FIGURE_GUIDE.md` cannot drift from the code.

Three bindings, each one direction of a drift that is otherwise invisible:

* every registered analysis has a ``###`` heading in ``EVAL.md``, by **exact slug equality** with
  its module name -- so an analysis added to the registry without a section fails here rather
  than shipping undocumented;
* every resolved ``eval_config`` key is mentioned in ``EVAL.md``, backticked -- a knob an
  operator can set and no document explains is a knob nobody sets correctly;
* every figure in the committed ``figure_manifest.json`` has an entry in ``FIGURE_GUIDE.md``, and
  every documented figure is one a run emits.

**These run in the fast gate, deliberately.** The manifest is kept equal to a real run by the
``slow``-marked smoke suite, and these tests read the manifest rather than the run -- because a
drift guard that only runs under ``-m slow`` is a drift guard that does not run. Every check is a
pure function over (document text, expected names), so the failing case is exercised by calling
it with one extra name rather than by editing a committed document.

The fourth binding is the operations side: every way ``preflight.py`` refuses a run has a row in
``EVAL.md``'s recovery table, extracted from the module's AST rather than from a list somebody
maintains.

The fifth is inside the code: ``run.py``'s ``RUN_ARGS`` comment table lists the names ``only`` and
``skip`` accept, for an operator editing that dict rather than typing flags. ``--help`` is
interpolated from the registry and cannot go stale; a hand-written table can, and it is read at
exactly the moment someone is choosing a name.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

import pytest

from teb_vae.lag_attn_rws.eval import run as run_module
from teb_vae.lag_attn_rws.eval.config_schema import validate_eval_config

_EVAL_ROOT = Path(__file__).resolve().parents[1] / "eval"
_EVAL_DOC = _EVAL_ROOT / "EVAL.md"
_FIGURE_GUIDE = _EVAL_ROOT / "FIGURE_GUIDE.md"
_MANIFEST = _EVAL_ROOT / "figure_manifest.json"
_PREFLIGHT = _EVAL_ROOT / "preflight.py"


@pytest.fixture(scope="module")
def eval_doc() -> str:
    return _EVAL_DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def figure_guide() -> str:
    return _FIGURE_GUIDE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def manifest() -> Dict:
    return json.loads(_MANIFEST.read_text(encoding="utf-8"))


# =============================================================================
# (a) Analysis sections, by exact slug equality
# =============================================================================
def analysis_headings(text: str) -> List[str]:
    """Return the ``###`` heading slugs of a document, in order.

    Exact equality on the whole heading rather than a substring search: a section headed
    "### the forecast analysis" would satisfy any "is 'forecast' in the heading" test while
    leaving the binding between a module name and its section a matter of prose.
    """
    return re.findall(r"^###\s+(\S+)\s*$", text, flags=re.MULTILINE)


def missing_sections(text: str, analyses: List[str]) -> List[str]:
    """Return the analyses with no ``###`` section of exactly their module name."""
    headings = set(analysis_headings(text))
    return [name for name in analyses if name not in headings]


def registered_analyses() -> List[str]:
    """Every analysis a run can execute: the unskippable steps plus the selectable registry."""
    return [*run_module.UNSKIPPABLE_ANALYSES, *run_module.ANALYSIS_FUNCTIONS]


def test_the_binding_is_not_vacuous(eval_doc):
    """A regex that matched nothing would pass every assertion below on an empty document."""
    assert len(analysis_headings(eval_doc)) >= 14
    assert len(registered_analyses()) >= 14


def test_every_registered_analysis_has_a_section_of_exactly_its_name(eval_doc):
    missing = missing_sections(eval_doc, registered_analyses())

    assert missing == [], (
        f"EVAL.md has no '### <name>' section for {missing}. Every registered analysis is "
        f"documented under its own module name, so a reader can go from a directory in the "
        f"output to the paragraph explaining it."
    )


def test_an_analysis_without_a_section_is_caught(eval_doc):
    """Non-vacuity: the assertion above passes on an empty registry, and this is what says the
    check discriminates. Exercised by adding a name rather than by editing the document."""
    assert missing_sections(eval_doc, [*registered_analyses(), "newly_added"]) == ["newly_added"]


#: The one single-token ``###`` heading the operations sections contribute. The others there are
#: multi-token, and :func:`analysis_headings` anchors on ``\s*$``, so they are never captured.
OPERATIONAL_HEADINGS = {"Dependencies"}


def stale_headings(text: str, registered: Sequence[str]) -> List[str]:
    """Return the ``###`` headings that no registered analysis answers to.

    Filtered against the registry and the operations sections' own heading -- and against nothing
    else. A shape filter such as "only headings containing an underscore" would silently exempt
    every analysis whose module name is one word, which is two thirds of them.
    """
    known = set(registered) | OPERATIONAL_HEADINGS
    return [heading for heading in analysis_headings(text) if heading not in known]


def test_the_document_documents_nothing_that_is_not_registered(eval_doc):
    """The other direction: a section for an analysis nobody runs outlives its deletion, and
    reads to a maintainer as a feature that exists."""
    stale = stale_headings(eval_doc, registered_analyses())

    assert stale == [], f"EVAL.md documents analyses that are not registered: {stale}"


def test_a_section_for_an_unregistered_analysis_is_caught(eval_doc):
    """Non-vacuity for the direction above, and the reason the shape filter had to go: with it,
    a stale section for any single-word analysis -- forecast, coupling, latent, events -- passed.
    Exercised by adding a heading rather than by editing the committed document."""
    assert stale_headings(eval_doc + "\n### phantom\n", registered_analyses()) == ["phantom"]
    for name in registered_analyses():
        assert stale_headings(eval_doc, [n for n in registered_analyses() if n != name]) == [name]


# =============================================================================
# (a, continued) Every resolved eval_config key is explained
# =============================================================================
def unmentioned_keys(text: str, keys: List[str]) -> List[str]:
    """Return the config keys the document does not mention **backticked**.

    Backticked rather than merely present: ``seed`` and ``caps`` are ordinary English words, and
    a bare-substring test would pass on a document that never explained either.
    """
    return [key for key in keys if f"`{key}`" not in text]


def resolved_eval_config_keys() -> List[str]:
    """Every key a validated ``eval_config`` block resolves to, defaults included."""
    return sorted(validate_eval_config({}))


def test_every_resolved_eval_config_key_is_explained(eval_doc):
    missing = unmentioned_keys(eval_doc, resolved_eval_config_keys())

    assert missing == [], (
        f"EVAL.md does not mention eval_config key(s) {missing} in backticks. A knob an operator "
        f"can set and no document explains is a knob nobody sets correctly."
    )


def test_a_config_key_without_a_mention_is_caught(eval_doc):
    assert unmentioned_keys(eval_doc, ["newly_added_key"]) == ["newly_added_key"]


def test_the_two_deliberately_absent_keys_stay_absent():
    """Both would let an operator move a threshold until a difference appeared or vanished, so
    their absence is part of the contract rather than an omission -- and the document says so."""
    keys = resolved_eval_config_keys()

    assert "alpha" not in keys and "trajectory_bin_hours" not in keys


# =============================================================================
# (b) Every emitted figure is documented
# =============================================================================
def undocumented_figures(guide: str, manifest: Dict) -> List[str]:
    """Return the manifest's figures and families with no entry in the figure guide."""
    missing = [
        f"{analysis}/{name}"
        for analysis, names in manifest["figures"].items()
        for name in names
        if f"{analysis}/{name}" not in guide
    ]
    missing += [
        f"family:{family}"
        for family, record in manifest.get("families", {}).items()
        if record["guide_marker"] not in guide
    ]
    return missing


def test_the_manifest_is_not_empty(manifest):
    """A manifest with no figures would pass the binding below vacuously."""
    assert sum(len(names) for names in manifest["figures"].values()) >= 14
    assert set(manifest.get("families", {})) == {"grouped_variants", "sample_pages"}


def test_every_figure_in_the_manifest_has_a_guide_entry(figure_guide, manifest):
    missing = undocumented_figures(figure_guide, manifest)

    assert missing == [], (
        f"FIGURE_GUIDE.md has no entry for {missing}. The manifest is kept equal to a real run "
        f"by the slow smoke suite, so a figure listed there is one an operator will actually "
        f"open -- and the compensated-lag convention is exactly what gets reverse-engineered "
        f"wrong without an entry."
    )


def test_a_figure_added_to_the_manifest_without_a_guide_entry_is_caught(figure_guide, manifest):
    """The third deliberate omission. Exercised against a copy of the manifest so the committed
    one is untouched."""
    planted = {
        "figures": {**manifest["figures"], "coupling": ["a_brand_new_figure.pdf"]},
        "families": manifest.get("families", {}),
    }

    assert undocumented_figures(figure_guide, planted) == ["coupling/a_brand_new_figure.pdf"]


def test_the_guide_documents_nothing_a_run_does_not_emit(figure_guide, manifest):
    """A guide entry for a figure nothing writes is a promise the run does not keep, and it
    outlives the analysis that was deleted."""
    documented = set(re.findall(r"`([a-z_]+/[a-z_]+\.pdf)`", figure_guide))
    emitted = {
        f"{analysis}/{name}"
        for analysis, names in manifest["figures"].items()
        for name in names
    }

    assert documented - emitted == set()


# =============================================================================
# The operations contract: every refusal has a recovery row
# =============================================================================
def preflight_refusal_prefixes(source: str) -> List[str]:
    """Return the leading fragment of every ``EvalPreconditionUnmet`` message in ``source``.

    Walked from the AST rather than matched with a regex: the messages are f-strings and
    concatenations, so the leading literal has to be found structurally. Each fragment is cut at
    the first newline, comma or colon, which is the part stable enough to key a table row on --
    everything after it names paths and values that differ per run.

    The re-raise of a reused trainer guard carries no literal of its own and yields nothing here;
    those four are covered by :func:`reused_trainer_guards`.
    """
    prefixes: List[str] = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)):
            continue
        if getattr(node.exc.func, "id", None) != "EvalPreconditionUnmet":
            continue
        argument = node.exc.args[0]
        while isinstance(argument, ast.BinOp):
            argument = argument.left
        if isinstance(argument, ast.JoinedStr):
            leading = argument.values[0]
            text = str(leading.value) if isinstance(leading, ast.Constant) else ""
        elif isinstance(argument, ast.Constant):
            text = str(argument.value)
        else:
            text = ""
        fragment = re.split(r"[\n,:]", text)[0].strip()
        if fragment:
            prefixes.append(fragment)
    return sorted(set(prefixes))


def refusal_raise_sites(source: str) -> int:
    """Count every ``raise`` of an ``EvalPreconditionUnmet``, however it is spelt.

    Matched on the tail of the callee name so an attribute-qualified raise still counts. This is
    the denominator :func:`preflight_refusal_prefixes` has to reach: that walker yields nothing
    for a message it cannot key -- one built from a bare name, or an f-string opening with an
    interpolation -- and a floor that does not track the module would let such a guard ship with
    no recovery row while every documentation test stayed green.
    """
    sites = 0
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)):
            continue
        name = getattr(node.exc.func, "id", None) or getattr(node.exc.func, "attr", None)
        if str(name).endswith("EvalPreconditionUnmet"):
            sites += 1
    return sites


def reused_trainer_guards(source: str) -> List[str]:
    """Return the names of the training-entry-point guards preflight reuses.

    They raise the trainer's own message rather than one of their own, so they are keyed in the
    recovery table by the check name preflight records them under.
    """
    names: List[str] = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_reuse_trainer_guard"):
            continue
        last = node.args[-1]
        if isinstance(last, ast.Constant):
            names.append(str(last.value))
    return sorted(set(names))


def rows_missing_from(text: str, keys: List[str]) -> List[str]:
    """Return the keys with no row in the recovery table."""
    table = text[text.index("### Guard recovery table"):]
    return [key for key in keys if f"`{key}`" not in table]


#: Raise sites that carry no literal of their own: ``_reuse_trainer_guard``'s re-raise of the
#: trainer's own message, which :func:`reused_trainer_guards` keys instead.
_RERAISED_REFUSALS = 1


def test_the_extraction_found_the_guards(eval_doc):
    """Non-vacuity in both halves: an extraction that found nothing passes everything below."""
    source = _PREFLIGHT.read_text(encoding="utf-8")

    assert len(reused_trainer_guards(source)) == 4
    assert "### Guard recovery table" in eval_doc
    # Equality against the module's own raise-site count, not a constant floor. A guard whose
    # message the extractor cannot key -- `EvalPreconditionUnmet(_SOME_MESSAGE)`, or an f-string
    # opening with an interpolation and so having no stable prefix at all -- fails here, telling
    # its author to reword, instead of vanishing from the coverage check below.
    assert len(preflight_refusal_prefixes(source)) == (
        refusal_raise_sites(source) - _RERAISED_REFUSALS
    ), (
        "every EvalPreconditionUnmet must open with a stable string literal, so its message can "
        "be keyed to a row in EVAL.md's guard recovery table"
    )


def test_every_preflight_refusal_has_a_recovery_row(eval_doc):
    """What keeps the table complete as guards are added: a new refusal with no row means an
    operator meets a message the documentation has never heard of."""
    source = _PREFLIGHT.read_text(encoding="utf-8")
    keys = preflight_refusal_prefixes(source) + reused_trainer_guards(source)

    missing = rows_missing_from(eval_doc, keys)

    assert missing == [], f"EVAL.md's guard recovery table has no row for {missing}"


def test_a_new_refusal_without_a_row_is_caught(eval_doc):
    assert rows_missing_from(eval_doc, ["a brand new refusal"]) == ["a brand new refusal"]


def test_the_exit_code_semantics_are_written_down(eval_doc):
    """The single most surprising property of the pipeline: three things that look like failures
    deliberately leave the exit code at zero, which is why an offline gate exists separately."""
    operations = eval_doc[eval_doc.index("## Operations"):]

    assert "if and only if a step raised" in operations
    for deliberate in ("sanity", "coverage", "inert-cap"):
        assert deliberate in operations, deliberate
    # And the two facts about a re-run that a reader gets wrong by default.
    assert "non-destructive but not additive" in operations
    assert "read the backup" in operations.lower()
    # What a refusal leaves behind, stated rather than implied.
    assert "no `summary.json`" in operations


# =============================================================================
# (e) The launch dict's own table of selectable analyses
# =============================================================================
def launch_table_analyses(source: str) -> List[str]:
    r"""Return the analysis names ``RUN_ARGS``' comment table lists, in order.

    Anchored on two spaces after the ``#`` and a colon after the name, which is what separates a
    table row from the prose around it and from a row's own continuation lines. Read from the
    ``RUN_ARGS`` assignment onward so a name mentioned anywhere else in the module -- the registry
    itself is above it -- cannot stand in for a row.
    """
    block = source[source.index("RUN_ARGS: Dict[str, Any] = {"):]
    return re.findall(r"^\s*#\s{2,}(\w+):", block, flags=re.MULTILINE)


def test_the_launch_table_lists_every_selectable_analysis_and_only_those():
    """Both directions at once, because the table is an ordered list of exactly the registry: a
    missing row is a name an operator will not know to type, and a stale row is one that raises
    at startup after they typed it."""
    listed = launch_table_analyses(Path(run_module.__file__).read_text(encoding="utf-8"))

    assert listed == list(run_module.ANALYSES), (
        f"run.py's RUN_ARGS comment table lists {listed}, but --only/--skip accept "
        f"{list(run_module.ANALYSES)}. The table is what an operator reads while editing the "
        f"dict; --help is interpolated from the registry and cannot disagree with it."
    )
    # The unskippable step is named in the table's prose as the one thing neither key takes; a
    # row for it would read as a fifteenth choice that raises when chosen.
    assert not set(run_module.UNSKIPPABLE_ANALYSES) & set(listed)


def test_a_missing_or_stale_launch_table_row_is_caught():
    """Non-vacuity: a regex matching nothing would pass the equality above on an empty registry.
    Exercised against synthetic sources rather than by editing the module."""
    assert launch_table_analyses("RUN_ARGS: Dict[str, Any] = {\n    #   forecast:  what\n") == [
        "forecast"
    ]
    # A continuation line, and the prose around the table, are not rows.
    assert launch_table_analyses(
        "RUN_ARGS: Dict[str, Any] = {\n"
        "    # Which analyses run. `band_partition` is not selectable.\n"
        "    #   forecast:         Is the forecast any good. Skill against persistence and\n"
        "    #                     climatology, resolved by horizon step.\n"
    ) == ["forecast"]


def test_the_interpretation_rules_are_all_present(eval_doc):
    """Every rule the spec requires the contract to carry, each keyed on the phrase that would
    have to survive a rewrite for the rule to still be stated."""
    for rule in (
        "not causal",                      # the readout's standing
        "prediction space",                # specificity is read there, not in KL space
        "unfloored KL",                    # only it is a rate
        "prior_variance_not_pinned",       # and only off its clamp
        "compensated lag",                 # the lag convention
        "per recording",                   # the aggregation unit
        "out-of-distribution",             # every class contrast
        "healthy_no_bg_cs",                # ...and the wider-than-expected subgroup scope
        "not comparable",                  # eval vs training metrics
        "estimate, not a bound",           # the sufficiency gap
        "per event",                       # deceleration rates
        "Lag ablation is absent",          # and necessity with it
        "/480 rescale",                    # the rescaled per-sample score
        "corrected within a clock",           # two clocks, two Holm families
        "a subset of the evaluated cohort",   # ...and why the second one is capped
        "counted and never dropped",          # the two onset diagnostics that filter nothing
    ):
        assert rule in eval_doc, f"EVAL.md no longer states: {rule!r}"
