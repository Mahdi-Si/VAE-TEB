r"""The documentation contract: `EVAL.md` and `FIGURE_GUIDE.md` cannot drift from the code.

Seven bindings, each one direction of a drift that is otherwise invisible:

* every registered analysis has a ``###`` heading in ``EVAL.md``, by **exact slug equality** with
  its module name -- and the registry read is the one a run of *this* cell selects from, the shared
  fifteen plus the binding's own three, so an analysis registered on ``CFS_BINDING`` is documented
  by being registered rather than by being remembered;
* every resolved ``eval_config`` key is mentioned, backticked -- a knob an operator can set and no
  document explains is a knob nobody sets correctly;
* every figure in the committed ``figure_manifest.json`` has an entry in ``FIGURE_GUIDE.md``, and
  every documented figure is one a run emits;
* every guard in ``preflight.GUARD_RECOVERY`` has a row in the recovery table, read from the
  mapping rather than from a list somebody maintains beside it;
* the launch table an operator reads while editing ``RUN_ARGS`` is exactly the registry;
* the divergence register is **rendered** from ``divergences.json`` rather than hand-kept, and the
  rendering is compared verbatim -- a register that could be edited in place would drift from the
  manifest it exists to make readable, which is the failure the manifest itself exists to prevent;
* and the interpretation rules this cell's output cannot be read without are each present.

**These run in the fast gate, deliberately.** The manifest is kept equal to a real run by the
``slow``-marked smoke suite, and these tests read the manifest rather than the run -- because a
drift guard that only runs under ``-m slow`` is a drift guard that does not run. Every check is a
pure function over (document text, expected names), so the failing case is exercised by calling it
with one extra name rather than by editing a committed document.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

import pytest

from teb_vae.lag_attn_cfs.eval import preflight
from teb_vae.lag_attn_cfs.eval import run as run_module
from teb_vae.lag_attn_cfs.eval.binding import CFS_BINDING
from teb_vae.lag_attn_cfs.eval.config_schema import validate_eval_config

_EVAL_ROOT = Path(__file__).resolve().parents[1] / "eval"
_EVAL_DOC = _EVAL_ROOT / "EVAL.md"
_FIGURE_GUIDE = _EVAL_ROOT / "FIGURE_GUIDE.md"
_MANIFEST = _EVAL_ROOT / "figure_manifest.json"
_DIVERGENCES = _EVAL_ROOT / "divergences.json"


@pytest.fixture(scope="module")
def eval_doc() -> str:
    return _EVAL_DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def figure_guide() -> str:
    return _FIGURE_GUIDE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def manifest() -> Dict:
    return json.loads(_MANIFEST.read_text(encoding="utf-8"))


# =================================================================================================
# (a) Analysis sections, by exact slug equality
# =================================================================================================
def analysis_headings(text: str) -> List[str]:
    """Return the ``###`` heading slugs of a document, in order.

    Exact equality on the whole heading rather than a substring search: a section headed "### the
    forecast analysis" would satisfy any "is 'forecast' in the heading" test while leaving the
    binding between a module name and its section a matter of prose.
    """
    return re.findall(r"^###\s+(\S+)\s*$", text, flags=re.MULTILINE)


def missing_sections(text: str, analyses: Sequence[str]) -> List[str]:
    """Return the analyses with no ``###`` section of exactly their module name."""
    headings = set(analysis_headings(text))
    return [name for name in analyses if name not in headings]


def registered_analyses() -> List[str]:
    """Every analysis a run of this cell can execute.

    The unskippable data-side step plus the **merged** registry: the shared fifteen with
    ``CFS_BINDING``'s own three merged in, which is what ``run.main`` selects against. Read off the
    merge rather than listed, so registering a fourth reaches this contract from one place.
    """
    return [
        *run_module.UNSKIPPABLE_ANALYSES,
        *run_module.merged_analysis_functions(CFS_BINDING),
    ]


def test_the_binding_is_not_vacuous(eval_doc) -> None:
    """A regex that matched nothing would pass every assertion below on an empty document."""
    assert len(analysis_headings(eval_doc)) >= 19
    assert len(registered_analyses()) >= 19


def test_every_registered_analysis_has_a_section_of_exactly_its_name(eval_doc) -> None:
    missing = missing_sections(eval_doc, registered_analyses())

    assert missing == [], (
        f"EVAL.md has no '### <name>' section for {missing}. Every registered analysis is "
        f"documented under its own module name, so a reader can go from a directory in the output "
        f"to the paragraph explaining it."
    )


def test_an_analysis_without_a_section_is_caught(eval_doc) -> None:
    """Non-vacuity: the assertion above passes on an empty registry, and this is what says the
    check discriminates. Exercised by adding a name rather than by editing the document."""
    assert missing_sections(eval_doc, [*registered_analyses(), "newly_added"]) == ["newly_added"]


def test_this_cells_own_three_analyses_are_documented_here(eval_doc) -> None:
    """Named rather than left to the loop above, which would still pass if the binding stopped
    registering them. These three are the questions only a causal cell can ask."""
    headings = set(analysis_headings(eval_doc))
    registry = run_module.merged_analysis_functions(CFS_BINDING)

    for name in ("warmup", "source_null", "spectral_skill"):
        assert name in headings, name
        assert name in registry, name


def test_the_analysis_the_fork_did_not_port_has_no_section(eval_doc) -> None:
    """``coherence`` is not ported at all, and a section for it would be a promise no run keeps."""
    assert "coherence" not in analysis_headings(eval_doc)
    assert "coherence" not in run_module.merged_analysis_functions(CFS_BINDING)


#: The one single-token ``###`` heading the operations sections contribute. The others there are
#: multi-token, and :func:`analysis_headings` anchors on ``\s*$``, so they are never captured.
OPERATIONAL_HEADINGS = {"Dependencies"}


def stale_headings(text: str, registered: Sequence[str]) -> List[str]:
    """Return the ``###`` headings that no registered analysis answers to.

    Filtered against the registry and the operations sections' own heading -- and against nothing
    else. A shape filter such as "only headings containing an underscore" would silently exempt
    every analysis whose module name is one word, which is most of them.
    """
    known = set(registered) | OPERATIONAL_HEADINGS
    return [heading for heading in analysis_headings(text) if heading not in known]


def test_the_document_documents_nothing_that_is_not_registered(eval_doc) -> None:
    """The other direction: a section for an analysis nobody runs outlives its deletion, and reads
    to a maintainer as a feature that exists. It matters more in a fork than anywhere else, because
    a section carried over from the pipeline this one was copied from is exactly such a section."""
    stale = stale_headings(eval_doc, registered_analyses())

    assert stale == [], f"EVAL.md documents analyses that are not registered: {stale}"


def test_a_section_for_an_unregistered_analysis_is_caught(eval_doc) -> None:
    """Non-vacuity for the direction above, and the reason a shape filter would not do: with one,
    a stale section for any single-word analysis -- forecast, coupling, latent, events -- passes."""
    assert stale_headings(eval_doc + "\n### phantom\n", registered_analyses()) == ["phantom"]
    for name in registered_analyses():
        assert stale_headings(eval_doc, [n for n in registered_analyses() if n != name]) == [name]


# =================================================================================================
# (b) Every resolved eval_config key is explained
# =================================================================================================
def unmentioned_keys(text: str, keys: Sequence[str]) -> List[str]:
    """Return the config keys the document does not mention **backticked**.

    Backticked rather than merely present: ``seed`` and ``caps`` are ordinary English words, and a
    bare-substring test would pass on a document that never explained either.
    """
    return [key for key in keys if f"`{key}`" not in text]


def resolved_eval_config_keys() -> List[str]:
    """Every key a validated ``eval_config`` block resolves to, defaults included."""
    return sorted(validate_eval_config({}))


def test_every_resolved_eval_config_key_is_explained(eval_doc) -> None:
    missing = unmentioned_keys(eval_doc, resolved_eval_config_keys())

    assert missing == [], (
        f"EVAL.md does not mention eval_config key(s) {missing} in backticks. A knob an operator "
        f"can set and no document explains is a knob nobody sets correctly."
    )


def test_a_config_key_without_a_mention_is_caught(eval_doc) -> None:
    assert unmentioned_keys(eval_doc, ["newly_added_key"]) == ["newly_added_key"]


def test_this_cells_own_key_is_explained_and_its_unset_default_is_stated(eval_doc) -> None:
    """``clock_margin_min_nats`` is the one key the fork adds, it ships ``null``, and the whole
    point of shipping it that way is lost if the document does not say what ``null`` does. The loop
    above would pass on a mention that never explained the default."""
    assert "clock_margin_min_nats" in resolved_eval_config_keys()
    assert validate_eval_config({})["clock_margin_min_nats"] is None
    assert unmentioned_keys(eval_doc, ["clock_margin_min_nats"]) == []
    assert "INCONCLUSIVE" in eval_doc


def test_the_three_deliberately_absent_keys_stay_absent() -> None:
    """The first two would let an operator move a threshold until a difference appeared or
    vanished. The third is this cell's own and is worse: an operator who could set the anchor
    stride could change the population every number in the run is computed over."""
    keys = resolved_eval_config_keys()

    assert "alpha" not in keys
    assert "trajectory_bin_hours" not in keys
    assert "anchor_stride" not in keys


# =================================================================================================
# (c) Every emitted figure is documented
# =================================================================================================
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


def test_the_manifest_is_not_empty(manifest) -> None:
    """A manifest with no figures would pass the binding below vacuously."""
    assert sum(len(names) for names in manifest["figures"].values()) >= 14
    assert set(manifest.get("families", {})) == {"grouped_variants", "sample_pages"}


def test_the_manifest_carries_this_cells_own_figures(manifest) -> None:
    """The families under ``warmup/``, ``source_null/`` and ``spectral_skill/`` are the only ones
    in the manifest the raw pipeline's run does not produce, and they are the ones whose axes a
    reader of that pipeline would read backwards."""
    for analysis in ("warmup", "source_null", "spectral_skill"):
        assert manifest["figures"].get(analysis), analysis


def test_every_figure_in_the_manifest_has_a_guide_entry(figure_guide, manifest) -> None:
    missing = undocumented_figures(figure_guide, manifest)

    assert missing == [], (
        f"FIGURE_GUIDE.md has no entry for {missing}. The manifest is kept equal to a real run by "
        f"the slow smoke suite, so a figure listed there is one an operator will actually open -- "
        f"and the coefficient-time lag axis is exactly what gets reverse-engineered wrong without "
        f"an entry."
    )


def test_a_figure_added_to_the_manifest_without_a_guide_entry_is_caught(
    figure_guide, manifest
) -> None:
    """Exercised against a copy of the manifest so the committed one is untouched."""
    planted = {
        "figures": {**manifest["figures"], "coupling": ["a_brand_new_figure.pdf"]},
        "families": manifest.get("families", {}),
    }

    assert undocumented_figures(figure_guide, planted) == ["coupling/a_brand_new_figure.pdf"]


#: Figures a ``gaussian_nll`` checkpoint emits and an ``mse`` one structurally cannot, so they are
#: documented here and absent from a manifest seeded by the fixture -- which trains under ``mse``,
#: whose decoder log-variance head is never fitted at all.
#:
#: Named rather than pattern-matched, and compared by **equality** below rather than as an upper
#: bound: a figure that stopped being emitted for any other reason must still fail. The pair is the
#: whole of ``calibration``, which is the one analysis whose output is conditional on the objective.
CONDITIONAL_ON_LIKELIHOOD = {
    "calibration/pit_reliability.pdf",
    "calibration/logvar_distribution.pdf",
}


def test_the_guide_documents_nothing_a_run_does_not_emit(figure_guide, manifest) -> None:
    """A guide entry for a figure nothing writes is a promise the run does not keep, and it
    outlives the analysis that was deleted -- with one stated exception, above."""
    documented = set(re.findall(r"`([a-z_]+/[a-z_0-9]+\.pdf)`", figure_guide))
    emitted = {
        f"{analysis}/{name}"
        for analysis, names in manifest["figures"].items()
        for name in names
    }

    assert documented - emitted == CONDITIONAL_ON_LIKELIHOOD


def test_the_conditional_pair_is_documented_as_conditional(figure_guide, manifest) -> None:
    """Non-vacuity for the exception above, in both halves: the two figures must be documented (or
    the set difference would be empty for the wrong reason), and the guide must say what makes them
    conditional -- an entry that promised them unconditionally would read as a run that failed."""
    for name in CONDITIONAL_ON_LIKELIHOOD:
        assert f"`{name}`" in figure_guide, name
        assert name not in {
            f"{analysis}/{figure}"
            for analysis, figures in manifest["figures"].items()
            for figure in figures
        }
    assert "An `mse` checkpoint emits no such figure at all" in figure_guide


# =================================================================================================
# (d) The operations contract: every refusal has a recovery row
# =================================================================================================
def rows_missing_from(text: str, keys: Sequence[str]) -> List[str]:
    """Return the keys with no row in the recovery table."""
    table = text[text.index("### Guard recovery table"):]
    return [key for key in keys if f"`{key}`" not in table]


def test_every_preflight_refusal_has_a_recovery_row(eval_doc) -> None:
    """What keeps the table complete as guards are added: a new refusal with no row means an
    operator meets a message the documentation has never heard of.

    Keyed on the guard **function** rather than on the leading fragment of its message, which is
    the fork's own improvement on the sibling: ``preflight.GUARD_RECOVERY`` is a module-level
    mapping that ``tests/test_eval_preflight.py`` already pins against the module's AST, so this
    test binds the document to a table the code already keeps honest instead of re-deriving the
    guard set from string literals a rewording could break.
    """
    missing = rows_missing_from(eval_doc, sorted(preflight.GUARD_RECOVERY))

    assert missing == [], f"EVAL.md's guard recovery table has no row for {missing}"


def test_the_guard_set_is_not_empty() -> None:
    """Non-vacuity: an empty mapping would pass the coverage assertion above."""
    assert len(preflight.GUARD_RECOVERY) >= 10


def test_a_new_refusal_without_a_row_is_caught(eval_doc) -> None:
    assert rows_missing_from(eval_doc, ["a_brand_new_guard"]) == ["a_brand_new_guard"]


def test_the_recovery_table_carries_the_causal_guards_by_name(eval_doc) -> None:
    """The three this cell adds. A table that lost one would still pass a count, and each of them
    refuses a run for a reason a reader of the raw pipeline's table has never met."""
    table = eval_doc[eval_doc.index("### Guard recovery table"):]

    for name in (
        "check_causal_transform",
        "check_no_reach_budget",
        "check_warmup_budget_matches_checkpoint",
    ):
        assert f"`{name}`" in table, name


def test_the_exit_code_semantics_are_written_down(eval_doc) -> None:
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


def test_the_document_says_which_question_each_green_check_answers(eval_doc) -> None:
    """Two green checks that answer two questions are only confusing if nothing says so. The
    cross-reference is asserted in this direction here and in the other by
    ``tests/test_check_run.py``, so neither module can quietly become the only one."""
    assert "check_run.py" in eval_doc
    assert "verify.py" in eval_doc
    assert "while a run is in flight" in eval_doc
    assert "is this finished checkpoint acceptable" in eval_doc


# =================================================================================================
# (e) The launch dict's own table of selectable analyses
# =================================================================================================
def launch_table_analyses(source: str) -> List[str]:
    r"""Return the analysis names ``RUN_ARGS``' comment table lists, in order.

    Anchored on two spaces after the ``#`` and a colon after the name, which is what separates a
    table row from the prose around it and from a row's own continuation lines. Read from the
    ``RUN_ARGS`` assignment onward so a name mentioned anywhere else in the module -- the registry
    itself is above it -- cannot stand in for a row.
    """
    block = source[source.index("RUN_ARGS: Dict[str, Any] = {"):]
    return re.findall(r"^\s*#\s{2,}(\w+):", block, flags=re.MULTILINE)


def test_the_launch_table_lists_every_selectable_analysis_and_only_those() -> None:
    """Both directions at once, because the table is an ordered list of exactly the registry: a
    missing row is a name an operator will not know to type, and a stale row is one that raises at
    startup after they typed it.

    The registry compared against is the **merged** one, because that is what ``main`` selects
    against -- so this is also what would catch a help text or a table built from the shared
    fifteen while the run accepted eighteen.
    """
    selectable = list(run_module.merged_analysis_functions(CFS_BINDING))
    listed = launch_table_analyses(Path(run_module.__file__).read_text(encoding="utf-8"))

    assert listed == selectable, (
        f"run.py's RUN_ARGS comment table lists {listed}, but --only/--skip accept {selectable}. "
        f"The table is what an operator reads while editing the dict; --help is interpolated from "
        f"the same registry and cannot disagree with it."
    )
    # The unskippable step is named in the table's prose as the one thing neither key takes; a row
    # for it would read as a nineteenth choice that raises when chosen.
    assert not set(run_module.UNSKIPPABLE_ANALYSES) & set(listed)


def test_the_help_text_names_the_registry_the_run_selects_from() -> None:
    """The other half of the pairing above. ``--help`` is interpolated rather than written, so it
    cannot go stale -- but it can be interpolated from the *wrong* tuple, and then it tells an
    operator that ``--only warmup`` is invalid while the run accepts it."""
    help_text = run_module.build_parser().format_help()

    for name in run_module.merged_analysis_functions(CFS_BINDING):
        assert name in help_text, name


def test_a_missing_or_stale_launch_table_row_is_caught() -> None:
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


# =================================================================================================
# (f) The divergence register is rendered, not written
# =================================================================================================
def render_divergence_register(manifest: Dict) -> str:
    """Render the divergence manifest as the markdown block ``EVAL.md`` must carry.

    One bullet per module, in the manifest's own key order, so the rendering is a pure function of
    the committed data. An ``equivalent`` entry carries no reason of its own -- that is what the
    state means -- so what is rendered for it is the assertions that keep the claim true, which is
    the only thing about such an entry a reader can check.
    """
    lines = []
    for name, entry in manifest["modules"].items():
        reason = entry.get("reason", "").strip()
        if not reason:
            cited = ", ".join(f"`{item}`" for item in entry.get("exercised_by", ()))
            reason = f"Behaviour-equivalent to the sibling module, exercised by {cited}."
        lines.append(f"- **`{name}`** (*{entry['state']}*) {reason}")
    return "\n".join(lines)


@pytest.fixture(scope="module")
def divergences() -> Dict:
    return json.loads(_DIVERGENCES.read_text(encoding="utf-8"))


def test_the_divergence_register_is_the_rendering_of_the_manifest(eval_doc, divergences) -> None:
    """The register is data with a renderer, not prose. Editing ``divergences.json`` without
    re-rendering fails here, and so does editing the register in place -- which is the failure the
    manifest exists to prevent, one level up: a prose register goes stale in the direction nobody
    notices, and a rendered one cannot."""
    block = render_divergence_register(divergences)

    assert block in eval_doc, (
        "EVAL.md's divergence register is not the current rendering of eval/divergences.json. "
        "Re-render it from the manifest rather than editing the section, and change the manifest "
        "in the same commit as the module whose state moved."
    )


def test_the_rendering_would_notice_a_changed_entry(eval_doc, divergences) -> None:
    """Non-vacuity: a renderer whose output were, say, the empty string would be trivially present
    in any document. Exercised against a copy so the committed manifest is untouched."""
    modules = dict(divergences["modules"])
    name = next(iter(modules))
    modules[name] = {**modules[name], "state": "invented"}

    assert render_divergence_register({"modules": modules}) not in eval_doc


def test_the_register_names_the_five_readouts_that_did_not_survive_the_fork(eval_doc) -> None:
    """The five a reader of the raw pipeline's contract goes looking for and does not find. Each is
    a deliberate removal with a reason, and an absence nobody wrote down is indistinguishable from a
    step that failed -- which is the whole reason the register is data rather than prose.

    Asserted against the *rendered* section rather than against the manifest, because the rendering
    is what a reader meets: an entry that existed in the file and never reached the document would
    pass a manifest-side check and leave the document silent.
    """
    section = eval_doc[eval_doc.index("## The divergence register"):]
    section = section[:section.index("\n## ")]

    for claim in (
        "analyses/coherence.py",            # the phase-domain pair, not ported at all
        "deceleration forecast skill",      # ...and the two readouts that scored a clinical trace
        "contraction-triggered response",
        "BPM_UNIT",                         # the conversion out of z units, deleted not repointed
        "physiological latency",            # the lag axis, and what it is not
    ):
        assert claim in section, claim


def test_the_register_names_every_absent_module(eval_doc, divergences) -> None:
    """The two modules the fork did not port at all, asserted by name: they are the entries a
    reader most needs, because an absence is what nobody goes looking for."""
    absent = [
        name for name, entry in divergences["modules"].items() if entry["state"] == "absent"
    ]

    assert set(absent) == {"spectra.py", "analyses/coherence.py"}
    for name in absent:
        assert f"`{name}`" in eval_doc, name


# =================================================================================================
# (g) The rules the contract has to carry
# =================================================================================================
def test_the_interpretation_rules_are_all_present(eval_doc) -> None:
    """Every rule the contract must carry, each keyed on the phrase that would have to survive a
    rewrite for the rule to still be stated. Some are the sibling's; the rest are this cell's, and
    each of those exists because a reader of the raw pipeline would otherwise carry the wrong
    contract across."""
    for rule in (
        "prediction space",                  # specificity is read there, not in KL space
        "unfloored KL",                      # only it is a rate
        "prior_variance_not_pinned",         # and only off its clamp
        "per recording",                     # the aggregation unit
        "out-of-distribution",               # every class contrast
        "healthy_no_bg_cs",                  # ...and the wider-than-expected subgroup scope
        "not comparable",                    # eval vs training metrics, and across the target axis
        "estimate, not a bound",             # the sufficiency gap
        "stored-coefficient time",           # the lag convention, and what it is not
        "coefficient-time attribution",      # ...stated as the refusal it is
        "budget-local",                      # the percentage, and the nats one step removed
        "availability clock",                # the hazard no permutation control can see
        "expected finding",                  # a small source-lag warmth is not a fault
        "/2940 rescale",                     # the rescaled per-anchor score
        "band-resolved skill, not coherence",  # what the frequency readout is and is not
    ):
        assert rule in eval_doc, f"EVAL.md no longer states: {rule!r}"


def test_the_forecast_claim_is_stated_as_exact(eval_doc) -> None:
    """The one thing this cell may say that the four two-sided ones may not, and the reason the
    sibling's refusal sentence could not simply be copied: a copied refusal would be a *false*
    disclosure rather than a conservative one."""
    assert "The forecast claim is exact" in eval_doc
    assert "one-sided" in eval_doc
    # And the narrower refusal that does survive, so the exactness is not read as a blank cheque.
    assert "791" in eval_doc


def test_the_dense_decoding_geometry_is_stated(eval_doc) -> None:
    """Every number in a run is computed over the anchors this call decodes, and the training
    tiling is not it. A document that did not say so would leave a reader reconciling an
    evaluation table against a training CSV computed over a different population."""
    assert "anchor_phase=0, anchor_stride=1" in eval_doc
    assert "137" in eval_doc
