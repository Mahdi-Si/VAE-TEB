r"""``DESIGN.md`` must describe the module that exists, not the one it described when written.

A design record is only worth having if something fails when it goes stale, and the part of it
most likely to go stale silently is the configuration surface: a key gets renamed, or a knob the
document says is deliberately absent creeps back in, and the prose still reads correctly. So §14
lists both sets explicitly and this file drives them against the shipped config.

The remaining tests are structural: the fourteen-item authoring checklist must still have fourteen
items, the deviation record must be present, and the document must not mention the roadmap that
produced it -- docstrings and design records describe the model and its invariants, never the
planning artefacts.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_DESIGN = _PACKAGE_DIR / "DESIGN.md"
_CONFIG = _PACKAGE_DIR / "configs" / "default.yaml"

#: A dotted config path inside backticks, one per bullet, in the §14 lists. Case-sensitive on
#: purpose: the config's own key is ``VAE_model``, and a pattern that lowercased it would silently
#: drop every model key from both lists and leave the tests below asserting almost nothing.
_KEY_PATTERN = re.compile(r"^-\s+`([A-Za-z_]+(?:\.[A-Za-z_]+)+)`\s*$", re.MULTILINE)


@pytest.fixture(scope="module")
def design() -> str:
    return _DESIGN.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def shipped() -> dict:
    """The shipped config, read raw. ``load_config`` is not used: ``default.yaml`` has no
    ``base:`` chain, and this asserts what is written in the file operators edit."""
    return yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))


def _section(text: str, heading: str) -> str:
    """The body of one ``**bold**`` subsection of §14, up to the next one or the end."""
    start = text.index(f"**{heading}**") + len(heading) + 4
    remainder = text[start:]
    end = remainder.find("\n**")
    return remainder if end < 0 else remainder[:end]


def _has(config, dotted: str) -> bool:
    """Whether a dotted path is present, distinguishing an explicit ``None`` from absence."""
    node = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


# ---------------------------------------------------------------------------------------
# The configuration surface
# ---------------------------------------------------------------------------------------
def test_the_document_lists_keys_in_both_directions(design):
    """A guard on the guard: if the extraction silently matched nothing -- or matched only the
    keys that happen to be lowercase -- every assertion below would pass on a short list."""
    required = _KEY_PATTERN.findall(_section(design, "Required"))

    assert len(required) > 25
    assert len([key for key in required if key.startswith("model_config.VAE_model.")]) > 10
    assert len(_KEY_PATTERN.findall(_section(design, "Deliberately absent"))) > 5


def test_every_key_the_document_requires_exists_in_the_shipped_config(design, shipped):
    required = _KEY_PATTERN.findall(_section(design, "Required"))

    missing = [key for key in required if not _has(shipped, key)]

    assert missing == [], f"DESIGN.md §14 requires keys the shipped config does not have: {missing}"


def test_every_key_the_document_calls_absent_is_absent(design, shipped):
    """The direction that catches a knob creeping back in. Each of these is either a sibling
    mechanism this architecture has no code for, or something this net made unconditional -- and a
    key would read to a maintainer as a control that exists."""
    absent = _KEY_PATTERN.findall(_section(design, "Deliberately absent"))

    present = [key for key in absent if _has(shipped, key)]

    assert present == [], f"DESIGN.md §14 calls these absent but the config sets them: {present}"


def test_the_reach_budget_key_is_documented_as_required(design):
    """The one config axis this module's causal standing depends on. Named explicitly so a
    reorganisation of §14 cannot drop it without failing."""
    required = _KEY_PATTERN.findall(_section(design, "Required"))

    assert "model_config.VAE_model.causal_reach_budget_s" in required


# ---------------------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------------------
def test_the_authoring_checklist_is_verified_item_by_item(design):
    """Fourteen items in ``train/MODEL_MIGRATION_GUIDE.md`` §3.9, and fourteen verdicts here."""
    checklist = design[design.index("## 13. Closeout") :]
    rows = [line for line in checklist.splitlines() if line.startswith("| ") and "| yes" in line]

    assert len(rows) == 15  # fourteen checklist items plus the "four verifications" row


def test_the_document_records_deviations_and_limitations(design):
    for heading in ("## 11. Deliberate limitations", "## 12. Deviation record"):
        assert heading in design


def test_the_interpretation_traps_are_recorded(design):
    """The two a figure or a headline number can get wrong without anything failing."""
    traps = design[design.index("## 8.") : design.index("## 9.")]

    assert "source_conditioned_kl_raw" in traps  # only the unfloored KL is an information rate
    assert "compensated" in traps and "20" in traps  # the two lag quantities
    assert "logvar_prior_floor_frac" in traps  # the clamp that inflates the readout


def test_the_context_sufficiency_limitation_is_carried(design):
    """The bullet must say what was measured and what is still unmeasured about it.

    It began life as "the gap is not measured", and the evaluation's oracle probe now measures it.
    A bullet left in the first form would read as a standing gap in the work; one saying only
    "measured" would drop the part that still matters -- the estimate's two bias directions oppose
    and neither is quantified, so it is not a bound and nothing downstream may treat it as one.
    """
    limitations = design[design.index("## 11.") : design.index("## 12.")]

    assert "lean-limit" in limitations
    assert "context-sufficiency" in limitations
    # What was measured, and by what.
    assert "oracle" in limitations and "target_state" in limitations
    # What remains unmeasured: both directions, and the refusal to call it a bound.
    assert "biases the gap down" in limitations and "biases it up" in limitations
    assert "not a bound" in limitations


def test_the_limitations_record_what_the_evaluation_closed_and_what_it_did_not(design):
    """A limitations section that only ever grows is a section nobody reads. Once the evaluation
    landed, several of these bullets stopped being true -- and the ones that did *not* have to
    stay, by name, or they become oversights rather than decisions."""
    limitations = design[design.index("## 11.") : design.index("## 12.")]

    # The evaluation is no longer "the minimum readout set only".
    assert "minimum readout set" not in limitations
    # What it closed, and what it deliberately did not -- each named rather than implied.
    for absence in (
        "Lag ablation",          # blocked model-side, with the reason restated
        "Necessity is not measured",
        "raw target history",    # the oracle's conditioning, hence estimate not bound
        # Answered by the coherence analysis; what remains is below delta-f and time-resolved.
        "spectral analysis",
        "distributional distances",
        "held-out clinical discrimination",
    ):
        assert absence.lower() in limitations.lower(), f"DESIGN.md §11 no longer names: {absence}"


def test_the_spectral_limitation_records_what_the_coherence_analysis_closed(design):
    """The one bullet in §11 whose stated reason has since been *answered* rather than merely
    outlived.

    A limitation that is quietly deleted the day it stops binding leaves no record that it ever
    did, and a reader of the next spectral question re-derives the same objection from scratch. A
    limitation left standing after it has been closed is worse: it reads as an oversight and sends
    that reader looking for an analysis that already exists. So the bullet has to do both -- name
    what closed it, and name what genuinely remains.
    """
    limitations = design[design.index("## 11.") : design.index("## 12.")].lower()

    # What closed it, and the resolution that did.
    assert "coherence" in limitations
    assert "7.8" in limitations
    assert "tau" in limitations or r"\tau" in limitations
    # And the three absences that survive, each named rather than implied.
    assert "non-stationary" in limitations
    assert "absolute band power" in limitations


def test_the_closeout_names_the_evaluation_commands(design):
    """§13's command block is where an operator looks for how to run this module. The gate was
    there; the evaluation and its acceptance check were not, so a finished checkpoint had no
    documented path to a verified number."""
    closeout = design[design.index("## 13.") : design.index("## 14.")]

    assert "teb_vae.lag_attn_rws.eval.run" in closeout
    assert "teb_vae.lag_attn_rws.eval.verify" in closeout
    assert "EVAL.md" in closeout


def test_the_results_document_names_the_command_that_fills_it():
    """Its tables are generated, not transcribed, and the three sourcing rules that make the copy
    safe are what stop a renamed directory relabelling a measurement."""
    results = (_PACKAGE_DIR / "RESULTS.md").read_text(encoding="utf-8")

    assert "eval.verify --runs" in results
    assert "metrics_history.csv" in results
    # Collapsed arms marked rather than dropped, and which pred_gap the tables carry.
    assert "marked" in results and "pred_gap_mc_nats" in results


# ---------------------------------------------------------------------------------------
# No roadmap in the shipped tree
# ---------------------------------------------------------------------------------------
def test_the_design_record_does_not_mention_the_roadmap(design):
    """A design record describes the model and its invariants, never the planning artefact that
    produced it -- which does not survive the module and would leave dangling references."""
    for token in ("SPEC_AND_SPRINTS", "Sprint ", "S5-T", "sprint 5"):
        assert token not in design, f"DESIGN.md mentions {token!r}"


def test_no_module_or_test_in_the_package_mentions_the_roadmap():
    """The same rule, across the code. Checked here because this is the file about documentation
    staying honest, and the roadmap reference most likely to appear is in a docstring.

    This file is excluded because the tokens it searches for are literals in it -- the one place
    they legitimately appear.
    """
    offenders = []
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        if path == Path(__file__).resolve():
            continue
        source = path.read_text(encoding="utf-8")
        for token in ("SPEC_AND_SPRINTS", "S5-T0", "S0-T0", "Sprint 0", "sprint plan"):
            if token in source:
                offenders.append(f"{path.relative_to(_PACKAGE_DIR)}: {token}")

    assert offenders == []
