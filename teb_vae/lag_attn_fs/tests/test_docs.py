r"""``DESIGN.md`` and ``RESULTS.md`` must describe the module that exists, not the one they described.

A design record is only worth having if something fails when it goes stale, and for a model that is
a **subclass** the stale-able surface is larger than usual: almost every claim in the document is
inherited, so a change in the parent can falsify a sentence here without touching a file in this
package. Three parts of it are pinned mechanically.

The **configuration surface** drifts when a key is renamed or when a knob the document calls
deliberately absent creeps back in, and the prose reads correctly either way -- so §14 lists both
sets explicitly and this file drives them against the shipped config in both directions, including
the direction that catches a *new* model key nobody documented.

The **parameter arithmetic** is the one place absolute numbers appear. §1 states four totals and a
delta between them, and the whole point of the delta is that it is the decoder's output head and
nothing else. All four are checked against ``sum(p.numel() ...)`` on constructed models -- both this
one and the model it is compared against, at both reach budgets -- so a legitimate change to a shared
downstream component re-costs the document rather than failing an unrelated assertion, and so the
delta cannot quietly stop being what §1 says it is.

**``RESULTS.md``** is bound on shape rather than on values: every column that names a measurement
must name one the task actually emits or one the document itself defines, every arm must have exactly
one inventory row, and every launch line must point at a config that exists. Without those, a
multi-day run would be recorded against a column nothing produces and the gap would appear only when
someone tried to read the table.

The remaining tests are structural, and one is about hygiene: neither document, and no module or test
in this package, may mention the planning artefact that produced it. Docstrings and design records
describe the model and its invariants; a planning document does not survive the module and would
leave dangling references behind it.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterator, Set

import pytest
import yaml

from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
from teb_vae.lag_attn_fs.trainer import LagAttnFsTrainer
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

from .conftest import shipped_gated_kwargs

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_DIR.parents[1]
_DESIGN = _PACKAGE_DIR / "DESIGN.md"
_RESULTS = _PACKAGE_DIR / "RESULTS.md"
_CONFIG_DIR = _PACKAGE_DIR / "configs"
_CONFIG = _CONFIG_DIR / "default.yaml"

#: A dotted config path inside backticks, one per bullet, in the §14 lists. Case-sensitive on
#: purpose: the config's own key is ``VAE_model``, and a pattern that lowercased it would silently
#: drop every model key from both lists and leave the assertions below checking almost nothing.
_KEY_PATTERN = re.compile(r"^-\s+`([A-Za-z_]+(?:\.[A-Za-z_]+)+)`\s*$", re.MULTILINE)

#: A backticked lowercase identifier -- a metric name or a derived quantity. Deliberately does not
#: match a dotted path, so a config key used as a results column header is not mistaken for a metric.
_IDENTIFIER_PATTERN = re.compile(r"`([a-z][a-z0-9_]*)`")

#: A markdown table separator row, e.g. ``|---|---:|---|``.
_SEPARATOR_PATTERN = re.compile(r"^\|[\s:|-]+\|$")

#: Config paths as the launch lines write them.
_LAUNCH_CONFIG_PATTERN = re.compile(r"teb_vae/lag_attn_fs/configs/[\w.]+\.yaml")

#: Any integer of seven digits or more, in either notation the document uses: plain markdown
#: (``3,393,993``) or LaTeX with braced separators (``3{,}377{,}997``). Both appear, because a number
#: inside a maths span must brace its separators to keep the spacing, and pinning only one notation
#: would leave half of §1's arithmetic unchecked.
_LARGE_NUMBER_PATTERN = re.compile(r"(\d{1,3}(?:(?:\{,\}|,)\d{3})+)")

#: The metric names the task emits, without their ``train/`` or ``val/`` stage prefix. A results
#: column names a measurement, not a stage, so the comparison is against the suffixes.
TRACKED_SUFFIXES = frozenset(
    name.split("/")[-1] for name in LagAttnFsTrainer.TRACKED_METRICS
)

#: Every study ``RESULTS.md`` must carry a table for. A study that lost its table would leave the
#: run phase deciding what to record, which is the one thing the document exists to prevent -- and
#: with no evaluation pipeline for this package, these tables are the *only* record of a run.
STUDY_HEADINGS = (
    "## Pre-registered acceptance criteria",
    "## Before launching: what reverts, and when to stop",
    "## Parameter budget",
    "## The gradient-clipping threshold",
    "## Arm inventory",
    "## Headline baseline",
    "## Bottleneck health",
    "## Forecasting or reconstructing?",
    "## The log-variance clamp",
    "## The KL-weight sweep",
    "## The prior-anchor weight",
)

#: The six bottleneck-health readouts, by the name the task emits them under. A headline number can
#: look healthy while the bottleneck is not, and each of these is a different way for that to happen.
BOTTLENECK_HEALTH_METRICS = (
    "source_conditioned_kl_raw",
    "kld_active_frac",
    "mu_post_prior_gap_rms",
    "logvar_prior_floor_frac",
    "mu_prior_sat_frac",
    "delta_mu_sat_frac",
)

#: The four readouts this model adds and the raw-target siblings do not. Named here so a
#: reorganisation of the document cannot quietly drop the section they justify.
FORECAST_GAP_METRICS = (
    "pred_gap_tau_first",
    "pred_gap_tau_last",
    "pred_gap_st",
    "pred_gap_ph",
)

#: Tokens that name the planning artefact rather than the model. Searched in both documents and in
#: every module of the package.
_ROADMAP_TOKENS = ("SPEC_AND_SPRINTS", "S0-T0", "S2-T0", "S5-T0", "Sprint 0", "sprint plan")


@pytest.fixture(scope="module")
def design() -> str:
    return _DESIGN.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def results() -> str:
    return _RESULTS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def shipped() -> dict:
    """The shipped config, read raw.

    ``load_config`` is not used: ``default.yaml`` has no ``base:`` chain, and this asserts what is
    written in the file an operator edits.
    """
    return yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def measured_totals() -> Dict[str, int]:
    """Parameter totals of both models at both reach budgets, measured.

    Built from the production keyword set rather than from the configs, so this file binds the
    document to the *architecture* and ``test_config_load.py`` binds the config to the driver -- two
    independent routes to the same widths.
    """

    def total(cls, budget) -> int:
        model = cls(**shipped_gated_kwargs(budget))
        return sum(parameter.numel() for parameter in model.parameters())

    return {
        "fs_guarded": total(SeqVaeLagAttnFs, 120.0),
        "fs_unguarded": total(SeqVaeLagAttnFs, None),
        "rws_guarded": total(SeqVaeLagAttnRws, 120.0),
        "rws_unguarded": total(SeqVaeLagAttnRws, None),
    }


def _section(text: str, heading: str) -> str:
    """The body of one ``**bold**`` subsection of §14, up to the next one or the end.

    Args:
        text: The whole document.
        heading: The bold heading, without its asterisks.

    Returns:
        The subsection body.
    """
    start = text.index(f"**{heading}**") + len(heading) + 4
    remainder = text[start:]
    end = remainder.find("\n**")
    return remainder if end < 0 else remainder[:end]


def _markdown_section(text: str, heading: str) -> str:
    """The body of one ``##`` section, from its heading to the next one or the end."""
    start = text.index(heading)
    remainder = text[start + len(heading) :]
    end = remainder.find("\n## ")
    return remainder if end < 0 else remainder[:end]


def _flat(text: str) -> str:
    """Collapse all runs of whitespace to single spaces.

    Every phrase assertion below runs against this rather than against the raw document. The file is
    hard-wrapped at 100 columns, so any phrase long enough to be worth pinning is eventually split
    across a line by an edit elsewhere in its paragraph -- and a test that then fails is reporting a
    reflow, not a lost claim.

    Leading blockquote markers go too, and they are the reason this needs a docstring rather than a
    one-liner: a ``lean-limit`` note is a blockquote, so a phrase wrapped inside one has a stray
    ``>`` in the middle of it after a naive whitespace collapse.
    """
    unquoted = (re.sub(r"^\s*>\s?", "", line) for line in text.splitlines())
    return " ".join(" ".join(unquoted).split())


def _integers_stated_in(text: str) -> Set[int]:
    """Every large integer the text states, in either of the two notations it uses."""
    return {
        int(match.replace("{,}", "").replace(",", ""))
        for match in _LARGE_NUMBER_PATTERN.findall(text)
    }


def _states_integer(text: str, value: int) -> bool:
    """Whether the text states this integer, in either notation."""
    return value in _integers_stated_in(text)


def _has(config: Any, dotted: str) -> bool:
    """Whether a dotted path is present, distinguishing an explicit ``None`` from absence."""
    node = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def _table_header_rows(text: str) -> Iterator[str]:
    """Yield every markdown table header row -- the line immediately above a separator."""
    lines = text.splitlines()
    for index in range(len(lines) - 1):
        if lines[index].startswith("|") and _SEPARATOR_PATTERN.match(lines[index + 1].strip()):
            yield lines[index]


def _declared_derived_quantities(results_text: str) -> Set[str]:
    """The column names ``RESULTS.md`` defines for itself, from its derived-quantities block."""
    start = results_text.index("**Derived quantities.**")
    remainder = results_text[start:]
    end = remainder.find("\n## ")
    block = remainder if end < 0 else remainder[:end]
    return {
        name
        for line in block.splitlines()
        if line.startswith("- ")
        for name in _IDENTIFIER_PATTERN.findall(line)
    }


# ---------------------------------------------------------------------------------------
# DESIGN.md: the configuration surface, in both directions
# ---------------------------------------------------------------------------------------
def test_the_document_lists_keys_in_both_directions(design):
    """A guard on the guard: if the extraction silently matched nothing -- or matched only the keys
    that happen to be lowercase -- every assertion below would pass on a short list."""
    required = _KEY_PATTERN.findall(_section(design, "Required"))

    assert len(required) > 50
    assert len([key for key in required if key.startswith("model_config.VAE_model.")]) > 40
    assert len(_KEY_PATTERN.findall(_section(design, "Deliberately absent"))) > 10


def test_every_key_the_document_requires_exists_in_the_shipped_config(design, shipped):
    required = _KEY_PATTERN.findall(_section(design, "Required"))

    missing = [key for key in required if not _has(shipped, key)]

    assert missing == [], f"DESIGN.md §14 requires keys the shipped config does not have: {missing}"


def test_every_model_key_in_the_shipped_config_is_documented(design, shipped):
    """The direction that catches a *new* key. A model key nobody documented is a knob whose meaning
    lives only in a YAML comment, and the constructor's signature sweep drops one that reaches
    nothing without a word -- so the run trains a different architecture than its config describes."""
    required = set(_KEY_PATTERN.findall(_section(design, "Required")))

    undocumented = [
        f"model_config.VAE_model.{key}"
        for key in shipped["model_config"]["VAE_model"]
        if f"model_config.VAE_model.{key}" not in required
    ]

    assert undocumented == [], f"DESIGN.md §14 does not document: {undocumented}"


def test_every_key_the_document_calls_absent_is_absent(design, shipped):
    """The direction that catches a knob creeping back in. Each of these would read to a maintainer
    as a control that exists, and two of them -- the decoder width and the block split -- would be a
    *second* source of truth for a number the model derives or declares elsewhere."""
    absent = _KEY_PATTERN.findall(_section(design, "Deliberately absent"))

    present = [key for key in absent if _has(shipped, key)]

    assert present == [], f"DESIGN.md §14 calls these absent but the config sets them: {present}"


def test_the_keys_that_would_become_a_second_source_of_truth_are_documented_as_absent(design):
    """Named explicitly so a reorganisation of §14 cannot drop them.

    The decoder's width follows the target gate and is recoverable from the stamped keep-index; the
    block split is a class attribute the task verifies against the data. A config key for either
    would be a second value free to disagree with the first, and disagreement in the second case
    breaks nothing -- it mislabels two reported columns.
    """
    absent = _KEY_PATTERN.findall(_section(design, "Deliberately absent"))

    for key in ("decoder_out_channels", "target_block_split", "forecast_channels"):
        assert f"model_config.VAE_model.{key}" in absent


def test_the_plotting_block_keeps_the_inherited_name_in_both_lists(design, shipped):
    """The trap the document exists to stop a reader falling into: the shared callback reads this
    literal, so a block renamed to match this package disables the figure with no error anywhere."""
    required = _KEY_PATTERN.findall(_section(design, "Required"))
    absent = _KEY_PATTERN.findall(_section(design, "Deliberately absent"))

    assert "advanced_config.callbacks.lag_attn_rws_plotting.enabled" in required
    assert "advanced_config.callbacks.lag_attn_fs_plotting.enabled" in absent
    assert LagAttnFsTrainer.PLOT_CONFIG_KEY == "lag_attn_rws_plotting"
    assert "lag_attn_rws_plotting" in shipped["advanced_config"]["callbacks"]


def test_the_reach_budget_key_is_documented_as_required(design):
    """The one config axis that decides both this module's causal standing *and* the units of every
    number it reports."""
    required = _KEY_PATTERN.findall(_section(design, "Required"))

    assert "model_config.VAE_model.causal_reach_budget_s" in required


# ---------------------------------------------------------------------------------------
# DESIGN.md: the parameter arithmetic, pinned against constructed models
# ---------------------------------------------------------------------------------------
def test_the_documented_parameter_totals_are_the_measured_ones(design, measured_totals):
    """All four, checked against ``sum(p.numel() ...)`` rather than against literals in a test."""
    section = _markdown_section(design, "## 1. ")

    for label, value in measured_totals.items():
        assert _states_integer(section, value), (
            f"DESIGN.md §1 does not state the measured {label} total {value:,}"
        )


def test_the_documented_delta_is_the_decoder_head_and_nothing_else(design, measured_totals):
    """The claim the delta carries, checked as arithmetic rather than as prose.

    $514 \\times (C - R)$: the two per-channel output rows plus their biases, at the decoder core's
    $256$-wide hidden state. If a future change
    makes the two models differ anywhere but the decoder head, this fails rather than letting §1 keep
    asserting a decomposition that no longer holds.
    """
    section = _markdown_section(design, "## 1. ")
    guarded_delta = measured_totals["fs_guarded"] - measured_totals["rws_guarded"]
    unguarded_delta = measured_totals["fs_unguarded"] - measured_totals["rws_unguarded"]

    assert guarded_delta == 514 * (78 - 16)
    assert unguarded_delta == 514 * (109 - 16)
    for delta in (guarded_delta, unguarded_delta):
        assert _states_integer(section, delta), (
            f"DESIGN.md §1 does not state the delta {delta:,}"
        )
    assert "514" in section, "§1 no longer states the per-channel cost the delta decomposes into"


def test_the_document_warns_that_the_siblings_stated_total_is_the_unguarded_one(
    design, measured_totals
):
    """The specific way this arithmetic is easy to get wrong: the comparison model gains availability
    input adapters under a finite budget, so the number its own record states is not the baseline
    either delta is against."""
    section = _markdown_section(design, "## 1. ")
    adapter_cost = measured_totals["rws_guarded"] - measured_totals["rws_unguarded"]

    assert adapter_cost > 0
    assert _states_integer(section, adapter_cost)
    assert "unguarded" in section


# ---------------------------------------------------------------------------------------
# DESIGN.md: the claims a reader could take too far
# ---------------------------------------------------------------------------------------
def test_the_document_states_the_two_ways_the_nats_are_incomparable(design):
    """Both halves, because the second is the one a reader of the first would not guess: the reach
    budget moves the surviving-channel count, hence the decoder width, hence the block every nat is
    summed over."""
    section = _markdown_section(design, "## 5. ")

    assert "Not comparable to the raw model's" in section
    assert "Not comparable across reach budgets within this model" in section
    assert "mutually unloadable checkpoints" in section


def test_the_target_gathered_never_delayed_rule_has_its_own_section(design):
    """The sharpest correctness trap in the module: a delayed target changes *which* future the model
    is asked to forecast, and every shape downstream is identical."""
    section = _markdown_section(design, "## 7. ")

    assert "gather and not the delay" in section
    assert "all 78 surviving channels carry a non-zero delay" in section
    # The gather-before-unfold ordering, which is a memory decision rather than a correctness one.
    assert "commute" in section


def test_the_smear_argument_is_stated_with_its_not_a_leak_reasoning(design):
    """Section 8 in full: the blend fraction, why it is not leakage, why it does not bias the
    readout, and what it *does* affect. A reader who meets only the first sentence would reasonably
    conclude the model was cheating."""
    section = _markdown_section(design, "## 8. ")

    assert "not a causality violation" in section
    assert "b(\\tau, \\rho_c)" in section or "\\frac{\\rho_c - 4\\tau}{2\\rho_c}" in section
    assert "largely cancels in the difference" in section  # the shared-decoder argument
    assert "one causal budget" in section  # why the target is the gated subset
    assert "test_smear.py" in section  # the figures are reproduced by a test


def test_the_smear_section_cross_references_the_preprint(design):
    """The argument's home outside this package. The reference must name a file that exists, or the
    cross-reference is worse than none."""
    section = _markdown_section(design, "## 8. ")

    assert "doc/latex_template" in section
    assert (_REPO_ROOT / "teb_vae/lag_attn_rws/doc/latex_template/sections/reach.tex").is_file()


def test_the_four_structural_constraints_are_recorded_with_their_fixture_caveat(design):
    """All four, and the caveat that makes them meaningful: the delta heads are zero-initialised, so
    an un-perturbed model passes every KL assertion vacuously."""
    section = _markdown_section(design, "## 6. ")

    for claim in ("No decoder bypass", "Source purity", "Exact zero KL", "invoked twice"):
        assert claim in section, f"DESIGN.md §6 no longer names: {claim}"
    assert "perturb_posterior" in section
    assert "causal_norm" in section  # the step-wise causality claim's precondition


def test_the_added_readouts_are_documented_as_partial_sums(design):
    """The only property that makes them worth reporting: a second definition of the per-element term
    or the mask would let them stop being a decomposition of the number beside them."""
    section = _markdown_section(design, "## 9. ")

    for metric in FORECAST_GAP_METRICS:
        assert f"`{metric}`" in section, f"DESIGN.md §9 no longer names {metric}"
        assert metric in TRACKED_SUFFIXES, f"{metric} is not a tracked metric"
    assert "partial sums" in _flat(section)

    # The declared split, pinned against the class attribute rather than restated: it cannot be
    # derived from c_y, so the document is one of only two places the number lives.
    stated = re.search(r"TARGET_BLOCK_SPLIT\s*=\s*(\d+)", section)
    assert stated is not None, "DESIGN.md §9 no longer states the block split"
    assert int(stated.group(1)) == SeqVaeLagAttnFs.TARGET_BLOCK_SPLIT


def test_the_document_records_deviations_and_limitations(design):
    for heading in ("## 11. Deliberate limitations", "## 12. Deviation record"):
        assert heading in design


@pytest.mark.parametrize(
    "phrase",
    [
        "method, not a keyword",           # the width hook, and why it is not a constructor key
        "inherited and present",           # future_index -- absent from the design, present in fact
        "test_nets_are_framework_free",    # why the model unfolds its own stream
        "allclose",                        # the lag-map identity is round-off, not bitwise
        "did not move",                    # the clip, re-derived and unchanged
        "was wrong, and the sweep says so",  # the scale-matched beta
        "confirmed, not revised",          # logvar_clamp
    ],
)
def test_the_deviation_record_names_each_required_deviation(design, phrase):
    """Seven things a reader would otherwise have to rediscover from the code or from a run. Three of
    them are places a *stated rationale* did not survive measurement, which is the half of a deviation
    record most likely to be quietly dropped -- it reads as an admission rather than as a finding."""
    record = _flat(design[design.index("## 12. Deviation record") :])

    assert phrase.lower() in record.lower(), f"DESIGN.md §12 no longer names: {phrase}"


def test_the_lean_limits_carry_their_replacement_triggers(design):
    """A ``lean-limit`` note without a measurable trigger is a permanent excuse. Exactly two here:
    the smeared target, and the absent evaluation."""
    flat = _flat(design)

    assert len(re.findall(r"^> lean-limit: ", design, re.MULTILINE)) == 2
    assert "replace with a one-sided filter bank when" in flat
    assert "replace with the evaluation's per-recording paired statistics when" in flat


def test_the_document_states_that_this_is_an_experiment_rather_than_a_remedy(design):
    """The one framing error that would make a correct negative result read as a failure."""
    section = _markdown_section(design, "## 1. ")

    assert "experiment, not a remedy" in section
    assert "expected to reproduce" in section


def test_the_running_section_names_no_evaluation_entry_point(design):
    """The evaluation is deferred whole, and a launch line for one would be the most convincing
    possible way to imply otherwise -- so the section says the absence out loud and carries no
    command that would contradict it."""
    section = _markdown_section(design, "## 13. ")
    commands = [line for line in section.splitlines() if "-m teb_vae" in line or "python -m" in line]

    assert commands, "§13 carries no launch lines"
    assert all("trainer" in line for line in commands), (
        f"§13 names a non-trainer entry point: {[l for l in commands if 'trainer' not in l]}"
    )
    assert not any(module in section for module in ("eval.run", "eval.verify", "fs.eval"))
    assert "There is no `eval` entry point" in section


# ---------------------------------------------------------------------------------------
# RESULTS.md: shape, columns and arms
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("heading", STUDY_HEADINGS)
def test_every_study_has_a_table_with_named_columns(results, heading):
    """A study without a table leaves the run phase deciding what to record. With no evaluation
    pipeline these tables are the only durable record a run produces beyond its own CSV."""
    section = _markdown_section(results, heading)

    headers = list(_table_header_rows(section))
    assert headers, f"{heading} carries no table"
    assert all(len(header.split("|")) > 2 for header in headers)


def test_every_column_names_a_metric_the_task_emits_or_a_quantity_the_document_defines(results):
    """A column nothing produces is a run recorded against a name that will be NaN in every row, and
    the gap appears only when someone tries to read the table."""
    derived = _declared_derived_quantities(results)
    named = {
        name
        for header in _table_header_rows(results)
        for name in _IDENTIFIER_PATTERN.findall(header)
    }

    unknown = sorted(named - TRACKED_SUFFIXES - derived)

    assert unknown == [], (
        f"these columns name neither a tracked metric nor a declared derived quantity: {unknown}"
    )


def test_the_derived_quantity_block_is_not_empty(results):
    """The guard on the guard above: an extraction that matched nothing would let every column
    through as "declared"."""
    derived = _declared_derived_quantities(results)

    assert len(derived) >= 5
    assert "params" in derived and "collapsed" in derived


def test_the_bottleneck_health_table_carries_all_six_readouts(results):
    """The watch list in full. A subset would let a run look healthy on the headline while the
    bottleneck was not."""
    section = _markdown_section(results, "## Bottleneck health")
    header = next(_table_header_rows(section))

    for metric in BOTTLENECK_HEALTH_METRICS:
        assert f"`{metric}`" in header, f"the bottleneck-health table dropped {metric}"
        assert metric in TRACKED_SUFFIXES, f"{metric} is not a tracked metric"


def test_the_forecast_resolution_table_carries_the_four_added_readouts(results):
    """The section that exists because a scalar summed over 2340 coefficients cannot separate
    forecasting from reconstruction of the already-determined component."""
    section = _markdown_section(results, "## Forecasting or reconstructing?")
    header = next(_table_header_rows(section))

    for metric in FORECAST_GAP_METRICS:
        assert f"`{metric}`" in header, f"the forecast-resolution table dropped {metric}"


def test_every_arm_has_exactly_one_inventory_row(results):
    """Both directions: an arm missing from the inventory would be run and reported nowhere, and a
    duplicated one would be filled in twice with nothing reconciling the two."""
    inventory = _markdown_section(results, "## Arm inventory")
    arms = sorted(path.name for path in _CONFIG_DIR.glob("sweep_*.yaml"))

    rows = [line for line in inventory.splitlines() if line.startswith("| `sweep_")]
    counts = {name: sum(f"`{name}`" in row for row in rows) for name in arms}

    assert arms, "no sweep arms found; the glob is checking nothing"
    assert counts == {name: 1 for name in arms}
    assert len(rows) == len(arms)


def test_every_launch_line_names_a_config_that_exists(results):
    """A launch line is copied and pasted; one naming a moved or renamed file fails at the shell with
    a message about a path rather than about an arm."""
    referenced = sorted(set(_LAUNCH_CONFIG_PATTERN.findall(results)))

    assert referenced, "RESULTS.md carries no launch lines"
    missing = [path for path in referenced if not (_REPO_ROOT / path).is_file()]
    assert missing == [], f"RESULTS.md names config files that do not exist: {missing}"


def test_the_results_document_states_where_its_numbers_come_from(results):
    """Its tables are transcribed rather than generated -- there is no evaluation pass to generate
    them -- so the three sourcing rules that make the copy safe are load-bearing here, not optional."""
    assert "metrics_history.csv" in results
    assert "resolved_config.yaml" in results  # rows keyed from the run's own record
    assert "marked" in results  # a collapsed arm is marked, not dropped


def test_the_results_document_carries_the_readings_that_survive_a_missing_evaluation(results):
    """Four statements a reader must meet before the first table, because each is a way to misread
    every table after it."""
    assert "negative `pred_gap` is a PASS" in results
    assert "Do not select on KL magnitude" in results
    assert "in-sample" in results
    assert "no evaluation pipeline for this package" in results


# ---------------------------------------------------------------------------------------
# No planning artefact in the shipped tree
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("document", ("DESIGN.md", "RESULTS.md"))
def test_the_records_do_not_mention_the_planning_document(document):
    """A design record describes the model and its invariants, never the artefact that produced it --
    which does not survive the module and would leave dangling references."""
    text = (_PACKAGE_DIR / document).read_text(encoding="utf-8")

    for token in _ROADMAP_TOKENS:
        assert token not in text, f"{document} mentions {token!r}"


def test_no_module_or_test_in_the_package_mentions_the_planning_document():
    """The same rule, across the code. Checked here because this is the file about documentation
    staying honest, and the reference most likely to appear is in a docstring.

    This file is excluded because the tokens it searches for are literals in it -- the one place they
    legitimately appear.
    """
    offenders = []
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        if path == Path(__file__).resolve():
            continue
        source = path.read_text(encoding="utf-8")
        for token in _ROADMAP_TOKENS:
            if token in source:
                offenders.append(f"{path.relative_to(_PACKAGE_DIR)}: {token}")

    assert offenders == []
