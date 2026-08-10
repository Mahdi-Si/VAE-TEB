r"""``DESIGN.md`` and ``RESULTS.md`` must describe the module that exists, not the one they described.

A design record is only worth having if something fails when it goes stale. Two parts of it go stale
silently. The **configuration surface** drifts when a key is renamed or when a knob the document
calls deliberately absent creeps back in, and the prose still reads correctly either way -- so §13
lists both sets explicitly and this file drives them against the shipped config in both directions,
including the direction that catches a *new* model key nobody documented. And the **parameter
total** is the one absolute number the whole record pins; it is checked here against
``sum(p.numel() ...)`` on a constructed shipped-geometry model rather than against a literal in a
test, so a legitimate shared change to an imported downstream component re-costs the document rather
than failing an unrelated assertion.

``RESULTS.md`` is a skeleton the training phase fills in, so what is checkable now is its *shape*:
every study has a table with named columns, every column that names a measurement names one the task
actually emits or one the document itself defines, every arm has exactly one row and one launch
line, and every launch line points at a config file that exists. Without those, a multi-day run
would be recorded against a column nothing produces and the gap would only appear when someone tried
to read the table.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set, Tuple

import pytest
import yaml

from teb_vae.lag_attn_rws.trainer import _TRACKED_METRICS
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws

from .conftest import SHIPPED_KWARGS

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_DESIGN = _PACKAGE_DIR / "DESIGN.md"
_RESULTS = _PACKAGE_DIR / "RESULTS.md"
_CONFIG_DIR = _PACKAGE_DIR / "configs"
_CONFIG = _CONFIG_DIR / "default.yaml"

#: A dotted config path inside backticks, one per bullet, in the §13 lists. Case-sensitive on
#: purpose: the config's own key is ``VAE_model``, and a pattern that lowercased it would silently
#: drop every model key from both lists and leave the assertions below checking almost nothing.
_KEY_PATTERN = re.compile(r"^-\s+`([A-Za-z_]+(?:\.[A-Za-z_]+)+)`\s*$", re.MULTILINE)

#: The parameter total as §1 states it: ``**2,066,943 parameters**``.
_TOTAL_PATTERN = re.compile(r"\*\*([\d,]+) parameters\*\*")

#: The same number as the §5 table's total row writes it, in LaTeX.
_TOTAL_ROW_PATTERN = re.compile(r"\|\s*\*\*Total\*\*\s*\|.*?\$\\mathbf\{([\d{},]+)\}\$")

#: A backticked lowercase identifier -- a metric name, a derived quantity, or a config key.
_IDENTIFIER_PATTERN = re.compile(r"`([a-z][a-z0-9_]*)`")

#: A markdown table separator row, e.g. ``|---|---:|---|``.
_SEPARATOR_PATTERN = re.compile(r"^\|[\s:|-]+\|$")

#: Config paths as the launch lines write them.
_LAUNCH_CONFIG_PATTERN = re.compile(r"teb_vae/lag_attn_transformer_rws/configs/[\w.]+\.yaml")

#: The metric names the framework collects, without their ``train/`` or ``val/`` stage prefix. A
#: results column names a measurement, not a stage, so the comparison is against the suffixes.
TRACKED_SUFFIXES = frozenset(name.split("/")[-1] for name in _TRACKED_METRICS)

#: Every study ``RESULTS.md`` must carry a table for. A study that lost its table would leave the
#: training phase deciding what to record, which is the thing the skeleton exists to prevent.
STUDY_HEADINGS = (
    "## Parameter budget",
    "## The gradient-clipping threshold",
    "## Distributed smoke, memory and throughput",
    "## Headline baseline",
    "## Bottleneck health",
    "## Arm inventory",
    "## Phase 1",
    "## Phase 2a",
    "## Phase 2b",
    "## Phase 3",
    "## The decoder-side pair",
    "## The prior-anchor weight",
)

#: The six bottleneck-health readouts, by the name the task emits them under. All six exist from
#: epoch 0: a headline number can look healthy while the bottleneck is not -- a prior variance
#: pinned on its clamp inflates the KL, a latent collapsed into one dimension holds the total up
#: while carrying nothing, and a bounded head sitting on its bound is a mis-set hyperparameter.
BOTTLENECK_HEALTH_METRICS = (
    "source_conditioned_kl_raw",
    "kld_active_frac",
    "mu_post_prior_gap_rms",
    "logvar_prior_floor_frac",
    "mu_prior_sat_frac",
    "delta_mu_sat_frac",
)


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


def _section(text: str, heading: str) -> str:
    """The body of one ``**bold**`` subsection of §13, up to the next one or the end.

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
    """The body of one ``##`` section, up to the next ``##`` heading or the end."""
    start = text.index(heading)
    remainder = text[start + len(heading) :]
    end = remainder.find("\n## ")
    return remainder if end < 0 else remainder[:end]


def _has(config: Any, dotted: str) -> bool:
    """Whether a dotted path is present, distinguishing an explicit ``None`` from absence."""
    node = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def _table_header_rows(markdown: str) -> Iterator[str]:
    """Yield every markdown table header row -- the line immediately above a separator row.

    Column headers are where a results table names its measurements; the cells below carry file
    names, config keys and prose, none of which are claims about what the task emits.

    Args:
        markdown: The document text.

    Yields:
        One header line per table.
    """
    lines = markdown.splitlines()
    for index, line in enumerate(lines[:-1]):
        if line.startswith("|") and _SEPARATOR_PATTERN.match(lines[index + 1].strip()):
            yield line


def _declared_derived_quantities(results_text: str) -> Set[str]:
    """The column names ``RESULTS.md`` defines for itself, from its derived-quantities block.

    Args:
        results_text: The whole results document.

    Returns:
        Every backticked identifier in the bullets under ``**Derived quantities.**``.
    """
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


def _latex_int(text: str) -> int:
    """Read an integer written with LaTeX thousands separators, e.g. ``2{,}066{,}943``."""
    return int(text.replace("{", "").replace("}", "").replace(",", ""))


@pytest.fixture(scope="module")
def measured_totals() -> Dict[str, int]:
    """Parameter totals of the shipped architecture and the two Phase 1 arms, measured.

    Built from :data:`SHIPPED_KWARGS` rather than from the configs, so this test binds the
    documents to the *architecture* and ``tests/test_sweep_configs.py`` binds the arms to their
    config files -- two independent routes to the same numbers.
    """

    def total(**overrides: Any) -> int:
        model = SeqVaeLagAttnTrfRws(**dict(SHIPPED_KWARGS, **overrides))
        return sum(parameter.numel() for parameter in model.parameters())

    #: The two Phase 1 arms make the source encoder the target encoder, so their source depth is
    #: the shipped *target* depth rather than a literal -- the same rule the arm files follow.
    symmetric_depth = int(SHIPPED_KWARGS["target_attention_blocks"])

    return {
        "shipped": total(),
        "a1": total(
            encoder_conv_kernels=(),
            encoder_conv_dilations=(),
            source_attention_blocks=symmetric_depth,
            source_attention_window=None,
        ),
        "a2": total(source_attention_blocks=symmetric_depth, source_attention_window=None),
    }


# ---------------------------------------------------------------------------------------
# DESIGN.md: the configuration surface, in both directions
# ---------------------------------------------------------------------------------------
def test_the_document_lists_keys_in_both_directions(design):
    """A guard on the guard: if the extraction silently matched nothing -- or matched only the
    keys that happen to be lowercase -- every assertion below would pass on a short list."""
    required = _KEY_PATTERN.findall(_section(design, "Required"))

    assert len(required) > 40
    assert len([key for key in required if key.startswith("model_config.VAE_model.")]) > 30
    assert len(_KEY_PATTERN.findall(_section(design, "Deliberately absent"))) > 10


def test_every_key_the_document_requires_exists_in_the_shipped_config(design, shipped):
    required = _KEY_PATTERN.findall(_section(design, "Required"))

    missing = [key for key in required if not _has(shipped, key)]

    assert missing == [], f"DESIGN.md §13 requires keys the shipped config does not have: {missing}"


def test_every_model_key_in_the_shipped_config_is_documented(design, shipped):
    """The direction that catches a *new* key. A model key nobody documented is a knob whose
    meaning lives only in a YAML comment, and the signature sweep drops one that reaches nothing
    without a word -- so the run trains a different architecture than its config describes."""
    required = set(_KEY_PATTERN.findall(_section(design, "Required")))

    undocumented = [
        f"model_config.VAE_model.{key}"
        for key in shipped["model_config"]["VAE_model"]
        if f"model_config.VAE_model.{key}" not in required
    ]

    assert undocumented == [], f"DESIGN.md §13 does not document: {undocumented}"


def test_every_key_the_document_calls_absent_is_absent(design, shipped):
    """The direction that catches a knob creeping back in. Each of these is either a mechanism of
    the encoder this one replaces, or something this architecture made structural or derived -- and
    a key would read to a maintainer as a control that exists."""
    absent = _KEY_PATTERN.findall(_section(design, "Deliberately absent"))

    present = [key for key in absent if _has(shipped, key)]

    assert present == [], f"DESIGN.md §13 calls these absent but the config sets them: {present}"


def test_the_five_replaced_encoder_keys_are_documented_as_absent(design):
    """Named explicitly so a reorganisation of §13 cannot drop them. Each describes a piece of the
    encoder being replaced, and each would be dropped by the signature sweep in silence."""
    absent = _KEY_PATTERN.findall(_section(design, "Deliberately absent"))

    for key in (
        "lstm_layers",
        "encoder_extra_dilations",
        "encoder_extra_kernel",
        "conv_norm_groups",
        "causal_norm",
    ):
        assert f"model_config.VAE_model.{key}" in absent


def test_the_seven_encoder_keys_are_documented_as_required(design):
    """The configuration surface this architecture adds. Each varies across a planned arm, so an
    undocumented one is a knob an operator would not know to sweep."""
    required = _KEY_PATTERN.findall(_section(design, "Required"))

    for key in (
        "encoder_conv_kernels",
        "encoder_conv_dilations",
        "encoder_num_heads",
        "encoder_d_ff",
        "target_attention_blocks",
        "source_attention_blocks",
        "source_attention_window",
    ):
        assert f"model_config.VAE_model.{key}" in required


def test_the_reach_budget_key_is_documented_as_required(design):
    """The one config axis this module's causal standing depends on."""
    required = _KEY_PATTERN.findall(_section(design, "Required"))

    assert "model_config.VAE_model.causal_reach_budget_s" in required


# ---------------------------------------------------------------------------------------
# DESIGN.md: the parameter total, pinned once, against a constructed model
# ---------------------------------------------------------------------------------------
def test_the_documented_parameter_total_is_the_measured_one(design, measured_totals):
    """The single place the absolute number is pinned. Checked against ``sum(p.numel() ...)``
    rather than against a literal in a test, so a legitimate change to an imported downstream
    component re-costs this document rather than failing an unrelated assertion here."""
    stated = _TOTAL_PATTERN.search(design)

    assert stated is not None, "DESIGN.md §1 no longer states the parameter total"
    assert int(stated.group(1).replace(",", "")) == measured_totals["shipped"]


def test_the_budget_table_agrees_with_the_prose(design, measured_totals):
    """Two statements of one number, in two notations. Either could be edited alone."""
    row = _TOTAL_ROW_PATTERN.search(design)

    assert row is not None, "DESIGN.md §5 no longer has a total row"
    assert _latex_int(row.group(1)) == measured_totals["shipped"]


# ---------------------------------------------------------------------------------------
# DESIGN.md: the deviation record and the causality distinction
# ---------------------------------------------------------------------------------------
def test_the_document_records_deviations_and_limitations(design):
    for heading in ("## 10. Deliberate limitations", "## 11. Deviation record"):
        assert heading in design


@pytest.mark.parametrize(
    "phrase",
    [
        "final `RMSNorm(128)` per encoder",  # the deviation from the proposal's parameter arithmetic
        "constructed conditionally",          # the availability parameters
        "reachable",                          # ...and the reason that is *not* about DDP
        "ignores the loader's `weight`",      # the encoder attention's deliberate omission
        "provisional",                        # the clipping threshold
    ],
)
def test_the_deviation_record_names_each_required_deviation(design, phrase):
    """Four deviations a reader would otherwise have to rediscover from the code, and one
    provisional value. The availability entry is checked on *two* strings because the interesting
    half is the justification: a zero-multiplied parameter is reachable, so the conditions are
    parameter economy and honesty rather than DDP -- and a record that stated only the fact would
    leave the wrong reason in a reader's head."""
    record = design[design.index("## 11. Deviation record") :]

    assert phrase.lower() in record.lower(), f"DESIGN.md §11 no longer names: {phrase}"


def test_the_token_causal_distinction_is_carried_with_both_lean_limits(design):
    """The module delivers token causality and reports it as token causality. The distinction is
    the one claim a reader could take too far, and the guard bounds the leak rather than removing
    it -- so both halves are named, and both ``lean-limit`` lines are present."""
    assert "Token causality is not raw-signal causality" in design

    causality = _markdown_section(design, "## 9. ")
    assert "H_t = f(X_{\\le t})" in causality
    assert "n_{\\mathrm{raw}}(t)" in causality
    assert "quantile" in causality  # the guard bounds the leak; it does not remove it

    # Exactly two, one in §9 and one in §10: the two-sided features, and the still-unpromoted
    # shared primitives. Each is checked on the phrase that carries its *replacement trigger*,
    # which is the part of a lean-limit note that stops it becoming a permanent excuse.
    assert len(re.findall(r"^> lean-limit: ", design, re.MULTILINE)) == 2
    assert "replace with genuinely one-sided" in design
    assert "promote to a common package when" in design


def test_the_document_states_that_the_kl_is_not_transfer_entropy(design):
    """The label would assert exactly the property §9 says the inputs do not have."""
    assert "not** called transfer entropy" in design


# ---------------------------------------------------------------------------------------
# RESULTS.md: shape, columns and arms
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("heading", STUDY_HEADINGS)
def test_every_study_has_a_table_with_named_columns(results, heading):
    """A study without a table leaves the training phase deciding what to record, which is the one
    thing the skeleton exists to prevent."""
    section = _markdown_section(results, heading)

    headers = list(_table_header_rows(section))
    assert headers, f"{heading} carries no table"
    assert all(len(header.split("|")) > 2 for header in headers)


def test_every_column_names_a_metric_the_task_emits_or_a_quantity_the_document_defines(results):
    """A column nothing produces is a multi-day run recorded against a name that will be NaN in
    every row, and the gap appears only when someone tries to read the table."""
    derived = _declared_derived_quantities(results)
    named = {
        name for header in _table_header_rows(results) for name in _IDENTIFIER_PATTERN.findall(header)
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
    """The proposal's watch list in full, and all six are emitted by the task from epoch 0. A
    subset would let a run look healthy on the headline while the bottleneck was not."""
    section = _markdown_section(results, "## Bottleneck health")
    header = next(_table_header_rows(section))

    for metric in BOTTLENECK_HEALTH_METRICS:
        assert f"`{metric}`" in header, f"the bottleneck-health table dropped {metric}"
        assert metric in TRACKED_SUFFIXES, f"{metric} is not a tracked metric"


def test_every_arm_has_exactly_one_inventory_row(results):
    """Both directions: an arm missing from the inventory would be run and reported nowhere, and a
    duplicated one would be filled in twice with nothing reconciling the two."""
    inventory = _markdown_section(results, "## Arm inventory")
    arms = sorted(path.name for path in _CONFIG_DIR.glob("sweep_*.yaml"))

    rows = [line for line in inventory.splitlines() if line.startswith("| `sweep_")]
    counts = {name: sum(f"`{name}`" in row for row in rows) for name in arms}

    assert counts == {name: 1 for name in arms}
    assert len(rows) == len(arms)


def test_every_launch_line_names_a_config_that_exists(results):
    """A launch line is copied and pasted; one naming a moved or renamed file fails at the shell
    with a message about a path rather than about an arm."""
    referenced = sorted(set(_LAUNCH_CONFIG_PATTERN.findall(results)))
    repo_root = _PACKAGE_DIR.parents[1]

    assert referenced, "RESULTS.md carries no launch lines"
    missing = [path for path in referenced if not (repo_root / path).is_file()]
    assert missing == [], f"RESULTS.md names config files that do not exist: {missing}"


def test_every_arm_and_the_baseline_have_a_launch_line(results):
    """The other direction: an arm with a row and no launch line is an arm nobody can run from
    this document."""
    referenced = {Path(path).name for path in _LAUNCH_CONFIG_PATTERN.findall(results)}
    expected = {path.name for path in _CONFIG_DIR.glob("sweep_*.yaml")} | {"default.yaml"}

    assert expected <= referenced, f"no launch line for: {sorted(expected - referenced)}"


def test_the_headline_selection_rule_is_stated_before_any_table(results):
    """A stronger target prior lowers the KL without the coupling having weakened, so the headline
    number can move the wrong way on a better model. The rule and its precondition are stated at
    the top, where a reader reaches them before the first table."""
    preamble = results[: results.index("## Launch lines")]

    assert "Do not select on KL magnitude" in preamble
    assert "predictive gain" in preamble
    assert "precondition" in preamble and "nll_base_block" in preamble


def test_the_stated_arm_totals_are_the_measured_ones(results, measured_totals):
    """The Phase 1 table quotes three absolute parameter totals, which is what a reader compares
    the arms on before any run finishes."""
    phase_one = _markdown_section(results, "## Phase 1")

    for label, key in (("A1", "a1"), ("A2", "a2"), ("A3", "shipped")):
        row = next(line for line in phase_one.splitlines() if line.startswith(f"| {label} "))
        assert f"{measured_totals[key]:,}" in row, f"the {label} row quotes a stale total"


# ---------------------------------------------------------------------------------------
# The gradient-norm procedure, and the value it will replace
# ---------------------------------------------------------------------------------------
def test_the_clipping_threshold_is_a_positive_finite_float_carrying_its_status(shipped):
    """The measurement itself belongs to the training phase, so this does not gate on it -- but the
    shipped value must be usable, and the file must say whether it was measured or carried over."""
    value = shipped["advanced_config"]["trainer"]["gradient_clip_val"]

    assert isinstance(value, float)
    assert value > 0.0 and math.isfinite(value)

    text = _CONFIG.read_text(encoding="utf-8")
    marker = text[: text.index("gradient_clip_val")]
    assert "PROVISIONAL" in marker or re.search(r"q_?\d\d|percentile", marker), (
        "configs/default.yaml no longer says whether gradient_clip_val was measured or carried over"
    )


def test_the_procedure_is_executable_rather_than_a_result(results):
    """Four things a procedure needs to be followed without rediscovering it: where to launch, how
    long to run, what to read, and where the answer goes."""
    section = _markdown_section(results, "## The gradient-clipping threshold")

    assert "2,000" in section  # the step budget
    assert "train/grad_norm" in section and "pre-clip" in section  # what to read, and which form
    assert "q_{99}" in section  # which quantile sets the threshold
    assert "configs/default.yaml" in section  # where the chosen value goes


# ---------------------------------------------------------------------------------------
# No roadmap in the shipped tree
# ---------------------------------------------------------------------------------------
#: Files the roadmap-token ban does not apply to. This module names every banned token as a
#: literal, and a planning document is the artefact the ban exists to keep *out* of everything
#: else -- it does not survive the work, and a reference to it from a shipped file would dangle.
#:
#: Both roadmaps are exempt: the model's, and the evaluation package's own. They are the same kind
#: of document and they live inside the package for the same reason -- the work they describe is
#: this package's -- so a ban that caught one of them would be catching the artefact it protects
#: rather than a leak from it.
_ROADMAP_BAN_EXEMPT = frozenset(
    {Path(__file__).name, "SPEC_AND_SPRINTS.md", "EVAL_SPEC_AND_SPRINTS.md"}
)

#: Tokens no shipped file may carry: the planning document's name, its section word, and its task
#: identifiers. The word "task" is deliberately not banned -- ``task.py`` is a module of this
#: package and the design record has to name it.
_ROADMAP_TOKENS = (
    re.compile(r"SPEC_AND_SPRINTS"),
    re.compile(r"Sprint "),
    re.compile(r"S\d-T\d\d"),
)


def _roadmap_offenders() -> List[Tuple[str, str]]:
    """Return ``(relative path, matched token)`` for every shipped file naming the roadmap."""
    offenders = []
    for path in sorted([*_PACKAGE_DIR.rglob("*.py"), *_PACKAGE_DIR.rglob("*.md")]):
        if path.name in _ROADMAP_BAN_EXEMPT:
            continue
        source = path.read_text(encoding="utf-8")
        for pattern in _ROADMAP_TOKENS:
            found = pattern.search(source)
            if found is not None:
                offenders.append((str(path.relative_to(_PACKAGE_DIR)), found.group(0)))
    return offenders


def test_no_shipped_file_mentions_the_planning_document():
    """Docstrings and design records describe the model and its invariants, never the planning
    artefact that produced them -- which does not survive the module and would leave every such
    reference dangling."""
    assert _roadmap_offenders() == []


def test_the_roadmap_ban_can_fire(tmp_path, monkeypatch):
    """A three-pattern rule that silently matched nothing would pass on a tree full of references.
    Driven by pointing the walk at a directory that contains one of each."""
    (tmp_path / "leaky.py").write_text('"""See S4-T05 in SPEC_AND_SPRINTS."""\n', encoding="utf-8")
    (tmp_path / "leaky.md").write_text("Delivered in Sprint 9.\n", encoding="utf-8")
    monkeypatch.setattr("teb_vae.lag_attn_transformer_rws.tests.test_docs._PACKAGE_DIR", tmp_path)

    offenders = _roadmap_offenders()

    assert sorted(token for _, token in offenders) == ["S4-T05", "SPEC_AND_SPRINTS", "Sprint "]
