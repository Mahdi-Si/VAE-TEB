r"""``DESIGN.md`` and ``RESULTS.md`` must describe the package that exists, not the one they described.

This package has no evaluation pipeline, so a run produces exactly two durable artefacts: its own
``train_results/metrics_history.csv`` and the table someone transcribes from it into ``RESULTS.md``.
There is no verdict file, no bootstrap interval and no per-recording table to catch a mistake in the
copy. That makes the document the record, and a record with no test is a record that goes stale in
the direction nobody notices.

``DESIGN.md`` has the larger stale-able surface, because this model is assembled entirely out of two
parents and almost every claim in it is therefore inherited: a change in either parent can falsify a
sentence here without touching a file in this package. Four parts of it are pinned mechanically.

**The configuration surface**, in both directions -- every key §18 requires must exist in the shipped
config, every key it calls deliberately absent must not, and every ``model_config.VAE_model`` key the
config carries must be documented. The prose reads correctly either way, so nothing else would catch
a renamed key or a knob creeping back in.

**The three linearisations.** The model, the task and the driver are each written out in the document
as an arrow chain, and each is compared against the real ``__mro__``. A diamond that silently
reordered would change what the model trains on and raise nothing, and a document that recorded the
old order would be the only place a reader could go to find out.

**The parameter arithmetic.** Eight totals and four deltas, checked against ``sum(p.numel() ...)`` on
constructed models rather than against literals here, so a legitimate change to a shared imported
component re-costs the document instead of failing an unrelated assertion. Two of the deltas carry a
claim -- the encoder axis is the two history encoders and the target axis is the decoder's output
head -- and the arithmetic is what keeps those claims true rather than merely written.

**The claims a reader could take too far**, each asserted against the code it describes rather than
against itself: the unconditional causality claim against the constructor signature that makes it
unconditional, the block split against the class attribute, and the sibling that did *not* gain the
width seam against the class that does not carry it.

``RESULTS.md`` is pinned on three of its own.

**The column set.** Every backticked lowercase identifier in a table header must name a metric the
task genuinely emits or a quantity the document itself defines in its derived-quantities block. A
column naming neither is a run recorded against a name that would be NaN in every row, and the gap
appears only when someone tries to read the table -- typically after the multi-day run that was
supposed to fill it.

**The parameter arithmetic.** Six totals and two deltas, checked against ``sum(p.numel() ...)`` on
constructed models rather than against literals here, so a legitimate change to a shared downstream
component re-costs the document instead of failing an unrelated assertion. Both deltas carry a claim
-- the encoder axis is the two history encoders and the target axis is the decoder's output head --
and the arithmetic is what keeps those claims true rather than merely written.

**The readings that survive a missing evaluation.** Four statements a reader must meet before the
first table, because each is a way to misread every table after it, plus the two shipped-package
edits whose revert path is code rather than configuration. Neither sibling has an analogue for the
second: this is the only package in the family that edited another package's net layer to exist, and
one of those edits sits underneath a completed four-arm result.

The remaining tests are structural, and one is about hygiene: neither the document nor any module or
test in this package may mention the planning artefact that produced it. A planning document does not
survive the module and would leave dangling references behind it.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

import pytest
import yaml

from teb_vae.lag_attn_fs.nets.feature_target import FeatureForecastTarget
from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
from teb_vae.lag_attn_rws import collapse
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E
from teb_vae.lag_attn_transformer_fs.nets.model import SeqVaeLagAttnTrfFs
from teb_vae.lag_attn_transformer_fs.task import SeqVaeLagAttnTrfFsTask
from teb_vae.lag_attn_transformer_fs.trainer import LagAttnTrfFsTrainer
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws

from .conftest import shipped_gated_kwargs
from teb_vae.lag_attn_fs.tests.conftest import shipped_gated_kwargs as sibling_gated_kwargs

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_DIR.parents[1]
_DESIGN = _PACKAGE_DIR / "DESIGN.md"
_RESULTS = _PACKAGE_DIR / "RESULTS.md"
_CONFIG = _PACKAGE_DIR / "configs" / "default.yaml"

#: A backticked lowercase identifier -- a metric name or a derived quantity. Deliberately does not
#: match a dotted path, so a config key used as a table header is not mistaken for a metric.
_IDENTIFIER_PATTERN = re.compile(r"`([a-z][a-z0-9_]*)`")

#: The target-axis delta as ``DESIGN.md`` §13 factorises it: ``514 \times (78 - 16)``. Captured so
#: the stated arithmetic can be *evaluated* rather than merely found -- a section carrying the right
#: delta beside a wrong factorisation of it is the failure a search for the number cannot see.
_HEAD_COST_PATTERN = re.compile(r"(\d+) \\times \((\d+) - (\d+)\)")

#: A dotted config path inside backticks, one per bullet, in the ``DESIGN.md`` §18 lists.
#: Case-sensitive on purpose: the config's own key is ``VAE_model``, and a pattern that lowercased it
#: would silently drop every model key from both lists and leave the assertions below checking almost
#: nothing.
_KEY_PATTERN = re.compile(r"^-\s+`([A-Za-z_]+(?:\.[A-Za-z_]+)+)`\s*$", re.MULTILINE)

#: A markdown table separator row, e.g. ``|---|---:|---|``.
_SEPARATOR_PATTERN = re.compile(r"^\|[\s:|-]+\|$")

#: Config paths as the launch lines write them.
_LAUNCH_CONFIG_PATTERN = re.compile(r"teb_vae/lag_attn_transformer_fs/configs/[\w.]+\.yaml")

#: Any integer of seven digits or more, in either notation the document uses: plain markdown
#: (``2,089,211``) or LaTeX with braced separators (``1{,}304{,}782``). Both appear, because a number
#: inside a maths span must brace its separators to keep the spacing, and pinning only one notation
#: would leave half of the arithmetic unchecked.
_LARGE_NUMBER_PATTERN = re.compile(r"(\d{1,3}(?:(?:\{,\}|,)\d{3})+)")

#: A signed decimal as the tables write it, including the unicode minus the document uses in prose.
_NUMBER_PATTERN = re.compile(r"[+\-−]?\d+(?:\.\d+)?")

#: The metric names the task emits, without their ``train/`` or ``val/`` stage prefix. A results
#: column names a measurement, not a stage, so the comparison is against the suffixes.
TRACKED_SUFFIXES = frozenset(
    name.split("/")[-1] for name in LagAttnTrfFsTrainer.TRACKED_METRICS
)

#: Every section ``RESULTS.md`` must carry, and each is a table a run fills in. A section that
#: disappeared would leave the run phase deciding what to record, which is the one thing the document
#: exists to prevent -- and with no evaluation pipeline these tables are the only record a run
#: produces beyond its own CSV.
STUDY_HEADINGS = (
    "## Pre-registered acceptance criteria",
    "## Before launching: what reverts, and when to stop",
    "## Parameter budget",
    "## The loss-scale constants",
    "## Distributed smoke, memory and throughput",
    "## Headline baseline",
    "## Bottleneck health",
    "## Forecasting or reconstructing?",
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

#: The four readouts the feature target adds and the raw-target siblings do not. Named here so a
#: reorganisation of the document cannot quietly drop the section they justify.
FORECAST_GAP_METRICS = (
    "pred_gap_tau_first",
    "pred_gap_tau_last",
    "pred_gap_st",
    "pred_gap_ph",
)

#: Tokens that name the planning artefact rather than the model. Searched in both documents and in
#: every module of the package.
_ROADMAP_TOKENS = (
    "SPEC_AND_SPRINTS",
    "S0-T0",
    "S1-T0",
    "S2-T0",
    "S3-T0",
    "S4-T0",
    "Sprint 0",
    "sprint plan",
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


@pytest.fixture(scope="module")
def measured_totals() -> Dict[str, int]:
    """Parameter totals of all three models at both reach budgets, measured.

    Built from each suite's own production keyword set rather than from the configs, so this file
    binds the document to the *architectures* while ``test_config_load.py`` binds the configs to the
    driver -- two independent routes to the same widths. The two comparison models are built from
    different keyword sets by necessity: the conv-LSTM constructor takes five keywords this one
    refuses.
    """

    def total(cls, kwargs) -> int:
        return sum(parameter.numel() for parameter in cls(**kwargs).parameters())

    return {
        "trf_fs_guarded": total(SeqVaeLagAttnTrfFs, shipped_gated_kwargs(120.0)),
        "trf_fs_unguarded": total(SeqVaeLagAttnTrfFs, shipped_gated_kwargs(None)),
        "trf_rws_guarded": total(SeqVaeLagAttnTrfRws, shipped_gated_kwargs(120.0)),
        "trf_rws_unguarded": total(SeqVaeLagAttnTrfRws, shipped_gated_kwargs(None)),
        "fs_guarded": total(SeqVaeLagAttnFs, sibling_gated_kwargs(120.0)),
        "fs_unguarded": total(SeqVaeLagAttnFs, sibling_gated_kwargs(None)),
    }


@pytest.fixture(scope="module")
def measured_raw_lstm_totals() -> Dict[str, int]:
    """The fourth cell of the grid, at both reach budgets, measured.

    Separate from :func:`measured_totals` rather than merged into it, because ``RESULTS.md`` quotes
    six totals and ``DESIGN.md`` quotes eight: the design record draws the whole $2 \\times 2$ so the
    encoder delta can be read at a fixed target *and* at the fixed other target, and a fixture that
    forced both documents to state the same set would make one of them state a number it has no use
    for.
    """

    def total(budget) -> int:
        model = SeqVaeLagAttnRws(**sibling_gated_kwargs(budget))
        return sum(parameter.numel() for parameter in model.parameters())

    return {"rws_guarded": total(120.0), "rws_unguarded": total(None)}


def _markdown_section(text: str, heading: str) -> str:
    """The body of one ``##`` section, from its heading to the next one or the end."""
    start = text.index(heading)
    remainder = text[start + len(heading) :]
    end = remainder.find("\n## ")
    return remainder if end < 0 else remainder[:end]


def _bold_section(text: str, heading: str) -> str:
    """The body of one ``**bold**`` subsection, up to the next one or the end.

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


def _flat(text: str) -> str:
    """Collapse all runs of whitespace to single spaces.

    Every phrase assertion below runs against this rather than against the raw document. The files
    are hard-wrapped at 100 columns, so any phrase long enough to be worth pinning is eventually
    split across a line by an edit elsewhere in its paragraph -- and a test that then fails is
    reporting a reflow, not a lost claim.

    Leading blockquote markers go too: a ``lean-limit`` note is a blockquote, so a phrase wrapped
    inside one would carry a stray ``>`` in the middle of it after a naive whitespace collapse.
    """
    unquoted = (re.sub(r"^\s*>\s?", "", line) for line in text.splitlines())
    return " ".join(" ".join(unquoted).split())


def _has(config: Any, dotted: str) -> bool:
    """Whether a dotted path is present, distinguishing an explicit ``None`` from absence."""
    node = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def _linearisation_stated_for(text: str, cls: type) -> List[str]:
    """The MRO the document writes out for a class, as a list of class names.

    Reads the arrow chain the document states -- ``` `A -> B -> C` ``` -- and returns
    ``["A", "B", "C"]``. The chain is allowed to be shorter than the real MRO and to wrap across a
    line, which it does for the two Lightning-side diamonds.

    Args:
        text: The whole document.
        cls: The class whose chain to find, matched by name at the chain's head.

    Returns:
        The names in the order the document states them.
    """
    stated = re.search(rf"`{cls.__name__} -> ([^`]+)`", text)
    assert stated is not None, f"the document states no linearisation for {cls.__name__}"
    return [cls.__name__] + [name.strip() for name in stated.group(1).split("->")]


def _integers_stated_in(text: str) -> Set[int]:
    """Every large integer the text states, in either of the two notations it uses."""
    return {
        int(match.replace("{,}", "").replace(",", ""))
        for match in _LARGE_NUMBER_PATTERN.findall(text)
    }


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


def _row_cells(text: str, label: str) -> Optional[List[str]]:
    """The cells of the one table row whose first column contains ``label``.

    Args:
        text: The section to search.
        label: A substring of the row's leading cell.

    Returns:
        The row's cells with surrounding whitespace stripped, or ``None`` if no row matches.
    """
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and label in cells[0]:
            return cells
    return None


def _number(cell: str) -> Optional[float]:
    """The first signed decimal in a table cell, with the document's unicode minus normalised."""
    match = _NUMBER_PATTERN.search(cell.replace("**", "").replace(",", ""))
    if match is None:
        return None
    return float(match.group(0).replace("−", "-"))


def _reading(cells: List[str], index: int, name: str) -> float:
    """One filled-in table cell, as a number.

    Skips the test rather than failing when the cell is blank: a table nobody has transcribed a run
    into yet is the state this document ships in, and a suite that failed on it would be reporting
    the absence of a run as a defect in the document.

    Args:
        cells: The row's cells.
        index: The column to read.
        name: The column's name, for the skip message.

    Returns:
        The cell's value.
    """
    value = _number(cells[index]) if index < len(cells) else None
    if value is None:
        pytest.skip(f"the recorded row carries no {name} yet")
    return value


# ---------------------------------------------------------------------------------------
# DESIGN.md: the configuration surface, in both directions
# ---------------------------------------------------------------------------------------
def test_the_design_lists_keys_in_both_directions(design):
    """A guard on the guard: if the extraction silently matched nothing -- or matched only the keys
    that happen to be lowercase -- every assertion below would pass on a short list."""
    required = _KEY_PATTERN.findall(_bold_section(design, "Required"))

    assert len(required) > 50
    assert len([key for key in required if key.startswith("model_config.VAE_model.")]) > 40
    assert len(_KEY_PATTERN.findall(_bold_section(design, "Deliberately absent"))) > 10


def test_every_key_the_design_requires_exists_in_the_shipped_config(design, shipped):
    required = _KEY_PATTERN.findall(_bold_section(design, "Required"))

    missing = [key for key in required if not _has(shipped, key)]

    assert missing == [], f"DESIGN.md §18 requires keys the shipped config does not have: {missing}"


def test_every_model_key_in_the_shipped_config_is_documented(design, shipped):
    """The direction that catches a *new* key. A model key nobody documented is a knob whose meaning
    lives only in a YAML comment, and the constructor's signature sweep drops one that reaches
    nothing without a word -- so the run trains a different architecture than its config describes."""
    required = set(_KEY_PATTERN.findall(_bold_section(design, "Required")))

    undocumented = [
        f"model_config.VAE_model.{key}"
        for key in shipped["model_config"]["VAE_model"]
        if f"model_config.VAE_model.{key}" not in required
    ]

    assert undocumented == [], f"DESIGN.md §18 does not document: {undocumented}"


def test_every_key_the_design_calls_absent_is_absent(design, shipped):
    """The direction that catches a knob creeping back in. Each of these would read to a maintainer
    as a control that exists."""
    absent = _KEY_PATTERN.findall(_bold_section(design, "Deliberately absent"))

    present = [key for key in absent if _has(shipped, key)]

    assert present == [], f"DESIGN.md §18 calls these absent but the config sets them: {present}"


def test_the_five_replaced_encoder_keys_are_documented_as_absent(design):
    """Named explicitly so a reorganisation of §18 cannot drop them. Each describes a piece of the
    encoder this model does not have, and each would be dropped by the signature sweep in silence --
    so a config block copy-pasted from the conv-LSTM feature sibling would read correct and build a
    different model."""
    absent = _KEY_PATTERN.findall(_bold_section(design, "Deliberately absent"))
    constructor_keys = set(inspect.signature(SeqVaeLagAttnTrfFs.__init__).parameters)

    for key in (
        "lstm_layers",
        "encoder_extra_dilations",
        "encoder_extra_kernel",
        "conv_norm_groups",
        "causal_norm",
    ):
        assert f"model_config.VAE_model.{key}" in absent
        assert key not in constructor_keys, f"{key} is a keyword of this constructor after all"


def test_the_two_keys_that_would_become_a_second_source_of_truth_are_documented_as_absent(design):
    """The decoder's width follows the target gate and is recoverable from the stamped keep-index;
    the block split is a class attribute the task verifies against the data. A config key for either
    would be a second value free to disagree with the first.

    ``decoder_out_channels`` is asserted absent from the *constructor* as well, which is the stronger
    half and the one that makes the exclusion list of the parity pins twelve names rather than
    thirteen: it is not a keyword here at all, so no config could reach it even by accident.
    """
    absent = _KEY_PATTERN.findall(_bold_section(design, "Deliberately absent"))

    for key in ("decoder_out_channels", "target_block_split"):
        assert f"model_config.VAE_model.{key}" in absent
    assert (
        "decoder_out_channels" not in inspect.signature(SeqVaeLagAttnTrfFs.__init__).parameters
    )


def test_the_seven_encoder_keys_are_documented_as_required(design):
    """The configuration surface this architecture has and the conv-LSTM feature sibling does not.
    Each is inherited from the raw domain where it was swept, so an undocumented one is a knob a
    reader would not know was already answered elsewhere."""
    required = _KEY_PATTERN.findall(_bold_section(design, "Required"))
    constructor_keys = set(inspect.signature(SeqVaeLagAttnTrfFs.__init__).parameters)

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
        assert key in constructor_keys, f"{key} names no constructor argument"


def test_the_reach_budget_key_is_documented_as_required(design):
    """The one config axis that decides this module's causal standing, the decoder's width *and* the
    units of every number it reports."""
    required = _KEY_PATTERN.findall(_bold_section(design, "Required"))

    assert "model_config.VAE_model.causal_reach_budget_s" in required


def test_the_plotting_block_keeps_the_shared_drivers_name_in_both_lists(design, shipped):
    """The trap the document exists to stop a reader falling into: the shared callback assembly reads
    this literal, so a block renamed to match this package disables the figure with no error
    anywhere."""
    required = _KEY_PATTERN.findall(_bold_section(design, "Required"))
    absent = _KEY_PATTERN.findall(_bold_section(design, "Deliberately absent"))

    assert "advanced_config.callbacks.lag_attn_rws_plotting.enabled" in required
    assert "advanced_config.callbacks.lag_attn_transformer_fs_plotting.enabled" in absent
    assert LagAttnTrfFsTrainer.PLOT_CONFIG_KEY == "lag_attn_rws_plotting"
    assert "lag_attn_rws_plotting" in shipped["advanced_config"]["callbacks"]


# ---------------------------------------------------------------------------------------
# DESIGN.md: the three linearisations, against the real MROs
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cls", [SeqVaeLagAttnTrfFs, SeqVaeLagAttnTrfFsTask, LagAttnTrfFsTrainer]
)
def test_each_documented_linearisation_is_the_real_one(design, cls):
    """Three arrow chains, each compared against ``__mro__`` rather than against prose.

    Every one of them is a diamond whose two branches were measured to be disjoint *today*, which is
    a fact about today's code rather than a property of the construction -- so a silent reorder is
    exactly the failure mode, and it would change what the model trains on while raising nothing.
    The chains are allowed to stop short of ``object``; what is compared is the prefix they state.
    """
    stated = _linearisation_stated_for(design, cls)
    real = [base.__name__ for base in cls.__mro__]

    assert stated == real[: len(stated)], (
        f"DESIGN.md states {stated} for {cls.__name__}; the real MRO begins {real[: len(stated)]}"
    )


def test_the_model_class_really_defines_nothing(design):
    """The claim the whole record rests on: with an empty class body the twenty forward keys, the
    posterior structure, the lag map and the metric set cannot have moved, because they are the
    parents' own code objects. Checked as a fact about the class, not as a sentence."""
    own = set(vars(SeqVaeLagAttnTrfFs)) - {"__doc__", "__module__", "__dict__", "__weakref__"}

    assert own == set(), f"SeqVaeLagAttnTrfFs defines {sorted(own)}"
    assert "empty class body" in _flat(_markdown_section(design, "## 1. "))


def test_the_mixin_section_states_why_neither_inheritance_works(design):
    """The measured reason this is a mixin rather than two inheritances -- that
    ``(SeqVaeLagAttnFs, SeqVaeLagAttnTrfRws)`` runs the conv-LSTM constructor -- and the two
    properties that keep it a move: the base order, and the absent ``__init__``."""
    section = _flat(_markdown_section(design, "## 6. "))

    assert "move, not an abstraction" in section
    assert "order of the bases is load-bearing" in section
    assert "conv-LSTM constructor" in section
    # The mixin really is a plain object with no constructor of its own, which is what keeps the
    # signature sweep seeing the architecture parent's parameters.
    assert FeatureForecastTarget.__bases__ == (object,)
    assert "__init__" not in vars(FeatureForecastTarget)


def test_the_diamond_section_names_where_each_inherited_thing_resolves(design):
    """Four attributes and one method, each asserted where §7 says it comes from. Two of the three
    re-pointed class attributes are silent when wrong -- the driver would build a conv-LSTM model, or
    write another model's checkpoint stem into a shared output tree."""
    section = _markdown_section(design, "## 7. ")

    assert LagAttnTrfFsTrainer.CHECKPOINT_STEM == "lag-attn-trf-fs"
    assert '"lag-attn-trf-fs"' in section
    assert LagAttnTrfFsTrainer.MODEL_CLS is SeqVaeLagAttnTrfFs
    assert LagAttnTrfFsTrainer.TASK_CLS is SeqVaeLagAttnTrfFsTask
    assert LagAttnTrfFsTrainer.TARGET_FIELDS == ("fhr_st", "fhr_ph")
    assert '`TARGET_FIELDS = ("fhr_st", "fhr_ph")`' in section
    assert len(LagAttnTrfFsTrainer.TRACKED_METRICS) == 78
    assert "78 entries" in section


def test_the_compile_resolution_is_recorded_as_a_decision(design):
    """It arrives by resolution order rather than by anything written down: the feature parent does
    not define the hook, so lookup passes through and ``torch.compile`` becomes permitted on a model
    whose feature-domain ancestor never exercised it."""
    from teb_vae.lag_attn_fs.trainer import LagAttnFsTrainer
    from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer

    section = _flat(_markdown_section(design, "## 7. "))

    assert "compile_model_requested" not in vars(LagAttnFsTrainer)
    assert (
        LagAttnTrfFsTrainer.compile_model_requested
        is LagAttnTrfRwsTrainer.compile_model_requested
    )
    assert "resolves to the conv-Transformer side, and that is a decision" in section


# ---------------------------------------------------------------------------------------
# DESIGN.md: the parameter arithmetic, pinned against constructed models
# ---------------------------------------------------------------------------------------
def test_the_design_parameter_totals_are_the_measured_ones(
    design, measured_totals, measured_raw_lstm_totals
):
    """All eight cells of the grid, checked against ``sum(p.numel() ...)`` rather than against
    literals in a test, so a legitimate change to a shared imported component re-costs the document
    rather than failing an unrelated assertion here."""
    section = _markdown_section(design, "## 13. ")
    stated = _integers_stated_in(section)

    for label, value in {**measured_totals, **measured_raw_lstm_totals}.items():
        assert value in stated, (
            f"DESIGN.md §13 does not state the measured {label} total {value:,}"
        )


def test_the_documented_target_axis_delta_is_the_decoder_head_and_nothing_else(
    design, measured_totals
):
    r"""$514 \times (C - 16)$ at both budgets: the two per-channel output rows plus their biases. If
    a future change makes the two conv-Transformer models differ anywhere but the decoder head, this
    fails rather than letting §13 keep asserting a decomposition that no longer holds.

    The *stated* products are evaluated rather than merely searched for. A test that checked only
    that the totals appear would pass on a section carrying the right delta beside a wrong
    factorisation of it, which is exactly the half a reader takes on trust.
    """
    section = _markdown_section(design, "## 13. ")
    guarded = measured_totals["trf_fs_guarded"] - measured_totals["trf_rws_guarded"]
    unguarded = measured_totals["trf_fs_unguarded"] - measured_totals["trf_rws_unguarded"]
    stated = _integers_stated_in(section)

    assert guarded == 514 * (78 - 16)
    assert unguarded == 514 * (109 - 16)
    assert guarded in stated and unguarded in stated

    factorisations = {
        int(cost) * (int(wide) - int(narrow))
        for cost, wide, narrow in _HEAD_COST_PATTERN.findall(section)
    }
    assert factorisations == {guarded, unguarded}, (
        f"§13 factorises the target-axis delta as {sorted(factorisations)}, and the measured deltas "
        f"are {sorted({guarded, unguarded})}"
    )


def test_the_documented_encoder_axis_delta_is_the_two_history_encoders(
    design, measured_totals, measured_raw_lstm_totals
):
    """The reduction the conv-Transformer encoders buy at a fixed target. Checked as arithmetic
    rather than as prose: it must be the same number the raw domain sees at the same budget, since
    everything downstream of the encoders is a shared module in both pairs -- which is the claim §13
    makes and the reason the delta is quoted at all."""
    section = _markdown_section(design, "## 13. ")
    feature_delta = measured_totals["fs_guarded"] - measured_totals["trf_fs_guarded"]
    raw_delta = measured_raw_lstm_totals["rws_guarded"] - measured_totals["trf_rws_guarded"]

    assert feature_delta == raw_delta, (
        "the encoder swap no longer costs the same in both target domains, so the reduction is not "
        "the two history encoders alone"
    )
    assert feature_delta in _integers_stated_in(section)


def test_the_design_reconciles_the_guarded_delta_with_the_sibling_record(design, measured_totals):
    """Two correct numbers that read as a contradiction unless the decomposition is written down: the
    availability *projections* cost 13,696 while the constructor delta is 6,272, because the input
    linears narrow at the same time. §13 states the reconciliation so no edit to that sibling
    document is needed."""
    section = _markdown_section(design, "## 13. ")
    adapter_delta = measured_totals["trf_rws_guarded"] - measured_totals["trf_rws_unguarded"]
    stated = _integers_stated_in(section)

    assert adapter_delta == 6_272
    # The three terms, as arithmetic rather than as three numbers that happen to be printed: the two
    # availability projections at the surviving widths, plus the two start embeddings, minus what the
    # input linears give back by narrowing from the declared widths to the surviving ones.
    assert 13_696 == 128 * (78 + 29)
    assert 7_680 == 128 * ((109 - 78) + (58 - 29))
    assert adapter_delta == 13_696 + 256 - 7_680
    assert {adapter_delta, 13_696, 7_680} <= stated
    assert "256" in section
    assert "narrow" in section


# ---------------------------------------------------------------------------------------
# DESIGN.md: the claims a reader could take too far
# ---------------------------------------------------------------------------------------
def test_the_design_states_the_two_ways_the_nats_are_incomparable(design):
    """Both halves, because the second is the one a reader of the first would not guess: the reach
    budget moves the surviving-channel count, hence the decoder width, hence the block every nat is
    summed over."""
    section = _flat(_markdown_section(design, "## 5. "))

    assert "Not comparable to the raw models'" in section
    assert "Not comparable across reach budgets within this model" in section
    assert "mutually unloadable checkpoints" in section


def test_the_causality_claim_is_unconditional_and_says_what_makes_it_so(design):
    """The one claim of this package that is genuinely stronger than the feature sibling's rather
    than inherited, and the flag a reader would otherwise go looking for. Asserted against the
    constructor signature, so the prose cannot outlive the property."""
    section = _flat(_markdown_section(design, "## 12. "))

    assert "causal_norm" not in inspect.signature(SeqVaeLagAttnTrfFs.__init__).parameters
    assert "`causal_norm` is not a constructor keyword of this model at all" in section
    assert "not raw-signal causality" in section  # the half a reader could take too far
    assert "quantile" in section  # the guard bounds the leak; it does not remove it
    assert "not** called transfer entropy" in section


def test_the_target_gathered_never_delayed_rule_has_its_own_section(design):
    """The sharpest correctness trap in the module: a delayed target changes *which* future the model
    is asked to forecast, and every shape downstream is identical."""
    section = _flat(_markdown_section(design, "## 8. "))

    assert "gather and not the delay" in section
    assert "all 78 surviving channels carry a non-zero delay" in section
    assert "hand-written slice" in section  # why the reference is not the shared helper
    assert "commute" in section  # the gather-before-unfold ordering, a memory decision


def test_the_smear_argument_is_cross_referenced_rather_than_restated(design):
    """It is a property of the target and the filter bank, so it belongs to the feature-domain record
    and is identical in both feature models. The cross-reference must name a document that exists, or
    it is worse than none."""
    section = _flat(_markdown_section(design, "## 8. "))

    assert "lag_attn_fs/DESIGN.md` §8" in section
    assert "unaffected by the encoder" in section
    assert (_REPO_ROOT / "teb_vae" / "lag_attn_fs" / "DESIGN.md").is_file()


def test_the_four_structural_constraints_are_recorded_with_their_fixture_caveat(design):
    """All four, and the caveat that makes them meaningful: the delta heads are zero-initialised, so
    an un-perturbed model passes every KL assertion vacuously."""
    section = _flat(_markdown_section(design, "## 9. "))

    for claim in ("No decoder bypass", "Source purity", "Exact zero KL", "invoked twice"):
        assert claim in section, f"DESIGN.md §9 no longer names: {claim}"
    assert "perturb_posterior" in section
    assert "base_decode: mean" in section  # the shipped flags the zero-KL claim is conditional on


def test_the_initialisation_order_puts_the_width_seam_before_the_generic_pass(design):
    """The whole safety argument for editing a shipped sibling's net: the decoder is built at the
    resolved width *before* the generic pass, the depthwise repair and the calibration, so at default
    arguments the same value is passed at the same point and the RNG stream cannot move."""
    section = _markdown_section(design, "## 10. ")

    assert section.index("the decoder at that width") < section.index("initialization(self)")
    assert "n_depthwise_init" in section
    # The one initialisation policy the target domain's width change actually reaches.
    assert "head_init_calibration" in section and "78" in section


def test_the_ddp_section_records_the_walk_that_is_not_ported_and_what_replaces_it(design):
    """Every ``forward`` that runs here belongs to a module this package imports, so the tensor-branch
    AST walk is not copied -- and the premise that makes that sound is the assertion that this
    package's net layer defines no ``forward`` at all."""
    section = _flat(_markdown_section(design, "## 11. "))

    assert "AST walk is deliberately not ported" in section
    assert "no `forward` at all" in section
    assert "broadcast_buffers=False" in section
    assert "future_index" in section  # present, never read, and broadcast for nothing


def test_the_added_readouts_are_documented_as_partial_sums(design):
    """The only property that makes them worth reporting: a second definition of the per-element term
    or the mask would let them stop being a decomposition of the number beside them."""
    section = _markdown_section(design, "## 14. ")

    for metric in FORECAST_GAP_METRICS:
        assert f"`{metric}`" in section, f"DESIGN.md §14 no longer names {metric}"
        assert metric in TRACKED_SUFFIXES, f"{metric} is not a tracked metric"
    assert "partial sums" in _flat(section)

    # The declared split, pinned against the class attribute rather than restated: it cannot be
    # derived from c_y, so the document is one of only two places the number lives.
    stated = re.search(r"TARGET_BLOCK_SPLIT\s*=\s*(\d+)", section)
    assert stated is not None, "DESIGN.md §14 no longer states the block split"
    assert int(stated.group(1)) == SeqVaeLagAttnTrfFs.TARGET_BLOCK_SPLIT


def test_the_design_states_that_this_is_an_experiment_rather_than_a_remedy(design):
    """The one framing error that would make a correct negative result read as a failure."""
    section = _flat(_markdown_section(design, "## 1. "))

    assert "experiment, not a remedy" in section
    assert "expected to reproduce" in section


# ---------------------------------------------------------------------------------------
# DESIGN.md: limitations, deviations, and how to run it
# ---------------------------------------------------------------------------------------
def test_the_design_records_deviations_and_limitations(design):
    for heading in ("## 15. Deliberate limitations", "## 16. Deviation record"):
        assert heading in design


@pytest.mark.parametrize(
    "phrase",
    [
        "the RNG stream cannot move",          # why editing a shipped sibling's net was safe
        "gain the seam",                       # the e2e sibling that did not, and why nothing needs it
        "moved verbatim into",                 # the mixin move out of the conv-LSTM feature model
        "The gradient clip moved",             # the constant the encoder did move
        "the equality is now a measurement",   # the constant it did not
        "does not reproduce at dev-box scale", # the throughput claim
        "not weaker",                          # the causality claim got stronger
    ],
)
def test_the_deviation_record_names_each_required_deviation(design, phrase):
    """Seven things a reader would otherwise have to rediscover from the code or from a run. Three of
    them are places a *stated rationale* did not survive measurement, which is the half of a deviation
    record most likely to be quietly dropped -- it reads as an admission rather than as a finding."""
    record = _flat(design[design.index("## 16. Deviation record") :])

    assert phrase.lower() in record.lower(), f"DESIGN.md §16 no longer names: {phrase}"


def test_the_e2e_sibling_really_did_not_gain_the_width_seam(design):
    """The asymmetry §16 records, asserted against the class rather than against the sentence. That
    model is standalone and builds its own decoder inline for a raw target it is the only consumer
    of, so the hook is absent there -- while the sibling this model *does* derive from carries it."""
    record = design[design.index("## 16. Deviation record") :]

    assert not hasattr(SeqVaeLagAttnTrfE2E, "_default_decoder_out_channels")
    assert hasattr(SeqVaeLagAttnTrfRws, "_default_decoder_out_channels")
    assert SeqVaeLagAttnTrfE2E.__bases__ != (SeqVaeLagAttnTrfRws,)
    assert "SeqVaeLagAttnTrfE2E" in record


def test_the_lean_limits_carry_their_replacement_triggers(design):
    """A ``lean-limit`` note without a measurable trigger is a permanent excuse. Exactly two here: the
    two-sided features, and the absent evaluation."""
    flat = _flat(design)

    assert len(re.findall(r"^> lean-limit: ", design, re.MULTILINE)) == 2
    assert "when a causal front end exists" in flat
    assert "when a feature-domain `eval` package exists" in flat


def test_the_design_running_section_names_no_evaluation_entry_point(design):
    """The evaluation is deferred whole, and a launch line for one would be the most convincing
    possible way to imply otherwise -- so the section says the absence out loud and carries no command
    that would contradict it."""
    section = _markdown_section(design, "## 17. ")
    commands = [line for line in section.splitlines() if "-m teb_vae" in line]

    assert commands, "§17 carries no launch lines"
    assert all("trainer" in line for line in commands), (
        f"§17 names a non-trainer entry point: {[line for line in commands if 'trainer' not in line]}"
    )
    assert "There is no `eval` entry point" in section


def test_every_companion_document_the_design_defers_to_exists(design):
    """This record's opening claim is that four sibling documents are *not* restated here, so every
    one of them is load-bearing: a moved file turns the deferral into a dead end, and the reader who
    followed it has no way to tell a missing document from an unwritten one."""
    referenced = sorted({match for match in re.findall(r"teb_vae/[\w/]+\.md", design)})

    assert len(referenced) >= 4, f"the design record defers to only {referenced}"
    missing = [path for path in referenced if not (_REPO_ROOT / path).is_file()]
    assert missing == [], f"DESIGN.md defers to documents that do not exist: {missing}"


def test_every_launch_line_in_the_design_names_a_config_that_exists(design):
    """A launch line is copied and pasted; one naming a moved or renamed file fails at the shell with
    a message about a path rather than about a run."""
    referenced = sorted(set(_LAUNCH_CONFIG_PATTERN.findall(design)))

    assert len(referenced) == 3, f"DESIGN.md §17 names {referenced}, not all three shipped configs"
    missing = [path for path in referenced if not (_REPO_ROOT / path).is_file()]
    assert missing == [], f"DESIGN.md names config files that do not exist: {missing}"


# ---------------------------------------------------------------------------------------
# The column set, and the sections that carry it
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("heading", STUDY_HEADINGS)
def test_every_section_carries_a_table_with_named_columns(results, heading):
    """A section without a table leaves the run phase deciding what to record. With no evaluation
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
    assert {"params", "collapsed", "epochs", "steps"} <= derived


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
        assert metric in TRACKED_SUFFIXES, f"{metric} is not a tracked metric"


def test_the_memory_section_defines_its_own_column_and_the_escalation_order(results):
    """``peak_memory_gb`` is not a tracked metric and never will be -- it is read from the CUDA
    allocator, not from the task -- so the document has to say what it means, and *when* it is read:
    a figure taken before the first backward measures a model that has not allocated its gradients."""
    section = _flat(_markdown_section(results, "## Distributed smoke, memory and throughput"))

    assert "torch.cuda.max_memory_allocated()" in section
    assert "after the first" in section and "training step" in section
    assert "accumulate_grad_batches" in section  # the first escalation lever
    assert "gradient checkpointing" in section  # the second, and its DDP caveat
    assert "use_reentrant=False" in section
    assert "Never a narrower model or a shorter context" in section


# ---------------------------------------------------------------------------------------
# The parameter arithmetic, pinned against constructed models
# ---------------------------------------------------------------------------------------
def test_the_documented_parameter_totals_are_the_measured_ones(results, measured_totals):
    """All six, checked against ``sum(p.numel() ...)`` rather than against literals in a test."""
    section = _markdown_section(results, "## Parameter budget")
    stated = _integers_stated_in(section)

    for label, value in measured_totals.items():
        assert value in stated, (
            f"the parameter budget does not state the measured {label} total {value:,}"
        )


def test_the_encoder_axis_delta_is_stated_and_is_the_two_history_encoders(results, measured_totals):
    """The reduction the conv-Transformer encoders buy, at a fixed target. Checked as arithmetic
    rather than as prose: it must be the same number the raw domain sees at the same budget, since
    everything downstream of the encoders is a shared module in both pairs."""
    section = _markdown_section(results, "## Parameter budget")
    feature_delta = measured_totals["fs_guarded"] - measured_totals["trf_fs_guarded"]
    raw_delta = 5_094_458 - measured_totals["trf_rws_guarded"]  # the conv-LSTM raw model, guarded

    assert feature_delta == raw_delta, (
        "the encoder swap no longer costs the same in both target domains, so the reduction is not "
        "the two history encoders alone"
    )
    assert feature_delta in _integers_stated_in(section)


def test_the_target_axis_delta_is_the_decoder_head_and_nothing_else(results, measured_totals):
    r"""$514 \times (C - 16)$: the two per-channel output rows plus their biases. If a future change
    makes the two conv-Transformer models differ anywhere but the decoder head, this fails rather
    than letting the section keep asserting a decomposition that no longer holds."""
    section = _markdown_section(results, "## Parameter budget")
    guarded_delta = measured_totals["trf_fs_guarded"] - measured_totals["trf_rws_guarded"]

    assert guarded_delta == 514 * (78 - 16)
    assert guarded_delta in _integers_stated_in(section)
    assert "514" in section, "the section no longer states the per-channel cost"


def test_the_guarded_delta_is_reconciled_with_the_sibling_design_record(results, measured_totals):
    """Two correct numbers that read as a contradiction unless the decomposition is written down: the
    availability *projections* cost 13,696 while the constructor delta is 6,272, because the input
    linears narrow at the same time."""
    section = _markdown_section(results, "## Parameter budget")
    adapter_delta = measured_totals["trf_rws_guarded"] - measured_totals["trf_rws_unguarded"]
    stated = _integers_stated_in(section)

    assert adapter_delta == 6_272
    assert adapter_delta in stated
    assert 13_696 in stated
    assert "narrow" in section


# ---------------------------------------------------------------------------------------
# The readings that survive a missing evaluation
# ---------------------------------------------------------------------------------------
def test_the_document_carries_the_four_reading_rules(results):
    """Four statements a reader must meet before the first table, because each is a way to misread
    every table after it. Restated here rather than cross-referenced to the feature sibling: a reader
    of this file must be able to read it without opening that one."""
    flat = _flat(results)

    assert "in-sample" in flat
    assert "Do not select on KL magnitude" in flat
    assert "no evaluation pipeline for this package" in flat
    assert "identically zero" in flat  # what WOULD indicate a build error
    assert "339" in flat  # the shard both splits are drawn from


def test_the_two_readings_are_named_before_any_table(results):
    """Which comparison answers which question, stated before the first table rather than after it.
    A reader who takes the encoder-axis row as a target-axis result is comparing 2340 coefficients
    against 480 samples."""
    preamble = results[: results.index("## Pre-registered acceptance criteria")]
    flat = _flat(preamble)

    assert "encoder axis" in flat and "target axis" in flat
    assert "lag_attn_fs" in flat and "lag_attn_transformer_rws" in flat
    assert "fixed target" in flat and "fixed encoder" in flat


def test_the_document_states_both_ways_the_nats_are_incomparable(results):
    """Both halves, because the second is the one a reader of the first would not guess: the reach
    budget moves the surviving-channel count, hence the decoder width, hence the block every nat is
    summed over."""
    flat = _flat(results)

    assert "2340" in flat and "480" in flat
    assert "two arms of *this* model at different budgets" in flat
    assert "checkpoints will not load into each other" in flat


def test_the_revert_table_carries_a_row_for_each_shipped_package_edit(results):
    """The two rows neither sibling's revert table has an analogue for. This is the only package in
    the family that edited another package's ``nets/`` to exist, and neither edit is revertible by
    configuration -- so each names its file, its revert path and what re-running costs."""
    section = _markdown_section(results, "### What reverts, and how")

    mixin = _row_cells(section, "mixin moved out of")
    seam = _row_cells(section, "width seam")

    assert mixin is not None, "the revert table has no row for the feature-target mixin move"
    assert seam is not None, "the revert table has no row for the decoder width seam"
    for row, path in (
        (mixin, "teb_vae/lag_attn_fs/nets/feature_target.py"),
        (seam, "teb_vae/lag_attn_transformer_rws/nets/model.py"),
    ):
        assert path in " ".join(row), f"the revert row does not name the file it lives in: {path}"
        assert "Code" in " ".join(row), "a code-only revert must not read as a config change"

    # What re-running costs, which is the half a revert table usually omits.
    assert "torch.equal" in " ".join(mixin)  # the inertness proof to reach for
    assert "n_depthwise_init" in " ".join(seam)  # where an RNG perturbation surfaces
    assert "lag_attn_transformer_e2e" in " ".join(seam)  # the third suite


def test_the_shipped_package_files_the_revert_table_names_exist(results):
    """A revert row naming a moved file is worse than none: it sends the operator to a path rather
    than to the change."""
    section = _markdown_section(results, "### What reverts, and how")
    referenced = re.findall(r"teb_vae/[\w/]+\.py", section)

    assert referenced, "the revert table names no source file"
    missing = [path for path in set(referenced) if not (_REPO_ROOT / path).is_file()]
    assert missing == [], f"the revert table names files that do not exist: {missing}"


def test_the_document_states_where_its_numbers_come_from(results):
    """Its tables are transcribed rather than generated -- there is no evaluation pass to generate
    them -- so the three sourcing rules that make the copy safe are load-bearing here, not optional."""
    flat = _flat(results)

    assert "metrics_history.csv" in flat
    assert "resolved_config.yaml" in flat  # rows keyed from the run's own record
    assert "marked, not dropped" in flat  # a collapsed arm is marked


def test_the_per_step_sampling_caveat_is_stated(results):
    """The two columns that are not epoch means. A threshold derived from them as if they were
    per-epoch aggregates would be derived from the wrong distribution -- and the clip threshold is
    exactly such a derivation."""
    flat = _flat(results)

    assert "one optimizer step per epoch" in flat
    assert "`grad_clip_frac` is a $0$/$1$ exceedance indicator" in flat


def test_every_launch_line_names_a_config_that_exists(results):
    """A launch line is copied and pasted; one naming a moved or renamed file fails at the shell with
    a message about a path rather than about a run."""
    referenced = sorted(set(_LAUNCH_CONFIG_PATTERN.findall(results)))

    assert referenced, "RESULTS.md carries no launch lines"
    missing = [path for path in referenced if not (_REPO_ROOT / path).is_file()]
    assert missing == [], f"RESULTS.md names config files that do not exist: {missing}"


def test_no_launch_line_names_an_evaluation_entry_point(results):
    """The evaluation is deferred whole, and a launch line for one would be the most convincing
    possible way to imply otherwise."""
    launch = _markdown_section(results, "## Launch lines")
    commands = [line for line in launch.splitlines() if "-m teb_vae" in line]

    assert commands, "the launch section carries no commands"
    assert all("trainer" in line for line in commands)
    assert "There is no `eval` entry point" in launch


# ---------------------------------------------------------------------------------------
# The criteria table, re-evaluated against the row it reports
# ---------------------------------------------------------------------------------------
def test_the_criteria_table_lists_the_pre_registered_six(results):
    """The set is registered before the runs, so the result cannot be chosen after the fact. The
    numbering is what a later reader cites, so a criterion cannot be renumbered away."""
    section = _markdown_section(results, "## Pre-registered acceptance criteria")
    rows = [line for line in section.splitlines() if re.match(r"^\|\s*[1-9]\s*\|", line)]

    assert len(rows) == 6, f"expected six pre-registered criteria, found {len(rows)}"
    assert "*sign* of `pred_gap` is not among the things these runs establish" in _flat(results)


@pytest.mark.slow
def test_the_recorded_headline_row_satisfies_the_criteria_it_is_read_against(results):
    """The verdict is a committed artefact rather than a manual reading.

    Re-evaluates criteria 3-6 against the numbers the headline and bottleneck tables actually carry,
    so a PASS the row does not support fails here. It reads the *converged* values the document
    commits to -- the last-ten-epoch means -- rather than the run's full series, which is not
    committed and lives only in the run directory; criterion 6's tail clause is therefore applied to
    those values held constant across the patience window, which is the strongest statement the
    committed row supports.

    Marked ``slow`` because it is meaningless until a run has filled the tables in, and a suite run
    before that should report an empty table rather than a failure.
    """
    headline = _row_cells(_markdown_section(results, "## Headline baseline"), "this model")
    health = _row_cells(_markdown_section(results, "## Bottleneck health"), "this model")
    assert headline is not None and health is not None

    nll_base_block = _reading(headline, 4, "nll_base_block")
    pred_gap = _reading(headline, 6, "pred_gap")
    kl_raw = _reading(health, 1, "source_conditioned_kl_raw")
    active_frac = _reading(health, 2, "kld_active_frac")
    prior_floor_frac = _reading(health, 4, "logvar_prior_floor_frac")

    # 3: the conditional prior stays off its clamp floor.
    assert prior_floor_frac < 0.2
    # 5: the forecast gap is a number, not an artefact.
    assert pred_gap != 0.0
    assert abs(pred_gap) < nll_base_block
    # 6: evaluated, not eyeballed. d_z is the shipped latent width.
    patience = collapse.KL_COLLAPSE_PATIENCE_EPOCHS
    assert not collapse.is_collapsed(
        [kl_raw] * patience, [active_frac], d_z=shipped_gated_kwargs(120.0)["d_z"]
    )
    assert headline[-1] == "no", "the headline row records a collapse verdict the criterion refutes"


@pytest.mark.slow
def test_the_recorded_log_variance_row_sits_inside_the_shipped_clamp(results):
    """Criterion 4, re-evaluated against the committed row. The clamp floor is read off the shipped
    config rather than restated, so a config that moved it re-costs the reading."""
    from teb_vae.lag_attn.config import load_config

    section = _markdown_section(results, "## Bottleneck health")
    rows = [
        cells
        for cells in (_row_cells(part, "this model") for part in section.split("| Arm |"))
        if cells is not None and len(cells) == 6
    ]
    mean_logvar_full = _number(rows[0][1]) if rows else None
    if mean_logvar_full is None:
        pytest.skip("the log-variance row is not filled in yet")

    floor = float(
        load_config(str(_PACKAGE_DIR / "configs" / "default.yaml"))["model_config"]["VAE_model"][
            "logvar_clamp"
        ][0]
    )

    assert mean_logvar_full >= floor + 0.5


# ---------------------------------------------------------------------------------------
# No planning artefact in the shipped tree
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("document", ("DESIGN.md", "RESULTS.md"))
def test_the_records_do_not_mention_the_planning_document(document):
    """A design record describes the model and its invariants and a results record describes runs and
    their readings; neither describes the artefact that scheduled them, which does not survive the
    module and would leave dangling references."""
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
