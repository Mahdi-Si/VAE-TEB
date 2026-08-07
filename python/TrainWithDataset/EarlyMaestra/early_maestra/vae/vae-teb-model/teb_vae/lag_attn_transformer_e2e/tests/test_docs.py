r"""``DESIGN.md`` must describe the module that exists, not the one it described when it was written.

A design record is only worth having if something fails when it goes stale, and the parts of one
that go stale *silently* are the parts worth driving from code. Three do so here.

The **configuration surface** drifts when a key is renamed, or when a knob the document calls
deliberately absent creeps back in, and the prose reads correctly either way -- so §13 lists both
sets explicitly and this file walks them against the shipped config in both directions, including
the direction that catches a *new* model key nobody documented. It also binds the absent list to
``trainer.INERT_MODEL_KEYS``, the table the pre-flight refuses by name: a key that stopped being
refused, or one that started being, would otherwise leave the document describing a guard that is
not there.

The **measured numbers** are the second. This module is the only place the absolute parameter total
is pinned -- deliberately, so that a legitimate change to an imported downstream component re-costs
this document rather than failing an unrelated assertion in a package that imports almost
everything. The front-end reach, the per-stage parameter subtotals and the composed raw receptive
field are checked the same way, against a constructed shipped-geometry model.

The third is the **roadmap ban**. Nothing that ships may name the planning document that produced
it: that artefact does not survive the work, and a reference to it from a shipped file would dangle.
Checked with a walk over every ``.py`` and ``.md`` in the package, and paired with a control that
points the walk at a directory containing one of each banned token -- a three-pattern rule that
silently matched nothing would pass on a tree full of references.

What is deliberately *not* checked here is prose. A design record that could be fully validated by
a test would be a data file, and the parts of it worth reading -- why the offset is right, why the
bias is load-bearing -- are exactly the parts no assertion can hold.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E
from teb_vae.lag_attn_transformer_e2e.trainer import INERT_MODEL_KEYS, LagAttnTrfE2ETrainer

from .conftest import SHIPPED_KWARGS
from .test_frontend_reach import SHIPPED_REACH_SAMPLES

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_DESIGN = _PACKAGE_DIR / "DESIGN.md"
_CONFIG_DIR = _PACKAGE_DIR / "configs"
_CONFIG = _CONFIG_DIR / "default.yaml"
_TINY = _CONFIG_DIR / "tiny.yaml"

#: A dotted config path inside backticks, one per bullet, in the §13 lists. Case-sensitive on
#: purpose: the config's own key is ``VAE_model``, and a pattern that lowercased it would silently
#: drop every model key from both lists and leave the assertions below checking almost nothing.
_KEY_PATTERN = re.compile(r"^-\s+`([A-Za-z_]+(?:\.[A-Za-z_]+)+)`\s*$", re.MULTILINE)

#: The parameter total as §1 states it: ``**2,151,743 parameters**``.
_TOTAL_PATTERN = re.compile(r"\*\*([\d,]+) parameters\*\*")

#: The front end's reach and budget, as both the printed stage table in §3.1 and the startup-log
#: excerpt in §9 write it. Two independent statements of one pair of numbers, either editable alone.
#: ``\s+`` rather than a literal space because one of the two is wrapped prose and the other is a
#: fixed-width table -- a pattern that assumed either layout would find only one of them.
_REACH_PATTERN = re.compile(
    r"reach (\d+) raw samples\s+\(([\d.]+) s\)\s+against a budget of\s+(\d+)"
)

#: §3.2's composed source raw reach, in samples and seconds.
_COMPOSED_PATTERN = re.compile(
    r"\\mathbf\{([\d{},]+)\} \\text\{ raw samples\} \\;=\\; ([\d.]+) \\text\{ s\}"
)

#: §7's depthwise-initialisation counts, this model's against the sibling's.
_DEPTHWISE_PATTERN = re.compile(
    r"`n_depthwise_init` is \*\*(\d+)\*\* here against the sibling's \*\*(\d+)\*\*"
)

#: §11's smoke-model parameter total, in LaTeX.
_SMOKE_TOTAL_PATTERN = re.compile(r"The smoke model is \$([\d{},]+)\$ parameters")

#: Seconds per raw sample at the shipped $4$ Hz grid; the document reports every reach both ways.
SECONDS_PER_RAW_SAMPLE = 0.25

#: Rows of §5's budget table, keyed by the label the row starts with, valued by how to measure the
#: number that row claims. Written out rather than derived so that a row silently deleted from the
#: table fails here instead of being skipped.
BUDGET_ROWS = (
    "One front end",
    "Both front ends",
    "Both encoders, imported unchanged",
    "Everything else, unchanged",
)


@pytest.fixture(scope="module")
def design() -> str:
    return _DESIGN.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def shipped() -> dict:
    """The shipped config, read raw.

    ``load_config`` is not used: ``default.yaml`` has no ``base:`` chain, and this asserts what is
    written in the file an operator edits.
    """
    return yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def shipped_model() -> SeqVaeLagAttnTrfE2E:
    """One constructed production-geometry model, shared by every measurement below."""
    return SeqVaeLagAttnTrfE2E(**SHIPPED_KWARGS)


@pytest.fixture(scope="module")
def measured(shipped_model: SeqVaeLagAttnTrfE2E) -> Dict[str, int]:
    """Every structural number §1, §3, §5 and §7 claim, measured on the built model.

    Measured from :data:`SHIPPED_KWARGS` rather than from the config, so this file binds the
    document to the *architecture*; ``tests/test_config_load.py`` binds the config to the
    architecture separately, and the two routes meet at the same numbers.
    """
    front_end = shipped_model.target_frontend
    per_stream = sum(p.numel() for p in front_end.parameters())
    encoders = sum(
        p.numel()
        for module in (shipped_model.target_encoder, shipped_model.source_encoder)
        for p in module.parameters()
    )
    total = sum(p.numel() for p in shipped_model.parameters())
    source_reach = shipped_model.source_encoder.receptive_field
    assert source_reach is not None, "the source encoder lost its bound; §3.2 assumes one"

    return {
        "total": total,
        "One front end": per_stream,
        "Both front ends": 2 * per_stream,
        "Both encoders, imported unchanged": encoders,
        "Everything else, unchanged": total - 2 * per_stream - encoders,
        "reach": front_end.reach_samples,
        "reach_budget": front_end.reach_budget,
        # Both reaches are *counts*, so the two supports overlap on the anchor token's own
        # ``raw_per_step`` samples and the composition subtracts one token rather than none.
        "composed_source_reach": front_end.reach_samples
        + shipped_model.raw_per_step * (source_reach - 1),
        "n_depthwise_init": shipped_model.n_depthwise_init,
    }


@pytest.fixture(scope="module")
def smoke_total() -> int:
    """Parameter total of the model ``configs/tiny.yaml`` actually builds.

    Through the real driver's signature sweep, not a hand-assembled keyword set: the claim in §11
    is about what the smoke fit runs, and a sweep that dropped a key would make a hand-built model
    agree with the document while disagreeing with the run.
    """
    import tempfile

    config = load_config(str(_TINY))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        kwargs = LagAttnTrfE2ETrainer(config_file_path=str(path))._build_model_kwargs()
    return sum(p.numel() for p in SeqVaeLagAttnTrfE2E(**kwargs).parameters())


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


def _latex_int(text: str) -> int:
    """Read an integer written with LaTeX thousands separators, e.g. ``2{,}151{,}743``."""
    return int(
        text.replace("\\mathbf", "").replace("{", "").replace("}", "").replace(",", "")
    )


def _row_int(design_text: str, label: str) -> int:
    """The number in the last cell of the one budget-table row starting with ``label``.

    Args:
        design_text: The whole document.
        label: The row's first cell, verbatim.

    Returns:
        The row's value.

    Raises:
        AssertionError: If the row is missing or duplicated, either of which would make a
            comparison against it meaningless.
    """
    rows = [
        line
        for line in design_text.splitlines()
        if line.startswith(f"| {label} |") or line.startswith(f"| {label}, ")
    ]
    assert len(rows) == 1, f"§5 has {len(rows)} rows labelled {label!r}, expected exactly one"
    cells = re.findall(r"\$([^$]+)\$", rows[0])
    assert cells, f"§5's {label!r} row carries no number"
    return _latex_int(cells[-1])


def _lean_limit_blocks(design_text: str) -> List[str]:
    """Every ``> lean-limit:`` block, flattened to one line each.

    Flattened because a limitation note is wrapped prose: a phrase that must be present may sit
    across a line break today and not tomorrow, and a test that noticed the rewrap would be
    checking the wrapping rather than the claim.

    Args:
        design_text: The whole document.

    Returns:
        One whitespace-collapsed string per block, in document order.
    """
    blocks: List[str] = []
    current: List[str] = []
    for line in design_text.splitlines():
        if line.startswith("> lean-limit:"):
            if current:
                blocks.append(" ".join(current))
            current = [line[2:].strip()]
        elif line.startswith(">") and current:
            current.append(line[1:].strip())
        elif current:
            blocks.append(" ".join(current))
            current = []
    if current:
        blocks.append(" ".join(current))
    return [re.sub(r"\s+", " ", block) for block in blocks]


# ---------------------------------------------------------------------------------------
# The configuration surface, in both directions
# ---------------------------------------------------------------------------------------
def test_the_document_lists_keys_in_both_directions(design):
    """A guard on the guard: if the extraction silently matched nothing -- or matched only the keys
    that happen to be lowercase -- every assertion below would pass on a short list."""
    required = _KEY_PATTERN.findall(_section(design, "Required"))

    assert len(required) > 40
    assert len([key for key in required if key.startswith("model_config.VAE_model.")]) > 30
    assert len(_KEY_PATTERN.findall(_section(design, "Deliberately absent"))) > 20


def test_every_key_the_document_requires_exists_in_the_shipped_config(design, shipped):
    required = _KEY_PATTERN.findall(_section(design, "Required"))

    missing = [key for key in required if not _has(shipped, key)]

    assert missing == [], f"DESIGN.md §13 requires keys the shipped config does not have: {missing}"


def test_every_model_key_in_the_shipped_config_is_documented(design, shipped):
    """The direction that catches a *new* key. A model key nobody documented is a knob whose meaning
    lives only in a YAML comment, and the signature sweep drops one that reaches nothing without a
    word -- so the run trains a different architecture than its config describes."""
    required = set(_KEY_PATTERN.findall(_section(design, "Required")))

    undocumented = [
        f"model_config.VAE_model.{key}"
        for key in shipped["model_config"]["VAE_model"]
        if f"model_config.VAE_model.{key}" not in required
    ]

    assert undocumented == [], f"DESIGN.md §13 does not document: {undocumented}"


def test_every_key_the_document_calls_absent_is_absent(design, shipped):
    """The direction that catches a knob creeping back in. Each of these either describes the input
    representation this model replaces, or is something this architecture made structural or
    derived -- and a key present would read to a maintainer as a control that exists."""
    absent = _KEY_PATTERN.findall(_section(design, "Deliberately absent"))

    present = [key for key in absent if _has(shipped, key)]

    assert present == [], f"DESIGN.md §13 calls these absent but the config sets them: {present}"


def test_every_key_the_preflight_refuses_is_documented_as_absent(design):
    """The document's absent list and the pre-flight's refusal table are two statements of one
    decision, and the interesting failure is them disagreeing: a key that stopped being refused
    would leave §13 describing a guard that no longer fires, and a key that started being refused
    without being listed would refuse a launch this record says is legal."""
    absent = set(_KEY_PATTERN.findall(_section(design, "Deliberately absent")))

    undocumented = sorted(
        f"model_config.VAE_model.{key}"
        for key in INERT_MODEL_KEYS
        if f"model_config.VAE_model.{key}" not in absent
    )

    assert undocumented == [], f"the pre-flight refuses keys §13 does not list: {undocumented}"
    # §11 says "the eight inert keys"; that count is a claim about this table.
    assert len(INERT_MODEL_KEYS) == 8


def test_the_front_ends_shape_has_no_configuration_surface(design, shipped):
    """The widths derive from ``d_model`` and the kernels are the constructor's own default, so
    there is nothing for an operator to set and nothing for the signature sweep to drop in silence.
    Listed as deliberately absent so the choice reads as a choice rather than an omission.

    The backward reach is the one front-end quantity that *is* configured, and it must be in the
    required list rather than the absent one -- the two lists are read as a pair, and a key in
    neither is exactly the silent-drop failure this file exists to catch."""
    absent = _KEY_PATTERN.findall(_section(design, "Deliberately absent"))
    required = _KEY_PATTERN.findall(_section(design, "Required"))

    assert "model_config.VAE_model.frontend_kernels" in absent
    assert "model_config.VAE_model.frontend_reach_budget_s" not in absent
    assert "model_config.VAE_model.frontend_reach_budget_s" in required
    assert [
        key for key in shipped["model_config"]["VAE_model"] if key.startswith("frontend")
    ] == ["frontend_reach_budget_s"]


def test_the_seven_encoder_keys_are_documented_as_required(design):
    """*Same encoder, different input* is the whole claim, so every key that shapes the encoder must
    still be live here. One quietly dropped would make the comparison a comparison of two
    encoders."""
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


def test_the_two_loader_lists_and_the_trim_are_documented_as_required(design):
    """The three keys this architecture's *inputs* depend on. ``load_fields`` and
    ``normalize_fields`` are what the pre-flight guards; ``trim_minutes`` is what makes the raw grid
    $4800$ rather than $5280$, and it is the one whose mismatch the shard-side guard exists for."""
    required = _KEY_PATTERN.findall(_section(design, "Required"))

    for key in (
        "dataset_config.dataloader_config.normalize_fields",
        "dataset_config.dataloader_config.dataset_kwargs.load_fields",
        "dataset_config.dataloader_config.dataset_kwargs.trim_minutes",
    ):
        assert key in required


def test_the_shipped_loader_lists_are_what_the_document_says_they_are(shipped):
    """§13's prose about those keys, checked against the file. Dropping the four feature blocks *is*
    the change, and an unnormalized ``up`` shifts every source-side readout with nothing raising."""
    dataloader = shipped["dataset_config"]["dataloader_config"]
    load_fields = dataloader["dataset_kwargs"]["load_fields"]

    assert set(load_fields) == {"fhr", "up", "weight", "guid"}
    assert set(dataloader["normalize_fields"]) == {"fhr", "up"}


# ---------------------------------------------------------------------------------------
# The measured numbers
# ---------------------------------------------------------------------------------------
def test_the_documented_parameter_total_is_the_measured_one(design, measured):
    """The single place the absolute number is pinned. Checked against ``sum(p.numel() ...)`` rather
    than against a literal in a test, so a legitimate change to an imported downstream component
    re-costs this document rather than failing an unrelated assertion in a package that imports
    almost everything it runs."""
    stated = _TOTAL_PATTERN.findall(design)

    assert len(stated) == 1, f"§1 should state the parameter total exactly once, found {stated}"
    assert int(stated[0].replace(",", "")) == measured["total"]


def test_the_budget_table_agrees_with_the_prose(design, measured):
    """Two statements of one number, in two notations. Either could be edited alone."""
    row = re.search(r"\|\s*\*\*Total\*\*\s*\|.*?\$(\\mathbf\{[\d{},]+\})\$", design)

    assert row is not None, "DESIGN.md §5 no longer has a total row"
    assert _latex_int(row.group(1)) == measured["total"]


@pytest.mark.parametrize("label", BUDGET_ROWS)
def test_every_budget_row_is_the_measured_one(design, measured, label):
    """Not only the total. The interesting row is *everything else*: it is the one that would move
    if a supposedly-untouched imported component changed, and a table whose parts no longer sum is
    how a reader discovers that the story about what changed is wrong."""
    assert _row_int(design, label) == measured[label]


def test_the_front_end_stage_rows_sum_to_the_front_end(design, measured, shipped_model):
    """The stage subtotals are the arithmetic that produces the front-end figure, so they are
    checked as arithmetic rather than as four independent literals."""
    stages = [_row_int(design, f"Front-end stage {index}") for index in range(1, 5)]
    norm = _row_int(design, "Final `RMSNorm(128)`")

    measured_stages = [
        sum(p.numel() for p in stage.parameters())
        for stage in shipped_model.target_frontend.stage_modules
    ]

    assert stages == measured_stages
    assert norm == shipped_model.target_frontend.output_norm.weight.numel()
    assert sum(stages) + norm == measured["One front end"]


def test_the_documented_reach_is_the_pinned_one(design, measured):
    """Stated twice -- in §3.1's printed stage table and in §9's startup-log excerpt -- because they
    are two different artefacts a reader might trust. Both are checked, against the built model and
    against the constant the reach test pins, so the three cannot drift apart."""
    statements = _REACH_PATTERN.findall(design)

    assert len(statements) >= 2, "§3.1's table and §9's log excerpt should both state the reach"
    assert measured["reach"] == SHIPPED_REACH_SAMPLES
    for reach, seconds, budget in statements:
        assert int(reach) == measured["reach"]
        assert int(budget) == measured["reach_budget"]
        assert float(seconds) == pytest.approx(
            measured["reach"] * SECONDS_PER_RAW_SAMPLE, abs=0.05
        )


def test_the_documented_composed_source_raw_reach_is_the_derived_one(design, measured):
    r"""The one quantity neither sibling could record. It only means something once the input is the
    raw signal, and it is a composition rather than a measurement, so the arithmetic is the thing
    that can go wrong: $R_{\mathrm{frontend}} + r(R_U - 1)$, one token of overlap subtracted because
    both reaches are counts."""
    stated = _COMPOSED_PATTERN.search(design)

    assert stated is not None, "DESIGN.md §3.2 no longer states the composed raw reach"
    assert _latex_int(stated.group(1)) == measured["composed_source_reach"]
    assert float(stated.group(2)) == pytest.approx(
        measured["composed_source_reach"] * SECONDS_PER_RAW_SAMPLE, abs=0.05
    )


def test_the_composed_reach_is_still_inside_the_lag_search_range(measured, shipped_model):
    """§3.2 calls this the tightest structural constraint in the model, which is a claim about the
    architecture rather than about the document -- so it is measured here rather than read. A source
    state reaching further back than the lag range would already be doing the alignment the lag
    attention exists to do, and the reported lag would stop being a statement about where the
    coupling came from."""
    lag_span_samples = shipped_model.max_lag * shipped_model.raw_per_step

    assert measured["composed_source_reach"] < lag_span_samples
    # 19.5 s of margin at the shipped geometry, which §3.2 states; one more step of source window
    # would spend all of it.
    margin_seconds = (lag_span_samples - measured["composed_source_reach"]) * SECONDS_PER_RAW_SAMPLE
    assert margin_seconds == pytest.approx(19.5, abs=0.05)


def test_the_documented_depthwise_counts_are_the_measured_ones(design, measured):
    """Twelve against the sibling's four: four encoder stem convolutions plus eight front-end ones.
    The count is the only evidence the variance-preserving pass was not a silent no-op, and the
    document's version of it is the one a reader checks a future change against."""
    stated = _DEPTHWISE_PATTERN.search(design)

    assert stated is not None, "DESIGN.md §7 no longer states the depthwise counts"
    assert int(stated.group(1)) == measured["n_depthwise_init"]
    # The sibling's count is stated as a comparison; the difference is two per front-end stage.
    from teb_vae.lag_attn_transformer_e2e.nets.frontend import NUM_STAGES

    assert int(stated.group(1)) - int(stated.group(2)) == 2 * NUM_STAGES


def test_the_documented_smoke_total_is_what_tiny_builds(design, smoke_total):
    """§11 quotes it to say that the smoke fit shrinks widths and nothing else. A stale number there
    would hide the case that matters: a ``tiny.yaml`` that quietly shrank the geometry would build a
    narrower front end than the production run's, and the smoke would exercise a stack nobody
    ships."""
    stated = _SMOKE_TOTAL_PATTERN.search(design)

    assert stated is not None, "DESIGN.md §11 no longer states the smoke model's parameter total"
    assert _latex_int(stated.group(1)) == smoke_total


def test_the_smoke_config_runs_the_production_front_end(shipped):
    """The claim §11 makes about *why* ``warmup_period`` could not shrink, checked against the file
    rather than against the prose: it is the reach budget, so a smaller one builds a different front
    end."""
    tiny = load_config(str(_TINY))

    for key in ("sequence_length", "raw_per_step", "warmup_period", "horizon"):
        assert (
            tiny["model_config"]["VAE_model"][key]
            == shipped["model_config"]["VAE_model"][key]
        ), f"configs/tiny.yaml no longer inherits {key}; §11 says the geometry stays real"


# ---------------------------------------------------------------------------------------
# The causality claim, the limitations and the deviation record
# ---------------------------------------------------------------------------------------
def test_the_document_records_deviations_and_limitations(design):
    for heading in ("## 10. Deliberate limitations", "## 11. Deviation record"):
        assert heading in design


@pytest.mark.parametrize(
    "phrase",
    [
        "five taps",                # the anti-alias tap count, and what it buys
        "carries a bias",           # the one non-bias-free projection, and why
        "standalone `nn.Module`",   # not a mode or subclass of the model compared against
        "refused by absence",       # how the inert keys raise
        "SwappedModel",             # the planted defect only this architecture can suffer
        "do not use `caplog`",      # loguru does not route through the stdlib logger
        "carried over, not measured",  # the clipping threshold's status
    ],
)
def test_the_deviation_record_names_each_required_deviation(design, phrase):
    """Seven differences a reader would otherwise have to rediscover from the code. Each is a place
    where the built module departs from the obvious reading of the design, and where the *reason*
    is the part worth carrying: a record stating only the fact leaves the wrong reason in a reader's
    head, which is worse than stating nothing."""
    record = design[design.index("## 11. Deviation record") :]

    assert phrase.lower() in record.lower(), f"DESIGN.md §11 no longer names: {phrase}"


def test_the_raw_causality_claim_is_carried_with_both_halves_of_the_distinction(design):
    """This model delivers the stronger of the two properties, and the weaker one has to stay named
    or a reader cannot tell what changed. The guard on the sibling bounds its leak rather than
    removing it -- ``quantile`` is the word that says so -- and this record has to keep saying it,
    because "the sibling could have fixed this with a budget" is the misreading available here."""
    assert "Raw-signal causality" in design

    causality = _markdown_section(design, "## 9. ")
    assert "H_t = f(X_{\\le t})" in causality       # token causality, which both models have
    assert "n_{\\mathrm{raw}}(t)" in causality       # raw-signal causality, which only this one has
    assert "quantile" in causality                   # the budget bounds the leak; it does not remove it


def test_the_document_states_that_the_kl_is_not_transfer_entropy(design):
    """The readout is a KL between two of the model's own distributions. Renaming it on the strength
    of an architectural argument is how a quantity starts being read as something nobody
    measured -- and this package's improved causal standing is exactly the argument that would
    tempt the rename."""
    assert "not** called transfer entropy" in design


def test_the_five_lean_limits_each_carry_a_replacement_trigger(design):
    """A limitation note without a trigger is a permanent excuse. Each is checked on the phrase that
    carries *its own* condition, so a note edited down to the fact alone fails here."""
    blocks = _lean_limit_blocks(design)

    assert len(blocks) == 5, f"§10 should carry five lean-limit notes, found {len(blocks)}"
    joined = "\n".join(blocks)
    for trigger in (
        "replace with a `ModelBinding` and an `eval/` package when",   # no eval package
        "replace with `configs/sweep_*.yaml`",                          # no arms
        "replace with a measured schedule when",                        # the front-end schedule
        "Promote to a common package when",                             # the unpromoted primitives
        "replace with the hook when this package gains an evaluation",   # the un-routed call site
    ):
        assert trigger in joined, f"§10 no longer carries the trigger: {trigger}"


def test_the_unrouted_builder_call_site_still_exists(design):
    """The fifth note names a file in another package, which is the kind of claim that rots without
    anyone noticing. If that call site is ever routed through the hook, the note becomes false and
    should be deleted -- so it is checked rather than trusted."""
    metrics = (
        _PACKAGE_DIR.parents[0] / "lag_attn_rws" / "eval" / "metrics.py"
    ).read_text(encoding="utf-8")

    assert "_build_target_streams" in metrics and "_build_source_stream" in metrics
    assert "_build_forward_inputs" not in metrics
    assert "teb_vae/lag_attn_rws/eval/metrics.py" in design


def test_the_resolved_config_claim_names_the_key_and_where_the_reach_lives(design):
    """A run's ``resolved_config.yaml`` is the durable record, and this architecture's reach is not
    in it -- the budget key it *does* carry belongs to the guard this model does not need. Saying so
    is what stops a later reader concluding the reach was never recorded."""
    assert "resolved_causal_budget: null" in design
    assert "SHIPPED_REACH_SAMPLES" in design


def test_the_clipping_threshold_is_a_positive_finite_float_carrying_its_status(shipped):
    """The measurement itself belongs to the first production run, so this does not gate on it --
    but the shipped value must be usable, and the file must say whether it was measured here or
    carried over from the model this one is compared against."""
    value = shipped["advanced_config"]["trainer"]["gradient_clip_val"]

    assert isinstance(value, float)
    assert value > 0.0 and math.isfinite(value)

    text = _CONFIG.read_text(encoding="utf-8")
    marker = text[: text.index("gradient_clip_val")]
    assert "PROVISIONAL" in marker or re.search(r"q_?\d\d|percentile", marker), (
        "configs/default.yaml no longer says whether gradient_clip_val was measured or carried over"
    )


def test_the_reclipping_procedure_is_executable_rather_than_a_result(design):
    """Four things a procedure needs to be followed without rediscovering it: what to read, in which
    form, which statistic sets the threshold, and where the answer goes. The sampling caveat is the
    fifth and the easiest to lose: the column carries one optimizer step per epoch, not an epoch
    aggregate, so its percentiles are per-step percentiles over a thinned sample."""
    section = _markdown_section(design, "## 12. ")

    assert "train/grad_norm" in section and "pre-clip" in section
    assert "one optimizer step per epoch" in section
    assert "q_{99}" in section
    assert "mean over epochs" in section
    assert "configs/default.yaml" in section


# ---------------------------------------------------------------------------------------
# No roadmap in the shipped tree
# ---------------------------------------------------------------------------------------
#: Files the roadmap-token ban does not apply to. This module names every banned token as a
#: literal, and a planning document is the artefact the ban exists to keep *out* of everything
#: else -- it does not survive the work, and a reference to it from a shipped file would dangle.
_ROADMAP_BAN_EXEMPT = frozenset({Path(__file__).name, "SPEC_AND_SPRINTS.md"})

#: Tokens no shipped file may carry: the planning document's name, its section word, and its task
#: identifiers. The word "task" is deliberately not banned -- ``task.py`` is a module of this
#: package and the design record has to name it.
_ROADMAP_TOKENS = (
    re.compile(r"SPEC_AND_SPRINTS"),
    re.compile(r"Sprint "),
    re.compile(r"S\d-T\d\d"),
)


#: Suffixes the ban walks. The configs are in the list because the rule is about every *shipped*
#: file -- code, docstring, comment, config and design record alike -- and a YAML comment is exactly
#: the kind of place a task identifier gets left behind, being the one file type nobody greps.
_ROADMAP_BAN_SUFFIXES = ("*.py", "*.md", "*.yaml")


def _roadmap_offenders() -> List[Tuple[str, str]]:
    """Return ``(relative path, matched token)`` for every shipped file naming the roadmap."""
    offenders = []
    walked = [path for suffix in _ROADMAP_BAN_SUFFIXES for path in _PACKAGE_DIR.rglob(suffix)]
    for path in sorted(walked):
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
    (tmp_path / "leaky.yaml").write_text("# added in Sprint 3\nkey: 1\n", encoding="utf-8")
    monkeypatch.setattr(
        "teb_vae.lag_attn_transformer_e2e.tests.test_docs._PACKAGE_DIR", tmp_path
    )

    offenders = _roadmap_offenders()

    assert sorted(token for _, token in offenders) == [
        "S4-T05",
        "SPEC_AND_SPRINTS",
        "Sprint ",
        "Sprint ",
    ]
    # One offender per suffix the walk covers, or a suffix could be dropped from the tuple and only
    # this count would notice.
    assert {Path(path).suffix for path, _ in offenders} == {".py", ".md", ".yaml"}
