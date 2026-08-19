r"""The two records describe the package that exists, not the one they described.

``DESIGN.md`` has the larger stale-able surface here, because this model is assembled entirely out of
imported parts and almost every claim in it is therefore inherited: a change in either parent can
falsify a sentence without touching a file in this package. Three parts of it are pinned
mechanically.

**The three linearisations.** The model, the task and the driver are each written out as an arrow
chain and each is compared against the real ``__mro__``. All three are diamonds or mixin-first
compositions, and the model's base order decides whether the tiled forward or the dense one wins -- a
silent reorder would change what the model trains on, and a document recording the old order would be
the only place a reader could go to find out.

**The parameter arithmetic.** Eight totals and two decompositions, checked against
``sum(p.numel() ...)`` on constructed models rather than against literals here, so a legitimate change
to a shared imported component re-costs the document instead of failing an unrelated assertion. Both
decompositions carry a claim -- the encoder axis is the two history encoders and the
input-representation axis is a horizon embedding and two adapters with the decoder head contributing
nothing -- and the arithmetic, attributed parameter name by parameter name, is what keeps those claims
true rather than merely written.

**The claims a reader could take too far**, each asserted against the code it describes rather than
against itself: the unconditional causality claim against the constructor signature that makes it
unconditional, the absent width keyword against the same signature, the empty bodies against
``vars``, and the metric surface against the driver's tuple.

``RESULTS.md`` is a pre-registration, and the rest of this file binds it to the code.

The document is written **before** the headline run, so that the criteria a run is judged against
cannot be chosen once the numbers are in view. That only means something if the document cannot
quietly drift from what the code emits, which is what the tests below check:

* every section the study needs is present, as a heading, so a run cannot arrive and find the
  document has stopped asking for something -- and the one heading the family carries that this
  record deliberately omits is asserted **absent**, with the omission stated;
* every metric name the document promises to fill in is one the task genuinely emits, and every
  readout this row added is named -- both directions, because a name the framework never emits is a
  cell nobody can fill, and an emitted readout nobody registered is one nobody will read;
* the revert record names files that exist, and names exactly the seven this row's arrival edited;
* the three rules that are easiest to lose in a rewrite -- that there is no evaluation package, that
  the nats are comparable to no cell outside this row, and that the sign of the gap is not a
  criterion -- are stated;
* and the document does not mention the planning artefact this package was built from. That scan
  covers every module and test in the package, not only the record.

One section is this cell's own, and it is the reason the cell exists: the record has to name **both**
edges of the square it sits at, and say which quantities are comparable along each. The encoder edge
compares loss *levels*, because both cells sum the same block over the same anchor count; the
input-representation edge cannot, and a record that read a level across it would be comparing two
different questions.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import List

import pytest

from teb_vae.lag_attn_crws.nets.causal_raw_inputs import CausalRawInputs
from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws
from teb_vae.lag_attn_crws.trainer import LagAttnCrwsTrainer
from teb_vae.lag_attn_transformer_crws.nets.model import SeqVaeLagAttnTrfCrws
from teb_vae.lag_attn_transformer_crws.task import SeqVaeLagAttnTrfCrwsTask
from teb_vae.lag_attn_transformer_crws.trainer import LagAttnTrfCrwsTrainer
from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer

# The parsing helpers, the constant lists and the measured totals come from the conv-LSTM cell of
# this row's copy rather than being restated. They describe the *row* -- eight models' parameter
# budgets, two markdown notations, the seven registered files, the eight dropped readouts -- and
# none is a property of an encoder; a second copy of the fixture would be a second place the eight
# constructors' keyword sets have to stay right, which is the same reason this package's conftest
# imports its data half rather than repeating it.
from teb_vae.lag_attn_crws.tests.test_docs import (  # noqa: E402
    BOTTLENECK_HEALTH_METRICS,
    CAUSAL_RAW_METRICS,
    DROPPED_READOUTS,
    FORBIDDEN_TOKENS,
    FORWARD_DICT_KEYS,
    OMITTED_HEADING,
    REVERT_FILES,
    STUDY_HEADINGS,
    _differing_names,
    _flat,
    _integers_stated_in,
    _markdown_section,
    _unseparated,
    measured_models,  # noqa: F401  -- bound here so pytest serves it to the tests below
    measured_totals,  # noqa: F401
)

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_DIR.parents[1]
_DESIGN = _PACKAGE_DIR / "DESIGN.md"
_RESULTS = _PACKAGE_DIR / "RESULTS.md"

#: Every suffix the tracked metric surface carries, so a documented name can be checked against what
#: a run genuinely logs rather than against a second hand-kept list.
_TRACKED_SUFFIXES = frozenset(
    name.split("/")[-1] for name in LagAttnTrfCrwsTrainer.TRACKED_METRICS
)


@pytest.fixture(scope="module")
def results() -> str:
    return _RESULTS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def design() -> str:
    return _DESIGN.read_text(encoding="utf-8")


def _linearisation_stated_for(text: str, cls: type) -> List[str]:
    """The linearisation the document writes out for a class, as a list of class names.

    Reads the arrow chain the document states -- ``` `A -> B -> C` ``` -- and returns
    ``["A", "B", "C"]``. The chain is allowed to be shorter than the real MRO and to wrap across a
    line, which the two Lightning-side diamonds both do.

    Args:
        text: The whole document.
        cls: The class whose chain to find, matched by name at the chain's head.

    Returns:
        The names in the order the document states them.
    """
    stated = re.search(rf"`{cls.__name__} -> ([^`]+)`", text)
    assert stated is not None, f"the document states no linearisation for {cls.__name__}"
    return [cls.__name__] + [name.strip() for name in stated.group(1).split("->")]


# =================================================================================================
# DESIGN.md: the three linearisations, against the real MROs
# =================================================================================================
@pytest.mark.parametrize(
    "cls", [SeqVaeLagAttnTrfCrws, SeqVaeLagAttnTrfCrwsTask, LagAttnTrfCrwsTrainer]
)
def test_each_documented_linearisation_is_the_real_one(design, cls):
    """Three arrow chains, each compared against ``__mro__`` rather than against prose.

    The model's is the one with teeth: the mixin must come first, or the dense forward wins and a
    $(B, T_{\\mathrm{valid}}, H, R)$ target is scored against an $A_{\\max}$-wide forecast. The chains
    are allowed to stop short of ``object``; what is compared is the prefix they state.
    """
    stated = _linearisation_stated_for(design, cls)
    real = [base.__name__ for base in cls.__mro__]

    assert stated == real[: len(stated)], (
        f"DESIGN.md states {stated} for {cls.__name__}; the real MRO begins {real[: len(stated)]}"
    )


def test_the_model_defines_only_a_constructor_and_the_task_defines_nothing(design):
    """The claim the whole record rests on: with nothing else defined here, a difference against
    the conv-LSTM cell of this row is attributable to the encoder alone and a difference against
    the conv-Transformer raw-signal cell to the input representation alone."""
    noise = {"__doc__", "__module__", "__dict__", "__weakref__", "__abstractmethods__", "_abc_impl"}
    model_own = set(vars(SeqVaeLagAttnTrfCrws)) - noise
    task_own = set(vars(SeqVaeLagAttnTrfCrwsTask)) - noise

    assert model_own == {"__init__"}, f"SeqVaeLagAttnTrfCrws defines {sorted(model_own)}"
    assert task_own == set(), f"SeqVaeLagAttnTrfCrwsTask defines {sorted(task_own)}"
    assert "a constructor and nothing else" in _flat(_markdown_section(design, "## 1. "))
    assert "defines **zero** callables" in _flat(_markdown_section(design, "## 7. "))


def test_the_driver_re_points_exactly_the_three_colliding_attributes(design):
    """All three collide, so resolution order alone would take the causal side, and each failure is
    silent: a conv-LSTM model built under this package's name, the same one layer up, or two
    models' checkpoints interleaved in one output tree."""
    section = _markdown_section(design, "## 7. ")

    assert LagAttnTrfCrwsTrainer.MODEL_CLS is SeqVaeLagAttnTrfCrws
    assert LagAttnTrfCrwsTrainer.TASK_CLS is SeqVaeLagAttnTrfCrwsTask
    assert LagAttnTrfCrwsTrainer.CHECKPOINT_STEM == "lag-attn-trf-crws"
    assert '"lag-attn-trf-crws"' in section
    assert LagAttnTrfCrwsTrainer.PLOT_CONFIG_KEY == "lag_attn_rws_plotting"
    assert "`PLOT_CONFIG_KEY` stays `\"lag_attn_rws_plotting\"`" in section
    own = {
        name for name in vars(LagAttnTrfCrwsTrainer) if not name.startswith("__")
    } - {"_abc_impl"}
    assert own == {"MODEL_CLS", "TASK_CLS", "CHECKPOINT_STEM"}, own


def test_the_documented_split_of_the_two_parents_is_where_each_member_resolves(design):
    """§7's table says which half comes from which parent, and several of its rows arrive by
    resolution order rather than by anything written down."""
    from teb_vae.lag_attn_rws.trainer import LagAttnRwsTrainer

    section = _flat(_markdown_section(design, "## 7. "))

    # The shared ancestor's, and re-pointed by neither parent.
    assert LagAttnTrfCrwsTrainer.TARGET_FIELDS is LagAttnRwsTrainer.TARGET_FIELDS
    assert "TARGET_FIELDS" not in vars(LagAttnCrwsTrainer)
    assert "TARGET_FIELDS" not in vars(LagAttnTrfRwsTrainer)
    assert "neither parent re-points it" in section
    # The causal-input parent's.
    assert LagAttnTrfCrwsTrainer.TRACKED_METRICS is LagAttnCrwsTrainer.TRACKED_METRICS
    assert LagAttnTrfCrwsTrainer.preflight.__func__ is LagAttnCrwsTrainer.preflight.__func__
    assert f"{len(LagAttnTrfCrwsTrainer.TRACKED_METRICS)} entries" in section
    # The one that passes through the causal parent because it defines no such hook.
    assert "compile_model_requested" not in vars(LagAttnCrwsTrainer)
    assert (
        LagAttnTrfCrwsTrainer.compile_model_requested
        is LagAttnTrfRwsTrainer.compile_model_requested
    )
    assert "resolves to the conv-Transformer side, and that is a decision" in section


# =================================================================================================
# DESIGN.md: the parameter arithmetic, pinned against constructed models
# =================================================================================================
def test_the_design_states_the_measured_totals(design, measured_totals):
    """Eight totals: both cells of this row and both raw-signal cells they are read against, each
    guarded and ungated."""
    stated = _integers_stated_in(_markdown_section(design, "## 13. "))

    for label, total in measured_totals.items():
        assert total in stated, (
            f"DESIGN.md §13 does not state the measured {label} total {total:,}"
        )


def test_the_headline_paragraph_carries_the_measured_total_and_both_comparisons(
    design, measured_totals
):
    """§1 is where a reader meets the number, and where a stale one would be read first."""
    stated = _integers_stated_in(_markdown_section(design, "## 1. "))

    assert measured_totals["trf_crws_guarded"] in stated
    assert measured_totals["crws_guarded"] in stated  # the encoder-axis comparison
    assert measured_totals["trf_rws_guarded"] in stated  # the input-representation one


def test_the_encoder_axis_delta_is_the_two_history_encoders(design, measured_totals):
    """Checked as arithmetic rather than as prose. It must be the same number at both targets and at
    both guards, since everything downstream of the encoders is a shared module in every pair --
    which is where the grid's claim that its axes are independent becomes a number."""
    causal_raw = measured_totals["crws_guarded"] - measured_totals["trf_crws_guarded"]
    raw = measured_totals["rws_guarded"] - measured_totals["trf_rws_guarded"]
    ungated = measured_totals["crws_ungated"] - measured_totals["trf_crws_ungated"]

    assert causal_raw == raw == ungated, (
        "the encoder swap no longer costs the same at every target and every guard"
    )
    assert causal_raw in _integers_stated_in(_markdown_section(design, "## 13. "))


def test_the_input_representation_delta_decomposes_into_the_two_terms_the_design_states(
    design, measured_totals, measured_models
):
    """The two input adapters, with the decoder head contributing nothing -- and it must be the same
    delta the conv-LSTM pair shows, because every module outside the encoders is shared. Attributed
    by parameter name, not only summed.

    **The horizon embedding used to be the second term and is now exactly zero**, because this cell
    forecasts $30$ steps like the raw-signal sibling it is compared against. It is computed rather
    than deleted so a future horizon divergence reappears as a failing sum."""
    section = _markdown_section(design, "## 13. ")
    delta = measured_totals["trf_crws_guarded"] - measured_totals["trf_rws_guarded"]

    horizon_embedding = (30 - 30) * 256
    adapters = 128 * (98 - 78) * 2 + 128 * (51 - 29) * 2 - 256

    assert horizon_embedding == 0, "the two horizons diverged; §13 needs its second term back"
    assert delta == horizon_embedding + adapters
    assert delta == measured_totals["crws_guarded"] - measured_totals["rws_guarded"]
    differing = _differing_names(
        measured_models["trf_crws_guarded"], measured_models["trf_rws_guarded"]
    )
    for name in differing:
        assert name.startswith(("target_adapter.", "source_adapter.")), name
    # Absent rather than present-with-a-value: the two embeddings are the same size now, which is
    # the parameter-level statement of the vanished second term.
    assert "horizon_core.horizon_embedding" not in differing
    assert (
        measured_models["trf_crws_guarded"].decoder.mean_head.out_features
        == measured_models["trf_rws_guarded"].decoder.mean_head.out_features
        == 16
    )
    for value in (delta, adapters, -horizon_embedding):
        assert str(value) in _unseparated(section), value


def test_the_guard_delta_is_the_two_availability_projections_alone(
    design, measured_totals, measured_models
):
    """Nothing in this target domain widens a head, so every parameter the budget adds is under an
    adapter -- asserted by name -- and the arithmetic §13 states for it is evaluated."""
    section = _markdown_section(design, "## 13. ")
    guard = measured_totals["trf_crws_guarded"] - measured_totals["trf_crws_ungated"]
    sibling = measured_totals["trf_rws_guarded"] - measured_totals["trf_rws_ungated"]

    assert guard == 128 * 98 + 128 * 51 - 128 * 4
    for name in _differing_names(
        measured_models["trf_crws_guarded"], measured_models["trf_crws_ungated"]
    ):
        assert "adapter" in name, name
    for value in (guard, sibling):
        assert str(value) in _unseparated(section), value


def test_the_records_agree_on_the_parameter_table(design, results, measured_totals):
    """The same totals appear in both documents, so a run's record and the design record cannot
    quote different budgets for one model."""
    in_design = _integers_stated_in(_markdown_section(design, "## 13. "))
    in_results = _integers_stated_in(_markdown_section(results, "## Parameter budget"))

    for label in (
        "trf_crws_guarded",
        "trf_crws_ungated",
        "crws_guarded",
        "trf_rws_guarded",
        "trf_rws_ungated",
    ):
        assert measured_totals[label] in in_design and measured_totals[label] in in_results, label


# =================================================================================================
# DESIGN.md: the claims a reader could take too far
# =================================================================================================
def test_the_causality_claim_is_unconditional_and_says_what_makes_it_so(design):
    """The one claim of this package that is genuinely stronger than the conv-LSTM cell of this
    row's rather than inherited, and it is a claim about a *keyword's absence*. Asserted against the
    constructor signature, so the prose cannot outlive the property."""
    section = _flat(_markdown_section(design, "## 8. "))

    assert "causal_norm" not in inspect.signature(SeqVaeLagAttnTrfCrws.__init__).parameters
    assert "causal_norm" in inspect.signature(SeqVaeLagAttnCrws.__init__).parameters
    assert "`causal_norm` is not a constructor keyword of this model at all" in section
    # And the half a reader could take too far: three facts meet here and they are independent.
    assert "Two causalities meet in this cell and they are independent" in section
    assert "not called a transfer entropy" in section


def test_the_width_keyword_is_absent_and_the_record_says_it_is_this_cells_own_property(design):
    """The one property this cell has that its conv-LSTM twin does not: no configuration can put the
    decoder and the raw target on different widths. Asserted against both signatures."""
    section = _flat(_markdown_section(design, "## 6. "))

    assert "decoder_out_channels" not in inspect.signature(SeqVaeLagAttnTrfCrws.__init__).parameters
    assert "decoder_out_channels" in inspect.signature(SeqVaeLagAttnCrws.__init__).parameters
    assert "There is no `decoder_out_channels` keyword" in section
    assert "_default_decoder_out_channels" not in vars(CausalRawInputs)
    assert "_default_decoder_out_channels" not in vars(SeqVaeLagAttnTrfCrws)


def test_the_design_names_both_edges_and_says_what_is_comparable_along_each(design):
    """The trap the square makes easy: both cells sum the same block over the same anchor count
    across the *encoder* edge, so a loss level is comparable there; across the *input-representation*
    edge the block is 240 against 480 and the anchor count 10.1 against 240, so it is not."""
    section = _flat(_markdown_section(design, "## 5. "))

    assert "The encoder edge, against `lag_attn_crws`: a loss *level* is comparable" in section
    assert (
        "The input-representation edge, against `lag_attn_transformer_rws`: a loss level is *not* comparable"
        in section
    )
    assert "Comparable across warm-up budgets within this model" in section
    assert "mutually unloadable checkpoints" in section


def test_the_mixin_section_states_why_inheritance_does_not_work(design):
    """The measured reason this is a mixin rather than an inheritance -- that
    ``(SeqVaeLagAttnCrws, SeqVaeLagAttnTrfRws)`` runs the conv-LSTM constructor -- and the property
    that keeps it a mixin: no constructor of its own."""
    from teb_vae.lag_attn_cfs.nets.causal_inputs import CausalWarmupInputs

    section = _flat(_markdown_section(design, "## 6. "))

    assert "runs the conv-LSTM constructor" in section
    assert "order of the bases is load-bearing" in section
    assert "*move*, not an abstraction" in section
    assert "One mixin, not two" in section
    assert "__init__" not in vars(CausalRawInputs)
    assert "__init__" not in vars(CausalWarmupInputs)


def test_the_lean_limits_carry_their_replacement_triggers(design):
    """A ``lean-limit`` note without a measurable trigger is a permanent excuse. Exactly three here,
    all inherited from the conv-LSTM cell of this row: the anchor floor that is a policy, the members
    written out rather than shared, and the absent evaluation package."""
    flat = _flat(design)

    assert len(re.findall(r"^> lean-limit: ", design, re.MULTILINE)) == 3
    assert "when a run shows the anchor count rather than the source pathway" in flat
    assert "when a third consumer of any of them appears" in flat
    assert "when a result from these cells is to be reported as a measurement" in flat


def test_the_design_states_that_there_is_no_evaluation_package(design):
    """Stated as a limitation with its follow-up named, and asserted against the directory."""
    section = _flat(_markdown_section(design, "## 14. "))

    assert "no `eval/` package" in section
    assert "`ModelBinding`" in section
    assert "No run checker ships" in section
    assert not (_PACKAGE_DIR / "eval").exists()
    assert not (_PACKAGE_DIR / "check_run.py").exists()
    assert not (_PACKAGE_DIR / "sample_page.py").exists()
    assert not (_PACKAGE_DIR / "plotting.py").exists()


def test_every_companion_document_the_design_defers_to_exists(design):
    """This record defers most of its claims to sibling documents, so each of them is load-bearing:
    a moved file turns the deferral into a dead end. A sibling under ``teb_vae/`` is cited by its
    package-relative path, as the whole family's records cite each other; a document inside *this*
    package is cited relative to the package. All three roots are tried."""
    referenced = sorted({match for match in re.findall(r"[\w/]+\.md", design) if "/" in match})

    assert len(referenced) >= 5, f"the design record defers to only {referenced}"
    roots = (_REPO_ROOT, _REPO_ROOT / "teb_vae", _PACKAGE_DIR)
    missing = [
        path for path in referenced if not any((root / path).is_file() for root in roots)
    ]
    assert missing == [], f"DESIGN.md defers to documents that do not exist: {missing}"


def test_every_launch_line_in_the_design_names_a_config_that_exists(design):
    """A launch line is copied and pasted; one naming a moved file fails at the shell with a message
    about a path rather than about a run."""
    referenced = sorted(
        set(re.findall(r"teb_vae/lag_attn_transformer_crws/configs/[\w.]+\.yaml", design))
    )
    shipped = sorted(path.name for path in (_PACKAGE_DIR / "configs").glob("*.yaml"))

    assert [Path(path).name for path in referenced] == shipped, (
        f"DESIGN.md §16 names {referenced}, not the shipped configs {shipped}"
    )
    missing = [path for path in referenced if not (_REPO_ROOT / path).is_file()]
    assert missing == [], missing


def test_the_documented_config_inventory_is_the_real_one(design):
    """§12 names the directory's contents, and a config added without a word there is an arm nobody
    reading the record knows exists -- and the one arm the conv-LSTM cell ships and this one does
    not is named as absent with its reason."""
    section = _flat(_markdown_section(design, "## 12. "))
    shipped = sorted(path.name for path in (_PACKAGE_DIR / "configs").glob("*.yaml"))

    for name in shipped:
        assert f"`{name}`" in section, name
    assert "No horizon arm ships" in section
    assert "sweep_horizon_30.yaml" not in shipped
    assert "nineteen" in section
    from teb_vae.lag_attn_transformer_crws.tests.test_config_load import PARITY_EXEMPT_PATHS

    assert len(PARITY_EXEMPT_PATHS) == 19


# =================================================================================================
# The document exists and asks for everything
# =================================================================================================
def test_the_record_exists_and_is_not_a_stub(results):
    assert len(results) > 4000, "RESULTS.md is too short to be a pre-registration"


@pytest.mark.parametrize("heading", STUDY_HEADINGS)
def test_every_required_section_is_present(heading, results):
    assert heading in results, heading


def test_the_distributed_smoke_heading_is_omitted_and_the_omission_is_stated(results):
    """The one heading the family carries that this record does not."""
    assert OMITTED_HEADING not in results
    assert "**No distributed-smoke table.**" in results
    assert "no production run is in scope" in _flat(results)


def test_the_two_criteria_tiers_are_both_present_and_distinguished(results):
    """The distinction is the point: Tier 1 asks whether the machinery did what it was built to do
    and a failure voids the run; Tier 2 is the science, and a fixed threshold on any of it would be
    a guess dressed as a gate."""
    assert "### Tier 1" in results
    assert "### Tier 2" in results
    assert "reported and interpreted, not passed or failed" in _flat(results)


def test_the_four_tier_one_criteria_are_registered(results):
    """Named individually rather than counted, so a criterion cannot be dropped and replaced. The
    fourth is this cell's own: the step-granular ramp this architecture sets."""
    tier_one = results.split("### Tier 1")[1].split("### Tier 2")[0]

    assert "`anchors_per_sample`" in tier_one
    assert "spike breaker never latches" in tier_one
    assert "identical metric row set" in tier_one
    assert "`lr_warmup_steps`" in tier_one
    assert "the two absences are deliberate" in _flat(tier_one)
    assert "target_warm_frac" not in tier_one


def test_the_five_tier_two_quantities_are_registered(results):
    tier_two = results.split("### Tier 2")[1].split("### The two edges")[0]

    for name in (
        "source_conditioned_kl_raw",
        "kld_active_frac",
        "logvar_prior_floor_frac",
        "kld_source_null",
        "shuffle_penalty",
        "pred_gap",
    ):
        assert name in tier_two, name


# =================================================================================================
# Every documented metric is one a run emits
# =================================================================================================
@pytest.mark.parametrize("name", CAUSAL_RAW_METRICS)
def test_every_added_readout_is_named_in_the_record(name, results):
    """The document is the only place a reader learns what these four columns are for; a run's CSV
    carries the names and nothing else."""
    assert f"`{name}`" in results, name


@pytest.mark.parametrize("name", CAUSAL_RAW_METRICS)
def test_every_added_readout_is_one_the_run_actually_tracks(name):
    """The other direction. A documented name the framework never emits is a cell nobody can
    fill."""
    assert name in _TRACKED_SUFFIXES, name


@pytest.mark.parametrize("name", DROPPED_READOUTS)
def test_every_dropped_readout_is_really_absent(name):
    """The eight columns a reader of the causal-feature records would expect and will not find."""
    assert name not in _TRACKED_SUFFIXES, name


@pytest.mark.parametrize("name", BOTTLENECK_HEALTH_METRICS)
def test_the_bottleneck_health_table_carries_every_readout(name, results):
    section = results.split("## Bottleneck health")[1].split("\n## ")[0]

    assert f"`{name}`" in section, name
    assert name in _TRACKED_SUFFIXES, name


def test_every_backticked_metric_name_in_the_record_is_one_the_run_emits(results):
    """The general form of the checks above, over the whole document. Restricted to names that look
    like metric identifiers, so config keys, file names and prose survive it. The two forward-dict
    keys and the eight dropped readouts share the shape and not the namespace and are admitted by
    name; the dropped ones are asserted absent from the tracked surface above, so admitting them
    here cannot hide a column that crept back in."""
    candidates = set(re.findall(r"`([a-z][a-z0-9_]{4,})`", results))
    metric_shaped = {
        name
        for name in candidates
        if name.startswith(("pred_gap", "nll_", "kld_", "logvar_", "mu_", "source_", "anchor"))
        or name in {"total_loss", "main_loss", "shuffle_penalty"}
    }

    unknown = sorted(
        name
        for name in metric_shaped
        if not any(tracked.startswith(name) for tracked in _TRACKED_SUFFIXES)
        and name not in FORWARD_DICT_KEYS
        and name not in DROPPED_READOUTS
    )
    assert unknown == [], unknown


# =================================================================================================
# The revert record, and the three rules
# =================================================================================================
def test_the_revert_record_is_by_file_and_names_exactly_the_seven_that_exist(results):
    """A list rather than an archaeology exercise, and an *exact* one: this row's arrival edited
    seven existing files and nothing else. Every named file exists and carries both strings."""
    section = results.split("### What reverts, and how")[1].split("\n### ")[0]

    paths = sorted(
        set(re.findall(r"^\| `((?:teb_vae|hdf5_dataset|scripts)/[\w./]+\.py)`", section, re.M))
    )
    assert paths == sorted(REVERT_FILES), paths
    for path in paths:
        source = (_REPO_ROOT / path).read_text(encoding="utf-8")
        assert '"lag_attn_crws"' in source and '"lag_attn_transformer_crws"' in source, path


def test_the_record_states_that_there_is_no_evaluation_package(results):
    """Stated once, near the top, so no number on the page is read as though it had a confidence
    interval."""
    flat = _flat(results)

    assert "There is no evaluation package" in flat
    assert "in-sample and carries no uncertainty" in flat
    assert "metrics_history.csv" in results
    assert "no evaluation line and no run-checker line" in flat
    assert not (_PACKAGE_DIR / "eval").exists()


def test_the_record_states_that_the_nats_are_comparable_only_within_this_row(results):
    """Both halves: not across the input-representation edge, because the block halved and the
    anchor count fell, and to the twin cell, because it ships the identical geometry."""
    flat = _flat(results)

    assert "240" in results and "480" in results
    assert "comparable only within this row" in flat
    assert "The one model whose loss level is comparable to this one's is `lag_attn_crws`" in flat


def test_the_record_states_that_the_sign_of_the_gap_is_not_a_criterion(results):
    """The finding this family is expected to reproduce, recorded before the run rather than
    explained after it."""
    assert "negative `pred_gap` is not a failure" in results


def test_the_record_names_both_edges_of_the_square(results):
    """The reason this cell exists. Against the conv-LSTM cell of this row the configs differ in the
    encoder alone; against the conv-Transformer raw-signal cell, in the input representation alone.
    A record that named only one would leave the other difference unattributed."""
    assert "lag_attn_crws" in results
    assert "lag_attn_transformer_rws" in results
    assert "### The two edges" in results


def test_the_record_says_which_quantities_are_comparable_along_which_edge(results):
    """The trap the square makes easy: both cells sum the same block over the same anchor count
    across the *encoder* edge, so a loss level is comparable there; across the *input-representation*
    edge the block is 240 against 480, so it is not, and only a sign and a trajectory can be read."""
    section = results.split("### The two edges")[1].split("\n## ")[0]

    assert "comparable: same block, same anchor count" in section
    assert "**not** comparable as a level" in section
    assert "no evaluation package for either cell" in _flat(section)


def test_the_record_states_the_unconditional_causality_claim(results):
    """The one architectural claim this cell can make that the conv-LSTM cell of this row cannot,
    and it is a claim about a *keyword's absence* rather than about a value -- so it belongs in the
    record rather than only in a test name."""
    assert "causal_norm" in results
    assert "unconditionally" in results


def test_the_record_states_which_loss_scale_constants_the_encoder_edge_moved(results):
    """The asymmetry the re-derivation found: the constants stated in nats of the summed block did
    not move across an edge that changes neither the block nor the anchor count, and the gradient
    clip -- a gradient statistic -- did. Pinned against the numbers ``tests/test_spike_breaker.py``
    brackets the constants with."""
    from teb_vae.lag_attn_transformer_crws.tests import test_spike_breaker as breaker

    section = results.split("## The loss-scale constants")[1].split("\n## ")[0]

    assert "Moved on the encoder edge?" in section
    assert "`gradient_clip_val`" in section and "1100.0" in section
    assert "`additive_margin`" in section and "5.0e+2" in section
    assert "re-measured" in section
    assert str(round(breaker.MEASURED_GRAD_Q99)) in section
    assert str(round(breaker.MEASURED_GRAD_MAX)) in section
    assert str(round(breaker.MEASURED_EXCURSION_MAX)) in section
    assert "rounded to 100" in _flat(section)


def test_every_launch_line_names_a_module_that_exists(results):
    """A launch line is copied and pasted; one naming a moved module fails at the shell with a
    message about an import rather than about a run."""
    section = results.split("## Launch lines")[1].split("\n## ")[0]

    modules = sorted(set(re.findall(r"-m (teb_vae[\w.]+)", section)))
    assert modules == ["teb_vae.lag_attn_transformer_crws.trainer"], modules
    for dotted in modules:
        assert (_REPO_ROOT / Path(*dotted.split("."))).with_suffix(".py").is_file(), dotted


def test_every_launch_line_names_a_config_that_exists(results):
    section = results.split("## Launch lines")[1].split("\n## ")[0]

    configs = sorted(set(re.findall(r"(teb_vae/[\w/]+/configs/[\w.]+\.yaml)", section)))
    shipped = sorted(path.name for path in (_PACKAGE_DIR / "configs").glob("*.yaml"))
    assert [Path(config).name for config in configs] == shipped, configs
    for config in configs:
        assert (_REPO_ROOT / config).exists(), config


# =================================================================================================
# Nothing points at the planning artefact
# =================================================================================================
@pytest.mark.parametrize("document", ("DESIGN.md", "RESULTS.md"))
@pytest.mark.parametrize("token", FORBIDDEN_TOKENS)
def test_neither_record_mentions_the_planning_document(token, document):
    assert token not in (_PACKAGE_DIR / document).read_text(encoding="utf-8"), (
        f"{document} mentions {token!r}"
    )


def test_no_module_or_test_in_the_package_mentions_the_planning_document():
    """The scan that matters more than the one above: a docstring or a test name pointing at a plan
    is a piece of code that stops being self-explaining the day the plan is archived."""
    offenders = []
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        # This file declares the token list, so it names every one of them by construction.
        if path.resolve() == Path(__file__).resolve():
            continue
        source = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_TOKENS:
            if token in source:
                offenders.append(f"{path.relative_to(_PACKAGE_DIR)}: {token}")

    assert offenders == [], offenders
