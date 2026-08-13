r"""The two records describe the package that exists, not the one they described.

``DESIGN.md`` has the larger stale-able surface here, because this model is assembled entirely out of
imported parts and almost every claim in it is therefore inherited: a change in either parent can
falsify a sentence without touching a file in this package. Three parts of it are pinned
mechanically.

**The three linearisations.** The model, the task and the driver are each written out as an arrow
chain and each is compared against the real ``__mro__``. All three are diamonds, and the model's base
order decides whether the decoder is built at the surviving channel count or at the raw grid — a
silent reorder would change what the model trains on, and a document recording the old order would be
the only place a reader could go to find out.

**The parameter arithmetic.** Six totals and three decompositions, checked against
``sum(p.numel() ...)`` on constructed models rather than against literals here, so a legitimate change
to a shared imported component re-costs the document instead of failing an unrelated assertion. Two
of the decompositions carry a claim — the encoder axis is the two history encoders and the target
axis is a decoder head, a horizon embedding and two adapters — and the arithmetic is what keeps those
claims true rather than merely written.

**The claims a reader could take too far**, each asserted against the code it describes rather than
against itself: the unconditional causality claim against the constructor signature that makes it
unconditional, the empty bodies against ``vars``, and the metric surface against the driver's tuple.

``RESULTS.md`` is a pre-registration, and the rest of this file binds it to the code.

The document is written **before** the headline run, so that the criteria a run is judged against
cannot be chosen once the numbers are in view. That only means something if the document cannot
quietly drift from what the code emits, which is what the tests below check:

* every section the study needs is present, as a heading, so a run cannot arrive and find the
  document has stopped asking for something;
* every metric name the document promises to fill in is one the task genuinely emits, and every
  readout this package added is named -- both directions, because a name the framework never emits
  is a cell nobody can fill, and an emitted readout nobody registered is one nobody will read;
* the revert record names files that exist;
* the two rules that are easiest to lose in a rewrite -- that there is no evaluation package, and
  that the nats are comparable to no other target domain -- are stated;
* and the document does not mention the planning artefact this package was built from. That scan
  covers every module and test in the package, not only the record: a design that documented itself
  by pointing at a plan would leave the code unexplained the day the plan is archived.

One section is this cell's own, and it is the reason the cell exists: the record has to name **both**
edges of the square it sits at, and say which quantities are comparable along each. The encoder edge
compares loss *levels*, because both cells sum the same block over the same anchor count; the target
edge cannot, and a record that read a level across it would be comparing two different questions.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import List

import pytest

from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs
from teb_vae.lag_attn_transformer_cfs.task import SeqVaeLagAttnTrfCfsTask
from teb_vae.lag_attn_transformer_cfs.trainer import LagAttnTrfCfsTrainer

# The parsing helpers and the measured totals come from the causal sibling's copy rather than being
# restated. They describe the *grid* -- four models' parameter budgets and two markdown notations --
# and neither is a property of an encoder; a second copy of the fixture would be a second place the
# four constructors' keyword sets have to stay right, which is the same reason this package's
# conftest imports its data half rather than repeating it.
from teb_vae.lag_attn_cfs.tests.test_docs import (  # noqa: E402
    _flat,
    _integers_stated_in,
    _markdown_section,
    _unseparated,
    measured_totals,  # noqa: F401  -- bound here so pytest serves it to the tests below
)

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_DIR.parents[1]
_DESIGN = _PACKAGE_DIR / "DESIGN.md"
_RESULTS = _PACKAGE_DIR / "RESULTS.md"

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
    "## The warm-up and the tiling",
)

#: The eight readouts this package adds, seven per stage and one on the evaluation stages alone.
CAUSAL_METRICS = (
    "target_warm_frac",
    "anchors_per_sample",
    "source_lag_warmth_frac_st",
    "source_lag_warmth_frac_ph",
    "pred_gap_warm_lo",
    "pred_gap_warm_mid",
    "pred_gap_warm_hi",
    "kld_source_null",
)

#: The six bottleneck-health readouts. A headline number can look healthy while the bottleneck is
#: not, and each of these is a different way for that to happen.
BOTTLENECK_HEALTH_METRICS = (
    "source_conditioned_kl_raw",
    "kld_active_frac",
    "mu_post_prior_gap_rms",
    "logvar_prior_floor_frac",
    "mu_prior_sat_frac",
    "delta_mu_sat_frac",
)

#: Tokens that name the planning artefact rather than the design. A record that cited one would go
#: stale the day the plan was archived, and the code would be left unexplained.
FORBIDDEN_TOKENS = (
    "SPEC_AND_SPRINTS",
    "sprint",
    "Sprint",
    "S0-T",
    "S1-T",
    "S2-T",
    "S3-T",
    "S4-T",
    "S5-T",
    "S6-T",
    "S7-T",
    "per the spec",
    "the roadmap",
)

#: Every suffix the tracked metric surface carries, so a documented name can be checked against what
#: a run genuinely logs rather than against a second hand-kept list.
_TRACKED_SUFFIXES = frozenset(
    name.split("/")[-1] for name in LagAttnTrfCfsTrainer.TRACKED_METRICS
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
    "cls", [SeqVaeLagAttnTrfCfs, SeqVaeLagAttnTrfCfsTask, LagAttnTrfCfsTrainer]
)
def test_each_documented_linearisation_is_the_real_one(design, cls):
    """Three arrow chains, each compared against ``__mro__`` rather than against prose.

    The model's is the one with teeth: the mixins must come first, or the decoder is built at
    $R = 16$ and a 98-channel block is scored against it. The chains are allowed to stop short of
    ``object``; what is compared is the prefix they state.
    """
    stated = _linearisation_stated_for(design, cls)
    real = [base.__name__ for base in cls.__mro__]

    assert stated == real[: len(stated)], (
        f"DESIGN.md states {stated} for {cls.__name__}; the real MRO begins {real[: len(stated)]}"
    )


def test_the_model_defines_only_a_constructor_and_the_task_defines_nothing(design):
    """The claim the whole record rests on: with nothing else defined here, a difference against
    the conv-LSTM causal cell is attributable to the encoder alone and a difference against the
    two-sided conv-Transformer cell to the transform alone."""
    noise = {"__doc__", "__module__", "__dict__", "__weakref__", "__abstractmethods__", "_abc_impl"}
    model_own = set(vars(SeqVaeLagAttnTrfCfs)) - noise
    task_own = set(vars(SeqVaeLagAttnTrfCfsTask)) - noise

    assert model_own == {"__init__"}, f"SeqVaeLagAttnTrfCfs defines {sorted(model_own)}"
    assert task_own == set(), f"SeqVaeLagAttnTrfCfsTask defines {sorted(task_own)}"
    assert "a constructor and nothing else" in _flat(_markdown_section(design, "## 1. "))
    assert "defines **zero** callables" in _flat(_markdown_section(design, "## 7. "))


def test_the_driver_re_points_exactly_the_three_colliding_attributes(design):
    """All three collide, so resolution order alone would take the causal side, and each failure is
    silent: a conv-LSTM model built under this package's name, the same one layer up, or two
    models' checkpoints interleaved in one output tree."""
    section = _markdown_section(design, "## 7. ")

    assert LagAttnTrfCfsTrainer.MODEL_CLS is SeqVaeLagAttnTrfCfs
    assert LagAttnTrfCfsTrainer.TASK_CLS is SeqVaeLagAttnTrfCfsTask
    assert LagAttnTrfCfsTrainer.CHECKPOINT_STEM == "lag-attn-trf-cfs"
    assert '"lag-attn-trf-cfs"' in section
    assert LagAttnTrfCfsTrainer.PLOT_CONFIG_KEY == "lag_attn_rws_plotting"
    assert "`PLOT_CONFIG_KEY` stays `\"lag_attn_rws_plotting\"`" in section


def test_the_documented_split_of_the_two_parents_is_where_each_member_resolves(design):
    """§7's table says which half comes from which parent, and two of its rows arrive by resolution
    order rather than by anything written down."""
    from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer
    from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer

    section = _flat(_markdown_section(design, "## 7. "))

    assert LagAttnTrfCfsTrainer.TARGET_FIELDS == LagAttnCfsTrainer.TARGET_FIELDS
    assert LagAttnTrfCfsTrainer.TRACKED_METRICS == LagAttnCfsTrainer.TRACKED_METRICS
    assert LagAttnTrfCfsTrainer.preflight.__func__ is LagAttnCfsTrainer.preflight.__func__
    assert f"{len(LagAttnTrfCfsTrainer.TRACKED_METRICS)} entries" in section
    # The one that passes through the causal parent because it defines no such hook.
    assert "compile_model_requested" not in vars(LagAttnCfsTrainer)
    assert (
        LagAttnTrfCfsTrainer.compile_model_requested
        is LagAttnTrfRwsTrainer.compile_model_requested
    )
    assert "resolves to the conv-Transformer side, and that is a decision" in section


# =================================================================================================
# DESIGN.md: the parameter arithmetic, pinned against constructed models
# =================================================================================================
def test_the_design_states_the_measured_totals(design, measured_totals):
    """Six totals: both cells of the causal row guarded and ungated, and the two-sided cell each is
    read against."""
    stated = _integers_stated_in(_markdown_section(design, "## 13. "))

    for label in (
        "trf_cfs_guarded",
        "trf_cfs_ungated",
        "cfs_guarded",
        "cfs_ungated",
        "trf_fs_guarded",
    ):
        assert measured_totals[label] in stated, (
            f"DESIGN.md §13 does not state the measured {label} total "
            f"{measured_totals[label]:,}"
        )


def test_the_headline_paragraph_carries_the_measured_total_and_both_comparisons(
    design, measured_totals
):
    """§1 is where a reader meets the number, and where a stale one would be read first."""
    stated = _integers_stated_in(_markdown_section(design, "## 1. "))

    assert measured_totals["trf_cfs_guarded"] in stated
    assert measured_totals["cfs_guarded"] in stated  # the encoder-axis comparison
    assert measured_totals["trf_fs_guarded"] in stated  # the target-axis one


def test_the_encoder_axis_delta_is_the_two_history_encoders(design, measured_totals):
    """Checked as arithmetic rather than as prose. It must be the same number at both targets and at
    both guards, since everything downstream of the encoders is a shared module in every pair --
    which is where the grid's claim that its axes are independent becomes a number."""
    causal = measured_totals["cfs_guarded"] - measured_totals["trf_cfs_guarded"]
    two_sided = measured_totals["fs_guarded"] - measured_totals["trf_fs_guarded"]
    ungated = measured_totals["cfs_ungated"] - measured_totals["trf_cfs_ungated"]

    assert causal == two_sided == ungated, (
        "the encoder swap no longer costs the same at every target and every guard"
    )
    assert causal in _integers_stated_in(_markdown_section(design, "## 13. "))


def test_the_target_axis_delta_decomposes_into_the_three_terms_the_design_states(
    design, measured_totals
):
    """The decoder head, the halved horizon's embedding and the two adapters — and it must be the
    same delta the conv-LSTM pair shows, because every module outside the encoders is shared."""
    section = _markdown_section(design, "## 13. ")
    delta = measured_totals["trf_cfs_guarded"] - measured_totals["trf_fs_guarded"]

    head = 514 * (98 - 78)
    horizon_embedding = -15 * 256
    adapters = 128 * (98 - 78) * 2 + 128 * (51 - 29) * 2 - 256

    assert delta == head + horizon_embedding + adapters
    assert delta == measured_totals["cfs_guarded"] - measured_totals["fs_guarded"]
    for value in (delta, head, adapters):
        assert str(value) in _unseparated(section), value


def test_the_guard_delta_is_stated_with_the_sign_it_actually_has(design, measured_totals):
    """Two correct numbers that read as a contradiction unless the reason is written down: the
    warm-up budget drops four channels so the availability projections dominate, while the two-sided
    reach budget drops thirty-one so the narrowing does."""
    section = _markdown_section(design, "## 13. ")
    causal = measured_totals["trf_cfs_guarded"] - measured_totals["trf_cfs_ungated"]

    assert causal == 128 * 98 + 128 * 51 - 514 * 4 - 128 * 4
    assert str(causal) in _unseparated(section)


def test_the_records_agree_on_the_parameter_table(design, results, measured_totals):
    """The same totals appear in both documents, so a run's record and the design record cannot
    quote different budgets for one model."""
    in_design = _integers_stated_in(_markdown_section(design, "## 13. "))
    in_results = _integers_stated_in(_markdown_section(results, "## Parameter budget"))

    for label in ("trf_cfs_guarded", "trf_cfs_ungated", "cfs_guarded", "trf_fs_guarded"):
        assert measured_totals[label] in in_design and measured_totals[label] in in_results, label


# =================================================================================================
# DESIGN.md: the claims a reader could take too far
# =================================================================================================
def test_the_causality_claim_is_unconditional_and_says_what_makes_it_so(design):
    """The one claim of this package that is genuinely stronger than the conv-LSTM causal cell's
    rather than inherited, and it is a claim about a *keyword's absence*. Asserted against the
    constructor signature, so the prose cannot outlive the property."""
    section = _flat(_markdown_section(design, "## 8. "))

    assert "causal_norm" not in inspect.signature(SeqVaeLagAttnTrfCfs.__init__).parameters
    assert "`causal_norm` is not a constructor keyword of this model at all" in section
    # And the half a reader could take too far: two causalities meet here and they are independent.
    assert "Two causalities meet in this cell and they are independent" in section
    assert "not called a transfer entropy" in section


def test_the_design_names_both_edges_and_says_what_is_comparable_along_each(design):
    """The trap the square makes easy: both cells sum the same block over the same anchor count
    across the *encoder* edge, so a loss level is comparable there; across the *target* edge the
    block is 1470 against 2340 and the horizon 15 against 30, so it is not."""
    section = _flat(_markdown_section(design, "## 5. "))

    assert "The encoder edge, against `lag_attn_cfs`: a loss *level* is comparable" in section
    assert "The target edge, against `lag_attn_transformer_fs`: a loss level is *not* comparable" in section
    assert "Not comparable across warm-up budgets within this model" in section
    assert "mutually unloadable checkpoints" in section


def test_the_mixin_section_states_why_neither_inheritance_works(design):
    """The measured reason this is two mixins rather than two inheritances -- that
    ``(SeqVaeLagAttnCfs, SeqVaeLagAttnTrfRws)`` runs the conv-LSTM constructor -- and the property
    that keeps them mixins."""
    from teb_vae.lag_attn_cfs.nets.causal_feature_target import CausalFeatureForecastTarget
    from teb_vae.lag_attn_cfs.nets.causal_inputs import CausalWarmupInputs

    section = _flat(_markdown_section(design, "## 6. "))

    assert "runs the conv-LSTM constructor" in section
    assert "order of the bases is load-bearing" in section
    assert "move*, not an abstraction" in section or "move, not an abstraction" in section
    # Neither mixin carries a constructor, which is what keeps the signature sweep seeing this
    # architecture's own parameters.
    assert "__init__" not in vars(CausalWarmupInputs)
    assert "__init__" not in vars(CausalFeatureForecastTarget)


def test_the_lean_limits_carry_their_replacement_triggers(design):
    """A ``lean-limit`` note without a measurable trigger is a permanent excuse. Exactly three here,
    all inherited from the target domain: the per-segment warm-up, the uncorrected group delay, and
    the lag floor that never varies."""
    flat = _flat(design)

    assert len(re.findall(r"^> lean-limit: ", design, re.MULTILINE)) == 3
    assert "when a measured run shows the anchor floor" in flat
    assert "when a lag result is to be reported as a physiological delay" in flat
    assert "when a run's `source_lag_warmth_frac_ph` falls below" in flat


def test_the_design_names_no_evaluation_entry_point(design):
    """The evaluation is deferred whole, and a launch line for one would be the most convincing
    possible way to imply otherwise."""
    section = _markdown_section(design, "## 16. ")
    commands = [line for line in section.splitlines() if "-m teb_vae" in line]

    assert commands, "§16 carries no launch lines"
    assert all(("trainer" in line) or ("check_run" in line) for line in commands)
    assert "There is no `eval` entry point" in section


def test_every_companion_document_the_design_defers_to_exists(design):
    """This record defers four claims to sibling documents, so each of them is load-bearing: a moved
    file turns the deferral into a dead end. A sibling under ``teb_vae/`` is cited by its
    package-relative path, as the whole family's records cite each other, so both roots are tried."""
    referenced = sorted({match for match in re.findall(r"[\w/]+\.md", design) if "/" in match})

    assert len(referenced) >= 4, f"the design record defers to only {referenced}"
    missing = [
        path
        for path in referenced
        if not (_REPO_ROOT / path).is_file() and not (_REPO_ROOT / "teb_vae" / path).is_file()
    ]
    assert missing == [], f"DESIGN.md defers to documents that do not exist: {missing}"


def test_every_launch_line_in_the_design_names_a_config_that_exists(design):
    """A launch line is copied and pasted; one naming a moved file fails at the shell with a message
    about a path rather than about a run."""
    referenced = sorted(
        set(re.findall(r"teb_vae/lag_attn_transformer_cfs/configs/[\w.]+\.yaml", design))
    )

    assert len(referenced) == 3, f"DESIGN.md §16 names {referenced}, not all three launch configs"
    missing = [path for path in referenced if not (_REPO_ROOT / path).is_file()]
    assert missing == [], missing


def test_the_documented_config_inventory_is_the_real_one(design):
    """§12 names the directory's contents, and a config added without a word there is an arm nobody
    reading the record knows exists."""
    section = _flat(_markdown_section(design, "## 12. "))
    shipped = sorted(path.name for path in (_PACKAGE_DIR / "configs").glob("*.yaml"))

    for name in shipped:
        assert f"`{name}`" in section, name
    assert len(re.findall(r"`[\w.]+\.yaml`", section)) == len(shipped)


# =================================================================================================
# The document exists and asks for everything
# =================================================================================================
def test_the_record_exists_and_is_not_a_stub(results):
    assert len(results) > 4000, "RESULTS.md is too short to be a pre-registration"


@pytest.mark.parametrize("heading", STUDY_HEADINGS)
def test_every_required_section_is_present(heading, results):
    assert heading in results, heading


def test_the_two_criteria_tiers_are_both_present_and_distinguished(results):
    """The distinction is the point: Tier 1 asks whether the machinery did what it was built to do
    and a failure voids the run; Tier 2 is the science, and a fixed threshold on any of it would be
    a guess dressed as a gate."""
    assert "### Tier 1" in results
    assert "### Tier 2" in results
    assert "reported and interpreted, not passed or failed" in results


def test_the_five_tier_one_criteria_are_registered(results):
    """Named individually rather than counted, so a criterion cannot be dropped and replaced."""
    tier_one = results.split("### Tier 1")[1].split("### Tier 2")[0]

    assert "`target_warm_frac` is exactly `1.0`" in tier_one
    assert "`anchors_per_sample`" in tier_one
    assert "spike breaker never latches" in tier_one
    assert "identical metric row set" in tier_one
    # The recomposition criterion, and the reason it is stated split-against-split.
    assert "pred_gap_warm_lo" in tier_one and "pred_gap_st" in tier_one
    assert "cancellation" in results


def test_the_four_tier_two_quantities_are_registered(results):
    tier_two = results.split("### Tier 2")[1]

    for name in (
        "source_conditioned_kl_raw",
        "kld_active_frac",
        "logvar_prior_floor_frac",
        "kld_source_null",
        "shuffle_penalty",
    ):
        assert name in tier_two, name


# =================================================================================================
# Every documented metric is one a run emits
# =================================================================================================
@pytest.mark.parametrize("name", CAUSAL_METRICS)
def test_every_added_readout_is_named_in_the_record(name, results):
    """The document is the only place a reader learns what these eight columns are for; a run's CSV
    carries the names and nothing else."""
    assert f"`{name}`" in results, name


@pytest.mark.parametrize("name", CAUSAL_METRICS)
def test_every_added_readout_is_one_the_run_actually_tracks(name):
    """The other direction. A documented name the framework never emits is a cell nobody can
    fill."""
    assert name in _TRACKED_SUFFIXES, name


@pytest.mark.parametrize("name", BOTTLENECK_HEALTH_METRICS)
def test_the_bottleneck_health_table_carries_every_readout(name, results):
    section = results.split("## Bottleneck health")[1].split("\n## ")[0]

    assert f"`{name}`" in section, name
    assert name in _TRACKED_SUFFIXES, name


def test_every_backticked_metric_name_in_the_record_is_one_the_run_emits(results):
    """The general form of the two checks above, over the whole document. Restricted to names that
    look like metric identifiers, so config keys, file names and prose survive it."""
    candidates = set(re.findall(r"`([a-z][a-z0-9_]{4,})`", results))
    metric_shaped = {
        name
        for name in candidates
        if name.startswith(("pred_gap", "nll_", "kld_", "logvar_", "mu_", "source_", "anchor"))
        or name in {"total_loss", "main_loss", "shuffle_penalty", "target_warm_frac"}
    }

    # A *family* name is admitted -- ``source_lag_warmth_frac`` stands for its two per-block
    # columns, and prose that had to spell both out every time would be worse prose. What is not
    # admitted is a name no tracked column starts with, which is the typo this catches.
    unknown = sorted(
        name
        for name in metric_shaped
        if not any(tracked.startswith(name) for tracked in _TRACKED_SUFFIXES)
    )
    assert unknown == [], unknown


# =================================================================================================
# The revert record, and the two rules
# =================================================================================================
def test_the_revert_record_is_by_file_and_every_file_exists(results):
    """A list rather than an archaeology exercise. Each row names a path outside this package that
    the package's arrival edited; a row naming a file that is gone is a revert instruction that
    would fail halfway through."""
    section = results.split("### What reverts, and how")[1].split("\n### ")[0]
    repo_root = _PACKAGE_DIR.parents[1]

    paths = sorted(set(re.findall(r"`((?:teb_vae|hdf5_dataset|scripts)/[\w./]+)`", section)))
    assert len(paths) >= 10, paths
    for path in paths:
        assert (repo_root / path).exists(), path


def test_the_revert_record_names_the_shared_seams_outside_this_package(results):
    """Named individually, because the ones outside the obvious directory are the ones a revert
    forgets: the loader's boundary reader, the fixture builder and the two committed binaries."""
    section = results.split("### What reverts, and how")[1].split("\n### ")[0]

    for path in (
        "teb_vae/lag_attn_rws/nets/raw_masks.py",
        "teb_vae/lag_attn_rws/nets/losses.py",
        "teb_vae/lag_attn_rws/nets/controls.py",
        "teb_vae/lag_attn_fs/nets/feature_target.py",
        "teb_vae/lag_attn/nets/encoders.py",
        "teb_vae/lag_attn_rws/plotting.py",
        "teb_vae/lag_attn_rws/sample_page.py",
        "hdf5_dataset/hdf5_dataset.py",
        "scripts/make_tiny_shard.py",
    ):
        assert f"`{path}`" in section, path


def test_the_record_states_that_there_is_no_evaluation_package(results):
    """Stated once, near the top, so no number on the page is read as though it had a confidence
    interval."""
    assert "no evaluation package" in results
    assert "metrics_history.csv" in results
    assert "uncertainty" in results


def test_the_record_states_that_the_nats_are_comparable_to_no_other_target_domain(results):
    """Both halves: across the target axis, because the block differs, and across budgets within
    this model, because ``C_keep`` is what the budget decides."""
    assert "1470" in results
    assert "2340" in results
    assert "two budgets are not comparable" in results


def test_the_record_states_that_the_sign_of_the_gap_is_not_a_criterion(results):
    """The finding this family is expected to reproduce, recorded before the run rather than
    explained after it."""
    assert "negative `pred_gap` is not a failure" in results


def test_no_launch_line_names_an_evaluation_entry_point(results):
    """There is none, and a line naming one would be a launch that fails on the box."""
    section = results.split("## Launch lines")[1].split("\n## ")[0]

    assert ".eval" not in section
    assert "eval.run" not in section


def test_every_launch_line_names_a_config_that_exists(results):
    section = results.split("## Launch lines")[1].split("\n## ")[0]
    repo_root = _PACKAGE_DIR.parents[1]

    configs = sorted(set(re.findall(r"(teb_vae/[\w/]+/configs/[\w.]+\.yaml)", section)))
    assert configs, "no launch line names a config at all"
    for config in configs:
        assert (repo_root / config).exists(), config


# =================================================================================================
# Nothing points at the planning artefact
# =================================================================================================
@pytest.mark.parametrize("token", FORBIDDEN_TOKENS)
def test_the_record_does_not_mention_the_planning_document(token, results):
    assert token not in results, token


def test_the_record_names_both_edges_of_the_square(results):
    """The reason this cell exists. Against the conv-LSTM causal cell the configs differ in the
    encoder alone; against the conv-Transformer two-sided cell, in the target domain alone. A record
    that named only one would leave the other difference unattributed."""
    assert "lag_attn_cfs" in results
    assert "lag_attn_transformer_fs" in results
    assert "### The two edges" in results


def test_the_record_says_which_quantities_are_comparable_along_which_edge(results):
    """The trap the square makes easy: both cells sum the same block over the same anchor count
    across the *encoder* edge, so a loss level is comparable there; across the *target* edge the
    block is 1470 against 2340, so it is not, and only a sign and a trajectory can be read."""
    section = results.split("### The two edges")[1].split("\n## ")[0]

    assert "comparable: same block, same anchor count" in section
    assert "**not** comparable as a level" in section


def test_the_record_states_the_unconditional_causality_claim(results):
    """The one architectural claim this cell can make that the conv-LSTM causal cell cannot, and it
    is a claim about a *keyword's absence* rather than about a value -- so it belongs in the record
    rather than only in a test name."""
    assert "causal_norm" in results
    assert "unconditionally" in results


def test_the_record_states_which_loss_scale_constants_the_encoder_edge_moved(results):
    """The asymmetry the re-derivation found: the constants stated in nats of the summed block must
    not move across an edge that changes neither the block nor the anchor count, and the gradient
    clip -- a gradient statistic -- must."""
    section = results.split("## The loss-scale constants")[1].split("\n## ")[0]

    assert "Moved on the encoder edge?" in section
    assert "`gradient_clip_val`" in section
    assert "`additive_margin`" in section
    assert "re-measured" in section


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
