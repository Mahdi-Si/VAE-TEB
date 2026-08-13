r"""The two records describe the package that exists, not the one they described.

This package has no evaluation pipeline, so a run produces exactly two durable artefacts: its own
``train_results/metrics_history.csv`` and what someone transcribes from it. There is no verdict file,
no bootstrap interval and no per-recording table to catch a mistake in the copy, which makes the
documents the record -- and a record with no test is a record that goes stale in the direction nobody
notices.

``DESIGN.md`` is pinned on three things a reader would otherwise take on trust. **The parameter
arithmetic**: six totals and three decompositions, checked against ``sum(p.numel() ...)`` on
constructed models rather than against literals here, so a legitimate change to a shared imported
component re-costs the document instead of failing an unrelated assertion -- and the *stated
factorisations* are evaluated rather than merely found, because a section carrying the right delta
beside a wrong decomposition of it is exactly the half a search for the number cannot see. **The
linearisation**, compared against the real ``__mro__``: the base order decides whether the decoder is
built at the surviving channel count or at the raw grid, and a document recording the old order would
be the only place a reader could go to find out. And **the claims a reader could take too far**, each
asserted against the code it describes rather than against itself -- the block split against the
class attribute, the metric surface against the driver's own tuple, the tiling against the
constructor's signature.

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
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Dict, List, Set

import pytest

from teb_vae.lag_attn_cfs.model_kwargs import WARMUP_MODEL_KWARGS
from teb_vae.lag_attn_cfs.nets.causal_feature_target import CausalFeatureForecastTarget
from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs
from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer

from .conftest import shipped_warmup_kwargs

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_DIR.parents[1]
_DESIGN = _PACKAGE_DIR / "DESIGN.md"
_RESULTS = _PACKAGE_DIR / "RESULTS.md"

#: Any integer of seven digits or more, in either notation the records use: plain markdown
#: (``5,143,262``) or LaTeX with braced separators (``5{,}143{,}262``). Both appear, because a number
#: inside a maths span must brace its separators to keep the spacing, and pinning one notation would
#: leave half of the arithmetic unchecked.
_LARGE_NUMBER_PATTERN = re.compile(r"(\d{1,3}(?:(?:\{,\}|,)\d{3})+)")

#: A factorised product as the records write it -- ``514 \times (98 - 78)``. Captured so the stated
#: arithmetic can be *evaluated* rather than merely found.
_FACTORISATION_PATTERN = re.compile(r"(\d+) \\times \((\d+) - (\d+)\)")

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
    name.split("/")[-1] for name in LagAttnCfsTrainer.TRACKED_METRICS
)


@pytest.fixture(scope="module")
def results() -> str:
    return _RESULTS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def design() -> str:
    return _DESIGN.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def measured_totals() -> Dict[str, int]:
    """Parameter totals of all four cells the two records quote, measured in one process.

    Built from each suite's own production keyword set rather than from the configs, so this file
    binds the documents to the *architectures* while ``test_config_load.py`` binds the configs to
    the driver -- two independent routes to the same widths. The three comparison models are built
    from different keyword sets by necessity: their constructors' schemas differ by the five
    conv-LSTM-only keywords and the seven encoder ones.
    """
    from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
    from teb_vae.lag_attn_fs.tests.conftest import shipped_gated_kwargs
    from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs
    from teb_vae.lag_attn_transformer_cfs.tests.conftest import (
        shipped_warmup_kwargs as trf_warmup_kwargs,
    )
    from teb_vae.lag_attn_transformer_fs.nets.model import SeqVaeLagAttnTrfFs
    from teb_vae.lag_attn_transformer_fs.tests.conftest import (
        shipped_gated_kwargs as trf_gated_kwargs,
    )

    def total(cls, kwargs) -> int:
        return sum(parameter.numel() for parameter in cls(**kwargs).parameters())

    def ungated(kwargs) -> Dict[str, object]:
        # The four resolved channel tuples removed and nothing else, so the guarded and ungated
        # arms differ in the guard alone rather than in a second hand-written keyword set.
        return {key: value for key, value in kwargs.items() if key not in WARMUP_MODEL_KWARGS}

    return {
        "cfs_guarded": total(SeqVaeLagAttnCfs, shipped_warmup_kwargs()),
        "cfs_ungated": total(SeqVaeLagAttnCfs, ungated(shipped_warmup_kwargs())),
        "trf_cfs_guarded": total(SeqVaeLagAttnTrfCfs, trf_warmup_kwargs()),
        "trf_cfs_ungated": total(SeqVaeLagAttnTrfCfs, ungated(trf_warmup_kwargs())),
        "fs_guarded": total(SeqVaeLagAttnFs, shipped_gated_kwargs(120.0)),
        "fs_ungated": total(SeqVaeLagAttnFs, shipped_gated_kwargs(None)),
        "trf_fs_guarded": total(SeqVaeLagAttnTrfFs, trf_gated_kwargs(120.0)),
    }


def _markdown_section(text: str, heading: str) -> str:
    """The body of one ``##`` section, from its heading to the next one or the end."""
    start = text.index(heading)
    remainder = text[start + len(heading):]
    end = remainder.find("\n## ")
    return remainder if end < 0 else remainder[:end]


def _flat(text: str) -> str:
    """Collapse whitespace, and drop leading blockquote markers.

    Every phrase assertion below runs against this rather than against the raw document. The files
    are hard-wrapped, so any phrase long enough to be worth pinning is eventually split across a
    line by an edit elsewhere in its paragraph -- and a test that then failed would be reporting a
    reflow rather than a lost claim. The blockquote markers go too, because a ``lean-limit`` note is
    a blockquote and a phrase wrapped inside one would otherwise carry a stray ``>``.
    """
    unquoted = (re.sub(r"^\s*>\s?", "", line) for line in text.splitlines())
    return " ".join(" ".join(unquoted).split())


def _unseparated(text: str) -> str:
    r"""``text`` with both thousands notations stripped, so a number can be searched for as digits.

    The LaTeX form goes **first**: ``5{,}143{,}262`` under a plain comma strip would become
    ``5{}143{}262``, and every subsequent search for a number inside a maths span would silently
    find nothing.
    """
    return text.replace("{,}", "").replace(",", "")


def _integers_stated_in(text: str) -> Set[int]:
    """Every large integer the text states, in either of the two notations it uses."""
    return {int(_unseparated(match)) for match in _LARGE_NUMBER_PATTERN.findall(text)}


def _linearisation_stated_for(text: str, cls: type) -> List[str]:
    """The linearisation the document writes out for a class, as a list of class names.

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
# DESIGN.md: the parameter arithmetic, pinned against constructed models
# =================================================================================================
def test_the_design_states_the_measured_totals(design, measured_totals):
    """Six totals -- both new cells guarded and ungated, and the two two-sided comparisons --
    checked against ``sum(p.numel() ...)`` rather than against literals in a test, so a legitimate
    change to a shared imported component re-costs the document."""
    section = _markdown_section(design, "## 13. ")
    stated = _integers_stated_in(section)

    for label in (
        "cfs_guarded",
        "cfs_ungated",
        "trf_cfs_guarded",
        "trf_cfs_ungated",
        "fs_guarded",
    ):
        assert measured_totals[label] in stated, (
            f"DESIGN.md §13 does not state the measured {label} total "
            f"{measured_totals[label]:,}"
        )


def test_the_headline_paragraph_carries_the_measured_total_and_both_comparisons(
    design, measured_totals
):
    """§1 is where a reader meets the number, and where a stale one would be read first."""
    section = _integers_stated_in(_markdown_section(design, "## 1. "))

    assert measured_totals["cfs_guarded"] in section
    assert measured_totals["fs_guarded"] in section  # the target-axis comparison
    assert measured_totals["trf_cfs_guarded"] in section  # the encoder-axis one


def test_the_target_axis_delta_decomposes_into_the_three_terms_the_design_states(
    design, measured_totals
):
    r"""The decoder head, the halved horizon's embedding and the two adapters. Each is *evaluated*
    from the document's own factorisation rather than merely searched for, because a section
    carrying the right total beside a wrong decomposition of it is the half a reader takes on
    trust."""
    section = _markdown_section(design, "## 13. ")
    delta = measured_totals["cfs_guarded"] - measured_totals["fs_guarded"]

    head = 514 * (98 - 78)
    horizon_embedding = -15 * 256
    adapters = 128 * (98 - 78) * 2 + 128 * (51 - 29) * 2 - 256

    assert delta == head + horizon_embedding + adapters, (
        f"the target-axis delta is {delta:+,}, which no longer decomposes into the decoder head "
        f"({head:+,}), the horizon embedding ({horizon_embedding:+,}) and the adapters "
        f"({adapters:+,})"
    )
    # The stated factorisation of the head term, evaluated.
    factorisations = {
        int(cost) * (int(wide) - int(narrow))
        for cost, wide, narrow in _FACTORISATION_PATTERN.findall(section)
    }
    assert head in factorisations, (
        f"§13 factorises nothing as the measured head cost {head:,}; it states {factorisations}"
    )
    for value in (delta, head, adapters):
        assert str(value) in _unseparated(section), value


def test_the_encoder_axis_delta_is_the_two_history_encoders(design, measured_totals):
    """Checked as arithmetic rather than as prose: the reduction must be the same number the
    two-sided feature pair sees at its own budget, since everything downstream of the encoders is a
    shared module in both pairs -- which is the claim §13 makes and the reason it is quoted."""
    from teb_vae.lag_attn_fs.tests.conftest import shipped_gated_kwargs
    from teb_vae.lag_attn_transformer_fs.nets.model import SeqVaeLagAttnTrfFs
    from teb_vae.lag_attn_transformer_fs.tests.conftest import (
        shipped_gated_kwargs as trf_gated_kwargs,
    )

    causal = measured_totals["cfs_guarded"] - measured_totals["trf_cfs_guarded"]
    two_sided = measured_totals["fs_guarded"] - measured_totals["trf_fs_guarded"]
    ungated = measured_totals["cfs_ungated"] - measured_totals["trf_cfs_ungated"]

    assert causal == two_sided == ungated, (
        "the encoder swap no longer costs the same at every target and every guard, so the "
        "reduction is not the two history encoders alone"
    )
    assert causal in _integers_stated_in(_markdown_section(design, "## 13. "))
    # The two comparison classes really are the ones the deltas were taken between.
    assert shipped_gated_kwargs(120.0)["c_y"] == 109
    assert SeqVaeLagAttnTrfFs(**trf_gated_kwargs(120.0)).decoder_out_channels == 78


def test_the_guard_delta_is_stated_with_the_sign_it_actually_has(design, measured_totals):
    """Two correct numbers that read as a contradiction unless the reason is written down: the
    warm-up budget drops four channels so the availability projections dominate, while the
    two-sided reach budget drops thirty-one so the narrowing does."""
    section = _markdown_section(design, "## 13. ")
    causal = measured_totals["cfs_guarded"] - measured_totals["cfs_ungated"]
    two_sided = measured_totals["fs_guarded"] - measured_totals["fs_ungated"]

    assert causal > 0 > two_sided, (
        f"the guard now costs {causal:+,} here and {two_sided:+,} there; the section explains a "
        f"sign difference that no longer exists"
    )
    # The three terms, as arithmetic rather than as three numbers that happen to be printed.
    assert causal == 128 * 98 + 128 * 51 - 514 * 4 - 128 * 4
    for value in (causal, two_sided, 128 * 98 + 128 * 51):
        assert str(abs(value)) in _unseparated(section), value


def test_the_records_agree_on_the_parameter_table(design, results, measured_totals):
    """The same three totals appear in both documents, so a run's record and the design record
    cannot quote different budgets for one model."""
    in_design = _integers_stated_in(_markdown_section(design, "## 13. "))
    in_results = _integers_stated_in(_markdown_section(results, "## Parameter budget"))

    for label in ("cfs_guarded", "cfs_ungated", "fs_guarded"):
        assert measured_totals[label] in in_design and measured_totals[label] in in_results, label


# =================================================================================================
# DESIGN.md: the linearisation, and the claims a reader could take too far
# =================================================================================================
def test_the_documented_linearisation_is_the_real_one(design):
    """Compared against ``__mro__`` rather than against prose. The base order decides whether the
    decoder is built at the surviving channel count or at the raw grid, and the document would be
    the only place a reader could go to find out that it had moved."""
    stated = _linearisation_stated_for(design, SeqVaeLagAttnCfs)
    real = [base.__name__ for base in SeqVaeLagAttnCfs.__mro__]

    assert stated == real[: len(stated)], (
        f"DESIGN.md states {stated}; the real MRO begins {real[: len(stated)]}"
    )


def test_the_model_class_really_defines_only_a_constructor(design):
    """The claim §1 and §6 both rest on: with nothing else defined here the forward keys, the
    posterior structure, the lag map and the objective's metric set cannot have moved, because they
    are the mixins' and the base's own code objects."""
    own = set(vars(SeqVaeLagAttnCfs)) - {"__doc__", "__module__", "__dict__", "__weakref__"}

    assert own == {"__init__"}, f"SeqVaeLagAttnCfs defines {sorted(own)}"
    assert "a constructor and nothing else" in _flat(_markdown_section(design, "## 1. "))


def test_the_two_replaced_delay_keywords_are_gone_and_the_four_new_ones_are_there(design):
    """A warm-up is a leading *mask* and ``ChannelDelay`` is a *shift*, so a warm-up routed under a
    delay name would train a different model with every shape intact. Asserted against the
    constructor rather than against the sentence that says so."""
    parameters = set(inspect.signature(SeqVaeLagAttnCfs.__init__).parameters)
    section = _flat(_markdown_section(design, "## 6. "))

    assert {"target_delays", "source_delays"} & parameters == set()
    for name in ("target_warmup_steps", "source_warmup_steps", "anchor_stride", "lag_floor"):
        assert name in parameters, name
        assert f"`{name}`" in design, name
    assert "order of the bases is load-bearing" in section


def test_the_block_split_is_pinned_against_the_class_attribute(design):
    """It cannot be derived -- $c_y$ is the two blocks' *sum* -- so the document is one of only two
    places the number lives, and a stale value would mislabel two reported columns and break
    nothing."""
    section = _markdown_section(design, "## 10. ")
    stated = re.search(r"`TARGET_BLOCK_SPLIT` re-pointed\s+from \$43\$ to \$(\d+)\$", _flat(section))

    assert stated is not None, "DESIGN.md §10 no longer states the block split"
    assert int(stated.group(1)) == CausalFeatureForecastTarget.TARGET_BLOCK_SPLIT
    assert CausalFeatureForecastTarget.SOURCE_BLOCK_SPLIT == 36
    assert "`SOURCE_BLOCK_SPLIT = 36`" in section


def test_the_documented_metric_count_is_the_drivers_own(design):
    """A hand-kept number in prose beside a computed tuple is the pair most likely to drift."""
    section = _markdown_section(design, "## 10. ")

    assert f"**{len(LagAttnCfsTrainer.TRACKED_METRICS)}**" in section
    # And the one deliberate asymmetry in the surface, asserted in both directions.
    assert "train/kld_source_null" not in LagAttnCfsTrainer.TRACKED_METRICS
    assert "val/kld_source_null" in LagAttnCfsTrainer.TRACKED_METRICS


def test_the_geometry_section_states_the_pairing_and_the_measured_channel_counts(design, budget):
    """Against the budget the committed fixture resolves to, not against constants: a fixture
    rebuilt at another quantile changes both the warm-up vectors and the stored channel count, and
    the document would be describing a boundary the data no longer has."""
    section = _flat(_markdown_section(design, "## 3. "))

    assert f"{budget.target.kept_width} of {budget.target.declared_width}" in section
    assert f"$B = {budget.target.max_warmup}$" in section
    assert "F \\ge B - 1" in section
    for name, kept, declared in budget.target.block_counts():
        assert f"`{name}` ${kept}/{declared}$" in section, name


def test_the_source_section_states_that_the_source_is_never_gated(design, budget):
    """The compromise this design makes, and the reason the two warmth columns exist. Asserted
    against the resolved budget: the source keep-index is the identity."""
    section = _flat(_markdown_section(design, "## 8. "))

    assert budget.source.kept_width == budget.source.declared_width
    assert f"all {budget.source.declared_width} source channels are kept" in section
    assert "small value there is the expected finding, not a failure" in section
    assert "the coupling readout is measuring a clock" in section


def test_the_design_states_the_two_ways_the_nats_are_incomparable(design):
    """Both halves, because the second is the one a reader of the first would not guess: the
    warm-up budget moves the surviving-channel count, hence the decoder width, hence the block every
    nat is summed over."""
    section = _flat(_markdown_section(design, "## 5. "))

    assert "Not comparable across the target axis" in section
    assert "Not comparable across warm-up budgets within this model" in section
    assert "mutually unloadable checkpoints" in section


def test_the_design_states_that_this_is_an_experiment_rather_than_a_remedy(design):
    """The one framing error that would make a correct negative result read as a failure."""
    section = _flat(_markdown_section(design, "## 1. "))

    assert "experiment, not a remedy" in section
    assert "expected to reproduce" in section
    assert "sign of `pred_gap` is a criterion nowhere" in section


def test_the_design_records_why_the_latent_gap_readout_is_re_pointed(design):
    """The inherited readout whose own docstring promises an invariant that tiling breaks. It is
    overridden on the task, and the document is where a reader learns the column changed meaning."""
    section = _flat(_markdown_section(design, "## 10. "))

    assert "_mu_gap_rms" in section
    assert "mu_post_prior_gap_rms" in section
    assert "restores the property the function already claims" in section


def test_the_lean_limits_carry_their_replacement_triggers(design):
    """A ``lean-limit`` note without a measurable trigger is a permanent excuse. Exactly three here:
    the per-segment warm-up, the uncorrected group delay, and the lag floor that never varies."""
    flat = _flat(design)

    assert len(re.findall(r"^> lean-limit: ", design, re.MULTILINE)) == 3
    assert "when a measured run shows the anchor floor" in flat
    assert "when a lag result is to be reported as a physiological delay" in flat
    assert "when a run's `source_lag_warmth_frac_ph` falls below" in flat


def test_the_design_names_no_evaluation_entry_point(design):
    """The evaluation is deferred whole, and a launch line for one would be the most convincing
    possible way to imply otherwise -- so the section says the absence out loud and carries no
    command that would contradict it."""
    section = _markdown_section(design, "## 16. ")
    commands = [line for line in section.splitlines() if "-m teb_vae" in line]

    assert commands, "§16 carries no launch lines"
    assert all(("trainer" in line) or ("check_run" in line) for line in commands)
    assert "There is no `eval` entry point" in section


def test_every_companion_document_the_design_defers_to_exists(design):
    """This record's opening claim is that several sibling documents are *not* restated here, so
    every one of them is load-bearing: a moved file turns the deferral into a dead end, and the
    reader who followed it has no way to tell a missing document from an unwritten one.

    A sibling under ``teb_vae/`` is cited by its package-relative path, as the whole family's records
    cite each other, so both roots are tried.
    """
    referenced = sorted({match for match in re.findall(r"[\w/]+\.md", design) if "/" in match})

    assert len(referenced) >= 4, f"the design record defers to only {referenced}"
    missing = [
        path
        for path in referenced
        if not (_REPO_ROOT / path).is_file() and not (_REPO_ROOT / "teb_vae" / path).is_file()
    ]
    assert missing == [], f"DESIGN.md defers to documents that do not exist: {missing}"


def test_every_launch_line_in_the_design_names_a_config_that_exists(design):
    """A launch line is copied and pasted; one naming a moved file fails at the shell with a
    message about a path rather than about a run."""
    referenced = sorted(set(re.findall(r"teb_vae/lag_attn_cfs/configs/[\w.]+\.yaml", design)))

    assert len(referenced) == 3, f"DESIGN.md §16 names {referenced}, not all three launch configs"
    missing = [path for path in referenced if not (_REPO_ROOT / path).is_file()]
    assert missing == [], missing


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
@pytest.mark.parametrize("document", ("DESIGN.md", "RESULTS.md"))
@pytest.mark.parametrize("token", FORBIDDEN_TOKENS)
def test_neither_record_mentions_the_planning_document(token, document):
    """A design record describes the model and its invariants and a results record describes runs
    and their readings; neither describes the artefact that scheduled them, which does not survive
    the module and would leave dangling references behind it."""
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
