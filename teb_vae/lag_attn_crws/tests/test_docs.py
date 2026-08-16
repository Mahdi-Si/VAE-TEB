r"""The two records describe the package that exists, not the one they described.

This package has no evaluation pipeline, so a run produces exactly two durable artefacts: its own
``train_results/metrics_history.csv`` and what someone transcribes from it. There is no verdict file,
no bootstrap interval and no per-recording table to catch a mistake in the copy, which makes the
documents the record -- and a record with no test is a record that goes stale in the direction nobody
notices.

``DESIGN.md`` is pinned on four things a reader would otherwise take on trust. **The parameter
arithmetic**: eight totals -- both cells of this row and both raw-signal cells they are read against,
each guarded and ungated -- checked against ``sum(p.numel() ...)`` on constructed models rather than
against literals here, so a legitimate change to a shared imported component re-costs the document
instead of failing an unrelated assertion; and the *stated decomposition* is evaluated parameter name
by parameter name rather than merely found, because a section carrying the right delta beside a wrong
decomposition of it is exactly the half a search for the number cannot see. **The linearisation**,
compared against the real ``__mro__``: the base order decides whether the tiled forward or the dense
one wins, and a document recording the old order would be the only place a reader could go to find
out. **The binding record**: almost every member of this package is another package's object reached
by reference, and the record names each one, its owner and the test that pins it -- so every member
it names is asserted to be that owner's object in code and every test file it names is asserted to
exist. And **the claims a reader could take too far**, each asserted against the code it describes
rather than against itself -- the width hook against the class dictionaries, the metric surface
against the driver's own tuple, the dropped readouts against the same tuple, the channel counts
against the budget the committed fixture resolves to.

``RESULTS.md`` is a pre-registration, and the rest of this file binds it to the code.

The document is written **before** the headline run, so that the criteria a run is judged against
cannot be chosen once the numbers are in view. That only means something if the document cannot
quietly drift from what the code emits, which is what the tests below check:

* every section the study needs is present, as a heading, so a run cannot arrive and find the
  document has stopped asking for something -- and the one heading the family carries that this
  record deliberately omits is asserted **absent**, with the omission stated;
* every metric name the document promises to fill in is one the task genuinely emits, and every
  readout this package added is named -- both directions, because a name the framework never emits
  is a cell nobody can fill, and an emitted readout nobody registered is one nobody will read;
* the revert record names files that exist, and names exactly the seven this package's arrival edited;
* the three rules that are easiest to lose in a rewrite -- that there is no evaluation package, that
  the nats are comparable to no cell outside this row, and that the sign of the gap is not a
  criterion -- are stated;
* and the document does not mention the planning artefact this package was built from. That scan
  covers every module and test in the package, not only the record: a design that documented itself
  by pointing at a plan would leave the code unexplained the day the plan is archived.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set, Tuple

import pytest
import torch

from teb_vae.lag_attn_cfs.model_kwargs import WARMUP_MODEL_KWARGS, warmup_model_kwargs
from teb_vae.lag_attn_cfs.nets.causal_feature_target import CausalFeatureForecastTarget
from teb_vae.lag_attn_cfs.nets.causal_inputs import CausalWarmupInputs
from teb_vae.lag_attn_cfs.task import SeqVaeLagAttnCfsTask
from teb_vae.lag_attn_crws.nets.causal_raw_inputs import CausalRawInputs
from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws
from teb_vae.lag_attn_crws.task import SeqVaeLagAttnCrwsTask
from teb_vae.lag_attn_crws.trainer import LagAttnCrwsTrainer
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
from teb_vae.lag_attn_rws.trainer import LagAttnRwsTrainer

from .conftest import make_streams, shipped_warmup_kwargs, tiny_warmup_kwargs

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_DIR.parents[1]
_DESIGN = _PACKAGE_DIR / "DESIGN.md"
_RESULTS = _PACKAGE_DIR / "RESULTS.md"

#: Any integer of seven digits or more, in either notation the records use: plain markdown
#: (``5,101,114``) or LaTeX with braced separators (``5{,}101{,}114``). Both appear, because a number
#: inside a maths span must brace its separators to keep the spacing, and pinning one notation would
#: leave half of the arithmetic unchecked.
_LARGE_NUMBER_PATTERN = re.compile(r"(\d{1,3}(?:(?:\{,\}|,)\d{3})+)")

#: A factorised product as the records write it -- ``128 \times (98 - 78)``. Captured so the stated
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
    "## Headline baseline",
    "## Bottleneck health",
    "## Forecasting or reconstructing?",
    "## The warm-up and the tiling",
)

#: The one heading the family's records carry that this one deliberately does not. No production run
#: is in scope, so every cell of that table would be unreachable, and a heading with no reachable
#: number is worse than no heading. The omission is asserted, and so is the sentence stating it.
OMITTED_HEADING = "## Distributed smoke, memory and throughput"

#: The four readouts this package adds: three per stage and one on the evaluation stages alone.
CAUSAL_RAW_METRICS = (
    "anchors_per_sample",
    "source_lag_warmth_frac_st",
    "source_lag_warmth_frac_ph",
    "kld_source_null",
)

#: The eight readouts the causal-feature cell carries that this cell **drops** rather than re-points:
#: the feature target's four gap splits and the causal-feature cell's four warm-up columns. Each
#: resolves the gap over a channel axis, a stored block or the target's own warm-up, and a raw target
#: has none of the three. Named so the records can say so by name and the scan below can admit them
#: as *stated absences* rather than as typos.
DROPPED_READOUTS = (
    "pred_gap_tau_first",
    "pred_gap_tau_last",
    "pred_gap_st",
    "pred_gap_ph",
    "pred_gap_warm_lo",
    "pred_gap_warm_mid",
    "pred_gap_warm_hi",
    "target_warm_frac",
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

#: The two forward-dict keys the records name in backticks. They share the shape of a metric name and
#: none of the namespace -- ``anchor_index`` is a tensor the forward returns, not a column a step logs
#: -- so the scan below admits exactly these two rather than exempting the shape.
FORWARD_DICT_KEYS = ("anchor_index", "anchor_valid")

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

#: The seven files outside this package that its arrival edited -- one kind of file, two strings
#: each. The revert record must name exactly these, because a revert that named more would touch a
#: file this package never did and one that named fewer would leave a registration behind.
REVERT_FILES = tuple(
    f"teb_vae/{package}/tests/test_nets_are_framework_free.py"
    for package in (
        "lag_attn_rws",
        "lag_attn_transformer_rws",
        "lag_attn_transformer_e2e",
        "lag_attn_fs",
        "lag_attn_transformer_fs",
        "lag_attn_cfs",
        "lag_attn_transformer_cfs",
    )
)

#: Every member the design record's binding table names, its owner, and the attribute it is bound
#: under here. Asserted in code -- ``is`` the owner's object -- and asserted named in the record, so
#: the table cannot describe a binding that no longer exists or omit one that does.
BOUND_MEMBERS: Tuple[Tuple[str, type, type], ...] = (
    ("SOURCE_BLOCK_SPLIT", CausalFeatureForecastTarget, CausalRawInputs),
    ("TARGET_BLOCK_SPLIT", CausalFeatureForecastTarget, CausalRawInputs),
    ("_resolve_block_warm_steps", CausalFeatureForecastTarget, CausalRawInputs),
    ("_anchors_per_sample", CausalFeatureForecastTarget, CausalRawInputs),
    ("_source_lag_warmth", CausalFeatureForecastTarget, CausalRawInputs),
    ("anchor_phase", SeqVaeLagAttnCfsTask, SeqVaeLagAttnCrwsTask),
    ("_phase_field", SeqVaeLagAttnCfsTask, SeqVaeLagAttnCrwsTask),
    ("resolve_anchor_geometry", SeqVaeLagAttnCfsTask, SeqVaeLagAttnCrwsTask),
    ("_build_forward_inputs", SeqVaeLagAttnCfsTask, SeqVaeLagAttnCrwsTask),
    ("_mu_gap_rms", SeqVaeLagAttnCfsTask, SeqVaeLagAttnCrwsTask),
    ("_added_metrics", SeqVaeLagAttnCfsTask, SeqVaeLagAttnCrwsTask),
    ("input_stream_panels", SeqVaeLagAttnCfsTask, SeqVaeLagAttnCrwsTask),
    ("input_budget_figure", SeqVaeLagAttnCfsTask, SeqVaeLagAttnCrwsTask),
)

#: Every suffix the tracked metric surface carries, so a documented name can be checked against what
#: a run genuinely logs rather than against a second hand-kept list.
_TRACKED_SUFFIXES = frozenset(
    name.split("/")[-1] for name in LagAttnCrwsTrainer.TRACKED_METRICS
)


@pytest.fixture(scope="module")
def results() -> str:
    return _RESULTS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def design() -> str:
    return _DESIGN.read_text(encoding="utf-8")


def _reach_gated(shipped: Mapping[str, Any]) -> Dict[str, Any]:
    """The raw-signal cells' production constructor call: the shipped widths plus the reach budget.

    Built the way those cells' drivers build it -- the shipped 120 s budget resolved against the
    production Morlet bank -- rather than written out, so the surviving-channel counts come from the
    filter bank every time. Local rather than imported, because the raw-signal suites keep this
    assembly inside their trainer tests rather than in a conftest helper.

    Args:
        shipped: One raw-signal suite's ``SHIPPED_KWARGS``.

    Returns:
        Constructor kwargs carrying the four resolved reach tuples.
    """
    from teb_vae.lag_attn.channel_reach import resolve_stream_budgets

    budget = resolve_stream_budgets(
        {
            "causal_reach_budget_s": 120.0,
            "use_up_st": shipped["use_up_st"],
            "warmup_period": shipped["warmup_period"],
            "c_y": shipped["c_y"],
            "c_u": shipped["c_u"],
        }
    )
    return dict(
        shipped,
        target_keep_index=budget.target_keep_index,
        target_delays=budget.target_delays,
        source_keep_index=budget.source_keep_index,
        source_delays=budget.source_delays,
    )


def _ungated(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """The four resolved warm-up tuples removed and nothing else, so the guarded and ungated arms
    differ in the guard alone rather than in a second hand-written keyword set."""
    return {key: value for key, value in kwargs.items() if key not in WARMUP_MODEL_KWARGS}


@pytest.fixture(scope="module")
def measured_models() -> Dict[str, torch.nn.Module]:
    """The eight models the two records quote, constructed in one process.

    Built from each suite's own production keyword set rather than from the configs, so this file
    binds the documents to the *architectures* while ``test_config_load.py`` binds the configs to
    the driver -- two independent routes to the same widths. The four comparison models are built
    from different keyword sets by necessity: their constructors' schemas differ by the five
    conv-LSTM-only keywords and the seven encoder ones, and the raw-signal cells take delays where
    this row takes warm-ups.
    """
    from teb_vae.lag_attn_rws.tests.conftest import SHIPPED_KWARGS as RWS_SHIPPED
    from teb_vae.lag_attn_transformer_crws.nets.model import SeqVaeLagAttnTrfCrws
    from teb_vae.lag_attn_transformer_crws.tests.conftest import (
        shipped_warmup_kwargs as trf_warmup_kwargs,
    )
    from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws
    from teb_vae.lag_attn_transformer_rws.tests.conftest import (
        SHIPPED_KWARGS as TRF_RWS_SHIPPED,
    )

    return {
        "crws_guarded": SeqVaeLagAttnCrws(**shipped_warmup_kwargs()),
        "crws_ungated": SeqVaeLagAttnCrws(**_ungated(shipped_warmup_kwargs())),
        "trf_crws_guarded": SeqVaeLagAttnTrfCrws(**trf_warmup_kwargs()),
        "trf_crws_ungated": SeqVaeLagAttnTrfCrws(**_ungated(trf_warmup_kwargs())),
        "rws_guarded": SeqVaeLagAttnRws(**_reach_gated(RWS_SHIPPED)),
        "rws_ungated": SeqVaeLagAttnRws(**dict(RWS_SHIPPED)),
        "trf_rws_guarded": SeqVaeLagAttnTrfRws(**_reach_gated(TRF_RWS_SHIPPED)),
        "trf_rws_ungated": SeqVaeLagAttnTrfRws(**dict(TRF_RWS_SHIPPED)),
    }


@pytest.fixture(scope="module")
def measured_totals(measured_models) -> Dict[str, int]:
    """Parameter totals of the eight models, measured."""
    return {
        label: sum(parameter.numel() for parameter in model.parameters())
        for label, model in measured_models.items()
    }


def _named_counts(model: torch.nn.Module) -> Dict[str, int]:
    """Parameter count per parameter name, so a delta can be attributed rather than merely summed."""
    return {name: parameter.numel() for name, parameter in model.named_parameters()}


def _differing_names(left: torch.nn.Module, right: torch.nn.Module) -> Dict[str, int]:
    """Every parameter name whose count differs between two models, with the signed difference."""
    a, b = _named_counts(left), _named_counts(right)
    return {
        name: a.get(name, 0) - b.get(name, 0)
        for name in sorted(set(a) | set(b))
        if a.get(name) != b.get(name)
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

    The LaTeX form goes **first**: ``5{,}101{,}114`` under a plain comma strip would become
    ``5{}101{}114``, and every subsequent search for a number inside a maths span would silently
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
    """Eight totals -- both cells of this row and both raw-signal cells they are read against, each
    guarded and ungated -- checked against ``sum(p.numel() ...)`` rather than against literals in a
    test, so a legitimate change to a shared imported component re-costs the document."""
    section = _markdown_section(design, "## 13. ")
    stated = _integers_stated_in(section)

    for label, total in measured_totals.items():
        assert total in stated, (
            f"DESIGN.md §13 does not state the measured {label} total {total:,}"
        )


def test_the_headline_paragraph_carries_the_measured_total_and_both_comparisons(
    design, measured_totals
):
    """§1 is where a reader meets the number, and where a stale one would be read first."""
    section = _integers_stated_in(_markdown_section(design, "## 1. "))

    assert measured_totals["crws_guarded"] in section
    assert measured_totals["rws_guarded"] in section  # the input-representation comparison
    assert measured_totals["trf_crws_guarded"] in section  # the encoder-axis one


def test_the_input_representation_delta_decomposes_into_the_two_terms_the_design_states(
    design, measured_totals, measured_models
):
    r"""The horizon embedding and the two input adapters -- and **not** the decoder head, which is
    ``raw_per_step`` wide in both cells. Evaluated from the document's own factorisation rather than
    merely searched for, and then attributed parameter name by parameter name: a section carrying the
    right total beside a wrong decomposition of it is the half a reader takes on trust."""
    section = _markdown_section(design, "## 13. ")
    delta = measured_totals["crws_guarded"] - measured_totals["rws_guarded"]

    horizon_embedding = -15 * 256
    adapters = 128 * (98 - 78) * 2 + 128 * (51 - 29) * 2 - 256

    assert delta == horizon_embedding + adapters, (
        f"the input-representation delta is {delta:+,}, which no longer decomposes into the "
        f"horizon embedding ({horizon_embedding:+,}) and the two adapters ({adapters:+,})"
    )
    # The same delta in the conv-Transformer pair, because every module outside the encoders is
    # shared -- which is the claim §13 makes and the reason both pairs are quoted.
    assert delta == measured_totals["trf_crws_guarded"] - measured_totals["trf_rws_guarded"]

    # Attributed by name: every parameter whose count differs is under one of the two terms, and
    # the decoder head is on neither side of the difference.
    differing = _differing_names(measured_models["crws_guarded"], measured_models["rws_guarded"])
    assert differing, "the two guarded models have identical parameter tables"
    for name in differing:
        assert name.startswith(("horizon_core.horizon_embedding", "target_adapter.", "source_adapter.")), (
            f"{name} differs between the guarded models and belongs to neither stated term"
        )
    assert differing["horizon_core.horizon_embedding"] == horizon_embedding
    assert sum(value for name, value in differing.items() if "adapter" in name) == adapters
    assert (
        measured_models["crws_guarded"].decoder.mean_head.out_features
        == measured_models["rws_guarded"].decoder.mean_head.out_features
        == 16
    )

    # The stated factorisations of the adapter term, evaluated.
    factorisations = {
        int(cost) * (int(wide) - int(narrow))
        for cost, wide, narrow in _FACTORISATION_PATTERN.findall(section)
    }
    assert {128 * (98 - 78), 128 * (51 - 29)} <= factorisations, (
        f"§13 does not factorise both adapter widths; it states {factorisations}"
    )
    for value in (delta, adapters, -horizon_embedding):
        assert str(value) in _unseparated(section), value


def test_the_ungated_delta_decomposes_into_two_terms_as_well(design, measured_totals):
    """Ungated against ungated the axis is the same horizon-embedding term plus the two input
    linears at the narrower stored widths, and §13 states it -- so a reader comparing the ungated
    arms is not left to derive it."""
    section = _markdown_section(design, "## 13. ")
    delta = measured_totals["crws_ungated"] - measured_totals["rws_ungated"]

    assert delta == -15 * 256 + 128 * (102 - 109) + 128 * (51 - 58)
    assert delta == measured_totals["trf_crws_ungated"] - measured_totals["trf_rws_ungated"]
    assert str(abs(delta)) in _unseparated(section)


def test_the_encoder_axis_delta_is_the_two_history_encoders(design, measured_totals):
    """Checked as arithmetic rather than as prose: the reduction must be the same number the
    raw-signal pair sees at its own budget, and the same on the ungated arm, since everything
    downstream of the encoders is a shared module in both pairs -- which is the claim §13 makes and
    the reason it is quoted."""
    causal_raw = measured_totals["crws_guarded"] - measured_totals["trf_crws_guarded"]
    raw = measured_totals["rws_guarded"] - measured_totals["trf_rws_guarded"]
    ungated = measured_totals["crws_ungated"] - measured_totals["trf_crws_ungated"]

    assert causal_raw == raw == ungated, (
        "the encoder swap no longer costs the same at every target and every guard, so the "
        "reduction is not the two history encoders alone"
    )
    assert causal_raw in _integers_stated_in(_markdown_section(design, "## 13. "))


def test_the_guard_delta_is_the_two_availability_projections_alone(
    design, measured_totals, measured_models
):
    """Unlike the feature cells nothing in this target domain widens a head, so every parameter the
    budget adds is under an adapter -- asserted by name -- and the arithmetic §13 states for it is
    evaluated. The raw-signal sibling's smaller guard cost is stated beside it, so two correct numbers
    a factor of three apart are not read as a contradiction."""
    section = _markdown_section(design, "## 13. ")
    guard = measured_totals["crws_guarded"] - measured_totals["crws_ungated"]
    sibling = measured_totals["rws_guarded"] - measured_totals["rws_ungated"]

    assert guard == 128 * 98 + 128 * 51 - 128 * 4
    assert guard == measured_totals["trf_crws_guarded"] - measured_totals["trf_crws_ungated"]
    for name in _differing_names(measured_models["crws_guarded"], measured_models["crws_ungated"]):
        assert "adapter" in name, name
    for value in (guard, sibling):
        assert str(value) in _unseparated(section), value


def test_the_records_agree_on_the_parameter_table(design, results, measured_totals):
    """The same totals appear in both documents, so a run's record and the design record cannot
    quote different budgets for one model."""
    in_design = _integers_stated_in(_markdown_section(design, "## 13. "))
    in_results = _integers_stated_in(_markdown_section(results, "## Parameter budget"))

    for label in ("crws_guarded", "crws_ungated", "trf_crws_guarded", "rws_guarded", "rws_ungated"):
        assert measured_totals[label] in in_design and measured_totals[label] in in_results, label


# =================================================================================================
# DESIGN.md: the linearisation, the binding record, and the claims a reader could take too far
# =================================================================================================
def test_the_documented_linearisation_is_the_real_one(design):
    """Compared against ``__mro__`` rather than against prose. The base order decides whether the
    tiled forward or the dense one wins, and the document would be the only place a reader could go
    to find out that it had moved."""
    stated = _linearisation_stated_for(design, SeqVaeLagAttnCrws)
    real = [base.__name__ for base in SeqVaeLagAttnCrws.__mro__]

    assert stated == real[: len(stated)], (
        f"DESIGN.md states {stated}; the real MRO begins {real[: len(stated)]}"
    )


def test_the_model_class_really_defines_only_a_constructor(design):
    """The claim §1 and §6 both rest on: with nothing else defined here the forward keys, the
    posterior structure, the lag map and the objective's metric set cannot have moved, because they
    are the mixin's and the base's own code objects."""
    own = set(vars(SeqVaeLagAttnCrws)) - {"__doc__", "__module__", "__dict__", "__weakref__"}

    assert own == {"__init__"}, f"SeqVaeLagAttnCrws defines {sorted(own)}"
    assert "a constructor and nothing else" in _flat(_markdown_section(design, "## 1. "))


def test_the_two_replaced_delay_keywords_are_gone_and_the_four_new_ones_are_there(design):
    """A warm-up is a leading *mask* and ``ChannelDelay`` is a *shift*, so a warm-up routed under a
    delay name would train a different model with every shape intact. Asserted against the
    constructor rather than against the sentence that says so."""
    parameters = set(inspect.signature(SeqVaeLagAttnCrws.__init__).parameters)
    section = _flat(_markdown_section(design, "## 6. "))

    assert {"target_delays", "source_delays"} & parameters == set()
    for name in ("target_warmup_steps", "source_warmup_steps", "anchor_stride", "lag_floor"):
        assert name in parameters, name
        assert f"`{name}`" in design, name
    assert "order of the bases is load-bearing" in section


def test_the_width_hook_is_the_load_bearing_absence(design, tiny_warmup):
    """The document's central structural claim is about a member that is *not* there. Asserted
    against the class dictionaries and against the built decoder, so the prose cannot outlive the
    property -- and the one limit of it, the still-settable width keyword, is asserted to be stated
    beside it rather than left for an operator to discover at the first batch."""
    section = _flat(_markdown_section(design, "## 6. "))

    assert "_default_decoder_out_channels" not in vars(CausalRawInputs)
    assert "_default_decoder_out_channels" not in vars(SeqVaeLagAttnCrws)
    assert (
        SeqVaeLagAttnCrws._default_decoder_out_channels
        is SeqVaeLagAttnRws._default_decoder_out_channels
    )
    model = SeqVaeLagAttnCrws(**tiny_warmup)
    assert model.decoder.mean_head.out_features == model.geometry.r == 16
    assert "load-bearing absence" in section
    assert "`decoder_out_channels`" in section
    assert "decoder_out_channels" in inspect.signature(SeqVaeLagAttnCrws.__init__).parameters


def test_the_mixin_overrides_exactly_the_two_target_coupled_members(design):
    """§6 says five of the input half's seven members are inherited untouched and two are overridden;
    asserted against ``vars`` of both classes rather than counted in prose."""
    inherited = (
        "_set_causal_inputs",
        "_build_adapter",
        "build_lag_mask",
        "_build_anchor_index",
        "forward",
        "_validate_causal_geometry",
    )
    for name in inherited:
        assert name not in vars(CausalRawInputs), name
        assert getattr(CausalRawInputs, name) is getattr(CausalWarmupInputs, name), name
    for name in ("_check_anchor_floor", "_resolve_warmup_readout_constants"):
        assert name in vars(CausalRawInputs), name
        assert f"`{name}`" in design, name


def test_the_binding_record_names_every_bound_member_and_its_owner(design):
    """Every member the record's binding table names is asserted to be its owner's object in code,
    and every bound member in code is asserted to be named in the table -- both directions, because
    a table describing a binding that no longer exists and a binding the table omits are the two
    ways a reader loses track of whose code a behaviour is."""
    section = _flat(_markdown_section(design, "## 6. "))
    assert "The binding record" in section

    for name, owner, consumer in BOUND_MEMBERS:
        bound = inspect.getattr_static(consumer, name)
        original = inspect.getattr_static(owner, name)
        # A staticmethod is re-wrapped, so compare the underlying functions; everything else is
        # the same object.
        if isinstance(original, staticmethod):
            assert isinstance(bound, staticmethod), f"{name} was bound without staticmethod()"
            assert bound.__func__ is original.__func__, name
        else:
            assert bound is original, name
        assert f"`{name}`" in section, f"the binding record does not name {name}"
        assert f"`{owner.__name__}`" in section, owner.__name__

    # The two module-level bindings the driver and the conftest reach.
    for name in ("`WARMUP_MODEL_KWARGS`", "`warmup_model_kwargs`", "`resolve_warmup_budget`"):
        assert name in section, name
    from teb_vae.lag_attn_crws.tests import conftest

    assert conftest.WARMUP_MODEL_KWARGS is WARMUP_MODEL_KWARGS
    assert conftest.warmup_model_kwargs is warmup_model_kwargs


def test_every_test_file_the_binding_record_names_exists(design):
    """The third column of the table names the test that pins each binding. A row naming a file that
    is gone would send a reader to a pin that no longer holds anything."""
    section = _markdown_section(design, "## 6. ")
    named = sorted(set(re.findall(r"`(tests/test_\w+\.py)`", section)))

    assert len(named) >= 6, named
    missing = [path for path in named if not (_PACKAGE_DIR / path).is_file()]
    assert missing == [], f"the binding record names tests that do not exist: {missing}"


def test_the_two_members_that_cannot_be_bound_are_named_with_the_reason(design):
    """``__init__`` and ``compute_loss_and_metrics`` call zero-argument ``super()``, which closes over
    the class that defines it. Asserted that both are the task's own and that the record says why."""
    section = _flat(_markdown_section(design, "## 6. "))

    for name in ("__init__", "compute_loss_and_metrics"):
        assert name in vars(SeqVaeLagAttnCrwsTask), name
        assert (
            vars(SeqVaeLagAttnCrwsTask)[name] is not vars(SeqVaeLagAttnCfsTask).get(name)
        ), name
    assert "zero-argument `super()`" in section
    assert "raises `TypeError` on the first step of the first run" in section


def test_the_geometry_section_states_the_policy_and_the_measured_channel_counts(design, budget):
    """Against the budget the committed fixture resolves to, not against constants: a fixture
    rebuilt at another quantile changes both the warm-up vectors and the stored channel count, and
    the document would be describing a boundary the data no longer has. And the policy's two exact
    halves -- *target-stream* and *by the first forecast step* -- because the obvious paraphrase of
    each is false and an operator acting on it would move the floor."""
    section = _flat(_markdown_section(design, "## 3. "))

    assert f"{budget.target.kept_width} of {budget.target.declared_width}" in section
    assert f"$B = {budget.target.max_warmup}$" in section
    assert "F \\ge B - 1" in section
    for name, kept, declared in budget.target.block_counts():
        assert f"`{name}` ${kept}/{declared}$" in section, name
    assert budget.source.kept_width == budget.source.declared_width == 51
    assert f"All ${budget.source.declared_width}$ source channels are kept" in section
    assert "not a constraint" in section
    assert "every kept **target-stream** input channel is warm **by the first forecast step**" in section
    # The two costs, both stated as numbers: what the policy costs and what the horizon lever buys.
    assert "$255$ anchors against $152$" in section
    assert "$137$ anchors against $152$" in section


def test_the_source_section_states_that_the_source_is_never_gated(design, budget):
    """The compromise this design makes, and the reason the two warmth columns exist. Asserted
    against the resolved budget: the source keep-index is the identity."""
    section = _flat(_markdown_section(design, "## 8. "))

    assert budget.source.kept_width == budget.source.declared_width
    assert f"all {budget.source.declared_width} source channels are kept" in section
    assert "small value there is the expected finding, not a failure" in section
    assert "the coupling readout is measuring a clock" in section


def test_the_design_states_what_the_nats_are_comparable_to(design):
    """Three halves: comparable to the twin cell of this row and to nothing else, *not* to the direct
    control as a level, and -- unlike the feature cells -- comparable across warm-up budgets, since
    a raw decoder is R wide at every budget."""
    section = _flat(_markdown_section(design, "## 5. "))

    assert "Comparable to `lag_attn_transformer_crws` and to nothing else" in section
    assert "not comparable to `lag_attn_rws`'s as a level either" in section
    assert "Comparable across warm-up budgets within this model" in section
    assert "mutually unloadable checkpoints" in section
    assert "240" in section and "480" in section


def test_the_design_states_that_this_is_an_experiment_rather_than_a_remedy(design):
    """The one framing error that would make a correct negative result read as a failure."""
    section = _flat(_markdown_section(design, "## 1. "))

    assert "experiment, not a remedy" in section
    assert "expected to reproduce" in section
    assert "sign of `pred_gap` is a criterion nowhere" in section


def test_the_anchored_gather_section_states_gather_not_index_select(design):
    """The one genuinely new piece of arithmetic, and the one distinction that decides whether it is
    right: a per-sample anchor set needs a ``gather``. Asserted that the record says so and that the
    function's own source is a ``gather``."""
    from teb_vae.lag_attn_crws.nets.causal_raw_inputs import gather_anchored_future_target

    section = _flat(_markdown_section(design, "## 4. "))
    source = inspect.getsource(gather_anchored_future_target)

    assert "`gather`, not `index_select`" in section
    assert ".gather(1," in source and "index_select" not in source.split('"""')[-1]
    assert "`gather_anchored_future_target`" in section
    assert "equals `build_future_target` under `torch.equal`" in section


def test_the_documented_metric_count_is_the_drivers_own(design):
    """A hand-kept number in prose beside a computed tuple is the pair most likely to drift."""
    section = _markdown_section(design, "## 10. ")

    assert f"**{len(LagAttnCrwsTrainer.TRACKED_METRICS)}**" in section
    assert len(LagAttnCrwsTrainer.TRACKED_METRICS) == len(LagAttnRwsTrainer.TRACKED_METRICS) + 7
    # And the one deliberate asymmetry in the surface, asserted in both directions.
    assert "train/kld_source_null" not in LagAttnCrwsTrainer.TRACKED_METRICS
    assert "val/kld_source_null" in LagAttnCrwsTrainer.TRACKED_METRICS


@pytest.mark.parametrize("name", DROPPED_READOUTS)
def test_every_dropped_readout_is_named_as_dropped_and_really_absent(name, design, results):
    """The five columns a reader of the causal-feature records would expect and will not find. Named
    in both records so their absence reads as a decision, and asserted absent from the tracked
    surface so the records cannot call dropped a column a run still emits."""
    assert f"`{name}`" in _markdown_section(design, "## 10. "), name
    assert name in results, name
    assert name not in _TRACKED_SUFFIXES, name


def test_the_design_records_why_the_latent_gap_readout_is_re_pointed(design):
    """The inherited readout whose own docstring promises an invariant that tiling breaks. It is
    overridden on the task, and the document is where a reader learns the column changed meaning."""
    section = _flat(_markdown_section(design, "## 10. "))

    assert "_mu_gap_rms" in section
    assert "mu_post_prior_gap_rms" in section
    assert "restores the property the function already claims" in section


def test_the_forward_return_section_states_the_key_count_and_the_shapes(design, tiny_warmup):
    """Twenty-two keys and the raw block's shape, checked against a forward rather than restated."""
    section = _flat(_markdown_section(design, "## 9. "))
    model = SeqVaeLagAttnCrws(**tiny_warmup).eval()
    with torch.no_grad():
        outputs = model(*make_streams(tiny_warmup))

    assert len(outputs) == 22
    assert "twenty-two keys" in section
    assert outputs["anchor_index"].dtype is torch.long
    assert outputs["anchor_valid"].dtype is torch.bool
    assert outputs["mu_full"].shape[-1] == 16
    assert "$(B, 11, 15, 16)$" in section
    assert "No `decoder_state` and no `delta_mu_src`" in section
    assert "decoder_state" not in outputs and "delta_mu_src" not in outputs


def test_the_page_section_states_the_row_count_and_the_one_sided_caveat(design):
    """Nine titled rows -- the sibling's seven and the two imported input rows -- and the caveat that
    is this cell's own string, asserted against the module that carries it."""
    from teb_vae.lag_attn_crws import sample_page

    section = _flat(_markdown_section(design, "## 11. "))

    assert "Nine titled rows" in section
    assert "one-sided" in section
    assert "one-sided here" in sample_page.LAG_TIME_CAVEAT
    assert "`causal_warmup_budget`" in section
    assert "`forecast_extra_rows`" in section
    assert not hasattr(SeqVaeLagAttnCrwsTask, "forecast_extra_rows")


def test_the_lean_limits_carry_their_replacement_triggers(design):
    """A ``lean-limit`` note without a measurable trigger is a permanent excuse. Exactly three here:
    the anchor floor that is a policy, the members written out rather than shared, and the absent
    evaluation package."""
    flat = _flat(design)

    assert len(re.findall(r"^> lean-limit: ", design, re.MULTILINE)) == 3
    assert "when a run shows the anchor count rather than the source pathway" in flat
    assert "when a third consumer of any of them appears" in flat
    assert "when a result from these cells is to be reported as a measurement" in flat


def test_the_design_states_that_there_is_no_evaluation_package(design):
    """Stated as a limitation with its follow-up named, and asserted against the directory -- so a
    pipeline arriving later has to change this record rather than silently outdate it."""
    section = _flat(_markdown_section(design, "## 14. "))

    assert "no `eval/` package" in section
    assert "`ModelBinding`" in section
    assert "`eval/metrics.py::model_inputs`" in section
    assert "No run checker ships" in section
    assert not (_PACKAGE_DIR / "eval").exists()
    assert not (_PACKAGE_DIR / "check_run.py").exists()


def test_every_companion_document_the_design_defers_to_exists(design):
    """This record's opening claim is that several sibling documents are *not* restated here, so
    every one of them is load-bearing: a moved file turns the deferral into a dead end, and the
    reader who followed it has no way to tell a missing document from an unwritten one.

    A sibling under ``teb_vae/`` is cited by its package-relative path, as the whole family's records
    cite each other; a document inside *this* package is cited relative to the package. All three
    roots are tried.
    """
    referenced = sorted({match for match in re.findall(r"[\w/]+\.md", design) if "/" in match})

    assert len(referenced) >= 5, f"the design record defers to only {referenced}"
    roots = (_REPO_ROOT, _REPO_ROOT / "teb_vae", _PACKAGE_DIR)
    missing = [
        path for path in referenced if not any((root / path).is_file() for root in roots)
    ]
    assert missing == [], f"DESIGN.md defers to documents that do not exist: {missing}"


def test_every_launch_line_in_the_design_names_a_config_that_exists(design):
    """A launch line is copied and pasted; one naming a moved file fails at the shell with a
    message about a path rather than about a run."""
    referenced = sorted(set(re.findall(r"teb_vae/lag_attn_crws/configs/[\w.]+\.yaml", design)))
    shipped = sorted(path.name for path in (_PACKAGE_DIR / "configs").glob("*.yaml"))

    assert [Path(path).name for path in referenced] == shipped, (
        f"DESIGN.md §16 names {referenced}, not the shipped configs {shipped}"
    )
    missing = [path for path in referenced if not (_REPO_ROOT / path).is_file()]
    assert missing == [], missing


def test_the_documented_config_inventory_is_the_real_one(design):
    """§12 names the directory's contents, and a config added without a word there is an arm nobody
    reading the record knows exists."""
    section = _flat(_markdown_section(design, "## 12. "))
    shipped = sorted(path.name for path in (_PACKAGE_DIR / "configs").glob("*.yaml"))

    for name in shipped:
        assert f"`{name}`" in section, name
    named = set(re.findall(r"`([\w.]+\.yaml)`", section))
    assert named <= set(shipped) | {"lag_attn_rws/configs/default.yaml"}, named - set(shipped)
    assert "twenty" in section  # the exemption count, matched against test_config_load's list
    from teb_vae.lag_attn_crws.tests.test_config_load import PARITY_EXEMPT_PATHS

    assert len(PARITY_EXEMPT_PATHS) == 20


# =================================================================================================
# The document exists and asks for everything
# =================================================================================================
def test_the_record_exists_and_is_not_a_stub(results):
    assert len(results) > 4000, "RESULTS.md is too short to be a pre-registration"


@pytest.mark.parametrize("heading", STUDY_HEADINGS)
def test_every_required_section_is_present(heading, results):
    assert heading in results, heading


def test_the_distributed_smoke_heading_is_omitted_and_the_omission_is_stated(results):
    """The one heading the family carries that this record does not. A heading with no reachable
    number is worse than no heading -- and an omission nobody stated reads as an oversight."""
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


def test_the_three_tier_one_criteria_are_registered_and_the_two_absences_stated(results):
    """Named individually rather than counted, so a criterion cannot be dropped and replaced. Three
    where the causal-feature cells register five, and the record must say why the other two do not
    exist on a raw target rather than leave a reader to notice the gap."""
    tier_one = results.split("### Tier 1")[1].split("### Tier 2")[0]

    assert "`anchors_per_sample`" in tier_one
    assert "spike breaker never latches" in tier_one
    assert "identical metric row set" in tier_one
    assert "the two absences are deliberate" in _flat(tier_one)
    assert "target_warm_frac" not in tier_one


def test_the_five_tier_two_quantities_are_registered(results):
    tier_two = results.split("### Tier 2")[1].split("\n---")[0]

    for name in (
        "source_conditioned_kl_raw",
        "kld_active_frac",
        "logvar_prior_floor_frac",
        "kld_source_null",
        "shuffle_penalty",
        "pred_gap",
    ):
        assert name in tier_two, name


def test_the_record_names_the_input_representation_edge_and_says_it_is_not_a_level(results):
    """The reason this cell exists: the direct control shares the target and the objective. And the
    trap the halved block makes easy -- a level read across it -- is stated where the table is."""
    section = results.split("### The input-representation edge")[1].split("\n---")[0]

    assert "`lag_attn_rws`" in section
    assert "**not** comparable as a level" in section
    assert "sign and the trajectory only" in section
    assert "sweep_horizon_30.yaml" in section


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


@pytest.mark.parametrize("name", BOTTLENECK_HEALTH_METRICS)
def test_the_bottleneck_health_table_carries_every_readout(name, results):
    section = results.split("## Bottleneck health")[1].split("\n## ")[0]

    assert f"`{name}`" in section, name
    assert name in _TRACKED_SUFFIXES, name


def test_every_backticked_metric_name_in_the_record_is_one_the_run_emits(results):
    """The general form of the two checks above, over the whole document. Restricted to names that
    look like metric identifiers, so config keys, file names and prose survive it.

    Two namespaces share the shape and not the meaning and are admitted by name rather than by
    shape: the two forward-dict keys the anchor axis returns, and the five causal-feature readouts
    this record names as *dropped* -- which are asserted absent from the tracked surface elsewhere in
    this file, so admitting them here cannot hide a column that crept back in.
    """
    candidates = set(re.findall(r"`([a-z][a-z0-9_]{4,})`", results))
    metric_shaped = {
        name
        for name in candidates
        if name.startswith(("pred_gap", "nll_", "kld_", "logvar_", "mu_", "source_", "anchor"))
        or name in {"total_loss", "main_loss", "shuffle_penalty"}
    }

    # A *family* name is admitted -- ``source_lag_warmth_frac`` stands for its two per-block
    # columns, and prose that had to spell both out every time would be worse prose. What is not
    # admitted is a name no tracked column starts with, which is the typo this catches.
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
    """A list rather than an archaeology exercise, and an *exact* one: this package's arrival edited
    seven existing files and nothing else, so a row naming an eighth would revert a file this package
    never touched and a missing row would leave a registration behind. Every named file exists."""
    section = results.split("### What reverts, and how")[1].split("\n### ")[0]

    # The table's rows, not the prose around it: the prose names the objective-fingerprint script
    # as the check that nothing else moved, and that script is not a file this package edited.
    paths = sorted(
        set(re.findall(r"^\| `((?:teb_vae|hdf5_dataset|scripts)/[\w./]+\.py)`", section, re.M))
    )
    assert paths == sorted(REVERT_FILES), paths
    for path in paths:
        assert (_REPO_ROOT / path).is_file(), path
    # And the two strings each of them gained really are in each of them.
    for path in paths:
        source = (_REPO_ROOT / path).read_text(encoding="utf-8")
        assert '"lag_attn_crws"' in source and '"lag_attn_transformer_crws"' in source, path


def test_the_revert_record_defers_the_shared_seams_to_the_cell_that_landed_them(results):
    """The anchor seams in the shared tree were landed by the causal-feature cell and are *not* this
    package's to revert; a record that listed them would invite a revert that broke that cell."""
    section = _flat(results.split("### What reverts, and how")[1].split("\n### ")[0])

    assert "teb_vae/lag_attn_cfs/RESULTS.md" in section
    assert "not restated here" in section
    assert "bound by reference" in section


def test_the_record_states_that_there_is_no_evaluation_package(results):
    """Stated once, near the top, so no number on the page is read as though it had a confidence
    interval. The narrow form the causal-feature records now carry -- "in-sample and carries no
    uncertainty" beside a pipeline that exists -- would be wrong here, because there is none."""
    flat = _flat(results)

    assert "There is no evaluation package" in flat
    assert "in-sample and carries no uncertainty" in flat
    assert "metrics_history.csv" in results
    assert "no evaluation line and no run-checker line" in flat
    assert not (_PACKAGE_DIR / "eval").exists()


def test_the_record_states_that_the_nats_are_comparable_only_within_this_row(results):
    """Both halves: not to the direct control, because the block halved and the anchor count fell,
    and to the twin cell, because it ships the identical geometry."""
    flat = _flat(results)

    assert "240" in results and "480" in results
    assert "comparable only within this row" in flat
    assert "The one model whose loss level is comparable to this one's is `lag_attn_transformer_crws`" in flat


def test_the_record_states_that_the_sign_of_the_gap_is_not_a_criterion(results):
    """The finding this family is expected to reproduce, recorded before the run rather than
    explained after it."""
    assert "negative `pred_gap` is not a failure" in results


def test_the_record_states_that_the_lag_caveat_is_one_sided(results):
    """The reading the missing target-side group delay makes correct here and nowhere else in the
    family, and the one an operator would otherwise copy from the causal-feature record."""
    flat = _flat(results)

    assert "one-sided" in flat
    assert "no target-side term" in flat


def test_the_record_states_that_no_run_checker_ships_and_the_guard_is_read_by_hand(results):
    """The causal-feature cell scores its criteria by code because its runs are blocked; this one
    ships no checker because no run is in scope, and the record must say how the guard is read
    until one exists rather than imply a script that is not there."""
    flat = _flat(results)

    assert "No run checker" in flat
    assert "read by hand" in flat
    assert not (_PACKAGE_DIR / "check_run.py").exists()


def test_the_loss_scale_constants_section_states_the_measured_percentiles(results):
    """The two re-derived constants and the two that came back to parity, with the statistic each
    was set from and the bracket the margin sits in -- both numbers pinned in
    ``tests/test_spike_breaker.py`` and restated here so a reader of this record sees them."""
    from teb_vae.lag_attn_crws.tests import test_spike_breaker as breaker

    section = results.split("## The loss-scale constants")[1].split("\n## ")[0]

    assert "`gradient_clip_val`" in section and "1000.0" in section
    assert "`additive_margin`" in section and "5.0e+2" in section
    assert "`ema_floor`" in section and "`horizon_embed_std`" in section
    assert str(round(breaker.MEASURED_GRAD_Q99)) in section
    assert str(round(breaker.MEASURED_EXCURSION_MAX)) in section
    assert "provisional" in section


def test_every_launch_line_names_a_module_that_exists(results):
    """A launch line is copied and pasted; one naming a moved module fails at the shell with a
    message about an import rather than about a run."""
    section = results.split("## Launch lines")[1].split("\n## ")[0]

    modules = sorted(set(re.findall(r"-m (teb_vae[\w.]+)", section)))
    assert modules == ["teb_vae.lag_attn_crws.trainer"], modules
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
