r"""One full pipeline run of the conv-Transformer model, end to end.

Everything else in this package's evaluation suite drives one seam at a time. This is the pass an
operator makes -- the trained checkpoint, this package's committed delta repointed at generated
shards, every analysis selected, retention caps on so the opt-in figures render -- and what it
proves is the claim the whole binding seam rests on: **the shared pipeline carries a different
model**. Not that the numbers are good; the shards are white noise and a checkpoint trained for a
handful of steps forecasts nothing. That every inherited analysis and this model's own
``encoder_attention`` reach a verdict, that this model's headline scalars and grouped fan-out reach
the summary, and that the run's own record says which architecture produced it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from teb_vae.lag_attn_rws.eval import run as shared_run
from teb_vae.lag_attn_transformer_rws.eval import run as trf_run
from teb_vae.lag_attn_transformer_rws.eval import verify as trf_verify
from teb_vae.lag_attn_transformer_rws.eval.binding import TRF_BINDING

from .conftest import write_repointed_overrides

pytestmark = pytest.mark.slow

#: Retention caps for this run, all small: the opt-in figures -- the forecast overlay, the lag
#: heatmap, the per-sample pages -- render only where something was retained, and a run without
#: them would assert the artifact layout while never exercising the parts of it that cost memory.
#:
#: ``encoder_attention`` is this model's own, and is the one cap that buys a whole *analysis*
#: rather than a figure: absent, it records a skip and the run would assert an eighteen-analysis
#: layout while never running the eighteenth. Eight rather than a smaller number, because the draw
#: is stratified over the eight subgroup shards with a floor of one -- so eight is exactly what
#: reaches all three clinical classes, which is what makes the grouped fan-out below a test rather
#: than a coin toss.
SMOKE_CAPS = {"waveforms": 4, "attention": 2, "pages": 2, "oracle": 8, "encoder_attention": 8}


@pytest.fixture(scope="module")
def smoke_run(trained_run, multi_class_shards, tmp_path_factory) -> Dict[str, Any]:
    """The full pipeline against the trained checkpoint, through this package's entry point."""
    overrides = write_repointed_overrides(
        tmp_path_factory.mktemp("smoke_overrides"), multi_class_shards
    )
    delta = yaml.safe_load(overrides.read_text(encoding="utf-8"))
    delta["eval_config"]["caps"] = dict(SMOKE_CAPS)
    overrides.write_text(yaml.safe_dump(delta, sort_keys=False), encoding="utf-8")

    output_dir = tmp_path_factory.mktemp("trf_smoke")
    exit_code = trf_run.main(
        trained_run, output_dir, overrides=overrides, device="cpu", num_samples=2
    )
    results_dir = Path(output_dir) / trf_run.RESULTS_DIRNAME
    summary = json.loads(
        (results_dir / trf_run.SUMMARY_FILENAME).read_text(encoding="utf-8")
    )
    return {
        "exit_code": exit_code,
        "results_dir": results_dir,
        "summary": summary,
    }


# =============================================================================
# The run
# =============================================================================
def test_the_full_run_completes_with_exit_code_zero(smoke_run) -> None:
    assert smoke_run["exit_code"] == 0
    assert smoke_run["summary"]["failed"] == []


def test_every_registered_analysis_reports_a_status(smoke_run) -> None:
    """A skip is acceptable and is recorded; a raise is not. Eighteen analyses plus the
    unskippable channel map and the loader probe -- a registry entry with no step record is an
    analysis the run silently lost."""
    steps = {record["name"]: record["status"] for record in smoke_run["summary"]["steps"]}

    expected = {"probe", *trf_run.UNSKIPPABLE_ANALYSES, *trf_run.analysis_registry()}
    assert expected <= set(steps), sorted(expected - set(steps))
    raised = {name: status for name, status in steps.items() if status not in ("ok", "skipped")}
    assert raised == {}, raised


def test_the_opt_in_families_actually_rendered(smoke_run) -> None:
    """The caps were set, so the run must contain what they buy -- otherwise this file would
    assert a layout while never exercising the parts of it that retain anything."""
    results_dir = smoke_run["results_dir"]

    assert list(results_dir.glob("samples/*/*.pdf")), "no per-sample pages rendered"
    assert (results_dir / "forecast" / "forecast_overlay.pdf").is_file()
    assert (results_dir / "attention" / "lag_heatmap.pdf").is_file()
    grouped = [
        path
        for path in results_dir.rglob("*.pdf")
        if path.name.endswith(("_by_clinical_class.pdf", "_by_subgroup.pdf"))
    ]
    assert grouped, "no grouped variants rendered against a multi-class split"


def test_this_models_own_analysis_ran_and_reached_the_headline(smoke_run) -> None:
    """The one analysis the sibling cannot have, end to end. Two claims, and the second is the one
    the seam exists for: it ran and wrote its own subdirectory, *and* its scalars reached the
    headline block -- which is the only block an arm table reads, so a number that stopped there
    would be a number no comparison could use."""
    results = smoke_run["summary"]["results"]
    directory = smoke_run["results_dir"] / "encoder_attention"

    assert results["encoder_attention"]["n_samples"] == SMOKE_CAPS["encoder_attention"]
    assert results["encoder_attention"].get("skipped") is not True
    assert (directory / "encoder_attention_entropy.pdf").is_file()
    assert (directory / "encoder_attention_heatmap.pdf").is_file()
    for name, _path in TRF_BINDING.headline_scalars:
        assert results["headline"][name] is not None, name


def test_the_runners_grouped_fan_out_reached_this_models_own_analysis(smoke_run) -> None:
    """The other half of "on the shared cohort grid": the per-recording frame this analysis
    declares is fanned out by the *runner*, in the same cohort order and palette as every other
    analysis's, into files whose names cannot collide with the two class-resolved figures the
    analysis draws itself."""
    results = smoke_run["results_dir"]
    grouped = smoke_run["summary"]["results"]["encoder_attention"]["grouped"]
    record = grouped["encoder_attention_per_recording"]["clinical_class"]

    assert record["skipped"] is False, record
    assert (results / "encoder_attention" / "encoder_attention_per_recording_by_clinical_class.csv").is_file()
    assert (results / "encoder_attention" / "encoder_attention_per_recording_by_clinical_class.pdf").is_file()
    assert record["groups"] == [
        name for name in ("healthy", "acidosis", "hie") if name in record["groups"]
    ]


def test_the_shared_headline_block_is_untouched_by_this_models_additions(smoke_run) -> None:
    """Appended, never merged into the shared registry: the sibling's every headline path must
    resolve on a sibling run, so an entry there would read as a number that model failed to
    produce. Here the two sets are disjoint and the shared names are all still present."""
    from teb_vae.lag_attn_rws.eval import report_seam

    headline = smoke_run["summary"]["results"]["headline"]
    shared = {name for name, _ in report_seam.HEADLINE_SCALARS}
    local = {name for name, _ in TRF_BINDING.headline_scalars}

    assert shared & local == set()
    assert shared <= set(headline)


# =============================================================================
# What the run says about the model that produced it
# =============================================================================
def test_which_architecture_produced_the_run_is_legible_from_the_run(smoke_run) -> None:
    """A cross-model table has to key its rows on something, and this is what it keys them on.

    The ``model_class`` stamp is written by the checkpoint contract and lives in the blob; the
    dumped config carries every constructor keyword and not the class they build. So the run
    copies the stamp into ``run_context``, and the row keys on the *artifact* rather than on a
    checkpoint the comparison may no longer have beside it -- or on a directory name, which a
    rename would relabel."""
    blob = shared_run.read_checkpoint(smoke_run["summary"]["checkpoint"])

    assert blob["model_class"] == TRF_BINDING.model_cls.__name__
    assert smoke_run["summary"]["run_context"]["model_class"] == TRF_BINDING.model_cls.__name__
    dumped = (smoke_run["results_dir"] / "resolved_config.yaml").read_text(encoding="utf-8")
    assert "model_class" not in dumped
    # And the path the cross-model table keys on resolves against a real summary. Pinned here
    # rather than in the verify suite, whose runs are synthetic: a constant agreeing with a
    # fixture it also wrote proves only that the fixture was copied from it.
    from teb_vae.lag_attn_rws.eval import verify as shared_verify

    assert shared_verify._dig(smoke_run["summary"], *trf_verify.MODEL_CLASS_PATH) == (
        TRF_BINDING.model_cls.__name__
    )


def test_the_encoder_half_of_the_disclosure_is_this_models(smoke_run) -> None:
    """And the half that is *not* shared. The sibling records ``causal_norm``; this architecture
    has no time-pooling normaliser for that key to describe, so it discloses what is true here
    instead -- and reporting the sibling's key anyway would read as a setting someone could
    change."""
    causality = smoke_run["summary"]["causality"]

    assert causality["time_pooling_normalisers"] == 0
    assert causality["time_pooling_normalisers_are_structural"] is True
    assert "n_depthwise_init" in causality
    assert "source_reach_vs_lag_range" in causality
    assert "causal_norm" not in causality
    assert "n_causalized_norms" not in causality


def test_preflight_reconciled_the_encoder_geometry(smoke_run) -> None:
    """The seven keys are only a guard if they were actually compared. A record that passed
    because every key was absent is indistinguishable, in the artifact, from one that checked."""
    compared = smoke_run["summary"]["preflight"]["checks"]["config_matches_checkpoint"]["compared"]

    for key in (
        "encoder_conv_kernels",
        "encoder_conv_dilations",
        "encoder_num_heads",
        "encoder_d_ff",
        "target_attention_blocks",
        "source_attention_blocks",
        "source_attention_window",
    ):
        assert key in compared, f"{key} was never reconciled against the checkpoint"
    assert "causal_norm" not in compared
