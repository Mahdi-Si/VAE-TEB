r"""One full pipeline run of the conv-Transformer model, end to end.

Everything else in this package's evaluation suite drives one seam at a time. This is the pass an
operator makes -- the trained checkpoint, this package's committed delta repointed at generated
shards, every analysis selected, retention caps on so the opt-in figures render -- and what it
proves is the claim the whole binding seam rests on: **the shared pipeline carries a different
model**. Not that the numbers are good; the shards are white noise and a checkpoint trained for
eight steps forecasts nothing. That the seventeen inherited analyses and this model's own
``encoder_attention`` each reach a verdict, that the artifact layout is the sibling's, and that the
run's own record says which architecture produced it.

The offline re-run at the end is the other half. Everything after the collection pass reads the
tables rather than the model, so a finished directory can be re-analysed with no checkpoint, no
model and no GPU -- and that is asserted by making construction *fatal* rather than by observing
that it did not seem to happen.

**The committed figure manifest is kept equal to this run**, which is the bridge into the fast
gate: ``test_eval_docs.py`` binds every figure to a ``FIGURE_GUIDE.md`` entry, and it cannot afford
a pipeline run, so it reads the manifest instead. Drift in either direction fails here -- a figure
a run stopped emitting leaves a stale row, and a new figure is missing from it. Regenerate after a
deliberate figure change by deleting ``eval/figure_manifest.json`` and running this file once.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from teb_vae.lag_attn_rws.eval import collect, preflight, probe as shared_probe
from teb_vae.lag_attn_rws.eval import run as shared_run
from teb_vae.lag_attn_rws.eval.binding import ModelBinding
from teb_vae.lag_attn_transformer_rws.eval import run as trf_run
from teb_vae.lag_attn_transformer_rws.eval import verify as trf_verify
from teb_vae.lag_attn_transformer_rws.eval.analyses import encoder_attention as encoder_analysis
from teb_vae.lag_attn_transformer_rws.eval.binding import TRF_BINDING

from .conftest import write_repointed_overrides

pytestmark = pytest.mark.slow

#: The committed manifest this run must equal.
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "eval" / "figure_manifest.json"

#: The runner's grouped variants are a *family* rather than fixed filenames: it fans one violin
#: figure per cohort axis over whatever each analysis declared, so the set grows with the analyses
#: and the guide documents them together. Normalised out of the per-analysis lists by suffix.
GROUPED_SUFFIXES = ("_by_clinical_class.pdf", "_by_subgroup.pdf")

#: The two class-resolved figures ``encoder_attention`` draws **itself**, which wear the family's
#: suffix and are not members of it: the fan-out resolves per-recording scalars into violins, and
#: these two are per-head and per-distance fields. They keep their manifest rows, and therefore
#: their guide entries, rather than being absorbed into a family entry describing violins.
SELF_DRAWN_CLASS_FIGURES = frozenset(
    f"{encoder_analysis.ANALYSIS_DIRNAME}/{name}"
    for name in (encoder_analysis.ENTROPY_CLASS_FIGURE, encoder_analysis.DISTANCE_CLASS_FIGURE)
)

#: The families the manifest records instead of filenames, each with the marker string its
#: ``FIGURE_GUIDE.md`` entry must contain -- which is what the fast documentation test checks.
FAMILIES: Dict[str, Dict[str, str]] = {
    "grouped_variants": {
        "pattern": "*_by_clinical_class.pdf and *_by_subgroup.pdf, beside the table each resolves",
        "guide_marker": "_by_clinical_class.pdf",
    },
    "sample_pages": {
        "pattern": "samples/<selection>/sample<index>_<guid>_epoch<epoch>.pdf",
        "guide_marker": "The per-sample pages",
    },
}

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
        "run_dir": Path(output_dir),
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


def test_the_artifact_layout_is_the_siblings(smoke_run) -> None:
    """By name: the summary and its heartbeat, the two preflight-side records, the dumped config
    and the log, the two durable tables with their sidecars, and the unskippable channel map."""
    results_dir = smoke_run["results_dir"]

    for name in (
        trf_run.SUMMARY_FILENAME,
        trf_run.STEPS_FILENAME,
        preflight.PREFLIGHT_FILENAME,
        shared_probe.PROBE_FILENAME,
        "resolved_config.yaml",
        trf_run.LOG_FILENAME,
        "per_sample.csv",
        "per_anchor.parquet",
        "per_sample_vectors.npz",
        "coherence_spectra.npz",
        collect.COLLECTION_FILENAME,
        "band_partition.json",
        "band_channel_map.csv",
    ):
        assert (results_dir / name).is_file(), f"the run left no {name}"


def test_one_subdirectory_per_analysis(smoke_run) -> None:
    subdirectories = {path.name for path in smoke_run["results_dir"].iterdir() if path.is_dir()}

    missing = set(trf_run.analysis_registry()) - subdirectories
    assert missing == set(), f"no artifact subdirectory for {sorted(missing)}"


def test_the_sanity_block_is_present_and_three_valued(smoke_run) -> None:
    """The run's self-consistency record. Three-valued rather than boolean: a check that could not
    be computed on this population is not the same as one that failed, and collapsing the two
    would turn every un-testable identity into a green tick."""
    sanity = smoke_run["summary"]["results"]["sanity"]

    assert sanity
    verdicts = [
        entry.get("passed") for entry in sanity.values() if isinstance(entry, dict)
    ]
    assert verdicts, sanity
    assert all(verdict in (True, False, None) for verdict in verdicts), verdicts


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


# =============================================================================
# The committed figure manifest
# =============================================================================
def observed_figures(results_dir: Path) -> Dict[str, List[str]]:
    """Every figure the run emitted, grouped by analysis directory, families normalised out.

    Args:
        results_dir: The finished run's results directory.

    Returns:
        ``analysis -> sorted filenames``, with the per-sample pages and the runner's grouped
        variants removed -- both are documented as families, whose members have no fixed names.
    """
    figures: Dict[str, List[str]] = {}
    for pdf in sorted(results_dir.rglob("*.pdf")):
        relative = pdf.relative_to(results_dir).as_posix()
        if relative.startswith("samples/"):
            continue
        if pdf.name.endswith(GROUPED_SUFFIXES) and relative not in SELF_DRAWN_CLASS_FIGURES:
            continue
        parts = relative.split("/")
        analysis = parts[0] if len(parts) > 1 else "."
        figures.setdefault(analysis, []).append(pdf.name)
    return {analysis: sorted(names) for analysis, names in sorted(figures.items())}


def test_the_committed_manifest_equals_what_the_run_produced(smoke_run) -> None:
    """Both directions at once: a stale row and a missing row are the same failure. When the
    manifest does not exist yet it is seeded from this run and the test fails asking for a review
    -- committing a file nobody has read is not a contract."""
    observed = observed_figures(smoke_run["results_dir"])

    if not MANIFEST_PATH.is_file():
        MANIFEST_PATH.write_text(
            json.dumps(
                {
                    "_comment": (
                        "Every figure a full evaluation run of this model emits, by analysis "
                        "directory, with dynamically-named figure families recorded as families. "
                        "Kept equal to a real run by tests/test_eval_smoke.py; read by the fast "
                        "documentation tests in tests/test_eval_docs.py. Regenerate by deleting "
                        "this file and running the smoke suite once."
                    ),
                    "figures": observed,
                    "families": FAMILIES,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        pytest.fail(f"seeded {MANIFEST_PATH} from this run; review it, commit it, and re-run.")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["figures"] == observed, (
        "the committed figure manifest disagrees with what a real run produces; if the change is "
        "deliberate, delete eval/figure_manifest.json and re-run this suite to reseed it"
    )
    assert manifest["families"] == FAMILIES


def test_this_models_own_class_resolved_figures_survive_the_family_filter(smoke_run) -> None:
    """Non-vacuity for the carve-out above: the two figures ``encoder_attention`` draws itself
    wear the grouped family's suffix, so a filter keyed on the suffix alone would absorb them into
    a family entry that describes per-recording violins -- and they would lose their guide entries
    while every documentation test stayed green."""
    observed = observed_figures(smoke_run["results_dir"])

    emitted = {
        f"{analysis}/{name}" for analysis, names in observed.items() for name in names
    }
    assert SELF_DRAWN_CLASS_FIGURES <= emitted, sorted(SELF_DRAWN_CLASS_FIGURES - emitted)
    # And the runner's own fan-out over the same analysis is still filtered out, so the carve-out
    # is narrow rather than a suffix exemption for the whole directory.
    assert "encoder_attention/encoder_attention_per_recording_by_clinical_class.pdf" not in emitted


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
def test_the_run_context_carries_this_models_own_facts(smoke_run) -> None:
    """The block the arm tables and the first-run checklist read: the parameter count, the
    training epoch, the anchor-coverage distribution and the observed objective magnitude. The
    parameter count is what makes it *this* model's -- the two architectures differ in it, so a
    run context copied from the sibling would be visible here."""
    run_context = smoke_run["summary"]["run_context"]

    assert run_context["n_parameters"] > 0
    assert run_context["train_epoch"] is not None
    assert run_context["anchor_coverage_frac"]
    assert run_context["observed_loss_scale"]["nll_full_block"] is not None


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


def test_the_causality_refusal_ships_verbatim(smoke_run) -> None:
    """Unconditional, and identical for both models: it is a property of the two-sided feature
    bank rather than of either encoder. No number in this run may be labelled a transfer entropy,
    and the sentence that says so is the one thing a reader who opens only the summary must see."""
    causality = smoke_run["summary"]["causality"]

    assert causality["statement"] == preflight.NOT_CAUSAL_STATEMENT
    assert causality["not_causal"] is True
    assert causality["max_channel_reach_s"] > causality["horizon_seconds"]


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


def test_the_run_recorded_the_analyses_it_selected(smoke_run) -> None:
    summary = smoke_run["summary"]

    assert summary["analyses_selected"] == list(trf_run.analysis_registry())
    assert summary["analyses_unskippable"] == list(trf_run.UNSKIPPABLE_ANALYSES)


# =============================================================================
# The offline re-run
# =============================================================================
def test_a_re_run_against_the_finished_directory_builds_no_model(smoke_run) -> None:
    """The property that makes ``--only <name> --output-dir <finished>`` worth having: everything
    after the collection pass reads the tables. Asserted by making construction fatal -- observing
    that a model "did not seem to be built" would pass on a run that built one and discarded it.
    """

    class _Exploding:
        __name__ = "SeqVaeLagAttnTrfRws"

        def __init__(self, *args, **kwargs):
            raise AssertionError("the model was rebuilt on an offline re-run")

    refusing = ModelBinding(
        model_cls=_Exploding,
        task_cls=TRF_BINDING.task_cls,
        tag=TRF_BINDING.tag,
        geometry_keys=TRF_BINDING.geometry_keys,
        encoder_disclosure=TRF_BINDING.encoder_disclosure,
        overrides_path=TRF_BINDING.overrides_path,
    )

    exit_code = trf_run.main(
        None, smoke_run["run_dir"], only="coupling", device="cpu", binding=refusing
    )

    summary = json.loads(
        (smoke_run["results_dir"] / trf_run.SUMMARY_FILENAME).read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert summary["checkpoint"] is None
    assert summary["analyses_selected"] == ["coupling"]
    # The readouts of the run being re-read survive: an offline pass reports the same findings as
    # the pass that collected them.
    assert summary["results"]["readouts"]


def test_the_entry_point_forwards_this_models_binding(monkeypatch) -> None:
    """The one line this module exists for. A default that drifted would evaluate the sibling's
    architecture from this package's command and say so nowhere."""
    seen: Dict[str, Any] = {}

    def _capture(*args: Any, **kwargs: Any) -> int:
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(shared_run, "main", _capture)
    trf_run.main(None, None)

    assert seen["binding"] is TRF_BINDING


def test_the_help_text_lists_the_flags_and_this_models_analyses(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        trf_run._cli(["--help"])
    assert excinfo.value.code == 0

    out = capsys.readouterr().out
    for flag in ("--checkpoint", "--output-dir", "--overrides", "--device", "--num-samples",
                 "--max-batches", "--only", "--skip"):
        assert flag in out
    assert "teb_vae.lag_attn_transformer_rws.eval.run" in out
    for name in trf_run.ANALYSES:
        assert name in out


def test_the_launch_dict_and_the_command_line_resolve_per_key() -> None:
    """The launch dict is what an IDE's Run button uses, and it is resolved *per key* -- the common
    iteration is varying one thing, so a flag must override that one value and leave the rest of
    the dict standing."""
    values, sources = shared_run.resolve_arguments(
        ["--checkpoint", "a.ckpt"],
        run_args={**trf_run.RUN_ARGS, "device": "cpu", "only": "encoder_attention"},
        parser=trf_run.build_parser(),
    )

    assert (values["checkpoint"], sources["checkpoint"]) == ("a.ckpt", "cli")
    assert (values["device"], sources["device"]) == ("cpu", "config")
    assert (values["only"], sources["only"]) == ("encoder_attention", "config")
    assert (values["output_dir"], sources["output_dir"]) == (None, "default")


def test_the_shipped_launch_dict_resolves() -> None:
    """It ships in ``run.py`` and no normal test run exercises it, so a key renamed on the parser
    would otherwise be found by an operator pressing Run rather than by this suite. The unknown-key
    refusal inside the resolver is what turns that rename into a startup error."""
    values, _ = shared_run.resolve_arguments([], parser=trf_run.build_parser(),
                                             run_args=trf_run.RUN_ARGS)

    assert set(values) == set(trf_run.RUN_ARGS)


def test_the_launch_dict_offers_exactly_the_command_lines_own_settings() -> None:
    """A key here that is not a flag is a setting that would appear in no artifact -- the override
    delta is the durable record, and this dict is a launch convenience rather than a second
    configuration surface."""
    dests = {
        action.dest for action in trf_run.build_parser()._actions if action.dest != "help"
    }

    assert set(trf_run.RUN_ARGS) == dests


def test_the_probe_help_text_names_this_package(capsys) -> None:
    from teb_vae.lag_attn_transformer_rws.eval import probe as trf_probe

    with pytest.raises(SystemExit) as excinfo:
        trf_probe._cli(["--help"])
    assert excinfo.value.code == 0

    out = capsys.readouterr().out
    assert "teb_vae.lag_attn_transformer_rws.eval.probe" in out
    assert "--config" in out
