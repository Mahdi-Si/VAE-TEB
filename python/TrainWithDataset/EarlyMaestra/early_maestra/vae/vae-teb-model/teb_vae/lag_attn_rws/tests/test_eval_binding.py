r"""What this model's binding says, and the proof that saying it changed no number.

The evaluation package takes a :class:`~teb_vae.lag_attn_rws.eval.binding.ModelBinding` rather
than naming one model, so a second architecture can reuse it instead of forking it. Two things
have to hold for that to be safe, and both are here.

**The default must not drift.** ``RWS_BINDING`` is what every existing call site gets by
omission, so an edit to it silently changes what this model's runs mean -- which class is
rebuilt, which constructor keys are reconciled, which holdout delta is merged. Each field is
pinned against the literal the code used before the binding existed. The pin lives in *this*
suite deliberately: the rws model's contract is an rws fact, and a change to it should fail where
that contract is maintained rather than in a sibling package that merely happens to import the
seam.

**And the seam must be a pure refactor**, which is what the ``slow`` gate at the bottom checks:
one full pipeline run against the tiny fixture, digested artifact by artifact, against a manifest
captured before any of this code existed. It is a refactor-neutrality gate rather than an
analysis test -- it asserts nothing about whether the numbers are *right*, only that they are the
same ones.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch
import yaml

from teb_vae.lag_attn_rws.eval import preflight, report_seam, run as run_module
from teb_vae.lag_attn_rws.eval.binding import ModelBinding
from teb_vae.lag_attn_rws.eval.config_schema import DEFAULT_OVERRIDES_PATH
from teb_vae.lag_attn_rws.eval.preflight import GEOMETRY_KEYS, rws_encoder_disclosure
from teb_vae.lag_attn_rws.tests.conftest import write_repointed_overrides

#: The constructor keys this model reconciles against a checkpoint, written out rather than
#: imported: importing :data:`GEOMETRY_KEYS` and comparing it to itself would pass on any edit.
#: An **ordered** sequence, not a set -- the order is what the ``compared`` record is built in and
#: what a reader of two runs' preflight files compares down.
RWS_GEOMETRY_KEYS = (
    "sequence_length",
    "d_model",
    "d_z",
    "horizon",
    "raw_per_step",
    "warmup_period",
    "c_y",
    "c_u",
    "use_up_st",
    "max_lag",
    "num_heads",
    "d_head",
    "horizon_attention_blocks",
    "causal_norm",
)


# =============================================================================
# The binding's fields
# =============================================================================
def test_the_binding_names_this_packages_model_and_task() -> None:
    """By name rather than by identity: the assertion should read as the contract does, and an
    import swapped for another class with the same name is a different failure entirely."""
    assert run_module.RWS_BINDING.model_cls.__name__ == "SeqVaeLagAttnRws"
    assert run_module.RWS_BINDING.task_cls.__name__ == "SeqVaeLagAttnRwsTask"


def test_the_tag_is_the_one_the_output_directory_was_always_built_from() -> None:
    """``<tag>-eval`` is where a run with no configured tag lands. A change here silently moves
    every such run's output directory."""
    assert run_module.RWS_BINDING.tag == "lag_attn_rws"


def test_the_geometry_keys_are_exactly_these_in_this_order() -> None:
    assert run_module.RWS_BINDING.geometry_keys == RWS_GEOMETRY_KEYS
    # One constant, one value: the module keeps the tuple as the rws set and the binding is what
    # carries it into the reconciliation, so the two cannot come apart.
    assert run_module.RWS_BINDING.geometry_keys is GEOMETRY_KEYS


def test_the_encoder_disclosure_is_this_encoders() -> None:
    assert run_module.RWS_BINDING.encoder_disclosure is rws_encoder_disclosure


def test_the_overrides_path_is_the_committed_delta_and_it_is_there() -> None:
    path = run_module.RWS_BINDING.overrides_path

    assert path is DEFAULT_OVERRIDES_PATH
    assert Path(path).is_file()
    assert Path(path).name == "eval_overrides.yaml"
    assert Path(path).parent.parent.name == "eval"


def test_this_model_registers_no_analysis_of_its_own() -> None:
    """Empty, and asserted empty rather than left unstated: the shared registry is the *shared*
    models' contract, so an analysis added there reaches every model that uses this pipeline --
    including one for which the question it answers means nothing. A model-specific analysis
    belongs in that model's ``extra_analyses``, which is the only place it is named."""
    assert dict(run_module.RWS_BINDING.extra_analyses) == {}


def test_this_model_adds_no_headline_scalar_of_its_own() -> None:
    """Empty for the same reason ``extra_analyses`` is, and asserted for a sharper one: every path
    in the *shared* headline registry has to resolve on a run of this model, and
    ``test_eval_report.py`` fails when one does not. A model-specific entry therefore cannot go
    there, and one that appeared here would mean an analysis nothing in this package registers."""
    assert run_module.RWS_BINDING.headline_scalars == ()


def test_the_headline_block_is_unchanged_when_a_binding_registers_nothing() -> None:
    """The neutrality claim behind the field, stated where the shared default lives: the extras are
    appended, so a binding that adds none produces the block this pipeline produced before the
    parameter existed -- key for key and in the same order."""
    from teb_vae.lag_attn_rws.eval import report_seam

    results = {"readouts": {"mc_pred_gap": 1.5}, "verdicts": []}
    without = report_seam.build_headline(results)
    with_empty = report_seam.build_headline(results, run_module.RWS_BINDING.headline_scalars)

    assert with_empty == without
    assert list(with_empty) == list(without)
    # And not vacuous: a registered entry does reach the block.
    extended = report_seam.build_headline(results, (("added", ("readouts", "mc_pred_gap")),))
    assert extended["added"] == 1.5


def test_the_binding_cannot_be_edited_after_it_is_declared() -> None:
    """A binding is a declaration, not a setting: a field mutated mid-run would leave a summary
    describing a contract that was not in force when the tables were collected."""
    with pytest.raises(Exception):
        run_module.RWS_BINDING.tag = "something_else"  # type: ignore[misc]


# =============================================================================
# The merged registry
# =============================================================================
def test_the_merged_registry_is_the_shared_one_when_nothing_is_added() -> None:
    merged = run_module.merged_analysis_functions(run_module.RWS_BINDING)

    assert merged == dict(run_module.ANALYSIS_FUNCTIONS)
    assert list(merged) == list(run_module.ANALYSIS_FUNCTIONS)
    # A copy, so a caller that edits what it was handed cannot reach the shared registry.
    assert merged is not run_module.ANALYSIS_FUNCTIONS


def test_extra_analyses_are_appended_after_the_shared_ones_in_declaration_order() -> None:
    """Appended rather than interleaved: reordering the shared registry to place one model's
    addition would change the *sibling's* run order too."""
    def _first(context, **kwargs):
        return {}

    def _second(context, **kwargs):
        return {}

    merged = run_module.merged_analysis_functions(
        _binding_with({"aaa_first": _first, "zzz_second": _second})
    )

    assert list(merged)[-2:] == ["aaa_first", "zzz_second"]
    assert list(merged)[: len(run_module.ANALYSIS_FUNCTIONS)] == list(run_module.ANALYSIS_FUNCTIONS)


def test_an_extra_analysis_may_not_take_a_shared_name() -> None:
    """Silently replacing a shared implementation would leave two models reporting different
    things under one name, which is indistinguishable in the output from them agreeing."""
    shared = next(iter(run_module.ANALYSIS_FUNCTIONS))

    with pytest.raises(ValueError, match=shared):
        run_module.merged_analysis_functions(_binding_with({shared: lambda context, **kw: {}}))


def test_an_extra_headline_scalar_may_not_take_a_shared_name() -> None:
    """The same rule as the registry's, one seam over. The extras resolve last, so a reused name
    would silently replace a shared reading -- and the headline block is the only thing every arm
    table, the acceptance gate and the cross-model row read, so the substitution would be invisible
    everywhere it mattered."""
    shared_name = report_seam.HEADLINE_SCALARS[0][0]

    with pytest.raises(ValueError, match=shared_name):
        report_seam.build_headline({}, ((shared_name, ("anything", "at", "all")),))


def test_the_encoder_disclosure_may_not_take_a_shared_causality_key() -> None:
    """The disclosure is merged into the middle of the causality record, so a reused key would
    either replace a shared one -- including ``statement``, the refusal sentence that record exists
    to carry -- or be dropped by a key below it. Both are silent in an artifact whose whole purpose
    is to be read literally."""
    model = _DisclosureProbeModel()

    with pytest.raises(ValueError, match="statement"):
        preflight.causality_disclosure(
            {}, model, lambda _model: {"statement": "an encoder's own sentence"}
        )


def test_a_disclosure_of_this_encoders_own_keys_is_still_accepted() -> None:
    """Not vacuous: the guard must refuse only the shared names, not every key a disclosure adds."""
    record = preflight.causality_disclosure(
        {}, _DisclosureProbeModel(), lambda _model: {"an_encoder_only_key": 1}
    )

    assert record["an_encoder_only_key"] == 1
    assert record["statement"] == preflight.NOT_CAUSAL_STATEMENT


class _DisclosureProbeModel:
    """The two attributes :func:`causality_disclosure` reads off a net, and nothing else."""

    horizon = 30
    source_delay_steps = 0


def _binding_with(extra: Dict[str, Any]) -> ModelBinding:
    """The rws binding with a different ``extra_analyses``, for the merge cases above."""
    return ModelBinding(
        model_cls=run_module.RWS_BINDING.model_cls,
        task_cls=run_module.RWS_BINDING.task_cls,
        tag=run_module.RWS_BINDING.tag,
        geometry_keys=run_module.RWS_BINDING.geometry_keys,
        encoder_disclosure=run_module.RWS_BINDING.encoder_disclosure,
        overrides_path=run_module.RWS_BINDING.overrides_path,
        extra_analyses=extra,
    )


# =============================================================================
# The refactor-neutrality gate
#
# The reference is data rather than a test, because "a run from before the refactor" cannot be
# expressed as an assertion at HEAD: it was captured by running the pipeline on the pre-binding
# tree and committing the digests. Digests rather than the artifacts themselves -- the set is
# about forty megabytes and what is pinned is equality, which a digest expresses exactly.
# =============================================================================
#: The captured reference.
REFERENCE_MANIFEST = Path(__file__).resolve().parent / "data" / "eval_reference_manifest.json"

#: matplotlib stamps every figure with the wall clock. Blanked before hashing, because the
#: drawing either side of it is byte-identical and excluding fifty-four figures instead would
#: leave most of the emitted tree uncovered.
_PDF_CREATION_DATE = re.compile(rb"/CreationDate\s*\([^)]*\)")


def artifact_digest(path: Path) -> str:
    """Digest one emitted artifact, normalising the two kinds of run-varying container metadata.

    ``.npz`` bundles are hashed over their arrays in sorted key order rather than over the zip
    container: the member order follows the interpreter's hash seed, so two processes write the
    same numbers into a different byte sequence.

    Args:
        path: The artifact.

    Returns:
        Its SHA-256 digest, hex encoded.
    """
    if path.suffix == ".pdf":
        return hashlib.sha256(
            _PDF_CREATION_DATE.sub(b"/CreationDate ()", path.read_bytes())
        ).hexdigest()
    if path.suffix == ".npz":
        digest = hashlib.sha256()
        with np.load(path, allow_pickle=False) as bundle:
            for key in sorted(bundle.files):
                array = bundle[key]
                digest.update(key.encode("utf-8"))
                digest.update(str(array.dtype).encode("utf-8"))
                digest.update(str(array.shape).encode("utf-8"))
                digest.update(array.tobytes())
        return digest.hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shard_paths_by_basename(value: Any) -> Any:
    """Return ``value`` with every shard path reduced to its file name.

    The fixture writes its shards into a per-session temporary directory, so the directory is a
    property of the pytest run rather than of the evaluation. The file *name* is the identity and
    is compared; the directory is not.

    Args:
        value: Any JSON-shaped structure.

    Returns:
        The same structure, with ``*.hdf5`` strings replaced by their basenames.
    """
    if isinstance(value, dict):
        return {key: shard_paths_by_basename(item) for key, item in value.items()}
    if isinstance(value, list):
        return [shard_paths_by_basename(item) for item in value]
    if isinstance(value, str) and value.endswith(".hdf5"):
        return Path(value).name
    return value


@pytest.fixture(scope="module")
def reference() -> Dict[str, Any]:
    """The committed manifest. **Never regenerate this file to make the gate pass**: it is the
    only record of what the pipeline produced before the binding existed, and rewriting it turns
    a failed refactor into a green suite."""
    assert REFERENCE_MANIFEST.is_file(), (
        f"{REFERENCE_MANIFEST} is missing. It is a one-time capture from the pre-binding tree and "
        f"cannot be reconstructed from the current one."
    )
    return json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))


#: The model keys the reference manifest was captured under, restated here because the shipped
#: config has since moved off six of the seven and this gate must not move with it.
#: ``source_dropout`` is the exception and is pinned rather than restored: it still ships at
#: ``null``, which resolves to the pre-key model at every site it touches, so the entry records
#: the capture condition instead of overriding it -- see the exemption in
#: :func:`test_the_legacy_keys_are_the_ones_the_shipped_config_has_moved_off`.
#:
#: The gate asks one question -- *is the binding seam a pure refactor?* -- and answers it by
#: digesting a pipeline run against a capture from before the seam existed. That comparison is
#: only meaningful if everything except the seam is held at the capture's values. When the
#: bottleneck bundle changed what the model computes (the causal guard on, the base branch decoded
#: at the prior mean, an independent posterior log-variance head), every digest moved for reasons
#: that have nothing to do with the seam, and the gate would have been reporting a refactor
#: failure that did not happen.
#:
#: The three decoder keys are the same story one revision later. ``tiny.yaml`` names neither the
#: decoder's width nor its depth, so both are inherited live from ``default.yaml`` -- and the
#: capacity revision moved all three, which changes the decoder's *weights* and therefore every
#: forecast digest in the manifest. Pinned at the capture's geometry, they hold the checkpoint at the
#: model the manifest describes. ``d_z`` needs no entry: ``tiny.yaml`` sets it itself, so the flip in
#: ``default.yaml`` never reaches this fixture.
#:
#: So the checkpoint this gate evaluates is trained at the legacy values, and the shipped values
#: are asserted elsewhere -- in the config-load tests, which is where a claim about a committed
#: file belongs. Regenerating the manifest instead would have destroyed the only record of what
#: the pipeline produced before the seam was opened; see the ``reference`` fixture.
LEGACY_MODEL_KEYS: Dict[str, Any] = {
    "causal_reach_budget_s": None,
    "base_decode": "sample",
    "posterior_logvar_mode": "residual",
    "source_dropout": None,
    "decoder_hidden": 128,
    "horizon_depth": 3,
    "horizon_attention_blocks": 0,
}


@pytest.fixture(scope="module")
def legacy_trained_run(tmp_path_factory) -> Path:
    """``trained_run``, rebuilt at :data:`LEGACY_MODEL_KEYS`.

    Deliberately a copy of the shared fixture's construction rather than a parameter on it: every
    other test in this suite must see the *shipped* model, and a switch on the shared fixture
    would let a future edit point them at the legacy one by accident.
    """
    from teb_vae.lag_attn.config import load_config
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
    from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask
    from teb_vae.lag_attn_rws.tests.conftest import (
        TASK_HPARAMS,
        _REPO_ROOT,
        absolutize_dataset_paths,
    )
    from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME, LagAttnRwsTrainer

    run_dir = tmp_path_factory.mktemp("legacy_run")
    checkpoint_dir = run_dir / "model_checkpoints"
    checkpoint_dir.mkdir()

    tiny = Path(_REPO_ROOT) / "teb_vae" / "lag_attn_rws" / "configs" / "tiny.yaml"
    config = absolutize_dataset_paths(load_config(str(tiny)))
    config["model_config"]["VAE_model"].update(LEGACY_MODEL_KEYS)
    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    driver = LagAttnRwsTrainer(config_file_path=str(config_path))
    model_kwargs = driver._build_model_kwargs()
    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(**model_kwargs)
    # Load-bearing, exactly as in the shared fixture: the delta heads are zero-initialised, so an
    # unperturbed checkpoint is indistinguishable in weight space from one that never loaded.
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for parameter in model.posterior_head.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.1)

    task = SeqVaeLagAttnRwsTask(
        model, lr=1e-3, model_kwargs=model_kwargs,
        **dict(TASK_HPARAMS, likelihood=config["model_config"]["VAE_model"]["likelihood"]),
    )
    blob = {"state_dict": task.state_dict(), "epoch": 0, "global_step": 0,
            "hyper_parameters": dict(task.hparams)}
    task.on_save_checkpoint(blob)
    torch.save(blob, checkpoint_dir / "lag-attn-rws-epoch=00.ckpt")

    (checkpoint_dir / RESOLVED_CONFIG_FILENAME).write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    return checkpoint_dir / "lag-attn-rws-epoch=00.ckpt"


def test_the_legacy_keys_are_the_ones_the_shipped_config_has_moved_off(shipped_vae_config):
    """The guard on the guard. If a key here silently matched the shipped value again, this gate
    would go back to testing the shipped model without anyone choosing that -- and if a key were
    renamed, the override would land nowhere and the gate would quietly drift."""
    for key, legacy in LEGACY_MODEL_KEYS.items():
        assert key in shipped_vae_config, f"{key} is not a live config key any more"
        if key != "source_dropout":
            assert shipped_vae_config[key] != legacy, (
                f"{key} ships at its legacy value {legacy!r}; this gate no longer holds anything "
                f"fixed and the entry should be dropped"
            )


@pytest.fixture(scope="module")
def shipped_vae_config() -> Dict[str, Any]:
    """The shipped ``VAE_model`` block, for the guard above."""
    from teb_vae.lag_attn.config import load_config
    from teb_vae.lag_attn_rws.tests.conftest import _REPO_ROOT

    shipped = Path(_REPO_ROOT) / "teb_vae" / "lag_attn_rws" / "configs" / "default.yaml"
    return load_config(str(shipped))["model_config"]["VAE_model"]


@pytest.fixture(scope="module")
def rerun(reference, legacy_trained_run, multi_class_shards, tmp_path_factory) -> Path:
    """One full pipeline run under exactly the parameters the reference was captured with."""
    run = reference["run"]
    trained_run = legacy_trained_run
    assert run["checkpoint_fixture"] == "trained_run"
    assert run["shards_fixture"] == "multi_class_shards"

    overrides = write_repointed_overrides(
        tmp_path_factory.mktemp("reference_overrides"), multi_class_shards
    )
    delta = yaml.safe_load(overrides.read_text(encoding="utf-8"))
    delta["eval_config"]["caps"] = dict(run["caps"])
    overrides.write_text(yaml.safe_dump(delta, sort_keys=False), encoding="utf-8")

    output_dir = tmp_path_factory.mktemp("reference_eval")
    exit_code = run_module.main(
        trained_run,
        output_dir,
        overrides=overrides,
        device=run["device"],
        num_samples=run["num_samples"],
    )
    assert exit_code == 0
    return Path(output_dir) / run_module.RESULTS_DIRNAME


def _environment_note(reference: Dict[str, Any]) -> str:
    """What to check first when the digests move: a numeric-stack upgrade moves them too."""
    import matplotlib
    import pandas
    import pyarrow
    import torch

    current = {
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "pandas": str(pandas.__version__),
        "matplotlib": str(matplotlib.__version__),
        "pyarrow": str(pyarrow.__version__),
    }
    moved = {
        name: (reference["environment"].get(name), value)
        for name, value in current.items()
        if reference["environment"].get(name) != value
    }
    if not moved:
        return "the numeric stack is the one the reference was captured under, so a difference here is the refactor's"
    return f"the numeric stack has moved since the capture, which moves digests too: {moved}"


@pytest.mark.slow
def test_the_emitted_artifact_set_is_the_one_the_reference_recorded(reference, rerun) -> None:
    """Both directions. A file the run stopped emitting is as much a regression as a changed one,
    and a *new* artifact must be a decision rather than something the digest loop skipped."""
    emitted = {
        path.relative_to(rerun).as_posix() for path in rerun.rglob("*") if path.is_file()
    }
    recorded = set(reference["digests"]) | set(reference["excluded"])

    assert emitted - recorded == set(), f"artifacts the reference does not know about: {sorted(emitted - recorded)}"
    assert recorded - emitted == set(), f"artifacts the run no longer emits: {sorted(recorded - emitted)}"


@pytest.mark.slow
def test_every_recorded_artifact_is_byte_identical(reference, rerun) -> None:
    """The gate. Every CSV, every parquet table, every figure and every array bundle the pipeline
    writes, against digests captured before the binding existed. This asserts nothing about
    whether a number is correct -- only that the refactor did not move it."""
    moved = []
    for relative, expected in sorted(reference["digests"].items()):
        path = rerun / relative
        assert path.is_file(), f"the run emitted no {relative}"
        if artifact_digest(path) != expected:
            moved.append(relative)

    assert moved == [], (
        f"{len(moved)} artifact(s) differ from the pre-binding reference: {moved}\n"
        f"{_environment_note(reference)}.\n"
        f"Do not regenerate {REFERENCE_MANIFEST.name} to make this pass -- it is the only record "
        f"of what the pipeline produced before the seam was opened."
    )


@pytest.mark.slow
def test_the_causality_record_is_unchanged_including_its_key_order(reference, rerun) -> None:
    """Compared parsed rather than by digest, because ``preflight.json`` legitimately carries the
    absolute checkpoint path. Key order is part of the assertion: the encoder's half of this
    record moved behind a callable, and where those keys sit in the record is what a reader of
    two runs' preflight files compares down."""
    written = json.loads((rerun / "preflight.json").read_text(encoding="utf-8"))["causality"]

    assert written == reference["causality"]
    assert list(written) == list(reference["causality"])


@pytest.mark.slow
def test_no_readout_in_the_summary_moved(reference, rerun) -> None:
    """The same claim as the digests make, stated where a reader would look for it: the whole
    ``results`` block, every readout, verdict and per-recording aggregate, against the reference.
    Shard paths are compared by file name, because the directory is the fixture's rather than the
    evaluation's."""
    written = json.loads((rerun / "summary.json").read_text(encoding="utf-8"))["results"]

    assert shard_paths_by_basename(written) == reference["results"]


def test_the_two_normalisations_still_detect_a_real_change(tmp_path) -> None:
    """The normalisations exist so the gate can cover figures and array bundles instead of
    excluding them, and each is a place a blanket exclusion could hide. So each is shown to still
    move on a change to the content it is meant to be comparing: a byte inside the figure's own
    drawing, and a value inside an array. Without this the gate could pass over fifty-seven
    artifacts it was no longer really reading."""
    figure = tmp_path / "figure.pdf"
    stamped = b"%PDF-1.4\n/CreationDate (D:20260101000000-04'00')\n/Type /Page\ntrailer\n"
    figure.write_bytes(stamped)
    first = artifact_digest(figure)

    # A different wall clock, same drawing: the digest must not move, or every figure is excluded.
    figure.write_bytes(stamped.replace(b"D:20260101000000", b"D:20261231235959"))
    assert artifact_digest(figure) == first

    # A different drawing: it must.
    figure.write_bytes(stamped.replace(b"/Type /Page", b"/Type /Font"))
    assert artifact_digest(figure) != first

    bundle = tmp_path / "arrays.npz"
    np.savez(bundle, alpha=np.array([1.0, 2.0]), beta=np.array([3.0]))
    first = artifact_digest(bundle)

    # The same arrays written in a different order: equal, which is the point of the sorted hash.
    np.savez(bundle, beta=np.array([3.0]), alpha=np.array([1.0, 2.0]))
    assert artifact_digest(bundle) == first

    # One value changed: not equal.
    np.savez(bundle, alpha=np.array([1.0, 2.5]), beta=np.array([3.0]))
    assert artifact_digest(bundle) != first


def test_the_reference_covers_the_run_rather_than_a_handful_of_it(reference) -> None:
    """Fast, and the reason the loop above is not vacuous: a truncated or emptied manifest would
    otherwise iterate over nothing and pass. The count is a floor rather than an equality, so a
    deliberate new artifact is a manifest decision rather than a second edit here."""
    assert len(reference["digests"]) >= 120
    assert all(len(digest) == 64 for digest in reference["digests"].values())
    # The tables the numbers actually live in, by name, so a manifest covering only figures fails.
    assert "per_sample.csv" in reference["digests"]
    assert "per_anchor.parquet" in reference["digests"]
    assert sum(name.endswith(".csv") for name in reference["digests"]) >= 60
    assert sum(name.endswith(".pdf") for name in reference["digests"]) >= 40
    assert reference["generated_from"]["git_tree"]


def test_the_exclusions_are_named_decisions(reference) -> None:
    """An exclusion list built by pattern would grow silently. Every entry is a file name with a
    stated reason, and nothing is excluded that a digest could have covered."""
    assert set(reference["excluded"]) == {
        "eval.log",
        "steps.json",
        "summary.json",
        "preflight.json",
        "collection.json",
        "resolved_config.yaml",
        "band_partition.json",
    }
    assert all(reason.strip() for reason in reference["excluded"].values())
    assert set(reference["excluded"]) & set(reference["digests"]) == set()
