"""Argument resolution, the run directory, and one end-to-end pass over the tiny fixture.

The four resolution paths are tested individually because the per-key rule is easy to get
subtly wrong in a way nothing else notices: an all-or-nothing fallback would discard the whole
dict the moment a single flag appeared, and the run would silently use defaults for everything
the operator thought they had configured.
"""
from __future__ import annotations

import json

import pytest

from teb_vae.lag_attn.eval import report as report_module
from teb_vae.lag_attn.eval import run as run_module
from teb_vae.lag_attn.eval.tests.conftest import EVAL_TINY_CONFIG


# ---------------------------------------------------------------------------
# Argument resolution
# ---------------------------------------------------------------------------
def test_command_line_only():
    values, sources = run_module.resolve_arguments(
        ["--config", "a.yaml", "--checkpoint", "b.ckpt"], run_args={}
    )
    assert values["config"] == "a.yaml" and values["checkpoint"] == "b.ckpt"
    assert sources["config"] == "cli"
    assert values["device"] is None and sources["device"] == "default"


def test_run_args_only():
    values, sources = run_module.resolve_arguments(
        [], run_args={"config": "a.yaml", "checkpoint": "b.ckpt", "device": "cpu"}
    )
    assert values["checkpoint"] == "b.ckpt"
    assert sources["checkpoint"] == "RUN_ARGS"
    assert values["device"] == "cpu"


def test_a_partial_command_line_overrides_only_its_own_key():
    """The common IDE iteration: vary the checkpoint, leave everything else to the dict."""
    values, sources = run_module.resolve_arguments(
        ["--checkpoint", "other.ckpt"],
        run_args={"config": "a.yaml", "checkpoint": "b.ckpt", "only": "forecast"},
    )
    assert (values["checkpoint"], sources["checkpoint"]) == ("other.ckpt", "cli")
    assert (values["config"], sources["config"]) == ("a.yaml", "RUN_ARGS")
    assert (values["only"], sources["only"]) == ("forecast", "RUN_ARGS")


def test_a_required_argument_absent_from_both_sources_names_both():
    with pytest.raises(SystemExit):
        run_module.resolve_arguments(["--config", "a.yaml"], run_args={})


def test_an_unknown_run_args_key_raises_at_startup():
    """Otherwise a typo there silently does nothing -- the same class of failure as in the YAML."""
    with pytest.raises(ValueError, match="not command-line arguments") as excinfo:
        run_module.resolve_arguments([], run_args={"checkpint": "b.ckpt"})
    assert "'checkpint'" in str(excinfo.value)
    assert "checkpoint" in str(excinfo.value), "the message must list the valid set"


def test_shipped_run_args_carry_no_key_that_is_not_a_flag():
    """RUN_ARGS must not become a rival configuration surface."""
    parser = run_module.build_parser()
    dests = {action.dest for action in parser._actions if action.dest != "help"}
    assert set(run_module.RUN_ARGS) <= dests


# ---------------------------------------------------------------------------
# Analysis selection
# ---------------------------------------------------------------------------
def test_select_analyses_defaults_to_everything():
    assert run_module.select_analyses(("a", "b"), None, None) == ["a", "b"]


def test_select_analyses_honours_only_and_skip():
    assert run_module.select_analyses(("a", "b", "c"), "c,a", None) == ["a", "c"]
    assert run_module.select_analyses(("a", "b", "c"), None, "b") == ["a", "c"]
    assert run_module.select_analyses(("a", "b", "c"), "a,b", "b") == ["a"]


def test_select_analyses_rejects_an_unknown_name():
    """A misspelled --only would otherwise silently run everything."""
    with pytest.raises(ValueError, match="unknown analyses"):
        run_module.select_analyses(("a", "b"), "forcast", None)


def test_a_subset_missing_a_declared_dependency_is_rejected_naming_it():
    """A subset that cannot be read correctly must fail loudly, not produce plausible output."""
    with pytest.raises(ValueError, match="declared dependency") as excinfo:
        run_module.select_analyses(
            ("a", "b", "c"), "b", None, dependencies={"b": ("a",)}
        )
    assert "b needs 'a'" in str(excinfo.value)


def test_skip_can_violate_a_dependency_that_only_survives_alone():
    """--only and --skip can each be innocent and jointly drop a dependency, so the check runs
    on the final subset rather than per flag."""
    with pytest.raises(ValueError, match="declared dependency"):
        run_module.select_analyses(("a", "b"), None, "a", dependencies={"b": ("a",)})


def test_a_subset_carrying_its_dependency_is_accepted():
    assert run_module.select_analyses(
        ("a", "b", "c"), "a,b", None, dependencies={"b": ("a",)}
    ) == ["a", "b"]


def test_the_shipped_dependency_table_names_only_real_analyses():
    """A table entry naming an analysis that does not exist could never be satisfied."""
    for name, needed in run_module.ANALYSIS_DEPENDENCIES.items():
        assert name in run_module.ANALYSES, f"{name} is not a registered analysis"
        for dependency in needed:
            assert dependency in run_module.ANALYSES, f"{dependency} is not registered"


def test_the_full_analysis_set_satisfies_the_shipped_dependency_table():
    """The default run must never be the thing the dependency check rejects."""
    assert run_module.select_analyses(run_module.ANALYSES, None, None) == list(
        run_module.ANALYSES
    )


# ---------------------------------------------------------------------------
# Run directory and loader policy
# ---------------------------------------------------------------------------
def test_run_directory_is_timestamped_with_a_collision_guard(tmp_path):
    """Two runs launched in the same second must not write into each other's directory."""
    config = {
        "general_config": {"tag": "eval_tag", "folders_config": {"out_dir_base": str(tmp_path)}}
    }
    first = run_module.make_output_dir(config)
    second = run_module.make_output_dir(config)
    assert first != second
    assert first.name == second.name == run_module.RESULTS_DIRNAME
    assert first.exists() and second.exists()


def test_explicit_output_dir_is_used_as_given(tmp_path):
    results = run_module.make_output_dir({}, tmp_path / "chosen")
    assert results == tmp_path / "chosen" / run_module.RESULTS_DIRNAME


def test_num_workers_is_forced_to_zero_with_a_warning():
    config = {"dataset_config": {"dataloader_config": {"num_workers": 4}}}
    run_module.force_single_process_loader(config)
    loader_config = config["dataset_config"]["dataloader_config"]
    assert loader_config["num_workers"] == 0
    assert loader_config["persistent_workers"] is False


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
def test_end_to_end_against_the_tiny_fixture(tiny_checkpoint, tmp_path, monkeypatch, repo_root):
    """The whole pipeline, asserting the directory tree and a summary that parses."""
    monkeypatch.chdir(repo_root)
    output_dir = tmp_path / "run"

    exit_code = run_module.main(
        config=str(repo_root / EVAL_TINY_CONFIG),
        checkpoint=str(tiny_checkpoint),
        output_dir=str(output_dir),
        device="cpu",
        argument_sources={"config": "cli", "checkpoint": "cli"},
    )
    assert exit_code == 0

    results = output_dir / run_module.RESULTS_DIRNAME
    for name in ("summary.json", "preflight.json", "loader_probe.json", "resolved_config.yaml"):
        assert (results / name).is_file(), f"{name} missing from the run directory"

    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    assert summary["exit_code"] == 0 and summary["n_failed"] == 0
    assert summary["results"]["geometry"]["c_y"] == 109
    assert summary["results"]["objective"]["likelihood"] == "gaussian_nll"
    assert summary["results"]["numerics"]["cudnn_benchmark"] is False
    assert summary["results"]["eval_config"]["seed"] == 42
    # The tell for a truncating loader: how many samples each step actually processed.
    assert summary["results"]["probe"]["n_samples"] == 4
    assert summary["results"]["arguments"]["sources"]["checkpoint"] == "cli"

    preflight_record = json.loads((results / "preflight.json").read_text(encoding="utf-8"))
    assert preflight_record["checks"]["weights_loaded"]["passed"] is True
    assert preflight_record["health_probe"]["residual_ratio"] > 0.0

    # Every registered analysis ran, wrote its directory, and reached summary.json. Registration
    # is three lines, and an analysis that is "done" but unreachable from the CLI would only ever
    # be exercised directly -- never through report.step, which is where a failure is captured.
    assert set(summary["results"]["analyses_selected"]) == set(run_module.ANALYSES)
    for name in run_module.ANALYSES:
        assert name in summary["results"], f"{name} did not reach summary.json"
        # An analysis that recorded a skip legitimately writes nothing -- and must not leave a
        # half-built directory behind either, which is what the else branch pins. Everything
        # that did run has to have produced its directory.
        if (summary["results"][name] or {}).get("skipped"):
            assert not (results / name).exists(), (
                f"{name}/ was created by an analysis that recorded a skip"
            )
        else:
            assert (results / name).is_dir(), f"{name}/ was not created"
    assert [record["name"] for record in summary["steps"]] == [
        "probe", "band_partition", *run_module.ANALYSES
    ]

    # The committed fixture is synthesised rather than pipeline-produced, so it carries no sel_*
    # channel provenance. That is a recorded skip, not a failure: whether a shard predates
    # _write_selection_attrs is a property of the file, and the rest of the run is unaffected.
    band = summary["results"]["band_partition"]
    assert band["skipped"] is True
    assert band["attempts"], "the skip must say which shard failed and why"

    # And the analysis that consumes the partition skips *with* it, rather than failing the run.
    # This is the one place the two are asserted to agree; they are written in different modules.
    frequency_band = summary["results"]["frequency_band"]
    assert frequency_band["skipped"] is True
    assert "sel_*" in frequency_band["reason"]

    assert summary["results"]["forecast"]["n_samples"] == 4
    assert (results / "forecast" / "per_sample.csv").is_file()
    assert (results / "uplift" / "per_sample.csv").is_file()
    assert (results / "residual" / "per_anchor.csv").is_file()
    assert (results / "scalars" / "test_metrics.csv").is_file()

    # The joint collapse verdict is promoted to a top-level field: it is the run's headline
    # conclusion, and a reader should not have to know which analysis produced it.
    assert summary["results"]["collapse"]["verdict"] in {
        "collapsed", "inconclusive", "healthy"
    }

    # ---- The finalised summary -------------------------------------------------
    block = summary["results"]
    for key in report_module.REQUIRED_RESULT_KEYS:
        assert key in block, f"{key} missing from summary.json"

    # The manifest is what makes the documentation tests non-circular, so it must match disk.
    manifest = block["artifacts"]
    on_disk = {
        path.relative_to(results).as_posix()
        for path in results.rglob("*")
        if path.is_file() and path.name != "summary.json"
    }
    assert set(manifest["files"]) == on_disk
    assert manifest["n_figures"] > 0 and all(
        name.endswith(".pdf") for name in manifest["figures"]
    )

    # Headline scalars are flattened out of the analysis blocks and are finite.
    assert block["headline"]["feat_mse"] == pytest.approx(
        block["forecast"]["mean_feat_mse_total"]
    )
    assert block["sanity"]["checks"]["headline_finite"]["verdict"] == "pass"
    assert set(block["sanity"]["checks"]) == {
        "per_file_counts", "classes_present", "argmax_lag", "headline_finite",
        "target_not_truncated",
    }

    # Effective n per analysis: the tell for two analyses on different populations.
    assert block["coverage"]["per_analysis"]["forecast"]["n_samples"] == 4

    # Absent, not zero -- the smoke run is on CPU.
    assert "max_memory_allocated_gb" not in block

    # The pages the samples analysis emitted are on disk and named in the manifest.
    assert block["samples"]["n_figures"] == 2, "eval_tiny.yaml caps samples at 2"
    assert any(name.startswith("samples/") for name in manifest["figures"])


def test_skip_leaves_an_analysis_out_of_the_run(
    tiny_checkpoint, tmp_path, monkeypatch, repo_root
):
    """``--skip`` must actually remove the step, not merely reorder it."""
    monkeypatch.chdir(repo_root)
    output_dir = tmp_path / "partial"

    run_module.main(
        config=str(repo_root / EVAL_TINY_CONFIG),
        checkpoint=str(tiny_checkpoint),
        output_dir=str(output_dir),
        device="cpu",
        skip="uplift,residual",
    )
    summary = json.loads(
        (output_dir / run_module.RESULTS_DIRNAME / "summary.json").read_text(encoding="utf-8")
    )
    names = [record["name"] for record in summary["steps"]]
    assert "uplift" not in names and "residual" not in names
    assert "forecast" in names and "scalars" in names


def test_end_to_end_refuses_an_unloaded_checkpoint(tmp_path, monkeypatch, repo_root):
    """An untrained checkpoint is weight-space indistinguishable from one that never loaded."""
    import torch

    from teb_vae.lag_attn.eval.tests.conftest import build_tiny_checkpoint_blob

    monkeypatch.chdir(repo_root)
    path = tmp_path / "zero_init.ckpt"
    torch.save(build_tiny_checkpoint_blob(perturb=False), path)

    with pytest.raises(RuntimeError, match="still exactly zero"):
        run_module.main(
            config=str(repo_root / EVAL_TINY_CONFIG),
            checkpoint=str(path),
            output_dir=str(tmp_path / "run"),
            device="cpu",
        )


def test_max_samples_flag_overrides_the_config(tiny_checkpoint, tmp_path, monkeypatch, repo_root):
    monkeypatch.chdir(repo_root)
    output_dir = tmp_path / "capped"
    run_module.main(
        config=str(repo_root / EVAL_TINY_CONFIG),
        checkpoint=str(tiny_checkpoint),
        output_dir=str(output_dir),
        device="cpu",
        max_samples=2,
    )
    summary = json.loads(
        (output_dir / run_module.RESULTS_DIRNAME / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["results"]["probe"]["n_samples"] == 2

    # It must reach the *analyses* too, not only the probe. The flag says it overrides
    # eval_config.max_samples, and an override that bounded the coverage record alone would
    # leave every headline number computed over the whole split while the summary claimed a cap.
    assert summary["results"]["eval_config"]["max_samples"] == 2
    assert summary["results"]["forecast"]["n_samples"] == 2
    assert summary["results"]["coverage"]["per_analysis"]["forecast"]["n_samples"] == 2


@pytest.mark.parametrize("bad_cap", [0, -1])
def test_a_nonpositive_max_samples_flag_is_rejected_rather_than_silently_capping_to_one(
    tiny_checkpoint, tmp_path, monkeypatch, repo_root, bad_cap
):
    """The flag must meet the same ``minimum=1`` bound the YAML key does.

    Assigned onto the resolved block after validation, ``0`` and ``-1`` both slipped past it and
    evaluated a single batch -- ``seen >= 0`` fires after the first yield -- which is
    indistinguishable in the output from the legal ``--max-samples 1``. A ``-1`` typed meaning
    "no cap" therefore produced the smallest run the pipeline can do and said nothing about it.
    """
    monkeypatch.chdir(repo_root)
    with pytest.raises(ValueError, match="max_samples"):
        run_module.main(
            config=str(repo_root / EVAL_TINY_CONFIG),
            checkpoint=str(tiny_checkpoint),
            output_dir=str(tmp_path / f"bad{bad_cap}"),
            device="cpu",
            max_samples=bad_cap,
        )


def test_rerunning_into_a_finished_run_preserves_the_prior_summary(
    tiny_checkpoint, tmp_path, monkeypatch, repo_root
):
    """The documented single-analysis re-run must not destroy the run it re-runs into.

    ``report.write`` opens ``summary.json`` with mode ``'w'`` and the manifest treats every
    earlier file as stale, so a ``--only cross_subgroup`` pass into a finished directory replaced
    a complete summary with one whose ``headline`` was entirely ``null`` -- and exited 0. The
    per-sample CSVs survive that, but the sanity block, the coverage record and the two promoted
    verdicts exist nowhere else.
    """
    monkeypatch.chdir(repo_root)
    output_dir = tmp_path / "reused"
    results_dir = output_dir / run_module.RESULTS_DIRNAME

    run_module.main(
        config=str(repo_root / EVAL_TINY_CONFIG),
        checkpoint=str(tiny_checkpoint),
        output_dir=str(output_dir),
        device="cpu",
        max_samples=2,
    )
    first = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    assert first["results"]["forecast"]["n_samples"] == 2, "the first pass did not produce results"

    run_module.main(
        config=str(repo_root / EVAL_TINY_CONFIG),
        checkpoint=str(tiny_checkpoint),
        output_dir=str(output_dir),
        device="cpu",
        max_samples=2,
        only="cross_subgroup",
    )

    backups = sorted(results_dir.glob("summary.bak.*.json"))
    assert backups, "the prior summary was overwritten without being preserved"
    preserved = json.loads(backups[-1].read_text(encoding="utf-8"))
    assert preserved["results"]["forecast"]["n_samples"] == 2, (
        "the preserved copy is not the prior run's complete summary"
    )
