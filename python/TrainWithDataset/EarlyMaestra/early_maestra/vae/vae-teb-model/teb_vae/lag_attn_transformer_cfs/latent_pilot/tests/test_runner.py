r"""The three ways this pilot is launched, and the ways each of them breaks silently.

**Execution-machine tests**: they spawn subprocesses with the current interpreter. They need no
fixtures, no checkpoint and no GPU, but they do need the repository importable::

    python -m pytest teb_vae/lag_attn_transformer_cfs/latent_pilot/tests/test_runner.py -q

The convention under test is the one the surrounding package already uses -- an editable dictionary,
a guarded ``__main__``, and a resolver that treats a parser default as the absence of a value -- so
what is checked here is that **this** runner obeys it, from outside the process, where the failures
that matter live:

* **Direct-file execution.** Running ``run.py`` as a script puts *its own* directory on
  ``sys.path``, not the repository root, so every ``teb_vae.`` import fails before ``__main__`` is
  reached. This file is one level deeper than ``trainer.py``, a depth that is easy to copy wrongly
  and impossible to notice afterwards: the symptom is an import error that reads like a broken
  environment.
* **A foreign working directory.** Under a Run button the working directory is whatever the IDE
  chose. Every relative path in the shipped configuration and in ``RUN_ARGS`` resolves against the
  repository root instead, and the way to see that is to launch from somewhere else entirely.
* **Import-time work.** Importing the runner must read nothing, build nothing, create no directory
  and parse no arguments -- because importing happens during collection, during ``--help``, and in
  every editor that indexes the file.

The three launch modes are asserted to fail *identically* on the same bad argument. Agreeing on a
refusal is the cheapest evidence that they share one resolver, which is the property that keeps a
command line and a dictionary from drifting into two behaviours.

Two tests here state the stage registry's contract rather than observing it -- every stage
selectable by an operator has a handler, and no handler is registered under a name that is not a
stage. They are written before the handlers are, which is the order that makes them worth having:
the dispatcher is finished when they pass.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from teb_vae.lag_attn_transformer_cfs.latent_pilot import config as pilot_config
from teb_vae.lag_attn_transformer_cfs.latent_pilot import run as pilot_run

#: The runner as a file, for the direct-execution mode.
RUNNER_FILE = Path(pilot_run.__file__).resolve()

#: A run directory inside the package that does not exist. Inside on purpose: an outside path is
#: refused by the containment check, and this must reach the *missing directory* refusal instead --
#: which is the one that proves the configuration was found and resolved first.
MISSING_RUN_DIR = pilot_config.PILOT_ROOT / "runs" / "no-such-run"


def _pythonpath() -> str:
    """The repository root ahead of whatever PYTHONPATH already holds.

    Returns:
        The value to pass to a subprocess. Prepending rather than assigning matters: an execution
        machine may need its own entries to import the environment at all.
    """
    return os.pathsep.join(
        [str(pilot_config.REPO_ROOT), os.environ.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)


def _launch(arguments, *, cwd, module: bool = False) -> subprocess.CompletedProcess:
    """Run the pilot runner in a subprocess and return the finished process.

    Args:
        arguments: Command-line arguments after the program.
        cwd: Working directory to launch from.
        module: Launch with ``-m`` instead of by file path.

    Returns:
        The completed process, with output captured.
    """
    command = (
        [sys.executable, "-m", "teb_vae.lag_attn_transformer_cfs.latent_pilot.run"]
        if module else [sys.executable, str(RUNNER_FILE)]
    )
    environment = dict(os.environ)
    # The module form needs the root importable; the file form must not, which is the point of the
    # guarded bootstrap inside run.py.
    if module:
        environment["PYTHONPATH"] = _pythonpath()
    return subprocess.run(
        command + list(arguments),
        cwd=str(cwd), env=environment, capture_output=True, text=True, timeout=600,
    )


# =============================================================================
# Import-time behaviour
# =============================================================================
def test_importing_the_runner_creates_nothing_and_prints_nothing(tmp_path):
    """From a foreign working directory, so a stray relative write would land where it is seen."""
    finished = subprocess.run(
        [
            sys.executable, "-c",
            "import teb_vae.lag_attn_transformer_cfs.latent_pilot.run as r; "
            "assert isinstance(r.RUN_ARGS, dict)",
        ],
        cwd=str(tmp_path),
        # Prepended, not assigned: replacing PYTHONPATH drops whatever the execution environment
        # put there -- a remote interpreter sets it -- and the import failure that follows is that
        # environment's, not this package's.
        env={**os.environ, "PYTHONPATH": _pythonpath()},
        capture_output=True, text=True, timeout=600,
    )
    assert finished.returncode == 0, finished.stderr
    assert finished.stdout == ""
    assert sorted(Path(tmp_path).iterdir()) == []


def test_importing_every_pilot_module_starts_no_work(tmp_path):
    """The whole package, not only the runner: a module that read a dataset or built a model on
    import would do it during collection of the logic subset too."""
    modules = (
        "config", "data", "model", "extract", "train", "evaluate", "analyze", "report", "run"
    )
    finished = subprocess.run(
        [
            sys.executable, "-c",
            "import importlib\n"
            + "\n".join(
                f"importlib.import_module("
                f"'teb_vae.lag_attn_transformer_cfs.latent_pilot.{name}')"
                for name in modules
            ),
        ],
        cwd=str(tmp_path),
        # Prepended, not assigned: replacing PYTHONPATH drops whatever the execution environment
        # put there -- a remote interpreter sets it -- and the import failure that follows is that
        # environment's, not this package's.
        env={**os.environ, "PYTHONPATH": _pythonpath()},
        capture_output=True, text=True, timeout=600,
    )
    assert finished.returncode == 0, finished.stderr
    assert sorted(Path(tmp_path).iterdir()) == []


# =============================================================================
# The three launch modes
# =============================================================================
def test_direct_file_execution_bootstraps_the_repository_root(tmp_path):
    """No ``PYTHONPATH``, a foreign working directory, and the runner must still import the
    repository, resolve its shipped configuration and reach the refusal below."""
    finished = _launch(
        ["--stage", "report", "--run-dir", str(MISSING_RUN_DIR)], cwd=tmp_path
    )
    assert finished.returncode != 0
    assert "does not exist" in finished.stderr
    assert "ModuleNotFoundError" not in finished.stderr


def test_the_file_form_does_not_shadow_the_repositorys_own_packages(tmp_path):
    """This package's modules are named ``train``, ``model``, ``data``, ``config`` and ``report``.

    Running the file puts their directory at ``sys.path[0]``, where ``train.py`` shadows the
    repository's top-level ``train`` package -- and the import that then fails,
    ``train.graph_models_utils``, is reached several stages into a run, long after the launch has
    looked successful. So the bootstrap has to drop its own directory as well as add the root, and
    it has to do that even when an inherited PYTHONPATH already carries the root further down the
    list, which is the condition this subprocess reproduces.
    """
    program = (
        "import importlib.util, sys\n"
        f"sys.path.insert(0, r'{RUNNER_FILE.parent}')\n"
        f"spec = importlib.util.spec_from_file_location('run', r'{RUNNER_FILE}')\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "import train.graph_models_utils\n"
        "print('imported the repository package')\n"
    )
    finished = subprocess.run(
        [sys.executable, "-c", program],
        cwd=str(tmp_path),
        env={**os.environ, "PYTHONPATH": _pythonpath()},
        capture_output=True, text=True, timeout=600,
    )

    assert finished.returncode == 0, finished.stderr
    assert "imported the repository package" in finished.stdout


def test_module_execution_refuses_identically(tmp_path):
    finished = _launch(
        ["--stage", "report", "--run-dir", str(MISSING_RUN_DIR)], cwd=tmp_path, module=True
    )
    assert finished.returncode != 0
    assert "does not exist" in finished.stderr


def test_the_programmatic_call_refuses_identically():
    """In-process, through the same ``main`` the two subprocess modes reach."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        pilot_run.main(stage="report", run_dir=str(MISSING_RUN_DIR))


def test_a_relative_configuration_is_found_from_a_foreign_working_directory(tmp_path):
    """The shipped ``RUN_ARGS`` names its configuration relative to the repository root. Launched
    from elsewhere, a resolver that used the working directory would fail on the config long before
    it reached the run directory -- so the *message* is what distinguishes the two."""
    finished = _launch(
        ["--stage", "report", "--run-dir", str(MISSING_RUN_DIR)], cwd=tmp_path
    )
    assert "pilot config" not in finished.stderr


def test_an_unknown_stage_is_refused_before_anything_is_created(tmp_path):
    finished = _launch(["--stage", "finetuning"], cwd=tmp_path)
    assert finished.returncode != 0
    assert "unknown stage" in finished.stderr
    assert sorted(Path(tmp_path).iterdir()) == []


def test_a_production_stage_without_paths_names_the_setting(tmp_path):
    """A stage that needs an input it was not given names the dotted setting, before a first pass
    over any shard.

    Against a configuration of this test's own, never the shipped template. Filling that template
    in is exactly what it asks an operator to do, so a test that depended on its paths still being
    unset would pass on a fresh checkout and fail on every machine actually set up to run the
    pilot -- and it would fail *there*, in the ``tests`` stage that gates all the others. An empty
    ``latent_pilot`` block resolves to the declared defaults, whose production paths are unset.
    """
    config = tmp_path / "paths_unset.yaml"
    config.write_text("latent_pilot: {}\n", encoding="utf-8")

    finished = _launch(["--config", str(config), "--stage", "extract"], cwd=tmp_path)

    assert finished.returncode != 0
    assert "paths.checkpoint" in finished.stderr
    # Refused before ``open_run``, so the refusal costs no run directory anywhere.
    assert sorted(Path(tmp_path).iterdir()) == [config]


def _smoke_without_the_pipeline(monkeypatch, *, absent):
    """Drive ``stage_smoke`` with its two expensive halves replaced.

    The generator runs a real one-epoch fit and the pipeline runs seven stages; neither is what the
    decision under test does. What is under test is which of them runs, and what the record says.

    Args:
        monkeypatch: The pytest fixture.
        absent: Whether the fixtures should look absent.

    Returns:
        ``(result, calls)``: the stage record and the names of the halves that were reached.
    """
    from teb_vae.lag_attn_transformer_cfs.latent_pilot.tests.fixtures import generate as fixtures

    calls = []

    monkeypatch.setattr(
        pilot_config, "missing_inputs",
        lambda settings, stages: ["  a fixture is absent."] if absent else [],
    )
    monkeypatch.setattr(
        fixtures, "generate",
        lambda *a, **k: calls.append("generate") or {"root": "/generated"},
    )
    monkeypatch.setattr(fixtures, "manifest_matches", lambda manifest, settings: [])
    monkeypatch.setattr(
        pilot_run, "run_pipeline",
        lambda *a, **k: calls.append("pipeline") or {"run_dir": "/smoke-run"},
    )
    return pilot_run.stage_smoke({}), calls


def test_the_smoke_stage_writes_its_fixtures_when_they_are_absent(monkeypatch):
    """A checkout that has never generated them is the ordinary case, not a refusal."""
    result, calls = _smoke_without_the_pipeline(monkeypatch, absent=True)

    assert calls == ["generate", "pipeline"]
    assert result["fixtures_generated"] is True
    assert result["clinical"] is False


def test_the_smoke_stage_leaves_fixtures_that_are_already_there(monkeypatch):
    """Regenerating would spend the fit again and move the ground under a run that read them."""
    result, calls = _smoke_without_the_pipeline(monkeypatch, absent=False)

    assert calls == ["pipeline"]
    assert result["fixtures_generated"] is False


def test_a_set_override_reaches_the_settings_the_same_way_the_dictionary_does():
    """Not a subprocess: what matters is that the two sources land in one resolved value."""
    from_dictionary = pilot_run.resolve_run_args(
        {"overrides": {"optim": {"max_epochs": 2}}}, argv=None
    )
    from_command_line = pilot_run.resolve_run_args(
        {}, argv=["--set", "optim.max_epochs=2"]
    )
    assert from_dictionary["overrides"] == from_command_line["overrides"]


# =============================================================================
# The stage registry
# =============================================================================
def test_every_stage_has_a_handler():
    """A stage the operator can select and the dispatcher cannot run is a stage that fails after
    the settings have been resolved and the run directory named."""
    missing = [name for name in pilot_config.STAGES if name not in pilot_run.STAGE_HANDLERS]
    assert missing == [], f"stages with no registered handler: {missing}"


def test_no_handler_is_registered_under_a_name_that_is_not_a_stage():
    assert set(pilot_run.STAGE_HANDLERS) <= set(pilot_config.STAGES)


def test_a_failing_stage_stops_the_sequence(monkeypatch, smoke_fixtures):
    """And the stages after it do not run: a sequence that carried on would report a fitted model
    that was never fitted."""
    ran = []

    def _record(name):
        def handler(context):
            ran.append(name)
            return {}
        return handler

    def _fail(context):
        ran.append("preflight")
        raise RuntimeError("preflight failed on purpose")

    handlers = {name: _record(name) for name in pilot_config.STAGES}
    handlers["preflight"] = _fail
    monkeypatch.setattr(pilot_run, "STAGE_HANDLERS", handlers)

    with pytest.raises(RuntimeError, match="on purpose"):
        pilot_run.main(
            config_path=pilot_config.PILOT_ROOT / "configs" / "smoke.yaml",
            stage=pilot_config.ALL_STAGE,
        )
    assert ran[-1] == "preflight"
    assert "extract" not in ran and "report" not in ran
