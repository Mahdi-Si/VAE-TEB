r"""S4-T05: the stage and report-section registries.

Sprints 5, 6 and 7 each add one analysis (predictive calibration, interventional lag
attribution, neural CMI). Naively all three would edit the same three surfaces -- the
``if args.stage == ...`` chain in ``run_pipeline_v2.main``, the hardcoded section order in
``final_report_v2._render_markdown``, and the four stage tables. That is neither additive nor
parallelisable. Two registries fix it:

* ``register_stage(StageSpec(...))`` -- the four stage tables (``--stage`` choices, the dict
  driver's order, its on-by-default set, and the arm-scoped set) are all **derived** from one
  ``OrderedDict``, so a new stage is dispatchable without touching ``main()``;
* ``register_section(SectionSpec(...))`` -- report sections are appended from their own module,
  and a section that raises renders ``n/a`` rather than losing the whole report.

The load-bearing property tested here is that a ``fatal=False`` stage which raises is caught,
logged as ``failed (non-fatal)``, and does **not** abort the run: a diverging CMI fit must
never kill a headline grading pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import pytest  # noqa: E402

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (  # noqa: E402
    final_report_v2 as fr,
    run_pipeline_v2 as rp,
)

# The tables exactly as they were hardcoded before the refactor. The registry must reproduce
# them byte-for-byte, or an existing stage silently changed behaviour.
_OLD_STAGES = {"build", "r0_realizability", "data_previews", "train", "beta_select",
               "eval", "test_plots", "report"}
_OLD_ORDER = ("solve_te", "am_check", "recover", "r0_realizability", "build",
              "data_previews", "scatter_preview", "beta_select", "train", "eval",
              "test_plots", "report")
_OLD_DEFAULT_ON = {"r0_realizability", "build", "data_previews", "train", "eval",
                   "test_plots", "report"}
_OLD_MODEL_DEPENDENT = {"train", "beta_select", "eval", "test_plots", "report"}


@pytest.fixture
def clean_registry():
    """Snapshot/restore the stage registry so a test's dummy stage does not leak."""
    saved = dict(rp._STAGE_REGISTRY)
    yield
    rp._STAGE_REGISTRY.clear()
    rp._STAGE_REGISTRY.update(saved)


@pytest.fixture
def clean_sections():
    """Snapshot/restore the report-section registry."""
    saved = list(fr._SECTION_REGISTRY)
    yield
    fr._SECTION_REGISTRY[:] = saved


# ---------------------------------------------------------------------------
# The four tables are derived, and unchanged
# ---------------------------------------------------------------------------
def test_stage_tables_reproduce_the_hardcoded_ones() -> None:
    """The builtins still reproduce the pre-refactor tables, byte for byte.

    Guarded as a *prefix / subset*, not as equality. The stage registry is a module global
    and the Sprint 5-7 plugins (``calibration``, ``lag_intervention``, ``cmi``) register
    themselves on import, so any test in the session that imports one -- or that calls
    ``build_parser`` / ``run_pipeline``, both of which invoke ``_load_stage_plugins`` --
    appends to it. Equality would then fail for a reason that has nothing to do with a
    builtin changing, which is what this guard exists to catch. The builtins occupy
    ``order`` 0-11, so they remain the ordered prefix of ``stage_order()``.
    """
    builtin = {spec.name for spec in rp._BUILTIN_STAGE_SPECS}

    assert _OLD_STAGES <= set(rp.stage_names())
    assert rp.stage_order()[: len(_OLD_ORDER)] == _OLD_ORDER
    assert _OLD_DEFAULT_ON <= {n for n, on in rp.stage_defaults().items() if on}
    assert _OLD_MODEL_DEPENDENT <= set(rp.model_dependent_stages())

    # No builtin gained or lost a property: restrict each table back to the builtins.
    assert {n for n in rp.stage_names() if n in builtin} == _OLD_STAGES
    assert {n for n, on in rp.stage_defaults().items() if on and n in builtin} == _OLD_DEFAULT_ON
    assert {n for n in rp.model_dependent_stages() if n in builtin} == _OLD_MODEL_DEPENDENT


def test_every_cli_stage_has_a_runner() -> None:
    for name in rp.stage_names():
        assert rp._STAGE_REGISTRY[name].run is not None, f"{name} has no runner"


def test_non_cli_stages_are_the_four_diagnostics() -> None:
    """``--stage`` never dispatched these; they have their own flags / dict-driver blocks."""
    non_cli = [n for n, s in rp._STAGE_REGISTRY.items() if not s.cli]
    assert set(non_cli) == {"solve_te", "am_check", "recover", "scatter_preview"}


def test_parser_choices_come_from_the_registry(clean_registry) -> None:
    rp.register_stage(rp.StageSpec("zzz_probe", 99, False, True, lambda ctx: 0))
    parser = rp.build_parser()
    args = parser.parse_args(["--stage", "zzz_probe"])
    assert args.stage == "zzz_probe"
    assert "zzz_probe" in rp.stage_names()
    assert "zzz_probe" in rp.model_dependent_stages()
    assert "zzz_probe" in rp.stage_order()


def test_duplicate_registration_raises(clean_registry) -> None:
    with pytest.raises(ValueError, match="already registered"):
        rp.register_stage(rp.StageSpec("eval", 99, False, True, lambda ctx: 0))


def test_registry_stays_sorted_by_order(clean_registry) -> None:
    rp.register_stage(rp.StageSpec("zzz_early", -5, False, False, lambda ctx: 0, cli=False))
    assert rp.stage_order()[0] == "zzz_early"


# ---------------------------------------------------------------------------
# fatal / non-fatal dispatch
# ---------------------------------------------------------------------------
def _ctx() -> rp.StageContext:
    return rp.StageContext(config={"experiment": {"tag": "t"}}, benchmark="G1_raw")


def test_non_fatal_stage_swallows_its_exception(capsys) -> None:
    def _boom(ctx):
        raise RuntimeError("the CMI fit diverged")

    spec = rp.StageSpec("probe", 99, False, True, _boom, fatal=False)
    assert rp._dispatch_stage(spec, _ctx()) == 0
    err = capsys.readouterr().err
    assert "failed (non-fatal)" in err
    assert "the CMI fit diverged" in err


def test_fatal_stage_propagates_its_exception() -> None:
    def _boom(ctx):
        raise RuntimeError("eval could not load the checkpoint")

    spec = rp.StageSpec("probe", 99, False, True, _boom, fatal=True)
    with pytest.raises(RuntimeError, match="could not load"):
        rp._dispatch_stage(spec, _ctx())


def test_dispatch_returns_the_stage_exit_code() -> None:
    spec = rp.StageSpec("probe", 99, False, True, lambda ctx: 7)
    assert rp._dispatch_stage(spec, _ctx()) == 7


def test_builtin_stages_are_all_fatal() -> None:
    """Today's behaviour: ``eval`` raising must still fail the run. Only analyses are lenient.

    Scoped to the builtins: the Sprint 5-7 analysis plugins register ``fatal=False`` by design,
    so that a diverging CMI fit or a failed calibration collection cannot abort a headline run.
    """
    for spec in rp._BUILTIN_STAGE_SPECS:
        assert spec.fatal is True, f"{spec.name} unexpectedly non-fatal"


def test_analysis_plugins_are_non_fatal() -> None:
    """The converse guard: a registered analysis stage must never take the run down."""
    rp._load_stage_plugins()
    for name in ("calibration", "lag_intervention", "cmi"):
        spec = rp._STAGE_REGISTRY.get(name)
        if spec is None:  # not shipped yet
            continue
        assert spec.fatal is False, f"{name} must be non-fatal"
        assert spec.model_dependent is True, f"{name} must be arm-scoped"


def test_arms_report_is_the_one_model_free_plugin() -> None:
    """``arms_report`` reads every arm and writes once at the tag root.

    Being ``model_dependent=False`` is what lets ``--stage arms_report`` run without ``--arm``.
    It also means the generic per-arm plugin loop skips it, so ``run_pipeline`` dispatches it in
    its own post-sweep block -- the two facts are the same fact, and this pins it.
    """
    rp._load_stage_plugins()
    spec = rp._STAGE_REGISTRY.get("arms_report")
    if spec is None:
        pytest.skip("arms_report_v3 not shipped")
    assert spec.fatal is False
    assert spec.model_dependent is False
    assert "arms_report" not in rp.model_dependent_stages()
    # It must sort after every per-arm analysis stage, so the artifacts it reads already exist.
    for name in ("eval", "report", "calibration", "lag_intervention", "cmi"):
        other = rp._STAGE_REGISTRY.get(name)
        if other is not None:
            assert spec.order > other.order, f"arms_report must run after {name}"


def test_plugin_stages_are_dispatchable_when_run_as_a_script() -> None:
    """``python run_pipeline_v2.py --stage calibration`` must not be an "invalid choice".

    Run as a script the driver is bound as ``__main__``, while the plugins reach
    ``register_stage`` through the dotted module name. Without the ``sys.modules`` alias at the
    top of ``run_pipeline_v2``, that dotted import creates a *second* module object with its own
    empty ``_STAGE_REGISTRY``; the plugins register into a registry nobody reads and ``--stage``
    silently drops them. This exercises the real ``--help`` path in a subprocess.
    """
    import subprocess
    import sys as _sys

    out = subprocess.run(
        [_sys.executable, str(Path(rp.__file__).resolve()), "--help"],
        capture_output=True, text=True, timeout=300, cwd=str(_REPO_ROOT),
    )
    assert out.returncode == 0, out.stderr
    assert "calibration" in out.stdout
    assert "lag_intervention" in out.stdout
    assert "cmi" in out.stdout
    assert "arms_report" in out.stdout


# ---------------------------------------------------------------------------
# StageContext
# ---------------------------------------------------------------------------
def test_stage_context_resolves_arm_scoped_and_tag_root_dirs(tmp_path) -> None:
    config = {"experiment": {"tag": "G1_raw_v3"}, "paths": {"results_dir": str(tmp_path)}}
    arm_ctx = rp.StageContext(config=config, benchmark="G1_raw", arm="v3_prod")
    bare_ctx = rp.StageContext(config=config, benchmark="G1_raw", arm=None)
    assert arm_ctx.run_dir().name == "v3_prod"
    assert arm_ctx.run_dir().parent == arm_ctx.tag_root()
    assert bare_ctx.run_dir() == bare_ctx.tag_root()


# ---------------------------------------------------------------------------
# Plugin loading
# ---------------------------------------------------------------------------
def test_missing_plugin_modules_are_not_an_error(monkeypatch, capsys) -> None:
    """Sprints 5-7 are not shipped yet; their absence must be silent, not a warning storm."""
    monkeypatch.setattr(rp, "_PLUGINS_LOADED", False)
    monkeypatch.setattr(rp, "_STAGE_PLUGIN_MODULES", ("definitely_not_a_module",))
    rp._load_stage_plugins()
    assert "definitely_not_a_module" not in capsys.readouterr().err


def test_a_broken_plugin_warns_but_does_not_gate(monkeypatch, capsys) -> None:
    import importlib

    def _explode(name):
        raise ValueError("plugin is broken")

    monkeypatch.setattr(rp, "_PLUGINS_LOADED", False)
    monkeypatch.setattr(rp, "_STAGE_PLUGIN_MODULES", ("broken_v3",))
    monkeypatch.setattr(importlib, "import_module", _explode)
    rp._load_stage_plugins()   # must not raise
    err = capsys.readouterr().err
    assert "broken_v3" in err and "failed to import" in err


# ---------------------------------------------------------------------------
# Report-section registry
# ---------------------------------------------------------------------------
def _section_ctx(tmp_path) -> fr.SectionContext:
    return fr.SectionContext(config={}, benchmark="G1_raw", results_dir=tmp_path,
                             metrics={"run_tag": "unit"})


def test_registered_sections_render_in_order(clean_sections, tmp_path) -> None:
    fr.register_section(fr.SectionSpec("second", 20, lambda c: ["## Second", ""]))
    fr.register_section(fr.SectionSpec("first", 10, lambda c: ["## First", ""]))
    text = "\n".join(fr._render_registered_sections(_section_ctx(tmp_path)))
    assert text.index("## First") < text.index("## Second")


def test_a_raising_section_renders_n_a_and_the_rest_survive(clean_sections, tmp_path) -> None:
    def _boom(ctx):
        raise KeyError("calibration_predictive")

    fr.register_section(fr.SectionSpec("bad", 10, _boom))
    fr.register_section(fr.SectionSpec("good", 20, lambda c: ["## Good", ""]))
    text = "\n".join(fr._render_registered_sections(_section_ctx(tmp_path)))
    assert "## bad" in text and "n/a" in text and "KeyError" in text
    assert "## Good" in text


def test_duplicate_section_registration_raises(clean_sections, tmp_path) -> None:
    fr.register_section(fr.SectionSpec("dup", 10, lambda c: []))
    with pytest.raises(ValueError, match="already registered"):
        fr.register_section(fr.SectionSpec("dup", 11, lambda c: []))


def test_no_registered_sections_is_an_empty_render(clean_sections, tmp_path) -> None:
    fr._SECTION_REGISTRY.clear()
    assert fr._render_registered_sections(_section_ctx(tmp_path)) == []
