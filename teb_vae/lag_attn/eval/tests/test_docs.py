r"""``EVAL.md`` and ``FIGURE_GUIDE.md`` cover what the pipeline actually produces.

Both halves are deliberately driven off the *code and the run*, never off a hand-maintained list
in the test, because a hand-maintained list is a second thing to forget to update and would make
the test pass by construction.

* The analysis coverage comes from globbing ``analyses/*.py``, and matches against **section
  headings** whose slug is the module name -- not a bare substring, which any sentence mentioning
  the word would satisfy.
* The config coverage comes from the **resolved** ``eval_config``, so a key added to the schema
  and to the shipped YAML is covered whether or not anyone remembered this test.
* The figure coverage comes from the artifact manifest a real smoke run writes, which is the
  reason the manifest exists at all.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from teb_vae.lag_attn.eval import run as run_module
from teb_vae.lag_attn.eval.tests.conftest import EVAL_TINY_CONFIG

#: The two documents under test, relative to the package directory.
EVAL_DOC = "EVAL.md"
FIGURE_DOC = "FIGURE_GUIDE.md"


@pytest.fixture(scope="module")
def package_dir() -> Path:
    """The ``eval`` package directory, which is where both documents live."""
    return Path(run_module.__file__).resolve().parent


@pytest.fixture(scope="module")
def eval_doc(package_dir) -> str:
    return (package_dir / EVAL_DOC).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def figure_doc(package_dir) -> str:
    return (package_dir / FIGURE_DOC).read_text(encoding="utf-8")


def _headings(document: str) -> set:
    """Return every markdown heading in ``document``, lowercased and stripped of markup.

    Backticks and asterisks only. Stripping ``_`` as emphasis would turn the heading ``te_lag``
    into ``telag`` and quietly fail to match the module it documents -- snake_case identifiers
    and markdown emphasis share a character.
    """
    found = set()
    for line in document.splitlines():
        match = re.match(r"^#{1,6}\s+(.*?)\s*$", line)
        if match:
            found.add(re.sub(r"[`*]", "", match.group(1)).strip().lower())
    return found


def _analysis_modules(package_dir: Path) -> list:
    """Every analysis module name, from the filesystem rather than from a literal list."""
    return sorted(
        path.stem
        for path in (package_dir / "analyses").glob("*.py")
        if path.stem != "__init__"
    )


# ---------------------------------------------------------------------------
# EVAL.md
# ---------------------------------------------------------------------------
def test_the_documents_exist(package_dir):
    assert (package_dir / EVAL_DOC).is_file()
    assert (package_dir / FIGURE_DOC).is_file()


def test_every_analysis_module_has_its_own_section_heading(eval_doc, package_dir):
    """A heading whose slug is the module name -- prose mentioning the word would not do."""
    headings = _headings(eval_doc)
    modules = _analysis_modules(package_dir)
    assert modules, "no analysis modules were found, so this test proves nothing"

    missing = [name for name in modules if name not in headings]
    assert not missing, (
        f"{EVAL_DOC} has no section heading for {missing}. Every analysis needs one: a reader "
        f"who finds an unfamiliar directory in a run has nowhere else to look."
    )


def test_every_registered_analysis_is_documented(eval_doc):
    """The registry and the filesystem can disagree; both must be covered."""
    headings = _headings(eval_doc)
    missing = [name for name in run_module.ANALYSES if name not in headings]
    assert not missing, f"{EVAL_DOC} does not document the registered analyses {missing}"


def test_every_resolved_eval_config_key_is_documented(eval_doc, repo_root, monkeypatch):
    """Driven off the resolved config, so a new key is covered without touching this test."""
    from teb_vae.lag_attn.config import load_config
    from teb_vae.lag_attn.eval.config_schema import validate_eval_config

    monkeypatch.chdir(repo_root)
    resolved = validate_eval_config(load_config(str(repo_root / EVAL_TINY_CONFIG)))

    missing = [key for key in resolved if f"`{key}`" not in eval_doc]
    assert not missing, (
        f"{EVAL_DOC} does not document the eval_config key(s) {missing}. An undocumented key is "
        f"one an operator has to read the validator to understand."
    )


def test_the_k_fold_repoint_and_its_consequence_are_stated(eval_doc):
    """A semantic change to what 'test' means, which a reader must not have to infer."""
    lowered = eval_doc.lower()
    assert "k_fold_cross_validation_dataset" in lowered
    assert "semantic change" in lowered
    assert "pretraining" in lowered


def test_the_masked_versus_raw_saturation_discrepancy_is_documented(eval_doc):
    """The two readings routinely disagree, and the disagreement is informative, not a defect."""
    assert "_raw" in eval_doc and "_masked" in eval_doc
    assert "saturation_flag_threshold" in eval_doc


def test_the_health_probe_floor_is_marked_as_uncalibrated(eval_doc):
    """It was chosen a priori; a reader must not take it for a measured threshold."""
    lowered = eval_doc.lower()
    assert "a priori" in lowered
    assert "recalibrate" in lowered, "the procedure for replacing it must be stated"


def test_the_band_partition_is_documented(eval_doc):
    assert "band_partition.json" in eval_doc and "band_channel_map.csv" in eval_doc
    # The absent ph_diag kind is expected, not a defect, and that has to be written down.
    assert "ph_diag" in eval_doc


# ---------------------------------------------------------------------------
# FIGURE_GUIDE.md
# ---------------------------------------------------------------------------
#: The seven traps, each keyed by a phrase that cannot appear by accident.
TRAP_MARKERS = (
    # A shuffled source moves the posterior *more*, so the KL-space reading looks like failure.
    "K_{\\mathrm{shuffled}}",
    # The seconds axis is provisional.
    "up_shift_secs",
    # An attribution needs the head structure.
    "head_structured_latent",
    # A transfer entropy needs the causal normalisation.
    "causal_norm",
    # The final H_d anchors collapse toward the prior by construction.
    "kld_support='anchor'",
    # W_o is frozen, so the projected combination is not on the latent's path.
    "attended_source",
    # reduce_mean divides by d_z.
    "measure_transfer_entropy",
)


@pytest.mark.parametrize("marker", TRAP_MARKERS)
def test_every_trap_is_stated_explicitly(figure_doc, marker):
    assert marker in figure_doc, f"{FIGURE_DOC} does not state the trap keyed by {marker!r}"


def test_the_traps_lead_the_document(figure_doc):
    """A trap stated in an appendix is one the reader meets after drawing the wrong conclusion."""
    head = figure_doc[: figure_doc.find("## 2.")]
    assert head, "the guide has no section 2, so the trap section cannot be first"
    for marker in TRAP_MARKERS:
        assert marker in head, f"{marker!r} is stated after the first figure section"


def test_the_prediction_space_ordering_is_named_as_the_criterion(figure_doc):
    assert "L_{\\mathrm{feat}} < L_{\\mathrm{base}} < L_{\\mathrm{feat,\\ shuffled}}" in figure_doc


def test_every_figure_the_smoke_run_emitted_is_documented(
    tiny_checkpoint, tmp_path, monkeypatch, repo_root, figure_doc
):
    """Driven off the artifact manifest a real run writes -- this is why the manifest exists.

    A hardcoded filename list would pass by construction and would stop covering the moment an
    analysis gained a figure.
    """
    monkeypatch.chdir(repo_root)
    output_dir = tmp_path / "run"
    run_module.main(
        config=str(repo_root / EVAL_TINY_CONFIG),
        checkpoint=str(tiny_checkpoint),
        output_dir=str(output_dir),
        device="cpu",
    )
    summary = json.loads(
        (output_dir / run_module.RESULTS_DIRNAME / "summary.json").read_text(encoding="utf-8")
    )
    emitted = summary["results"]["artifacts"]["figures"]
    assert emitted, "the smoke run emitted no figures, so this test proves nothing"

    missing = []
    for path in emitted:
        name = Path(path).name
        # The per-sample pages are named per recording, so the guide documents their *pattern*.
        stem = "sample<index>_<guid>_epoch<epoch>.pdf" if name.startswith("sample") else name
        if stem not in figure_doc:
            missing.append(path)
    assert not missing, (
        f"{FIGURE_DOC} has no entry for {missing}. Every emitted figure needs one: an undocumented "
        f"figure is one a reader has to reverse-engineer from the code that drew it."
    )


def test_each_analysis_directory_appears_in_the_figure_guide(figure_doc):
    """A reader who finds a directory in a run must be able to look it up by name."""
    # scalars/ emits no figures by design, so it is the one analysis with nothing to document
    # here; its CSV-only output is covered by EVAL.md instead.
    expected = [name for name in run_module.ANALYSES if name != "scalars"]
    missing = [name for name in expected if f"{name}/" not in figure_doc]
    assert not missing, f"{FIGURE_DOC} does not name the output directories {missing}"
