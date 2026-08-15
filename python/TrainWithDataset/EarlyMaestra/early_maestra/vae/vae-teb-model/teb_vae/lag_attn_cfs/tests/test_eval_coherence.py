r"""``coherence`` and ``spectra`` are not ported at all -- asserted as an absence.

The sibling suite's file of this name tests a cross-spectral estimator over $1{,}225$ lines. This
one tests that no such estimator exists here, and the inversion is the point: the single largest
risk in this fork is a raw-domain assumption surviving the copy, and carrying those two modules for
line comparability would have imported $2{,}374$ lines of them deliberately -- a $4\,$Hz rate, a
$480$-sample block, ``nperseg = 512`` and the fetal-HRV band edges -- into a feature-domain
evaluator where a later reasonable edit could reach them.

**What is absent is everything a modulus cannot support**: phase agreement, group delay, and the
residual spectrum's exact three-way split into irreducible, timing and amplitude terms. A stored
scattering coefficient is $|x \star \psi_\lambda|$; the analysing filter's phase was discarded
before the value was written, so none of the three has an analogue here at any window length.

**What replaces the half that does exist** is ``spectral_skill``, on the frequency axis this target
domain already carries in its channels. It is deliberately *not* called ``coherence``, so a reader
who knows the raw pipeline cannot carry the wrong contract across, and the tests below assert both
halves of that: the name is gone, and the thing that answers the neighbouring question is present
under its own name.

Four independent statements, because each would fail on a different mistake:

* the two files do not exist -- the assertion that fires the day somebody reaches for a copy;
* no shipped module imports either of them, or any spectral symbol, in any form the AST walk of
  ``test_eval_self_contained.py`` resolves;
* ``coherence`` is not in the registry and ``--only coherence`` refuses, naming what *is*
  available -- so the failure tells an operator what to type instead of merely stopping;
* the sanity block emits neither cross-spectral check, since two checks that could only ever be
  INCONCLUSIVE would read as an analysis that failed rather than one that does not exist.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from teb_vae.lag_attn_cfs.eval import report_seam
from teb_vae.lag_attn_cfs.eval import run as run_module

from .test_eval_self_contained import EVAL_ROOT, _module_name_for, imported_names

#: The two modules of the source package that do not enter this one, as paths relative to
#: ``eval/``. Written out rather than derived: the point of this file is that these two names in
#: particular are gone, and a derived list would go quiet the day one of them came back under
#: another spelling.
NOT_PORTED = ("analyses/coherence.py", "spectra.py")

#: The analysis that answers the neighbouring question, and must be present for the absence above
#: to be a replacement rather than a deletion.
REPLACEMENT = "spectral_skill"

#: Import names no shipped module may reach for, matched against an import's **final component**
#: rather than as substrings. ``scipy.signal``'s ``welch``, ``csd`` and ``coherence`` are what the
#: raw estimator is built on, so a reach for the construction without reaching for the module is
#: still reported -- while ``analyses.spectral_skill``, whose name contains none of these as a
#: component, is not swept up by the readout that replaces them.
FORBIDDEN_IMPORT_NAMES = frozenset(
    {"coherence", "spectra", "spectral", "welch", "csd", "periodogram"}
)

#: The sanity checks the sibling's ``build_sanity`` emits and this one must not.
FORBIDDEN_SANITY_CHECKS = ("coherence_parseval", "coherence_detrended_share")


def _shipped_modules() -> List[Path]:
    """Every ``.py`` that ships under ``eval/``."""
    return sorted(EVAL_ROOT.rglob("*.py"))


# =================================================================================================
# The files are not there
# =================================================================================================
@pytest.mark.parametrize("relative", NOT_PORTED)
def test_the_module_is_not_in_this_package(relative: str) -> None:
    """The assertion that fails the day someone reaches for a copy of either module."""
    assert not (EVAL_ROOT / relative).exists(), (
        f"{relative} exists in this package. Neither module is ported: a stored coefficient is a "
        f"modulus, so phase agreement, group delay and the residual's three-way split have no "
        f"analogue here at any window length. The frequency-resolved question is answered by "
        f"{REPLACEMENT!r}, on the frequency axis the channels already carry."
    )


def test_the_replacement_is_present_so_the_absence_is_a_replacement() -> None:
    """Non-vacuity for every assertion above it. Two missing files would also satisfy a package
    that had simply dropped the frequency-resolved question, and that is a different -- and worse
    -- outcome than the one this fork decided on."""
    assert (EVAL_ROOT / "analyses" / f"{REPLACEMENT}.py").is_file()
    assert REPLACEMENT in run_module.merged_analysis_functions(run_module.CFS_BINDING)


# =================================================================================================
# Nothing imports them, in any form
# =================================================================================================
@pytest.mark.parametrize(
    "module", _shipped_modules(), ids=lambda path: str(path.relative_to(EVAL_ROOT))
)
def test_no_shipped_module_imports_a_spectral_estimator(module: Path) -> None:
    """Walked per module so a failure names the file. The walk resolves relative and lazy imports,
    which is what a name-based scan of the source text would miss -- and a lazy
    ``from scipy.signal import coherence`` inside the one function that wanted it is exactly the
    shape a later change reaches for."""
    names = imported_names(module.read_text(encoding="utf-8"), _module_name_for(module))
    offending = sorted(
        {
            name
            for name in names
            if name.rsplit(".", 1)[-1].lower() in FORBIDDEN_IMPORT_NAMES
        }
    )

    assert offending == [], f"{module.relative_to(EVAL_ROOT)} imports {offending}"


def test_the_import_scan_would_report_a_reach_for_the_estimator() -> None:
    """Non-vacuity for the sweep above, which passes trivially on a package that imports nothing.
    Both shapes: the module by name, and the SciPy primitive it is built on."""
    reaches = (
        "from teb_vae.lag_attn_cfs.eval import spectra\n",
        "def f():\n    from scipy.signal import coherence\n",
    )
    for source in reaches:
        names = imported_names(source, "teb_vae.lag_attn_cfs.eval.analyses.forecast")
        assert any(
            name.rsplit(".", 1)[-1].lower() in FORBIDDEN_IMPORT_NAMES for name in names
        ), source

    # And the replacement is not swept up by it, which is the one false positive that would make
    # the sweep unusable and get it loosened rather than obeyed.
    allowed = imported_names(
        f"from teb_vae.lag_attn_cfs.eval.analyses import {REPLACEMENT}\n",
        "teb_vae.lag_attn_cfs.eval.binding",
    )
    assert not any(
        name.rsplit(".", 1)[-1].lower() in FORBIDDEN_IMPORT_NAMES for name in allowed
    )


# =================================================================================================
# The registry refuses the name, and says what to type instead
# =================================================================================================
def test_coherence_is_not_registered_in_either_registry() -> None:
    """Neither selectable nor unskippable: a name that ran unconditionally would be worse than one
    that could be asked for."""
    merged = run_module.merged_analysis_functions(run_module.CFS_BINDING)

    assert "coherence" not in merged
    assert "coherence" not in run_module.UNSKIPPABLE_ANALYSES
    assert "spectra" not in merged


def test_only_coherence_refuses_and_names_the_valid_analyses() -> None:
    """The failure an operator who knows the raw pipeline will actually hit. It has to name the
    available analyses -- including the replacement -- rather than merely refusing, because
    otherwise the operator's next move is to read this source."""
    available = list(run_module.merged_analysis_functions(run_module.CFS_BINDING))

    with pytest.raises(ValueError) as raised:
        run_module.select_analyses(available, "coherence", None)

    message = str(raised.value)
    assert "coherence" in message
    assert REPLACEMENT in message


def test_a_valid_name_still_selects_so_the_refusal_is_not_blanket() -> None:
    """Non-vacuity: a ``select_analyses`` that raised on everything would pass the test above."""
    available = list(run_module.merged_analysis_functions(run_module.CFS_BINDING))

    assert run_module.select_analyses(available, REPLACEMENT, None) == [REPLACEMENT]


# =================================================================================================
# The sanity block carries neither cross-spectral check
# =================================================================================================
def _sanity_checks(results: Dict[str, Any]) -> Dict[str, Any]:
    """Return the named checks of the sanity block, off a minimal results dict."""
    block = report_seam.build_sanity(results, report_seam.build_headline(results))
    return dict(block["checks"])


def test_the_sanity_block_emits_neither_cross_spectral_check() -> None:
    """Both are properties of an estimator this package does not have. Emitting them as
    INCONCLUSIVE would read as two analyses that failed rather than two that do not exist -- and
    ``verify.py`` refuses on the sanity block, so the difference is a gate outcome."""
    checks = _sanity_checks({})

    for name in FORBIDDEN_SANITY_CHECKS:
        assert name not in checks
    assert not any("coherence" in str(name) for name in checks)


def test_the_sanity_block_still_carries_the_checks_this_package_does_have() -> None:
    """Non-vacuity: an empty sanity block would satisfy the assertion above and would also mean the
    gate had nothing to refuse on."""
    checks = _sanity_checks({})

    assert "kl_identity" in checks
    assert "per_anchor_recombines" in checks


# =================================================================================================
# The divergence manifest says the same thing, in the file that is machine-checked
# =================================================================================================
def test_the_manifest_records_both_as_absent_and_points_at_the_replacement() -> None:
    """Prose alone is not a control, and this is the file ``EVAL.md``'s divergence register is
    rendered from. Each reason has to name what answers the neighbouring question, or a reader of
    the register learns only that something is missing."""
    manifest = json.loads(
        (EVAL_ROOT / "divergences.json").read_text(encoding="utf-8")
    )["modules"]

    for relative in NOT_PORTED:
        entry = manifest[relative]
        assert entry["state"] == "absent"
        assert REPLACEMENT in entry["reason"]
        assert "modulus" in entry["reason"] or "phase" in entry["reason"]
