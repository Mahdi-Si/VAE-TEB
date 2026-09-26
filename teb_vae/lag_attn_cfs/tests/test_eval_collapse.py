r"""The collapse criterion as this cell's acceptance gate applies it, and the import it costs.

The criterion itself is :mod:`teb_vae.lag_attn_rws.collapse`, imported rather than forked and
tested where it is defined. Two properties belong to this cell's gate:

* a run whose metrics CSV carries no ``val/kld_active_frac`` column is **unknown** rather than
  healthy, because the second clause cannot be answered from it;
* importing the gate pulls in no numeric stack, so a finished run's ``summary.json`` can be checked
  on a box with no ``torch`` installed.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from teb_vae.lag_attn_cfs.eval import verify

_REPO_ROOT = Path(__file__).resolve().parents[3]


# =================================================================================================
# What the criterion cannot answer, and must not pretend to
# =================================================================================================
def test_an_absent_active_fraction_series_is_unknown_rather_than_not_collapsed(tmp_path) -> None:
    """Clause 2 needs the final active fraction, so a run whose CSV carries only the KL column can
    be answered with clause 1 alone -- and this cell's arm collector refuses to render a
    one-clause answer as a verdict. The alternative is a ``no`` in the cell an operator scans a
    sweep table down, on evidence the run never provided."""
    from .test_eval_verify import write_arm

    write_arm(
        tmp_path, "no_active",
        csv_columns=[verify.EPOCH_COLUMN, verify.KL_SERIES_COLUMN],
    )
    write_arm(tmp_path, "complete")

    arms = {
        arm["run"].split("/")[0]: arm
        for arm in (
            verify.collect_arm(path, tmp_path)
            for path in sorted(tmp_path.rglob(verify.SUMMARY_FILENAME))
        )
    }

    assert arms["no_active"]["collapsed"] is None
    assert any(verify.ACTIVE_FRAC_COLUMN in note for note in arms["no_active"]["incomplete"])
    # Non-vacuity: the same shapes with both series present do produce a verdict.
    assert arms["complete"]["collapsed"] is False


# =================================================================================================
# The property the import exists to preserve
# =================================================================================================
def test_importing_the_gate_pulls_in_no_numeric_stack() -> None:
    """Run in a subprocess: this session has already imported ``torch``, so an in-process check
    would pass no matter what the modules do.

    The point is the gate. It reads a finished run's ``summary.json``, applies this arithmetic, and
    must do so on a machine that has never had a deep-learning stack on it -- which is what makes a
    summary produced on the production box checkable anywhere the file can be copied. Asserted on
    ``verify`` rather than on ``collapse`` alone, because it is the *composition* that has to hold:
    the criterion staying stdlib-only buys nothing if the module importing it does not.
    """
    source = (
        "import sys\n"
        "import teb_vae.lag_attn_cfs.eval.verify as gate\n"
        "leaked = sorted(name for name in sys.modules if name.split('.')[0] "
        "in {'torch', 'lightning', 'numpy', 'scipy', 'h5py', 'pandas', 'matplotlib'})\n"
        "assert gate.is_collapsed is not None\n"
        "print(','.join(leaked))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "", (
        f"importing the acceptance gate pulled in {completed.stdout.strip()}; it applies this "
        f"criterion on a box with none of those installed"
    )
