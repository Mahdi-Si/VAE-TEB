r"""The collapse criterion's properties its own arithmetic cannot check, and the import it costs.

:mod:`teb_vae.lag_attn_rws.collapse` is **imported rather than forked**. It is stdlib-only -- its
only imports are ``__future__`` and ``typing`` -- so importing it keeps the acceptance gate free of
``torch``, and the criterion itself is model-free arithmetic over a per-epoch series both cfs cells
and both raw cells already log under the same two names. The layering walk forbids
``teb_vae.lag_attn_rws.eval``, not the model package around it, so this import is inside the rule
rather than an exemption to it -- and a second copy of a threshold two packages must agree on is
the drift the fork's whole anti-drift apparatus exists to prevent.

That module restates ``KL_COLLAPSE_MIN_ACTIVE_DIMS * KLD_ACTIVE_EPS`` as a literal instead of
importing the epsilon, precisely so the gate can apply the criterion on a box with no ``torch``
installed. That trade is only safe if both halves of it are pinned here: the literal must equal the
product it stands for, and the module -- and the gate that imports it -- must actually be free of
``torch``.

The third property is the one a reader of the arithmetic would have to reconstruct: the criterion
reads the **tail** of a run, never its best window. The KL starts at exactly $0$ by construction
(the zero-initialised posterior residual) and the $\beta$ warm-up holds it there deliberately, so
an any-window reading would classify every healthy run as collapsed and a best-window reading would
classify every collapsed one as healthy.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from teb_vae.lag_attn_cfs.eval import verify
from teb_vae.lag_attn_rws.collapse import (
    KL_COLLAPSE_MIN_ACTIVE_DIMS,
    KL_COLLAPSE_PATIENCE_EPOCHS,
    KL_COLLAPSE_THRESHOLD_NATS,
    is_collapsed,
)
from teb_vae.lag_attn_rws.nets.losses import KLD_ACTIVE_EPS

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: A latent width the second clause is applied against. Only the ratio matters: clause 2 fires
#: below ``KL_COLLAPSE_MIN_ACTIVE_DIMS / d_z``, so at 64 the floor is $2/64 = 0.03125$.
_D_Z = 64


# =================================================================================================
# The restated constants
# =================================================================================================
def test_the_literal_threshold_equals_the_product_it_stands_for() -> None:
    """The invariant that replaces the import. A change to the epsilon fails here."""
    assert KL_COLLAPSE_THRESHOLD_NATS == KL_COLLAPSE_MIN_ACTIVE_DIMS * KLD_ACTIVE_EPS
    assert KL_COLLAPSE_THRESHOLD_NATS == 0.02


def test_the_gate_reads_the_criterion_from_that_module_rather_than_owning_a_copy() -> None:
    """Imported, not forked: two packages applying two copies of one threshold is exactly how the
    same run comes to be collapsed in one table and healthy in another."""
    assert verify.is_collapsed is is_collapsed


# =================================================================================================
# The tail window
# =================================================================================================
def test_clause_one_reads_the_end_of_the_series_and_not_its_best_point() -> None:
    """A run that was healthy in the middle and died is collapsed; a run that opened at zero -- as
    every healthy run does, by construction -- and recovered is not. An any-window or best-window
    reading would get both backwards."""
    healthy_open = [0.0, 0.0, 0.0, 0.5, 2.0, 4.0, 5.0, 5.4]
    died_at_the_end = [0.0, 0.5, 4.0, 0.01, 0.008, 0.005, 0.003, 0.001]
    active = [0.6] * 8

    assert is_collapsed(healthy_open, active, _D_Z) is False
    assert is_collapsed(died_at_the_end, active, _D_Z) is True


def test_a_run_shorter_than_the_patience_window_cannot_fire_clause_one() -> None:
    """Its tail would be the whole series, which always includes the deliberate zero-KL opening."""
    short = [0.0] * (KL_COLLAPSE_PATIENCE_EPOCHS - 1)

    assert is_collapsed(short, [0.6] * len(short), _D_Z) is False


def test_clause_two_fires_on_the_final_active_fraction_alone() -> None:
    """The two clauses are one statement at the same threshold, and either suffices: a latent
    carrying its nats in one dimension is collapsed whatever the total KL reads."""
    healthy_kl = [5.0] * 8

    assert is_collapsed(healthy_kl, [0.6] * 7 + [1.0 / _D_Z], _D_Z) is True
    assert is_collapsed(healthy_kl, [0.6] * 7 + [3.0 / _D_Z], _D_Z) is False


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
