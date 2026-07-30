r"""The collapse criterion's two properties that its own arithmetic cannot check.

:mod:`teb_vae.lag_attn_rws.collapse` restates ``KL_COLLAPSE_MIN_ACTIVE_DIMS * KLD_ACTIVE_EPS``
as a literal instead of importing the epsilon, so that the offline acceptance gate can apply the
same criterion on a box with no ``torch`` installed. That trade is only safe if both halves of it
are pinned: the literal must equal the product it stands for, and the module must actually be
free of ``torch``.

The behavioural tests live beside the sweep-config lint, which is the consumer that drove the
definition.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from teb_vae.lag_attn_rws.collapse import (
    KL_COLLAPSE_MIN_ACTIVE_DIMS,
    KL_COLLAPSE_THRESHOLD_NATS,
)
from teb_vae.lag_attn_rws.nets.losses import KLD_ACTIVE_EPS

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_the_literal_threshold_equals_the_product_it_stands_for() -> None:
    """The invariant that replaces the import. A change to the epsilon fails here."""
    assert KL_COLLAPSE_THRESHOLD_NATS == KL_COLLAPSE_MIN_ACTIVE_DIMS * KLD_ACTIVE_EPS
    assert KL_COLLAPSE_THRESHOLD_NATS == 0.02


def test_importing_the_criterion_pulls_in_no_torch() -> None:
    """Run in a subprocess: this session has already imported ``torch``, so an in-process check
    would pass no matter what the module does.

    The point is the offline gate. It reads a finished run's ``summary.json`` and applies this
    arithmetic, and it must do so on a machine that has never had a deep-learning stack on it.
    """
    source = (
        "import sys\n"
        "import teb_vae.lag_attn_rws.collapse as collapse\n"
        "leaked = sorted(name for name in sys.modules if name.split('.')[0] "
        "in {'torch', 'lightning', 'numpy', 'scipy', 'h5py'})\n"
        "assert collapse.KL_COLLAPSE_THRESHOLD_NATS == 0.02\n"
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
        f"importing collapse.py pulled in {completed.stdout.strip()}; the offline gate applies "
        f"this criterion on a box with none of those installed"
    )
