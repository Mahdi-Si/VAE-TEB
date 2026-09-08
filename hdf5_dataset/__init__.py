"""Dataset construction, and the pin that decides which kymatio builds it.

The scattering coefficients in the shards were produced by the **kymatio checkout committed in this
repository** (``<repo>/kymatio``), not by a release from PyPI. The two are not interchangeable: the
filter bank is what every stored coefficient is a projection onto, so a different build would change
the data under the same file names, with no error anywhere to say so.

Importing this package puts that checkout at the front of ``sys.path``, which is what makes the
guarantee hold rather than depend on the machine:

* ``<repo>/kymatio`` is the checkout and ``<repo>/kymatio/kymatio`` the package inside it, so the
  *checkout* is the path entry -- pointing at the repository root instead leaves the bare checkout
  directory acting as a namespace portion for the name ``kymatio``, which loses to any regular
  package found further along ``sys.path`` and takes ``kymatio.__version__`` with it.
* Front, because that is what beats an official ``pip install kymatio`` sitting in site-packages.

Only the path is arranged here. Nothing is imported: :mod:`hdf5_dataset.causal_scattering` checks
that the module it actually got is the vendored one, which is the check that still fires when
something imported kymatio before this package was reached.
"""
from __future__ import annotations

import os
import sys

#: The kymatio checkout committed in this repository -- the directory holding ``setup.py``, whose
#: ``kymatio`` subdirectory is the importable package.
VENDORED_KYMATIO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kymatio"
)

if VENDORED_KYMATIO in sys.path:
    sys.path.remove(VENDORED_KYMATIO)
sys.path.insert(0, VENDORED_KYMATIO)
