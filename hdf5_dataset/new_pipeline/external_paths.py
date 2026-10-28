r"""Put the two packages that live outside this repository on ``sys.path``.

The dataset build imports ``early_maestra`` (the record adaptor) and, through it, ``mimo`` (the
sequence code). Neither is in this repository. On a box they sit in their own folders, for
example::

    <somewhere>/TrainWithDataset/EarlyMaestra/early_maestra/__init__.py
    <somewhere>/TrainWithDataset/MIMO_Sequence_Trainer/mimo/__init__.py

The folder that goes on ``sys.path`` is the one that *holds* each package: ``EarlyMaestra`` and
``MIMO_Sequence_Trainer``. :func:`add_external_roots` finds them in this order:

1. the environment variable ``TEB_EXTERNAL_ROOTS`` (folders separated by ``os.pathsep``);
2. :data:`EXTERNAL_ROOTS` below, which you edit for the box;
3. the folders of :data:`CONTAINERS`, each of which may hold ``EarlyMaestra/`` and
   ``MIMO_Sequence_Trainer/``.

A package that the first source provides is not searched for in the later ones. The roots go
into ``sys.path`` just after the repository root, so they win over an older copy elsewhere on
the path and never shadow a module of this repository. Worker processes
inherit them: ``spawn`` copies the parent's ``sys.path``, and ``fork`` copies the whole process.
The function only reads the file system; it imports nothing, so a test that stubs
``early_maestra`` in ``sys.modules`` is not affected.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

logger = logging.getLogger(__name__)

# ================================ FILL IN: the box's folders ================================
#: Folders that HOLD ``early_maestra/`` or ``mimo/``, e.g.
#: ``r"/home/<user>/tmp/TrainWithDataset/EarlyMaestra"`` and
#: ``r"/home/<user>/tmp/TrainWithDataset/MIMO_Sequence_Trainer"``. Empty: use the search below.
EXTERNAL_ROOTS: List[str] = []

#: The environment variable that overrides :data:`EXTERNAL_ROOTS`.
ENV_VAR = "TEB_EXTERNAL_ROOTS"

#: This repository: ``hdf5_dataset/new_pipeline/external_paths.py`` -> up three.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Package name -> the folder name that usually holds it.
PACKAGES: Dict[str, str] = {"early_maestra": "EarlyMaestra", "mimo": "MIMO_Sequence_Trainer"}

#: Folders searched for ``EarlyMaestra/`` and ``MIMO_Sequence_Trainer/`` when the two sources
#: above do not give a package: a local copy in the repository, the repository's ``tmp/``, the
#: folder beside the repository, and the home ``tmp/``.
CONTAINERS: List[Path] = [
    REPO_ROOT / "mimo",
    REPO_ROOT / "tmp" / "TrainWithDataset",
    REPO_ROOT.parent / "TrainWithDataset",
    REPO_ROOT.parent,
    Path.home() / "tmp" / "TrainWithDataset",
]


def _holds(root: Path, package: str) -> bool:
    """Whether ``root`` holds ``package`` as a regular package."""
    return (root / package / "__init__.py").is_file()


def _explicit_roots() -> List[Path]:
    """The roots from :data:`ENV_VAR` if it is set, else from :data:`EXTERNAL_ROOTS`."""
    value = os.environ.get(ENV_VAR, "").strip()
    entries: Iterable[str] = value.split(os.pathsep) if value else EXTERNAL_ROOTS
    return [Path(entry).expanduser() for entry in entries if str(entry).strip()]


def find_external_roots() -> Dict[str, Path]:
    """Return the root that holds each package of :data:`PACKAGES`, where one is found.

    Returns:
        Package name -> the folder to put on ``sys.path``. A package that no source holds is
        missing from the result.
    """
    found: Dict[str, Path] = {}
    explicit = _explicit_roots()
    for root in explicit:
        if not root.is_dir():
            logger.warning(f"external root {root} does not exist")
            continue
        held = [package for package in PACKAGES if _holds(root, package)]
        if not held:
            logger.warning(f"external root {root} holds none of {sorted(PACKAGES)}")
        for package in held:
            found.setdefault(package, root)
    for package, folder in PACKAGES.items():
        if package in found:
            continue
        for container in CONTAINERS:
            root = container / folder
            if _holds(root, package):
                found[package] = root
                break
    return found


def add_external_roots() -> Dict[str, Path]:
    """Put the root of each external package on ``sys.path``, once, just after the repository.

    Returns:
        Package name -> the root used, as :func:`find_external_roots` returns it.
    """
    found = find_external_roots()
    repo = str(REPO_ROOT)
    for package, root in found.items():
        entry = str(root.resolve())
        if entry in sys.path:
            continue
        position = sys.path.index(repo) + 1 if repo in sys.path else 0
        sys.path.insert(position, entry)
        logger.debug(f"{package}: added {entry} to sys.path")
    return found
