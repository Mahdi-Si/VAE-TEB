r"""The dependency on the causal-feature sibling runs one way.

This package reaches into ``teb_vae/lag_attn_cfs`` -- its input mixin, its task members and its test
fixtures are bound by reference or imported by name -- and the sibling must never import back. A
sibling module importing this package would make the two one package: a change on either side could
then break the other, and the sibling is scored through by every causal cell in the family.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: The sibling this package reaches into.
_SIBLING = "teb_vae/lag_attn_cfs"

#: This package's import path, as a sibling module importing it would spell it.
_THIS_PACKAGE = "lag_attn_crws"


def test_the_causal_sibling_never_imports_this_package():
    """A property of the code as it stands, checked on every non-test module of the sibling.

    ``tests/`` is excluded on the sibling's side: its import guard has to enumerate every package in
    the family by name, and a test that reads its own subject as a violation is a test nobody keeps.
    """
    importers = [
        str(path.relative_to(_REPO_ROOT)).replace("\\", "/")
        for path in sorted((_REPO_ROOT / _SIBLING).rglob("*.py"))
        if "tests" not in path.parts
        and re.search(rf"^\s*(from|import)\s+.*{_THIS_PACKAGE}", path.read_text(encoding="utf-8"),
                      re.MULTILINE)
    ]

    assert importers == [], importers
