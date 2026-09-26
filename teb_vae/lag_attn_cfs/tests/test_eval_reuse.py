r"""The reuse seam binds the shared primitives rather than forking them, and this pins that.

This evaluation package is a **fork** of ``teb_vae/lag_attn_rws/eval``, so the one thing that
cannot be allowed to drift is the set of pieces both forks are supposed to share. The seam names
them in exactly one place; if a later edit adds a name here, drops one, or -- worst -- copies one
of these modules into this package, two summaries would go on reading as though they described the
same population while the definition of a cohort, an interval or a significance level had quietly
diverged.

The assertions are therefore about **identity**, not about values. ``CLASS_NAMES`` comparing equal
across the two packages proves nothing a copy would fail; ``is`` proves there is one object.
"""
from __future__ import annotations

from teb_vae.lag_attn_cfs.eval import _reuse
from teb_vae.lag_attn_rws.eval import _reuse as sibling_reuse


def test_the_bound_names_are_exactly_the_siblings() -> None:
    """A name added on one side and not the other is a primitive one fork owns and the other
    reimplements, which is the first step of the drift the fork's measures exist to prevent."""
    assert _reuse.__all__ == sibling_reuse.__all__


def test_every_bound_name_resolves_to_the_same_object_both_packages_see() -> None:
    """``is`` rather than ``==``: a copied module would compare equal on every constant below and
    would still be a second definition."""
    for name in _reuse.__all__:
        assert getattr(_reuse, name) is getattr(sibling_reuse, name), name
