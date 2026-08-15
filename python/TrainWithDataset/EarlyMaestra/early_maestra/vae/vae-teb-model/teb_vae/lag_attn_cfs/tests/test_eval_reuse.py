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


def test_the_cohort_definition_is_one_object_rather_than_two_equal_ones() -> None:
    """The named case, because it is the one whose divergence would be silent: two packages with
    two subgroup orderings emit two by-subgroup tables that look comparable and are not."""
    from teb_vae.lag_attn.eval import labels as shared_labels

    assert _reuse.labels is shared_labels
    assert _reuse.labels.CLASS_NAMES is shared_labels.CLASS_NAMES
    assert _reuse.labels.CANONICAL_SUBGROUPS is shared_labels.CANONICAL_SUBGROUPS
    assert len(_reuse.labels.CANONICAL_SUBGROUPS) == 8


def test_the_two_single_symbol_bindings_come_from_the_modules_beside_them() -> None:
    """``subsample_indices`` is the seeded stratified draw every cap goes through, and
    ``configure_numerics`` is the pinned numeric environment. Bound singly because nothing else in
    either module is used here -- and pinned to their own modules so a rebind to a local helper
    fails rather than passing."""
    assert _reuse.subsample_indices is _reuse.masks.subsample_indices
    assert _reuse.configure_numerics is _reuse.numerics.configure_numerics


def test_the_seam_binds_the_shared_package_by_name() -> None:
    """The one module whose whole job is naming another package must name the shared one."""
    assert _reuse.band_partition.__name__.startswith("teb_vae.lag_attn.eval")
    assert _reuse.stats.__name__.startswith("teb_vae.lag_attn.eval")


def test_this_file_is_one_of_the_named_exemptions_to_the_no_sideways_import_rule() -> None:
    """No module under ``eval/`` may import the pipeline this package was forked from -- a
    half-fork has two implementations *and* a dependency -- and that rule is enforced by the AST
    layering walk in ``test_eval_self_contained.py``, in absolute, relative, aliased and lazy form
    alike. It is not restated here: two scans of one rule are two chances for one of them to be
    edited alone.

    This *test* file does import the forked-from package, deliberately: pinning the shared
    primitives to the same objects both packages bind cannot be written any other way. The walk
    therefore carries a table of exempted test files with a reason each and asserts the table and
    the suite agree in both directions. Named here as well, so a reader of this file learns it is
    an exception rather than a precedent.
    """
    from pathlib import Path

    from .test_eval_self_contained import SIBLING_EVAL_TEST_EXEMPTIONS

    assert Path(__file__).name in SIBLING_EVAL_TEST_EXEMPTIONS
