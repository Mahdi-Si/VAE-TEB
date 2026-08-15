r"""Every entry point of this package starts from an IDE's Run button.

The merge itself is the cfs cell's :mod:`teb_vae.lag_attn_cfs.eval.launch` and is tested there;
what this file checks is the convention's four rules against **this** package's runners, each of
which is a way the Run button breaks silently rather than loudly:

**A launch dict exists and is keyed by the parser's own ``dest`` set.** A key that is not an
argument does nothing, and does it on the one launch path with no command line to misspell in.

**No argument is ``required=True``, and no argument carries a non-``None`` argparse default.** The
first fires before the launch dict is ever read, so it makes the Run button unusable no matter what
the dict says. The second is subtler: the merge treats any non-``None`` parsed value as having come
from the command line, so an argparse default silently makes that key's launch-dict entry
unreachable -- the operator edits the dict, nothing changes, and nothing says why.

:data:`ENTRY_POINTS` is written out rather than discovered, and the guard below is what keeps that
from being a hole: it discovers every module under ``eval/`` carrying a ``__main__`` block and
asserts the tuple names exactly those. So a runner that lands without joining the tuple fails here --
which is the whole reason the list is not discovered by the parametrised tests themselves, since a
discovery walk would simply not see a runner that forgot the convention.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any, Tuple

import pytest

#: Every module in this package that an operator launches directly. Written out rather than
#: discovered, and cross-checked against the directory by
#: :func:`test_the_tuple_names_every_module_with_a_main_block`.
#:
#: Two, and neither is a copy of anything: ``run`` supplies this cell's binding and its own help
#: text, ``verify`` delegates the gate and adds this cell's one sweep axis and the cross-cell
#: table. The probe is deliberately not here -- it is the cfs cell's, reached through the binding.
ENTRY_POINTS: Tuple[str, ...] = (
    "teb_vae.lag_attn_transformer_cfs.eval.run",
    "teb_vae.lag_attn_transformer_cfs.eval.verify",
)

#: Where those modules live.
EVAL_ROOT = Path(__file__).resolve().parents[1] / "eval"


def _module(name: str) -> Any:
    """Import an entry point by name."""
    return importlib.import_module(name)


def _has_main_block(source: str) -> bool:
    """Whether a module guards a block on ``__name__ == '__main__'``."""
    return any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        for node in ast.walk(ast.parse(source))
    )


def test_the_tuple_names_every_module_with_a_main_block() -> None:
    """Both directions: a runner that landed without joining the tuple would otherwise be checked
    by none of the tests below, and an entry naming a module that is not a runner is a check on
    nothing."""
    runnable = set()
    for path in sorted(EVAL_ROOT.rglob("*.py")):
        if not _has_main_block(path.read_text(encoding="utf-8")):
            continue
        stem = path.relative_to(EVAL_ROOT).with_suffix("").as_posix().replace("/", ".")
        runnable.add(f"teb_vae.lag_attn_transformer_cfs.eval.{stem}")

    assert set(ENTRY_POINTS) == runnable, (
        f"only in ENTRY_POINTS: {sorted(set(ENTRY_POINTS) - runnable)}; "
        f"only in the package: {sorted(runnable - set(ENTRY_POINTS))}. A module with a __main__ "
        f"block is a module an operator launches, and every one of them must obey the launch "
        f"convention."
    )


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_every_entry_point_ships_a_launch_dict(name: str) -> None:
    """Without one there is nothing to fill in, and the Run button can only fail."""
    module = _module(name)

    assert isinstance(getattr(module, "RUN_ARGS", None), dict), (
        f"{name} has no RUN_ARGS dict, so it cannot be launched without a command line"
    )


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_every_launch_dict_key_is_an_argument(name: str) -> None:
    """A key that is not a ``dest`` silently does nothing. The resolver refuses it, and this is
    that refusal exercised against what each module actually ships."""
    module = _module(name)
    dests = {action.dest for action in module.build_parser()._actions if action.dest != "help"}

    assert set(module.RUN_ARGS) == dests, (
        f"{name}: RUN_ARGS keys and parser dests disagree; "
        f"only in RUN_ARGS: {sorted(set(module.RUN_ARGS) - dests)}, "
        f"only on the parser: {sorted(dests - set(module.RUN_ARGS))}"
    )


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_no_argument_is_required_by_argparse(name: str) -> None:
    """``required=True`` fires before the launch dict is consulted, so it makes the Run button
    unusable whatever the dict says. Required-ness belongs after the merge."""
    required = [action.dest for action in _module(name).build_parser()._actions if action.required]

    assert required == [], (
        f"{name}: {required} are required=True, so launching without a command line raises "
        f"before RUN_ARGS is read. Enforce them after the merge instead."
    )


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_no_argument_carries_a_non_none_argparse_default(name: str) -> None:
    """The merge reads any non-``None`` parsed value as coming from the command line, so an
    argparse default makes that key's launch-dict entry unreachable. Real defaults are applied
    after the merge -- ``verify``'s ``--out`` is the standing case."""
    defaulted = {
        action.dest: action.default
        for action in _module(name).build_parser()._actions
        if action.dest != "help" and action.default is not None
    }

    assert defaulted == {}, (
        f"{name}: {defaulted} carry argparse defaults, which shadow RUN_ARGS. Default to None and "
        f"apply the real default after resolve_launch_args."
    )


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_the_usage_line_names_this_package_rather_than_the_cfs_cell(name: str) -> None:
    """The parsers are enumerated locally rather than borrowed, and this is the reason it has to
    stay that way: a borrowed parser would print the cfs cell's module path in the usage line of a
    command an operator ran against this one."""
    parser = _module(name).build_parser()

    assert "lag_attn_transformer_cfs" in parser.prog
    assert "lag_attn_cfs." not in parser.prog


def test_the_runners_help_text_names_this_models_registry() -> None:
    """``--only`` and ``--skip`` interpolate the registry the binding resolves to. Derived on every
    build rather than restated, so an analysis registered on the binding appears in the help text
    by being registered in one place."""
    from teb_vae.lag_attn_transformer_cfs.eval import run as run_module

    helps = {
        action.dest: action.help
        for action in run_module.build_parser()._actions
        if action.dest in {"only", "skip"}
    }

    assert set(helps) == {"only", "skip"}
    for dest, text in helps.items():
        for analysis in run_module.ANALYSES:
            assert analysis in text, f"--{dest} does not name {analysis}"
        # The unskippable step is not selectable, so it must not be offered as a choice.
        for unskippable in run_module.UNSKIPPABLE_ANALYSES:
            assert f"{unskippable}," not in text and not text.endswith(unskippable)
