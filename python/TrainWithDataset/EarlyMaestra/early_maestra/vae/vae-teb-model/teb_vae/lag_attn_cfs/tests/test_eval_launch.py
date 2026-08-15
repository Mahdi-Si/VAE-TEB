r"""Every entry point starts from an IDE's Run button, and the merge that lets it.

Three things have to hold for a runner in this repository, and each is a way the Run button breaks
silently rather than loudly:

**A launch dict exists and is keyed by the parser's own ``dest`` set.** A key that is not an
argument does nothing, and does it on the one launch path with no command line to misspell in.

**No argument is ``required=True``, and no argument carries a non-``None`` argparse default.** The
first fires before the launch dict is ever read, so it makes the Run button unusable no matter what
the dict says. The second is subtler and was a real defect in the sibling: the merge treats any
non-``None`` parsed value as having come from the command line, so an argparse default silently
makes that key's launch-dict entry unreachable -- the operator edits the dict, nothing changes, and
nothing says why.

**A required argument is enforced after the merge**, with a message naming both ways to supply it.

:data:`ENTRY_POINTS` is written out rather than discovered, and the guard below is what keeps that
from being a hole: it discovers every module under ``eval/`` carrying a ``__main__`` block and
asserts the tuple names exactly those. So a runner that lands without joining the tuple fails here --
which is the whole reason the list is not discovered by the parametrised tests themselves, since a
discovery walk would simply not see a runner that forgot the convention.
"""
from __future__ import annotations

import argparse
import ast
import importlib
from pathlib import Path
from typing import Any, Tuple

import pytest

from teb_vae.lag_attn_cfs.eval import launch

#: Every module in this package that an operator launches directly. Written out rather than
#: discovered, and cross-checked against the directory by
#: :func:`test_the_tuple_names_every_module_with_a_main_block`.
ENTRY_POINTS: Tuple[str, ...] = (
    "teb_vae.lag_attn_cfs.eval.probe",
    "teb_vae.lag_attn_cfs.eval.run",
    "teb_vae.lag_attn_cfs.eval.verify",
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
    """The guard that makes an empty tuple honest, and the one that keeps it honest afterwards.

    Both directions: a runner that landed without joining the tuple would otherwise be checked by
    none of the tests below, and an entry naming a module that is not a runner is a check on
    nothing.
    """
    runnable = set()
    for path in sorted(EVAL_ROOT.rglob("*.py")):
        if not _has_main_block(path.read_text(encoding="utf-8")):
            continue
        stem = path.relative_to(EVAL_ROOT).with_suffix("").as_posix().replace("/", ".")
        runnable.add(f"teb_vae.lag_attn_cfs.eval.{stem}")

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
    after the merge."""
    defaulted = {
        action.dest: action.default
        for action in _module(name).build_parser()._actions
        if action.dest != "help" and action.default is not None
    }

    assert defaulted == {}, (
        f"{name}: {defaulted} carry argparse defaults, which shadow RUN_ARGS. Default to None and "
        f"apply the real default after resolve_launch_args."
    )


# =================================================================================================
# The merge itself
# =================================================================================================
def _parser() -> argparse.ArgumentParser:
    """A two-argument parser standing in for a runner's."""
    parser = argparse.ArgumentParser(prog="probe-parser")
    parser.add_argument("--alpha", default=None)
    parser.add_argument("--beta", default=None)
    return parser


def test_the_command_line_wins_per_key_rather_than_wholesale() -> None:
    """The common iteration is varying one thing, so a flag must override that one value and leave
    the rest of the dict standing."""
    values, sources = launch.resolve_launch_args(
        _parser(), {"alpha": "from_dict", "beta": "from_dict"}, ["--alpha", "from_cli"]
    )

    assert (values["alpha"], sources["alpha"]) == ("from_cli", launch.CLI_SOURCE)
    assert (values["beta"], sources["beta"]) == ("from_dict", launch.DICT_SOURCE)


def test_an_absent_value_is_none_and_says_it_is_a_default() -> None:
    """The third source exists so a run's provenance has no unlabelled case."""
    values, sources = launch.resolve_launch_args(_parser(), {}, [])

    assert values == {"alpha": None, "beta": None}
    assert set(sources.values()) == {launch.DEFAULT_SOURCE}


def test_a_launch_dict_key_that_is_not_an_argument_raises_naming_it() -> None:
    """The typo guard. Named, because the whole point is that the operator cannot see the mistake
    in a command line they never typed."""
    with pytest.raises(ValueError, match="alpah"):
        launch.resolve_launch_args(_parser(), {"alpah": 1}, [])


def test_every_parser_dest_appears_in_the_resolved_values() -> None:
    """The values are splatted into ``main``, so a missing key is a ``TypeError`` at launch."""
    values, _ = launch.resolve_launch_args(_parser(), {"alpha": 1}, [])

    assert set(values) == {"alpha", "beta"}


def test_a_missing_required_argument_names_both_ways_to_supply_it() -> None:
    """An operator who launched from the Run button cannot act on "pass --config"; the message has
    to name the dict as well."""
    message = launch.missing_required({"config": None, "other": 1}, ("config", "other"))

    assert message is not None
    assert "--config" in message
    assert "RUN_ARGS" in message
    assert "other" not in message


def test_nothing_missing_is_no_message() -> None:
    """Not vacuous: the check must stay silent when the values are there."""
    assert launch.missing_required({"config": "a.yaml"}, ("config",)) is None


def test_the_resolver_costs_no_numeric_stack() -> None:
    """The acceptance gate imports this module and must stay free of ``torch``: an entry point that
    cost a numeric stack to *parse its own arguments* would be one nobody runs."""
    imported = {
        alias.name
        for node in ast.walk(ast.parse(Path(launch.__file__).read_text(encoding="utf-8")))
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(ast.parse(Path(launch.__file__).read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom)
    }

    assert imported <= {"argparse", "typing", "__future__"}
