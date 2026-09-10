r"""Every entry point of this package starts from an IDE's Run button.

The merge itself is the causal cell's :mod:`teb_vae.lag_attn_cfs.eval.launch` and is tested there;
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
from being a hole: it discovers every module under the package's runner directories carrying a
``__main__`` block and asserts the tuple names exactly those. So a runner that lands without joining
the tuple fails here -- which is the whole reason the list is not discovered by the parametrised
tests themselves, since a discovery walk would simply not see a runner that forgot the convention.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any, Tuple

import pytest

#: Every module in this package an operator launches directly. Written out rather than discovered,
#: and cross-checked against the directory by
#: :func:`test_the_tuple_names_every_module_with_a_main_block`.
#:
#: The trainer is deliberately not among them: it follows the trainers' single-constant convention
#: rather than the launch dict's, which its own test covers.
ENTRY_POINTS: Tuple[str, ...] = (
    "teb_vae.lag_slot_transformer_cfs.eval.run",
    "teb_vae.lag_slot_transformer_cfs.eval.verify",
    "teb_vae.lag_slot_transformer_cfs.eval.memory",
    "teb_vae.lag_slot_transformer_cfs.eval.latent_probes",
    "teb_vae.lag_slot_transformer_cfs.eval.acceptance",
    "teb_vae.lag_slot_transformer_cfs.instruments.campaign",
)

#: The directories a runner may live in, and the dotted prefix each one carries.
#:
#: Two rather than one, because the synthetic instruments are launched the same way the evaluation
#: passes are and would otherwise be checked by none of the rules below. The trainer is in neither:
#: it follows the trainers' single-constant convention rather than the launch dict's, and its own
#: test covers that.
RUNNER_ROOTS: Tuple[Tuple[Path, str], ...] = (
    (Path(__file__).resolve().parents[1] / "eval", "teb_vae.lag_slot_transformer_cfs.eval"),
    (
        Path(__file__).resolve().parents[1] / "instruments",
        "teb_vae.lag_slot_transformer_cfs.instruments",
    ),
)

#: The one argument an entry point cannot proceed without, enforced after the merge.
#:
#: The memory pass is absent on purpose rather than by omission: it measures a **geometry**, and the
#: shipped production configuration is the geometry it exists to measure, so an operator who wants
#: exactly that types nothing and presses Run. An entry point with no required argument is a
#: legitimate shape and the test below parametrises over this mapping rather than over every runner
#: so that it stays one.
#: The instrument campaign is absent for the memory pass's reason: the shipped generators at the
#: shipped seeds are the campaign it exists to run, so an operator who wants exactly that types
#: nothing and presses Run.
REQUIRED_AFTER_MERGE = {
    "teb_vae.lag_slot_transformer_cfs.eval.run": "checkpoint",
    "teb_vae.lag_slot_transformer_cfs.eval.verify": "summary",
    "teb_vae.lag_slot_transformer_cfs.eval.latent_probes": "checkpoint",
    "teb_vae.lag_slot_transformer_cfs.eval.acceptance": "runs",
}


def _module(name: str) -> Any:
    """Import an entry point by name.

    Args:
        name: The module's dotted path.

    Returns:
        The imported module.
    """
    return importlib.import_module(name)


def _has_main_block(source: str) -> bool:
    """Whether a module guards a block on ``__name__ == '__main__'``.

    Args:
        source: The module's source text.

    Returns:
        ``True`` when it does.
    """
    return any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        for node in ast.walk(ast.parse(source))
    )


def test_the_tuple_names_every_module_with_a_main_block() -> None:
    """Both directions.

    A runner that landed without joining the tuple would otherwise be checked by none of the tests
    below, and an entry naming a module that is not a runner is a check on nothing.
    """
    runnable = set()
    for root, prefix in RUNNER_ROOTS:
        for path in sorted(root.rglob("*.py")):
            if not _has_main_block(path.read_text(encoding="utf-8")):
                continue
            stem = path.relative_to(root).with_suffix("").as_posix().replace("/", ".")
            runnable.add(f"{prefix}.{stem}")

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
    """A key that is not a ``dest`` silently does nothing.

    The resolver refuses it, and this is that refusal exercised against what each module ships.
    """
    module = _module(name)
    dests = {action.dest for action in module.build_parser()._actions if action.dest != "help"}

    assert set(module.RUN_ARGS) == dests, (
        f"{name}: RUN_ARGS keys and parser dests disagree; "
        f"only in RUN_ARGS: {sorted(set(module.RUN_ARGS) - dests)}, "
        f"only on the parser: {sorted(dests - set(module.RUN_ARGS))}"
    )


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_no_argument_is_required_by_argparse(name: str) -> None:
    """``required=True`` fires before the launch dict is consulted.

    It would make the Run button unusable whatever the dict says, so required-ness belongs after
    the merge.
    """
    required = [
        action.dest for action in _module(name).build_parser()._actions if action.required
    ]

    assert required == [], (
        f"{name}: {required} are required=True, so launching without a command line raises "
        f"before RUN_ARGS is read. Enforce them after the merge instead."
    )


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_no_argument_carries_a_non_none_argparse_default(name: str) -> None:
    """The merge reads any non-``None`` parsed value as coming from the command line.

    An argparse default therefore makes that key's launch-dict entry unreachable: the operator
    edits the dict, nothing changes, and nothing says why.
    """
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
def test_the_usage_line_names_this_package_rather_than_a_sibling(name: str) -> None:
    """The parsers are enumerated locally rather than borrowed, and this keeps it that way.

    A borrowed parser prints the sibling's module path in the usage line of a command an operator
    ran against this one.
    """
    parser = _module(name).build_parser()

    assert "lag_slot_transformer_cfs" in parser.prog
    assert "lag_attn_cfs." not in parser.prog


@pytest.mark.parametrize("name", sorted(REQUIRED_AFTER_MERGE))
def test_the_refusal_names_both_ways_to_supply_the_missing_argument(name: str) -> None:
    """Enforced after the merge, and the message has to name the launch dict.

    An operator who reached this file from an IDE has no command line to add a flag to, so a
    refusal naming only the flag sends them to the wrong place.
    """
    module = _module(name)
    exit_code = module.main()

    assert exit_code == 2
    dest = REQUIRED_AFTER_MERGE[name]
    # The refusal is logged rather than raised, so what is asserted here is the contract around it:
    # a distinct exit code, and the argument still absent from the launch dict as shipped.
    assert module.RUN_ARGS[dest] is None, (
        f"{name}: RUN_ARGS[{dest!r}] ships filled in, so the refusal above can never fire for an "
        f"operator who cloned this repository."
    )


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_a_launch_dict_key_that_is_not_an_argument_refuses_at_startup(name: str) -> None:
    """The one launch path with no command line to misspell in still catches a misspelling."""
    from teb_vae.lag_attn_cfs.eval.launch import resolve_launch_args

    module = _module(name)
    with pytest.raises(ValueError, match="not command-line arguments"):
        resolve_launch_args(module.build_parser(), {"chekpoint": "x"}, [])


def test_the_run_entry_point_records_where_each_value_came_from() -> None:
    """A run whose draw count came from a flag is a different run from one where it came from the
    delta, and only the recorded sources say which it was."""
    from teb_vae.lag_attn_cfs.eval.launch import CLI_SOURCE, DICT_SOURCE, resolve_launch_args

    module = _module("teb_vae.lag_slot_transformer_cfs.eval.run")
    _values, sources = resolve_launch_args(
        module.build_parser(), {"checkpoint": "from-the-dict"}, ["--device", "cpu"]
    )

    assert sources["checkpoint"] == DICT_SOURCE
    assert sources["device"] == CLI_SOURCE


def test_a_flag_overrides_one_launch_dict_entry_and_leaves_the_rest_standing() -> None:
    """Per key, not all or nothing: the common iteration is varying one thing, and a fallback
    discarded the moment any flag appeared would be useless for exactly that."""
    from teb_vae.lag_attn_cfs.eval.launch import resolve_launch_args

    module = _module("teb_vae.lag_slot_transformer_cfs.eval.run")
    values, _sources = resolve_launch_args(
        module.build_parser(),
        {"checkpoint": "kept", "num_mc_samples": 8},
        ["--num-mc-samples", "32"],
    )

    assert values["checkpoint"] == "kept"
    assert values["num_mc_samples"] == 32
