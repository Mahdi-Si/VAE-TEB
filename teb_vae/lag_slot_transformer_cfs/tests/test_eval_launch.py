r"""Every entry point of this package starts from an IDE's Run button.

The merge itself is the causal cell's :mod:`teb_vae.lag_attn_cfs.eval.launch` and is tested there;
what this file checks is the convention against **this** package's runners, each rule being a way
the Run button breaks silently rather than loudly:

**A launch dict exists and is keyed by the parser's own ``dest`` set.** A key that is not an
argument does nothing, and does it on the one launch path with no command line to misspell in.

**No argument is ``required=True``, and no argument carries a non-``None`` argparse default.** The
first fires before the launch dict is ever read, so it makes the Run button unusable no matter what
the dict says. The second is subtler: the merge treats any non-``None`` parsed value as having come
from the command line, so an argparse default silently makes that key's launch-dict entry
unreachable -- the operator edits the dict, nothing changes, and nothing says why.

**The argument a runner cannot proceed without is refused after the merge**, with exit code two.

:data:`ENTRY_POINTS` is discovered: every module under the package's runner directories carrying a
``__main__`` block, so a runner that lands without obeying the convention fails here.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any, Tuple

import pytest

#: The directories a runner may live in, and the dotted prefix each one carries.
#:
#: Two rather than one, because the synthetic instruments are launched the same way the evaluation
#: passes are. The trainer is in neither: it follows the trainers' single-constant convention
#: rather than the launch dict's, and its own test covers that.
RUNNER_ROOTS: Tuple[Tuple[Path, str], ...] = (
    (Path(__file__).resolve().parents[1] / "eval", "teb_vae.lag_slot_transformer_cfs.eval"),
    (
        Path(__file__).resolve().parents[1] / "instruments",
        "teb_vae.lag_slot_transformer_cfs.instruments",
    ),
)

#: The one argument an entry point cannot proceed without, enforced after the merge.
#:
#: The memory pass and the instrument campaign are absent on purpose: the shipped production
#: geometry and the shipped generators are what they exist to measure, so an operator who wants
#: exactly that types nothing and presses Run.
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


def _entry_points() -> Tuple[str, ...]:
    """Every module under :data:`RUNNER_ROOTS` an operator launches directly.

    Returns:
        The dotted module names, sorted.
    """
    runnable = set()
    for root, prefix in RUNNER_ROOTS:
        for path in sorted(root.rglob("*.py")):
            if _has_main_block(path.read_text(encoding="utf-8")):
                stem = path.relative_to(root).with_suffix("").as_posix().replace("/", ".")
                runnable.add(f"{prefix}.{stem}")
    return tuple(sorted(runnable))


#: Every module in this package an operator launches directly.
ENTRY_POINTS: Tuple[str, ...] = _entry_points()


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_every_entry_point_obeys_the_launch_convention(name: str) -> None:
    """A launch dict keyed by the parser's dests, nothing ``required``, no non-``None`` default.

    Args:
        name: The entry point's dotted module name.
    """
    module = _module(name)
    assert isinstance(getattr(module, "RUN_ARGS", None), dict), (
        f"{name} has no RUN_ARGS dict, so it cannot be launched without a command line"
    )
    actions = [action for action in module.build_parser()._actions if action.dest != "help"]
    dests = {action.dest for action in actions}

    assert set(module.RUN_ARGS) == dests, (
        f"{name}: only in RUN_ARGS: {sorted(set(module.RUN_ARGS) - dests)}, "
        f"only on the parser: {sorted(dests - set(module.RUN_ARGS))}"
    )
    assert [action.dest for action in actions if action.required] == [], (
        f"{name}: required=True fires before RUN_ARGS is read; enforce it after the merge"
    )
    defaulted = {action.dest: action.default for action in actions if action.default is not None}
    assert defaulted == {}, f"{name}: {defaulted} carry argparse defaults, which shadow RUN_ARGS"


@pytest.mark.parametrize("name", sorted(REQUIRED_AFTER_MERGE))
def test_a_missing_required_argument_is_refused_after_the_merge(name: str) -> None:
    """Enforced after the merge rather than by argparse, with a distinct exit code.

    Args:
        name: The entry point's dotted module name.
    """
    assert name in ENTRY_POINTS, f"{name} is no longer discovered as a runner"
    assert _module(name).main() == 2
