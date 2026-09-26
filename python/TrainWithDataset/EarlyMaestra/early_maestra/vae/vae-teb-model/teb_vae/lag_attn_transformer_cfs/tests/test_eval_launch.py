r"""Every evaluation entry point of this package starts from an IDE's Run button.

The merge itself is the cfs cell's :mod:`teb_vae.lag_attn_cfs.eval.launch` and is tested there;
what this file checks is the convention's three rules against **this** package's runners, each of
which is a way the Run button breaks silently rather than loudly:

**The launch dict is keyed by the parser's own ``dest`` set.** A key that is not an argument does
nothing, and does it on the one launch path with no command line to misspell in.

**No argument is ``required=True``, and no argument carries a non-``None`` argparse default.** The
first fires before the launch dict is ever read, so it makes the Run button unusable no matter what
the dict says. The second is subtler: the merge treats any non-``None`` parsed value as having come
from the command line, so an argparse default silently makes that key's launch-dict entry
unreachable -- the operator edits the dict, nothing changes, and nothing says why.
"""
from __future__ import annotations

import importlib
from typing import Tuple

import pytest

#: Every module in this package's ``eval/`` that an operator launches directly: ``run`` supplies
#: this cell's binding and its own help text, ``verify`` delegates the gate and adds this cell's one
#: sweep axis and the cross-cell table.
ENTRY_POINTS: Tuple[str, ...] = (
    "teb_vae.lag_attn_transformer_cfs.eval.run",
    "teb_vae.lag_attn_transformer_cfs.eval.verify",
)


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_the_launch_dict_and_the_parser_obey_the_run_button_convention(name: str) -> None:
    """The three rules at once, on the parser each module actually ships. Real defaults are applied
    after the merge -- ``verify``'s ``--out`` is the standing case."""
    module = importlib.import_module(name)
    actions = [action for action in module.build_parser()._actions if action.dest != "help"]
    dests = {action.dest for action in actions}

    assert set(module.RUN_ARGS) == dests, (
        f"{name}: RUN_ARGS keys and parser dests disagree; "
        f"only in RUN_ARGS: {sorted(set(module.RUN_ARGS) - dests)}, "
        f"only on the parser: {sorted(dests - set(module.RUN_ARGS))}"
    )
    required = [action.dest for action in actions if action.required]
    assert required == [], (
        f"{name}: {required} are required=True, so launching without a command line raises "
        f"before RUN_ARGS is read. Enforce them after the merge instead."
    )
    defaulted = {action.dest: action.default for action in actions if action.default is not None}
    assert defaulted == {}, (
        f"{name}: {defaulted} carry argparse defaults, which shadow RUN_ARGS. Default to None and "
        f"apply the real default after resolve_launch_args."
    )
