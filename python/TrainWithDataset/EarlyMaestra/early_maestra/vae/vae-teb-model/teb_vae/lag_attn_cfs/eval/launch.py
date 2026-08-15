r"""Launching an entry point without a command line, and recording where each value came from.

Every runnable module in this package must start from an IDE's Run button with nothing typed, and
the way they all do it is the same: a module-level ``RUN_ARGS`` dict beside the ``__main__`` guard,
keyed by argparse ``dest``, merged **per key** under whatever the command line supplied. Per key
rather than all-or-nothing because the common iteration is varying one thing -- a fallback that was
discarded the moment any flag appeared would be useless for exactly that.

This module is that merge, once, for all of them. It is deliberately **layer 0 and stdlib only**
(``argparse``, ``typing``): the acceptance gate imports it and must stay free of ``torch``, and an
entry point that costs a numeric stack to *parse its own arguments* would be one nobody runs.

``RUN_ARGS`` is a launch convenience and not a second configuration surface. Anything that shapes
what a run *measures* -- the seed, the caps, the draw counts -- belongs in the override delta, which
is dumped into the run directory and is therefore the durable record; a value injected from Python
would appear in no artifact and could not be recovered from the output afterwards.
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, Optional, Sequence, Tuple

#: Where a resolved value came from, in ``summary.json`` and in the launch log line.
CLI_SOURCE = "cli"
DICT_SOURCE = "config"
DEFAULT_SOURCE = "default"


def resolve_launch_args(
    parser: argparse.ArgumentParser,
    run_args: Optional[Dict[str, Any]] = None,
    argv: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Merge a command line over a launch dict, per key, and say where each value came from.

    Args:
        parser: The entry point's own parser. Its ``dest`` set is both the valid key set for
            *run_args* and what *argv* is parsed with, so the usage line in an error names the
            module the operator actually launched.
        run_args: The fallback dict, keyed by argparse ``dest``. ``None`` or empty means the
            command line is the only source.
        argv: Command-line arguments. ``None`` reads ``sys.argv[1:]``.

    Returns:
        ``(values, sources)``. *values* is keyed by every ``dest`` the parser defines, so it can be
        splatted straight into the entry point's ``main``; *sources* maps each key to
        :data:`CLI_SOURCE`, :data:`DICT_SOURCE` or :data:`DEFAULT_SOURCE`. The sources are recorded
        in the run's own artifacts, so a run's provenance is unambiguous after the fact rather than
        reconstructed from a shell history.

    Raises:
        ValueError: If *run_args* carries a key that is not an argparse ``dest``. A typo there
            would otherwise silently do nothing, which is the same class of failure the
            ``eval_config`` validator guards against -- and it would do nothing *quietly*, on the
            one launch path that has no command line to misspell in the first place.
    """
    fallback = dict(run_args or {})

    valid_dests = {action.dest for action in parser._actions if action.dest != "help"}
    unknown = sorted(set(fallback) - valid_dests)
    if unknown:
        raise ValueError(
            f"RUN_ARGS carries key(s) that are not command-line arguments: "
            f"{', '.join(repr(key) for key in unknown)}. Valid keys are: "
            f"{', '.join(sorted(valid_dests))}. RUN_ARGS is a launch convenience, not a second "
            f"configuration surface -- settings that shape the run belong in the override delta, "
            f"which is dumped into the run directory."
        )

    parsed = vars(parser.parse_args(list(argv) if argv is not None else None))
    values: Dict[str, Any] = {}
    sources: Dict[str, str] = {}
    for key in sorted(valid_dests):
        if parsed.get(key) is not None:
            values[key], sources[key] = parsed[key], CLI_SOURCE
        elif fallback.get(key) is not None:
            values[key], sources[key] = fallback[key], DICT_SOURCE
        else:
            values[key], sources[key] = None, DEFAULT_SOURCE
    return values, sources


def missing_required(
    values: Dict[str, Any], required: Sequence[str], *, run_args_name: str = "RUN_ARGS"
) -> Optional[str]:
    """Return the refusal message for absent required arguments, or ``None`` when all are present.

    Required-ness is enforced *after* the merge rather than by ``required=True`` on the parser,
    which is the whole point: ``required=True`` fires before a launch dict is ever consulted, so it
    makes the Run button unusable no matter what the dict says.

    Args:
        values: The resolved values from :func:`resolve_launch_args`.
        required: The ``dest`` names that must carry a value.
        run_args_name: What to call the launch dict in the message.

    Returns:
        A message naming every absent argument and both ways to supply it, or ``None``.
    """
    absent = [name for name in required if values.get(name) is None]
    if not absent:
        return None
    flags = ", ".join(f"--{name.replace('_', '-')}" for name in absent)
    keys = ", ".join(f"{name!r}" for name in absent)
    return (
        f"{flags} is required. Pass it on the command line, or -- to launch this file from an "
        f"IDE's Run button -- set {keys} in {run_args_name} near the bottom of this module."
    )
