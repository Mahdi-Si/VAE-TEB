r"""Config loading for the lag-attention model: ``base:`` inheritance and resolution.

A variant is a config, not a subclass. The mechanism that makes that affordable is a single
key, ``base:``, naming a parent config whose contents this one inherits:

.. code-block:: yaml

    # v3_tiny.yaml
    base: v3.yaml
    general_config:
      epochs: 1          # only what actually differs

:func:`load_config` walks that chain and deep-merges each child over its parent, so a variant
file carries only its deltas. Nested dicts merge key-by-key -- setting one ``VAE_model`` key
does not drop the rest of the block -- while lists and scalars replace wholesale, because a
list is a value here (``cuda_devices: [0]`` must *replace* ``[0, 1, 2, 3, 4, 5, 6]``, not
append to it).

:func:`resolve_config_file` is the seam the experiment driver requires.
:class:`train.graph_model_base.GraphModelBase` reads YAML from a **path** and accepts no dict,
so a merged config only reaches it by being written back out and passed as a file. That
resolved file doubles as the run's provenance: it is the exact configuration that ran, with
every inherited default made explicit.
"""
from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, List, Optional

import yaml

#: The key naming a parent config. Resolved relative to the child's own directory, so a
#: config tree can be moved or copied without rewriting its links.
BASE_KEY = "base"


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    r"""Recursively merge ``over`` into a deep copy of ``base`` (``over`` wins).

    Nested dicts are merged key-by-key rather than replaced wholesale, so a variant delta that
    sets a single ``VAE_model`` kwarg does not drop the rest of the base ``VAE_model`` block.
    Lists and scalars replace: they are leaf values, not containers to be extended.

    Args:
        base: The inherited mapping. Not mutated.
        over: The overriding mapping. Not mutated.

    Returns:
        A new mapping; no value is shared with either input.
    """
    out = deepcopy(base)
    for key, val in over.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = deepcopy(val)
    return out


def _read_yaml(path: str) -> Dict[str, Any]:
    """Read one YAML file into a dict.

    Args:
        path: Filesystem path to the YAML document.

    Returns:
        The parsed mapping. An empty file yields an empty dict.

    Raises:
        ValueError: If the document's top level is not a mapping.
    """
    with open(path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(
            f"config {path!r} must contain a top-level YAML mapping, got "
            f"{type(loaded).__name__}."
        )
    return loaded


def load_config(config_path: str, _seen: Optional[List[str]] = None) -> Dict[str, Any]:
    r"""Load a config, resolving its ``base:`` chain into one merged mapping.

    Each ``base:`` is resolved **relative to the config that names it**, then loaded (itself
    recursively) and merged under the child. The ``base`` key is consumed here and never
    appears in the result: it is a loader directive, and leaving it in would draw an
    "unknown key" warning from ``validate_config`` and end up in the MLflow param dump.

    Args:
        config_path: Path to the leaf config.
        _seen: Internal. The chain of absolute paths already visited, used to detect cycles.

    Returns:
        The fully merged config mapping.

    Raises:
        ValueError: If the ``base:`` chain contains a cycle, if ``base`` is not a string, or
            if any document in the chain is not a top-level mapping.
        FileNotFoundError: If any config in the chain does not exist.
    """
    absolute = os.path.abspath(config_path)
    seen = list(_seen or [])
    if absolute in seen:
        chain = " -> ".join(seen + [absolute])
        raise ValueError(f"circular 'base:' reference in config chain: {chain}")
    seen.append(absolute)

    config = _read_yaml(absolute)
    base_ref = config.pop(BASE_KEY, None)
    if base_ref is None:
        return config
    if not isinstance(base_ref, str):
        raise ValueError(
            f"config {absolute!r} has a non-string '{BASE_KEY}:' value "
            f"({base_ref!r}); it must be a path to a parent config."
        )

    base_path = os.path.join(os.path.dirname(absolute), base_ref)
    return _deep_merge(load_config(base_path, _seen=seen), config)


def resolve_config_file(config_path: str, out_dir: str) -> str:
    r"""Resolve a config's ``base:`` chain and write the merged result to ``out_dir``.

    ``GraphModelBase.__init__`` takes a config **path**, not a dict, so a merged config
    reaches it only via a file. The written document is also the run's provenance record --
    every inherited value is explicit in it, which a config that only says ``base: v3.yaml``
    is not.

    Args:
        config_path: Path to the leaf config.
        out_dir: Directory receiving ``resolved_config.yaml``. Created if absent.

    Returns:
        Path to the written resolved config.

    Raises:
        ValueError: If the ``base:`` chain contains a cycle or is otherwise malformed.
        FileNotFoundError: If any config in the chain does not exist.
    """
    config = load_config(config_path)
    os.makedirs(out_dir, exist_ok=True)
    resolved_path = os.path.join(out_dir, "resolved_config.yaml")
    with open(resolved_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return resolved_path
