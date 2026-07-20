r"""A rerun with the same config, checkpoint and seed must produce identical numbers.

The spec makes this a requirement and nothing else asserts it. It is not obviously true: the
forward samples $z$ unconditionally, so without :func:`configure_numerics` seeding the global
generators the two runs draw different latents and every forecast number moves. Suppressing the
sampling instead would be worse -- a mean-$z$ evaluation reports a forecast the model never
actually makes -- so the sampling is seeded rather than removed.

TF32 and ``cudnn.benchmark`` are the other half. ``default.yaml`` sets ``benchmark: true``, and
nondeterministic algorithm selection would move the low-order bits between runs. Both are
disabled explicitly by ``configure_numerics``; :func:`test_numerics_are_load_bearing` asserts the
seeding actually is what holds this together, so the test cannot pass for the wrong reason.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from teb_vae.lag_attn.eval import run as run_module
from teb_vae.lag_attn.eval.tests.conftest import EVAL_TINY_CONFIG

#: Keys whose value legitimately differs between two runs: wall clock, paths, and the run
#: directory. Compared, they would fail every time and say nothing.
VOLATILE_KEYS = frozenset(
    {"elapsed_s", "output_dir", "checkpoint", "config", "max_memory_allocated_gb"}
)


def _numeric_leaves(value: Any, prefix: str = "") -> List[Tuple[str, float]]:
    """Flatten a parsed summary into ``(dotted path, number)`` pairs.

    Booleans are excluded: they are ``int`` subclasses in Python and would compare as $0$ / $1$
    against genuine numbers, which is legal but muddles what a mismatch means.

    Args:
        value: A parsed ``summary.json`` fragment.
        prefix: Dotted path accumulated so far.

    Returns:
        Every numeric leaf, keyed by path.
    """
    leaves: List[Tuple[str, float]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in VOLATILE_KEYS:
                continue
            leaves.extend(_numeric_leaves(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            leaves.extend(_numeric_leaves(item, f"{prefix}[{index}]"))
    elif isinstance(value, bool):
        return leaves
    elif isinstance(value, (int, float)):
        leaves.append((prefix, float(value)))
    return leaves


def _run_once(checkpoint: Path, output_dir: Path, repo_root: Path) -> Dict[str, Any]:
    """Run the pipeline once and return its parsed summary."""
    exit_code = run_module.main(
        config=str(repo_root / EVAL_TINY_CONFIG),
        checkpoint=str(checkpoint),
        output_dir=str(output_dir),
        device="cpu",
    )
    assert exit_code == 0, "a failed run cannot be compared for reproducibility"
    summary_path = output_dir / run_module.RESULTS_DIRNAME / "summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def test_two_runs_produce_identical_numbers(
    tiny_checkpoint, tmp_path, monkeypatch, repo_root
) -> None:
    """Bit-identical, not merely close. Anything else means something is unseeded."""
    monkeypatch.chdir(repo_root)

    first = _numeric_leaves(_run_once(tiny_checkpoint, tmp_path / "run_a", repo_root))
    second = _numeric_leaves(_run_once(tiny_checkpoint, tmp_path / "run_b", repo_root))

    assert [key for key, _ in first] == [key for key, _ in second], (
        "the two runs reported different keys, so they are not the same pipeline"
    )
    assert len(first) > 10, "too few numeric leaves compared for this to mean anything"

    differences = [
        (key, left, right)
        for (key, left), (_, right) in zip(first, second)
        if left != right
    ]
    assert not differences, f"reruns disagreed on {differences[:5]}"


def test_numerics_are_load_bearing(tiny_checkpoint, tmp_path, monkeypatch, repo_root) -> None:
    r"""Without the seeding, the sampled $z$ differs and the run's numbers move.

    This is what stops the reproducibility test above passing for the wrong reason -- if nothing
    in the pipeline were actually stochastic, it would pass with ``configure_numerics`` removed.
    """
    monkeypatch.chdir(repo_root)

    seeded = _run_once(tiny_checkpoint, tmp_path / "seeded", repo_root)

    # Re-run with the seeding neutered, so the global generator is left wherever the previous
    # run happened to leave it.
    monkeypatch.setattr(run_module, "configure_numerics", lambda seed: {"seed": int(seed)})
    torch.manual_seed(999)
    unseeded = _run_once(tiny_checkpoint, tmp_path / "unseeded", repo_root)

    seeded_leaves = dict(_numeric_leaves(seeded))
    unseeded_leaves = dict(_numeric_leaves(unseeded))
    moved = [
        key
        for key in seeded_leaves
        if key in unseeded_leaves and seeded_leaves[key] != unseeded_leaves[key]
    ]
    assert moved, (
        "removing the seeding changed nothing, so the reproducibility assertion is not "
        "actually testing the seeding"
    )
