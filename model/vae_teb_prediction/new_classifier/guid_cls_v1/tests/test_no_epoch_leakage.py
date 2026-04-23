"""Static check: model code must not consume raw ``epoch`` from the batch.

PRD §3.3 forbids feeding absolute time-to-delivery to the model. This test
greps the four model modules (tokenizer, transformer, heads, top-level
classifier) and fails if any of them reads ``batch["epoch"]`` or
``batch.epoch``.

Cross-delivery censoring is allowed inside :class:`GuidSequenceDataset`
(precompute + dataset modules) because the result is a *mask*, not a
classifier feature.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_MODEL_FILES = (
    "segment_tokenizer.py",
    "temporal_transformer.py",
    "heads.py",
    "guid_classifier.py",
)

_FORBIDDEN = (
    re.compile(r"""batch\[['"]epoch['"]\]"""),
    re.compile(r"""\.epoch\b"""),
)


@pytest.mark.parametrize("filename", _MODEL_FILES)
def test_module_does_not_read_epoch(filename: str) -> None:
    """Static guard against epoch leakage into the model code path."""
    path = Path(__file__).resolve().parent.parent / filename
    text = path.read_text(encoding="utf-8")
    for pattern in _FORBIDDEN:
        m = pattern.search(text)
        assert m is None, (
            f"{filename}: forbidden epoch reference matched {m.group(0)!r}; "
            f"PRD §3.3 forbids feeding raw epoch to the model"
        )
