"""The migration guide's citations resolve and its named capabilities exist.

Two guards keep ``MODEL_MIGRATION_GUIDE.md`` honest as the consumer tree moves:

- every ``path:line`` (or ``path:start-end``) citation points at a real file with the
  cited line(s) in range — a drifted citation fails the range check;
- every framework capability the guide tells consumers to adopt is named in the guide
  **and** resolves to a ``def``/``class`` symbol under ``train/`` or ``utils/``.

``utils/`` is searched alongside ``train/`` because framework capabilities may legitimately
be leaf helpers: ``log_artifact_to_mlflow`` lives in ``utils/mlflow_utils.py`` so that both
``train/callbacks.py`` and ``utils/seqvae_plot_callbacks.py`` can share it without ``utils/``
importing ``train/`` (see ``test_layering.py``).
"""
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GUIDE = Path(__file__).resolve().parents[1] / "MODEL_MIGRATION_GUIDE.md"

# Package dirs that make up the framework layers a capability may live in.
_FRAMEWORK_DIRS = ("train", "utils")

# Capabilities the guide directs consumers to adopt; each must resolve to a def/class
# under one of _FRAMEWORK_DIRS.
_CAPABILITIES = [
    "build_trainer",
    "select_ddp_strategy",
    "configure_determinism",
    "validate_config",
    "configure_param_groups",
    "build_lr_scheduler",
    "on_save_checkpoint",
    "log_artifact_to_mlflow",
    "GraphDataModule",
    "MLflowRunLoggingCallback",
    "check_model_class",
    "MetricsHistoryCsvCallback",
]

# A repo-relative python path followed by a line number or line range, e.g.
# ``model/.../trainer_raw_v4.py:530-565``. Requires the ``:line`` suffix, so bare module
# paths in prose (no line) are not treated as citations.
_CITATION = re.compile(r"([A-Za-z0-9_][A-Za-z0-9_./\-]*\.py):(\d+)(?:-(\d+))?")


def _read(path):
    return path.read_text(encoding="utf-8", errors="replace")


def _line_count(path):
    # errors="replace" keeps the line structure intact for counting even if a source
    # file carries a stray non-UTF-8 byte.
    with open(path, encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def test_guide_exists():
    assert _GUIDE.is_file()


def test_citations_resolve_and_in_range():
    citations = _CITATION.findall(_read(_GUIDE))
    # The guide is citation-dense; a garbled or truncated guide must fail here.
    assert len(citations) >= 20

    for rel_path, start, end in citations:
        target = _REPO_ROOT / rel_path
        assert target.is_file(), f"cited file missing: {rel_path}"
        n = _line_count(target)
        lo = int(start)
        hi = int(end) if end else lo
        span = f"{start}-{end}" if end else start
        assert 1 <= lo <= hi <= n, f"{rel_path}:{span} out of range (file has {n} lines)"


def test_capabilities_named_and_resolve():
    text = _read(_GUIDE)
    sources = "\n".join(
        _read(p)
        for d in _FRAMEWORK_DIRS
        for p in sorted((_REPO_ROOT / d).glob("*.py"))
    )
    for name in _CAPABILITIES:
        assert name in text, f"capability not named in guide: {name}"
        symbol = re.compile(rf"^\s*(def|class)\s+{re.escape(name)}\b", re.MULTILINE)
        assert symbol.search(sources), (
            f"capability does not resolve in {'/, '.join(_FRAMEWORK_DIRS)}/: {name}"
        )
