r"""Shared pytest configuration for the ``synthetic_v2`` test suite."""

from __future__ import annotations


def pytest_configure(config) -> None:
    r"""Register custom markers so unknown-mark warnings do not fire.

    ``slow`` tags the heavy end-to-end integration test (S6-T05), which runs the real
    scattering transform on a tiny grid. Deselect it with ``-m "not slow"``.
    """
    config.addinivalue_line(
        "markers", "slow: heavy integration test (real transform); skip with -m 'not slow'"
    )
