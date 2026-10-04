r"""The patch cell's own analyses: ports of four shared ones and ten new raw-signal ones.

Every module obeys the shared protocol (``teb_vae/lag_attn_cfs/eval/analyses/__init__.py``):

.. code-block:: python

    def run_<name>_analysis(context, *, eval_config, output_dir, probe=None) -> Dict[str, Any]

returning at least ``n_samples``, ``composition`` and ``plan`` (with ``capped``), run failure-isolated
by the shared runner. ``context`` is the shared ``AnalysisContext``: ``collection`` (tables, vectors,
retained arrays, ``record``, ``results``), the merged ``config``, and ``task``/``loader`` (``None`` on
an offline re-run -- record a skip then). ``task.orig_model`` is the eval view
(:mod:`..view`); the raw substrate is :mod:`..raw`. Write figures and CSVs under
``output_dir / "<name>"``; caps live in ``eval_config["caps"]`` under ``<name>_*`` keys.

**Headline scalars.** A module may declare ``HEADLINE: Tuple[Tuple[str, str], ...]`` of
``(headline_name, key)``; the binding registers each as the path ``(<name>, "headline", key)``, so the
analysis returns ``{"headline": {key: float, ...}}``. Names must be unique and must not reuse a shared
headline name. Nothing else about a module is read by the binding, so no analysis author edits
``binding.py``.

Layering (plan D7): no Lightning, no ``model/*``, no ``teb_vae.lag_attn_rws.eval``, and no import of
another patch analysis module (a port may call its shared namesake). Shared helpers come from
``teb_vae.lag_attn_cfs.eval`` layers 0-1 and :mod:`..raw`.
"""
from __future__ import annotations

from typing import Any, Dict


def skip_record(name: str, reason: str) -> Dict[str, Any]:
    """The protocol's clean skip: no population, nothing capped, the reason on record.

    Args:
        name: The analysis name.
        reason: Why nothing was computed.

    Returns:
        ``{"n_samples": None, "composition": {}, "plan": {...}, "skipped": True, "reason": ...}``.
    """
    return {
        "n_samples": None,
        "composition": {},
        "plan": {"capped": False, "analysis": name},
        "skipped": True,
        "reason": str(reason),
    }


__all__ = ["skip_record"]
