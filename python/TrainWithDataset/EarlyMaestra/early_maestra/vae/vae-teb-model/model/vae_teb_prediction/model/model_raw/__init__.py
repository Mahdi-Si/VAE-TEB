"""Raw-signal VAE-TEB v4 (``SeqVaeRawV4``) package.

This package is the raw-signal-input fork of the lag-attentive VAE-TEB model. It replaces the
fixed scattering/phase front end of :class:`SeqVaeLagAttnV3` with a learned, strictly-causal,
anti-aliased multiscale front end over the raw $4$ Hz FHR/UP signals, and replaces the
feature-domain forecast with a raw future-FHR forecast, while inheriting the entire v3
information architecture (guarantees G0-G11) unchanged.

The authoritative design lives in
``model/vae_teb_prediction/model/vae-teb-raw-model.md`` and the roadmap in
``model/vae_teb_prediction/model/vae-teb-raw-v4-spec-and-sprints.md``.

Sprint 0 lands the package scaffold, config-driven geometry, the reuse surface, and the config
skeleton; Sprint 1 lands the causal raw front end.
"""
from __future__ import annotations
