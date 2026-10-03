"""P4-04: the objective's target gather, its AR(1) wiring, and its forecast mask.

The shared objective itself is pinned in the sibling suites; these tests pin only what this
package's ``compute_loss`` adds in front of it: which token each cell is scored against, that the
AR(1) coefficient reaches it, and that ``weight`` reaches its mask.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn_transformer_patch.nets import patch_target
from teb_vae.lag_attn_transformer_patch.nets.patching import patch_summaries, patchify

from .conftest import build, make_stub_batch

R, H, T = 16, 30, 300
PHASES = torch.tensor([3, 11])  # distinct per sample, so a gather that mixes rows is caught


def _inputs(gap: int | None = None):
    """``(fhr, weight, y_patch, u_patch)`` as the task builds them, with an optional scored gap."""
    batch = make_stub_batch(seq_len=T)
    weight = batch.weight.clone()
    if gap is not None:
        weight[:, gap] = 0.0
    y = patchify(batch.fhr, weight, raw_per_step=R, validity="fhr_weight")
    u = patchify(batch.up, weight, raw_per_step=R, validity="finite")
    return batch.fhr, weight, y, u


def _total(model, fo, fhr, weight) -> torch.Tensor:
    return model.compute_loss(fo, fhr, weight=weight)["metrics"]["total_loss"]


def test_target_and_persistence_are_standardized_summaries_of_the_right_tokens(monkeypatch):
    """Cell ``(a, tau)`` is scored against token ``a + 1 + tau`` and the persistence input is token
    ``a`` (B.10.7: one target definition), both standardized; tiled (stride 15) and dense (stride 1)
    alike. Non-identity loc/scale/eps, so a dropped affine or eps is caught."""
    loc, scale, eps = (0.3, -1.2), (2.0, 0.5), 0.05
    model = build(
        persistence_residual=True,
        target_summary_loc=loc,
        target_summary_scale=scale,
        variability_eps=eps,
    )
    fhr, weight, y, u = _inputs()
    expected = (patch_summaries(fhr.reshape(2, T, R), weight, eps=eps) - torch.tensor(loc)) / (
        torch.tensor(scale)
    )
    targets = []
    monkeypatch.setattr(
        patch_target,
        "compute_raw_objective",
        lambda fo, target, **_: targets.append(target) or {"metrics": {}},
    )
    rows = torch.arange(2)[:, None]
    with torch.no_grad():
        for phase, stride in ((PHASES, 15), (0, 1)):
            fo = model(y, u, phase, stride)
            model.compute_loss(fo, fhr, weight=weight)
            anchors = fo["anchor_index"]
            steps = anchors[:, :, None] + 1 + torch.arange(H)
            torch.testing.assert_close(targets[-1], expected[rows[..., None], steps])
            torch.testing.assert_close(fo["persistence"], expected[rows, anchors])


def test_ar_residual_at_phi_zero_is_the_factorized_score_bitwise():
    """At ``phi = 0`` the AR(1) model scores exactly as the factorized one; once ``phi != 0`` it
    does not, so the coefficient demonstrably reaches the shared objective."""
    ar, factorized = build(), build(forecast_ar_residual=False)
    fhr, weight, y, u = _inputs()
    with torch.no_grad():
        fo = ar(y, u, PHASES, 15)
        assert torch.equal(_total(ar, fo, fhr, weight), _total(factorized, fo, fhr, weight))
        ar.target_ar_logit.fill_(0.5)
        assert not torch.allclose(_total(ar, fo, fhr, weight), _total(factorized, fo, fhr, weight))


def test_an_invalid_target_patch_is_not_scored():
    """Garbage in both the FHR and the forecast at every cell aimed at a ``weight = 0`` token in the
    scored region leaves ``total_loss`` bitwise unchanged (under ``phi != 0``, so the AR lag is
    exercised); the same garbage aimed at the next, valid token moves it."""
    gap = 100  # anchors 70..99 aim at it; the 1-in-30 loss keeps them above the coverage floor
    model = build()
    fhr, weight, y, u = _inputs(gap)
    with torch.no_grad():
        model.target_ar_logit.fill_(0.5)
        fo = model(y, u, 0, 1)
        clean = _total(model, fo, fhr, weight)

        def garbage_at(token: int) -> torch.Tensor:
            raw = fhr.clone()
            raw[:, R * token : R * (token + 1)] = 1e3
            aimed = (fo["anchor_index"][:, :, None] + 1 + torch.arange(H)) == token
            poisoned = dict(fo)
            for key in ("mu_full", "mu_base"):
                poisoned[key] = fo[key].masked_fill(aimed[..., None], 1e3)
            return _total(model, poisoned, raw, weight)

        assert torch.equal(garbage_at(gap), clean)
        assert not torch.allclose(garbage_at(gap + 1), clean)
