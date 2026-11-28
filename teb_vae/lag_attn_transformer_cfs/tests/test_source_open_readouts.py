r"""The gate readouts under the soft and input UA masks with running-gap removal, and the controls.

Under ``source_mask_mode`` ``'soft'`` or ``'input'`` with ``source_running_gap_keys='remove'`` an
anchor whose every key the removal took has its gate closed, $g^{\mathrm{eff}}_t = g_t\,o_t$ with
$o_t = \exists \ell: \mathrm{lag\_mask}_{t\ell} \wedge \lnot R_{t\ell}$. What is asserted here:

* the forward returns $o_t$ as ``source_open`` and $g_t$ as ``event_gate_learned`` there, so a
  reader can tell an anchor the removal shut from one the model shut;
* without the removal the two modes return neither key, as before;
* every control arm writes its own ``attended_source_heads``, ``source_invalid_mass``,
  ``source_open`` and ``event_gate_learned``, never the matched forward's;
* the source-null arm encodes in eval mode and restores each module's own mode;
* the source-null KL reads the model's ``scored_weight``;
* the occluded arm gives the event gate the occluded UA's IRF evidence;
* the time-shift pairs score a model whose forward kwargs carry no ``source_valid``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from teb_vae.lag_attn_cfs.eval.analyses import time_shift
from teb_vae.lag_attn_rws.nets import controls

from .conftest import build, make_streams, make_stub_batch, make_task, tiny_warmup_kwargs

#: Three samples, so the permutation control has a non-trivial derangement.
BATCH = 3
#: The steps at which every key of sample 0 is removed.
CLOSED = (12, 13, 14)
#: The readouts every arm must recompute rather than carry.
READOUTS = ("attended_source_heads", "source_invalid_mass", "source_open", "event_gate_learned")


def _kwargs(mode, **overrides):
    return tiny_warmup_kwargs(
        source_quality_mask=True, source_mask_mode=mode, posterior_event_gate=True, **overrides
    )


def _matched(mode, perturb, remove=True):
    """A perturbed model, its forward under an all-valid UA, the stream and the removal mask."""
    kwargs = _kwargs(mode, **({"source_running_gap_keys": "remove"} if remove else {}))
    model = build(kwargs).eval()
    perturb(model)
    y_st, y_ph, u = make_streams(kwargs, batch=BATCH)
    valid = torch.ones(u.shape, dtype=torch.bool)
    removed = torch.zeros(u.shape[0], u.shape[1], int(model.lag_attn.L), dtype=torch.bool)
    removed[0, list(CLOSED)] = True
    removed[1, :, 0] = True
    with torch.no_grad():
        out = (
            model(y_st, y_ph, u, None, None, valid, removed) if remove
            else model(y_st, y_ph, u, None, None, valid)
        )
    return model, out, u, removed


def _open(model, u, removed):
    lag_mask = model.build_lag_mask(u.shape[1], u.device).to(torch.bool)
    return (lag_mask[None] & ~removed).any(dim=-1)


@pytest.mark.parametrize("mode", ["soft", "input"])
def test_the_removal_shut_anchors_carry_source_open_and_the_learned_gate(perturb_posterior, mode):
    model, out, u, removed = _matched(mode, perturb_posterior)

    expected = _open(model, u, removed)
    assert torch.equal(out["source_open"], expected)
    assert not bool(expected[0, list(CLOSED)].any())
    learned, applied = out["event_gate_learned"][..., 0], out["event_gate"][..., 0]
    # Removal-shut: the applied gate is zero, the learned one is the model's own (nonzero) value.
    assert float(applied[0, list(CLOSED)].abs().max()) == 0.0
    assert float(learned[0, list(CLOSED)].min()) > 0.0
    # Open: the two gates are one value.
    assert torch.equal(applied[expected], learned[expected])


@pytest.mark.parametrize("mode", ["soft", "input"])
def test_without_the_removal_the_modes_return_neither_key(perturb_posterior, mode):
    _model, out, _u, _removed = _matched(mode, perturb_posterior, remove=False)

    assert "source_open" not in out
    assert "event_gate_learned" not in out
    assert "source_invalid_mass" in out


@pytest.mark.parametrize("mode", ["soft", "input"])
def test_the_permutation_arm_carries_the_donors_source_open(perturb_posterior, mode):
    model, out, u, removed = _matched(mode, perturb_posterior)
    perm_index = torch.tensor([1, 2, 0])

    with torch.no_grad():
        shuffled = controls.perm_forward_outputs(model, out, perm_index=perm_index)

    assert torch.equal(shuffled["source_open"], _open(model, u, removed[perm_index]))
    for key in READOUTS:
        assert shuffled[key] is not out[key], key
    assert not torch.equal(shuffled["attended_source_heads"], out["attended_source_heads"])


@pytest.mark.parametrize("mode", ["hard", "soft", "input"])
def test_the_null_and_occluded_arms_write_their_own_readouts(perturb_posterior, mode):
    model, out, u, _removed = _matched(mode, perturb_posterior)
    gated = u if model.source_gate is None else model.source_gate(u)

    with torch.no_grad():
        nulled = controls.source_null_forward_outputs(model, out, u)
        reference = controls.occluded_forward_outputs(
            model, out, gated, anchors=out["anchor_index"]
        )
        occluded = controls.occluded_forward_outputs(
            model, out, gated, occlusion=torch.ones(u.shape[1], dtype=torch.bool),
            anchors=out["anchor_index"],
        )

    present = [key for key in READOUTS if key in out]
    assert "attended_source_heads" in present and "event_gate_learned" in present
    for arm in (nulled, reference, occluded):
        assert sorted(key for key in READOUTS if key in arm) == sorted(present)
        for key in present:
            assert arm[key] is not out[key], key
    # Re-run with nothing occluded, the arm is the matched forward; with everything occluded, not.
    for key in present:
        assert torch.equal(reference[key], out[key]), key
    assert not torch.equal(occluded["attended_source_heads"], out["attended_source_heads"])
    assert not torch.equal(nulled["attended_source_heads"], out["attended_source_heads"])


def test_the_null_arm_encodes_in_eval_mode_and_restores_each_modules_mode(perturb_posterior):
    model, out, u, _removed = _matched("soft", perturb_posterior)
    model.train()
    modules = model.source_kv_modules()
    assert modules and all(module.training for module in modules)
    seen = []
    encode = model.encode_source_kv

    def recording_encode(*args, **kwargs):
        seen.append([module.training for module in modules])
        return encode(*args, **kwargs)

    model.encode_source_kv = recording_encode
    with torch.no_grad():
        controls.source_null_forward_outputs(model, out, u)

    assert seen == [[False] * len(modules)]
    assert all(module.training for module in modules)


def test_the_null_kld_reads_the_models_scored_weight(perturb_posterior, monkeypatch):
    model, out, u, _removed = _matched("soft", perturb_posterior)
    weight = torch.ones(u.shape[:2])
    assert float(controls.source_null_kld(model, out, u, weight)) > 0.0
    calls = []

    def scored_weight(w):
        calls.append(w)
        return torch.zeros_like(w)

    monkeypatch.setattr(model, "scored_weight", scored_weight)

    assert float(controls.source_null_kld(model, out, u, weight)) == 0.0
    assert len(calls) == 1 and calls[0] is weight


def test_the_occluded_arm_gives_the_gate_the_occluded_uas_evidence():
    kwargs = tiny_warmup_kwargs(
        sequence_length=300, posterior_event_gate=True, source_quality_mask=True,
        source_mask_mode="soft", irf_readout=True, irf_gate_evidence=True,
    )
    model = build(kwargs).eval()
    with torch.no_grad():
        model.irf_evidence_gate.weight.fill_(1.0)
    ok = torch.ones(2, 300, dtype=torch.bool)
    level = torch.randn(2, 300, generator=torch.Generator().manual_seed(2))
    y_st, y_ph, u = make_streams(kwargs)
    with torch.no_grad():
        out = model(y_st, y_ph, u, 0, 1, ok[..., None].expand(-1, -1, kwargs["c_u"]),
                    ua_level=level, ua_valid=ok, target_step_valid=ok)
        gated = u if model.source_gate is None else model.source_gate(u)
        band = torch.zeros(300, dtype=torch.bool)
        band[150:] = True
        reference = controls.occluded_forward_outputs(model, out, gated, anchors=out["anchor_index"])
        occluded = controls.occluded_forward_outputs(
            model, out, gated, occlusion=band, anchors=out["anchor_index"]
        )
        expected = model.irf_evidence_for(
            {**out, "irf_ua_level": level.masked_fill(band, 0.0), "irf_ua_valid": ok & ~band}
        )

    assert reference["irf_evidence"] is out["irf_evidence"]
    assert torch.equal(occluded["irf_evidence"], expected)
    assert not torch.equal(occluded["irf_evidence"], out["irf_evidence"])


def test_the_time_shift_pairs_score_an_irf_model_without_the_quality_mask():
    """The IRF inputs make the forward's kwargs non-empty while ``source_valid`` is absent."""
    module = make_task(tiny_warmup_kwargs(posterior_event_gate=True, irf_readout=True)).eval()
    torch.manual_seed(0)
    for parameter in module.orig_model.posterior_head.delta_mu_head.parameters():
        parameter.data.normal_(0.0, 0.2)

    def batch(seed):
        stub = make_stub_batch(seed=seed)
        stub.up_valid = torch.ones_like(stub.up, dtype=torch.bool)
        return module.transfer_batch_to_device(stub, module.device, dataloader_idx=0)

    own = time_shift.score_pairs(module, batch(1), batch(1))
    other = time_shift.score_pairs(module, batch(1), batch(2))

    np.testing.assert_allclose(own["kld_shifted"], own["kld_aligned"], rtol=1e-6)
    assert not np.allclose(other["kld_shifted"], other["kld_aligned"])
