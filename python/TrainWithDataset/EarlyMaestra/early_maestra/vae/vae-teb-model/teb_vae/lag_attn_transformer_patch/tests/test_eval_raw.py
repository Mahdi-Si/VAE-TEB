"""The raw-signal substrate (``eval/raw.py``) on the tiny eval view: its forward is the task's, its
raw-UP integrated gradients are complete and confined to the lag window, and its edits keep gaps."""
from __future__ import annotations

import torch

from teb_vae.lag_attn_cfs.eval import metrics
from teb_vae.lag_attn_rws.tests.conftest import TASK_HPARAMS
from teb_vae.lag_attn_transformer_patch.eval import raw
from teb_vae.lag_attn_transformer_patch.eval.view import (
    SeqVaeLagAttnTrfPatch,
    SeqVaeLagAttnTrfPatchEvalTask,
)

from .conftest import TINY_KWARGS, make_stub_batch

#: ``fhr_weight`` so a ``weight`` gap reaches UP too: a builder that ignored the model's
#: ``source_validity`` would then differ from the other one.
KWARGS = dict(TINY_KWARGS, source_validity="fhr_weight")
ANCHOR, GAP = 150, 147  # a scored anchor, and a gap token inside its lag window (max_lag 8)
R = KWARGS["raw_per_step"]


def _setup(perturb_posterior):
    """The perturbed eval view, its eval task and a stub batch with a planted gap at ``GAP``."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfPatch(**KWARGS).eval()
    perturb_posterior(model)  # at init every KL is 0
    task = SeqVaeLagAttnTrfPatchEvalTask(model, lr=1e-3, model_kwargs=KWARGS, **TASK_HPARAMS)
    batch = make_stub_batch(2, KWARGS["sequence_length"])
    batch.weight[:, GAP] = 0.0
    return model, task, batch


def test_forward_raw_is_the_tasks_input_path_and_edits_keep_gaps(perturb_posterior) -> None:
    model, task, batch = _setup(perturb_posterior)
    y_st, y_ph, u_stream, _, _ = metrics.model_inputs(task, batch)
    with torch.no_grad():
        torch.manual_seed(1)
        expected = model(y_st, y_ph, u_stream, anchor_phase=0, anchor_stride=1)
        torch.manual_seed(1)
        actual = raw.forward_raw(model, batch.fhr, batch.up, batch.weight)
    assert expected.keys() == actual.keys()
    for key, value in expected.items():
        assert torch.equal(value, actual[key]), key

    signal = torch.randn(1, 4 * R)
    signal[0, R + 3] = float("nan")  # a gap sample inside the overwritten token 1
    token = raw.replace_tokens(signal, torch.tensor([[False, True, False, False]]), 0.5, raw_per_step=R)[0, R:2 * R]
    assert torch.isnan(token[3]) and (token[torch.arange(R) != 3] == 0.5).all()


def test_raw_up_ig_of_kld_is_complete_and_zero_outside_the_lag_window(perturb_posterior) -> None:
    """128 steps from a flat zero UP: completeness holds, and the attribution is exactly zero after
    the anchor's token, before its lag window (the first token's delta reads one sample back), on a
    gap token and on non-finite samples. The attributed readout at the input is the evaluation's own
    on the ORIGINAL raw: a NaN UP span and a NaN FHR sample inside a weight-1 patch stay masked,
    rather than turning into valid zeros when the IG input is made finite."""
    model, _, batch = _setup(perturb_posterior)
    fhr, up, weight = batch.fhr[:1].clone(), batch.up[:1].clone(), batch.weight[:1]
    up[0, R * 145:R * 145 + 5] = float("nan")   # inside the lag window, a weight-1 token
    fhr[0, R * 148 + 7] = float("nan")          # inside a weight-1 FHR patch
    fhr3, up3, fhr_valid, up_valid = raw.RawReadout.prepare(model, fhr, up, weight)
    wrapper = raw.RawReadout(raw.AnchorReadout(model, raw.ATTENTION_CELL, readout="kld").eval())
    columns = torch.tensor([ANCHOR - int(model.warmup_period)])
    assert int(wrapper.anchor_steps(columns)) == ANCHOR

    result = raw.raw_ig(
        wrapper, (fhr3, fhr3[..., :0], up3), (model.summary_target(fhr, weight), weight),
        columns, torch.zeros(1, dtype=torch.long), stream="up", baseline=torch.zeros_like(up3), n_steps=128,
        valid=(fhr_valid, up_valid),
    )
    with torch.no_grad():
        evaluated = raw.forward_raw(model, fhr, up, weight)["kld_per_t"][0, ANCHOR]
    assert abs(float(result["value_input"]) - float(evaluated)) <= 1e-6
    attribution = result["map"].flatten()
    assert torch.isfinite(attribution).all() and torch.isfinite(result["delta"]).all()
    assert (attribution[~up_valid.flatten()] == 0).all()
    moved = float(result["value_input"] - result["value_entry"])
    assert abs(moved) > 1e-3
    assert abs(float(attribution.sum()) - moved) < 5e-2 * abs(moved)

    # TINY_KWARGS' ``lag_kv_source: adapter`` gives the source a one-token reach, so the window is
    # exactly the lag tokens ANCHOR - max_lag .. ANCHOR.
    first = R * (ANCHOR - (int(model.lag_attn.L) - 1)) - 1
    assert (attribution[:first] == 0).all()
    assert (attribution[R * (ANCHOR + 1):] == 0).all()
    assert (attribution[R * GAP:R * (GAP + 1)] == 0).all()
