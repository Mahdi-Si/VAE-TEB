r"""S6-T02/T03/T04: the ``lag_intervention`` stage, its true-band gate, and its figure.

Three separable claims are pinned here.

**The intervention is sound.** An all-keep band mask must reproduce the clean forward exactly
(``delta_L == 0``), a masked band must contribute no attention mass, and the per-cell true band
:math:`\mathcal{L}^\star` must be selected per :math:`D` even though the model's mask API is not
per-sample -- which is why the collector buckets each batch by its distinct ``delay`` values.

**The gate is sound.** ``inband_gate_pass`` is exercised on synthetic per-cell profiles with a
known answer: one where masking the true band hurts more than masking :math:`\{\ell \ge D\}`
(agreeing) and one where it hurts less (disagreeing). It is *never* asserted against a pilot
checkpoint: at 400 steps the source pathway has not switched on (Sprint 3 measured
``shuffle_penalty`` :math:`\approx 4\times10^{-6}`), so every :math:`\Delta L` sits at zero and
the gate fails everywhere. That is a training-progress signal, recorded in the spec, not a bug.

**The gate is never asserted where it is meaningless.** A :math:`\mathrm{TE}_{\mathrm{inj}} = 0`
null cell has no true lag, so its ``inband_gate_pass`` is ``None``.
"""
from __future__ import annotations

import json
import types

import numpy as np
import pytest
import torch

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (
    lag_intervention_v3 as li,
)

_B, _T, _C_Y, _L = 4, 24, 8, 12
_HORIZON, _WARMUP = 4, 2
_BANDS = {"0-2": (0, 2), "3-5": (3, 5), "6-11": (6, 11)}


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------
def test_band_keep_mask_excludes_the_band() -> None:
    keep = li.band_keep_mask(_L, [3, 4, 5])
    assert keep.dtype == torch.bool and keep.shape == (_L,)
    assert not keep[3:6].any()
    assert keep[:3].all() and keep[6:].all()


def test_band_keep_mask_clips_out_of_range_lags() -> None:
    keep = li.band_keep_mask(4, [2, 3, 99, -1])
    assert keep.tolist() == [True, True, False, False]


def test_true_band_masks_split_at_the_true_lag() -> None:
    r""":math:`\mathcal{L}^\star = [\max(0, D-H), D)` and its complement :math:`\{\ell \ge D\}`."""
    d = 5
    m_star, m_comp = li._true_band_masks(_L, d, _HORIZON)
    # Ablating L* = {1..4} (since D-H = 1) keeps lag 0 and everything >= 5.
    assert m_star.tolist() == [True] + [False] * 4 + [True] * 7
    # Ablating the complement keeps only lags < D.
    assert m_comp.tolist() == [True] * 5 + [False] * 7


def test_true_band_masks_are_complementary_over_the_split() -> None:
    r"""Every lag in :math:`[D-H, L)` is ablated by exactly one of the two masks."""
    d, n = 5, _L
    m_star, m_comp = li._true_band_masks(n, d, _HORIZON)
    ablated_star = {i for i in range(n) if not m_star[i]}
    ablated_comp = {i for i in range(n) if not m_comp[i]}
    assert ablated_star.isdisjoint(ablated_comp)
    assert ablated_star | ablated_comp == set(range(max(0, d - _HORIZON), n))


# ---------------------------------------------------------------------------
# Collector, over a stub model that reacts to the mask in a known way.
# ---------------------------------------------------------------------------
class _StubModel:
    r"""A model whose forecast error grows with the attention mass it *loses*.

    ``sensitivity[l]`` is the loss penalty incurred by ablating lag ``l``. The clean forward is
    perfect (``mu_full == y_plus``), so ``feat_loss == 0`` and ``delta_L`` reads back the
    penalty directly -- which makes the arithmetic of the collector checkable by hand.
    """

    def __init__(self, sensitivity: np.ndarray, y_plus: torch.Tensor) -> None:
        self.sensitivity = torch.as_tensor(sensitivity, dtype=torch.float32)
        self._y_plus = y_plus
        self.lag_attn = types.SimpleNamespace(L=_L)
        self.calls = []

    # Presence of this attribute is the stage's v3 capability probe.
    def _combined_lag_mask(self, *a, **k):  # pragma: no cover - probe only
        return None, None

    def eval(self):
        return self

    def __call__(self, y_st, y_ph, u_stream, *, lag_band_mask=None):
        self.calls.append(None if lag_band_mask is None else lag_band_mask.clone())
        b, t = y_st.shape[0], y_st.shape[1]
        if lag_band_mask is None:
            penalty = torch.zeros(b)
        else:
            lost = ~lag_band_mask.to(torch.bool)
            penalty = self.sensitivity[lost].sum().expand(b).clone()

        # mu_full = truth + a constant offset whose square is the penalty.
        offset = penalty.sqrt().view(b, 1, 1, 1)
        mu_full = torch.zeros(b, t, _HORIZON, _C_Y)
        mu_full[:, : self._y_plus.shape[1]] = self._y_plus + offset

        # A flat, valid attention profile so te_lag_map mass is well-defined.
        lag_map = torch.ones(b, t, _L) / _L
        if lag_band_mask is not None:
            lag_map = lag_map * lag_band_mask.to(lag_map.dtype).view(1, 1, -1)
        return {
            "mu_full": mu_full,
            "kld_per_t": torch.ones(b, t),
            "te_lag_map": lag_map,
        }


class _StubRunner:
    def __init__(self, batches, model) -> None:
        self._batches = batches
        self.model = model
        self.warmup_steps = _WARMUP
        self.horizon = _HORIZON

    def inference_mode(self):
        import contextlib

        return contextlib.nullcontext()

    def iter_batches(self, loader, max_samples=None):
        seen = 0
        for b in self._batches:
            if max_samples is not None and seen >= max_samples:
                break
            yield b
            seen += int(b.fhr_st.shape[0])

    def build_future_target(self, batch):
        return batch.y_plus


def _make_batch(*, delays, te_levels, cell_ids, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    n = len(delays)
    y_plus = torch.randn(n, _T - _HORIZON, _HORIZON, _C_Y, generator=g)
    return types.SimpleNamespace(
        fhr_st=torch.zeros(n, _T, 4),
        fhr_ph=torch.zeros(n, _T, 4),
        up_st=torch.zeros(n, _T, 3),
        up_ph=torch.zeros(n, _T, 5),
        y_plus=y_plus,
        delay=torch.tensor(delays, dtype=torch.long),
        te_true=torch.tensor(te_levels, dtype=torch.float32),
        cell_id=torch.tensor(cell_ids, dtype=torch.long),
    )


def _collect(sensitivity, batches, max_samples=None):
    model = _StubModel(np.asarray(sensitivity, dtype=np.float32), batches[0].y_plus)
    runner = _StubRunner(batches, model)
    # Every batch shares y_plus shape; the stub reuses batch 0's target for mu_full.
    return model, li.collect_lag_intervention(
        model, runner, None, bands=_BANDS, max_samples=max_samples
    )


def test_noop_control_yields_zero_delta_l() -> None:
    r"""An all-keep mask reduces to the causal validity mask: ``delta_L == 0``."""
    batch = _make_batch(delays=[5] * _B, te_levels=[2.0] * _B, cell_ids=[1] * _B)
    _, res = _collect(np.zeros(_L), [batch])
    assert res["noop_max_abs_delta"] == pytest.approx(0.0, abs=1e-6)


class _NoisyStubModel(_StubModel):
    r"""Adds reparameterisation noise to ``mu_full``, exactly as the real ``forward`` does.

    ``SeqVaeLagAttnV3.forward`` samples :math:`z \sim q` on every call with no ``self.training``
    guard, so :math:`\mu_{\mathrm{full}}` is stochastic. Measured on the ``v3_prod`` pilot that
    noise moves the forecast loss by :math:`2.9\times10^{-4}` -- two orders of magnitude above
    the in-band lift being measured. Unpaired, :math:`\Delta L` is pure noise.
    """

    def __call__(self, y_st, y_ph, u_stream, *, lag_band_mask=None):
        out = super().__call__(y_st, y_ph, u_stream, lag_band_mask=lag_band_mask)
        out["mu_full"] = out["mu_full"] + torch.randn_like(out["mu_full"]) * 0.5
        return out


def test_common_random_numbers_cancel_the_posterior_noise() -> None:
    r"""With a stochastic forward, only the shared seed makes the no-op control read zero.

    This is the regression guard for the confound the no-op control caught on the real pilot:
    every forward in a batch must draw the same :math:`\varepsilon`, or ``delta_L`` measures
    sampling noise instead of the mask.
    """
    batch = _make_batch(delays=[5] * _B, te_levels=[2.0] * _B, cell_ids=[1] * _B)
    model = _NoisyStubModel(np.zeros(_L, dtype=np.float32), batch.y_plus)
    runner = _StubRunner([batch], model)
    res = li.collect_lag_intervention(model, runner, None, bands=_BANDS, seed=7)
    assert res["noop_max_abs_delta"] == pytest.approx(0.0, abs=1e-6)
    # ... and every band's delta_L is exactly zero too, since the stub has no sensitivity.
    for name in _BANDS:
        assert res[f"delta_L_{name}"] == pytest.approx(np.zeros(_B), abs=1e-6)


def test_unpaired_forwards_would_have_hidden_the_effect() -> None:
    r"""Demonstrates the confound: two *independent* noisy forwards disagree substantially."""
    batch = _make_batch(delays=[5] * _B, te_levels=[2.0] * _B, cell_ids=[1] * _B)
    model = _NoisyStubModel(np.zeros(_L, dtype=np.float32), batch.y_plus)
    args = (batch.fhr_st, batch.fhr_ph, torch.zeros(_B, _T, 8))
    torch.manual_seed(0)
    a = model(*args)["mu_full"]
    b = model(*args)["mu_full"]                       # a fresh epsilon draw
    assert not torch.allclose(a, b, atol=1e-3), "the noisy stub is not actually stochastic"

    # Seeding identically recovers the pairing.
    c = li._seeded_forward(model, *args, seed=11)["mu_full"]
    d = li._seeded_forward(model, *args, seed=11)["mu_full"]
    assert torch.equal(c, d)


def test_isolated_rng_restores_the_global_state() -> None:
    r"""The stage reseeds the global RNG; it must put it back."""
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    with li._isolated_rng():
        torch.manual_seed(999)
        _ = torch.randn(4)
    assert torch.equal(torch.get_rng_state(), before)


def test_delta_l_reads_back_the_ablated_sensitivity() -> None:
    r"""Ablating band ``G`` costs exactly the sensitivity mass inside ``G``."""
    sens = np.zeros(_L, dtype=np.float32)
    sens[3:6] = 0.25                       # all the signal lives in band "3-5"
    batch = _make_batch(delays=[5] * _B, te_levels=[2.0] * _B, cell_ids=[1] * _B)
    _, res = _collect(sens, [batch])

    assert res["delta_L_0-2"] == pytest.approx(np.zeros(_B), abs=1e-5)
    assert res["delta_L_3-5"] == pytest.approx(np.full(_B, 0.75), rel=1e-4)
    assert res["delta_L_6-11"] == pytest.approx(np.zeros(_B), abs=1e-5)


def test_masked_band_contributes_no_attention_mass() -> None:
    r"""The clean forward's ``te_lag_map`` mass over a band matches the flat stub profile."""
    batch = _make_batch(delays=[5] * _B, te_levels=[2.0] * _B, cell_ids=[1] * _B)
    _, res = _collect(np.zeros(_L), [batch])
    # A flat profile over L=12: band "0-2" holds 3/12 of the mass.
    assert res["mass_0-2"] == pytest.approx(np.full(_B, 3 / _L), rel=1e-6)
    assert res["mass_3-5"] == pytest.approx(np.full(_B, 3 / _L), rel=1e-6)
    assert res["mass_6-11"] == pytest.approx(np.full(_B, 6 / _L), rel=1e-6)
    # inband + outband partition the mass exactly.
    assert res["mass_inband"] + res["mass_outband"] == pytest.approx(np.ones(_B), rel=1e-6)


def test_delay_bucketing_applies_the_right_true_band_per_sample() -> None:
    r"""A batch mixing :math:`D` must not apply one sample's :math:`\mathcal{L}^\star` to all.

    The mask API is ``(L,)`` / ``(T, L)``; a ``(B, T, L)`` mask would be collapsed to sample 0's
    row. The collector therefore buckets by ``delay`` and gathers the matching rows.
    """
    sens = np.zeros(_L, dtype=np.float32)
    sens[0] = 1.0                       # only lag 0 carries signal
    # D = 1 -> L* = {0} (ablating it costs 1.0). D = 6 -> L* = {2..5} (costs 0.0).
    batch = _make_batch(delays=[1, 1, 6, 6], te_levels=[2.0] * 4, cell_ids=[1, 1, 2, 2])
    _, res = _collect(sens, [batch])

    inband = res["delta_L_inband"]
    assert inband[:2] == pytest.approx(np.full(2, 1.0), rel=1e-4), "D=1 rows lost lag 0"
    assert inband[2:] == pytest.approx(np.zeros(2), abs=1e-5), "D=6 rows must not lose lag 0"

    # Masking {l >= D} costs lag 0 only when D == 0, i.e. never here.
    assert res["delta_L_outband"] == pytest.approx(np.zeros(4), abs=1e-5)


def test_max_samples_is_honoured() -> None:
    batches = [
        _make_batch(delays=[5] * _B, te_levels=[2.0] * _B, cell_ids=[1] * _B, seed=s)
        for s in range(4)
    ]
    _, res = _collect(np.zeros(_L), batches, max_samples=_B)
    assert res["n_samples"] == _B


def test_v1_model_is_rejected_loudly() -> None:
    r"""A model without ``lag_band_mask`` support cannot silently produce zero ``delta_L``.

    The committed ``pl_module_v2`` alias is ``SeqVaeLagAttnV1``, which ignores unknown forward
    kwargs at its own peril. Probing for ``_combined_lag_mask`` is what stops a v1 checkpoint
    from being graded as if every band mattered exactly nothing.
    """
    batch = _make_batch(delays=[5] * _B, te_levels=[2.0] * _B, cell_ids=[1] * _B)

    class _V1Model:
        lag_attn = types.SimpleNamespace(L=_L)

        def eval(self):
            return self

    model = _V1Model()
    assert not hasattr(model, "_combined_lag_mask")
    runner = _StubRunner([batch], model)
    with pytest.raises(AttributeError, match="lag_band_mask"):
        li.collect_lag_intervention(model, runner, None, bands=_BANDS)


# ---------------------------------------------------------------------------
# S6-T03: the gate, on synthetic per-cell profiles with a known answer.
# ---------------------------------------------------------------------------
def _profiles(*, inband_rel, outband_rel, te=2.0, n_cells=3, mass_follows=True):
    r"""Synthesise the per-sample arrays :func:`summarise_lag_intervention` consumes."""
    n = n_cells
    out = {
        "cell_id": np.arange(n),
        "te_true": np.full(n, float(te)),
        "delay": np.full(n, 5),
        "feat_loss": np.ones(n),
        "delta_L_rel_inband": np.full(n, float(inband_rel)),
        "delta_L_rel_outband": np.full(n, float(outband_rel)),
        "delta_L_inband": np.full(n, float(inband_rel)),
        "delta_L_outband": np.full(n, float(outband_rel)),
        "mass_inband": np.full(n, 0.7),
        "mass_outband": np.full(n, 0.3),
        "n_samples": n,
        "n_lags": _L,
        "noop_max_abs_delta": 0.0,
    }
    # Per-band delta_L and mass: make them agree (or not) across cells so rho is determined.
    for j, name in enumerate(_BANDS):
        delta = np.linspace(0.1, 0.9, n) * (j + 1)
        out[f"delta_L_rel_{name}"] = delta
        out[f"delta_L_{name}"] = delta
        out[f"mass_{name}"] = delta if mass_follows else delta[::-1]
    return out


def test_gate_passes_when_the_true_band_matters_most() -> None:
    r"""Agreeing arrangement: masking :math:`\mathcal{L}^\star` hurts more than masking the rest."""
    summary = li.summarise_lag_intervention(
        _profiles(inband_rel=0.40, outband_rel=0.05), bands=_BANDS, margin=0.0
    )
    overall = summary["overall"]
    assert overall["inband_gate_pass"] is True
    assert overall["inband_gate_pass_frac"] == 1.0
    assert overall["mean_inband_lift"] == pytest.approx(0.35, rel=1e-3)
    lo, hi = overall["inband_lift_ci"]
    assert lo <= overall["mean_inband_lift"] <= hi


def test_gate_fails_when_the_wrong_lags_matter() -> None:
    r"""Disagreeing arrangement: the forecast leans on :math:`\{\ell \ge D\}`."""
    summary = li.summarise_lag_intervention(
        _profiles(inband_rel=0.02, outband_rel=0.30), bands=_BANDS, margin=0.0
    )
    overall = summary["overall"]
    assert overall["inband_gate_pass"] is False
    assert overall["inband_gate_pass_frac"] == 0.0
    assert overall["mean_inband_lift"] < 0.0


def test_margin_is_applied_strictly() -> None:
    r"""A lift equal to the margin does not pass: the inequality is strict."""
    prof = _profiles(inband_rel=0.10, outband_rel=0.00)
    assert li.summarise_lag_intervention(prof, bands=_BANDS, margin=0.10)["overall"][
        "inband_gate_pass"
    ] is False
    assert li.summarise_lag_intervention(prof, bands=_BANDS, margin=0.05)["overall"][
        "inband_gate_pass"
    ] is True


def test_null_cells_are_never_gated() -> None:
    r"""A :math:`\mathrm{TE}_{\mathrm{inj}} = 0` cell has no true lag, so no verdict."""
    summary = li.summarise_lag_intervention(
        _profiles(inband_rel=0.0, outband_rel=0.4, te=0.0), bands=_BANDS, margin=0.0
    )
    for cell in summary["per_cell"].values():
        assert cell["is_signal"] is False
        assert cell["inband_gate_pass"] is None
        assert cell["inband_lift"] is None
    assert summary["overall"]["n_signal_cells"] == 0
    assert summary["overall"]["inband_gate_pass_frac"] is None


def test_rho_is_reported_not_gated_below_eight_cells() -> None:
    summary = li.summarise_lag_intervention(
        _profiles(inband_rel=0.4, outband_rel=0.1, n_cells=3), bands=_BANDS, margin=0.0
    )
    assert summary["rho_reported_not_gated"] is True
    for entry in summary["rho_by_band"].values():
        assert entry["gated"] is False
        assert entry["n_cells"] == 3


def test_rho_tracks_agreement_between_intervention_and_attention() -> None:
    r"""Attention mass following :math:`\Delta L` gives :math:`\rho = +1`; reversing it, :math:`-1`."""
    agree = li.summarise_lag_intervention(
        _profiles(inband_rel=0.4, outband_rel=0.1, n_cells=10, mass_follows=True),
        bands=_BANDS, margin=0.0, seed=0,
    )
    disagree = li.summarise_lag_intervention(
        _profiles(inband_rel=0.4, outband_rel=0.1, n_cells=10, mass_follows=False),
        bands=_BANDS, margin=0.0, seed=0,
    )
    for name in _BANDS:
        assert agree["rho_by_band"][name]["rho"] == pytest.approx(1.0, abs=1e-9)
        assert agree["rho_by_band"][name]["gated"] is True
        assert disagree["rho_by_band"][name]["rho"] == pytest.approx(-1.0, abs=1e-9)
    assert agree["rho_reported_not_gated"] is False


def test_summary_without_cell_id_degrades() -> None:
    assert "error" in li.summarise_lag_intervention({}, bands=_BANDS)


# ---------------------------------------------------------------------------
# Stage / section / figure wiring.
# ---------------------------------------------------------------------------
def test_stage_is_registered_opt_in_and_non_fatal() -> None:
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import run_pipeline_v2 as rp

    spec = rp._STAGE_REGISTRY["lag_intervention"]
    assert spec.default_on is False, "the stage costs |G|+1 forwards/batch; it must be opt-in"
    assert spec.fatal is False
    assert spec.model_dependent is True


def test_report_section_renders_na_without_the_json(tmp_path) -> None:
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import final_report_v2 as fr

    ctx = fr.SectionContext(config={}, benchmark="G1_raw", results_dir=tmp_path)
    lines = li._render_lag_intervention_section(ctx)
    assert any("n/a" in ln for ln in lines)


def test_report_section_renders_the_gate(tmp_path) -> None:
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import final_report_v2 as fr

    summary = li.summarise_lag_intervention(
        _profiles(inband_rel=0.4, outband_rel=0.1, n_cells=10), bands=_BANDS, margin=0.0
    )
    summary.update(arm="v3_prod", split="val", n_samples_skipped=0, noop_atol=1e-6)
    (tmp_path / "lag_intervention.json").write_text(json.dumps(summary), encoding="utf-8")

    ctx = fr.SectionContext(config={}, benchmark="G1_raw", results_dir=tmp_path)
    text = "\n".join(li._render_lag_intervention_section(ctx))
    assert "Interventional lag attribution" in text
    assert "in-band gate" in text
    assert "0-2" in text, "the rho-by-band table did not render"


def test_figure_renders_from_a_synthetic_summary(tmp_path) -> None:
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import visualize_v2 as viz

    summary = li.summarise_lag_intervention(
        _profiles(inband_rel=0.4, outband_rel=0.1, n_cells=4), bands=_BANDS, margin=0.0
    )
    summary.update(arm="v3_prod", split="val")
    written = viz.plot_lag_intervention(summary, tmp_path / "lag_intervention")
    assert written and all(p.is_file() and p.stat().st_size > 0 for p in written)


def test_figure_handles_an_empty_summary(tmp_path) -> None:
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import visualize_v2 as viz

    written = viz.plot_lag_intervention({}, tmp_path / "empty")
    assert written and all(p.is_file() for p in written)
