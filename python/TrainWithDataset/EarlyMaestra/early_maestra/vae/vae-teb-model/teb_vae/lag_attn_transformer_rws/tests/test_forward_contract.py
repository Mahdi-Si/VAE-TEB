r"""The forward return contract: the exact key set, the shapes, and what the driver must reach.

The key set is asserted by equality, not by subset: an extra key is how a bypass tensor would
first appear, and a missing one is how a downstream consumer starts reading defaults.

Three of the assertions here are about *silent* failures rather than shapes. The experiment
driver builds constructor kwargs by an ``inspect.signature`` sweep and drops anything the
signature does not name, so a renamed channel tuple would make a resolved reach budget vanish with
no error at all. The diagnostic figure reads the source delay through
``getattr(model, "source_delay_steps", 0)``, so a model that stopped exposing it would shift the
figure's lag axis by up to two minutes and report nothing. And the adapters' availability patterns
must come from the gate the forward actually applies, not from a second reading of the same
constructor arguments.
"""
from __future__ import annotations

import inspect

import pytest
import torch

from teb_vae.lag_attn_rws.channel_reach import resolve_stream_budgets
from teb_vae.lag_attn_rws.trainer import _CHANNEL_TUPLE_KEYS, _NON_CONSTRUCTOR_KEYS
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws
from teb_vae.lag_attn_transformer_rws.tests.conftest import BATCH, SEQ_LEN

_DOCUMENTED_KEYS = {
    "mu_prior",
    "logvar_prior",
    "raw_logvar_prior",
    "mu_post",
    "logvar_post",
    "z_prior",
    "z_post",
    "target_state",
    "source_state",
    "attended_source_heads",
    "attn_weights",
    "mu_base",
    "logvar_base",
    "mu_full",
    "logvar_full",
    "kld_per_t",
    "kld_per_t_per_head",
    "source_kl_lag_map",
    "mu_prior_sat_frac",
    "delta_mu_sat_frac",
}

#: The budget the guarded probes resolve, and the delay it produces. The same pair the sibling's
#: lag-consistency test pins, because it is the shipped guarded configuration.
_BUDGET_S, _EXPECTED_DELAY = 120.0, 30


def _forward(tiny_kwargs, inputs, perturb=None):
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfRws(**tiny_kwargs).eval()
    if perturb is not None:
        perturb(model)
    torch.manual_seed(0)
    with torch.no_grad():
        return model, model(*inputs)


def guarded_kwargs(tiny_kwargs: dict) -> dict:
    """The tiny keyword set carrying the production reach budget's resolved channel tuples.

    Resolved rather than hand-written: the tuples are hundreds of entries long and are produced by
    the same function the experiment driver calls, so a probe built on a hand-made pair would test
    a guard no run can ask for. The sequence and warm-up lengths grow to $64$ and $30$ because the
    budget's own resolution refuses a delay longer than the warm-up.

    Args:
        tiny_kwargs: The tiny constructor keyword set.

    Returns:
        A new keyword set with the four channel tuples merged in.
    """
    budget = resolve_stream_budgets(
        {
            "causal_reach_budget_s": _BUDGET_S,
            "use_up_st": True,
            "warmup_period": 30,
            "c_y": int(tiny_kwargs["c_y"]),
            "c_u": int(tiny_kwargs["c_u"]),
        }
    )
    assert budget is not None
    return dict(
        tiny_kwargs,
        sequence_length=64,
        warmup_period=30,
        target_keep_index=budget.target_keep_index,
        target_delays=budget.target_delays,
        source_keep_index=budget.source_keep_index,
        source_delays=budget.source_delays,
    )


# ---------------------------------------------------------------------------------------
# The key set and the shapes
# ---------------------------------------------------------------------------------------
def test_the_forward_returns_exactly_the_documented_key_set(tiny_kwargs, inputs):
    _, out = _forward(tiny_kwargs, inputs)
    assert set(out.keys()) == _DOCUMENTED_KEYS
    # The two pathways this architecture never had must not appear under the sibling's old names.
    assert "decoder_state" not in out
    assert "delta_mu_src" not in out


def test_the_latent_and_state_shapes(tiny_kwargs, inputs):
    model, out = _forward(tiny_kwargs, inputs)
    for key in ("mu_prior", "logvar_prior", "raw_logvar_prior", "mu_post", "logvar_post",
                "z_prior", "z_post"):
        assert out[key].shape == (BATCH, SEQ_LEN, model.d_z), key
    for key in ("target_state", "source_state"):
        assert out[key].shape == (BATCH, SEQ_LEN, model.d_model), key


def test_the_attention_shapes(tiny_kwargs, inputs):
    model, out = _forward(tiny_kwargs, inputs)
    num_lags = model.max_lag + 1
    d_head = model.d_model // model.num_heads
    assert out["attn_weights"].shape == (BATCH, SEQ_LEN, model.num_heads, num_lags)
    assert out["attended_source_heads"].shape == (BATCH, SEQ_LEN, model.num_heads, d_head)


def test_the_kl_readout_shapes(tiny_kwargs, inputs):
    model, out = _forward(tiny_kwargs, inputs)
    num_lags = model.max_lag + 1
    assert out["kld_per_t"].shape == (BATCH, SEQ_LEN)
    assert out["kld_per_t_per_head"].shape == (BATCH, SEQ_LEN, model.num_heads)
    assert out["source_kl_lag_map"].shape == (BATCH, SEQ_LEN, num_lags)


def test_decoding_covers_the_valid_anchor_range_only(tiny_kwargs, inputs):
    """(B, T - H, H, R), not (B, T, H, R): the tail anchors are never decoded."""
    model, out = _forward(tiny_kwargs, inputs)
    expected = (BATCH, model.geometry.t_valid, model.horizon, model.raw_per_step)
    assert expected[1] == SEQ_LEN - model.horizon
    for key in ("mu_base", "logvar_base", "mu_full", "logvar_full"):
        assert out[key].shape == expected, key


def test_one_epsilon_serves_both_latents_when_the_residual_is_zero(tiny_kwargs, inputs):
    """At init q == p, so the shared draw makes the samples bitwise equal."""
    _, out = _forward(tiny_kwargs, inputs)
    assert torch.equal(out["z_prior"], out["z_post"])


def test_one_epsilon_serves_both_latents_when_the_distributions_differ(
    tiny_kwargs, inputs, perturb_posterior
):
    """The stronger claim: even off-init, both samples recover the *same* epsilon. Two independent
    draws would pass the at-init test above and still corrupt every base-minus-full readout with
    sampling noise."""
    _, out = _forward(tiny_kwargs, inputs, perturb=perturb_posterior)
    assert not torch.equal(out["mu_post"], out["mu_prior"])  # genuinely off-init

    eps_prior = (out["z_prior"] - out["mu_prior"]) * torch.exp(-0.5 * out["logvar_prior"])
    eps_post = (out["z_post"] - out["mu_post"]) * torch.exp(-0.5 * out["logvar_post"])
    assert torch.allclose(eps_prior, eps_post, atol=1e-5)


def test_the_saturation_diagnostics_are_scalars_in_unit_range(tiny_kwargs, inputs):
    _, out = _forward(tiny_kwargs, inputs)
    for key in ("mu_prior_sat_frac", "delta_mu_sat_frac"):
        assert out[key].dim() == 0
        assert 0.0 <= float(out[key]) <= 1.0


# ---------------------------------------------------------------------------------------
# The constructor surface the driver reaches through
# ---------------------------------------------------------------------------------------
def test_the_signature_names_every_channel_tuple_the_driver_injects():
    """The driver's ``inspect.signature`` sweep drops unknown keys *silently*, so a renamed or
    dropped channel tuple would make a configured reach budget resolve, be injected, and land
    nowhere -- a model built with no guard at all, reported as guarded in its own run record."""
    parameters = set(inspect.signature(SeqVaeLagAttnTrfRws.__init__).parameters)

    assert _CHANNEL_TUPLE_KEYS <= parameters, (
        f"the driver injects {sorted(_CHANNEL_TUPLE_KEYS)} and this constructor is missing "
        f"{sorted(_CHANNEL_TUPLE_KEYS - parameters)}"
    )


def test_init_weights_is_a_constructor_argument_the_driver_never_forwards():
    """It stays in the signature -- a test builds an uninitialised model with it -- while the
    driver's exclusion list keeps it out of config: weight initialisation is not a config
    decision."""
    parameters = set(inspect.signature(SeqVaeLagAttnTrfRws.__init__).parameters)

    assert "init_weights" in parameters
    assert "init_weights" in _NON_CONSTRUCTOR_KEYS


def test_a_copy_pasted_sibling_config_key_is_refused(tiny_kwargs):
    """``lstm_layers`` means nothing here. Refused loudly rather than absorbed by a ``**kwargs``,
    which is what would let a hand-copied config silently build a different model."""
    with pytest.raises(TypeError, match="lstm_layers"):
        SeqVaeLagAttnTrfRws(**dict(tiny_kwargs, lstm_layers=2))


@pytest.mark.parametrize(
    "key", ["encoder_extra_dilations", "encoder_extra_kernel", "conv_norm_groups", "causal_norm"]
)
def test_the_other_sibling_only_keys_are_refused(tiny_kwargs, key):
    """There is no extra dilation schedule, no convolution pre-norm and no time-pooling normaliser
    left to causalise, so each of these would reach nothing if it were quietly accepted."""
    with pytest.raises(TypeError, match=key):
        SeqVaeLagAttnTrfRws(**dict(tiny_kwargs, **{key: 1}))


# ---------------------------------------------------------------------------------------
# The guarded surface
# ---------------------------------------------------------------------------------------
def test_an_unguarded_model_reports_no_source_delay(tiny_kwargs):
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfRws(**tiny_kwargs)

    assert model.target_gate is None and model.source_gate is None
    assert model.source_delay_steps == 0


def test_the_model_reports_its_own_source_delay(tiny_kwargs):
    """Read by the diagnostic figure through a silent ``getattr`` default, so a model that stopped
    exposing it would mis-annotate the lag axis by up to two minutes with nothing failing."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfRws(**guarded_kwargs(tiny_kwargs))

    assert model.source_gate is not None
    assert model.source_delay_steps == model.source_gate.max_delay == _EXPECTED_DELAY


def test_the_adapters_are_built_from_their_own_gates(tiny_kwargs):
    """Width and availability pattern both read off the gate object the forward actually applies.

    A second reading of the constructor arguments would be a second source of truth, and the gate
    fills in a missing keep-index or a missing delay vector itself -- so the two could disagree
    about a stream that was still built and still trained.
    """
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfRws(**guarded_kwargs(tiny_kwargs))

    for gate, adapter in (
        (model.target_gate, model.target_adapter),
        (model.source_gate, model.source_adapter),
    ):
        assert gate is not None
        assert adapter.linear.in_features == gate.out_channels
        assert adapter.availability.shape == (model.sequence_length, gate.out_channels)
        expected = (
            torch.arange(model.sequence_length).unsqueeze(-1) >= gate.delay.delay_steps
        ).to(adapter.availability.dtype)
        assert torch.equal(adapter.availability, expected)


def test_an_unguarded_adapter_carries_no_availability_terms(tiny_kwargs):
    """The same convention the gate itself uses: the unguarded case is represented by not having
    the thing, so nothing inert appears in a parameter or buffer listing."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfRws(**tiny_kwargs)

    for adapter in (model.target_adapter, model.source_adapter):
        assert adapter.mask_proj is None and adapter.start_embed is None
        assert not [name for name, _ in adapter.named_buffers() if "availability" in name]


def test_a_guarded_forward_returns_the_same_contract(tiny_kwargs):
    """The guard changes what the model reads, not what it returns."""
    kwargs = guarded_kwargs(tiny_kwargs)
    seq_len = int(kwargs["sequence_length"])
    generator = torch.Generator().manual_seed(0)
    guarded_inputs = (
        torch.randn(BATCH, seq_len, 43, generator=generator),
        torch.randn(BATCH, seq_len, 66, generator=generator),
        torch.randn(BATCH, seq_len, 58, generator=generator),
    )
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfRws(**kwargs).eval()

    with torch.no_grad():
        out = model(*guarded_inputs)

    assert set(out) == _DOCUMENTED_KEYS
    assert out["target_state"].shape == (BATCH, seq_len, model.d_model)
    assert torch.isfinite(out["mu_prior"]).all()
