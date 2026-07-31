r"""The source pathway never sees the target, the prior never sees the source, and both are causal.

$\mathrm{KL}(q \Vert p)$ reads as "what the source added" only if $p$, and everything decoded from
$z^p$, is a function of the target's history alone. So replacing the source stream with noise must
leave the prior *and the base forecast* bitwise unchanged -- and, from the other side, the source
encoder must be handed nothing derived from the target.

Three probes, because each catches a different failure. Bitwise equality under resampling catches a
pathway that is connected and used. An autograd probe catches one that is connected but happens to
contribute nothing on this input. Forward pre-hooks asserting **object identity** at the adapters
catch a concatenation that mixed a stream in before either of the other two could see it.

The last section runs the positional causality probe on the *assembled* model rather than on a bare
encoder: through the channel gate, the availability-aware adapter and the prior head, on
``mu_prior`` and ``target_state``. The encoder-level test cannot cover that composition, and the
gate in particular is an index operation whose failure mode -- a delay applied along the wrong axis
-- is invisible to every shape check.

The bitwise comparisons work because the model is run in ``eval()`` with the generator re-seeded
before each forward: the single ``randn_like`` draw is then the only RNG consumer, so both runs
share their $\epsilon$.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws
from teb_vae.lag_attn_transformer_rws.tests.conftest import (
    CAUSALITY_TOL,
    MOVEMENT_TOL,
    SEQ_LEN,
    relative_change,
    resample_after,
)

#: Target-only quantities. Every one of these must be bitwise unmoved by any source at all.
_TARGET_ONLY_KEYS = (
    "mu_prior",
    "logvar_prior",
    "raw_logvar_prior",
    "target_state",
    "z_prior",
    "mu_base",
    "logvar_base",
)


def _model(tiny_kwargs, cls=SeqVaeLagAttnTrfRws) -> SeqVaeLagAttnTrfRws:
    torch.manual_seed(0)
    return cls(**tiny_kwargs).eval()


class CrossWiredModel(SeqVaeLagAttnTrfRws):
    """Deliberately broken: each stream is leaked into the other before the real forward runs.

    The negative control for this whole file. Every assertion here is of the form "this pathway did
    *not* move", which a model that computed nothing at all would also satisfy; a model that mixes
    the streams is what those assertions have to be able to catch.
    """

    def forward(self, y_st, y_ph, u_stream):  # type: ignore[override]
        """Leak the target into the source stream and the source into the target's, then forward."""
        into_source = torch.cat([y_st, y_ph], dim=-1)[..., : u_stream.shape[-1]]
        into_target = u_stream[..., :1]
        return super().forward(y_st + into_target, y_ph, u_stream + into_source)


# ---------------------------------------------------------------------------------------
# The source cannot reach the prior
# ---------------------------------------------------------------------------------------
def test_resampling_the_source_leaves_the_prior_and_base_forecast_unchanged(
    tiny_kwargs, inputs
):
    model = _model(tiny_kwargs)
    y_st, y_ph, u_stream = inputs

    torch.manual_seed(0)
    with torch.no_grad():
        base = model(y_st, y_ph, u_stream)
    noise_u = torch.randn(u_stream.shape, generator=torch.Generator().manual_seed(99))
    torch.manual_seed(0)
    with torch.no_grad():
        resampled = model(y_st, y_ph, noise_u)

    for key in _TARGET_ONLY_KEYS:
        assert torch.equal(base[key], resampled[key]), key
    # And the source pathway did notice the change -- otherwise the test proves nothing.
    assert not torch.equal(base["source_state"], resampled["source_state"])


def test_no_gradient_path_runs_from_the_source_stream_to_the_prior(tiny_kwargs, inputs):
    """The stronger claim. Bitwise equality on one input is consistent with a pathway that exists
    and contributed zero; an autograd probe asks whether the pathway is in the graph at all."""
    model = _model(tiny_kwargs)
    y_st, y_ph, u_stream = inputs
    u_stream = u_stream.clone().requires_grad_(True)

    out = model(y_st, y_ph, u_stream)
    for key in ("mu_prior", "logvar_prior", "target_state", "mu_base"):
        (grad,) = torch.autograd.grad(
            out[key].sum(), u_stream, retain_graph=True, allow_unused=True
        )
        assert grad is None, f"{key} is differentiable with respect to the source stream"


def test_the_same_probe_finds_the_source_on_a_source_driven_quantity(tiny_kwargs, inputs):
    """The positive direction, so the ``grad is None`` assertions above are not vacuous."""
    model = _model(tiny_kwargs)
    y_st, y_ph, u_stream = inputs
    u_stream = u_stream.clone().requires_grad_(True)

    out = model(y_st, y_ph, u_stream)
    (grad,) = torch.autograd.grad(out["source_state"].sum(), u_stream, allow_unused=True)

    assert grad is not None and float(grad.abs().max()) > 0.0


# ---------------------------------------------------------------------------------------
# The target cannot reach the source state
# ---------------------------------------------------------------------------------------
def test_resampling_the_target_leaves_the_source_state_unchanged(tiny_kwargs, inputs):
    model = _model(tiny_kwargs)
    y_st, y_ph, u_stream = inputs
    generator = torch.Generator().manual_seed(41)

    torch.manual_seed(0)
    with torch.no_grad():
        base = model(y_st, y_ph, u_stream)
    torch.manual_seed(0)
    with torch.no_grad():
        resampled = model(
            torch.randn(y_st.shape, generator=generator),
            torch.randn(y_ph.shape, generator=generator),
            u_stream,
        )

    assert torch.equal(base["source_state"], resampled["source_state"])
    assert not torch.equal(base["target_state"], resampled["target_state"])


def test_no_gradient_path_runs_from_the_target_stream_to_the_source_state(tiny_kwargs, inputs):
    model = _model(tiny_kwargs)
    y_st, y_ph, u_stream = inputs
    y_st = y_st.clone().requires_grad_(True)
    y_ph = y_ph.clone().requires_grad_(True)

    out = model(y_st, y_ph, u_stream)
    grads = torch.autograd.grad(
        out["source_state"].sum(), [y_st, y_ph], allow_unused=True
    )

    assert all(grad is None for grad in grads)


# ---------------------------------------------------------------------------------------
# What each adapter is handed, by identity
# ---------------------------------------------------------------------------------------
def _capture_adapter_inputs(model, inputs):
    """Run one forward with pre-hooks on both adapters, returning what each received."""
    seen: dict[str, list[torch.Tensor]] = {"source": [], "target": []}
    handles = [
        model.source_adapter.register_forward_pre_hook(
            lambda module, args: seen["source"].append(args[0])
        ),
        model.target_adapter.register_forward_pre_hook(
            lambda module, args: seen["target"].append(args[0])
        ),
    ]
    try:
        with torch.no_grad():
            model(*inputs)
    finally:
        for handle in handles:
            handle.remove()
    return seen


def test_the_source_adapter_receives_the_source_object_itself(tiny_kwargs, inputs):
    """Instrumented at the adapters -- the trust boundary where raw streams enter. Identity, not
    equality: an equal tensor could be the output of a concatenation that mixed a target block in
    and then happened to agree on this input."""
    model = _model(tiny_kwargs)
    seen = _capture_adapter_inputs(model, inputs)

    assert len(seen["source"]) == 1
    assert seen["source"][0] is inputs[2]


def test_the_target_adapter_receives_only_the_concatenated_target_features(tiny_kwargs, inputs):
    model = _model(tiny_kwargs)
    seen = _capture_adapter_inputs(model, inputs)

    assert len(seen["target"]) == 1
    assert torch.equal(seen["target"][0], torch.cat([inputs[0], inputs[1]], dim=-1))
    # Its width is c_y, which cannot hold an extra c_u block.
    assert seen["target"][0].shape[-1] == model.c_y


def test_the_source_encoder_consumes_only_the_source_adapter_output(tiny_kwargs, inputs):
    """One step deeper: what reaches the source encoder is the adapter's projection of the source
    stream, so no target tensor can join between adapter and encoder."""
    model = _model(tiny_kwargs)
    captured: list[torch.Tensor] = []
    adapter_out: list[torch.Tensor] = []

    handles = [
        model.source_adapter.register_forward_hook(
            lambda module, args, output: adapter_out.append(output)
        ),
        model.source_encoder.register_forward_pre_hook(
            lambda module, args: captured.append(args[0])
        ),
    ]
    try:
        with torch.no_grad():
            model(*inputs)
    finally:
        for handle in handles:
            handle.remove()

    assert len(captured) == 1 and len(adapter_out) == 1
    assert captured[0] is adapter_out[0]


def test_the_two_encoders_share_no_parameter_tensor(tiny_kwargs):
    """Separate instances, not one module used twice: a shared encoder would make the source state
    a function of the target and every purity assertion above would be about the same tensor."""
    model = _model(tiny_kwargs)
    target_ids = {id(parameter) for parameter in model.target_encoder.parameters()}
    source_ids = {id(parameter) for parameter in model.source_encoder.parameters()}

    assert target_ids and source_ids
    assert target_ids.isdisjoint(source_ids)


# ---------------------------------------------------------------------------------------
# The assembled model's causality
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("cut", [0, 1, SEQ_LEN - 2])
def test_the_assembled_prior_reads_only_the_targets_past(tiny_kwargs, inputs, cut):
    r"""$H_t = f(X_{\le t})$ through the gate, the adapter, the encoder and the prior head.

    The paired probe: the outputs at the cut must be bit-stable while the outputs at the last step
    must move. The second half is the negative control -- this architecture has no time-pooling
    normaliser to flip, so the control is positional rather than a switch that exists only for
    tests.
    """
    model = _model(tiny_kwargs)
    y_st, y_ph, u_stream = inputs

    torch.manual_seed(0)
    with torch.no_grad():
        reference = model(y_st, y_ph, u_stream)
    torch.manual_seed(0)
    with torch.no_grad():
        perturbed = model(
            resample_after(y_st, cut, seed=5), resample_after(y_ph, cut, seed=6), u_stream
        )

    for key in ("mu_prior", "target_state"):
        at_cut = relative_change(reference[key][:, cut], perturbed[key][:, cut])
        assert at_cut < CAUSALITY_TOL, (
            f"{key} at t={cut} moved by {at_cut:.3e} when only the strict future changed"
        )
    at_end = relative_change(reference["target_state"][:, -1], perturbed["target_state"][:, -1])
    assert at_end > MOVEMENT_TOL, (
        f"the target state at the last step moved by only {at_end:.3e} -- the perturbation never "
        f"reached the model, so the bit-stability above proves nothing"
    )


# ---------------------------------------------------------------------------------------
# The negative control: a model that genuinely mixes the streams
# ---------------------------------------------------------------------------------------
def test_a_cross_wired_model_fails_the_bitwise_purity_assertions(tiny_kwargs, inputs):
    model = _model(tiny_kwargs, cls=CrossWiredModel)
    y_st, y_ph, u_stream = inputs
    noise_u = torch.randn(u_stream.shape, generator=torch.Generator().manual_seed(99))

    torch.manual_seed(0)
    with torch.no_grad():
        base = model(y_st, y_ph, u_stream)
    torch.manual_seed(0)
    with torch.no_grad():
        resampled = model(y_st, y_ph, noise_u)

    assert not torch.equal(base["mu_prior"], resampled["mu_prior"])
    assert not torch.equal(base["target_state"], resampled["target_state"])


def test_a_cross_wired_model_fails_the_autograd_probe(tiny_kwargs, inputs):
    model = _model(tiny_kwargs, cls=CrossWiredModel)
    y_st, y_ph, u_stream = inputs
    u_stream = u_stream.clone().requires_grad_(True)

    out = model(y_st, y_ph, u_stream)
    (grad,) = torch.autograd.grad(out["mu_prior"].sum(), u_stream, allow_unused=True)

    assert grad is not None


def test_a_cross_wired_model_fails_the_adapter_identity_hook(tiny_kwargs, inputs):
    model = _model(tiny_kwargs, cls=CrossWiredModel)
    seen = _capture_adapter_inputs(model, inputs)

    assert seen["source"][0] is not inputs[2]
