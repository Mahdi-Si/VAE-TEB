r"""The source pathway never sees the target, and the prior never sees the source.

$\mathrm{KL}(q \Vert p)$ reads as "what the source added" only if $p$, and everything decoded from
$z^p$, is a function of the target's history alone. So replacing the source signal with noise must
leave the prior *and the base forecast* bitwise unchanged -- and, from the other side, the source
front end must be handed nothing derived from the target.

This file is load-bearing here in a way it is not for the model this one is compared with, and the
reason is the input representation itself. There, the two streams have different widths -- $109$
target channels against $58$ source ones -- so a swapped or crossed pair fails a shape check on the
first forward. Here both inputs are $(B, 4800)$ raw signals. A transposed argument pair, a front end
handed the wrong tensor, a copy-paste that ran the target front end twice: every one of those
produces correctly shaped output and a plausible loss curve, and the *only* thing that notices is a
purity probe. The identity hooks below are what turn "the numbers moved" into "this specific tensor
reached this specific module".

Three probes, because each catches a different failure. Bitwise equality under resampling catches a
pathway that is connected and used. An autograd probe catches one that is connected but happens to
contribute nothing on this input. Forward pre-hooks asserting **object identity** at the two front
ends catch a swap before either of the other two could see it.

The bitwise comparisons work because the model is run in ``eval()`` with the generator re-seeded
before each forward: the single ``randn_like`` draw is then the only RNG consumer, so both runs
share their $\epsilon$.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E

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


def _model(tiny_kwargs, cls=SeqVaeLagAttnTrfE2E) -> SeqVaeLagAttnTrfE2E:
    torch.manual_seed(0)
    return cls(**tiny_kwargs).eval()


class CrossWiredModel(SeqVaeLagAttnTrfE2E):
    """Deliberately broken: each stream is leaked into the other before the real forward runs.

    The negative control for this whole file. Every assertion here is of the form "this pathway did
    *not* move", which a model that computed nothing at all would also satisfy; a model that mixes
    the streams is what those assertions have to be able to catch.

    A tenth of the other signal, rather than the other signal outright: a full swap would be caught
    by the identity hooks alone, and the point is that the resampling and autograd probes catch a
    *partial* contamination too.
    """

    def forward(self, y_raw, u_raw, weight):  # type: ignore[override]
        """Leak a tenth of each stream into the other, then forward."""
        return super().forward(y_raw + 0.1 * u_raw, u_raw + 0.1 * y_raw, weight)


class SwappedModel(SeqVaeLagAttnTrfE2E):
    """Deliberately broken in the way only this input representation can be: the two raw signals
    are transposed on their way into the front ends.

    Both are ``(B, 4800)``, so nothing about the shapes objects. In the sibling this mistake cannot
    survive the first forward; here it produces a model that trains, converges, and reports the
    source-conditioned KL of the target against itself.
    """

    def forward(self, y_raw, u_raw, weight):  # type: ignore[override]
        """Forward with the two raw signals exchanged."""
        return super().forward(u_raw, y_raw, weight)


# ---------------------------------------------------------------------------------------
# The source cannot reach the prior
# ---------------------------------------------------------------------------------------
def test_resampling_the_source_leaves_the_prior_and_base_forecast_unchanged(
    tiny_kwargs, raw_inputs
):
    model = _model(tiny_kwargs)
    y_raw, u_raw, weight = raw_inputs

    torch.manual_seed(0)
    with torch.no_grad():
        base = model(y_raw, u_raw, weight)
    noise_u = torch.randn(u_raw.shape, generator=torch.Generator().manual_seed(99))
    torch.manual_seed(0)
    with torch.no_grad():
        resampled = model(y_raw, noise_u, weight)

    for key in _TARGET_ONLY_KEYS:
        assert torch.equal(base[key], resampled[key]), key
    # And the source pathway did notice the change -- otherwise the test proves nothing.
    assert not torch.equal(base["source_state"], resampled["source_state"])


def test_no_gradient_path_runs_from_the_source_signal_to_the_prior(tiny_kwargs, raw_inputs):
    """The stronger claim. Bitwise equality on one input is consistent with a pathway that exists and
    contributed zero; an autograd probe asks whether the pathway is in the graph at all."""
    model = _model(tiny_kwargs)
    y_raw, u_raw, weight = raw_inputs
    u_raw = u_raw.clone().requires_grad_(True)

    out = model(y_raw, u_raw, weight)
    for key in ("mu_prior", "logvar_prior", "target_state", "mu_base"):
        (grad,) = torch.autograd.grad(
            out[key].sum(), u_raw, retain_graph=True, allow_unused=True
        )
        assert grad is None, f"{key} is differentiable with respect to the source signal"


def test_the_same_probe_finds_the_source_on_a_source_driven_quantity(tiny_kwargs, raw_inputs):
    """The positive direction, so the ``grad is None`` assertions above are not vacuous."""
    model = _model(tiny_kwargs)
    y_raw, u_raw, weight = raw_inputs
    u_raw = u_raw.clone().requires_grad_(True)

    out = model(y_raw, u_raw, weight)
    (grad,) = torch.autograd.grad(out["source_state"].sum(), u_raw, allow_unused=True)

    assert grad is not None and float(grad.abs().max()) > 0.0


# ---------------------------------------------------------------------------------------
# The target cannot reach the source state
# ---------------------------------------------------------------------------------------
def test_resampling_the_target_leaves_the_source_state_unchanged(tiny_kwargs, raw_inputs):
    model = _model(tiny_kwargs)
    y_raw, u_raw, weight = raw_inputs
    noise_y = torch.randn(y_raw.shape, generator=torch.Generator().manual_seed(41))

    torch.manual_seed(0)
    with torch.no_grad():
        base = model(y_raw, u_raw, weight)
    torch.manual_seed(0)
    with torch.no_grad():
        resampled = model(noise_y, u_raw, weight)

    assert torch.equal(base["source_state"], resampled["source_state"])
    assert not torch.equal(base["target_state"], resampled["target_state"])


def test_no_gradient_path_runs_from_the_target_signal_to_the_source_state(tiny_kwargs, raw_inputs):
    model = _model(tiny_kwargs)
    y_raw, u_raw, weight = raw_inputs
    y_raw = y_raw.clone().requires_grad_(True)

    out = model(y_raw, u_raw, weight)
    (grad,) = torch.autograd.grad(out["source_state"].sum(), y_raw, allow_unused=True)

    assert grad is None


# ---------------------------------------------------------------------------------------
# What each front end is handed, by identity
# ---------------------------------------------------------------------------------------
def _capture_frontend_inputs(model, inputs):
    """Run one forward with pre-hooks on both front ends, returning what each received."""
    seen: dict[str, list[torch.Tensor]] = {"source": [], "target": []}
    handles = [
        model.source_frontend.register_forward_pre_hook(
            lambda module, args: seen["source"].append(args[0])
        ),
        model.target_frontend.register_forward_pre_hook(
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


def test_each_front_end_receives_its_own_signal_object(tiny_kwargs, raw_inputs):
    """Instrumented at the front ends -- the trust boundary where the raw signals enter. Identity,
    not equality: with two same-shaped raw tensors an *equal* tensor could be the wrong one, and this
    is the assertion a transposed argument pair cannot survive."""
    model = _model(tiny_kwargs)
    seen = _capture_frontend_inputs(model, raw_inputs)

    assert len(seen["target"]) == 1 and len(seen["source"]) == 1
    assert seen["target"][0] is raw_inputs[0]
    assert seen["source"][0] is raw_inputs[1]


def test_the_source_encoder_consumes_only_the_source_front_ends_output(tiny_kwargs, raw_inputs):
    """One step deeper: what reaches the source encoder is the source front end's own output, so no
    target tensor can join between the two."""
    model = _model(tiny_kwargs)
    captured: list[torch.Tensor] = []
    frontend_out: list[torch.Tensor] = []

    handles = [
        model.source_frontend.register_forward_hook(
            lambda module, args, output: frontend_out.append(output)
        ),
        model.source_encoder.register_forward_pre_hook(
            lambda module, args: captured.append(args[0])
        ),
    ]
    try:
        with torch.no_grad():
            model(*raw_inputs)
    finally:
        for handle in handles:
            handle.remove()

    assert len(captured) == 1 and len(frontend_out) == 1
    assert captured[0] is frontend_out[0]


def test_the_two_front_ends_and_the_two_encoders_share_no_parameter_tensor(tiny_kwargs):
    """Separate instances, not one module used twice: a shared front end or encoder would make the
    source state a function of the target and every purity assertion above would be about the same
    tensor."""
    model = _model(tiny_kwargs)

    for target, source in (
        (model.target_frontend, model.source_frontend),
        (model.target_encoder, model.source_encoder),
    ):
        target_ids = {id(parameter) for parameter in target.parameters()}
        source_ids = {id(parameter) for parameter in source.parameters()}
        assert target_ids and source_ids
        assert target_ids.isdisjoint(source_ids)


# ---------------------------------------------------------------------------------------
# The negative controls
# ---------------------------------------------------------------------------------------
def test_a_cross_wired_model_fails_the_bitwise_purity_assertions(tiny_kwargs, raw_inputs):
    model = _model(tiny_kwargs, cls=CrossWiredModel)
    y_raw, u_raw, weight = raw_inputs
    noise_u = torch.randn(u_raw.shape, generator=torch.Generator().manual_seed(99))

    torch.manual_seed(0)
    with torch.no_grad():
        base = model(y_raw, u_raw, weight)
    torch.manual_seed(0)
    with torch.no_grad():
        resampled = model(y_raw, noise_u, weight)

    assert not torch.equal(base["mu_prior"], resampled["mu_prior"])
    assert not torch.equal(base["target_state"], resampled["target_state"])


def test_a_cross_wired_model_fails_the_autograd_probe(tiny_kwargs, raw_inputs):
    model = _model(tiny_kwargs, cls=CrossWiredModel)
    y_raw, u_raw, weight = raw_inputs
    u_raw = u_raw.clone().requires_grad_(True)

    out = model(y_raw, u_raw, weight)
    (grad,) = torch.autograd.grad(out["mu_prior"].sum(), u_raw, allow_unused=True)

    assert grad is not None


def test_a_cross_wired_model_fails_the_identity_hook(tiny_kwargs, raw_inputs):
    model = _model(tiny_kwargs, cls=CrossWiredModel)
    seen = _capture_frontend_inputs(model, raw_inputs)

    assert seen["target"][0] is not raw_inputs[0]
    assert seen["source"][0] is not raw_inputs[1]


def test_a_swapped_argument_pair_is_caught_by_the_identity_hook_alone(tiny_kwargs, raw_inputs):
    """The failure this file exists for, and the one every other probe in the package would miss.

    A swapped pair keeps both streams pure -- the prior still reads one signal only, the source state
    still reads the other only -- so both bitwise probes and both autograd probes pass. What is wrong
    is *which* signal each read, and only object identity can say so.
    """
    swapped = _model(tiny_kwargs, cls=SwappedModel)
    y_raw, u_raw, weight = raw_inputs

    # Purity holds on the swapped model, which is exactly why the other probes cannot see it: the
    # prior is unmoved by resampling the tensor the *swapped* model treats as its source.
    noise_y = torch.randn(y_raw.shape, generator=torch.Generator().manual_seed(99))
    torch.manual_seed(0)
    with torch.no_grad():
        base = swapped(y_raw, u_raw, weight)
    torch.manual_seed(0)
    with torch.no_grad():
        resampled = swapped(noise_y, u_raw, weight)
    assert torch.equal(base["mu_prior"], resampled["mu_prior"])

    seen = _capture_frontend_inputs(swapped, raw_inputs)
    assert seen["target"][0] is raw_inputs[1]
    assert seen["source"][0] is raw_inputs[0]
