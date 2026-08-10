r"""The feature-domain forecaster: the raw-signal model with its target changed and nothing else.

:class:`SeqVaeLagAttnFs` is a composition, not a copy. Both encoders, the channel gates, the input
adapters, the prior and posterior heads, the lag cross-attention, the shared decoder, the paired
reparameterisation and the seven-term objective are :class:`SeqVaeLagAttnRws`'s, reached by
inheritance rather than by convention -- so a change to any of them lands in both models at once,
and a comparison between them stays a comparison of target domains.

What differs is the target domain, and that lives in
:class:`~teb_vae.lag_attn_fs.nets.feature_target.FeatureForecastTarget`: the decoder's width, the
gathered-never-delayed target block, the four resolved forecast gaps and the delegation that hands
the shared objective both. None of those five members mentions an encoder, which is why they are a
mixin rather than a method of this class -- a second forecaster reaching the same target through
different encoders composes the same two pieces in the same order.

This class body is therefore empty, and that is the load-bearing property rather than an
economy. With nothing defined here, the twenty forward keys, the posterior's structure, the lag
map and the objective's metric set cannot have moved: they are the base's own code objects, pinned
by the base's own suite.
"""
from __future__ import annotations

from teb_vae.lag_attn_fs.nets.feature_target import FeatureForecastTarget
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws


class SeqVaeLagAttnFs(FeatureForecastTarget, SeqVaeLagAttnRws):
    r"""Single-latent lag-attentive VAE forecasting stored target coefficients.

    Constructed exactly as its base class is -- same keywords, same defaults, same refusals --
    with no keyword of its own. The decoder width follows the target gate, so a run configured
    with a reach budget gets a decoder for the channels that budget kept, and one configured
    without a budget gets a decoder for all $c_y$.

    The mixin comes **first** in the bases, which is what makes its width hook win method
    resolution over the base's ``raw_per_step`` one; reversing the two would build a raw-width
    decoder and forecast a feature block against it.
    """
