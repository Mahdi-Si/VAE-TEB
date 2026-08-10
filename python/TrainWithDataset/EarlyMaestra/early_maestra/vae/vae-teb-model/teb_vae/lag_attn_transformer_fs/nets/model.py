r"""The conv-Transformer feature forecaster: two halves from two parents and nothing of its own.

:class:`SeqVaeLagAttnTrfFs` is a composition. Both causal conv-Transformer encoders, the channel
gates, the availability-aware input adapters, the prior and posterior heads, the lag
cross-attention, the shared horizon decoder and the paired reparameterisation are
:class:`SeqVaeLagAttnTrfRws`'s; the decoder's width, the gathered feature target, the four resolved
forecast gaps and the objective delegation are
:class:`~teb_vae.lag_attn_fs.nets.feature_target.FeatureForecastTarget`'s. Nothing is written here.

**Why a mixin and not two inheritances.** The feature target was first built as four methods on
``SeqVaeLagAttnFs``, which subclasses the conv-LSTM model. Inheriting from both models --
``class SeqVaeLagAttnTrfFs(SeqVaeLagAttnFs, SeqVaeLagAttnTrfRws)`` -- linearises as
``TrfFs -> Fs -> Rws -> TrfRws -> Module`` and so runs the **conv-LSTM** constructor: a model that
builds, trains and reports, and is not this architecture. The five target-domain members mention no
encoder, so they moved into a plain-object mixin instead, and both feature models are now their own
architecture plus that.

**The order of the bases is load-bearing.** The mixin comes first, which is what makes its width
hook win method resolution over the base's ``raw_per_step`` one. Reversed, the decoder would be
built at $R = 16$ and a $C_{\mathrm{keep}}$-wide feature block scored against it -- and since the
objective takes the block width as an argument, nothing would raise.

**The empty class body is the guarantee.** With nothing defined here, the twenty forward keys, the
absent ``decoder_state`` and ``delta_mu_src``, every latent shape, the posterior's head-structured
split and the lag map cannot have moved: they are the base's own code objects, pinned by the base's
own suite over the same functions. The only thing this class changes about the forward is the last
axis of the four forecast tensors, which is why the suite has no forward-contract module of its own.

The unit consequence, restated where a reader will look for it: the reconstruction is summed over
$H \cdot C_{\mathrm{keep}} = 2340$ coefficients at the shipped budget against the raw variant's
$H \cdot R = 480$ samples, so the nats are comparable to neither the raw variant's nor another reach
budget's.
"""
from __future__ import annotations

from teb_vae.lag_attn_fs.nets.feature_target import FeatureForecastTarget
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws


class SeqVaeLagAttnTrfFs(FeatureForecastTarget, SeqVaeLagAttnTrfRws):
    r"""Single-latent lag-attentive conv-Transformer VAE forecasting stored target coefficients.

    Constructed exactly as :class:`SeqVaeLagAttnTrfRws` is -- same keywords, same defaults, same
    refusals -- with no keyword of its own and, in particular, no ``decoder_out_channels``. The
    decoder width follows the target gate, so a run configured with a reach budget gets a decoder
    for the channels that budget kept and one configured without a budget gets a decoder for all
    $c_y$; and because no keyword records it, no second field can disagree with the gate.
    """
