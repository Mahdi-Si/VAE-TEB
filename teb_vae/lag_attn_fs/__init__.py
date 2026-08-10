r"""The feature-domain lag-attention sequential VAE.

Structurally this is ``teb_vae.lag_attn_rws`` with one thing changed: what it forecasts. A
single-latent VAE over decimated scattering/phase-harmonic features predicts the next
$H_d = 30$ decimated steps of the **feature** future -- the target channels surviving the
configured causal reach budget, $78$ of $109$ at the shipped $120$ s -- twice from one shared
decoder: once from the target-only prior latent and once from the source-conditioned posterior
latent. The per-step KL between the two is the source-conditioned coupling readout, and because
the decoder receives nothing but $z$, the prior latent is forced to carry the whole predictive
state.

Why the package exists. Two sibling models differ in *two* ways at once. ``lag_attn`` forecasts
feature channels through a latent that only corrects a ``decoder_state`` bypass; ``lag_attn_rws``
forecasts raw FHR through a latent that carries everything. Nothing in the tree separates the
latent factorisation from the target domain. This model is the missing cell -- feature target, no
bypass -- so that against ``lag_attn`` it isolates the bypass removal and against
``lag_attn_rws`` it isolates the target domain.

Layout::

    nets/       the network, framework-free: torch + stdlib + entmax only
    tests/      fast hermetic pytest
    configs/    YAML configs resolved through ``teb_vae.lag_attn.config``

The import rule: everything shared is imported from ``teb_vae.lag_attn.nets`` and
``teb_vae.lag_attn_rws.nets`` -- the geometry, the forecast and KL masks, the objective, the
prior and posterior heads, the lag cross-attention, the decoder, the channel gates, the lag
report -- and nothing here re-implements any of it. A change to one of those belongs in the
package that owns it, where all three models get it.

**The target is gathered, never delayed.** The input :class:`~teb_vae.lag_attn.nets.delays.
ChannelGate` does two things -- gather the channels surviving the reach budget, then delay each
survivor by $\delta_c = \lceil \rho_c / \Delta \rceil$ steps so its forward reach falls behind
the anchor's causal endpoint. The forecast target takes the gather and **not** the delay: a
delayed target would silently ask anchor $t$ to forecast the future of anchor $t - \delta_c$,
and no shape check anywhere would notice.

**The target is smeared, and that is not a leak.** A stored coefficient at decimated step $s$ is
a weighted average of raw signal over a window *centred* at raw index $16 s$, so a share of the
short-horizon target is a deterministic function of raw signal the model has already observed.
No future information enters the model; part of the answer is computable from legitimate
history, which is the opposite of leakage. Restricting the target to the reach-budget survivors
halves that share -- mean blend $0.091$ over the horizon against $0.173$ on all $109$ channels --
and makes the far horizon genuinely clean.

**The nats are not comparable across target domains, nor across reach budgets within this
model.** The reconstruction is summed over $H_d \cdot C_{\mathrm{keep}} = 2340$ coefficients
against the raw model's $H_d \cdot R = 480$ samples, and $C_{\mathrm{keep}}$ moves with the
budget, so every budget sums a different block. The numbers rank arms at one budget and nothing
else.
"""
