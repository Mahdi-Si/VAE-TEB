r"""The causal-input raw-target lag-attention sequential VAE.

The seventh cell of the encoder-by-target grid, and the only one whose forecast claim and lag claim
are both exact::

                            conv-LSTM encoders          conv-Transformer encoders
    raw FHR target          lag_attn_rws                lag_attn_transformer_rws
    two-sided feature       lag_attn_fs                 lag_attn_transformer_fs
    causal feature          lag_attn_cfs                lag_attn_transformer_cfs
    causal in / raw out     lag_attn_crws   <- this     lag_attn_transformer_crws

**Neither side of the objective contains its own future.** The two-sided cells read coefficients
that at decimated step $t$ average raw samples on *both* sides of $t$ -- up to $965$ s into $t$'s own
future on the slowest channel -- so a forecast of what follows $t$ is computed from inputs that have
already seen it. ``teb_vae.lag_attn_cfs`` removes that at source by reading the causal dataset
variant, where a coefficient at $t$ is a function of $\{x(s) : s \le t\}$ and of nothing else, but
pairs it with a **causal feature** target, which reintroduces a different caveat: a causal
coefficient still lags by its own composed group delay of $13.3$ to $791.0$ s, uncompensated, so a
lag-resolved reading is an attribution over stored-coefficient time rather than over physical delay.

Pairing causal inputs with the **raw** target removes both. The target is the signal itself: no
warm-up, no group delay, no channel selection, honest at every step. So this cell -- and its
conv-Transformer twin -- forecast the next minute of raw FHR from inputs that carry no future, and
``lag_attn_rws`` is the direct control: same target, same objective, same horizon family, differing
only in whether the inputs contain the answer.

Nothing about the architecture is new. The encoders, the channel gates, the availability adapters,
the prior and posterior heads, the lag cross-attention and the shared horizon decoder are
``teb_vae.lag_attn_rws``'s; the input warm-up mask, the lag validity floor, the tiled anchor set and
the forward that decodes at it are ``teb_vae.lag_attn_cfs``'s. What is local is the raw future
window gathered at those anchors, which no raw-target sibling needed because none of them tiled.

Layout::

    DESIGN.md       the as-built record: the geometry, the anchored raw target, the readouts, the
                    budget, and every member bound by reference from a sibling
    nets/           the network: the causal-input mixin over the raw-signal architecture, and the
                    raw future window gathered at the decoded anchors
    task.py         the Lightning task: the tiling phase, the source-null readout, the page seams
    trainer.py      the experiment driver and its pre-flight refusals
    configs/        the shipped configuration and its arms, resolved through ``teb_vae.lag_attn.config``
    sample_page.py  the diagnostic page rows the tiled anchor set makes different
    tests/          fast hermetic pytest
    RESULTS.md      the pre-registered criteria the headline run is scored against

**The input warm-up is a policy here, not a validity requirement.** In the causal-feature cells the
anchor floor $F$ must clear the resolved budget $B$, because below it the objective would score
assumed pre-recording history as though it were signal. A raw sample carries no such region, so the
floor constrains nothing about the target and is retained as the declared *input-warmth* statement:
every kept input channel of **both streams** is warm by the first forecast step. Every part of that
wording is exact and no obvious paraphrase is. It is *both streams* because the channel alignment
shifts both and the constructor checks both; before it, the source was ungated and kept channels
waiting up to $278$ steps that were cold hundreds of steps past this floor by design, with
``source_lag_warmth_frac_st`` / ``_ph`` measuring that residual instead of the floor refusing it.
And it is *by the first forecast step* rather than at the anchor: at a floor of $133$ against
$B = 134$ the slowest kept target-stream channel is still cold at the anchor itself and becomes
honest exactly at $t + 1$, which is the first step the forecast covers -- so that half asks only for
$F \ge 133$, and what puts the shipped floor at $134$ is the shifted-input half, which the
zero-marginal-warm-up lemma makes bind at exactly $B$.

**Stored causal time is still not physical time, on the input side.** One-sidedness and zero latency
are different properties and only the first is bought by the causal transform. The target has no
group delay at all, so the caveat that reaches every lag-resolved reading here is one-sided rather
than two-sided: it attaches to the *inputs*, whose composed group delay reaches $791$ s and which
``causal_delay_s`` records without anything yet compensating for it.

**Nats from this cell are comparable to no sibling.** The raw block is $H \cdot R = 15 \times 16 =
240$ samples against ``lag_attn_rws``'s $480$, and the tiled anchor set decodes about $10.1$ anchors
per step against a dense $240$. Every loss-scale constant stated in nats has to be re-derived rather
than transferred.
"""
