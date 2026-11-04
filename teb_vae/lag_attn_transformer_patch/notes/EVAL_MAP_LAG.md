# EVAL_MAP_LAG: lag, coupling and control analyses on the patch cell

This note covers how the lag, coupling and control half of `teb_vae/lag_attn_cfs/eval` reads on
`SeqVaeLagAttnTrfPatch`, and indexes the raw-signal lag proposals N1–N6 that the analysis modules
cite. `../eval/EVAL.md` describes the built analyses and how each shared column reads.

## 1. Facts about the patch model that decide the mapping

| Fact | Value | Consequence |
|---|---|---|
| Source delay steps | 0 | The compensated lag axis is $\tau_\ell = 4\ell$ s. Lag $\ell$ is the UP patch whose last sample lies exactly $4\ell$ s before the anchor patch's last sample. There is no filter delay. |
| `source_gate`, `target_gate` | `None` | The ungated branches in the shared `occlusion` and `time_shift` code apply. |
| Source adapter | `PatchEmbedding`; no availability buffer | There is no availability clock. On a zero stream every K/V token is identical, so the null-arm attention is the attention's learned per-lag key bias. The shared "clock" readouts therefore measure the KL that a flat UP elicits. |
| K/V reach | Set by `lag_kv_source` (`IMPLEMENTATION_PLAN.md` A.3) | Under `adapter` (1 token), occluding a band of lags changes K/V at exactly those lags. Under the shipped `encoder`, an edit at lag $\ell$ also reaches the keys at smaller lags (later tokens) within the causal encoder reach. |
| Lag-support margin | $F - (L - 1) = 30 - 37 = -7$ at the shipped geometry | Anchors 30–36 (7 of 240 dense) see a truncated lag window, so the support-corrected and untruncated profiles differ from the raw profile. |
| Informative lags | Anchor $t$ forecasts tokens $t+1 \ldots t+30$ | A UP→FHR delay of $d$ tokens is informative at lags $[d - H, d - 1]$. The lag map spreads one delay over up to $H$ lags; the delay is $d = \ell + 1 + \tau$, which needs the horizon axis (N4). |
| Token layout | `[value(16), delta(16), m_t − 1]`, validity 0 = valid | Zeroing a whole token turns an invalid token into "valid and flat at the z-mean". On a valid token it zeroes only the values. |
| Units | UP and FHR are loader z-scored | The z-mean of UP is above resting tone, because contractions skew UP positive. "Zero" UP is therefore a weak tonic plateau, not "no contraction". This is why occlusion has a `baseline` arm (`EVAL_PLAN.md` D5). |
| Step | 16 samples at 4 Hz = 4 s | The shared `SECONDS_PER_STEP` and the clinical time grids apply unchanged. |

## 2. Shared lag modules

Once the collection pass runs on the patch forward, these shared analyses read the patch tables
unchanged: `lag_kl`, `attention`, `lag_clocks`, `lag_kld_scaled`, `lag_high_kl`, `source_null`,
`perm_control`, `coupling` and `events`, with the helpers `lag_shape`, `lag_hist` and
`events.detect_contractions`. `band_partition` is excluded, because a token has no spectral
channels. `occlusion` and `time_shift` are ported (`../eval/analyses/`).

Two readings change on this model:

- **The "clock" in `source_null`, `lag_kld_scaled` and `lag_high_kl`.** There is no availability
  staircase or announcement buffer. Read every "clock" as the flat-UP response, and every
  clock-excess as the KL beyond what a flat UP elicits.
- **The lag-axis wording.** The shared constants `COEFFICIENT_LAG_AXIS_LABEL`, `GROUP_DELAY_CAVEAT`
  and `GROUP_DELAY_NOTE` (`lag_axis.py`) speak of stored-coefficient time. The patch binding's
  `causality_text` replaces the preflight wording; the shared figures keep the old label (a lean
  limit in `../eval/EVAL.md`).

## 3. Proposal index

| # | Question | Implemented in |
|---|---|---|
| N1 | What forecast response does one contraction elicit, as a function of delay? (the effective FIR kernel) | `impulse_response` |
| N2 | Does the attention follow a contraction as it recedes (a diagonal), or sit at fixed lags? | `event_locked` |
| N3 | Before an FHR nadir, does attention sit on the contraction that preceded it, and does the implied delay track the measured one? | `decelerations` |
| N4 | At which UP→FHR delay does removing UP cost the forecast? (interventional delay map) | `delay_map` |
| N5 | Does the coupling scale with contraction amplitude and duration? (dose–response) | `decelerations` |
| N6 | Do heads split the work, and which head carries the forecast benefit? | Not implemented (§4) |
| — | Does the lag readout move with the UP content? (raw delay sweep) | `raw_shift` |

## 4. Planned: N6, per-head lag specialisation and head ablation

- **Question.** Do the heads split the work (a contraction tracker, a fixed-latency head, a tonic
  head)? Which head carries the forecast benefit?
- **Computation.** For each head $m$:
  - the N2 tracking index and the slope of the attended lag against time since the peak;
  - event selectivity: attention mass on contraction tokens divided by mass on contraction-free
    tokens;
  - the KL share from `kld_per_t_per_head`;
  - **ablation**: replace the head's `attended_source_heads[..., m, :]` with its null-arm value,
    re-pose through `posterior_head`, decode, and record $\Delta$gap and $\Delta K$. This is a
    variant of `controls._attend_and_pose` (`lag_attn_rws/nets/controls.py`).
  - per-head, per-class lag histograms from the existing `lag_profile_per_head` vectors.
- **Figure.** A heads × {tracking, selectivity, KL share, ablation $\Delta$nats} panel, plus the
  per-head lag histograms by class.
- **Cost.** $M = 4$ ablation arms per batch.
- **Reading.** A head with high tracking, a large ablation $\Delta$ and a modest KL share is the
  useful contraction reader. A head with a high KL share and an ablation $\Delta$ near zero moves
  the belief without helping the forecast.
