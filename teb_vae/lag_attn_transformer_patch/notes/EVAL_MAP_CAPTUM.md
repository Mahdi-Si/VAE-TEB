# EVAL_MAP_CAPTUM: attribution on the patch cell

This note covers how the shared CFS Captum machinery (`teb_vae/lag_attn_cfs/eval/attributions.py`,
`attribution_pass.py`, `gradcam.py`) applies to `SeqVaeLagAttnTrfPatch`, and indexes the raw-signal
attribution proposals P1–P8 that the analysis modules cite. `../eval/EVAL.md` describes the built
analyses; `teb_vae/lag_attn_cfs/eval/ATTRIBUTION.md` describes the shared pass.

## 1. The shared pass on the patch model

- The shared `attribution` analysis runs unchanged. The eval view (`eval/view.py`) presents the
  patch model under the CFS three-stream forward: `y_st = y_patch`, `y_ph = y_patch[..., :0]`,
  `u = u_patch`. The eval task supplies the patch streams and the summary target, and
  `PatchSummaryTarget._build_forecast_target` gives the block-score readouts their target.
- In patch-token mode the shared baselines zero every channel of a token, including the validity
  channel $m_t - 1$. Under an `all_zero` baseline the path therefore crosses the `torch.where` in
  `PatchEmbedding`. The raw-signal analyses avoid this: they attribute over raw samples, and the
  validity is held fixed (§2).

## 2. `patchify` under attribution

- **value** = `where(valid, raw, 0)` and **delta** $= (x_n - x_{n-1}) \cdot m_n \cdot m_{n-1}$. Both
  are linear in raw with fixed masks, so the gradient is the mask. Invalid and non-finite samples
  get exactly 0.
- **validity** = `amin(mask) − 1` is a boolean function of `weight` and `isfinite(raw)`, with no
  gradient. `RawReadout.prepare` (`eval/raw.py`) returns finite raw plus the original per-sample
  validity, which is held fixed along the path, so a NaN sample stays masked.
- **The token-boundary read.** `delta[0]` of token $t$ reads raw sample $16t - 1$, the last sample
  of token $t - 1$. A source readout at anchor $t$ with `max_lag` $= 37$ therefore has raw support
  $[16(t-37) - 1,\ 16t + 15]$.
- **Raw IG and patch IG share one path.** With validity fixed, `patchify` is affine in the raw
  samples, so the straight raw path maps to the straight patch path. Both IGs are complete with the
  same total; they allocate the total differently. `raw_attribution` gets both from one
  `LayerIntegratedGradients` call on the input of `source_adapter`.
- **The `(B, T, R)` reshape.** With raw passed as `(B, T, 16)`, the shared reductions read "time" as
  the token and "channel" as the sample position inside the patch. The raw-sample lag is
  $s = 16\ell + 15 - k$ for lag $\ell$ and sample position $k$.

## 3. Grad-CAM views on the patch model

The shared resolvers find every layer by the names inherited from `SeqVaeLagAttnTrfRws`.

| View | Module | Meaning on the patch cell |
|---|---|---|
| `target` | Input of `target_encoder.attention_blocks[-1]` | Token-level FHR evidence. The target encoder is full-causal, so it can reach $t = 0$. |
| `source` | Forward hook on `out["source_state"]` | `source_state` is the K/V representation: the source encoder output under the shipped `lag_kv_source: encoder`, the `PatchEmbedding` output under `adapter`. |
| `attention` | Output of `lag_attn.attn_dropout`, flipped to lag order | Gradient-weighted attention over the 38 lags |

Other layers used by the raw analyses: `lag_attn.W_k` / `W_v` (the key/value split, P6) and the
input of `source_adapter` (patch and raw IG, §2). No in-model layer has a raw-sample time axis, so
raw resolution comes from IG, not from Grad-CAM.

## 4. Proposal index

| # | Question | Readouts | Implemented in |
|---|---|---|---|
| P1 | Where in raw UP history does the KL or the gain come from, and is it where the attention says? | `kld`, `pred_gap`, `delta_level` (the mean over $\tau$ of $\mu^{full} - \mu^{base}$ in the level channel), against `source_kl_lag_map` | `raw_attribution` |
| P2 | What inside a patch matters: sample position, value or delta? | As P1 | `raw_attribution` |
| P3 | What FHR history drives the prior? | `base_level`, `base_var`, `mu_prior` | `fhr_drivers` |
| P4 | Which phase of a contraction drives the model? | `kld`, `pred_gap`, from a resting-tone baseline | `event_locked` |
| P5 | Does the gain sit where the KL sits? | Agreement of the P1 lag profiles across readouts | `raw_attribution` |
| P6 | Does UP decide *where* to look (keys) or carry *what* is read (values)? | `LayerIntegratedGradients` on `lag_attn.W_k` / `W_v` | `raw_attribution` |
| P7 | Do heads specialise in lags? | IG of `kld_per_t_per_head[m]` against that head's attention | `raw_attribution` |
| P8 | Which FHR context makes UP matter? | `src_effect` $= K_t(\mathrm{fhr}, \mathrm{up}) - K_t(\mathrm{fhr}, 0)$ | `fhr_drivers` |

The structural checks on these rows are exactly zero on this model by construction:
`after_anchor_max_abs`, `outside_window_max_abs` and `validity_channel_max_abs`. Each row also
records its completeness residual.
