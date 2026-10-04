# lag_attn_transformer_patch — results

No production run exists yet. This page holds the pre-registered reading for each comparison, the
planted-delay instrument record and the parameter and cost record. The readings were written on
2026-10-03, before any production number, and must not be edited after one exists. Record results
below them.

## Pre-registered readings

**What is comparable.**
- Nats from this cell are comparable to no other cell. The scored block is H·C = 30 × 2 = 60
  standardized summaries per anchor, against CFS's 30 × 76 coefficients.
- Nats are comparable between this package's arms, because every arm scores the same target. They
  are not comparable once `target_summary_*` or `variability_eps` differ between runs. Fix those
  constants once, from the production training fold, before the first run.
- The comparison against CFS is read on quantities that do not depend on the target:
  - the downstream classifier (`teb_vae/classifier`, `VaeSource` keys `mu_prior`, `delta_mu` and
    `kld_per_t`, same cohort, same folds);
  - the source share of the KL, `1 − kld_source_null / source_conditioned_kl_raw` on val;
  - the permutation gap (`kld_shuffled` against the matched KL).
- Both cells must use the same shard folds and the same stats file.

**A difference counts only if it exceeds the classifier's own fold-to-fold spread.** For a
quantity the eval package reports with a bootstrap interval over recordings (`eval/EVAL.md`), it
also counts only if that interval excludes zero. No `metrics_history.csv` scalar carries an
interval. (This sentence was updated on 2026-10-03, when the eval package was added and before
any production number existed.)

| Comparison | Quantity | Reading fixed in advance |
|---|---|---|
| default vs CFS default | Classifier metric on the same cohort and folds; source share of the KL | **Representation helps:** the patch classifier is at least as good as CFS, *and* the source share is higher. **Representation costs:** the classifier is worse; then read `sweep_warmup_134` before concluding anything. **No difference:** the representation is neutral for the downstream task, and the cell's value is its lag readout (A.3) and its 104 extra dense anchors. |
| `sweep_warmup_134` vs default | Classifier metric; val `nll_full_block` per block | Default beats CFS but this arm does not: the gain came from the 76% more anchors, not the representation. This arm also beats CFS: the gain is the representation. The arm's nats will be **higher** than the default's, because it drops early-segment anchors, which are the easiest. That alone is not a finding. |
| `sweep_lag_kv_encoder` vs default | Concentration of the val KL lag profile (support-corrected argmax and the mass in its ±3-lag neighbourhood); classifier metric; source share | Prediction (A.3): the encoder arm's profile is broader. If its classifier metric and source share are no worse, lag locality buys only an interpretable lag map. If they are worse, locality also carries predictive information. If the encoder arm is better on both, A.3's argument does not hold on this data. |
| `sweep_persistence` vs default | Classifier metric from `mu_prior` alone; val `pred_gap`; `nll_full_block` | Prediction (A.5): persistence lowers the nats and lowers the classifier metric from `mu_prior`, because the current level leaves the latent. If the classifier metric does not drop, persistence is harmless and the level was not doing diagnostic work in the latent. |
| `sweep_source_validity_fhr` vs default | Source share of the KL; classifier metric | Expect no difference beyond the fold spread: FHR gaps and UP gaps are mostly independent sensors. If the `fhr_weight` arm is better, UP during FHR gaps is mostly artefact, so ship `fhr_weight`. If it is worse, real contractions during FHR gaps carry information. |

**Health checks that void a run before any reading:**
- `train/anchors_per_sample` is not 16.0, or `val/anchors_per_sample` is not 240.0 (sweep_warmup_134: 10 and 136);
- the KL lag map does not sum to `kld_per_t`;
- the loss-spike breaker skipped more than a handful of batches. The provisional `additive_margin`
  and `gradient_clip_val` were not re-derived.

## Planted-delay instrument record (P5-01)

This is an instrument reading at tiny widths on the committed 8-segment planted shard, not a
result. The val set is the training set. The shard couples raw UP to raw FHR at δ = 45 steps.
Before any model saw it, the orchestrator checked the raw signals: the 4 s patch means of UP and
FHR correlate at r = −0.76, peaking at exactly lag 45. H = 30 and `max_lag` = 60, so the
informative band is [15, 44]: 30 of 61 lags, and a flat profile puts 0.492 in it.
`lag_recovery_check.py` reads the last-epoch weights.

| Run | Epochs | Wall | Raw band / argmax | Corrected band / argmax | KL per anchor | `pred_gap` | `kld_shuffled` / matched | `kld_source_null` / matched |
|---|---:|---:|---|---|---:|---:|---:|---:|
| first draft of `planted.yaml` (S = 15, `source_dropout` 0.2, AR on) | 40 | 35 s | 0.505 / 13 | 0.497 / 13 | 0.0096 | −0.27 | 1.00 | 0.75 |
| same | 400 | 284 s | 0.486 / 11 | 0.477 / 48 | 0.040 | −0.48 | ≈ 1.0 | ≈ 1.0 |
| **`planted.yaml` as shipped**: the CFS instrument leaves (S = 1, `source_dropout` null, AR off) | 40 | 32 s | 0.588 / 27 | 0.581 / 27 | 0.035 | −0.35 | 0.97 | 5.44 |

In the third run every head peaks inside the band: lags 29, 27, 34 and 42, with KL band shares
0.53 to 0.62.

**What the instrument shows.**
- With the first draft's leaves, the source path stays nearly closed and the lag profile is flat.
- With the CFS instrument's three leaves, the profile moves into the band on every head. On
  2026-10-03 the user chose to ship these leaves in `planted.yaml`, so the two cells' instrument
  readings are comparable. A re-run of the shipped file reproduced the row: band 0.588, argmax 27.
- In neither setting is the KL source-specific: a shuffled source moves the posterior about as much
  as the real one, and `pred_gap` stays negative.
- This matches the sibling cells' own record on their planted fixture
  (`lag_attn_transformer_cfs/RESULTS.md`, "The identifiability record"): pooled argmax 0, band mass
  about 0.2, and `kld_source_null` at about 92% of the KL. In that family, only an interventional
  readout found the plant: occluding lags [15, 44] cost +14.94 nats on the conv-LSTM cell.
- The pass/fail reading is the user's (plan P5-01). One open choice: should the interventional
  occlusion readout be ported? In this representation a zeroed source row reads as valid-and-flat
  UP, which is the "no contraction" intervention.

## Eval instrument reading on the planted shard

**An instrument reading at tiny widths on the committed 8-segment planted shard, not a result.** The
val set is the training set, and every interval is over 8 recordings. Read on 2026-10-03: the
shipped `planted.yaml` checkpoint at 40 epochs (`lag-attn-trf-patch-epoch=39.ckpt`) through
`eval/run.py` with `eval/configs/planted_overrides.yaml` on one GTX 1660 Ti, 8 min 39 s wall. All 35
steps ok; `time_shift` and `cross_subgroup` recorded skips (one segment per recording, one cohort).
The plant is δ = 45 tokens = 180 s; the informative lags are [15, 44]. Every value below is from
that run's `summary.json` and the analysis CSVs. Intervals are 95% percentile bootstraps over recordings.

| Analysis | Readout | Planted reading |
|---|---|---|
| shared lag readouts | KL and attention argmax lag | 27 (inside [15, 44]) |
| shared controls | `kld_source_null` / matched KL; `pred_gap_mc_nats` | 0.192 / 0.035 nats = 5.4; +0.0014 (s.e. 0.0008) |
| `delay_map` | Argmax of the interventional delay curve | **d = 45 tokens = 180 s**: 1.41e-4 nats per cell [0.90e-4, 2.0e-4]; d = 44 and 46 at 1.40e-4 and 1.26e-4; 0.0042 nats per anchor summed over the 30-cell diagonal |
| `occlusion` (1 anchor per segment) | `baseline` arm, band deltas | planted [15, 44]: +0.0008 [−0.015, 0.018]; peak band `near`: +0.0068 [−0.005, 0.019]. No band resolved |
| `impulse_response` | Level kernel, `rest` background | Peak −0.13 bpm at d = 28 s; −0.043 bpm at 180 s (`segment` background: −0.020 bpm at 144 s). The data's contraction-triggered FHR dip is −12.9 bpm at 177 s, so the kernel is ≈ 1% of it at its peak and 0.3% at 180 s. Causality check 0 |
| `raw_shift` | Centroid slope on whole-patch shifts (1 = tracks content) | Head-mean attention 0.0066 (R² 0.05); single heads −0.034 to 0.060; KL-map centroid 0.035 (R² 0.44). Sub-patch: −0.033 |
| `raw_attribution` (4 segments × 4 anchors) | Token-binned \|IG\| of raw UP | `pred_gap` and Δlevel peak at **lag 29**, the model's own lag-map argmax on those rows; planted-band share 0.57 and 0.53 (flat 0.49). `kld` peaks at lag 52, band share 0.48, `lag_corr` 0.26. Value channel 0.88–0.89 of \|IG\|. Completeness ≤ 0.006 |
| `event_locked` (43 contractions) | Tracking index (1 = none) | Heads 0 and 2: **1.61** [1.46, 1.75] and **1.58** [1.41, 1.87]. Heads 1 and 3: 0.0004 and 0.22. Head mean 0.82; KL map 1.04 [0.95, 1.18] |
| `decelerations` (38 paired) | Measured vs implied delay | Measured median 170 s (pooled); triggered-FHR minimum at 177 s. Implied (attention) median 182 s, median \|error\| 15 s, but Spearman **ρ = −0.26** (per-recording mean −0.33). Enrichment on the causal contraction 0.91 [0.83, 0.98] against a permuted-delay control of 1.05 |
| `channel_skill` | Level forecast | RMSE 6.3 bpm against persistence 10.5 bpm; beats persistence at every lead. Full against base, MSE skill −0.022 [−0.025, −0.018] |
| `signal_loss` | FHR gap of 120 s ending at the anchor (shard has 0% `missing` tokens) | ΔKL **+0.41** nats, Δ`logvar_prior` +0.60, level forecast moves 5.6 bpm, its log-variance +0.07. The same gap ending 60 or 180 s earlier: \|ΔKL\| ≤ 0.0007 |
| `fhr_drivers` | Lags holding 90% / 50% of \|IG\| | Base level, 90%: 444 s. `src_effect` (the KL the UP adds), 50%: 5.9 s |

**What the instrument says about where the lags come from.** These are readings of one tiny model,
not findings.
- **The attention sits at fixed lags and does not track UP content.** Shifting UP by whole patches
  moves the head-mean attention centroid by 0.7% of the shift, and no head by more than 6%. On the
  paired events the attention is not enriched on the causal contraction (0.91, below its permuted
  control), and the implied delay anti-correlates with the measured one. Heads 0 and 2 do show a
  diagonal on the contraction-locked map (≈ 1.6); under the shift intervention they move by 3% and
  6% of the shift.
- **The observational lag readouts cannot name the delay.** They sit inside the informative band
  (KL argmax 27, IG peak 29), but the KL is not UP-specific (a flat UP elicits 5.4× the matched KL,
  and the clock-excess profile is degenerate), and the KL-IG profile correlates only 0.26 with the
  model's own lag map. A lag in [15, 44] is consistent with the plant without locating it.
- **An interventional readout finds the plant.** Single-lag occlusion over 16 anchors per segment,
  read along the diagonals d = ℓ + 1 + τ, peaks at exactly d = 45 (180 s), its neighbours next. Band
  occlusion at one anchor per segment does not resolve it.
- **The forecast barely uses the plant at 40 epochs.** The delay-map peak is 1.4e-4 nats per cell,
  the impulse response is about 1% of the data's contraction-triggered dip and peaks at 28 s rather
  than 180 s, `pred_gap_mc` is +0.0014 nats per anchor, and the full level forecast is 2% worse than
  the base in MSE. The model reacts far more to missing FHR at the anchor than to UP at any lag.

## Parameter and cost record (P5-03)

Measured 2026-10-03 on one NVIDIA GeForce GTX 1660 Ti (Max-Q, 6 GB), torch 2.14.0+cu130, float32.

**Build.** Each model is built from its shipped `configs/default.yaml` `VAE_model` block through the
trainer's own signature sweep (`_build_model_kwargs`). The CFS cell's four warm-up tuples cannot be
resolved here, because the `REPOINT_ME_causal_int` shards are not on this machine. They are set by
hand at the documented shipped widths: target 76 of 80 (32 `fhr_st` + 44 `fhr_ph`) and source 46 of
46. The `W'_c` values come from the legacy causal fixture's identical filters. Parameter counts
depend on the widths only. `target_novelty_frac` and `target_scored_horizon` are left `None`; both
are masks and buffers and add no parameters.

### Parameters

| Top-level module | patch | CFS | delta |
|---|---:|---:|---:|
| `target_adapter` | 71,808 | 86,912 | −15,104 |
| `source_adapter` | 71,808 | 79,232 | −7,424 |
| `target_encoder` | 1,676,928 | 1,676,928 | 0 |
| `query_proj` | 8,320 | 8,320 | 0 |
| `lag_attn` | 71,576 | 71,576 | 0 |
| `prior_head` | 108,010 | 124,650 | −16,640 |
| `posterior_head` | 116,484 | 116,484 | 0 |
| `horizon_core` | 1,849,346 | 1,849,346 | 0 |
| `decoder` | 106,740 | 147,056 | −40,316 |
| `target_ar_logit` | 2 | 76 | −74 |
| **Total** | **4,081,022** | **4,160,580** | **−79,558 (−1.9%)** |

Every non-zero delta:
- `target_adapter`: input `Linear` 33→128 (4,352) against 76→128 (9,856). No `mask_proj` (CFS 76×128 = 9,728).
  Adds a learned `missing` token (+128). Net −15,104.
- `source_adapter`: the same with in_dim 33 against 46: `Linear` −1,664, `mask_proj` −5,888, `missing` +128. Net −7,424.
- `prior_head`: no clock input (`prior_availability_input: false`): `clock_proj` 128×128 and `clock_norm` are removed. −16,640.
- `decoder`: the mean and logvar heads output 2 channels against 76 (2 × −19,018). There is no
  `persistence_weight` (`persistence_residual: false`, CFS 30×76 = 2,280). −40,316.
- `target_ar_logit`: one AR(1) coefficient per target channel, 2 against 76. −74.

The encoders, the lag attention, the posterior head and the horizon core are identical in shape.

### Cost: one training step

The step is the forward, `compute_loss(...)["metrics"]["total_loss"]` and `.backward()` at batch 16.
The models run in train mode at training geometry: stride 15, a random per-sample phase in [0, 15),
and `weight` all ones. The loss uses the shipped hparams with `beta = 1`. Each figure is the median
of 20 steps after 3 warm-up steps, with `torch.cuda.synchronize()` on both sides of each step. Peak
memory is `torch.cuda.max_memory_allocated` over the 20 steps. The GPU was idle before and after
each run.

CFS inputs are random `(16, 300, c)` stored features: `fhr_st` 36, `fhr_ph` 44 and `up` 46. Patch
inputs are random raw `fhr` and `up` of shape `(16, 4800)`. They go through `patchify` as `task.py`
does. "Step" includes `patchify`; "model only" feeds pre-patchified tensors.

| Cell | warm-up F | anchors / sample | step ms (median, min–max) | model only ms | peak MiB |
|---|---:|---:|---:|---:|---:|
| CFS (shipped) | 134 | 10 | 144.5 (143.7–146.2) | — | 1,269 |
| patch (shipped) | 30 | 16 | 174.5 (173.7–175.7) | 174.6 | 1,429 |
| patch, `sweep_warmup_134` geometry | 134 | 10 | 143.4 (142.4–145.1) | 143.0 | 1,234 |

- Batch 16 fits both cells with room to spare (under 1.5 GB of 6 GB).
- The shipped patch step costs +21% time and +13% peak memory over CFS. All of it comes from the
  60% more anchors: F = 30 leaves 16 tiles against 10. At F = 134 the patch cell matches CFS
  (−1% time, −3% memory).
- `patchify` cost is within run-to-run noise (under 1 ms). The CFS figure does not include the
  offline feature transform its shards carry.
