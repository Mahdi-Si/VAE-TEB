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

**A difference counts only if it exceeds the classifier's own fold-to-fold spread.** There is no
evaluation package and no confidence interval on any `metrics_history.csv` scalar (DESIGN.md §4).

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
