# EVAL_MAP_CAPTUM: the CFS Captum machinery, mapped onto the patch cell

Written 2026-10-03 against `main` @ `1536732`. This is a read-only map; no code was changed. Paths are
relative to `teb_vae/`. Short names: `A` = `lag_attn_cfs/eval/attributions.py`, `P` =
`lag_attn_cfs/eval/attribution_pass.py`, `G` = `lag_attn_cfs/eval/gradcam.py`, `C` =
`lag_attn_cfs/eval/class_contrast.py`, `CWI` = `lag_attn_cfs/nets/causal_inputs.py::CausalWarmupInputs`.
`captum==0.9.0` is installed in `.venv`. The dry-run scripts are in the session scratchpad, and the
short version is in §3.2.

## 0. Verdict

- **The Captum core is reusable as-is.** That covers `integrated_gradients`, `layer_attribution`,
  `ablate_lag_bands`, `expand_rows`, `entry_point`, `attributed_forward`, `gradcam.gradcam`, the
  reductions and `class_contrast`. The requirement is that the patch model is presented under the
  CFS 3-stream calling convention: `y_st = y_patch`, `y_ph = y_patch[..., :0]`, `u = u_patch`.
  A ~15-line eval-only class swap does this (§3.2), with no edits to the core or the model. All of
  the following ran unchanged on the tiny patch model: IG, the per-head layer split, Grad-CAM, the
  band ablation and the target-only check.
- **Three things are missing on the patch side.** (1) A patch `model_inputs`: `metrics.py:249`
  reads `fhr_st/fhr_ph/up_ph`. (2) `_build_forecast_target`: the model has none, and the block-score
  readouts call it at `A:618`. (3) A **validity-preserving baseline**: `baselines_for` (`A:671`)
  zeroes the `m_t − 1` channel too.
- **Raw-signal attribution works through a 10-line wrapper** that patchifies inside the readout,
  with the raw inputs passed as `(B, T, R)`. Every existing reduction then reads it unchanged:
  "time" means token, "channel" means the sample inside the patch. In this mode validity is
  structural: it is derived from `weight`, so it never moves along the path.
- **IG dry run (kld, one anchor).** Raw-UP and patch-token IG give the same sum to 1e-6, and both
  are complete. The relative completeness error is 2.6e-3 / 3.7e-2 at 64 steps, 3.9e-4 at 128 and
  2.3e-6 at 256. **Use 128 steps.** At production width on a GTX 1660 Ti, the cost is 0.36 s/row
  (source-null) and 0.58 s/row (all-zero) at 128 steps, and Grad-CAM is about 3 ms/row.

## 1. What the CFS pass attributes

### 1.1 Readouts (one scalar per row at one anchor; `AnchorReadout.forward`, `A:558-667`)

The forward runs under `attributed_forward` (`A:434`). It shadows `_build_anchor_index` with one
anchor per row and `_reparameterize_shared` with `(μp, μq)`, so no ε is drawn. Every input and
`mu_post/logvar_post` is tied in with `0·sum` (`A:595`), so unused inputs get an exact 0.

| Readout | Built from (forward dict) | Line |
|---|---|---|
| `kld` | `kld_per_t` gathered at the anchor (dense latent) | `A:600` |
| `kld_dim`, `mu_post_dim`, `mu_prior_dim` | `model.kld_tensor(...)` or `mu_*`, at `[anchor, coordinate]` | `A:604-615` |
| `nll_full`, `nll_base`, `pred_gap` (= base − full), `mse_full`, `mse_gap`, `nll_horizon` (one τ) | `mu_/logvar_{full,base}[:, 0]` against `model._build_forecast_target(target_features, anchor)` under `forecast_mask(model.scored_weight(weight), geometry, coverage_floor)` and `forecast_likelihood_terms(model)`. The `mse_*` readouts drop `ar_coef`. | `A:616-657` |
| `lag_band` | `attn_weights` at the anchor, head-mean, summed over `[lo, hi]` | `A:660-664` |

Sets: `MAIN_READOUTS` = kld, pred_gap, nll_full, mse_full (`A:237`). `TARGET_ONLY_READOUTS`
(`A:233`). `EXAMPLE_READOUTS` (`A:174`). `TRACE_READOUTS` = kld, pred_gap (`A:166`).
`COHORT_READOUTS` = kld, pred_gap (`C:56`). `GRADCAM_READOUTS` = `MAIN_READOUTS` (`C:59`).

### 1.2 Inputs, baselines and methods

| Item | CFS | Line |
|---|---|---|
| Attributed inputs | `(y_st (B,T,36), y_ph (B,T,66), u (B,T,c_u))`, the stored feature streams at declared width. Extra args: `(target_features (B,T,c_y), weight (B,T), columns, coordinates)`. | `A:558`, `P:677` |
| Baselines | `source_null`: targets kept, `u = 0`. `all_zero`: everything 0. | `A:671-689` |
| Entry point | `x0 = b + 1e-3 (x − b)` (`BASELINE_ENTRY_FRACTION`). The jump `f(x0) − f(b)` is reported. | `A:190`, `A:692` |
| IG | `IntegratedGradients`, `n_steps = IG_STEPS = 64`, `internal_batch_size = max(64, N)`, with convergence delta. Under `source_null` only `u` is handed to Captum (`_source_only`, `A:1204`). | `A:186-187`, `A:1111-1201` |
| Layer IG | `LayerIntegratedGradients` on `posterior_head.fusion` (per head, `head_structured`), `source_null` only | `A:1308-1407`, `A:1358` |
| Ablation | `FeatureAblation`, source zeroed per lag band relative to the anchor, plus `rest`. Occlusion sign. | `A:1217-1305` |
| Grad-CAM | Three views, one forward and one backward (§4) | `G:115-186` |
| Rows | `expand_rows`: one row per (segment, anchor), via `repeat_interleave` | `A:1082` |

### 1.3 Anchor selection, cohort and class contrast

- **High-KL clean anchors.** `informative_anchors` (`P:145`) takes `per_anchor.parquet`, sets a
  pooled `kld_per_t` 0.7-quantile threshold (`A:151`) and requires `coverage ≥ 0.95` (`A:152`).
  `select_segments` (`P:185`) then draws one segment per recording, class-balanced and seeded
  (`traces.select_recordings`). `informative_columns` (`A:759`) keeps ≤ 4 per segment (`A:144`),
  with history validity ≥ 0.95 over `[t−L+1, t]` and spacing ≥ H. `EXAMPLES_PER_CLASS = 3`
  (`A:155`). Without the table, `spread_columns` (`A:734`) is the fallback.
- **Cohort.** `run_cohort` (`P:1490`) uses cap `caps.attribution_cohort_segments` (150). It runs IG
  of `COHORT_READOUTS` under `source_null` plus Grad-CAM, with no layer, ablation, horizon or
  examples.
- **Class contrast.** `run_class_contrast` (`C:484`) averages per recording, then computes a 95 %
  bootstrap over recordings (1000 resamples), Kruskal-Wallis (Holm per family `ig:<r>` /
  `gradcam:<r>`) and pairwise Mann-Whitney with Holm and Cliff's δ. The metrics
  (`recording_metrics`, `C:172`) are `source_total`, `source_abs_total`, `lag_centroid_s`,
  `lagband_<b>`, and Grad-CAM `<view>_centroid_s/_total`.

### 1.4 Outputs and checks

- **Files** (all under `attribution/`; table at `ATTRIBUTION.md:256-278`): `attribution_rows.csv`,
  `_vectors.npz`, `_maps.npz`, `_examples.csv`, `_lag_channel.npz`, `_recordings.csv`,
  `_summary.csv`, `_bands.csv`, `_lag_bands.csv`, `_layer.csv`, `_null.csv`, `_blocks.csv`,
  `_traces.csv`, `_trace_anchors.csv`, the cohort rows and vectors, `gradcam_rows.csv` and
  `_vectors.npz`, and the class recordings, stats, pairwise and lag-channel files.
- **Figures** (`ATTRIBUTION.md:303-321`, `FIGURE_GUIDE.md:871-1037`): maps, per-class example
  pages, lag_profile, bands, layer, null, channels, lag_channel, time_profile, checks,
  time_to_delivery, blocks, horizon, classes, class_maps, gradcam_classes and traces.
- **Checks per row** (`_reduce_rows`, `P:876-990`):
  - `completeness_rel` = |δ| / max(|f(x)−f(x0)|, |f(x)|) (`P:953`)
  - `after_anchor_max_abs` (`P:963`)
  - `gated_off_max_abs`, which uses `warm_from_step` (`P:967`, `A:859`)
  - `lag_corr` / `lag_js` against `source_kl_lag_map[t_a]` (`A:821`, `A:1471`)
- **Block-level checks**:
  - `target_only_check` (`P:1057`): the source attribution of `nll_base` must be 0.
  - The `checks` block (`P:2052-2057`) carries the medians and maxima, and
    `n_rows_over_tolerance` against `COMPLETENESS_TOLERANCE = 1e-2` (`P:96`).

## 2. Feature-specific coupling, and what the patch version needs

| # | Coupling (where) | Patch cell | Needed |
|---|---|---|---|
| 1 | `model_inputs` → `task._build_target_streams` / `_build_source_stream` read `batch.fhr_st/fhr_ph/up_ph` (`metrics.py:249`, `lag_attn_rws/task.py:306,334`). Used at `P:677` (every pass, trace and cohort). | The patch batch has only `fhr/up/weight`, so this raises `AttributeError` | A patch `model_inputs` returning `(y_patch, y_patch[..., :0], u_patch, summaries (B,T,2), weight)`. Build the patches with `task._build_forward_inputs(batch)[:2]` and the summaries with `model.summary_target(batch.fhr, batch.weight)`. |
| 2 | 5-arg feature forward `model(y_st, y_ph, u, phase, stride)`: `A:587`, dense call at `P:684` | The patch forward is `(y_patch, u_patch, phase, stride)` (`patch_inputs.py:39`). The CFS call misbinds `u` → `anchor_phase`. | The class swap in §3.2: `forward = CWI.forward`, which is exactly what `PatchStreamInputs.forward` delegates to with an empty `y_ph` |
| 3 | `model._build_forecast_target(target_features, anchors)` (`A:618`) | Absent: `hasattr` is `False` | Gather `summaries[b, t_a+1+τ]`, τ < H. This is the same index as `PatchSummaryTarget.compute_loss` (`patch_target.py:93-95`) |
| 4 | `baselines_for` zeroes **every** channel (`A:671`) | Channel 32 = `m_t − 1`. Zeroing it marks gap tokens valid, so the path crosses `PatchEmbedding`'s `torch.where` (`patching.py:94`) at α = 0.5, and the adapter's `Linear` also reads ch 32 as a feature. Measured: 1.75e-3 of attribution leaks onto ch 32. This affects FHR under `all_zero` always, and UP under `source_null` only on the `source_validity: fhr_weight` arm. | Either **raw mode** (validity derived from `weight`, structural) or a baseline that zeroes `[..., :-1]` and keeps `[..., -1]`. Add a check: `validity_channel_max_abs == 0`. |
| 5 | Warm-up staircase: `warm_from_step` (`A:859`), `masked_field` cold-cell blanking (`A:939`), `gated_off_max_abs` (`P:967`) | `target/source_warmup_steps` are `None`, so these return `None` and 0 (trivial) | The patch analogue is invalid tokens (`m_t = 0`). Their attribution is exactly 0 (verified: raw FHR gap token 12). New check `invalid_token_max_abs`, and blank invalid tokens in maps. |
| 6 | ST/PH blocks: `BLOCKS` (`A:245-253`), `block_sums` (`A:914`), `source_block_split` via `SOURCE_BLOCK_SPLIT` and `use_up_st` (`A:906`), `n_scattering = y_st width` (`P:703`), "phase-harmonic block" labels (`A:2597-2601`, `A:1809`) | Everything lands in "target_scattering" / "source_phase" | Value / delta / validity groups. The cheapest route: pass `channel_groups = {stream: {value: 0..15, delta: 16..31, validity: [32]}}` into `_reduce_rows` (`band_<stream>_<g>` columns). Optionally split `y_st = y_patch[..., :R]`, `y_ph = y_patch[..., R:]` (CWI re-concatenates), which turns the target blocks into value \| delta for free. |
| 7 | Frequency-band channel map `band_channel_map.csv` (`channel_groups_from_map`, `A:1502`; band_partition) | Not applicable | Use the groups in row 6. The joined `spectral_skill` columns are recorded absent. |
| 8 | Lag axis "stored-coefficient time": `COEFFICIENT_LAG_AXIS_LABEL` (`lag_axis.py:71`), `compensated_seconds_axis(L, delay_steps)` (`lag_axis.py:102`), `ATTRIBUTION_CAVEAT` group-delay sentence (`A:260`) | **No transform group delay.** Token lag ℓ = raw samples `[16ℓ, 16ℓ+15]` behind the anchor token's last sample, i.e. 4ℓ to 4ℓ+3.75 s. `delay_steps` = 0. | Relabel to "lag (s, 4 s patches)" and drop the group-delay caveat. Raw maps have a 0.25 s axis. |
| 9 | `occlusion_bands` sized for L = 91 (`eval/configs/eval_overrides.yaml:372-376`) | L = 38 (lags 0..37): `mid` and `far` fall outside the window, giving NaN groups and constant-0 `lag_band` readouts | Patch bands, e.g. `anchor [0,2]`, `near [3,7]`, `mid [8,15]`, `far [16,37]` (0-11, 12-31, 32-63, 64-151 s), or `partition_width: 8` |
| 10 | `IG_STEPS = 64` (`A:186`) | 64 steps leaves 3.7e-2 on a tiny row and a 1.1e-1 max at production width | 128 (§3.3) |
| 11 | `SECONDS_PER_STEP = 4.0` (`lag_report.py:54`), `lag_attn.L` (`P:691`), `DENSE_ANCHOR_GEOMETRY = (0,1)` (`metrics.py:132`) | Same: T = 300 tokens of 4 s, L = 38, dense A = 240, anchors 30..269 | none |
| 12 | `layer_attribution`: `posterior_head.fusion` (`A:1358`); `model_lag_readout`: `source_kl_lag_map` (`A:821`); `lag_band`: `attn_weights` (`A:660`); `source_channel_shifts` (`P:1816`) | All present (inherited from `SeqVaeLagAttnTrfRws`). `head_structured = True`, `source_gate = None`. | none |
| 13 | Example pages draw the `(T, c)` input heatmaps (`P:582-588`, `A:1985`). Raw rows come via `traces.attach_raw_signals` (reads `batch.fhr/up`). | The 33-channel token heatmap is legible but not what a reader wants | Raw rows plus raw-resolution maps `(T·R)`: proposal P1's figure |
| 14 | `per_anchor.parquet` (`kld_per_t`, `coverage`) for selection | Comes from the collection pass (core-infra slice) | Same columns on the patch collection |
| 15 | Task 4-tuple `_build_forward_inputs` (`patch task.py:149`) | Not used by attribution | none |

## 3. Wrapping the patch model for Captum

### 3.1 Differentiability of `patchify` (`patching.py:16-50`, `e2e/nets/frontend.py:225-301`)

- **value** = `where(valid, raw, 0)`. **delta** = `(x_n − x_{n−1})·m_n·m_{n−1}`. Both are linear in
  raw with fixed masks; the gradient is the mask. Invalid and non-finite samples get exactly 0.
- **validity** = `amin(mask) − 1`, a boolean function of `weight` and `isfinite(raw)`, with no
  gradient. Along a straight raw path between finite endpoints it is constant, so it is
  **structurally held**.
- **The token-boundary read.** `delta[0]` of token t reads raw sample 16t−1 (the last sample of
  token t−1). So raw support of a source readout at anchor t is `[16(t−37)−1, 16t+15]`. This was
  measured exactly: `[1487, 2095]` at t = 130.
- **NaN.** `alpha·NaN` is NaN, so attribution w.r.t. raw would be NaN at NaN samples even though
  the gradient is 0. `nan_to_num` the raw inputs first. The loader already sanitises, but guard
  anyway.
- **Raw IG and patch IG share one path.** For value and delta, patchify is linear, so the straight
  raw path maps to the straight patch path with validity fixed. The two IGs therefore have the
  same total. They allocate differently: the delta pull-back differs. Measured: per-token |raw| vs
  |patch| correlation 0.96, and sums equal to 1e-6. So the value/delta split of a raw run is one
  extra patch-mode call, or a `LayerIntegratedGradients` call on a `patchify` module's output.

### 3.2 The wrapper (dry-run code, verified)

```python
from teb_vae.lag_attn_cfs.nets.causal_inputs import CausalWarmupInputs
from teb_vae.lag_attn_transformer_patch.nets.patching import patchify

class _CfsCall:                     # eval-only: the patch model under the CFS 3-stream forward
    forward = CausalWarmupInputs.forward          # (y_st=y_patch, y_ph=empty, u_patch, phase, stride)
    def _build_forecast_target(self, summaries, anchors):            # summaries (B,T,2) as target_features
        steps = anchors.long()[:, :, None] + torch.arange(1, self.horizon + 1, device=anchors.device)
        return summaries[torch.arange(summaries.shape[0], device=summaries.device)[:, None, None], steps]

@contextmanager
def cfs_call(model):                # same instance, so attributed_forward's shadows and hooks still apply
    cls = model.__class__; model.__class__ = type("PatchAsCfs", (_CfsCall, cls), {})
    try: yield model
    finally: model.__class__ = cls

class RawReadout(nn.Module):        # raw (B,T,R) streams -> patchify -> the patch-mode AnchorReadout
    def __init__(self, inner):
        super().__init__(); self.inner = inner
        self.model, self.cell, self.readout = inner.model, inner.cell, inner.readout
    def anchor_steps(self, columns): return self.inner.anchor_steps(columns)
    def forward(self, fhr, empty, up, summaries, weight, columns, coordinates):   # A:558's signature
        m, r = self.model, self.model.raw_per_step
        y = patchify(fhr.flatten(1), weight, raw_per_step=r, validity="fhr_weight")
        u = patchify(up.flatten(1), weight, raw_per_step=r, validity=m.source_validity)
        return self.inner(y, y[..., :0], u, summaries, weight, columns, coordinates) + 0.0 * empty.sum()

with cfs_call(model):
    inner = core.AnchorReadout(model, core.ATTENTION_CELL, readout="kld").eval()
    # patch-token mode: core unchanged
    core.integrated_gradients(inner, (y, y[..., :0], u), (summ, weight), columns, baseline="source_null")
    # raw mode: inputs (B,T,16); the target side under all_zero keeps validity by construction
    core.integrated_gradients(RawReadout(inner), (fhr.view(B, T, 16), fhr.view(B, T, 16)[..., :0],
                              up.view(B, T, 16)), (summ, weight), columns, baseline="source_null", n_steps=128)
```

- **Why a class swap and not a wrapper module.** `attributed_forward` (`A:434`) shadows seams on
  the instance it is given, and Grad-CAM registers hooks via `model.__call__` (`G:153-157`). A
  delegating wrapper module would leave the inner net unshadowed (dense decode, sampled ε, silently
  wrong). Calling `CWI.forward(model, ...)` directly would skip the forward hook. The swap keeps
  both, and it restores the class on exit (verified).
- **What else this unlocks.** With the swap plus a patch `model_inputs` (§2 #1), the whole existing
  `run_pass` and `run_cohort` run in **patch-token mode**. That covers selection, layer split,
  ablation, lag agreement, Grad-CAM, class contrast and traces. The only gap is §2 #4 (all-zero
  validity).
- **What stays new for raw mode.** `attribute_batch` builds `core.AnchorReadout` itself and calls
  the model on its inputs directly (`P:684`). A raw-mode pass therefore needs either a
  `wrapper_factory` / `prepare(y_st, y_ph, u, weight)` hook threaded through `attribute_batch`
  (about 6 core lines), or its own small pass. The second is recommended, because the raw analyses
  (§5) need their own aggregation anyway. It would reuse `expand_rows`, `integrated_gradients`,
  `layer_attribution`, `ablate_lag_bands`, `gradcam`, `lag_profile`, `offset_channel_map` and
  `agreement`.
- **The `(B, T, R)` reshape is the trick.** With raw as `(B,T,16)`:
  - `time_profile` gives per-token sums.
  - `channel_profile` gives the within-patch sample position.
  - `offset_channel_map` (`A:2492`) gives a `(L, 16)` lag map. Flattened, that is the raw-sample
    lag axis, with s = 16ℓ + 15 − k.
- **Two small breakages.** The empty-result shape in `integrated_gradients` (`A:1154`) assumes 3-D
  inputs, which holds here. The `layer_attribution` / `gradcam` paths work as they are (verified).

### 3.3 Dry-run results

**Tiny model.** `build(max_lag=37)` (d_model 32, L 38) with `perturb_posterior` applied. At init
the posterior head is zero-initialised, so KL ≡ 0 and IG would be vacuous; it must be perturbed.
Stub batch `make_stub_batch(batch=2, seq_len=300)`, anchor column 100 = stored step 130, CPU.
Here `rel` = |δ| / |f(x) − f(x0)|, which is stricter than the pass's `completeness_rel`.

| Run | Result |
|---|---|
| Patch-token, `kld`, `source_null`, 64 steps | f(x) 16.507 / 3.364, f(b) 14.092 / 3.240, f(x0) 14.707 / 3.154. IG sum 1.8041 / 0.2023. **rel 2.6e-3 / 3.7e-2.** 0.67 s for 2 rows. After-anchor 0, outside the lag window 0, validity channel 0. Value/delta \|·\| share 0.39 / 0.61. |
| Raw UP, `kld`, `source_null` | IG sum 1.8041005 vs patch 1.8041009, same rel. **Support exactly `[16(t−37)−1, 16t+15]`.** After-anchor 0. |
| Raw UP, steps 64 / 128 / 256 | rel (row 2) 3.7e-2 / **3.9e-4** / 2.3e-6. 0.64 / 1.05 / 1.97 s. |
| Raw, `kld`, `all_zero` | rel 1.2e-2 / 9.2e-2. FHR after-anchor 0. **FHR gap token: exactly 0.** |
| Raw, `pred_gap`: `source_null` / `all_zero` | rel 4.6e-3, 1.7e-2 / **0.43**. `all_zero` `pred_gap` is unconverged, as is the known CFS issue (`ATTRIBUTION.md:191`). |
| Source-null path f(α) at α = 0 / 1e-6 / 1e-3 / 1 | 14.092 / 15.378 / 14.707 / 16.507. Starting from exact zero, rel is 0.041 / 1.24. **The cause** is the zero-init bias of `PatchEmbedding.adapter.linear` feeding `LayerNorm(0)`. With a random bias, or from a flat −0.5 UP baseline, the path is smooth: f(0) = f(1e-6). A trained checkpoint is probably smooth; keep the entry point and report the jump. |
| `nll_base` source attribution | max \|·\| = **0.0** (target-only purity holds; `prior_availability_input` is off) |
| Per-head layer split | [−0.031, 0.010, 0.182, 1.638]. Sum 1.7995, equal to f(x) − f(x0) 1.7995. |
| Ablation, bands near [0,9] / far [10,37] | Δ = +0.574 / −4.185 (row 1). **`rest` = 0 exactly**: the adapter K/V has a reach of 1 token. |
| Patch-token, `all_zero` (validity zeroed) | rel 8.9e-2, **validity-channel attribution 1.75e-3 ≠ 0**. With a validity-kept baseline the IG sum equals the raw FHR sum (−0.59353 vs −0.59354). |
| `attributed_forward` vs dense forward at means | `kld_per_t` and `mu_full` at the anchor are equal |

**Production width.** `default.yaml` model keys, 4.08 M params, random init plus perturbed
posterior, GTX 1660 Ti 6 GB, 16 rows:

| Run | Time | rel median / max |
|---|---|---|
| `source_null` raw-UP `kld`, 64 steps | 0.18 s/row | 2.1e-2 / 1.1e-1 |
| `source_null` raw-UP `kld`, 128 steps | 0.36 s/row | 3.1e-3 / 1.4e-2 |
| `all_zero`, 128 steps | 0.58 s/row | 3.7e-3 / 2.6e-2 |
| Grad-CAM | about 3 ms/row | n/a |

Peak memory was 3.5 GiB. Re-measure on the trained checkpoint: `entmax` (`use_entmax: true`) makes
the path less smooth than the tiny fixture's softmax.

## 4. Grad-CAM on the patch model

The resolvers find every layer by the names inherited from `SeqVaeLagAttnTrfRws` (verified run:
each view sums to 1; the target view is `(N, 300)`, source and attention are `(N, 38)`).

| View | CFS resolver | Patch module found | Meaning on the patch cell |
|---|---|---|---|
| `target` | `target_encoder.attention_blocks[-1]` input (`G:61-75`) | `CausalTransformerBlock` (the encoder is `conv_blocks` → `attention_blocks` → `output_norm`) | Token-level FHR evidence. It is full-causal, so it can reach t = 0. |
| `source` | Forward hook on `out["source_state"]` (`G:154`) | `source_state` = `source_adapter(u_patch)` = the `PatchEmbedding` output, **which is the K/V stream itself** under `lag_kv_source: adapter`. No source encoder is built. | A CAM over lag ℓ is evidence from raw UP patch t − ℓ: one token, no smearing |
| `attention` | `lag_attn.attn_dropout` output, flipped to lag order (`G:157`, `G:174`) | Present (`LagCrossAttention`) | Gradient-weighted α over 38 lags |

Other natural layers:
- `lag_attn.W_k` / `W_v` (separate `Linear`s): a key-vs-value LayerIG split (P6).
- `posterior_head.fusion` (per head): already the layer-IG layer.
- The `patchify` output, as a module: a value/delta split of raw IG.
- `source_adapter.adapter.linear`: per-input-channel weights.

No in-model layer has a raw-sample time axis, so "raw Grad-CAM" would just be gradient×input. Use IG
for raw resolution and Grad-CAM for cohort-scale token resolution.

## 5. Proposed raw-signal analyses

These are ordered by priority. All are source-side unless stated. They use the §3.2 wrapper, the
high-KL clean anchors, one segment per recording, and per-recording means before class statistics
(`C` unchanged). Costs use 0.36 / 0.58 s per row at 128 steps.

| # | Question | Readout | Input / baseline | Method | Aggregation and figure | Cost | Reading |
|---|---|---|---|---|---|---|---|
| **P1** | **Where in raw UP history does the KL / gain come from, and is it where the attention says?** (lag-resolved raw-UP map) | `kld`, `pred_gap`, new `delta_level` = mean_τ (μ_full − μ_base)[level] (the UP-driven forecast-level shift) | Raw UP `(B,T,16)`, FHR fixed, `source_null` (raw 0 ≡ the controls' null) | IG, 128 steps, entry 1e-3 | Re-index to raw lag s = 16·t_a + 15 − n ∈ [0, 608]: mean signed and \|·\| over anchors, then recordings, then class. Token-binned \|IG\| against `source_kl_lag_map[t_a]` (`agreement`, `A:1471`), and per head K^(m)α^(m) vs IG of `kld_per_t_per_head[m]`. Figure: the raw-lag \|IG\| profile (0.25 s) over the model's K·α step profile, pooled and by class, plus an anchor × lag heatmap for one recording. | 24 seg × 4 anchors × 3 readouts ≈ 2 min; cohort ≈ 9 min | **High `lag_corr` and low JS**: the attention lag is where UP content causally moves the KL. **IG at ℓ ≈ 0-1 while α peaks far**: attention parked on a sink, and the KL is driven by recent UP. **IG peak where α is small**: content acting through keys (selection), not values (see P6). |
| **P2** | **What inside a patch matters?** Sample position; value vs delta | Same as P1 | Raw `(B,T,16)`, plus patch-token IG with a validity-kept baseline (same path, §3.1) | IG (raw) and IG (patch) | Sum over lag-window tokens: a 16-position profile for raw, value and delta. Value share = Σ\|value\| / Σ\|value+delta\|. Edge mass = position 0 delta (the boundary read). Figure: position profiles plus a share bar per readout and class. | One extra IG call per row (≈ +35 s main) | **Delta-dominant**: the model reads UP slope (onset or rise), not level. **Flat position profile**: a level summary. **Edge-heavy**: the patch boundary itself carries signal (grid artefact risk; test with a ±8-sample shift). |
| **P3** | **What FHR history drives the prior?** | New `base_level`, `base_var` = mean_τ μ_base[..., c]; `mu_prior_dim` (top-\|μ\| coordinate); `nll_base` | Raw FHR `(B,T,16)`, UP irrelevant (exactly 0, verified). Two baselines: `all_zero` (population-mean FHR) and **flat at the segment's own pre-anchor median** (removes level, keeps pattern). | IG, 128 steps | \|IG\| by raw lag back to t = 0 (the target encoder is full-causal, and the conv stem reaches 21 tokens = 84 s). Cumulative-share curve: the lag holding 50 % and 90 %. Share within 84 s vs beyond. Figure: lag profiles on a log-y axis plus the cumulative curve by class. | 96 rows × 4 readouts × 0.58 s ≈ 4 min | **90 % within about 1 min**: a near-persistence prior. **Long tail**: the prior tracks baseline or variability trends. This defines what UP must add (read with P1). The level baseline splits "FHR level" from "FHR pattern". |
| **P4** | **Contraction-locked attribution: which phase of a contraction drives the model?** | `kld`, `pred_gap`, `delta_level` | Raw UP. `source_null`, plus a **tone baseline** (UP replaced by the segment's flat 10th-percentile level: contractions removed, tone kept; measured smooth at α → 0). | IG. Contractions via `events.detect_contractions` (`events.py:265`; `onset_raw/peak_raw/end_raw` at 4 Hz). Anchors placed at peak + {0, 15, 30, 60, 90} s (token-rounded). | Event-triggered average of raw-UP IG on a time-relative-to-peak axis, per anchor offset: a 2-D map (offset × time rel. peak), rise / plateau / fall / outside shares, and the mean UP waveform above it. Figure: three rows (mean UP, attribution map, phase shares by class). | About 6 contractions/segment × 5 offsets ≈ 30 rows ≈ 11 s/segment, so 50 segments ≈ 9 min | **Mass on the rising limb or peak at offsets of 15-40 s**: a late-deceleration-like coupling. **Falling limb**: recovery coupling. **No phase locking**: the model reads tone or noise, not contractions. IG(tone → x) vs IG(0 → x) separates contraction content from UP level. |
| **P5** | **Does the gain sit where the KL sits?** (per-anchor coupling of the P1 maps) | `kld` vs `pred_gap` vs `delta_level` maps from P1 | (P1's rows) | none (free) | Per row: Pearson of the \|IG\| raw-lag profiles across readouts, and the KL-to-gain lag shift (argmax cross-correlation). By class and by KL decile. Figure: scatter plus histogram of shifts. | 0 | **Same lags**: the belief change is the useful one. **Gain at other lags**: the KL is spent on non-predictive UP (cf. `lag_high_kl`). |
| **P6** | **Key or value?** Is the lag chosen by UP content (keys → α) or is content carried (values)? | `kld`, `pred_gap` | `source_null` on patch tokens | `LayerIntegratedGradients` on `[lag_attn.W_k, lag_attn.W_v]`, taken at the anchor's window. The two parts sum to the total under `source_null` (the query is target-only). | Per row the key share and value share, plus a per-lag key/value split. Figure: share by class; lag profiles of each. | 1 LayerIG call per row (≈ +35 s) | **Key-dominant**: UP decides *where* to look (a timing signal). **Value-dominant**: the content at a roughly fixed lag drives the update. |
| **P7** | **Do heads specialise in lags?** | New `kld_head[m]` = `kld_per_t_per_head[..., m]` (a 3-line readout) | Raw UP, `source_null` | IG, M = 4 rows per anchor | Per head: raw-lag \|IG\| vs that head's own α^(m) K^(m). Head × lag heatmap and per-head `lag_corr`. | 4× the P1 kld cost (≈ 1 min) | **Distinct per-head lag peaks that match α^(m)**: multi-lag coupling. **All heads identical**: one effective lag channel. |
| **P8** | **Which FHR context makes UP matter?** (interaction) | `src_effect` = kld(fhr, up) − kld(fhr, 0); the same for `pred_gap` | Raw FHR, `all_zero` (UP held at observed and at null inside the readout) | IG; two forwards per step | FHR raw-lag \|IG\| of the source effect. Recent (< 30 s) vs far share. Correlate with the P3 profile. | 0.7-1.2 s/row ≈ 2 min | **Recent FHR dominant**: UP *explains* an ongoing FHR change (explaining away). **Far history**: UP is weighted by FHR state (e.g. variability regime). |

Implementation notes for the proposals:
- **One new module.** Put the readouts `delta_level`, `base_level/var`, `kld_head` and `src_effect`
  in a patch eval module (~300 lines): a `RawReadout` subclass with its own `forward` branch for the
  new names, falling back to `AnchorReadout` for the rest, plus the raw pass and these figures.
  P1+P2+P5 share one set of IG calls. P4 adds `events.detect_contractions` and the tone baseline:
  one extra `baselines` kind, built once per row from raw UP, held fixed through Captum's expansion
  in the same way as `summaries`.
- **Checks.** Keep `completeness_rel` and `after_anchor_max_abs`, the latter on raw: zero beyond
  16t+15. Add:
  - `outside_lag_window_max_abs`: zero before 16(t−37)−1 on the UP side.
  - `invalid_token_max_abs`: zero.
  - `validity_channel_max_abs`: patch mode; zero.
  - `raw_vs_patch_sum_rel`: equal paths, ≤ 1e-4.

## 6. Config and tests

- **Config.** `eval_overrides.yaml` (`:243-267`) keeps `caps.attribution_segments: 24` and
  `attribution_cohort_segments: 150` unchanged; cost is about 2× CFS per row at 128 steps, still
  minutes. `occlusion_bands` must be redone for L = 38 (§2 #9). `IG_STEPS` → 128, either as a
  module constant override or as a new `eval_config` key.
- **Tests to port from `lag_attn_cfs/tests/test_eval_attribution.py`** (fixtures `build()` +
  `make_stub_batch(seq_len=300)` + `perturb_posterior` + the §3.2 shim):
  - The wrapper reads the forward's own quantities (`:85`).
  - No RNG, and the model is restored (`:120`). Add: the class is restored after `cfs_call`.
  - Causality (`:219`). Raw form: support exactly `[16(t−37)−1, 16t+15]`.
  - Target-only purity (`:244`).
  - Completeness from the entry point (`:256`).
  - Batched = one at a time (`:286`).
  - The per-head split sums (`:339`).
  - Ablation reproduces the null and `rest` = 0 (`:354`).
  - New: invalid-token and validity-channel zeros; raw sum equals patch sum.
- **Tests to port from `lag_attn_transformer_cfs/tests/test_eval_class_attribution.py`.** `:39`,
  `:57` and `:64` port unchanged once the shim exists (the views are found by name). `:86` and `:97`
  are model-free and portable as-is.
