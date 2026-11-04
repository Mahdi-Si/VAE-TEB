# `lag_attn_transformer_patch` — approach and specification

Part A explains the approach and the reasons for it. Part B specifies the model as built. Part C
lists the task IDs and the implementation decisions that the code and the other documents refer to.
`DESIGN.md` is the as-built record and says how to run the package.

---

# Part A — Approach

## A.1 What this package is

This package is a cell of the causal model grid. It is `teb_vae/lag_attn_transformer_cfs` (the
"CFS cell") with a different **representation**.

| | CFS cell | This package |
|---|---|---|
| Input token | Causal scattering and phase-harmonic (ST/PH) coefficients, 76 target and 46 source channels | One 4 s raw patch, 16 samples, per stream |
| Forecast target | Next 30 stored ST/PH coefficients over 76 channels | Next 30 per-patch summaries over 2 channels |
| Encoders, prior, lag attention, posterior, decoder, KL readouts | Shared | Shared, imported unchanged |

The model class is `SeqVaeLagAttnTrfPatch`.

## A.2 Why patches replace ST/PH

The reasons below are mathematical.

1. **A patch loses no information.** A patch is a reshape of the raw samples. ST/PH apply a
   modulus and a low-pass filter, and these steps cannot be inverted.
2. **A patch has no filter delay.** Token $t$ reads raw samples $16t \ldots 16t+15$ and nothing
   else. A causal band-pass filter has a group delay of about one over its bandwidth, so a slow
   ST/PH channel reports old content. The CFS cell needs alignment clocks, forecast clocks and
   novelty readouts only to manage these delays.
3. **A patch needs almost no warm-up.** The filter support of ST/PH forces a warm-up of 134 of 300
   steps in the CFS cell. A patch has a support of one token. Only the encoder's convolution stem
   needs a warm-up, and its reach is 21 tokens.
4. **A patch target is a true forecast.** Every target patch lies strictly after the last input
   token. A stored ST/PH coefficient mixes samples from before and after the anchor.

## A.3 Lag resolution and the key/value source

The lag attention must tell lag $\ell$ from lag $\ell + 1$. Assume that its key at lag $\ell$ is a
function of the UP signal over a window of reach $R$ that ends at $t - \ell$. Then the keys at two
lags that are closer than $R$ share most of their input. By the data-processing inequality, the
attention cannot resolve lag much more finely than $R$.

| Key and value source | Reach $R$ |
|---|---|
| UP patch embedding (`lag_kv_source: adapter`) | 1 token (4 s) |
| Gated causal conv stem (`lag_kv_source: conv_stem`) | The stem reach, 21 tokens at the shipped kernels and dilations |
| Windowed source encoder (`lag_kv_source: encoder`) | $R_U = R_{\mathrm{conv}} + N_U (W_U - 1)$, with $N_U$ = `source_attention_blocks` and $W_U$ = `source_attention_window` |

With local values (`adapter`), the attention is a data-dependent FIR filter (finite impulse
response filter) over the UP history, and the lag map is the kernel of that filter. The shipped
`default.yaml` uses `lag_kv_source: encoder`. The arm `sweep_lag_kv_encoder.yaml` selects
`adapter`.

## A.4 Why the target is per-patch summaries

The target decides what the latent state must encode. FHR is a slow part plus a fast part:

- The **slow part** is the baseline and the decelerations. Fetal state and UP determine it.
- The **fast part** is beat-to-beat variation. Only its variance can be forecast.

The candidate targets behave as follows:

1. **Raw 480 samples with a factorized Gaussian.** The model pays for one slow error 480 times. The
   score then overstates the density of strongly correlated samples.
2. **Raw 480 samples with an AR(1) residual at 4 Hz.** A slow error $e$ becomes innovations of
   about $(1 - \phi)e$. With $\phi$ near 1, the score almost ignores a correct deceleration.
3. **Future PH coefficients.** The relative phase of two band-limited signals decorrelates within
   about one inverse bandwidth. After that time, a PH coefficient is a random sign that no model can
   forecast.
4. **Per-patch summaries.** The coordinates are the quantities the downstream task needs, and an
   AR(1) residual, when it is switched on, acts over 30 patch steps instead of 480 raw samples.

This package uses option 4 with two channels per patch:

- **`level`**: the mean of the patch. It is a 4 s epoch average and acts as a low-pass filter of
  about 0.1 Hz. Computerized CTG analysis (Dawes–Redman) uses a similar 3.75 s epoch.
- **`variability`**: the log of the RMS of the 15 first differences inside the patch. This is a
  short-term variability proxy. Differences, not the standard deviation, are used because a slope
  inside a deceleration must not count as variability.

## A.5 Objective switches

| Switch | Shipped value | Meaning |
|---|---|---|
| `forecast_ar_residual` | `false` | The factorised Gaussian NLL scores each of the $H \cdot C$ cells on its own. `true` scores AR(1) innovations along the horizon, with $\phi_c = \tanh(a_c)$. |
| `horizon_weight_halflife_steps` | `null` | A weighted score is not a log-density. |
| `persistence_residual` | `false` | With persistence the latent can stop encoding the current FHR level. With `false` the level reaches the decoder only through the latent, which the classifier reads. |
| `prior_availability_input` | `false` | The prior clock cancels the warm-up availability staircase of ST/PH. Patches have no staircase. |
| Tiled anchors, random phase | stride 15 | An unbiased subsample of the dense loss, at a lower decoder cost. |

## A.6 Scope

**In scope:** the model, the task, the trainer, configs, tests, a planted-delay check, classifier
compatibility, the evaluation binding with its raw-signal analyses (`eval/`), and a design record.

**Out of scope for v1:**
- a cross-segment ("Level 2") model, because `teb_vae/classifier` already has a sequence scope;
- UP dropout, robust UP scaling and a UP flat-run detector;
- patch sizes other than 16 samples.

---

# Part B — Specification

## B.1 Geometry

All numbers are for the shipped default. The loader trims 1 min from each end of a 22 min segment.

| Quantity | Symbol | Value |
|---|---|---:|
| Raw samples per token | $R$ | 16 |
| Tokens per segment | $T$ | 300 (raw length 4800) |
| Horizon | $H$ | 30 tokens = 120 s |
| Warm-up (anchor floor) | $F$ | 30 tokens |
| Anchor ceiling | $T_{\mathrm{valid}} = T - H$ | 270 |
| Dense anchors (val and test) | | 240 |
| Training stride | $S$ | 15, so $A_{\max} = \lceil 240 / 15 \rceil = 16$ |
| Lag window | `max_lag` | 37 (38 lags, 148 s) |
| Input features per token | $F_{\mathrm{in}} = 2R + 1$ | 33 |
| Target channels | $C$ | 2 |
| Scored block per anchor | $H \cdot C$ | 60 |

The encoder stem reaches 21 tokens. The warm-up of 30 tokens covers it.

## B.2 Data flow

```
batch.fhr (B,4800), batch.up (B,4800), batch.weight (B,300)
        │  task: patchify()            [nets/patching.py, pure torch]
        ▼
y_patch (B,300,33)    u_patch (B,300,33)
        │  model.forward(y_patch, u_patch, anchor_phase, anchor_stride)
        ▼
target_adapter = PatchEmbedding ──► target_encoder ──► prior p(z|Y≤t)
source_adapter = PatchEmbedding ──► encode_source_kv (lag_kv_source) ──► lag attention keys/values
        ▼
posterior q(z|Y≤t,U≤t) ─► shared decoder ×2 ─► mu/logvar_{base,full} (B,A,30,2)
        │  model.compute_loss(outputs, fhr_raw, weight=...)
        ▼
patch_summaries(patchify(fhr)) gathered at t+1..t+H ─► shared objective (block_width=2)
```

## B.3 `patchify` (`nets/patching.py`)

`patchify(raw, weight, *, raw_per_step, validity)` returns a `(B, T, 2R+1)` tensor.

1. `teb_vae.lag_attn_transformer_e2e.nets.frontend.featurize(raw, w)` returns one stacked
   `(B, 3, L)` tensor; `.unbind(1)` gives `(value, mask, delta)` at the raw rate.
2. Under `validity="fhr_weight"`, `w = weight`. The target stream always uses this rule.
3. Under `validity="finite"`, `w = ones_like(weight)`, so only non-finite samples are invalid. The
   source stream uses the model's `source_validity`, which defaults to `"finite"`, because `weight`
   describes FHR, not UP.
4. `value` and `delta` are reshaped to `(B, T, R)`.
5. The token validity $m_t$ is the minimum of the per-sample mask over the patch.
6. The output is `[value (R), delta (R), m_t − 1 (1)]` along the last axis.

**Why the validity channel is $m_t - 1$.** A fully valid stream then has a zero in that channel,
and an invalid token has a non-zero input vector. Two consequences follow:

- An all-zero stream means "valid, flat at the mean level, no slope". This is the null for the
  shared source-null control (`controls.source_null_kld`), which feeds `zeros` of the stream shape.
- No invalid token reaches a normalization layer as an exact zero vector.

This follows the existing `AvailabilityInputAdapter` convention $W_m(m - \mathbf{1})$.

`patch_summaries(values, valid, *, eps)` returns `(…, 2)`:

- `level` is the mean of the 16 values.
- `variability` is $\log\big(\sqrt{\mathrm{mean}(d^2)} + \varepsilon\big)$, with $d$ the 15 first
  differences inside the patch. The first element of `delta` is not used, because it crosses the
  patch boundary.
- Invalid cells (`valid < 0.5`) are 0.0 in both channels.

The target builder and the persistence path both call this one function.

## B.4 `PatchEmbedding` (`nets/patching.py`)

- It wraps `AvailabilityInputAdapter(in_dim=33, d_model, sequence_length, dropout, delays=None)`
  from `teb_vae.lag_attn.nets.encoders`.
- It adds one learned parameter `missing` of shape `(d_model,)`, initialized with
  `START_EMBED_STD` from the same module. The generic `initialization()` pass does not touch a bare
  `nn.Parameter`.
- Forward: `e = adapter(x)`, then `torch.where(x[..., -1:] + 1 > 0.5, e, missing)`.
- The forward has no Python branch on a tensor value. DDP runs with `find_unused_parameters=False`.

## B.5 Model composition

```python
class SeqVaeLagAttnTrfPatch(PatchStreamInputs, PatchSummaryTarget, SeqVaeLagAttnTrfRws): ...
```

- **`SeqVaeLagAttnTrfRws`** (`teb_vae/lag_attn_transformer_rws/nets/model.py`) supplies the
  encoders, heads, lag attention, decoder and `lag_kv_source`.
- **`PatchStreamInputs(CausalWarmupInputs)`** (`nets/patch_inputs.py`) reuses the tiled-anchor
  forward of `teb_vae/lag_attn_cfs/nets/causal_inputs.py` with no gates, no warm-up vectors and no
  alignment. It overrides `_build_adapter` to return `PatchEmbedding`, and it makes
  `_check_anchor_floor` and `_resolve_warmup_readout_constants` no-ops. Its
  `forward(y_patch, u_patch, anchor_phase=None, anchor_stride=None)` delegates to
  `CausalWarmupInputs.forward(self, y_patch, y_patch[..., :0], u_patch, anchor_phase,
  anchor_stride)`. The empty second block is the reason for the wrapper.
- **`PatchSummaryTarget`** (`nets/patch_target.py`):
  - sets the decoder width to 2 through `_default_decoder_out_channels`;
  - owns `compute_loss` and `_build_forecast_target`;
  - binds `_set_likelihood_structure`, `_register_likelihood_structure` and `_anchors_per_sample`
    from `CausalFeatureForecastTarget`, and `forecast_likelihood_kwargs` from
    `FeatureForecastTarget`, by reference;
  - supplies `scored_weight` (the identity), `_anchor_target_values` (the persistence input) and a
    no-op `_check_persistence_target`.
- **Base order is load-bearing.** The mixins come first. See `lag_attn_transformer_cfs/DESIGN.md`
  §6 for the failure the reverse order causes.

**Constructor.** The keyword list is that of `SeqVaeLagAttnTrfCrws.__init__`
(`teb_vae/lag_attn_transformer_crws/nets/model.py`) with these changes:

- **Removed** keywords: `c_y`, `c_u`, `use_up_st`, `target_keep_index`, `target_warmup_steps`,
  `source_keep_index`, `source_warmup_steps`, `target_align_delays` and `source_align_delays`. The
  trainer pre-flight refuses each one by name.
- **Added** keywords:

  | Keyword | Default | Meaning |
  |---|---|---|
  | `forecast_ar_residual` | `False` | AR(1) residual likelihood, $\phi_c = \tanh(a_c)$ seeded at 0 |
  | `persistence_residual` | `False` | The persistence residual; `sweep_persistence.yaml` sets it |
  | `target_summary_loc` | `(0.0, 0.0)` | Per-channel location of the affine standardization |
  | `target_summary_scale` | `(1.0, 1.0)` | Per-channel scale of the affine standardization |
  | `variability_eps` | `0.01` | $\varepsilon$ in loader z-units; `summary_stats.py` prints it |
  | `source_validity` | `"finite"` | `"finite"` or `"fhr_weight"`; read by the task |

- The constructor passes `c_y = c_u = 2 * raw_per_step + 1`, `use_up_st=False` and
  `target_delays = source_delays = None` to the base.
- It captures the forwarded keywords with the `locals()` pattern, excluding
  `FORWARDED_EXCLUSIONS + PATCH_ONLY`.

## B.6 `compute_loss`

1. `summary_target(fhr_raw, weight)` calls `patchify(..., validity="fhr_weight")`, then
   `patch_summaries`, then the affine standardization. The result is `(B, T, 2)`.
2. `_build_forecast_target` gathers at `anchor_index`: token $a + 1 + \tau$ for
   $\tau \in [0, H)$, giving `(B, A, H, 2)`. When the dict has no anchor set, the dense range
   `arange(anchor_ceiling)` is used.
3. `teb_vae.lag_attn_rws.nets.losses.compute_loss` scores the block with
   `block_width=decoder_out_channels` (2), `horizon_weight=getattr(self, "horizon_weight", None)`
   and `**forecast_likelihood_kwargs()`. The last gives `cell_mask=None`, and `ar_coef=tanh(a)`
   when AR is on, else `None`.
4. The shared objective builds the forecast mask from `weight`, so a target patch that is invalid
   in `weight` is not scored. A non-finite sample inside a valid patch still gives a finite target,
   because `featurize` sets it to 0.
5. `anchors_per_sample` is merged into the metrics.

## B.7 Task (`task.py`)

The task is `SeqVaeLagAttnTrfPatchTask(SeqVaeLagAttnTrfCrwsTask)`. The parent carries `seed`,
`_stage`, `compute_loss_and_metrics`, the bound CFS members `anchor_phase`, `_phase_field`,
`resolve_anchor_geometry` and `_mu_gap_rms`, and the LR ramp. The task writes:

- `_build_forward_inputs`, which returns `(y_patch, u_patch, phase, stride)`;
- `_added_metrics`, which calls `controls.source_null_kld` with `inputs[1]` on val and test, because
  the CFS version reads `inputs[2]`;
- the page seams `forecast_rows`, `forecast_extra_rows` and `input_stream_panels`, which draw the
  evaluation's patch page (`eval/analyses/samples.py`);
- `input_budget_figure`, which returns `None`.

## B.8 Trainer (`trainer.py`)

The trainer is `LagAttnTrfPatchTrainer(LagAttnTrfRwsTrainer)` with these attributes:

- `MODEL_CLS = SeqVaeLagAttnTrfPatch`
- `TASK_CLS = SeqVaeLagAttnTrfPatchTask`
- `CHECKPOINT_STEM = "lag-attn-trf-patch"`

Its pre-flight:
- refuses every key in `REMOVED_MODEL_KEYS` (the B.5 removals plus `target_delays`,
  `source_delays`, `target_scored_horizon`, `target_forecast_shift` and `target_novelty_frac`) and
  every key with the prefix `causal_`, `target_weight_` or `target_phase_fast_`;
- requires `fhr` and `up` in both `load_fields` and `normalize_fields`;
- requires `weight`, `guid` and `epoch` in `load_fields`;
- checks the raw length against the shard after the trim;
- requires `lambda_boundary` to be 0.

`PLOT_CONFIG_KEY` keeps its inherited value. `RUN_CONFIG` and `main` follow the sibling trainers.

## B.9 Configuration

`configs/default.yaml` is `teb_vae/lag_attn_transformer_cfs/configs/default.yaml` with only this
delta. `tests/test_config_load.py` asserts it (`CFS_DELTA`).

| Leaf | This package |
|---|---|
| `warmup_period` | 30 |
| Every `causal_*` key, `c_y`, `c_u`, `use_up_st`, `target_weight_*`, `target_phase_fast_*` | absent |
| `posterior_event_gate`, `lambda_event_gate`, `event_weight_floor`, `event_weight_reach_steps` | absent |
| `persistence_residual` | `false` |
| `prior_availability_input` | `false` |
| `horizon_weight_halflife_steps` | `null` |
| `forecast_ar_residual` | `false` |
| `lag_kv_source` | `encoder` |
| `source_dropout` | `null` |
| `free_bits` | 0.02 |
| `beta_schedule.warmup_epochs` | 100 |
| `general_config.lr_milestone` | `[80, 110]` |
| `target_summary_loc`, `target_summary_scale`, `variability_eps` | `[0, 0]`, `[1, 1]`, `0.01` (provisional, see `DESIGN.md` §3) |
| `source_validity` | `finite` |
| `load_fields` | `[fhr, up, weight, guid, epoch]` |
| `normalize_fields` | `[fhr, up]` |
| `gradient_clip_val` | 280 (the CFS value scaled by 60/2280; provisional) |
| `additive_margin` | 175 (the CFS value scaled by 60/2280; provisional) |
| Identity keys (`tag`, `run_name`, `tags.variant`, output tree, experiment name) | this package's own |

The loss-scale constants are provisional. Re-derive them with the procedure in
`lag_attn_transformer_e2e/DESIGN.md` §12. The shards may be any existing build, because only `fhr`,
`up` and `weight` are read. For a comparison against CFS, use the same shard folds and the same
stats file.

**Arms.** Each arm is the default plus one named leaf plus the identity keys:

| File | Leaf | Question |
|---|---|---|
| `sweep_warmup_134.yaml` | `warmup_period: 134` | Does a difference come from the representation, or from the extra anchors that $F = 30$ allows? |
| `sweep_lag_kv_encoder.yaml` | `lag_kv_source: adapter` | Does lag locality matter? |
| `sweep_persistence.yaml` | `persistence_residual: true` | Does the level leave the latent? |
| `sweep_source_validity_fhr.yaml` | `source_validity: fhr_weight` | Does the UP mask choice matter? |
| `planted.yaml` | `max_lag: 60`, the planted shard, and the leaves in `DESIGN.md` §2 | Can the model recover a known delay? (a check, not an arm) |

## B.10 Invariants every implementation must keep

1. **Raw causality.** States at token $t$ are bitwise unchanged when you perturb raw sample
   $16t + 16$ or later. They move when you perturb sample $16t + 15$. Test in float64.
2. **Source purity.** The prior never reads the source. The source path never reads the target.
3. **Exact zero KL at init.** Posterior deltas are zeroed after the generic initialization. Use
   the `perturb_posterior` fixture for every KL test, because a fresh model passes any KL test
   trivially.
4. **One decoder, invoked twice**, with no source bypass.
5. **DDP reachability.** Every trainable parameter receives a gradient on these batches:
   - a fully valid batch;
   - a batch with an invalid patch;
   - a fully invalid batch.

   This includes `missing`.
6. **The lag attribution identity.** `source_kl_lag_map` sums over lags to `kld_per_t`.
7. **One definition of the target.** The target builder and the persistence path call
   `patch_summaries`.

---

# Part C — Implementation record

## C.1 Task IDs

The code and the configs cite these IDs.

| ID | Title | Artefact |
|---|---|---|
| P0-01 | Map the co-mixin contract | `notes/CONTRACT.md` |
| P1-01, P1-02 | `patchify`, `patch_summaries`, `PatchEmbedding` | `nets/patching.py` |
| P1-03 | Target statistics script | `summary_stats.py` |
| P1-04 | Package skeleton | `__init__.py`, `nets/__init__.py`, `tests/__init__.py`, `tests/conftest.py` |
| P2-01 | `PatchStreamInputs` mixin | `nets/patch_inputs.py` |
| P2-02 | `PatchSummaryTarget` mixin | `nets/patch_target.py` |
| P2-03 | Model constructor | `nets/model.py` |
| P3-01 | Task | `task.py` |
| P3-02 | Trainer and configs | `trainer.py`, `configs/` |
| P4-01 … P4-08 | Tests | `tests/` (`DESIGN.md` §2 lists the files) |
| P5-01 | Planted-delay check | `lag_recovery_check.py`, `tests/test_lag_recovery.py` |
| P5-02 | Classifier compatibility | `teb_vae/classifier/tests/test_patch_source.py` (no classifier source change) |
| P6-01 | Design record | `DESIGN.md` |

## C.2 Where the invariants are tested

| B.10 item | Test |
|---|---|
| 1 | `tests/test_model.py::test_states_at_token_t_read_no_raw_sample_after_16t_plus_15` |
| 2, 4 | `tests/test_model.py::test_source_purity_and_one_decoder_invoked_twice` |
| 3, 6 | `tests/test_model.py::test_kl_is_exactly_zero_at_init_and_its_lag_map_sums_to_it` |
| 5 | `tests/test_model.py::test_every_trainable_parameter_gets_a_gradient` |
| 7 | `tests/test_objective.py::test_target_and_persistence_are_standardized_summaries_of_the_right_tokens` |

## C.3 Implementation decisions

Each row records where the code differs from a first reading of Part B. `notes/CONTRACT.md` §10
has the detail.

| # | Topic | What the code does |
|---|---|---|
| C1 | `featurize` output | It returns one stacked `(B, 3, L)` tensor; `patchify` unpacks it with `.unbind(1)` |
| C2 | Persistence | The base raises in `_check_persistence_target` when `persistence_residual=True`; `PatchSummaryTarget` overrides it as a no-op, as CFT does |
| C3 | Hook list | The hooks that `CausalWarmupInputs` reads are listed in `CONTRACT.md` §1 |
| C4 | Forwarded keywords | `FORWARDED_EXCLUSIONS` lacks the four new keys, so the constructor extends it locally with `PATCH_ONLY` |
| C5 | Horizon weight | The loss passes `getattr(self, "horizon_weight", None)`; it is `None` at the shipped `null` |
| C6 | Task parent | The task subclasses `SeqVaeLagAttnTrfCrwsTask`, which has the binds, `seed`, `_stage` and `compute_loss_and_metrics` |
| C7 | Seed | `create_model` applies `general_config.seed` to the task, as CRWS does |
| C8 | Pre-flight | One short refusal, plus the two raw e2e guards and the CRWS `_check_phase_key_fields`, `_check_raw_target_fields` and `_check_boundary_term_is_off` as they are |
| C9 | `use_up_st` | The constructor passes `use_up_st=False` to the base |
| C10 | Tracked metrics and startup message | `TRACKED_METRICS` adds `anchors_per_sample` and `kld_source_null`; `causal_standing_message` states the patch causality |
| C11 | Classifier | The classifier has no package registry; no classifier source changes, one new test |
| C12 | Summary constants | `default.yaml` ships the identity values `[0, 0]`, `[1, 1]` and `0.01`, marked PROVISIONAL. `tiny.yaml` and `planted.yaml` carry values from `summary_stats.py` on their own fixtures |
| C13 | Removed CFS keys | `target_phase_fast_cutoff_hz` and `target_phase_fast_horizon` are absent from `default.yaml`; the pre-flight refuses `target_phase_fast_*`, `target_delays` and `source_delays` by name |
| C14 | `planted.yaml` | It carries the three CFS planted-check leaves `anchor_stride: 1`, `source_dropout: null` and `forecast_ar_residual: false` |

---

## Appendix — file pointers

| Need | Where |
|---|---|
| Architecture parent | `teb_vae/lag_attn_transformer_rws/nets/model.py` (`_build_adapter`, `encode_source_kv`, `lag_kv_source`) |
| Tiled-anchor forward | `teb_vae/lag_attn_cfs/nets/causal_inputs.py` (`_build_anchor_index`, `forward`) |
| Raw-target precedent | `teb_vae/lag_attn_crws/nets/causal_raw_inputs.py`, `teb_vae/lag_attn_crws/task.py` |
| Constructor precedent | `teb_vae/lag_attn_transformer_crws/nets/model.py` |
| Shared objective | `teb_vae/lag_attn_rws/nets/losses.py` (`compute_loss`, `raw_sample_score`) |
| Controls | `teb_vae/lag_attn_rws/nets/controls.py` |
| Raw featurization and pre-flight guards | `teb_vae/lag_attn_transformer_e2e/nets/frontend.py` (`featurize`), `teb_vae/lag_attn_transformer_e2e/trainer.py` |
| Adapter and start-embedding constant | `teb_vae/lag_attn/nets/encoders.py` (`AvailabilityInputAdapter`, `START_EMBED_STD`) |
| CFS shipped config | `teb_vae/lag_attn_transformer_cfs/configs/default.yaml` |
| Fixtures | `teb_vae/lag_attn/tests/fixtures/` (`tiny_shard.hdf5`, `tiny_shard_causal_planted.hdf5`, stats files) |
| Downstream classifier | `teb_vae/classifier/sources.py` (`VaeSource`) |
