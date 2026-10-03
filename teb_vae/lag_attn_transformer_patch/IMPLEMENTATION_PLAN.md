# `lag_attn_transformer_patch` — approach and implementation plan

Written 2026-10-03. **Status: plan only. No code exists in this package yet.**

This document is the work order for a coding agent that orchestrates a swarm of sub-agents. Part A
explains the approach and the reasons for it. Part B specifies the model. Part C is the task list,
with dependencies, owned files and acceptance criteria.

---

## 0. Rules for the orchestrator

1. Read Part A and Part B completely before you assign a task.
2. Run task P0-01 first. Every later task depends on its notes.
3. Assign one task to one agent. Give that agent the task text, Part B and the P0-01 notes.
4. Run tasks in parallel only where §C.1 marks them as parallel.
5. Do not change any file outside `teb_vae/lag_attn_transformer_patch/`, except where a task names
   the file. Shared packages are imported, never edited.
6. Do not commit, push or open a pull request. The user reviews the work first.
7. Run only local checks: the unit tests and the `tiny.yaml` smoke fit on CPU or on one GPU.
8. Do not start a production training run. The user runs training and evaluation on another machine.
9. When a task finds that this plan is wrong, stop that task. Write the conflict into §C.3 and ask
   the user.
10. Update the tracker in §C.2 when a task starts and when it finishes.

Run the gate from the repository root:

```bash
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_patch/tests -q -m "not slow"
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_patch/tests -q -m slow
```

---

# Part A — Approach

## A.1 What this package is

This package is a new cell of the causal model grid. It is
`teb_vae/lag_attn_transformer_cfs` (the "CFS cell") with a different **representation**. Nothing
else changes on purpose.

| | CFS cell | This package |
|---|---|---|
| Input token | Causal scattering and phase-harmonic (ST/PH) coefficients, 76 target and 46 source channels | One 4 s raw patch, 16 samples, per stream |
| Forecast target | Next 30 stored ST/PH coefficients over 76 channels | Next 30 per-patch summaries over 2 channels |
| Encoders, prior, lag attention, posterior, decoder, KL readouts | Shared | Shared, imported unchanged |

The working name of the model class is `SeqVaeLagAttnTrfPatch`.

## A.2 Why patches replace ST/PH

The reasons below are mathematical. They do not depend on any earlier training result.

1. **A patch loses no information.** A patch is a reshape of the raw samples. ST/PH apply a
   modulus and a low-pass filter, and these steps cannot be inverted.
2. **A patch has no group delay.** Token $t$ reads raw samples $16t \ldots 16t+15$ and nothing
   else. A causal band-pass filter has a group delay of about one over its bandwidth, so a slow
   ST/PH channel reports content that is hundreds of seconds old. The CFS cell needs alignment
   clocks, forecast clocks and novelty readouts only to manage these delays.
3. **A patch needs almost no warm-up.** The filter support of ST/PH forces a warm-up of 134 of 300
   steps. A patch has a support of one token. Only the encoder's convolution stem needs a warm-up,
   and its reach is 21 tokens.
4. **A patch target is a true forecast.** Every target patch lies strictly after the last input
   token. A stored ST/PH coefficient mixes samples from before and after the anchor.

## A.3 Why the lag attention reads the patch embedding

The lag attention must tell lag $\ell$ from lag $\ell + 1$. Assume that its key at lag $\ell$ is a
function of the UP signal over a window of reach $R$ that ends at $t - \ell$. Then the keys at two
lags that are closer than $R$ share most of their input. By the data-processing inequality, the
attention cannot resolve lag much more finely than $R$.

| Key and value source | Reach $R$ | Lag resolution |
|---|---:|---|
| UP patch embedding (`lag_kv_source: adapter`) | 4 s | One token |
| `lag_attn_transformer_e2e` front end | 80.5 s | About 20 tokens |
| Deep source encoder (`lag_kv_source: encoder`) | Several minutes | Very poor |

With local values, the attention is a data-dependent FIR filter (finite impulse response filter)
over the UP history. The lag map is then the kernel of that filter. This is the main reason for the
patch design, and the reason not to use the e2e front end.

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
4. **Per-patch summaries.** The coordinates are the quantities the downstream task needs. An AR(1)
   residual over 30 patch steps is then mild and correct.

This package uses option 4 with two channels per patch:

- **`level`**: the mean of the patch. It is a 4 s epoch average and acts as a low-pass filter of
  about 0.1 Hz. Computerized CTG analysis (Dawes–Redman) uses a similar 3.75 s epoch.
- **`variability`**: the log of the RMS of the 15 first differences inside the patch. This is a
  short-term variability proxy. Differences, not the standard deviation, are used because a slope
  inside a deceleration must not count as variability.

## A.5 Objective switches, and why

| Switch | Value | Reason |
|---|---|---|
| `forecast_ar_residual` | `true` | One latent forecasts all 30 steps, so the errors correlate along the horizon. AR(1) innovations give the exact joint density. |
| `horizon_weight_halflife_steps` | `null` | A weighted score is not a log-density. AR(1) already removes the repeated counting. |
| `persistence_residual` | `false` | With persistence the latent can stop encoding the current FHR level. The baseline level is diagnostic, and the classifier reads the latent. |
| `prior_availability_input` | `false` | The prior clock cancels the warm-up availability staircase of ST/PH. Patches have no staircase. |
| Tiled anchors, random phase | stride 15 | This is an unbiased subsample of the dense loss and it saves decoder cost. |

## A.6 Scope

**In scope:** the model, the task, the trainer, configs, tests, a planted-delay check, classifier
compatibility, and a short design record.

**Out of scope for v1:**
- the shared evaluation pipeline in `teb_vae/lag_attn_cfs/eval` (it is feature-domain specific);
- a cross-segment ("Level 2") model, because `teb_vae/classifier` already has a sequence scope;
- UP dropout, robust UP scaling and a UP flat-run detector;
- patch sizes other than 16 samples;
- every item of the external research proposal other than patch tokenization.

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
source_adapter = PatchEmbedding ──► (lag_kv_source: adapter) ──► lag attention keys/values
        ▼
posterior q(z|Y≤t,U≤t) ─► shared decoder ×2 ─► mu/logvar_{base,full} (B,A,30,2)
        │  model.compute_loss(outputs, fhr_raw, weight=...)
        ▼
patch_summaries(patchify(fhr)) gathered at t+1..t+H ─► shared objective (block_width=2, ar_coef)
```

## B.3 `patchify` (new, `nets/patching.py`)

`patchify(raw, weight, *, raw_per_step, validity)` returns a `(B, T, 2R+1)` tensor.

1. Call `teb_vae.lag_attn_transformer_e2e.nets.frontend.featurize(raw, w)` to get
   `(value, mask, delta)` at the raw rate. Import it; do not copy it.
2. For the target stream, set `w = weight`.
3. For the source stream, set `w = ones_like(weight)` under the default `source_validity:
   "finite"`, so that only non-finite samples are invalid. `weight` describes FHR, not UP.
4. Reshape `value` and `delta` to `(B, T, R)`.
5. Compute the token validity $m_t$ as the minimum of the per-sample mask over the patch.
6. Concatenate `[value (R), delta (R), m_t − 1 (1)]` along the last axis.

**Why the validity channel is $m_t - 1$.** A fully valid stream then has a zero in that channel,
and an invalid token has a non-zero input vector. Two consequences follow:

- An all-zero stream means "valid, flat at the mean level, no slope". This is the correct null for
  the shared source-null control (`controls.source_null_kld`), which feeds `zeros` of the stream
  shape.
- No invalid token reaches a normalization layer as an exact zero vector. The e2e design record
  (§8) measured a gradient norm of $2.1 \times 10^{17}$ from such a token.

This follows the existing `AvailabilityInputAdapter` convention $W_m(m - \mathbf{1})$.

`patch_summaries(values, valid, *, eps)` returns `(…, 2)`:

- `level` is the mean of the 16 values.
- `variability` is $\log\big(\sqrt{\mathrm{mean}(d^2)} + \varepsilon\big)$, with $d$ the 15 first
  differences inside the patch. Do not use the first element of `delta`, because it crosses the
  patch boundary.

Both the target builder and any persistence path must call this one function.

## B.4 `PatchEmbedding` (new, `nets/patching.py`)

- Wrap the existing adapter stack: `AvailabilityInputAdapter(in_dim=33, d_model, sequence_length,
  dropout, delays=None)` from `teb_vae.lag_attn.nets.encoders`.
- Add one learned parameter `missing` of shape `(d_model,)`. Initialize it with
  `START_EMBED_STD` from the same module. The generic `initialization()` pass does not touch a bare
  `nn.Parameter`; confirm this in a test.
- Forward: `e = adapter(x)`, `m = x[..., -1] + 1`, `out = torch.where(m[..., None] > 0.5, e,
  missing)`.
- Do not branch on a tensor value in Python. DDP runs with `find_unused_parameters=False`.

## B.5 Model composition

```python
class SeqVaeLagAttnTrfPatch(PatchStreamInputs, PatchSummaryTarget, SeqVaeLagAttnTrfRws): ...
```

- **`SeqVaeLagAttnTrfRws`** (`teb_vae/lag_attn_transformer_rws/nets/model.py`) supplies the
  encoders, heads, lag attention, decoder and `lag_kv_source`.
- **`PatchStreamInputs(CausalWarmupInputs)`** (new, `nets/patch_inputs.py`) reuses the tiled-anchor
  forward of `teb_vae/lag_attn_cfs/nets/causal_inputs.py` with no gates, no warm-up vectors and no
  alignment. It overrides `_build_adapter` to return `PatchEmbedding`. It defines
  `forward(y_patch, u_patch, anchor_phase=None, anchor_stride=None)`, which delegates to
  `CausalWarmupInputs.forward(self, y_patch, y_patch[..., :0], u_patch, anchor_phase,
  anchor_stride)`. The empty second block is the reason for the wrapper; document it there.
- **`PatchSummaryTarget`** (new, `nets/patch_target.py`) does four things:
  - It sets the decoder width to 2 through `_default_decoder_out_channels`.
  - It owns `compute_loss`.
  - It registers the AR(1) parameter.
  - It supplies every hook that `CausalWarmupInputs` reads from its target co-mixin (P0-01 lists
    them).
- **Base order is load-bearing.** The mixins come first. See `lag_attn_transformer_cfs/DESIGN.md`
  §6 for the failure the reverse order causes.

**Constructor.** Copy the keyword list of `SeqVaeLagAttnTrfCrws.__init__`
(`teb_vae/lag_attn_transformer_crws/nets/model.py`) and change it as follows.

- **Remove** these keywords. The trainer pre-flight refuses each one by name:
  - `c_y`, `c_u`, `use_up_st`
  - `target_keep_index`, `target_warmup_steps`, `source_keep_index`, `source_warmup_steps`
  - `target_align_delays`, `source_align_delays`
- **Add** these keywords:

  | Keyword | Default | Meaning |
  |---|---|---|
  | `forecast_ar_residual` | `False` (config: `true`) | AR(1) residual likelihood, $\phi_c = \tanh(a_c)$ seeded at 0 |
  | `persistence_residual` | `False` | Kept only as an ablation arm |
  | `target_summary_loc` | `(0.0, 0.0)` | Affine standardization of the two target channels |
  | `target_summary_scale` | `(1.0, 1.0)` | Same |
  | `variability_eps` | `0.01` | $\varepsilon$ in z units; set from P1-03 |
  | `source_validity` | `"finite"` | `"finite"` or `"fhr_weight"` |

- **Pass to the base** `c_y = c_u = 2 * raw_per_step + 1`, and `target_delays = source_delays =
  None`.
- **Capture the forwarded keywords** with the `locals()` pattern the CFS and CRWS constructors use.

## B.6 `compute_loss`

1. Call `patchify(fhr_raw, weight, validity="fhr_weight")`, then `patch_summaries`, then the affine
   standardization. The result is `(B, T, 2)`.
2. Gather at `anchor_index`: index $a + 1 + \tau$ for $\tau \in [0, H)$, giving `(B, A, H, 2)`.
   Fall back to the dense range when the dict has no anchor set, as CRWS does.
3. Call `teb_vae.lag_attn_rws.nets.losses.compute_loss` with these arguments:
   - `block_width=2`
   - `ar_coef=tanh(a)` when AR is on, else `None`
   - `horizon_weight=None`, `channel_weight=None`, `cell_mask=None`
4. Use `weight` for the forecast mask, as now. A target patch that is invalid in `weight` is not
   scored. A non-finite sample inside a valid patch must still give a finite target; `featurize`
   already zeroes it.
5. Merge `anchors_per_sample` into the metrics. Bind the CFS helper by reference if it does not read
   feature-specific attributes; otherwise compute it in one line.

## B.7 Task (`task.py`)

The task is `SeqVaeLagAttnTrfPatchTask(SeqVaeLagAttnTrfRwsTask)`. Follow the pattern of
`teb_vae/lag_attn_crws/task.py`:

- **Bind by reference** from `SeqVaeLagAttnCfsTask`: `anchor_phase`, `_phase_field` (inside
  `staticmethod(...)`), `resolve_anchor_geometry` and `_mu_gap_rms`. `_mu_gap_rms` calls
  `model.scored_weight`, so the target mixin must supply it as the identity.
- **Write your own** members:
  - `__init__(..., seed)`, which saves `seed`.
  - `compute_loss_and_metrics`, which sets `_stage`. A bound copy fails on `super()`.
  - `_build_forward_inputs`, which returns `(y_patch, u_patch, phase, stride)`.
  - `_added_metrics`, which calls `controls.source_null_kld` with `inputs[1]`, because the CFS
    version reads `inputs[2]`.
- **Plotting:** v1 ships no forecast page. Disable the plotting callback in the configs, and record
  this as a `lean-limit:` note in `DESIGN.md`.

## B.8 Trainer (`trainer.py`)

The trainer is `LagAttnTrfPatchTrainer(LagAttnTrfRwsTrainer)` with these attributes:

- `MODEL_CLS = SeqVaeLagAttnTrfPatch`
- `TASK_CLS = SeqVaeLagAttnTrfPatchTask`
- `CHECKPOINT_STEM = "lag-attn-trf-patch"`

Its pre-flight imports the e2e guards, adapts their key lists, and does not copy them:
- Refuse every removed constructor key (B.5) and every causal-feature key (`causal_*`,
  `target_weight_*`, `target_scored_horizon`, `target_forecast_shift`, `target_novelty_frac`).
- Require `fhr` and `up` in both `load_fields` and `normalize_fields`.
- Require `weight`, `guid` and `epoch` in `load_fields`.
- Check the raw length against the shard after the trim.

Keep `PLOT_CONFIG_KEY` at its inherited value. Provide `RUN_CONFIG` and `main` as the sibling
trainers do.

## B.9 Configuration

Start from `teb_vae/lag_attn_transformer_cfs/configs/default.yaml` and apply only this delta. Every
other leaf keeps its CFS value, so that the two cells differ in the representation alone.

| Leaf | CFS value | This package |
|---|---|---|
| `warmup_period` | 134 | 30 |
| `causal_warmup_budget_steps` and every other `causal_*` key | present | removed |
| `c_y`, `c_u`, `use_up_st`, `target_weight_*`, `target_scored_horizon` keys | present | removed |
| `lag_kv_source` | `adapter` | `adapter` (unchanged; stated for the reader) |
| `persistence_residual` | `true` | `false` |
| `prior_availability_input` | `true` | `false` |
| `horizon_weight_halflife_steps` | 30.0 | `null` |
| `forecast_ar_residual` | `true` | `true` |
| `target_summary_loc`, `target_summary_scale`, `variability_eps` | absent | from P1-03 |
| `source_validity` | absent | `finite` |
| `load_fields` | feature blocks | `[fhr, up, weight, guid, epoch]` |
| `normalize_fields` | feature blocks + raw | `[fhr, up]` |
| `gradient_clip_val` | 10500 | 280 (scaled by 60/2280; re-derive) |
| `additive_margin` | 6.6e3 | 175 (scaled by 60/2280; re-derive) |
| Identity keys (`run_name`, `tags.variant`, output tree, experiment name, checkpoint stem) | CFS | this package's own |

The loss-scale constants are provisional. Re-derive them from the first production run with the
procedure in `lag_attn_transformer_e2e/DESIGN.md` §12. The shards may be any existing build,
because only `fhr`, `up` and `weight` are read. For a comparison against CFS, use the same shard
folds and the same stats file.

**Arms.** Each arm is the default plus one named leaf plus the identity keys:

| File | Leaf | Question |
|---|---|---|
| `sweep_warmup_134.yaml` | `warmup_period: 134` | Is any gain from the representation, or from 76% more anchors? |
| `sweep_lag_kv_encoder.yaml` | `lag_kv_source: encoder` | Does lag locality matter? |
| `sweep_persistence.yaml` | `persistence_residual: true` | Does the level leave the latent? |
| `sweep_source_validity_fhr.yaml` | `source_validity: fhr_weight` | Does the UP mask choice matter? |
| `planted.yaml` | `max_lag: 60`, planted shard | Can the model recover a known delay? (an instrument, not an arm) |

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
7. **One definition of the target.** The target builder and any persistence path call
   `patch_summaries`.

---

# Part C — Tasks

## C.1 Phases and dependencies

```
Phase 0  P0-01                                  (alone)
Phase 1  P1-01  P1-02  P1-03  P1-04             (parallel)
Phase 2  P2-01  P2-02                           (parallel) → P2-03
Phase 3  P3-01  P3-02                           (parallel, need P2-03)
Phase 4  P4-01 … P4-08                          (parallel, need Phase 3)
Phase 5  P5-01  P5-02  P5-03                    (parallel, need Phase 4 green)
Phase 6  P6-01                                  (needs Phase 5)
```

## C.2 Tracker

| ID | Title | Depends on | Status |
|---|---|---|---|
| P0-01 | Map the co-mixin contract | — | done |
| P1-01 | `patchify` and `patch_summaries` | P0-01 | done |
| P1-02 | `PatchEmbedding` | P0-01 | done |
| P1-03 | Target statistics script | P0-01 | done |
| P1-04 | Package skeleton | P0-01 | done |
| P2-01 | `PatchStreamInputs` mixin | P1-01, P1-02 | done |
| P2-02 | `PatchSummaryTarget` mixin | P1-01 | done |
| P2-03 | Model constructor | P2-01, P2-02 | done |
| P3-01 | Task | P2-03 | done |
| P3-02 | Trainer and configs | P2-03, P1-03 | done |
| P4-01 | Tests: patching | Phase 3 | done |
| P4-02 | Tests: causality | Phase 3 | done |
| P4-03 | Tests: invariants and purity | Phase 3 | done |
| P4-04 | Tests: objective and target | Phase 3 | done |
| P4-05 | Tests: DDP reachability and strategy | Phase 3 | done |
| P4-06 | Tests: config load and arms | Phase 3 | done |
| P4-07 | Tests: task and controls | Phase 3 | done |
| P4-08 | Tests: train smoke | Phase 3 | done |
| P5-01 | Planted-delay check | Phase 4 | done |
| P5-02 | Classifier compatibility | Phase 4 | done (no classifier source change; new test `teb_vae/classifier/tests/test_patch_source.py`) |
| P5-03 | Parameter and cost record | Phase 4 | done |
| P6-01 | `DESIGN.md` and `RESULTS.md` | Phase 5 | done |

## C.3 Conflicts found during implementation

Write each conflict as: the task ID, the plan statement, what the code does, and the decision you
need from the user. Leave this section empty when there is none.

Found by P0-01 (detail: `notes/CONTRACT.md` §10). Each was resolved by the orchestrator with the
smallest change that keeps the plan's intent; the user may overrule any of them.

| # | Plan statement | What the code does | Decision taken |
|---|---|---|---|
| C1 | B.3: `featurize` gives `(value, mask, delta)` | It returns one stacked `(B, 3, L)` tensor | Unpack with `.unbind(1)` |
| C2 | B.5 lists four jobs of `PatchSummaryTarget` | The base raises in `_check_persistence_target` when `persistence_residual=True` | `PatchSummaryTarget` overrides it as a no-op, as CFT does |
| C3 | P0-01 places some hooks in `causal_feature_target.py` | They are in `causal_inputs.py`; CWI reads fewer hooks than listed | Use `CONTRACT.md` §1 as the hook list |
| C4 | B.5: forward keywords with the `locals()` pattern | `FORWARDED_EXCLUSIONS` lacks the four new keys, so the base raises `TypeError` | Extend the exclusion tuple locally |
| C5 | B.6: `horizon_weight=None` | The kept `horizon_weight_halflife_steps` keyword would register a buffer that the loss then ignores | Pass `getattr(self, "horizon_weight", None)`; it is `None` at the shipped `null` |
| C6 | B.7: bind four CFS members onto a `SeqVaeLagAttnTrfRwsTask` subclass | The `_stage` class default is needed by `VaeSource` | Subclass `SeqVaeLagAttnTrfCrwsTask`, which already has the binds, `seed`, `_stage` and `compute_loss_and_metrics`. Override only `_build_forward_inputs` and `_added_metrics` |
| C7 | B.8 does not mention the seed | `LagAttnTrfRwsTrainer` never gives the task `general_config.seed`, so the anchor phase keys on 0 | Override `create_model` to apply the seed, as CRWS does |
| C8 | B.8: import the e2e guards and adapt their key lists | `_check_no_inert_model_keys` reads a module global and takes no key list | Write one short refusal. Import the two raw e2e guards and the CRWS `_check_phase_key_fields`, `_check_raw_target_fields` and `_check_boundary_term_is_off` as they are |
| C9 | B.5 removes `use_up_st` | The base default is then `True`, which has no effect | Pass `use_up_st=False` to the base |
| C10 | B.8 does not mention tracked metrics or the startup message | `anchors_per_sample` and `kld_source_null` never reach `metrics_history.csv`; the inherited startup line is false for patches | Extend `TRACKED_METRICS` as CRWS does; override `causal_standing_message` with one true sentence |
| C11 | P5-02: name each changed classifier file | The classifier has no package registry | No classifier source change is expected; add one test |
| C12 | B.9: `target_summary_*` from P1-03 | The production shards are not on this machine. The tiny fixture is synthetic: level scale 0.26 and variability loc 0.32, about 14 bpm RMS beat-to-beat, far from real FHR | `default.yaml` ships the identity values `[0, 0]`, `[1, 1]` and `0.01`, marked PROVISIONAL. `tiny.yaml` carries the tiny-fixture values. The user must run `summary_stats.py` on the production shards before the first production run |
| C13 | The B.9 table lists the removed CFS keys | It omits `target_phase_fast_cutoff_hz` and `target_phase_fast_horizon`, which set the CFS feature target's scored horizon and which the patch constructor does not take | They are dropped from `default.yaml`, and the trainer pre-flight refuses `target_phase_fast_*`, `target_delays` and `source_delays` by name |
| C14 | B.9: `planted.yaml` = `max_lag: 60` plus the planted shard | With S = 15, `source_dropout` 0.2 and AR on, the source path stays closed on the instrument. The CFS instrument uses S = 1, no source dropout and AR off | User's decision: `planted.yaml` also carries those three CFS instrument leaves, so the two cells' readings are comparable |

## C.4 Task details

### P0-01 — Map the co-mixin contract (read-only)

**Goal.** List every attribute and method that `CausalWarmupInputs` and the CFS task members of
B.7 read from the model, and say which class supplies each one today.

1. Read `teb_vae/lag_attn_cfs/nets/causal_inputs.py` completely.
2. Read `teb_vae/lag_attn_crws/nets/causal_raw_inputs.py`, the raw-target precedent.
3. Read the members of `teb_vae/lag_attn_cfs/nets/causal_feature_target.py` that are called from
   step 1. Examples: `_check_anchor_floor`, `_resolve_warmup_readout_constants`,
   `_anchor_target_values`, `scored_weight`, `anchor_ceiling`, `_set_likelihood_structure`,
   `_register_likelihood_structure`, `forecast_likelihood_kwargs`, `SOURCE_BLOCK_SPLIT`,
   `TARGET_BLOCK_SPLIT`.
4. Read `teb_vae/lag_attn_rws/nets/controls.py`: `perm_forward_outputs`, `source_null_kld`,
   `source_null_forward_outputs`, and the occlusion helper. Record what each reads.
5. Read `SeqVaeLagAttnCfsTask` members `anchor_phase`, `_phase_field`, `resolve_anchor_geometry`,
   `_mu_gap_rms` and `_added_metrics`.
6. Read `teb_vae/classifier/sources.py::VaeSource` and record what it reads from the task and the
   model.

**Output.** Write `teb_vae/lag_attn_transformer_patch/notes/CONTRACT.md` with a table: member,
reader, current supplier, and what the patch model must supply (reuse by reference, trivial
override, or new code).

**Done when** every member in the table has a decision, and every "bind by reference" decision
names a member that reads no feature-specific attribute.

### P1-01 — `patchify` and `patch_summaries`

**Files.** `nets/patching.py`. Spec: B.3.

**Done when** both functions exist, import nothing from a framework layer (no Lightning), and have
docstrings with the shapes and the $m_t - 1$ rule.

### P1-02 — `PatchEmbedding`

**Files.** `nets/patching.py`. Spec: B.4. Coordinate with P1-01 on the same file. P1-01 owns the
functions and P1-02 owns the class.

**Done when** the class exists, the forward contains no Python branch on a tensor value, and the
`missing` parameter survives `teb_vae.lag_attn.nets.blocks.initialization`.

### P1-03 — Target statistics script

**Files.** `summary_stats.py` (an `if __name__ == "__main__"` entry with a `RUN_ARGS` dict, as
other packages do).

1. Read training-fold shards through the same loader and stats file as training.
2. Compute `patch_summaries` on valid tokens only.
3. Compute $\varepsilon$ as the monitor resolution (0.25 bpm) divided by the FHR standard deviation
   from the stats file.
4. Weight recordings equally, then print `target_summary_loc`, `target_summary_scale` and
   `variability_eps` as YAML.

**Done when** the script runs on `teb_vae/lag_attn/tests/fixtures/tiny_shard.hdf5` with
`tiny_stats.hdf5` and prints finite values. The user runs it on the production shards.

### P1-04 — Package skeleton

**Files.** `__init__.py`, `nets/__init__.py`, `tests/__init__.py`, `tests/conftest.py`.

- `conftest.py` imports the shared data fixtures from the sibling suites, as
  `teb_vae/lag_attn_transformer_e2e/tests/conftest.py` does. Do not restate them.
- It defines a `build` factory for the model at tiny widths and at shipped geometry.

**Done when** `pytest teb_vae/lag_attn_transformer_patch/tests` collects with no error.

### P2-01 — `PatchStreamInputs`

**Files.** `nets/patch_inputs.py`. Spec: B.5 and the P0-01 notes.

**Done when** the mixin builds `PatchEmbedding` for both streams, constructs with no gates and no
warm-up vectors, and its `forward` returns the CFS forward's key set with
`mu_* : (B, A_max, H, 2)`.

### P2-02 — `PatchSummaryTarget`

**Files.** `nets/patch_target.py`. Spec: B.5 and B.6.

- Reuse `_set_likelihood_structure` and `_register_likelihood_structure` only if P0-01 shows that
  they read no feature-specific attribute. Otherwise write the AR(1) parameter in about 10 lines.
- Under `persistence_residual: true`, `_anchor_target_values` must return the standardized
  summaries of the anchor's own patch, computed through `patch_summaries`.

**Done when** `compute_loss` returns the shared objective's metric dict plus `anchors_per_sample`
on the tiny model.

### P2-03 — Model constructor

**Files.** `nets/model.py`. Spec: B.5.

**Done when** `SeqVaeLagAttnTrfPatch` constructs at tiny and shipped geometry. Its only member is
`__init__`. Its decoder `out_features` follow width 2.

### P3-01 — Task

**Files.** `task.py`. Spec: B.7.

**Done when** one `compute_loss_and_metrics` call runs on a stub batch at each of the stages
`train`, `val` and `test`, and the `val` stage reports `kld_source_null`.

### P3-02 — Trainer and configs

**Files.** `trainer.py`, `configs/default.yaml`, `configs/tiny.yaml`, and the B.9 arm files.

- `tiny.yaml` reads `teb_vae/lag_attn/tests/fixtures/tiny_shard.hdf5` and `tiny_stats.hdf5`. It
  shrinks widths only and keeps the shipped geometry.
- `planted.yaml` reads `tiny_shard_causal_planted.hdf5` and `tiny_stats_causal_planted.hdf5`.

**Done when** `python -m teb_vae.lag_attn_transformer_patch.trainer --config
teb_vae/lag_attn_transformer_patch/configs/tiny.yaml` completes one epoch and writes a checkpoint
under its own stem.

### P4-01 — Tests: patching

**File.** `tests/test_patching.py`.

The tests assert:
- shapes;
- the $m_t - 1$ channel;
- an all-NaN patch gives finite output with validity −1;
- `patch_summaries` against a NumPy reference;
- `variability` ignores the boundary-crossing difference;
- `missing` replaces exactly the invalid tokens;
- an all-zero stream gives no `missing` token.

### P4-02 — Tests: causality

**File.** `tests/test_causality.py`.

- Use B.10 item 1, through `patchify` and the full model, for `target_state`, `source_state` and
  `mu_prior`.
- Pair each bitwise assertion with a movement assertion, as the e2e suite does.
- Add a check that no history-path module pools over time (reuse the e2e
  `refuse_time_pooling_norms` helper or its test pattern).

### P4-03 — Tests: invariants and purity

**File.** `tests/test_invariants.py`.

Cover B.10 items 2, 3, 4 and 6. Reuse the sibling suites' helpers by import where they exist.

### P4-04 — Tests: objective and target

**File.** `tests/test_objective.py`.

The tests assert:
- the anchored gather equals the dense gather at stride 1;
- the target at anchor $a$, step $\tau$ equals `patch_summaries` of token $a + 1 + \tau$;
- with `forecast_ar_residual` at $\phi = 0$ the score equals the factorized score bitwise;
- an invalid target patch is not scored.

### P4-05 — Tests: DDP reachability and strategy

**File.** `tests/test_ddp_reachability.py`.

Cover B.10 item 5 on the three batch kinds. Reuse the AST walk from
`teb_vae/lag_attn_transformer_e2e/tests/test_ddp_reachability.py` by import.

### P4-06 — Tests: config load and arms

**File.** `tests/test_config_load.py`.

The tests assert:
- every `VAE_model` key reaches the constructor or the task;
- every removed key is refused by name;
- each arm differs from `default.yaml` in exactly its declared leaves;
- `default.yaml` differs from the CFS `default.yaml` only in the B.9 delta and the identity keys.

### P4-07 — Tests: task and controls

**File.** `tests/test_task.py`.

The tests assert:
- `_build_forward_inputs` arity and shapes;
- the anchor phase is deterministic;
- `perm_forward_outputs` and `source_null_kld` run on the model's own forward dict;
- the source-null stream is "valid and flat" (no `missing` token).

### P4-08 — Tests: train smoke (`slow`)

**File.** `tests/test_train_smoke.py`.

Run three epochs of `tiny.yaml`. The tests assert:
- the loss is finite;
- `train/grad_norm` is finite and non-zero;
- every `PatchEmbedding` tensor differs from a freshly built model's after the fit.

### P5-01 — Planted-delay check

**Files.** `lag_recovery_check.py` and `tests/test_lag_recovery.py` (`slow`).

1. Train `planted.yaml` for a few epochs. The planted shard couples raw UP to raw FHR at 45 steps
   (180 s); the root attribute `planted_delay_steps` stamps this value.
2. Read the head-averaged `source_kl_lag_map` over the validation anchors.
3. Report the informative band $[\delta - H, \delta - 1] = [15, 44]$ against the rest of the lag
   window.
4. Reuse the band logic of `teb_vae/lag_attn_cfs/lag_recovery_check.py` by import if it is
   target-independent. Otherwise write the reading here.

**Done when** the script prints the band mass and the lag argmax, and the test asserts only that
the script runs and the numbers are finite. The pass/fail reading belongs to the user.

### P5-02 — Classifier compatibility

**Files.** Only what `teb_vae/classifier` needs to load this package. Name each changed file in the
tracker.

**Done when** `VaeSource` loads a `tiny.yaml` checkpoint of this package and extracts `mu_prior`,
`delta_mu` and `kld_per_t` with the step mask starting at `warmup_period = 30`. Add one test in
`teb_vae/classifier/tests/`.

### P5-03 — Parameter and cost record

1. Measure the parameter count at shipped geometry and decompose the change against the shipped
   CFS cell.
2. Time one forward and backward pass at batch 16 against the CFS cell on the same device.
3. Record both in `RESULTS.md`.

### P6-01 — `DESIGN.md` and `RESULTS.md`

**Files.** `DESIGN.md` and `RESULTS.md`.

`DESIGN.md` is the as-built record:
- what moved from this plan and why;
- every `lean-limit:` note, including the disabled forecast page, the missing eval package and the
  UP validity rule;
- how to run it.

`RESULTS.md` holds:
- the pre-registered reading for each arm, written before any production number exists;
- the parameter and cost record from P5-03.

Keep this plan as it is; mark the tracker complete.

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
