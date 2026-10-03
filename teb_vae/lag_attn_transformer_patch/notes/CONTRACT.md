# P0-01 — Co-mixin contract (read-only map)

Written 2026-10-03 against `main` @ `ba350c8`. Paths are relative to `teb_vae/`. Short names:
`CWI` = `lag_attn_cfs/nets/causal_inputs.py::CausalWarmupInputs`,
`CFT` = `lag_attn_cfs/nets/causal_feature_target.py::CausalFeatureForecastTarget`,
`FFT` = `lag_attn_fs/nets/feature_target.py::FeatureForecastTarget`,
`CRI` = `lag_attn_crws/nets/causal_raw_inputs.py::CausalRawInputs`,
`TrfRws` = `lag_attn_transformer_rws/nets/model.py::SeqVaeLagAttnTrfRws`.

Decision words:
- **inherit**: comes from `TrfRws` (or `CWI`) through the MRO. Nothing to write.
- **reuse by reference**: assign in the class body, as CRI does (CRI:173-185). Wrap static
  methods in `staticmethod(...)`.
- **trivial override**: the one-liner given.
- **new code**: what it must compute.
- **not needed**: no reader on the patch path once the listed override exists.

---

## 1. What `CausalWarmupInputs` reads from `self` but does not define

### 1a. Target co-mixin hooks

| Member | Reader (file:line) | Current supplier | Patch decision |
|---|---|---|---|
| `_check_anchor_floor` (static) | CWI `_validate_causal_geometry` causal_inputs.py:345, called as `(warmup_period, target_warmup_steps or (), target_gate shifts or ())`. A 4th arg is passed only if `target_forecast_shift` is set (:343-344). | CFT:177 (4 args); CRI:190 (3 args) | **trivial override**: `@staticmethod def _check_anchor_floor(*_): return None`. There is no per-channel warm-up. Both existing versions return at once on an empty `kept_warmup_steps` (CFT:232, CRI:261), which is all the patch would pass. |
| `_resolve_warmup_readout_constants` | CWI `_validate_causal_geometry` :346 | Defined **in CWI** :460 (feature version), not in CFT. Overridden by CRI:313 (source only). | **trivial override**: `def _resolve_warmup_readout_constants(self): pass`. Its outputs are `target_warm_frac`, `warm_tertile_id`, `novelty_tertile_id` and `source_block_warm_st/_ph`. Only CFT `_resolved_forecast_gaps` / `_source_lag_warmth` and the CFS eval and page code read them, and the patch uses none of these. With no source warm-up, the source patterns would be constant all-True. This override makes the next six rows unnecessary. |
| `_resolve_target_warm_frac` | CWI :476 | CFT:305 | **not needed** (its only reader is overridden) |
| `_resolve_warm_tertiles` | CWI :496 | CFT:392 | **not needed** |
| `target_novelty_frac` (attr) | CWI :517 | CFT `_set_target_novelty` :440/463 | **not needed** |
| `_resolve_novelty_tertiles` | CWI :528 | CFT:409 | **not needed** |
| `SOURCE_BLOCK_SPLIT` (=36) | CWI :550 | CFT:175; CRI:173 | **not needed** |
| `_resolve_block_warm_steps` (static) | CWI :590 | CFT:645; CRI:177 | **not needed** |
| `_anchor_target_values(target, anchors)` | CWI `forward` :852-856, only when `self.persistence_residual` | CFT:876 | **new code**. `target` is the pre-gate `(B,T,33)` stream (`cat([y_patch, y_patch[..., :0]])`). Gather it at `anchors (B,A)`. Then call `patch_summaries(values=x[..., :R], valid=x[..., -1] + 1)`, apply the affine `target_summary_loc/scale`, and return `(B,A,2)`. Use the same function as the loss target (B.10.7). |

### 1b. Base / own members (all exist today)

| Member | Reader in CWI | Supplier | Patch decision |
|---|---|---|---|
| `anchor_stride`, `lag_floor`, `target_warmup_steps`, `source_warmup_steps`, `target_forecast_shift` | :314, :683, :735, :333, :367, :636-638, :286 | CWI `_set_causal_inputs` :262-270 | **inherit**. Call `_set_causal_inputs` with every keep-index and warm-up set to `None` (see §2). |
| `anchor_ceiling` (property) | :314, :482, :744 | **CWI** :272 (not CFT) | **inherit**. With `target_forecast_shift=None` it equals `geometry.t_valid` = 270. |
| `_prior_clock_dim` | base `__init__` TrfRws:535, only if `prior_availability_input` | CWI:446 returns `d_model`. It overrides TrfRws:847, which raises. | **inherit**. The patch therefore *admits* `prior_availability_input: true`. The config sets `false`. |
| `_combined_source_steps`, `_prior_clock` | :558, :878 | CWI:348, :375 | **inherit**. `_combined_source_steps` returns `None`. `_prior_clock` is unused at flag `false`. |
| `geometry` (`.t_valid`) | :288-289, :317-320 | TrfRws:374 (`TrimmedRawGeometry(raw_len=T*R, decimation=R, horizon=H, warmup=F)`) | **inherit** |
| `warmup_period`, `horizon`, `sequence_length`, `d_model`, `d_z` | :314, :332, :477-478, :736, :744, :588, :458, :911 | TrfRws:386, :384, :381, :382, :383 | **inherit** |
| `c_y`, `c_u`, `use_up_st` | :487, :518, :546, :550 (all inside the overridden readout method) | TrfRws:387-389 | Pass `c_y = c_u = 2R+1` to the base. `use_up_st`: pass `False` (see §10, C9). |
| `target_gate`, `source_gate` | :335, :429, :489, :547, :637, :857-859, :368 | TrfRws:465/:468 via `_build_channel_gate` :684. It returns `None` when keep-index and delays are both `None` (:708-709). | **inherit** (both `None`) |
| `super()._build_adapter` | :641 | TrfRws:714 | The patch overrides `_build_adapter` and does **not** call super (§2) |
| `super().build_lag_mask`, `lag_attn.L` | :678, :682 | TrfRws:791, :549 | **inherit**. At `lag_floor=0` the base mask object comes back unchanged (:679-680). |
| `source_kv_modules`, `encode_source_kv` | :434, :440, :866 | TrfRws:761, :775 | **inherit** |
| `persistence_residual` | :853 | TrfRws:419 | **inherit** (forwarded keyword) |
| `target_encoder`, `target_adapter` | :861 | TrfRws:484, :476 | **inherit**. The adapter is `PatchEmbedding` through the override. |
| `prior_head`, `prior_availability_input` | :876-879 | TrfRws:529, :528 | **inherit** |
| `query_uses_logvar`, `query_proj`, `lag_attn`, `posterior_head` | :884-895 | TrfRws:542, :544, :549, :560 | **inherit** |
| `_reparameterize_shared` | :896 | TrfRws:1064 | **inherit** |
| `mu_scale`, `delta_mu_scale` | :903, :905 | TrfRws:392-393 | **inherit** |
| `decoder` | :912-917 | TrfRws:596. Its width comes from `_default_decoder_out_channels` (:595). | **inherit** |
| `kld_tensor`, `te_analysis` | :922-930 | TrfRws:1238, :575 | **inherit** |
| `register_buffer` | :493, :525, :584 | `nn.Module` | **inherit** |

### 1c. Members the plan names that CWI does **not** read

| Member | Real reader(s) | Current supplier | Patch decision |
|---|---|---|---|
| `scored_weight(weight)` | CFS task `_mu_gap_rms` task.py:464 (bound by the patch). Also CFT.compute_loss :864 and CFS eval. | CFT:795 (`pooled_scored_weight`, identity when there is no shift) | **trivial override**: `def scored_weight(self, weight): return weight` |
| `forecast_likelihood_kwargs()` | FFT.compute_loss feature_target.py:452; CFT `_gap_by_kept_channel` :976; CFS task `forecast_rows` :162; CFS eval | FFT:105 | **reuse by reference** (`= FeatureForecastTarget.forecast_likelihood_kwargs`). It reads only `getattr(self, "target_ar_logit"/"target_cell_mask", None)` and returns `{"cell_mask": None, "ar_coef": tanh(a) or None}`. The patch `compute_loss` can splat it. |
| `_set_likelihood_structure(*, target_scored_horizon, forecast_ar_residual)` | CFS model constructors (TrfCfs model.py:239) | CFT:514 | **reuse by reference**. It reads nothing from `self`; it only sets `target_scored_horizon` and `forecast_ar_residual`. Call it before `super().__init__` with `target_scored_horizon=None`. |
| `_register_likelihood_structure()` | TrfCfs model.py:261 (after the base) | CFT:542 | **reuse by reference**. With `target_scored_horizon=None` it reads only `decoder_out_channels` and `forecast_ar_residual` and builds `target_ar_logit = Parameter(zeros(2))`. Because it runs after `initialization`, φ starts at exactly 0. Its reads of `c_y`, `target_gate` and `horizon` (:560-575) are in the scored-horizon branch, which the patch never reaches. |
| `TARGET_BLOCK_SPLIT` | CFS task `forecast_rows` :147 (not bound), cfs `sample_page`, eval | CFT:160; CRI:174 | **not needed**. Omit it. |
| `_check_persistence_target()` | **base** TrfRws `__init__` :420-421. The base version raises (:876-903). | CFT:683 (no-op) | **trivial override**: `def _check_persistence_target(self): pass`. The plan omits this (C2). |
| `_default_decoder_out_channels()` | base `__init__` TrfRws:595 | TrfRws:657 (=R=16); FFT:125 | **trivial override**: `return 2`. Use a constant: it is called inside the base `__init__`, before any post-super attribute exists. |
| `compute_loss(...)` | RWS task `compute_loss_and_metrics` task.py:654 and :708 (perm re-score) | TrfRws:1263; CRI:386 | **new code** (§4) |
| `_anchors_per_sample(fo, target)` | patch `compute_loss` | CFT:982-1017. Already bound by CRI:184. | **reuse by reference**. It reads `fo["anchor_index"]`, `fo["anchor_valid"]`, the target's dtype and device, and `geometry.t_valid` (dense fallback only). It returns `valid.sum()/B`. |
| `_source_lag_warmth` | CRI.compute_loss :491 | CFT:1019 | **not needed**. It needs `source_block_warm_*`, and with no source warm-up it is ≡ 1.0. |

---

## 2. Constructor path

**CWI has no `__init__`.** Its init path is two calls:
1. `_set_causal_inputs(...)` (CWI:171-270), **before** the base `__init__`. It takes
   `horizon, target_keep_index, target_warmup_steps, source_keep_index, source_warmup_steps,
   anchor_stride, lag_floor, target_forecast_shift=None`. It refuses `S ∉ [1,H]`, `lag_floor < 0`,
   a warm-up without its keep-index, and a shift without a keep-index or with mixed signs. It sets
   plain attributes, which is legal before `Module.__init__`.
2. `_validate_causal_geometry()` (CWI:291-346), **after** the base `__init__`. It refuses
   `S > anchor_ceiling − F`, then calls `_check_anchor_floor(*floor_args)` and
   `_resolve_warmup_readout_constants()`.

**These values make the gates, warm-up and alignment no-ops:**
- `target_keep_index = source_keep_index = None`
- `target_warmup_steps = source_warmup_steps = None`
- `target_forecast_shift = None` (the default)
- `lag_floor = 0`
- base `target_delays = source_delays = None`

The effects are:
- `_build_channel_gate` returns `None` (TrfRws:708-709).
- `_build_adapter` sees `warmup None`.
- `anchor_ceiling = t_valid`.
- `_combined_source_steps() → None`.
- `_check_anchor_floor(30, (), ())` returns at once.
- `build_lag_mask` returns the base object.

**`_build_adapter` in `SeqVaeLagAttnTrfRws`** (model.py:714-741):
```python
def _build_adapter(self, gate: Optional[ChannelGate], declared_width: int, dropout: float) -> AvailabilityInputAdapter
```
- It is called positionally:
  - :476 `self._build_adapter(self.target_gate, self.c_y, dropout)`
  - :477-479 `self._build_adapter(self.source_gate, self.c_u, self.source_dropout)`
- The base builds `AvailabilityInputAdapter(in_dim=width, d_model=self.d_model,
  sequence_length=self.sequence_length, dropout=dropout, delays=None | gate delays)`.
- CWI's override (:597-655) tells the two streams apart with `gate is self.target_gate`. With both
  gates `None` that test is True for **both** calls. This is harmless only because the warm-ups are
  `None`.
- The patch override returns `PatchEmbedding(in_dim=declared_width, d_model=self.d_model,
  sequence_length=self.sequence_length, dropout=dropout)` and must not call super. The two calls
  differ only in dropout (`dropout` vs `source_dropout`).
- Only CFS eval `analyses/occlusion.py:273` (out of scope) reads `model.source_adapter`. Nothing
  outside tests reads adapter attributes or checks `isinstance(..., AvailabilityInputAdapter)`.

**How `lag_kv_source: adapter` reaches the source adapter output:**
1. TrfRws:500-523 builds neither `source_encoder` nor `source_kv_stem`.
2. `source_kv_body()` returns `None` (:755-759).
3. `source_kv_modules()` returns `(self.source_adapter,)` (:772-773).
4. `encode_source_kv(x)` returns `source_adapter(x)` (:786-789).
5. CWI forward :866: `h_u = self.encode_source_kv(source)`. So `h_u = PatchEmbedding(u_patch)`,
   `(B,T,d_model)`. It is K and V for `lag_attn` (:889-891) and is returned as `source_state`.

**How `SeqVaeLagAttnTrfCrws.__init__` is composed** (transformer_crws/nets/model.py:85-215):
- Class: `class SeqVaeLagAttnTrfCrws(CausalRawInputs, SeqVaeLagAttnTrfRws)`. The mixin comes first.
- Signature: the TrfRws keywords minus `target_delays`/`source_delays`/`persistence_residual`, plus
  `target_warmup_steps, source_warmup_steps, target_align_delays, source_align_delays,
  anchor_stride=1, lag_floor=0`. Its defaults are `warmup_period=134, c_y=102, c_u=51,
  max_lag=90, lag_kv_source="encoder"`.
- Body:
  - `forwarded = {n: v for n, v in locals().items() if n not in FORWARDED_EXCLUSIONS}` (:186-190)
  - `self._set_causal_inputs(horizon=..., target_keep_index=..., target_warmup_steps=..., source_keep_index=..., source_warmup_steps=..., anchor_stride=..., lag_floor=...)` (:193-201)
  - `super().__init__(**forwarded, target_delays=target_align_delays, source_delays=source_align_delays)` (:208-212)
  - `self._validate_causal_geometry()` (:215)
- `FORWARDED_EXCLUSIONS = ("self","__class__") + CAUSAL_ONLY_KEYWORDS` (CWI:93-111). The
  keywords are: `target_warmup_steps, source_warmup_steps, anchor_stride, lag_floor,
  target_weight_st, target_weight_ph, target_align_delays, source_align_delays,
  target_novelty_frac, target_forecast_shift, target_scored_horizon, forecast_ar_residual`.
- The TrfCfs version (transformer_cfs/nets/model.py:211-261) also calls `_set_channel_weights`,
  `_set_target_novelty` and `_set_likelihood_structure` before super, and
  `_register_channel_weights` and `_register_likelihood_structure` after
  `_validate_causal_geometry`.

**Patch recipe:**
```python
PATCH_ONLY = ("target_summary_loc", "target_summary_scale", "variability_eps", "source_validity")
forwarded = {n: v for n, v in locals().items() if n not in FORWARDED_EXCLUSIONS + PATCH_ONLY}
# stash PATCH_ONLY as plain python values (tuples/float/str): no buffers before Module.__init__
self._set_causal_inputs(horizon=horizon, target_keep_index=None, target_warmup_steps=None,
                        source_keep_index=None, source_warmup_steps=None,
                        anchor_stride=anchor_stride, lag_floor=lag_floor)
self._set_likelihood_structure(target_scored_horizon=None, forecast_ar_residual=forecast_ar_residual)
super().__init__(**forwarded, c_y=2 * raw_per_step + 1, c_u=2 * raw_per_step + 1, use_up_st=False)
self._validate_causal_geometry()
self._register_likelihood_structure()
```
- `forecast_ar_residual` is already in `FORWARDED_EXCLUSIONS`. The four `PATCH_ONLY` keys are
  not, and the base would raise `TypeError` on them.
- `persistence_residual` **is** forwarded. Under `true`, the base calls
  `_check_persistence_target()` at TrfRws:420, before the gates are built.

---

## 3. `CausalWarmupInputs.forward` output dict (CWI:932-963)

The signature is `forward(y_st, y_ph, u_stream, anchor_phase=None, anchor_stride=None)`.
- The patch wrapper calls it with `(y_patch, y_patch[..., :0], u_patch, …)`.
- `target = cat([y_st, y_ph], -1)` (:842) is then `(B,T,33)`.
- The batch and device are read from `y_st` (:832-833).

The table uses `B`, `T=300`, `L=max_lag+1`, `M=num_heads`, `A=A_max`, `H` and `C=2`.

| Key | Shape | Depends on target co-mixin? |
|---|---|---|
| `mu_prior`, `logvar_prior`, `raw_logvar_prior`, `mu_post`, `logvar_post`, `z_prior`, `z_post` | `(B,T,d_z)` | no |
| `target_state`, `source_state` | `(B,T,d_model)` | no. `source_state` is the `PatchEmbedding` output under `adapter`. |
| `attended_source_heads` | `(B,T,M,d_head)` | no |
| `attn_weights` | `(B,T,M,L)` | no |
| `mu_base`, `logvar_base`, `mu_full`, `logvar_full` | `(B,A,H,C)` | **width C** from `_default_decoder_out_channels`. Under persistence the means include the persistence term. |
| `kld_per_t` | `(B,T)` | no |
| `kld_per_t_per_head` | `(B,T,M)` | no |
| `source_kl_lag_map` | `(B,T,L)` | no |
| `mu_prior_sat_frac`, `delta_mu_sat_frac` | scalar | no |
| `anchor_index` | `(B,A)` long | `anchor_ceiling` (`target_forecast_shift`; `None` for the patch) |
| `anchor_valid` | `(B,A)` bool | same |
| `persistence` | `(B,A,C)` | **yes**. It comes from `_anchor_target_values` and is present only under `persistence_residual`. |

At shipped geometry (F=30, ceiling 270):
- train (S=15): `A = 16`. Every phase gets 16 real anchors because 240 % 15 = 0.
- val/test (S=1, φ=0): `A = 240`.

---

## 4. Raw-target precedent and `compute_loss`

**How CRWS supplies the hooks** (causal_raw_inputs.py):
- `CRI(CWI)` is the **only** model mixin. It has no target co-mixin, and the decoder width stays
  the base `raw_per_step`.
- It binds `SOURCE_BLOCK_SPLIT`, `TARGET_BLOCK_SPLIT`, `staticmethod(_resolve_block_warm_steps)`,
  `_anchors_per_sample` and `_source_lag_warmth` from CFT by reference (:173-185).
- It overrides `_check_anchor_floor` (3-arg, :190) and `_resolve_warmup_readout_constants`
  (source patterns only, :313).
- It does **not** supply `_anchor_target_values` or `_check_persistence_target`. Persistence is
  structurally off: the keyword is absent from the CRWS signature.
- Do **not** subclass CRI for the patch. With `class X(PatchStreamInputs(CRI), PatchSummaryTarget,
  TrfRws)`, `CRI.compute_loss` would win the MRO over `PatchSummaryTarget.compute_loss`.

**How `CRI.compute_loss(fo, fhr_raw, *, weight, beta…lambda_boundary)` works** (:386-492):
1. It gets `anchors = fo.get("anchor_index")`.
2. It builds the target:
   - with no anchors: `build_future_target(fhr_raw, geometry, future_index=self.future_index)`,
     the dense `[0, T_valid)` range;
   - otherwise: `gather_anchored_future_target(...)` (:61-134), which bounds-checks the anchors.
3. It calls `compute_raw_objective(..., block_width=self.geometry.r,
   horizon_weight=getattr(self, "horizon_weight", None))`.
4. It merges `anchors_per_sample` and `source_lag_warmth_frac_st/_ph` into the metrics.

**The shared objective**, `lag_attn_rws/nets/losses.py:739`:
```python
compute_loss(forward_outputs, target, *, weight, geometry, block_width, coverage_floor, logvar_clamp,
             beta=1.0, beta_prior=0.0, lambda_full=1.0, lambda_base=1.0, likelihood="gaussian_nll",
             free_bits=0.0, lambda_ms=0.0, lambda_deriv=0.0, lambda_boundary=0.0,
             channel_weight=None, horizon_weight=None, cell_mask=None, ar_coef=None) -> Dict[str, Any]
```
- `target` is `(B,A,H,X)`.
- Masks are built **inside** from `weight (B,T)`, `geometry` and
  `fo["anchor_index"/"anchor_valid"]` (:891-912). There are no mask arguments:
  `m_{t,τ} = 1[t≥w]·v_t·v_{t+1+τ}`, with an anchor zeroed below `coverage_floor`.
- `block_width` feeds only the four log-variance diagnostics (:989-1017).
- `ar_coef` is `(C,)` (φ). It turns the score into innovations `r_τ − φ r_{τ−1}`, with the lag
  zeroed at masked steps (`raw_sample_score` :222-237). `None` gives the factorised score, bitwise.
- `cell_mask` is `(H,C)`; `horizon_weight` is `(H,)`; `channel_weight` is `(C,)`. Each
  `None` = unweighted, bitwise.
- `lambda_boundary ≠ 0` together with an anchor set → `ValueError` (:893-899).
- Return value: `{"metrics": {...}, "likelihood": str}`. The metric keys (:1059-1094) are:
  `total_loss, nll_full_block, nll_full_sample, nll_base_block, nll_base_sample, pred_gap,
  source_conditioned_kl_raw, source_conditioned_kl_train, kld_active_frac, prior_rate,
  aux_multiscale, aux_derivative, aux_boundary, kld_beta, beta_prior, lambda_ms, lambda_deriv,
  lambda_boundary, anchor_coverage_frac, mean_logvar_full, mean_logvar_base,
  logvar_full_floor_frac, logvar_full_ceil_frac, mean_logvar_prior, mean_logvar_post,
  logvar_prior_floor_frac, delta_mu_rms`.

**Patch `compute_loss`** has the same signature as CRI's. The RWS task calls it as
`compute_loss(fo, batch.fhr, weight=batch.weight, beta=…, lambda_boundary=…)` (rws/task.py:654,
:708):
1. Build the target `(B,T,2)` from `patchify(fhr_raw, weight, validity="fhr_weight")`, then
   `patch_summaries`, then the affine standardization.
2. Gather at `fo["anchor_index"]` (index `a+1+τ`). If there is no anchor set, use
   `arange(anchor_ceiling)` as CRWS does. With `anchor_ceiling = 270`, the largest index is
   269+30 = 299 < T.
3. Call `compute_raw_objective(fo, target, weight=weight, geometry=self.geometry, block_width=2,
   coverage_floor=self.coverage_floor, logvar_clamp=self.logvar_clamp, …,
   horizon_weight=getattr(self, "horizon_weight", None), **self.forecast_likelihood_kwargs())`.
   See §10, C5 for `horizon_weight`.
4. Merge `anchors_per_sample` with `result["metrics"]["anchors_per_sample"] =
   self._anchors_per_sample(fo, target)`.

**`anchors_per_sample` in CFS** is `CFT._anchors_per_sample` (:982-1017):
`anchor_valid.sum()/B`, falling back to `geometry.t_valid` or `anchor_index.shape[1]`. It reads
**no** feature attribute, so bind it by reference as CRI:184 does. It reads 16.0 at train and
240.0 at val.

---

## 5. `lag_attn_rws/nets/controls.py`

All four are `@torch.no_grad()`. Their shared tail is `_attend_and_pose(model, fo, h_u)` (:104-153).
- It reads `model.query_uses_logvar`, `model.lag_attn`, `model.query_proj`, `model.build_lag_mask`
  and `model.posterior_head`.
- It reads `fo["mu_prior"]`, `fo["logvar_prior"]` (only if `query_uses_logvar`),
  `fo["target_state"]` and `fo["raw_logvar_prior"]`.

| Function (line) | Signature | Reads beyond the tail |
|---|---|---|
| `perm_forward_outputs` (:157) | `(model, fo, *, perm_index=None, generator=None, groups=None, anchors=None)` | `fo["source_state"]` (permuted on batch); `model.decoder`; `model.geometry.t_valid` (only if `anchors is None`); `fo.get("persistence")`. Draws fresh ε. Returns a copy with `mu_post, logvar_post, z_post, attn_weights, mu_full, logvar_full, perm_index` replaced. The RWS task passes `anchors=fo.get("anchor_index")` (rws/task.py:696-707). |
| `source_null_forward_outputs` (:283) | `(model, fo, u_stream)` | Shape, dtype and device of `u_stream` (declared width, pre-gate). Builds a zeros `(1,T,F_in)` stream and passes it through `model.source_gate` (None) and `model.encode_source_kv`, then expands it to B. No decode, no ε. Replaces `mu_post, logvar_post, attn_weights`. Patch: zeros mean validity channel 0, so m = 1, i.e. valid and flat. No `missing` token. |
| `source_null_kld` (:512) | `(model, fo, u_stream, weight)` | The above, plus `model.geometry`, `model.coverage_floor`, `fo["anchor_index"/"anchor_valid"]`, `fo["mu_prior"/"logvar_prior"]`. Uses the **raw** `weight` (not `scored_weight`). Returns `source_conditioned_kl_raw` of the null arm. |
| `occluded_forward_outputs` (:401) | `(model, fo, source, *, occlusion=None, anchors=None, generator=None)` | `source` is **post-gate** `(B,T,c_kept)`. Rows where the `(T,)` or `(B,T)` `occlusion` is true are zeroed. Also reads `model.encode_source_kv`, `model.decoder`, `model.geometry.t_valid` (if `anchors is None`) and `fo.get("persistence")`. Replaces `OCCLUSION_KEYS` (:389). Patch caveat: a zeroed patch row reads as valid and flat, not as `missing`. The only caller is cfs eval `analyses/occlusion.py` (out of scope). |

None of the four reads a feature-specific attribute. All of them run on the patch forward dict.

---

## 6. Task members

**`SeqVaeLagAttnCfsTask`** (lag_attn_cfs/task.py):

| Member (line) | Reads | Feature-specific? | Patch decision |
|---|---|---|---|
| `anchor_phase(batch)` (:298) | `self.orig_model.anchor_stride`, `self._phase_field(batch, "guid"/"epoch")`, `self.current_epoch`, `self.hparams.get("seed", 0)`, and the module globals `_KEY_SEPARATOR`, `_as_key`, `_as_float` (resolved through the function's own globals). Returns `(B,)` long on the host, pinned if CUDA. | no | **reuse by reference** |
| `_phase_field(batch, name)` (:366, static) | `getattr(batch, name)` | no | **reuse by reference**, wrapped in `staticmethod(...)` |
| `resolve_anchor_geometry(stage, batch)` (:390) | `DENSE_STAGES` (module), `self.anchor_phase`, `orig_model.anchor_stride`. Returns `(0,1)` on val/test and `(φ, S)` on train. | no | **reuse by reference** |
| `_mu_gap_rms(fo, weight)` (:436) | `model.scored_weight`, `model.geometry`, `model.coverage_floor`, `fo["anchor_index"/"anchor_valid"/"mu_post"/"mu_prior"]` | only `scored_weight` (supplied as identity) | **reuse by reference** |
| `_build_forward_inputs(batch)` (:409) | `self._build_target_streams` (`batch.fhr_st/fhr_ph`, `c_y` check), `self._build_source_stream` (`up_ph/up_st`, `use_up_st`, `c_u`), `self._stage` | **yes** | **new code**. Returns `(y_patch, u_patch, φ, S)`. Reads `batch.fhr`, `batch.up`, `batch.weight`, `orig_model.raw_per_step`, `orig_model.source_validity`, and `self.resolve_anchor_geometry(self._stage, batch)`. |
| `_added_metrics(inputs, fo, weight, stage)` (:480) | `inputs[2]` is the source (in the patch tuple that slot is φ) | layout | **new code**: `{} if stage == "train" else {"kld_source_null": controls.source_null_kld(self.orig_model, fo, inputs[1], weight)}` |
| `compute_loss_and_metrics` (:527) | sets `self._stage`, then zero-arg `super()` | n/a | **new code**. A bound copy raises `TypeError` because zero-arg `super` closes over `SeqVaeLagAttnCfsTask`. |
| `_stage` class attribute (:561) | default `"val"` | no | **new code**: `_stage: str = DENSE_STAGES[0]` (§10, C6) |
| `transfer_batch_to_device` (:273) | keeps `epoch` on the host, zero-arg `super` | no | Optional. It cannot be bound, and CRWS does not have it either. Without it `epoch` is moved to the device and `_as_float` does B `.item()` syncs per train step. |

`SeqVaeLagAttnFsTask` adds nothing the patch needs. The patch's `_build_raw_target` is
`SeqVaeLagAttnRwsTask`'s (rws/task.py:432), which returns `(batch.fhr, batch.weight)`.

**How the CRWS task binds** (lag_attn_crws/task.py:82-217): `class
SeqVaeLagAttnCrwsTask(SeqVaeLagAttnRwsTask)`, with these assignments:
```python
anchor_phase = SeqVaeLagAttnCfsTask.anchor_phase
_phase_field = staticmethod(SeqVaeLagAttnCfsTask._phase_field)
resolve_anchor_geometry = SeqVaeLagAttnCfsTask.resolve_anchor_geometry
_build_forward_inputs = SeqVaeLagAttnCfsTask._build_forward_inputs
_mu_gap_rms = SeqVaeLagAttnCfsTask._mu_gap_rms
_added_metrics = SeqVaeLagAttnCfsTask._added_metrics
input_stream_panels = SeqVaeLagAttnCfsTask.input_stream_panels   # property object
input_budget_figure = SeqVaeLagAttnCfsTask.input_budget_figure
```
Its own members are:
- `__init__(base_model, *, seed=0, **kw)` → `super().__init__`, then `save_hyperparameters("seed")`
- the `forecast_rows` property
- `compute_loss_and_metrics` (sets `_stage`)
- `_stage = DENSE_STAGES[0]`
- `warmup_budget = None`

**The transformer tasks:**
- `SeqVaeLagAttnTrfRwsTask(SeqVaeLagAttnRwsTask)` (transformer_rws/task.py) adds only
  `build_lr_scheduler`, the `lr_warmup_steps` ramp.
- `SeqVaeLagAttnTrfCfsTask(SeqVaeLagAttnCfsTask, SeqVaeLagAttnTrfRwsTask)` and
  `SeqVaeLagAttnTrfCrwsTask(SeqVaeLagAttnCrwsTask, SeqVaeLagAttnTrfRwsTask)` are diamonds with
  **empty bodies**.

**Simpler alternative to B.7:** use `class SeqVaeLagAttnTrfPatchTask(SeqVaeLagAttnTrfCrwsTask)`.
It inherits `__init__(seed)`, `compute_loss_and_metrics`, `_stage`, the four bound members and the
LR ramp. It overrides only `_build_forward_inputs`, `_added_metrics` and optionally
`forecast_rows`, whose CRWS version draws `(B,A,H,R)` rows; plotting is disabled anyway. This has
the same behaviour as B.7 with three fewer members.

**RWS task step** (rws/task.py:624-753):
1. `inputs = _build_forward_inputs(batch)`, then `fhr, weight = _build_raw_target(batch)`.
2. `fo = self.model(*inputs)`, then `orig_model.compute_loss(fo, fhr, weight=…)`.
3. It adds `metrics["main_loss"]`, `mu_prior_sat_frac`, `delta_mu_sat_frac` and
   `mu_post_prior_gap_rms`.
4. On non-train stages it runs `perm_forward_outputs` and re-scores. Batch size and device come
   from `inputs[0]`.
5. `_added_metrics(...)` must not reuse a metric name; a collision raises.

---

## 7. `classifier/sources.py::VaeSource`

**Package dispatch: there is no registry.**
- `trainer_class(package)` (:148-159) imports `teb_vae.<package>.trainer` and requires **exactly
  one** class that is defined in that module (`c.__module__ == module.__name__`) and has
  `TASK_CLS`.
- Only `"lag_attn"` is refused by name. `VaeCfg.package` is a free string.
- So the classifier needs no code change for a new package, only a test (P5-02).

**Loading** (`load_task` :176-199):
- `torch.load(weights_only=False)`.
- `check_model_class(blob, MODEL_CLS.__name__)`. It only warns if `model_class` is absent.
- It requires `blob["model_kwargs"]` and `blob["hyper_parameters"]`.
- It builds `TASK_CLS(MODEL_CLS(**model_kwargs), model_kwargs=…, **{hp[n] for n in
  task_parameters(TASK_CLS)})`.
  - `task_parameters` (:162-173) walks the `__init__`s up the MRO while each one takes
    `**kwargs`.
  - `seed` must therefore be in `hyper_parameters`, which `save_hyperparameters("seed")` ensures.
- Weights load through `load_checkpoint_strict(task.orig_model, blob)`.

**Config and data:**
- `resolved_config.yaml` is found next to the checkpoint or in the run root
  (`cohort.py:resolved_config_for`).
- `checkpoint_loader_kwargs` (:202-222):
  - takes `dataset_kwargs` and appends `weight, guid, epoch` to `load_fields`;
  - takes `normalize_fields` from the config;
  - refuses `epoch_max`, `cs_label`, `bg_label` and `allowed_guids`;
  - reads `stats_path` from `dataset_config.stat_path`.
- `trim_minutes` must agree across the stats file, the loader and the classifier (:117-127).

**Forward:**
- `batch = task.transfer_batch_to_device(batch, device, 0)`.
- The frozen path calls `inputs = task._build_forward_inputs(batch)` and `out =
  model(*inputs)` **outside** a step (:367-369), so it relies on the class default `_stage = "val"`.
  That gives the dense anchor set.
- The co-training path hooks a forward inside `compute_loss_and_metrics` and requires exactly one
  forward (:371-379).

**Extraction** (`_output` :327-341):
- `delta_mu = out["mu_post"] - out["mu_prior"]`.
- `kld_per_dim = model.kld_tensor(...)`.
- `attn_summary(out["attn_weights"], lag_bins)`, which expects `(B,T,M,L)`.
- Any other key is read as `out[name]`, e.g. `mu_prior` and `kld_per_t`.
- Every output is reshaped to `(n, steps, -1)` with `steps = weight.shape[1]` (:409), so the keys
  must be dense on T. The ones listed are.

**Step mask** (:388-391): `(weight > 0) & (t >= model.warmup_period) & (t <
getattr(model, "anchor_ceiling", geometry.t_valid))`. With `causal_all`, `geometry.t` is used
instead. For the patch this is `[30, 270)`. Model attributes read: `warmup_period`,
`geometry.t/.t_valid`, `anchor_ceiling`, `kld_tensor`, `encode_source_kv`.

**`kld_excess` gate** (:269-274): it requires a forward parameter **named `u_stream`** and then
passes `inputs[2]` as the stream (:384). Keep the patch parameter named `u_patch`: then
`kld_excess` is refused cleanly instead of being handed the phase tensor.

**Feature-specific reads:** none in `VaeSource`. `Hdf5Source` and its config hold the ST/PH and
raw field names.

**Test precedent:**
- `classifier/tests/conftest.py:78-111`: `save_checkpoint` and the `vae_checkpoint` fixture use
  `lag_attn_transformer_cfs.tests.conftest`'s `TINY_KWARGS` and `make_task`. They write `best.ckpt`
  and `resolved_config.yaml`.
- `test_sources.py:39-81` checks the mask `t>=134` and ceiling 270.

---

## 8. `lag_attn_transformer_e2e/trainer.py` pre-flight

**The class:** `LagAttnTrfE2ETrainer(LagAttnTrfRwsTrainer)` (:95).
- `MODEL_CLS`, `TASK_CLS`, `CHECKPOINT_STEM = "lag-attn-trf-e2e"` (:99-101).
- `PLOT_CONFIG_KEY` is inherited: `"lag_attn_rws_plotting"` (rws/trainer.py:192).
- `@classmethod preflight(cls, config) -> None` (:103-134) calls the three guards below in order.
  The base hook is a no-op (rws/trainer.py:221).
- Other overrides: `causal_standing_message` (:136) and `_reference_frontend` (:168).
- `main(config_path)` (:318) calls `run_training(config_path, trainer_cls=…)`, which is
  `lag_attn_rws.trainer.main`.
- `RUN_CONFIG` is at :355.

**The guards** all have the signature `(config: Dict[str, Any]) -> None`:

| Guard / constant | Line | What it checks | Adaptable without copying? |
|---|---|---|---|
| `INERT_MODEL_KEYS: Dict[str, str]` | :74-92 | keys `c_y, c_u, use_up_st, causal_reach_budget_s, target_keep_index, target_delays, source_keep_index, source_delays` | n/a |
| `_check_no_inert_model_keys` | :198-222 | refuses any `INERT_MODEL_KEYS` key in `model_config.VAE_model` | **no**: it reads the module global and takes no key-list parameter |
| `_check_raw_source_normalized` | :225-258 | `"up"` (hardcoded) is in both `dataset_kwargs.load_fields` and `dataloader_config.normalize_fields` | reuse as-is |
| `_check_raw_length_against_shard` | :261-315 | `fhr.shape[1]` of `vae_train_datasets[0]` minus `2·240·trim_minutes` equals `sequence_length·raw_per_step`. It returns **silently** if there are no shards, if `sequence_length` or `raw_per_step` is absent from `VAE_model`, or if the h5 read fails. | reuse as-is; the patch config must state both keys |

**Other importable guards:**
- From `lag_attn_crws/trainer.py`:
  - `_check_boundary_term_is_off` (:346)
  - `_check_phase_key_fields` (:376; `PHASE_KEY_FIELDS = ("guid","epoch")` :90)
  - `_check_raw_target_fields(config, *, fields)` (:402; it also requires `weight`)
- From `lag_attn_rws/trainer.py`: `_check_raw_target_normalized(config, *, fields=("fhr",))`
  (:802).

**What the shared `main` already runs** (rws/trainer.py:640-650), before `trainer_cls.preflight`:
- `_check_stat_path`
- `_check_declared_widths_against_shard`, a no-op unless `c_y` and `c_u` are set
- `_check_raw_target_normalized(fields=TARGET_FIELDS)`, which is `("fhr",)`
- `_check_causal_budget_resolves`

**The kwargs sweep** (rws/trainer.py:265-305) forwards only keys on the constructor signature. An
unknown key is **dropped silently**, which is why the B.8 refusals are needed.

**Seed:** CFS and CRWS give the task its seed in `create_model` with
`self.apply_config_hyperparameters({"seed": general_config.seed}, self.pl_model)`
(crws/trainer.py:258-260, cfs/trainer.py:263-265). `LagAttnTrfRwsTrainer` does not.

**`TRACKED_METRICS`:** CRWS adds `train/val anchors_per_sample` and `val/kld_source_null`
(crws/trainer.py:114-147). The TrfRws trainer does not.

---

## 9. `lag_attn_transformer_e2e/nets/frontend.py::featurize`

- Signature: `featurize(raw: Tensor (B, L), weight: Tensor (B, T)) -> Tensor` (:225).
- It returns **one tensor** of shape `(B, 3, L)`, channel-major, in `raw`'s dtype. It is
  `stack((value, mask, delta), dim=1)` (:303). It is **not** a tuple; unpack with `value, mask,
  delta = featurize(raw, w).unbind(1)`.
- It raises `ValueError` if either input is not 2-D, if the batch sizes differ, or if `L` is not a
  positive multiple of `T`. `r = L // T` is derived, not passed.
- `mask`: `(weight >= VALID_THRESHOLD).repeat_interleave(r) & isfinite(raw)`, as a float. The
  threshold is `VALID_THRESHOLD = 1.0` (rws/nets/raw_masks.py:43), which is the same threshold the
  forecast mask uses.
- `value`: `where(valid, raw, 0)`. There is no normalization: the loader already z-scored the
  signal. Invalid and non-finite samples become an exact 0.
- `delta`: `(value − prev) · mask · prev_mask` with replicate padding, so `delta[..., 0] == 0`.
  Inside a patch `delta[..., 16t]` crosses the patch boundary; use `delta[..., 1:]` per patch.
- Every output is finite. A non-finite sample inside a `weight = 1` patch makes the input token
  `m_t = 0`, so it gets the `missing` embedding. The target summaries stay finite (that sample is
  0), and the patch is still scored if `weight = 1` (B.6.4).
- `refuse_time_pooling_norms(module, *, label="front end")` is at frontend.py:379-403 and is
  importable.

---

## 10. Conflicts with the plan

**C1 (minor).**
- Plan: B.3 step 1 says "`featurize(raw, w)` to get `(value, mask, delta)`".
- Code: it returns one stacked `(B, 3, L)` tensor (frontend.py:303).
- Suggested resolution: unpack with `.unbind(1)`. No plan change is needed beyond the wording.

**C2 (plan omission).**
- Plan: B.5 lists the four jobs of `PatchSummaryTarget`, and none of them is
  `_check_persistence_target`.
- Code: with `persistence_residual=True` the base raises at TrfRws:420-421/876-903, so
  `sweep_persistence.yaml` would not construct.
- Suggested resolution: `PatchSummaryTarget` overrides it as a no-op, as CFT:683 does.

**C3 (plan wording).**
- Plan: P0-01 step 3 places `_resolve_warmup_readout_constants` and `anchor_ceiling` in
  `causal_feature_target.py`, and lists `scored_weight`, `forecast_likelihood_kwargs`,
  `_set/_register_likelihood_structure` and `TARGET_BLOCK_SPLIT` as hooks CWI reads.
- Code: the first two live in `causal_inputs.py` (:460, :272). CWI reads none of the others; §1c
  lists their real readers.
- Suggested resolution: P2-02 uses §1 as the hook list. The real CWI target hooks are
  `_check_anchor_floor` and `_anchor_target_values`, plus the six `_resolve_*`/split members, which
  become unnecessary once `_resolve_warmup_readout_constants` is overridden.

**C4 (plan omission).**
- Plan: B.5 copies the CRWS keyword list and passes the result to the base through `locals()`.
- Code: `FORWARDED_EXCLUSIONS` does not cover the four new keys `target_summary_loc`,
  `target_summary_scale`, `variability_eps` and `source_validity`, so the base raises `TypeError`.
  The CRWS list also lacks `persistence_residual`; the plan does add it.
- Suggested resolution: extend the exclusion tuple locally, as in the §2 recipe.

**C5 (minor).**
- Plan: B.6 step 3 passes `horizon_weight=None`.
- Code: the copied signature keeps `horizon_weight_halflife_steps`. A config that sets it would
  register a `horizon_weight` buffer (TrfRws:454-459) that the loss then silently ignores.
- Suggested resolution: pass `getattr(self, "horizon_weight", None)`, as CRI and FFT do. It is
  `None` under the shipped `null`. Alternatively, drop the keyword from the signature.

**C6 (plan omission).**
- Plan: B.7 lists the members to write and does not include the class attribute
  `_stage: str = DENSE_STAGES[0]`.
- Code: `VaeSource` (sources.py:367-368) and the plot callback call `_build_forward_inputs`
  outside a step, so without the class default they fail with `AttributeError`.
- Suggested resolution: declare `_stage` as CRWS does (crws/task.py:209). A simpler route is to
  subclass `SeqVaeLagAttnTrfCrwsTask` (§6), which provides it.

**C7 (plan omission).**
- Plan: B.8 does not mention the seed.
- Code: `LagAttnTrfRwsTrainer.create_model` never puts `general_config.seed` into the task
  hyperparameters, so `anchor_phase` always keys on `seed = 0`.
- Suggested resolution: override `create_model` as `super().create_model()` followed by
  `self.apply_config_hyperparameters({"seed": ...}, self.pl_model)`, as in crws/trainer.py:258-260.
  It cannot be bound, because it uses zero-arg `super`.

**C8 (conflict).**
- Plan: B.8 says pre-flight "imports the e2e guards, adapts their key lists, and does not copy
  them".
- Code: `_check_no_inert_model_keys` reads the module global `INERT_MODEL_KEYS` and has no
  parameter. The other two hardcode `"up"`/`"fhr"`.
- Suggested resolution:
  - Write one refusal of about 10 lines over the patch's removed keys plus prefix matches
    (`causal_*`, `target_weight_*`), and `target_scored_horizon`, `target_forecast_shift`,
    `target_novelty_frac`.
  - Import `_check_raw_source_normalized` and `_check_raw_length_against_shard` as-is. The config
    must state `sequence_length` and `raw_per_step`, or the length check is silently skipped.
  - Import CRWS `_check_phase_key_fields`, `_check_raw_target_fields(config, fields=("fhr",))` and
    `_check_boundary_term_is_off`. Together these cover `weight`, `guid`, `epoch` and
    `lambda_boundary = 0`; the shared objective raises otherwise (losses.py:893).

**C9 (minor).**
- Plan: B.5 removes `use_up_st` from the signature.
- Code: the base default is then `True` (TrfRws:132). The value is inert here, because its only
  readers are the overridden readout method, the unused RWS `_build_source_stream` and pages.
- Suggested resolution: pass `use_up_st=False` to the base so the model does not claim an ST
  block.

**C10 (plan omission).**
- Plan: B.8 does not mention `TRACKED_METRICS` or `causal_standing_message`.
- Code:
  - The inherited `TRACKED_METRICS` lacks `anchors_per_sample` and `kld_source_null`, so neither
    reaches `metrics_history.csv`.
  - The inherited `causal_standing_message` (rws/trainer.py:239-262) logs "input features at step
    t read up to 974 s into their own future", which is false for patches.
- Suggested resolution: extend `TRACKED_METRICS` as CRWS does, and override the message as e2e
  does (:136).

**C11 (note, P5-02).**
- Plan: P5-02 says to name each changed classifier file.
- Code: the classifier has no package registry.
- Suggested resolution: expect zero changed classifier source files and one new test. The patch
  `trainer.py` must define exactly one class with `TASK_CLS` in its own module. Keep the forward
  parameter named `u_patch`, not `u_stream`; see §7 on `kld_excess`.
