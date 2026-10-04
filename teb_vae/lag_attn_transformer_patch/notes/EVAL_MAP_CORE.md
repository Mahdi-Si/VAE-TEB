# EVAL_MAP_CORE: the CFS evaluation core, mapped for the patch model

Written 2026-10-03 against `main` @ `1536732`. This is a read-only map; no repo file was edited.
Paths are relative to `teb_vae/`. Abbreviations: `E/` = `lag_attn_cfs/eval/`, `TE/` = `lag_attn_transformer_cfs/eval/`,
`P` = `SeqVaeLagAttnTrfPatch`, `PT` = `SeqVaeLagAttnTrfPatchTask`.
Out of scope here (mapped by other agents): what each analysis computes, Captum/attribution, and the lag and clinical analyses.

**Bottom line.** The core is target-agnostic except for about 12 touch points. All but one sit
behind **two seams that the model and task already expose**. The first is `metrics.model_inputs`,
which calls the task's three `_build_*` builders. The second is the CFS three-stream forward call
`model(y_st, y_ph, u_stream, anchor_phase=, anchor_stride=)`. With 7 small scratch shims (§5.3, no
repo edit), the **whole pipeline ran end to end** on the tiny patch checkpoint. 27 of 28 steps ran,
all 10 verdicts were computed, and `anchor_geometry_intact` passed.

---

## 1. Run flow (`E/run.py`)

### 1.1 Entry → outputs

| # | Step | file:line | Isolated? |
|---|---|---|---|
| 0 | `TE/run.py:_cli` → `resolve_arguments` → `main(**values)`, which calls `kwargs.setdefault("binding", TRF_CFS_BINDING)` | TE/run.py:159, :99 | – |
| 1 | `resolved_config_for(ckpt)` → `load_config` → `merge_eval_overrides_with_provenance(run_cfg, binding.overrides_path)` | E/run.py:1328-1339 | no |
| 2 | `force_single_process_loader`; `validate_eval_config` (refuses `occlusion_bands` that reach past `max_lag`) | :1348-1352; config_schema.py:539, :282 | no |
| 3 | `merged_analysis_functions(binding)` → `select_analyses(only, skip)` | :1355-1356 | no |
| 4 | `configure_numerics(seed)`, `make_output_dir(config, out, binding)` (fallback tag `binding.tag`), log sink, dump config | :1358-1372 | no |
| 5 | `read_checkpoint` → `load_task(ckpt, dev, blob, binding)`: class-name guard, then `binding.model_cls(**model_kwargs)`, `binding.task_cls(model, ...)`, strict load | :1380-1381; probe.py:677-756 | no |
| 6 | `preflight.run_preflight(... binding)` → `write_preflight` | :1384-1395; preflight.py:1685 | **no** (by design) |
| 7 | `attach_warmup_budget` (CFS page clocks; `None` when ungated) | :1398, :950 | no |
| 8 | `source_delay_steps`, `arm_record`, `describe_arm` | :1409-1438 | no |
| 9 | `metrics.likelihood_structure_record(model)` | :1443 | **no** |
| 10 | `load_or_collect_tables`: reuse tables, or loader → **`probe` step** → `binding.collect or collect.collect_tables` | :1448, :1655-1820 | probe yes; **collect no** |
| 11 | `report.results.update(collection.results)`; `cohort.build_cohort_block` | :1471-1478 | no |
| 12 | `AnalysisContext(collection, config, task, loader)`; `run_analyses(UNSKIPPABLE)`, then `run_analyses(selected)` | :1479-1491 | **yes**, per analysis |
| 13 | `revise_clock_verdict` (re-decides `coupling_exceeds_availability_clock` from `results.source_null.difference`) | :1496, :838 | no |
| 14 | `build_run_context` (calls `warmup_budget_record(model)`) | :1519, :1105, :979 | **no** |
| 15 | `finalise`: `lag_heads`, `headline` (shared plus `binding.headline_scalars`), `coverage`, `sanity`, manifest; each block under `_safe` | :1523; report_seam.py:960 | yes, per block |
| 16 | Write `summary.json`; the exit code is `report.exit_code()` (non-zero if any step failed) | :1608-1621 | – |

Outputs go to `<out>/eval_results/`: `summary.json`, `steps.json`, `preflight.json`,
`loader_probe.json`, `collection.json`, `per_sample.csv`, `per_anchor.parquet`,
`per_sample_vectors.npz`, `per_anchor_vectors.npz`, `retained_arrays.npz`, `eval.log`, and one
subdirectory per analysis. A run with no `--checkpoint` re-reads the tables. It builds no model and
no loader (:1714-1748).

### 1.2 `ModelBinding` (E/binding.py:53-135, frozen dataclass)

| Field | Type / default | Consumed at |
|---|---|---|
| `model_cls` | `type` | probe.py:713, :716 (`check_model_class(blob, model_cls.__name__)`, **name equality**), :734; preflight.py:849 (`warmup_model_kwargs`); error messages |
| `task_cls` | `type` | probe.py:735 (`task_cls(model, model_kwargs=, beta_schedule=, kld_beta=, beta_prior=, lambda_full=, lambda_base=, likelihood=, free_bits=)`) |
| `tag` | `str` | run.py:464 (`<tag>-eval` when `general_config.tag` is absent) |
| `geometry_keys` | `Tuple[str,...]` | preflight.py:1041. A key is compared only if it is in **both** the config's `VAE_model` and `model_kwargs`; otherwise it is skipped silently |
| `encoder_disclosure` | `model -> dict` | preflight.py:1524. Must not return any `SHARED_CAUSALITY_KEYS` (:201) |
| `overrides_path` | `Path` | run.py:1338 |
| `extra_analyses` | `Mapping`, `{}` | run.py:281. Merged as shared (minus trailing) + extras + `cross_subgroup`. A name collision raises |
| `headline_scalars` | `((name, path),...)`, `()` | report_seam.py:341. Appended after the shared ones; a name collision raises |
| `excluded_analyses` | `Tuple`, `()` | run.py:327-336. Removed after the merge; an unknown name raises. **Does not reach `UNSKIPPABLE_ANALYSES`** (`band_partition`) |
| `collect` | callable or `None` | run.py:1788. Same signature and return type as `collect.collect_tables` |

`TRF_CFS_BINDING` (TE/binding.py:275) reuses the CFS `EXTRA_ANALYSES` and `HEADLINE_SCALARS` by
identity. Its `GEOMETRY_KEYS` (TE/binding.py:82) is 26 keys.

### 1.3 Registry, selection, headline, verdicts

- **Registry.** The shared `ANALYSIS_FUNCTIONS` (run.py:223) are `forecast coupling perm_control latent lag_kl attention calibration residual distributions trajectory time_to_delivery second_stage events sufficiency cross_subgroup`. The binding extras (binding.py:242) are `samples recording_traces attribution warmup source_null time_shift occlusion lag_clocks lag_kld_scaled lag_high_kl spectral_skill`. `TRAILING_ANALYSES = ("cross_subgroup",)` (:278). `UNSKIPPABLE_ANALYSES = {band_partition}` (:194) always runs first.
- **`--only/--skip`.** `select_analyses` (:637). A comma-separated list checked against the merged registry; an unknown name raises before the checkpoint is loaded. Order is always the registry's. An unskippable name is refused.
- **Headline.** `build_headline` (report_seam:341) digs `HEADLINE_SCALARS` (:163, 44 shared paths) plus the binding's paths out of `results`. A missing path gives `None`, which `check_headline_finite` treats as allowed; a NaN fails it. `verdict_<name>` holds the status for each of the 10 names in `HEADLINE_VERDICTS` (:280).
- **Verdicts.** Computed once in the collection pass: `metrics.evaluate` → `build_verdicts` (metrics:3767) in the order of `VERDICT_REGISTRY` (:3298). One verdict, `coupling_exceeds_availability_clock`, is re-decided after the analyses by `revise_clock_verdict` (run.py:838) from `results["source_null"]["difference"]`. `anchor_geometry_verdict` (:3586) needs `anchors_per_sample == anchor_ceiling − F` and `target_warm_frac == 1.0` (:3652).
- **Sanity** (report_seam:901). `kl_identity`, `per_anchor_recombines`, `argmax_lag`, the lag identities, `per_file_counts`, `classes_present`, `target_not_truncated` and `headline_finite`. Failed checks are **not** reflected in the exit code; `verify.py` checks them.

### 1.4 Analysis contract (E/analyses/__init__.py)

- **Signature** (fixed, enforced by `tests/test_eval_protocol.py`): `run_<name>_analysis(context, *, eval_config, output_dir, probe) -> dict`.
- **Call** (run.py:794): `report.step(name, fn, context, eval_config=..., output_dir=results_dir, probe=probe_record)`. An exception becomes a failed `StepRecord` and the run continues. The result is stored at `results[name]`, plus `"grouped"` when it declared `grouped_frames`. `steps.json` is rewritten after every step.
- **`AnalysisContext`** (:74-105, frozen): `collection` (`per_sample`, `per_anchor`, vectors, retained arrays, `results`, `record`), `config` (merged), `task` and `loader`. `task` and `loader` are `None` on an offline re-run; the model-reading analyses must record a skip in that case.
- **Return value.** Must carry `n_samples`, `composition` and `plan` (`plan` includes `capped`). Optional `grouped_frames`: `[{path, value_columns, stem?, directory?, references?}]`, which the runner fans out by class and by subgroup (run.py:683). Figures and CSVs are written by the analysis itself under `output_dir/<name>/` through `figures_seam`. Layering: `analyses/*` may not import Lightning, a model, `task` or `trainer`, or another analysis.

---

## 2. Feature-target touch points in the core, and what the patch model needs

Status meanings: **R** = refusal (`EvalPreconditionUnmet`); **C** = crash outside isolation (aborts the run); **c** = crash inside a step; **W** = runs but the output is wrong or mislabelled; **ok** = runs correctly.
Handling: (i) new `ModelBinding` field (shared edit); (ii) subclass or override from the patch package; (iii) skip or exclude for patches; (g) a getattr fallback or one-line root-cause fix in shared code.

### 2.1 Config, preflight and probe

| Touch point | file:line | What it does | On P | Handling (size) |
|---|---|---|---|---|
| `TARGET_FIELDS=("fhr_st","fhr_ph")` → `check_target_normalized` | preflight.py:157, :556-575 | Requires the target fields to be in both `load_fields` and `normalize_fields` | **R** (first refusal in the dry run): P normalizes `[fhr, up]` | (i) `target_fields=("fhr","up")`, about 4 lines; or (ii) set `preflight.TARGET_FIELDS` from the patch run.py, 1 line, but this changes a module global |
| `check_declared_widths` via `config_view_for_shard_guards` (c_y, c_u, use_up_st) → rws `_check_declared_widths_against_shard` | preflight.py:456-488, :578; lag_attn_rws/trainer.py:738 | Requires `c_y == fhr_st+fhr_ph` and `c_u == up_st+up_ph` on the shard | **R**: 33 ≠ 102 | (i) a shard-guard hook. For P, run the patch trainer's `_check_raw_length_against_shard` on a view with `vae_test_datasets` copied to train. About 8 lines |
| `objective_channel_weights` (`target_channel_weight`, `TARGET_BLOCK_SPLIT`, `target_weight_st/ph`, `target_gate`, `c_y`) | preflight.py:1395-1438, called at :1781 | Discloses how the objective's channel-weight mass splits across ST and PH | **C**: `AttributeError` raised inside `run_preflight`, so it is a crash, not a refusal | (g) return `{"applies": False}` when `target_channel_weight` is absent. 2 lines |
| `STORED_BLOCKS` → `group_delay_summary` | preflight.py:133, :1322 | Per-block `causal_delay_s` read from the shard | **W**: records ST/PH delays the model never reads | (iii) leave it and say so, or (i) a statement hook |
| `CAUSALITY_STATEMENT`, `lag_axis` label and `GROUP_DELAY_CAVEAT` | preflight.py:182, :1536-1544; lag_axis.py | "stored coefficients, one-sided filter bank, group delay to …" | **W**: false for patches. Patch t reads raw samples Rt..Rt+R−1, so there is no group delay. `encoder_disclosure` cannot override these keys (:201, :1525) | (i) an optional `causality_statement` field, about 4 lines; or (iii) document it |
| `check_causal_transform` (shard `transform == "causal"`) | preflight.py:362 | Shard-variant identity | ok on the CFS holdout. The check is irrelevant to P, but P's DESIGN says "any build works" | keep it: it keeps the cross-cell comparison on the same shards |
| `check_trim_minutes == 1.0` | :336 | Trim rebase | ok, and needed: 5280 − 480 = 4800 = T·R | keep |
| `REQUIRED_EVAL_LOAD_FIELDS`, `check_load_fields`, `check_no_reach_budget`, `check_repointed`, `check_stat_path`, `check_test_shards_exist` | :144, :399, :431, :261, :542, :290 | Clinical fields and paths | ok | keep |
| `check_warmup_budget_matches_checkpoint` (`causal_warmup_budget_steps`, keep and warm-up tuples, align keys, `causal_representation`) | :758-971 | Re-resolves the budget and compares it with the stamped tuples | ok: budget unset and no tuples, so `gated: False`. It logs a **W** warning about "one-sided filter … stored channel" | keep (the warning text is wrong for P) |
| `reconcile_with_checkpoint` → `resolve_target_scored_horizon` (`target_phase_fast_*`) | :1041-1089 | Geometry-key equality plus the scored-horizon vector | ok: the rule is unset, giving `None == None`. The 27 keys in §4 all matched | binding `geometry_keys` (§4) |
| `verify_weights_loaded` (`posterior_head` delta heads, `horizon_core` FiLM) | :1120-1228 | Weight-space load check | ok | keep |
| `anchor_geometry`, `lag_support` | :1234, :1269 | Floor, ceiling, stride, lag margin | ok: 240 anchors, F=30, margin 30−8 ≥ 0 | keep |
| Probe `REQUIRED_BATCH_FIELDS` (adds `fhr_st, fhr_ph, up_st, up_ph`) | probe.py:101-117, :338 | First-batch field check | ok **only if** the overrides still load the four ST/PH blocks (unread and unnormalized, which is harmless) | keep them in `load_fields` (0 lines), or (i) `required_batch_fields`, 3 lines |
| `forward_contract` (`inputs[3], inputs[4]`, `c_y`, `target_warmup_steps`, `target_warm_frac`) | probe.py:779-862 | Measures the forward contract. Used by the **probe CLI only** (:1134), not by `run.main` | **c**: `IndexError` at :805, because P's task returns a 4-tuple | (g) `inputs[-2], inputs[-1]` plus getattr, 2 lines; or skip the CLI |

### 2.2 `run.main` body (outside isolation)

| Touch point | file:line | On P | Handling |
|---|---|---|---|
| `metrics.likelihood_structure_record` → `target_block_membership` (`TARGET_BLOCK_SPLIT`, `arange(c_y)` when ungated) | run.py:1443; metrics.py:474-517, :2131-2153 | **C**: no `TARGET_BLOCK_SPLIT`. Even with it, `arange(c_y=33)` against 2 channels is a shape error | (g) ungated → `arange(decoder_out_channels)` (the same value for CFS, where ungated c_y equals the decoder width), plus split constant 1 on P. Then `ar_coef_mean_st/_ph` mean level/variability. 2 lines |
| `warmup_budget_record` (`model.target_warm_frac`, `c_y`, `c_u`, gates, warm-up steps) | run.py:979-1016, via `build_run_context` :1220 | **C** at the **end** of the run, after every analysis, so no `summary.json` is written | P sets `target_warm_frac = 1.0` (true by construction), 1 line in `PatchSummaryTarget`; or (g) getattr |
| `attach_warmup_budget`, `source_delay_steps`, `arm_record` (`causal_align_reference*`) | :950, :1409, :1026 | ok (None, 0, None) | keep |
| `PRED_GAP_CONVENTION` text ("H*C_keep … wavelet-modulus") | report_seam.py:296 | **W** (text only) | (iii) document |

### 2.3 Collection pass (`collect.collect_tables` → `metrics.evaluate` → `evaluate_batch`, metrics.py:2190-2830)

**The collection pass does not call `model.compute_loss`.** It rebuilds the target itself with
`target = model._build_forecast_target(target_features, anchors)` (:2276). Every score then comes
from the shared raw-objective primitives (`masked_raw_block_per_anchor`, `forecast_mask`,
`kl_mask`) under `model.forecast_likelihood_kwargs()`.

For each batch it collects: dense forward outputs at (φ, S) = (0, 1); the base, full, shuffled and
base-shuffled branches at K Monte Carlo draws; mean-decoded scores; the training-path blocks;
persistence, climatology and segment-mean baselines; KL, the source-null arm (`controls.source_null_forward_outputs`)
and lag profiles; per-channel vectors; calibration sums; and the retained `target, mu_*, logvar_*,
up_raw, fhr_raw, weight, anchor_index, attn_weights`.

| Touch point | file:line | On P | Handling (size) |
|---|---|---|---|
| `model_inputs(task, batch)` → `task._build_target_streams` / `_build_source_stream` / `_build_raw_target` → `(y_st, y_ph, u_stream, target_features, weight)` | metrics.py:249-271, call :2263. **The same function is imported by** oracle:465, occlusion:349, time_shift:154, samples:487, recording_traces:326, attribution_pass:677 | **C**: P's task inherits the **RWS** builders, which feed `fhr_st/fhr_ph`/`up_ph` and raise on `c_y` (rws task.py:325) | (ii) **eval task subclass** overrides the 3 builders to return `(y_patch, y_patch[...,:0])`, `u_patch` and `(summary_target(fhr, weight) (B,T,2), weight)`. About 15 lines; no shared edit |
| Three-stream forward `model(y_st, y_ph, u_stream, anchor_phase=, anchor_stride=)` | metrics:2268, oracle:466, time_shift:157, samples:488, attribution_pass:684, attributions:587 | **C**: P's forward is `(y_patch, u_patch, phase, stride)`, so the call raises `TypeError` | (ii) **eval model view** whose `forward` takes the CFS form and calls `CausalWarmupInputs.forward(self, y, y[...,:0], u, φ, S)`. That is exactly what `PatchStreamInputs.forward` does. About 6 lines. It must keep `__name__ == "SeqVaeLagAttnTrfPatch"` for the guard at probe.py:716, or (i) add 1 line so `load_task` accepts a declared checkpoint class name |
| `model._build_forecast_target(features, anchors)` | metrics:2276, oracle:616 | **C**: absent on P | Add it **to `PatchSummaryTarget`**, gathering tokens `a+1 … a+H`, and have `compute_loss` call it so loss and eval share one definition. About 8 lines in the patch package |
| `anchor_support` → `model.scored_weight`, `geometry`, `coverage_floor` | :2156 | ok (identity weight) | – |
| `forecast_likelihood_terms` (`cell_mask`, `ar_coef`) | :402 | ok: `cell_mask=None`, `ar_coef=tanh(a)` of shape (2,) | – |
| `controls.perm_forward_outputs`, `source_null_forward_outputs(model, outputs, u_stream)` | :2303, :2020 | ok. Note: an all-zero `u_patch` means "valid and flat" (DESIGN §4), consistent with P's own `val/kld_source_null` | – |
| `mc_predictive_block`, `mean_decoded_block` (`model.decoder` + persistence) | :580, :814 | ok | – |
| `baseline_forecasts(target_features, weight, model, anchors)` (`target_gate`, `target_forecast_shift`, `target_warmup_steps`) | :891-1010 | ok given (B,T,2) summaries (no gate, no shift). **W**: climatology = 0 is the population mean only when `target_summary_loc/scale` come from `summary_stats.py`. default.yaml ships identity placeholders | none; note it in the patch EVAL.md |
| `target_block_membership` → `pred_gap_st/_ph` | :2534-2536 | **C**, as in §2.2 | same 2-line (g) fix. Columns then mean `pred_gap_level` / `pred_gap_variability` |
| `model.warm_tertile_id` → `pred_gap_warm_{lo,mid,hi}` (per sample and per anchor) | :2541-2545, :2723-2725 | **C**: absent | (g) getattr → NaN columns (4 lines). An all-zeros view attribute runs but puts the whole gap in "lo", which is **W**. Exclude `warmup` in either case |
| `model.target_warm_frac` column | :2554-2558 | **C**: absent | P attribute `= 1.0` (as in §2.2) |
| `source_lag_warmth_per_sample` (`source_block_warm_st/_ph`, (T,) bool) | :2058-2128, call :2560 | **C**: absent | view or model attributes all-True (honest: 1.0); or (g) getattr → NaN |
| `build_lag_mask`, `kld_tensor`, `te_analysis`; outputs `source_kl_lag_map`, `attn_weights`, `kld_per_t(_per_head)`, `mu_full/base`, `logvar_*` | :2389, :2615-2700 | ok (CFS key set) | – |
| Batch `up`, `fhr` (retained raw); `target`, `weight`, `guid`, `epoch`, `cs_label`, … (Collector) | :2768-2771; collect.py:463, :616-640, :760 | ok | – |
| `geometry_record` (`c_y` → `target_declared_width`, `target_gate.keep_index`) | collect.py:1004-1057 | **W**: records 33, which is the *input token* width; the target is 2 | (iii) a label note, or (g) record `decoder_out_channels` |
| `normalization_record`, `NORMALIZED_BLOCKS = fhr_st..up_ph` | collect.py:250, :1089 | **W**: records ST/PH stats, not `fhr/up` | (i) or (g) read the blocks to record from the config's `normalize_fields`. 2 lines |
| `expected_anchors_per_sample`, `bounds_record` | metrics:4102; collect.py:1060 | ok (240) | – |

### 2.4 Analyses the core forces or registers (detail is left to the other agents)

| Item | file:line | On P | Handling |
|---|---|---|---|
| `band_partition` (UNSKIPPABLE: shard `sel_*`, `use_up_st`, declared widths, `target_keep_index`) | run.py:194; analyses/band_partition.py:546 | **W**: emits an ST/PH channel map for a model with 2 summary channels. `excluded_analyses` cannot remove it | (i) let `excluded_analyses` also filter `UNSKIPPABLE_ANALYSES`. About 3 lines in run.py:1482 and :327 |
| `spectral_skill` | binding.py:282 | **c** (dry run): "kept-axis channel map describes 102 channels but the per-channel readouts are 2 wide" | (iii) `excluded_analyses` |
| `warmup` (tertiles, `source_lag_warmth`, geometry guards) | binding.py:255 | **W**: meaningless for patches | (iii) exclude. Lose 2 headline guards, or re-home `anchors_per_sample` and `target_warm_frac` from collection `readouts` |
| `samples` page seams | analyses/samples.py:487 | soft-fail inside the step: 28 pages failed, "x (4800,) vs y (600,)" | the other agents' scope (a summary-row page) |
| `oracle`/`sufficiency`, `occlusion`, `time_shift`, `recording_traces`, `attribution` | as listed | ran once the hooks were in. `time_shift`, `recording_traces` and `attribution` scored 0 rows on the 4-sample fixture, so they are **not exercised** | the other agents' scope |

---

## 3. Representation-agnostic: runs unchanged given a binding

- **Shared helpers:** `launch.py`, `_reuse.py`, `config_schema.py` (merge, provenance, loader forcing, `eval_config`; the bands must lie inside `max_lag`), `frames.py`, `cohort.py`, `dataset_rows.py`, `figures_seam.py` (it draws the coefficient lag label, but a step is 4 s in both cells), `report_seam.py` (all but the convention text) and `verify.py`. For `verify.py`, the gate and arm tables read `summary.json` only. The cross-cell table hardcodes `SeqVaeLagAttnCfs` and `SeqVaeLagAttnTrfCfs` at verify.py:569-570.
- **`run.py` orchestration:** output directory, numerics, registry merge and selection, `run_analyses`, grouped fan-out, `revise_clock_verdict`, `finalise`, the offline re-run and provenance (collect.py:1336-1603).
- **`probe.run_probe`** (population, per-file, GUID span), `read_checkpoint`, `load_task` (given name-matching classes) and `resolved_config_for`.
- **Preflight guards** for paths, stats, trim, transform, load fields, reach, reconcile, the ungated budget, weights loaded, anchor geometry and lag support. Also the causality disclosure's structure, `trf_cfs_encoder_disclosure` (verified on P: `lag_kv_source=adapter` gives reach 1), and `verify_weights_loaded`.
- **Collection, apart from the §2.3 rows:** branch scoring, MC marginalisation, the permutation and source-null controls, KL and lag profiles, attention entropy, calibration, the verdict registry, the Collector tables and sidecars, retention, and the event and contraction ages from raw `up`.

---

## 4. The minimum `TRF_PATCH_BINDING`

```python
TRF_PATCH_BINDING = ModelBinding(
    model_cls=PatchEvalModel,          # the eval view (§5.3); __name__ must stay "SeqVaeLagAttnTrfPatch"
    task_cls=PatchEvalTask,            # PT plus the 3 builders
    tag="lag_attn_trf_patch",
    geometry_keys=GEOMETRY_KEYS,       # below
    encoder_disclosure=trf_cfs_encoder_disclosure,   # reused; optionally wrapped to add raw_per_step
    overrides_path=<pkg>/eval/configs/eval_overrides.yaml,  # = TE's file: bands fit max_lag 37; keep ST/PH in load_fields for the probe
    extra_analyses=EXTRA_ANALYSES,     # CFS object, by identity
    excluded_analyses=("warmup", "spectral_skill"),
    headline_scalars=tuple(e for e in HEADLINE_SCALARS if e[1][0] not in ("warmup", "spectral_skill")),
)
```

**GEOMETRY_KEYS.** I took the constructor's keywords ∩ `configs/default.yaml` `VAE_model` keys (49
keys), then applied the TE rule. That gives TE's 26 keys **minus** `c_y, c_u, use_up_st` (P's
constructor does not accept them) **plus** the four keys that define the target or source:
`target_summary_loc, target_summary_scale, variability_eps, source_validity`. The 27 keys are:
`sequence_length d_model d_z horizon raw_per_step warmup_period max_lag num_heads d_head horizon_attention_blocks anchor_stride lag_floor prior_availability_input lag_kv_source persistence_residual forecast_ar_residual encoder_conv_kernels encoder_conv_dilations encoder_num_heads encoder_d_ff target_attention_blocks source_attention_blocks source_attention_window target_summary_loc target_summary_scale variability_eps source_validity`.
All 27 reconciled on the tiny checkpoint. `model_kwargs` stores lists, as the YAML does, so the
equality comparison works. `horizon_weight_halflife_steps` is excluded for TE's reason.

### 4.1 New hooks, ranked by preference

| # | Hook | Recommended | Size |
|---|---|---|---|
| H1 | Batch → model inputs and target (`model_inputs` contract) | (ii) `PatchEvalTask._build_target_streams/_build_source_stream/_build_raw_target` | about 15 lines in the patch package |
| H2 | Three-stream forward | (ii) `PatchEvalModel.forward` → `CausalWarmupInputs.forward` | about 6 lines + the class-name note |
| H3 | Forecast target gather | Put `_build_forecast_target` on `PatchSummaryTarget` and reuse it in `compute_loss` (one definition) | about 8 lines, patch nets |
| H4 | `target_warm_frac = 1.0`, all-True `source_block_warm_*` | Class attributes or properties on the eval view (true by construction) | 5 lines |
| H5 | Channel split `target_block_membership` | (g) ungated → `arange(decoder_out_channels)`, plus `TARGET_BLOCK_SPLIT = 1` on the view | 2 shared + 1 |
| H6 | `warm_tertile_id` | (g) getattr → NaN columns (shared), plus exclude `warmup` | about 4 shared |
| H7 | Target-field normalisation guard | (i) `ModelBinding.target_fields` (default `preflight.TARGET_FIELDS`) | about 4 shared |
| H8 | Shard width guard | (i) `ModelBinding.shard_guard: Callable[[config, model], None]` (default `check_declared_widths`). P passes a raw-length check | about 6 shared + 6 |
| H9 | Channel-weight disclosure | (g) `{"applies": False}` when the attributes are absent | 2 shared |
| H10 | `band_partition` | (i) `excluded_analyses` also filters `UNSKIPPABLE_ANALYSES` | about 3 shared |
| H11 | Causality statement and lag caveat | (i) optional `causality_statement`, or (iii) document | 4 shared, or 0 |
| H12 | `normalization_record` blocks; `geometry_record` declared width; probe CLI `forward_contract` | (g) | 2 + 1 + 2 shared |

The alternative with **zero shared edits** is H5–H9 done as monkeypatches from the patch `run.py`
(the §5.3 shim). It works and was verified, but it mutates `preflight.*` and `metrics.*` globals
for the whole process. That is acceptable for a one-model CLI process; it is not acceptable for a
pytest session that also imports CFS.

Total footprint, either way: about 60 lines in the patch package (`eval/binding.py`,
`eval/model_view.py`, `eval/run.py` modelled on TE/run.py, `eval/configs/eval_overrides.yaml`, and
`eval/verify.py` delegating) plus about 25 lines of shared getattr and binding-field edits. No fork
of `collect`, `metrics` or `run` is needed.

---

## 5. Dry run (tiny checkpoint, CPU, no repo edit)

Checkpoint: `output/teb_vae_trf_patch_tiny/2026-10-03--[17-25]-lag_attn_trf_patch_tiny/model_checkpoints/lag-attn-trf-patch-epoch=00.ckpt`.
The trainer ran in 6.3 s. The scratch scripts, which are temporary, were `dryrun.py` and
`shimrun.py` in the session scratchpad.

### 5.1 With TE's shipped `eval_overrides.yaml`

The first failure is `validate_eval_config` (config_schema.py:282): `occlusion_bands.near = [5, 14]
reaches past the model's max_lag=8`. This is specific to tiny, before any model is built. At the
shipped geometry (`max_lag: 37`) the bands pass, and the first refusal would be
`check_repointed` (`REPOINT_ME` paths), which is environmental.

### 5.2 With repointed local overrides (causal fixture shard, its stats, bands inside 8, the same `load_fields`)

| Stage | Result |
|---|---|
| `validate_eval_config`, `merged_analysis_functions`, `load_task` | OK |
| `check_repointed`, `test_shards_exist`, `stat_path`, `trim_minutes`, `causal_transform`, `load_fields`, `no_reach_budget` | OK |
| **`check_target_normalized`** | **R, the first failure in `run_preflight`** (preflight.py:575): `fhr_st`, `fhr_ph` are not in `normalize_fields` |
| `check_declared_widths` | R: `c_y=33` but the shard gives 102 (36+66); `c_u=33` but the shard gives 15 (`up_ph` only, because `use_up_st=False`) |
| `reconcile_with_checkpoint` (27 keys), `warmup_budget` (ungated), `verify_weights_loaded`, `causality_disclosure` | OK |
| `objective_channel_weights` | AttributeError `target_channel_weight` (crash, not a refusal) |
| `attach_warmup_budget`, `source_delay_steps` | OK |
| `metrics.likelihood_structure_record` (run.py:1443, not isolated) | AttributeError `TARGET_BLOCK_SPLIT` |
| `warmup_budget_record` (end of run, not isolated) | AttributeError `target_warm_frac` |
| loader, `run_probe` | OK (passes only because ST/PH are loaded) |
| `probe.forward_contract` | IndexError `inputs[4]` (probe.py:805) |
| `metrics.model_inputs` / `evaluate_batch` | RuntimeError at rws task.py:325: "target stream is 102 channels … built with c_y=33". This is the RWS feature builders. Past it, the three-stream forward would raise `TypeError` |

### 5.3 With the 7 shims (H1–H5 as an eval view and task, H7–H9 as monkeypatches; `warm_tertile_id` all zeros)

```python
class PatchEvalModel(SeqVaeLagAttnTrfPatch):          # __name__ reset to "SeqVaeLagAttnTrfPatch"
    TARGET_BLOCK_SPLIT = 1; target_warm_frac = 1.0
    def forward(self, y_st, y_ph, u_stream=None, anchor_phase=None, anchor_stride=None, **kw):
        if not torch.is_tensor(u_stream): return super().forward(y_st, y_ph, u_stream, anchor_phase)
        return CausalWarmupInputs.forward(self, y_st, y_ph, u_stream, anchor_phase, anchor_stride)
    def _build_forecast_target(self, s, a):            # tokens a+1..a+H, as compute_loss
        steps = a.long()[:, :, None] + torch.arange(1, self.horizon + 1, device=s.device)
        return s[torch.arange(s.shape[0], device=s.device)[:, None, None], steps]
    warm_tertile_id = property(lambda m: torch.zeros(m.decoder_out_channels, dtype=torch.long))
    source_block_warm_st = source_block_warm_ph = property(lambda m: torch.ones(m.sequence_length, dtype=torch.bool))
class PatchEvalTask(SeqVaeLagAttnTrfPatchTask):
    def _build_target_streams(self, b): y = patchify(b.fhr, b.weight, raw_per_step=self.orig_model.raw_per_step); return y, y[..., :0]
    def _build_source_stream(self, b): m = self.orig_model; return patchify(b.up, b.weight, raw_per_step=m.raw_per_step, validity=m.source_validity)
    def _build_raw_target(self, b): return self.orig_model.summary_target(b.fhr, b.weight), b.weight
# + preflight.TARGET_FIELDS=("fhr","up"); check_declared_widths, objective_channel_weights no-ops;
#   metrics.target_block_membership -> arange(decoder_out_channels) < TARGET_BLOCK_SPLIT
```

Result of `run.main(ckpt, out, binding=…, device="cpu")`: `summary.json` was written, exit code 1,
in about 45 s. **27 of 28 steps ran.** `spectral_skill` raised, as expected for the feature domain.
`samples` failed every page softly at the page seam. `time_shift`, `recording_traces` and
`attribution` scored 0 rows on the 4-sample fixture, so they were not exercised.

Verdicts on the untrained tiny model: `predictive_improvement` PASS, `anchor_geometry_intact` PASS,
the others FAIL or INCONCLUSIVE. The `argmax_lag` sanity check fails (a flat profile on an untrained
model).

Recorded geometry: 240 anchors, `block_width` 60, `scored_cells` 60, `ar_coef_per_channel` of
length 2. The **W** items in §2 showed up as predicted: `target_declared_width` 33, normalization
holds ST/PH blocks, and the `band_partition` map has 102 kept rows.

**Not verified:** that the eval's `nll_*_block` equals `compute_loss`'s value on the same forward.
H3 makes that true by construction; a parity test like `E/tests/test_eval_parity.py` would confirm
it.
