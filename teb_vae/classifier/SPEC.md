# CTG Outcome Classifier — Specification and Implementation Roadmap

**Status:** v1.2 spec, 2026-09-28. Nothing in this package is implemented yet.
**Package:** `teb_vae/classifier/` (this file lives there).
**Companion:** [`RESEARCH.md`](RESEARCH.md) is the literature and best-practice report this spec relies on. Citations like **[R: Thresholds]** point to its sections, and §19 lists the primary references.

---

## 0. How to use this document

This is the normative spec for a downstream classification module. It predicts adverse neonatal outcome from CTG. It works at segment level and continuously at GUID (recording) level. It runs under k-fold cross-validation, for binary and 3-class tasks. Inputs are representations from any TEB-VAE family in `teb_vae/`, or from the scattering (ST) / phase-harmonic (PH) coefficients stored in the HDF5 shards.

- **MUST / SHOULD / MAY** carry RFC-2119 meaning.
- **[FACT]** marks something verified in the code or data on 2026-09-28, with a `file:line` reference. **[DECISION]** marks a design choice this spec makes. **[OPEN]** marks a question for the project owner (collected in §18). Defaults of [OPEN] items are chosen so implementation can proceed.
- Implement **phase by phase** (§16). Each phase has acceptance criteria. Do not build later phases' options early. Progress is tracked in the phase table at the top of §16: mark a phase `done` only when its acceptance criteria pass.
- Reuse the existing code listed in §3 before writing new code. Do not copy files out of `tmp/`: that folder is temporary and is never committed. Read it for reference only.
- Keep this package small. There is one function per option, selected by a config string. There is no plugin registry, no factory hierarchy, and no base class that has only one subclass.

---

## 1. Goals and non-goals

### 1.1 Goals

- **G1. Pluggable inputs.** Per-timestep feature streams on the 4-second step grid, from:
  - (a) any TEB-VAE checkpoint: posterior/prior means, log-variances, Δμ, per-step KL, lag attention, encoder states;
  - (b) ST/PH coefficients from the shards;
  - (c) a cache of either.
- **G2. Context inputs.** Time from labour onset, labour stage (first / second / unknown) and time since second-stage onset. Also optional clinical covariates (e.g. maternal temperature), static or time-stamped, which may be missing for some GUIDs or segments.
- **G3. Prediction units.**
  - **Segment:** a score per 22-min segment.
  - **GUID offline:** one score per recording.
  - **GUID online:** a causal score after every observed segment, which gives a continuous risk trajectory and alarms.
- **G4. Tasks.** Binary (several class mappings) and 3-class (healthy / acidosis / HIE), as a multiclass or an ordinal head.
- **G5. Training.** Choice of labeling strategy, loss, sampler and training regime: frozen cached features, frozen online, partial fine-tune, LP-FT, or co-training with the VAE objective.
- **G6. k-fold protocol.** Uses the dataset's predefined `fold_k/{train,val,test}` splits. Per fold: fit on train; select, calibrate and choose thresholds on val; report on test. Nothing is ever tuned on test.
- **G7. Complete evaluation.**
  - Validation and test, across all folds, at segment, GUID and online levels.
  - Threshold policies, including an **FPR cap** (e.g. α = 0.3) with an optional finite-sample guarantee.
  - Sensitivity, specificity, FPR, PPV, NPV, AUROC, AUPRC, pAUC, calibration, 3-class metrics, time-resolved and alarm metrics.
  - Subgroups, patient-level bootstrap confidence intervals, and model comparison.
- **G8. Reproducible outputs.** One long-format prediction table and one long-format metrics table per run. Every figure and summary is rendered from those two tables.
- **G9. Guards.** Leakage and validity guards enforced by code and tests (§12).

### 1.2 Non-goals (v1)

- New VAE architectures, or changes to any VAE package.
- Re-splitting the cohort. The predefined folds are used; a re-split mode is deferred (§18).
- External or temporal validation. The spec only makes it possible: the saved model and thresholds can score a new shard set.
- A hyperparameter-search framework. Sweeps are separate configs run by the same CLI.
- Clinical deployment tooling.

---

## 2. Data facts the implementation depends on

All of these are **[FACT]** unless marked otherwise. Sources:
- `hdf5_dataset/new_pipeline/create_new_pipeline.py` (**CNP**)
- `hdf5_dataset/hdf5_dataset.py` (**HD**)
- `hdf5_dataset/dataset_explained_research.md` (**DER**)
- the sample shard `tmp/data/hie_cs.hdf5`

### 2.1 Time grid and segments

| Quantity | Value | Source |
|---|---|---|
| Raw sampling | 4 Hz | CNP:140-146, HD:441 |
| Segment length | 5280 raw samples = **1320 s = 22 min** | CNP:140-146 |
| Trim (production) | 1 min each end → 4800 samples, **1200 s observed** | HD:445 `decimated_trim_steps`; configs `trim_minutes: 1.0` |
| Decimation | ×16 → **1 step = 4 s**; 330 steps untrimmed, **T = 300 after trim** | HD:441-442 |
| `epoch` | untrimmed segment **start**, in seconds relative to delivery (negative) | CNP; `latent_pilot/data.py:1196` |
| Absolute time of step t | `epoch + 60·trim_minutes + 4·t` (s rel. delivery) | `latent_pilot/data.py:1196 anchor_seconds` |
| Observed window | `[epoch + 60, epoch + 1260]` | derived |
| Stride (code default) | 1200 s (`OVERLAP_PERCENTAGE = 1/11`) | CNP:141 |
| **Stride in the sample shard** | **660 s (50% overlap)**. Epoch differences are all multiples of 660; the file was built by a variant of the script | measured |
| Extraction window | last 12.4 h (`MIN_DOMAIN_START_DATASET = -44640`) | CNP:150 |
| Segments per GUID | **variable**. Sample: 19–61 (median 38). Up to about 38 slots at the 1200-s stride | measured |

> **"22" is the segment length in minutes, not the number of segments per GUID.** A GUID has N_g ≥ 1 segments. They are ordered by `epoch`, and gaps between them can be large: in the sample, 2–9 contiguous runs per GUID and gaps up to about 3 h. **[OPEN-1]** in §18 asks whether you intended a fixed 22-slot grid. The design below handles any N_g.

- **Missing segments** come from quality gates. The implementation MUST NOT assume adjacency in the file means adjacency in time. The gates (CNP:2974-3006) drop a segment if any of these hold:
  - mean `weight` < 0.90;
  - an FHR flat run > 120 s, a UP flat run > 300 s, or total FHR flat time > 300 s;
  - a duplicate `domain_start`;
  - a post-delivery start;
  - a transform failure.
- **Slot index.** `slot = round((epoch − epoch_grid_origin) / stride_s)`. The stride MUST be inferred from the data as the mode of positive within-GUID epoch differences, never hard-coded.
  - Known bug to avoid: `hdf5_dataset/guid_hdf5_dataset.py:401-403` hard-codes 1200 s, which causes 239 index collisions on the sample shard.
- **Other sample-shard observations.** In the sample shard (an all-CS HIE subgroup) the latest segment starts at −2640 s, and 2 of 15 GUIDs have **no** segment overlapping the last hour. Decision-time metrics MUST handle GUIDs that have no late data (§11.5).

### 2.2 HDF5 schema (per segment row)

| Key | Shape (stored) | Shape (loaded, trim 1.0) | Notes |
|---|---|---|---|
| `fhr`, `up` | (N, 5280) f4 | (4800,) | raw bpm / mmHg, z-scored at load if requested |
| `fhr_st`, `up_st` | (N, C, 330) | (300, C) | causal build: C = 36. Two-sided build: C = 43 |
| `fhr_ph` | (N, C, 330) | (300, C) | `integer_harmonic_v1`: 44. Legacy: 66 |
| `up_ph` | (N, C, 330) | (300, C) | 10 (integer) / 15 (legacy) |
| `fhr_up_ph` | two-sided only | (300, 79) | absent in causal files |
| `target` | (N, 330) | (300,) | **class × weight**: 1 = healthy, 2 = acidosis, 3 = HIE; 0 = invalid step |
| `weight` | (N, 330) | (300,) | {0, 1} per step; 0 where raw FHR is 0 or padding |
| `epoch` | (N,) | scalar | s rel. delivery |
| `time_from_labor_onset` | (N,) | scalar | `epoch − onset_s`; NaN if unknown |
| `second_stage_onset` | (N,) | scalar | `epoch − ss_onset_s`, signed; NaN if unknown |
| `cs_label`, `bg_label` | (N,) u1 | bool | subgroup constants |
| `guid` | (N,) str | str | raw `.mat` stem; mixed case and hyphenation |

- **Block widths differ by build.** The feature config MUST read channel counts from the file, not hard-code them. `HD:285 _check_layouts_agree` refuses to mix builds.
- **Per-block causal attributes:** `causal_warmup_steps` (per channel, in untrimmed steps) and `causal_delay_s`. Cold cells (`t < warmup[c]`) MUST be masked when ST/PH are consumed directly. Use `CombinedHDF5Dataset(..., emit_validity_mask=True)` or `channel_valid_mask` (HD:1365).
- **Normalisation is not stored in the shard.** It is applied at load time from a `stats_path` file (HD:708-830, 1485-1616): ST channel 0 is linear, other ST channels are log, PH is asinh, then z-score. The stats file MUST share the loader's `trim_minutes`.
- **Loader.** `CombinedHDF5Dataset(paths, load_fields, allowed_guids, cs_label, bg_label, epoch_min, epoch_max, label, cache_size, pin_memory, dtype, stats_path, normalize_fields, trim_minutes, emit_validity_mask)` (HD:1172-1189). Do **not** use its `label=` filter: it tests exact equality on the weight-scaled target and silently drops partially valid segments (`latent_pilot/data.py:281-284`).

### 2.3 Labels and subgroups

- **Segment class code** comes from `teb_vae/lag_attn/eval/labels.py:68 clinical_class_code(target_row, weight_row)`: the modal `round(target/weight)` over valid steps. The label is constant within a GUID. It is assigned from the source folder, i.e. the subgroup file.
- **Subgroups** (CNP:129-138):

  | Subgroup file | class | cs | bg |
  |---|---|---|---|
  | `healthy_no_bg_no_cs` | 1 | 0 | 0 |
  | `healthy_no_bg_cs` | 1 | 1 | 0 |
  | `healthy_bg_cs` | 1 | 1 | 1 |
  | `healthy_bg_no_cs` | 1 | 0 | 1 |
  | `acidosis_cs` | 2 | 1 | 1 |
  | `acidosis_no_cs` | 2 | 0 | 1 |
  | `hie_cs` | 3 | 1 | 1 |
  | `hie_no_cs` | 3 | 0 | 1 |

  - `bg_label` means "a blood gas is available". It is not a value.
  - No pH, base excess, Apgar or temperature exists anywhere in the data. The only external CSV columns the pipeline reads are `trace_guid`, `labor_onset_hours` and `second_stage_onset_hours` (CNP:1724-1768).
- **Label-proxy fields.** `bg_label = 0` means healthy by construction. `cs_label` is a delivery outcome. `source_file_basename` encodes the subgroup. None of these may ever be a model input (§12).

### 2.4 Clocks and labour stage

- **Time from labour onset at segment end:** `tlo_end = time_from_labor_onset + 1260` (s). It is negative before onset. It is NaN when the GUID is absent from the metadata CSV: 13% of segments in the sample.
- **Second-stage onset.** Let `ss = second_stage_onset`. The observed window relative to second-stage onset is `[ss + 60, ss + 1260]`.

  | Stage | Condition |
  |---|---|
  | first | `ss + 1260 ≤ 0` |
  | second | `ss + 60 ≥ 0` |
  | straddle | otherwise |
  | unknown | `ss` is NaN (64% of segments in the sample) |

  - **NaN means unknown, not "never reached second stage".** In a CS cohort the two are confounded.
  - A second-stage onset recorded exactly at delivery is a known sentinel artefact. Treat it as unknown. `teb_vae/lag_attn_cfs/eval/cohort.py:573 second_stage_eligibility` detects it but only counts it; the classifier deliberately sets `stage = unknown`.
- **Causal vs future information:**
  - `1[ss + 1260 ≥ 0]` (in second stage by the segment end) is **causal**.
  - `max(ss + 1260, 0)` (time spent in second stage) is **causal**.
  - The magnitude of a negative `ss` (time *until* second stage) is **future information** and MUST NOT be an input. The previous classifier leaked it through a signed `ψ(SSO)` feature (§3.3).
- **TLO missingness is manipulated by cohort construction.** 75% of the healthy no-BG group is forced to have TLO (CNP:156, 2348). So `has_tlo` is label-correlated, and §7.3 requires a confound check.

### 2.5 Cohort and folds

**Directory layout.** `k_fold_cross_validation_dataset/fold_{1..10}/{train,val,test}/<subgroup>.hdf5`. `pre_training_dataset/` holds healthy-BG GUIDs used for VAE pretraining.

**Split mechanics (CNP:2525-2709):**
- 10 folds, `RANDOM_STATE = 42`.
- Stratified by subgroup and by tertiles of labour duration.
- **Grouping is by GUID only.** There is no patient ID.
- In the default "augmented" mode, every fold's test set gets **the same** extra healthy no-BG GUIDs (CNP:2697-2701). Pooled out-of-fold (OOF) test metrics MUST de-duplicate them (§11.7).

**Class balance and prevalence:**
- Selection (CNP:2194-2479) makes train/val roughly **1:1 healthy : unhealthy**.
- Test is augmented with population-proportional healthy no-BG GUIDs, so **test prevalence is lower than val prevalence**. Calibration and PPV/NPV must account for this (§11.7).
- Within the unhealthy class, HIE is rare. An earlier cohort build reported fold 1 as H 2081 / A 1848 / HIE 231 GUIDs (`tmp/new_classifier/possible_improvements.md:48-51`).

**Eligibility is asymmetric:**
- Unhealthy GUIDs need ≥ 2 h valid; healthy GUIDs need ≥ 3 h, judged in the last 6.37 h (CNP:152-153, 1919-2038).
- So recording length and segment count are label-correlated. §10.9 requires a metadata-only "shortcut" baseline to measure this.

**Pretraining exposure:**
- The VAE is pretrained on `pre_training_dataset/`.
- On the resume path, that pool is "every BG file minus the fold GUIDs" with no eligibility check (CNP:3434-3456).
- Disjointness between the pretraining GUIDs and the classification test GUIDs MUST be verified at run time (§12, L3).

### 2.6 VAE representation facts (`lag_attn_transformer_cfs`, shipped `configs/default.yaml`)

- **Forward pass.** `outs = task.model(*task._build_forward_inputs(batch))`. It uses dense anchors (phase 0, stride 1) whenever the task is not in a training stage (`teb_vae/lag_attn_cfs/task.py:409-434, 560-567`). The output keys are listed at `teb_vae/lag_attn_cfs/nets/causal_inputs.py:932-962`.
- **Loading.** `teb_vae/lag_attn_cfs/eval/probe.py:677 load_task(checkpoint_path, device, *, blob=None, binding=CFS_BINDING)` with `binding=TRF_CFS_BINDING` (`teb_vae/lag_attn_transformer_cfs/eval/binding.py:272`). The model rebuilds from `checkpoint["model_kwargs"]` alone. **The checkpoint's `model_kwargs` are authoritative over the design docs**, which are partly stale.
- **Latents are dense** `(B, T=300, d_z=64)` on the same 4-s grid as ST/PH. Steps depend only on steps ≤ t.
- **Warm-up.**
  - Steps `< warmup_period` (134 in the shipped config) are warm-up. Decoder supervision covers anchors in `[134, 270)`.
  - Features at steps 270–299 are causal but unsupervised.
  - Read `warmup_period` and `sequence_length` from `model_kwargs`; never hard-code them.
- **Coverage consequence** (derived, **important**). After masking warm-up, each segment contributes steps `[134, 300)`, i.e. **664 s** of features:
  - At the **660-s stride** these windows tile the timeline almost exactly.
  - At the **1200-s stride**, about 45% of the timeline is never covered by post-warm-up features.
  - This is the likely reason the sample shard uses 660 s. **[OPEN-2]**
- **Stored-channel delay.** Stored channels lag physical time by their group delay (13–791 s, uncompensated). This is a caveat for time-resolved plots, not something to correct.

**Keys available (trf_cfs):**

| Key | Shape |
|---|---|
| `mu_prior`, `logvar_prior`, `mu_post`, `logvar_post`, `z_prior`, `z_post` | (B, T, 64) |
| `target_state`, `source_state` | (B, T, 128) |
| `attended_source_heads` | (B, T, 4, 32) |
| `attn_weights` | (B, T, 4, 38), entmax, exact zeros |
| `kld_per_t` | (B, T) |
| `kld_per_t_per_head` | (B, T, 4) |
| `source_kl_lag_map` | (B, T, 38) |
| `anchor_index`, `anchor_valid` | (B, A) |

- Under `base_decode: mean`, `z_prior` is the same tensor object as `mu_prior`, so use the means.
- Per-dimension KL: `model.kld_tensor(mu_prior=, logvar_prior=, mu_post=, logvar_post=)` returns (B, T, 64) (`teb_vae/lag_attn_transformer_rws/nets/model.py:1238`).
- A source-null counterfactual is available: `teb_vae/lag_attn_rws/nets/controls.py:283 source_null_forward_outputs`.

**What trained runs showed** (earlier configs; `DIAGNOSIS.md`, `LAG_READOUT_DIAGNOSIS.md`):
- **Latent usage collapsed.** About 2 dimensions carry more than 80% of the KL, `kld_active_frac` was about 0.11, and the prior log-variance sat at its floor about 95–98% of the time.
- **`kld_per_t` is mostly an availability clock.** 67–86% of it survives replacing the source with its mean.
- **The lag readout was degenerate** (argmax lag = 0).
- **The UP pathway overfit.**

**Consequences for the classifier (all features must be standardised per channel):**
1. `mu_prior` (FHR-only) is the most robust default.
2. `mu_post − mu_prior` (Δμ) is secondary.
3. KL and attention are ablations. KL is best used as an **attention cue**, not a pooled value (§9.3).

### 2.7 Other VAE families (G1)

- **Shared structure.** Every family exposes `task.model(*task._build_forward_inputs(batch))` and stamps `model_class` and `model_kwargs` into its checkpoint.
- **Differences:**

  | Family | Difference |
  |---|---|
  | `lag_attn_{rws,fs,cfs,crws}` and their `lag_attn_transformer_*` twins | Same key set as trf_cfs. The rws/fs variants have no `anchor_*` keys |
  | `lag_attn_transformer_e2e` | Consumes raw `(fhr, up, weight)` |
  | `lag_slot_transformer_cfs` | Latents on an **anchor axis** `(B, A, d_z)`; `kld_per_anchor[_dim]`; no `attn_weights` or `kld_per_t`. Features must be scattered to T via `anchor_index` |
  | `lag_attn` (original) | Different key names (single `z`, `te_lag_map`). **Unsupported in v1** unless an adapter is added |

- **Bindings that exist:**

  | Binding | Location |
  |---|---|
  | `CFS_BINDING` | `lag_attn_cfs/eval/binding.py` |
  | `RWS_BINDING` | `lag_attn_rws/eval/run.py:224` |
  | `TRF_BINDING` | `lag_attn_transformer_rws` |
  | `TRF_CFS_BINDING` | `lag_attn_transformer_cfs` |
  | `LAG_RESIDUAL_BINDING` | `lag_slot_transformer_cfs` |

  The rws families have their own `load_task` at `lag_attn_rws/eval/run.py:618`. For other families, build a binding from `teb_vae.<pkg>.trainer.MODEL_CLS / TASK_CLS`.

### 2.8 Environment

- The project venv (`.venv`) runs **Python 3.14.7, scikit-learn 1.9.0, torch 2.14.0+cu130, pandas 3.0.5, lightning 2.6.5, torchmetrics 1.9.0, scipy 1.18.1**.
- `requirements.txt` pins older versions (sklearn 1.8.0, torch 2.7.1, pandas 2.3.1). **[OPEN-10]**
- The implementation targets the venv. It needs **no new dependencies**: sklearn, scipy, torch, lightning, pandas/pyarrow, matplotlib and pydantic are all installed.
- Watch out for these:
  - **pandas 3:** default string dtype. GUID columns must be explicitly `str`.
  - **sklearn ≥ 1.9:** `cv='prefit'` is removed. Use `FrozenEstimator` if `CalibratedClassifierCV` is used.
  - **mlflow autolog:** only tested up to torch 2.13. Log manually.

---

## 3. What already exists, and what to reuse

### 3.1 Reuse map

| Need | Existing code | Action |
|---|---|---|
| Rebuild any cfs-family model from a checkpoint | `teb_vae/lag_attn_cfs/eval/probe.py:677 load_task`; bindings (§2.7) | **Import** |
| Loader config inherited from a checkpoint's resolved config (stats, trim, fields) | `teb_vae/lag_attn_transformer_cfs/latent_pilot/data.py:216 pilot_loader_config` | **Import or port**. Keep its refusals of inherited cohort filters (`data.py:291-302`) |
| Segment-level HDF5 access | `hdf5_dataset/hdf5_dataset.py:1091 CombinedHDF5Dataset` | **Import** |
| Segment metadata table and per-GUID consolidation with exclusion reasons | `latent_pilot/data.py:397 segment_frame`, `:545 recording_frame`, `:615 exclusion_counts` | **Import**. Extend with the columns of §6.1. Port only if the signature does not fit |
| Class code from target/weight | `teb_vae/lag_attn/eval/labels.py:68 clinical_class_code` | **Import** |
| Split disjointness, patient groups, pretraining exposure record | `latent_pilot/data.py:633 check_split_disjoint`, `:705 attach_patient_groups`, `:759 exposure_record` | **Import** |
| Train-only, recording-weighted feature scaler with floor | `latent_pilot/extract.py:592 fit_scaler` (+ `_hierarchical_moments`) | **Generalise.** The existing one is keyed to one latent; the new one is per channel, over any stream |
| Keyed, fingerprinted extraction cache (npz + index parquet, `assert_same_keys`) | `latent_pilot/extract.py:139-509` | **Pattern to follow.** Storage per §8.4 |
| ROC points, confusion counts, rate NaN rules, stratified cluster (paired) bootstrap | `latent_pilot/evaluate.py:1256 confusion_counts`, `:1449 roc_points`, `:1691 paired_bootstrap`, `:1866 metric_intervals`, `:2028 subgroup_table` | **Import** where signatures fit; they are model-agnostic |
| Config `base:` chains and deep merge | `teb_vae/lag_attn/config.py:85 load_config`, `:38 _deep_merge`, `:126 resolve_config_file` | **Import** |
| Length-bucketed GUID batching | `hdf5_dataset/length_bucket_sampler.py:151 VariableBatchBucketSampler` | **Import** (seeded, `set_epoch`) |
| Figure layer: style, rendering, colours, standard panels | `teb_vae/lag_attn_cfs/eval/figures_seam.py`: `configure_figure_style`, `render_figure`, `new_figure`, `group_colors`, `CLINICAL_CLASS_COLORS`, `SUBGROUP_COLORS`, `ribbon_plot`, `violin_panel`, `windowed_comparison_figure`, `caveat_note(text=…)` | **Import** (§11.10). **Not** `utils/style.get_class_colors`, which paints healthy blue, nor a direct `save_figure` |
| Cohort ordering and labels | `teb_vae/lag_attn/eval/labels.py`: `ordered_groups`, `distinct_groups`, `class_name`, `subgroup_of`, `GROUP_COLUMNS` | **Import** |
| Group statistics | `teb_vae/lag_attn/eval/stats.py`: `holm_adjust`, `kruskal_across_groups`, `pairwise_comparisons`, `wilcoxon_paired`, `windowed_group_comparisons`, `delta_magnitude`, `bootstrap_ci` (means only) | **Import** (§11.6.2, §11.14) |
| Fail-soft reporting and manifests | `teb_vae/lag_attn/eval/report.py`: `Report` (`step`, `exit_code`, `write`), `json_safe`, `build_manifest`; `teb_vae/lag_attn_cfs/eval/report_seam.py`: `write_steps`; `teb_vae/lag_attn_cfs/eval/launch.py:29 resolve_launch_args` | **Import** (§11.14) |
| Second-stage sentinel detection | `teb_vae/lag_attn_cfs/eval/cohort.py:573 second_stage_eligibility`. It **counts only**; the classifier then sets `stage = unknown` itself, a deliberate divergence | **Import** |
| Verify gate pattern | `teb_vae/lag_attn_cfs/eval/verify.py` (`CRITERIA`, PASS/FAIL/INCONCLUSIVE) | **Follow the pattern** (§11.15) |
| Sample-page selection | `teb_vae/lag_attn_cfs/analyses/samples.py:691 per_class_rows`, `:753 extreme_rows`; `teb_vae/lag_attn/eval/masks.py:404 subsample_indices` | **Port** the selection (analysis modules must not be imported); import `subsample_indices` |
| Training framework | `train/graph_model_base.py GraphModelBase`, `train/pl_model_base.py LightningModelBase`, `train/callbacks.py` (`MetricsLoggingCallback`, `MetricsHistoryCsvCallback`, `LossPlotCallback`, `HyperparameterLoggingCallback`, `MLflowRunLoggingCallback`), Lightning `EMAWeightAveraging`, `ModelCheckpoint`; test doubles `train/test_utils.py` | **Subclass and import** (§10.10) |
| Warm-up masking and log-scale helpers for per-step figures | `teb_vae/lag_attn_cfs/eval/attributions.py:849 masked_field`, `:882 signed_log_norm`, `:896 unsigned_log_norm`, `:906 symlog_axis` | **Port** (~40 lines). Importing `attributions` loads captum at module level |
| Samples-page layout (stacked full-width rows on one time axis) | `teb_vae/lag_attn_rws/sample_page.py:837 build_diagnostic_figure` (and the cfs extension) | **Follow the layout** for per-GUID trajectory pages |
| Logging, MLflow seam, seeding/determinism | `utils/custom_logger.setup_logging`, `utils/mlflow_utils.log_artifact_to_mlflow`, `train/graph_model_base.py:255 configure_determinism` | **Import** |
| Tiny fixture shards | `scripts/make_tiny_shard.py`, `latent_pilot/tests/fixtures/generate.py` | **Reuse** for test fixtures |

**Layering.** `teb_vae/` sits at the top (`utils ← train ← teb_vae`, enforced by `train/tests/test_layering.py`). Code in `teb_vae/classifier/` may import `utils`, `train`, `hdf5_dataset` and sibling `teb_vae.*` packages. It must **not** import `model`.

### 3.2 `latent_pilot`: what it is, and how this module relates

`teb_vae/lag_attn_transformer_cfs/latent_pilot/` is a one-fold, one-seed, binary, **recording-level** pilot. It compares:
- a linear probe on pooled `mu_post`, against
- fine-tuning `posterior_head.delta_mu_head`.

It gates the fine-tune with forecast-preservation checks.

Its data, time, extraction, scaler, bootstrap and test-lock primitives are high quality, so import them.

This module **supersedes** the pilot's scope. The pilot lacks:
- k-fold;
- 3-class;
- segment and online outputs;
- FPR-capped thresholds;
- calibration;
- covariates;
- pluggable sources;
- co-training.

Two pilot ideas carry over as-is:
- the **selection lock**: test predictions are produced only after selection is frozen;
- the **preservation gate**, which monitors forecast degradation when the encoder is trained (§10.6).

### 3.3 Lessons from the previous classifier (`tmp/new_classifier/`, reference only)

**Results:** best fold-1 validation AUROC was 0.683, from a 3-layer causal transformer over segment tokens. A linear probe on the last 3 segments reached 0.626. Later variants fell to 0.56–0.61.

**Do:**
- Use a causal transformer over segment tokens, with a relative-time attention bias computed from segment timestamps.
- Use per-position (online) outputs, and check them against an explicit prefix sweep.
- Choose thresholds on validation only.
- Initialise the head bias from the class prior.
- Keep `epoch` out of the model inputs, enforced by a test.
- Always run the linear probe as a floor. A neural model below the probe means a wiring or training bug.
- Add a 3-class auxiliary head, which probably helped binary AUROC.

**Don't:**
- **Broadcast the GUID label to every position with uniform weight.** Position 0 cannot beat the prior, and the model collapsed to the class prior (the "plateau"). → horizon/decay weighting, a bag loss, `k_warm` (§6.5, §10.3).
- **Select checkpoints on total loss while AUROC peaks elsewhere**, with no class balancing and tuning via optimizer brakes. → early-stop on GUID-level validation log loss or AUROC (§10.5).
- **Use a signed `ψ(second_stage_onset)` input.** It leaks time-to-delivery before second stage. Keeping the second-stage zero-sentinel GUIDs in training leaks `epoch` directly. → §2.4 and §12.
- **Filter evaluation cohorts after training.** Dropping GUIDs without second stage from val/test biases the CS/HIE strata. → report cohorts, never silently filter them (§11.6).
- **Widen the test window relative to val** (`epoch_min_test`). Committed FPR accumulates with exposure, so test FPR overshoots the cap. → one window for all splits.
- **Gap-fill epochs** in evaluation. `fill_missing_epochs` dropped real off-grid rows and moved alarms. → never impute rows; use running-max scores (§11.2).
- **Let the threshold search fall back silently to 0.5.** → raise instead.
- **Headline averages over time bins.** → report metrics at named checkpoints.
- **Embed DataFrames in JSON.** → parquet tables.
- **Swallow errors broadly** with `except Exception` around whole report sections. → fail loudly.
- **Report no confidence intervals, no pooled OOF analysis, no PPV/NPV, no lead time.** → all are in §11.
- **Fit latent statistics on an unshuffled first-16k-segment subset.** That makes the scaler depend on file order. → fit on the full train fold (§8.5).
- **Apply weight decay to biases, LayerNorms and the prior-initialised head bias.** → exclude them (§10.5).

The full bug list is in Appendix A.

---

## 4. Functional requirements

| ID | Requirement |
|---|---|
| FR-1 | Build a cohort table per fold and split from the HDF5 shards: segments, GUIDs, labels, clocks, stage, exclusions with reasons, slot index. Persist it as parquet. |
| FR-2 | Validate folds: GUIDs disjoint across train/val/test within a fold; both classes present per split; shared test GUIDs across folds detected; pretraining exposure recorded. |
| FR-3 | Map class codes to the configured task (§6.3). Assign training targets and weights by the configured labeling strategy (§6.5), and evaluation windows by the configured evaluation strategy. |
| FR-4 | Provide feature streams from a VAE checkpoint (any supported family), from shard fields (ST/PH/raw), or from cache. Each stream comes with a per-step validity mask. |
| FR-5 | Encode context features: TLO, stage, time in second stage, Δt, valid fraction. Join optional static or time-stamped covariates causally, with missingness handling. |
| FR-6 | Train a classifier per fold (and per seed) with the configured scope, model, loss, sampler and regime. Early-stop on validation. |
| FR-7 | Fit calibration on validation, per fold. |
| FR-8 | Compute thresholds on validation per policy: FPR cap (empirical and NP-umbrella), Youden, sensitivity target, fixed. |
| FR-9 | Write val and test predictions for segment, GUID and online levels, with raw and calibrated scores, in the schema of §11.1. |
| FR-10 | Compute every analysis of the catalogue (§11.12: cohort, ROC/PR, thresholds, metric types, subgroups, calibration, confusion/3-class, alarms, heterogeneity, errors, baselines) per fold and pooled, for val and test, with patient-level bootstrap CIs. Compare runs. |
| FR-11 | Render every figure of `FIGURE_REGISTRY` and a `summary.md` from the tables alone (§11.10–§11.13). |
| FR-12 | Always run the controls: linear probe, metadata-shortcut baseline, shuffled-label control. |
| FR-13 | Enforce the leakage guards of §12 by code, and cover them with tests. |
| FR-14 | Support resume: stages are idempotent per fold and seed, keyed by a settings digest. |
| FR-15 | Train through the repo's `train/` framework, with the same callbacks, `metrics_history.csv`, loss plots, loguru logs, checkpoints and MLflow conventions as the VAE families. Nest MLflow runs per unit under a run-level parent (§10.10). |
| FR-16 | Provide a torch-free verify gate over the evaluation outputs, including output completeness (§11.15). |

**Non-functional requirements:**
- One full 10-fold, frozen-cached run on one RTX 4080 SHOULD finish in under 2 h, excluding the one-off feature extraction.
- Extraction is done **once per unique segment** across folds for frozen sources (§8.4).
- Deterministic given a seed (`seed_everything(workers=True)`), except for documented cuDNN nondeterminism.

---

## 5. Architecture and big picture

### 5.1 The module in one paragraph

**Inputs.** The module takes a cohort of CTG recordings (GUIDs). Each GUID is a variable number of 22-min segments with gaps. The recordings are already split into k folds of train/val/test. For each segment it builds a per-4-second feature stream: VAE latents, KL, attention, or ST/PH coefficients. It pairs that stream with causal context: labour clock, stage and optional covariates.

**Model and outputs.** A small classifier turns each segment into a token. Optionally, a causal sequence model runs over the tokens. Together they give a **segment score**, a **continuous GUID score after every segment**, and a **final GUID score**.

**Per fold.**
- The model is fit on train.
- Early stopping, calibration and thresholds are fit on val.
- Only then is test scored.

**Evaluation.** Everything downstream reads two tables, `predictions/*.parquet` and `evaluation/tables/metrics.parquet`.
- **Scores into decisions.** Threshold policies turn scores into decisions, e.g. FPR ≤ 0.3 chosen on val.
- **Metric types.** Decisions are read through three metric types: **instantaneous**, **committed cumulative** and **committed overall**.
- **Axes and outputs.** These are evaluated on several time axes, per fold and pooled across folds, with patient-level confidence intervals. The results are rendered as figures and `summary.md`.

### 5.2 Run lifecycle

| Step | Stage (`run.py --stage`) | What happens | Section |
|---|---|---|---|
| 1 | `cohort` | Read every `fold_k/{train,val,test}` shard. Build segment and GUID tables (labels, clocks, stage, slot, exclusions). Map class codes to the task, and apply the labeling strategy. Validate folds (disjointness, pretraining exposure, shared test GUIDs). Run the missingness-confound check | §6, §7, §12 |
| 2 | `extract` | Frozen sources only. Compute features once per unique segment across all folds, then cache them with a fingerprint | §8 |
| 3 | `train` | For each fold × seed: fit the scaler on train; train the model (§9–§10) and the mandatory baselines; early-stop on val; fit calibration on val; select every threshold policy on val (§11.3). Write `selection_lock.json` | §10, §11.3 |
| 4 | `predict` | Write val and test prediction rows at the segment, online and GUID levels. Test requires the lock | §11.1 |
| 5 | `evaluate` | Build the score views (§11.2). Apply policies and metric types (§11.5) on every time axis, level and subgroup. Aggregate per fold and pooled, with bootstrap CIs | §11.2–§11.9 |
| 6 | `report` | Render figures and `summary.md` from the metrics tables only, and log to the MLflow parent. `compare` contrasts runs | §11.10–§11.13 |
| 7 | `verify` | Torch-free gate over `summary.json`: completeness, leakage, controls, FPR overshoot | §11.15 |

### 5.3 Data and model flow

```
                  ┌──────────── cohort.py ────────────┐
 HDF5 shards ───► │ segment table · GUID table · tasks │──► cohort/*.parquet
 (fold_k/split)   │ label strategies · fold checks     │
 covariate CSVs ─►│ context features (tlo, stage, cov) │
                  └───────────────┬────────────────────┘
                                  │ (row ids, labels, weights, context)
                  ┌──────────── sources.py ───────────┐
 checkpoint ────► │ VaeSource | Hdf5Source | cache     │──► feature cache (per unique segment)
                  │ derived feats · step masks · scaler│
                  └───────────────┬────────────────────┘
                                  │ X[N_seg,T,C], M[N_seg,T]
                  ┌──────────── data.py ──────────────┐
                  │ SegmentDataset | GuidDataset       │ (collate: pad segments, seg_mask)
                  └───────────────┬────────────────────┘
                  ┌──────────── model.py ─────────────┐
                  │ step proj → [temporal] → pooling   │ segment token
                  │ ⊕ context fusion                   │
                  │ [sequence aggregator (causal)]     │ per-position states
                  │ head(s): binary | multiclass | ord │
                  └───────────────┬────────────────────┘
       losses.py · train.py (regimes, early stop, EMA, seeds) · calibration
                                  │
                  predictions/*.parquet  (segment | guid | online rows)
                                  │
     thresholds.py · metrics.py (engine, online/alarm, pooling, bootstrap, DeLong)
                                  │
                  evaluation/tables/metrics.parquet ──► report.py (figures, summary.md)
```

### 5.4 What each split is used for, per fold

```
 fold k ─┬─ train ──► fit: feature scaler · covariate scaler/vocab · class weights & priors · model weights
         │
         ├─ val ────► select: early stopping / best.ckpt · calibration map (temperature/Platt)
         │            · every threshold policy (FPR cap, NP umbrella, Youden, …) per basis (§11.3)
         │            · alarm-rule parameters (k_of_n) · probe C
         │            ──► selection_lock.json  (config digest + checkpoint digest)   ◄── nothing below may change it
         │
         └─ test ───► score only: apply frozen model + calibration + thresholds ──► predictions rows
                                                                                          │
 all folds' test rows (shared test GUIDs de-duplicated) ──► pooled OOF analysis  ◄────────┘
 all folds' val rows ─────────────────────────────────────► "optimistic" validation analysis
```

### 5.5 Evaluation and analysis pipeline

```
 predictions/segments.parquet  (one row per scored segment: p_seg, p_online, clocks, labels, strata)
 predictions/guids.parquet     (one row per GUID: final / aggregated scores)
            │
            ▼  §11.2 score views
 ┌─────────────────────────────────────────────────────────────────────────────────────────┐
 │ segment score s_n     online score s_g(n)     running max r_g(n)     final score s_g     │
 └─────────────────────────────────────────────────────────────────────────────────────────┘
            │
            ▼  §11.3 thresholds (chosen on VAL, per fold, per policy, per basis) → frozen for TEST
            │
            ▼  analysis grid (every cell computed on val and test, per fold and pooled)
 ┌───────────────┬──────────────────────────────────┬──────────────────────────────────────────┐
 │ level         │ segment · guid (offline) · online │                                          │
 │ metric type   │ threshold-free (AUROC, AUPRC, pAUC, calibration)                             │
 │ (§11.5)       │ instantaneous · committed cumulative · committed overall                     │
 │ time axis     │ to_delivery · from_onset · rel_second_stage · position · elapsed            │
 │ time points   │ checkpoints (e.g. 6,4,3,2,1,0.5 h) · bins (0.5 h) · end of recording         │
 │ policy        │ np30 · emp30 · emp15 · youden · … (each evaluated under all 3 metric types)  │
 │ task / head   │ binary · 3-class (argmax, collapsed adverse score, per-class OvR)            │
 │ subgroup      │ 18 families (§11.6.1): class · source_file · class×cs · healthy bg×cs · cs ·  │
 │               │ bg · stage · tertiles (length, span, quality, labour) · has_tlo · fold · …   │
 └───────────────┴──────────────────────────────────────────────────────────────────────────────┘
            │
            ▼  §11.5 alarm analysis (lead time, cumulative detection, fraction of healthy alarmed, burden)
            ▼  §11.7 aggregation: per-fold · pooled OOF (confusion pooling) · GUID bootstrap CIs
            ▼        prevalence-shift adjusted PPV/NPV · §11.8 model comparison
 evaluation/tables/metrics.parquet  +  metrics/summary.json
            │
            ▼  §11.10–11.11
 figures/ (ROC/PR, metric-type curves vs time, alarm curves, calibration, confusion, trajectory pages)
 summary.md (cohort flow, headline table, baselines/controls, thresholds & overshoot, TRIPOD+AI list)
```

**Reading guide.** The *threshold-free* rows say how well the scores rank. The three *metric types* say what a clinician would experience at the chosen operating point:
- **instantaneous:** is it flagging now?
- **committed cumulative:** of those monitored so far, how many have been flagged?
- **committed overall:** of all labours, how many have been flagged by now?

The *alarm analysis* adds how early the flags come, and how many false alarms a healthy labour receives.

### 5.6 Package layout **[DECISION]**

```
teb_vae/classifier/
  SPEC.md  RESEARCH.md
  __init__.py
  config.py      # pydantic v2 schema (extra="forbid"), load via teb_vae.lag_attn.config.load_config, digest
  cohort.py      # §6, §7: tables, tasks, label strategies, context features, covariate join, fold checks
  sources.py     # §8: feature sources, derived features, masks, scaler, cache
  data.py        # datasets, collate, samplers
  model.py       # §9: encoder, pooling, fusion, aggregators, heads
  losses.py      # §10.2-10.3
  train.py       # §10: ClassifierTask(LightningModelBase), ClassifierTrainer(GraphModelBase), callbacks, regimes, calibration fit, ensembling
  baselines.py   # §10.9: linear probe, shortcut baseline, shuffled-label control (sklearn)
  thresholds.py  # §11.3
  metrics.py     # §11.4-11.9, §11.12, §11.14: metric engine, analysis registry (blocks C/R/T/M/S/K/X/A/H/E/B/Q), pooling, bootstrap, comparison
  report.py      # §11.10-11.13: FIGURE_REGISTRY, figures via the VAE figure seam, pages, summary.md, MLflow parent logging
  run.py         # §14: CLI, stages, fold/seed unit loop + teardown, MLflow parent, run dir, provenance, RUN_ARGS
  verify.py      # §11.15: torch-free gate over evaluation/summary.json
  configs/       # default.yaml, smoke.yaml, st_ph.yaml, segment_scope.yaml, cotrain.yaml
  tests/         # §15
```

**Why pydantic.** The project's configs are plain dicts. The pilot's dict validator is about 1.5k lines. `pydantic.BaseModel(model_config={"extra": "forbid"})` gives the same strictness (unknown keys rejected, types and ranges checked) in far less code, and pydantic 2 is already installed. The YAML still uses `base:` chains through `load_config`, and the resolved YAML is written into the run directory.

---

## 6. Cohort, tasks and labeling strategies (`cohort.py`)

### 6.1 Segment table (one row per stored segment, per fold × split)

| Column | Type | Definition |
|---|---|---|
| `fold`, `split` | int, str | from the directory |
| `ds_index` | int | index into the `CombinedHDF5Dataset` built over this split's files (row identity) |
| `source_file` | str | shard basename (subgroup). **Never an input** |
| `guid` | str | raw stem |
| `guid_norm` | str | uppercase with hyphens removed (the same rule as CNP:435-444 `_normalize_guid`), for external joins |
| `epoch_s` | float | segment start |
| `t_end_s` | float | `epoch_s + 60·trim + 4·T` (end of the observed window) |
| `slot` | int | §2.1 |
| `seg_pos` | int | 0-based rank by `epoch_s` within the GUID (within the split) |
| `class_code` | int | 1/2/3 from `clinical_class_code` |
| `cs`, `bg` | bool | label proxies, **evaluation only** |
| `tlo_end_s` | float | §2.4, NaN if unknown |
| `ss_rel_s` | float | `second_stage_onset`, NaN if unknown or sentinel |
| `stage` | cat | first / straddle / second / unknown (§2.4) |
| `valid_frac` | float | mean `weight` over the observed window |
| `hours_to_delivery` | float | `−t_end_s/3600`. **Evaluation and label construction only** |
| `excluded`, `exclusion_reason` | bool, str | see below |

**Exclusion reasons:**
- `duplicate_epoch` (keep the first, report the count)
- `crosses_delivery` (`t_end_s > 0`)
- `low_valid_frac` (`valid_frac < data.min_valid_frac`)
- `outside_window` (`epoch_s < data.epoch_min_s`)
- `label_conflict`

Build the table by iterating a `CombinedHDF5Dataset` with metadata-only `load_fields` (`target`, `weight`, `epoch`, `guid`, `time_from_labor_onset`, `second_stage_onset`, `cs_label`, `bg_label`), as `segment_frame` does. Record `ds_index` so features can be fetched later by index.

### 6.2 GUID table

One row per (fold, split, guid):
- `class_code`
- `n_segments`
- `first_epoch_s`, `last_t_end_s`
- `has_tlo`, `has_ss`
- `cs`, `bg`
- `shared_test` (the GUID appears in more than one fold's test split)
- `excluded` with a reason, via `recording_frame` semantics: conflicting codes or metadata exclude the GUID; there is no majority vote.

`min_segments_per_guid` (default 1) is applied here, and the count is reported.

### 6.3 Tasks (class mappings) **[DECISION]**

`labels.task` maps `class_code` → target. A code mapped to `exclude` removes the GUID from **training and evaluation of that task**, and the count is reported.

| `labels.task` | 1 healthy | 2 acidosis | 3 HIE | K | Notes |
|---|---|---|---|---|---|
| `adverse_vs_healthy` (default) | 0 | 1 | 1 | 2 | primary binary task |
| `hie_vs_rest` | 0 | 0 | 1 | 2 | rare positive; NP thresholds need enough negatives |
| `hie_vs_healthy` | 0 | exclude | 1 | 2 | |
| `acidosis_vs_healthy` | 0 | 1 | exclude | 2 | |
| `three_class` | 0 | 1 | 2 | 3 | `labels.head: multiclass` (default) or `ordinal` **[OPEN-6]** |
| `cs_outcome` | — | — | — | 2 | target = `cs`. For exploring the treatment paradox only; `cs` is still never an input |

**Head options:**
- `labels.head ∈ {binary, multiclass, ordinal}`. `binary` requires K = 2. `ordinal` requires an ordered K ≥ 3.
- `labels.aux_3class_weight` (λ₃, default 0) adds a 3-class auxiliary head to a binary task.

**Cohort filters (`cohort.*`):**
- `include_healthy_no_bg` (default true). The no-BG healthy group has no blood-gas confirmation and a manipulated TLO rate. **[OPEN-8]**
- `subgroups`: an allow-list of shard basenames.

### 6.4 Clock of a segment for label construction

Use `Δ_n = hours_to_delivery` of segment n, measured at its observed end. It is used **only** to build training weights and targets (§6.5) and evaluation windows (§6.6). It is never an input.

### 6.5 Labeling strategies for training (`labels.strategy`) **[DECISION]**

All strategies produce, per segment (segment scope) or per position (sequence scope), a target `y_n` and a loss weight `ω_n ≥ 0`. Negative GUIDs always get `y_n = 0` and `ω_n = 1`. For positive GUIDs:

| Strategy | Positive-GUID target / weight | Rationale |
|---|---|---|
| `propagate` | `y_n = y_g`, `ω_n = 1` | Weakest supervision. Kept as a baseline; the old classifier's plateau shows why it is not the default |
| `horizon` | `y_n = y_g`, `ω_n = 1[Δ_n ≤ H]` (otherwise excluded from the loss) | CTG convention (last 60 min) [R: Weak labels] |
| `horizon_decay` (**default**) | `y_n = y_g`, `ω_n = 2^{−max(0, Δ_n − H)/h}` | Temporal label smoothing (Yèche 2023); `latent_pilot` recency weights |
| `final_only` | only the last position / segment of each GUID is scored | offline GUID classification |
| `mil` | no per-segment target. A bag score `s_bag = τ·log mean_n exp(s_n/τ)` (or the attention-pool output) gets `ℓ(y_g, s_bag)` | Ilse 2018; "some segment is pathological" |

**Defaults:** `H = labels.horizon_h = 1.0` h, `h = labels.decay_halflife_h = 0.5` h.

**`k_warm`** (default 0 for `horizon*`, 3 for `propagate`) drops the first k positions of each GUID from the per-position loss in sequence scope.

**GUID-equal weighting:**
- Per-segment losses are normalised per GUID, so each GUID contributes equally: loss_g = Σω·ℓ / Σω, then the mean over GUIDs.
- GUIDs with Σω = 0 contribute only through the final or bag terms.
- Without this, long labours dominate the loss [R: Plain BCE — clustering].

**Symmetry caution.** `horizon` excludes early segments of positive GUIDs but keeps all segments of negative GUIDs. The model can then learn "late-labour appearance" as a positive cue. That is acceptable for training, but evaluation MUST be time-matched (§6.6), and the per-checkpoint metrics of §11.5 expose any such bias.

### 6.6 Evaluation windows (`labels.eval_window`)

Segment-level metrics are computed on a **time-matched** set: the same window applies to positive and negative GUIDs.
- `all`: every retained segment.
- `horizon`: segments with `Δ_n ≤ H`, for both classes.
- `bins`: per `eval.bin_h` bins of `Δ` (§11.5).
- `stage:first` / `stage:second`.

GUID-level and online metrics are defined in §11.2 and §11.5, not by this key.

### 6.7 Fold validation (FR-2)

- **Within-fold disjointness.** GUIDs must be disjoint across train, val and test within a fold. Use `check_split_disjoint`, and group by patient when `data.patient_map` (GUID → patient JSON) is supplied, via `attach_patient_groups`.
- **Class presence.** Every split must contain every class of the task, via `require_both_classes`. For 3-class tasks, check per class.
- **Shared test GUIDs.** Flag `shared_test`. Pooled metrics apply `data.shared_test_policy ∈ {first_fold (default), exclude}` (§11.7).
- **Pretraining exposure.**
  1. Read the VAE checkpoint's resolved config (`vae_train_datasets` and friends).
  2. Collect those shards' GUIDs.
  3. Intersect them with every classification split.
  4. Any overlap with **test** is an error unless `data.allow_pretrain_overlap: true`.
  5. Record the result via `exposure_record`.
- **Written reports.** `cohort/fold_summary.csv` holds GUID and segment counts per fold × split × class × subgroup, plus exclusions by reason.

---

## 7. Context features: time, stage and clinical covariates (`cohort.py`)

### 7.1 Allowed inputs **[DECISION]**

| Input | Allowed | Encoding |
|---|---|---|
| Feature streams (§8) | yes | — |
| `tlo_end` | yes (`context.tlo`) | `ψ(h) = sign(h)·log1p(|h|)` on hours, plus a missing flag (§7.3) |
| stage at segment end | yes (`context.stage`) | one-hot {first, straddle, second, unknown} |
| time in second stage | yes (`context.time_in_ss`) | `ψ(max(ss_rel + 1260, 0)/3600)`; 0 if not in second stage; unknown shares the stage flag |
| Δt since previous observed segment | yes, sequence scope only (`context.delta_t`) | `log1p(Δt_h)`; 0 for the first position |
| elapsed monitoring time | ablation only (`context.elapsed`, default off) | `log1p(h)`. **Label-correlated via asymmetric eligibility (§2.5)** |
| `valid_frac`, fraction of masked steps | yes (`context.valid_frac`) | raw fraction. Signal loss is informative [R: failure modes] |
| covariates (§7.3) | yes | value + missing flag |
| `epoch`, `hours_to_delivery`, `t_end_s` | **never** | — |
| negative part of `ss_rel` (time until second stage) | **never** | — |
| `cs`, `bg`, `source_file`, `class_code` | **never** | — |
| GUID-level `n_segments`, `first_epoch`, recording length | **never as a feature**. Sequence models see elapsed positions implicitly; the shortcut baseline (§10.9) measures what that leaks | — |

These rules are enforced by a column **allow-list** in `data.py` and by the invariance test T-L1 (§15).

### 7.2 Where context enters the model

- **Segment scope:** context features of that segment are fused (§9.4).
- **Sequence scope:** each token carries its own segment's context.
  - `Δt` and relative time also enter the attention bias (§9.5).
  - The TLO time embedding, if enabled, is a Time2Vec of `tlo_end` hours: `φ₀ = ω₀t + α₀`, `φᵢ = sin(ωᵢt + αᵢ)`, i = 1..7. When TLO is missing, the vector is multiplied by 0 and the missing flag is set.

### 7.3 Clinical covariates

There are none in the shards (§2.3). They come from optional external tables joined on `guid_norm`:
- **Static** (`context.covariates.static_csv`): columns `guid` plus variables (e.g. parity, gestational age, induction). Constant per GUID.
- **Timed** (`context.covariates.timed_csv`): long format with `guid, time_s, variable, value`.
  - `time_s` is in seconds relative to **delivery**. It is used only to align observations to segments, never as an input.
  - Example: maternal temperature measurements.
  - **As-of join, strictly causal:** for segment n, take the latest observation with `time_s ≤ t_end_s(n)` and age ≤ `max_age_h` (default 2 h).
  - Features per variable: value, missing flag, and optionally `log1p(age_h)`.

**Processing:**
1. Numeric variables are standardised using **train-fold** mean and std. Categorical variables are one-hot encoded, with the vocabulary taken from the train fold.
2. Missing values are imputed as 0 after standardisation, and the missing flag is set.
3. **`missing: indicator | no_indicator`.** The indicator helps when missingness is informative, but it harms calibration when missingness depends on the outcome (Sisk 2023) [R: covariates].
4. **Missingness confound check (mandatory).** For every covariate and for `has_tlo`, compute the train-fold missing rate per class. If |Δ| > `context.missing_confound_max` (default 0.10), the run logs a WARNING, writes the value to the manifest, and runs the `no_indicator` ablation if `context.auto_ablate_missing: true`.
5. **Covariate dropout during training:**
   - `dropout_p` (per variable, default 0.2): set the value to missing.
   - `block_dropout_p` (whole block, default 0.1).
   - This keeps a CTG-only path alive (ModDrop), and lets one model be evaluated both with and without covariates (`eval.covariates_off: true` adds a second prediction pass).
6. **Only prospectively available variables may be used.** The config MUST declare each variable with `available_at: prospective`. Delivery mode, intervention flags and retrospective labour duration are rejected.

---

## 8. Feature sources (`sources.py`)

### 8.1 Contract **[DECISION]**

```python
@dataclass(frozen=True)
class StepFeatures:
    values: torch.Tensor      # (B, T', C) float, channels concatenated in config order
    step_mask: torch.Tensor   # (B, T') bool, True = usable step
    attn: torch.Tensor | None # (B, T', C_a) optional attention-only cues (never pooled)
    channels: tuple[str, ...] # names, e.g. ("mu_prior[0]", ..., "kld_per_t")

class FeatureSource(Protocol):
    step_seconds: float                  # 4.0 × time_pool
    required_fields: tuple[str, ...]     # batch keys needed from CombinedHDF5Dataset
    trainable: bool                      # True only for partial/lpft/cotrain regimes
    fingerprint: dict                    # everything that changes the features (§8.4)
    def __call__(self, batch) -> StepFeatures: ...
```

There are two implementations, `VaeSource` and `Hdf5Source`, plus cache read/write functions. There is no base class.

### 8.2 `VaeSource`

**Resolving the binding** from `source.vae.package`:

| Package | Binding |
|---|---|
| `lag_attn_transformer_cfs` | `TRF_CFS_BINDING` |
| `lag_attn_cfs` | `CFS_BINDING` |
| `lag_attn_rws` | `RWS_BINDING` (via its own `load_task`) |
| `lag_attn_transformer_rws` | `TRF_BINDING` |
| `lag_slot_transformer_cfs` | `LAG_RESIDUAL_BINDING` |
| other `lag_attn*` families | `ModelBinding` built from `teb_vae.<pkg>.trainer.MODEL_CLS/TASK_CLS`. The implementer MUST check the `ModelBinding` fields; the class is defined twice, at `lag_attn_rws/eval/binding.py:28` and `lag_attn_cfs/eval/binding.py:54`. Add a contract test per family |
| `lag_attn` | raise `NotImplementedError` (v1) |

**Loader for online use.** The loader config for online regimes comes from the checkpoint's resolved config, via `pilot_loader_config` semantics: the same `stats_path`, `normalize_fields`, `trim_minutes` and model input fields. The classification shards are then substituted.

**Keys** (`source.vae.keys`). Each entry is `{name, transform: none|log1p|asinh, role: value|attention}`. Direct and derived keys:

| Name | Shape per segment | How |
|---|---|---|
| `mu_prior`, `mu_post`, `logvar_prior`, `logvar_post`, `target_state`, `source_state`, `kld_per_t`, `kld_per_t_per_head`, `source_kl_lag_map` | (T, ·) | direct output key |
| `attn_weights` | (T, 4·L) | flattened |
| `delta_mu` | (T, d_z) | `mu_post − mu_prior` |
| `kld_per_dim` | (T, d_z) | `model.kld_tensor(...)` |
| `attn_summary` | (T, 4·(2+n_bins)) | per head: expected lag, entropy, and mass in lag bins `source.vae.lag_bins` (default steps [0,4], [5,12], [13,24], [25, L−1]) |
| `kld_excess` | (T,) | `kld_per_t − kld_per_t(source_null)` via `controls.source_null_forward_outputs`. Removes the availability clock (§2.6). Costs a second forward pass |
| `nll_forecast` | (T,) | **not in v1.** It is known only H steps later and needs a lead shift. Deferred |

**Default keys [DECISION]:**
- `mu_prior` (value)
- `delta_mu` (value)
- `kld_per_t` (`transform: log1p`, `role: attention`)

This follows the evidence (§2.6) and [R: auxiliary VAE signals].

**Step mask** = AND of:
- `t ≥ warmup_period` (from `model_kwargs`);
- `weight[t] > 0`;
- `t < 270` if `source.vae.step_support: supervised`, or `t < T` if `causal_all` (**default**, since these steps are causal);
- `anchor_valid` scattered to T, for anchor-axis families.

**Slot family.** Values on the anchor axis are scattered to `T` at `anchor_index`. Unscattered steps are masked. Stride-1 dense anchors are required, and asserted.

**e2e family.** `required_fields` includes raw `fhr`, `up` and `weight`, via its task's `_build_forward_inputs`.

**Frozen vs trainable:**
- **Frozen:** `torch.no_grad()`, `model.eval()`.
- **Trainable:** `model.train()`, with grad enabled only on the parameters selected by `train.unfreeze` (§10.1). Dropout stays **off** in frozen modules: they are put in `eval()` individually.

### 8.3 `Hdf5Source` (ST / PH / raw)

- **Fields.** `source.hdf5.fields` is any subset of `fhr_st, fhr_ph, up_st, up_ph, fhr_up_ph`, plus `raw_fhr`, `raw_up` (reshaped `(300, 16)` per step).
- **Loading and normalisation.** Uses `stats_path` and `trim_minutes`. For causal builds, `emit_validity_mask=True`.
- **Masks:**
  - Step mask: `weight > 0`.
  - Per-channel cold-cell masks are applied by zeroing values. `step_mask` is additionally False where **all** channels are cold.
  - Optional `source.hdf5.min_step: auto` sets `step_mask = t ≥ max_c warmup_c` over the selected channels. This mirrors VAE warm-up and is the default for causal builds.
- **Channel counts** are read from the file and recorded in the fingerprint.
- This makes ST/PH a drop-in replacement for VAE latents: same grid, same downstream model.

### 8.4 Cache (for `frozen_cached`)

**Why and what:**
- Frozen features are identical across folds, because the VAE is shared. So extraction runs **once per unique segment** `(guid, epoch_s)` over the union of all folds and splits.
- Layout: `cache_root/<fingerprint_hash>/features.h5` with:
  - `values` `(N_unique, T', C)`, float16 by default;
  - `step_mask` `(N_unique, T')` bool;
  - optional `attn` `(N_unique, T', C_a)`;
  - `index.parquet`, row-aligned: `guid`, `epoch_s`, `source_file`, `ds_index` per (fold, split) occurrence.
- **Fingerprint** (JSON in the cache dir): source kind, checkpoint path and SHA-256, `model_kwargs` digest, keys and transforms, `step_support`, `time_pool`, stats file digest, trim, shard paths with size and mtime, and code version (git SHA).
  - A mismatch refuses to reuse the cache.
  - Rows are compared with `assert_same_keys`-style checks.
- **`time_pool = k`** (default 1). Masked mean over non-overlapping k-step windows; the pooled step is valid if any member is valid. This shrinks the cache by k×.
  - Size example: 50k segments × 300 × 129 ch × 2 B ≈ 3.9 GB at k = 1.
- Extraction MUST batch with `torch.no_grad()`, fp32 compute and fp16 storage, and show a progress bar.
- `run.py --stage extract` is resumable per chunk.

### 8.5 Normalisation

- **Per-channel scaler**, fitted per fold on **train-split valid steps only**, with recording weighting: each GUID's mean first, then the mean over GUIDs, as in `_hierarchical_moments`.
- Apply the `transform` (e.g. `log1p`) **before** fitting.
- **Scale floor:** `max(1e-3, 0.1·median positive std)`. A zero-variance channel logs a warning and is dropped from the model input; the drop is recorded.
- The scaler is saved as `scaler.json` next to the model and applied inside the dataset. For trainable sources it is refitted on train at the start of each training stage (§10.6).
- Context covariates have their own train-fold scaler (§7.3).

---

## 9. Model (`model.py`)

### 9.1 Scopes (`model.scope`) **[DECISION]**

- **`segment`:** the model maps one segment to a score. GUID offline and online scores are derived post hoc with fixed aggregators (§11.2).
  - Cheapest.
  - Gives clean segment-level metrics.
  - This is the configuration of the previous best simple baselines.
- **`sequence` (default):** the model maps the causal prefix of a GUID's segments to a score **at every position** n. It also produces a per-segment score from the same token (the segment-local head, §9.6).
  - Non-causal mode (`model.sequence.causal: false`) is allowed only with `labels.strategy ∈ {final_only, mil}`, and its online metrics are disabled.

### 9.2 Step encoder

- `x_t = step_proj(values_t)`: Linear(C → d_step) + LayerNorm + GELU + Dropout. `d_step = 64`.
- Optional `model.step.temporal`:
  - `none` (default);
  - `causal_conv`: depthwise-separable, kernel 5, dilations (1, 2, 4), residual;
  - `transformer`: 1 layer, causal, 2 heads, masked.
- Masked steps are zeroed after projection and excluded by every pooling op.

### 9.3 Segment pooling (`model.pooling.kind`)

| Kind | Formula | Notes |
|---|---|---|
| `mean` | masked mean of x_t | baseline |
| `mean_max` | [masked mean ‖ masked max] | robustness baseline [R: hierarchy] |
| `gated_attention` (**default**) | a_t ∝ exp(wᵀ(tanh(V u_t) ⊙ σ(U u_t))), e = Σ a_t x_t | Ilse 2018. **`u_t = [x_t ‖ attn cues_t]`**: KL and other `role: attention` channels shape the weights but are not pooled |
| `query` | single learned query, multi-head attention over steps (PMA, k = 1) | Set Transformer |
| `conjunctive` | ŷ = Σ a_t · f(x_t): per-step logits weighted by attention | MILLET (Early 2024). Segment scope only. Gives faithful per-step evidence for figures |

- A fully masked segment has no valid steps. It is excluded upstream (`low_valid_frac`), and the code asserts this never reaches the model.
- Attention weights are returned for figures (§11.10).

### 9.4 Context fusion (`context.fusion`)

The segment token is `z = MLP([e ‖ c])`, where `c` is the context vector (§7.1).

| Mode | Behaviour |
|---|---|
| `concat` | c is concatenated before the token MLP |
| `film` | `z = γ(c) ⊙ MLP(e) + β(c)` (Perez 2018) |
| `token` | sequence scope only. The covariate block becomes an extra per-position input added to the token |
| `late` (**default for covariates**) | c is concatenated only at the head input; least prone to overfitting [R: covariates] |

- **Time and stage** features always use `concat`.
- **Clinical covariates** use the configured mode.
- Token size: `d_token = 128`, with dropout 0.1.

### 9.5 Sequence aggregator (`model.sequence.kind`, sequence scope)

| Kind | Description |
|---|---|
| `causal_transformer` (**default**) | Pre-norm, 2 layers, d = 128, 4 heads, FFN 256, dropout 0.1–0.2. **Key-padding mask** for absent or padded segments, plus a **causal mask**. **Relative-time bias:** a per-head learned bias `β_h[b(i,j)]`, where `b = bucket(|t_end_i − t_end_j|)`, log-bucketed with 32 buckets saturating at 12 h (this worked in the previous classifier). It uses relative times only, never absolute time to delivery. Output: state h_n per position |
| `gru` | GRU over tokens with an input `[z_n ‖ log1p(Δt_n)]` and GRU-D-style decay `h ← h·exp(−relu(w)·Δt)` before each step |
| `attention_mil` | Gated-attention pooling over the prefix: position n pools tokens ≤ n with a causal mask (cumulative masked softmax). Non-causal when `causal: false` |

- A mask that is fully masked because of padding MUST NOT produce NaN. Guard the softmax, and assert this in tests.
- Online outputs MUST equal an explicit prefix sweep (test T-M2).

### 9.6 Heads

- **Per-position GUID head** (sequence scope) and **per-segment head** (segment scope): MLP(d → 64 → K_out) with dropout 0.1.
  - The last layer is zero-initialised.
  - The bias is initialised from the train-fold prior: `logit(π)` for binary, `log π_k` for multiclass, cumulative logits for ordinal.
- **Output types:**
  - `binary`: 1 logit.
  - `multiclass`: K logits, softmax.
  - `ordinal` (CORAL, Cao 2020): shared score g(x) plus K−1 ordered biases, `P(Y > k) = σ(g + b_k)`. The biases are kept ordered by a cumulative-softplus parametrisation. The alarm score is `g` (equivalently `P(Y ≥ 1)`), a single ranking for all cut-points [R: ordinal].
- **Segment-local head** (sequence scope, optional, `model.segment_head: true`): a small MLP on z_n *before* the aggregator. It gives true segment-level scores from the same model. Its loss weight is `train.loss_weights.segment`.
- **Multi-task:** a binary main head plus a 3-class auxiliary head (λ₃), sharing the trunk.

### 9.7 Size guidance

Target trunk ≤ 0.5 M parameters for frozen regimes. Over-parameterising the aggregator is the main overfitting risk at this sample size [R: hierarchy].

---

## 10. Training (`train.py`, `losses.py`, `baselines.py`)

### 10.1 Regimes (`train.regime`) **[DECISION]**

| Regime | Encoder | Data path | Notes |
|---|---|---|---|
| `frozen_cached` (**default**) | not run | cache (§8.4) | Fastest. All ablations of heads, labels and losses use it |
| `frozen_online` | eval, no_grad | HDF5 → source each batch | Needed for `kld_excess` without caching. Also enables `z` sampling as augmentation (`source.vae.sample_z_train: true`: sample `z_post` instead of `mu_post` during training only) |
| `partial` | params matching `train.unfreeze` prefixes trainable; the rest frozen, in `eval()` | online | Discriminative LR: `train.backbone_lr` (default 1e-5) vs head `lr`. Examples: `["posterior_head.delta_mu_head"]` (pilot), `["target_encoder.blocks.5"]` (top block), `["prior_head"]` |
| `lpft` | stage 1: frozen (head only, `lpft_head_epochs`); stage 2: `partial` with the stage-1 head | cache → online | Kumar 2022. Stage 2 resets early stopping. Stage 1 may run on the cache |
| `cotrain` | `train.unfreeze` params trainable **and** the VAE objective is optimised jointly | online | §10.6. The fragile end of the ladder [R: Freeze first] |

**Rules:**
- Every non-frozen regime MUST also produce the frozen baseline at the same seed, i.e. epoch-0 evaluation. The report shows Δ vs frozen.
- The parameter allowlist for trainable modules MUST be explicit.
- Optimizer groups are built from the allowlist only.
- A pre-step check asserts that no other parameter has `requires_grad`, and that frozen modules are in `eval()`. This follows the pilot's guard (`latent_pilot/model.py:465-590`).

### 10.2 Losses (`train.loss.name`)

With p = σ(s), y ∈ {0, 1}, and w_y the class weights:

| Name | Formula | Use |
|---|---|---|
| `bce` (**default**) | −[y log p + (1−y) log(1−p)] | calibration-friendly [R: Plain BCE] |
| `weighted_bce` | w_y · bce, with w from `train.loss.weighting` | |
| `focal` | −α_t (1−p_t)^γ log p_t; γ = 2; **α must be set explicitly** (torchvision's 0.25 down-weights positives) | ablation |
| `logit_adjusted` | bce on s + τ·log(π₁/π₀) | equivalent to a threshold shift for binary; ablation only |
| `auc_margin` | AUC-M square surrogate (Yuan 2021), copied (~40 lines) | fine-tune stage after BCE; needs positives per batch → `sampler: class_balanced` |
| `pauc` | one-way pAUC surrogate over FPR ≤ `eval` α (Zhu 2022), copied | same as above |
| `ce` / `weighted_ce` / `focal_ce` | multiclass analogues | 3-class |
| `coral` | Σ_k BCE(1[y > k], σ(g + b_k)) | ordinal |
| `cumulative_link` | proportional-odds negative log-likelihood | ordinal alternative |

**Supporting options:**
- **`train.loss.weighting`:** `none` (default), `inverse`, `sqrt_inverse`, or `effective_number` (β = 0.999; Cui 2019).
  - Weights are computed at **GUID level** on the train fold.
  - For 3-class, **`sqrt_inverse` is the recommended default**, because HIE is rare.
- **Label smoothing** (`train.loss.label_smoothing`, default 0). It floors predicted risk, so use it with care.
- **Prior correction after reweighting.** If any weighting or `logit_adjusted` is used, predictions are corrected before calibration with `s ← s − log(w₁/w₀)` (binary) or the vector analogue (Saerens 2002). Calibration is then fit (§10.8).

### 10.3 Loss composition

**Sequence scope:**

```
L = λ_final · mean_g ℓ(y_g, s_g(N_g))
  + λ_pos   · mean_g [ Σ_n ω_n ℓ(y_n, s_g(n)) / Σ_n ω_n ]      (strategy weights, k_warm applied)
  + λ_bag   · mean_g ℓ(y_g, s_bag,g)                            (strategy = mil, or λ_bag > 0)
  + λ_seg   · mean_g [ Σ_n ω_n ℓ(y_n, s^seg_n) / Σ_n ω_n ]      (segment-local head)
  + λ_3     · (same terms for the 3-class auxiliary head)
```

- Defaults: `train.loss_weights = {final: 1.0, positions: 0.5, bag: 0.0, segment: 0.3}`, `λ_3 = labels.aux_3class_weight`.
- **Segment scope:** only the `positions` term applies, over segments (s_g(n) is replaced by s_n).

### 10.4 Batching and sampling

- **Sequence scope:** one item is one GUID (all retained segments of the split, ordered).
  - Batches are length-bucketed with `VariableBatchBucketSampler` (`train.batch_guids` is the base size).
  - Collate right-pads to N_max with `seg_mask`.
- **Segment scope:** one item is one segment. Batch size `train.batch_segments` (default 256).
- **`train.sampler`:**
  - `natural` (default): train/val are about 1:1 at GUID level by construction (§2.5).
  - `class_balanced`: GUID-level balanced draws; required for `auc_margin` and `pauc`.
  - Using `class_balanced` together with `weighting ≠ none` logs a WARNING (double correction).
- **Segment dropout** (sequence scope, `train.segment_dropout`, default 0.1): drop non-final tokens during training, then recompute Δt and relative-time buckets from the remaining `t_end` values. The previous classifier had this bug and fixed it.

### 10.5 Optimisation and selection

- **Optimizer.** AdamW, lr 1e-3, betas (0.9, 0.999), weight decay 1e-2 **excluding** biases, norms, embeddings and the head bias.
- **Schedule.** Linear warm-up of 200 steps, then cosine to 0.05·lr. Max 150 epochs. Grad clip 1.0 by norm. fp32 for frozen regimes; bf16 autocast MAY be enabled for online regimes.
- **EMA of weights** (`train.ema`, default 0.999; `null` disables). Implemented by Lightning's `EMAWeightAveraging` callback (§10.10.2). Used for validation and the final checkpoint.
- **Early stopping** (`advanced_config.callbacks.early_stopping`, built by `GraphModelBase`, §10.10.2):
  - Monitor: `val/guid_logloss` (default: GUID-level, final-position, unweighted, after prior correction) or `val/guid_auroc`.
  - Patience 25. Tie-break: lower val loss, then the earlier epoch.
  - **Do not** select on sensitivity-at-FPR or AUPRC. They swing by several points per epoch at about 20–100 validation positives [R: Plain BCE].
- **Checkpoint.** The best EMA weights are saved as `model_checkpoints/best.ckpt` (`ModelCheckpoint(filename="best")`, §10.10.2). Gradient clipping is `advanced_config.trainer.gradient_clip_val`. Predictions are always made from `best.ckpt`, never from last-epoch weights. The previous v0 classifier had this bug.

### 10.6 Co-training specifics

**Objective:**
- The joint loss is `λ_cls·L_cls + λ_vae·L_vae + λ_sp·‖θ − θ₀‖²` over trainable parameters (L2-SP anchor to the pretrained weights).
- `L_vae` is the VAE task's own `compute_loss_and_metrics` on the same batch of segments, run in its **training** geometry (tiled anchors).
- Implementation: subclass the family's task, capture the model outputs with a forward hook during `super().compute_loss_and_metrics`, and reuse the dense latents for the classifier. Do **not** run the model twice.

**Required pinning:**
- β schedule `constant` at the pretrained final value. A warm start resets `current_epoch`, and the β warm-up would restart at 0 (`lag_attn_rws/task.py:469-513`).
- The LR warm-up of the VAE task is disabled. The classifier's schedule governs everything.

**Batching (sequence scope):**
- All segments of the batch's GUIDs are flattened into VAE sub-batches (`train.cotrain.vae_chunk`, default 32) and scattered back.
- `train.cotrain.grad_segments: all | last_k:<k>`. `last_k` runs the other segments under no_grad to cap memory.
  - `ponytail:` crude memory cap; replace it with gradient checkpointing if long GUIDs matter.

**Stop-gradient option.** `train.cotrain.detach_head_input: true` means the classifier reads `sg(features)` through its own trainable layers, while the VAE still trains on `L_vae`. This keeps the generative model interpretable [R: Freeze first].

**Preservation monitoring**, adapted from the pilot, runs each epoch on a fixed validation subset:
- forecast MSE (mean-decoded) relative to the frozen baseline;
- `kld_active_frac`, `logvar_prior_floor_frac`, `delta_mu_sat_frac`.

The **gate**: an epoch is selectable only if relative forecast MSE ≤ `train.cotrain.gates.forecast_mse_rel` (default +10%). Epoch 0 (frozen) is always a candidate.

**Data rules:**
- Labels come only from the classification shards of the fold's **train** split.
- Unlabeled or excluded rows are masked by multiplying by zero, not by Python branching. This matters under DDP with `find_unused_parameters=False`.

**Refit.** The scaler is refitted on train at the start of each training stage, because features drift.

### 10.7 Seeds and ensembles

- `run.seeds` (default `[42]`). Per fold, the seed is `seed + 1000·fold`.
- With more than one seed, the **ensemble** takes the mean of **uncalibrated logits** across seeds, then calibrates once on val (§10.8).
- Per-seed predictions are also saved. The report shows the per-seed spread.

### 10.8 Calibration (`calibration.method`)

| Method | Details |
|---|---|
| `temperature` (**default**) | Binary and multiclass: fit T > 0 by LBFGS on val NLL (prior-corrected logits). Ordinal: scale g and refit the K−1 offsets |
| `platt` | Binary: a, b on the logit |
| `none` | |

- Calibration is fit **per fold** on **GUID-level final-position val predictions**. The fitted map is also applied to segment and online scores.
- Isotonic calibration is **not** offered. There are too few positives per fold for the 3-class tasks.
- The calibrated probability is calibrated to **val prevalence**. See §11.7 for the prevalence shift on test.

### 10.9 Mandatory baselines and controls (`baselines.py`, FR-12)

These run automatically per fold with the same cohort, splits and evaluation engine:

1. **Linear probe.** `sklearn.linear_model.LogisticRegression` (L2, C chosen on val from {0.01, 0.1, 1}) on per-GUID features:
   - the masked mean of the source channels over the GUID's last `baselines.probe_last_n` segments (default 3), plus `mean_max`;
   - one GUID-level score, plus per-segment scores from the same probe applied to single segments.
   - Every learned model is reported **against this floor**.
2. **Shortcut baseline.** Logistic regression on metadata only: `n_segments`, recording span, `has_tlo`, `has_ss`, and mean `valid_frac`. **These are never model inputs.**
   - If its val AUROC exceeds `baselines.shortcut_warn_auroc` (default 0.60), the report prints a prominent warning.
   - That would mean cohort construction lets the models exploit length or missingness shortcuts (§2.5).
3. **Shuffled-label control.** The main model trained with GUID labels permuted within the train split (one seed, frozen regime). Expected AUROC is about 0.5. A value > 0.6 is flagged as a pipeline leak.

### 10.10 Training framework integration (`train/`), callbacks, monitoring and MLflow **[DECISION]**

The classifier trains through the **same framework as every VAE family**:
- `train/graph_model_base.py` (`GraphModelBase`);
- `train/pl_model_base.py` (`LightningModelBase`);
- `train/callbacks.py`;
- `utils/custom_logger.py`;
- MLflow via `MLFlowLogger`.

The reference pattern is the `lag_attn_rws` family root: `teb_vae/lag_attn_rws/trainer.py:155` and `task.py:56`. Deviations are listed explicitly in §10.10.7.

#### 10.10.1 Classes

**`ClassifierTask(LightningModelBase)`** (`train.py`) is the Lightning module.

- **Constructor.** `ClassifierTask(base_model, *, lr, weight_decay, spike_breaker, compile_model=False, classifier_kwargs, ...)`.
  - All constructor kwargs are JSON-able scalars or dicts, because they become hparams and MLflow params (`save_hyperparameters(ignore=['base_model'])`, `pl_model_base.py:91`).
  - `compile_model` is **always False**, because GUID lengths vary.
- **Implements `compute_loss_and_metrics(batch, batch_idx, stage) -> (loss, metrics)`** (`pl_model_base.py:171` contract).
  - `loss` is `total_loss` (§10.3).
  - `metrics` holds the unprefixed names below. The base prefixes them `stage/…` and logs them `on_epoch` (and `on_step` for train) (`_log_metrics`, `pl_model_base.py:664-675`).
    - `total_loss`, `loss_final`, `loss_positions`, `loss_bag`, `loss_segment`, `loss_aux3`: each component, 0 when disabled;
    - `acc_bin` (batch accuracy at p = 0.5, monitoring only);
    - `mean_num_segments`, `frac_masked_steps`;
    - `attn_entropy` and `attn_max_weight` (step pooling, and sequence attention when present);
    - `logit_mean_pos`, `logit_mean_neg`;
    - `head_bias`.
  - Also returns an unprefixed `main_loss` equal to `total_loss`, which the spike breaker watches (`pl_model_base.py:384`).
- **Overrides:**
  - `configure_param_groups()`: two groups, decay and no-decay. No-decay holds biases, norms, embeddings and the head bias. Trainable-backbone regimes add a third group at `train.backbone_lr`.
  - `build_optimizer()`: AdamW with betas **(0.9, 0.999)**. The base hard-codes (0.9, 0.95) (`pl_model_base.py:275-302`).
  - `build_lr_scheduler()`: **per-step** linear warm-up plus cosine, returned with `interval: "step"`. Steps come from `self.trainer.estimated_stepping_batches`. Copy the pattern of `teb_vae/lag_attn_transformer_rws/task.py:43`. The base scheduler is epoch-wise MultiStepLR.
  - `on_before_optimizer_step`: logs `train/grad_norm` (fp32, pre-clip) and `train/grad_clip_frac` every 25 steps and on the last batch. Copy `teb_vae/lag_attn_rws/task.py:216-301`.
  - `on_train_epoch_end`: logs `train/weight_norm` (one `self.log`).
  - `on_save_checkpoint(ckpt)`: call `super()`, which stamps `model_class`. Then add:
    - `classifier_kwargs`, needed to rebuild the model without a config;
    - `source_fingerprint` (§8.4), the `scaler` payload and the task/labels config;
    - `feature_channels`.
  - `on_validation_epoch_start`: clears the prediction buffers.
  - `validation_step`: calls super, then appends per-GUID final scores and labels, per-segment scores, and (3-class) probabilities to `self.val_buffer` for the epoch callback (§10.10.2).
- **Naming constraint.** Submodules must **not** be named `model`, `net`, `network` or `module`, because `train/graph_models_utils.py:40-76 _clean_state_dict` strips those prefixes in a loop. Use names such as `step_encoder`, `pooling`, `aggregator`, `head`, `head_aux3`.

**`ClassifierTrainer(GraphModelBase)`** (`train.py`): one instance per **unit** = (fold, seed, kind), with kind ∈ {`model`, `shuffled`}. Baselines are sklearn and run outside Lightning.

- **Class attributes, mirroring the families:**
  - `TASK_CLS = ClassifierTask`
  - `CHECKPOINT_STEM = "classifier"`
  - `TRACKED_METRICS` (§10.10.3)
  - `PLOT_CONFIG_KEY = "classifier_plotting"`
- **`__init__(unit_config_path, *, unit_dir)`:**
  - Call `super().__init__`.
  - Then re-point `base_folder`, `output_base_dir`, `train_results_dir`, `test_results_dir` and `model_checkpoint_dir` to `<run>/folds/fold_<k>/seed_<s>[/shuffled]/`.
  - This avoids the base's minute-resolution run stamp (`_resolve_run_stamp`, `graph_model_base.py:70-101`), under which units started in the same minute collide. The re-pointing follows `train/test_utils.py:169 make_graph_model`.
- **`create_model()`:**
  - Build the network (`model.py`) and `ClassifierTask`.
  - For trainable regimes, also build the `VaeSource` model via its binding (§8.2).
  - Call `apply_config_hyperparameters({"lr": …})` (`graph_model_base.py:561`).
- **`train_model()`:**
  - Build the loaders: `GuidDataset` + `VariableBatchBucketSampler`, or `SegmentDataset`.
  - Build the callbacks (§10.10.2), then `self.build_trainer(callbacks)` and `fit`.
  - Load `best.ckpt` and write `fold_results.json`: best path, best score, and `trainer.validate(ckpt_path=best)` metrics.
- **`_build_trainer_kwargs()` override:**
  - `LearningRateMonitor(logging_interval="step")`.
  - Honour `run.device: cpu`. The base picks the accelerator from `torch.cuda.is_available()` only (`graph_model_base.py:507-513`).
  - `use_distributed_sampler=False`, because the batch sampler is custom.
- **Reused unchanged:** `setup_config`, `validate_config`, `configure_determinism`, `_init_mlflow_logger`, `_log_run_metadata_to_mlflow`, provenance tags, `build_trainer`, `upload_run_logs`.
- **Do not use `train/data_module.py GraphDataModule`.** It assumes VAE shards (`dataset_config.vae_*_datasets`).

**Outer loop** (`run.py`, the `train` stage, sequential over units). Per unit:
1. Derive the framework-shaped unit config (§13.2) and write it to `seed_<s>/model_checkpoints/resolved_config.yaml`.
2. `ClassifierTrainer(...)` → `setup_config()` → `create_model()` → `train_model()`.
3. **Teardown.** The base leaks across units within one process:
   - call `gm.upload_run_logs()` explicitly, then `atexit.unregister(gm.upload_run_logs)`;
   - `gm._system_metrics_monitor.finish()` if one exists;
   - `del gm.pl_model`, `gc.collect()`, `torch.cuda.empty_cache()`;
   - re-run `setup_logging` for the run-level `run.log`, because `setup_logging` replaces all sinks.
4. Fit calibration and thresholds on val (§10.8, §11.3), then write `selection_lock.json`.
5. Append one line to the run-level `kfold_progress.log`: `ts | fold | seed | kind | status | train_s | best_epoch | best_<monitor> | val/guid_auroc | error`.

A unit failure is recorded (status `failed` with a traceback in `fold_results.json`). With `run.fail_fast: true` (default false) the run stops; otherwise the remaining units continue. `evaluate` refuses to pool a run with failed units unless `--allow-partial`, and the summary lists the missing units.

#### 10.10.2 Callbacks, in registration order

**Order matters.** In Lightning 2.6.5, callback `on_validation_epoch_end` runs **before** the module's (`lightning/pytorch/loops/evaluation_loop.py:378-379`). A metric logged in the module's own epoch-end hook would reach `MetricsHistoryCsvCallback` and `LossPlotCallback` one epoch late. GUID-level validation metrics are therefore computed in a callback registered **first**.

| # | Callback | Source | Enabled by | Output |
|---|---|---|---|---|
| 1 | **`GuidEpochMetricsCallback`** (new) | `train.py` | always | Reads `pl_module.val_buffer` and logs via `pl_module.log(..., on_epoch=True)` (see below). Appends one JSON line per epoch to `train_results/epoch_summary.jsonl` |
| 2 | `MetricsLoggingCallback(TRACKED_METRICS)` + `MetricsHistoryCsvCallback` | `train/callbacks.py:290, 329` | always | `train_results/metrics_history.csv`, rewritten every validation epoch. Rank 0, skips sanity. This replaces the earlier spec name `train_history.csv` |
| 3 | `LossPlotCallback(output_dir, plot_frequency, metric_filters=("*/total_loss", "*/loss_*", "val/guid_*"), mlflow_logger=…)` | `train/callbacks.py:43` | always | `train_results/loss_plot_epoch.html` (uploaded) |
| 4 | `HyperparameterLoggingCallback(tracked_keys=("lr",), …)` | `train/callbacks.py:199` | always | `train_results/hyperparameters.html` |
| 5 | **`ClassifierPlotCallback`** (new) | `train.py`, modelled on `teb_vae/lag_attn_rws/plotting.py:258` | `advanced_config.callbacks.classifier_plotting.{enabled, every_n_epochs, file_format}` | Every N epochs, rank 0, skips sanity, wrapped in try/except with a logged warning: val ROC + PR (GUID final score); score histograms by class; reliability diagram; confusion at the val-quantile FPR-cap threshold; pooling-attention entropy histogram. Written to `train_results/classifier_diagnostics/epoch{E:04d}_{name}.pdf` and uploaded via `log_artifact_to_mlflow` |
| 6 | `ModelCheckpoint(dirpath=model_checkpoints, filename="best", save_top_k=1, monitor=<early_stopping.monitor>, mode=<mode>, auto_insert_metric_name=False, save_last=advanced_config.callbacks.model_checkpoint.save_last)` | Lightning | always | `model_checkpoints/best.ckpt` (+ `last.ckpt`). Predictions always load `best_model_path` (A6) |
| 7 | `EarlyStopping` | built by the base from `advanced_config.callbacks.early_stopping` (`graph_model_base.py:442-449`) | config | — |
| 8 | `LearningRateMonitor("step")` | base (overridden to step) | always | `lr-AdamW[/pg<i>]` |
| 9 | `EMAWeightAveraging(decay=train.ema)` | Lightning 2.6.5 (`lightning/pytorch/callbacks/weight_averaging.py:366`) | `train.ema` not null | EMA weights are used for validation and written into the checkpoint. No custom EMA code |
| 10 | `MLflowRunLoggingCallback` | base, when MLflow is on (`train/callbacks.py:412`) | `tracking.mlflow.enabled` | Architecture text and parameter counts. **`log_model: false`** in unit configs, otherwise every unit registers a model version |
| 11 | `LpftUnfreezeCallback` (P7) | `train.py`, modelled on the old `TwoStageVaeUnfreeze` | `train.regime: lpft` | At `lpft_head_epochs`: unfreeze the allowlist, `optimizer.add_param_group`, reset EarlyStopping (`wait_count = 0`, `best_score` reset), log `train/stage = 2`. Appends to `train_results/stage_transitions.jsonl` (epoch, n_params trainable/frozen, prefixes, backbone_lr, iso time) |
| 12 | `PreservationGateCallback` (P7) | `train.py` | `train.regime: cotrain` | Per epoch on the fixed validation subset: logs `val/forecast_mse_rel`, `val/kld_active_frac`, `val/logvar_prior_floor_frac`, `val/delta_mu_sat_frac`. Sets `val/gate_ok ∈ {0, 1}`. The checkpoint monitor becomes `val/guid_logloss_gated` (= +∞ when the gate fails) |

**What `GuidEpochMetricsCallback` logs.** The same formulas as the evaluation engine, imported from `metrics.py` so training and evaluation can never disagree:

| Metric | Definition |
|---|---|
| `val/guid_logloss` | the early-stopping monitor (§10.5) |
| `val/guid_auroc`, `val/guid_auprc`, `val/guid_pauc30` | McClish pAUC at FPR ≤ 0.3 |
| `val/guid_brier`, `val/guid_ece` | ECE with equal-mass bins |
| `val/seg_auroc` | on the time-matched eval window |
| `val/sens_at_fpr30` | empirical, at the validation quantile. **Monitoring only**, never used for selection |
| `val/score_mean_pos`, `val/score_mean_neg`, `val/prevalence` | |
| 3-class | `val/macro_f1`, `val/{precision,recall,f1}_class{c}`, `val/auroc_ovr_class{c}`, `val/auroc_macro` |
| online | `val/online_auroc_last` (last position) and `val/online_auroc_pos{k}` for k ∈ {1, 3, 5}: early-position AUROC, which detects the old prior-collapse "plateau" (§3.3) |
| overfitting monitor | every `advanced_config.callbacks.classifier_plotting.train_eval_every` epochs: the same GUID metrics on a fixed, seeded train subset of `train_eval_guids` GUIDs (default 256), logged as `train/guid_auroc` and `train/guid_logloss` |

Each `epoch_summary.jsonl` line holds:
- `epoch`, `global_step`;
- all the scalars above;
- `confusion_bin@0.5`, `confusion_bin@q30` (the validation-quantile threshold), and `confusion_3class` (argmax);
- `support` per class, `n_val_guids`;
- `grad_norm_mean_epoch`, `grad_norm_max_epoch` (from task buffers), `lr`;
- `stage` (lpft/cotrain).

#### 10.10.3 `TRACKED_METRICS` (columns of `metrics_history.csv`)

```
train/{total_loss, loss_final, loss_positions, loss_bag, loss_segment, loss_aux3, acc_bin,
       grad_norm, grad_clip_frac, weight_norm, attn_entropy, mean_num_segments, guid_auroc, guid_logloss}
val/{total_loss, loss_final, loss_positions, loss_bag, loss_segment, loss_aux3,
     guid_logloss, guid_auroc, guid_auprc, guid_pauc30, guid_brier, guid_ece, seg_auroc, sens_at_fpr30,
     online_auroc_last, online_auroc_pos1, online_auroc_pos3, score_mean_pos, score_mean_neg}
lr
# 3-class adds: val/{macro_f1, auroc_macro, recall_class0..2}
# spike breaker (when enabled): train/{spike_skipped, spike_ema_loss}
# cotrain adds: train/{vae_total_loss, vae_kld, vae_nll}, val/{forecast_mse_rel, kld_active_frac, logvar_prior_floor_frac, gate_ok}
```

**Documented quirk.** `MetricsLoggingCallback` reads during validation, before the train epoch is reduced. So `train/*` cells in the CSV are the **last-step** values, while MLflow holds the epoch means as `train/x_epoch`. The same quirk exists in every family (`teb_vae/lag_attn_rws/task.py:236-244`). The loss plots therefore use MLflow/epoch keys where available.

The train smoke test (T-F2) asserts that every `TRACKED_METRICS` name is present and not all-NaN, as the families do (`teb_vae/lag_attn_transformer_cfs/tests/test_train_smoke.py:283-296`).

#### 10.10.4 Files per unit and per run

**Per unit, `folds/fold_<k>/seed_<s>/`:**

| Location | Contents |
|---|---|
| `train_results/` | `full.log` (+ `.rank<N>`), `run.jsonl` (loguru JSON sink), `metrics_history.csv`, `epoch_summary.jsonl`, `loss_plot_epoch.html`, `hyperparameters.html`, `classifier_diagnostics/*.pdf`, `stage_transitions.jsonl` (P7), profiler output (`advanced_config.trainer.profiler`, default `null` for the classifier) |
| `model_checkpoints/` | `best.ckpt`, optionally `last.ckpt`, `resolved_config.yaml` (the unit's framework-shaped config) |
| `setup.json` | n GUIDs and segments per split and class; train priors; class weights; head-bias init; source fingerprint and cache path; feature channels and dropped channels; optimizer, scheduler, trainer and dataloader settings; VAE checkpoint SHA-256; parameter counts (trainable/frozen) |
| other | `scaler.json`, `calibration.json`, `thresholds.json`, `selection_lock.json`, `fold_results.json` |

**Per run:**
- `kfold_progress.log`;
- `kfold_summary.json`: per-unit best metrics, and the mean/SD/min/max across folds of the best validation metrics;
- `execution_metadata.json`: git SHA and dirty flag, python, platform, CUDA, devices, fold and seed ids, wall time per stage.

#### 10.10.5 MLflow design (folds × seeds)

- **Parent run** (`run.py`, via `mlflow.MlflowClient`):
  - Created once per run directory, **fail-closed**: if the server is unreachable, log a WARNING and set `enabled: false` for every unit and for evaluate.
  - Name: `run.name`.
  - Tags: the same provenance as `GraphModelBase._collect_provenance_tags` (`graph_model_base.py:769-814`), plus `kind=parent`, `task`, `source_kind`, `vae_package`, `vae_ckpt_sha`, `n_folds`, `seeds`.
  - Params: the flattened `classifier:` block.
  - Artifacts: `config.resolved.yaml`.
- **Child run per unit.** The derived unit config sets `advanced_config.tracking.mlflow`:
  - `run_name: fold<k>-seed<s>-<kind>`;
  - `tags: {mlflow.parentRunId: <parent id>, fold: k, seed: s, kind: model|shuffled}`;
  - `log_model: false`, `log_checkpoints: false`.

  `MLFlowLogger(tags=…)` then creates a true nested run with **no base-class change**.
- **Params per unit.** The base flattens only `general_config` and `model_config` into params (`_log_run_metadata_to_mlflow`, `graph_model_base.py:816-853`). So `config.py` copies `classifier:` to `model_config.classifier` in the unit config, and every classifier key becomes a child param.
- **Per-unit metrics** are logged automatically by Lightning (`self.log`), with epoch means as `train/x_epoch`. After calibration and threshold selection, the unit also logs: `val/thr_<policy>`, `val/sens_<policy>`, `val/fpr_<policy>`, `calib_temperature`.
- **Evaluate and report log to the parent**, using `MlflowClient.log_metric` / `log_artifact` directly. `utils/mlflow_utils.log_artifact_to_mlflow` is a **no-op without a trainer** (`utils/mlflow_utils.py:19-37`), so it cannot be used here. The parent receives:
  - headline metrics as `pooled_test/<level>/<metric>[@<policy>]` and `foldmean_test/…` (e.g. `pooled_test/guid/auroc`, `pooled_test/guid/sens@np30`, `pooled_test/guid/fpr@np30`);
  - baselines as `probe/…`, `shortcut/…`, `shuffled/…`;
  - `summary.md`, `summary.json`, `evaluation/tables/metrics.parquet` and `evaluation/figures/**` as artifacts (figures under `figures/`).
- **Logs.** `full.log` and `run.jsonl` of each unit are uploaded by the base's `upload_run_logs`, called explicitly at teardown (§10.10.1).
- **`compare`** creates no MLflow run. It writes to the directory given by `--out`.

#### 10.10.6 Logging

- `utils/custom_logger.setup_logging` is used for every unit (per-unit sinks in `train_results/`) and for the run-level `run.log` / `run.jsonl`.
- Every stage logs start and end with durations. Every refusal (leakage guard, missing lock, digest mismatch) logs an ERROR naming the guard ID (§12) before raising.
- `stage_state.json` records per-stage, per-unit status (`pending | running | done | failed`) with timestamps. The `--stage all` resume logic reads it.

#### 10.10.7 Framework deviations and gotchas (must be handled; covered by tests T-F1…T-F5)

| # | Base behaviour | Classifier handling |
|---|---|---|
| F1 | `torch.compile` on by default | `compile_model=False` always |
| F2 | AdamW betas (0.9, 0.95); one weight decay for all params | Override `build_optimizer` and `configure_param_groups` |
| F3 | Epoch-wise MultiStepLR | Per-step warm-up + cosine override |
| F4 | Callback epoch-end runs before the module's | `GuidEpochMetricsCallback` first (§10.10.2) |
| F5 | CPU cannot be forced from config | Override `_build_trainer_kwargs` |
| F6 | Minute-resolution run stamp; directory collisions | Re-point directories per unit |
| F7 | `atexit` upload keeps each driver (and its GPU model) alive; `SystemMetricsMonitor` never stopped; `setup_logging` replaces sinks | Explicit teardown per unit |
| F8 | `log_artifact_to_mlflow` is a no-op without a trainer | Parent logging via `MlflowClient` |
| F9 | MLflow server down → silently continues without logging | Parent fail-closed; the status is written to `manifest.json` and `summary.md` |
| F10 | `MLflowRunLoggingCallback` registers a model version when `log_model` is true | `log_model: false` in unit configs |
| F11 | `_clean_state_dict` strips `model.`/`net.`/`network.`/`module.` prefixes | Submodule naming rule (§10.10.1) |
| F12 | `validate_config` warns only for unknown keys under `advanced_config.trainer` / `advanced_config` | The strict pydantic `classifier:` schema covers the rest. The config-load test asserts no `config:` warnings (T-F1) |
| F13 | `GraphDataModule` is VAE-specific | Own loaders |
| F14 | `train/*` CSV cells are last-step values | Documented; plots use epoch means |

---

## 11. Evaluation and analysis (`thresholds.py`, `metrics.py`, `report.py`)

Evaluation reads **only** the prediction tables. It can be re-run without retraining: `--stage evaluate`, `--stage report`.

The big picture is in §5.5. In short:
1. **Predictions (§11.1)** are turned into **score views (§11.2)**.
2. **Thresholds** are chosen on validation per policy and basis (§11.3).
3. Every cell of the analysis grid is computed: level × metric type (threshold-free, instantaneous, committed cumulative, committed overall) × time axis and point × policy × task/head × subgroup, for val and test, per fold (§11.4–§11.6).
4. Results are aggregated per fold and pooled with CIs (§11.7), then compared (§11.8), stored (§11.9) and rendered (§11.10–§11.11).

### 11.1 Prediction tables (FR-9)

**`predictions/segments.parquet`** has one row per (run, model, seed | `ens`, fold, split, segment).

| Group | Columns |
|---|---|
| keys | `run_id, model_id, seed, fold, split, guid, seg_pos, slot, epoch_s, t_end_s` |
| eval-only clocks | `hours_to_delivery, tlo_end_h, ss_rel_h, stage` |
| labels | `class_code, y` (task target), `in_eval_window`, `label_weight` (ω_n) |
| strata (never inputs) | `cs, bg, clinical_class` (name), `subgroup` (shard basename), `has_tlo, has_ss, shared_test`, plus the §11.6.1 family memberships computed at evaluate time |
| scores | `logit_seg, p_seg, p_seg_cal` (segment head, or the segment-scope model) |
| online scores (sequence scope) | `logit_online, p_online, p_online_cal`: the GUID score after this position; `p_online_runmax_cal`: running max |
| 3-class (when present) | `p_c0, p_c1, p_c2` (+ `_cal`), `ord_score` |
| quality | `valid_frac, n_valid_steps` |

**`predictions/guids.parquet`** has one row per (run, model, seed | ens, fold, split, guid).
- Keys, labels and strata as above.
- `n_segments`, `last_t_end_s`.
- Scores per aggregator: `score_final` (sequence final position), `score_max`, `score_mean`, `score_lse`, `score_last` (segment scope), each raw and calibrated.
- 3-class probabilities.
- For each threshold policy: `pred_<policy>`, `first_alarm_t_s_<policy>`, `lead_time_h_<policy>`.

**Other files:**
- **`thresholds.json`** per fold: `{level: {policy_id: {threshold, val_sens, val_fpr, val_spec, n_pos, n_neg, k (NP order statistic), method}}}`.
- **`calibration.json`** per fold.

Pandas string columns MUST be explicit `str` (pandas 3).

### 11.2 Score definitions **[DECISION]**

| Level | Score | Notes |
|---|---|---|
| **Segment** | `p_seg`: the segment-local score | segment scope, or the segment head in sequence scope |
| **GUID offline, sequence scope** | `score_final` = online score at the last position | |
| **GUID offline, segment scope** | aggregator over the GUID's segments: `max`, `mean`, `lse` (τ = 1), `last`, `topk_mean` (k = 3) | `noisy_or` is deliberately **not** offered, because it inflates risk with segment count [R: hierarchy] |
| **GUID online** | s_g(n) = online score after segment n | segment scope uses the running aggregator over segments ≤ n |
| **Alarm score** | running max, `r_g(n) = max_{m ≤ n} s_g(m)` | with the default latch rule, "alarmed by n" ⇔ `r_g(n) > thr`, so no epoch imputation is ever needed |

**`eval.alarm_rule`:**
- `latch` (default): the first crossing latches the alarm.
- `k_of_n`: an alarm when k of the last n positions exceed thr. It is a hyperparameter, chosen on val.

### 11.3 Threshold policies (`eval.thresholds`) (FR-8)

Thresholds are chosen **per fold on the validation split**, per level, and frozen before test is scored.

**Threshold basis.** Each policy says **which metric type, at which point, carries the FPR constraint**. Keys: `basis`, `axis` (default `to_delivery`) and `at` (`end`, or hours, e.g. `1.0`, meaning c* = −1 h). The previous pipeline chose one threshold per metric type at 1 h before delivery. `basis` reproduces that, and extends it.

The negative-score vector, i.e. the scores the FPR cap is imposed on, depends on the basis:

| `basis` | Negative scores used on val | Counting unit |
|---|---|---|
| `guid_final` | the final or aggregated GUID score `s_g` of every negative GUID | negative GUIDs |
| `instantaneous` | the snapshot score (§11.5.1) of each negative GUID with a segment in the bin ending at c* | negative GUIDs present in the bin |
| `committed_cumulative` | the running max `r_g` up to c*, for negative GUIDs **available** at c* | available negative GUIDs |
| `committed_overall` (**default**) | `r_g` up to c* for available negatives, and **−∞** for negatives not yet monitored (they cannot alarm) | **all** negative GUIDs of the split |
| `segment` | segment scores `s_n` in the eval window | segments. **NP guarantees are void here** (segments are correlated), so only `empirical` is allowed |

**Computation.**
- Every basis reduces to the same function `fpr_threshold(neg_scores, alpha, method, delta)` on that vector. There is no binary search and no epoch filling. This fixes A1, A7 and A8.
- For `committed_overall`, padding with −∞ makes the empirical rule allow ⌊α·n_all⌋ alarms. The NP order statistic is taken over n_all, which is still a valid guarantee, because unmonitored negatives never alarm.
- With `at: end`, `committed_overall` equals "alarmed at any point during the recording": the running-max GUID decision.

**Evaluation beyond the basis.** Whatever its basis, **every policy's threshold is evaluated under all three metric types** (§11.5) and at every time point. This shows how a threshold chosen for one view behaves in the others. The previous pipeline evaluated each metric type only under its own threshold.

**Scores are thresholded on calibrated logits with a strict `>` rule.** The fraction of tied validation scores is reported.

| Policy | Definition |
|---|---|
| `fpr_cap`, `method: empirical` | Let n₀ = number of validation negatives and `v₍₁₎ ≤ … ≤ v₍ₙ₀₎` their sorted scores. Choose the smallest threshold `thr` among candidate values such that `#{v > thr}/n₀ ≤ α`, which maximises validation sensitivity under the cap. This overshoots α on new data about 50% of the time (Tong 2018) [R: Thresholds] |
| `fpr_cap`, `method: np_umbrella` | **Neyman–Pearson umbrella** (Tong, Feng & Li 2018): `k* = min{k : Σ_{j=k}^{n₀} C(n₀,j)(1−α)^j α^{n₀−j} ≤ δ}`, `thr = v₍k*₎`, alarm if score > thr. Guarantees P(population FPR > α) ≤ δ. Requires `n₀ ≥ ⌈log δ / log(1−α)⌉` (9 for α = 0.3, δ = 0.05). If violated: raise, unless `allow_fallback: true`, in which case use empirical and flag it |
| `youden` | argmax_thr (TPR − FPR) on validation; ties go to the lower threshold (pilot semantics, `latent_pilot/evaluate.py:1337`) |
| `sens_target` | smallest-FPR threshold with validation TPR ≥ β |
| `fixed` | a given calibrated probability (e.g. 0.5) |

`np_umbrella` in scipy is `k = next(k for k in range(1, n0+1) if binom.sf(k-1, n0, 1-alpha) <= delta)`. Unit-test it against a brute-force binomial sum.

**Default policy set [DECISION] [OPEN-5]:**

| id | policy | basis @ at | role |
|---|---|---|---|
| `np30` | fpr_cap α = 0.30, np_umbrella δ = 0.05 | committed_overall @ end | **primary** |
| `emp30` | fpr_cap α = 0.30, empirical | committed_overall @ end | literature-comparable |
| `emp15` | fpr_cap α = 0.15, empirical | committed_overall @ end | clinical anchor [R: CTG literature] |
| `inst30_1h` | fpr_cap α = 0.30, empirical | instantaneous @ 1 h | reproduces the previous pipeline's instantaneous threshold |
| `cum30_1h` | fpr_cap α = 0.30, empirical | committed_cumulative @ 1 h | reproduces the previous pipeline's committed-cumulative threshold |
| `youden` | youden | guid_final | threshold-agnostic reference |

For a segment-scope model with no online score, `instantaneous` and `committed_*` use the running segment aggregator (§11.2).

**Oracle (reported, never used for decisions).** Test TPR at exactly FPR ∈ `eval.report_tpr_at_fpr` (default 0.05, 0.10, 0.15, 0.20, 0.30), read off the test ROC. It is labelled *oracle*, and allows comparison with MCNN, FHR-LINet and OxSys.

**3-class:**
- `argmax` confusion (threshold-free).
- Every binary policy applied to a **collapsed adverse score**: `p_c1 + p_c2` for multiclass, `g` for ordinal.
- Optional per-class one-vs-rest thresholds (`eval.ovr_thresholds: true`).

**Always report:**
- validation sens/FPR next to test sens/FPR at the same threshold;
- **FPR overshoot**, test FPR − α;
- per-fold thresholds.

A search failure raises. There is never a silent fallback.

### 11.4 Metric catalogue (FR-10)

**Threshold-free (binary):**
- AUROC.
- AUPRC, with its prevalence baseline.
- pAUC over [0, α], McClish-standardised: `roc_auc_score(max_fpr=α)`.
- Log loss, Brier and scaled Brier (1 − Brier/Brier_ref).
- Calibration intercept and slope (logistic recalibration of y on the logit).
- ICI/E50/E90 (Austin & Steyerberg; LOWESS-free variant: a spline or binned smooth is acceptable).
- ECE with **equal-mass** bins (10), never equal-width.
- Score distributions per class.

**Thresholded (per policy):**
- TP, FP, TN, FN.
- Sensitivity, specificity, FPR, PPV, NPV.
- Balanced accuracy, MCC, LR+ and LR−.
- F1, reported but marked secondary: it is not a proper score (STRATOS) [R: Grouped CV].
- The threshold value and its validation counterparts.

**Decision curve:** net benefit `NB(p_t) = TP/N − FP/N · p_t/(1−p_t)` over p_t ∈ [0.05, 0.6], against treat-all and treat-none.

**3-class:**
- 3×3 confusion (argmax), counts and row-normalised.
- Per-class OvR AUROC and AUPRC; macro OvR AUROC; Hand–Till AUROC (`roc_auc_score(multi_class='ovo')`).
- Balanced accuracy and per-class recall.
- Quadratic-weighted kappa (`cohen_kappa_score(weights='quadratic')`).
- Ranked probability score.
- Macro-F1 (secondary).
- The binary collapses `adverse_vs_healthy` and `hie_vs_rest`, computed from the 3-class output.
- For a binary head evaluated on subtype subsets (healthy ∪ acidosis, healthy ∪ HIE): per-subtype sensitivity.

**Implementation:**
- Use sklearn: `roc_curve(drop_intermediate=False)`, `roc_auc_score`, `average_precision_score`, `confusion_matrix`, `brier_score_loss`, `log_loss`, `cohen_kappa_score`, `calibration_curve(strategy='quantile')`.
- Use the pilot's `confusion_counts` and rate-NaN rules: a rate is NaN when its class-count denominator is 0.
- torchmetrics is used **only** for in-training monitoring. Pass probabilities, not logits, because of its auto-sigmoid pitfall.

### 11.5 Online and time-resolved analysis

**Time axes (`eval.time_axes`):**

| Axis | Definition |
|---|---|
| `to_delivery` | hours before delivery; evaluation only |
| `from_onset` | `tlo_end`; GUIDs without TLO are excluded and counted |
| `rel_second_stage` | `ss_rel + 1260` in h; GUIDs with unknown second stage are excluded and counted |
| `position` | `seg_pos`, 1..N |
| `elapsed` | hours since the first segment |

#### 11.5.1 The three metric types **[DECISION]**

These carry over the previous pipeline's instantaneous / committed-cumulative / committed-overall analysis. The definitions are corrected for its defects (A1, A5, A8, A15) and generalised to every axis.

**Notation**
- Each axis defines a clock c that increases with real time: `to_delivery` → `t_end` in hours, negative; `from_onset` → `tlo_end`; `rel_second_stage` → `(ss_rel + 1260)/3600`; `position` → `seg_pos`; `elapsed` → hours since the first segment.
- c_g(n) is the clock of GUID g's segment n. An evaluation point is c*. A bin is b = (c_lo, c_hi].
- s_g(n) is the online score after segment n (for segment scope: the segment score s_n, or the running aggregator; see below).
- r_g(n) = max_{m ≤ n} s_g(m) is the running max.
- n*(g, c*) is the last segment with c_g(n) ≤ c*.
- The latched alarm state is A_g(c*) = 1[r_g(n*) > thr]. Under `k_of_n` it is the k-of-n state, latched after the first firing.

**The three types.** Each is computed per class. Positives give sensitivity; negatives give FPR, with specificity = 1 − FPR.

| Metric type | Population at the time point | Decision | Sensitivity / FPR | Behaviour and question answered |
|---|---|---|---|---|
| **Instantaneous** (bin b) | GUIDs with ≥ 1 segment whose clock falls in b. Each GUID contributes **one** row: its last segment in b (the *snapshot*) | **raw, not latched**: d_g = 1[s_g(n_b) > thr] | TP/P and FP/N over that population | Not monotone: a GUID can alarm and then stop. *"Is the model flagging right now?"* |
| **Committed cumulative** (point c*) | **Available** GUIDs: monitoring had started by c* (first segment clock ≤ c*) | latched A_g(c*) | alarmed / available, per class | Not monotone, because the denominator grows as more GUIDs come under monitoring. *"Of the labours monitored so far, what fraction has been alarmed?"* |
| **Committed overall** (point c*) | **All** GUIDs of the split that are eligible on this axis (e.g. `has_tlo` for `from_onset`); not-yet-monitored GUIDs count as not alarmed | latched A_g(c*) | alarmed / all, per class | **Monotone non-decreasing** in c* (asserted). At `end` it equals the GUID-level running-max decision. *"Of all adverse (healthy) labours, what fraction had been alarmed by this time?"* The FPR a unit would actually experience |

**Corrections to the previous pipeline:**
- **Instantaneous** used the *latched* `clinical_pred`, which made it a noisy copy of committed. Here it uses the raw score. The latched state at a time point *is* committed cumulative.
- **Denominators.** Bins used to count rows. Here each GUID counts once per bin, so long recordings do not dominate.
- **Epoch filling** is never done. Running max and "last segment ≤ c*" make the committed types exact on irregular, gappy data.

**Variants:**
- **Segment-level instantaneous** (`level=segment`) counts every segment in b, with its segment score s_n. It is reported for segment-scope models, with GUID-cluster bootstrap CIs.
- **3-class** runs each type for:
  - (i) the collapsed adverse score, under every binary policy;
  - (ii) per class, one-vs-rest with per-class thresholds when `eval.ovr_thresholds: true`;
  - (iii) argmax, instantaneous only, giving top-1 accuracy and macro-F1 per bin.
- **Threshold-free companions.** At the same populations the engine computes:
  - **snapshot AUROC/AUPRC** (instantaneous population, score s_g(n_b));
  - **cumulative AUROC** (available population, score r_g(n*)).

  Thresholded and ranking views of the same time point therefore always share a denominator.

**Staleness.** For instantaneous checkpoints on the `to_delivery` axis (not bins), a GUID's snapshot must have ended within `eval.snapshot_max_staleness_h` (default 1 h) of c*. Otherwise the GUID is excluded, and the exclusion counted.

**Last minutes.** `eval.exclude_last_min` (default 0; the previous pipeline used 30) drops segments ending within that many minutes of delivery from **every** metric type. Report it when it is non-zero.

#### 11.5.2 Where the metric types are evaluated

- **Checkpoints** (`eval.checkpoints_h`, default `[6, 4, 3, 2, 1, 0.5]`) on the `to_delivery` axis, plus `end`. This gives the headline time-resolved table: every policy × every metric type × checkpoint, with n per class.
- **Bin grid** (`eval.bin_h`, default 0.5 h) on **every** axis in `eval.time_axes`:
  - Instantaneous uses bin b.
  - Committed types use the bin's right edge c_hi.
  - The **position** axis gives "metrics vs segment index".
  - The **rel_second_stage** axis gives the second-stage analysis that the old SSO module produced, without its silent cohort filter. GUIDs with unknown second stage are excluded **and counted** (L14).
- **Segment-level AUROC per bin**, time-matched (§6.6).
- **Bin reporting rules:**
  - Every bin and checkpoint reports n_pos and n_neg and a bootstrap CI.
  - Bins with fewer than `eval.min_bin_class_n` (default 5) of either class show as NaN and are drawn hollow.
- **Never averaged over bins.** Headline numbers are always read at named checkpoints (A15).

#### 11.5.3 Alarm analysis (per policy, latch or k_of_n)

- **Positives:** event sensitivity (alarmed at all before the last segment); lead time = `−t_end` of the first alarm, in hours; cumulative detection vs lead time; median lead time with IQR.
- **Negatives:** the fraction alarmed by τ (the patient-level FPR curve); time to first false alarm; alarm burden = the fraction of monitored segments after the first alarm, i.e. time-in-alarm.
- **Alarms per detection:** FP GUIDs / TP GUIDs (Tomašev 2019).

#### 11.5.4 Stage stratification

Every time-resolved metric is also reported for `stage ∈ {first, second}` subsets.

### 11.6 Subgroup analysis (`eval.subgroup_families`)

This section carries over and extends the previous pipeline's 22-filter subgroup analysis (`create_enhanced_subgroup_filters`, `evaluate_classifier.py:1271-1351`).

**Rules:**
- **Membership is computed per GUID from GUID-level fields.** The old code filtered rows by row-level target (§3.3).
- **Metric by population:**

  | Subgroup population | Metric reported |
  |---|---|
  | healthy-only | **specificity / FPR** |
  | unhealthy-only | **sensitivity** |
  | mixed-class | both, plus threshold-free metrics |

  This also fixes the old per-fold "healthy" line, which was always NaN because it plotted sensitivity.
- **No subgroup ever changes the threshold.**

#### 11.6.1 Subgroup families

| Family id | Members (GUID-level definition) | Population | Origin |
|---|---|---|---|
| `class` | `healthy` (code 1), `acidosis` (2), `hie` (3), `unhealthy` (2 ∪ 3) | single-class | old `acidosis`, `hie`, `unhealthy`, `healthy` |
| `source_file` | the 8 shard cells of §2.3 (`healthy_no_bg_no_cs`, …, `hie_no_cs`) | single-class | old healthy BG×CS cells and class×CS; exact cohort cells |
| `class_x_cs` | {healthy, acidosis, hie, unhealthy} × {cs+, cs−} | single-class | old `*_cs_pos/neg` |
| `healthy_x_bg` | healthy × {bg+, bg−} | single-class | old `healthy_bg_pos/neg` |
| `healthy_bg_x_cs` | the 4 healthy bg × cs cells | single-class | old `healthy_bg_{pos,neg}_cs_{pos,neg}` |
| `cs` | cs+, cs− (all classes) | mixed | old legacy `cs_positive/negative` (computed there but never plotted) |
| `bg` | bg+, bg− (all classes) | mixed | old legacy `bg_*` |
| `stage_last` | first / straddle / second / unknown, at the GUID's last segment | mixed | new |
| `reached_second_stage` | yes (any segment with `ss_rel + 1260 ≥ 0`) / no (known and never reached) / unknown | mixed | new; the old SSO eligibility, as a stratum instead of a filter |
| `has_tlo` | yes / no | mixed | new; also the confound readout |
| `admission_tlo_tertile` | tertiles of `tlo_end` at the first segment (early vs late admission in labour); unknown TLO is its own level | mixed | new |
| `labour_duration_tertile` | tertiles of onset→delivery (**retrospective; a stratum only, never an input**). This is the fold stratifier (§2.5) | mixed | new |
| `n_segments_tertile` | tertiles of `n_segments` | mixed | new; the length-shortcut readout |
| `span_tertile` | tertiles of recording span (first `epoch` → last `t_end`) | mixed | new |
| `valid_frac_tertile` | tertiles of the GUID's mean `valid_frac` (signal quality) | mixed | new [R: informative signal loss] |
| `late_coverage` | has ≥ 1 segment ending within 1 h of delivery: yes / no | mixed | new (§2.1: some GUIDs have no late data) |
| `shared_test` | yes / no | mixed | new (§2.5) |
| `fold` | 1…k | mixed | new; heterogeneity (§11.12, block H) |
| `covariate:<name>` | available / missing, per covariate | mixed | new (§7.3) |

**Documented but not computed.** `acidosis_x_bg` and `hie_x_bg` are empty by construction: every unhealthy GUID has bg = 1 (§2.3). The report states this instead of plotting an empty line.

**Tertile cut-points** are computed on the **pooled test population of the evaluated split**, written to `evaluation/tables/subgroup_cutpoints.json`, and reused for val. They are descriptive strata, not model inputs, so no train/val separation is needed.

**Restricted-pair populations** (reported with the families): `healthy ∪ acidosis`, `healthy ∪ hie`, `acidosis ∪ hie`. On each: AUROC with CI, and sensitivity of the positive subtype at each policy. The shared FPR is identical across pairs and is reported once. This carries over the old "binary head by underlying class" (`evaluate_3class_metrics.py:380`) and adds the restricted AUROC it lacked.

#### 11.6.2 What is computed per subgroup (per split, per fold, and pooled)

1. **Counts:** n GUIDs per class, n segments, prevalence.
2. **Threshold-free** (mixed populations only): AUROC, pAUC@α, AUPRC (with its prevalence baseline), Brier, calibration intercept and slope. Each has a GUID-bootstrap CI, at the GUID-final and snapshot/cumulative checkpoint levels.
3. **Thresholded, per policy** (single-class or mixed): sensitivity or specificity/FPR as appropriate, TP/FP/TN/FN, PPV and NPV (mixed only), Wilson and bootstrap CIs.
4. **The three metric types vs time** (§11.5.1), on every axis in `eval.time_axes`, at checkpoints and bins, with n per class per point.
5. **ΔAUROC vs complement** (subgroup vs everyone else in the split) and Δsensitivity/Δspecificity vs complement, from a paired GUID bootstrap. This reuses the §11.8 machinery.
6. **Score distributions** per class within the subgroup.
7. **Tests across the members of a family** (score distribution per class; one family is one test family). This follows the VAE pattern (`teb_vae/lag_attn_cfs/analyses/cross_subgroup.py:305-405`, helpers in `teb_vae/lag_attn/eval/stats.py`):
   - Kruskal–Wallis per family → Holm across families (`holm_adjust`);
   - for surviving families, pairwise Mann–Whitney + Cliff's δ (`pairwise_comparisons`, `delta_magnitude`) → Holm within the family.
   - α = 0.05 is a module constant, not a config key (the VAE rationale). Groups smaller than `MIN_GROUP_SIZE` (3) are excluded and recorded.

**Power guard.** A subgroup with fewer than `eval.min_subgroup_n` (default 10) GUIDs in a class needed by a metric reports NaN for that metric, with `underpowered = true`, and is drawn hollow. It is never silently dropped.

**Ordering and colours:**
- Groups are ordered by `teb_vae/lag_attn/eval/labels.py:183 ordered_groups`, **worst cohort first** (HIE, acidosis, healthy; subgroups in reversed canonical order). The same order orients every pairwise test, so δ > 0 means the more severe cohort scores higher.
- Class and `source_file` colours come from `teb_vae/lag_attn_cfs/eval/figures_seam.py:264 group_colors` (`CLINICAL_CLASS_COLORS`: healthy `#2E8B57`, acidosis `#E8A33D`, HIE `#C0392B`; `SUBGROUP_COLORS` tints).
- Other families use fixed palettes defined once in `report.py`. The old CS/BG colours are kept:

  | Member | Colour |
  |---|---|
  | cs+ | `#3498db` |
  | cs− | `#9b59b6` |
  | bg+ | `#f39c12` |
  | bg− | `#16a085` |
  | tertiles | a sequential 3-step palette |
  | stage | first `#4c72b0`, straddle `#8172b2`, second `#c44e52`, unknown grey |

- **Do not use `utils/style.get_class_colors`.** It paints healthy blue, which conflicts with the VAE figures.
- The prediction tables carry `clinical_class` (the name) and `subgroup` (the shard basename without extension) string columns next to `class_code`, so the VAE helpers (`GROUP_COLUMNS`, `ordered_groups`, `per_recording_*`, `emit_grouped_variants`) work unchanged.

### 11.7 Cross-fold aggregation, CIs and prevalence shift

**Per-fold vs pooled:**
- **Per fold:** every metric on val and on test.
- **Pooled OOF test (primary for threshold metrics):**
  - Concatenate all folds' test GUID rows, each with its own fold's calibrated score and threshold.
  - Pool the confusion counts, then compute sensitivity, specificity, FPR, PPV and NPV [Forman & Scholz 2010].
  - Shared test GUIDs: under `first_fold`, only the lowest-fold row is kept; under `exclude`, they are dropped. The other variant is also reported as a sensitivity analysis.
- **AUROC and pAUC:** the per-fold mean ± SD is primary. Pooled OOF AUROC on **calibrated** scores is secondary and labelled as such.
- **Validation aggregates** are computed the same way and labelled *optimistic (used for selection)*.

**Bootstrap (`eval.bootstrap`):**
- GUID-level (patient-level if a map is given), outcome-stratified, percentile intervals, B = 2000.
- Reuse the pilot's `paired_bootstrap` and `metric_intervals` (same units, strata and draws).
- Two variants:
  - (a) **Fixed thresholds** (default): test-sampling uncertainty.
  - (b) **Refit** (`refit_threshold: true`): each replicate resamples the fold's val GUIDs, re-derives the calibration and threshold, then resamples test GUIDs. The CI then includes threshold noise [R: Grouped CV]. Heavier; run it for final reports.
- Undefined draws are counted and reported, never silently dropped.

**Other intervals:**
- Wilson CIs for single proportions per fold (`scipy.stats.binomtest(...).proportion_ci('wilson')`).
- **Across-fold uncertainty caveat:** naive CV CIs under-cover (Bates 2024). The summary states this.

**Prevalence shift (§2.5):**
- Report the test prevalence per fold.
- PPV and NPV are reported as observed on test, and **re-weighted to a reference prevalence π_ref** (`eval.reference_prevalence`; default null = skip) via Bayes: `PPV = sens·π/(sens·π + (1−spec)(1−π))`.
- Calibration on test is reported raw, and after prior-shift correction `logit p' = logit p − logit π_val + logit π_test`.
- FPR-cap thresholds are unaffected by prevalence.

### 11.8 Model comparison (`run.py compare`)

- Across runs that share cohort digests:
  - paired GUID bootstrap on pooled OOF test (ΔAUROC, Δsensitivity at each policy);
  - DeLong on pooled GUID-level scores (a fast DeLong implementation, ~40 lines, written from the paper, not copied from an unlicensed repo);
  - Nadeau–Bengio corrected resampled t-test on per-fold AUROCs, with variance factor `(1/k + 1/(k−1))`, k = number of folds.
- A comparison refuses to run if the cohort digests differ.

### 11.9 `evaluation/tables/metrics.parquet` schema (long format)

```
run_id, model_id, seed ('ens' allowed), fold ('pooled' allowed), split, level (segment|guid|online|alarm),
task, head, label_strategy, eval_window, subgroup, subgroup_value, axis, t (checkpoint/bin centre; NaN if n/a),
metric_type (threshold_free|instantaneous|committed_cumulative|committed_overall|alarm),
denominator (bin_present|available|all|n/a), policy_id, policy_basis, threshold, metric, value, ci_lo, ci_hi, n_pos, n_neg, n_boot_undefined
```

Plus `evaluation/summary.json`: the headline table, keyed by (level, policy, metric), holding the pooled test value with CI, the per-fold mean ± SD, and the validation value.

### 11.10 Figure conventions (`report.py`, FR-11)

The complete list of figures is in the analysis catalogue (§11.12). The rules here apply to every figure.

**Rendering and style:**
- Every figure is rendered from `evaluation/tables/*.parquet` and `predictions/*.parquet` only. Never from in-memory model state.
- Use the VAE figure seam (`teb_vae/lag_attn_cfs/eval/figures_seam.py`, which re-exports `teb_vae/lag_attn/eval/figures.py`):
  - `configure_figure_style(fmt)`, called **once** at the start of report;
  - `new_figure(n_rows, n_cols, height_per_row=…)`;
  - `render_figure(fig, path_stem)`: an extension-less stem, 200 dpi, panel letters, tight layout with footnote space, and it **always closes** the figure;
  - `legend_with_headroom`, `ribbon_plot`, `violin_panel`, `grouped_violin_figure`, `heatmap_with_colorbar`, `significance_strip`, `windowed_comparison_figure`.
- Do not call `utils/style.save_figure` directly: it defaults to 600 dpi and adds no panel letters.
- **Formats.** `eval.figure_formats` (default `[pdf]`) is looped by calling `set_figure_format` per format. PNG is available by adding `png`.
- **Every panel tolerates empty or all-NaN input** and shows `EMPTY_NOTE`; it never raises.
- Figure stems are module-level constants in `report.py`. The registry (`FIGURE_REGISTRY`, §11.15) lists every stem the verify gate expects for the run's configuration.

**Annotations:**
- Every time-resolved plot follows the old pipeline's orientation: the hours-before-delivery axis is **inverted**, so delivery is on the right.
- It carries:
  - a dashed grey vertical line at each **threshold-basis time point** (`at`, e.g. "decision time 1 h");
  - a dotted zero line on the `rel_second_stage` axis (label "second-stage onset");
  - an **n strip** under the axis (GUIDs per class per bin);
  - a **footer** with exclusion counts (e.g. "excluded: 41 GUIDs with unknown second stage (NaN 38, sentinel 3)").
- Legends carry `(N = <GUIDs>)` per line (per fold), or `(N = <GUIDs>, k folds)` (pooled).
- Operating points are drawn as ◦ at the validation-chosen point and ● at the realised test point, with the α line dashed.
- The previous pipeline's line colours are kept for metric lines: sensitivity `#2ecc71` (circle), specificity `#3498db` (square), FPR `#e74c3c` (triangle). Class colours follow §11.6.2.

**Cross-fold display** (`eval.fold_band`):
- Every pooled curve shows:
  - thin per-fold lines (alpha 0.25);
  - the pooled estimate (thick);
  - its **GUID-bootstrap 95% band**.
- `fold_band: minmax` (default) **also** shades the per-fold min–max envelope, as the previous aggregated plots did.
- ROC curves also show the **vertical average** across folds with a ±1 SD band (old `aggregate_results.py:164`) next to the pooled ROC.
- Per-fold curves are drawn only where that fold has data. There is no `np.interp` edge extrapolation (A14). Per-point `n_folds` is stored in the tables.

**Per-fold figure sets** (`eval.per_fold_figures`, default `true`):
- The **core set** is also rendered per fold under `evaluation/figures/fold_<k>/`: R1, R2, M1, M2, S2 (class family), X1, K1, E1 (§11.12).
- The full set is rendered for `pooled` only.
- Validation figures (`split = val`) are rendered with the same stems under `evaluation/figures/val/`. They are labelled "validation (used for selection)".

**Scale management.** There are no five-variant copies and no per-(subgroup × class × metric-type) PNG explosion. The previous pipeline wrote about 72 + 48 PNGs per fold and about 250 aggregated. Instead:
- families are **faceted**: rows = subgroup family, columns = metric type, one figure per axis × policy;
- per-class × subgroup curves are written to tables (`metrics.parquet`) and drawn only for the primary policy.

**Per-GUID pages.**
- Masking and log-scale helpers come from `teb_vae/lag_attn_cfs/attributions.py:849-927` (`masked_field`, `signed_log_norm`, `unsigned_log_norm`, `symlog_axis`). That module **imports captum at load time**, so these ~40 lines are **ported** into `report.py` with a pointer comment, rather than importing `attributions`.
- `figures_seam.caveat_note(fig, text=…)` MUST be called with explicit text. Its default describes the VAE lag axis and would mislabel a classifier page. Use: "Input coefficients are one-sided, with a per-channel group delay of up to N s. The time axis is stored time."

### 11.11 `summary.md`

Generated per run. It contains:
1. The config digest, source fingerprint, git SHA and environment versions.
2. The cohort flow table: per fold × split × class, with exclusions by reason.
3. The headline table: pooled test with CI, the per-fold mean ± SD, and validation values, for the primary policy and levels.
4. The same table for baselines and controls, with the shortcut and shuffled-label warnings.
5. Threshold and overshoot table.
6. Calibration table.
7. 3-class table.
8. The time-resolved table: primary policy × {instantaneous, committed cumulative, committed overall} × checkpoints, with sensitivity, FPR, n and CIs; plus snapshot and cumulative AUROC.
9. Subgroup table.
10. Subgroup forest highlights (S3), restricted pairs (S5), alarm and lead-time summary (A1–A4), fold heterogeneity (H1), and the top errors (E1).
11. Limitations, auto-filled: internal CV only, prevalence shift, shared test GUIDs, missingness-confound warnings, and "NP guarantee at GUID level only".
12. A TRIPOD+AI mini-checklist, listing which items the run's outputs support (data flow, outcome definition, predictors and timing, sample size, missing data, analysis, performance with CIs, calibration) [R: TRIPOD+AI].

### 11.12 Complete analysis catalogue

**Coverage.** Every analysis the module produces is listed here. It carries over every capability of the previous pipeline, checked item by item against its code (`tmp/new_classifier/evaluate_classifier.py` and `guid_cls_v1/*`), and adds new ones.

**Columns:**
- **Old** names the previous equivalent (or "new").
- **Outputs** gives the table (always written) and the figure stem (in `FIGURE_REGISTRY`).

**Scope.**
- Unless stated otherwise, each analysis runs for **val and test**, **per fold and pooled**, for the **binary task**, and for 3-class through the collapsed adverse score.
- "× policies" means for every threshold policy.
- "× axes" means every axis in `eval.time_axes`.

Every table goes into `evaluation/tables/metrics.parquet` (long format, §11.9) unless a separate table is named.

#### Block C — Cohort and dataset statistics (stage `cohort`; per fold × split, and cross-fold)

| ID | Analysis | Definition | Outputs | Old |
|---|---|---|---|---|
| C1 | Cohort flow | GUIDs and segments per fold × split × class × source_file; exclusions by reason (§6.1); task exclusions (§6.3) | `cohort/fold_summary.csv`, summary §1 | partial |
| C2 | Dataset summary | Totals; segments per GUID (min, max, mean, median, SD); time range (h, HH:MM); unique slots; class, cs and bg counts at GUID and segment level; TLO/SS availability per class; second-stage sentinel count | `cohort/dataset_summary.json` | `dataset_summary.json` |
| C3 | Label cross-table | class × cs × bg: n_guids, n_segments (12 cells) | `cohort/label_cross_table.csv` | `label_cross_table.csv` |
| C4 | Cohort overview page | 2×2 panels: (a) histogram of segments per GUID with mean and median lines; (b) segment count vs hours before delivery (inverted); (c) GUID counts across the 8 source_file cells; (d) text box of totals | fig `cohort_overview` | `dataset_overview.pdf` |
| C5 | Subgroup overview | Per subgroup: GUID bar (left axis) and hatched segment bar (right axis) | fig `cohort_subgroups` | `subgroup_overview.pdf` |
| C6 | Segments per time bin | Stacked by class; second panel stacked by cs/bg within each class, including the healthy 4-way bg × cs | fig `cohort_time_bins` | `epochs_per_time_bin*.pdf` |
| C7 | Ranked segments per GUID | GUIDs sorted by n_segments, bars coloured by class, mean line | fig `cohort_ranked_lengths` | `epochs_per_guid_ranked.pdf` |
| C8 | Coverage heatmap | Rows = GUIDs (grouped by class, sorted by span), columns = slots on the delivery axis; cell = present / excluded / absent. Shows gaps and late coverage | fig `cohort_coverage` | new |
| C9 | Gap and run statistics | Per class: distribution of within-GUID gaps (h), number of contiguous runs, fraction with late coverage | table + fig `cohort_gaps` | new |
| C10 | Clock availability | Per class × fold: has_tlo, has_ss, sentinel, admission-TLO distribution, labour-duration distribution | table + fig `cohort_clocks` | new |
| C11 | Cross-fold count spread | Mean / SD / min / max of every C1–C3 count across folds | `cohort/count_spread.csv` | `_aggregate_dataset_statistics` |
| C12 | Leakage and validity readouts | Exposure (L3), missingness confound (L10), shared test GUIDs (L12), stride and post-warm-up coverage fraction (§2.6) | `cohort/exposure.json`, `confound.json`, summary §10 | new |

#### Block R — ROC, PR and ranking

| ID | Analysis | Definition | Outputs | Old |
|---|---|---|---|---|
| R1 | GUID final-score ROC | ROC of `s_g` (sequence final, or each segment aggregator). Pooled curve with bootstrap band; thin per-fold curves; vertical average ± SD; operating points ◦/● for every policy; AUC pooled and mean ± SD | table `roc_points.parquet` (fold, split, level, variant, fpr, tpr, thr); fig `roc_guid` | `roc.png`, aggregated ROC |
| R2 | ROC at checkpoints, **committed-cumulative** population | For each checkpoint in `eval.checkpoints_h` ∪ {end}: ROC of `r_g(n*)` over GUIDs **available** at c*. Small multiples | fig `roc_checkpoints_cumulative` | old CC-ROC (fixed; A5) |
| R3 | ROC at checkpoints, **committed-overall** population | Same as R2, but not-yet-monitored GUIDs score −∞ | fig `roc_checkpoints_overall` | new |
| R4 | Snapshot ROC per bin (**instantaneous** population) | ROC of each GUID's snapshot score in a bin; drawn for the checkpoint bins | fig `roc_checkpoints_snapshot` | new |
| R5 | Segment-level ROC | Time-matched eval window (§6.6), GUID-cluster bootstrap band | fig `roc_segment` | new |
| R6 | PR curves | Same variants as R1 and R2, with AP and the **prevalence baseline** line | fig `pr_guid`, `pr_checkpoints` | new for binary |
| R7 | AUROC / AUPRC / pAUC vs time | Snapshot and cumulative AUROC (and pAUC@α) vs time, × axes, bootstrap CIs, n strip | fig `auroc_vs_time_<axis>` | per-class AUROC vs time (delivery axis only) |
| R8 | Decision-horizon ribbon | Sensitivity at each FPR-cap policy vs decision time c* ∈ `eval.decision_horizons_h` (default 0.5, 1, 2, 4, 6 h). The threshold is re-selected on val at each horizon (committed_overall basis) | fig `decision_horizon` | PRD §18 item, never built |
| R9 | Oracle TPR at fixed FPR | Test TPR at FPR ∈ `report_tpr_at_fpr`, read off R1 and R2. Labelled *oracle* | table | new |
| R10 | Stage-specific ROC | Snapshot ROC restricted to first-stage vs second-stage snapshots | fig `roc_stage` | new |
| R11 | Subgroup ROC | R1 per member of the mixed families (cs, bg, stage_last, has_tlo, tertiles) and per restricted pair (healthy∪acidosis, healthy∪hie, acidosis∪hie), with bootstrap bands | fig `roc_subgroups_<family>` | new |
| R12 | Score distributions | Violin/box per class (GUID final score, and snapshot at checkpoints); 3-class: one panel per probability, one box per true class | fig `score_distributions` | 3-class box plots, histograms |
| R13 | Windowed score comparison | `windowed_comparison_figure`: score by class per 0.5 h window, Kruskal–Wallis + Holm strip, Cliff's δ heatmap | fig `score_windows_<axis>` | new (VAE convention) |

#### Block T — Thresholds

| ID | Analysis | Definition | Outputs | Old |
|---|---|---|---|---|
| T1 | Threshold table | Per fold × policy: threshold, basis, `at`, val sens/FPR/spec, n_pos/n_neg, NP k*, tie fraction. Cross-fold mean / SD / min / max | `evaluation/tables/thresholds.parquet` | `threshold_info.json`, threshold mean/std |
| T2 | Val→test drift | Validation vs test sens and FPR per fold and policy, with **FPR overshoot** (test FPR − α) and its pooled distribution | fig `threshold_drift` | new |
| T3 | Threshold stability | (a) spread of per-fold thresholds; (b) bootstrap-refit spread (§11.7 refit variant); (c) test sens/FPR under a threshold perturbation of ±1 and ±2 validation order statistics | fig `threshold_stability` | new |
| T4 | Metric-type cross-evaluation | Every policy (whatever its basis) evaluated under all 3 metric types at every checkpoint | fig `metric_type_comparison` (3b) | `metric_type_comparison.png` (v0) |
| T5 | Per-class OvR thresholds (3-class) | Per class × basis: threshold, val sens/FPR | `thresholds.parquet` (class column) | `perclass_thresholds.json` |

#### Block M — Metric types and time-resolved performance (§11.5)

| ID | Analysis | Definition | Outputs | Old |
|---|---|---|---|---|
| M1 | Metric types vs time | Three stacked rows (instantaneous, committed cumulative, committed overall): sensitivity, specificity and FPR lines with CIs, α line, basis-time line, n strip, × axes × policies (figures for the primary policy and every `*_1h` basis policy; tables for all) | fig `metric_types_<axis>_<policy>` | 5 variants × 3 types, per fold and aggregated |
| M2 | Checkpoint table | policy × metric type × checkpoint: sens, spec, FPR, PPV, NPV, n, CIs | table + summary §8 | `decision_point_metrics` |
| M3 | Second-stage axis mirror | M1 on `rel_second_stage`, with the zero line and the exclusion footer by reason | fig `metric_types_rel_second_stage_<policy>` | SSO mirror (without its cohort filter) |
| M4 | Labour-onset axis | M1 on `from_onset` | fig `metric_types_from_onset_<policy>` | new (TLO was never used) |
| M5 | Position axis | M1 on `position` (metrics vs segment index) | fig `metric_types_position_<policy>` | new |
| M6 | Segment-level instantaneous | Every segment in a bin, segment score, GUID-cluster CI | fig `segment_instantaneous_<axis>` | old row-level instantaneous |
| M7 | Monotonicity and consistency checks | committed_overall monotone; at `end`, committed_overall = the running-max GUID decision; instantaneous at the last segment = the `score_final` decision (sequence scope) | `sanity` block (§11.14) | old monotonicity log |

#### Block S — Subgroups (§11.6)

| ID | Analysis | Definition | Outputs | Old |
|---|---|---|---|---|
| S1 | Subgroup table | For every family member: counts, threshold-free metrics (mixed), sens or spec at every policy with CIs, underpowered flags | `evaluation/tables/subgroups.parquet` | subgroup long CSV (counts lost, A4) |
| S2 | Subgroup curves vs time | Facet figure: rows = families (class; class_x_cs; healthy_x_bg; healthy_bg_x_cs; stage_last; has_tlo; tertile families), columns = the 3 metric types; lines = members with N in the legend; single-class families show sens or spec per §11.6. × axes, primary policy | fig `subgroups_vs_time_<axis>_<policy>` | 8 subgroup PNGs per type (diagnosis, cs/bg stratifications, healthy combos) |
| S3 | Subgroup forest | One panel per family: AUROC and pAUC (mixed), sens/spec at the primary policy, calibration slope; CIs, underpowered hollow | fig `subgroup_forest` | new |
| S4 | Δ vs complement | ΔAUROC and Δsens/Δspec with paired-bootstrap CIs, Holm-adjusted within the family | table + fig `subgroup_delta` | new |
| S5 | Restricted pairs | healthy∪acidosis, healthy∪hie, acidosis∪hie: AUROC, ROC (R11), subtype sensitivity at each policy, × metric types vs time | fig `restricted_pairs_<axis>` | "binary by underlying class" (sens only) |
| S6 | Family tests | Kruskal–Wallis → Holm → pairwise Mann–Whitney + Cliff's δ → Holm, on score distributions per class | table `subgroup_tests.parquet` | new (VAE convention) |
| S7 | Covariate availability strata | S1/S3 for `covariate:<name>` and `has_tlo`; with `eval.covariates_off`, the covariate-on vs covariate-off comparison | fig `covariate_strata` | new |

#### Block K — Calibration

| ID | Analysis | Definition | Outputs | Old |
|---|---|---|---|---|
| K1 | Reliability (binary) | Equal-mass bins with a count histogram; slope, intercept, ECE, ICI in the legend; before and after calibration | fig `calibration_guid` | new for binary |
| K2 | Reliability per class (3-class) | 1×3 panels, equal-mass bins, per-class ECE | fig `calibration_per_class` | `calibration_perclass.png` (equal-width) |
| K3 | Calibration per subgroup | Reliability curves for cs±, bg±, stage_last, has_tlo; slope and intercept in S3 | fig `calibration_subgroups` | new |
| K4 | Calibration per fold | Slope and intercept forest across folds; temperature per fold | fig `calibration_folds` | new |
| K5 | Prevalence-shift view | Test calibration raw vs prior-shift corrected; PPV/NPV at test prevalence and at π_ref | table + fig `prevalence_shift` | new |
| K6 | Decision curve | Net benefit vs p_t, with treat-all and treat-none | fig `decision_curve` | new |
| K7 | Brier vs time | Per-class and macro Brier vs time (3-class); binary Brier of snapshot scores vs time | fig `brier_vs_time_<axis>` | `brier_perclass` vs time |

#### Block X — Confusion and 3-class

| ID | Analysis | Definition | Outputs | Old |
|---|---|---|---|---|
| X1 | Binary confusion | Per policy; counts and rates; pooled (summed counts) and per fold | fig `confusion_binary` | new |
| X2 | 3×3 confusion (argmax) | Counts and row-normalised; pooled = summed counts; mean of row-normalised per-fold matrices also shown | fig `confusion_3class` | `confusion_matrix_3class.png`, mean confusion |
| X3 | Confusion evolution | 3×3 snapshot confusion at up to 8 checkpoints/bins, **one GUID per bin** | fig `confusion_evolution_<axis>` | `confusion_evolution` (row-level) |
| X4 | 3-class OvR ROC and PR | Per class + macro; pooled with a per-fold band; AP with prevalence baseline | fig `roc_ovr`, `pr_ovr` | aggregated OvR ROC; per-class PR |
| X5 | Per-class metrics vs time | (a) argmax per-class recall vs time; (b) OvR-threshold per-class sens/spec/FPR vs time under the 3 metric types (`ovr_thresholds`, **default true when `task: three_class`**) | fig `per_class_vs_time_<axis>` | `per_class_vs_time/{mt}_panel` |
| X6 | Per-class AUROC vs time | Snapshot OvR AUROC per class vs time, × axes (including `rel_second_stage`, which closes the old gap) | fig `per_class_auroc_vs_time_<axis>` | `auroc_vs_time` (delivery only) |
| X7 | Top-1 / F1 vs time | Top-1 accuracy; macro, weighted and per-class F1 vs time (snapshot, one GUID per bin) | fig `f1_vs_time_<axis>` | `top1_acc`, `macro_f1` vs time |
| X8 | Ordinal metrics | QWK, ranked probability score, Hand–Till AUROC; ordinal alarm score AUROC | table | new |
| X9 | Collapse consistency | Binary head vs P(acidosis) + P(HIE) (multi-task runs): AUROC of each, their correlation, disagreement at the primary policy | table + fig `collapse_vs_binary` | new |
| X10 | Per-class × subgroup | X5 for every single-class family member: table always; figure for the primary policy only | table; fig `per_class_subgroups_<axis>` | per-class × subgroup grid |

#### Block A — Alarms and lead time (§11.5.3)

| ID | Analysis | Definition | Outputs | Old |
|---|---|---|---|---|
| A1 | Event sensitivity | Fraction of positive GUIDs alarmed before their last segment, per policy | table | new |
| A2 | Lead-time distribution | Delivery − first alarm (h) for true alarms: median, IQR, histogram; cumulative detection vs lead time | fig `lead_time` | `first_detection_epoch` saved but never analysed |
| A3 | False-alarm curve | Fraction of healthy GUIDs alarmed by τ (the committed-overall FPR curve), and time-to-first-false-alarm distribution | fig `false_alarms` | new |
| A4 | Alarm burden | Healthy: fraction of monitored segments in alarm; alarms per detection (FP/TP GUIDs) | table | new |
| A5 | Alarm rule comparison | latch vs k_of_n (when configured) on A1–A4 | table | new |

#### Block H — Heterogeneity and stability

| ID | Analysis | Definition | Outputs | Old |
|---|---|---|---|---|
| H1 | Per-fold forest | AUROC, pAUC, sensitivity at each policy, FPR overshoot, prevalence, n per fold; an I²-style dispersion of per-fold AUROC | fig `fold_forest` | fold mean/std only |
| H2 | Seed spread | Per-seed metrics vs the ensemble (when there is more than one seed) | fig `seed_spread` | new |
| H3 | Shared-test sensitivity analysis | Pooled metrics with `first_fold` vs `exclude` | table | new |
| H4 | Validation vs test | All headline metrics side by side (optimism) | summary table | `validation_*` aggregates |

#### Block E — Error analysis and interpretation

| ID | Analysis | Definition | Outputs | Old |
|---|---|---|---|---|
| E1 | Top errors table | Per fold: the top-K (`eval.error_analysis.top_k`, default 10) false positives (healthy with the highest max score) and false negatives (positives with the lowest), with guid, subgroup, stage, max score, first-alarm time, n_segments, valid_frac, has_tlo | `evaluation/tables/errors.parquet` | new |
| E2 | Per-GUID trajectory pages | §11.10 page layout for a seeded sample per class and fold, plus E1 GUIDs. Selection follows `teb_vae/lag_attn_cfs/analyses/samples.py:691 per_class_rows` / `:753 extreme_rows`, ported. Manifest `pages.csv` | figs under `pages/` | new |
| E3 | Score vs signal quality | Score and error rate vs `valid_frac` deciles; FP rate by quality tertile | fig `score_vs_quality` | new |
| E4 | Score vs length | Score vs n_segments and span, by class: the visual check for the length shortcut next to the shortcut baseline | fig `score_vs_length` | new |
| E5 | Pooling attention summaries | Distribution of step-attention mass over the segment (early / late half, warm-up boundary), by class; per-GUID segment-attention (MIL/transformer) vs time to delivery | fig `attention_summary` | new |
| E6 | Feature attribution (optional) | `eval.attribution.enabled` (default false): captum Integrated Gradients of the GUID score w.r.t. the per-step features, averaged per channel group (`mu_prior`, `delta_mu`, …) and per class. Uses captum only here | fig `attribution_channels` | new |

#### Block B — Baselines and controls (§10.9)

| ID | Analysis | Definition | Outputs | Old |
|---|---|---|---|---|
| B1 | Baseline table | Probe, shortcut and shuffled control: AUROC with CI, sens at each policy | summary §4 | linear probe (fold 1 only) |
| B2 | Model vs probe | Paired ΔAUROC and Δsens@primary, with CIs | fig `model_vs_probe` | new |
| B3 | Warnings | Shortcut AUROC > `shortcut_warn_auroc`; shuffled AUROC > 0.6 | summary + verify | new |

#### Block Q — Model comparison (`run.py compare`, §11.8)

| ID | Analysis | Outputs |
|---|---|---|
| Q1 | Overlaid R1/R2 ROC and M1 curves for up to 6 runs | figs `compare_roc`, `compare_metric_types_<axis>` |
| Q2 | Paired ΔAUROC / Δsens forest overall and per subgroup family (paired GUID bootstrap, DeLong p, Nadeau–Bengio p) | fig `compare_forest`, table `comparison.parquet` |
| Q3 | Ablation table: one row per run, the headline metrics and key config differences (only keys that differ) | `comparison.md` |

### 11.13 Evaluation output tree

```
<run>/evaluation/
  summary.json            # VAE envelope (§11.14): results.headline, steps, n_failed, failed, exit_code, artifacts, ...
  steps.json              # heartbeat, rewritten after every analysis
  eval.log  resolved_config.yaml  provenance.json
  tables/
    metrics.parquet  thresholds.parquet  roc_points.parquet  subgroups.parquet  subgroup_tests.parquet
    subgroup_cutpoints.json  errors.parquet  alarms.parquet  inclusion.csv
  figures/                # pooled full set (stems from §11.12), one file per format
    cohort/  roc/  thresholds/  metric_types/  subgroups/  calibration/  confusion/  multiclass/
    alarms/  heterogeneity/  errors/  baselines/
    fold_<k>/…            # core set per fold (eval.per_fold_figures)
    val/…                 # validation versions ("used for selection")
  pages/                  # per-GUID trajectory pages + pages.csv manifest
summary.md                # at run root
```

`inclusion.csv` has one row per restricted analysis, with columns (analysis, split, fold, axis, n_included, n_excluded, reason counts). This is L14.

### 11.14 Evaluation execution conventions (aligned with the VAE eval pipeline)

These follow `teb_vae/lag_attn_cfs/eval/run.py` and `teb_vae/lag_attn/eval/report.py`.

- **Analysis registry.**
  - Every block in §11.12 is a function `run_<id>(ctx, *, eval_config, out_dir) -> dict`.
  - The return value holds `n_guids`, `composition` and `plan`, plus the metric rows the analysis appends.
  - Functions are registered in an ordered dict in `metrics.py` / `report.py`.
  - `--only` and `--skip` select analyses; unknown names are refused.
- **Fail-soft steps.**
  - Each analysis runs under `teb_vae/lag_attn/eval/report.py:794 Report.step`, which keeps the full traceback.
  - After each step, `steps.json` is rewritten (`report_seam.write_steps`).
  - A failed analysis never stops the others.
  - **Exit code 1 if and only if a step raised.** Validity failures (confound, controls, overshoot) go to the `sanity` block and are turned into refusals by `verify`, not by the exit code.
  - This replaces the old broad `except Exception` swallowing (A9): failures are recorded and counted, never hidden.
- **`summary.json` envelope** (like the VAE's):
  - `results.headline`: the flat table the gate reads (§11.9 `summary.json` content);
  - `results.<block>`: per-block results;
  - `steps`, `n_failed`, `failed`, `exit_code`;
  - `arguments.values/sources` (from `teb_vae/lag_attn_cfs/eval/launch.py:29 resolve_launch_args`);
  - `config_digest`, `numerics`;
  - `sanity`, `coverage` (populations that disagree across analyses);
  - `artifacts` (`build_manifest`: every file with its size, figures subset);
  - `run_context` (source fingerprint, checkpoint SHA, git SHA, env versions).
  - It is written with `json_safe(..., allow_nan=False)`.
- **Re-runs.**
  - A prior `summary.json` / `steps.json` is renamed to `*.bak.<stamp>.json` (`preserve_prior_summary` pattern).
  - Tables are re-read from `predictions/*.parquet` with a provenance sidecar check: the config and checkpoint digests must match, otherwise refuse (the `collect.py:1566` pattern).
- **Statistics helpers.**
  - `teb_vae/lag_attn/eval/stats.py`: `holm_adjust`, `kruskal_across_groups`, `pairwise_comparisons`, `wilcoxon_paired`, `windowed_group_comparisons`, `delta_magnitude`. Its `bootstrap_ci` (unstratified mean) is used **only** for means over GUIDs (lead time, alarm burden).
  - AUROC and rate CIs use the pilot's stratified cluster bootstrap (§11.7).
  - Every interval records `{method, resamples, seed, n, n_dropped}`.

### 11.15 Verify gate (`python -m teb_vae.classifier.verify <run>`)

**Design:**
- Modelled on `teb_vae/lag_attn_cfs/eval/verify.py`. It is torch-free, reads only `evaluation/summary.json`, `stage_state.json` and `manifest.json`, and has a `CRITERIA` registry.
- Each criterion returns PASS, FAIL or INCONCLUSIVE. **INCONCLUSIVE is never a pass**, but does not block.
- Exit 1 on any FAIL. `--json-out PATH` writes the verdicts.
- `--runs A B …` prints a markdown table of headline metrics per run, arms keyed on config differences.

**Criteria:**
1. `exit_code == 0`, i.e. no analysis step raised.
2. All planned units are `done` (`stage_state.json`); no failed units, unless the run was made with `--allow-partial` (then INCONCLUSIVE).
3. Cohort: within-fold disjointness passed; pretraining exposure has no test overlap (L2, L3).
4. `selection_lock.json` existed before every test prediction (lock timestamp < test prediction timestamp) (L5).
5. Headline metrics are finite.
6. NP-policy FPR overshoot: the pooled test FPR ≤ α + a binomial tolerance (95%, n_neg). Otherwise FAIL, naming the policy.
7. Shuffled-label control AUROC ≤ 0.60, and its CI includes 0.5.
8. Model AUROC > shortcut baseline AUROC (point estimate); INCONCLUSIVE if the CIs overlap.
9. committed_overall is monotone; the end-point consistency checks of M7 pass.
10. Missingness confound below threshold, or the `no_indicator` ablation present.
11. **Expected outputs.** `artifacts.figures` ⊇ `FIGURE_REGISTRY(config)`, and every table in §11.13 exists and is non-empty. This is stricter than the VAE gate, which does not check completeness.
12. Summary sections 1–12 of §11.11 are present in `summary.md`.

---

## 12. Leakage and validity guards (FR-13)

| # | Guard | Enforcement |
|---|---|---|
| L1 | Forbidden inputs (§7.1) never reach the model | Column allow-list in `data.py`. Test T-L1: permute or randomise `epoch_s`, `hours_to_delivery`, `cs`, `bg`, `source_file` and the negative part of `ss_rel` in a batch, and assert predictions are bit-identical |
| L2 | GUID (or patient) disjointness within folds | `check_split_disjoint` at cohort build. Hard error |
| L3 | VAE pretraining GUIDs ∩ classification test GUIDs = ∅ | §6.7. Hard error unless explicitly allowed; always in the manifest |
| L4 | Scalers, covariate vocabularies, class weights and priors fit on **train** only | Fit functions take a train-only frame and assert `split == 'train'` |
| L5 | Early stopping, calibration, thresholds and alarm-rule parameters fit on **val** only | Test predictions are written only after `selection_lock.json` (config digest + best checkpoint digest per fold/seed) exists, following `latent_pilot/config.py:1409-1492`. `--stage predict --split test` refuses otherwise |
| L6 | One window (`epoch_min_s`) for all splits | Config has a single key; no per-split override exists |
| L7 | Second-stage sentinel (onset exactly at delivery) treated as unknown | `cohort.py`, counted in the manifest |
| L8 | Segments crossing delivery excluded | `crosses_delivery` exclusion |
| L9 | Duplicate (guid, epoch) rows removed | `duplicate_epoch` exclusion |
| L10 | Missingness-confound check on TLO and covariates | §7.3; value in manifest and summary |
| L11 | Shortcut baseline and shuffled-label control run | §10.9. Their results appear in `summary.md` |
| L12 | Shared test GUIDs handled in pooled metrics | §11.7 |
| L13 | Stats-file trim equals loader trim; source trim equals classifier trim | Asserted at source construction (the pilot refuses mismatches, `latent_pilot/model.py:206-274`) |
| L14 | No evaluation-time cohort filtering without a reported count | The metric engine requires an `inclusion` record for every restricted analysis |

---

## 13. Configuration

All keys under the top-level `classifier:` key. Unknown keys are an error. Paths resolve against the repository root. The complete default:

```yaml
# teb_vae/classifier/configs/default.yaml
classifier:
  run:
    name: trf_cfs_adverse_seq
    out_root: runs/classifier
    seeds: [42]
    folds: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    device: cuda:0
    num_workers: 4
    fail_fast: false
    plot_frequency: 5               # → general_config.plot_frequency (ClassifierPlotCallback cadence)

  data:
    kfold_root: /path/to/k_fold_cross_validation_dataset   # [OPEN] placeholder
    fold_dir: "fold_{k}"
    split_dirs: {train: train, val: val, test: test}
    subgroups: all                  # or list of shard basenames
    stride_s: auto                  # inferred; error if multimodal
    epoch_min_s: -44640             # one window for every split (L6)
    min_valid_frac: 0.1
    patient_map: null               # JSON guid -> patient id
    shared_test_policy: first_fold  # first_fold | exclude
    allow_pretrain_overlap: false

  cohort:
    include_healthy_no_bg: true
    min_segments_per_guid: 1

  labels:
    task: adverse_vs_healthy        # §6.3
    head: binary                    # binary | multiclass | ordinal
    aux_3class_weight: 0.3          # λ3; 0 disables
    strategy: horizon_decay         # propagate | horizon | horizon_decay | final_only | mil
    horizon_h: 1.0
    decay_halflife_h: 0.5
    k_warm: 0
    eval_window: all                # all | horizon | stage:first | stage:second

  context:
    tlo: {enabled: true, missing: indicator}
    stage: {enabled: true}
    time_in_ss: {enabled: true}
    delta_t: {enabled: true}
    elapsed: {enabled: false}       # ablation only (§7.1)
    valid_frac: {enabled: true}
    covariates:
      static_csv: null
      timed_csv: null
      variables: []                 # [{name: temp_c, kind: numeric, available_at: prospective}]
      max_age_h: 2.0
      age_feature: true
      missing: indicator            # indicator | no_indicator
      dropout_p: 0.2
      block_dropout_p: 0.1
    fusion: late                    # concat | film | token | late (covariates)
    missing_confound_max: 0.10
    auto_ablate_missing: false

  source:
    kind: vae                       # vae | hdf5
    vae:
      package: lag_attn_transformer_cfs
      checkpoint: /path/to/best.ckpt
      keys:
        - {name: mu_prior}
        - {name: delta_mu}
        - {name: kld_per_t, transform: log1p, role: attention}
      step_support: causal_all      # causal_all | supervised
      lag_bins: [[0, 4], [5, 12], [13, 24], [25, 37]]
      sample_z_train: false         # frozen_online only
    hdf5:
      fields: [fhr_st, fhr_ph, up_st, up_ph]
      stats_path: /path/to/stats.hdf5
      trim_minutes: 1.0
      min_step: auto
    time_pool: 1
    cache_root: runs/classifier_cache
    cache_dtype: float16

  model:
    scope: sequence                 # segment | sequence
    step: {d: 64, temporal: none, dropout: 0.1}
    pooling: {kind: gated_attention, d_attn: 64}
    token: {d: 128, dropout: 0.1}
    sequence: {kind: causal_transformer, causal: true, layers: 2, heads: 4, d_ff: 256,
               dropout: 0.1, time_bias_buckets: 32, time_bias_max_h: 12.0}
    segment_head: true
    head: {hidden: 64, dropout: 0.1, bias_init: prior}
    segment_aggregators: [max, mean, lse, last, topk_mean]   # post-hoc GUID scores (§11.2)
    lse_tau: 1.0

  train:
    regime: frozen_cached           # frozen_cached | frozen_online | partial | lpft | cotrain
    unfreeze: []
    backbone_lr: 1.0e-5
    lpft_head_epochs: 30
    loss: {name: bce, weighting: none, beta_en: 0.999, focal_gamma: 2.0, focal_alpha: null,
           logit_adjust_tau: 1.0, label_smoothing: 0.0}
    loss_weights: {final: 1.0, positions: 0.5, bag: 0.0, segment: 0.3}
    sampler: natural                # natural | class_balanced
    batch_guids: 16
    batch_segments: 256
    segment_dropout: 0.1
    optimizer: {lr: 1.0e-3, weight_decay: 1.0e-2, betas: [0.9, 0.999]}
    schedule: {warmup_steps: 200, kind: cosine, min_lr_frac: 0.05}
    max_epochs: 150
    ema: 0.999                      # EMAWeightAveraging; null disables
    # grad clip, early stopping, checkpointing, MLflow: advanced_config (§13.1)
    cotrain:
      vae_weight: 1.0
      cls_weight: 1.0
      l2sp: 1.0e-4
      detach_head_input: false
      vae_chunk: 32
      grad_segments: all
      gates: {forecast_mse_rel: 0.10}

  calibration: {method: temperature}  # temperature | platt | none

  baselines:
    probe_last_n: 3
    shortcut_warn_auroc: 0.60
    shuffled_control: true

  eval:
    levels: [segment, guid, online]
    thresholds:
      # basis: guid_final | instantaneous | committed_cumulative | committed_overall | segment (§11.3)
      - {id: np30, policy: fpr_cap, alpha: 0.30, method: np_umbrella, delta: 0.05, allow_fallback: false,
         basis: committed_overall, axis: to_delivery, at: end}
      - {id: emp30, policy: fpr_cap, alpha: 0.30, method: empirical, basis: committed_overall, at: end}
      - {id: emp15, policy: fpr_cap, alpha: 0.15, method: empirical, basis: committed_overall, at: end}
      - {id: inst30_1h, policy: fpr_cap, alpha: 0.30, method: empirical, basis: instantaneous, at: 1.0}
      - {id: cum30_1h, policy: fpr_cap, alpha: 0.30, method: empirical, basis: committed_cumulative, at: 1.0}
      - {id: ovr30_1h, policy: fpr_cap, alpha: 0.30, method: empirical, basis: committed_overall, at: 1.0}  # previous v1 primary
      - {id: youden, policy: youden, basis: guid_final}
    primary_policy: np30
    report_tpr_at_fpr: [0.05, 0.10, 0.15, 0.20, 0.30]
    metric_types: [instantaneous, committed_cumulative, committed_overall]   # all evaluated for every policy
    exclude_last_min: 0
    alarm_rule: {kind: latch}       # latch | k_of_n {k, n}
    checkpoints_h: [6, 4, 3, 2, 1, 0.5]
    snapshot_max_staleness_h: 1.0
    time_axes: [to_delivery, from_onset, rel_second_stage, position, elapsed]
    bin_h: 0.5
    min_bin_class_n: 5
    subgroup_families: [class, source_file, class_x_cs, healthy_x_bg, healthy_bg_x_cs, cs, bg, stage_last,
                        reached_second_stage, has_tlo, admission_tlo_tertile, labour_duration_tertile,
                        n_segments_tertile, span_tertile, valid_frac_tertile, late_coverage, shared_test, fold]
    restricted_pairs: [[healthy, acidosis], [healthy, hie], [acidosis, hie]]
    min_subgroup_n: 10
    ovr_thresholds: auto            # auto = true for three_class tasks
    decision_horizons_h: [0.5, 1, 2, 4, 6]
    per_fold_figures: true
    fold_band: minmax               # minmax | none
    figure_formats: [pdf]           # add png if wanted
    error_analysis: {top_k: 10}
    attribution: {enabled: false, n_steps: 32}
    covariates_off: false
    bootstrap: {resamples: 2000, refit_threshold: false, seed: 0}
    reference_prevalence: null
    trajectory_pages: {per_class: 3, top_errors: 3}
```

### 13.1 `advanced_config` (framework block, same schema as the VAE families)

This block sits next to `classifier:` in the same YAML file. `GraphModelBase.validate_config` checks it (`train/graph_model_base.py:289-388`), and it is copied verbatim into every unit config.

```yaml
advanced_config:
  trainer:
    precision: 32-true
    gradient_clip_val: 1.0
    gradient_clip_algorithm: norm
    deterministic: true
    log_every_n_steps: 10
    num_sanity_val_steps: 2
    profiler: null
  logging: {level: INFO, json_log: true, rotation: 50 MB, retention: 5}
  spike_breaker: {enabled: false}
  tracking:
    mlflow:
      enabled: false
      tracking_uri: http://localhost:5000
      experiment_name: ctg-classifier
      log_model: false              # F10
      log_checkpoints: false
      log_config_artifact: true
      tags: {}
  callbacks:
    early_stopping: {enabled: true, monitor: val/guid_logloss, mode: min, patience: 25, min_delta: 0.0}
    model_checkpoint: {monitor: val/guid_logloss, mode: min, save_last: true}
    classifier_plotting: {enabled: true, every_n_epochs: 5, file_format: pdf,
                          train_eval_every: 5, train_eval_guids: 256}
```

Verify every key against `_KNOWN_TRAINER_KEYS` / `_KNOWN_ADVANCED_BLOCKS` (`graph_model_base.py:45, 56`) while implementing. Test T-F1 fails on any `config:` warning.

### 13.2 Unit config derivation (`config.py`)

For each unit (fold k, seed s, kind), `config.py` writes a framework-shaped YAML that `GraphModelBase` can load:

```yaml
general_config:
  tag: <run.name>-fold<k>-seed<s>[-shuffled]
  seed: <s + 1000·k>
  cuda_devices: [<index from run.device>]    # [] + accelerator override for cpu (F5)
  epochs: <train.max_epochs>
  lr: <train.optimizer.lr>
  lr_milestone: []
  plot_frequency: <run.plot_frequency>
  batch_size: {train: <train.batch_guids | batch_segments>, test: <same>}
  folders_config: {out_dir_base: <run>/folds/fold_<k>/seed_<s>}
model_config:
  classifier: <the full resolved classifier: block>   # flattened into MLflow params (§10.10.5)
advanced_config: <§13.1 block, with tracking.mlflow.run_name/tags set per unit>
```

The strict pydantic schema validates `classifier:`. The base validates `general_config` and `advanced_config`.

### 13.3 Shipped configs

**Other shipped configs** each have `base: default.yaml` plus overrides:
- `smoke.yaml`: fixtures, CPU, 2 folds, tiny budgets.
- `st_ph.yaml`: `source.kind: hdf5`.
- `segment_scope.yaml`.
- `three_class.yaml`: `task: three_class`, `head: multiclass`, `weighting: sqrt_inverse`.
- `cotrain.yaml`.

---

## 14. CLI, run layout, provenance and tracking (`run.py`)

### 14.1 CLI

```
python -m teb_vae.classifier.run --config teb_vae/classifier/configs/default.yaml \
       --stage {cohort,extract,train,predict,evaluate,report,verify,all} \
       [--folds 1,2] [--seeds 42,43] [--run-dir PATH] [--set key.path=value ...] [--device cuda:0] \
       [--only R1,M1,...] [--skip E6,...] [--allow-partial]
python -m teb_vae.classifier.verify RUN_DIR [--json-out PATH] [--runs RUN_A RUN_B ...]
python -m teb_vae.classifier.run compare --runs RUN_A RUN_B [--level guid] [--policy np30]
```

- `--set` values are parsed as YAML (dotted `key=value`; the parser pattern of `teb_vae/lag_attn_cfs/lag_recovery_check.py:213`). Run-shaping settings belong in YAML; `--set` is for quick experiments and is recorded in `summary.json` `arguments.sources`.
- A module-level `RUN_ARGS` dict supports the IDE Run button, as in `latent_pilot/run.py:1855`, with the same guarded repo-root `sys.path` bootstrap.

**Stages:**

| Stage | What it does |
|---|---|
| `cohort` | tables and fold checks (§6) |
| `extract` | feature cache; frozen_cached only |
| `train` | per fold × seed: model and baselines; writes `best.ckpt`, `calibration.json`, `thresholds.json`, then writes `selection_lock.json` |
| `predict` | val and test prediction rows; test requires the lock |
| `evaluate` | every analysis of §11.12 (tables), under fail-soft steps (§11.14) |
| `report` | figures (`FIGURE_REGISTRY`), `summary.md`, MLflow parent logging |
| `verify` | the gate of §11.15 |

- Each stage is idempotent and skips completed (fold, seed) units whose settings digest matches.
- A digest mismatch in an existing run directory is an error unless `--run-dir` is new.
- Folds run sequentially.
  - `ponytail:` no multi-GPU fold parallelism; add a subprocess pool only if wall-clock matters.

### 14.2 Run directory

```
<out_root>/<YYYY-mm-dd--HH-MM-SS>-<name>/
  config.resolved.yaml   manifest.json   stage_state.json   run.log   run.jsonl
  kfold_progress.log   kfold_summary.json   execution_metadata.json
  cohort/  segments.parquet  guids.parquet  fold_summary.csv  dataset_summary.json  label_cross_table.csv
           count_spread.csv  exposure.json  confound.json
  folds/fold_<k>/seed_<s>/                     # one training unit (§10.10.4)
      train_results/  full.log  run.jsonl  metrics_history.csv  epoch_summary.jsonl  loss_plot_epoch.html
                      hyperparameters.html  classifier_diagnostics/  stage_transitions.jsonl
      model_checkpoints/  best.ckpt  last.ckpt  resolved_config.yaml
      setup.json  scaler.json  calibration.json  thresholds.json  selection_lock.json  fold_results.json
      shuffled/…                               # shuffled-label control unit (seed 0 only)
  folds/fold_<k>/ens/  calibration.json  thresholds.json          # when >1 seed
  baselines/fold_<k>/  probe.skops  shortcut.skops  probe_fit.json  shortcut_fit.json
  predictions/  segments.parquet  guids.parquet  provenance.json
  evaluation/   (§11.13)
  summary.md
```


**`manifest.json`:**
- git SHA and dirty flag;
- config digest;
- source fingerprint;
- checkpoint SHA-256;
- shard paths with size, mtime and `source_guid_digest` attributes;
- environment versions;
- cohort counts and exclusions;
- exposure result;
- confound check results;
- baseline warnings.

`skops` is installed; use it, not pickle, for sklearn models.

### 14.3 Tracking

- Always: loguru per unit and per run (§10.10.6), `metrics_history.csv` and `epoch_summary.jsonl` per unit, and `kfold_progress.log` per run.
- When `advanced_config.tracking.mlflow.enabled`: a parent run with nested child runs per unit, plus the evaluation metrics and artifacts on the parent. The full design is §10.10.5. No autolog (§2.8).


---

## 15. Testing (`tests/`, pytest from the repository root, hermetic)

**Fixtures:**
- `tests/fixtures/make_fixture.py` generates a tiny k-fold tree: 3 folds × {train, val, test}, 4 subgroup files, about 60 GUIDs, and variable segment counts with gaps.
- It uses a **660-s stride** and a **planted signal**: positives have a class-dependent shift in the ST channels, in segments within 1 h of delivery.
- It includes TLO and second-stage values with NaNs, one second-stage sentinel, a duplicate epoch, and one shared test GUID across folds.
- Reuse `scripts/make_tiny_shard.py` / `latent_pilot/tests/fixtures/generate.py` for the HDF5 writing.
- A tiny VAE checkpoint comes from the trf_cfs test fixtures (`teb_vae/lag_attn_transformer_cfs/tests/conftest.py`).

**Required tests** (one file per module; known-answer where possible):

| ID | Test |
|---|---|
| T-C1 | Segment table on the fixture: slot inference (660 s), stage classification at the trim-aware boundaries, sentinel → unknown, exclusions counted |
| T-C2 | Task mappings and exclusions; GUID label consistency error |
| T-C3 | Label strategies: ω values for known Δ (horizon, decay half-life, k_warm) |
| T-C4 | Fold checks: overlap raises; shared test detected; pretraining overlap raises |
| T-C5 | Covariate as-of join is causal: an observation after `t_end` is never used; max_age respected |
| T-S1 | `VaeSource` on the tiny checkpoint: shapes, step mask (warm-up, weight, support), derived keys (`delta_mu = post − prior`, `attn_summary` masses sum to 1) |
| T-S2 | `Hdf5Source`: cold-cell masking, channel counts read from the file |
| T-S3 | Cache: extraction once per unique segment; fingerprint mismatch refuses; row alignment |
| T-S4 | Scaler: train-only assertion; recording weighting; floor |
| T-M1 | No NaN with fully padded sequences, or with a single-segment GUID |
| T-M2 | Causal online outputs equal an explicit prefix sweep (atol 1e-5) |
| T-M3 | CORAL biases stay ordered; P(Y > k) is monotone in k |
| T-L1 | Forbidden-input invariance (§12, L1) |
| T-L2 | Test prediction without `selection_lock.json` raises |
| T-T1 | Empirical FPR cap: validation FPR ≤ α; maximal sensitivity; strict `>` with ties |
| T-T2 | NP umbrella: k* matches a brute-force binomial sum on a grid of (n, α, δ); raises when n < n_min; simulated population FPR > α with frequency ≤ δ (+ MC tolerance) |
| T-T4 | Metric types on a hand-built 4-GUID example with known answers. Instantaneous uses raw (unlatched) decisions and one row per GUID per bin. Committed cumulative uses the available denominator. Committed overall is monotone and equals the running-max GUID decision at `end`. The −∞ padding for the `committed_overall` basis gives ⌊α·n_all⌋ alarms |
| T-T3 | Running-max latch: "alarmed by n" ⇔ r_g(n) > thr; lead time computed from the first crossing |
| T-E1 | Metric engine matches sklearn on random data (AUROC, AP, pAUC McClish, Brier, kappa) |
| T-E2 | Pooled confusion equals the sum of per-fold confusions; shared-test dedupe |
| T-E3 | Bootstrap reproducible given seed; stratified draws keep both classes; undefined draws counted |
| T-E4 | Prevalence re-weighting formula (known answer) |
| T-F1 | Config load: every `configs/*.yaml` → resolved → unit config derivation → `make_graph_model` → `validate_config`, with a loguru sink asserting **no** `config:` warnings (the pattern of `teb_vae/lag_attn_transformer_cfs/tests/test_config_load.py:254-323`). Also the pydantic schema rejects unknown `classifier:` keys |
| T-F2 | Train smoke (`@pytest.mark.slow`, CPU, 2 epochs, MLflow disabled): every `TRACKED_METRICS` name is present and not all-NaN in `metrics_history.csv`; `best.ckpt` reloads via `check_model_class` + `load_checkpoint_strict`; `epoch_summary.jsonl` has one line per epoch |
| T-F3 | Callback order: `GuidEpochMetricsCallback` is first, and `val/guid_logloss` of epoch e is in the CSV row of epoch e (not e+1) |
| T-F4 | MLflow nesting with `FakeMLflowLogger`/`FakeMLflowExperiment` (`train/test_utils.py:86-108`): unit configs carry `mlflow.parentRunId`, `fold`, `seed`, `kind`; `log_model` is false; the parent is fail-closed when the client raises |
| T-F5 | Per-unit teardown: after 2 units in one process, no `atexit` upload is still registered and no model tensors remain on the device |
| T-V1 | Verify gate: a synthetic `summary.json` hits each criterion (PASS/FAIL/INCONCLUSIVE); a missing registry figure → FAIL |
| T-E5 | Subgroup membership known answers on the fixture (every family of §11.6.1; empty `acidosis_x_bg` documented, not computed); healthy-only groups report spec, unhealthy-only report sens |
| T-E6 | ROC vertical average and pooled ROC on synthetic folds; R2/R3 populations (available vs −∞ padding) |
| T-E7 | Fail-soft: an analysis that raises → `n_failed = 1`, other analyses complete, exit code 1, traceback in `summary.json` |
| T-X1 | **Smoke, end-to-end** (`smoke.yaml`, CPU, < 3 min): all stages including `verify`; planted-signal GUID AUROC > 0.9; shuffled control < 0.65; linear probe runs; `summary.md` produced; **every figure in `FIGURE_REGISTRY` produced**; verify exits 0 |
| T-X2 | Same smoke with `source.kind: hdf5` (ST/PH) |
| T-X3 | Resume: re-running `train` skips completed units; changed config digest errors |

---

## 16. Implementation plan (phases and acceptance criteria)

Each phase ends with its tests green and a short `CHANGELOG` entry at the bottom of this file. Do not start a phase before the previous one is accepted.

**Progress tracker.** Update this table when a phase's acceptance criteria are met: set Status to `done`, fill Date, and add a `CHANGELOG` entry. Use `in progress` while working on a phase.

| Phase | Scope | Status | Date |
|---|---|---|---|
| P0 | Config, cohort, folds | todo | |
| P1 | Feature sources and cache | todo | |
| P2 | Evaluation engine and baselines | todo | |
| P3 | Neural segment scope + training framework | todo | |
| P4 | Sequence scope and online evaluation | todo | |
| P5 | Context, covariates, labeling strategies | todo | |
| P6 | 3-class, full analysis catalogue, verify | todo | |
| P7 | Adaptation regimes | todo | |
| P8 | Seeds, ensembles, comparison | todo | |

**P0 — Config, cohort, folds (§5.6, §6, §7.1-7.2, §12 L2/L3/L6-L9).**
- *Deliverables:* `config.py`, `cohort.py` (without covariates), `run.py --stage cohort`, and the fixture generator.
- *Accept:* T-C1–T-C4 pass. Running on `tmp/data/hie_cs.hdf5` (as a one-file split) produces a segment table with stride 660, 15 GUIDs and 614 segments, the trim-aware stage counts **170 first / 4 straddle / 44 second / 396 unknown**, and a TLO NaN rate of **13.5%**. Remember the sample is one subgroup only.

**P1 — Feature sources and cache (§8).**
- *Deliverables:* `sources.py`, and `--stage extract`.
- *Accept:* T-S1–T-S4. Extraction on real shards reports throughput and cache size. `Hdf5Source` also works.

**P2 — Evaluation engine and baselines, end to end (§10.9, §11.1-11.4, §11.7, §11.9).**
Built before any neural model, so that every later model is measured by the same code.
- *Deliverables:* `baselines.py` (probe, shortcut), `thresholds.py`, and `metrics.py`, covering GUID and segment levels, pooled and per-fold, and bootstrap. The evaluation execution conventions (§11.14: registry, `Report.step`, `steps.json`, the `summary.json` envelope), and blocks C, R1/R6/R9, T1/T2, B1 of §11.12. The minimum `report.py` (figure seam) and a minimal `verify.py` (criteria 1, 3, 5, 11).
- *Accept:* T-T1, T-T2, T-E1–T-E4. First real 10-fold report with the linear probe and shortcut baseline.

**P3 — Neural segment scope (§9.1-9.4, §9.6, §10.1-10.5, §10.8).**
- *Deliverables:* `data.py` SegmentDataset, `model.py` (step encoder, pooling, fusion, heads), `losses.py` (bce, weighted, focal, ce, weighting, prior correction), and calibration. **Framework integration (§10.10):** `ClassifierTask`, `ClassifierTrainer`, the outer unit loop with teardown, callbacks 1–10, `TRACKED_METRICS`, per-unit files, MLflow parent/child, `advanced_config` and unit-config derivation (§13.1–13.2). The shuffled-label control unit.
- *Accept:* T-M1, T-M3, T-L1, T-L2, T-F1–T-F5, T-X1 (segment scope). On real data, report the segment-scope model against the probe. An MLflow parent with 10 child runs is visible when enabled.

**P4 — Sequence scope and online evaluation (§9.5, §10.3-10.4, §11.2, §11.5).**
- *Deliverables:* GuidDataset and collate, the aggregators, and the loss composition. Blocks M, A, R2–R5, R7, R8, T3–T4 of §11.12.
- *Accept:* T-M2, T-T3, T-T4, T-X1 (sequence scope). The three metric types appear on every axis in the smoke report.

**P5 — Context, covariates and labeling strategies complete (§6.5-6.6, §7.3).**
- *Deliverables:* all strategies, the covariate join, dropout, the confound check, fusion modes, `eval.covariates_off`.
- *Accept:* T-C5, T-C3. The ablation report works.

**P6 — 3-class and ordinal, full analysis catalogue, figures, summary and verify (§6.3, §9.6, §11.4, §11.6, §11.10–11.15).**
- *Deliverables:* blocks S, K, X, H, E (E6 optional), R10–R13, Q; the full `FIGURE_REGISTRY`; the complete `verify.py`; the `compare` figures.
- *Accept:* T-E5–T-E7, T-V1. Every registry figure renders on the smoke run. `summary.md` is complete, including the TRIPOD+AI mini-checklist. Verify exits 0 on smoke.

**P7 — Adaptation regimes (§10.1, §10.6).**
- *Deliverables:* `frozen_online`, `partial`, `lpft`, `cotrain`, preservation gates, `kld_excess`.
- *Accept:* contract test on the tiny checkpoint for gradient reach and for no gradient outside the allowlist; the co-train smoke step runs; the gate rejects a synthetic degradation.

**P8 — Seeds, ensembles and comparison (§10.7, §11.8).**
- *Deliverables:* multi-seed ensemble, `compare` subcommand, DeLong, Nadeau–Bengio, and the refit bootstrap.
- *Accept:* T-X3. `compare` on two smoke runs.

---

## 17. Recommended experiment plan

This runs after P4, using `frozen_cached` unless noted. Every row is reported against the linear probe and the shortcut baseline, on the same folds.

1. **Source ablation** (sequence scope, default everything else):
   - `mu_prior`
   - + `delta_mu`
   - + `kld_per_t` (attention role)
   - `mu_post`
   - `target_state`
   - `attn_summary`
   - `kld_excess` (P7)
   - ST/PH direct (`st_ph.yaml`)
   - ST/PH + `mu_prior`
2. **Scope:** segment (max / lse aggregators) vs sequence (transformer / GRU / attention_mil).
3. **Labels:** `propagate` vs `horizon` vs `horizon_decay` vs `mil` vs `final_only`; H ∈ {0.5, 1, 2} h.
4. **Context:** none vs TLO vs TLO + stage vs + time_in_ss; with and without the missing indicator; `elapsed` (shortcut probe).
5. **Loss:** bce vs weighted_bce vs focal (α explicit); for 3-class, ce + sqrt_inverse vs coral; `pauc` fine-tune (optional).
6. **Tasks:** adverse_vs_healthy (primary), hie_vs_rest, three_class.
7. **Adaptation (P7):** best frozen config vs partial (top block) vs lpft vs cotrain (with detach on and off).
8. **Seeds:** the final config with 5 seeds, as an ensemble.

**Reporting:** the primary endpoint is set in advance (**[OPEN-5]**; default: pooled OOF test sensitivity at `np30`, GUID offline, adverse_vs_healthy), plus AUROC per-fold mean ± SD. Everything else is secondary.

---

## 18. Open questions for the project owner

Each has a default, so implementation is not blocked.

| # | Question | Default used by this spec |
|---|---|---|
| OPEN-1 | "22 segments per GUID": is 22 meant as a fixed number of slots (e.g. the last 22 × stride), or was this the 22-min segment length? | Variable N_g, all retained segments in the 12.4 h window. A `data.max_segments` (keep the last N) can be added trivially if wanted |
| OPEN-2 | Which stride will the production k-fold shards use: 660 s (tiles post-warm-up windows) or 1200 s (~45% of time unseen by VAE features)? | Stride is inferred. The report prints the coverage fraction |
| OPEN-3 | Clinical covariates: which variables, from which table, static or timed, and their time reference? | Hooks only (§7.3); none enabled |
| OPEN-4 | Is there a patient ID (several GUIDs per mother or pregnancy)? | GUID = patient |
| OPEN-5 | Primary endpoint and FPR-cap method: NP umbrella (guaranteed, conservative) or empirical (comparable to literature, overshoots ~50%)? And δ? Which metric type and time point carries the cap: committed overall at end of recording, or 1 h before delivery as in the previous pipeline? | Primary `np30` (α = 0.3, δ = 0.05, committed overall @ end), with `emp30`, `emp15`, `inst30_1h`, `cum30_1h`, `ovr30_1h` (previous v1 primary) also reported. Note that 0.3 is about twice clinical practice (≈ 0.15) |
| OPEN-6 | Is healthy < acidosis < HIE a valid ordinal scale for this cohort? `latent_pilot` explicitly forbids treating codes as ordinal | `multiclass` default; `ordinal` available as an ablation |
| OPEN-7 | Should `cs_outcome` be a studied task? | Available, not run by default |
| OPEN-8 | Keep healthy no-BG GUIDs (no blood-gas confirmation; manipulated TLO rate)? | Kept; `class` × `bg` subgroup reported; `include_healthy_no_bg: false` ablation recommended |
| OPEN-9 | Compute budget: how many folds × seeds × configs are affordable? | 10 folds × 1 seed for ablations; 5 seeds for the final config |
| OPEN-10 | Reconcile `requirements.txt` with the venv (Python 3.14 / sklearn 1.9 / torch 2.14 / pandas 3)? | Code targets the venv. No new dependencies |
| OPEN-11 | A re-split mode (`StratifiedGroupKFold` on GUID or patient) for cohorts without predefined folds? | Deferred. If added, use sklearn ≥ 1.8 (shuffle bug fixed) and check per-fold class counts |

---

## 19. References

Full discussion is in [`RESEARCH.md`](RESEARCH.md). The references that ground decisions in this spec:

**CTG literature**
- Georgieva et al. 2017, OxSys 1.5, *AOGS*. https://obgyn.onlinelibrary.wiley.com/doi/10.1111/aogs.13136
- Mendis et al. 2024, FHR-LINet / MCNN operating points. https://pmc.ncbi.nlm.nih.gov/articles/PMC11144251/
- Mendis et al. 2025, cross-database CTG, *IEEE JTEHM*. https://pmc.ncbi.nlm.nih.gov/articles/PMC12250915/
- McCoy et al. 2025, InceptionTime, *AJOG*. https://pmc.ncbi.nlm.nih.gov/articles/PMC11499302/
- Ben M'Barek et al. 2025, DeepCTG 2.0. https://lepennec.perso.math.cnrs.fr/Reprint/Health/2024-CBM-BMJMKSCLPS.pdf
- Asfaw et al. 2023, early-labour CNN (Oxford). https://pmc.ncbi.nlm.nih.gov/articles/PMC10294944/
- O'Sullivan et al. 2021, review of AI for CTG. https://pmc.ncbi.nlm.nih.gov/articles/PMC8576107/
- Fridman & Ben Shachar 2026, SSL PatchTST for CTG (preprint). https://arxiv.org/abs/2601.06149
- Chudáček et al. 2014, CTU-UHB database. https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/

**Representation → classifier**
- Kumar et al. 2022, LP-FT. https://arxiv.org/abs/2202.10054
- Ilse et al. 2018, attention-based MIL. https://arxiv.org/abs/1802.04712
- Early et al. 2024, MILLET. https://arxiv.org/abs/2311.10049
- Lee et al. 2019, Set Transformer. https://arxiv.org/abs/1810.00825
- Shukla & Marlin 2021, mTAN. https://arxiv.org/abs/2101.10318
- Horn et al. 2020, SeFT. https://arxiv.org/abs/1909.12064
- Che et al. 2018, GRU-D. https://arxiv.org/abs/1606.01865
- Perez et al. 2018, FiLM. https://arxiv.org/abs/1709.07871
- Neverova et al. 2016, ModDrop. https://arxiv.org/abs/1501.00102
- Sisk et al. 2023, missing indicators. https://arxiv.org/abs/2206.12295
- Yèche et al. 2023, temporal label smoothing. https://arxiv.org/abs/2208.13764
- Kvamme & Borgan 2019, discrete-time survival. https://arxiv.org/abs/1910.06724
- Kingma et al. 2014, semi-supervised VAE (M2). https://arxiv.org/abs/1406.5298
- Bowman et al. 2016, posterior collapse. https://arxiv.org/abs/1511.06349
- Nalisnick et al. 2019, generative likelihood and OOD. https://arxiv.org/abs/1810.09136

**Losses, imbalance, calibration**
- van den Goorbergh et al. 2022, *JAMIA*. https://academic.oup.com/jamia/article/29/9/1525/6605096
- Carriero et al. 2025, *Stat Med*. https://onlinelibrary.wiley.com/doi/full/10.1002/sim.10320
- Lin et al. 2017, focal loss. https://arxiv.org/abs/1708.02002
- Cui et al. 2019, class-balanced loss. https://arxiv.org/abs/1901.05555
- Menon et al. 2021, logit adjustment. https://arxiv.org/abs/2007.07314
- Cao, Mirjalili & Raschka 2020, CORAL. https://arxiv.org/abs/1901.07884
- Shi, Cao & Raschka 2023, CORN. https://arxiv.org/abs/2111.08851
- Yuan et al. 2021, AUC-M. https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Large-Scale_Robust_Deep_AUC_Maximization_A_New_Surrogate_Loss_and_ICCV_2021_paper.html
- Zhu et al. 2022, pAUC optimisation. https://proceedings.mlr.press/v162/zhu22g.html
- Guo et al. 2017, temperature scaling. https://arxiv.org/abs/1706.04599
- Saerens et al. 2002, prior-shift correction. https://doi.org/10.1162/089976602753284446
- Van Calster et al. 2019, calibration hierarchy. https://doi.org/10.1186/s12916-019-1466-7
- Austin & Steyerberg 2019, ICI. https://doi.org/10.1002/sim.8281
- Nixon et al. 2019, ECE pitfalls. https://arxiv.org/abs/1904.01685

**Evaluation**
- Tong, Feng & Li 2018, Neyman–Pearson umbrella algorithm, *Sci Adv*. https://www.science.org/doi/10.1126/sciadv.aao1659
- Forman & Scholz 2010, "Apples-to-apples in cross-validation". https://dl.acm.org/doi/10.1145/1882471.1882479
- Bates, Hastie & Tibshirani 2024, CV confidence intervals. https://arxiv.org/abs/2104.00673
- Varma & Simon 2006. https://doi.org/10.1186/1471-2105-7-91
- Cawley & Talbot 2010. https://jmlr.org/papers/v11/cawley10a.html
- Van Calster et al. 2024, STRATOS performance measures. https://arxiv.org/abs/2412.10288
- Vickers & Elkin 2006, decision curves. https://doi.org/10.1177/0272989X06295361
- DeLong et al. 1988. https://doi.org/10.2307/2531595
- Nadeau & Bengio 2003. https://doi.org/10.1023/A:1024068626366
- Field & Welsh 2007, cluster bootstrap. https://doi.org/10.1111/j.1467-9868.2007.00593.x
- Brown, Cai & DasGupta 2001, Wilson intervals. https://doi.org/10.1214/ss/1009213286
- Hand & Till 2001, multiclass AUC. https://doi.org/10.1023/A:1010920819831
- Lauritsen et al. 2021, early-warning framing. https://pmc.ncbi.nlm.nih.gov/articles/PMC8593052/
- Hyland et al. 2020, ICU alarm metrics. https://www.nature.com/articles/s41591-020-0789-4
- Tomašev et al. 2019, alarms per detection. https://doi.org/10.1038/s41586-019-1390-1
- Collins et al. 2024, TRIPOD+AI, *BMJ*. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11025451/
- Moons et al. 2025, PROBAST+AI, *BMJ*.
- Riley et al. 2024, sample size for prediction models. https://pubmed.ncbi.nlm.nih.gov/38253388/

**Libraries**
- scikit-learn 1.8/1.9 (`StratifiedGroupKFold` shuffle fix #32478; `metric_at_thresholds`; `CalibratedClassifierCV(method='temperature')`; `FrozenEstimator`).
- scipy (`binom`, `bootstrap`, `binomtest`).
- torchmetrics (monitoring only).
- CORAL reference: https://github.com/raschka-research-group/coral_pytorch.
- MAPIE `BinaryClassificationController` is an optional cross-check for FPR control, not a dependency.

**Internal documents**
- `teb_vae/lag_attn_transformer_cfs/{DESIGN.md, DIAGNOSIS.md, LAG_READOUT_DIAGNOSIS.md, LATENT_CLASS_FINETUNING_PILOT.md}`
- `hdf5_dataset/dataset_explained_research.md`
- previous classifier: `tmp/new_classifier/{PRD.md, classifier_description.md, possible_improvements.md, guid_cls_v1/IMPLEMENTATION.md}` (reference only; temporary folder)

---

## Appendix A — Previous-pipeline defects not to repeat

| # | Defect | Where (tmp/new_classifier) |
|---|---|---|
| A1 | `fill_missing_epochs` drops off-grid real rows and shifts alarms | `evaluate_classifier.py:574-737` |
| A2 | Test specificity always 0.0 (missing key in the summary) | `evaluate_classifier.py:3597-3606, 4548` |
| A3 | Validation "accuracy" silently became balanced accuracy (key mismatch) | `evaluate_classifier.py:4457-4461` |
| A4 | Subgroup CSV lost all counts (column-name mismatch) | `guid_cls_v1/evaluate_guid_classifier.py:1379-1394` |
| A5 | Committed ROC operating-point marker misaligned | `evaluate_classifier.py:3813-3849` |
| A6 | "Best checkpoint" = most recently modified file; v0 k-fold evaluated last-epoch weights | `evaluate_classifier.py:141-153`; `kfold_classifier_trainer.py:391, 466, 517` |
| A7 | Threshold search silently falls back to 0.5 on NaN | `evaluate_classifier.py:863-893` |
| A8 | Validation threshold search filled gaps; test evaluation did not | `clinical_metrics_utils.py:843` vs `evaluate_guid_classifier.py:1325` |
| A9 | Broad `except Exception` hid missing report sections | `evaluate_guid_classifier.py:1426, 1558, 1599` |
| A10 | Signed second-stage feature and zero-sentinel GUIDs in training leaked time-to-delivery | `guid_cls_v1/guid_dataset.py:91-104` |
| A11 | Latent statistics fitted on an unshuffled first-N-batches subset; the cap was not in the cache signature | `guid_cls_v1/precompute_latents.py:616-619, 1005` |
| A12 | Class weights counted padded rows as healthy | `kfold_classifier_trainer.py:212-214` |
| A13 | Weight decay on biases, norms and the prior-initialised bias | `guid_cls_v1/lightning_module.py:783-789` |
| A14 | `np.interp` held edge values flat when averaging time curves across folds | `guid_cls_v1/aggregate_results.py:328-410` |
| A15 | Headline numbers averaged over time bins | `evaluate_classifier.py:3588-3606` |

## Appendix B — Formulas

- **Stage at segment end:** see §2.4. **Time in second stage** = `max(ss_rel + 1260, 0)`.
- **Label decay:** `ω = 2^{−max(0, Δ − H)/h}`.
- **LSE pooling:** `τ·log( (1/N) Σ exp(s_n/τ) )`.
- **Gated attention:** `a_t = softmax_t( wᵀ[tanh(V u_t) ⊙ σ(U u_t)] )` over valid t.
- **CORAL:** `P(Y > k | x) = σ(g(x) + b_k)`, with `b_1 ≥ b_2 ≥ …` enforced by `b_k = b_1 − Σ_{j<k} softplus(δ_j)`. The class probabilities are differences of consecutive cumulative probabilities.
- **NP umbrella:** `k* = min{k ∈ 1..n : Σ_{j=k}^{n} C(n,j)(1−α)^j α^{n−j} ≤ δ}`, `thr = v₍k*₎`. The minimum n is `⌈log δ / log(1−α)⌉`.
- **McClish pAUC:** `½·(1 + (A_p − A_min)/(A_max − A_min))`, with `A_min = α²/2` and `A_max = α`; this is what `roc_auc_score(max_fpr=α)` computes.
- **Net benefit:** `NB = TP/N − (FP/N)·p_t/(1 − p_t)`.
- **Prevalence-adjusted PPV:** `sens·π / (sens·π + (1 − spec)(1 − π))`.
- **Prior-shift correction:** `logit p' = logit p − logit π_train + logit π_target`. After class weighting with w: `s' = s − log(w₁/w₀)`.
- **Nadeau–Bengio corrected variance:** `(1/k + n_test/n_train)·σ̂²`. For k-fold, n_test/n_train = 1/(k−1).

---

## CHANGELOG

- 2026-09-28 — v1.2: added training framework integration (§10.10: `ClassifierTask`/`ClassifierTrainer`, callbacks, `TRACKED_METRICS`, MLflow parent/child, gotchas), the full subgroup catalogue (§11.6), figure conventions (§11.10), the complete analysis catalogue (§11.12), the evaluation output tree (§11.13), execution conventions (§11.14), the verify gate (§11.15), `advanced_config` and unit-config derivation (§13.1–13.2), and tests T-F*/T-V1/T-E5–7.
- 2026-09-28 — v1.1: added the big picture (§5.1–5.5), the three metric types (§11.5.1–11.5.2), and threshold bases per metric type (§11.3).
- 2026-09-28 — v1.0 spec written (research: `RESEARCH.md`; code survey of `teb_vae/`, `hdf5_dataset/new_pipeline/`, `latent_pilot/`, `tmp/new_classifier/`).
