# lag_attn_cfs / lag_attn_transformer_cfs / lag_attn_crws / lag_attn_transformer_crws — reference for LLM agents

Dense reference, written 2026-09-04; CFS interpretation corrections added 2026-09-05. Repo root = this repository. Implementation pointers use `path:line` (line numbers as of the original writing date; if a pointer drifts, grep the named symbol). §1 = causal feature-stream (cfs) models + the full data pipeline they share. §2 = causal raw-waveform-stream (crws) models as a delta over §1. Read §1.0 before §1.1. Code determines current behavior; its comments and this document are not independent evidence of mathematical correctness. Known limitations and stale statements are listed in §1.13 / §2.8.

Domain: intrapartum fetal monitoring. FHR = fetal heart rate (bpm, 4 Hz), UP = uterine pressure (mmHg, 4 Hz). Every model here has a 30-step nominal forecast horizon and target-only/source-conditioned branches. For CFS, the actual forecast labels are FHR-derived coefficients selected by the target clock: the current `physical` mode reads some labels as late as 460 s after the anchor, not just the next 120 s of stored coefficients. The per-step KL is a source-conditioned latent divergence; attention partitions it into a lag attribution, not an identified transfer-entropy or physiological-delay measurement. `pred_gap` is the base-minus-full score gap; its training-path and matched Monte Carlo versions use different decoding policies (§1.8 / §1.11).

---

## 1. lag_attn_cfs + lag_attn_transformer_cfs

### 1.0 Review status and implementation work

**User-specified timeline convention (binding):** treat the dataset-created UA/UP timeline as canonical. The 20-second UP shift applied at dataset creation is part of how the stored signals are; treat them as if recorded that way. No downstream code, document, lag formula, plot axis, simulation, or critique may add it back, subtract it, budget it, audit it, or interpret it. All filter, alignment, and forecast-clock calculations use the stored timeline. **Done 2026-09-05:** the former `MECHANICAL_SHIFT_SECONDS` / `lag_original_sensor_seconds` helpers in `teb_vae/lag_attn/nets/lag_report.py`, the `mechanical_shift_seconds` terms in `lag_attn_cfs/warmup_budget.py`, the `kl_lag_original_sensor_seconds` metric, the legacy `eval_config.up_shift_secs` key in `teb_vae/lag_attn/eval`, and every caption or config comment carrying a $-20$ s term were removed. If any such term is found again, remove it; do not parameterise it.

The [independent CFS critique](CFS_SCATTERING_PHASE_FORECAST_CRITIQUE.md) contains derivations, numerical counterexamples, and verification limits. The [CFS implementation task list](CFS_SCATTERING_PHASE_FIX_TASKS.md) tracks fixes, dependencies, migration requirements, and acceptance criteria. The [architecture/code companion](CFS_CRWS_MODEL_ARCHITECTURE_CODE.md) remains a snapshot of the current executable network, not of the proposed replacement.

**Status (promoted 2026-09-05, by decision rather than by held-out evidence):** both CFS `default.yaml` files now carry the corrected representation and exact-clock task -- `integer_harmonic_v1` (44/10 phase channels, $c_y=80$, $c_u=46$, kept 76/46 at the shipped budget, decoder 76 wide, block $30\times76=2280$), no input-channel alignment, the `stored` forecast clock (ceiling 270, dense anchors 136) and `anchor_stride: 13` (11 tiles). The configuration that shipped before is `sweep_legacy_dualref_physclock.yaml` (102/51, dual references, `physical`, stride 5, legacy shards). Where a section below quotes 102/51, 98, 2940, dual references or the physical clock as *shipped*, it now describes that legacy arm; the mechanisms are unchanged. The corrected shards do not exist yet (`REPOINT_ME_causal_int`); build them with `create_new_pipeline.py` under `phase_operator=integer_harmonic_v1`. CFS changes remain versioned so shared CRWS/two-sided consumers retain their existing contracts.

| Topic | Correct interpretation / next action |
|---|---|
| One-sided vs two-sided | Keep one-sided inputs. Future two-sided labels are a legitimate separately defined target task; they are not automatically an input leak. CFS-04/09/12. |
| Phase powers | The current $2^{3/2}$ family is discontinuous at the principal-angle branch. Corrected integer $2,4$ families would leave 44 FHR / 10 UP phase pairs before model-side selection; rebuild/version metadata, stats, and checkpoints. CFS-01–03. |
| Leg alignment | `envelope` plus carrier compensation is a defensible narrowband same-event approximation; retain `none` as a different-statistic ablation. It does not establish an exact content timestamp. CFS-01/12. |
| Channel alignment | $0.875\tau_g$ is an energy-centroid convention, not a universal signal delay. Current fast target encoder inputs are delayed 340 s; current levels still reach the persistence path. Test unshifted and grouped/fresh-plus-slow inputs. CFS-05/09/12/13. |
| Target clock | `stored` forecasts later available coefficients; `input` can forecast already available values relative to a delayed history; `physical` is approximate delay compensation with later label endpoints. CFS-05/09. |
| Warm-up / quality | 95% energy is an approximate initialization policy. Passing it is not independence from padding, and endpoint validity does not certify full filter support. CFS-06/07. |
| Novelty / readouts | Stored novelty is a fixed-horizon envelope proxy. KL/attention are model diagnostics; use matched held-out predictive scores and raw-domain controls for stronger claims. CFS-08/10/11. |

The critique reproduced a 401.4 s slow-channel AM delay against the 351.9 s energy-centroid prediction and a finite fractional-phase jump for inputs only $2\times10^{-8}$ apart. It also found approximately 13.3047 s for the actual low-pass mean delay. These are numerical checks, not model-performance comparisons. Its 31 selected passing tests establish selected feature-level mechanics, not validation of other preprocessing operations or exact physical timing.

### 1.1 Package lineage & file map

Encoder × target grid (`teb_vae/lag_attn_crws/RESULTS.md:23-31`):

| target \ encoder | conv-LSTM | conv-Transformer |
|---|---|---|
| raw FHR (two-sided feature inputs) | `lag_attn_rws` | `lag_attn_transformer_rws` |
| two-sided feature target | `lag_attn_fs` | `lag_attn_transformer_fs` |
| **causal (one-sided) feature target** | **`lag_attn_cfs`** | **`lag_attn_transformer_cfs`** |
| **causal inputs, raw FHR target** | **`lag_attn_crws`** | **`lag_attn_transformer_crws`** |

Inheritance / composition (MRO is load-bearing, mixins first):

- `SeqVaeLagAttnCfs(CausalWarmupInputs, CausalFeatureForecastTarget, SeqVaeLagAttnRws)` — `teb_vae/lag_attn_cfs/nets/model.py:40`. Only member is `__init__` (`:54-215`).
- `SeqVaeLagAttnTrfCfs(CausalWarmupInputs, CausalFeatureForecastTarget, SeqVaeLagAttnTrfRws)` — `teb_vae/lag_attn_transformer_cfs/nets/model.py:56-58`. Not a subclass of the cfs model: that would run the conv-LSTM constructor (`:13-18`).
- `CausalFeatureForecastTarget(FeatureForecastTarget)` — `teb_vae/lag_attn_cfs/nets/causal_feature_target.py:116`; `FeatureForecastTarget` — `teb_vae/lag_attn_fs/nets/feature_target.py:62`.
- Tasks: `SeqVaeLagAttnCfsTask(SeqVaeLagAttnFsTask)` `teb_vae/lag_attn_cfs/task.py:78`; `SeqVaeLagAttnFsTask(SeqVaeLagAttnRwsTask)` `teb_vae/lag_attn_fs/task.py:47`; `SeqVaeLagAttnRwsTask(LightningModelBase)` `teb_vae/lag_attn_rws/task.py:56`; `SeqVaeLagAttnTrfCfsTask(SeqVaeLagAttnCfsTask, SeqVaeLagAttnTrfRwsTask)` empty body `teb_vae/lag_attn_transformer_cfs/task.py:41-52`.
- Trainers (experiment drivers): `LagAttnCfsTrainer(LagAttnRwsTrainer)` `teb_vae/lag_attn_cfs/trainer.py:133`; `LagAttnTrfCfsTrainer(LagAttnCfsTrainer, LagAttnTrfRwsTrainer)` `teb_vae/lag_attn_transformer_cfs/trainer.py:134-155` (3 class attrs only); base `LagAttnRwsTrainer(GraphModelBase)` `teb_vae/lag_attn_rws/trainer.py:155`, shared `main()` `:600-662`.

| File | Owns |
|---|---|
| `hdf5_dataset/new_pipeline/create_new_pipeline.py` | HDF5 dataset builder (prescreen, GUID selection, folds, shards) |
| `hdf5_dataset/kymatio_phase_scattering.py` | two-sided scattering + phase harmonics (`KymatioPhaseScattering1D`) |
| `hdf5_dataset/causal_scattering.py`, `causal_scattering_torch.py` | one-sided gammatone bank, causal chain, channel plan, leg alignment, novelty, torch batch path |
| `hdf5_dataset/hdf5_dataset.py` | loader `CombinedHDF5Dataset`, trim, normalisation, `read_causal_warmup` |
| `hdf5_dataset/calculate_dataset_stats.py` | normalisation stats file (`stat_path`) |
| `mimo/EarlyMaestra/early_maestra/adaptor/mimo_adaptor.py` (+`efm.py`) | raw `.mat` records → 4 Hz UP/FHR, segmentation |
| `teb_vae/lag_attn/nets/{attention,blocks,encoders,decoders,heads,delays,controls,lag_report}.py` | shared network primitives |
| `teb_vae/lag_attn/config.py` | YAML `base:` chain loader |
| `teb_vae/lag_attn_rws/nets/{model,losses,raw_masks,raw_targets,geometry,heads,controls}.py` | base architecture, objective, masks, geometry |
| `teb_vae/lag_attn_rws/{task,trainer}.py` | Lightning task, driver, `main()`, DDP |
| `teb_vae/lag_attn_fs/nets/feature_target.py` | feature-target mixin (decoder width = kept channels, target gather) |
| `teb_vae/lag_attn_cfs/nets/causal_inputs.py` | `CausalWarmupInputs`: warm-up mask, alignment shifts, lag floor, anchor tiling, tiled `forward` |
| `teb_vae/lag_attn_cfs/nets/causal_feature_target.py` | `CausalFeatureForecastTarget`: block split 36, floor refusal, forecast clock gather, channel weights, readouts |
| `teb_vae/lag_attn_cfs/causal_warmup.py` | `resolve_warmup_budget`: shards → keep-index / warm-up / alignment / forecast-clock tuples |
| `teb_vae/lag_attn_cfs/model_kwargs.py` | budget → constructor kwargs |
| `teb_vae/lag_attn_cfs/warmup_budget.py` | budget tradeoff curve + run-level figure |
| `teb_vae/lag_attn_cfs/{task,trainer,sample_page,check_run,lag_recovery_check}.py` | cfs task/driver/diagnostic page/in-flight check/identifiability probe |
| `teb_vae/lag_attn_cfs/eval/` | evaluation pipeline (fork of `lag_attn_rws/eval`) |
| `teb_vae/lag_attn_transformer_rws/nets/{blocks,encoders,model}.py` | conv-Transformer encoder + `SeqVaeLagAttnTrfRws` |
| `teb_vae/lag_attn_transformer_cfs/{nets/model,task,trainer}.py`, `eval/{binding,run,verify}.py` | composition only |
| `*/configs/default.yaml`, `tiny.yaml`, `planted.yaml`, `smoke_hie.yaml`, `sweep_*.yaml` | configs |
| `*/DESIGN.md`, `RESULTS.md`, `DIAGNOSIS.md`, `LAG_READOUT_DIAGNOSIS.md` | design record / results forms / diagnoses |

Constants everywhere: raw $f_s = 4$ Hz, decimation 16 ⇒ $\Delta = 4$ s/step (`hdf5_dataset/hdf5_dataset.py:376-377`, `teb_vae/lag_attn/nets/lag_report.py:55`). Stored segment 5280 raw = 330 steps = 22 min; loader trim 1 min each side ⇒ 4800 raw / $T=300$ steps (`teb_vae/lag_attn_rws/nets/geometry.py:3-11`). Horizon $H=30$ steps = 120 s. The dataset-created UA timeline is canonical for downstream use (see §1.0).

### 1.2 Raw data & HDF5 dataset (`create_new_pipeline.py`)

- Pipeline steps in `create_new_pipeline()` `:3166-3498`: (1) prescreen GUIDs on last 6 h (`prescreen_all_guids` `:1930-2023`) → `guid_screening_results.csv`; (2) select GUIDs (`select_classification_guids` `:2083-2368`); (3) 10-fold stratified CV (`create_cv_splits` `:2414-2601`, `N_FOLDS=10, VAL_RATIO=1/9, RANDOM_STATE=42` `:154-158`) → `classification_dataset_records.pickle`; (4) classification shards; (5) pretraining shards from healthy-BG leftovers.
- Raw records: `<records_base_path>/<FOLDER>/EFMOut/*.mat` (`:1957`, `:3511`), 8 folders → subgroups `FOLDER_TO_SUBGROUP` `:89-98` (acidosis/hie/healthy_bg/healthy_no_bg × cs/no_cs). `SUBGROUP_META` `:123-132` (target healthy=1, acidosis=2, hie=3; bg flag). Adaptor: FHR from `Repaired HR1|HR2|external FECG`, UP from `Aligned UA` (`mimo/EarlyMaestra/early_maestra/adaptor/efm.py:135-148`); resampled to 4 Hz; channel map `{'UP':0,'FHR':1}` `mimo_adaptor.py:225`. The dataset-creation UA adjustment is accepted and excluded from downstream analysis. Other operations remain reviewable: `interpolate_bad_values` uses future knots for interior gaps (retrospective; pinned in `hdf5_dataset/tests/test_preprocessing_availability.py`, together with the pointwise prefix equivalence of the rest of `_sanitize_signals`), but the adaptor zero-fills any segdat segment containing a NaN and pads its targets, so on the real path no NaN reaches the sanitiser and that branch is dead; the adaptor's `subsample_factor > 1` branch uses whole-block Fourier `resample` iff a record's own rate exceeds 4 Hz, and keeps no flag, so whether it ran for a shard needs the raw records (CFS-04, open). `mimo/` is not git-tracked.
- Segmentation (MIMO `prepare_data`, `:1699-1711`): `BASE_BLOCK_SIZE=3520`, `SIGNAL_LENGTH=5280` (1320 s), `SEQUENCE_LENGTH=330`, `STEP_SIZE=4800` (20 min stride, 2 min overlap) `:134-140`. Sanitise `:1744-1762` (interp non-finite, clip fhr [0,500], up [-50,500]). Dedup by `domain_start` `:505-546`. Quality: reject `mean(weight)<0.90` (`WEIGHT_THRESHOLD` `:151`), flat FHR >480 samples, flat UP >1200, cumulative flat FHR runs ≥240 summing >1200 `:1766-1805`. Build keeps only `epoch < 0` (pre-delivery) and `> MIN_DOMAIN_START_DATASET=-44640` s (12.4 h) `:144,2805,2883-2888`.
- HDF5 schema (`create_initial_hdf5` `:669-953`; chunks (32,…), lzf), stored layout `(N, C, T)`; loader transposes to `(T, C)`:

| key | dtype/shape | meaning |
|---|---|---|
| `fhr`, `up` | f4 (5280,) | raw 4 Hz on the canonical dataset timeline |
| `fhr_st` | f4 (43 or 36, 330) | FHR scattering: ch0 = $S_0$, rest first-order (descending centre freq) |
| `fhr_ph` | f4 (66, 330) | FHR phase-harmonic pairs (+ `sel_*` attrs) |
| `fhr_up_ph` | f4 (79, 330) | cross UP→FHR phase pairs, **two-sided only**, forbidden for causal models |
| `up_st` | f4 (43 or 36, 330) | UP scattering |
| `up_ph` | f4 (15, 330) | UP phase pairs |
| `target` | f4 (330,) | `pre_defined_target * weight` |
| `weight` | f4 (330,) | binary validity per decimated step (`mimo_sequence.py::calc_sample_weights`) |
| `epoch` | f4 scalar | segment `domain_start` in s relative to delivery (negative) |
| `cs_label`, `bg_label` | u1 | caesarean / blood-gas flags |
| `time_from_labor_onset`, `second_stage_onset` | f4 | `epoch − onset_hours·3600` (NaN if unknown) `:930-945` |
| `guid` | str | recording id |

Root attrs: `transform` (`two_sided`|`causal`), `source_pickle_path`, `source_guid_digest`; causal adds `causal_kernel_taps=32768`, `gammatone_order=4`, `causal_warmup_quantile=0.95`, `causal_leg_alignment` (`:782-804`). Per-block causal attrs (`_write_causal_attrs`): `causal_warmup_steps` int32 (C,) untrimmed steps, `causal_delay_s` f32 (C,), `causal_novelty_curve` f32 (C, T+1) — the horizon-free envelope-mass share within $w$ stored steps, $w=0..T$ (current builds; written as a dense attribute under `libver='latest'`). Legacy shards carry `causal_novelty_frac` f32 (C,) instead, a scalar at $H=30$; the loader reads either and the resolver labels which it got.
- Widths (`resolve_channel_layout` `:1371-1399`): two-sided `{fhr_st:43, fhr_ph:66, fhr_up_ph:79, up_st:43, up_ph:15}` ⇒ $c_y=109$, $c_u=58$; causal `{fhr_st:36, fhr_ph:66, up_st:36, up_ph:15, fhr_up_ph:None}` ⇒ $c_y=102$, $c_u=51$.
- Output paths: `<out>/k_fold_cross_validation_dataset/fold_{1..10}/{train,val,test}/<subgroup>.hdf5` (`:3373-3425`); `<out>/pre_training_dataset/{train,test}_dataset_{cs,no_cs}.hdf5` (`:3443-3495`). Aligned/causal builds must go to a separate `output_base_path` (`:3214`).
- Run: `RUN_ARGS` (must fill `tlo_csv_path`; `records_base_path`, `output_base_path` = `.../new_pipeline_6h_causal_int` and the legacy build's `classification_pickle_path` are pre-filled since 2026-09-05); shipped `transform="causal"`, `leg_alignment="envelope"`, `phase_operator="integer_harmonic_v1"`, so the Run button builds the integer dataset both CFS `default.yaml` files expect; `--transform`, `--leg-alignment`, `--device`, `--test-mode {holdout,augmented}` flags `:3522-3572`; entry `_cli` `:3631-3660`.
- Loader (`hdf5_dataset/hdf5_dataset.py`): `CombinedHDF5Dataset(paths, load_fields, ..., stats_path, normalize_fields, trim_minutes)` `:1070-1173`; trim: raw `data[240:-240]`, coefficient blocks `data[:,15:-15]`, `target/weight` `[15:-15]` (`:1695-1707`, `decimated_trim_steps` `:380-398`); coefficient blocks transposed to `(T,C)` `:1723-1724`. Normalisation `normalize_tensor_data` `:606-725`: `fhr`,`up` z-score; `*_st` channel 0 linear, others $\log(\max(x,0)+10^{-6})$; `*_ph` $\operatorname{asinh}$; then $(g(x)-\mu_c)/(\sigma_c+10^{-8})$ with stats from `stat_path` (config key is `stat_path`; loader kwarg `stats_path`; a wrong path only warns and disables normalisation → guarded by `teb_vae/lag_attn_rws/trainer.py:705-735`). Causal stats exclude each channel's warm-up region (`calculate_dataset_stats.py:231-248`). Warm-up rebase $W'=\max(W-15,0)$ `:319-370`; `read_causal_warmup(paths, trim_minutes)` `:450-603` validates every shard (causal, agreeing quantile/warmup/delay/leg-alignment) and returns `CausalWarmup(warmup_steps, delay_s, novelty_frac, leg_alignment, kept_steps)` `:401-447`.

### 1.3 Two-sided features: kymatio scattering + phase harmonic

- `KymatioPhaseScattering1D(J=11, Q=4, T=16, shape=5280, max_order=1)` built at `create_new_pipeline.py:1287-1295`; class `hdf5_dataset/kymatio_phase_scattering.py:11-99`. Wraps `kymatio.torch.Scattering1D(average=True, T=16)` `:92-95`; first-order Morlet bank from `scattering_filter_factory` `:115-129`, 42 filters, centre freqs descending (`hz[0]=1.4915`, `hz[-1]=0.000515` Hz; see `hdf5_dataset/tests/test_causal_torch.py:155-172`). Reflect padding to 8192 (`_reflect_pad` `:174-205`; pad 1456 each side, `causal_scattering.py:170-190`).
- Scattering block (43,330): $S_0=\phi\star x$, $S_1^{(k)}=\phi\star|\psi_k\star x|$, decimated ×16 (kymatio internal). Channel 0 = $S_0$, 1..42 = $S_1$ in bank order.
- Phase harmonics (`_build_coupling_indices` `:134-160`, `_compute_phase_correlation` `:275-301`): for every pair $(i,j)$ with $\xi_j \ge \xi_i$, power $p=\xi_j/\xi_i$, $[y]^p=|y|e^{ip\arg y}$ (`_accelerate_phase` `:211-218`):
  $$C_{ij}(t)=\Re\Big\{\phi\star\big([y_i]^{p}\,\overline{y_j}\big)\Big\}\big|_{\downarrow16},\qquad y_k=\psi_k\star x .$$
  Low-pass + decimation in `_apply_phi_filter` `:233-273` (frequency-domain truncation, i.e. a $d\times$ analytic projection; see `causal_scattering.py:1276-1306`).
- Self-pair selection (pipeline `_phase_pair_mask` `create_new_pipeline.py:1106-1174`): keep iff $\xi_i \ge f_{\min}/f_s$, $\xi_j \le f_{\max}/f_s$, $|p-2^{k/Q}|<0.05\cdot 2^{k/Q}$ for $k\in$ `PHASE_HARMONIC_K_STEPS=(4,6,8)` ⇒ $p\in\{2, 2^{1.5}, 4\}$ (`:207,215`); `FHR_PHASE_BAND_HZ=(0.008,1.00)` → 66 `fhr_ph` pairs, `UP_PHASE_BAND_HZ=(0.008,0.05)` → 15 `up_ph` pairs (`:191-205`). Pairs ordered ascending `(i,j)`.
- **Known mathematical defect shared by the current causal operator:** for noninteger $p$, $|z|e^{ip\operatorname{Arg}z}$ has a branch jump $2|z||\sin(\pi p)|$. Polar construction and retaining the real product do not generally remove it. The $2^{3/2}$ family affects 22/66 FHR and 5/15 UP phase pairs before model-side selection. The corrected CFS version must store/use integer harmonic indices rather than silently change this shared legacy selection (CFS-01–03). The product also retains amplitude factors and is not generally a centered covariance or normalized phase-locking statistic.
- Cross pairs `fhr_up_ph` = `select_fhr_up_cross_coefficients_v2` `kymatio_phase_scattering.py:635-755` (band A: UP<cap, FHR∈[0.008,0.04) Hz, k∈{0..4}; band B: FHR∈[0.04,0.25], k∈{8,12,16}) → 79 (`create_new_pipeline.py:1307-1332`). Not stored on causal builds.
- Four forward passes per batch `(B,2,5280)` produce `fhr_st, fhr_ph, fhr_up_ph, up_st, up_ph` (`:2925-2951`, slices `:3020-3031`).
- **Non-causality of centered inputs**: Morlet taps span $[-1023.75, +1024]$ s; the slowest `fhr_st` $L_{95}$ is approximately 965 s (`causal_scattering.py:5-13`). $L_{95}$ is an energy-based reach, not the maximum support. The two-sided models use `causal_reach_budget_s` (`teb_vae/lag_attn/channel_reach.py:1-30`: drop channels above the budget, delay survivors by $\lceil L_{95}/\Delta\rceil$); residual tails mean this does not establish exact causality. Causal cells require `causal_reach_budget_s: null` (`causal_warmup.py:966-977`). A centered future label with declared support/availability is a separate valid supervised target, provided it never enters the inputs.

### 1.4 One-sided causal features + alignment modes

- Bank (`hdf5_dataset/causal_scattering.py`): `build_causal_bank(build_filter_bank(5280))` `:489-546`, complex gammatone matched to each production Morlet:
  $$\psi^c_k(t)=a_k(t)\big(e^{i2\pi\xi_k t}-\kappa_k\big),\quad a_k(t)=t^{\gamma-1}e^{-2\pi b_k t}\mathbb 1_{t>0},\quad \gamma=\texttt{GAMMATONE\_ORDER}=4,$$
  $b_k=\sigma_k\sqrt{\ln2}/\sqrt{2^{1/\gamma}-1}$ (`gammatone_rate` `:394-418`), zero-mean (`:444-464`), $L^1$-normalised; `CAUSAL_KERNEL_TAPS=32768` `:145`; low-pass $\phi$ = gamma envelope summing to 1. The uncorrected gamma envelope gives center group delay $\tau_g=\gamma/(2\pi b)$ and energy centroid $(2\gamma-1)/(4\pi b)=0.875\,\tau_g$ (`:364-372`, `ALIGNMENT_DELAY_FACTOR` `:138`). These are distinct summaries; neither gives a universal physical timestamp for every corrected/nonlinear channel (CFS-05).
- Chain (`scattering_block_causal` `:1189-1211`, `phase_block_causal` `:1379-1442`): causal convolution with `'edge'` history pad (`causal_convolve` `:828-857`), $S_0=\phi\star x$, $S_1=\phi\star|\psi_k\star x|$, $\Phi_{ij}=\Re\{\phi\star([y_i]^p\overline{y_j})\}$, plain subsample `[::16]`. Torch batched twin `CausalTorchBank.transform_batch` / `transform_batch_numpy` `hdf5_dataset/causal_scattering_torch.py:90-153, 407-534` (used by the builder `create_new_pipeline.py:2604-2665`).
- Channel plan (`build_channel_plan` `:1549-1625`, `CausalChannelPlan(kept, warmup_steps, delay_s)` `:1451-1485`): per filter warm-up = taps enclosing `CAUSAL_WARMUP_QUANTILE=0.95` energy (`causal_support_samples` `:754-771`); composed $W(S_0)=W_\phi$, $W(S_1^{(k)})=W_k+W_\phi$, $W(\Phi_{ij})=\max(W_i,W_j)+W_\phi$ (`:1488-1546`); `warmup_steps=ceil(W/16)`; channels with `warmup_steps > 330` are **dropped at write time** → 7 slowest wavelets per scattering block (43→36); phase blocks unchanged (66, 15) (`:1618`). `delay_s` composed the same way from $\tau_g$; recorded, never compensated in the dataset (`:1473`, `:1570`).
- **Warm-up interpretation:** these quantile-composed waits are approximate initialization thresholds, not exact independence from assumed prehistory. The actual low-pass retains 15.34% of its $L^1$ mass beyond its own 95%-energy threshold. The slow retained scattering channel's composed sensitivity envelope retains approximately 15.20% beyond its 596 s rounded wait. These are sensitivity diagnostics, not measured prediction errors; `target_warm_frac==1` only certifies the configured threshold (CFS-06/07).
- Shipped plan numbers (`hdf5_dataset/tests/test_causal_torch.py:312-334, 401-410`): warm-up (untrimmed steps) `fhr_st` 5..293, `fhr_ph` 8..149, `up_ph` 56..149; nominal delays $S_0$ approximately 13.3047 s on the independently rebuilt bank/local aligned shard, slowest kept wavelet approximately 791 s, `fhr_ph` 20.5..402.16 s, `up_ph` 150.79..402.16 s. Earlier prose quoted 13.3405 s for $S_0$; resolve from actual metadata. Max filter index used by a pair = 30 (402.1604 s = composed delay of the filter near 0.00824 Hz).
- `causal_novelty_curve` (`novelty_curve` in `causal_scattering.py`): per stored channel, $N_c(w)$ = the share of the composed envelope $g_c=|\psi_k|\star\phi$ (slow leg for a phase pair) within $w$ stored steps, tabulated for $w=0..T$ at write time. A property of the bank with **no forecast horizon baked in** — the horizon is a model-side choice, so the resolver looks the run's own $H$ and each kept channel's forecast advance up: $\nu_c=N_c(H+s_c)$ (`_novelty_from_curve`; `WarmupBudget.novelty_source='curve'`, `novelty_horizon_steps`). At $w=30$: $S_0\approx1.0$, slowest retained target channel approximately 0.026; under the physical clock's advances the slowest stored `fhr_st` share rises from 0.25% to 57.5% and the spread collapses (0.998→0.425), which is why the split must follow the gather. This is an **envelope-mass proxy**, not an exact fraction of nonlinear coefficient value or a universal conservative bound, and every consumer labels it so. Legacy shards carry only the scalar `causal_novelty_frac` at `LEGACY_NOVELTY_HORIZON_STEPS = 30` on the stored clock; the resolver labels those `'legacy_scalar'` and the preflight `novelty_proxy` record flags a mismatched gather with a `tertile_note` (CFS-08).
- **Leg alignment** (dataset-side, `LEG_ALIGNMENT_MODES=("none","envelope")` `:1037`): inside each phase pair the fast leg $j$ has smaller delay than slow leg $i$; skew $\Delta_{ij}=\tau_i-\tau_j=\tau_i(1-1/p_{ij})$ (`pair_leg_skew` `:944-977`). `'envelope'` delays the fast leg and de-rotates it (`leg_alignment_shift` `:980-1034`):
  $$\tilde y_j[t]=y_j[t-s_{ij}]\,e^{i2\pi\xi_j s_{ij}},\qquad s_{ij}=\operatorname{round}(\Delta_{ij}f_s)\ge0 .$$
  `'none'` multiplies legs at the same index. Changes only the values of `fhr_ph`/`up_ph` and the root attr `causal_leg_alignment`; widths, warm-ups, nominal delays identical (`:1517-1522`; `create_new_pipeline.py:3673-3677`). Envelope alignment supports an approximate same-event narrowband interpretation; it does not make `causal_delay_s` universally exact. Historical unaligned correlation misses (`causal_warmup.py:596-604`) concern a particular measurement, not a theorem that the unaligned statistic is useless. Default in the transform functions is `'none'` for shard compatibility; shipped builder `RUN_ARGS` uses `"envelope"`.
- **Channel alignment** (model-side, `channel_alignment_delays` `:1628-1691`, restated in `teb_vae/lag_attn_cfs/causal_warmup.py:762-837`): the current implementation uses the energy-centroid convention $\kappa\tau_c$ to choose input shifts; read channel $c$ at step $t-d_c$,
  $$d_c=\operatorname{round}\!\Big(\kappa\,\frac{\tau_{\mathrm{ref}}-\tau_c}{\Delta}\Big)\ge0,\qquad \kappa=0.875,\ \Delta=4\ \mathrm{s};$$
  channels with $\tau_c>\tau_{\mathrm{ref}}$ are **dropped** (negative input shift would read a future stored value). Applied by `ChannelGate(delays=...)` → `ChannelDelay` (`teb_vae/lag_attn/nets/delays.py:40-233`). Nothing in the dataset compensates channel delay; it is a model-load decision (§1.5). One-sided inputs need not be globally aligned to be causal. The 85-step maximum shift withholds 340 s of fast target trajectories from the encoder, although current levels still reach persistence. Snapping a reference to an existing channel is a resolver policy, not a mathematical necessity (CFS-05/09/13).
- Cross-check script: `hdf5_dataset/compare_causal_scattering.py` (arms A shard / B numpy on Morlet / C causal bank; per-channel CSV, REPORT.md).

### 1.5 Input configurations for the models

What a run reads is decided by (a) which shard variant, (b) five `model_config.VAE_model` keys resolved by `resolve_warmup_budget` (`teb_vae/lag_attn_cfs/causal_warmup.py:908-1149`), (c) the source-block toggle. Streams fed to `forward`: target $=[\texttt{fhr\_st}\,\|\,\texttt{fhr\_ph}]$ $(B,T,c_y)$, source $=[\texttt{up\_st}\,\|\,\texttt{up\_ph}]$ $(B,T,c_u)$ (`teb_vae/lag_attn_rws/task.py:306-430`).

| Case | How selected | Effect |
|---|---|---|
| two-sided inputs | shard `transform=two_sided` (43/66/79/43/15) | used by `lag_attn_rws/fs` cells; causal cells **refuse** (`read_causal_warmup` raises on non-causal, `hdf5_dataset.py:507-513`); guard is `causal_reach_budget_s` (`channel_reach.py`) |
| one-sided, unaligned phase legs | shard built `leg_alignment=none` | root attr `causal_leg_alignment='none'`; config `causal_leg_alignment: none` or `null` to accept (`causal_warmup.py:596-627`) |
| one-sided, envelope-aligned legs (**shipped**) | shard built `leg_alignment=envelope` | config `causal_leg_alignment: envelope` (`lag_attn_cfs/configs/default.yaml:430`) |
| warm-up gating | `causal_warmup_budget_steps: 134` | keeps target channels with $W'_c\le134$ → 98/102 (`fhr_st` 32/36, `fhr_ph` 66/66; dropped `fhr_st` waits 162,194,233,278); **source never gated by budget** (`causal_warmup.py:917-921`, `:1031-1033, 1080-1084`) |
| no channel alignment (**shipped since 2026-09-05**) | `causal_align_reference: null`, `causal_align_reference_source: null` | pure gather, `target_align_delays=None`; every channel read at its own availability time; the default itself (the former `sweep_align_unaligned.yaml` was deleted) |
| single clock | `causal_align_reference: target_max`, `causal_align_reference_source: null` | $\tau^y_{\mathrm{ref}}=402.1604$ s applied to both streams; target shifts $d_c\in[0,85]$; source 47/51 (up_st 32..35 dropped); `sweep_align_target_max.yaml` (now an arm that ADDS the reference) |
| dual clock (legacy default, now `sweep_legacy_dualref_physclock.yaml`) | `causal_align_reference: target_max`, `causal_align_reference_source: 288.2672` | source aligned using the 288.2672 s nominal reference (snapped); source 39/51 (`up_st` 30/36, `up_ph` 9/15), shifts 0..60; input-reference offset $-113.89$ s, scaled convention $\kappa\cdot=-99.66$ s, not an exact measured physical offset (`WarmupBudget.inter_stream_offset_s`) |
| explicit float reference | `causal_align_reference: <s>` | snapped to nearest kept target delay within $\Delta/2$ else `ValueError` (`_resolve_reference_delay` `:630-685`) |
| forecast clock `stored` (**shipped since 2026-09-05**) | `causal_target_forecast_clock: stored`/absent | each target channel scored at its own stored index, $s_c=0$; ceiling $T_{\mathrm{valid}}=270$, dense anchors 136; stride 13 tiles them into 11 |
| forecast clock `physical` (legacy default, now `sweep_legacy_dualref_physclock.yaml`) | `causal_target_forecast_clock: physical` | Approximate compensation: $s_c=\operatorname{round}(\kappa(\tau_c-\tau_{\min})/\Delta)\ge0$, measured $\tau_{\min}\approx13.3047$ s, max 85 → ceiling $270-85=185$, dense anchors 51; last label endpoint 460 s (`causal_inputs.py:248-265`) |
| forecast clock `input` | `causal_target_forecast_clock: input` | $s_c=-d_c\le0$; requires aligned target; for $d_c=85$ all 30 labels precede the actual anchor. Continuation of restricted delayed history, not unavailable future observations; `sweep_target_clock_input.yaml` |
| source without scattering block | `use_up_st: false` | source = `up_ph` only ($c_u$=15); **refused with a warm-up budget** (`lag_attn_cfs/trainer.py:422-449`) |
| reach + warm-up both set | | refused (`causal_warmup.py:966-977`) |

Resolution order inside `resolve_warmup_budget`: read shards (`read_causal_warmup`) → cross-check trim vs `sequence_length` (`:1001-1010`) → build declared vectors per stream (`_build_stream` `:511-593`, `TARGET_BLOCKS=("fhr_st","fhr_ph")`, `SOURCE_BLOCKS=("up_st","up_ph")` `:164-165`) → budget keep on target → leg-alignment check → target ref → source ref → `_align_stream` both → forecast clock over survivors → stride feasibility `warmup_period + anchor_stride <= ceiling` (`:1103-1119`). Returns `WarmupBudget` (`:322-481`: `target/source: StreamWarmup(keep_index, warmup_steps, align_delays, declared_delay_s, declared_novelty_frac)`, `reference_delay_s`, `source_reference_delay_s`, `target_forecast_shift`, `.summary()` for the startup log).

`warmup_model_kwargs(budget, model_cls)` (`teb_vae/lag_attn_cfs/model_kwargs.py:71-170`) maps to constructor kwargs `target_keep_index, target_warmup_steps, source_keep_index, source_warmup_steps` (+ `target_novelty_frac` for feature cells, + `target_forecast_shift` when clock ≠ stored, + `target_align_delays, source_align_delays` when a reference is set). These tuples land in the checkpoint's `model_kwargs` (`teb_vae/lag_attn_rws/task.py:758-770`).

Effective per-channel input availability announced to the encoder: $\delta^{\mathrm{adapter}}_c=W'_c+d_c$ (`causal_inputs.py:573-631`); mask $m_{t,c}=\mathbb 1[t\ge\delta_c]$ inside `AvailabilityInputAdapter` (`teb_vae/lag_attn/nets/encoders.py:95-380`):
$$e_t=W_x(x_t\odot m_t)+W_m(m_t-\mathbf 1)+\mathbb 1[\textstyle\sum_c m_{t,c}=0]\,e_{\mathrm{start}} .$$
Shipped: $\min_c(W'_c+d_c)=80$ target / 55 source, so both availability terms exist (`lag_attn_cfs/DESIGN.md:573-599`). This mask encodes the configured initialization policy, not support-aware signal quality or validation of other preprocessing operations on the canonical timeline (CFS-04/06/07).

### 1.6 Model: SeqVaeLagAttnCfs (conv-LSTM cell)

Data flow (`CausalWarmupInputs.forward` `teb_vae/lag_attn_cfs/nets/causal_inputs.py:760-938`), $B$ batch, $T=300$, $L=91$ lags, $H=30$, $C_{\mathrm{keep}}=98$, $d_{model}=128$, $d_z=64$, 4 heads × $d_{head}=32$:

1. `anchor_index, anchor_valid = _build_anchor_index(B, φ, S)` `:664-758`: $\mathcal A(\varphi)=\{F+\varphi+kS<\text{ceiling}\}$, width $A_{\max}=\lceil(\text{ceiling}-F)/S\rceil$ (promoted default: $F=134$, $S=13$, ceiling $T_{\mathrm{valid}}=270$ → 11; val/test $S=1,\varphi=0$ → 136. Legacy arm: $S=5$, ceiling 185 → 11; val/test 51). Padded slots repeat last valid anchor, flagged `False`.
2. `target=cat([y_st,y_ph])` $(B,T,102)$; `persistence = _anchor_target_values(target, anchors)` $(B,A,98)$ if `persistence_residual` (`causal_feature_target.py:799-851`).
3. `target_gate` (gather 98 survivors, delay each by $d_c$) → `target_adapter` (availability mask) → `target_encoder` = `CausalConvLstmEncoder` (kernels (3,7,11)+15×2, dilations (1,2,4,8,16), 2-layer LSTM; `teb_vae/lag_attn_rws/nets/model.py:511-524`, class `teb_vae/lag_attn/nets/encoders.py:383-560`) → $h_y$ $(B,T,128)$.
4. `source_gate` (39 of 51) → `source_adapter` → `encode_source_kv` (`lag_attn_rws/nets/model.py:821-835`): under shipped `lag_kv_source: conv_stem` a `CausalConvStem` (same conv schedule, no LSTM; RF 387 steps, `:544-560`) → $h_u$ $(B,T,128)$ = keys **and** values.
5. `prior_head(h_y, clock=_prior_clock(u_stream))` → $\mu^p,\ell^p,\ell^p_{raw}$ $(B,T,64)$; `FullLatentPriorHead` (`teb_vae/lag_attn_rws/nets/heads.py`); clock = encode of a zeroed source stream, batch-1, eval-mode, detached (`causal_inputs.py:351-420`), on under `prior_availability_input: true`.
6. Query `query_proj(mu_prior)` → `LagCrossAttention(h_u, mask)` (`teb_vae/lag_attn/nets/attention.py:82-280`): scores $s_{t,m,\ell}=\tfrac1{\sqrt d}(\langle q,k_{t-\ell}\rangle+\langle q,r_\ell\rangle)+b_{m,\ell}$, entmax15, returns `alpha` $(B,T,4,91)$ (lag 0 = current step) and `attended_heads` $(B,T,4,32)$; mask = `build_lag_mask` floored at `lag_floor` (`causal_inputs.py:633-659`).
7. `posterior_head(h_y, attended_heads, mu_prior, raw_logvar_prior)` → $\mu^q=\mu^p+s_\mu\tanh(\cdot)$, $\ell^q$ independent head (`posterior_logvar_mode: independent`), head-structured (latent group $m$ written by head $m$) (`teb_vae/lag_attn/nets/heads.py:139-412`).
8. `_reparameterize_shared` (`lag_attn_rws/nets/model.py:1112-1153`): one $\epsilon$; $z^q=\mu^q+\sigma^q\epsilon$; shipped `base_decode: mean` ⇒ `z_prior` **is** `mu_prior` (same object).
9. Shared decoder twice on gathered latents: `decoder(z_prior.gather(anchors), persistence)` → `mu_base, logvar_base`; same on `z_post` → `mu_full, logvar_full`, each $(B,A_{\max},H,C_{\mathrm{keep}})$. Decoder = `BaselineFutureDecoder(HorizonDecoderCore(depth 4, kernel 3, FiLM per block, 2 horizon self-attention blocks), d_model=d_z, out_channels=C_keep, dropout 0, persistence_residual)` (`lag_attn_rws/nets/model.py:633-657`; `teb_vae/lag_attn/nets/decoders.py:236-506`); persistence $\mu_{\tau,c}=w_{\tau,c}y_{t,c}+f_\theta(z)_{\tau,c}$, $w$ seeded $2^{-\tau/5}$.
10. `kld_tensor` → `te_analysis` → `kld_per_t` $(B,T)$, `source_kl_lag_map` $(B,T,L)$ (= $\sum_m K^{(m)}_t\alpha^{(m)}_{t,\ell}$), `kld_per_t_per_head` $(B,T,4)$ (`teb_vae/lag_attn/nets/heads.py:415-464`).

Return dict (`causal_inputs.py:907-938`): `mu_prior, logvar_prior, raw_logvar_prior, mu_post, logvar_post, z_prior, z_post` $(B,T,d_z)$; `target_state, source_state` $(B,T,d_{model})$; `attended_source_heads` $(B,T,M,d_{head})$; `attn_weights` $(B,T,M,L)$; `mu_base, logvar_base, mu_full, logvar_full` $(B,A_{\max},H,C_{\mathrm{keep}})$; `kld_per_t, kld_per_t_per_head, source_kl_lag_map`; scalars `mu_prior_sat_frac, delta_mu_sat_frac`; `anchor_index` (long), `anchor_valid` (bool); `persistence` $(B,A_{\max},C_{\mathrm{keep}})$ when enabled. No `decoder_state` or `delta_mu_src` source bypass (`lag_attn_rws/nets/model.py:25-40`); the target-only persistence connection is an explicit decoder path outside the latent.

Structural invariants: source pathway never sees target; prior never sees source values (only the zeroed-source clock); exact zero KL at init (`_zero_init_delta_heads` `:951-997`, `test_zero_kl_init.py`); attention and decoder dropout 0; `W_o` frozen (`:675-676`); `causal_norm: true` swaps GroupNorm → `CausalGroupNorm` on history paths (`:659-670`, `teb_vae/lag_attn/nets/blocks.py:299-393`); `compile` must stay off (LSTM, boolean-mask indexing).

Constructor (`teb_vae/lag_attn_cfs/nets/model.py:54-215`): full `SeqVaeLagAttnRws` signature minus `target_delays/source_delays`, plus `CAUSAL_ONLY_KEYWORDS` (`causal_inputs.py:92-103`): `target_warmup_steps, source_warmup_steps, anchor_stride, lag_floor, target_weight_st, target_weight_ph, target_align_delays, source_align_delays, target_novelty_frac, target_forecast_shift`. Order: `_set_causal_inputs` → `_set_channel_weights` → `_set_target_novelty` → `super().__init__(**forwarded, target_delays=target_align_delays, source_delays=source_align_delays)` → `_validate_causal_geometry` (stride vs span; floor check) → `_register_channel_weights`. Defaults differing from base: `warmup_period=134, c_y=102, c_u=51`. The driver builds kwargs by `inspect.signature` sweep, so any config key not in the signature is **silently dropped** (`lag_attn_rws/trainer.py:285-293`).

Anchor floor refusal (`causal_feature_target.py:177-303`):
$$F\ \ge\ \max\Big(\max_c(W'_c-s_c)-1,\ \max_c(W'_c+d_c)\Big)$$
shipped: scored half 133, input-warmth half 134 → $F=134$ (536 s).

Geometry object `TrimmedRawGeometry(raw_len=4800, decimation=16, horizon=30, warmup=134)` (`teb_vae/lag_attn_rws/nets/geometry.py:37-134`): $T=300$, $T_{\mathrm{valid}}=270$, forecast of anchor $t$ covers decimated steps $t+1..t+H$.

### 1.7 Model: SeqVaeLagAttnTrfCfs (conv-Transformer cell)

Same two mixins over `SeqVaeLagAttnTrfRws` (`teb_vae/lag_attn_transformer_rws/nets/model.py:121-1340`). Differences from the conv-LSTM cell:

| Aspect | conv-LSTM (`lag_attn_rws`) | conv-Transformer (`lag_attn_transformer_rws`) |
|---|---|---|
| encoder | `CausalConvLstmEncoder` (dilated causal conv stack ‖ LSTM, GroupNorm/LayerNorm) | `CausalConvTransformerEncoder` (`nets/encoders.py:106-219`): `GatedCausalConvBlock` stem (depthwise kernels (5,9), dilations (1,2), reach 21 steps) → `CausalTransformerBlock`×N (RMSNorm, RoPE, SwiGLU, LayerScale 1e-2, SDPA `is_causal` or band mask) → RMSNorm |
| target/source blocks | — | `target_attention_blocks: 6`, `source_attention_blocks: 3`, `source_attention_window: 16` (RF 66 steps = 264 s; **inert under shipped `lag_kv_source: conv_stem`**) |
| removed ctor keys | — | `lstm_layers, encoder_extra_dilations, encoder_extra_kernel, conv_norm_groups, causal_norm` (absent → passing raises) |
| added ctor keys | — | `encoder_conv_kernels, encoder_conv_dilations, encoder_num_heads, encoder_d_ff, target_attention_blocks, source_attention_blocks, source_attention_window` |
| LR schedule | epoch milestones | + step-granular `general_config.lr_warmup_steps: 2000` (`lag_attn_transformer_rws/task.py:32-125`) |
| compile | refused | honoured from config (shipped `false`) |
| K/V source under `conv_stem` | stem RF 387 steps (not local) | stem RF 21 steps (local) |

MRO `SeqVaeLagAttnTrfCfs → CausalWarmupInputs → CausalFeatureForecastTarget → FeatureForecastTarget → SeqVaeLagAttnTrfRws → Module` (`lag_attn_transformer_cfs/tests/test_construct.py:206-210`); reversed order would build a 16-wide decoder. Config diff vs cfs `default.yaml`: remove the 5 keys, add 7 encoder keys + `lr_warmup_steps`, `gradient_clip_val` 15000→14000, identity keys; everything else leaf-identical (`lag_attn_transformer_cfs/tests/test_config_load.py:151-164, 480-515`). Extra sweep arms only here: `sweep_source_dropout_02/03.yaml` (`source_dropout: 0.2/0.3`). Params shipped 4,284,556 (cfs 4,655,987) (`lag_attn_transformer_cfs/DESIGN.md:576-587`).

### 1.8 Objective

Single home: `teb_vae/lag_attn_rws/nets/losses.py::compute_loss` `:679-1015`, reached via `FeatureForecastTarget.compute_loss` (`lag_attn_fs/nets/feature_target.py:326-438`) after `CausalFeatureForecastTarget.compute_loss` substitutes the pooled validity (`causal_feature_target.py:753-797`). Reported Gaussian score units are nats per anchor; weighting makes the training block a composite score rather than the original joint log-density. The full regularized two-branch objective is not simply an exact ordinary conditional-VAE ELBO.

$$\mathcal L=\lambda_{\mathrm{full}}D_1+\lambda_{\mathrm{base}}D_0+\beta(e)\,\mathrm{KL}_{\mathrm{train}}+\beta_p R_p+\lambda_{\mathrm{ms}}\mathcal L_{\mathrm{ms}}+\lambda_{\Delta}\mathcal L_{\Delta}+\lambda_{\mathrm{boundary}}\mathcal L_{\mathrm{boundary}}$$ (`losses.py:900-908`)

- Target block $Y^+[b,a,\tau,k]=Y[b,\,t_a+1+\tau+s_k,\,\mathrm{keep}[k]]$ $(B,A,H,C_{\mathrm{keep}})$ (`causal_feature_target.py:621-716`; stored clock: `feature_target.py:121-198`). Target is the loader-normalised concatenated `[fhr_st‖fhr_ph]` stream, gate keep-index only, **no delay** (`lag_attn_fs/task.py:88-143`).
- Per-coefficient score (`raw_sample_score` `losses.py:132-226`): `gaussian_nll` $\tfrac12[\log2\pi+\ell+(y-\mu)^2e^{-\ell}]$ (or `mse` $(y-\mu)^2$), then `* channel_weight[c]`, `* horizon_weight[τ]`.
- $D_1$ = `nll_full_block`, $D_0$ = `nll_base_block`: score × forecast mask, summed over $(H,C)$, averaged over contributing anchors (`masked_raw_likelihood` `:293-348`). `nll_*_sample = block/(H·C)`.
- Masks (`teb_vae/lag_attn_rws/nets/raw_masks.py`): `forecast_mask` `:142-236` $m_{a,\tau}=\mathbb 1[t_a\ge w]\,v_{t_a}\,v_{t_a+1+\tau}\,\mathbb 1[\mathrm{coverage}\ge\texttt{coverage\_floor}]\,\mathrm{valid}_a$ with $v=\mathbb 1[\texttt{weight}\ge1.0]$ (`VALID_THRESHOLD` `:43`); `kl_mask` `:270-337` = contributing anchors scattered back to $(B,T)$. Under the physical clock `weight` is first pooled: $\tilde w_u=\min_{j\in[u+s_{\min},u+s_{\max}]}w_j$ (`pooled_scored_weight` `causal_feature_target.py:82-113`).
- Mask limitation: shift-span pooling protects target endpoints, not all raw samples contributing through a filter's history. The inspected local HIE/CS shard retained 99.9281% of the same candidate anchors after this pooling, so large pooling-induced sample loss was not demonstrated there; support contamination remains a distinct concern (CFS-07).
- $\mathrm{KL}$ per step/dim (`kld_tensor` `:448-482`): $\tfrac12[\ell^p-\ell^q+(e^{\ell^q}+(\mu^q-\mu^p)^2)e^{-\ell^p}-1]$; `kl_train` clamps each dim at `free_bits` before masking, sums $d_z$, averages over KL support (`masked_source_kl` `:351-406`); `source_conditioned_kl_raw` (no floor) is the information-rate readout; `kld_active_frac` = dims with mean KL > `KLD_ACTIVE_EPS=1e-2`.
- $R_p=\sum_d\tfrac12(e^{\ell^p}-1-\ell^p)$ prior scale anchor (`masked_prior_rate` `:409-445`), shipped $\beta_p=0.1$.
- KL interpretation: $\mathbb E_{Y,U}\mathrm{KL}(q(Z\mid Y,U)\Vert p(Z\mid Y))=I_q(Z;U\mid Y)+\mathbb E_Y\mathrm{KL}(q(Z\mid Y)\Vert p(Z\mid Y))$, where the aggregate $q(Z\mid Y)$ averages over $U\mid Y$. Thus the latent readout includes prior mismatch and is not automatically future-feature transfer entropy. The source-conditioned branch does not observe future labels; “posterior” names its role in this architecture. Attention-weighted KL and source-null subtraction do not establish causal attribution (CFS-10/11).
- Shape terms (`:514-676`): multiscale $L_1$ (rates (1,4,16)), derivative Huber ($\delta=1$), boundary gap; all shipped 0.0 on cfs (channel axis has no order); `lambda_boundary≠0` refused with any anchor set (`:825-831`, preflight `lag_attn_cfs/trainer.py:371-393`).
- $\beta(e)$: `beta_schedule {kind: linear_warmup, start 0.0, end 1.0, warmup_epochs 50}` → $\beta(e)=\mathrm{start}+(\mathrm{end}-\mathrm{start})\min(1,e/50)$ (`lag_attn_rws/task.py:469-513`).
- Channel weight (`_resolve_channel_weights` `causal_feature_target.py:508-566`): per kept channel `target_weight_st` (declared idx < `TARGET_BLOCK_SPLIT=36`) or `target_weight_ph`, rescaled by $C_{\mathrm{keep}}/\sum w$; shipped (1.0, 0.1) → 2.5389 / 0.25389. Horizon weight (`horizon_decay_weight` `losses.py:87-129`): $w_\tau=H\,2^{-\tau/\lambda}/\sum_{\tau'}2^{-\tau'/\lambda}$, $\lambda=15$ → 1.806..0.473. Both make the block a weighted score, not a log-density; eval applies neither (`lag_attn_cfs/configs/default.yaml:489-497`).
- At current widths, phase receives $6.6/(32+6.6)\approx17.1\%$ of channel-weight mass. Positive weighted marginal log scores can still be proper marginal scores; this does not make them a joint information measure. Retune/ablate weights after changing phase channels rather than transferring the old loss scale without evidence (CFS-10/14).
- Decoding-policy limitation: training uses the prior mean but samples the full latent. Equal latent distributions therefore need not yield equal training-path predictions/gaps. Evaluation's `mc_predictive_block` samples both with common random numbers; use that matched score for source-gain claims. Its one-draw score is **not** the mean-base training path: the former "$K=1$ reduces the estimator exactly to the training-path score" statements in both eval packages were false under `base_decode: mean` and were corrected 2026-09-05 (`test_eval_metrics.py::test_at_one_draw_the_base_branch_is_not_the_training_path_under_mean_decoding` pins the difference, the CRN identity, and the $-\infty$ log-variance recovery). `preflight.json` now also carries `objective_weights` (the phase share of the renormalised channel-weight mass, 17.1% at the shipped geometry, read off the model). Validate MC convergence beyond the default eight draws (CFS-10, open).
- Reported metrics (`losses.py:979-1012`): `total_loss, nll_full_block, nll_full_sample, nll_base_block, nll_base_sample, pred_gap (= D_0 − D_1), source_conditioned_kl_raw/train, kld_active_frac, prior_rate, aux_*, kld_beta, beta_prior, lambda_*, anchor_coverage_frac, mean_logvar_*, logvar_full_floor/ceil_frac, logvar_prior_floor_frac, delta_mu_rms` + fs adds `pred_gap_tau_first/last, pred_gap_st/ph` (`feature_target.py:257-324`) + cfs adds `pred_gap_warm_{lo,mid,hi}, pred_gap_novel_{lo,mid,hi}, target_warm_frac (=1.0 guard), anchors_per_sample ([10,11] train / 51 val guard), source_lag_warmth_frac_st/_ph` (`causal_feature_target.py:1019-1102`) + task adds `main_loss, mu_prior_sat_frac, delta_mu_sat_frac, mu_post_prior_gap_rms`, val-only `nll_shuffled_block, kld_shuffled, shuffle_penalty` (permutation control, `lag_attn_rws/task.py:518-553, 689-733`), val-only `kld_source_null` (zeroed-source re-encode, `lag_attn_cfs/task.py:438-480`), train-only `spike_skipped, spike_ema_loss, grad_norm, grad_clip_frac`. Full list `LagAttnCfsTrainer.TRACKED_METRICS` (`lag_attn_cfs/trainer.py:149-157`).

### 1.9 Training

- Entry: `python -m teb_vae.lag_attn_cfs.trainer --config teb_vae/lag_attn_cfs/configs/default.yaml`; prod `TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_cfs.trainer --config ...` (ranks = `len(cuda_devices)`) (`lag_attn_cfs/trainer.py:1-20`). IDE Run button: `RUN_CONFIG = "teb_vae/lag_attn_cfs/configs/default.yaml"` (the promoted corrected default); repo-root guard `:56-57`; chdir `:552-554`. Transformer twin: `teb_vae/lag_attn_transformer_cfs/trainer.py:196`.
- `main()` order (`lag_attn_rws/trainer.py:600-662`): `resolve_config_file` (base chain merge, `teb_vae/lag_attn/config.py:85-150`: dicts merge, lists/scalars replace) → driver ctor → guards `_check_stat_path`, `_check_declared_widths_against_shard`, `_check_raw_target_normalized(TARGET_FIELDS=("fhr_st","fhr_ph"))`, `_check_causal_budget_resolves` → `trainer_cls.preflight` (cfs: no `fhr_up_ph`, `lambda_boundary==0`, `guid`+`epoch` in `load_fields`, `use_up_st` with budget, floor pairs with resolved budget; `lag_attn_cfs/trainer.py:268-296`) → `setup_config` (run dir, seed, logs, MLflow) → write `resolved_config.yaml` beside checkpoints (`:665-702`) → `GraphDataModule` → `create_model` → `train_model`.
- `create_model` (`lag_attn_rws/trainer.py:307-380`; cfs `:214-263`): `_build_model_kwargs` = signature sweep of `VAE_model` keys + `warmup_model_kwargs(resolve_warmup_budget(config))`; task built with `beta_schedule, kld_beta, beta_prior, lambda_*, likelihood, free_bits, compile_model`; cfs logs the resolved anchor geometry, pushes `seed` into hparams (tile phase), hands `warmup_budget` to the task.
- Task step (`lag_attn_rws/task.py:624-753`): `_build_forward_inputs` → cfs returns 5-tuple `(y_st, y_ph, u_stream, anchor_phase, anchor_stride)` (`lag_attn_cfs/task.py:370-392`); phase $\varphi_b=\mathrm{blake2b}(\texttt{guid}\|\lfloor\texttt{epoch}\rfloor\|\texttt{train\_epoch}\|\texttt{seed})\bmod S$ on train (`:263-325`), `(0,1)` dense on val/test (`DENSE_STAGES` `:75`, `:351-368`). Framework `LightningModelBase.training_step` (`train/pl_model_base.py`) runs the spike breaker watching `metrics['main_loss']`.
- Optimiser: AdamW `lr 3e-4`, `weight_decay 1e-4` default, `lr_milestone [400,800]` ×0.1; transformer cells add `LambdaLR` step warm-up (`lag_attn_transformer_rws/task.py:112-125`).
- DDP (`lag_attn_rws/trainer.py:397-473`): `DDPStrategy(find_unused_parameters = likelihood != 'gaussian_nll', broadcast_buffers=False, gradient_as_bucket_view=True)`; `'auto'` on one device; no `static_graph` (spike breaker changes the graph).
- Callbacks (`train_model` `:475-574`): metrics CSV (`train_results/metrics_history.csv`), loss HTML, hyperparameter plot, `ModelCheckpoint(monitor val/total_loss, save_top_k 3)` filename `lag-attn-cfs-{epoch:02d}`, secondary checkpoint on `val/nll_full_block` (stem `lag-attn-cfs-val-nll_full_block-…`), diagnostic page callback under config block `lag_attn_rws_plotting` (name must not change, `PLOT_CONFIG_KEY` `:192`). Early stopping patience 50 (`default.yaml:989-993`).
- Checkpoint contract: `model_class` stamp + `model_kwargs` (`lag_attn_rws/task.py:758-770`); load with `train/graph_models_utils.py` (`check_model_class`, `load_checkpoint_strict`).
- Existing sweep configs (`run_name`/`tags.variant`; `lag_attn_cfs/tests/test_sweep_configs.py`): some change clock and anchor stride together, and removing input alignment also changes source-channel retention. They are not automatically isolated one-factor comparisons (CFS-12).

| file | delta |
|---|---|
| `sweep_anchor_stride_1.yaml` | `anchor_stride: 1` (dense, 51 anchors under physical clock) |
| `sweep_horizon_15.yaml` | `horizon: 15`, `anchor_stride: 15` |
| `sweep_floor_150.yaml` | `warmup_period: 150` |
| `sweep_horizon_depth_3.yaml` | `horizon_depth: 3` (RF 15 < H+1) |
| `sweep_lag_bias_decay.yaml` | `alibi_slope_scale: 1.0` |
| `sweep_align_target_max.yaml` | `causal_align_reference: target_max` (adds the single-reference input alignment to the unaligned default) |
| `sweep_lag_kv_adapter.yaml` | `lag_kv_source: adapter` (1-step K/V reach) |
| `sweep_target_clock_input.yaml` | clock `input` + `causal_align_reference: target_max` (the input clock copies an input shift) |
| `sweep_legacy_dualref_physclock.yaml` | the pre-2026-09-05 default: `ratio_power_v0`, `c_y: 102`, `c_u: 51`, `target_max` + `288.2672`, clock `physical`, `anchor_stride: 5`, `REPOINT_ME_causal` shards |
| `tiny.yaml` | 1 epoch, 1 device, d_model 32, d_z 8, max_lag 8, `likelihood: mse`, fixtures `teb_vae/lag_attn/tests/fixtures/tiny_shard_causal_int.hdf5` + `tiny_stats_causal_int.hdf5` (the integer-operator fixture, since the promotion) |
| `planted.yaml` (base tiny) | identifiability instrument: planted lag 45 stored steps, `anchor_stride 1`, clock `stored`, single clock; run via `lag_recovery_check.py` |
| `smoke_hie.yaml` | dev-box run at production geometry on `output/causal_scattering/hie_cs_alighned/hie_cs_alighned.hdf5` (2617 segments), both splits same shard |

### 1.10 Configuration reference (`teb_vae/lag_attn_cfs/configs/default.yaml`; transformer deltas noted)

| key | shipped | meaning / consumer |
|---|---|---|
| `general_config.seed` | 42 | determinism + tile phase (`lag_attn_cfs/trainer.py:257-259`) |
| `general_config.cuda_devices` | [0..6] | 7 ranks |
| `general_config.lr` / `lr_milestone` / `epochs` / `batch_size` | 3e-4 / [400,800] / 5000 / 128,128 | framework |
| `general_config.lr_warmup_steps` | 2000 (**transformer cells only**) | `lag_attn_transformer_rws/trainer.py:60` (`LR_WARMUP_STEPS_KEY`, applied in its `create_model`) |
| `general_config.plot_frequency` | 5 | loss HTML + diagnostic page cadence |
| `VAE_model.beta_schedule` | linear_warmup 0→1 over 50 epochs | `lag_attn_rws/task.py:469-513` |
| `free_bits` | 0.0 | KL floor per dim |
| `beta_prior` | 0.1 | prior scale anchor weight |
| `likelihood` | gaussian_nll | `mse` → DDP find-unused path |
| `lambda_full`, `lambda_base` | 1.0, 1.0 | |
| `lambda_ms`, `lambda_deriv`, `lambda_boundary` | 0, 0, 0 (boundary must be 0) | |
| `coverage_floor` | 0.9 | min valid window fraction per anchor |
| `d_model`, `d_z`, `num_heads`, `d_head` | 128, 64, 4, 32 | `num_heads·d_head==d_model`, `d_z % num_heads==0` |
| `horizon`, `raw_per_step`, `sequence_length` | 30, 16, 300 | geometry |
| `warmup_period` | 134 | anchor floor $F$ (536 s) |
| `c_y`, `c_u`, `use_up_st` | 80, 46, true | declared widths under `integer_harmonic_v1` (checked vs shard); the legacy arm declares 102, 51 |
| `lstm_layers`, `dropout`, `source_dropout` | 2, 0.1, null | conv-LSTM only; `source_dropout` null → pathway 0.1, posterior-fusion 0 |
| `decoder_hidden` | 256 | |
| `logvar_clamp`, `mu_scale`, `delta_mu_scale`, `delta_logvar_scale` | [-5,3], 5, 3, 2 | bounds |
| `base_decode` | mean | prior branch decodes $\mu^p$ |
| `posterior_logvar_mode` | independent | with `head_init_calibration` keeps zero-KL init |
| `causal_norm` | true (conv-LSTM only) | CausalGroupNorm |
| `causal_reach_budget_s` | null (required) | two-sided guard, refused with warm-up budget |
| `causal_warmup_budget_steps` | 134 | target keep threshold on $W'_c$ |
| `causal_align_reference` | null | no input alignment; `target_max` (402.1604 s) on the alignment and legacy arms |
| `causal_align_reference_source` | null | dual source clock only on the legacy arm (288.2672) |
| `causal_target_forecast_clock` | stored | exact availability-time labels, ceiling 270, 136 dense anchors; `physical` (approximate compensation, ceiling 185, labels through 460 s) on the legacy arm |
| `causal_leg_alignment` | envelope | expected shard attr |
| `causal_phase_operator` | integer_harmonic_v1 | expected shard attr; `ratio_power_v0` on the legacy arm |
| `anchor_stride` | 13 | train tiling on the stored clock's 136-anchor span (11 tiles); must satisfy $F+S\le$ ceiling; 5 on the legacy arm |
| `lag_floor` | 0 | lag mask floor |
| `target_weight_st`, `target_weight_ph` | 1.0, 0.1 | block weights (ratio) |
| `horizon_weight_halflife_steps` | 15.0 | horizon decay |
| `max_lag`, `use_entmax`, `attention_grad_checkpoint` | 90, true, false | $L=91$ (364 s) |
| `lag_kv_source` | conv_stem | K/V from conv stem; deep source encoder not built |
| `lag_bias_init`, `alibi_slope_scale` | alibi_decay, 0.0 | flat learnable $(M,L)$ bias seed |
| `query_uses_logvar` | false | |
| `prior_availability_input` | true | prior clock from zeroed source |
| `horizon_depth`, `horizon_kernel`, `horizon_film`, `horizon_attention_blocks` | 4, 3, true, 2 | decoder RF $=1+(k-1)(2^d-1)=31\ge H+1$ |
| `persistence_residual` | true | feature cells only |
| `horizon_embed_std`, `head_init_calibration`, `a_head_gain` | 0.8, true, 2.0 | init policy |
| `encoder_extra_dilations`, `encoder_extra_kernel`, `conv_norm_groups` | [8,16], 15, null | conv-LSTM only |
| `encoder_conv_kernels/dilations`, `encoder_num_heads`, `encoder_d_ff`, `target/source_attention_blocks`, `source_attention_window` | [5,9], [1,2], 4, 512, 6, 3, 16 | transformer only |
| `core_model_checkpoint` | null | warm start (strict) |
| `dataset_config.vae_train_datasets/vae_test_datasets` | `…/REPOINT_ME_causal/pre_training_dataset/{train,test}_dataset_{cs,no_cs}.hdf5` | placeholders |
| `dataset_config.stat_path` | `…/REPOINT_ME_causal/stats.hdf5` | key is `stat_path` |
| `dataloader_config.normalize_fields` | [fhr, up, fhr_st, fhr_ph, up_st, up_ph] | |
| `dataset_kwargs.load_fields` | [fhr, up, fhr_st, fhr_ph, up_ph, up_st, weight, guid, epoch] | `guid`,`epoch` required |
| `dataset_kwargs.trim_minutes` | 1.0 | 330→300 steps |
| `dataset_kwargs.epoch_min` | -48000 | |
| `advanced_config.trainer.precision` | "32-true" | |
| `trainer.gradient_clip_val` | 15000 (cfs) / 14000 (trf-cfs) | norm clip |
| `trainer.compile` | false | |
| `trainer.num_sanity_val_steps` | 0 (required) | |
| `spike_breaker` | enabled, multiplier 5, ema_decay 0.02, warmup 100, `ema_floor 1.0e+9` (relative test off), `additive_margin 9.0e+3`, max skips 25 | `train/pl_model_base.py` |
| `callbacks.early_stopping` | val/total_loss, patience 50 | |
| `callbacks.model_checkpoint` | val/total_loss, top-3, `secondary_monitor: val/nll_full_block` | |
| `callbacks.lag_attn_rws_plotting` | enabled, 2 examples, pdf | |
| `tracking.mlflow.run_name` = `tags.variant` | `lag_attn_cfs_intphase_unaligned_stored` | arm identity guard (`lag_attn_cfs_dualref288_physclock` is the legacy arm) |

### 1.11 Evaluation (compact)

- `python -m teb_vae.lag_attn_cfs.eval.run --checkpoint <run>/model_checkpoints/<name>.ckpt` (`eval/EVAL.md:19-24`); config = checkpoint's `resolved_config.yaml` + `eval/configs/eval_overrides.yaml` (holdout causal shards, `eval_config` block: seed 42, `num_mc_samples 8`, `clock_margin_min_nats 0.15`, occlusion bands). `RUN_ARGS` (`eval/run.py:1908+`): `checkpoint, output_dir, overrides, device, num_samples, max_batches, only, skip`. Forward called densely (`anchor_phase=0, anchor_stride=1`). Offline re-run: `--output-dir <finished run> --only <analysis>`. `eval/verify.py summary.json` (torch-free gate), `check_run.py --run-dir` (in-flight from CSV). Transformer twin binds the same pipeline: `lag_attn_transformer_cfs/eval/binding.py:267-276` (`TRF_CFS_BINDING`), `eval/run.py:83-100`.
- Outputs in `<run>/eval_results/`: `summary.json` (headline scalars incl. `pred_gap_mc_nats`, `pred_gap_train_path_nats`; 10 verdicts; sanity; run_context), `preflight.json` (checks, the budget record with `content_lag_s` and `novelty_proxy`, the causality disclosure, `objective_weights`), `per_sample.csv`, `per_anchor.parquet`, `per_anchor_vectors.npz`, `<analysis>/` dirs (`EVAL.md:75-110`).
- Analyses (`eval/run.py` RUN_ARGS comment): forecast, coupling, perm_control, latent, lag_kl, attention, calibration, residual, distributions, trajectory, time_to_delivery, second_stage, events, sufficiency, samples, warmup, source_null, occlusion, lag_clocks, lag_kld_scaled, lag_high_kl, spectral_skill, cross_subgroup (+ band_partition always). Metric definitions: `eval/metrics.py`; figures: `eval/FIGURE_GUIDE.md`.
- Approximate source-to-label content lag on the canonical dataset timeline, before source-encoder history mixing: $L_{j,c}(\ell,h)=\Delta(\ell+1+h+s_c+d^u_j)+\delta^u_j-\delta^y_c$, with source input shifts $d^u_j$, target shifts $s_c$, and explicitly approximate filter-content-delay summaries $\delta$. No UA preprocessing correction enters this calculation. The shared `physical_lag_seconds` (`teb_vae/lag_attn/nets/lag_report.py`) now computes exactly $\Delta(\ell+1+h)+\kappa(\tau^u_{\mathrm{ref}}-\tau^y_{\mathrm{ref}})$ with no shift term, and needs the reference of the **scored target**, not automatically the target encoder reference. Under current `physical` labels the required convention gives $L\approx\Delta(\ell+1+h)+\kappa(\tau^u_{\mathrm{ref}}-\tau_{\min})$, up to rounding. `warmup_budget.py` selects the scored reference and carries no shift term (2026-09-05). Canonical nominal nearest separation approximately 244.6 s, union across lags/horizons approximately 244.6–720.6 s, intersection across every horizon approximately 360.6–604.6 s. These are approximate coefficient-content windows, not identified physiological delays. This formula is the user-required convention, not a claim that pending report-code changes are complete.
- `stored`, `input`, and `physical` ask different forecast questions; do not compare absolute NLL across them as if labels and anchor populations were identical. Current physical-clock centering is not strictly future even under its own convention: first-element nominal center $4-0.875\tau_{\min}\approx-7.64$ s relative to the anchor (CFS-09/12).

### 1.12 Tests & pinned invariants

- Fixtures: `teb_vae/lag_attn/tests/fixtures/{tiny_shard.hdf5, tiny_shard_causal.hdf5, tiny_shard_causal_planted.hdf5, tiny_stats*.hdf5}`; committed real causal fixture `hdf5_dataset/tests/data/causal_fixture.hdf5` (8 segments).
- `lag_attn_cfs/tests/conftest.py`: fixture literals `SHIPPED_BUDGET_STEPS=134`, `SHIPPED_WARMUP_PERIOD=134`, `SHIPPED_ALIGN_REFERENCE="target_max"`, `SHIPPED_LEG_ALIGNMENT="envelope"`, `SHIPPED_PHASE_OPERATOR="ratio_power_v0"`, `SHIPPED_HORIZON=30`, `SHIPPED_SEQUENCE_LENGTH=300`, `SHIPPED_TRIM_MINUTES=1.0`, widths 36/66/36/15 -- these describe the committed LEGACY tiny fixture the unit tests are built on, not the promoted `default.yaml` (`INT_C_Y=80`, `INT_C_U=46` and `INT_CAUSAL_SHARD` describe the integer fixture; `test_config_load.py` reads the config independently); tiny geometry `TINY_SEQ_LEN=24, TINY_HORIZON=4, TINY_WARMUP_PERIOD=5, TINY_STRIDE=4, TINY_BUDGET_STEPS=6` (`:227-239`); helpers `tiny_warmup_kwargs`, `tiny_align_kwargs`, `shipped_warmup_kwargs`, `make_stub_batch`, `make_task`.
- Key tests: `test_causality.py` (no future leak), `test_anchors.py`, `test_invariants.py`, `test_forward_contract.py` (return keys), `test_checkpoint_contract.py`, `test_objective.py`, `test_zero_kl_init.py`, `test_causal_warmup.py` (κ=0.875 agreement, shifts 0..85), `test_warmup_budget.py`, `test_config_load.py` (parity vs `lag_attn_fs` outside 24 exemptions; stride 5 ↔ physical clock), `test_sweep_configs.py`, `test_eval_launch.py` (`ENTRY_POINTS` `:54-62`: eval probe/run/verify, check_run, lag_recovery_check, warmup_budget, scripts.make_tiny_shard), `test_docs.py` (re-measures param totals in DESIGN/RESULTS). `test_eval_*.py` and `test_train_smoke.py` are the slow half; run `.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_cfs/tests/test_<module>.py -q`.
- Shared-net stability: `teb_vae/lag_attn/tests/test_shared_code_stability.py` pins ancestor param count 363,222 and forward fingerprints.
- Verification boundary: current feature-level prefix/future-perturbation tests do not exercise other repair or active resampling on the canonical timeline; the accepted UA adjustment is excluded. Passing geometry/warm-up tests certifies implemented indexing and policy, not an exact physical-clock or padding-independence claim. Add the independent counterexample and raw-prefix gates in CFS-01/04.

### 1.13 Design decisions, diagnoses, results, stale statements

- Params: cfs shipped 4,655,987; off-state (all §17 switches off) 5,146,334 = bitwise pre-revision model (`lag_attn_cfs/DESIGN.md:72-74`). trf-cfs 4,284,556 / off-state 5,054,992.
- Revision switches (config-gated, off-state bitwise old model; `lag_attn_cfs/DESIGN.md §17`): `lag_kv_source: conv_stem`, `prior_availability_input`, `persistence_residual`, `horizon_weight_halflife_steps`, `alibi_slope_scale: 0.0`, dual source reference 288.2672, physical forecast clock, stride 5. Motivated by `lag_attn_transformer_cfs/LAG_READOUT_DIAGNOSIS.md` (argmax lag 0 caused by: unaligned arm on prod box, geometry censoring, deep K/V RF ≈68 steps making 76% of lag window degenerate, 67.5% of KL surviving source-null) and `DIAGNOSIS.md` (H=15 run: source-branch generalisation failure train `pred_gap` +56 vs val −41; `pred_gap` estimator asymmetry under `base_decode: mean`; 85.7% KL source-null; 2 latent dims ≈82% KL).
- `RESULTS.md` (both cells, 2026-08-27): forms only, no production run of the revised architecture. Tier-1 gates: `target_warm_frac==1.0`, `anchors_per_sample`∈[10,11] train / 51 val, finite loss, breaker never latches, gap splits recompose, deterministic re-eval. Identifiability (planted fixture, lag 45): pooled argmax 0 on every arm; band-mass 0.208→0.383 (cfs) / 0.192→0.234 (trf-cfs); occlusion of band [15,44] costs +14.94 nats.
- Training-path `pred_gap` sign alone is not a source-use criterion because of mean-base/sampled-full asymmetry (`lag_attn_rws/PRIOR_SCALE_AND_PREDICTIVE_GAP.md`). A negative **matched held-out** predictive gap must not be dismissed as expected from that asymmetry; it means the evaluated full predictive density scores worse on that support. Use matched baselines, controls, and uncertainty (CFS-10).
- Open findings from the 2026-09-05 review: fractional phase discontinuity; upstream availability; approximate initialization thresholds; signal-dependent delay; stale-clock novelty proxies; withheld recent trajectories under global alignment; KL/lag interpretation. See §1.0 and [the task list](CFS_SCATTERING_PHASE_FIX_TASKS.md). The proposal is integer-phase one-sided inputs, envelope leg alignment, an unshifted-input/stored-target CFS candidate, then controlled grouped/fresh-plus-slow experiments. None is claimed to be the new default or a measured performance winner yet.
- Stale (code/yaml win): comment blocks inside both CFS `default.yaml` still narrate the dual-reference and physical-clock mechanisms as if shipped (the header note says the leaves at each key are what ships: unaligned, stored, stride 13); `DESIGN.md` §4/§9/§15 quote $(B,5,30,98)$/136 dense anchors (actual $A_{\max}=11$, dense 51); `DESIGN.md:1465-1467` clip 6000/margin 3e3 (actual 15000/9e3); historical lag windows and bands that apply a UA preprocessing correction are superseded by the canonical-timeline calculation in §1.11; `lag_attn_transformer_cfs/eval/binding.py:58-59` "twenty-two" geometry keys (25); `hdf5_dataset/PIPELINE.md` describes the legacy builder (5760/360, 106 channels) not `create_new_pipeline.py`; `teb_vae/lag_attn/encoders.py` "10 GroupNorms" holds only without extra dilations; `RESULTS.md:358` / `eval/verify.py:22` arm counts (9 sweep files exist).

---

## 2. lag_attn_crws + lag_attn_transformer_crws

### 2.1 What crws is vs cfs

Same one-sided inputs as §1.5 (same three tensors, same warm-up budget, same alignment machinery, same tiling), but the **target is the raw 4 Hz FHR** future: decoder emits $R=16$ raw samples per horizon token, block $H\cdot R=30\times16=480$ samples per anchor (vs 2940 coefficients in cfs). Neither side of the objective contains its own future and the target has no warm-up and no group delay (`lag_attn_crws/trainer.py:200-205`). Direct control: `lag_attn_rws` (same target/objective, two-sided inputs).

- `SeqVaeLagAttnCrws(CausalRawInputs, SeqVaeLagAttnRws)` `teb_vae/lag_attn_crws/nets/model.py:53`; `SeqVaeLagAttnTrfCrws(CausalRawInputs, SeqVaeLagAttnTrfRws)` `teb_vae/lag_attn_transformer_crws/nets/model.py:68`. **One mixin, not two**: no `CausalFeatureForecastTarget`, so `_default_decoder_out_channels` stays `raw_per_step` (`lag_attn_rws/nets/model.py:708-727`); composing the feature mixin by mistake builds a 98-wide decoder against a $(B,A,H,16)$ target (`crws/nets/model.py:9-17`).
- `CausalRawInputs(CausalWarmupInputs)` `teb_vae/lag_attn_crws/nets/causal_raw_inputs.py:137`; binds by reference `SOURCE_BLOCK_SPLIT=36`, `TARGET_BLOCK_SPLIT=36`, `_resolve_block_warm_steps`, `_anchors_per_sample`, `_source_lag_warmth` from the cfs target mixin (`:173-185`).
- Tasks: `SeqVaeLagAttnCrwsTask(SeqVaeLagAttnRwsTask)` `teb_vae/lag_attn_crws/task.py:82` (directly from rws, **not** via fs; binds `anchor_phase, _phase_field, resolve_anchor_geometry, _build_forward_inputs, _mu_gap_rms, _added_metrics, input_stream_panels, input_budget_figure` from the cfs task `:103-124`; own `forecast_rows` and `compute_loss_and_metrics` `:143-203`); `SeqVaeLagAttnTrfCrwsTask(SeqVaeLagAttnCrwsTask, SeqVaeLagAttnTrfRwsTask)` empty `teb_vae/lag_attn_transformer_crws/task.py:43`.
- Trainers: `LagAttnCrwsTrainer(LagAttnRwsTrainer)` `teb_vae/lag_attn_crws/trainer.py:128` (`TARGET_FIELDS` inherited `("fhr",)`, `CHECKPOINT_STEM="lag-attn-crws"`); `LagAttnTrfCrwsTrainer(LagAttnCrwsTrainer, LagAttnTrfRwsTrainer)` `teb_vae/lag_attn_transformer_crws/trainer.py:81-103` (3 attrs; `_build_model_kwargs`/`create_model` run on both parents via `super()`).
- Constructor keywords: base signature minus `target_delays/source_delays`, plus 6 causal keys `target_warmup_steps, source_warmup_steps, target_align_delays, source_align_delays, anchor_stride, lag_floor` (`crws/nets/model.py:83-137`); **no** `persistence_residual`, `target_weight_*`, `target_novelty_frac`, `target_forecast_shift` (structurally excluded, `:19-26`). `decoder_out_channels` still exists on the conv-LSTM cell's signature (would fail loudly at `raw_sample_score`), absent on the transformer cell.

### 2.2 Inputs: CausalRawInputs

Same HDF5 keys as §1.5 plus `fhr` (raw, must be in `load_fields` **and** `normalize_fields`) and `weight` (required, `_check_raw_target_fields` `crws/trainer.py:402-433`). `_build_raw_target` is the rws one: `(batch.fhr (B,4800), batch.weight (B,300))` (`lag_attn_rws/task.py:432-464`).

Shipped alignment differs from the feature cells (`crws/configs/default.yaml`: `causal_align_reference: 42.21`, no source reference, no forecast clock):
- Clock convention (historical design rationale: `lag_attn_crws/DESIGN.md:169-206`): with a raw target $\tau^y\equiv0$, the canonical-timeline approximation is $L(\ell,h)=\Delta(\ell+1+h)+\kappa\tau_{\mathrm{ref}}$, with no UA preprocessing correction. At `target_max` the nominal smallest lead is approximately 355.9 s. The shipped 42.21 s reference snaps to 42.2066 s (filter near 0.1109 Hz; $\kappa\tau_{\mathrm{ref}}\approx36.93$ s), giving a nominal canonical range approximately $[40.9,516.9]$ s. These are convention-based summaries of the existing geometry, not a CRWS architecture change or an endorsement of historical offset-adjusted reports.
- Cost (`DESIGN.md:207-236`, `:925-926`): budget 134 keeps 98/102 target-stream inputs, all 51 source; alignment then drops every channel above 42.2066 s → target-stream input **38/102** (`fhr_st` 17/36, `fhr_ph` 21/66), source **17/51** (`up_st` 17/36, **`up_ph` 0/15** — whole block gone, fastest `up_ph` is 150.79 s); shifts 0..6 steps, every kept channel honest by step 6. Consequence: every reachable lag is warm; `source_lag_warmth_frac_st` ≡ 1.0 and `_ph` ≡ 1.0 over zero channels (`DESIGN.md:556-600`). `use_up_st: false` refused (source would be empty).
- Floor (`CausalRawInputs._check_anchor_floor` `causal_raw_inputs.py:190-311`): same two inequalities, re-justified as an input-warmth *policy* (raw target has no validity constraint); shipped requirement is 6 steps, `warmup_period: 134` kept equal to cfs so the two rows differ in one variable.
- Readouts resolved: only `source_block_warm_st/_ph` (`_resolve_warmup_readout_constants` `:313-381`); no target warm fraction, no tertiles.

### 2.3 Model: SeqVaeLagAttnCrws

Forward is §1.6's tiled forward verbatim (inherited, `causal_raw_inputs.py:15-18`), with `persistence=None`, no prior-clock difference (flag still available: `prior_availability_input: true` shipped), decoder width 16. Outputs `mu_base, logvar_base, mu_full, logvar_full` $(B,A_{\max},H,16)$; shipped `anchor_stride: 30` on span $270-134=136$ → $A_{\max}=5$, tiles 4–5 per sample (~4.53 mean), dense 136 on val/test (ceiling stays $T_{\mathrm{valid}}=270$; no forecast clock). Anchored raw target `gather_anchored_future_target(fhr_raw, geometry, anchors, future_index)` `:61-134`: $X^+[b,a,\tau,r]=x[b,\ \texttt{future\_index}[\mathcal A[b,a],\tau,r]]$, `future_index[t,τ,r]=16(t+1)+16τ+r` (`lag_attn_rws/nets/raw_targets.py:27-44`). Dense fallback when no anchor set. `compute_loss` `:386-492` = shared objective with `block_width=geometry.r=16`, `horizon_weight`, no channel weight; adds `anchors_per_sample`, `source_lag_warmth_frac_st/_ph`.

### 2.4 Model: SeqVaeLagAttnTrfCrws

Composition only (`lag_attn_transformer_crws/nets/model.py:85-215`): §1.7's encoder swap with the crws input mixin; signature = `SeqVaeLagAttnTrfRws` minus delays plus the 6 causal keys; no `decoder_out_channels`, no `persistence_residual`. Diamond MRO `TrfCrwsTask → CrwsTask → TrfRwsTask → RwsTask → LightningModelBase` (`lag_attn_transformer_crws/task.py:20-27`); trainer resolution: `TARGET_FIELDS, TRACKED_METRICS, preflight` from causal parent, `compile_model_requested, _build_trainer_kwargs` from transformer parent (`trainer.py:33-53`). Block $30\times16=480$; nats comparable only to `lag_attn_crws`.

### 2.5 Objective

Identical seven-term objective (§1.8) via `compute_raw_objective` on the raw block; differences: `likelihood gaussian_nll`, `lambda_ms: 0.1`, `lambda_deriv: 0.1` (shape terms **on**, meaningful on a raw trajectory; multiscale $L_1$ rates (1,4,16) over the flattened 480-sample block, derivative Huber $\delta=1$), `lambda_boundary 0` (refused with tiles), horizon weight $\lambda=15$ (the only weight; `DESIGN.md:377-405`), no channel weight, no persistence. `total_loss` is mixed-unit; read `nll_*` for nats. `pred_gap` from the CSV is horizon-weighted with no unweighted twin (no eval package on this row).

### 2.6 Training & configs

- Launch: `python -m teb_vae.lag_attn_crws.trainer --config teb_vae/lag_attn_crws/configs/default.yaml` / `RUN_CONFIG` `crws/trainer.py:543`; transformer `lag_attn_transformer_crws/trainer.py:141`. Same `main()`, DDP, callbacks as §1.9; preflight adds the `weight` field check (`crws/trainer.py:269-298`); `create_model` logs geometry with raw block width `:207-264`. Tracked metrics = rws set + `anchors_per_sample, source_lag_warmth_frac_st/_ph` + val `kld_source_null` (`:114-147`).
- Config deltas (non-comment diff, cfs→crws `default.yaml`): `lambda_ms 0→0.1`, `lambda_deriv 0→0.1`, `causal_align_reference target_max→42.21`, `causal_align_reference_source` and `causal_target_forecast_clock` **absent**, `anchor_stride 5→30`, `target_weight_st/ph` absent, `persistence_residual` absent, `gradient_clip_val 15000→12000`, `spike_breaker.additive_margin 9e3→2.5e3`, identity keys (`tag lag_attn_crws_baseline`, `run_name`/`variant lag_attn_crws_ref42`, `experiment_name seqvae-teb-lag-attn-crws`). Everything else identical (budget 134, floor 134, H 30, `lag_kv_source conv_stem`, `prior_availability_input true`, `alibi_slope_scale 0.0`, halflife 15, early stopping, secondary checkpoint).
- crws→transformer_crws: remove `lstm_layers, causal_norm, encoder_extra_dilations, encoder_extra_kernel, conv_norm_groups`; add `lr_warmup_steps: 2000` + 7 encoder keys ([5,9], [1,2], 4 heads, d_ff 512, 6/3 blocks, window 16); `gradient_clip_val 12000→11100`; identity keys.
- Variant configs: `tiny.yaml` (1 epoch, d_model 32, d_z 8, max_lag 8, `mse`, causal fixtures), `smoke_causal.yaml` (600 epochs, batch 4, clip 1e9, fixture; the run the loss-scale constants were measured on), `sweep_anchor_stride_1.yaml`, crws-only `sweep_horizon_15.yaml` (`horizon 15, anchor_stride 15`; restores the 240-sample block).
- Params: crws shipped 4,589,907 (off-state 5,081,146; `lag_attn_rws` 5,094,458); trf-crws 4,218,476 (off-state 4,989,804; `lag_attn_transformer_rws` 5,003,116) (`lag_attn_crws/DESIGN.md:86-90`, `lag_attn_transformer_crws/DESIGN.md:65-72`).

### 2.7 Tests & invariants (deltas)

`lag_attn_crws/tests/conftest.py`: `SHIPPED_ALIGN_REFERENCE = 42.21` `:167`, `SHIPPED_HORIZON = 30` `:76`, imports shipped literals from the cfs conftest (`IMPORTED_FROM_CAUSAL` `:78`); fixtures `build`, `make_raw_signal`, `budget`, `unaligned_budget`. Own tests: `test_causal_raw_inputs.py` (anchored gather equals dense builder at dense anchors), `test_raw_target.py`, `test_anchors.py`, `test_causality.py`, `test_config_load.py` (parity vs `lag_attn_rws` outside 21 exemptions), `test_sweep_configs.py`, `test_warmup_budget.py`, `test_task.py` (MRO/binding provenance). Transformer twin: `test_ddp_reachability.py`, `test_trainer.py` (both `create_model` halves fire). No eval package on this row (`lag_attn_crws/RESULTS.md:43-49`).

### 2.8 Design decisions & results

- Two feature-cell mechanisms declined on this row (`crws/RESULTS.md:10-18`, `DESIGN.md §14`): persistence residual (no per-channel level in a raw block) and the dual source reference (single 42.21 s reference is already source-only in effect since $\tau^y=0$).
- `RESULTS.md` (2026-08-27) both cells: forms only, no runs; in-sample only (both splits one shard in smoke configs); Tier-1 gates as cfs minus target-warm columns; nats comparable only within the row.
- Stale (code wins): `crws/nets/model.py:143` and `transformer_crws/nets/model.py:147-148` docstrings say default `horizon` 15 (constructor default is 30, `:89`/`:91`); `transformer_crws/nets/model.py:53-57` says block $15\times16=240$ (shipped $30\times16=480$, `lag_attn_crws/RESULTS.md:50`); `crws/trainer.py:102-104` says `anchors_per_sample` ∈[10,11] at shipped tiling (copied from cfs; at stride 30 it is 4–5, `lag_attn_crws/RESULTS.md:51`); `crws/trainer.py:479-480` "144 of the 152 anchors" is H=15-era arithmetic.
