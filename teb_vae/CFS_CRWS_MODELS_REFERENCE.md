# lag_attn_cfs / lag_attn_transformer_cfs / lag_attn_crws / lag_attn_transformer_crws — reference

This is a dense reference for the four causal models and the data pipeline that they share. §1 covers the causal feature-stream (CFS) models and the full data pipeline. §2 covers the causal raw-waveform-stream (CRWS) models as a delta over §1. Implementation pointers name a file and a symbol; grep the symbol to find the current line. The [architecture/code companion](CFS_CRWS_MODEL_ARCHITECTURE_CODE.md) holds the network code, the module trees and the shape traces.

Domain: intrapartum fetal monitoring. FHR is the fetal heart rate (bpm, 4 Hz). UP is the uterine pressure (mmHg, 4 Hz). Every model has a target-only branch and a source-conditioned branch. For CFS, the forecast labels are FHR-derived coefficients that the target forecast clock selects (§1.5). The per-step KL is a source-conditioned latent divergence, and the attention weights split it over lags. `pred_gap` is the base-minus-full score gap; the training path and the matched Monte Carlo evaluation use different decoding policies (§1.8, §1.11).

The stored UP/FHR timeline in the HDF5 shards is canonical. Every filter, alignment, forecast-clock and lag calculation below starts from the stored timeline and uses only the feature grid and the filter-delay terms.

---

## 1. lag_attn_cfs + lag_attn_transformer_cfs

### 1.1 Package lineage and file map

Encoder × target grid:

| target \ encoder | conv-LSTM | conv-Transformer |
|---|---|---|
| raw FHR (two-sided feature inputs) | `lag_attn_rws` | `lag_attn_transformer_rws` |
| two-sided feature target | `lag_attn_fs` | `lag_attn_transformer_fs` |
| **causal (one-sided) feature target** | **`lag_attn_cfs`** | **`lag_attn_transformer_cfs`** |
| **causal inputs, raw FHR target** | **`lag_attn_crws`** | **`lag_attn_transformer_crws`** |

Inheritance and composition (the MRO is load-bearing; mixins come first):

- `SeqVaeLagAttnCfs(CausalWarmupInputs, CausalFeatureForecastTarget, SeqVaeLagAttnRws)` in `teb_vae/lag_attn_cfs/nets/model.py`. Its only member is `__init__`.
- `SeqVaeLagAttnTrfCfs(CausalWarmupInputs, CausalFeatureForecastTarget, SeqVaeLagAttnTrfRws)` in `teb_vae/lag_attn_transformer_cfs/nets/model.py`. It is not a subclass of the CFS model, because that would run the conv-LSTM constructor.
- `CausalFeatureForecastTarget(FeatureForecastTarget)` in `teb_vae/lag_attn_cfs/nets/causal_feature_target.py`; `FeatureForecastTarget` in `teb_vae/lag_attn_fs/nets/feature_target.py`.
- Tasks: `SeqVaeLagAttnCfsTask(SeqVaeLagAttnFsTask)` (`teb_vae/lag_attn_cfs/task.py`); `SeqVaeLagAttnFsTask(SeqVaeLagAttnRwsTask)` (`teb_vae/lag_attn_fs/task.py`); `SeqVaeLagAttnRwsTask(LightningModelBase)` (`teb_vae/lag_attn_rws/task.py`); `SeqVaeLagAttnTrfCfsTask(SeqVaeLagAttnCfsTask, SeqVaeLagAttnTrfRwsTask)` with an empty body (`teb_vae/lag_attn_transformer_cfs/task.py`).
- Trainers (experiment drivers): `LagAttnCfsTrainer(LagAttnRwsTrainer)` (`teb_vae/lag_attn_cfs/trainer.py`); `LagAttnTrfCfsTrainer(LagAttnCfsTrainer, LagAttnTrfRwsTrainer)` (`teb_vae/lag_attn_transformer_cfs/trainer.py`, class attributes only); base `LagAttnRwsTrainer(GraphModelBase)` (`teb_vae/lag_attn_rws/trainer.py`), which also holds the shared `main()`.

| File | Owns |
|---|---|
| `hdf5_dataset/new_pipeline/create_new_pipeline.py` | HDF5 dataset builder (prescreen, GUID selection, folds, shards) |
| `hdf5_dataset/kymatio_phase_scattering.py` | two-sided scattering + phase harmonics (`KymatioPhaseScattering1D`) |
| `hdf5_dataset/causal_scattering.py`, `causal_scattering_torch.py` | one-sided gammatone bank, causal chain, phase operators, channel plan, leg alignment, novelty curve, torch batch path |
| `hdf5_dataset/hdf5_dataset.py` | loader `CombinedHDF5Dataset`, trim, normalisation, quality masks, `read_causal_warmup` |
| `hdf5_dataset/calculate_dataset_stats.py` | normalisation stats file (`stat_path`) |
| `mimo/EarlyMaestra/early_maestra/adaptor/mimo_adaptor.py` (+`efm.py`) | raw `.mat` records → 4 Hz UP/FHR, segmentation |
| `teb_vae/lag_attn/nets/{attention,blocks,encoders,decoders,heads,delays,controls,lag_report}.py` | shared network primitives |
| `teb_vae/lag_attn/config.py` | YAML `base:` chain loader |
| `teb_vae/lag_attn_rws/nets/{model,losses,raw_masks,raw_targets,geometry,heads,controls}.py` | base architecture, objective, masks, geometry |
| `teb_vae/lag_attn_rws/{task,trainer}.py` | Lightning task, driver, `main()`, DDP |
| `teb_vae/lag_attn_fs/nets/feature_target.py` | feature-target mixin (decoder width = kept channels, target gather, likelihood kwargs) |
| `teb_vae/lag_attn_cfs/nets/causal_inputs.py` | `CausalWarmupInputs`: warm-up mask, alignment shifts, lag floor, anchor tiling, source key mask, tiled `forward` |
| `teb_vae/lag_attn_cfs/nets/causal_feature_target.py` | `CausalFeatureForecastTarget`: block split 36, floor refusal, forecast-clock gather, channel weights, likelihood structure, readouts |
| `teb_vae/lag_attn_cfs/causal_warmup.py` | `resolve_warmup_budget`: shards → keep-index / warm-up / alignment / forecast-clock tuples |
| `teb_vae/lag_attn_cfs/scored_horizon.py` | `resolve_target_scored_horizon`: per-channel scored horizon $H_c$ |
| `teb_vae/lag_attn_cfs/model_kwargs.py` | budget → constructor kwargs |
| `teb_vae/lag_attn_cfs/warmup_budget.py` | budget tradeoff curve, run-level figure, content-lag summaries |
| `teb_vae/lag_attn_cfs/{task,trainer,sample_page,check_run,lag_recovery_check}.py` | CFS task, driver, diagnostic page, in-flight check, identifiability probe |
| `teb_vae/lag_attn_cfs/eval/` | evaluation pipeline |
| `teb_vae/lag_attn_transformer_rws/nets/{blocks,encoders,model}.py` | conv-Transformer encoder + `SeqVaeLagAttnTrfRws` |
| `teb_vae/lag_attn_transformer_cfs/{nets/model,task,trainer}.py`, `eval/` | composition and the evaluation binding |
| `*/configs/default.yaml`, `tiny.yaml`, `planted.yaml`, `smoke_hie.yaml`, `sweep_*.yaml` | configs |

Constants: raw $f_s = 4$ Hz and decimation 16 give $\Delta = 4$ s per step. A stored segment is 5280 raw samples = 330 steps = 22 min. The loader trims 1 min from each side, which gives 4800 raw samples and $T=300$ steps (`teb_vae/lag_attn_rws/nets/geometry.py`). The horizon $H$ is the config key `horizon`; §1.10 gives the value in each `default.yaml`.

### 1.2 Raw data and HDF5 dataset (`create_new_pipeline.py`)

- Pipeline steps in `create_new_pipeline()`: (1) prescreen GUIDs on the last 6 h (`prescreen_all_guids`) → `guid_screening_results.csv`; (2) select GUIDs (`select_classification_guids`); (3) 10-fold stratified CV (`create_cv_splits`, `N_FOLDS=10`) → `classification_dataset_records.pickle`; (4) classification shards; (5) pretraining shards from healthy-BG leftovers.
- Raw records: `<records_base_path>/<FOLDER>/EFMOut/*.mat`. Eight folders map to subgroups through `FOLDER_TO_SUBGROUP` (acidosis / hie / healthy_bg / healthy_no_bg × cs / no_cs). `SUBGROUP_META` gives the target (healthy=1, acidosis=2, hie=3) and the BG flag. The adaptor reads FHR from `Repaired HR1|HR2|external FECG` and UP from `Aligned UA` (`efm.py`), resamples to 4 Hz, and uses the channel map `{'UP':0,'FHR':1}` (`mimo_adaptor.py`). The adaptor zero-fills any segment that contains a NaN. In the builder, `interpolate_bad_values` bridges interior non-finite runs linearly to the next valid sample (a retrospective operation); `hdf5_dataset/tests/test_preprocessing_availability.py` pins this behaviour and the pointwise prefix equivalence of the rest of `_sanitize_signals`. `mimo/` is not git-tracked.
- Segmentation (MIMO `prepare_data`): `BASE_BLOCK_SIZE=3520`, `SIGNAL_LENGTH=5280` (1320 s), `SEQUENCE_LENGTH=330`, `STEP_SIZE=4800` (20 min stride, 2 min overlap). Sanitise: interpolate non-finite values, clip FHR to [0,500] and UP to [-50,500]. Deduplicate by `domain_start`. Quality rejection: `mean(weight)<0.90` (`WEIGHT_THRESHOLD`), flat FHR >480 samples, flat UP >1200, cumulative flat FHR runs ≥240 that sum to >1200. The build keeps only `epoch < 0` (pre-delivery) and `epoch > MIN_DOMAIN_START_DATASET=-44640` s (12.4 h).
- HDF5 schema (`create_initial_hdf5`; chunks (32,…), lzf). The stored layout is `(N, C, T)`; the loader transposes coefficient blocks to `(T, C)`:

| key | dtype/shape | meaning |
|---|---|---|
| `fhr`, `up` | f4 (5280,) | raw 4 Hz on the stored timeline |
| `fhr_st` | f4 (43 or 36, 330) | FHR scattering: ch0 = $S_0$, the rest first-order (descending centre frequency) |
| `fhr_ph` | f4 (66 or 44, 330) | FHR phase-harmonic pairs (+ `sel_*` attributes) |
| `fhr_up_ph` | f4 (79, 330) | cross UP→FHR phase pairs, **two-sided only**; the causal models refuse it |
| `up_st` | f4 (43 or 36, 330) | UP scattering |
| `up_ph` | f4 (15 or 10, 330) | UP phase pairs |
| `target` | f4 (330,) | `pre_defined_target * weight` |
| `weight` | f4 (330,) | binary validity per decimated step (`mimo_sequence.py::calc_sample_weights`) |
| `epoch` | f4 scalar | segment `domain_start` in s relative to delivery (negative) |
| `cs_label`, `bg_label` | u1 | caesarean / blood-gas flags |
| `time_from_labor_onset`, `second_stage_onset` | f4 | `epoch − onset_hours·3600` (NaN if unknown) |
| `guid` | str | recording id |

  Widths: 43 scattering channels on two-sided builds, 36 on causal builds; 66/15 phase pairs under `ratio_power_v0` (and on two-sided builds), 44/10 under `integer_harmonic_v1`. Signal-quality builds (root attribute `quality_version`) also store per-sample quality bit fields (`up_quality`, `fhr_quality`) and per-coefficient contamination fractions $\rho$ (`*_rho`), which the loader's `ua_step_mask` and `fhr_step_mask` options read.
- Root attributes: `transform` (`two_sided`|`causal`), `source_pickle_path`, `source_guid_digest`. Causal builds add `causal_kernel_taps=32768`, `gammatone_order=4`, `causal_warmup_quantile=0.95`, `causal_leg_alignment` and `causal_phase_operator`. Per-block causal attributes (`_write_causal_attrs`): `causal_warmup_steps` int32 (C,) in untrimmed steps, `causal_delay_s` f32 (C,), and `causal_novelty_curve` f32 (C, T+1), the horizon-free envelope-mass share within $w$ stored steps for $w=0..T$. Legacy shards carry `causal_novelty_frac` f32 (C,) instead, a scalar at $H=30$; the loader reads either, and the resolver labels which one it got. Phase blocks carry `sel_phase_operator` and, under the integer operator, `sel_harmonic` (int32).
- Widths (`resolve_channel_layout`): two-sided `{fhr_st:43, fhr_ph:66, fhr_up_ph:79, up_st:43, up_ph:15}` ⇒ $c_y=109$, $c_u=58$. Causal `ratio_power_v0` `{fhr_st:36, fhr_ph:66, up_st:36, up_ph:15}` ⇒ $c_y=102$, $c_u=51$. Causal `integer_harmonic_v1` `{fhr_st:36, fhr_ph:44, up_st:36, up_ph:10}` ⇒ $c_y=80$, $c_u=46$.
- Output paths: `<out>/k_fold_cross_validation_dataset/fold_{1..10}/{train,val,test}/<subgroup>.hdf5` and `<out>/pre_training_dataset/{train,test}_dataset_{cs,no_cs}.hdf5`. Causal builds must go to a separate `output_base_path`.
- Run: edit `RUN_ARGS` in the file and press Run, or use the CLI. Fill `tlo_csv_path` first. `RUN_ARGS` sets `transform="causal"`, `leg_alignment="envelope"` and `phase_operator="integer_harmonic_v1"`. CLI flags include `--records-base-path`, `--output-base-path`, `--tlo-csv-path`, `--classification-pickle-path`, `--transform`, `--leg-alignment`, `--phase-operator`, `--signal-quality`, `--device` and `--test-mode {holdout,augmented}`.
- Loader (`hdf5_dataset/hdf5_dataset.py`): `CombinedHDF5Dataset(paths, load_fields, ..., stats_path, normalize_fields, trim_minutes)`. Trim: raw `data[240:-240]`, coefficient blocks `data[:,15:-15]`, `target`/`weight` `[15:-15]` (`decimated_trim_steps`). Normalisation (`normalize_tensor_data`): `fhr` and `up` are z-scored; `*_st` channel 0 is linear and the others use $\log(\max(x,0)+10^{-6})$; `*_ph` use $\operatorname{asinh}$; then $(g(x)-\mu_c)/(\sigma_c+10^{-8})$ with stats from `stat_path`. The config key is `stat_path` and the loader kwarg is `stats_path`; a wrong path only warns and disables normalisation, so `teb_vae/lag_attn_rws/trainer.py` guards it. Causal stats exclude each channel's warm-up region (`calculate_dataset_stats.py`). The warm-up is rebased as $W'=\max(W-15,0)$. `read_causal_warmup(paths, trim_minutes)` validates every shard (causal, and agreeing quantile, warm-up, delay, leg alignment and phase operator) and returns a `CausalWarmup` record.
- Quality masks (signal-quality builds only): `emit_quality_mask` adds `up_valid` and `fhr_valid` on the trimmed raw grid. `ua_step_mask` and `fhr_step_mask` add `<block>_step_valid` $(T,C)$ masks from the quality bits (`bits`), the contamination fraction (`rho_epsilon`) and an optional forward extension (`hold_steps`).

### 1.3 Two-sided features: kymatio scattering + phase harmonics

- `KymatioPhaseScattering1D(J=11, Q=4, T=16, shape=5280, max_order=1)` (`hdf5_dataset/kymatio_phase_scattering.py`) wraps `kymatio.torch.Scattering1D(average=True, T=16)`. The first-order Morlet bank comes from `scattering_filter_factory`: 42 filters, centre frequencies descending. Reflect padding to 8192 (`_reflect_pad`).
- Scattering block (43, 330): $S_0=\phi\star x$, $S_1^{(k)}=\phi\star|\psi_k\star x|$, decimated ×16 inside kymatio. Channel 0 = $S_0$, channels 1..42 = $S_1$ in bank order.
- Phase harmonics (`_build_coupling_indices`, `_compute_phase_correlation`): for every pair $(i,j)$ with $\xi_j \ge \xi_i$, power $p=\xi_j/\xi_i$, $[y]^p=|y|e^{ip\arg y}$ (`_accelerate_phase`):
  $$C_{ij}(t)=\Re\Big\{\phi\star\big([y_i]^{p}\,\overline{y_j}\big)\Big\}\big|_{\downarrow16},\qquad y_k=\psi_k\star x .$$
  `_apply_phi_filter` does the low-pass and decimation by frequency-domain truncation.
- Self-pair selection (`_phase_pair_mask` in the builder): keep a pair if $\xi_i \ge f_{\min}/f_s$, $\xi_j \le f_{\max}/f_s$ and $|p-2^{k/Q}|<0.05\cdot 2^{k/Q}$ for $k\in$ `PHASE_HARMONIC_K_STEPS=(4,6,8)`, which gives $p\in\{2, 2^{1.5}, 4\}$. `FHR_PHASE_BAND_HZ=(0.008,1.00)` gives 66 `fhr_ph` pairs and `UP_PHASE_BAND_HZ=(0.008,0.05)` gives 15 `up_ph` pairs. Pairs are ordered by ascending `(i,j)`.
- Noninteger powers: for noninteger $p$, $|z|e^{ip\operatorname{Arg}z}$ has a branch jump of $2|z||\sin(\pi p)|$ at the principal-angle cut. The $2^{3/2}$ family uses noninteger $p$. The causal `integer_harmonic_v1` operator uses only the integer families $k\in\{2,4\}$ (§1.4); `ratio_power_v0` and the two-sided builds keep all three families.
- Cross pairs `fhr_up_ph` = `select_fhr_up_cross_coefficients_v2` (band A: UP below the cap, FHR in [0.008, 0.04) Hz, k∈{0..4}; band B: FHR in [0.04, 0.25] Hz, k∈{8,12,16}) → 79 pairs. Causal builds do not store them.
- Centred inputs are non-causal: a centred Morlet tap uses future samples. The two-sided models use `causal_reach_budget_s` (`teb_vae/lag_attn/channel_reach.py`), which drops channels whose 95%-energy forward reach exceeds the budget and delays the survivors by $\lceil L_{95}/\Delta\rceil$. The causal cells require `causal_reach_budget_s: null`, and the resolver refuses it together with a warm-up budget.

### 1.4 One-sided causal features and alignment modes

- Bank (`hdf5_dataset/causal_scattering.py`): `build_causal_bank(build_filter_bank(5280))` gives a complex gammatone matched to each production Morlet:
  $$\psi^c_k(t)=a_k(t)\big(e^{i2\pi\xi_k t}-\kappa_k\big),\quad a_k(t)=t^{\gamma-1}e^{-2\pi b_k t}\mathbb 1_{t>0},\quad \gamma=\texttt{GAMMATONE\_ORDER}=4,$$
  with $b_k=\sigma_k\sqrt{\ln2}/\sqrt{2^{1/\gamma}-1}$ (`gammatone_rate`), zero mean, $L^1$-normalised, and `CAUSAL_KERNEL_TAPS` $=2^{15}=32768$. The low-pass $\phi$ is a gamma envelope that sums to 1. The gamma envelope has centre group delay $\tau_g=\gamma/(2\pi b)$ and energy centroid $(2\gamma-1)/(4\pi b)=\kappa\,\tau_g$ with $\kappa=$ `ALIGNMENT_DELAY_FACTOR` $=1-1/(2\gamma)=0.875$. The code uses $\kappa$ as a delay convention.
- Chain (`scattering_block_causal`, `phase_block_causal`): causal convolution with an `'edge'` history pad (`causal_convolve`), $S_0=\phi\star x$, $S_1=\phi\star|\psi_k\star x|$, $\Phi_{ij}=\Re\{\phi\star([y_i]^p\overline{y_j})\}$, then a plain subsample `[::16]`. The torch batched twin is `CausalTorchBank.transform_batch` / `transform_batch_numpy` (`hdf5_dataset/causal_scattering_torch.py`); the builder uses it.
- Phase operators: `PHASE_OPERATOR_LEGACY = 'ratio_power_v0'` (the ratio power $p=\xi_j/\xi_i$, families `PHASE_K_STEPS=(4,6,8)`) and `PHASE_OPERATOR_INTEGER = 'integer_harmonic_v1'` (families `PHASE_K_STEPS_INTEGER=(4,8)`). Under the integer operator, `harmonic_index` takes an integer $k$ from the bank and refuses $|\xi_j/\xi_i-k|\ge0.05k$ or $k<2$; `resolve_phase_power` is shared by NumPy and Torch. The integer operator refuses a fractional pair by name, and the two-sided arm refuses the integer operator. The loader refuses file lists with mixed operators, and the stats file must record the same operator.
- Channel plan (`build_channel_plan`, `CausalChannelPlan(kept, warmup_steps, delay_s)`): the per-filter warm-up is the number of taps that enclose `CAUSAL_WARMUP_QUANTILE=0.95` of the energy (`causal_support_samples`). The composed warm-ups are $W(S_0)=W_\phi$, $W(S_1^{(k)})=W_k+W_\phi$ and $W(\Phi_{ij})=\max(W_i,W_j)+W_\phi$, and `warmup_steps=ceil(W/16)`. The builder drops channels with `warmup_steps > 330` at write time: the 7 slowest wavelets per scattering block (43→36). `delay_s` is composed the same way from $\tau_g$. The dataset records it and never compensates it.
- The warm-up is a quantile-based initialisation threshold, not an exact boundary of independence from the assumed prehistory.
- `causal_novelty_curve` (`novelty_curve` in `causal_scattering.py`): for each stored channel, $N_c(w)$ is the share of the composed envelope $g_c=|\psi_k|\star\phi$ (the slow leg for a phase pair) within $w$ stored steps, tabulated for $w=0..T$ at write time. It carries no forecast horizon. The resolver reads the run's own $H$ and each kept channel's forecast advance: $\nu_c=N_c(H+s_c)$ (`_novelty_from_curve`; `WarmupBudget.novelty_source='curve'`, `novelty_horizon_steps`). It is an envelope-mass proxy. Legacy shards carry only the scalar `causal_novelty_frac` at `LEGACY_NOVELTY_HORIZON_STEPS = 30` on the stored clock; the resolver labels those `'legacy_scalar'`, and the evaluation preflight's `novelty_proxy` record adds a `tertile_note` when the gather differs.
- **Leg alignment** (dataset-side, `LEG_ALIGNMENT_MODES=("none","envelope")`): inside each phase pair the fast leg $j$ has a smaller delay than the slow leg $i$. The skew is $\Delta_{ij}=\tau_i-\tau_j=\tau_i(1-1/p_{ij})$ (`pair_leg_skew`). `'envelope'` delays the fast leg and de-rotates it (`leg_alignment_shift`):
  $$\tilde y_j[t]=y_j[t-s_{ij}]\,e^{i2\pi\xi_j s_{ij}},\qquad s_{ij}=\operatorname{round}(\Delta_{ij}f_s)\ge0 .$$
  `'none'` multiplies the legs at the same index. The mode changes only the values of `fhr_ph`/`up_ph` and the root attribute `causal_leg_alignment`; widths, warm-ups and nominal delays stay the same. The transform functions default to `'none'` for shard compatibility; the builder `RUN_ARGS` uses `"envelope"`.
- **Channel alignment** (model-side, `channel_alignment_delays`, restated in `teb_vae/lag_attn_cfs/causal_warmup.py`): when a reference is set, the resolver reads channel $c$ at step $t-d_c$ with
  $$d_c=\operatorname{round}\!\Big(\kappa\,\frac{\tau_{\mathrm{ref}}-\tau_c}{\Delta}\Big)\ge0,\qquad \kappa=0.875,\ \Delta=4\ \mathrm{s}.$$
  Channels with $\tau_c>\tau_{\mathrm{ref}}$ are **dropped**, because a negative shift would read a future stored value. `ChannelGate(delays=...)` → `ChannelDelay` (`teb_vae/lag_attn/nets/delays.py`) applies the shifts. The dataset does not compensate channel delay; alignment is a model-load decision (§1.5). Snapping a numeric reference to an existing channel is resolver policy.
- Cross-check script: `hdf5_dataset/compare_causal_scattering.py` (arm A = shard, arm B = NumPy on Morlet, arm C = causal bank; per-channel CSV and a report).

### 1.5 Input configurations for the models

Three things decide what a run reads: (a) the shard variant, (b) the `model_config.VAE_model` keys that `resolve_warmup_budget` (`teb_vae/lag_attn_cfs/causal_warmup.py`) resolves, and (c) the source-block toggle. The streams that reach `forward` are target $=[\texttt{fhr\_st}\,\|\,\texttt{fhr\_ph}]$ $(B,T,c_y)$ and source $=[\texttt{up\_st}\,\|\,\texttt{up\_ph}]$ $(B,T,c_u)$ (`teb_vae/lag_attn_rws/task.py`).

| Case | How selected | Effect |
|---|---|---|
| two-sided inputs | shard `transform=two_sided` | used by the `lag_attn_rws/fs` cells; the causal cells **refuse** it (`read_causal_warmup` raises on non-causal shards) |
| phase operator | `causal_phase_operator: integer_harmonic_v1` or `ratio_power_v0` | checked against the shards by name (`_check_phase_operator`); declared widths 80/46 or 102/51 |
| one-sided, unaligned phase legs | shard built with `leg_alignment=none` | root attribute `causal_leg_alignment='none'`; config `causal_leg_alignment: none`, or `null` to accept any |
| one-sided, envelope-aligned legs | shard built with `leg_alignment=envelope` | config `causal_leg_alignment: envelope` |
| target warm-up gating | `causal_warmup_budget_steps: 134` | keeps target channels with $W'_c\le134$: 76/80 on integer shards (`fhr_st` 32/36, `fhr_ph` 44/44), 98/102 on `ratio_power_v0` shards (`fhr_st` 32/36, `fhr_ph` 66/66) |
| source channel subset | `causal_source_channels: [indices]` | explicit keep-index into the declared source stream before alignment; the CFS `default.yaml` keeps 42/46 (`up_st` 32/36, `up_ph` 10/10); `null` keeps every source channel |
| no channel alignment | `causal_align_reference: null`, `causal_align_reference_source: null` | pure gather, `target_align_delays=None`; every channel is read at its own availability time |
| single reference | `causal_align_reference: target_max`, `causal_align_reference_source: null` | $\tau_{\mathrm{ref}}$ = the slowest kept target channel, applied to both streams (`sweep_align_target_max.yaml`) |
| two references | `causal_align_reference: target_max`, `causal_align_reference_source: <s>` | the source stream uses its own snapped reference; `WarmupBudget.inter_stream_offset_s` records $\tau^u_{\mathrm{ref}}-\tau^y_{\mathrm{ref}}$ (`sweep_legacy_dualref_physclock.yaml`) |
| explicit float reference | `causal_align_reference: <s>` | snapped to the nearest kept target delay within $\Delta/2$, else `ValueError` (`_resolve_reference_delay`) |
| forecast clock `stored` | `causal_target_forecast_clock: stored` or absent | each target channel is scored at its own stored index, $s_c=0$; ceiling $T_{\mathrm{valid}}=T-H$ |
| forecast clock `physical` | `causal_target_forecast_clock: physical` | approximate delay compensation: $s_c=\operatorname{round}(\kappa(\tau_c-\tau_{\min})/\Delta)\ge0$; ceiling $T_{\mathrm{valid}}-\max_c s_c$; the last label endpoint is $\Delta(H+\max_c s_c)$ after the anchor |
| forecast clock `input` | `causal_target_forecast_clock: input` | $s_c=-d_c\le0$; requires an aligned target; continues the delayed input history (`sweep_target_clock_input.yaml`) |
| source without scattering block | `use_up_st: false` | source = `up_ph` only; **refused with a warm-up budget** (`lag_attn_cfs/trainer.py`) |
| reach + warm-up both set | | refused |

Resolution order inside `resolve_warmup_budget`: read the shards (`read_causal_warmup`) → cross-check the trim against `sequence_length` → build the declared vectors per stream (`_build_stream`, `TARGET_BLOCKS=("fhr_st","fhr_ph")`, `SOURCE_BLOCKS=("up_st","up_ph")`) → keep the target by budget and the source by `causal_source_channels` → check leg alignment and phase operator → resolve the target reference, then the source reference → `_align_stream` on both → forecast clock over the survivors → stride feasibility `warmup_period + anchor_stride <= ceiling`. It returns a `WarmupBudget`: `target`/`source` `StreamWarmup(keep_index, warmup_steps, align_delays, declared_delay_s, declared_novelty_frac)`, `reference_delay_s`, `source_reference_delay_s`, `target_forecast_shift`, `phase_operator`, `representation()`, and `.summary()` for the startup log.

`warmup_model_kwargs(budget, model_cls)` (`teb_vae/lag_attn_cfs/model_kwargs.py`) maps the budget to the constructor kwargs `target_keep_index`, `target_warmup_steps`, `source_keep_index` and `source_warmup_steps`. Feature cells also get `target_novelty_frac`; `target_forecast_shift` is added when the clock is not `stored`, and `target_align_delays`/`source_align_delays` when a reference is set. The CFS trainer adds `target_scored_horizon` from `resolve_target_scored_horizon` when `target_phase_fast_cutoff_hz` and `target_phase_fast_horizon` are set. These tuples land in the checkpoint's `model_kwargs`, and the task stamps `causal_representation` (phase operator, leg alignment, forecast clock) into every checkpoint that carries a resolved budget.

The effective per-channel input availability announced to the encoder is $\delta^{\mathrm{adapter}}_c=W'_c+d_c$ (`causal_inputs.py`). `AvailabilityInputAdapter` (`teb_vae/lag_attn/nets/encoders.py`) builds the mask $m_{t,c}=\mathbb 1[t\ge\delta_c]$:
$$e_t=W_x(x_t\odot m_t)+W_m(m_t-\mathbf 1)+\mathbb 1[\textstyle\sum_c m_{t,c}=0]\,e_{\mathrm{start}} .$$
The adapter builds the $W_m$ and $e_{\mathrm{start}}$ terms only when they are not constant over $t$. The mask encodes the configured initialisation policy, not signal quality. Signal quality enters through the separate quality masks (§1.2, §1.6).

### 1.6 Model: SeqVaeLagAttnCfs (conv-LSTM cell)

Data flow (`CausalWarmupInputs.forward` in `teb_vae/lag_attn_cfs/nets/causal_inputs.py`). The symbols are: $B$ batch, $T=300$, $L=$ `max_lag` $+1$ lags, $H$ horizon, $C_{\mathrm{keep}}$ kept target channels, $d_{model}=128$, $d_z=64$, $M=4$ heads × $d_{head}=32$. The CFS `default.yaml` gives $H=10$, $L=91$, $C_{\mathrm{keep}}=76$.

1. `anchor_index, anchor_valid = _build_anchor_index(B, φ, S)`: $\mathcal A(\varphi)=\{F+\varphi+kS<\text{ceiling}\}$, width $A_{\max}=\lceil(\text{ceiling}-F)/S\rceil$. With $F=134$, $S=5$ and the stored-clock ceiling $T_{\mathrm{valid}}=290$, $A_{\max}=32$; validation and test use $S=1$, $\varphi=0$, which gives 156 anchors. Padded slots repeat the last valid anchor and are flagged `False`.
2. `target=cat([y_st,y_ph])` $(B,T,c_y)$; `persistence = _anchor_target_values(target, anchors)` $(B,A,C_{\mathrm{keep}})$ if `persistence_residual` (`causal_feature_target.py`).
3. `target_gate` (gather the survivors, delay each by $d_c$) → `target_adapter` (availability mask) → `target_encoder` = `CausalConvLstmEncoder` (kernels (3,7,11,15,15), dilations (1,2,4,8,16), 2-layer LSTM) → $h_y$ $(B,T,128)$.
4. `source_gate` → `source_adapter` → `encode_source_kv`: under `lag_kv_source: conv_stem`, a `CausalConvStem` (kernels (3,5,11,15,15), dilations (1,2,4,8,16), no LSTM; RF 387 steps) → $h_u$ $(B,T,128)$ = keys **and** values. Under `adapter`, $h_u$ is the adapter output; under `encoder`, a full source encoder.
5. `prior_head(h_y, clock=_prior_clock(u_stream))` → $\mu^p,\ell^p,\ell^p_{raw}$ $(B,T,64)$ (`FullLatentPriorHead`, `teb_vae/lag_attn_rws/nets/heads.py`). The clock is the encoding of a zeroed source stream at batch 1, in eval mode, detached (`causal_inputs.py`); it is on under `prior_availability_input: true`.
6. Query `query_proj(mu_prior)` → `LagCrossAttention(h_u, mask)` (`teb_vae/lag_attn/nets/attention.py`): scores $s_{t,m,\ell}=\tfrac1{\sqrt d}(\langle q,k_{t-\ell}\rangle+\langle q,r_\ell\rangle)+b_{m,\ell}$, entmax15; returns `alpha` $(B,T,M,L)$ (lag 0 = current step) and `attended_heads` $(B,T,M,32)$. The mask is `build_lag_mask` floored at `lag_floor`. Under `source_quality_mask`, `source_key_mask` turns the per-sample source validity into a $(B,T,L)$ key mask (§12 of the companion).
7. `posterior_head(h_y, attended_heads, mu_prior, raw_logvar_prior)` → $\mu^q=\mu^p+g_t\,s_\mu\tanh(\cdot/s_\mu)$; $\ell^q$ comes from an independent head (`posterior_logvar_mode: independent`). The head is head-structured: head $m$ writes latent group $m$ (`teb_vae/lag_attn/nets/heads.py`). $g_t$ is the event gate under `posterior_event_gate` (closed by `_close_gate` where no source key is valid), else 1.
8. `_reparameterize_shared` (`lag_attn_rws/nets/model.py`): one $\epsilon$; $z^q=\mu^q+\sigma^q\epsilon$. Under `base_decode: mean`, `z_prior` **is** `mu_prior` (the same object).
9. The shared decoder runs twice on gathered latents: `decoder(z_prior.gather(anchors), persistence)` → `mu_base, logvar_base`; the same on `z_post` → `mu_full, logvar_full`, each $(B,A_{\max},H,C_{\mathrm{keep}})$. Decoder = `BaselineFutureDecoder(HorizonDecoderCore(depth 4, kernel 3, FiLM per block, 2 horizon self-attention blocks), d_model=d_z, out_channels=C_keep, dropout 0, persistence_residual)` (`teb_vae/lag_attn/nets/decoders.py`). Persistence: $\mu_{\tau,c}=w_{\tau,c}y_{t,c}+f_\theta(z)_{\tau,c}$, with $w$ seeded at $2^{-\tau/5}$.
10. `kld_tensor` → `te_analysis` → `kld_per_t` $(B,T)$, `source_kl_lag_map` $(B,T,L)$ ($=\sum_m K^{(m)}_t\alpha^{(m)}_{t,\ell}$), `kld_per_t_per_head` $(B,T,M)$.

Return dict: `mu_prior, logvar_prior, raw_logvar_prior, mu_post, logvar_post, z_prior, z_post` $(B,T,d_z)$; `target_state, source_state` $(B,T,d_{model})$; `attended_source_heads` $(B,T,M,d_{head})$; `attn_weights` $(B,T,M,L)$; `mu_base, logvar_base, mu_full, logvar_full` $(B,A_{\max},H,C_{\mathrm{keep}})$; `kld_per_t, kld_per_t_per_head, source_kl_lag_map`; scalars `mu_prior_sat_frac, delta_mu_sat_frac`; `anchor_index` (long), `anchor_valid` (bool); `persistence` $(B,A_{\max},C_{\mathrm{keep}})$ when enabled; `event_gate` $(B,T,1)$ under `posterior_event_gate`; `source_key_mask` $(B,T,L)$ and `source_open` $(B,T)$ under `source_quality_mask`. The forward has no `decoder_state` and no `delta_mu_src` source bypass; the target-only persistence connection is an explicit decoder path outside the latent.

Structural invariants: the source pathway never sees the target; the prior never sees source values (only the zeroed-source clock); KL is exactly zero at init (`_zero_init_delta_heads`, `test_zero_kl_init.py`); attention and decoder dropout are 0; `W_o` is frozen; `causal_norm: true` swaps GroupNorm for `CausalGroupNorm` on history paths (`teb_vae/lag_attn/nets/blocks.py`); `compile` must stay off (LSTM, boolean-mask indexing).

Constructor (`teb_vae/lag_attn_cfs/nets/model.py`): the full `SeqVaeLagAttnRws` signature minus `target_delays/source_delays`, plus `CAUSAL_ONLY_KEYWORDS` (`causal_inputs.py`): `target_warmup_steps, source_warmup_steps, anchor_stride, lag_floor, target_weight_st, target_weight_ph, target_align_delays, source_align_delays, target_novelty_frac, target_forecast_shift, target_scored_horizon, forecast_ar_residual, source_quality_mask`. Order: `_set_causal_inputs` → `_set_channel_weights` → `_set_target_novelty` → `_set_likelihood_structure` → `super().__init__(**forwarded, target_delays=target_align_delays, source_delays=source_align_delays)` → `_validate_causal_geometry` (stride against span; floor check) → `_register_channel_weights` → `_register_likelihood_structure`. The constructor defaults are `warmup_period=134`, `c_y=102`, `c_u=51`, `horizon=30`. The driver builds kwargs by an `inspect.signature` sweep, so it **silently drops** any config key that is not in the signature (`lag_attn_rws/trainer.py`).

Anchor floor refusal (`causal_feature_target.py`):
$$F\ \ge\ \max\Big(\max_c(W'_c-s_c)-1,\ \max_c(W'_c+d_c)\Big).$$
With no shifts and budget 134, this requires $F\ge134$ (536 s).

Likelihood structure (feature cells): `target_scored_horizon` gives $H_c$ per declared channel (`scored_horizon.py`): $H_c=H_{\mathrm{fast}}$ for an `fhr_ph` channel whose slow-leg centre frequency $\xi_{i(c)}$ exceeds `target_phase_fast_cutoff_hz`, else $H$. It is registered as the buffer `target_cell_mask` $(H,C_{\mathrm{keep}})$. `forecast_ar_residual` adds the parameter `target_ar_logit` $(C_{\mathrm{keep}},)$, with $\phi_c=\tanh(a_c)$ seeded at $a_c=0$. `FeatureForecastTarget.forecast_likelihood_kwargs` passes both to every scoring site.

Geometry object `TrimmedRawGeometry(raw_len=4800, decimation=16, horizon=H, warmup=134)` (`teb_vae/lag_attn_rws/nets/geometry.py`): $T=300$, $T_{\mathrm{valid}}=T-H$, and the forecast of anchor $t$ covers decimated steps $t+1..t+H$.

### 1.7 Model: SeqVaeLagAttnTrfCfs (conv-Transformer cell)

The same two mixins compose over `SeqVaeLagAttnTrfRws` (`teb_vae/lag_attn_transformer_rws/nets/model.py`). Differences from the conv-LSTM cell:

| Aspect | conv-LSTM (`lag_attn_rws`) | conv-Transformer (`lag_attn_transformer_rws`) |
|---|---|---|
| encoder | `CausalConvLstmEncoder` (dilated causal conv stack ‖ LSTM, GroupNorm/LayerNorm) | `CausalConvTransformerEncoder` (`nets/encoders.py`): `GatedCausalConvBlock` stem (depthwise kernels (5,9), dilations (1,2), reach 21 steps) → `CausalTransformerBlock`×N (RMSNorm, RoPE, SwiGLU, LayerScale 1e-2, SDPA `is_causal` or band mask) → RMSNorm |
| attention block counts | — | `target_attention_blocks`, `source_attention_blocks`, `source_attention_window` (the last two act only under `lag_kv_source: encoder`) |
| removed ctor keys | — | `lstm_layers, encoder_extra_dilations, encoder_extra_kernel, conv_norm_groups, causal_norm, decoder_out_channels` (passing one raises) |
| added ctor keys | — | `encoder_conv_kernels, encoder_conv_dilations, encoder_num_heads, encoder_d_ff, target_attention_blocks, source_attention_blocks, source_attention_window` |
| LR schedule | epoch milestones | + step-granular `general_config.lr_warmup_steps` (`LambdaLR` warm-up, `lag_attn_transformer_rws/task.py`) |
| compile | refused | read from config |
| K/V source under `conv_stem` | `CausalConvStem`, RF 387 steps | `GatedCausalConvStem`, RF 21 steps |

MRO: `SeqVaeLagAttnTrfCfs → CausalWarmupInputs → CausalFeatureForecastTarget → FeatureForecastTarget → SeqVaeLagAttnTrfRws → Module`. The reversed order would build a 16-wide decoder.

The model keys of the trf-cfs `default.yaml` that differ from the CFS `default.yaml`, apart from the encoder keys: `horizon: 30`, `anchor_stride: 15`, `max_lag: 37`, `lag_kv_source: adapter`, `logvar_clamp: [-6, 3]`, `source_dropout: 0.2`, `horizon_weight_halflife_steps: 30`, `posterior_event_gate: true`, `lambda_event_gate: 0.05`, `source_quality_mask: true`, `source_event_mask: false`, `target_quality_mask: true`, `forecast_ar_residual: true`, `target_phase_fast_cutoff_hz: 0.03`, `target_phase_fast_horizon: 5`, `event_weight_floor: 0.0`, `event_weight_reach_steps: 55`, `event_weight_ua_valid: true`. Its dataset block sets `emit_quality_mask: true`, `ua_step_mask` and `fhr_step_mask` (`bits: invalid`), and points at a signal-quality build.

### 1.8 Objective

The single home is `teb_vae/lag_attn_rws/nets/losses.py::compute_loss`. `FeatureForecastTarget.compute_loss` (`lag_attn_fs/nets/feature_target.py`) reaches it after `CausalFeatureForecastTarget.compute_loss` substitutes the pooled validity. Gaussian scores are in nats per anchor. The channel and horizon weights make the training block a weighted score, not a joint log-density.

$$\mathcal L=\lambda_{\mathrm{full}}D^{w}_1+\lambda_{\mathrm{base}}D_0+\beta(e)\,\mathrm{KL}_{\mathrm{train}}+\beta_p R_p+\lambda_{\mathrm{ms}}\mathcal L_{\mathrm{ms}}+\lambda_{\Delta}\mathcal L_{\Delta}+\lambda_{\mathrm{boundary}}\mathcal L_{\mathrm{boundary}}+\lambda_g\,\beta(e)\,\mathbb E_t[g_t]$$

- Target block $Y^+[b,a,\tau,k]=Y[b,\,t_a+1+\tau+s_k,\,\mathrm{keep}[k]]$ $(B,A,H,C_{\mathrm{keep}})$ (`causal_feature_target.py`; stored clock: `feature_target.py`). The target is the loader-normalised concatenated `[fhr_st‖fhr_ph]` stream, gathered by the gate keep-index only, with **no delay** (`lag_attn_fs/task.py`).
- Per-coefficient score (`raw_sample_score`): `gaussian_nll` $\tfrac12[\log2\pi+\ell+(y-\mu)^2e^{-\ell}]$ (or `mse` $(y-\mu)^2$), then `* channel_weight[c]` and `* horizon_weight[τ]`. Under `forecast_ar_residual` the residual $r_\tau=y_\tau-\mu_\tau$ is replaced by the AR(1) innovation $r_\tau-\phi_c r_{\tau-1}$ (`ar_innovation`, with a variance correction after masked steps). Under a per-channel scored horizon, `cell_mask` zeroes the unscored cells, and the scored-cell count replaces $H\cdot C$.
- $D_1$ = `nll_full_block` and $D_0$ = `nll_base_block`: the score × forecast mask, summed over $(H,C)$, averaged over contributing anchors (`masked_raw_likelihood`); `nll_*_sample = block/(H·C)`. $D^{w}_1$ = `nll_full_event_block` is the full block as a weighted mean over anchors (`_event_weighted_full_block`): anchors within `event_weight_reach_steps` after a detected contraction onset weigh 1, and the others weigh `event_weight_floor`. Without `event_weight_floor`, $D^{w}_1=D_1$.
- Masks (`teb_vae/lag_attn_rws/nets/raw_masks.py`): `forecast_mask` $m_{a,\tau}=\mathbb 1[t_a\ge w]\,v_{t_a}\,v_{t_a+1+\tau}\,\mathbb 1[\mathrm{coverage}\ge\texttt{coverage\_floor}]\,\mathrm{valid}_a$ with $v=\mathbb 1[\texttt{weight}\ge1.0]$ (`VALID_THRESHOLD`); `kl_mask` = the contributing anchors scattered back to $(B,T)$. Under a non-stored clock, `weight` is first pooled: $\tilde w_u=\min_{j\in[u+s_{\min},u+s_{\max}]}w_j$ (`pooled_scored_weight`). Under `target_quality_mask`, the target step-validity mask multiplies `weight`, so it reaches the forecast mask, the coverage floor and the KL support. Shift-span pooling protects target endpoints, not every raw sample in a filter's history.
- $\mathrm{KL}$ per step and dimension (`kld_tensor`): $\tfrac12[\ell^p-\ell^q+(e^{\ell^q}+(\mu^q-\mu^p)^2)e^{-\ell^p}-1]$. `kl_train` clamps each dimension at `free_bits` before masking, sums over $d_z$ and averages over the KL support (`masked_source_kl`). `source_conditioned_kl_raw` (no floor) is the information-rate readout. `kld_active_frac` counts dimensions with mean KL > `KLD_ACTIVE_EPS=1e-2`.
- $R_p=\sum_d\tfrac12(e^{\ell^p}-1-\ell^p)$ is the prior scale anchor (`masked_prior_rate`), weighted by $\beta_p$ = `beta_prior`.
- KL interpretation: $\mathbb E_{Y,U}\mathrm{KL}(q(Z\mid Y,U)\Vert p(Z\mid Y))=I_q(Z;U\mid Y)+\mathbb E_Y\mathrm{KL}(q(Z\mid Y)\Vert p(Z\mid Y))$, where the aggregate $q(Z\mid Y)$ averages over $U\mid Y$. The latent readout therefore includes prior mismatch and is not by definition a future-feature transfer entropy.
- Event gate term: $\lambda_g$ = `lambda_event_gate` charges the mean gate over the open anchors, divided by the KL anchor count, and it rides the $\beta$ schedule. A nonzero $\lambda_g$ without `posterior_event_gate` is refused.
- Shape terms: multiscale $L_1$ (rates (1,4,16)), derivative Huber ($\delta=1$) and boundary gap. All are 0.0 in the CFS configs, because the channel axis has no order. `lambda_boundary≠0` is refused with any anchor set (also in the CFS trainer preflight).
- $\beta(e)$: `beta_schedule {kind: linear_warmup, start, end, warmup_epochs}` → $\beta(e)=\mathrm{start}+(\mathrm{end}-\mathrm{start})\min(1,e/\texttt{warmup\_epochs})$ (`lag_attn_rws/task.py`).
- Channel weight (`_resolve_channel_weights`): each kept channel gets `target_weight_st` (declared index < `TARGET_BLOCK_SPLIT=36`) or `target_weight_ph`, rescaled by $C_{\mathrm{keep}}/\sum w$. Horizon weight (`horizon_decay_weight`): $w_\tau=H\,2^{-\tau/\lambda}/\sum_{\tau'}2^{-\tau'/\lambda}$ with $\lambda$ = `horizon_weight_halflife_steps`. The evaluation applies neither weight; `preflight.json` records `objective_weights` (the phase share of the renormalised channel-weight mass, read off the model).
- Decoding policy: the training path decodes the base branch at $\mu^p$ and samples the full latent. The evaluation's `mc_predictive_block` samples both branches with common random numbers. At one draw, the evaluation's base score is not the training-path score under `base_decode: mean`; `test_eval_metrics.py::test_at_one_draw_the_base_branch_is_not_the_training_path_under_mean_decoding` pins this, the common-random-number identity and the $-\infty$ log-variance recovery.
- Reported metrics: `compute_loss` returns `total_loss`, `nll_full_block`, `nll_full_sample`, `nll_full_event_block`, `nll_base_block`, `nll_base_sample`, `pred_gap` ($=D_0-D_1$), `source_conditioned_kl_raw/train`, `kld_active_frac`, `prior_rate`, `event_gate_mean`, the `aux_*` terms, the weights and log-variance diagnostics. The FS mixin adds `pred_gap_tau_first/last` and `pred_gap_st/ph`; the CFS mixin adds the warm/novelty gap tertiles, `target_warm_frac`, `anchors_per_sample` and `source_lag_warmth_frac_st/_ph`; the task adds `main_loss`, the saturation fractions, the validation-only permutation control (`nll_shuffled_block`, `kld_shuffled`, `shuffle_penalty`) and `kld_source_null`, and the training-only spike-breaker and gradient readouts. The full list is `LagAttnCfsTrainer.TRACKED_METRICS` (`lag_attn_cfs/trainer.py`).

### 1.9 Training

- Entry: `python -m teb_vae.lag_attn_cfs.trainer --config teb_vae/lag_attn_cfs/configs/default.yaml`. Multi-GPU: `TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=<N> -m teb_vae.lag_attn_cfs.trainer --config ...`, with $N$ = `len(cuda_devices)`. IDE Run button: edit `RUN_CONFIG` at the bottom of the trainer (`"teb_vae/lag_attn_cfs/configs/default.yaml"`) and press Run. The transformer twin is `teb_vae/lag_attn_transformer_cfs/trainer.py`.
- `main()` order (`lag_attn_rws/trainer.py`): `resolve_config_file` (base-chain merge, `teb_vae/lag_attn/config.py`: dicts merge, lists and scalars replace) → driver constructor → guards `_check_stat_path`, `_check_declared_widths_against_shard`, `_check_raw_target_normalized(TARGET_FIELDS=("fhr_st","fhr_ph"))`, `_check_causal_budget_resolves` → `trainer_cls.preflight` (CFS: no `fhr_up_ph`, `lambda_boundary==0`, `guid` and `epoch` in `load_fields`, `use_up_st` with a budget, floor pairs with the resolved budget) → `setup_config` (run dir, seed, logs, MLflow) → write `resolved_config.yaml` beside the checkpoints → `GraphDataModule` → `create_model` → `train_model`.
- `create_model`: `_build_model_kwargs` = the signature sweep of `VAE_model` keys + `warmup_model_kwargs(resolve_warmup_budget(config))` + `target_scored_horizon`. The task is built with `beta_schedule, kld_beta, beta_prior, lambda_*, likelihood, free_bits`, the event-weight and quality-mask keys, and `compile_model`. The CFS driver logs the resolved anchor geometry, pushes `seed` into the hparams (tile phase) and hands `warmup_budget` to the task.
- Task step: `_build_forward_inputs` returns `(y_st, y_ph, u_stream, anchor_phase, anchor_stride)`, plus `source_valid` under `source_quality_mask` (`lag_attn_cfs/task.py`). On train, the phase is $\varphi_b=\mathrm{blake2b}(\texttt{guid}\|\lfloor\texttt{epoch}\rfloor\|\texttt{train\_epoch}\|\texttt{seed})\bmod S$; validation and test use `(0,1)` dense (`DENSE_STAGES`). `LightningModelBase.training_step` (`train/pl_model_base.py`) runs the spike breaker on `metrics['main_loss']`.
- Optimiser: AdamW with `lr` from config, `weight_decay` 1e-4 by default, and `lr_milestone` epoch milestones (×0.1). The transformer cells add a `LambdaLR` step warm-up.
- DDP (`lag_attn_rws/trainer.py`): `DDPStrategy(find_unused_parameters = likelihood != 'gaussian_nll', broadcast_buffers=False, gradient_as_bucket_view=True)`; `'auto'` on one device; no `static_graph` (the spike breaker changes the graph).
- Callbacks (`train_model`): metrics CSV (`train_results/metrics_history.csv`), loss HTML, hyperparameter plot, `ModelCheckpoint` on `val/total_loss` with `save_top_k` 3 and filename `lag-attn-cfs-{epoch:02d}`, a secondary checkpoint on `secondary_monitor` (`val/nll_full_block`), and the diagnostic page callback under the config block `lag_attn_rws_plotting` (the name must not change). Early stopping is configured under `callbacks.early_stopping`.
- Checkpoint contract: a `model_class` stamp + `model_kwargs` + `causal_representation`. Load with `train/graph_models_utils.py` (`check_model_class`, `load_checkpoint_strict`).
- Sweep configs in `lag_attn_cfs/configs/` (each chains `base: default.yaml`; identity keys `run_name`/`tags.variant`; inventory in `lag_attn_cfs/tests/test_sweep_configs.py`). Some arms change more than one factor; for example, the input clock needs an alignment reference.

| file | delta against `default.yaml` |
|---|---|
| `sweep_anchor_stride_1.yaml` | `anchor_stride: 1` |
| `sweep_horizon_15.yaml` | `horizon: 15`, `anchor_stride: 7` |
| `sweep_floor_150.yaml` | `warmup_period: 150` |
| `sweep_horizon_depth_3.yaml` | `horizon_depth: 3` |
| `sweep_lag_bias_decay.yaml` | `alibi_slope_scale: 1.0` |
| `sweep_align_target_max.yaml` | `causal_align_reference: target_max` |
| `sweep_lag_kv_adapter.yaml` | `lag_kv_source: adapter` |
| `sweep_target_clock_input.yaml` | `causal_target_forecast_clock: input` + `causal_align_reference: target_max` |
| `sweep_legacy_dualref_physclock.yaml` | `causal_phase_operator: ratio_power_v0`, `c_y: 102`, `c_u: 51`, `causal_source_channels: null`, `causal_align_reference: target_max`, `causal_align_reference_source: 288.2672`, clock `physical`, `REPOINT_ME_causal` shards |
| `tiny.yaml` | 1 epoch, 1 device, `d_model` 32, `d_z` 8, `max_lag` 8, `likelihood: mse`, fixtures `teb_vae/lag_attn/tests/fixtures/tiny_shard_causal_int.hdf5` + `tiny_stats_causal_int.hdf5` |
| `planted.yaml` | identifiability instrument on `tiny_shard_causal_planted.hdf5` (`ratio_power_v0`, `target_max`, `anchor_stride: 1`, `horizon: 30`); run through `lag_recovery_check.py` |
| `smoke_hie.yaml` | dev-box run on `output/causal_scattering/hie_cs_alighned/hie_cs_alighned.hdf5` (`ratio_power_v0`), with both splits on the same shard |

The trf-cfs `configs/` directory also holds `sweep_all_cells_scored.yaml`, `sweep_factorised_likelihood.yaml`, `sweep_lag_kv_conv_stem.yaml`, `sweep_no_event_gate.yaml`, `sweep_no_event_weight.yaml`, `sweep_no_quality_mask.yaml`, `sweep_source_dropout_03.yaml`, `sweep_source_event_mask.yaml`, `prod_q4_*.yaml` and `ua_s0_only.yaml`.

### 1.10 Configuration reference (`teb_vae/lag_attn_cfs/configs/default.yaml`; trf-cfs differences in §1.7)

| key | value | meaning / consumer |
|---|---|---|
| `general_config.seed` | 42 | determinism + tile phase |
| `general_config.cuda_devices` | [0..6] | 7 ranks |
| `general_config.lr` / `lr_milestone` / `epochs` / `batch_size` | 3e-4 / [400,800] / 5000 / 128,128 | framework |
| `general_config.accumulate_grad_batches` | 1 | framework |
| `general_config.lr_warmup_steps` | (transformer cells only) | `LR_WARMUP_STEPS_KEY` in `lag_attn_transformer_rws/trainer.py` |
| `general_config.plot_frequency` | 5 | loss HTML + diagnostic page cadence |
| `VAE_model.beta_schedule` | linear_warmup 0→1 over 50 epochs | `lag_attn_rws/task.py` |
| `free_bits` | 0.0 | KL floor per dimension |
| `beta_prior` | 0.1 | prior scale anchor weight |
| `likelihood` | gaussian_nll | `mse` → DDP find-unused path |
| `lambda_full`, `lambda_base` | 1.0, 1.0 | |
| `lambda_ms`, `lambda_deriv`, `lambda_boundary` | 0, 0, 0 (boundary must be 0) | |
| `coverage_floor` | 0.9 | minimum valid window fraction per anchor |
| `d_model`, `d_z`, `num_heads`, `d_head` | 128, 64, 4, 32 | `num_heads·d_head==d_model`, `d_z % num_heads==0` |
| `horizon`, `raw_per_step`, `sequence_length` | 10, 16, 300 | geometry |
| `warmup_period` | 134 | anchor floor $F$ (536 s) |
| `c_y`, `c_u`, `use_up_st` | 80, 46, true | declared widths under `integer_harmonic_v1` (checked against the shard) |
| `lstm_layers`, `dropout`, `source_dropout` | 2, 0.1, null | conv-LSTM only; `source_dropout` null → source pathway 0.1, posterior fusion 0 |
| `decoder_hidden` | 256 | |
| `logvar_clamp`, `mu_scale`, `delta_mu_scale`, `delta_logvar_scale` | [-5,3], 5, 3, 2 | bounds |
| `base_decode` | mean | the prior branch decodes $\mu^p$ |
| `posterior_logvar_mode` | independent | with `head_init_calibration`, keeps the zero-KL init |
| `causal_norm` | true (conv-LSTM only) | CausalGroupNorm |
| `causal_reach_budget_s` | null (required) | two-sided guard, refused with a warm-up budget |
| `causal_warmup_budget_steps` | 134 | target keep threshold on $W'_c$ |
| `causal_source_channels` | 42 indices (drops `up_st` 32..35) | source keep-index before alignment |
| `causal_align_reference` | null | no input alignment |
| `causal_align_reference_source` | null | second reference for the source stream |
| `causal_target_forecast_clock` | stored | availability-time labels, ceiling $T-H$ |
| `causal_leg_alignment` | envelope | expected shard attribute |
| `causal_phase_operator` | integer_harmonic_v1 | expected shard attribute |
| `anchor_stride` | 5 | train tiling; must satisfy $F+S\le$ ceiling |
| `lag_floor` | 0 | lag mask floor |
| `target_weight_st`, `target_weight_ph` | 1.0, 0.1 | block weights (ratio) |
| `horizon_weight_halflife_steps` | 5.0 | horizon decay $\lambda$ |
| `max_lag`, `use_entmax`, `attention_grad_checkpoint` | 90, true, false | $L=91$ (364 s) |
| `lag_kv_source` | conv_stem | K/V from the conv stem; no deep source encoder |
| `lag_bias_init`, `alibi_slope_scale` | alibi_decay, 0.0 | flat learnable $(M,L)$ bias seed |
| `query_uses_logvar` | false | |
| `prior_availability_input` | true | prior clock from the zeroed source |
| `horizon_depth`, `horizon_kernel`, `horizon_film`, `horizon_attention_blocks` | 4, 3, true, 2 | decoder RF $=1+(k-1)(2^d-1)=31\ge H+1$ |
| `persistence_residual` | true | feature cells only |
| `horizon_embed_std`, `head_init_calibration`, `a_head_gain` | 0.8, true, 2.0 | init policy |
| `encoder_extra_dilations`, `encoder_extra_kernel`, `conv_norm_groups` | [8,16], 15, null | conv-LSTM only |
| `posterior_event_gate`, `lambda_event_gate` | absent (off) | posterior event gate and its sparsity weight |
| `source_quality_mask`, `source_event_mask`, `target_quality_mask` | absent (off) | per-sample source key mask; contraction-only source keys; target step-validity mask |
| `forecast_ar_residual` | absent (off) | AR(1) residual likelihood |
| `target_phase_fast_cutoff_hz`, `target_phase_fast_horizon` | absent (off) | per-channel scored horizon (set both or neither) |
| `event_weight_floor`, `event_weight_reach_steps`, `event_weight_ua_valid` | absent (off) | contraction-locked anchor weight on the full branch |
| `encoder_conv_kernels/dilations`, `encoder_num_heads`, `encoder_d_ff`, `target/source_attention_blocks`, `source_attention_window` | absent | transformer only |
| `core_model_checkpoint` | null | warm start (strict) |
| `dataset_config.vae_train_datasets/vae_test_datasets` | `…/REPOINT_ME_causal_int/pre_training_dataset/{train,test}_dataset_{cs,no_cs}.hdf5` | placeholders |
| `dataset_config.stat_path` | `…/REPOINT_ME_causal_int/stats.hdf5` | the key is `stat_path` |
| `dataloader_config.normalize_fields` | [fhr, up, fhr_st, fhr_ph, up_st, up_ph] | |
| `dataset_kwargs.load_fields` | [fhr, up, fhr_st, fhr_ph, up_ph, up_st, weight, guid, epoch] | `guid`, `epoch` required |
| `dataset_kwargs.trim_minutes` | 1.0 | 330→300 steps |
| `dataset_kwargs.epoch_min` | -48000 | |
| `advanced_config.trainer.precision` | "32-true" | |
| `trainer.gradient_clip_val` | 4000 | norm clip |
| `trainer.compile` | false | |
| `trainer.num_sanity_val_steps` | 0 (required) | |
| `spike_breaker` | enabled, multiplier 5, ema_decay 0.02, warmup 100, `ema_floor 1.0e+9` (relative test off), `additive_margin 2200`, max skips 25 | `train/pl_model_base.py` |
| `callbacks.early_stopping` | val/total_loss, patience 50 | |
| `callbacks.model_checkpoint` | val/total_loss, top-3, `secondary_monitor: val/nll_full_block` | |
| `callbacks.lag_attn_rws_plotting` | enabled, 2 examples, pdf | |
| `tracking.mlflow.run_name` = `tags.variant` | `lag_attn_cfs_intphase_unaligned_stored` | arm identity guard |

### 1.11 Evaluation

- Run: `python -m teb_vae.lag_attn_cfs.eval.run --checkpoint <run>/model_checkpoints/<name>.ckpt`, or edit `RUN_ARGS` in `eval/run.py` and press Run. The config is the checkpoint's `resolved_config.yaml` + `eval/configs/eval_overrides.yaml` (holdout causal shards and the `eval_config` block, for example `seed: 42` and `num_mc_samples: 8`). `RUN_ARGS` / CLI: `checkpoint, output_dir, overrides, device, num_samples, max_batches, only, skip`. The forward runs densely (`anchor_phase=0, anchor_stride=1`). Offline re-run: `--output-dir <finished run> --only <analysis>`. `eval/verify.py summary.json` is a torch-free gate; `check_run.py --run-dir` checks a run in flight from its CSV. The transformer twin binds the same pipeline through `lag_attn_transformer_cfs/eval/binding.py` (`TRF_CFS_BINDING`).
- Outputs in `<run>/eval_results/`: `summary.json` (headline scalars, including `pred_gap_mc_nats` and `pred_gap_train_path_nats`; verdicts; sanity; run context), `preflight.json` (checks; the budget record with `content_lag_s` and `novelty_proxy`; `objective_weights`), `per_sample.csv`, `per_anchor.parquet`, `per_anchor_vectors.npz`, and one directory per analysis.
- Analyses in run order (`run.merged_analysis_functions(CFS_BINDING)`): forecast, coupling, perm_control, latent, lag_kl, attention, calibration, residual, distributions, trajectory, time_to_delivery, second_stage, events, sufficiency, samples, recording_traces, warmup, source_null, time_shift, occlusion, lag_clocks, lag_kld_scaled, lag_high_kl, spectral_skill, attribution, cross_subgroup. Metric definitions are in `eval/metrics.py`; figures are described in `eval/FIGURE_GUIDE.md`.
- Baselines (`baseline_forecasts` in `eval/metrics.py`): `persistence` (last observed step), `climatology` (0, the z-scored population mean) and `segment_mean` (the running mean of the observed warm steps up to the anchor). All three are causal and are built on the gathered kept channels.
- Approximate source-to-label content lag on the stored timeline, before source-encoder history mixing: $L_{j,c}(\ell,h)=\Delta(\ell+1+h+s_c+d^u_j)+\delta^u_j-\delta^y_c$, with source input shifts $d^u_j$, target shifts $s_c$ and filter-content-delay summaries $\delta$ (`content_lag_seconds` and `summarise_content_lag` in `warmup_budget.py`). The shared `physical_lag_seconds` (`teb_vae/lag_attn/nets/lag_report.py`) computes $\Delta(\ell+1+h)+\kappa(\tau^u_{\mathrm{ref}}-\tau^y_{\mathrm{ref}})$, where $\tau^y_{\mathrm{ref}}$ is the reference of the **scored** target; `warmup_budget.py` selects that reference.
- The `stored`, `input` and `physical` clocks define different forecast targets and anchor populations, so absolute NLLs are not comparable across them.

### 1.12 Tests and pinned invariants

- Fixtures: `teb_vae/lag_attn/tests/fixtures/{tiny_shard.hdf5, tiny_shard_causal.hdf5, tiny_shard_causal_int.hdf5, tiny_shard_causal_planted.hdf5, tiny_stats*.hdf5}` (built by `scripts/make_tiny_shard.py`); committed real causal fixture `hdf5_dataset/tests/data/causal_fixture.hdf5`.
- `lag_attn_cfs/tests/conftest.py`: the `SHIPPED_*` literals (`SHIPPED_BUDGET_STEPS=134`, `SHIPPED_WARMUP_PERIOD=134`, `SHIPPED_ALIGN_REFERENCE="target_max"`, `SHIPPED_LEG_ALIGNMENT="envelope"`, `SHIPPED_PHASE_OPERATOR="ratio_power_v0"`, `SHIPPED_HORIZON=30`, `SHIPPED_SEQUENCE_LENGTH=300`, `SHIPPED_TRIM_MINUTES=1.0`) describe the committed legacy tiny fixture that the unit tests are built on, not `default.yaml`. `INT_C_Y`, `INT_C_U` and `INT_CAUSAL_SHARD` describe the integer fixture; `test_config_load.py` reads the config on its own. Tiny geometry: `TINY_SEQ_LEN=24, TINY_HORIZON=4, TINY_WARMUP_PERIOD=5, TINY_STRIDE=4, TINY_BUDGET_STEPS=6`. Helpers include `tiny_warmup_kwargs`, `tiny_align_kwargs`, `shipped_warmup_kwargs`, `make_stub_batch` and `make_task`.
- Key tests: `test_causality.py` (no future leak; a perturbation after the anchor reaches no earlier input), `test_anchors.py`, `test_invariants.py`, `test_forward_contract.py` (return keys), `test_checkpoint_contract.py`, `test_objective.py`, `test_zero_kl_init.py`, `test_causal_warmup.py`, `test_warmup_budget.py`, `test_config_load.py` (parity against `lag_attn_fs` outside declared exemptions), `test_sweep_configs.py`, `test_eval_launch.py` (`ENTRY_POINTS`: every runner with a `__main__` block). The trf-cfs package adds `test_event_mechanisms.py`, `test_source_quality_mask.py` and `test_likelihood_structure.py`. The `test_eval_*.py` files and `test_train_smoke.py` are the slow half. Run one module with `.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_cfs/tests/test_<module>.py -q`.
- Feature-extraction tests: `hdf5_dataset/tests/test_phase_operator.py` (phase-operator continuity, legacy branch jump, NumPy/Torch agreement, leg-alignment phasor), `hdf5_dataset/tests/test_preprocessing_availability.py` (sanitiser prefix behaviour).
- Shared-net stability: `teb_vae/lag_attn/tests/test_shared_code_stability.py` pins the ancestor parameter count (363,222) and forward fingerprints.

---

## 2. lag_attn_crws + lag_attn_transformer_crws

### 2.1 What CRWS is compared with CFS

CRWS uses the same one-sided inputs as §1.5: the same three tensors, the same warm-up budget, the same alignment machinery and the same tiling. The **target is the raw 4 Hz FHR** future: the decoder emits $R=16$ raw samples per horizon token, so the block is $H\cdot R=30\times16=480$ samples per anchor. The target has no warm-up and no group delay. The direct control is `lag_attn_rws` (same target and objective, two-sided inputs). CRWS has no persistence residual (a raw block has no per-channel level) and no second source reference.

- `SeqVaeLagAttnCrws(CausalRawInputs, SeqVaeLagAttnRws)` (`teb_vae/lag_attn_crws/nets/model.py`); `SeqVaeLagAttnTrfCrws(CausalRawInputs, SeqVaeLagAttnTrfRws)` (`teb_vae/lag_attn_transformer_crws/nets/model.py`). There is **one mixin, not two**: no `CausalFeatureForecastTarget`, so `_default_decoder_out_channels` stays `raw_per_step`. Composing the feature mixin by mistake would build a $C_{\mathrm{keep}}$-wide decoder against a $(B,A,H,16)$ target.
- `CausalRawInputs(CausalWarmupInputs)` (`teb_vae/lag_attn_crws/nets/causal_raw_inputs.py`) binds by reference `SOURCE_BLOCK_SPLIT=36`, `TARGET_BLOCK_SPLIT=36`, `_resolve_block_warm_steps`, `_anchors_per_sample` and `_source_lag_warmth` from the CFS target mixin.
- Tasks: `SeqVaeLagAttnCrwsTask(SeqVaeLagAttnRwsTask)` (`teb_vae/lag_attn_crws/task.py`), directly from RWS and **not** through FS. It binds `anchor_phase, _phase_field, resolve_anchor_geometry, _build_forward_inputs, _mu_gap_rms, _added_metrics, input_stream_panels, input_budget_figure` from the CFS task and has its own `forecast_rows` and `compute_loss_and_metrics`. `SeqVaeLagAttnTrfCrwsTask(SeqVaeLagAttnCrwsTask, SeqVaeLagAttnTrfRwsTask)` has an empty body (`teb_vae/lag_attn_transformer_crws/task.py`).
- Trainers: `LagAttnCrwsTrainer(LagAttnRwsTrainer)` (`teb_vae/lag_attn_crws/trainer.py`; `TARGET_FIELDS` inherited `("fhr",)`, `CHECKPOINT_STEM="lag-attn-crws"`); `LagAttnTrfCrwsTrainer(LagAttnCrwsTrainer, LagAttnTrfRwsTrainer)` (`teb_vae/lag_attn_transformer_crws/trainer.py`; class attributes only; `_build_model_kwargs` and `create_model` run on both parents through `super()`).
- Constructor keywords: the base signature minus `target_delays`, `source_delays`, `persistence_residual` and `posterior_event_gate`, plus 6 causal keys `target_warmup_steps, source_warmup_steps, target_align_delays, source_align_delays, anchor_stride, lag_floor`. There is **no** `target_weight_*`, `target_novelty_frac`, `target_forecast_shift`, `target_scored_horizon`, `forecast_ar_residual` or `source_quality_mask`. `decoder_out_channels` still exists on the conv-LSTM cell's signature and is absent on the transformer cell.

### 2.2 Inputs: CausalRawInputs

CRWS reads the same HDF5 keys as §1.5, plus `fhr` (raw; it must be in `load_fields` **and** `normalize_fields`) and `weight` (required; `_check_raw_target_fields` in `crws/trainer.py`). `_build_raw_target` is the RWS one: `(batch.fhr (B,4800), batch.weight (B,300))` (`lag_attn_rws/task.py`).

The CRWS `default.yaml` sets `causal_align_reference: 42.21` and no source reference or forecast clock. The CRWS shards use `ratio_power_v0` ($c_y=102$, $c_u=51$).

- Clock convention: with a raw target, $\tau^y\equiv0$, and the stored-timeline approximation of the lag is $L(\ell,h)=\Delta(\ell+1+h)+\kappa\tau_{\mathrm{ref}}$. The 42.21 s reference snaps to 42.2066 s (a filter near 0.1109 Hz).
- Channel cost: budget 134 keeps 98/102 target-stream inputs and all 51 source inputs. Alignment then drops every channel slower than 42.2066 s, which leaves target-stream input **38/102** (`fhr_st` 17/36, `fhr_ph` 21/66) and source **17/51** (`up_st` 17/36, **`up_ph` 0/15**: the whole block is dropped). Shifts span 0..6 steps. `use_up_st: false` is refused, because the source would be empty.
- Floor (`CausalRawInputs._check_anchor_floor`): the same two inequalities, as an input-warmth policy (a raw target has no validity constraint). `warmup_period: 134` is kept equal to CFS.
- Readouts: only `source_block_warm_st/_ph` (`_resolve_warmup_readout_constants`); no target warm fraction and no tertiles.

### 2.3 Model: SeqVaeLagAttnCrws

The forward is the tiled forward of §1.6, inherited, with `persistence=None` and decoder width 16. The outputs `mu_base, logvar_base, mu_full, logvar_full` are $(B,A_{\max},H,16)$. With `anchor_stride: 30` on the span $270-134=136$, $A_{\max}=5$; validation and test use 136 dense anchors (the ceiling stays $T_{\mathrm{valid}}=270$; there is no forecast clock). The anchored raw target is `gather_anchored_future_target(fhr_raw, geometry, anchors, future_index)`: $X^+[b,a,\tau,r]=x[b,\ \texttt{future\_index}[\mathcal A[b,a],\tau,r]]$ with `future_index[t,τ,r]=16(t+1)+16τ+r` (`lag_attn_rws/nets/raw_targets.py`). Without an anchor set, it falls back to the dense builder. `compute_loss` is the shared objective with `block_width=geometry.r=16`, the horizon weight and no channel weight; it adds `anchors_per_sample` and `source_lag_warmth_frac_st/_ph`.

### 2.4 Model: SeqVaeLagAttnTrfCrws

This is composition only (`lag_attn_transformer_crws/nets/model.py`): the encoder swap of §1.7 with the CRWS input mixin. The signature is `SeqVaeLagAttnTrfRws` minus the delays, `persistence_residual` and `posterior_event_gate`, plus the 6 causal keys; there is no `decoder_out_channels`. Diamond MRO: `TrfCrwsTask → CrwsTask → TrfRwsTask → RwsTask → LightningModelBase`. Trainer resolution: `TARGET_FIELDS`, `TRACKED_METRICS` and `preflight` come from the causal parent; `compile_model_requested` and `_build_trainer_kwargs` come from the transformer parent. The block is $30\times16=480$ samples; its nats are comparable only to `lag_attn_crws`.

### 2.5 Objective

CRWS uses the objective of §1.8 through `compute_raw_objective` on the raw block. The differences: `lambda_ms: 0.1` and `lambda_deriv: 0.1` (the shape terms are on, because a raw trajectory has an order: multiscale $L_1$ rates (1,4,16) over the flattened 480-sample block, derivative Huber $\delta=1$); `lambda_boundary: 0` (refused with tiles); horizon weight $\lambda=15$ (the only weight); no channel weight; no persistence; no event gate. `total_loss` mixes units; read `nll_*` for nats. The CRWS packages have no eval package.

### 2.6 Training and configs

- Launch: `python -m teb_vae.lag_attn_crws.trainer --config teb_vae/lag_attn_crws/configs/default.yaml`, or edit `RUN_CONFIG` in `crws/trainer.py` and press Run; transformer: `lag_attn_transformer_crws/trainer.py`. `main()`, DDP and callbacks are as in §1.9. The preflight adds the `weight` field check, and `create_model` logs the geometry with the raw block width. Tracked metrics = the RWS set + `anchors_per_sample`, `source_lag_warmth_frac_st/_ph` + validation `kld_source_null`.
- Config deltas, CFS → CRWS `default.yaml`: `c_y 80→102`, `c_u 46→51`, `horizon 10→30`, `horizon_weight_halflife_steps 5→15`, `lambda_ms 0→0.1`, `lambda_deriv 0→0.1`, `causal_align_reference null→42.21`, `anchor_stride 5→30`, `gradient_clip_val 4000→12000`, `spike_breaker.additive_margin 2200→2500`; `causal_align_reference_source`, `causal_target_forecast_clock`, `causal_phase_operator`, `causal_source_channels`, `target_weight_st/ph` and `persistence_residual` are **absent**; identity keys (`tag lag_attn_crws_baseline`, `run_name`/`variant lag_attn_crws_ref42`, `experiment_name seqvae-teb-lag-attn-crws`). Budget 134, floor 134, `lag_kv_source conv_stem`, `prior_availability_input true`, `alibi_slope_scale 0.0`, early stopping and the secondary checkpoint are the same.
- CRWS → trf-CRWS: remove `lstm_layers, causal_norm, encoder_extra_dilations, encoder_extra_kernel, conv_norm_groups`; add `lr_warmup_steps: 2000` + the 7 encoder keys ([5,9], [1,2], 4 heads, `d_ff` 512, 6/3 blocks, window 16); `causal_align_reference 42.21→null` (unaligned: 98/102 target, 51/51 source); `lr 3e-4→3e-6`; `lambda_base 1.0→0.1`; `beta_schedule.end 1.0→15.0`; `batch_size 128→328`; `gradient_clip_val 12000→11100`; `cache_size 0→10000`; dataset paths and identity keys.
- Variant configs: `tiny.yaml` (1 epoch, `d_model` 32, `d_z` 8, `max_lag` 8, `mse`, causal fixtures), `smoke_causal.yaml` (fixture smoke run), `sweep_anchor_stride_1.yaml`, and the CRWS-only `sweep_horizon_15.yaml`.

### 2.7 Tests and invariants (deltas)

`lag_attn_crws/tests/conftest.py`: `SHIPPED_ALIGN_REFERENCE = 42.21`, `SHIPPED_HORIZON = 30`; it imports the shipped literals from the CFS conftest; fixtures `build`, `make_raw_signal`, `budget`. Own tests: `test_causal_raw_inputs.py` (the anchored gather equals the dense builder at dense anchors), `test_raw_target.py`, `test_anchors.py`, `test_causality.py`, `test_config_load.py` (parity against `lag_attn_rws` outside declared exemptions), `test_sweep_configs.py`, `test_task.py` (five forward inputs, stage-resolved anchor count). Transformer twin: `test_ddp_reachability.py`, `test_trainer.py` (both `_build_model_kwargs` halves fire).
