# Context & Codebase Map (`lag_attn_transformer_cfs` & Companion Modules)

This reference summarizes the verified architecture, dataset layout, clocks, HDF5 schema, configs, checkpoints, empirical audit findings, and codebase invariants gathered during the full review. Start here in future sessions instead of re-reading the repository from scratch.

> **Canonical Clock Directive (`up_shift_secs = -20`)**: During HDF5 shard creation, UA (`up`) is intentionally shifted `20 s` left (`up_shift_secs = -20`). Per explicit user directive, **leave this untouched and treat it as if it did not happen** (`KNOTS = 0..30`, `lag_floor = 0`, and all HDF5 shards remain unchanged; `recorded delay = stored delay - 20 s` when translating back to clinical/recorded clock).

---

## 1. Quick-Start Reading Order ("What to Read First")

| Topic / Need | Exact File(s) & Line Range(s) | What You Get There |
| :--- | :--- | :--- |
| **Current training run and next architecture** | `teb_vae/lag_attn_transformer_cfs/IMPLEMENTATION_PLAN.md` | UA-path-only S8 continuation (`run_s8_source_only.yaml`), evaluation protocol, and the frozen FHR base with a learned UA residual |
| **Model Class & Keyword Schema** | `teb_vae/lag_attn_transformer_cfs/nets/model.py:1-162` | MRO, `SeqVaeLagAttnTrfCfs.__init__`, `IRF_KEYWORDS`, why mixin order is load-bearing |
| **Causal Inputs & IRF Wiring** | `teb_vae/lag_attn_cfs/nets/causal_inputs.py:771-960`, `1825-1875` | `_build_irf_source_path`, `_irf_outputs`, `irf_source`, `irf_evidence_for`, `attend_lagged_source` |
| **IRF Head Math** | `teb_vae/lag_attn_cfs/nets/irf_source.py:1-157` | `ua_step_level`, `context_irf`, `IrfSourceHead.forward`, causal tone/high-pass/prequential gate |
| **Causal Target Block & AR(1)/Student-t** | `teb_vae/lag_attn_cfs/nets/causal_feature_target.py:1-80` | `CausalFeatureForecastTarget`, forecast clock shifts, novelty/warm-up tertiles, persistence gather |
| **Task Composition & Driver** | `teb_vae/lag_attn_transformer_cfs/task.py:1-52`, `teb_vae/lag_attn_transformer_cfs/trainer.py:1-183` | Diamond MRO (`SeqVaeLagAttnTrfCfsTask`), `LagAttnTrfCfsTrainer`, `RUN_CONFIG` (`:149`) |
| **Objective, Wrong-UA & IRF Loss Terms** | `teb_vae/lag_attn_rws/task.py:1095-1185`, `teb_vae/lag_attn_cfs/task.py:920-1206` | `main_loss`, `lambda_source_null`, `_irf_source_terms`, `_full_wrong_term`, `_other_recording_rows`, `do_perm` |
| **Metric Surface (`TRACKED_METRICS`)** | `teb_vae/lag_attn_cfs/trainer.py:116-181` | `_CAUSAL_SUFFIXES`, `_CAUSAL_VAL_ONLY_SUFFIXES`, `TRACKED_METRICS` |
| **Permutation / Occlusion / Null Controls** | `teb_vae/lag_attn_rws/nets/controls.py:213-345,528-682` | `perm_forward_outputs`, `occluded_forward_outputs`, `source_null_kld_with_grad` |
| **Standalone Stitched IRF Model** | `teb_vae/coupling_irf/DESIGN.md`, `teb_vae/coupling_irf/model.py:1-257` | Stitched 4-h 4-s TCN base + 60-min ridge IRF + `plant()` benchmark |
| **Dataloader & DDP Shuffling** | `train/data_module.py:1-121` | `GraphDataModule` (`train` `shuffle=True`, `val`/`test` `shuffle=False`) |
| **Configs (Baseline & S8–S10)** | `teb_vae/lag_attn_transformer_cfs/configs/default.yaml`, `sweep_next_s{8,9,10}*.yaml`, `sweep_alt_irf_source_only.yaml` | Full baseline leaves & warm-start rung diffs |
| **Warm-Start Surgery** | `teb_vae/neural_surgery_trf_cfs/surgery/make_s9init.py:1-192` | `MODES` (`readout`, `s10`, `source`), weight splicing & exactness check |
| **IRF Gate Calibration & Lag Analysis** | `teb_vae/lag_attn_transformer_cfs/eval/calibrate_irf_gate.py:1-154`, `eval/lag_analysis/{steps.py,report.py}` | `irf_gate_k` calibration, step accumulators, `wmedian` bootstrap reporting |
| **Audit & Empirical Reports** | `tmp/audit_s7/REPORT.md`, `tmp/lag_analysis_369/REPORT.md`, `tmp/coupling_irf/runs/v2/eval_test/ACCEPTANCE.md` | S7 forecast audit, checkpoint 369 lag study, `CouplingIRF` v2 test results |

---

## 2. Repo State & Core Commands

### Recent Commits (newest first)
- `d8fc4bb3` — *New IRF lag analysis pathway*
- `f272f5f8` — *New config with IRF*
- `e6dae170` — *Better init for s8 with run args*
- `ab08a4d3` — *s8 config for lag attn transformer cfs*
- `3e5169aa` — *revise import modules based on front path*
- `c3e2e65b` — *minimal dataset creating quality profile update*
- `294c0a9f` — *Lag detection analysis pipeline*
- `883cc79c` — *New lag analysis eval pipeline*

### Environment & Commands
- **Python binary**: `.venv/bin/python`
- **Non-slow unit test suite**:
  ```bash
  .venv/bin/python -m pytest teb_vae/lag_attn_transformer_cfs/tests -q -m "not slow"
  ```
- **Training (7-GPU DDP, S8 continuation: onset weighting off, `freeze_target_path: true`)**:
  ```bash
  TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
      -m teb_vae.lag_attn_transformer_cfs.trainer \
      --config teb_vae/lag_attn_transformer_cfs/configs/run_s8_source_only.yaml
  ```
- **Single-GPU smoke**:
  ```bash
  .venv/bin/python -m teb_vae.lag_attn_transformer_cfs.trainer \
      --config teb_vae/lag_attn_transformer_cfs/configs/tiny.yaml
  ```
- **Historical IRF warm-start checkpoint surgery**:
  ```bash
  .venv/bin/python -m teb_vae.neural_surgery_trf_cfs.surgery.make_s9init \
      --mode readout --source <run>/lag-attn-trf-cfs-epoch=369.ckpt
  ```
- **IRF gate calibration (`irf_gate_k`)**:
  ```bash
  .venv/bin/python -m teb_vae.lag_attn_transformer_cfs.eval.calibrate_irf_gate \
      --checkpoint <ckpt> --config <resolved_config.yaml>
  ```
- **Standalone `coupling_irf` cache / train / acceptance eval**:
  ```bash
  .venv/bin/python -m teb_vae.coupling_irf.data
  .venv/bin/python -m teb_vae.coupling_irf.train --out <dir>
  .venv/bin/python -m teb_vae.coupling_irf.evaluate --ckpt <dir>/model.pt --cache <steps.npz> --out <dir>
  ```

---

## 3. Dataset, Clocks, HDF5 Schema & Dataloader Gotchas

### Time Grid & Clocks
- **Raw signal**: 4 Hz (`5280` samples = 22 min), decimated $16\times$ to `330` steps of `4 s` each (`STEP_S = 4.0`).
- **Trimming**: `trim_minutes: 1.0` removes 15 steps (`60 s`) from both ends $\implies$ **`T = 300` steps (20 min)** / **`4800` raw samples**.
- **Two clocks**: `up_shift_secs = -20` is stamped at shard build time (`hdf5_dataset/new_pipeline/create_new_pipeline.py:2203,3418`, `hdf5_dataset/create_hdf5_dataset.py:306,941`, `hdf5_dataset/create_minimal_hdf5.py:178`). Everything in the model operates on the **stored clock** (`recorded delay = stored delay - 20 s`).

### Local Dataset (`tmp/dataset/split_last4h_gap10/`) & Production Stats
- Built from last 4 hours before delivery with gap $\le 10\%$, `trim_minutes: 1`, `quality_version: 5`, `transform: 'causal'`, `causal_phase_operator: 'integer_harmonic_v1'`, `causal_leg_alignment: 'envelope'`, `causal_kernel_taps: 32768`, `gammatone_order: 4`.
- **Splits (`split_guids.csv`)**: **5,915 total GUIDs** — `train: 4,729 GUIDs` (`22,170` kept segments per `tmp/lag_analysis_369/REPORT.md:4`), `test: 1,186 GUIDs` (`5,538` kept segments). `segments.csv` contains `38,420` segment rows total.
- **8 subgroup HDF5 shards per split** (`dataset_files.yaml`):
  `acidosis_cs`, `acidosis_no_cs`, `healthy_bg_cs`, `healthy_bg_no_cs`, `healthy_no_bg_cs`, `healthy_no_bg_no_cs`, `hie_cs`, `hie_no_cs`.
- **Pretraining vs. clinical test distribution**: Production model pretraining (`resolved_config_369/resolved_config.yaml:175-202`) trains **only on healthy shards** (`healthy_*`), so `acidosis_*` and `hie_*` in the local split are strictly **unseen outcome classes**.
- **Stats files**:
  - Local split stats: `tmp/dataset/split_last4h_gap10/stats.hdf5`
  - Production server stats used with `epoch=369`: `tmp/lag_analysis_369/server/stats.hdf5`

### HDF5 Shard Schema (Verified on `tmp/dataset/split_last4h_gap10/test/hie_cs.hdf5`, 27 keys)

| Key(s) | Shape (per $N$ segments) | Dtype | Notes / Attributes |
| :--- | :--- | :--- | :--- |
| `fhr`, `up` | `(N, 5280)` | `float32` | Processed 4 Hz raw signals |
| `fhr_raw`, `up_raw` | `(N, 5280)` | `float16` | Unrepaired raw signals |
| `fhr_st`, `up_st` | `(N, 36, 330)` | `float16` | One-sided gammatone scattering (`causal_delay_s`, `causal_warmup_steps`, `causal_novelty_curve`) |
| `fhr_ph` | `(N, 44, 330)` | `float16` | One-sided FHR phase harmonics (`causal_delay_s` `20.5 s .. 402.2 s`, `sel_*` attrs) |
| `up_ph` | `(N, 10, 330)` | `float16` | One-sided UA phase harmonics |
| `fhr_st_rho`, `fhr_ph_rho`, `up_st_rho`, `up_ph_rho` | `(N, C, 330)` | `uint8` | Quality contamination ratios quantized by `/ 255` (`quality_rho_scale: 255`) |
| `fhr_quality`, `up_quality` | `(N, 5280)` | `uint8` / `uint16` | Sample-level quality bitmasks (`bits`, `invalid_bits*` attrs) |
| `weight`, `target` | `(N, 330)` | `float32` | 4-s step validity weight and target array |
| `guid`, `epoch` | `(N,)` | `object` / `float32` | Recording ID and segment offset |
| `time_from_labor_onset`, `second_stage_onset` | `(N,)` | `float32` | Labor timing metadata |
| `cs_label`, `bg_label`, `fhr_repair_exact` | `(N,)` | `uint8` | Delivery mode / blood-gas / repair flags |
| `fhr_valid_frac`, `up_valid_frac`, `up_censored_frac`, `fhr_raw_match` | `(N,)` | `float32` | Segment summary quality scalars |

- **Normalization in `stats.hdf5`** (computed excluding warm-up, `trim_minutes = 1.0`):
  - `fhr_st`, `up_st`: channel 0 linear (regular); channels `1..35` transformed by `log(x + 1e-6)` (`log_epsilon = 1e-6`).
  - `fhr_ph`, `up_ph`: all channels transformed by `asinh(x)`.
  - **`fhr_up_ph` is strictly forbidden** (`FORBIDDEN_FIELD` in `teb_vae/lag_attn_cfs/trainer.py:84`).

### Dataloader & Timeline Stitching Gotchas
- **`GraphDataModule` (`train/data_module.py:99-121`)**: `train_dataloader()` uses `shuffle=True`, whereas `val_dataloader()` and `test_dataloader()` use **`shuffle=False`**. Because HDF5 shards store segments contiguously by `guid`, adjacent rows share the same GUID **71.5–81.1%** of the time on local shards (and ~70%+ even under DDP). Anything using `.roll(1, 0)` with same-GUID dropping silently discards most validation/test rows.
- **Multi-hour stitching**: Available via `teb_vae/coupling_irf/data.py` (`stitch`, packing up to 4 h per GUID on the 4-s grid) and `hdf5_dataset/guid_hdf5_dataset.py`.

---

## 4. Model Architecture & Where Each Method Lives (`SeqVaeLagAttnTrfCfs`)

### Class Inheritance & MRO (Order Is Load-Bearing)
- **Model** (`teb_vae/lag_attn_transformer_cfs/nets/model.py:64-66`):
  `SeqVaeLagAttnTrfCfs -> CausalWarmupInputs -> CausalFeatureForecastTarget -> FeatureForecastTarget -> SeqVaeLagAttnTrfRws -> nn.Module`
  *(Mixins MUST precede `SeqVaeLagAttnTrfRws` so `_default_decoder_out_channels` builds for `C_keep` feature channels instead of `raw_per_step = 16`, and tiled forward overrides dense forward.)*
- **Task** (`teb_vae/lag_attn_transformer_cfs/task.py:41`):
  `SeqVaeLagAttnTrfCfsTask -> SeqVaeLagAttnCfsTask -> SeqVaeLagAttnFsTask -> SeqVaeLagAttnTrfRwsTask -> SeqVaeLagAttnRwsTask -> LightningModelBase`
- **Trainer** (`teb_vae/lag_attn_transformer_cfs/trainer.py:84`):
  `LagAttnTrfCfsTrainer -> LagAttnCfsTrainer -> LagAttnTrfRwsTrainer -> LagAttnRwsTrainer -> GraphModelBase`

### Module Responsibility Map

| Component / Responsibility | Owning Module |
| :--- | :--- |
| Causal conv-Transformer encoders, adapters, prior/posterior heads, lag attention, FiLM horizon decoder | `teb_vae/lag_attn_transformer_rws/nets/model.py` |
| Channel warm-up budget mask, lag floor, tiled anchor forward, `IrfSourceHead` integration, `irf_evidence_gate` | `teb_vae/lag_attn_cfs/nets/causal_inputs.py` |
| Gathered shifted/stored target block, channel weights, scored horizon masks, AR(1)/Student-t likelihood, resolved gaps | `teb_vae/lag_attn_cfs/nets/causal_feature_target.py` |
| Ridge UA→FHR IRF head (`IrfSourceHead`), prequential gate, causal step-level UA aggregation (`ua_step_level`) | `teb_vae/lag_attn_cfs/nets/irf_source.py` |
| Loss & metrics assembly, `main_loss`, `lambda_full_wrong`, `do_perm` / `val/shuffle_penalty` | `teb_vae/lag_attn_rws/task.py`, `teb_vae/lag_attn_cfs/task.py` |
| Re-posing / occluding / source-null controls (`perm_forward_outputs`, `occluded_forward_outputs`, `source_null_kld_with_grad`) | `teb_vae/lag_attn_rws/nets/controls.py` |
| Warm-up budget resolution, `TRACKED_METRICS`, trainer CLI & DDP setup | `teb_vae/lag_attn_cfs/trainer.py`, `teb_vae/lag_attn_transformer_cfs/trainer.py` |

### Shipped Geometry & Production Hyperparameters
- **Raw/decoded geometry (`default.yaml` & `resolved_config_369`)**: `sequence_length = 300`, `horizon = 30` (`120 s`), `warmup_period = 134` (`T_valid = 270`, **136 dense anchors** on `val`/`test`; `anchor_stride = 15` gives ~9–10 tiled anchors on `train`), `lag_floor = 0`, `max_lag = 37` (**38 lags = 148 s**).
- **Channels**: Stored `c_y = 80` (`36` `fhr_st` + `44` `fhr_ph`) and `c_u = 46` (`36` `up_st` + `10` `up_ph`) $\to$ surviving warm-up budget (`134` steps): **`C_keep = 76` target** (`32` scattering + `44` phase) and **42 source** channels; fast phase channels (`cutoff > 0.03 Hz`) scored only up to `target_phase_fast_horizon = 5` steps; novelty minimum `0.05` on `fhr_st`.
- **Backbone & capacity (~4.18 M params)**: `d_model = 128`, `d_z = 64` (4 heads $\times$ 16 dims), `target_attention_blocks = 6`, `source_attention_blocks = 3`, `decoder_hidden = 256` (in `resolved_config_369`), `num_heads = 4`, `d_head = 32`, `use_entmax = true` ($\alpha = 1.5$), `lag_kv_source = adapter`, `source_quality_mask = true`, `source_mask_mode = soft`, `posterior_event_gate = true`.
- **Likelihood in `epoch=369` (`s8`)**: `gaussian_nll` interface with **Student-t (`forecast_student_nu = 8.0`)** + **AR(1) residual (`forecast_ar_residual = true`, `forecast_ar_max = 0.95`)** + **38-lag `persistence_residual`** skip (widened from 6 lags in `s7`).
- **IRF source wiring**: Controlled by `source_path` (`'lag_attn'` or `'irf'`), `irf_readout` (hybrid readout beside lag attention), `irf_gate_evidence` (linear `irf_evidence_gate` added to soft event-gate logits), `irf_channels` (default `0..20` in `s9`/`s10`), `irf_gate_k` (`4.45` in `s9`/`s10`), `irf_context_steps` (`0` = full past within the segment).

---

## 5. Standalone Companion Model: `teb_vae/coupling_irf/`

Built as a clean baseline/replacement for UA→FHR coupling & delay (`teb_vae/coupling_irf/DESIGN.md`, `teb_vae/coupling_irf/model.py`):
- **Inputs & Base**: Stitched 4-hour timelines on the 4-s grid (`fhr` in `(bpm - 140)/20`, `fhr_sd`, `fhr_ok`, `ua`, `ua_ok`). FHR-only 6-block dilated causal TCN base (`width = 64`, `dilations = (1..32)`, Student-t $\nu = 8$) detached when scoring UA.
- **IRF Head**: Half-wave rectified excess UA above trailing 10th-percentile resting tone (`75` steps), convolved with `J = 29` hat functions (`KNOTS = 0..30`, delays `4..116 s`), fitted via 60-min (`context_steps = 900`) rolling ridge regression with `PREQ_GAP = 15` (`60 s`) prequential calibration gate ($a_t = \mathrm{clip}(\text{slope} - k \cdot \text{se}, 0, 1)$, calibrated $k = 4.4$).
- **Run v2 test results** (`tmp/coupling_irf/runs/v2/eval_test/ACCEPTANCE.md`, 1,186 test GUIDs, stored clock):
  - **A1/A2 specificity**: `gain_expl = +0.0164` own vs `-0.000478` wrong UA; `gain_fore = +0.00433` own vs `-0.000267` wrong UA; gate open `40.7%` own vs `4.66%` wrong.
  - **A3/A4 planted delay**: Median error **`0.974–2.67 s`** across 8–88 s planted delays at 20 bpm; UA shift slope `-1.00`. Drops sharply to `15.4%` detected / `5.33 s` error / `gain_expl = -0.000518` when only **30%** of contractions trigger a deceleration.
  - **A5–A7**: Real-contraction delay Spearman `0.482` over GUIDs (`0.381` per contraction, `median abs error 15.6 s`, within-GUID decel AUC `0.577`); split-half gain reliability `0.417` (below `0.45` threshold); subgroup signal: `hie_cs`/`hie_no_cs` gate open `57.4%`/`56.2%` vs healthy `27.3–41.2%`, median `hie_cs` delay `60.5 s` vs `healthy_no_bg_no_cs` `40.0 s`.

---

## 6. Checkpoints, Config Ladder (`s1`–`s10`), & `tmp/` Artifacts

- **`tmp/resolved_config_369/`**:
  - `lag-attn-trf-cfs-epoch=369.ckpt` — trained rung-8 (`s8`) checkpoint (warm-started from `epoch=679-s8init.ckpt`, `lr=0.003`, `batch_size=228`, `persistence_lags=38`, `forecast_student_nu=8.0`, `base_decode=sample`).
  - `lag-attn-trf-cfs-epoch=369-{s9init,s9readout-init,s10init}.ckpt` — surgery initializations generated by `teb_vae/neural_surgery_trf_cfs/surgery/make_s9init.py`.
  - `resolved_config.yaml` — exact server training configuration of `epoch=369`.
- **`tmp/resolved_config_s7/`**: Rung-7 baseline (`lag-attn-trf-cfs-epoch=443.ckpt`, `-s8init.ckpt`, `resolved_config.yaml`).
- **Key `tmp/` evaluation reports**:
  - `tmp/audit_s7/REPORT.md` — S7 epoch 443 audit (forecast MSE/NLL vs AR/ridge/TCN, `pred_gap_mean`, outcome probe AUROC `0.79` vs `0.80` raw feature mean/std).
  - `tmp/lag_analysis_369/REPORT.md` — Preregistered 49-claim / 89-row lag audit on `epoch=369`.
  - `tmp/coupling_irf/runs/v2/eval_test/ACCEPTANCE.md` — Full A1–A7 evaluation table on test split.
- **Recent config ladder (`teb_vae/lag_attn_transformer_cfs/configs/`)**:
  - `sweep_next_s8_long_ar_skip.yaml` — `s7` + `persistence_lags: 38`, `constant` beta `1.0`.
  - `sweep_next_s9_irf_readout.yaml` — `s8` + `irf_readout: true` (`irf_channels: 0..20`, `irf_gate_k: 4.45`, `lambda_source_{irf,expl,wrong}: 1.0`, `irf_lr_scale: 30.0`).
  - `sweep_next_s10_specific_source.yaml` — `s9` + `lambda_full_wrong: 1.0`, `irf_gate_evidence: true`, `event_weight_floor: null`, `event_weight_ua_valid: false`.
  - `sweep_alt_irf_source_only.yaml` — diagnostic `source_path: irf` (lag attention removed; loses ~98% of forecast gain).

---

## 7. Verified Audit Findings, Repo Invariants & Pitfalls

*(See `teb_vae/lag_attn_transformer_cfs/IMPLEMENTATION_PLAN.md` for exact line-by-line fixes and next-generation architectural proposals.)*

### Audit findings
- **`epoch=369` (`s8`)**: Lag attention does **not** measure physical delay (`kappa = 0.06–0.33`, 95–100% of gain retained by a UA-blind fixed lag profile, `2/64` latent dims and head 3 carry `99.4%` of gain, `logvar_prior` collapses onto floor under `base_decode: sample`).
- **6 concrete implementation bugs found in `s9`/`s10` & IRF**:
  1. `IrfSourceHead` multiplies regression weights `beta` fitted on mean-subtracted `Zc` against uncentered `Z` / `Xf` (causing `-0.22 -> +0.29 SD` drift, `pred_gap_irf = -0.21`) and counts zeroed `PREQ_GAP` startup steps in `resid`.
  2. The legacy IRF losses roll the input by one row and drop most rows when GUIDs are contiguous. The validation shuffle and the full wrong-UA term (`_full_wrong_term`, now centred on the base NLL) use a cross-GUID derangement when all ranks can supply one.
  3. `lambda_full_wrong` backpropagates >99% of its squared gradient norm into FHR encoders and shared decoder weights (worsening `nll_base_block` by `+8.3` nats in 30 steps).
  4. `_irf_source_terms` and `_full_wrong_term` fold 4–7 copies of uncentered `~ -1465` nat NLLs directly into `main_loss` & `val/total_loss`, distorting `spike_breaker` (`additive_margin: 6600`) and checkpoint selection.
  5. `irf_gate_evidence` learns a negative weight gradient under `kl_own`, drifts with step index (`irf_context_steps = 0`), and `occluded_forward_outputs` does not recompute/zero `irf_evidence`.
  6. Broken tooling (`calibrate_irf_gate.py:98-99` rejects `irf_readout`, `report.py` `wmedian` bias, stale `RUN_CONFIG`, stale `lr` in `make_s9init.py`) and **14 failing non-slow pytest cases** in `tests/{test_task,test_sweep_configs,test_config_load,test_eval_binding,test_eval_run}.py`.

### Non-Negotiable Repo Invariants (Before Editing Code/Configs)
1. **Dual-edge config pinning (`tests/test_config_load.py`)**: Every key in `teb_vae/lag_attn_transformer_cfs/configs/default.yaml` is strictly compared leaf-for-leaf against `teb_vae/lag_attn_cfs/configs/default.yaml` and `teb_vae/lag_attn_transformer_fs/configs/default.yaml`. New `VAE_model` keys must appear at inert defaults in `default.yaml` and be registered in `test_config_load.py`.
2. **Sweep config rules (`tests/test_sweep_configs.py`)**: Every sweep preserves the default keys, moves at least one leaf besides identity, and keeps `run_name` and `variant` in sync. Warm starts may declare a constant $\beta=1$ and its `value` key. Every sweep must leave at least one valid anchor at the last tiling phase.
3. **Diamond task disjointness & metric tracking (`tests/test_task.py`)**: `SeqVaeLagAttnCfsTask` and `SeqVaeLagAttnTrfRwsTask` must remain disjoint; every key in `LagAttnTrfCfsTrainer.TRACKED_METRICS` must match its stage emission.
4. **DDP collective lockstep**: Validation-stage decisions (e.g., `_sync_perm_decision` in `lag_attn_rws/task.py:1150` and `_agreed` in `lag_attn_cfs/task.py:1203`) must match across ranks; skipping `_sync_perm_decision` on `train` avoids an `all_reduce` + GPU sync per training step.
5. **Refusals preserved**: `fhr_up_ph` is forbidden (`FORBIDDEN_FIELD`); `lambda_boundary != 0.0` is refused under anchor tiling; causal `target_forecast_shift` with IRF head is refused.
