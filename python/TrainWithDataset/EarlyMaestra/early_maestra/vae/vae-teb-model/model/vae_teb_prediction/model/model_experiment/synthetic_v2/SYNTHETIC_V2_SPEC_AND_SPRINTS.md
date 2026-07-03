# synthetic_v2 Raw -> Scattering -> Block-TE Pipeline - Spec and Roadmap

Status: Sprints 0-4 complete; Sprint 5 code + pilot smoke complete (headline run S5-T04 deferred to user);
Sprint 6 complete (eval machinery + tests + pilot-checkpoint smoke; headline gate numbers await S5-T04);
Sprint 7 complete (journal figures + plot_style_v2, TE-aware standard-testing sample plots, pulse_train variant,
final_report_v2, README; figure/report numbers await the S5-T04 headline run)
Last updated: 2026-07-01
Owner: Mahdi-Si

> Companion design/math reference: `SYNTHETIC_V2_RAW_TE_PIPELINE_EXPLAINED.md` (same folder).
> That document is the authoritative derivation of every equation and design choice; this
> document is the build contract: what to implement, in what order, and how each piece is
> proven correct. Read the EXPLAINED doc first for the "why"; read this for the "how" and the
> task-by-task plan. This roadmap incorporates a three-reviewer critique (technical correctness,
> completeness, atomicity) whose accepted findings are folded into the tasks and risks below.

---

## 1. Context

We are validating a variational transfer-entropy bottleneck model with lag attention
(`model/vae_teb_prediction/model/vae_teb_lag_attn_v1.py`; theory in
`knowledge/unsupervised_model/VAE-TEB-Lag-Attention.md`). The model reports two quantities on
fetal-heart-rate (FHR) / uterine-pressure (UP) recordings: a per-step latent KL divergence
$\bar K_t$ (its surrogate for the block transfer entropy $\mathrm{TE}^{(H)}_{U\to Y}$) and
lag-attention weights $\alpha_{t,\ell}$ (which past UP lag drives future FHR). To trust these on
real data we validate them on synthetic data whose TE and lag we know exactly.

The existing "mixed" pipeline (`model/vae_teb_prediction/model/model_experiment/synthetic/`,
documented in `synthetic/MIXED_DATASET_AND_BLOCK_TE_EXPLAINED.md`) writes the model's
*scattering-feature channels directly* from a linear-Gaussian state space and computes TE in
that feature domain. It never runs the production scattering transform, so it cannot answer the
one question that matters most for deployment: **does a controlled, known block transfer entropy
survive the real nonlinear scattering + phase-harmonic encoder and its normalisation?**

`synthetic_v2` closes that gap. It generates *one* raw 4 Hz FHR waveform and *one* raw 4 Hz UP
waveform as a physiologically-placed additive band composition with a single coupled pathway
(UP contraction strength -> FHR deceleration depth), passes each through the *real*
`KymatioPhaseScattering1D` transform and the *real* log/asinh + z-score normalisation, then feeds
the model. It grades the model against three transfer entropies: the injected latent TE
($\mathrm{TE}_{\mathrm{inj}}$, exact label), the raw-domain TE ($\mathrm{TE}_{\mathrm{raw}}$), and
the scattering-realizable TE ($\mathrm{TE}_{\mathrm{scat}}$) measured on the model-facing features.
The single most important early result is the preservation fraction
$\mathrm{frac}_\Phi = \mathrm{TE}_{\mathrm{scat}} / \mathrm{TE}_{\mathrm{inj}}$: it separates "the
transform lost the coupling" from "the model failed to recover it", a diagnosis the v1 pipeline
structurally cannot produce.

## 2. Goals and non-goals

### Goals

- Build a standalone `synthetic_v2/` module that generates raw 4 Hz FHR/UP signals with a single
  analytically-controlled UP->FHR block-TE pathway, at a chosen target TE (nats) and lag $D$.
- Pass each raw signal through the production `KymatioPhaseScattering1D` transform and the exact
  production normalisation, producing the model's four feature fields with the verified channel
  counts `fhr_st(43)`, `fhr_ph(44)`, `up_st(43)`, `up_ph(58)`.
- Measure, model-free, all three transfer entropies - injected ($\mathrm{TE}_{\mathrm{inj}}$),
  raw-domain ($\mathrm{TE}_{\mathrm{raw}}$), and scattering-realizable ($\mathrm{TE}_{\mathrm{scat}}$)
  - and the preservation fraction $\mathrm{frac}_\Phi$, before any GPU training is spent.
- Train the (unchanged) VAE-TEB model on the pooled synthetic cache and grade it: gamma-calibration
  of $\bar K$ against both $\mathrm{TE}_{\mathrm{inj}}$ and $\mathrm{TE}_{\mathrm{scat}}$, lag
  recovery against the true lag band, and null-control collapse.
- Deliver journal-level, tailored preview and report figures: raw signals with annotated bands,
  scattering-coefficient heatmaps with the coupled pulse-shape channel highlighted, the latent pair
  and its AM envelope/carrier decomposition, calibration/lag/frac_Phi diagnostics, and TE-aware
  sample diagnostics. The figure work must also bridge into the standard testing pipeline
  (`model/vae_teb_prediction/testing/run_tests.py`) so one-sample plots include the actual
  synthetic TE values (`TE_inj`, `TE_scat`, optional `TE_raw`) and true lag context.
- Keep the module standalone: any code taken from `synthetic/` is copied and adapted in
  `synthetic_v2/`; all runtime imports resolve to `synthetic_v2/` or to import-clean shared
  dependencies (the model and `hdf5_dataset/kymatio_phase_scattering.py`).

### Non-goals

- No changes to the model, loss, trainer internals, or attention (`vae_teb_lag_attn_v1.py` is
  used only through its `forward` contract).
- No `M` (number-of-informative-channels) dilution axis. v2 has exactly one source and one
  target; `M` is removed everywhere.
- `band` lag mode is deferred for the initial build. Only `fixed` per-cell lag is generated,
  enumerated, and scored here. The ported inverter retains band-averaging math
  (`mean_te_block_state_space_over_delays`) so band mode can be added later, but no v2 generation,
  provenance, or evaluation path implements it now.
- No cross-field `fhr_up_ph` (79 ch) consumed for training (the v1 model does not read it); it is
  produced only optionally for the realizability / raw-TE probes.
- No intermittent / time-gated TE (the v1 EXPLAINED doc Section 13 proposal). v2 coupling is
  constant in time.
- No importing of `hdf5_dataset/new_pipeline/create_new_pipeline.py` at runtime (it pulls
  `early_maestra`, which is not importable in the synthetic environment; see Risks).
- No modification of the existing `synthetic/` package; it is a copy/adapt source only.

## 3. Users and usage

Single research user (the model author), running on two machines: the local Windows workstation
(RTX 4080, `.venv` with torch 2.7.1+cu128) for iteration, dataset builds, the realizability probe,
and figure generation; and the production 8x A6000 (48 GB) / 128-core Linux box for full-scale
DDP training when needed. Triggered from the CLI via a driver script
(`run_pipeline_v2.py`) with stage flags. Not a service; no external callers; volume is a handful of
dataset builds and training runs, not continuous traffic. Outputs are `.npz` caches, a `meta.json`
manifest, checkpoints, figures (PDF/PNG), and evaluation reports (JSON/markdown) written under
`synthetic_v2/data/` and `synthetic_v2/results/`.

## 4. Current state

Phase-0 exploration mapped the reusable terrain. Concrete anchors:

Copy/adapt sources in `synthetic/`:

- `analytic_te.py` - block-TE math to port: `_simulate_state_space_gaussian` (655),
  `te_block_state_space_gaussian` (849), `mean_te_block_state_space_over_delays`,
  `B_y_for_mean_te_block_state_space` (inverter), `snr_per_step_for_te_block` (931),
  `realizable_te_block_from_arrays` (998, the R0 probe; informative source expected in `U[:,:,:M]`).
- `generators.py` - `_standardize_per_channel` (65), `_true_lag_trajectory` (312),
  band/oscillator generation patterns, and generator `meta` conventions.
- `mixed_dataset.py` - `MixCell` dataclass (159), `enumerate_mix_cells` (373),
  `build_mixed_split` (652), `write_mixed_cache` (788), provenance array conventions.
- `dataset.py` - `SyntheticTEDataset` (186), `attribute_dict_collate` (93), `build_u_stream`
  (442, concatenates `up_st(43)+up_ph(58)=101`), `make_dataloader` (459). Note: `__getitem__`
  (431-437) reads v1 provenance `sample_M`, `sample_delay_min/max`, `sample_band_id` that v2 drops.
- `pl_module_synth.py` - `SyntheticSeqVaeLagAttnPl` (73), `compute_loss_and_metrics` (160),
  logs `kld_nats`.
- `datamodule_synth.py` - `SyntheticTEDataModule` (64).
- `mixed_eval.py` - `collect_per_sample_kbar` (92, groups by `M`/`delay_max`/`band_id`; clean-window
  floor uses `delay_max`), `fit_calibration_slices` (452), `per_cell_lag_recovery` (603).
- `train_ddp.py`, `train_minimal.py`, `gpu_pool.py` - training drivers.
- `plot_style.py`, `visualize.py`, `visualize_mixed.py` - figure helpers and layout.
- `config_synth.yaml` - the config blocks to inherit (`model`, `loss`, `optim`, `dataset`, `ddp`).

Production dependencies (used, not modified):

- `hdf5_dataset/kymatio_phase_scattering.py` - `KymatioPhaseScattering1D` (60); `forward` (394)
  returning `'scattering'`, `'phase_corr'`, `'cross_phase_corr'`, `'autoc_idx'`;
  `select_fhr_phase_coefficients(min_freq=...)` (501); `select_fhr_up_cross_coefficients_v2` (635);
  `center_freqs` buffer (128, normalised xi). This module imports cleanly.
- `hdf5_dataset/hdf5_dataset.py` - `normalize_tensor_data` (21): ch0 untouched, `log(clamp(x,0)+eps)`
  on scattering ch 1..42, `asinh` on all phase channels, per-channel z-score. Logic will be copied
  locally to avoid a heavy import; the local copy is validated for parity against this function.

Model consumer (used, not modified):

- `vae_teb_lag_attn_v1.py` - `SeqVaeLagAttnV1` (1172); `forward(y_st, y_ph, u_stream)` (1650)
  returning a dict with `kld_per_t (B,T)`, `attn_weights (B,T,num_heads,91)`,
  `te_lag_map (B,T,91)`, `warmup_mask (T,)`. Defaults: `sequence_length=300`, `horizon=30`,
  `warmup_period=30`, `max_lag=90`, `c_y=87`, `c_u=101`, `use_up_st=True`.
- Checkpoint save/load should follow the project convention in `train/graph_models_utils.py`
  (per project CLAUDE.md).

Standard testing pipeline integration points:

- `testing/run_tests.py` already supports synthetic/non-HDF5 evaluation through `loader_override`
  and `guid_loader_override`. It runs `run_sample_diagnostics` and `run_kld_lag_diagnostics` when
  `skip_forecast_heatmaps=False` and `analysis_samples>0`, so v2 can reuse the existing testing
  surface instead of creating a second one-off sample-plot runner.
- `testing/collectors.py` has `_extract_te_true`, but `collect_predictions()` currently does not
  attach `te_true` to the sample dict, and it has no extractors for v2's `te_scat`, `te_raw`,
  `frac_phi`, `sample_delay`, or per-sample `cell_id`. These fields must be propagated for
  TE-aware diagnostics.
- `testing/plot_single_samples.py` has `plot_sample_lag_attention(..., true_te=..., true_lag_tt=...)`
  support, but the main `plot_sample_lag_attn_diagnostic()` title currently includes only GUID,
  epoch, class, forecast MSE, uplift, and residual ratio. It must render actual TE metadata and the
  true lag band for synthetic v2 sample PDFs.

Verified by running the transform on this workstation (RTX 4080, CUDA):

- Scattering output = `(1, 43, 330)` -> `fhr_st = up_st = 43` (1 order-0 + 42 first-order;
  `center_freqs` has 42 entries).
- `select_fhr_phase_coefficients(min_freq=0.006).optimal_mask.sum() = 44` -> `fhr_ph = 44`, so
  `c_y = 43 + 44 = 87` (matches model).
- `select_fhr_phase_coefficients(min_freq=0.002).optimal_mask.sum() = 58` -> `up_ph = 58`, so
  `c_u = 43 + 58 = 101` (matches model).
- `select_fhr_up_cross_coefficients_v2(...).cross_mask.sum() = 79` -> `fhr_up_ph = 79` (not fed to
  the model).
- A single `forward(x, compute_phase=True, compute_cross_phase=False, scattering_channel=c,
  phase_channels=[c])` returns BOTH `scattering` (channel c) and `phase_corr` (channel c). So the
  four model-facing fields need only TWO passes (one per channel); the cross field needs a third
  pass with `compute_cross_phase=True` (which suppresses `phase_corr`).
- `center_freqs` are normalised xi (max ~0.37); the true-Hz center is `xi * fs`. The filter nearest
  `f_pulse=0.02 Hz` is at xi ~0.0049 (0.0196 Hz), NOT `argmin(|center_freqs - 0.02|)` (which wrongly
  picks 0.078 Hz). Channel selection must convert with `fs`.

## 5. Proposed approach

End-to-end data flow (one cell = one `(target_te, lag D)` point, fixed lag):

```
analytic control  (analytic_te.py, ported)
  inverter B_y_for_mean_te_block_state_space(target_te, D) -> coupling B, TE_inj (exact label)
        |
latent pair on the DECIMATED grid (T'=330, 4 s/step)
  source c_k ~ AR(2), r=0.80, w=0.10 (contraction rhythm ~0.004 Hz)
  target d_k = 0.40 d_{k-1} + B c_{k-D} + eps        (only B-term carries TE)
        |
render to RAW 4 Hz  (raw_generators.py, new; am_carrier default)
  envelope = band-limited upsample of c,d (330 -> 5280); positive offsets a0/b0
  carrier  = pulse-shape template near f_pulse (default 0.02 Hz; may be raised - see AM caveat)
  UP_raw  = mu_UP  + A_u(c)*carrier + drift + toco-noise
  FHR_raw = mu_FHR - A_y(d)*carrier + FHRV(LF,MF,HF) + accels + noise
  length 5280 samples (22 min); DC drawn per-sample from physiological ranges
        |
REAL transform  (scattering_adapter.py, new)
  KymatioPhaseScattering1D(J=11,Q=4,T=16,shape=5280,max_order=1); TWO self-phase passes
  masks: fhr_ph(min_freq=0.006)->44, up_ph(min_freq=0.002)->58; scattering->43
  -> fhr_st(43) fhr_ph(44) up_st(43) up_ph(58) @ 330 steps; trim 15/end -> 300 (= latent[15:315])
        |
REAL normalisation  (local copy of normalize_tensor_data logic, parity-tested)
  ch0 raw; log(clamp(x,0)+1e-6) on st ch 1..42; asinh on ph; per-channel z-score
  stats from synthetic_pool (default) or a real fold's stats.hdf5
        |
three TE probes  (eval_v2.py, new) -- BEFORE training
  TE_inj (label), TE_raw (raw/bandpassed determinant ratio), TE_scat (feature ridge/held-out probe
  on the fs-correct coupled pulse-shape channels, aligned to latent[15:315])
  -> frac_Phi = TE_scat / TE_inj
        |
cache  (build_dataset_v2.py, new): npz (4 fields + weight + provenance) + meta.json
        |
MODEL forward(y_st=fhr_st, y_ph=fhr_ph, u_stream=[up_st||up_ph])
  -> kld_per_t (K-bar), attn_weights
        |
GRADE  (eval_v2.py): gamma-calibration (K-bar vs TE_inj and TE_scat), lag recovery, nulls
        |
STANDARD TESTING ROUTE  (testing/run_tests.py)
  build a v2 DataLoader, pass it through `loader_override`, and produce the usual testing figures
  with synthetic TE provenance in every per-sample PDF and CSV row
```

Module layout (all under `synthetic_v2/`): `analytic_te.py` (copy/adapt), `raw_generators.py` (new),
`scattering_adapter.py` (new), `build_dataset_v2.py` (new), `dataset_v2.py` / `datamodule_v2.py` /
`pl_module_v2.py` (copy/adapt), `eval_v2.py` (new), `visualize_v2.py` / `plot_style_v2.py` (new/copy),
`run_pipeline_v2.py` (new), `config_synth_v2.yaml` (new), `tests/` (one module per unit),
`README.md` (usage).

Key implementation decisions:

- Latent pair generated on the decimated grid (r=0.80 stays lag-identifiable and w=0.10 lands in
  the contraction band). Delay $D$ is in decimated steps; $16D$ raw samples. Features after trim are
  `latent[15:315]`; all latent-vs-feature correlations use that slice.
- `am_carrier` render is the default; `pulse_train` is a config option added later. Coupling is
  routed through a strictly positive amplitude envelope so the sign-blind scattering modulus still
  sees it.
- AM-separation caveat (from technical review): with `r=0.80` the AR(2) envelope has a broad
  ~0.016 Hz bandwidth, comparable to the default carrier `f_pulse=0.02 Hz` and wider than the
  analyzing wavelet's ~0.005 Hz passband. The clean "~5x separation" is optimistic. An analytic
  pre-check (Sprint 1) compares the envelope bandwidth to the wavelet passband before any pilot, and
  if inadequate the carrier is raised (e.g. `f_pulse` 0.04-0.06 Hz) and/or `w` narrowed, with a
  lag-identifiability recheck. The Sprint 3 frac_Phi gate is where this is proven, not assumed.
- The scattering adapter depends only on `kymatio_phase_scattering.py`; it reproduces mask selection
  by calling `select_fhr_phase_coefficients` directly and asserts exact channel counts (43/44/43/58).
  Coupled-channel identification converts `center_freqs` to Hz with `fs`. Normalisation is a local,
  parity-tested copy of `normalize_tensor_data`.
- The three TE probes run as a non-fatal (configurable-fatal) pre-flight and again in the final
  report; `TE_scat` and `frac_Phi` are stamped per sample so evaluation can regress $\bar K$ against
  $\mathrm{TE}_{\mathrm{scat}}$.

### Alternatives considered

- Thin wrapper over `synthetic/` (import at runtime) - rejected: EXPLAINED requires a standalone
  module, and `create_new_pipeline.py` is not import-safe here anyway.
- Feed the signed latent directly as the raw band (no carrier) - rejected: the scattering modulus is
  sign-blind, so this routes coupling through `|c|` and loses an uncontrolled fraction of the TE.
- Reconfigure the model's `c_y`/`c_u` to whatever the transform emits - unnecessary: the verified
  counts already match the model's fixed 87/101 contract; hold the model fixed and assert the counts.
- Compute TE only in the injected latent domain (as v1) - rejected: that is exactly the blind spot
  v2 exists to remove; feature-space TE must be measured.

## 6. Data and integrations

- Reads: none from any database. Optionally reads one real training fold's `stats.hdf5` (path from
  config) if `norm_stats_source: real_fold`; default `synthetic_pool` reads nothing external.
- Writes: `.npz` split caches (`train/val/test`) + `meta.json` under
  `synthetic_v2/data/G1_raw/<tag>/`; checkpoints, figures, and reports under
  `synthetic_v2/results/<run_tag>/`. All writes are to the module's own output tree; no shared
  production data is touched.
- Database: not applicable - file-based research pipeline, no DB reads or writes.
- No new schema. Cache arrays (Section 17 of EXPLAINED): `fhr_st(N,300,43)`, `fhr_ph(N,300,44)`,
  `up_st(N,300,43)`, `up_ph(N,300,58)`, `weight(N,300)`, `true_lag_tt(N,300)`,
  `sample_te_true(N)`, `sample_te_scat(N)`, `sample_frac_phi(N)`, `sample_te_raw(N)` (optional),
  `sample_delay(N)`, `sample_cell_id(N)`, `sample_held_out(N)`. No `sample_M`, no band fields.
- `config_synth_v2.yaml` carries, in addition to EXPLAINED Section 16: `scattering.batch_size`
  (GPU-batch size for the transform), a `seed`/`base_seed` block (DGP, inverter MC, shuffle),
  `eval.realizability.fatal`, and the `mix.inverter` knobs, all read from config (not hardcoded).
- Libraries (all already in `.venv`): torch 2.7.1+cu128, numpy, PyYAML, matplotlib, pytorch-lightning,
  kymatio (via `KymatioPhaseScattering1D`), h5py (only if `real_fold` stats used). No new packages.
- No LLM calls anywhere in this pipeline.

## 7. Non-functional requirements

- Compute target: builds, the TE probes, and figures run on the local RTX 4080 with GPU-batched
  scattering (`scattering.batch_size`, default ~32-64 at shape 5280, kept under the 4080's budget);
  training runs single-GPU locally for pilots and is DDP-capable for the prod 8x A6000 box.
- Build cost: full grid ~= 20 cells x (2000 + 400 + 600) ~= 60k samples, each needing two scattering
  passes on a 5280-sample signal. Batched on GPU this is minutes to low tens of minutes; the build
  is resumable (per-(cell,split) checkpointing) so a crash does not restart from zero.
- Determinism: fixed seeds for the DGP, the inverter Monte-Carlo (seed held across bisection so it
  does not chase noise), and the shuffle permutation; a cache is reproducible from its manifest, and
  a determinism test rebuilds a tiny cache from stored seeds and asserts identical arrays.
- Memory: `scattering.batch_size` bounds peak GPU memory; the batched-scatter path asserts it stays
  within budget.
- No security/tenancy/PII concerns (fully synthetic data, local files).
- Numerical safety: `log` clamps negatives (inert on non-negative scattering magnitudes), `logdet`
  via `slogdet`, ridge regularisation in the probe, z-score epsilon 1e-8, log epsilon 1e-6.

## 8. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation / where de-risked |
|------|-----------|--------|------------------------------|
| AM separation inadequate: AR(2) envelope (~0.016 Hz, r=0.80) is broadband vs carrier 0.02 Hz and wider than the ~0.005 Hz wavelet passband, so the coupled channel smooths away the lag-carrying fluctuations and frac_Phi is low | High | High | Sprint 1 analytic AM-separation pre-check BEFORE any pilot; Sprint 3 recovery task sweeps `f_pulse` (0.04-0.06 Hz), `w`, `am_offset_ratio`, cell strength with a lag-identifiability recheck; frac_Phi gate proves it. |
| Coupled-channel selection unit bug (`center_freqs` xi vs `f_pulse` Hz) picks a 4x-wrong channel | Confirmed | High | Select via `argmin(|center_freqs*fs - f_pulse|)` with an assertion the chosen channel's true Hz is within one Q-step of `f_pulse` (S2-T03); config documents that `phase_min_freq` is xi and `f_pulse` is Hz. |
| Injected TE does not survive scattering (frac_Phi << 1) | Medium | High | frac_Phi probe is a Sprint 3 gate BEFORE training; am_carrier maximizes preservation; recovery task retunes. |
| v1 loaders/eval `KeyError` on dropped provenance (`sample_M`/`band_id`/`delay_max`) | Confirmed | High | S4-T03/S6-T01 map `sample_delay -> delay_max` for the clean-window floor and strip M/band grouping and the M/band diagnostic structures; a loader test asserts a v2 batch has no `M`/`band_id`/`delay_max` keys. |
| `create_new_pipeline.py` import pulls `early_maestra` (not installed) | Confirmed | High | Do not import it. Call `KymatioPhaseScattering1D` + `select_*` directly; copy `normalize_tensor_data` logic locally (parity-tested). |
| Local normaliser diverges from production `normalize_tensor_data` | Medium | Medium | S2-T02 parity test against the real function on a fixed input. |
| Channel counts drift from 43/44/43/58 | Low | High | Verified live; S2-T01 asserts exact counts and fails loudly. |
| Latent/feature time misalignment (off-by-15 after trim) | Medium | Medium | All latent-vs-feature correlations use `latent[15:315]`, stated in S2-T03/S3-T02 acceptance. |
| Build slow / crashes at 60k samples | Medium | Medium | GPU-batched, resumable per (cell,split); pilot grid (Sprint 3) before full build (Sprint 4). |
| gamma-calibration confounded by KL weight beta (posterior collapse) | Medium | Medium | Optional beta-selection stage; report calibration vs both TE_inj and TE_scat; null cell pins the intercept. |
| Lag blur from 4 s phi-averaging | Low | Low | Score lag recovery with +/-1 step tolerance. |
| TE provenance exists in the v2 cache but is lost in the standard `run_tests.py` plots | Confirmed | Medium | S7-T06 extends `dataset_v2`, `collect_predictions`, `run_sample_diagnostics`, and `plot_sample_lag_attn_diagnostic` so sample PDFs and `sample_metrics.csv` carry TE_inj, TE_scat, TE_raw, frac_Phi, and true lag; S7-T07 runs the bridge smoke. |

## 9. Testing and validation strategy

This repo uses pytest (not the `runner_base.py` pattern from other repos) run through the project
interpreter `.venv/Scripts/python.exe`. Every new/ported unit gets a test module under
`synthetic_v2/tests/`. To keep the real transform affordable, scattering unit tests use a short
signal shape (for example `shape=1024`, small `J`) EXCEPT one full-`shape=5280` end-to-end
shape/count assertion (owned by S2-T01). TE-probe tests that need real physics use a tiny sample
count (for example N~=64) and a small pilot grid rather than a shrunk shape. Lightning smoke tests
train on a tiny in-test fixture cache, not on a real pilot artifact.

Per-unit tests: analytic_te (TE values, inverter round-trip, SNR monotonicity, seed-stable inverter);
raw_generators (band powers, DC ranges, upsample anti-alias, envelope positivity, null separability,
AM-separation pre-check); scattering_adapter (exact counts at full shape, normalisation parity,
monotone-map TE-invariance, fs-correct AM-channel tracking at `latent[15:315]`); te probes (frac_Phi
~=1 on a strong cell and ~=0 on a null; TE_raw sane; fatal-gate behavior); build_dataset_v2 (schema,
row-aligned shuffle, determinism-from-seeds, resume-skip, loader field-mapping, model-forward
compat); train_v2 (one-epoch smoke, ckpt round-trip via `graph_models_utils`, loss/likelihood switch);
eval_v2 (calibration slope fit, lag-mass, null-ratio); visualize_v2 (each figure file produced).
Non-test validations (config, drivers, figures): documented exact CLI commands whose written
artifacts are inspected (frac_Phi table / `realizability.json`, loss curve, `metrics.json`, report).

For standard testing-plot integration, a tiny v2 cache is loaded through `dataset_v2` and passed to
`testing/run_tests.py::run_full_test_pipeline` via `loader_override` with slow real-data-only analyses
disabled. The validation artifact is a one-sample diagnostic PDF and `sample_metrics.csv` containing
`te_true`, `te_scat`, `te_raw` when available, `frac_phi`, `sample_delay`, and the model's
`kld_mean`, proving the standard test path can show actual TE values without a custom plotting fork.

## 10. Rollout and observability

Ships as a self-contained module; nothing else depends on it, so there is no feature flag or phased
rollout - it is additive. Development follows the sprints: scaffolding that runs, generation, the
transform, the critical three-TE de-risk, build/train/eval, then figures and the README. Each sprint
is independently runnable and demoable.

Observability is via structured logs and written artifacts: the build logs each solved cell
(`target_te`, D, solved B, te_block_realised); the pre-flight writes a per-cell `realizability.json`
(TE_inj / TE_raw / TE_scat / frac_Phi); training logs `kld_nats` and loss terms per epoch and writes
loss-curve figures; evaluation writes `metrics.json` (gamma_inj, gamma_scat, intercept, LagMass,
argmax-lag error, null-ratio, mean frac_Phi) plus the figure gallery and a markdown report.
"Rollback" is deleting a cache/run tag directory; caches are keyed by tag so runs never overwrite.

## 11. Open questions

- Real-fold stats path: default is `synthetic_pool` (self-contained; this overrides the EXPLAINED
  Section 16 example, which showed `real_fold`). If `real_fold` is chosen for a run, the path to a
  real fold's `stats.hdf5` must be supplied. Owner: user, at run configuration time.
- Carrier frequency `f_pulse`: default 0.02 Hz may be raised to 0.04-0.06 Hz if the Sprint 1
  AM-separation pre-check or the Sprint 3 frac_Phi gate shows inadequate separation. Owner: resolved
  by the Sprint 1 pre-check and Sprint 3 recovery task.
- Headline grid size: EXPLAINED proposes `target_te_grid=[0,0.5,1,2,3]` x `lag_grid=[4,8,12,20]` at
  `n_per_cell_train=2000`. A smaller pilot grid runs first (Sprint 3); the full grid is locked in
  S4-T01 after the pilot frac_Phi result. Owner: user, after Sprint 3.
- beta (KL weight) for the headline run: fixed default from config vs the optional beta-sweep
  (S5-T03), run only if calibration looks collapsed. Owner: user, after first training.

---

<!-- Phase 5: Sprint Plans, Full Task List, Todo Checklist -->

## Sprint overview

| Sprint | Goal | Demoable outcome | Depends on |
|--------|------|------------------|------------|
| Sprint 0 | Standalone scaffold + analytic-TE port | `run_pipeline_v2.py --solve-te <te> <D>` prints achieved TE + SNR; tests pass | - |
| Sprint 1 | Raw generators + AM-separation pre-check | Generate one raw FHR/UP pair; plot annotated raw signals; AM-separation pre-check prints the margin | Sprint 0 |
| Sprint 2 | Scattering adapter + normalisation | raw -> 4 normalised fields (asserted 43/44/43/58), parity-checked; scattering heatmap with coupled channel marked | Sprint 1 |
| Sprint 3 | Cell enumeration, pilot build, three-TE de-risk | Pre-flight table over a pilot grid: TE_inj / TE_raw / TE_scat / frac_Phi; frac_Phi ~1 strong, ~0 null; recovery if low | Sprint 2 |
| Sprint 4 | Full build, cache, loaders | Locked full grid built to npz + meta.json (resumable, deterministic); a batch runs `model.forward` | Sprint 3 |
| Sprint 5 | Training | Headline training run produces a checkpoint + loss curves | Sprint 4 |
| Sprint 6 | Evaluation gates | gamma_inj / gamma_scat, lag recovery, null collapse from the checkpoint; end-to-end smoke | Sprint 5 |
| Sprint 7 | Journal figures, standard test plots, pulse_train, report, README | Full figure gallery + TE-aware `run_tests.py` sample PDFs + report + README; pulse_train frac_Phi vs am_carrier | Sprint 6 |

---

## Sprint 0: Standalone scaffold and analytic-TE port

Goal: Establish the `synthetic_v2/` package, config (with the batch-size, seed, and fatal-gate knobs),
and the ported single-pathway block-TE math, so a cell's exact injected TE and coupling B can be
solved and printed.
Demoable outcome: `run_pipeline_v2.py --solve-te 2.0 8` prints the solved B, achieved TE (~2.0), and
per-step SNR (~0.143); `pytest tests/test_analytic_te.py` is green.
Depends on: nothing.

### Tasks

#### S0-T01: Package skeleton, config, and solve-te demo hook

Description: Create the `synthetic_v2/` module files as importable stubs and write
`config_synth_v2.yaml` with the `benchmarks.G1_raw` blocks (`data`, `raw`, `scattering`, `mix`,
`eval`) per EXPLAINED Section 16, PLUS `scattering.batch_size`, a `seed`/`base_seed` block,
`eval.realizability.fatal`, `mix.inverter` knobs, and copied `model`/`loss`/`optim`/`dataset` blocks.
Pin `norm_stats_source: synthetic_pool` and document that `phase_min_freq` is normalised xi while
`raw.f_pulse` is true Hz. Add a `run_pipeline_v2.py --solve-te <target_te> <D>` hook that calls the
inverter and prints B, achieved TE, and SNR.

Acceptance criteria:
- `config_synth_v2.yaml` parses and contains every EXPLAINED Section 16 key plus `scattering.batch_size`,
  `seed`/`base_seed`, `eval.realizability.fatal`, `mix.inverter`; `render_mode: am_carrier`, no `m_grid`.
- All planned module files import without error under `.venv`.
- `run_pipeline_v2.py --help` lists stages including `--solve-te`.

Files affected:
- `synthetic_v2/config_synth_v2.yaml`; module stubs; `synthetic_v2/run_pipeline_v2.py` (--solve-te hook).
- `synthetic_v2/tests/test_config_v2.py`.

Validation: `.venv/Scripts/python.exe -m pytest .../synthetic_v2/tests/test_config_v2.py -q` (asserts
required keys and defaults); `run_pipeline_v2.py --solve-te 2.0 8` prints B, TE, SNR.

#### S0-T02: Port analytic_te.py (single pathway)

Description: Copy `synthetic/analytic_te.py` into `synthetic_v2/analytic_te.py` and adapt to a single
pathway: keep `_simulate_state_space_gaussian`, `te_block_state_space_gaussian`,
`mean_te_block_state_space_over_delays`, `B_y_for_mean_te_block_state_space`,
`snr_per_step_for_te_block`, `realizable_te_block_from_arrays`. Set `M=1` at call sites, read inverter
knobs from config, and use LaTeX docstrings per CLAUDE.md.

Acceptance criteria:
- `B_y_for_mean_te_block_state_space(target_te=2.0, delay=8, ...)` returns `B_y_scalar` and `te_block`
  within `tol` of 2.0.
- `snr_per_step_for_te_block(2.0, H=30, M=1)` ~= 0.143.
- No `M` grid remains; module imports cleanly.

Files affected:
- `synthetic_v2/analytic_te.py`.

Validation: standalone smoke `python -c "from ...analytic_te import *; assert abs(...te_block-2.0)<0.1"`
prints OK, and covered by S0-T03.

#### S0-T03: analytic_te tests

Description: Port/adapt tests to `synthetic_v2/tests/test_analytic_te.py`.

Acceptance criteria:
- Inverter round-trip for target TE in {0.5,1.0,2.0,3.0} at D=8 within 5%.
- SNR-law strictly increasing with TE at fixed H.
- `te_block_state_space_gaussian` at B=0 ~= 0 (within MC floor); inverter seed-stable across bisection.

Files affected:
- `synthetic_v2/tests/test_analytic_te.py`.

Validation: `.venv/Scripts/python.exe -m pytest .../tests/test_analytic_te.py -q` passes.

---

## Sprint 1: Raw signal generators and AM-separation pre-check

Goal: Generate the raw 4 Hz FHR/UP waveforms - latent pair on the decimated grid, physiological DC and
independent dressing bands, band-limited upsample, and the positive-envelope AM rendering - and prove
the AM envelope/carrier separation is adequate before any transform work.
Demoable outcome: `generate_cell_raw(...)` returns raw FHR/UP (5280) + `true_lag_tt`; a script plots the
annotated raw signals and the AM-separation pre-check prints the envelope-BW vs wavelet-passband margin.
Depends on: Sprint 0.

### Tasks

#### S1-T01: Coupled latent pair on the decimated grid

Description: In `raw_generators.py`, `simulate_latent_pair(n, T_tot, r, w, target_ar, B, D, sigma2_y,
sigma2_eta, seed)` producing AR(2) source `c_k` and AR(1) coupled target `d_k` on the decimated grid,
returning `c, d` shaped `(n, T_tot)` and `true_lag_tt` via a ported `_true_lag_trajectory`.

Acceptance criteria:
- `c` spectral peak near w=0.10 rad/step (~0.004 Hz); with B=0, `corr(d_k, c_{k-D})` ~0; with B>0
  clearly positive.
- `true_lag_tt` is a flat line at D per sample (fixed mode).

Files affected: `synthetic_v2/raw_generators.py`; `synthetic_v2/tests/test_raw_generators.py`.

Validation: `pytest .../test_raw_generators.py -k latent -q` passes.

#### S1-T02: DC baseline and independent dressing bands

Description: Per-sample DC draws (`mu_fhr~U[110,160]`, `mu_up~U[5,25]`), independent FHRV bands
(LF/MF/HF random-phase cosine-sum at configured powers), UP slow drift and toco noise, accelerations,
optional baseline wander - all independent of the coupled pathway.

Acceptance criteria:
- Empirical band powers match configured ratios within tolerance; DC in range; wander independent of
  `c,d`; reproducible under a fixed seed; independent across FHR/UP.

Files affected: `synthetic_v2/raw_generators.py`; `synthetic_v2/tests/test_raw_generators.py`.

Validation: `pytest .../test_raw_generators.py -k "bands or dc" -q` passes.

#### S1-T03a: Band-limited upsample 330 -> 5280

Description: Implement anti-aliased band-limited upsample of `c,d` from the decimated grid (330) to
raw (5280), keeping the envelope bandwidth below the carrier band.

Acceptance criteria:
- Upsampled envelope has no significant spectral energy at or above `f_pulse` (anti-alias check).
- Output length 5280; endpoints handled without edge blow-up.

Files affected: `synthetic_v2/raw_generators.py`; `synthetic_v2/tests/test_raw_generators.py`.

Validation: `pytest .../test_raw_generators.py -k upsample -q` passes.

#### S1-T03b: Positive AM envelopes, carrier, and raw composition

Description: Positive amplitude envelopes `A_u=a0+a1*c_tilde`, `A_y=b0+b1*d_tilde` (offsets from
`am_offset_ratio` guaranteeing positivity), a fixed unit-scale carrier near `f_pulse`, and the
additive composition into `generate_cell_raw` (UP = mu_UP + A_u*carrier + drift + noise; FHR = mu_FHR
- A_y*carrier + FHRV + accels + noise). Expose `render_mode` (am_carrier now).

Acceptance criteria:
- `A_u,A_y >= A_min > 0` everywhere; `generate_cell_raw` returns FHR/UP length 5280 and provenance
  (D, B, TE_inj).

Files affected: `synthetic_v2/raw_generators.py`; `synthetic_v2/tests/test_raw_generators.py`.

Validation: `pytest .../test_raw_generators.py -k render -q` passes.

#### S1-T04: AM-separation analytic pre-check

Description: Add `am_separation_margin(r, w, f_pulse, Q, fs)` that computes the AR(2) envelope -3 dB
bandwidth and the analyzing-wavelet passband at `f_pulse`, and returns their ratio/margin plus a
recommendation (raise `f_pulse`, narrow `w`) when the margin is inadequate. This is the Sprint 1 gate
for the AM-separation risk.

Acceptance criteria:
- For the default `r=0.80, w=0.10, f_pulse=0.02`, the function reports the (marginal) separation and
  flags it; for a raised `f_pulse` (e.g. 0.05) it reports an improved margin.
- The check is callable from `run_pipeline_v2.py` and prints the margin.

Files affected: `synthetic_v2/raw_generators.py`; `synthetic_v2/tests/test_raw_generators.py`.

Validation: `pytest .../test_raw_generators.py -k separation -q` passes; `run_pipeline_v2.py
--am-check` prints the margin.

#### S1-T05: Raw preview figure and null-separability check

Description: Add a raw-signal preview to `visualize_v2.py` (annotated FHR/UP: DC, decel dips,
contraction pulses, accels) and a null-separability test (B=0 -> lagged cross-correlation of the
rendered raw pair ~0 at the coupled lag).

Acceptance criteria:
- The preview command writes a PDF/PNG of the raw pair; null-separability holds within noise.

Files affected: `synthetic_v2/visualize_v2.py`; `synthetic_v2/tests/test_raw_generators.py`.

Validation: `pytest .../test_raw_generators.py -k null -q` passes; preview file written.

---

## Sprint 2: Scattering adapter and normalisation

Goal: Turn raw FHR/UP into the model's four normalised feature fields using the real transform, with
verified exact counts, fs-correct coupled-channel identification, and a parity-tested local
normaliser - all standalone (no `early_maestra`).
Demoable outcome: `transform_and_normalise(fhr_raw, up_raw)` returns `fhr_st(300,43)`, `fhr_ph(300,44)`,
`up_st(300,43)`, `up_ph(300,58)`; a scattering heatmap is plotted with the coupled channel highlighted.
Depends on: Sprint 1.

### Tasks

#### S2-T01: Transform wrapper (minimal passes) with count assertions and batch knob

Description: In `scattering_adapter.py`, wrap `KymatioPhaseScattering1D(J=11,Q=4,T=16,shape=5280,
max_order=1)`. For the four MODEL fields run TWO passes (one combined scattering+self-phase pass per
channel: FHR -> `fhr_st(43)`+`fhr_ph(44)`; UP -> `up_st(43)`+`up_ph(58)`), applying
`select_fhr_phase_coefficients(min_freq=0.006)` and `(0.002)`; add an OPTIONAL cross pass
(`compute_cross_phase=True`) only for the probe path. Trim 15/end -> 300. Batch over samples with
`scattering.batch_size` and bound peak memory. Import only from `kymatio_phase_scattering.py`. Lock
the xi-vs-Hz threshold semantics in a comment/assert.

Acceptance criteria:
- Output counts exactly 43/44/43/58 (asserted; raises on mismatch); length 300 after trim.
- The four model fields are produced with two passes; no import of `create_new_pipeline`/`early_maestra`.
- Batched transform over N samples respects `batch_size` and stays within a stated memory ceiling.

Files affected: `synthetic_v2/scattering_adapter.py`; `synthetic_v2/tests/test_scattering_adapter.py`.

Validation: `pytest .../test_scattering_adapter.py -k counts -q` passes (real `shape=5280`, the single
full-shape test).

#### S2-T02: Local normalisation, stats source, and production parity

Description: Copy the `normalize_tensor_data` logic into `scattering_adapter.py` (ch0 untouched,
`log(clamp(x,0)+1e-6)` on st ch 1..42, `asinh` on all phase channels, per-channel z-score eps 1e-8).
Implement `norm_stats_source`: `synthetic_pool` (default; stats from the generated pool) and
`real_fold` (load a fold's `stats.hdf5` from config). Add a parity test vs the production function.

Acceptance criteria:
- `synthetic_pool` gives ~N(0,1) per channel; `real_fold` loads/applies mu/sigma and raises clearly
  only when selected without a path; ch0 z-scored but not log/asinh.
- Local normaliser matches `hdf5_dataset.hdf5_dataset.normalize_tensor_data` on a fixed input within
  numerical tolerance (parity).

Files affected: `synthetic_v2/scattering_adapter.py`; `synthetic_v2/tests/test_scattering_adapter.py`.

Validation: `pytest .../test_scattering_adapter.py -k "norm or parity" -q` passes (small `shape=1024`
for norm; parity uses a fixed synthetic input array).

#### S2-T03: End-to-end transform + fs-correct AM-channel identification

Description: Wire `transform_and_normalise(fhr_raw, up_raw)` end to end, and add
`coupled_channel_indices(fs, f_pulse)` selecting the `up_st`/`fhr_st` first-order channel via
`argmin(|center_freqs*fs - f_pulse|)`. Verify the identified `up_st` channel tracks the decimated
latent `c` sliced to `[15:315]` on a strong am_carrier cell.

Acceptance criteria:
- `transform_and_normalise` returns the four fields at correct shapes for one raw pair.
- The chosen channel's true Hz is within one Q-step of `f_pulse` (asserted); its correlation with
  `c[15:315]` is strong on a strong cell (for example |corr| > 0.6), or the shortfall is documented
  and routed to the S3-T06 recovery task.

Files affected: `synthetic_v2/scattering_adapter.py`; `synthetic_v2/tests/test_scattering_adapter.py`.

Validation: `pytest .../test_scattering_adapter.py -k am -q` passes (moderate shape; correlation uses
`[15:315]`).

#### S2-T04: Scattering heatmap preview and TE-invariance test

Description: Add a scattering-heatmap figure to `visualize_v2.py` (43 ch x time, FHR and UP, coupled
channel highlighted) and a test that the monotone normalisation maps preserve a channel's ordering.

Acceptance criteria:
- The heatmap command writes a PDF/PNG with the coupled channel marked; the normalisation is strictly
  monotone per channel (asserted numerically).

Files affected: `synthetic_v2/visualize_v2.py`; `synthetic_v2/tests/test_scattering_adapter.py`.

Validation: `pytest .../test_scattering_adapter.py -k monotone -q` passes (small `shape=1024`); heatmap
file written.

---

## Sprint 3: Cell enumeration, pilot build, and the three-TE de-risk

Goal: Enumerate the cell grid and solve each coupling, generate a pilot pool, and measure all three
transfer entropies on the model-facing features BEFORE any GPU training - the gate that makes v2 worth
building - with a defined recovery path if preservation is low.
Demoable outcome: a pre-flight table over a pilot grid printing TE_inj, TE_raw, TE_scat, frac_Phi per
cell (frac_Phi ~1 on a strong cell, ~0 on a null), written to `realizability.json`.
Depends on: Sprint 2.

### Tasks

#### S3-T01: Cell enumeration, inverter solve, and pilot-sample generation

Description: In `build_dataset_v2.py`, add a `CellV2` dataclass (`cell_id`, `target_te`, `D`,
`B_y_scalar`, `te_block_realised`) and `enumerate_cells_v2(config)` crossing `target_te_grid` x
`lag_grid` (fixed lag; `target_te=0` -> null B=0; dedup solves by `(D, target_te)`; unsolvable cells
logged + dropped). Add `generate_pilot_samples(cell, n, split)` reusing `generate_cell_raw`, shared
with the Sprint 4 build so no generation logic is duplicated.

Acceptance criteria:
- Null cells kept with B=0, te_block_realised=0; non-null solve within `tol`; drops logged, no crash.
- The manifest lists each cell (target TE, D, solved B, realised TE); `generate_pilot_samples` returns
  raw pairs for a small N.

Files affected: `synthetic_v2/build_dataset_v2.py`; `synthetic_v2/tests/test_build_dataset_v2.py`.

Validation: `pytest .../test_build_dataset_v2.py -k enumerate -q` passes.

#### S3-T02: Coupled-channel slicing wrapper for the realizable-TE probe

Description: In `eval_v2.py`, slice the fs-correct decel/contraction pulse-shape channels out of
`fhr_st`/`up_st` into the single-channel arrays `realizable_te_block_from_arrays` expects
(`U[:,:,:1]`, `Y[:,:,:1]`), aligned to the feature grid (`latent[15:315]`), for a batch of samples.

Acceptance criteria:
- Returns `Y`,`U` shaped for the probe; alignment uses `[15:315]`; works batched.
- Acceptance notes that this conditions `Y^-` on one FHR channel (not all 87), so frac_Phi is a
  coupled-sub-process realizability estimate, not a tight bound on what the full model sees.

Files affected: `synthetic_v2/eval_v2.py`; `synthetic_v2/tests/test_te_preservation.py`.

Validation: `pytest .../test_te_preservation.py -k slice -q` passes.

#### S3-T03: frac_Phi probe (TE_scat) with tolerance gate

Description: Implement `measure_te_scat(features, D, cell)` running the ported
`realizable_te_block_from_arrays` (ridge + 70/30 held-out) on the sliced channels, returning `TE_scat`
and `frac_Phi = TE_scat / TE_inj`, with a configurable `frac_threshold` gate.

Acceptance criteria:
- On a strong am_carrier cell (target_te 2-3, D 8) frac_Phi >= `frac_threshold` (for example 0.7) at a
  stated N; on a null cell TE_scat ~= 0.

Files affected: `synthetic_v2/eval_v2.py`; `synthetic_v2/tests/test_te_preservation.py`.

Validation: `pytest .../test_te_preservation.py -k fracphi -q` passes (tiny N per cell).

#### S3-T04: TE_raw probe (raw-domain determinant ratio)

Description: Implement `measure_te_raw(fhr_raw, up_raw, D)` estimating
$\mathrm{TE}^{(H)}_{x_{UP}\to x_{FHR}}$ on the rendered raw (or bandpassed contraction/decel) signals
via the determinant-ratio / ridge held-out approach, so the trio TE_inj / TE_raw / TE_scat can be
reported and `frac_Phi>1` anomalies diagnosed (EXPLAINED Sec 10, 19).

Acceptance criteria:
- Returns a finite `TE_raw` that is ~0 on a null cell and positive on a strong cell; stamped into
  `realizability.json` (and optionally `sample_te_raw`).

Files affected: `synthetic_v2/eval_v2.py`; `synthetic_v2/tests/test_te_preservation.py`.

Validation: `pytest .../test_te_preservation.py -k teraw -q` passes (tiny N).

#### S3-T05: Pre-flight harness over the pilot grid (with fatal gate)

Description: Add `run_realizability_preflight` that, for a small pilot grid (for example
`target_te in {0,1,2,3}`, `D in {4,8}`, N per cell), generates, transforms, normalises, runs all three
TE probes, writes `realizability.json` and a printed table, and honors `eval.realizability.fatal`
(halt/raise when `frac_Phi < frac_threshold` if fatal, else warn).

Acceptance criteria:
- Runs end to end on the local GPU; `realizability.json` has per-cell TE_inj/TE_raw/TE_scat/frac_Phi;
  frac_Phi trends to 1 on strong cells, to 0 on null; `fatal: true` raises on a low-frac cell.

Files affected: `synthetic_v2/eval_v2.py`; `synthetic_v2/run_pipeline_v2.py` (`r0_realizability` stage);
`synthetic_v2/tests/test_te_preservation.py`.

Validation: `run_pipeline_v2.py --stage r0_realizability --pilot` writes `realizability.json`;
`pytest .../test_te_preservation.py -k preflight -q` passes on a tiny grid.

#### S3-T06: frac_Phi recovery / tuning

Description: If the pilot frac_Phi is below threshold, sweep the render knobs (`f_pulse` toward
0.04-0.06, carrier separation, `am_offset_ratio`, cell strength), re-measure via S1-T04 + S3-T03,
recheck lag identifiability, and record the chosen params in the manifest. Defines the sprint's
behavior when the gate is not met (rather than a dead end).

Acceptance criteria:
- The tuning routine runs a small sweep and reports frac_Phi and the AM-separation margin per setting;
  the selected params are written to the manifest; lag identifiability is rechecked for the chosen
  setting.

Files affected: `synthetic_v2/eval_v2.py` (or `raw_generators.py`); `synthetic_v2/tests/test_te_preservation.py`.

Validation: `pytest .../test_te_preservation.py -k recovery -q` passes (enumerates the sweep on a tiny
grid); a manifest field records the chosen `f_pulse`/knobs.

---

## Sprint 4: Full build, cache, and loaders

Goal: Lock the full grid from the pilot result, then generate -> scatter -> normalise all splits with a
resumable, deterministic, GPU-batched build; stamp v2 provenance (including per-cell frac_Phi);
shuffle row-aligned; write the cache; and load it into model batches.
Demoable outcome: the full cache (`train/val/test.npz` + `meta.json`) is built; a batch loads and runs
through `model.forward` without shape errors.
Depends on: Sprint 3.

### Tasks

#### S4-T01: Lock the full grid and finalize config

Description: Using the Sprint 3 frac_Phi/AM-separation results, finalize `target_te_grid`, `lag_grid`,
`n_per_cell_{train,val,test}`, the chosen `f_pulse`/render knobs, and `scattering.batch_size` in
`config_synth_v2.yaml`; record the rationale in the manifest/README.

Acceptance criteria:
- The config's full grid and render knobs reflect the pilot outcome; a config test asserts the grid
  is present and internally consistent (no `m_grid`, fixed lag).

Files affected: `synthetic_v2/config_synth_v2.yaml`; `synthetic_v2/tests/test_config_v2.py`.

Validation: `pytest .../test_config_v2.py -k grid -q` passes.

#### S4-T02a: build_split for one (cell, split) with provenance and per-cell frac_Phi

Description: `build_split(cell, split, n)` generates raw pairs (reusing `generate_pilot_samples`
primitives), scatters + normalises in GPU batches, invokes the S3 `measure_te_scat` probe for the
cell's pool to compute `te_scat`/`frac_phi` at build N, and stamps all provenance
(`sample_te_true`, `sample_te_scat`, `sample_frac_phi`, `sample_te_raw` optional, `sample_delay`,
`sample_cell_id`, `sample_held_out`, `true_lag_tt`, `weight`). Order: generate-all -> scatter/normalise
-> probe -> stamp.

Acceptance criteria:
- Returns arrays with the Section 17 shapes for one cell; every provenance field present and correctly
  shaped; no `sample_M`/band fields; frac_Phi computed at build N (not the pilot N).

Files affected: `synthetic_v2/build_dataset_v2.py`; `synthetic_v2/tests/test_build_dataset_v2.py`.

Validation: `pytest .../test_build_dataset_v2.py -k provenance -q` passes.

#### S4-T02b: Pool, shuffle, and write cache (deterministic)

Description: Concatenate all cells' arrays, apply one shared row-aligned permutation seeded from
config, and `write_cache` the four fields + provenance to `.npz` plus the manifest to `meta.json`
(pooled te_true, channel_map, per-cell manifest, raw/scattering blocks). Seed everything from the
`seed`/`base_seed` block.

Acceptance criteria:
- Cache arrays match Section 17; the shared shuffle keeps features and provenance row-aligned (verified
  by a stamped index); rebuilding a tiny cache from the stored seeds yields identical arrays
  (determinism).

Files affected: `synthetic_v2/build_dataset_v2.py`; `synthetic_v2/tests/test_build_dataset_v2.py`.

Validation: `pytest .../test_build_dataset_v2.py -k "schema or shuffle or determinism" -q` passes.

#### S4-T02c: Resumable per-(cell, split) build

Description: Add per-(cell, split) checkpoint files so a re-run skips completed work; the full build is
crash-safe.

Acceptance criteria:
- Re-running with existing checkpoints skips completed (cell, split) units and produces an identical
  final cache.

Files affected: `synthetic_v2/build_dataset_v2.py`; `synthetic_v2/tests/test_build_dataset_v2.py`.

Validation: `pytest .../test_build_dataset_v2.py -k resume -q` passes.

#### S4-T03: dataset_v2 / datamodule_v2 loaders and model-forward compat

Description: Copy/adapt `dataset.py` and `datamodule_synth.py` into `dataset_v2.py` / `datamodule_v2.py`
using v2 provenance (map `sample_delay -> delay_max` where the clean-window floor needs it; strip
`sample_M`/`delay_min/max`/`band_id` reads); `build_u_stream` concatenates `up_st(43)+up_ph(58)=101`.
Add a model-forward-compat check on a built batch.

Acceptance criteria:
- A `DataLoader` yields `fhr_st, fhr_ph, up_st, up_ph, weight` + v2 provenance; a loader test asserts
  NO `M`/`band_id`/`delay_min`/`delay_max` KEYS remain; `build_u_stream` -> `(B,300,101)`.
- `SeqVaeLagAttnV1.forward(y_st,y_ph,u_stream)` on a built batch returns `kld_per_t (B,300)` and
  `attn_weights (B,300,num_heads,91)` without shape/dtype errors.

Files affected: `synthetic_v2/dataset_v2.py`, `synthetic_v2/datamodule_v2.py`;
`synthetic_v2/tests/test_build_dataset_v2.py`.

Validation: `pytest .../test_build_dataset_v2.py -k "loader or forward" -q` passes.

---

## Sprint 5: Training

Goal: Train the unchanged VAE-TEB model on the v2 cache via a Lightning wrapper and a staged driver;
single-GPU for local runs, DDP-capable for the prod box; run the headline training.
Demoable outcome: a headline training run on the full cache produces a checkpoint and loss curves.
Depends on: Sprint 4.

### Tasks

#### S5-T01: pl_module_v2, trainer, ckpt round-trip, and loss switch

Description: Copy/adapt `pl_module_synth.py` into `pl_module_v2.py` (loss knobs `kld_beta`,
`lambda_full`, `lambda_base`, `likelihood`, `sigma_obs`, `free_bits`); provide a trainer entry
(single-GPU default, DDP when `devices>1`); use `train/graph_models_utils.py` for checkpoint save/load.

Acceptance criteria:
- The module wraps `SeqVaeLagAttnV1`, logs `kld_nats`, and completes one epoch on a tiny in-test
  fixture cache; a saved-then-reloaded model (via `graph_models_utils`) gives identical forward output
  on a fixed batch; `mse` and `gaussian_nll` both yield finite losses and `free_bits` clamps KL.

Files affected: `synthetic_v2/pl_module_v2.py`; `synthetic_v2/tests/test_train_v2.py`.

Validation: `pytest .../test_train_v2.py -k "module or ckpt or loss" -q` passes.

#### S5-T02: run_pipeline_v2 driver stages

Description: Implement ordered stages in `run_pipeline_v2.py` (`build`, `r0_realizability`,
`data_previews`, `train`, `eval`, `report`) with `--stage`/`--pilot` CLI, wiring build -> probe ->
previews -> train.

Acceptance criteria:
- `--help` lists the stages; `--stage train --pilot` trains a few epochs and writes a checkpoint under
  `results/<run_tag>/`.

Files affected: `synthetic_v2/run_pipeline_v2.py`; `synthetic_v2/tests/test_train_v2.py`.

Validation: `pytest .../test_train_v2.py -k smoke -q` passes; `run_pipeline_v2.py --stage train --pilot`
writes a checkpoint.

#### S5-T03: Optional beta-selection stage

Description: Optional `beta_select` stage: short runs over a small `beta_grid`, recording the beta with
the least-collapsed latent (highest `kld_nats` without wrecking reconstruction). Skippable (default
off; fixed `kld_beta` otherwise).

Acceptance criteria:
- Writes `beta_select.json` with per-beta `kld_nats` and loss; default-off path uses the config beta.

Files affected: `synthetic_v2/run_pipeline_v2.py`; `synthetic_v2/tests/test_train_v2.py`.

Validation: `pytest .../test_train_v2.py -k beta -q` passes (enumeration only, no full training).

#### S5-T04: Headline training run

Description: Run the full-grid training (single-GPU locally or DDP on the prod box) from the locked
config, producing the checkpoint and loss curves the evaluation grades.

Acceptance criteria:
- A checkpoint and loss-curve figures are written under `results/<run_tag>/`; `kld_nats` is logged per
  epoch; the run is reproducible from the config + seeds.

Files affected: `synthetic_v2/run_pipeline_v2.py` (train stage wiring); `results/<run_tag>/` artifacts.

Validation: `run_pipeline_v2.py --stage train` completes and writes the checkpoint + loss curves
(artifact inspection).

---

## Sprint 6: Evaluation gates

Goal: From the trained checkpoint, produce the verdicts - gamma-calibration of K-bar vs both TE_inj and
TE_scat, lag recovery vs the true lag band, null-control collapse - plus an end-to-end smoke.
Demoable outcome: a `metrics.json` with gamma_inj, gamma_scat, intercept, LagMass, argmax-lag error,
null-ratio, and mean frac_Phi from an eval pass on the test split.
Depends on: Sprint 5.

### Tasks

#### S6-T01: Per-sample K-bar collection over the clean window

Description: Copy/adapt `collect_per_sample_kbar` into `eval_v2.py`: one eval pass, average `kld_per_t`
over the clean window `[max(w, D-1), T-H)` per sample using the per-cell `sample_delay` (mapped to the
window floor), grouped by `sample_cell_id`; strip the v1 M/band grouping and the `per_dim_kl_by_M` /
`kld_time_by_band` structures. Carry `sample_te_true` and `sample_te_scat`.

Acceptance criteria:
- Returns per-sample K-bar + TE_inj + TE_scat grouped by cell; the window floor uses per-cell D; no
  reference to `M`/`band_id`/`delay_max` provenance.

Files affected: `synthetic_v2/eval_v2.py`; `synthetic_v2/tests/test_eval_v2.py`.

Validation: `pytest .../test_eval_v2.py -k kbar -q` passes.

#### S6-T02: gamma-calibration against TE_inj and TE_scat

Description: Fit `K-bar = alpha + gamma*TE` across cells against both TE_inj and TE_scat; report
`gamma_inj`, `gamma_scat`, `alpha`, R^2, and a monotonicity check.

Acceptance criteria:
- Two slopes + one intercept + R^2 reported; monotonicity of K-bar vs TE checked and flagged.

Files affected: `synthetic_v2/eval_v2.py`; `synthetic_v2/tests/test_eval_v2.py`.

Validation: `pytest .../test_eval_v2.py -k calib -q` passes (slope fit on synthetic points).

#### S6-T03: Lag recovery

Description: For each cell with lag D and true band `L* = {max(0,D-H),...,D-1}`, compute attention
LagMass inside `L*` and the argmax-lag error (+/-1 tolerance) from `attn_weights`.

Acceptance criteria:
- LagMass and argmax-lag error reported per cell; a synthetic attention fixture peaked in `L*` scores
  LagMass ~1; +/-1 tolerance applied.

Files affected: `synthetic_v2/eval_v2.py`; `synthetic_v2/tests/test_eval_v2.py`.

Validation: `pytest .../test_eval_v2.py -k lag -q` passes.

#### S6-T04: Null controls and metrics report

Description: Implement shuffle/reverse source-stream null controls, report `null_ratio = K-bar_null /
K-bar_signal`, run the frac_Phi/TE_raw probes on a null cell, and write `metrics.json` (gamma_inj,
gamma_scat, alpha, LagMass, argmax-lag error, null_ratio, per-cell frac_Phi, TE_raw summary).

Acceptance criteria:
- Null controls run; `null_ratio` trends to 0 for real signal cells; `metrics.json` contains all gates.

Files affected: `synthetic_v2/eval_v2.py`; `synthetic_v2/run_pipeline_v2.py` (`eval` stage);
`synthetic_v2/tests/test_eval_v2.py`.

Validation: `pytest .../test_eval_v2.py -k null -q` passes; `run_pipeline_v2.py --stage eval` writes
`metrics.json`.

#### S6-T05: End-to-end integration smoke

Description: Add a tiny-grid integration test/driver path that runs build -> r0_realizability -> train
(few epochs) -> eval -> report in one pass, proving the stages compose.

Acceptance criteria:
- The end-to-end tiny run completes and produces the cache, `realizability.json`, a checkpoint,
  `metrics.json`, and a report without manual intervention.

Files affected: `synthetic_v2/run_pipeline_v2.py`; `synthetic_v2/tests/test_eval_v2.py`.

Validation: `pytest .../test_eval_v2.py -k e2e -q` passes (tiny grid, small N, short T).

---

## Sprint 7: Journal figures, standard test plots, pulse_train, report, and README

Goal: Deliver the revised tailored figure set (the user's explicit priority), make the standard
`testing/run_tests.py` plots synthetic-TE-aware, assemble the combined final report, add the
`pulse_train` render variant with its measured frac_Phi, and write a usage README.
Demoable outcome: a full figure gallery (raw + scattering, latent + AM decomposition, calibration / lag
/ frac_Phi / TE_raw), the data-generation *story* figures that document the controls behind the previews
(band recipe, TE authoring, latent coupling, AM separation — S7-T10), standard one-sample diagnostics
annotated with actual TE values, a final report, and a README; `pulse_train` frac_Phi measured vs am_carrier.
Depends on: Sprint 6.

### Tasks

#### S7-T01: plot_style_v2 and the annotated raw-signal figure

Description: Copy/adapt `plot_style.py` into `plot_style_v2.py` as the synthetic v2 plotting source of
truth and finalize the raw-signal figure (FHR with baseline, decel dips, accels; UP with resting tone,
contraction pulses), house style, PDF+PNG. Multi-panel figures use the two-column stacked GridSpec /
dedicated colorbar-gutter pattern from the existing synthetic `plot_style.py`, not per-axes colorbars
that shrink heatmap rows.

Acceptance criteria:
- `plot_style_v2.apply_style()` is called once by `visualize_v2.py`; the raw figure annotates each
  band and the coupled events on both signals in the repo's publication style.
- PDF and 600-dpi PNG are written for every figure; multi-panel figures have aligned panel widths,
  consistent serif fonts, thin black spines, inward ticks, and no oversized titles.

Files affected: `synthetic_v2/plot_style_v2.py`, `synthetic_v2/visualize_v2.py`;
`synthetic_v2/tests/test_visualize_v2.py`.

Validation: `pytest .../test_visualize_v2.py -k raw -q` passes (file produced).

#### S7-T02: Raw + scattering paired preview

Description: The headline preview: raw FHR/UP on top, their scattering-coefficient heatmaps below, the
fs-correct coupled pulse-shape channel highlighted, and the decimated latent (`[15:315]`) overlaid on
that channel.

Acceptance criteria:
- One figure shows raw signal and its scattering transform (stacked) for both FHR and UP; the coupled
  channel is highlighted and visibly tracks the latent on a strong cell.

Files affected: `synthetic_v2/visualize_v2.py`; `synthetic_v2/tests/test_visualize_v2.py`.

Validation: `pytest .../test_visualize_v2.py -k paired -q` passes (file produced).

#### S7-T03: Latent/AM decomposition and diagnostics figures

Description: Add the latent-pair + AM envelope/carrier decomposition figure and the diagnostics panel
(K-bar vs TE_inj and TE_scat with fitted lines, frac_Phi and TE_raw per cell, attention-vs-lag with
`L*` marked). Include confidence intervals or bootstrap bands where there is a per-cell distribution,
and show cell counts so the plots are interpretable as evidence rather than examples.

Acceptance criteria:
- The decomposition figure shows `c,d`, the upsampled envelope, the carrier, and the rendered band; the
  diagnostics panel renders the two calibration lines, frac_Phi/TE_raw bars, and a lag-recovery heatmap.
- Calibration and lag panels display `n`, slope/intercept/R^2, true lag band, tolerance, and cell-level
  uncertainty where applicable.

Files affected: `synthetic_v2/visualize_v2.py`; `synthetic_v2/tests/test_visualize_v2.py`.

Validation: `pytest .../test_visualize_v2.py -k "decomp or diag" -q` passes (files produced).

#### S7-T04: pulse_train render variant and its frac_Phi

Description: Implement the `pulse_train` render mode (raised-cosine contraction/deceleration event
train), selectable via `render_mode`, and measure its frac_Phi against am_carrier on the same cells.

Acceptance criteria:
- `render_mode: pulse_train` produces valid raw signals passing the shape/positivity checks; its
  frac_Phi is measured and reported next to am_carrier (expected lower; documented, not assumed).

Files affected: `synthetic_v2/raw_generators.py`; `synthetic_v2/tests/test_raw_generators.py`,
`test_te_preservation.py`.

Validation: `pytest .../test_raw_generators.py -k pulse -q` and `.../test_te_preservation.py -k pulse -q`
pass.

#### S7-T05: Final report assembly

Description: `final_report_v2` assembles the manifest, the three-TE table (TE_inj/TE_raw/TE_scat/frac_Phi),
calibration and lag metrics, the figure gallery, and the standard testing sample-plot summary into a
single markdown/PDF report under `results/<run_tag>/`.

Acceptance criteria:
- The `report` stage writes a report referencing every figure and the `metrics.json` values, with a
  summary table (gamma_inj, gamma_scat, mean frac_Phi, LagMass, null_ratio).
- The report includes at least one representative sample PDF/PNG entry where actual TE_inj, TE_scat,
  optional TE_raw, frac_Phi, model K-bar, and true lag are visible.

Files affected: `synthetic_v2/eval_v2.py` or `synthetic_v2/final_report_v2.py`;
`synthetic_v2/run_pipeline_v2.py` (`report` stage); `synthetic_v2/tests/test_visualize_v2.py`.

Validation: `run_pipeline_v2.py --stage report` writes the report; `pytest .../test_visualize_v2.py -k
report -q` passes.

#### S7-T06: Standard testing metadata bridge for TE-aware sample plots

Description: Extend the v2 loader and standard testing collectors so synthetic TE provenance survives
the normal `run_tests.py` path. `dataset_v2.__getitem__` exposes `te_true` (alias of
`sample_te_true`/`TE_inj`), `te_scat`, optional `te_raw`, `frac_phi`, `sample_delay`, `cell_id`, and
`true_lag_tt`; `testing/collectors.py::collect_predictions` extracts those fields; and
`testing/analyses/qualitative.py` writes them to `sample_metrics.csv`. Update
`plot_sample_lag_attn_diagnostic()` so one-sample PDFs show actual TE values in the title/header and
draw the true lag band when `true_lag_tt` is present.

Acceptance criteria:
- A collected sample dict from a v2 loader contains `te_true`, `te_scat`, `frac_phi`, `sample_delay`,
  `cell_id`, `true_lag_tt`, and `te_raw` when present.
- `sample_metrics.csv` includes the same TE/provenance columns plus `kld_mean`.
- The main one-sample diagnostic PDF header includes `TE_inj`, `TE_scat`, `frac_Phi`, optional
  `TE_raw`, model `K-bar`, and `D`; attention/TE-lag panels mark the true lag band.

Files affected: `synthetic_v2/dataset_v2.py`; `model/vae_teb_prediction/testing/collectors.py`;
`model/vae_teb_prediction/testing/analyses/qualitative.py`;
`model/vae_teb_prediction/testing/plot_single_samples.py`; `synthetic_v2/tests/test_visualize_v2.py`.

Validation: `pytest .../test_visualize_v2.py -k "metadata_bridge or sample_te" -q` passes; manual
inspection of one generated PDF confirms the actual TE values are visible.

#### S7-T07: run_tests.py bridge smoke for synthetic v2

Description: Add a `run_pipeline_v2.py --stage test_plots` (or equivalently documented helper) that
builds a v2 `DataLoader` from the test split and calls
`testing/run_tests.py::run_full_test_pipeline` with `loader_override`, `skip_trajectory=True`,
`skip_interactive=True`, `analysis_samples` set from config, and real-data-only analyses disabled as
needed. This proves the standard evaluation entry point can produce v2 sample plots without HDF5 paths.

Acceptance criteria:
- The bridge writes the standard `samples_diag/` PDFs and `sample_metrics.csv` for a tiny v2 cache.
- The output contains actual TE/provenance columns and at least one PDF with TE_inj/TE_scat/frac_Phi
  and true lag annotations.
- The bridge does not require GUID-organised HDF5 data or a `stats.hdf5` file.

Files affected: `synthetic_v2/run_pipeline_v2.py`; `synthetic_v2/tests/test_visualize_v2.py`;
`synthetic_v2/README.md`.

Validation: `run_pipeline_v2.py --stage test_plots --pilot --analysis-samples 1` writes a
TE-annotated sample diagnostic and exits successfully.

#### S7-T08: TE-aware aggregate plot upgrades

Description: Make the non-sample plots more complete for paper-level review: add grouped K-bar vs
TE_inj/TE_scat plots by lag cell, TE_raw/frac_Phi distributions with uncertainty, lag-mass vs true
lag-band summaries, and residual/uplift/error summaries stratified by true TE and lag. When the
standard testing plots see TE columns, they should group or annotate by those columns rather than
falling back to label-only real-data views.

Acceptance criteria:
- Aggregate figures report sample counts, per-cell means/intervals, true TE labels, true lag labels,
  and pass/fail gate thresholds where relevant.
- Plots gracefully degrade on real data with no TE columns and on synthetic v2 with no class labels.
- Figure filenames and report captions distinguish example plots from population-level evidence.

Files affected: `synthetic_v2/visualize_v2.py`; `model/vae_teb_prediction/testing/visualizers.py`
where TE-column-aware fallbacks are needed; `synthetic_v2/tests/test_visualize_v2.py`.

Validation: `pytest .../test_visualize_v2.py -k "aggregate or te_columns" -q` passes; generated
aggregate figures include TE/lag group labels and counts.

#### S7-T09: Module README / usage doc

Description: Write `synthetic_v2/README.md` documenting the pipeline stages, the exact CLI commands per
stage (`--solve-te`, `--am-check`, `r0_realizability`, `build`, `train`, `eval`, `test_plots`,
`report`), the config knobs, the outputs, and pointers to the two design docs.

Acceptance criteria:
- The README lists every `run_pipeline_v2.py` stage with a runnable example command and the artifacts it
  produces; it links the EXPLAINED and this SPEC doc.

Files affected: `synthetic_v2/README.md`.

Validation: The documented commands are copy-runnable (manually verified against `run_pipeline_v2.py
--help`); no unresolved placeholders remain.

---

#### S7-T10: Data-generation story figures (the controls)

Description: Add four data-domain "story" figures to `visualize_v2.py` that document the *controls* behind
the generation stages (the existing gallery shows only the outputs): `plot_band_spectra` — the frequency
recipe (Welch PSD of raw FHR/UP with the physiological bands + coupled carrier + LF notch marked, §4-§5);
`plot_te_authoring` — the TE control (a Monte-Carlo `TE^(H)(B)` sweep with the inverter-solved `B` per
`target_te` overlaid, plus the `SNR ≈ exp(2·TE/(H·M))−1` extractability law with the ~1% floor, §9);
`plot_latent_coupling` — the coupling pathway/lag (`c`/`d`/`c` shifted by `D`, and the cross-correlation
peaking at `D` with the true lag band `L* = {max(0,D−H),…,D−1}` shaded, §6); `plot_am_separation` — the
carrier de-risk (AR(2) envelope PSD vs the analyzing-wavelet passband at `f_pulse=0.06` vs the rejected
`0.02` Hz, with the separation margin annotated, §7). All four reuse the pipeline's own math
(`raw_generators`, `analytic_te`) via lazy imports so `visualize_v2` stays free of the torch/kymatio
transform stack. Wire all four into `run_pipeline_v2.data_previews` so a plain run emits them into
`results/<tag>/figures/`.

Acceptance criteria:
- All four figures render PDF+PNG under the headless `Agg` backend from a single strong cell + `config`
  (no realizability preflight needed); `te_authoring` runs a modest self-contained Monte-Carlo.
- `data_previews` appends `band_spectra`, `latent_coupling`, `am_separation`, `te_authoring` to its
  written list; per-figure tests in `tests/test_visualize_v2.py` assert non-empty PDF+PNG.

Files affected: `synthetic_v2/visualize_v2.py`, `synthetic_v2/run_pipeline_v2.py`,
`synthetic_v2/tests/test_visualize_v2.py`.

Validation: `pytest tests/test_visualize_v2.py -k "spectra or authoring or coupling or separation" -q`
passes; each figure eyeballed at low DPI (carrier/band placement, `TE(B)` through the solved points,
xcorr peak at `D`, 0.06-vs-0.02 wavelet/envelope overlap).

---

## Full task list

- S0-T01: Package skeleton, config, and solve-te demo hook
- S0-T02: Port analytic_te.py (single pathway)
- S0-T03: analytic_te tests
- S1-T01: Coupled latent pair on the decimated grid
- S1-T02: DC baseline and independent dressing bands
- S1-T03a: Band-limited upsample 330 -> 5280
- S1-T03b: Positive AM envelopes, carrier, and raw composition
- S1-T04: AM-separation analytic pre-check
- S1-T05: Raw preview figure and null-separability check
- S2-T01: Transform wrapper (minimal passes) with count assertions and batch knob
- S2-T02: Local normalisation, stats source, and production parity
- S2-T03: End-to-end transform + fs-correct AM-channel identification
- S2-T04: Scattering heatmap preview and TE-invariance test
- S3-T01: Cell enumeration, inverter solve, and pilot-sample generation
- S3-T02: Coupled-channel slicing wrapper for the realizable-TE probe
- S3-T03: frac_Phi probe (TE_scat) with tolerance gate
- S3-T04: TE_raw probe (raw-domain determinant ratio)
- S3-T05: Pre-flight harness over the pilot grid (with fatal gate)
- S3-T06: frac_Phi recovery / tuning
- S4-T01: Lock the full grid and finalize config
- S4-T02a: build_split for one (cell, split) with provenance and per-cell frac_Phi
- S4-T02b: Pool, shuffle, and write cache (deterministic)
- S4-T02c: Resumable per-(cell, split) build
- S4-T03: dataset_v2 / datamodule_v2 loaders and model-forward compat
- S5-T01: pl_module_v2, trainer, ckpt round-trip, and loss switch
- S5-T02: run_pipeline_v2 driver stages
- S5-T03: Optional beta-selection stage
- S5-T04: Headline training run
- S6-T01: Per-sample K-bar collection over the clean window
- S6-T02: gamma-calibration against TE_inj and TE_scat
- S6-T03: Lag recovery
- S6-T04: Null controls and metrics report
- S6-T05: End-to-end integration smoke
- S7-T01: plot_style_v2 and the annotated raw-signal figure
- S7-T02: Raw + scattering paired preview
- S7-T03: Latent/AM decomposition and diagnostics figures
- S7-T04: pulse_train render variant and its frac_Phi
- S7-T05: Final report assembly
- S7-T06: Standard testing metadata bridge for TE-aware sample plots
- S7-T07: run_tests.py bridge smoke for synthetic v2
- S7-T08: TE-aware aggregate plot upgrades
- S7-T09: Module README / usage doc
- S7-T10: Data-generation story figures (band recipe, TE authoring, latent coupling, AM separation)

---

## Todo checklist

### Sprint 0: Standalone scaffold and analytic-TE port  — DONE (2026-06-30)
- [x] S0-T01: Package skeleton, config, and solve-te demo hook
- [x] S0-T02: Port analytic_te.py (single pathway)
- [x] S0-T03: analytic_te tests

> Sprint 0 delivered: `analytic_te.py` (ported single-pathway block-TE / inverter /
> SNR / R0), `config_synth_v2.yaml` (G1_raw block per §16 + `scattering.batch_size`,
> `seeds`, `eval.realizability.fatal`, `mix.inverter`; no `m_grid`), `run_pipeline_v2.py`
> (`--solve-te` hook), importable stubs for every planned module, and
> `tests/{test_config_v2,test_analytic_te}.py`. Verified: `--solve-te 2.0 8` →
> `te_block=2.0003`, `SNR=0.1427`; `test_config_v2` 10/10 green; `test_analytic_te`
> 9/9 green; all 11 modules import cleanly.

### Sprint 1: Raw signal generators and AM-separation pre-check  — DONE (2026-07-01)
- [x] S1-T01: Coupled latent pair on the decimated grid
- [x] S1-T02: DC baseline and independent dressing bands
- [x] S1-T03a: Band-limited upsample 330 -> 5280
- [x] S1-T03b: Positive AM envelopes, carrier, and raw composition
- [x] S1-T04: AM-separation analytic pre-check
- [x] S1-T05: Raw preview figure and null-separability check

> Sprint 1 delivered: `raw_generators.py` (single-pathway latent pair reusing
> `analytic_te._simulate_state_space_gaussian`; ported `true_lag_trajectory`; closed-form
> `ar2_stationary_std`; on-DFT-grid power bands `synth_band`/`synth_fhrv`, DC draws,
> accelerations, drift/wander, white noise; FFT `upsample_bandlimited` with Nyquist-bin fix
> and anti-alias cutoff; strictly-positive `am_envelope`; `generate_cell_raw` composing the
> raw FHR/UP pair; `am_separation_margin`/`am_separation_from_config`), the `plot_raw_preview`
> figure in `visualize_v2.py`, the `--am-check` hook in `run_pipeline_v2.py`, and
> `tests/test_raw_generators.py`. Verified: full suite 43/43 green (24 new + 19 Sprint 0);
> `--am-check` reports margin_peak≈0.522 (flagged MARGINAL), preservation≈0.211,
> mod_depth_rms≈0.0625; preview PDF/PNG written.
>
> **Flagged for Sprint 3 pilot / Sprint 4 lock (NOT changed here):** the default
> `am_offset_ratio=4.0` gives only ~7.6% (RMS ~6.25%) modulation depth, and with
> `f_pulse=0.02 Hz` the AM-separation margin_peak≈0.52 (<1) / preservation≈0.21 — a leading
> `frac_Phi` risk. Levers to sweep in S3-T06: lower `am_offset_ratio` (~1.5-2.5), raise
> `f_pulse` (but ~0.05 Hz overlaps the FHRV LF band 0.03-0.15 on the FHR side), or narrow
> `omega` to lower `f_env_peak` (lag-1 autocorr stays ~0.97, so identifiability cost is
> modest). Two design refinements beyond the plan: the AR(2) is overdamped so its PSD peaks
> near DC (~0.0015 Hz) below the pole angle 0.004 Hz (the low-frequency test asserts <0.01 Hz
> rather than a fixed bin); and UP/FHR use independent per-sample carrier phases (the
> scattering modulus is phase-blind) to remove a shared-carrier raw cross-correlation artifact.

### Sprint 2: Scattering adapter and normalisation  — DONE (2026-07-01)
- [x] S2-T01: Transform wrapper (minimal passes) with count assertions and batch knob
- [x] S2-T02: Local normalisation, stats source, and production parity
- [x] S2-T03: End-to-end transform + fs-correct AM-channel identification
- [x] S2-T04: Scattering heatmap preview and TE-invariance test

> Sprint 2 delivered: `scattering_adapter.py` — the `ScatteringAdapter` class wraps the
> production `KymatioPhaseScattering1D` (imports only `hdf5_dataset/kymatio_phase_scattering.py`,
> never `create_new_pipeline`/`early_maestra`), runs **two self-phase passes** (one per signal:
> `forward(x, compute_phase=True, scattering_channel=0, phase_channels=[0])` yields both
> `scattering` and `phase_corr`), applies the `select_fhr_phase_coefficients(min_freq=…)` masks,
> trims 15/end (330→300), and batches over samples with `scattering.batch_size`. Channel counts
> **verified live**: scattering/`up_st`/`fhr_st` = 43, `fhr_ph` (min_freq 0.006) = 44, `up_ph`
> (min_freq 0.002) = 58; `assert_production_counts()` fails loudly on drift (default-on at
> `n_raw=5280`). `transform_and_normalise` returns model-/cache-facing `(n,300,C)` fields.
> Normalisation is a local, **parity-tested** copy of `normalize_tensor_data` (ch0 untouched,
> `log(clamp+1e-6)` on `*_st` ch 1.., `asinh` on `*_ph`, per-channel z-score eps 1e-8) —
> max |diff| vs the production function ≈ 5e-7 (float32); `synthetic_pool` (default) and
> `real_fold` stats sources, the latter raising clearly without a path. `coupled_channel_indices`
> selects the pulse-shape `*_st` channel via `argmin(|center_freqs·fs − f_pulse|)` → **channel 26
> at 0.0196 Hz (ξ=0.00490)**, within one Q-step of `f_pulse=0.02`. Added
> `visualize_v2.plot_scattering_heatmap` (stacked FHR/UP heatmaps, coupled channel highlighted,
> Hz y-labels, shared colorbar gutter) and the one-off `run_pipeline_v2.py --scatter-preview`
> hook. `tests/test_scattering_adapter.py` 10/10 green; full v2 suite 61/61 green.
>
> **Flagged for Sprint 3 (frac_Phi gate / S3-T06 recovery, NOT changed here):** with the default
> `f_pulse=0.02` and `am_offset_ratio=4.0`, the coupled `up_st` channel tracks the decimated
> latent `c[15:315]` at only **|corr| ≈ 0.30–0.33** (consistent with Sprint 1's preservation
> ≈0.21 / margin_peak≈0.52) — clearly positive but well below the aspirational 0.6. The S2-T03
> tracking test therefore asserts `|corr| > 0.15` and documents the shortfall; raising `f_pulse`
> (0.04–0.06), lowering `am_offset_ratio` (~1.5–2.5), and/or narrowing `omega` are the S3-T06
> levers, proven by the Sprint 3 `frac_Phi` probe rather than assumed.

### Sprint 3: Cell enumeration, pilot build, and the three-TE de-risk  — DONE (2026-07-01)
- [x] S3-T01: Cell enumeration, inverter solve, and pilot-sample generation
- [x] S3-T02: Coupled-channel slicing wrapper for the realizable-TE probe
- [x] S3-T03: frac_Phi probe (TE_scat) with tolerance gate
- [x] S3-T04: TE_raw probe (raw-domain determinant ratio)
- [x] S3-T05: Pre-flight harness over the pilot grid (with fatal gate)
- [x] S3-T06: frac_Phi recovery / tuning

> Sprint 3 delivered: `build_dataset_v2.py` (`CellV2`; `solve_cell_coupling` — now the
> single owner of the inverter call, with `run_pipeline_v2.solve_te` delegating to it;
> `enumerate_cells_v2` crossing `target_te_grid`×`lag_grid` with a null `B=0` cell,
> `(target_te,D)` solve memoisation, and drop-on-unsolvable via the inverter's
> bracket `ValueError`; deterministic `cell_seed` = `base + split·100003 + cell·101`;
> `generate_pilot_samples` reusing `generate_cell_raw`, shared with the Sprint 4 build),
> and `eval_v2.py` (the three-TE probes: `slice_coupled_channels` → single-channel
> `(n,300,1)` at the fs-correct pulse-shape index; `measure_te_scat` = the ported R0
> realizable-gain (z-scored features) averaged over anchors×seeds → `frac_Phi`;
> `measure_te_raw` = carrier-band bandpass → **Hilbert-envelope demodulation** (the
> learned-filter-free analog of the scattering modulus) → Fourier decimation
> `5280→330→trim→300` → per-channel z-score → the same R0 ratio, with a `demodulate=False`
> control; `run_realizability_preflight` writing `realizability.json` and honouring
> `eval.realizability.fatal`; `sweep_render_knobs` writing `recovery.json`). Wired the
> `r0_realizability` stage + `--pilot`/`--full`/`--recover` flags into `run_pipeline_v2.py`
> (the stage defaults to the pilot grid; `--full` opts into the heavy full grid); added
> `eval.realizability.{n_anchors,n_seeds,pilot,recovery}` to the config. Tests:
> `test_build_dataset_v2.py` (8) + `test_te_preservation.py` (6) green; full v2 suite
> **75 passed** (was 61). `--solve-te 2.0 8` still reports `te_block=2.0003`, `SNR=0.1427`
> after the refactor. A high-effort code review (10 findings) was applied: fixed the
> `measure_te_raw` DC-inflated ridge (z-score) and the carrier-band demodulation (the AM
> coupling lives at `f_pulse`, not the VLF envelope band), the uncaught `--solve-te`
> bracket `ValueError`, the null-only-grid fatal-gate misfire, the full-grid CLI footgun,
> and the sweep abort-on-bad-`f_pulse`.
>
> **De-risk result (the point of Sprint 3).** At the **default** render knobs
> (`f_pulse=0.02`, `am_offset_ratio=4.0`, `omega=0.10`) the injected TE does **not** survive:
> pilot mean `frac_Phi(signal) ≈ 0.196`, every signal cell LOW, `headline_pass=False`
> (`results/G1_raw_baseline/realizability.json`). The `--recover` sweep
> (`results/G1_raw_baseline/recovery.json`) shows `f_pulse=0.02` is unrecoverable
> (max `frac_Phi≈0.59`, AM-margin<1 even at deep modulation), while **raising the carrier
> to 0.04–0.06 Hz** makes every setting valid (margin≥1, lag identifiable at
> `lag1≈0.97`). The picker targets `frac_Phi≈1` (faithful preservation, §10/§14.1 — a value
> ≫1 is estimator/φ-smoothing inflation, *not* created information, so "maximise `frac_Phi`"
> was corrected to "closest to 1 among valid"). **Chosen: `f_pulse=0.06`,
> `am_offset_ratio=4.0`, `omega=0.10` → `frac_Phi=1.122`** — a minimal single-knob change
> (only the carrier; modulation depth and contraction rhythm unchanged).
>
> **Confirmed at pilot scale** (`results/G1_raw_recovered/realizability.json`): with
> `f_pulse=0.06`, all **D≥8** cells PASS (`frac_Phi` 0.90–2.17, mean ≈ 1.5 over
> `lag∈{8,12}`, `headline_pass=True`), null cells give `te_scat≈0`. **Short-lag D=4 (16 s)
> cells remain under-preserved** (`frac_Phi≈0.39–0.70`): the 16 s lag is comparable to the
> scattering low-pass smoothing width, so a fixed `delay_max=D=4` probe cannot fully recover
> it. `TE_raw` (demodulated carrier-band envelope) is ≈0 on null cells (≈ −0.2, R0 noise) and
> **positive and tracking `TE_inj` on signal cells** (≈0.14→0.47 over `te_inj` 1→3 at
> `f_pulse=0.06`, D≥8); the `demodulate=False` control reads ≈0 even for strong cells,
> confirming the amplitude coupling is invisible to a linear probe without demodulation —
> i.e. exactly why the scattering modulus is needed.
>
> **Flagged for the Sprint 4 grid/config lock (S4-T01), NOT changed here (config defaults
> untouched):** set `raw.f_pulse: 0.06` (from 0.02); keep `am_offset_ratio=4.0`,
> `omega=0.10`; restrict the headline `lag_grid` to **D≥8** (e.g. `[8,12,20]`), moving D=4 to
> a robustness cell or widening the probe's `delay_max` by +1–2 steps for short lags. Expect
> `frac_Phi` generally ≥1 (inflated at low TE by R0 finite-sample noise); report calibration
> against both `TE_inj` and `TE_scat` so the inflation is visible (§14.1).

### Sprint 4: Full build, cache, and loaders  — DONE (2026-07-01)
- [x] S4-T01: Lock the full grid and finalize config
- [x] S4-T02a: build_split for one (cell, split) with provenance and per-cell frac_Phi
- [x] S4-T02b: Pool, shuffle, and write cache (deterministic)
- [x] S4-T02c: Resumable per-(cell, split) build
- [x] S4-T03: dataset_v2 / datamodule_v2 loaders and model-forward compat

> Sprint 4 delivered the model-ready dataset. **S4-T01 config lock** (`config_synth_v2.yaml`):
> `raw.f_pulse` 0.02 -> **0.06** (the Sprint-3 recovered carrier), `mix.lag_grid` [4,8,12,20] ->
> **[8,12,20]** (D=4 dropped: at a 16 s lag the source->target delay is comparable to the
> scattering low-pass width, so the coupling is under-preserved), `n_per_cell_{train,val,test}` ->
> **800/200/300** (a lighter first full build; scale up by re-running with larger counts),
> `experiment.tag` -> **`G1_raw_v1`** (keeps the Sprint-3 `G1_raw_baseline` de-risk marker
> untouched). `am_offset_ratio=4.0` / `omega=0.10` unchanged. New `test_config_v2.test_full_grid_locked`
> asserts the lock. Two Sprint-1 raw-generator tests that hard-coded the old 0.02 carrier were
> updated to follow the locked `f_pulse` (`test_am_separation_from_config` now expects
> `adequate=True`, `margin_peak=1.565`; `_env_xcorr_at_lag` bands `[0.5 f_pulse, 1.6 f_pulse]`).
>
> **S4-T02a/b/c build** (`build_dataset_v2.py`): a **three-stage, resumable, deterministic** build
> reconciling pooled normalisation with per-(cell,split) resumability. Stage 1 `build_split_parts`
> generates raw (seeded by `cell_seed`) + scatters each (cell,split) to an **un-normalised**
> channels-first part `_parts/<split>_cell<ID>.npz` (skipped if present -> crash-safe resume);
> Stage 2 `fit_pool_stats` accumulates per-channel mean/std over the **train** parts incrementally
> (O(C) memory), persisting `norm_stats.npz` -- fitting the normaliser **once on the pooled train
> split** (not per cell) so the same physical scattering value maps to the same z-score across
> cells (the model needs a consistent ch-0 baseline / absolute scale; TE is z-score-invariant so
> `frac_Phi` is unaffected); Stage 3 `assemble_split` normalises each part with the pooled stats,
> runs the `measure_te_scat` probe at build N (non-fatal), stamps §17 provenance, pools all cells,
> and applies one shared row-aligned permutation seeded from `seeds.shuffle`. `write_cache_v2`
> writes the four fields + `weight` + `true_lag_tt` + the six `sample_*` provenance arrays
> (`sample_te_true`, `sample_te_scat`, `sample_frac_phi`, `sample_delay`, `sample_cell_id`,
> `sample_held_out` -- **no** `sample_M`/`delay_min`/`delay_max`/`band_id`) plus `meta.json` (pooled
> `te_true`, `true_lag_band` union, `channel_map`, `coupled_channel`, per-cell manifest with
> `te_scat_measured`/`frac_phi`, `raw`/`scattering`/`seeds` blocks). All writes are atomic
> (temp + `os.replace`).
>
> **S4-T03 loaders** (`dataset_v2.py`, `datamodule_v2.py`, copied/adapted from `synthetic/`):
> `SyntheticTEDatasetV2` memory-maps the four `(N,300,C)` fields + `weight`/`true_lag_tt` and
> eager-loads the O(N) provenance; `__getitem__` exposes `te_true`/`te_scat`/`frac_phi`/`delay`
> (= `sample_delay`, the fixed lag D for the clean-window floor)/`cell_id`/`held_out`/`true_lag_tt`
> and drops every v1 grouping key. `build_u_stream` -> `(B,300,101)`; `SyntheticTEDataModuleV2`
> resolves `data_dir/benchmark/tag` and serves train/val/test. `--stage build [--pilot|--full]`
> wired into `run_pipeline_v2.py`.
>
> **Tests: 83 passed** (was 75; +8 S4 cases in `test_build_dataset_v2.py`: provenance/schema,
> meta manifest, shuffle row-alignment, determinism, resume, loader field-mapping, and a real
> `model.forward` compat check). The model import needs the repo root ahead of the sibling
> `model/vae_teb_prediction/utils` shadow package on `sys.path` (documented in the forward test).
>
> **Full build run** (locked grid, RTX 4080) -> `data/G1_raw/G1_raw_v1/{train,val,test}.npz`
> (12000/3000/4500 samples), `meta.json`, `norm_stats.npz` (~4.4 GB). Verified end-to-end: a
> `DataLoader` batch runs `SeqVaeLagAttnV1.forward` -> `kld_per_t (B,300)`, `attn_weights
> (B,300,4,91)`. **De-risk confirmed at full scale**: coupled channel = 20 (0.0554 Hz), pooled
> `te_true=1.30` nats, `te_block_realised` lands on target (0.499/0.999/2.00/2.99); all **12 signal
> cells PASS** the `frac_threshold=0.7` gate (`frac_Phi` 0.99-2.47, **mean ~1.80**), null cells give
> `te_scat ~= -0.15` (R0 noise, `frac_Phi=null`), none dropped. As flagged in §10/§14.1, `frac_Phi`
> runs **> 1** (largest at low target_te, e.g. cell 4 te=0.5/D=12 -> 2.47; closest to 1 at high te,
> cell 12 te=3/D=8 -> 0.99): this is R0 finite-sample / phi-smoothing inflation on the single coupled
> sub-channel, **not created information** -- Sprint 6 therefore calibrates K-bar against **both**
> `TE_inj` and `TE_scat` so the inflation is visible.

### Sprint 5: Training  — CODE + PILOT SMOKE DONE (2026-07-01); headline run (S5-T04) deferred to user
- [x] S5-T01: pl_module_v2, trainer, ckpt round-trip, and loss switch
- [x] S5-T02: run_pipeline_v2 driver stages
- [x] S5-T03: Optional beta-selection stage
- [~] S5-T04: Headline training run — training path implemented + pilot-smoked on the full cache;
      the full ~100-epoch headline run is left for the user to launch (`--stage train`, no `--pilot`).

> Sprint 5 delivered the training layer. **S5-T01** (`pl_module_v2.py`):
> `SyntheticSeqVaeLagAttnV2Pl` — a `train.pl_model_base.LightningModelBase` subclass ported
> from `synthetic/pl_module_synth.py`, reading `fhr_st`/`fhr_ph` + `build_u_stream` (up_st|up_ph
> → 101) from each batch, calling the **unchanged** `SeqVaeLagAttnV1.compute_loss` with the
> calibration knobs (`kld_beta`, `lambda_full`, `lambda_base`, `likelihood`, `sigma_obs`,
> `free_bits`, `detach_baseline_in_full`), logging `kld_nats = kld_loss·d_z` (the $\bar K$
> surrogate scale). Copies the two load-bearing v1 patterns: **eager** (`self.model =
> base_model`, no `torch.compile`) and the cross-rank-`MAX`-synced **spike-skip** step; adds the
> optional linear-warmup→`MultiStepLR` scheduler. Checkpoints follow the project convention —
> `save_checkpoint_v2` writes the v1-format dict (`model_state_dict` + `model_kwargs` + meta),
> reloaded by `train.graph_models_utils.load_checkpoint_strict`.
>
> **S5-T02** folded the v1 `train_ddp.py` driver into `pl_module_v2.train_v2` (single-GPU default,
> DDP when `devices>1`; `build_model` on CPU → wrap → `SyntheticTEDataModuleV2` → `Trainer`
> (`LearningRateMonitor`, `ModelCheckpoint(monitor=val/total_loss)`, `CSVLogger`) → `fit` →
> non-fatal post-fit `fit_latent_stats` → export `final.ckpt`/`best.ckpt` → render loss curves)
> and wired `--stage train` (+`--pilot`, `--epochs`, `--devices`) into `run_pipeline_v2.py`.
> `visualize_v2.plot_loss_curves_html` reads the Lightning `metrics.csv` (resolving the
> `_step`/`_epoch` fork) and renders a single interactive Plotly `training_curves.html` with one
> distinctly-coloured trace per logged metric (train solid / val dashed); it is the only
> training-curve output (no static PDF/PNG twin) and is refreshed live by `LossPlotHtmlCallback`.
> **S5-T03** `pl_module_v2.beta_select`
> (`--stage beta_select`, default OFF via `beta_select.enabled`, force-run when invoked): short
> per-β runs (`skip_checkpoint`) → `beta_select.json` picking the highest-`kld_nats` (least
> collapsed) β. Config gained `train.pilot_*` / `train.devices`, `beta_select.*`, and
> `loss.detach_baseline_in_full`.
>
> **Tests: full v2 suite 90 passed** (+6 new in `tests/test_train_v2.py` on a tiny
> in-test fixture cache + tiny model, CPU-forced: one-epoch module fit logging `kld_nats`;
> `save_checkpoint_v2`↔`load_checkpoint_strict` forward parity (`mu_post`/`kld_per_t`);
> `mse`/`gaussian_nll` both finite + `free_bits` KL-floor clamp; `train_v2` pilot smoke writes a
> checkpoint + loss curve; `beta_select` no-op (disabled) and enumeration (forced)). `--solve-te
> 2.0 8` still reports `te_block=2.0003`, `SNR=0.1427`.
>
> **S5-T04 pilot smoke** (`--stage train --pilot`, RTX 4080, `train.pilot_epochs=3` ×
> `pilot_limit_train_batches=4` × `pilot_batch_size=16`): ran end-to-end on the full
> `G1_raw_v1` cache — train total_loss 2.62→1.56, val 1.72→1.52, `kld_nats≈4.47`;
> `fit_latent_stats` aggregated 3.24M samples; `best.ckpt` re-exported from the Lightning best via
> `load_checkpoint_strict`; wrote `results/G1_raw_v1/{final,best}.ckpt`, `logs/version_0/metrics.csv`,
> and `figures/training_curves.html`. **Note:** the pilot writes into the headline tag dir
> (`results/G1_raw_v1/`); the full headline run overwrites those with the real (non-pilot)
> checkpoint. **Deferred to the user (S5-T04):** launch the full run with
> `run_pipeline_v2.py --stage train` (optionally `--devices N` for DDP on the prod box).

### Sprint 6: Evaluation gates  — DONE (2026-07-01)
- [x] S6-T01: Per-sample K-bar collection over the clean window
- [x] S6-T02: gamma-calibration against TE_inj and TE_scat
- [x] S6-T03: Lag recovery
- [x] S6-T04: Null controls and metrics report
- [x] S6-T05: End-to-end integration smoke

> Sprint 6 delivered the evaluation gates on a trained checkpoint, all in `eval_v2.py`
> (extended) + `run_pipeline_v2.py` (`eval`/`report` stages wired) + a new
> `tests/test_eval_v2.py`; the **model/loss/trainer are unchanged** (used only through the
> `forward` contract `kld_per_t` / `te_lag_map` / `attn_weights` / `warmup_mask`).
>
> **S6-T01** `collect_per_sample_kbar` (ported from v1 `mixed_eval`, stripped of the
> `M`/`band_id` axes and the `per_dim_kl_by_M` / `kld_time_by_band` structures): one eval
> pass, per-sample $\bar K$ = clean-window mean of `kld_per_t` over
> `[max(w, D-1), T-H)` (`_clean_window_mean`, floor from the per-sample `sample_delay`),
> grouped by `sample_cell_id`, carrying `te_inj`/`te_scat`/`frac_phi`/`held_out`. Null
> controls re-forward with a corrupted source (`_corrupt_source`: `shuffle` = cross-batch
> source permutation, `reverse` = time-flip). In the **same pass** it accumulates each
> cell's clean-window mean of `te_lag_map` into a lag profile (so S6-T03 needs no second
> forward). **S6-T02** `fit_calibration_slope` (OLS $\gamma=\mathrm{Cov}/\mathrm{Var}$,
> $\alpha$, $R^2$) + `fit_calibration` fitting $\bar K=\alpha+\gamma\,\mathrm{TE}$ vs
> **both** `TE_inj` and `TE_scat` with a Spearman-sign monotonicity flag (nulls kept as the
> near-origin anchor). **S6-T03** `score_lag_profile` (pure: LagMass in
> $\mathcal L^\star=\{\max(0,D-H),\dots,D-1\}$, argmax lag, $\pm$`lag_tolerance_steps`
> allowance) + `recover_lags` aggregating over signal cells. **S6-T04** `null_ratios`
> ($\bar K_{\mathrm{null}}/\bar K_{\mathrm{signal}}$ per cell + signal-mean), a model-free
> `_null_probe` (null-cell `sample_te_scat`; folds a sibling `realizability.json` when
> present), and `run_eval` assembling `metrics.json` (calibration / lag_recovery /
> null_controls / null_probe / frac_phi / per-cell table) via `_jsonable`. The `eval` and
> minimal `report` (`write_report` -> `report.md`) stages are wired into
> `run_pipeline_v2.py` with `--ckpt` / `--split` (checkpoint auto-discovery best->final;
> loader fallback test->val->train). The minimal report is a **seam** — Sprint 7 (S7-T05
> `final_report_v2`) supersedes it. **S6-T05** the `-k e2e` integration test composes
> build -> r0_realizability -> train (1 epoch, CPU) -> eval -> report on a tiny **real**
> grid (`grid_override`/`n_override`, shared adapter) and asserts every artifact.
>
> **Tests: full v2 suite 99 passed** (was 90; +9 in `tests/test_eval_v2.py` — a stub-model
> + fake-loader keep the calibration/lag/null gates fast, a tiny fixture cache + tiny
> checkpoint exercises `run_eval`, and the `slow`-marked `-k e2e` runs the real transform
> in ~25 s). Registered the `slow` marker in `tests/conftest.py` (deselect with
> `-m "not slow"`).
>
> **Pilot-checkpoint smoke** (`--stage eval` then `--stage report`, real full
> `G1_raw_v1` test split, 4500 samples / 15 cells, RTX 4080): the machinery runs end to
> end and writes `results/G1_raw_v1/{metrics.json,report.md}`. As expected for the
> **3-epoch pilot** checkpoint (undertrained, S5-T04 headline run deferred), the numbers
> are not yet meaningful — `gamma_inj≈-0.003`, `gamma_scat≈-0.002`, `mean_LagMass≈0.17`
> (<0.8), `null_ratio≈1.0` for both controls (the model has not yet learned to use the
> source), while the build-stamped `frac_Phi` reads back correctly (signal mean ≈1.72,
> [1.04, 2.48]) and the null-cell dressing-only `TE_scat≈-0.30` (≈0). **The real gate
> verdicts (gamma_scat -> 1, LagMass -> 1, null_ratio -> 0) are produced after the user's
> full `--stage train` headline run (S5-T04).**

### Sprint 7: Journal figures, standard test plots, pulse_train, report, and README  — DONE (2026-07-01)
- [x] S7-T01: plot_style_v2 and the annotated raw-signal figure
- [x] S7-T02: Raw + scattering paired preview
- [x] S7-T03: Latent/AM decomposition and diagnostics figures
- [x] S7-T04: pulse_train render variant and its frac_Phi
- [x] S7-T05: Final report assembly
- [x] S7-T06: Standard testing metadata bridge for TE-aware sample plots
- [x] S7-T07: run_tests.py bridge smoke for synthetic v2
- [x] S7-T08: TE-aware aggregate plot upgrades
- [x] S7-T09: Module README / usage doc
- [x] S7-T10: Data-generation story figures (band recipe, TE authoring, latent coupling, AM separation)

> Sprint 7 delivered the journal figure set, the TE-aware standard-testing bridge, the
> `pulse_train` render variant, the final report, and the README. **S7-T01** ported
> `synthetic/plot_style.py` into a standalone `plot_style_v2.py` (house style: serif, thin
> black spines, the `stacked_figure` + `attach_colorbar` colorbar-gutter, `save_figure`,
> `add_caption`; the v1 `M`-dilution maps dropped) and refactored `visualize_v2.py` onto it
> (palette aliases, `style_axes`, 600-dpi PNG); the raw preview annotates the coupled
> decel/contraction bands. **S7-T02** `plot_raw_scatter_paired` (raw FHR/UP + their scattering
> heatmaps stacked, the fs-correct coupled channel highlighted, the decimated latent `[15:315]`
> overlaid). **S7-T03** extended `generate_cell_raw`'s returned `latents` dict (additive:
> `A_u/A_y/carrier_u/carrier_y/u_c/y_d`) and added `plot_latent_am_decomposition` (latent →
> envelope → carrier → rendered band) + `plot_diagnostics_panel` (2×2: calibration vs both TEs,
> frac_Φ, LagMass, null ratio). **S7-T04** implemented the `pulse_train` renderer
> (`make_pulse_train`: one-sided raised-cosine event train at `f_pulse`, so a positive envelope
> renders clinically-realistic upward contractions / downward decelerations); `render_mode` now
> validates to `{am_carrier, pulse_train}` (the old empty-string-`NotImplementedError` test became
> an unknown-`render_mode` `ValueError` test); `config.raw.pulse_train.{rate_hz, duty}` added; its
> `frac_Φ` is measured via the existing `measure_te_scat` probe (expected lower, documented).
> **S7-T06** made the standard testing sample plots TE-aware, **additively and presence-guarded**
> so real-data (HDF5) runs are unchanged: `dataset_v2` exposes an optional `te_raw`;
> `collectors.py` gained guarded `_extract_scalar_field` / `_extract_int_field` / `_extract_delay`
> / `_extract_array_field` and attaches `te_true/te_scat/te_raw/frac_phi/sample_delay/cell_id/
> true_lag_tt/true_lag_band` to each sample dict (absent → `None` on real data);
> `analyses/qualitative.py` threads them into the plotter and adds the TE columns to
> `sample_metrics.csv` only when a `te_true` is present; `plot_single_samples.py`'s
> `plot_sample_lag_attn_diagnostic` gained the optional TE/lag params, plain-text title bits, and a
> true-lag-band overlay on the attention / TE-lag panels (mirroring the sibling
> `plot_sample_lag_attention`). **S7-T07** added the `test_plots` stage + `--analysis-samples`;
> `run_test_plots` builds a v2 `DataLoader` and runs the standard `run_sample_diagnostics` /
> `run_kld_lag_diagnostics` on a `TestRunner` constructed **directly** from the `vae_teb_lag_attn_v1`
> model (via the checkpoint's `model_kwargs`) — NOT through `run_full_test_pipeline`, because
> `testing/base.py` is pinned to the legacy `vae_teb_lag_attn_old` architecture and cannot align a
> v1 checkpoint; the shared pipeline is left untouched. **S7-T08** added the population-level
> aggregate figures (`plot_calibration_by_lag`, `plot_frac_phi_distribution`,
> `plot_lag_mass_summary` — grouped by lag, with counts / s.e.m. / thresholds). The TE-aware
> aggregates live in `visualize_v2` (wired + tested); `testing/visualizers.py` was intentionally
> left unchanged because the v2 bridge drives `run_sample_diagnostics` directly and never reaches
> those aggregate plots (an initial `resolve_group_column` hook was dropped in review as dead code).
> **S7-T05** `final_report_v2` collates `meta.json` / `metrics.json`
> / `realizability.json` / the figure gallery / the standard-testing `sample_metrics.csv` +
> representative sample into `report.md` (+ a rendered headline diagnostics figure), degrading
> gracefully on any missing artifact; the `report` stage now calls it (the minimal
> `eval_v2.write_report` remains an internal fallback). **S7-T09** wrote `README.md` (every stage +
> runnable command + config knobs + outputs, linking the two design docs).
>
> **Tests:** `tests/test_visualize_v2.py` added (raw / paired / decomp / diag / metadata_bridge /
> sample_te / aggregate / report, plus a `slow`-marked `test_plots` bridge smoke that
> builds a tiny cache + checkpoint and asserts TE-annotated PDFs + `sample_metrics.csv` columns);
> pulse cases added to `test_raw_generators.py` / `test_te_preservation.py`. New standalone-test
> path guard: the `testing` stack pulls the *old* model's bare `from utils.custom_logger import …`,
> which must resolve to the **repo-root** `utils` (the sibling `model/vae_teb_prediction/utils` has
> no `custom_logger` and pytest keeps re-inserting its dir); `_ensure_repo_root_utils()` fixes this
> at test runtime (the real `--stage test_plots` path runs with the repo root on `sys.path`, so it
> is unaffected).
>
> **Deferred (unchanged from Sprint 5/6):** the figures and report render against the *pilot*
> checkpoint until the user launches the full **S5-T04** headline run (`--stage train`, no
> `--pilot`); re-run `--stage {eval, test_plots, report}` afterwards for the real gate numbers.

---

This document is the living roadmap and guidebook for `synthetic_v2`. As each task is completed, check
its box above. Keep this file as the single source of truth for the build - update status here rather
than tracking progress elsewhere. The companion `SYNTHETIC_V2_RAW_TE_PIPELINE_EXPLAINED.md` remains the
authoritative math/design reference; this file governs the build order and the per-task validations.
