# `model_experiment/` — v2 synthetic-TE validation

This directory holds the synthetic-data validation harness for
`SeqVaeLagAttnV1`. The goal is to test whether the model's latent KL

$$
K_t
\;=\;
\mathrm{KL}\!\left(\,
q_\phi(z_t \mid Y_{\le t},\,U_{\le t})
\;\big\|\;
p_\psi(z_t \mid Y_{\le t})
\,\right)
$$

behaves as a transfer-entropy surrogate: it should be near zero under
null controls, monotone in the true block transfer entropy
$\mathrm{TE}^{(H)}_{U\to Y}$ (with $H = 30$), linearly calibrated to true
TE in nats under a suitable $\beta$ / likelihood, and localised on the
true source-lag band $\mathcal{L}^\star$.

---

## Why v2

The v1 attempt failed for one reason: the source process was i.i.d.
white noise, so the 30-step future target block $Y^{+}_{t,H}$ was
effectively unpredictable from $(Y_{\le t}, U_{\le t})$. The model's
residual branch $\Delta\hat Y^{src}$ stayed at zero-init, the latent
collapsed ($K_t \to 0$), and no monotonicity, calibration, or
lag-recovery claim could be tested. v2 fixes **the data, not the
model**: low-frequency smooth processes (AR(2) oscillators, smooth AR(1)
drivers, slow categorical regimes) with closed-form block TE in nats.
See `model_validation_v2_plan.md §0` for the full v1 verdict.

---

## Read order

1. **[`model_validation_v2.md`](model_validation_v2.md)** — theory:
   the predictable-target constraint, the new benchmarks, the
   calibration / lag-recovery / directionality protocol.
2. **[`model_validation_v2_plan.md`](model_validation_v2_plan.md)** —
   sprint tracker with decisions V2-D1…D9 and the atomic task list
   (the final progress log lists every artifact produced).
3. **[`../new_architecture.md`](../new_architecture.md)** —
   `SeqVaeLagAttnV1` specification (inputs, outputs, shapes,
   hyperparameters, latent stabilisers).
4. **[`../vae_teb_lag_attn_v1.py`](../vae_teb_lag_attn_v1.py)** — the
   model under test. The TE surrogate is `out["kld_per_t"]`; the lag
   attribution map is `out["te_lag_map"]`. The Sprint-5 calibration
   path is the `compute_loss(..., likelihood='gaussian_nll',
   sigma_obs='learned')` switch.
5. *(legacy)* `synthetic_te_validation_plan.md` and
   `model_validation.md` — v1 theory and tracker, retained as
   historical context. **Not loaded by any v2 runner.**

---

## Project conventions

From `CLAUDE.md` at the repo root:

- Google-style docstrings on every public function/class.
- LaTeX math notation in docstrings and Markdown: `$ … $` for inline,
  `$$ … $$` for displayed equations.
- Checkpoint loading via `train/graph_models_utils.py:load_checkpoint_strict`.
- Python env: `.venv` at the repo root
  (torch 2.7.1+cu128, CUDA 12.8, RTX 4080).
- Every runner under `synthetic/` exposes **both** a CLI and an
  edit-and-run `RUN_CONFIG` dict (V2-D8); the dispatch picks one based
  on whether any `--flag` was passed.

---

## Benchmarks at a glance

All benchmarks emit the model's native input shapes (V2-D1) — no model
surgery is needed. They differ only in the data-generating process, the
form of the ground-truth TE, and the meaning of the swept knob.

| Benchmark | DGP | Knob | Ground truth TE | True lag band $\mathcal{L}^\star$ | Purpose |
|---|---|---|---|---|---|
| **G1** | Multi-channel AR(2) oscillator → AR(1) target with delay $D{=}60$ | $B_y$ (coupling magnitude) | MC determinant ratio via `te_block_state_space_gaussian` | $\{30, \ldots, 59\}$ | **Primary headline** (V2-D3): closed-form TE in nats + predictable future + clean lag band |
| **G1-rev** | G1 with `reverse_roles=True` (anti-causal slot pairing) | — | $0$ by construction; `true_lag_band = []` | — | **Directionality control**: paired with G1 in `directionality.py` |
| **G1_twoband** | G1 with two delays $D_1{=}35,\ D_2{=}85$ | — | MC, two-band | $\{5..34\} \cup \{55..84\}$ | Diagnostic: tests whether the lag-attention can resolve **two** non-contiguous bands |
| **G2** | Scalar AR(1) source → AR(1) ARX target at delay $D{=}60$ | $c$ (ARX coupling) | **Closed form** `te_block_arx_gaussian` (sub-second; no MC) | $\{30, \ldots, 59\}$ | **Sanity / debug benchmark**; also the source for null controls |
| **G2_wrong_delay** | G2 with $D = 200 \gg L_{\max} + H = 120$ | — | Closed form (large but unreachable) | unreachable | Null control: model's max lag cannot reach the true source-target alignment (INFO-only row, see caveat) |
| **G2_zero_coupling** | G2 with $c = 0$ | — | Exactly $0$ | — | Null control: source and target are independent — any well-behaved surrogate must collapse $\bar K \to 0$ |
| **G3** | $K{=}10$-class slow regime switch with smooth phase-continuous oscillator templates; source one-hot leaks the regime at lead $\delta = 30$ | $p_{\mathrm{switch}}$ | $M \cdot H \cdot \tilde{H}(p, K)$ via `te_categorical_switch_block` (closed-form discrete entropy; mirrors the TEB rotating-MNIST setup) | $\{0, \ldots, 29\}$ | Categorical / nonlinear benchmark |

`G4` (switched sinusoid) is **deferred** per Sprint 7 — only landed if
G1+G2+G3 pass calibration. No code, no cache, no tests.

The DGP-specific knob is **continuous and crosses 0** in every sweep,
so each grid spans the null ($\mathrm{TE} = 0$) and the v2 operating
band ($\mathrm{TE}/\!H \in [0.05, 0.3]$ nats per step, V2-D6).

---

## File layout

```
model_experiment/
├── README_v2.md                  # THIS DOCUMENT
├── model_validation_v2.md        # v2 theory  ← read first
├── model_validation_v2_plan.md   # sprint tracker
├── synthetic/                    # the entire harness lives here
│   ├── config_synth.yaml         # single source of truth (benchmarks / model / loss / sweeps)
│   ├── generators.py             # G1 / G2 / G3 DGPs
│   ├── analytic_te.py            # closed-form / MC ground-truth TE
│   ├── dataset.py                # SyntheticTEDataset + collate
│   ├── build_dataset.py          # one-shot cache builder (CLI / RUN_CONFIG)
│   ├── visualize.py              # per-benchmark cache previews
│   ├── train_minimal.py          # single-GPU PyTorch loop (no Lightning)
│   ├── gpu_pool.py               # multi-GPU scheduler (one model per slot)
│   ├── evaluate_te.py            # K̄ vs TE, Metrics 1–4
│   ├── lag_recovery.py           # sliding-window LOLO (V2-D5)
│   ├── beta_sweep.py             # β rate-distortion + HP probes
│   ├── calibration.py            # Sprint-5/7.1 calibration γ → 1
│   ├── directionality.py         # G1 ↔ G1-rev paired runner
│   ├── null_controls.py          # wrong-delay + zero-coupling re-eval
│   ├── final_report.py           # collates everything into report.json + headline.pdf
│   ├── plot_style.py             # journal-style mpl rc / palette
│   ├── plot_training_curves.py   # loss-curve renderer for train_minimal
│   └── test_*.py                 # pytest suite (~133 tests; <20 min on CPU)
├── data/                         # caches (gitignored; regenerable)
│   └── <benchmark>/<tag>/        # train.npz / val.npz / test.npz / meta.json / preview.pdf
└── results/                      # per-run artifacts (checkpoints, CSV, figures, JSON)
    ├── <benchmark>/<run_tag>/    # final.ckpt + best.ckpt + metrics.csv + training_curves.pdf
    ├── <benchmark>/eval_te/      # K̄ vs TE summary + plots + metrics.json
    ├── <benchmark>/lag_recovery/ # LOLO + attention lag-mass + plots
    ├── <benchmark>/beta_sweep/   # rate-distortion + selected β
    ├── <benchmark>/calibration/  # γ → 1 panel + calibration.json
    ├── <benchmark>/null_controls/# wrong-delay + zero-coupling rows
    ├── directionality/           # G1 ↔ G1-rev comparison
    └── final_report/             # report.json + report_table.csv + headline.pdf
```

`paths.data_dir` and `paths.results_dir` in `config_synth.yaml` are
resolved relative to `model_experiment/`.

---

## Pipeline at a glance

```
config_synth.yaml ── benchmarks: { G1, G1-rev, G1_twoband, G2, G2_wrong_delay, G2_zero_coupling, G3 }
        │
        │ experiment.benchmark + experiment.tag
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — DATA                                                       │
│  build_dataset.py  ──►  data/<benchmark>/<tag>/{train,val,test}.npz   │
│                       + meta.json (te_true, true_lag_band, …)         │
│                       + preview.pdf (per-benchmark panels)            │
│                                                                       │
│  visualize.py      ──►  re-render preview / gallery without rebuild   │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — TRAINING                                                   │
│  train_minimal.py  ──►  results/<benchmark>/<tag>/final.ckpt          │
│                       + best.ckpt + metrics.csv + training_curves.pdf │
│                       + config_used.yaml                              │
│                                                                       │
│  gpu_pool.py       ──►  task-parallel training (sweep / β / HP /      │
│                         directionality) on multi-GPU boxes;           │
│                         one model per slot, no DDP                    │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  STAGE 3 — EVALUATION (no training; reads checkpoints)                │
│  evaluate_te.py    ──►  K̄ vs TE / sweep / shuffle+reverse null       │
│  lag_recovery.py   ──►  sliding-window LOLO + attention lag-mass      │
│  beta_sweep.py     ──►  rate-distortion + HP probes                   │
│  calibration.py    ──►  γ → 1 with Gaussian-NLL likelihood            │
│  directionality.py ──►  G1 ↔ G1-rev paired comparison                 │
│  null_controls.py  ──►  re-eval one G2 checkpoint on wrong-delay +   │
│                         zero-coupling caches                          │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  STAGE 4 — REPORT                                                     │
│  final_report.py   ──►  report.json + report_table.csv                │
│                       + headline.pdf (4-panel summary)                │
│                       + claim_tier ∈ {strong, moderate, weak,         │
│                                       deferred}                       │
└───────────────────────────────────────────────────────────────────────┘
```

Every stage tolerates missing upstream artifacts: `final_report.py`
emits DEFERRED rows and "not run" headline panels rather than crashing.

---

## Entry points and run order

All commands assume the repo-root `.venv` (`torch 2.7.1+cu128`,
RTX 4080) and a CWD anywhere — the runners resolve config / data /
results paths relative to `model_experiment/`. Substitute the `python`
path for `.venv\Scripts\python.exe` on Windows.

### 0. Smoke test (one-time)

Verify the model + env are healthy:

```powershell
python -m model.vae_teb_prediction.model.vae_teb_lag_attn_v1
pytest model/vae_teb_prediction/model/model_experiment/synthetic -q
```

### 1. Build the dataset cache

Generate once, persist, reuse (V2-D2). The active benchmark is
selected by `experiment.benchmark` in `config_synth.yaml`; the cache
directory is `data/<benchmark>/<tag>/`.

```powershell
# CLI mode
python -m model.vae_teb_prediction.model.model_experiment.synthetic.build_dataset `
    --tag G1_baseline                     # writes data/G1/G1_baseline/
# Overrides exposed by build_dataset: --tag --easy --m --force

# Edit-and-run mode (RUN_CONFIG dict at the bottom of build_dataset.py)
python -m model.vae_teb_prediction.model.model_experiment.synthetic.build_dataset
```

Output: `train.npz`, `val.npz`, `test.npz`, `meta.json`,
`preview.pdf`. `meta.json` carries the analytic `te_true`,
`true_lag_band`, the informative-channel map, and the generator seed.

### 2. Train one model

```powershell
python -m model.vae_teb_prediction.model.model_experiment.synthetic.train_minimal `
    --data-tag G1_baseline `
    --epochs 100
# Other overrides: --beta --batch-size --grad-checkpoint/--no-grad-checkpoint
#                  --lr --m --device --seed --run-tag
```

Output under `results/<benchmark>/<run_tag>/`:

- `final.ckpt` — last-epoch state + the optimiser + the resolved
  `loss_settings` (likelihood / sigma_obs / free_bits) so re-eval is
  reproducible.
- `best.ckpt` — lowest-val-loss state.
- `metrics.csv` — per-epoch losses + sat-fractions + `mean_logvar_*`
  collapse diagnostics + grad norm.
- `training_curves.{pdf,png}` — train / val curves auto-refreshed every
  `plotting.plot_every` epochs.
- `config_used.yaml` — frozen, fully-resolved config.

For the G2 calibration-debug slice the same command works with
`experiment.benchmark: G2` set in the YAML.

### 3. Run the per-cell sweep on one GPU (optional)

To get the headline `kbar_vs_te` / `kbar_vs_knob` / `predgap_vs_kbar`
plots, you need **multiple** checkpoints (one per sweep cell). On a
single GPU the loop is sequential:

```powershell
# Per-cell training is the inner loop; evaluate_te orchestrates it.
python -m model.vae_teb_prediction.model.model_experiment.synthetic.evaluate_te `
    --mode sweep --build-missing --train-missing
```

On a multi-GPU box use `gpu_pool` for task parallelism (one model per
slot, no DDP — see §4 below).

### 4. Scale up: multi-GPU training (recommended for sweeps)

`gpu_pool.py` enumerates cells, builds every needed cache **once**
(serial), then runs a worker subprocess per GPU slot via
`CUDA_VISIBLE_DEVICES`. Cells whose `final.ckpt` already exists are
skipped unless `--force` is passed, so an interrupted run resumes
cleanly.

```powershell
# 1) Train the active benchmark's full sweep
python -m model.vae_teb_prediction.model.model_experiment.synthetic.gpu_pool `
    --mode a_sweep --gpus 0,1,2,3,4,5,6 --benchmark G1

# 2) Train the β rate-distortion sweep at the fixed sweep cell
python -m model.vae_teb_prediction.model.model_experiment.synthetic.gpu_pool `
    --mode beta --gpus 0,1,2,3,4,5,6

# 3) Train a hyper-parameter probe
python -m model.vae_teb_prediction.model.model_experiment.synthetic.gpu_pool `
    --mode hp --axis lambda_base --gpus 0,1,2,3

# 4) Train the directionality pair (G1 and G1-rev)
python -m model.vae_teb_prediction.model.model_experiment.synthetic.gpu_pool `
    --mode directionality --gpus 0,1

# 5) Train the full calibration matrix (β × TE-target, ~27 cells)
python -m model.vae_teb_prediction.model.model_experiment.synthetic.gpu_pool `
    --mode calibration --benchmark G1 --gpus 0,1,2,3,4,5,6,7
# Same for G2: --benchmark G2 (closed-form ARX inverter, no MC budget).
# After this finishes, fit the slope without retraining:
python -m model.vae_teb_prediction.model.model_experiment.synthetic.calibration `
    --benchmark G1 --no-build-missing --no-train-missing
```

Repeating an index in `--gpus` (e.g. `0,0,1,1`) packs two concurrent
models onto that GPU. `gpu_pool` is **train-only**; downstream
evaluators just score the checkpoints it produces.

#### 8-GPU recipe (A100/H100/L40-class, ≥ 40 GB VRAM)

To fully saturate an 8-GPU machine, either set `runtime.gpus:
[0,1,2,3,4,5,6,7]` once in `config_synth.yaml` or pass `--gpus
0,1,2,3,4,5,6,7` per invocation. Five `gpu_pool` modes are now
parallelisable:

| Mode | Cells | Waves on 8 GPUs |
|---|---|---|
| `a_sweep` (per benchmark) | 10–15 | 2 |
| `beta` | 9 | 2 |
| `hp` (per axis) | 2–3 | 1 |
| `directionality` | 2 | 1 (uses 2/8 GPUs) |
| **`calibration`** (new) | **27** (3 TE × 9 β) | **4** |

For bigger-memory GPUs, edit `config_synth.yaml` to disable activation
checkpointing and grow the batch — both buy real speedups that compound
across the 8 slots:

```yaml
runtime:
  gpus: [0, 1, 2, 3, 4, 5, 6, 7]
model:
  attention_grad_checkpoint: false   # ~10-15% per-step speedup
optim:
  batch_size: 128                    # was 32 for the 12 GB laptop
dataset:
  num_workers: 2                     # overlap host->device with compute
  pin_memory: true                   # only useful with CUDA
  persistent_workers: true           # keep workers across epochs
```

Calibration is now first-class in the pool. The two-step workflow on an
8-GPU box is:

```powershell
# 1) Train every (β, TE-point) cell in parallel (~4 waves)
python -m model.vae_teb_prediction.model.model_experiment.synthetic.gpu_pool `
    --mode calibration --benchmark G1 --gpus 0,1,2,3,4,5,6,7 --build-missing

# 2) Read the checkpoints from disk, fit the slope, render the plot
python -m model.vae_teb_prediction.model.model_experiment.synthetic.calibration `
    --benchmark G1 --no-build-missing --no-train-missing
```

Step 2 is single-GPU (it's a serial read + OLS fit + plot, finishes in
minutes). Repeat with `--benchmark G2` for the ARX sanity check —
`gpu_pool` dispatches the benchmark via `calibration._CALIBRATION_BUILDERS`
so no YAML edit is needed.

### 5. Evaluate

Each evaluator is read-only — it loads the relevant checkpoint(s) and
writes plots + JSON. Order is **independent**, but `final_report.py`
needs the others' JSONs to fill its panels.

```powershell
# (a) K̄ vs TE Metrics 1–4 (single or sweep mode)
python -m model.vae_teb_prediction.model.model_experiment.synthetic.evaluate_te `
    --mode single --run-tag G1_baseline           # one checkpoint
python -m model.vae_teb_prediction.model.model_experiment.synthetic.evaluate_te `
    --mode sweep                                  # all sweep cells
# Output: results/<benchmark>/eval_te/{summary.csv, metrics.json,
#         kbar_vs_te.pdf, kbar_vs_<knob>.pdf, predgap_vs_kbar.pdf,
#         diagnostics_*.pdf, null_control.pdf, per_dim_kl_heatmap.pdf}

# (b) Lag recovery — sliding-window LOLO on the headline checkpoint
python -m model.vae_teb_prediction.model.model_experiment.synthetic.lag_recovery `
    --mode analyze --benchmark G1 --run-tag G1_baseline
python -m model.vae_teb_prediction.model.model_experiment.synthetic.lag_recovery `
    --mode width_sweep --widths 1,5,10,20         # picks w*
# Output: results/<benchmark>/lag_recovery/{summary.csv, metrics.json,
#         lolo_abar.pdf, lag_profile.pdf, lolo_vs_attn_overlay.pdf,
#         attn_heatmap.pdf, lolo_width_sweep.pdf}

# (c) β rate-distortion + HP analysis (after gpu_pool --mode beta / hp)
python -m model.vae_teb_prediction.model.model_experiment.synthetic.beta_sweep
# Output: results/<benchmark>/beta_sweep/{analysis.json, ratedist.pdf}

# (d) Calibration γ → 1 (Sprint 5 on G1; Sprint 7.1 on G2)
python -m model.vae_teb_prediction.model.model_experiment.synthetic.calibration `
    --benchmark G1 --build-missing --train-missing
# Output: results/<benchmark>/calibration/{calibration.json,
#         calibration_curve.pdf, summary.csv}

# (e) Directionality G1 ↔ G1-rev
python -m model.vae_teb_prediction.model.model_experiment.synthetic.directionality `
    --build-missing --train-missing
# Output: results/directionality/{summary.csv, metrics.json,
#         directionality_bars.pdf}

# (f) Null controls on a G2 checkpoint
python -m model.vae_teb_prediction.model.model_experiment.synthetic.null_controls `
    --build-missing
# Output: results/G2/null_controls/{summary.csv, metrics.json}
```

### 6. Headline report

Once any subset of the evaluators above has populated `results/`:

```powershell
python -m model.vae_teb_prediction.model.model_experiment.synthetic.final_report
# Output: results/final_report/{report.json, report_table.csv, headline.pdf}
```

`report.json.claim_tier.tier ∈ {strong, moderate, weak, deferred}` is
the manuscript-level verdict per `model_validation_v2.md §12`.

---

## Recommended end-to-end run order

For the **minimum viable result** (Sprint-plan §7) on one GPU:

```text
1. build_dataset       (G2_baseline)
2. gpu_pool --mode a_sweep --benchmark G2     # 5 c × 2 M = 10 cells
3. evaluate_te --mode sweep --benchmark G2     # K̄ vs TE, Metrics 1–4
4. calibration --benchmark G2                  # γ → 1 across 9 β × 3 TE points
5. lag_recovery --mode width_sweep             # pick w*, then --mode analyze
6. final_report                                # headline.pdf
```

For the **full v2 headline** add (in any order):

```text
7. build_dataset       (G1_baseline, G1-rev_baseline, G3_baseline, G1_twoband)
8. gpu_pool --mode a_sweep --benchmark G1
9. gpu_pool --mode directionality              # G1 + G1-rev pair
10. evaluate_te --mode sweep --benchmark G1
11. calibration --benchmark G1                 # 9 β × 3 TE points on G1
12. lag_recovery --mode analyze --benchmark G1
13. lag_recovery --mode analyze --benchmark G1_twoband
14. gpu_pool --mode a_sweep --benchmark G3
15. evaluate_te --mode sweep --benchmark G3
16. lag_recovery --mode analyze --benchmark G3
17. null_controls --build-missing              # uses a calibrated G2 ckpt
18. directionality                             # paired G1 ↔ G1-rev re-eval
19. beta_sweep                                 # if you also ran gpu_pool --mode beta
20. final_report
```

---

## Analyses, benchmarks and figures — interpretation guide

Everything below is **produced by the v2 pipeline**. For each artifact
this section gives the file path, the axes / values, and how to read it.

### A. Dataset previews (`build_dataset.py` → `visualize.py`)

Path: `data/<benchmark>/<tag>/preview.pdf`.

Panels are benchmark-aware (chosen at build time by
`_PRIMARY_DELAY` / `_panels_*` in `visualize.py`).

- **G1 / G1-rev / G1_twoband** (`_panels_state_space`):
  - **Target PSD overlay** — power spectral density of `fhr_st` /
    `fhr_ph` averaged over channels. The dominant peak should sit at
    the AR(2) angular frequency $\omega$ (default $0.05$ → period
    $\approx 125$ steps), well below Nyquist. Validates the
    low-frequency-target requirement (`model_validation_v2.md §2`).
  - **Oscillator phase portrait** — scatter of $(s_{t-1}, s_t)$ for
    the source state. A coherent ellipse means the AR(2) modes are
    operating in their oscillatory regime; a degenerate line means the
    damping is too low. Sanity check for V2-Q1.
  - **Parameter summary** — text dump of $r$, $\omega$, $A_y$,
    delays, $B_y$, and the computed block TE.
- **G2 / G2_wrong_delay / G2_zero_coupling** (`_panels_arx`):
  - **Delay-alignment scatter** — $y_t$ vs $u_{t-D}$ for the
    informative channels. A clear oblique cloud confirms the source
    drives the target at delay $D$; zero slope on G2_zero_coupling
    means the null is genuine.
  - **Per-channel lagged cross-correlation** — $\rho_{yu}(\tau)$ for
    each informative channel. Peak position should be near $\tau = D$
    (sometimes shifted by AR smoothing — the test suite allows a
    $D \pm 20$ window).
  - **Parameter summary** — $\rho_u, \rho_y, c, D, \sigma^2$.
- **G3** (`_panels_regime_switch`):
  - **Decoded-regime strip** — `argmax` of the per-channel one-hot
    source over time, colour-coded by regime id. Each row is a
    channel; sharp colour changes mark regime switches.
  - **Regime-coloured target trace** — `fhr_st[:,0]` overlaid with
    the regime strip in matching colours. Within a regime the target
    is a coherent oscillator at $\omega_k$ with amplitude $a_k$;
    across regime boundaries the phase is continuous (by design — the
    generator uses an integrated phase $\Phi_t$).
  - **Parameter summary** — $K, p_{\mathrm{switch}}, \delta$, template
    bounds.

If the preview looks wrong, **stop**. Every later figure is built on
top of it.

### B. Training curves (`train_minimal.py`)

Path: `results/<benchmark>/<run_tag>/training_curves.{pdf,png}` +
`metrics.csv`.

Two panels:

- **Loss panel** — train (solid) and val (dashed) for
  `feat_loss` / `base_loss` / `kld_loss` / `total_loss`. Look for
  (i) `base_loss` decreasing first as the FHR-only branch warms up,
  (ii) `feat_loss` then dropping below `base_loss` once the residual
  branch starts using the latent (gives `pred_gap > 0`), and (iii)
  `kld_loss` settling at a non-trivial positive level (collapse means
  it sits at zero from epoch 1).
- **Auxiliary panel** — `pred_gap`, `mu_prior_sat_frac`,
  `delta_mu_sat_frac` (latent-stabiliser tanh bounds — sustained
  values above ~0.05 mean the bound is throttling the latent and
  `mu_scale` / `delta_mu_scale` should be raised), and
  `mean_logvar_full` / `mean_logvar_base` (collapse signatures — a
  value pinned at `logvar_clamp[0] = -5` means the head collapsed).

### C. K̄ vs TE diagnostics (`evaluate_te.py`)

Path: `results/<benchmark>/eval_te/`.

#### Per-checkpoint figure: `diagnostics_<run_tag>.pdf` (2 × 2)

- **Top-left — Per-latent-dim KL** ($K_d$ in nats on the y-axis, dim
  index $d \in [0, d_z)$ on x). A bar plot. A **flat-zero row** is the
  posterior-collapse signature; a healthy plot shows a few dims
  carrying most of the KL. Mean shown as a dashed line.
- **Top-right — Paired vs shuffled K̄** (bar pair). Left bar is
  $\bar K$ with the real $U \to Y$ pairing; right bar is
  $\bar K$ after `shuffle_source_batch` randomises batch indices.
  **A faithful surrogate collapses the shuffled bar to near zero**;
  the gap is the genuine transfer signal. The panel title shows
  `te_true` for reference.
- **Bottom-left — Baseline vs full forecast loss** (bar pair). Left
  bar is $\mathcal{L}_{\mathrm{base}}$; right bar is
  $\mathcal{L}_{\mathrm{feat}}$. The title carries
  `pred_gap = L_base − L_feat`; **positive iff the residual branch is
  actually using the latent to improve forecasts**.
- **Bottom-right — Scalars** (text panel): `te_true`, `te_per_step`,
  `k_bar`, `k_bar_shuffled`, `kld_loss`, `mu_post_prior_gap`,
  `attn_entropy` (with the diffuse reference `ln(91) ≈ 4.511`; a
  smaller value means attention is sharper), epoch / n_test / warmup,
  `latent_stats_fitted`.

#### Sweep figures (`--mode sweep`)

- **`kbar_vs_te.pdf`** — scatter of $\bar K$ (nats, y) vs
  $\mathrm{TE}_{\mathrm{true}}$ (nats, x) across all sweep cells, one
  series per $M$ level. **Dashed line** is the fitted
  $\bar K = \alpha + \gamma \cdot \mathrm{TE}$. Title carries the
  Spearman $\rho$. PASS targets: monotone (Spearman $\rho \ge 0.95$)
  and calibrated ($|\gamma - 1| \le 0.2$).
- **`kbar_vs_<knob>.pdf`** — $\bar K$ (y) vs the swept knob (x):
  $B_y$ for G1, $c$ for G2, $p_{\mathrm{switch}}$ for G3. One line per
  $M$. The shape should match the analytic TE curve in $B_y / c / p$.
- **`predgap_vs_kbar.pdf`** — `pred_gap` (y) vs $\bar K$ (x). PASS:
  positive cluster (the latent helps prediction whenever K is
  non-trivial). A scatter in the upper half plane confirms Metric 4.
- **`per_dim_kl_heatmap.pdf`** — heat-map (settings × latent dim) of
  per-dim KL, settings ordered by analytic TE. Reveals whether the
  bottleneck **recruits more latent dimensions** as TE grows (a
  staircase pattern is the right shape).
- **`null_control.pdf`** — per-setting bar pair of paired vs shuffled
  K̄, with the analytic TE overlaid. PASS when shuffled bars are
  uniformly tiny and paired bars track the TE line.

#### Metrics JSON (`metrics.json`)

```text
metric1_null      : E_0 (mean |K̄| at TE=0), null/signal ratio,
                    k_bar_shuffled_mean, k_bar_reversed_mean
metric2_spearman  : monotonicity ρ across the sweep
metric3_calibration: alpha, gamma, r2  (deferred until calibration runs)
metric4_pred_gain : pred_gap means at TE=0 vs TE>0,
                    verdict_te0_near_zero, verdict_te_pos_positive
```

### D. Lag recovery — sliding-window LOLO (`lag_recovery.py`)

Path: `results/<benchmark>/lag_recovery/`.

The v2-D5 fidelity test: for each candidate lag $\ell$ on a coarse +
fine grid, a width-$w$ window of the **raw source input** centred on
$\ell$ is overwritten with $\mathcal{N}(0, 1)$, the forward pass is
re-run, and the degradation
$\delta_\ell = L_{\mathrm{feat}}(\text{corrupted}) -
L_{\mathrm{feat}}(\text{clean})$ is recorded. Normalised mass is
$A_\ell = [\delta_\ell]_+ / \sum_j [\delta_j]_+$.

- **`lolo_abar.pdf`** — bar plot, x: lag $\ell \in [0, L_{\max}]$,
  y: $A_\ell$. **The true band $\mathcal{L}^\star$ is shaded
  vermillion** (single band) or vermillion + green (two-band for
  `G1_twoband`). Title carries the window width $w$.
  PASS criterion: $\mathrm{LagMass}_{\mathrm{LOLO}} = \sum_{\ell \in
  \mathcal{L}^\star} A_\ell \ge 0.8$ (Criterion 4 in
  `model_validation_v2.md §8`).
- **`lag_profile.pdf`** — two stacked panels. Top: raw $\delta_\ell$
  (unsigned forecast-loss degradation, nats). Bottom: normalised
  $A_\ell$. Shows whether the in-band signal dominates the
  out-of-band noise.
- **`lolo_vs_attn_overlay.pdf`** — **the key diagnostic**:
  $A_\ell$ on the left (blue) axis, lag-sum-normalised attention
  weight $\bar\alpha_\ell$ on the right (orange) axis, shared
  lag x-axis with the true band shaded. The attention map is a
  heuristic; the LOLO map is faithful. A healthy model shows **both**
  peaked inside the band, but the LOLO is the headline.
- **`attn_heatmap.pdf`** — full $\bar\alpha_{t,\ell}$ across anchors
  $t$ and lags $\ell$, averaged over the test set. Diffuse rows mean
  uninformative attention; a bright vertical band at the right lag is
  the cleanest signature.
- **`lolo_width_sweep.pdf`** (`--mode width_sweep`) — stacked
  two-panel figure showing how `LagMass_LOLO` and `peak_lag` vary
  with $w \in \{1, 5, 10, 20\}$. The runner picks
  $w^\star = \min\{w : \mathrm{LagMass}(w) \ge 0.95 \cdot
  \max_w \mathrm{LagMass}(w)\}$. Use $w^\star$ in subsequent
  `--mode analyze` invocations.

### E. β rate-distortion + HP probes (`beta_sweep.py`)

Path: `results/<benchmark>/beta_sweep/`.

- **`ratedist.pdf`** — log-x scatter of $\bar K$ (rate, y) vs total
  forecast loss (distortion, x) across the β grid. The "knee" of the
  curve picks `selected_beta`: at very small β reconstruction is
  perfect but $\bar K$ overestimates TE; at very large β posterior
  collapses (`pred_gap → 0`, $\bar K → 0$). The selected β is the
  smallest β that still produces a non-collapsed latent.
- **`hp_*.pdf`** — analogue plots for the three hyper-parameter
  probes: `lambda_base ∈ {0.25, 0.5, 1.0}` (residual isolation
  weight), `d_z ∈ {8, 24}` (latent capacity), `warmup_period ∈
  {15, 30, 60}`.

### F. Calibration γ → 1 (`calibration.py`)

Path: `results/<benchmark>/calibration/`.

The Sprint-5 / 7.1 deliverable. For each per-step TE target in
`calibration.te_per_step_targets = [0.05, 0.15, 0.30]`, an inverter
(`analytic_te.B_y_for_te_block_state_space` for G1 or
`c_for_te_block_arx` for G2) bisects the coupling knob so the
analytic block TE hits the target. Then one model per (β, TE-point)
cell is trained with `compute_loss(likelihood='gaussian_nll',
sigma_obs='learned')` so $\bar K$ has **nat scale**. OLS over each
β-slice gives slope $\gamma$ and intercept $\alpha$; the selector
picks $\beta^\star = \arg\min_\beta |\gamma - 1| + 0.05 |\alpha|$
with $R^2$ as a tie-break.

- **`calibration_curve.pdf`** — single panel scatter of
  $(\mathrm{TE}_{\mathrm{true}}, \bar K)$ at the **selected** β, with
  the fitted line $\bar K = \alpha + \gamma \mathrm{TE}$ and the
  $y = x$ reference dashed. $(\alpha, \gamma, R^2, \beta^\star)$
  annotated in the corner. **An axes-fraction inset plots
  $\gamma$ vs $\beta$** across the full grid so the β choice is
  visible. PASS: $|\gamma - 1| \le 0.2$ (Criterion 3).
- **`calibration.json`** — full per-β table, the inverter provenance
  (target vs achieved block TE per TE-point), the selected-β record +
  rationale, the skipped-cell list, and a slim `cells` array
  consumed by `final_report.py`.

### G. Directionality G1 ↔ G1-rev (`directionality.py`)

Path: `results/directionality/`.

Two paired runs: G1 forward (oscillator state in the source slot →
real TE $> 0$) and G1-rev (oscillator state in the **target** slot →
slot-wise TE $= 0$ by construction).

- **`directionality_bars.pdf`** — bar pair of
  $\bar K_{\rm forward}$ vs $\bar K_{\rm reverse}$, with the ratio
  $\bar K_{\rm fwd} / \bar K_{\rm rev}$ annotated.
- **`metrics.json.comparison`** — `k_bar_forward`, `k_bar_reverse`,
  `directionality_ratio`, `verdict_direction_specific`
  (True ⟺ ratio $\ge 10$ — Criterion 5 in §8).

### H. Null controls (`null_controls.py`)

Path: `results/<source_benchmark>/null_controls/` (default
`<source_benchmark> = G2`).

Re-evaluates **one already-trained source-benchmark checkpoint** on
two control caches built from the same DGP family:

- **`wrong_delay`** — `G2_wrong_delay` cache with $D = 200 \gg
  L_{\max} + H = 120$. The lag-attention window cannot reach the
  true alignment. **INFO-only row** in the final report (the
  recurrent source LSTM can still propagate signal across more than
  $L_{\max}$ steps, so collapse here is not gating —
  `model_validation_v2.md §4.2`).
- **`zero_coupling`** — `G2_zero_coupling` cache with $c = 0$, so
  source and target are independent and the true TE is exactly $0$.
  **PASS-gated**: $\bar K \le 0.05$ nats is well below the v2 signal
  band of 0.05–0.3 nats.

Two CSV rows in `summary.csv`; same fields in `metrics.json.controls`.

### I. Final report (`final_report.py`)

Path: `results/final_report/`.

- **`report_table.csv`** — flat table with one row per (benchmark,
  metric): `value`, `criterion`, `status` ∈ {PASS, FAIL, INFO,
  DEFERRED}. The 12 v2 metric rows per benchmark:
  `null_E_0`, `spearman_rho`, `calibration_gamma`,
  `pred_gain_te0`, `pred_gain_te_pos`, `lag_mass_ratio_to_uniform`,
  `lag_mass_lolo`, `selected_beta`, `k_bar_shuffled`,
  `k_bar_reversed`, `k_bar_wrong_delay` (INFO),
  `k_bar_zero_coupling`. Plus one global `directionality` row.
- **`report.json`** — same content as the CSV plus the full collated
  artifacts and `claim_tier`:
  ```text
  strong   : null, monotonicity, lag, directionality,
             residual_usefulness, null_controls all pass
  moderate : null, monotonicity, null_controls, residual_usefulness pass
             (lag / calibration deferred or weak)
  weak     : only null + monotonicity + residual_usefulness pass
  deferred : no criterion has a converged verdict
  ```
- **`headline.pdf`** — 4-panel summary figure:
  1. **Calibration scatter** — $\bar K$ (y) vs
     $\mathrm{TE}_{\mathrm{true}}$ (x) across {G1, G2, G3} at each
     benchmark's selected β, with $y = x$ dashed. Each benchmark gets
     its own fitted $\gamma \cdot x + \alpha$ line. PASS shape: all
     three benchmark series collapse onto $y = x$.
  2. **Lag recovery** $A_\ell$ on G1 — same bar plot as panel D's
     `lolo_abar.pdf`, with the true band shaded and normalised
     attention overlaid on a twin axis.
  3. **Directionality bar pair** — same as panel G.
  4. **Rate-distortion** — $\bar K$ vs $\beta$ for G1, with the
     calibration-selected $\beta^\star$ marked by a vertical line.

  Every panel falls back to a `"not run / deferred"` placeholder
  when its input JSON is absent, so the headline renders at any
  point during the runtime gates.

---

## Success criteria — v2 claim tiers

Per `model_validation_v2.md §12` and `model_validation_v2_plan.md §8`:

| # | Criterion | Metric | PASS threshold | Sprint |
|---|---|---|---|---|
| 1 | Null | $E_0 = |\bar K|$ at TE=0 | $E_0 <$ smallest non-zero $\bar K$ | 6.5 |
| 2 | Monotonicity | Spearman ρ across the TE sweep | $\rho \ge 0.95$ | 5 / 7.1 / 7.2 |
| 3 | **Calibration** | $\gamma$ in $\bar K = \alpha + \gamma\,\mathrm{TE}$ | $|\gamma - 1| \le 0.2$ | 5.4 |
| 4 | Lag recovery | $\mathrm{LagMass}_{\mathrm{LOLO}}$ on $\mathcal{L}^\star$ | $\ge 0.8$ | 4 |
| 5 | Directionality | $\bar K_{X\to Y} / \bar K_{Y\to X}$ | $\ge 10$ | 6.1 |
| 6 | Residual usefulness | sign of `pred_gap` | $> 0$ iff $\mathrm{TE} > 0$ | 5 / 6.5 |

A **strong** claim needs 1, 2, 3, 4, 5, 6 to all hold (with at least
one passing null-control: shuffled / reversed / zero-coupling).
**Moderate** = 1, 2, 6 + null-controls pass; 3 fails. **Weak** =
only 1, 2, 6 hold.

---

## Status

| Sprint | Code | Runtime artifacts | Notes |
|---|---|---|---|
| 0 — plan lock-in & bookkeeping | done (2026-05-20) | n/a | model smoke green; this README |
| 1 — analytic TE for G1/G2/G3 | done (2026-05-20) | n/a | 49 tests in `test_analytic_te_v2.py` green |
| 2 — generators G1–G3 | done (2026-05-20) | n/a | 32 tests in `test_generators_v2.py` green; v1 gens deleted |
| 3 — wiring (build/visualize/eval) | done (2026-05-20) | n/a | 15 build/forward tests; G1_smoke cache OK |
| 4 — sliding-window LOLO | done (2026-05-20) | ⏳ G1 checkpoint width-sweep | 19 tests in `test_lag_recovery_v2.py` green |
| 5 — Gaussian-NLL + calibration | done (2026-05-20) | ⏳ G1 calibration train (9 β × 3 TE) | 24 new tests; MSE path bit-exact |
| 6 — directionality + null controls | done (2026-05-20) | ⏳ G1+G1-rev pair train; ⏳ G2 ckpt → null re-eval | 22 new tests; null_controls/metrics.json stub exists for G2 |
| 7 — breadth (G2 calib / G3 sweep) + headline | done (2026-05-21) | ⏳ G2 calibration (`--benchmark G2`); ⏳ G3 sweep; ⏳ G1 LOLO; ⏳ final_report | 18 new tests; deferred panel placeholders verified |

**G4** (switched sinusoid) is deferred per scope.

For the full progress log with file-level diffs see
`model_validation_v2_plan.md §11`.
