# v3 — Mixed-Population TE Validation for the Lag-Attentive VAE-TEB Model

**Status:** Active · **Created:** 2026-06-09 · **Owner:** Mahdi-Si
**Builds on:** `model_validation_v2.md` (theory) and `model_validation_v2_plan.md`
(single-cell G1–G4 tracker). **Read those first.**

This document specifies the **mixed-population (`G1_mix`)** experiment — train
*one* model on a heterogeneous pool and recover transfer entropy *per
sub-population* — which until now existed only in module docstrings. It records
the rationale, the math, the $\beta$-selection requirement (the key fix over the
original implementation), success criteria, and the added experiments.

Code: `synthetic/mixed_dataset.py` (build) · `synthetic/train_ddp.py` /
`synthetic/gpu_pool.py` (train) · `synthetic/mixed_calibration.py` ($\beta$
sweep + selection) · `synthetic/mixed_eval.py` (per-group recovery).

> **⚠️ Patch (2026-06-27) — see `model_validation_v3_mixed_patch_plan.md`.** The original grid
> collapsed for *every* cell: at $M{=}16$, block $\mathrm{TE}{=}1.5$, $H{=}30$ the
> per-step-per-channel TE is $\approx 0.003$ nats (source drive $\approx 0.6\%$ of the target
> innovation variance) — below a finite model's noise floor on a near-unit-root target with an
> unidentifiable lag, so $\bar K$ floored out and **no $\beta$ could calibrate**. The patch
> *concentrates the signal* so it is extractable and *gates on realizability*: aggressive DGP
> retune ($r{=}0.80$, $A_y{=}0.40$, $K_{\text{history}}{=}64$); a fixed identifiable headline band
> `tiny:[4,4]` with low $M\in\{1,2\}$ and `delay_walk:false`; a model-free **R0 gate** — a held-out
> ridge probe (`analytic_te.realizable_te_block_from_arrays`,
> `mixed_eval.run_realizability_preflight`, pipeline stage `r0_realizability`) that measures how
> much of the analytic TE a finite predictor recovers at the training sample size and **hard-stops
> before training** if the headline cells are not realizable; plus post-train **collapse /
> prediction-gain** gates (`mixed_eval._collapse_summary`; the `pred_gain_vs_te` figure now overlays
> the realizable curve and the $y{=}x$ optimal line). The random-walk-lag pool is kept as a separate
> robustness build (`G1_mix_walk`); high-$M$ / wide-band cells become a graceful-degradation axis.
> Gates R0–R6 and exact config keys are in the patch plan.

---

## 1. Paradigm and rationale

Every other v2 benchmark caches a **single** $(M, \mathrm{TE}, \text{lag})$ cell
and trains **one model per cell**. The mixed experiment asks the complementary,
harder question:

> Train **one** model on a **single heterogeneous pool** that mixes
> informative-channel counts $M$, block transfer entropies $\mathrm{TE}$, and
> lag bands. Does the model recover the true KLD / lag / TE **per
> sub-population**, and does it **generalise** to held-out
> $(M, \mathrm{TE}, \text{lag})$ triples it never trained on?

This tests whether $K_t$ has learned transfer entropy **as a function** of the
data-generating regime, not merely fit one operating point. The held-out-triple
extrapolation is the evidence that distinguishes "learned the mapping" from
"memorised the grid".

The pool is built by `mixed_dataset.py`: one cell per
$(M, \mathrm{TE}, \text{band})$ grid triple is solved and generated, then
concatenated with **per-sample provenance** (`sample_te_true`, `sample_M`,
`sample_delay_min/max`, `sample_band_id`, `sample_cell_id`, `sample_held_out`)
so `mixed_eval.py` can read each sample's own cell TE and grouping keys.

---

## 2. The quantity and the identity (why $\beta$ must be selected)

The model reports the per-step latent KL surrogate

$$
K_t \;=\; \mathrm{KL}\!\left(q_\phi(z_t\mid Y_{\le t},U_{\le t}) \,\big\|\,
p_\psi(z_t\mid Y_{\le t})\right),
$$

and the headline claim is calibration of its window mean $\bar K$ against the
analytic block transfer entropy,

$$
\bar K \;=\; \alpha \;+\; \gamma\,\mathrm{TE}^{(H)}_{U\to Y}, \qquad \gamma\to 1 .
$$

The governing identity is

$$
\mathbb{E}[K_t]
\;=\;
\underbrace{I_q(Z;U\mid Y)}_{\text{directed term (want = TE)}}
\;+\;
\underbrace{\mathbb{E}_Y\!\big[\mathrm{KL}(q_\phi(z\mid Y)\,\|\,p_\psi(z\mid Y))\big]}_{\text{prior-mismatch floor (want } \to 0)} .
$$

So $\bar K = \mathrm{TE}$ requires **(a)** the learned prior to match the
aggregate posterior over $U$ (floor $\to 0$) **and (b)** $Z$ to be a
sufficient-and-minimal statistic of $U$ for the future $Y^+$. Point **(b) is
exactly what $\beta$ controls**:

- $\beta$ too small $\Rightarrow$ the KL is nearly free, $z$ over-encodes,
  $I_q(Z;U\mid Y)\gg\mathrm{TE}$, so $\gamma\gg 1$;
- $\beta$ too large $\Rightarrow$ the latent collapses, $\gamma\to 0$.

**Therefore a calibrated $\beta$ must be *selected*, not fixed.** The original
mixed run trained at the global default $\beta=10^{-3}$ under `gaussian_nll` and
so could not deliver $\gamma\to 1$; the Gaussian-NLL switch is *necessary*
(it puts $\bar K$ and the likelihood on a common nat scale) but *not sufficient*.

### Null-subtracted calibration (validating the identity)

Shuffling the source $U$ destroys the directed term while leaving the floor
intact, so $\bar K_{\text{shuffle}}$ **estimates the floor**. `mixed_eval` fits
both the raw slope and the **null-subtracted** slope on
$\bar K - \bar K_{\text{shuffle}}$: a correct model has the subtracted intercept
$\to 0$ with $\gamma$ unchanged, and per $M$,

$$
\alpha_M \;\approx\; \bar K_{\text{shuffle}}(M) \;\approx\; \bar K(\mathrm{TE}=0; M).
$$

The `prior_mismatch` figure shows these three quantities per $M$.

---

## 3. Ground-truth transfer entropy

All cells use the G1 AR(2)-oscillator state-space DGP with a **within-signal
random-walk lag** (Section 3.1.1 of `model_validation_v2.md`): the lag
$d_{i,t}$ drifts as a bounded reflecting integer walk on
$\{d_{\min},\dots,d_{\max}\}$, locally constant over any $H=30$ block.

- **Per-cell TE.** The $M$ informative channel pairs are mutually independent, so
  the cell block TE is the per-channel block TE $\times M$. The coupling $B_y$ is
  solved once per distinct $(d_{\min}, d_{\max}, \mathrm{TE}/M)$ key by the
  mean-over-delays inverter `analytic_te.B_y_for_mean_te_block_state_space` and
  reused across $M$ (holding $\mathrm{TE}_{\text{true}}$ fixed as $M$ varies, so
  the $M$ axis isolates channel dilution). The realised cell TE is the
  histogram-weighted mean over the walk's lag occupancy.
- **Per-channel solve.** Computing TE channel-by-channel keeps each Monte-Carlo
  determinant ratio well-conditioned ($K$ regressors, not $K\cdot M$),
  avoiding the high-$M$ singularity.
- **Zero-coupling null cell.** A grid value $\mathrm{TE}=0$ short-circuits the
  inverter to $B_y=0$ (true TE $=0$ exactly) and **bypasses** the MC-floor trim;
  it anchors the calibration intercept $\alpha$ with an in-distribution null
  rather than an extrapolated one. (`mixed_dataset.enumerate_mix_cells`.)

---

## 4. The grid and the caches

Default `benchmarks.G1_mix.mix` (config `config_synth.yaml`):

| Axis | Values |
|---|---|
| $M$ (informative channels) | $\{8, 16, 32\}$ trained; $\{4, 64\}$ extrapolation-only |
| $\mathrm{TE}$ (nats) | $\{0.0, 0.1, 0.7, 1.5, 3.0\}$  ($0.0$ = null anchor) |
| lag band $[d_{\min},d_{\max}]$ | short $[1,8]$, mid $[1,15]$, long $[1,31]$ |
| samples / cell | train 1000 / val 200 / test 300 (scale to 2000/400/600 on the big box) |

Three test caches share one builder (`mixed_dataset.build_g1_mix`):

- `data/G1_mix/<tag>/` — the in-mix pool (all three splits, trained cells).
- `data/G1_mix/<tag>_holdout/` — interior held-out triples (`mix.holdout`); an
  **interpolation** test (every held-out marginal value still appears in a
  trained cell, enforced by `verify_holdout_marginals`).
- `data/G1_mix/<tag>_extrap_m{4,64}/` — test-only caches at $M$ **outside** the
  trained range (`mix.holdout_m`); a genuine **extrapolation** across the
  channel-dilution axis. Built via `build_g1_mix(..., extrap_m=M)`; every sample
  is stamped `sample_held_out=1` so `mixed_eval` scores it on the in-mix
  calibration.

Optional `data.randomize_channel_layout: true` places the informative channels
at random (per-cell, seeded) positions instead of $[0,M)$, removing the
positional leak by which a model could decode $M$ from *which* front channels
are active. The permutation is TE-invariant (z-scoring is per-channel) and
recorded in `meta.channel_perm_{y,u}`; default off.

---

## 5. The $\beta$-selection protocol (the core of v3)

`mixed_calibration.py` trains the pooled model **once per $\beta$**, evaluates
each checkpoint with `mixed_eval.evaluate_mixed`, fits the **per-$M$**
calibration slope at each $\beta$, and selects

$$
\beta^\star \;=\; \arg\min_\beta\;
    \operatorname{mean}_M |\gamma_M(\beta) - 1|
    \;+\; \lambda_\alpha \operatorname{mean}_M |\alpha_M(\beta)| ,
\qquad \lambda_\alpha = 0.05 ,
$$

the per-$M$ generalisation of `calibration.select_beta_by_calibration`. The
per-$M$ score is used (not a single pooled slope) because a pooled slope can
hide an $M$ with $\gamma\ll 1$ behind another with $\gamma\gg 1$.

The $\beta$ grid is centred on the nat-scale regime
($\{10^{-3}, 3\!\cdot\!10^{-3}, 10^{-2}, 3\!\cdot\!10^{-2}, 10^{-1},
3\!\cdot\!10^{-1}, 1, 3\}$, config `mix_calibration.beta_grid`; resolution order
`mix_calibration.beta_grid` $\to$ `calibration.beta_grid` $\to$
`beta_sweep.grid`).

**Compute (8× A6000).** Default is *task-parallel single-GPU* — one independent
pooled training per $\beta$, fanned across the GPUs by
`gpu_pool.py --mode mix_beta`. (`train_ddp`'s own docstring notes DDP is
wasteful for many small runs.) A sequential 8-GPU DDP fallback per $\beta$ is
available via `mixed_calibration --mode ddp`.

Artifacts: `results/G1_mix/mixed_calibration/{summary.csv, calibration.json}`
plus `gamma_vs_beta.{pdf,png}` (one line per $M$, $\gamma=1$ reference,
$\beta^\star$ marked) and `selection_score_vs_beta.{pdf,png}`.

---

## 6. Success criteria / claim tiers (per sub-population)

| # | Criterion | Metric | Threshold |
|---|---|---|---|
| 1 | Null | $\bar K$ at the TE$=0$ cell **and** under shuffle/reverse | $\approx 0$ (below smallest non-zero $\bar K$) |
| 2 | Monotonicity | Spearman $\rho(\bar K, \mathrm{TE})$ **per $M$** | $\rho \ge 0.95$ |
| 3 | **Calibration** | per-$M$ $\gamma_M$ at $\beta^\star$ | $|\gamma_M - 1| \le 0.2$ for every $M$ |
| 3b | Identity | null-subtracted intercept; $\alpha_M$ vs shuffle floor | $\alpha_M^{\text{nullsub}} \approx 0$; $\alpha_M \approx \bar K_{\text{shuffle}}$ |
| 4 | Lag recovery | per-cell sliding-window LOLO LagMass on $\mathcal{L}^\star$ | $\ge 0.8$ |
| 5 | Prediction gain | sign of $\Delta\mathcal{L}=\mathcal{L}_{\text{base}}-\mathcal{L}_{\text{feat}}$ | $>0$ iff $\mathrm{TE}>0$ |
| 6 | Generalisation | held-out + $M$-extrapolation gap vs trained-$M$ mean | small (report) |

**Tiers.** *Strong:* 1, 2, 3, 4, 5 hold per $M$ → "$K_t$ is a calibrated TE
estimator across a heterogeneous population". *Moderate:* 3 fails but 1, 2, 4, 5
hold → "$K_t$ ranks TE correctly per sub-population but is not nat-calibrated".
*Weak:* only 1, 2, 5 → "$K_t$ responds to source information".

---

## 7. Added experiments (v3)

1. **$\beta$-sweep + per-$M$ calibration selection** (Section 5) — the core fix.
2. **Zero-coupling null cell** — in-distribution $\mathrm{TE}=0$ anchor per band.
3. **Null-subtracted + prior-mismatch diagnostics** — validate the identity.
4. **$M$-extrapolation** — test-only caches at $M\in\{4,64\}$ outside the trained
   range (genuine dilution-axis extrapolation).
5. **Channel-position randomization** — robustness to the positional leak.
6. **Posterior-collapse vs $\beta$** — read off `gamma_vs_beta` (collapse shows
   as $\gamma_M\to 0$ at large $\beta$).

---

## 8. Run recipes (8× A6000, 128 cores)

Per-box config overrides (edit `config_synth.yaml` on the box; not committed as
defaults): `optim.batch_size: 128`, `model.attention_grad_checkpoint: false`
(~10–15 % faster on 48 GB), `dataset: {num_workers: 2, pin_memory: true,
persistent_workers: true}`, and optionally `mix.n_per_cell_*: 2000/400/600`.
`mix.build_workers: auto` already fans the build across the 128 cores.

**Host RAM (the exit `-9` fix).** With `dataset.mmap: auto` (the default),
`SyntheticTEDataset` memory-maps the uncompressed split `.npz` instead of
reading it into RAM, so all DDP ranks *and* all DataLoader workers share
**one** page-cache copy of the pool (~227 KB/sample ⇒ ~10 GB at 1000/200 per
cell, ~21 GB at 2000/400). Before this fix each of the 8 ranks eagerly held a
private copy ($8\times$ the pool, plus worker copies), which tripped the
kernel OOM killer (rank SIGKILLed, exit `-9`) right after DDP registration.
The first epoch demand-pages from disk; later epochs run at RAM speed.
`dataset.mmap: false` restores the eager per-process load.

```bash
M=model.vae_teb_prediction.model.model_experiment.synthetic
# 1. Sweep beta (builds the in-mix + held-out pools once, then 8 trainings):
python -m $M.gpu_pool --mode mix_beta --gpus 0,1,2,3,4,5,6,7
# 2. Evaluate every beta + select beta* (per-M calibration):
python -m $M.mixed_calibration --no-build --no-train
# 3. (optional) Final DDP run of the headline model at beta*:
python -m $M.train_ddp --tag G1_mix_base --beta <beta*> --devices 8
# 4. Build the extrapolation caches and run the full extrapolation eval at beta*:
python -m $M.mixed_dataset --extrap-m-all
python -m $M.mixed_eval --run-tag mixed_calibration/beta_<token> \
    --in-mix-tag G1_mix_base --holdout-tag G1_mix_base_holdout
python -m $M.mixed_eval --run-tag mixed_calibration/beta_<token> \
    --in-mix-tag G1_mix_base --holdout-tag G1_mix_base_extrap_m64
```

One-shot alternative (build + sweep + select in one process):
`python -m $M.mixed_calibration --gpus 0,1,2,3,4,5,6,7`.

**Single-file end-to-end driver:** `synthetic/run_mixed_pipeline.py` runs every
stage above in the correct order — build the in-mix / holdout / extrap-$M$
caches, the **data-anatomy previews** (`synthetic/visualize_mixed.py`:
per-channel source/target panels with the true lag walk $d_t$, colour-matched
"this source section drives this target section $d_t$ steps later" annotations,
the AR-filtered *driven component* overlay, an innovation-cross-correlation
evidence panel, the TE $\times$ lag-band case gallery, and the full channel
atlas — written to `data/G1_mix/<tag>/previews/`), (optionally) the $\beta$
sweep + selection, the final `train_ddp` run at $\beta^\star$, the
`mixed_eval` passes (in-mix + holdout, plus one per extrapolation cache, each
in its own `mixed_eval_extrap_m<M>/` subdirectory), and finally the broad
model-diagnostic pipeline (`run_pipeline_tests.py` →
`testing.run_full_test_pipeline`: histograms, forecast quality, attention / lag
diagnostics, KL-PCA, residual usage, …) on the **same** $\beta^\star$
checkpoint, under `results/G1_mix/<run_tag>/testing_pipeline/<output_tag>/`.
It is pure edit-and-run: no CLI — set the `PIPELINE` dict in `__main__`
(stage toggles, `devices`, `beta`: `None` / float / `"auto"`) and run the file.
Builds skip complete caches and training skips an existing checkpoint, so the
driver is resumable after an interruption.

---

## 9. Reading the KLD scales (training curves vs `mixed_eval`)

Two KL reports exist, **both correct, $\times d_z$ apart** — do not compare
them as if they shared units:

1. **Loss-side `kld_loss`** (the "KLD" line in `metrics.csv` /
   `training_curves` / `loss_plot_epoch.html`): the masked mean over
   $(b, t, d)$,

   $$
   \texttt{kld\_loss}
   \;=\;
   \frac{\sum_{b,t,d} m_{b,t}\,\mathrm{KL}_{b,t,d}}{d_z \sum_{b,t} m_{b,t}},
   $$

   i.e. **nats per latent dimension per step**. This is the quantity $\beta$
   multiplies in the objective, so its per-dim normalisation is what makes
   $\beta$ comparable across `d_z` choices.

2. **Eval-side $\bar K$** (`mixed_eval` / `evaluate_te`): the per-sample mean
   of the **dim-summed** `kld_per_t` over the clean window
   $[\max(\text{warmup}, d_{\max}-1),\,T-H)$,

   $$
   \bar K_i
   \;=\;
   \frac{1}{|W_i|} \sum_{t \in W_i} \sum_{d=1}^{d_z} \mathrm{KL}_{i,t,d},
   $$

   i.e. **nats per step** — the correct unit for the TE surrogate, since for
   diagonal Gaussians $\mathrm{KL}(q\,\|\,p) = \sum_d \mathrm{KL}_d$ and the
   analytic block TE is also a whole-vector quantity in nats.

So with $d_z = 24$, a training-curve KLD of $\approx 2$ **is** an eval-side
$\bar K \approx 48$ nats — the "completely different range" between the loss
plot and the `mixed_eval` figures is the $d_z$ factor (plus the smaller window
difference: the training KL window is $[\text{warmup}, T)$ on train/val, the
eval window is the per-sample clean window on test). To make this readable at
a glance, training now also logs **`kld_nats`** $= d_z \cdot$ `kld_loss`
(`train_kld_nats` / `val_kld_nats` CSV columns, a dashed overlay in the
`training_curves` KL panel, and a `kld_nats` entry in each checkpoint's
`train_metrics`). Compare **`kld_nats`** with the $\bar K$-vs-TE figures; the
`mixed_eval` `kld_vs_te_overview` figure plots $\bar K$ (dim-summed nats) and
$\bar K / d_z$ (the `kld_loss` scale) side by side to make this $d_z$ factor
explicit.

**Why not switch back to `mse`?** The likelihood is *not* the cause of the
scale gap — $\bar K$ is a latent-space KL and is computed identically under
either likelihood. Under `mse` the reconstruction term is not a log-density,
so the rate–distortion trade-off has no common unit and $\bar K$ is
**scale-free**: only monotone trends are certified and the $\gamma \to 1$
calibration target is meaningless (v2 §10.1, Decision V2-D4). `gaussian_nll`
(with `sigma_obs: learned`) is what puts $\mathcal{L}_{\text{feat}}$ and the
KL on a common nat scale, which is why the `G1_mix` loss overlay pins it.
A $\bar K$ that far exceeds the true TE at $\beta = 10^{-3}$ is the expected
under-regularised regime ($I_q(Z;U\mid Y) \gg \mathrm{TE}$, §2) — the fix is
**selecting $\beta^\star$** (Section 5), not changing the likelihood.

---

## 10. What strong evidence looks like

> On a single heterogeneous pool with analytically known per-cell
> $\mathrm{TE}^{(30)}_{U\to Y}$, the model's mean latent KL $\bar K$ is near zero
> at the zero-coupling cell and under shuffled/reversed source, increases
> monotonically with true TE within every $M$, is **per-$M$ nat-calibrated**
> ($|\gamma_M-1|\le 0.2$) at the selected $\beta^\star$, localises source
> attribution to the known lag band, has $\Delta\mathcal{L}>0$ only where
> $\mathrm{TE}>0$, and extrapolates to held-out $(M,\mathrm{TE},\text{band})$
> triples and to untrained $M\in\{4,64\}$ with a small generalisation gap.

The headline figures are `mixed_calibration/gamma_vs_beta`, and from the
$\beta^\star$ eval: the master `kld_vs_te` (per-cell $\bar K$ vs TE, faceted
$M \times$ band, each $M$-row on its own adaptive scale), `kld_vs_te_overview`
(the dim-summed-nats and per-dim $\bar K / d_z$ scales side by side),
`calibration_health` (per-$M$ $\gamma_M$ with the $|\gamma-1|\le0.2$ target band
plus the local-$\gamma$ $\bar K/\mathrm{TE}$ heatmap), `prior_mismatch`,
`lag_profiles`, `pred_gain_vs_te`, and `generalization_gap`.

**Per-sample scatter suite** (every test sample as one point, colour $= M$,
rendered in every eval pass): `per_sample_scatter` ($2\times2$: raw $\bar K$
vs TE on linear and log scales — the linear panel carries a secondary
$\bar K/d_z$ axis on the loss-side `kld_loss` scale — plus pooled- and
per-$M$-calibrated $\widehat{\mathrm{TE}}$ vs $y=x$), `per_sample_nullsub`
(the sample-wise floor-subtracted response $\bar K - \bar K_{\text{shuffle}}$,
raw + calibrated), `per_sample_te_error` (boxplots of the single-sample
calibrated-TE error per TE level $\times$ $M$), `per_sample_kbar_ecdf`
(per-$M$ ECDFs per TE level — single-sample separability) and
`per_sample_null_scatter` (per-sample clean vs shuffled $\bar K$, log–log).
The `evaluate_te`-style cross-cell suite (`kbar_vs_te`, `kbar_vs_B_y`,
`predgap_vs_kbar`, `kbar_vs_te__byM`, `per_dim_kl_by_cell`,
`null_control_bars`) renders alongside them at zero extra compute by adapting
the per-cell rows into `evaluate_te`'s own renderers. The pipeline's
default-on `combined_figures` stage pools the `per_sample.csv` /
`per_cell.csv` of the in-mix and every `mixed_eval_extrap_m<M>` pass into
`results/G1_mix/<run_tag>/combined_figures/` (all five $M$ colours in one
figure); it is pure CSV $\to$ matplotlib and can be re-rendered without GPU
via `python -m ...synthetic.mixed_eval --combined-only --run-tag <tag>`.
