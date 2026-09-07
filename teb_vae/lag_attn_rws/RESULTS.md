# Calibration study — results

Status: **awaiting production runs.** Every number in this file comes from a training run, never
from the test suite. All but one study is a production-box run; the exception is **The
prior-anchor weight**, whose local arms are dev-box measurements on the committed HIE sample
shard, marked as such in every row, and taken to de-risk a production calibration rather than to
stand in for one. The structure below states, per study, exactly what must be recorded, so the
multi-day runs are filled in as arithmetic rather than re-derived from memory.

All runs launch from the repo root as

```bash
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_rws.trainer --config teb_vae/lag_attn_rws/configs/<arm>.yaml
```

and are identified by their `TEB_RUN_STAMP` directory and the `resolved_config.yaml` written
beside the checkpoints — the arms deliberately share the default tag so that each differs from
`default.yaml` in exactly its swept key (see the arm files' own headers).

## Filling these tables

Every table below is generated rather than transcribed. Evaluate each arm's checkpoint once, then
run the arm comparison over the directory holding the finished runs:

```bash
python -m teb_vae.lag_attn_rws.eval.run --checkpoint <arm-run>/model_checkpoints/<name>.ckpt
python -m teb_vae.lag_attn_rws.eval.verify --runs <dir-of-eval-runs> --out RESULTS_arms.md
```

`RESULTS_arms.md` carries the beta, `d_z`, reach and architecture tables, and the numbers below
are copied from it. Three sourcing rules make that copy safe: rows are keyed by the swept value
read from each run's own `resolved_config.yaml` — never from a directory name, so a renamed
directory cannot relabel a measurement; the epoch count, the final `val/kld_active_frac` and the
`Collapsed?` verdict come from each run's training `metrics_history.csv`, which is where the
per-epoch series the criterion consumes actually lives; and a collapsed arm is **marked** rather
than dropped, while an arm missing its CSV is reported as incomplete rather than silently
rowless.

Check each arm against the pre-registered acceptance criteria before quoting it:

```bash
python -m teb_vae.lag_attn_rws.eval.verify <arm-run>/eval_results/summary.json
```

`pred_gap` throughout this file is the `pred_gap_mc_nats` headline column — the Monte Carlo
marginalised score. The single-draw `pred_gap_train_path_nats` beside it is the objective-parity
check, not a second answer. `eval/EVAL.md` is the contract for every number here.

## Before launching: what reverts, and when to stop

The configuration these runs launch under changed three things at once — the prior scale anchor,
the gradient clip and the prior head's initialisation — and they do not revert the same way. This
section is written before the production time is spent, so a run that goes wrong is stopped
against a threshold agreed in advance rather than against a judgement made while seven GPUs are
busy, and so an operator can tell at a glance which mistakes cost a config edit and which cost a
retrain.

### What reverts, and how

| Change | Where it lives | Revert path | What reverting costs |
|---|---|---|---|
| The prior scale anchor at $10^{-1}$ | `model_config.VAE_model.beta_prior` in `configs/default.yaml` | **Config.** `0.0` multiplies the fourth term by zero and leaves the three-term objective exactly. The `prior_rate` column is written either way — it is a diagnostic, not a weighted term — so the prior's scale stays visible in the CSV of an unanchored run | The run restarts from epoch 0. Nothing else. |
| The gradient clip, $250 \to 5000$ | `advanced_config.trainer.gradient_clip_val` | **Config.** `250.0` | The run restarts, and returns to clipping every step at roughly a eleventh of the configured learning rate — see the clipping section below for what that was measured to cost. |
| The prior head's log-variance calibration | applied at construction when `model_config.VAE_model.head_init_calibration` is true | **Config, but not alone.** That one key also drives the decoder's output-head calibration; `false` reverts both, and nothing reverts the prior-head half by itself | A retrain. The calibration is baked into the weights at construction, so no checkpoint trained under it can be un-calibrated, and a resumed run carries it. Reverting the decoder half along with it gives back the $\approx 15$ nats/sample the raw-target NLL starts above the trivial predictor without it. |
| The eighth acceptance verdict `source_margin_positive`, the `source_margin` readout and the `source_margin_nats` / `prior_rate_nats` headline scalars | the evaluation package | **None by configuration.** Reverting the commits is the only path, because the verdict ordering raises on a criterion that is registered and not produced | Every evaluation directory collected under eight criteria would then need re-collecting — the same cost, in the other direction. |
| The regenerated artifact-stability reference manifest, `tests/data/eval_reference_manifest.json` | the test suite | **None by configuration**; it moves with the commit that regenerated it | — |
| The `prior_rate`, `beta_prior` and `grad_clip_frac` columns | the tracked metric surface | **None, and none needed** — they are additive | A `metrics_history.csv` written before them simply lacks them; nothing reads them as required, which is why the reference run below can be quoted without a `grad_clip_frac` column. |

Two consequences of that table are worth stating on their own, because both are cheap to discover
too late.

**An in-flight run cannot be moved between the calibrated and uncalibrated starts.** Resuming from
a checkpoint carries the initialisation with it, so a control arm at the uncalibrated start is a
separate launch from step 0 — never a resume with the key flipped.

**An evaluation directory collected before the eighth verdict cannot be re-reported.** The offline
`--only <analysis>` path refuses it with `StaleCachedVerdicts` rather than emitting a
seven-criterion summary, naming the fix: delete the collection from that directory, or point
`--output-dir` at a new one, and pass `--checkpoint` so the pass has a model to collect with. Any
directory from before the split therefore costs a full collection pass, not a report. `eval/EVAL.md`
carries this under its offline re-run guidance.

### Go/no-go while a run is in flight

Three signals, all in the run's own `train_results/metrics_history.csv`, all present from epoch 0.
They are read together because each is a different face of the same risk: the $20\times$ wider clip
lets through a step the old threshold was silently absorbing.

| Signal | Healthy | Hold and inspect | Stop the run |
|---|---|---|---|
| `spike_skipped` | $0$ in every epoch | any single non-zero epoch | non-zero in 3 or more epochs, or a consecutive run of skips reaching the `max_consecutive_skips` escape hatch of $25$ |
| `grad_clip_frac` | mean $\lesssim 0.05$ | mean above $0.25$ over any 20 consecutive epochs | $1.000$ over any 20 consecutive epochs |
| `val/total_loss` | falling, or flat within a fifth of a nat per 20-epoch window | a 20-epoch window mean more than $1$ nat/anchor above the window 20 epochs earlier, at any point after the beta ramp ends at epoch $50$ | any non-finite value, or three consecutive windows over that threshold |

**`val/total_loss` is a mixed-unit criterion.** It carries the three auxiliary shape terms beside
the four it always had, and those are L1 and Huber quantities on z-scored raw samples rather than
nats — so its *level* is no longer readable as nats per anchor, and comparing it across arms with
different `lambda_*` weights compares two different quantities. What survives is its **direction**,
which is what the row above gates on. For a pure-nats reading use `val/nll_full_block` and
`val/nll_base_block`; for the shape terms' own magnitudes, which is what re-derives their three
provisional weights, use `val/aux_multiscale`, `val/aux_derivative` and `val/aux_boundary`. A
`0.0`-weighted term is not computed and reports exact `0.0`, so a zero column means the term was off
rather than satisfied.

Each threshold is measured rather than chosen. **`spike_skipped`**: the unanchored 1018-epoch
reference run skipped nothing at all, and its largest single-batch excursion of `main_loss` from
its own EMA was $669$ nats against the $1000$ nat `additive_margin` — so the breaker has never
fired in this objective, and one firing is new behaviour rather than noise. **`grad_clip_frac`**:
$0.25$ is the criterion the clip change was validated against, and the validated dev-box arm's
worst 20-epoch running mean was $0.20$ against a whole-run mean of $0.045$; $1.000$ is what the
reference run was doing before the change. **`val/total_loss`**: over that reference run's $930$
post-ramp windows, the 20-epoch mean rose at all in $21$ of them and never by more than $0.153$
nats/anchor — including across the last $200$ epochs, where the run is converged and the windows
are noise. A $1$ nat threshold is therefore some $6.5\times$ the largest excursion a healthy run of
this objective has produced. It separates *descending or flat* from *rising*; a run that plateaus
far flatter than the reference should have its own noise band recorded here and the threshold
revisited against it.

**Read `spike_skipped` as a sample, not a rate.** Like `grad_clip_frac`, it reaches
`metrics_history.csv` through `trainer.callback_metrics` at validation end — before the training
epoch is reduced — so each row carries one optimizer step's indicator rather than the epoch's skip
rate. `grad_clip_frac` is where that is directly visible: it takes only $0$ and $1$ across all $200$
rows of the validated dev-box run, never a fraction. At $450$ steps per production epoch this cuts
both ways, and both ways argue for caution — a rare skip may appear in no row at all, and a row that
does read $1$ is a one-in-$450$ draw that came up positive, which is evidence of many. That is why a
single non-zero epoch is a hold rather than something to note and move past.

**What is not an abort signal.** `logvar_prior_floor_frac` is the measurement, not a health check:
the weakest arms are *expected* to pin, that outcome is the reading they exist to produce, and a
pinned arm is marked rather than dropped — so it runs to convergence like the others, because its
`nll_base_block` and coupling columns are what the comparison needs. The same holds for the
collapse criterion below: a collapsed arm is a recorded result. Stopping either early buys GPU
time and loses the row.

### When a stop fires

`spike_skipped` — stop, set `gradient_clip_val` back to `250.0`, relaunch, and record the finding
in the clipping section below rather than carrying it forward; the breaker firing under the wider
clip is the one outcome that section's derivation did not observe. `grad_clip_frac` at $1.000$ —
do **not** revert blindly: the gradient scale has moved by another order of magnitude, so re-derive
the threshold from that run's own `grad_norm` percentiles by the procedure in the sibling package's
`RESULTS.md` and relaunch at the new value. `val/total_loss` non-finite — stop and relaunch from
the last finite checkpoint at the reverted clip.

That third signal is the **validation** column deliberately. The breaker gates the training path
only, and on a batch it skips it replaces `train/total_loss` and `train/main_loss` with its own
EMA before they reach the logger — so a training curve can read finite and healthy while the
parameters are not. `spike_skipped` and the validation columns are what see through that, which is
why the first is a signal here and `train/total_loss` is not.

A relaunched arm writes a new `TEB_RUN_STAMP` directory and the arms deliberately share their tag,
so record which stamp superseded which beside the row it fills; otherwise two directories differing
only in their timestamp are indistinguishable afterwards.

## Baseline architecture

The baseline `default.yaml` now includes the landed architecture bundle: the plain conv stack
(D1), per-block FiLM in the horizon core (A2), the plain residual seams (R1-C1) and the
zero-parameter initialisation policies (`horizon_embed_std: 0.8`, `head_init_calibration: true`,
`a_head_gain: 2.0`). Every arm below — the beta, `d_z` and reach sweeps and the architecture A/B
arms — inherits all of them. The measured baseline size is **5,088,186 parameters**
(`DESIGN.md` §1). All arms are one-swept-key except two deliberate multi-key exceptions: the 240 s
reach arm (raised `warmup_period`) and the init-off arm, which reverts the whole three-key
init-policy bundle at once so it can be measured as a group on production data.

The five architecture A/B arms are created, not yet run; their results table is below the reach
sweep. Of the three first-run re-derivation items, `gradient_clip_val` is **done** — see the
gradient-clipping section below, which replaced the scaled $250$ with a measured $5000$. The
spike-breaker `additive_margin` and the `logvar_clamp` inspection remain: both were calibrated
under the old init scale and are re-derived under the new one, since `head_init_calibration`
moves the init NLL (and therefore the early gradient scale) by more than an order of magnitude.

## Collapse criterion

A completed run is **collapsed** when either

1. `val/source_conditioned_kl_raw` < **0.02 nats/anchor** at every one of its **last 5
   epochs**, or
2. its final `val/kld_active_frac` < **2 / d_z** (fewer than two active latent dimensions).

The single source of truth — constants, rationale, and the `is_collapsed` function that applies
it to a metrics CSV — is `collapse.py`. Both clauses say the same thing at the
same per-dimension activity epsilon (`KLD_ACTIVE_EPS = 1e-2`): the latent finished carrying
less than two dimensions' worth of source information. The criterion reads the tail of the run
only, because every run opens at exactly zero KL by construction.

## Arm inventory

| Config | Swept key (`model_config.VAE_model.`) | Value | Note |
|---|---|---|---|
| `sweep_beta_0p1.yaml` | `beta_schedule.end` | 0.1 | |
| `sweep_beta_0p3.yaml` | `beta_schedule.end` | 0.3 | |
| `sweep_beta_1p0.yaml` | `beta_schedule.end` | 1.0 | = shipped default; baseline run stands in |
| `sweep_beta_3p0.yaml` | `beta_schedule.end` | 3.0 | |
| `sweep_dz_24.yaml` | `d_z` | 24 | |
| `sweep_dz_32.yaml` | `d_z` | 32 | |
| `sweep_dz_48.yaml` | `d_z` | 48 | the pre-revision width; −62,823 params |
| `sweep_dz_64.yaml` | `d_z` | 64 | = shipped default; pinned, empty delta |
| `sweep_dz_96.yaml` | `d_z` | 96 | +126,063 params |
| `sweep_reach_null.yaml` | `causal_reach_budget_s` | null | unguarded; baseline run stands in |
| `sweep_reach_240.yaml` | `causal_reach_budget_s` | 240 | also `warmup_period: 60` (structural; see file) |
| `sweep_reach_120.yaml` | `causal_reach_budget_s` | 120 | max delay 30 = shipped warm-up |
| `sweep_reach_60.yaml` | `causal_reach_budget_s` | 60 | `up_ph` fully pruned below 100 s |
| `sweep_reach_32.yaml` | `causal_reach_budget_s` | 32 | most causal arm |
| `sweep_enc_kernel_7.yaml` | `encoder_extra_kernel` | 7 | −524,288 params (narrower long-dilation convs) |
| `sweep_norm_groups_1.yaml` | `conv_norm_groups` | 1 | per-timestep channel norm; 0 param change |
| `sweep_query_logvar.yaml` | `query_uses_logvar` | true | query reads `[μ^p ‖ ℓ^p]`; +6,144 params |
| `sweep_horizon_depth_5.yaml` | `horizon_depth` | 5 | +328,960 params (fifth refine block + FiLM) |
| `sweep_init_off.yaml` | (init bundle) | off | multi-key: `horizon_embed_std` 0.02, `head_init_calibration` false (decoder and prior-head calibration), `a_head_gain` 1.0 |

Surviving channel counts per finite budget (pinned by `tests/test_sweep_configs.py` against the
analytic filter bank): 240 s → 94 target / 43 source (worst delay 57 steps); 120 s → 78 / 29
(30); 60 s → 59 / 23 (14); 32 s → 43 / 19 (8). Arms of the reach sweep build different
input-adapter widths and therefore cannot share checkpoints.

The `d_z` sweep runs at the KL weight the beta sweep selects: that value is folded into
`default.yaml` first, and the arms inherit it — which is why the arm files carry only their
swept key.

## Distributed smoke and memory

To record, before any multi-day run:

- [ ] A short multi-rank `torchrun` completes several steps without deadlock (sampler sharding
      and the permutation-control rank reduction exercised).
- [ ] Peak memory at production geometry, read as `torch.cuda.max_memory_allocated()` after the
      first training step (the framework resets peak stats but never reads them back — this
      number is otherwise not reported anywhere).
- [ ] If memory binds: which levers were applied, in the documented order (`bf16-mixed` →
      `attention_grad_checkpoint` → batch 32 with `accumulate_grad_batches: 4`), and the
      measured effect of each.

Results: _pending._

## Precision validation

Required only if `bf16-mixed` is used for any headline number. The KL is a small difference of
larger quantities and bf16 carries eight mantissa bits, so it is the one memory lever that can
perturb the readout itself.

- [ ] `source_conditioned_kl_raw`, `mean_logvar_prior`, `pred_gap` compared between a short
      bf16 run and a short fp32 baseline; divergence reported.
- [ ] A stated tolerance, and the verdict on whether bf16 is admissible for headline runs.

The zero-parameter init policies (`horizon_embed_std`, `head_init_calibration`, `a_head_gain`)
live in the fp32 master weights under `bf16-mixed` and are applied once at construction, so they
have no structural interaction with the precision lever; they only shift the starting point the
fp32 optimiser state carries.

Results: _pending._

## Headline baseline

The default configuration trained to convergence; the run the module exists for.

- [ ] Both acceptance verdicts from `summary.json`, with their numbers
      (full < base, and full < base < shuffled).
- [ ] Comparison against the feature-target sibling on the shared quantities.
- [ ] `anchor_coverage_frac` distribution inspected; `coverage_floor: 0.9` confirmed or revised.
- [ ] Log-variance distributions inspected; `logvar_clamp: [-5, 3]` confirmed or revised.
- [ ] Observed `main_loss` scale; spike breaker `additive_margin` re-derived from it
      (provisional 1.0e+3 confirmed or revised).
- [ ] Latent-collapse verdict under the criterion above.

Results: _pending._

## KL-weight sweep (`beta_schedule.end`)

- [ ] Each arm trained to a stated minimum epoch count; metrics CSVs retained.
- [ ] Best `pred_gap` among non-collapsed arms identified.
- [ ] Collapsed arms reported as collapsed (criterion above), not dropped.
- [ ] Winning value folded into `default.yaml` before the `d_z` sweep.

| Arm | Epochs | `pred_gap` | `source_conditioned_kl_raw` | `kld_active_frac` | Collapsed? |
|---|---|---|---|---|---|
| 0.1 | | | | | |
| 0.3 | | | | | |
| 1.0 | | | | | |
| 3.0 | | | | | |

## Latent width sweep (`d_z`)

- [ ] Base NLL, active dimensions and `pred_gap` per arm.
- [ ] Verdict per size: is the baseline forecast capacity-starved?

`horizon_embed_std: 0.8` was calibrated at `d_z: 48` (projected-latent RMS ≈ 0.83 at init), which
the capacity revision moved to 64. By
xavier arithmetic the projected-latent RMS drifts across the `d_z` arms — ≈ 0.56 at `d_z: 24` to
≈ 0.82 at `d_z: 64` — so the horizon-token symmetry break is milder at small `d_z` and near-matched
at large `d_z`. Symmetry breaking tolerates the ≈2× mis-scale; if a `d_z` arm's result turns on it,
replace the fixed 0.8 with a measured-at-construction match (recorded here so the inheritance is
not silent).

| Arm | `nll_base_block` | Active dims | `pred_gap` | Capacity-starved? |
|---|---|---|---|---|
| 24 | | | | |
| 32 | | | | |
| 48 | | | | |
| 64 | | | | |

## Reach budget sweep (`causal_reach_budget_s`)

- [ ] Surviving channel counts per block per arm (cross-checked against the startup log).
- [ ] The budget at which `up_ph` disappears, and what dropping it costs.
- [ ] Whether a causal budget exists at which the coupling readout survives — the condition
      under which the transfer-entropy label becomes defensible.

| Arm | Channels kept (tgt/src) | `nll_base_block` | `pred_gap` | `source_conditioned_kl_raw` | Collapsed? |
|---|---|---|---|---|---|
| null | 109 / 58 | | | | |
| 240 | 94 / 43 | | | | |
| 120 | 78 / 29 | | | | |
| 60 | 59 / 23 | | | | |
| 32 | 43 / 19 | | | | |

## Architecture A/B arms

Each arm flips one structural knob (or, for init-off, the whole init bundle) against the baseline,
which already carries D1 + A2 + R1-C1 + the init policies. Created, not yet run.

- [ ] Each arm trained to a stated minimum epoch count against the baseline; metrics CSVs retained.
- [ ] For each arm, whether it beats the baseline on `pred_gap` without collapsing the latent
      (criterion above), and at what parameter cost.
- [ ] `encoder_extra_kernel: 7` — does the −524,288-param encoder cost the forecast anything?
- [ ] `conv_norm_groups: 1` — does a single-group conv norm change the exported representation?
- [ ] `query_uses_logvar: true` — does querying from the prior's certainty move the coupling?
- [ ] `horizon_depth: 5` — does the deeper horizon core earn its +328,960 params?
- [ ] `init_off` — the isolation run for the init bundle, on production data; folded back into the
      baseline verdict for R3-C1/R3-C2/R2-C1.

| Arm | Δparams vs baseline | `nll_base_block` | `pred_gap` | `source_conditioned_kl_raw` | `kld_active_frac` | Collapsed? |
|---|---|---|---|---|---|---|
| `encoder_extra_kernel: 7` | −524,288 | | | | | |
| `conv_norm_groups: 1` | 0 | | | | | |
| `query_uses_logvar: true` | +6,144 | | | | | |
| `horizon_depth: 5` | +328,960 | | | | | |
| `init_off` | 0 | | | | | |

## The prior-anchor weight

This package carries no `beta_prior` sweep arms on purpose: the sweep runs on the
conv-Transformer sibling, where the prior-variance collapse was measured, and this architecture
receives the chosen weight only. What is recorded here is the anchor's validation on this
architecture — five arms on the committed HIE sample shard bracketing the anchor weight and the
gradient clip, and the production confirmation at the chosen weight. The five local rows are
dev-box measurements, the one deliberate exception to this file's production-box rule: they exist
to de-risk the production calibration, and their run directories are the artifact of record.

Two controls rather than one, because the two questions they answer need different starts. The
**uncalibrated** control reverts `head_init_calibration` as well, so its prior head begins where
the production run that showed the collapse began — that is the arm that has to reproduce the
pathology. The **calibrated** control differs from the anchored arms in `beta_prior` alone, so it
is the only honest reference for what the anchor costs the base forecast.

One asymmetry to state rather than leave implicit: **no arm here reproduces the reported run's
start exactly.** That run's checkpoint records `head_init_calibration: true`, and at the commit it
ran from, the key drove the decoder output-head calibration alone — the prior-scale half did not
exist yet. Its start is therefore decoder-calibrated and prior-uncalibrated, which one boolean
cannot express: the uncalibrated control reverts both halves and the calibrated control applies
both. The uncalibrated control matches it on the **prior** half, which is the half whose collapse
is under test, and diverges on the decoder half — which is exactly why the base-forecast reading
below is taken against the calibrated control and not this one.

Two derived columns beside the tracked ones. `source_margin` is the per-recording paired
difference $D_{\mathrm{shuffled}} - D_{\mathrm{full}}$ from the evaluation's permutation control —
positive means the matched source beat a derangement-shuffled stranger — and the local rows carry
a training-side aggregate of it instead, for the reason given below. `abs(pred_gap)/K` is the
amplification ratio: nats of forecast movement per nat of source-conditioned divergence $K$ (val
`source_conditioned_kl_raw`), read together with the **sign** of `pred_gap`, since the same
magnitude means degradation where the gap is negative and gain where it is positive. The
reference row is the sibling architecture's unanchored 1018-epoch production run (mean over
epochs 900–1018), the measurement the anchor exists to answer: a run that fixes the floor
fraction but not the ratio has not demonstrated the fix.

### How the local rows were produced

Five arms over `output/hie_cs.hdf5`, one RTX 4080 device, $200$ epochs each, from
`configs/smoke_hie.yaml` and `configs/smoke_hie_control.yaml`, the rest differing from
`smoke_hie.yaml` in exactly the keys named in their row. **Each row states its own `beta_prior`
and clip, because two of those defaults moved underneath these arms as a result of what they
measured**: the shipped weight was $10^{-2}$ when the first three ran and is $10^{-1}$ now, and
the clip was $250$ for the first four and $5000$ for the last. Re-running the table means naming
both values per arm, not re-running the two config files. As they stand today `smoke_hie.yaml`
resolves to the last row. Normalisation
statistics are `output/hie_cs_stats.hdf5`, generated from this shard at `trim_minutes: 1.0`; its
recorded `fhr` mean and standard deviation, $145.754680$ and $15.824679$, are the shard's own to
six decimals, against the four-sample fixture's $140.046998$ and $9.958108$ — so it is generated,
not inherited, which is the failure that would otherwise pass every existing guard silently.

**Epochs here are not production epochs.** $339$ windows at batch $32$ is $11$ optimizer steps per
epoch, against $450$ in the production run. Every criterion phrased in epochs is therefore
restated in steps below, and a first pass at $60$ epochs — $660$ steps — was discarded because
both arms were still moving at the end of it and neither could be read.

Two further properties of the local setting, stated once so no row is over-read. Both splits are
the same shard, because `dataset_kwargs` is shared between the two loaders and cannot carry a
per-split GUID filter, so every `val/` column is in-sample. And `source_margin` in the local rows
is the **training-side aggregate** $\text{`nll\_shuffled\_block`} - \text{`nll\_full\_block`}$
from `metrics_history.csv`, not the evaluation's per-recording paired difference; the two answer
the same question at different resolutions, and only the latter carries a confidence interval.
The local rows have no evaluation pass, so their verdict cell is empty by construction.

- [x] Control (`beta_prior: 0.0`, prior calibration off): `logvar_prior_floor_frac` exceeds 0.5
      within five epochs, reproducing the collapse from the uncalibrated start. **Met, read in
      steps.** It crossed at epoch $15$ — $165$ optimizer steps — where the production run crossed
      between its epochs $0$ and $1$, i.e. within $450$–$900$ steps. The local collapse is
      *faster* per step than the one it reproduces; the five-epoch form of this criterion was
      written against a production-sized epoch and does not transfer.
- [ ] Anchored: floor fraction below 0.2 at the last epoch, `mean_logvar_prior` at least 1.0
      above the floor, `nll_base_block` within 2% of the control. **Not met at the then-shipped
      `beta_prior: 1.0e-2`; met at $0.1$, which is now the shipped value.** See below — this is
      the sprint's substantive finding.
- [x] The amplification ratio recorded for every local arm and for the production run.
- [x] Run directories named beside their rows.

| Run | Run directory | `beta_prior` | clip | `logvar_prior_floor_frac` | `mean_logvar_prior` | `prior_rate` | `nll_base_block` | `pred_gap` | `abs(pred_gap)/K` | `source_margin` | $K$ | `kld_active_frac` | Verdicts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| unanchored reference (sibling architecture) | `output/lag_attn_rws_transformer` | 0 | 250 | 0.992 | −4.986 | — | −197.19 | −7.37 | 4.3 | +1.06 | 1.70 | 0.292 | FAIL / PASS / FAIL |
| local control, uncalibrated start | `2026-08-04--[14-29]-…_control` | 0.0 | 250 | 0.996 | −4.989 | 95.9 | +93.28 | +0.47 | 1.26 | −0.47 | 0.378 | 0.237 | — |
| local control, calibrated start | `2026-08-04--[15-28]-…_bp0` | 0.0 | 250 | 0.978 | −4.968 | 95.4 | −0.71 | +1.29 | 2.40 | +0.31 | 0.536 | 0.252 | — |
| local anchored, old shipped weight | `2026-08-04--[14-50]-…_smoke` | 1.0e-2 | 250 | 0.955 | −4.934 | 94.6 | −3.21 | +1.36 | 2.82 | −1.30 | 0.482 | 0.248 | — |
| local anchored | `2026-08-04--[15-09]-…_bp0p1` | 0.1 | 250 | 0.058 | −3.278 | 56.2 | +4.34 | +8.14 | 2.72 | +10.02 | 3.00 | 0.726 | — |
| **local anchored, re-derived clip — the shipped configuration** | `2026-08-04--[15-50]-…_bp0p1_clip5000` | 0.1 | 5,000 | **0.046** | **−3.332** | 57.4 | −9.15 | +10.74 | 3.11 | +16.63 | 3.46 | 0.692 | — |
| production confirmation | | 0.1 | 5,000 | | | | | | | | | | |

Every figure is the mean over the last $10$ epochs, matching the reference row's shape. Directory
stems are elided at the tag; each is under `output/lag_attn_rws_hie_*`.

**The shipped `beta_prior: 1.0e-2` delays the collapse; it does not prevent it.** The floor
crossing moves from epoch $15$ to epoch $100$ — $165$ steps to $1{,}100$, a $6.7\times$ delay —
and then the prior pins anyway, finishing at $0.955$ against the unanchored $0.978$. Nothing in
that arm's final state distinguishes it from the failure it was added to prevent.

**The mechanism is a saturating restoring force, so the weight is a threshold rather than a
dial.** $\partial R_p / \partial \mathrm{lv}_p = \tfrac{1}{2}(e^{\mathrm{lv}_p} - 1)$ per latent
dimension, which tends to $-\tfrac{1}{2}$ as the log-variance falls: however far the prior sinks,
the anchor never pushes back harder than $0.5\,\beta_{\mathrm{prior}}$ per dimension. The
reconstruction's downward pressure, by contrast, *grows* as the decoder sharpens — the same
coupling between a sharpening decoder and a sinking prior that produced the collapse, seen from
the other side. So a weight either exceeds the pressure or is eventually overrun by it, and there
is no weight that merely slows the descent to an acceptable rate. $10^{-2}$ is on the losing side
of that threshold and $10^{-1}$ is on the winning side, with no in-between behaviour observed.

**At $0.1$ the anchor holds, and the latent stays wide.** `kld_active_frac` is $0.69$–$0.73$
against $0.24$–$0.25$ in all three collapsed arms and $0.292$ in the production run. The dead
latent coordinates that motivated the `d_z` sweep are therefore a *symptom* of the prior collapse
rather than an independent width problem — which is what the sweep would otherwise have been
launched to find out, and it was established here without touching $d_z$.

**The base forecast is not the price.** The design claim is that anchoring the prior's *scale*
leaves the prior mean, and therefore the base forecast, alone. The stated 2% test cannot decide
it: `nll_base_block` passes through zero as the decoder sharpens (the calibrated control finishes
at $-0.71$), and a percentage of a quantity crossing zero is not a quantity. Read in nats per
anchor instead, at the matched clip the $0.1$ anchor costs $5.05$ nats against the calibrated
control — $0.011$ nats per raw sample, under $1\%$ of the trivial predictor's $1.42$ — and under
the re-derived clip it *beats* that control, $-9.15$ against $-0.71$. The cost is real but small,
and most of what looked like cost at clip $250$ was the clip.

**On the amplification ratio, mind the sign.** The reference row's $4.3$ is nats of forecast
*degradation* per nat of $K$: its `pred_gap` is negative. Every local arm's `pred_gap` is
positive — the source helps — so the same ratio there is nats of *gain* per nat of $K$, and the
two are not comparable by magnitude. The local runs are in-sample, which is the most likely
reason the sign differs, and it is why no local row claims a coupling result. What the local runs
do establish is the floor fraction, the latent width and the base-forecast cost, none of which
depend on that sign.

**The head calibration alone is worth something.** With no anchor at all, calibrating the prior
head moves the floor crossing from epoch $15$ to epoch $49$ — a $3.3\times$ delay from an
initialisation policy that costs no parameters and no compute.

**The shipped default moved to $10^{-1}$ on this evidence**, in both packages. The argument for
leaving it at $10^{-2}$ pending the production sweep does not survive contact with what the sweep
is for: $10^{-2}$ is *measured* to fail its own purpose on the only data anyone has run it
against, and shipping a default known not to work — so that an operator launching `default.yaml`
today reproduces the collapse the term was added to prevent — is not caution, it is a trap. The
weight is set to the smallest value tested that held.

What that does **not** settle, and the config comment says so: this is one shard, in-sample, on
this architecture. The transformer's four `sweep_beta_prior_*` arms bracket $10^{-1}$ on both
sides and are what confirm or revise it on the architecture the collapse was measured on. Their
`0p1` arm now restates the shipped value and their `0p01` arm carries the old one, so the
comparison that produced this change is still run there rather than assumed.

## The gradient-clipping threshold — measured

`gradient_clip_val` shipped at $250.0$, scaled from the feature-target sibling's $0.5$ by the
$\approx 2 \times 480$ change in loss magnitude rather than derived. It now ships at
**$5000.0$**. The derivation is the first production run of this objective, on the
conv-Transformer sibling — same objective, same reduction, same units — and the full working is
in that package's `RESULTS.md`. In short: `train/grad_norm` recorded $q_{50} = 2775$,
$q_{99} = 4681$, $q_{99.9} = 5866$, maximum $7313$, minimum $703$, and **every** recorded step
exceeded $250$, so that run performed normalised-gradient descent at roughly a eleventh of its
configured learning rate. $5000$ is the smallest round value above $q_{99}$.

Validated on the dev box before it shipped, at `beta_prior: 0.1` held fixed across the pair, so
the $\approx 20\times$ change in effective step size is the only difference between the two arms:

| Arm | `grad_clip_frac`, whole run | `grad_clip_frac`, last third | `grad_norm` median | `grad_norm` max | `spike_skipped` | `total_loss`, last third |
|---|---:|---:|---:|---:|---:|---:|
| clip 250 | 1.0000 | 1.0000 | 2,320 | 12,509 | 0 | +86.97 → +20.99 |
| clip 5,000 | **0.0450** | 0.0746 | 2,052 | 8,165 | 0 | +74.97 → +16.12 |

Every pre-registered criterion is met: `grad_clip_frac` falls from $1.000$ to $0.045$, well under
the $0.25$ threshold; `spike_skipped` stays at zero; `total_loss` is finite throughout and falls
across the last third on both the training and validation paths (validation $+82.78 \to +3.14$).
The run did not destabilise, so no revision to the threshold is recorded. The comparison also
improved every quantity the anchor is judged on — the clip-$5000$ row of the table above is the
best arm on the floor fraction, the base forecast, the source margin and $K$ alike — which is
what an eleven-fold under-training of every step would predict.

**Read `grad_clip_frac` as a mean over epochs, not per row.** `metrics_history.csv` records one
optimizer step per epoch for this metric and for `grad_norm`, not the epoch's aggregate: the
metrics collector reads `trainer.callback_metrics` at validation end, before the training epoch is
reduced. So the column is $0$ or $1$ in every row and its mean over the run is the estimate of the
per-step exceedance fraction. Every `val/` column in this file is a true epoch mean; these two are
the exception.

Results: the local validation is complete; the production confirmation row is _pending._
