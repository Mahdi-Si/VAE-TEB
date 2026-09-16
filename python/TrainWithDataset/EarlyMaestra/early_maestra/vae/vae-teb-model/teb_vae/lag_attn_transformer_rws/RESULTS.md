# Causal conv-Transformer encoder study — results

Status: **awaiting production runs.** Every number in this file comes from a run on the production
box; nothing here is produced by the test suite. The structure below states, per study, exactly what
must be recorded, so the multi-day runs are filled in as arithmetic rather than re-derived from
memory. The one number already measured is the parameter budget, because it is a property of the
constructed model rather than of a run.

## What this study is comparing, and the rule for reading it

The question is whether replacing both history encoders — a dilated causal convolution stack in
parallel with an LSTM — with causal conv-Transformer encoders improves the forecast at **matched
capacity**. The comparison model is `teb_vae/lag_attn_rws` trained on the same shards with the
same objective, the same optimiser, the same schedule and the same seed. It is arm **A0** of this
study and it is not a config in this directory.

The question used to read "at $38.7\%$ fewer parameters": the capacity revision raised this
architecture's encoders and left the comparison model's alone, so the two now sit within $1.8\%$ of
each other and the axis is read at near parity. That is a better-posed comparison rather than a lost
headline — a forecast difference now attributes to the encoder's *structure* rather than to its size
— and it is why every table below carries `params` beside the forecast columns.

**Do not select on KL magnitude.** A stronger target prior lowers the source-conditioned KL without
the coupling having weakened — the prior simply predicts more of what the source was carrying — so
an encoder that improves the model can look worse by the headline number. Select on KL that comes
with **source-specific predictive gain**, and treat a competitive $D_0$ as a precondition: an arm
whose base reconstruction is worse than the baseline's has not earned a reading on `pred_gap` at
all. Every table below therefore carries `nll_base_block` and `pred_gap` beside the KL, so the trade
is visible rather than inferred.

## Launch lines

From the repository root. The arms deliberately share the baseline's tag, MLflow experiment and run
name — a per-arm identity would be a second delta — so a run is identified by its `TEB_RUN_STAMP`
directory and the `resolved_config.yaml` written beside its checkpoints.

```bash
STAMP="$(date '+%Y-%m-%d--[%H-%M]')"

# Baseline (A3, the recommended arm)
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_transformer_rws.trainer --config teb_vae/lag_attn_transformer_rws/configs/default.yaml

# Phase 1 — architecture
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_transformer_rws.trainer --config teb_vae/lag_attn_transformer_rws/configs/sweep_arch_a1.yaml
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_transformer_rws.trainer --config teb_vae/lag_attn_transformer_rws/configs/sweep_arch_a2.yaml

# Phase 2 — source locality and the causal input budget
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_transformer_rws.trainer --config teb_vae/lag_attn_transformer_rws/configs/sweep_window_8.yaml
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_transformer_rws.trainer --config teb_vae/lag_attn_transformer_rws/configs/sweep_window_32.yaml
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_transformer_rws.trainer --config teb_vae/lag_attn_transformer_rws/configs/sweep_window_64.yaml
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_transformer_rws.trainer --config teb_vae/lag_attn_transformer_rws/configs/sweep_window_full.yaml
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_transformer_rws.trainer --config teb_vae/lag_attn_transformer_rws/configs/sweep_reach_null.yaml
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_transformer_rws.trainer --config teb_vae/lag_attn_transformer_rws/configs/sweep_base_sample.yaml
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_transformer_rws.trainer --config teb_vae/lag_attn_transformer_rws/configs/sweep_logvar_residual.yaml
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_transformer_rws.trainer --config teb_vae/lag_attn_transformer_rws/configs/sweep_source_dropout_0p2.yaml
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_transformer_rws.trainer --config teb_vae/lag_attn_transformer_rws/configs/sweep_source_dropout_0p3.yaml

# Phase 3 — depth and width
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_transformer_rws.trainer --config teb_vae/lag_attn_transformer_rws/configs/sweep_target_blocks_4.yaml
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_transformer_rws.trainer --config teb_vae/lag_attn_transformer_rws/configs/sweep_target_blocks_8.yaml
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_transformer_rws.trainer --config teb_vae/lag_attn_transformer_rws/configs/sweep_source_blocks_2.yaml
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_transformer_rws.trainer --config teb_vae/lag_attn_transformer_rws/configs/sweep_source_blocks_4.yaml
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_transformer_rws.trainer --config teb_vae/lag_attn_transformer_rws/configs/sweep_ff_384.yaml

# The decoder-side pair — one arm per mechanism the capacity revision added
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_transformer_rws.trainer --config teb_vae/lag_attn_transformer_rws/configs/sweep_aux_off.yaml
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_transformer_rws.trainer --config teb_vae/lag_attn_transformer_rws/configs/sweep_horizon_attn_off.yaml

# The prior-anchor weight
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_transformer_rws.trainer --config teb_vae/lag_attn_transformer_rws/configs/sweep_beta_prior_0p001.yaml
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_transformer_rws.trainer --config teb_vae/lag_attn_transformer_rws/configs/sweep_beta_prior_0p01.yaml
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_transformer_rws.trainer --config teb_vae/lag_attn_transformer_rws/configs/sweep_beta_prior_0p1.yaml
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_transformer_rws.trainer --config teb_vae/lag_attn_transformer_rws/configs/sweep_beta_prior_1p0.yaml
```

## Before launching: what reverts, and when to stop

The configuration these arms launch under changed three things at once — the prior scale anchor,
the gradient clip and the prior head's initialisation — and they do not revert the same way. This
section is written before the production time is spent, so an arm that goes wrong is stopped
against a threshold agreed in advance rather than against a judgement made while seven GPUs are
busy, and so an operator can tell at a glance which mistakes cost a config edit and which cost a
retrain.

### What reverts, and how

| Change | Where it lives | Revert path | What reverting costs |
|---|---|---|---|
| The prior scale anchor at $10^{-1}$ | `model_config.VAE_model.beta_prior` in `configs/default.yaml`, restated or overridden by each `sweep_beta_prior_*` arm | **Config.** `0.0` multiplies the fourth term by zero and leaves the three-term objective exactly. The `prior_rate` column is written either way — it is a diagnostic, not a weighted term — so the prior's scale stays visible in the CSV of an unanchored run | The arm restarts from epoch 0. Nothing else. |
| The gradient clip, $250 \to 5000$ | `advanced_config.trainer.gradient_clip_val` | **Config.** `250.0` | The arm restarts, and returns to the normalised-gradient descent the section below measures: every step clipped, at roughly a eleventh of the configured learning rate. |
| The prior head's log-variance calibration | applied at construction when `model_config.VAE_model.head_init_calibration` is true | **Config, but not alone.** That one key also drives the decoder's output-head calibration; `false` reverts both, and nothing reverts the prior-head half by itself | A retrain. The calibration is baked into the weights at construction, so no checkpoint trained under it can be un-calibrated, and a resumed run carries it. Reverting the decoder half along with it gives back the $\approx 15$ nats/sample the raw-target NLL starts above the trivial predictor without it. |
| The eighth acceptance verdict `source_margin_positive`, the `source_margin` readout and the `source_margin_nats` / `prior_rate_nats` headline scalars | the shared evaluation pipeline, which this package binds rather than copies | **None by configuration.** Reverting the commits is the only path, because the verdict ordering raises on a criterion that is registered and not produced | Every evaluation directory collected under eight criteria would then need re-collecting — the same cost, in the other direction. |
| The regenerated artifact-stability reference manifest, `teb_vae/lag_attn_rws/tests/data/eval_reference_manifest.json` | the sibling package's test suite, which gates this package's evaluation too | **None by configuration**; it moves with the commit that regenerated it | — |
| The `prior_rate`, `beta_prior` and `grad_clip_frac` columns | the tracked metric surface, shared with the comparison model | **None, and none needed** — they are additive | A `metrics_history.csv` written before them simply lacks them, which is why the $1018$-epoch baseline run is quoted below with `grad_norm` percentiles and no `grad_clip_frac` column. |

Two consequences of that table are worth stating on their own, because both are cheap to discover
too late.

**An in-flight run cannot be moved between the calibrated and uncalibrated starts.** Resuming from
a checkpoint carries the initialisation with it, so an arm at the uncalibrated start is a separate
launch from step 0 — never a resume with the key flipped.

**An evaluation directory collected before the eighth verdict cannot be re-reported.** The offline
`--only <analysis>` path refuses it with `StaleCachedVerdicts` rather than emitting a
seven-criterion summary, naming the fix: delete the collection from that directory, or point
`--output-dir` at a new one, and pass `--checkpoint` so the pass has a model to collect with. Any
directory from before the split therefore costs a full collection pass, not a report. `eval/EVAL.md`
carries this under its offline re-run guidance.

### Go/no-go while an arm is in flight

Three signals, all in the arm's own `train_results/metrics_history.csv`, all present from epoch 0.
They are read together because each is a different face of the same risk: the $20\times$ wider clip
lets through a step the old threshold was silently absorbing.

| Signal | Healthy | Hold and inspect | Stop the arm |
|---|---|---|---|
| `spike_skipped` | $0$ in every epoch | any single non-zero epoch | non-zero in 3 or more epochs, or a consecutive run of skips reaching the `max_consecutive_skips` escape hatch of $25$ |
| `grad_clip_frac` | mean $\lesssim 0.05$ | mean above $0.25$ over any 20 consecutive epochs | $1.000$ over any 20 consecutive epochs |
| `val/total_loss` | falling, or flat within a fifth of a nat per 20-epoch window | a 20-epoch window mean more than $1$ nat/anchor above the window 20 epochs earlier, at any point after the beta ramp ends at epoch $50$ | any non-finite value, or three consecutive windows over that threshold |

**`val/total_loss` is a mixed-unit criterion.** It now carries the three auxiliary shape terms
beside the four it always had, and those are L1 and Huber quantities on z-scored raw samples rather
than nats — so its *level* is no longer readable as nats per anchor, and comparing it across two
arms that carry different `lambda_*` weights compares two different quantities. What survives that
is its **direction**, which is what the row above gates on, and the thresholds are in the same units
as the column they are applied to. For a pure-nats reading use `val/nll_full_block` and
`val/nll_base_block`; for the shape terms' own magnitudes use `val/aux_multiscale`,
`val/aux_derivative` and `val/aux_boundary`, which is also where the weights get re-derived. A
`0.0`-weighted term reports exact `0.0` rather than its would-be value, so a zero column means the
term was off and not that it was satisfied.

Each threshold is measured rather than chosen: the first and third from this package's own baseline
run, the middle one from the sibling's dev-box validation of the clip.
**`spike_skipped`**: that $1018$-epoch run skipped nothing at all, and its largest
single-batch excursion of `main_loss` from its own EMA was $669$ nats against the $1000$ nat
`additive_margin` — so the breaker has never fired in this objective, and one firing is new
behaviour rather than noise. **`grad_clip_frac`**: $0.25$ is the criterion the clip change was
validated against on the sibling's dev box, where the validated arm's worst 20-epoch running mean
was $0.20$ against a whole-run mean of $0.045$; $1.000$ is what this package's baseline run was
doing before the change, read as the fraction of its recorded `grad_norm` steps above $250$ since
the column itself postdates it. **`val/total_loss`**: over that run's $930$ post-ramp windows the
20-epoch mean rose at all in $21$ of them and never by more than $0.153$ nats/anchor — including
across the last $200$ epochs, where it is converged and the windows are noise. A $1$ nat threshold
is therefore some $6.5\times$ the largest excursion a healthy run of this objective has produced.
It separates *descending or flat* from *rising*; an arm that plateaus far flatter than the baseline
should have its own noise band recorded here and the threshold revisited against it.

**Read `spike_skipped` as a sample, not a rate**, for the same reason the section below gives for
`grad_norm`: it reaches `metrics_history.csv` through `trainer.callback_metrics` at validation end,
before the training epoch is reduced, so each row carries one optimizer step's indicator rather than
the epoch's skip rate. `grad_clip_frac` is where that is directly visible — across all $200$ rows of
the sibling's dev-box validation it takes only $0$ and $1$, never a fraction; this package's
baseline run predates the column entirely. At the baseline's $450$ steps per epoch this cuts both
ways, and both ways
argue for caution: a rare skip may appear in no row at all, and a row that does read $1$ is a
one-in-$450$ draw that came up positive, which is evidence of many. That is why a single non-zero
epoch is a hold rather than something to note and move past.

**What is not an abort signal.** `logvar_prior_floor_frac` is the measurement, not a health check.
The two weakest arms are *expected* to pin — that outcome is the reading they exist to produce, and
the anchor's own threshold behaviour predicts it — so a pinning arm is marked rather than dropped
and runs to convergence like the others, because its `nll_base_block` and coupling columns are what
the comparison needs. The same holds for the collapse criterion below: a collapsed arm is a
recorded result. Stopping either early buys GPU time and loses the row.

### When a stop fires

`spike_skipped` — stop, set `gradient_clip_val` back to `250.0`, relaunch, and record the finding
in the clipping section below rather than carrying it forward; the breaker firing under the wider
clip is the one outcome that section's derivation did not observe. `grad_clip_frac` at $1.000$ —
do **not** revert blindly: the gradient scale has moved by another order of magnitude, so repeat
the five-step derivation below against that run's own `grad_norm` percentiles and relaunch at the
new value. `val/total_loss` non-finite — stop and relaunch from the last finite checkpoint at the
reverted clip.

That third signal is the **validation** column deliberately. The breaker gates the training path
only, and on a batch it skips it replaces `train/total_loss` and `train/main_loss` with its own
EMA before they reach the logger — so a training curve can read finite and healthy while the
parameters are not. `spike_skipped` and the validation columns are what see through that, which is
why the first is a signal here and `train/total_loss` is not.

A relaunched arm writes a new `TEB_RUN_STAMP` directory and the arms deliberately share their tag,
run name and experiment, so record which stamp superseded which beside the row it fills; otherwise
two directories differing only in their timestamp are indistinguishable afterwards.

## Where the numbers come from

**Every column below is read from the run's own training artefacts**, not from an evaluation run,
and that is deliberate rather than provisional: these tables record how the arms *trained*, and the
one metric surface every arm emits from epoch $0$ is the only thing all of them can be compared on
while the sweep is still in flight. Reading them has three rules that make the copy safe.

1. **Rows are keyed by the swept value read from each run's own `resolved_config.yaml`**, never from
   a directory name — a renamed directory cannot then relabel a measurement.
2. **Per-epoch series come from `train_results/metrics_history.csv`**, which is where every tracked
   metric actually lands. Quote the final epoch unless a column says otherwise.
3. **A collapsed arm is marked, not dropped**, and an arm missing its CSV is reported as incomplete
   rather than silently left rowless.

The collapse criterion is the shared one in `teb_vae/lag_attn_rws/collapse.py`, applied unchanged
because the objective is unchanged: a completed run is collapsed when `val/source_conditioned_kl_raw`
is below $0.02$ nats per anchor at every one of its last $5$ epochs, or when its final
`val/kld_active_frac` is below $2 / d_z$. The criterion reads the tail only, because every run here
opens at exactly zero KL by construction.

**Derived quantities.** Column names that are not tracked metrics, defined once here so a table
cannot introduce an undefined one:

- `params` — `sum(p.numel() for p in model.parameters())` on the constructed model.
- `delta_params` — the same, minus the baseline's.
- `source_reach` — the source encoder's `receptive_field`, in steps; blank means unbounded.
- `epochs` — completed epochs, from the metrics CSV.
- `peak_memory_gb` — `torch.cuda.max_memory_allocated()` after the first training step.
- `examples_per_s`, `steps_per_s` — throughput at the same effective batch as the comparison model.
- `collapsed` — the verdict of the criterion above.
- `source_margin` — the per-recording paired difference $D_{\mathrm{shuffled}} - D_{\mathrm{full}}$
  from the evaluation's permutation control; positive means the matched source beat a
  derangement-shuffled stranger. An evaluation quantity, not a training column.
- `abs(pred_gap)/K` — the amplification ratio: nats of forecast degradation per nat of
  source-conditioned divergence, with $K$ the val `source_conditioned_kl_raw`. $4.3$ in the
  unanchored reference run; the quantity the prior-scale anchor must reduce.

**What the evaluation adds, and where it lands instead.** The questions these tables cannot answer —
what the source added on held-out recordings, where in the past it added it, whether the observation
model is calibrated, and what the encoders' attention actually attends to — belong to
`teb_vae/lag_attn_transformer_rws/eval/`, whose contract is `eval/EVAL.md`. Its output is a
per-checkpoint `eval_results/` directory, and its arm and cross-model tables are generated into their
own document rather than transcribed here:

```bash
python -m teb_vae.lag_attn_transformer_rws.eval.run --checkpoint <run>/model_checkpoints/<name>.ckpt
python -m teb_vae.lag_attn_transformer_rws.eval.verify --runs <dir-of-runs> --out RESULTS_arms.md
```

The selection rule at the top of this file is the same rule that document carries, deliberately: a
reader who reaches the arm table without this page must still meet it.

## Parameter budget — measured

The one table already filled in, because it is a property of the constructed model. Predicted from
the architecture's arithmetic, then measured on a shipped-geometry build; the two agree exactly.

| Component | Predicted | Measured |
|---|---:|---:|
| Convolution block, $k = 5$ | 50,176 | 50,176 |
| Convolution block, $k = 9$ | 50,688 | 50,688 |
| Attention block | 262,656 | 262,656 |
| Target encoder | 1,676,928 | 1,676,928 |
| Source encoder | 888,960 | 888,960 |
| **Total** | **4,996,844** | **4,996,844** |

Against $5{,}088{,}186$ for the comparison model — a $1.8\%$ reduction, with the two history
encoders at $2{,}565{,}888$ against $2{,}657{,}230$. That margin was $38.7\%$ before the capacity
revision, which raised this architecture's encoders and left the comparison model's alone; the two
rows are now near parity in budget, which is a better-posed encoder comparison rather than a lost
headline.

## The gradient-clipping threshold — measured, and now provisional again

`gradient_clip_val` shipped at a provisional $250.0$, carried over from the comparison model with a
stated reason (`DESIGN.md` §11) rather than measured. The $1018$-epoch baseline run supplied the
measurement, and `default.yaml` carries **$5000.0$**. Nothing in this section is produced by the
test suite.

**That measurement predates this capacity**, so the shipped value is provisional again and is
marked so in `configs/default.yaml`. It was read off a run at $d_z = 48$, four target blocks,
$d_{\mathrm{ff}} = 256$ and a $128$-wide three-block decoder, under the four-term objective — the
revision moved every one of those, and the three shape terms put gradient on the decoder that the
measured distribution never saw. It ships unchanged rather than rescaled by a guess, because a
measured value carried across a scale change is a known quantity and an invented one is not.
**Re-derive it from the first run at this geometry**, by the five steps below, before any arm is
launched off the baseline.

The same holds for the spike breaker's `additive_margin` ($1.0 \times 10^{3}$ nats). It is a
threshold on `main_loss`, which is now the mixed-unit criterion — at the shipped weights the aux
contribution is small beside a $480$-sample summed NLL, but that is an expectation and not a
measurement. The headline-baseline checklist above already carries its re-derivation; the three
`aux_*` columns are logged from epoch $0$, so the split between the nats and the shape terms is
readable directly.

The procedure, which is what a re-derivation after any change of scale repeats:

1. **Launch the baseline** at the line above and let it run to at least $2{,}000$ optimizer steps —
   the same budget as the learning-rate warm-up, so the reading covers the whole ramp and the first
   steps at full learning rate, which is where a pre-norm stack is most fragile.
2. **Read `train/grad_norm`** from `train_results/metrics_history.csv`. The task logs it *pre-clip*
   in `on_before_optimizer_step`, so it is the true norm and not the clipped one — that is the only
   place the quantity exists at all, and there is deliberately no `val/` variant.
3. **Record the quantiles** below over that window, plus the maximum and the fraction of steps
   already exceeding the incumbent threshold.
4. **Set the threshold above the healthy tail**: the smallest round value above $q_{99}$. A
   threshold below $q_{99}$ clips ordinary steps and silently rescales the whole update; one far
   above the maximum leaves the non-finite guard as the only protection.
5. **Write the chosen value into `configs/default.yaml`**, replacing the `PROVISIONAL` marker in the
   comment above `gradient_clip_val` with the measured quantiles and the run stamp they came from.
   The comment is the record; this table is the working.

Measured over the baseline run, whose checkpoint records $452{,}250$ optimizer steps at epoch
$1004$ — $450$ steps per epoch, so the $2{,}000$-step window of step 1 is its first five epochs.
The last column is the value written back into `advanced_config.trainer.gradient_clip_val`.

| Window | $q_{50}$ | $q_{90}$ | $q_{99}$ | max | fraction $> 250$ | chosen threshold |
|---|---:|---:|---:|---:|---:|---:|
| first 2,000 steps | 1,964 | 2,529 | 2,760 | 2,786 | 1.000 | 5,000 |
| steady state (after the ramp) | 2,777 | 3,613 | 4,683 | 7,313 | 1.000 | 5,000 |

**Read the fraction column first.** It is $1.000$ in both windows: every recorded step of the run
exceeded the threshold it was training under, and the smallest value anywhere in the run, $703$,
is still nearly three times it. That is not a clip protecting against blow-ups, it is
normalised-gradient descent — the update direction survives and its magnitude is discarded, at an
effective learning rate of roughly a eleventh of the configured $3 \times 10^{-4}$. $5000$ leaves
$0.5\%$ clipping.

**What the sample above actually is.** `metrics_history.csv` carries **one optimizer step per
epoch** for this metric, not the epoch's aggregate: `MetricsLoggingCallback` reads
`trainer.callback_metrics` at validation end, which is before the training epoch is reduced, so
for a metric logged `on_step=True, on_epoch=True` from `on_before_optimizer_step` the bare key
still holds the last step's value. The $1{,}019$ rows are therefore $1{,}019$ individual steps
thinned from $452{,}250$ — which is the distribution a *per-step* threshold should be set from,
so the derivation is on firmer ground than an epoch mean would have put it. The same applies to
`train/grad_clip_frac`: each row is one step's exceedance indicator, so it reads $0$ or $1$ per
row and its **mean over epochs** is the estimate of the per-step exceedance fraction. Every
`val/` column in this file is a true epoch mean; only these two are step samples.

Bounds from the CPU smoke fits on the committed four-sample shard at tiny widths — they bound the
*scale*, not the distribution, and are two orders below the production reading above, which is
what the caveat in `DESIGN.md` §9 predicted:

| | epoch 0 | epoch 1 | epoch 2 | max |
| --- | ---: | ---: | ---: | ---: |
| unguarded (`null`) | 95.0 | 107 | 113 | 113.3 |
| guarded, 120 s | 122 | 87.5 | 76.4 | 122.3 |

The change was validated on the sibling package's dev box before it shipped, at fixed
`beta_prior`, over the committed HIE sample shard; see that package's `RESULTS.md`.

## Distributed smoke, memory and throughput

To record before any multi-day run. Throughput is a headline claim of this architecture — removing
the recurrence removes the serialisation — so it is measured rather than asserted, at the same
effective batch as the comparison model.

- [ ] A short multi-rank `torchrun` completes several steps without deadlock (sampler sharding and
      the permutation-control rank reduction exercised).
- [ ] If memory binds, the escalation order is `batch_size` $32$ with `accumulate_grad_batches` $4$
      holding the effective batch, and only then gradient checkpointing on the target stack (which
      would need `use_reentrant=False` to survive DDP). Never a narrower model or a shorter context.
      Record which levers were applied and the measured effect of each.

| Model | `params` | `peak_memory_gb` | `examples_per_s` | `steps_per_s` |
|---|---:|---:|---:|---:|
| this model (`default.yaml`) | 4,996,844 | | | |
| comparison model (A0) | 5,088,186 | | | |

Results: _pending._

## Headline baseline

The shipped configuration trained to convergence, beside the comparison model on the same shards.
This is the run the module exists for, and `nll_base_block` is the number the whole comparison turns
on: does the stronger target prior lower $D_0$?

- [ ] Both models trained to a stated minimum epoch count on the same shards, same seed.
- [ ] The verdict on $D_0$, with the difference and its sign.
- [ ] Whether the KL moved, and — the part that decides it — whether `pred_gap` moved with it.
- [ ] Latent-collapse verdict under the criterion above.
- [ ] `anchor_coverage_frac` distribution inspected; `coverage_floor: 0.9` confirmed or revised.
- [ ] Log-variance distributions inspected; `logvar_clamp: [-5, 3]` confirmed or revised.
- [ ] Observed `main_loss` scale; the spike breaker's `additive_margin` re-derived from it
      (provisional 1.0e+3 confirmed or revised).

| Model | `epochs` | `nll_base_block` | `nll_full_block` | `pred_gap` | `source_conditioned_kl_raw` | `collapsed` |
|---|---:|---:|---:|---:|---:|---|
| this model (A3, `default.yaml`) | | | | | | |
| comparison model (A0) | | | | | | |

Results: _pending._

## Bottleneck health

The six columns are the full watch list, recorded for the baseline and for every arm that is quoted.
They exist because a headline number can look healthy while the bottleneck is not: a prior variance
pinned on its clamp inflates the KL, a latent collapsed into one dimension holds the total KL up
while carrying nothing, and a tanh-bounded head sitting on its bound is a silently mis-set
hyperparameter. All six are emitted by the task from epoch 0.

| Arm | `source_conditioned_kl_raw` | `kld_active_frac` | `mu_post_prior_gap_rms` | `logvar_prior_floor_frac` | `mu_prior_sat_frac` | `delta_mu_sat_frac` |
|---|---:|---:|---:|---:|---:|---:|
| `default.yaml` (A3) | | | | | | |

Results: _pending._

## Arm inventory

Every arm is `default.yaml` plus its declared delta and nothing else, linted by
`tests/test_sweep_configs.py`. A0 is the comparison model, not a file here.

| Config | Delta against `default.yaml` | `delta_params` | Question |
|---|---|---:|---|
| `sweep_arch_a1.yaml` | no conv stem; source 6 blocks, unbounded | +586,240 | is the convolutional bias worth anything? |
| `sweep_arch_a2.yaml` | source 6 blocks, unbounded | +787,968 | is the source encoder's asymmetry worth anything? |
| `sweep_window_8.yaml` | `source_attention_window` 8 | 0 | sharper lag identity, weaker source state? |
| `sweep_window_32.yaml` | `source_attention_window` 32 | 0 | the first arm whose reach passes the lag range |
| `sweep_window_64.yaml` | `source_attention_window` 64 | 0 | the diffuse end of the trade |
| `sweep_window_full.yaml` | `source_attention_window` null | 0 | the negative control for the locality argument |
| `sweep_reach_null.yaml` | `causal_reach_budget_s` null | −6,272 | what was the shipped guard buying, and what did the leak cost? |
| `sweep_base_sample.yaml` | `base_decode` sample | 0 | does the base branch's noise drive the prior-variance collapse? |
| `sweep_logvar_residual.yaml` | `posterior_logvar_mode` residual | 0 | does the shared raw log-variance drive it? the path that survives `base_decode: mean` |
| `sweep_source_dropout_0p2.yaml` | `source_dropout` 0.2 | 0 | does regularising the source pathway alone close the train-minus-held-out gap? |
| `sweep_source_dropout_0p3.yaml` | `source_dropout` 0.3 | 0 | the same, harder; the pair brackets the strength |
| `sweep_target_blocks_4.yaml` | `target_attention_blocks` 4 | −525,312 | the pre-revision depth: was raising the prior to six blocks needed? |
| `sweep_target_blocks_8.yaml` | `target_attention_blocks` 8 | +525,312 | does a deeper prior lower $D_0$ further? |
| `sweep_source_blocks_2.yaml` | `source_attention_blocks` 2 | −262,656 | how shallow can the source encoder be? |
| `sweep_source_blocks_4.yaml` | `source_attention_blocks` 4 | +262,656 | the deepest source encoder still inside the lag range |
| `sweep_ff_384.yaml` | `encoder_d_ff` 384 | −442,368 | the pre-revision-adjacent width: is the block's position-wise capacity what binds? |
| `sweep_aux_off.yaml` | `lambda_ms`, `lambda_deriv`, `lambda_boundary` all 0.0 | 0 | what did shaping the forecast mean beyond the per-sample likelihood buy? read in nats, not on `total_loss` |
| `sweep_horizon_attn_off.yaml` | `horizon_attention_blocks` 0 | −525,314 | does mixing all 30 horizon tokens at once beat the dilated stack that already spans them? |
| `sweep_beta_prior_0p001.yaml` | `beta_prior` 1.0e-3 | 0 | the weakest anchor; expected to pin, and the arm that says how far below the threshold it is |
| `sweep_beta_prior_0p01.yaml` | `beta_prior` 1.0e-2 | 0 | the weight this package shipped first, since measured to fail on the sibling — does the threshold sit in the same place here? |
| `sweep_beta_prior_0p1.yaml` | none — restates the shipped `beta_prior` 0.1, pinned | 0 | the shipped weight; the baseline run stands in |
| `sweep_beta_prior_1p0.yaml` | `beta_prior` 1.0 | 0 | the upper bracket, at parity with the converged KL weight: does a stronger anchor start taxing the base forecast? |

$d = 128$ is fixed across every arm: it is the width the prior head, the posterior fusion, the lag
attention's key-value projections and the decoder input all assume.

## Phase 1 — architecture

A1 and A2 differ by exactly the stem's 201,728 parameters, which is the comparison the pair exists
to make. A1 against `default.yaml` is **not** the stem cost, because it also gives the source
encoder a fourth block.

- [ ] Each arm trained to a stated minimum epoch count against the baseline; metrics CSVs retained.
- [ ] Does the conv stem earn its 201,728 parameters (A2 against A1)?
- [ ] Does the source encoder's asymmetry cost anything (A2 against A3)?
- [ ] Verdict: which arm becomes the baseline for Phases 2 and 3.

| Arm | `params` | `epochs` | `nll_base_block` | `pred_gap` | `source_conditioned_kl_raw` | `kld_active_frac` | `collapsed` |
|---|---:|---:|---:|---:|---:|---:|---|
| A1 (`sweep_arch_a1.yaml`) | 5,583,084 | | | | | | |
| A2 (`sweep_arch_a2.yaml`) | 5,784,812 | | | | | | |
| A3 (`default.yaml`) | 4,996,844 | | | | | | |
| A0 (comparison model) | 5,088,186 | | | | | | |

Results: _pending._

## Phase 2a — source locality

The sweep spans the regime change: the shipped 66-step reach sits inside the 90-step lag search
range and the two widest arms sit outside it. Whether the bounded window sharpens lag attribution
measurably, or the effect is inside seed noise, is the open question this sweep exists to answer —
and the lag-sharpness measurement itself belongs to the evaluation, which is planned separately, so
record the runs and the forecast columns now and the sharpness verdict when it lands.

- [ ] Each arm trained to a stated minimum epoch count; metrics CSVs retained.
- [ ] Forecast verdict per window, on `nll_base_block` and `pred_gap`.
- [ ] Lag-sharpness verdict — **deferred**, pending the evaluation; the runs are what it will read.

| Arm | `source_reach` | `epochs` | `nll_base_block` | `pred_gap` | `source_conditioned_kl_raw` | `collapsed` |
|---|---:|---:|---:|---:|---:|---|
| `sweep_window_8.yaml` | 42 | | | | | |
| `default.yaml` (16) | 66 | | | | | |
| `sweep_window_32.yaml` | 114 | | | | | |
| `sweep_window_64.yaml` | 210 | | | | | |
| `sweep_window_full.yaml` | unbounded | | | | | |

Results: _pending._

## Phase 2b — the causal input budget

The arm the availability representation exists to make runnable. At the 120 s budget the target
adapter reads 78 of 109 declared channels and the source 29 of 58, at a worst delay of 30 steps —
exactly the shipped `warmup_period`, so this is the deepest admissible budget and the hardest case
for the representation.

- [ ] The run completes with `train/grad_norm` finite throughout, and its distribution recorded
      beside the unguarded baseline's — the measurement that turns "the defect is fixed" from a
      smoke-shard observation into a production one.
- [ ] Whether the coupling readout survives the budget: the condition under which a causal reading
      of this KL becomes defensible at all.
- [ ] The surviving channel counts cross-checked against the startup log.

The direction to expect, stated in advance so a confirming result is not read as a failure:
`nll_base_block` should be **worse** under the shipped guard than under the null arm. The guarded
target branch has lost up to 974 s of lookahead it was quietly using, and that is the guard
working. The quantity to read is `pred_gap` and the source margin: if the source's contribution
was being subsumed by the target branch's leak, the gap improves under the guard even though both
branches score worse in absolute terms.

| Arm | Channels kept (tgt/src) | `epochs` | `nll_base_block` | `pred_gap` | `source_conditioned_kl_raw` | max `grad_norm` | `collapsed` |
|---|---|---:|---:|---:|---:|---:|---|
| `default.yaml` (120 s) | 78 / 29 | | | | | | |
| `sweep_reach_null.yaml` | 109 / 58 | | | | | | |

Results: _pending._

## Phase 3 — depth and width

Each arm is a single-key delta, and the parameter cost of each is exact: 262,656 per attention block
either way, 442,368 for the feed-forward narrowing across all nine blocks.

- [ ] Each arm trained to a stated minimum epoch count; metrics CSVs retained.
- [ ] Target depth: does $N_Y = 8$ lower $D_0$ enough to justify 525,312 parameters, and does
      $N_Y = 4$ -- the pre-revision depth -- cost anything?
- [ ] Source depth: does $N_U$ move `pred_gap` at all, at a fixed window?
- [ ] Is $d_{\mathrm{ff}} = 512$ earning its width at $d = 128$? The 384 arm is the test.

| Arm | `delta_params` | `epochs` | `nll_base_block` | `pred_gap` | `source_conditioned_kl_raw` | `kld_active_frac` | `collapsed` |
|---|---:|---:|---:|---:|---:|---:|---|
| `sweep_target_blocks_4.yaml` | −525,312 | | | | | | |
| `sweep_target_blocks_8.yaml` | +525,312 | | | | | | |
| `sweep_source_blocks_2.yaml` | −262,656 | | | | | | |
| `sweep_source_blocks_4.yaml` | +262,656 | | | | | | |
| `sweep_ff_384.yaml` | −442,368 | | | | | | |

Results: _pending._

## The decoder-side pair

Two ablate-one arms for the two mechanisms the capacity revision added below the encoders: the
auxiliary shape terms in the criterion, and the self-attention over the horizon tokens in the
shared decoder core. Both aim at the same defect from opposite ends — the per-sample Gaussian
likelihood's optimum is the conditional mean, and a fully parallel decoder is free to emit an
over-smoothed one — so they are read as a pair. An arm that loses nothing says its half was not
what closed the gap.

**Read this table in nats.** `sweep_aux_off.yaml` optimises a different criterion from the
baseline's, so its `total_loss` is not comparable with the baseline's at all; `nll_full_block` and
`nll_base_block` are, and they are what the columns carry. The three `aux_*` columns are quoted for
the **baseline** only: at weight $0$ a term is not computed and reports exact $0.0$, so the arm
cannot report what it gave up.

- [ ] Both arms trained to the same minimum epoch count as the baseline; metrics CSVs retained.
- [ ] The shape terms' verdict on `nll_full_block`, and on the per-epoch diagnostic figure — a
      forecast that scores the same in nats and looks visibly flatter is the outcome they exist to
      prevent, and it is not visible in any column here.
- [ ] The horizon attention's verdict, against its 525,314 parameters.
- [ ] The baseline's three `aux_*` magnitudes against `main_loss`, which is what re-derives the
      three provisional weights.

| Arm | `delta_params` | `epochs` | `nll_base_block` | `nll_full_block` | `pred_gap` | `aux_multiscale` | `aux_derivative` | `aux_boundary` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `default.yaml` (both on) | 0 | | | | | | | |
| `sweep_aux_off.yaml` | 0 | | | | | 0 | 0 | 0 |
| `sweep_horizon_attn_off.yaml` | −525,314 | | | | | | | |

Results: _pending._

## The prior-anchor weight

The four `sweep_beta_prior_*` arms bracket the shipped $10^{-1}$ over three orders of magnitude,
on the architecture in which the prior-variance collapse was measured. The shipped value was
revised from $10^{-2}$ on the sibling's dev-box evidence below; these arms are what confirm or
revise it *here*, which is the measurement that decides it. The reading order is fixed
before any run: **first** `logvar_prior_floor_frac` and `mean_logvar_prior` — did the arm hold
the prior off its clamp floor at all; **then** `nll_base_block` against the reference — did the
anchor tax the base forecast it is designed to leave alone; **only then** the coupling columns.
An arm that fixes the floor fraction but not the amplification ratio has not demonstrated the
fix, so `abs(pred_gap)/K` is a first-class column rather than an afterthought. Training columns
come from each run's `metrics_history.csv` at convergence; `pred_gap`, `source_margin` and $K$
come from the evaluation pass over each arm's checkpoint.

- [ ] Every arm records the full column set; a collapsed arm is marked, not dropped.
- [ ] At least one arm reports `logvar_prior_floor_frac` below 0.5 at convergence.
- [ ] The `nll_base_block` column shows what each weight cost the base forecast.
- [ ] The amplification ratio is recorded per arm and compared against the reference 4.3.

**What the sibling package's dev-box validation already showed**, on the committed HIE sample
shard at the shipped geometry — evidence for where to look, not a substitute for these arms, which
run on the architecture the collapse was measured on. At $10^{-2}$ the anchor **delays** the
collapse without preventing it: the floor crossing moved by $6.7\times$ in optimizer steps and the
prior then pinned anyway, finishing at $0.955$ against an unanchored $0.978$. At $10^{-1}$ it
held, at a floor fraction of $0.046$ and a `kld_active_frac` of $0.69$ against $0.25$ in every
collapsed arm. The reason it behaves as a threshold rather than a dial is that the restoring force
$\tfrac{1}{2}(e^{\mathrm{lv}_p} - 1)$ saturates at $\tfrac{1}{2}\beta_{\mathrm{prior}}$ per
dimension while the reconstruction's opposing pressure grows as the decoder sharpens. So expect
`sweep_beta_prior_0p001` and `sweep_beta_prior_0p01` to pin and the reading to be decided between
$10^{-1}$ and $1.0$ — and if $0p1$ pins here too, that is the interesting result, because it
would mean the threshold moves with the architecture.

The reference row is the unanchored 1018-epoch production run (columns are the mean over epochs
900–1018; `prior_rate` was not yet tracked, and its verdict triple is implied by the recorded
columns: no predictive improvement, positive source margin, so no source specificity).

| Arm | `beta_prior` | `logvar_prior_floor_frac` | `mean_logvar_prior` | `prior_rate` | `nll_base_block` | `pred_gap` | `abs(pred_gap)/K` | `source_margin` | $K$ | `kld_active_frac` | Verdicts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| unanchored reference | 0 | 0.992 | −4.986 | — | −197.19 | −7.37 | 4.3 | +1.06 | 1.70 | 0.292 | FAIL / PASS / FAIL |
| `sweep_beta_prior_0p001.yaml` | 1.0e-3 | | | | | | | | | | |
| `sweep_beta_prior_0p01.yaml` | 1.0e-2 | | | | | | | | | | |
| `sweep_beta_prior_0p1.yaml` (shipped) | 0.1 | | | | | | | | | | |
| `sweep_beta_prior_1p0.yaml` | 1.0 | | | | | | | | | | |

**The sibling's dev-box readings for the two middle arms**, on the committed HIE shard at 200
epochs, in-sample, on the convolutional-recurrent architecture — evidence for where to look, not a
substitute for these rows:

| `beta_prior` | `logvar_prior_floor_frac` | `mean_logvar_prior` | `kld_active_frac` | verdict |
|---:|---:|---:|---:|---|
| 1.0e-2 | 0.955 | −4.934 | 0.248 | pinned; a 6.7x delay in steps and nothing more |
| 0.1 | 0.046 | −3.332 | 0.692 | held |

Results: _pending._
