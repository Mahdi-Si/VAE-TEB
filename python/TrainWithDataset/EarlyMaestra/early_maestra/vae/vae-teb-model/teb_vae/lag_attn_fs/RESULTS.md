# Feature-domain forecaster — results

Status: **the local validation is complete; the production runs are a separate, manual effort.**
Every number in this file comes from a training run, never from the test suite. The filled tables
below are dev-box measurements on the committed HIE sample shard, marked as such in every row: they
exist to read this objective's acceptance criteria at the real widths before production time is
spent, not to stand in for a production run.

**The KL-weight sweep moved the shipped defaults underneath the arms that measured them.** The
package shipped $\beta = 5.0$ and $\beta_{\mathrm{prior}} = 0.5$, scale-matched to this objective's
$4.9\times$ larger reconstruction by argument rather than by measurement; the four arms are monotone
in $\beta$ on every column the selection rule reads, and the lower bracket won. `default.yaml` now
ships $\beta = 1.0$ and $\beta_{\mathrm{prior}} = 0.1$. **Every row below therefore states its own
$\beta$ and $\beta_{\mathrm{prior}}$**, because re-running the table means naming both per arm and
not re-running one config file.

There is **no evaluation pipeline for this package** — deliberately, and the consequence is
structural rather than cosmetic. Every column here is read from a run's own
`train_results/metrics_history.csv`. There are no verdicts, no bootstrap confidence intervals, no
trivial-predictor baselines, no calibration and no per-recording tables, and none of the rows below
carry an uncertainty. A difference between two arms in this file is a difference between two point
estimates from single runs.

## What this study is comparing, and the rule for reading it

This model is the missing cell of a $2 \times 2$: the raw-signal sibling `teb_vae/lag_attn_rws`
changed the latent factorisation **and** the target domain in one step, and this one holds the
factorisation and moves only the domain. Against `teb_vae/lag_attn` it isolates the removal of the
decoder bypass; against `lag_attn_rws` it isolates the target domain. Nothing in this file is a
remedy for anything: the raw models' negative held-out gain lives in the source encoder, the lag
attention and the posterior fusion, none of which a target-domain swap touches.

Four rules govern how every table below is read, and all four are stated before the first table on
purpose.

**A negative `pred_gap` is a PASS.** It is the predicted outcome, not a failure to be tuned away.
What would instead indicate a *build* error is `pred_gap` identically zero — the two branches would
then be decoding the same latent — or `abs(pred_gap)` exceeding `nll_base_block` in magnitude, which
no difference between two forecasts of the same block can honestly reach. Neither is a
hyperparameter question, and an arm producing either is stopped and diagnosed rather than recorded.

**Do not select on KL magnitude.** A larger `source_conditioned_kl_raw` is not a better model. The
rate can rise because the latent is carrying more of the *target's* own future, which is exactly
what a weakened $\beta$ produces and exactly what an in-sample run cannot distinguish from coupling.
Read the rate beside `shuffle_penalty` and `nll_shuffled_block`, which is the only control in the
training metric surface that asks whether the source the latent is conditioned on had to be *this*
recording's.

**The nats are comparable to other arms of this model at this reach budget and to nothing else.**
The reconstruction sums over $H \cdot C_{\mathrm{keep}} = 30 \times 78 = 2340$ coefficients against
the raw sibling's $H \cdot R = 480$ samples, and the factorised Gaussian over correlated wavelet
coefficients overcounts independent information either way. `causal_reach_budget_s` moves
$C_{\mathrm{keep}}$, hence the decoder width, hence the block every nat is summed over — so two
arms of *this* model at different budgets are as incomparable as this model and the raw one, and
their checkpoints will not load into each other.

**A short horizon is partly reconstruction, and the split columns are what separate it from
forecasting.** A stored coefficient at decimated step $s$ is a weighted average over a window
*centred* at raw index $16s$, so at horizon step $\tau = 0$ half of the target's support lies in
signal the model has already observed and by $\tau = 29$ none of it does. That is not leakage — no
future information enters the model — but a summed scalar cannot tell a model forecasting the clean
tail from one reconstructing the blended head. `pred_gap_tau_first` against `pred_gap_tau_last` is
that reading, and `pred_gap_st` against `pred_gap_ph` is the same question along the channel axis,
where the two stored blocks' filters have different reaches.

## Pre-registered acceptance criteria

Registered before the runs, so the result cannot be chosen after the fact. Read at convergence over
the last ten epochs unless a row says otherwise.

| # | Criterion | Read from | Verdict |
|---|---|---|---|
| 1 | the total recomposes from its four weighted terms | `total_loss` against `nll_full_block` + `nll_base_block` + `kld_beta` · `source_conditioned_kl_train` + `beta_prior` · `prior_rate` | **PASS**, all four arms: relative error $4 \times 10^{-9}$ to $8 \times 10^{-8}$ on both stages |
| 2 | the KL opens off zero and stays open | `source_conditioned_kl_raw` above $0.05$ nats/anchor by epoch $20$, and not falling across the last five | **PASS**, all four: $0.69$–$4.89$ at epoch $20$, and every arm's last-five mean is *above* its previous five |
| 3 | the conditional prior stays off its clamp floor | `logvar_prior_floor_frac` below $0.2$ | **PASS**, all four: $0.0121$ at $\beta_{\mathrm{prior}} = 0.1$ and exactly $0$ at every stronger anchor |
| 4 | the decoder's log-variance is inside its clamp | `mean_logvar_full` at least $0.5$ above the floor of $-5$ | **PASS**, all four: margins $3.87$–$3.89$ |
| 5 | the forecast gap is a number, not an artefact | `pred_gap` not identically zero, and `abs(pred_gap)` below `nll_base_block` | **PASS**, all four: $\lvert$gap$\rvert$ from $1.20$ to $15.38$ against a base block of $\approx 1950$ |
| 6 | the clamp itself is read rather than assumed | `logvar_full_floor_frac`, `logvar_full_ceil_frac` | **read**: $0.0134$–$0.0140$ at the floor, $0.00026$–$0.00030$ at the ceiling. See below |

No arm is collapsed under the shared criterion, no arm skipped a batch, and `total_loss` is finite in
every epoch of every arm.

**Criterion 1 is a relative tolerance, and has to be.** The unit test that owns this identity
asserts it with `torch.allclose(rtol=1e-6, atol=1e-6)`, where the relative term is what binds at any
realistic loss. Read from a run it can only be relative: `main_loss` here is a few thousand nats and
the CSV records float32, whose spacing at $4 \times 10^{3}$ is already $\approx 5 \times 10^{-4}$ —
an absolute $10^{-6}$ is below one representable step and no correct implementation could meet it.
The reading is $|{\rm total} - \sum {\rm terms}| / |{\rm total}| \le 10^{-6}$.

**Criterion 2's "not falling" is a statement about the tail, not about consecutive epochs.** The
per-epoch rate at this shard size is noisy — eleven optimizer steps stand behind each row — so the
reading is the last five epochs' mean against the five before it, and a criterion satisfied by a
sequence that happens to descend monotonically through noise would be measuring the noise.

## Launch lines

From the repository root.

```bash
# Dev box: the local validation, one device, the committed HIE shard
python -m teb_vae.lag_attn_fs.trainer --config teb_vae/lag_attn_fs/configs/smoke_hie.yaml

# Production box: the baseline and the four KL-weight arms
STAMP="$(date '+%Y-%m-%d--[%H-%M]')"
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_fs.trainer --config teb_vae/lag_attn_fs/configs/default.yaml
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_fs.trainer --config teb_vae/lag_attn_fs/configs/sweep_beta_1p0.yaml
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_fs.trainer --config teb_vae/lag_attn_fs/configs/sweep_beta_2p5.yaml
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_fs.trainer --config teb_vae/lag_attn_fs/configs/sweep_beta_5p0.yaml
TEB_RUN_STAMP="$STAMP" torchrun --nproc_per_node=7 -m teb_vae.lag_attn_fs.trainer --config teb_vae/lag_attn_fs/configs/sweep_beta_10p0.yaml
```

`sweep_beta_1p0.yaml` restates the shipped weights, so it differs from `default.yaml` in nothing and
the baseline run stands in for it. It is a file rather than a note because it keeps the comparison
that chose the value runnable after the fact — which is exactly what happened here: `sweep_beta_5p0`
was the restating arm when the sweep was written, `default.yaml` moved to the sweep's winner, and the
two swapped roles without either file changing a number.

**The local arms are `smoke_hie.yaml` plus the same two keys**, not the sweep files: those inherit
`default.yaml` and with it the production shard paths, the seven-device list and the production
epoch count. Each local arm is a two-key overlay on `smoke_hie.yaml` —

```yaml
base: <repo>/teb_vae/lag_attn_fs/configs/smoke_hie.yaml
general_config:
  tag: lag_attn_fs_hie_b1p0
  folders_config: {out_dir_base: output/lag_attn_fs_hie_b1p0}
model_config:
  VAE_model:
    beta_schedule: {end: 1.0}
    beta_prior: 0.1
```

— and nothing else moves. The overlay itself is a scratch file and is **not** the record: the
record is the fully resolved `resolved_config.yaml` the run writes beside its own checkpoints, in
which every inherited value is explicit and from which each row below is keyed. The tag and output
directory are per-arm here, unlike on the production box, because four dev-box runs launched inside
the same hour would otherwise be indistinguishable by the only field anything indexes on.

## Where the numbers come from

Three sourcing rules, so the copy from a run directory into this file is safe.

1. **Rows are keyed by the swept value read from each run's own `resolved_config.yaml`**, never from
   a directory name — a renamed directory cannot then relabel a measurement.
2. **Per-epoch series come from `train_results/metrics_history.csv`.** Quote the mean over the last
   ten epochs unless a column says otherwise; a single final epoch at eleven steps per epoch is one
   draw, not a converged value.
3. **A collapsed arm is marked, not dropped**, and an arm missing its CSV is reported as incomplete
   rather than silently left rowless.

**The collapse criterion** is the shared one in `teb_vae/lag_attn_rws/collapse.py`, applied unchanged
because the objective is unchanged: a completed run is collapsed when `source_conditioned_kl_raw` is
below $0.02$ nats per anchor at every one of its last $5$ epochs, or when its final
`kld_active_frac` is below $2 / d_z$. The criterion reads the tail only, because every run here opens
at exactly zero KL by construction.

**Derived quantities.** Column names that are not tracked metrics, defined once here so no table can
introduce an undefined one:

- `epochs` — completed epochs, from the metrics CSV.
- `steps` — optimizer steps: `epochs` $\times\, 11$ on the local shard at batch $32$, since $339$
  windows at that batch is eleven batches and `accumulate_grad_batches` is $1$.
- `params` — `sum(p.numel() for p in model.parameters())` on the constructed model.
- `collapsed` — the verdict of the criterion above.
- `source_margin` — the training-side aggregate `nll_shuffled_block` $-$ `nll_full_block`: positive
  means the matched source beat a derangement-shuffled stranger. The evaluation's per-recording
  paired version, which is the one that carries a confidence interval, does not exist for this
  package.
- `abs(pred_gap)/K` — the amplification ratio, with $K$ the val `source_conditioned_kl_raw`: nats of
  forecast movement per nat of source-conditioned divergence. Read together with the **sign** of
  `pred_gap`, since the same magnitude is degradation where the gap is negative and gain where it is
  positive.

**Every `val/` column on the local shard is in-sample.** `dataset_kwargs` is shared between the two
loaders and cannot carry a per-split GUID filter, so both splits are the same $339$ windows. That is
adequate for the six criteria above, all of which are optimisation diagnostics, and it is why no
local row claims a coupling result.

## Before launching: what reverts, and when to stop

Written before the production time is spent, so an arm that goes wrong is stopped against a
threshold agreed in advance, and so an operator can tell at a glance which mistakes cost a config
edit and which cost a retrain.

### What reverts, and how

| Change | Where it lives | Revert path | What reverting costs |
|---|---|---|---|
| The converged KL weight at $1.0$ | `model_config.VAE_model.beta_schedule.end` in `configs/default.yaml`, restated by each `sweep_beta_*` arm | **Config.** The four arms are the revert path, and `sweep_beta_5p0.yaml` is the scale-matched value this package shipped first | The arm restarts from epoch 0. Nothing else. |
| The prior scale anchor at $0.1$ | `model_config.VAE_model.beta_prior` | **Config**, but never alone: what the design fixes is the ratio `beta_prior` / `beta_schedule.end` $= 0.1$, and moving one without the other sweeps two axes at once. `0.0` multiplies the fourth term by zero and leaves the three-term objective exactly; `prior_rate` is written either way, so a collapsing prior stays visible in an unanchored run's CSV | The arm restarts from epoch 0. |
| The spike breaker's `additive_margin` at $5 \times 10^{3}$ | `advanced_config.spike_breaker.additive_margin` | **Config.** It is stated in nats of the summed block and does not transfer across block sizes; the derivation is the instrumented run below | The arm restarts. Setting it too low is the expensive direction: a breaker that skips every batch trains nothing and reports an EMA in place of the loss. |
| The gradient clip at $5000$ | `advanced_config.trainer.gradient_clip_val` | **Config**, but re-derive rather than revert: the threshold is measured from a run's own pre-clip `grad_norm` percentiles and a value carried from elsewhere is the mistake this package already avoided once | The arm restarts. |
| The decoder width | **nothing directly.** It follows `causal_reach_budget_s` through the resolved survivor set | **None by configuration that keeps the run comparable.** Changing the budget changes $C_{\mathrm{keep}}$, hence the block, hence the units of every nat in this file | A full retrain *and* a re-baselining: no earlier row is comparable to the new ones, and no earlier checkpoint loads. |
| The four forecast-gap columns | the tracked metric surface, `LagAttnFsTrainer.TRACKED_METRICS` | **None, and none needed** — they are additive | A `metrics_history.csv` written before them simply lacks them. |

Two consequences worth stating on their own.

**A budget change is not an arm.** Every other row above restarts a run; that one invalidates the
file. The reach budget is the only key in this configuration whose movement changes what the numbers
*mean* rather than what they are, and it is held fixed at $120$ s across everything below.

**An in-flight run cannot be moved between initialisation policies.** `head_init_calibration` is
applied once at construction and baked into the weights, so an arm at the uncalibrated start is a
separate launch from step 0, never a resume with the key flipped. It also drives *two* halves — the
decoder output heads and the prior head's log-variance — and there is deliberately no key for one
alone.

### Go/no-go while a run is in flight

Three signals, all in the run's own `train_results/metrics_history.csv`, all present from epoch 0.

| Signal | Healthy | Hold and inspect | Stop the arm |
|---|---|---|---|
| `spike_skipped` | $0$ in every epoch | any single non-zero epoch | non-zero in 3 or more epochs, or a consecutive run of skips reaching the `max_consecutive_skips` escape hatch of $25$ |
| `grad_clip_frac` | mean $\lesssim 0.05$ | mean above $0.25$ over any 20 consecutive epochs | $1.000$ over any 20 consecutive epochs |
| `val/total_loss` | falling, or flat within a fifth of a nat per anchor per 20-epoch window | a 20-epoch window mean above the window 20 epochs earlier, at any point after the beta ramp ends | any non-finite value, or three consecutive windows over that threshold |

All three thresholds are now measured on this objective rather than carried.

**`spike_skipped` and `grad_clip_frac`.** Across the four validation arms — $800$ epochs, $8{,}800$
optimizer steps — the breaker never fired once, and the clip fired on three sampled steps in total,
for whole-run `grad_clip_frac` means of $0.000$, $0.000$, $0.010$ and $0.005$. So a single non-zero
`spike_skipped` epoch is new behaviour rather than noise, and the $0.25$ hold band sits
twenty-five-fold above anything this objective has been observed to do.

**`val/total_loss`.** Deliberately tighter than the sibling's $1$ nat, because a one-nat band would be
meaningless against a loss of $4 \times 10^{3}$. Over each arm's $141$ post-ramp 20-epoch windows the
mean rose in **none** of them, at any $\beta$ in the bracket, and the smallest fall anywhere was
$51$ nats — that is, the loss never stopped falling in $564$ windows across a tenfold change in the
KL weight. What the band asks is therefore that the curve still be *descending*, which is a much
stronger property than this objective has yet been observed to lose. Restate it against an arm's own
noise once one plateaus, and record it here.

**What is not an abort signal.** `logvar_prior_floor_frac` is the measurement, not a health check —
the weakest arm is *expected* to pin, that outcome is the reading it exists to produce, and a pinning
arm runs to convergence like the others because its `nll_base_block` and coupling columns are what
the comparison needs. The same holds for the collapse criterion: a collapsed arm is a recorded
result. Stopping either early buys GPU time and loses the row. A negative `pred_gap` is not a signal
either; see the reading rules above.

### When a stop fires

`spike_skipped` — stop, and re-derive `additive_margin` from that run's own `main_loss` fluctuation
by the procedure below rather than reverting to a number from another block size; the breaker firing
in this objective is new behaviour that the instrumented run did not observe. `grad_clip_frac` at
$1.000$ — do **not** revert blindly: repeat the five-step derivation below against that run's own
`grad_norm` percentiles and relaunch at the new value. `val/total_loss` non-finite — stop and
relaunch from the last finite checkpoint.

The third signal is the **validation** column deliberately. The breaker gates the training path only,
and on a batch it skips it replaces `train/total_loss` and `train/main_loss` with its own EMA before
they reach the logger — so a training curve can read finite and healthy while the parameters are not.
`spike_skipped` and the validation columns are what see through that.

## Parameter budget — measured

The one table already filled in, because it is a property of the constructed model. The whole
difference from the comparison model is the decoder's output head: $514 \times (C - 16)$ parameters,
where $514$ is the two per-channel output rows plus their biases at the decoder core's $256$-wide
hidden state. The capacity revision doubled that per-channel cost with `decoder_hidden`, so the
delta doubled with it while remaining exactly the same decomposition.

| Model | Reach budget | Decoder width | `params` | Δ against the comparison model |
|---|---|---:|---:|---:|
| this model (`default.yaml`) | 120 s | 78 | 5,126,326 | +31,868 |
| this model, unguarded | null | 109 | 5,135,988 | +47,802 |
| comparison model (`lag_attn_rws`) | 120 s | 16 | 5,094,458 | — |
| comparison model, unguarded | null | 16 | 5,088,186 | — |

Read the budget column before the delta. The comparison model gains $6{,}272$ parameters when a
finite budget is configured — a guarded run builds availability input adapters — so the $+31{,}868$
delta is against $5{,}094{,}458$ and not against the $5{,}088{,}186$ that `lag_attn_rws/DESIGN.md`
states, which is its *unguarded* model. Note the guarded model here is the **smaller** of the two:
its decoder emits $78$ channels rather than $109$, and at $514$ parameters each that outweighs the
availability adapters it gains.

## The gradient-clipping threshold — measured

`gradient_clip_val` does not transfer across loss scales, and this objective's block is $4.9\times$
the comparison model's. It was therefore re-derived rather than inherited — and **the measurement's
surprise is that the number did not move**: the same rule returns the same $5000$, arrived at
independently. `configs/default.yaml` carries the derivation in the comment above the key; this
section is the working.

The procedure, which is what a re-derivation after any change of scale repeats:

1. **Launch with the clip itself set far above anything reachable** — $10^{9}$ — so nothing rescales
   the steps the norms are drawn from. A derivation run under the incumbent threshold measures the
   threshold.
2. **Read `train/grad_norm`.** The task logs it *pre-clip* in `on_before_optimizer_step`, so it is
   the true norm; there is deliberately no `val/` variant.
3. **Record the quantiles**, the maximum, and the fraction of steps already exceeding the incumbent.
4. **Set the threshold above the healthy tail**: the smallest round value above $q_{99}$. Below
   $q_{99}$ the clip rescales ordinary steps and the run performs normalised-gradient descent at a
   fraction of its configured learning rate; far above the maximum it leaves the non-finite guard as
   the only protection.
5. **Write the value into `configs/default.yaml`**, replacing the provisional marker with the
   measured quantiles and the run they came from. The comment is the record.

Measured over $120$ epochs on the committed `output/hie_cs.hdf5` shard — $339$ windows at batch $32$
is **eleven** optimizer steps per epoch, so $1{,}320$ steps in total, of which the CSV records $120$.

| Window | $q_{50}$ | $q_{90}$ | $q_{95}$ | $q_{99}$ | max | fraction $> 5000$ | chosen threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| the 120-epoch derivation run (clip at $10^{9}$) | 2,147 | 3,156 | 3,416 | 4,421 | 4,741 | 0.000 | 5,000 |
| β 1.0 arm, 200 epochs at the chosen clip | 2,048 | — | — | — | 4,929 | 0.000 | — |
| β 5.0 arm, 200 epochs | 2,050 | — | — | — | 6,033 | 0.010 | — |
| β 10.0 arm, 200 epochs | 2,008 | — | — | — | 5,125 | 0.005 | — |

$5000$ sits just above the derivation run's maximum, and the four validation arms are what say the
threshold is set in the right place rather than merely above one sample. **The clip binds, rarely.**
Across $800$ sampled steps it fired on three of them — $0$, $2$ and $1$ per arm — for a whole-run
`grad_clip_frac` of $0.000$ to $0.010$, twenty-five-fold inside the $0.25$ hold band. That is what a
blow-up guard should look like: it is catching a tail rather than rescaling the update.

The derivation run's $0.000$ exceedance is therefore not a contradiction but the thin-tail caveat the
config comment predicted: $120$ steps thinned from $1{,}320$ on one shard has a shorter tail than
$200$ thinned from $2{,}200$, and the longer runs found the excursions the shorter one did not. The
median is stable to within $2\%$ across all four arms and a tenfold change in $\beta$, so **no
revision to the threshold is recorded.**

**Why a $4.9\times$ larger loss produced the same gradient norms.** The reconstruction sums over $78$
channels, but the decoder's output head is per-channel, so the extra terms land on disjoint rows of
two `Linear` layers rather than accumulating onto one shared parameter. What scales with the block is
the loss value, not the norm of its gradient. That is an argument after the fact; the number is the
measurement.

**What the sample is.** `metrics_history.csv` records **one optimizer step per epoch** for this
metric and for `grad_clip_frac`, not the epoch's aggregate: the metrics collector reads
`trainer.callback_metrics` at validation end, before the training epoch is reduced. So these rows are
per-step draws thinned by epoch — the right distribution to set a per-step threshold from — and
`grad_clip_frac` is a $0$/$1$ exceedance indicator whose **mean over epochs** is the estimate of the
per-step exceedance fraction, not a per-row rate. Every `val/` column in this file is a true epoch
mean; those two are the exception.

**The spike breaker's margin, from the same run.** Over the $100$ post-warm-up epochs `main_loss` sat
at $4562 \pm 404$ (min $3493$, max $5683$) with epoch-to-epoch $|\Delta|$ of median $323$, q90 $745$,
q99 $1316$, max $1500$. `additive_margin` ships at $5 \times 10^{3}$: $3.3\times$ the largest ordinary
movement and $\approx 12\times$ the standard deviation. `ema_floor` stays at $10^{9}$, which switches
the relative test off — and that is confirmed rather than assumed at this block size: the
per-coefficient Gaussian NLL is bounded below by $\tfrac{1}{2}(\log 2\pi + \ell_{\min}) \approx -1.6$
at the shipped clamp floor, so the two reconstruction terms cannot exceed $\approx 7.5 \times 10^{3}$
in magnitude and the KL and the anchor are both nonnegative. Five orders of magnitude of headroom.

## Arm inventory

Every arm is `default.yaml` plus its declared delta and nothing else, linted by
`tests/test_sweep_configs.py`. Each moves **two** keys, and that is the axis rather than a second
delta: the prior anchor's restoring force saturates at `beta_prior` $/\,2$ per latent dimension while
the reconstruction it opposes is exactly what this target domain multiplied, so the ratio
`beta_prior` / `beta_schedule.end` $= 0.1$ is held fixed and a pinning prior has one explanation
rather than two.

| Config | `beta_schedule.end` | `beta_prior` | Question, and what it answered |
|---|---:|---:|---|
| `sweep_beta_1p0.yaml` | 1.0 | 0.1 | the value the comparison model ships: what does a $4.9\times$ weaker rate cost? **Nothing — it wins**, and `default.yaml` now restates it, so the baseline run stands in |
| `sweep_beta_2p5.yaml` | 2.5 | 0.25 | half the scale-matched value; is the bracket finer than the argument that chose it? Yes: it is second on every column, so the axis has real resolution |
| `sweep_beta_5p0.yaml` | 5.0 | 0.5 | the scale-matched value, and what this package shipped first; third on every column |
| `sweep_beta_10p0.yaml` | 10.0 | 1.0 | twice it — the arm that has to fail for the bracket to have found anything. It does: `kld_active_frac` $0.070$ against a collapse threshold of $0.042$ |

The scale-matched point is $2340 / 480 = 4.875$, and the **direction** is the part that is easy to
get backwards: a larger reconstruction at fixed $\beta$ makes $\beta\,\mathrm{KL}$ relatively
*weaker*, so the latent opens **wider**. Two arms sit below that point and two at or above it, which
is what let the sweep find the optimum at the lower edge rather than only bound it — and is the whole
reason the bracket was not built around the inherited $1.0$.

Nothing else is swept. The axes the comparison model sweeps — $d_z$, the reach budget, the
architecture knobs — are deliberately absent: with no evaluation pipeline there is nothing to score
them with that the four columns above do not already say, and the reach budget in particular cannot
be an arm here because it changes what every nat means.

## Headline baseline

The chosen configuration on the committed shard, read against the six pre-registered criteria. Every
figure is the mean over the last ten epochs of a $200$-epoch, $2{,}200$-step run.

| Arm | `epochs` | `steps` | `total_loss` | `nll_base_block` | `nll_full_block` | `pred_gap` | `source_conditioned_kl_raw` | `kld_active_frac` | `collapsed` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| the shipped pair, β 1.0 / β_p 0.1 | 200 | 2,200 | 3900.64 | 1945.89 | 1944.69 | **+1.196** | 3.463 | 0.667 | no |
| the pair this package first shipped, β 5.0 / β_p 0.5 | 200 | 2,200 | 3929.19 | 1951.97 | 1961.18 | −9.202 | 0.510 | 0.317 | no |

Results: **both arms meet every criterion; the lower pair wins the selection rule on every column it
reads.** Read `nll_base_block` first, as the rule says: the base forecast is $6.08$ nats per anchor
*better* at $\beta = 1.0$, so the arm has earned a reading on the gap rather than merely produced one.

**Do not read the positive `pred_gap` as a generalisation result.** These runs are in-sample — both
splits are the same $339$ windows — and the sibling package saw exactly this sign flip between its
in-sample dev-box arms and its held-out production run. What the local runs establish is the base
forecast, the latent width, the prior's standing against its clamp and the *ordering* of the arms.
The sign of the gap is not among them, and section 1.2 of the roadmap predicts a negative held-out
gap that no in-sample run can see.

## Bottleneck health

The full watch list, recorded for every arm. It exists because a headline number can look healthy
while the bottleneck is not: a prior variance pinned on its clamp inflates the KL, a latent collapsed
into one dimension holds the total up while carrying nothing, and a tanh-bounded head sitting on its
bound is a silently mis-set hyperparameter. All six are emitted from epoch 0.

| Arm | `source_conditioned_kl_raw` | `kld_active_frac` | `mu_post_prior_gap_rms` | `logvar_prior_floor_frac` | `mu_prior_sat_frac` | `delta_mu_sat_frac` |
|---|---:|---:|---:|---:|---:|---:|
| β 1.0 / β_p 0.1 (shipped) | 3.463 | 0.667 | 0.836 | 0.0121 | 0.000 | 0.000 |
| β 2.5 / β_p 0.25 | 1.270 | 0.535 | 0.482 | 0.0000 | 0.000 | 0.000 |
| β 5.0 / β_p 0.5 | 0.510 | 0.317 | 0.312 | 0.0000 | 0.000 | 0.000 |
| β 10.0 / β_p 1.0 | 0.234 | 0.063 | 0.278 | 0.0000 | 0.000 | 0.000 |

Results: **the two tanh-bounded latent heads never touch their bounds in any arm**, so `mu_scale` and
`delta_mu_scale` are not binding and neither is a mis-set hyperparameter at this target. The
prior-floor column is the one that behaves against expectation and is discussed below.

**`kld_active_frac` is where the upper arm nearly fails.** At $\beta = 10$ the latent finishes at
$0.070$ against the collapse criterion's $2 / d_z = 0.0417$ — not collapsed, but within a factor of
$1.7$ of a verdict, and its *training* split is already at $0.0375$, below the threshold. That arm is
the one the bracket needed: it says the shipped weight sits under a ceiling and not merely above a
floor, and it says the ceiling is closer than $10\times$ the scale-matched value.

## Forecasting or reconstructing?

The reading the four added columns exist for, and the one this file would otherwise be unable to make
at all. `pred_gap_tau_first` scores the horizon step whose target is half-determined by observed
history; `pred_gap_tau_last` scores the step whose target is not determined by it at all. A gap that
lives only in the first is a model reconstructing the smeared component of its own past. All four are
partial sums of the `pred_gap` beside them, over the same denominator, which is the only property that
makes them worth reporting — and both splits recompose to it.

| Arm | `pred_gap` | `pred_gap_tau_first` | `pred_gap_tau_last` | `pred_gap_st` | `pred_gap_ph` | `shuffle_penalty` |
|---|---:|---:|---:|---:|---:|---:|
| β 1.0 / β_p 0.1 (shipped) | +1.196 | −0.086 | **+0.076** | +0.725 | +0.471 | 13.906 |
| β 2.5 / β_p 0.25 | −3.959 | −0.224 | −0.107 | −1.582 | −2.377 | 5.897 |
| β 5.0 / β_p 0.5 | −9.202 | −0.383 | −0.286 | −4.032 | −5.170 | 2.855 |
| β 10.0 / β_p 1.0 | −15.382 | −0.586 | −0.489 | −6.878 | −8.504 | 1.997 |

Results: **the horizon split does the job it was added for, and it answers the smear objection
directly.** In the shipped arm the gap is *negative* at $\tau = 0$, where half the target's support
lies in observed history, and *positive* at $\tau = 29$, where none of it does. Whatever the source
contributes there, it is not a reconstruction of the blended component — that component is exactly
where the arm does worst. Every other arm is negative at both ends and more negative at $\tau = 0$
than at $\tau = 29$, which is the same ordering.

The block split is the same statement along the channel axis and adds nothing surprising: `fhr_ph`
tracks `fhr_st` in sign in every arm, at roughly its channel-count share. The two blocks' filters
have different reaches, so a large disagreement here would have been the interesting outcome; there
is none.

**`shuffle_penalty` needs its capacity caveat stated, because it cuts the other way.** In absolute
nats the matched source beats a derangement-shuffled stranger by $13.9$ at $\beta = 1.0$ against
$2.0$ at $\beta = 10.0$ — a five-fold difference that tracks the selection. Divided by the rate it
is buying, though, it runs the *opposite* way: $4.0$, $4.6$, $5.6$, $8.5$ nats of margin per nat of
`source_conditioned_kl_raw` as $\beta$ rises. Part of the larger absolute penalty is simply a
higher-capacity latent being perturbed, so the honest reading is that the shipped arm has more
source-specific signal in total and each of its nats is doing slightly less work. Nothing in the
selection rule turns on the ratio; it is recorded so a later reader does not discover it and think
it was hidden.

## The log-variance clamp — read, not assumed

`logvar_clamp: [-5.0, 3.0]` is the one inherited interval that is *going home* rather than being
transplanted: `doc/latex_template/sections/architecture.tex` records that the raw models took it from
a decoder that emitted feature coefficients. These columns are what confirm or revise it here, and
they are per-**coefficient** in this model because `block_width` is $C_{\mathrm{keep}}$ and not the
raw grid's $R$.

| Arm | `mean_logvar_full` | `mean_logvar_base` | `logvar_full_floor_frac` | `logvar_full_ceil_frac` | `mean_logvar_prior` | `logvar_prior_floor_frac` |
|---|---:|---:|---:|---:|---:|---:|
| β 1.0 / β_p 0.1 (shipped) | −1.126 | −1.127 | 0.01404 | 0.00030 | −3.711 | 0.0121 |
| β 2.5 / β_p 0.25 | −1.124 | −1.125 | 0.01382 | 0.00030 | −2.754 | 0.0000 |
| β 5.0 / β_p 0.5 | −1.120 | −1.121 | 0.01387 | 0.00028 | −1.972 | 0.0000 |
| β 10.0 / β_p 1.0 | −1.107 | −1.108 | 0.01343 | 0.00026 | −1.229 | 0.0000 |

Results: **the interval is confirmed, and it is barely moved by $\beta$.** The decoder's mean
log-variance sits at $-1.11$ to $-1.13$ across a tenfold change in the KL weight — $3.88$ above the
floor and $4.11$ below the ceiling — with $1.4\%$ of coefficients on the floor and $0.03\%$ on the
ceiling. Both ends are live rather than dead, which is what a clamp should look like: it is catching
a thin tail rather than shaping the distribution. **No revision to `logvar_clamp` is recorded**, and
the reason it needed reading at all is that the roadmap's own prediction was the opposite — the
interval was expected to be *more* comfortable in the domain it came from, and it is neither more nor
less.

The **prior's** log-variance is the column that moves, and it moves with $\beta_{\mathrm{prior}}$
exactly as the saturating-anchor argument says it should: $-3.71$, $-2.75$, $-1.97$, $-1.23$ as the
anchor strengthens tenfold. Only the weakest anchor puts any coordinate on the floor at all, and at
$1.2\%$ it is six-fold inside the $0.2$ criterion.

## The KL-weight sweep

The four arms on the committed shard, at the local scale. Read in a fixed order, registered here
before any of them runs: **first** whether the arm is collapsed, because a collapsed arm's coupling
columns say nothing; **then** `nll_base_block`, because an arm whose base forecast is worse than the
baseline's has not earned a reading on `pred_gap` at all; **only then** the gap and the rate together.

| Arm | `beta_schedule.end` | `beta_prior` | `epochs` | `nll_base_block` | `pred_gap` | `source_conditioned_kl_raw` | `kld_active_frac` | `source_margin` | `collapsed` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `sweep_beta_1p0.yaml` (**now shipped**) | 1.0 | 0.1 | 200 | **1945.89** | **+1.196** | 3.463 | 0.667 | 13.906 | no |
| `sweep_beta_2p5.yaml` | 2.5 | 0.25 | 200 | 1950.15 | −3.959 | 1.270 | 0.535 | 5.897 | no |
| `sweep_beta_5p0.yaml` (shipped first) | 5.0 | 0.5 | 200 | 1951.97 | −9.202 | 0.510 | 0.317 | 2.855 | no |
| `sweep_beta_10p0.yaml` | 10.0 | 1.0 | 200 | 1963.76 | −15.382 | 0.234 | 0.063 | 1.997 | no |

Results: **monotone in $\beta$ on every column, and the lower bracket wins.** That is the finding,
and the monotonicity is what makes it a finding rather than a draw between two noisy runs: four arms
over a tenfold range order identically on the base forecast, the gap, the rate, the latent width and
the source margin, with no crossing anywhere.

**The scale-matching argument predicted the right direction and the wrong conclusion.** Section 4.8
of the roadmap is correct that a larger reconstruction at fixed $\beta$ makes $\beta\,\mathrm{KL}$
relatively weaker and opens the latent wider — the rate column confirms it, falling from $3.46$ to
$0.23$ as $\beta$ rises. What it inferred from that was that the inherited $\beta$ would leave the
KL *over*-open, so the scale-matched $4.875$ was the value to ship. On this shard the wider latent is
simply better: it buys a better base forecast, a better gap at both ends of the horizon, and a
five-fold larger source margin, and its prior still sits $1.2\%$ from the clamp floor. The argument
bracketed the right axis and put the shipped value on the wrong side of it, which is what the sweep
existed to find out.

**What this does *not* establish.** One shard, in-sample, at $2{,}200$ optimizer steps against a
production run's hundreds of thousands. It establishes an ordering among four arms trained
identically, not a production optimum, and the four `sweep_beta_*.yaml` files exist so that ordering
is re-measured at production scale rather than assumed to survive. The direction to watch is that
in-sample runs reward capacity, so the true optimum on held-out recordings is more likely to sit
*above* $1.0$ than below it — and $2.5$, which is second on every column here, is the arm that would
inherit if it does.

**And it was measured at $d_z = 48$.** The capacity revision moved the shipped latent to $64$, so
every KL in these tables is now summed over a third more dimensions against a reconstruction block
unchanged at $2340$ coefficients, and the collapse criterion $2/d_z$ tightens from $0.0417$ to
$0.03125$ — which moves the upper arm from "within a factor of $1.7$ of a verdict" to a factor of
$2.2$, in the safe direction. The shipped pair is deliberately **not** rescaled for it: that would be
the scale-matching argument again, and the finding above is precisely that this argument names the
right axis and gets the side wrong. It stands as measured, and the four arms are what re-measure it
at the new geometry when fs training resumes.

Run directories, so a row can be traced back to the artefacts that produced it and a superseded
relaunch is not mistaken for a second measurement. All four are dev-box runs of `smoke_hie.yaml` plus
the arm's two keys, launched sequentially on one RTX 4080 between 21:35 and 23:19 on 2026-08-08, at
$\approx 25$ minutes each:

| Arm | Run directory | `epochs` | Superseded by |
|---|---|---:|---|
| β 1.0 / β_p 0.1 | `output/lag_attn_fs_hie_b1p0/2026-08-08--[22-00]-lag_attn_fs_hie_b1p0` | 200 | — |
| β 2.5 / β_p 0.25 | `output/lag_attn_fs_hie_b2p5/2026-08-08--[22-26]-lag_attn_fs_hie_b2p5` | 200 | — |
| β 5.0 / β_p 0.5 | `output/lag_attn_fs_hie_smoke/2026-08-08--[21-35]-lag_attn_fs_hie_smoke` | 200 | — |
| β 10.0 / β_p 1.0 | `output/lag_attn_fs_hie_b10p0/2026-08-08--[22-52]-lag_attn_fs_hie_b10p0` | 200 | — |

The third directory is named for `smoke_hie.yaml` because it *was* that config at the time: the file
resolved to $\beta = 5.0$ when it ran and resolves to $\beta = 1.0$ now. Its
`resolved_config.yaml` is what disambiguates it, which is why rows are keyed from that file and never
from a directory name.

Each arm wrote $20$ diagnostic pages — ten plotted epochs at two drawn samples — and the seven-row
figure renders on the real shard with no callback failure in any of the four runs.

## The prior-anchor weight

`beta_prior` is not swept independently here, and that is a design decision rather than an omission:
it moves with $\beta$ at a fixed ratio for the reason above. What this section records is whether the
transfer worked — whether an anchor scaled to a $4.9\times$ larger reconstruction still holds the
conditional prior off its clamp floor, which is what `logvar_prior_floor_frac < 0.2` asks.

The mechanism is a **saturating restoring force**, so the weight behaves as a threshold rather than a
dial: $\partial R_p / \partial \mathrm{lv}_p = \tfrac{1}{2}(e^{\mathrm{lv}_p} - 1)$ per latent
dimension, which tends to $-\tfrac{1}{2}$ as the log-variance falls, so however far the prior sinks
the anchor never pushes back harder than $\tfrac{1}{2}\,$`beta_prior` per dimension — while the
reconstruction's opposing pressure *grows* as the decoder sharpens. A weight either exceeds that
pressure or is eventually overrun by it; there is no weight that merely slows the descent.

| Arm | `beta_prior` | `logvar_prior_floor_frac` | `mean_logvar_prior` | `prior_rate` | `nll_base_block` | `kld_active_frac` | Held? |
|---|---:|---:|---:|---:|---:|---:|---|
| `sweep_beta_1p0.yaml` (**now shipped**) | 0.1 | 0.0121 | −3.711 | 65.96 | 1945.89 | 0.667 | **held** |
| `sweep_beta_2p5.yaml` | 0.25 | 0.0000 | −2.754 | 43.82 | 1950.15 | 0.535 | held |
| `sweep_beta_5p0.yaml` | 0.5 | 0.0000 | −1.972 | 26.98 | 1951.97 | 0.317 | held |
| `sweep_beta_10p0.yaml` | 1.0 | 0.0000 | −1.229 | 13.00 | 1963.76 | 0.063 | held |

Results: **the anchor holds at every weight in the bracket, including the weakest — and that is the
one result here that was not predicted.** The transfer question was whether an anchor scaled to a
$4.9\times$ larger reconstruction would still hold. It does, and so does one that was *not* scaled:
$\beta_{\mathrm{prior}} = 0.1$, the raw sibling's value carried across unchanged against a
reconstruction almost five times larger, finishes at a floor fraction of $0.0121$ — six-fold inside
the criterion — where the roadmap's threshold argument predicted it would be the arm most at risk of
being overrun.

Two things reconcile that with the argument rather than refuting it. The saturating restoring force
is per latent dimension and the opposing pressure reaches the prior's log-variance through
`base_decode` and `posterior_logvar_mode`, both of which this package ships in the configuration
that *removes* those paths — `mean` and `independent` — so the pressure the anchor has to exceed is
already much smaller than the one measured when the collapse was first found. And `mean_logvar_prior`
does move monotonically with the anchor, from $-3.71$ to $-1.23$, so the anchor is plainly doing
work; it simply has less to fight. The reading is that **this configuration is not anchor-limited at
any weight in this bracket**, which makes `beta_prior` a much less delicate key here than the raw
sibling's history suggests, and means the ratio being held fixed is a safeguard rather than a
constraint that binds.

**What the comparison model's dev-box arms already showed**, on this same shard at the raw target —
evidence for where to look, and now visibly *not* transferable in its detail. At `beta_prior`
$10^{-2}$ the anchor delayed the raw model's collapse without preventing it (floor fraction $0.955$,
against an unanchored $0.978$); at $10^{-1}$ it held, at $0.046$ with `kld_active_frac` $0.69$. The
$0.1$ arm here holds at $0.0121$ with `kld_active_frac` $0.667$ — the same weight, a nearly identical
latent width, and a floor fraction four times *better* against a reconstruction $4.9\times$ larger.
The threshold moved with the architecture's two collapse-removing keys, not with the block size.

**What the comparison model's dev-box arms already showed**, on this same shard at the raw target —
evidence for where to look, not a substitute for these rows, because the threshold is a property of
the reconstruction pressure and this objective's is $4.9\times$ larger. At `beta_prior` $10^{-2}$ the
anchor *delayed* the collapse without preventing it: the floor crossing moved by $6.7\times$ in
optimizer steps and the prior pinned anyway, finishing at $0.955$ against an unanchored $0.978$. At
$10^{-1}$ it held, at a floor fraction of $0.046$ and a `kld_active_frac` of $0.69$ against $0.25$ in
every collapsed arm. The shipped $0.5$ here is that value scaled by the same factor the reconstruction
grew by, and the `sweep_beta_1p0.yaml` arm — which carries the raw model's $0.1$ unscaled against a
$4.9\times$ reconstruction — is the arm that says whether the scaling was necessary.

## What the production runs still owe

Not scheduled here; recorded so the gap is explicit rather than discovered later.

- [ ] A short multi-rank `torchrun` completing several steps without deadlock — the sampler sharding
      and the permutation control's rank reduction exercised.
- [ ] Peak memory at production geometry, read as `torch.cuda.max_memory_allocated()` after the first
      training step. The output activations here are $1.68$ GiB at batch $128$ against the raw
      model's $0.25$, so this is the one place the target domain is expected to bind.
- [ ] The baseline and the four arms at production scale and on held-out recordings. **Everything in
      this file is in-sample**, and the finding that motivated the whole package — a source pathway
      that does not generalise — is invisible to an in-sample run by construction.
- [ ] `anchor_coverage_frac` inspected against `coverage_floor: 0.9` at production scale. On the
      committed shard it sits at $0.99996$ in every arm, so the floor admits essentially every
      anchor and cannot be judged there — the shard is too clean to exercise it.
- [ ] The evaluation pipeline, deferred whole. Until it exists there are no verdicts, no confidence
      intervals, no trivial-predictor baselines and no per-recording tables for this model.
