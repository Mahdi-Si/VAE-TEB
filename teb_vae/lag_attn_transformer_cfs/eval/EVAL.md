# The evaluation contract

Short by design. `SeqVaeLagAttnTrfCfs` is `SeqVaeLagAttnCfs` with both history encoders replaced, so
it is evaluated by *that* pipeline rather than by a copy of it, and this document says only what is
true of this package. The pipeline is documented in three companion files, none restated here:

| Document | What it covers |
|---|---|
| `teb_vae/lag_attn_cfs/eval/EVAL.md` | **The contract**: what a run is, the output layout, the layers, the configuration reference, one section per registered analysis, how the output is misread, and the guard recovery table. |
| `teb_vae/lag_attn_cfs/eval/FIGURE_GUIDE.md` | Every figure a run writes: what each panel shows, how to read it, how it is misread. `figure_manifest.json` beside it lists the fixed names and filename families. |
| `teb_vae/lag_attn_cfs/eval/ATTRIBUTION.md` | The Captum attribution pass: readouts, baselines, checks, outputs and its figures. |

## What this package supplies

Four files, and each holds a fact the shared pipeline cannot derive:

| File | What it carries |
|---|---|
| `binding.py` | `TRF_CFS_BINDING`: the classes to rebuild from a checkpoint, the `geometry_keys` reconciled against it, this encoder's own causality disclosure, and the override path below. |
| `configs/eval_overrides.yaml` | The causal holdout split and the evaluation-only settings. It is the cfs cell's file key for key, and value for value **except where a value is a function of the lag window**: `occlusion_bands` is cut to this cell's own `max_lag`, because a band reaching past the window is refused at config load. |
| `run.py` | The command line. It supplies the binding, a `prog=` string, and enumerates its own flags for one reason: `--only` and `--skip` must name *this* model's registry (`--help` lists it). |
| `verify.py` | The acceptance gate, delegated in full, beside the tables for the sweep arms this cell ships and the cross-cell table the two cfs cells are read down. |

Launch from the repository root:

```bash
python -m teb_vae.lag_attn_transformer_cfs.eval.run --checkpoint <run>/model_checkpoints/<name>.ckpt
python -m teb_vae.lag_attn_transformer_cfs.eval.verify <run>/eval_results/summary.json
python -m teb_vae.lag_attn_transformer_cfs.eval.verify --runs <dir-of-runs> --out RESULTS_arms.md
```

Or with no command line at all: `run.py` and `verify.py` each carry a `RUN_ARGS` dictionary at the
bottom of the file. Edit the values there and press the IDE's Run button; a flag given on the
command line wins over the dictionary, and the console line prints where each value came from.

## What resolves to the parent

Everything else. The preflight guards and their recovery table, the population and forward-contract
probe, the collection pass and its branches, every registered analysis, the readouts, the
ten-verdict registry, the headline registry (the Monte Carlo error block included), the sanity
block, the figure seam and the gate's criteria come from `teb_vae.lag_attn_cfs.eval` unchanged and
are reached through the binding. The analyses only a causal cell can have (`warmup`,
`source_null`, `time_shift`, `occlusion`, `lag_clocks`, `lag_kld_scaled`, `lag_high_kl` and
`spectral_skill`) are registered on the *cfs* binding's `EXTRA_ANALYSES` and picked up here by
binding that object rather than by re-registering it, so an analysis added there reaches this cell
from one place. `tests/test_eval_binding.py` asserts the identity.

**`occlusion` is the one that most needs saying here.** It is the interventional half of the lag
question: it removes the source's *values* in a lag band, leaves the availability announcement
untouched, re-encodes through the run's own K/V path and reports the per-horizon-step forecast cost.
Under the shipped `lag_kv_source: adapter` the keys and values are the projected source stream
itself, whose reach is one step, so any lag structure the attention reports is the attention's own
and not something a history encoder already aligned; the `sweep_lag_kv_conv_stem` arm restores a
local stem whose reach is its receptive field. `preflight.json` records the reach beside the
furthest searched lag (`source_reach_vs_lag_range`), which is the comparison that matters, rather
than either number alone.

The implementation is shared, so the two cfs cells' band deltas are computed identically. The
**bands are not**: each cell cuts them to its own lag window, so a band named `far` covers
different lags in the two cells. Compare the band deltas across cells by the lag spans the
occlusion table records, never by band name.

**This package defines no numeric function.** Not "few": none. `binding.py` is a frozen record,
`run.py` and `verify.py` delegate to the shared implementations, and there is no module here that
computes a quantity a summary reports. That is asserted about the code in
`tests/test_eval_binding.py` and `tests/test_eval_run.py` rather than promised by this paragraph,
because a prose claim of delegation is exactly the claim that decays first.

`geometry_keys` is the one declaration that differs and it is written out rather than derived. It is
the cfs cell's tuple **minus** `causal_norm` (not a constructor parameter of this model, because
these encoders carry no time-pooling normaliser to causalise) **plus** this architecture's encoder
keys (`encoder_conv_kernels`, `encoder_conv_dilations`, `encoder_num_heads`, `encoder_d_ff`,
`target_attention_blocks`, `source_attention_blocks`, `source_attention_window`) **plus**
`forecast_ar_residual`, which only this cell configures. Each must be both a constructor parameter
and a config key, because `preflight.reconcile` silently skips any key absent from either, so a key
that is only one of the two is a reconciliation that never happens and never says so. The count is
left to the code, which is where a reader can check it.

**Four switches that change what a number means are reconciled, and one is deliberately not.**
`prior_availability_input`, `lag_kv_source`, `persistence_residual` and `forecast_ar_residual` are
in the tuple, because the evaluation rebuilds the architecture from the checkpoint's own
`model_kwargs`: a config disagreeing about one of them would not fail, it would report one
architecture's (or one likelihood's) numbers under another's stated name. The other density term,
the per-channel scored horizon `target_scored_horizon`, is a resolved vector with no config key, so
the preflight re-resolves it from the config and compares it instead. `horizon_weight_halflife_steps`
is **absent**, on the same ground as the objective weights: it re-weights the *training* criterion's
horizon axis, and no evaluated readout applies it, since this pipeline scores every block
unweighted.

**Two of the encoder keys describe a stack the shipped arm does not build.**
`source_attention_blocks` and `source_attention_window` describe the deep source encoder, which
exists only under `lag_kv_source: encoder`; neither the shipped `adapter` nor the `conv_stem` arm
constructs it. They stay in the tuple because they are exactly what an `encoder` arm reconciles
against, and the causality disclosure reports zero source blocks, not the configured count, when
that encoder was not built.

## What the scored likelihood is

Since the final revision (`DESIGN.md`, amendment of 2026-09-23) this cell scores a forecast cell
$(\tau, c)$ only while $\tau < H_c$, the channel's own scored horizon, and scores its residual
under an AR(1) innovation with the per-channel coefficient $\phi_c$ when `forecast_ar_residual` is
on. Every density readout (block NLLs, `pred_gap`, the matched Monte Carlo gap, the splits and both
baselines) is scored under the checkpoint's own $H_c$ and $\phi_c$, read from the model rather than
configured, and `summary.json` states both under `likelihood_structure`. The `sweep_all_cells_scored`
and `sweep_factorised_likelihood` arms restore the previous choice of each.

## What the cross-cell table can and cannot say

The cross-cell table puts runs of this cell and of `lag_attn_cfs` side by side, keyed by the
`model_class` each run recorded. Whether a *level* in one row can be compared with a level in the
other depends on the two runs, not on the two cells, and the test is mechanical:

* **A level is comparable only when the scored block is the same.** A block score is a sum over the
  scored cells of a block, $\sum_{c} \min(H_c, H)$ coefficients, so its scale is set by the horizon
  $H$, the per-channel scored horizons $H_c$, the kept channel budget $C_{\mathrm{keep}}$ and the
  likelihood's AR term before the model is reached. Compare `preflight.json` (`horizon`,
  `anchors_per_sample`, `block_width`, the warm-up budget) and `summary.json`
  (`likelihood_structure`) between the two runs. Only when they match does a difference in
  `d_base_mc_nats`, `pred_gap_mc_nats`, `source_conditioned_kl_raw_nats` or
  `coupling_minus_clock_nats` measure the encoder.
* **At the shipped leaves they do not match.** The final revision changed this cell's horizon,
  stride, lag window, K/V arm and likelihood; the conv-LSTM cell was not revised. The shipped pair is
  therefore two different predictors scored on two different blocks, and the cross-cell table reads
  as signs and orderings only (does `pred_gap` have the same sign, does the verdict agree), with the
  level columns ignored. Rebuilding the level comparison needs a run of one cell at the other's
  geometry and likelihood.
* The *lag* readouts compare less still: the two cells search different lag windows through
  different K/V representations, so a lag-profile difference is about the K/V arm and the window as
  much as about the encoders.

**Against `lag_attn_transformer_fs` no level is comparable at any leaves.** That cell is the same
architecture over the **two-sided** transform: its channel set was never pruned by a warm-up budget
and its coefficients are not causal, so its block is a different sum over different quantities. If
that edge is ever wanted it must be a signs-and-orderings table with the level columns removed, and
that is a different table, not this one with a caveat attached.

The same rule holds one step further out and is stated in the shared contract: the percentage
columns are **budget-local**, because $C_{\mathrm{keep}}$ is whatever the warm-up budget decided, so
two arms of *either* cfs cell at two budgets are non-comparable to each other as well.

## Which alignment a checkpoint was built at

`causal_align_reference` and `causal_align_reference_source` are deliberately **not** among the
`geometry_keys`, and they cannot be: both are config keys that name no constructor parameter, so
`preflight.reconcile` would skip them silently. What reaches the checkpoint is their consequence,
the two shift vectors in `model_kwargs`, and `preflight.check_warmup_budget_matches_checkpoint`
re-resolves both references against the shards this run is about to read and compares the resolved
tuples. The alignment is therefore checked, just not by name. The shipped configuration is
unaligned; the `sweep_align_target_max` and `sweep_target_clock_input` arms are the aligned ones.

**It is also printed by name**, on the console block and in `summary.run_arm`, in three readings kept
separate: the *configured* reference label and source reference, the *built* `lag_kv_source`, and the
*resolved* target clock, source clock and inter-stream offset in seconds. Three because a config
naming one arm while the checkpoint carries another is exactly what the re-resolution guard above
catches structurally and what this line makes visible on the page.

## The gate

From the repository root:

```bash
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_cfs/tests -q -m "not slow"
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_cfs/tests -q -m slow
```
