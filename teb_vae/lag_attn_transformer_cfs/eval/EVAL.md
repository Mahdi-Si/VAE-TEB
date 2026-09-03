# The evaluation contract

Short by design. `SeqVaeLagAttnTrfCfs` is `SeqVaeLagAttnCfs` with both history encoders replaced and
nothing else changed, so it is evaluated by *that* pipeline rather than by a copy of it, and this
document says only what is true of this package. **The contract is
`teb_vae/lag_attn_cfs/eval/EVAL.md`** — what a run is, the output layout, the four layers, the
configuration reference, one section per registered analysis, how the output is misread, and the
guard recovery table are all there and are not restated here.

## What this package supplies

Four files, and each holds a fact the shared pipeline cannot derive:

| File | What it carries |
|---|---|
| `binding.py` | `TRF_CFS_BINDING`: the classes to rebuild from a checkpoint, the `geometry_keys` reconciled against it, this encoder's own causality disclosure, and the override path below. |
| `configs/eval_overrides.yaml` | The causal holdout split and the evaluation-only settings — deliberately the cfs cell's file key for key and value for value, so a difference between the two cells' summaries is never a difference between their configurations. |
| `run.py` | The command line. It supplies the binding, a `prog=` string, and enumerates its own flags for one reason: `--only` and `--skip` must name *this* model's registry. |
| `verify.py` | The acceptance gate, delegated in full, beside the sweep axes this cell ships arms for and the cross-cell table the two cfs cells are read down. |

Launch from the repository root:

```bash
python -m teb_vae.lag_attn_transformer_cfs.eval.run --checkpoint <run>/model_checkpoints/<name>.ckpt
python -m teb_vae.lag_attn_transformer_cfs.eval.verify <run>/eval_results/summary.json
python -m teb_vae.lag_attn_transformer_cfs.eval.verify --runs <dir-of-runs> --out RESULTS_arms.md
```

## What resolves to the parent

Everything else. The preflight guards and their recovery table, the population and forward-contract
probe, the collection pass and its five branches, every registered analysis, the readouts, the
ten-verdict registry, the headline registry, the sanity block, the figure seam and the gate's
criteria come from `teb_vae.lag_attn_cfs.eval` unchanged and are reached through the binding. The
analyses only a causal cell can have — `warmup`, `source_null`, `occlusion`, `lag_clocks`,
`lag_kld_scaled`, `lag_high_kl` and `spectral_skill` — are registered on the *cfs* binding and
picked up here by binding that pipeline rather than by re-registering them, so an eighth would
reach this cell from one place.

**`occlusion` is the one that most needs saying here.** It is the interventional half of the lag
question: it removes the source's *values* in a lag band, leaves the availability announcement
untouched, re-encodes through the run's own K/V path and reports the per-horizon-step forecast cost.
That makes it the readout on which this cell's shipped `lag_kv_source: conv_stem` arm is genuinely
informative — this architecture's local stem reaches 21 steps against a 91-lag window where the
conv-LSTM cell's reaches 387 — and it is registered on the shared binding precisely so the two cells'
band deltas are produced by one implementation and remain comparable as levels across the encoder
edge.

**This package defines no numeric function.** Not "few": none. `binding.py` is a frozen record,
`run.py` and `verify.py` delegate to the shared implementations, and there is no module here that
computes a quantity a summary reports. That is asserted about the code in
`tests/test_eval_docs.py` rather than promised by this paragraph, because a prose claim of
delegation is exactly the claim that decays first: the moment one number is computed locally the two
cells stop being measured by one implementation, and their difference stops being attributable to the
encoder.

`geometry_keys` is the one declaration that differs and it is written out rather than derived. It is
the cfs cell's tuple **minus** `causal_norm` — not a constructor parameter of this model, because
these encoders carry no time-pooling normaliser to causalise — **plus** the seven this architecture
adds: `encoder_conv_kernels`, `encoder_conv_dilations`, `encoder_num_heads`, `encoder_d_ff`,
`target_attention_blocks`, `source_attention_blocks` and `source_attention_window`. Each must be both
a constructor parameter and a config key, because `preflight.reconcile` silently skips any key absent
from either, so a key that is only one of the two is a reconciliation that never happens and never
says so. The count is left to the code, which is where a reader can check it.

**Three architecture switches are in both cells' tuples and a fourth is deliberately not.**
`prior_availability_input`, `lag_kv_source` and `persistence_residual` are reconciled, because the
evaluation rebuilds the architecture from the checkpoint's own `model_kwargs`: a config disagreeing
about one of them would not fail — it would report one architecture's numbers under another's stated
name, and every `nll_*`, every skill comparison and every lag readout would be measured on a
different predictor. `horizon_weight_halflife_steps` is **absent**, on the same ground as the
objective weights: it re-weights the *training* criterion's horizon axis, and no evaluated readout
applies it, since this pipeline scores every block unweighted.

**Two of the seven encoder keys apply to the `encoder` K/V arm alone.**
`source_attention_blocks` and `source_attention_window` describe the deep source encoder, which the
shipped `lag_kv_source: conv_stem` does not build; the stem reads `encoder_conv_kernels` and
`encoder_conv_dilations` instead. They stay in the tuple because they are exactly what the `encoder`
comparison arm reconciles against, and because removing them would make the leaf-for-leaf config
parity check read a divergence where there is none.

## What the encoder edge means, and what it does not

The cross-cell table exists to make one comparison readable and to make the other one unavailable.
The two are **not symmetric**, and reading either as the other is the misreading this section exists
to foreclose.

**Against `lag_attn_cfs`, a loss level *is* comparable.** Both cells sum the same
$H \cdot C_{\mathrm{keep}} = 2940$ target coefficients over the same 136 dense anchors under the
same objective, from the same shards at the same warm-up budget. Only the two history encoders
differ. So
`d_base_mc_nats`, `pred_gap_mc_nats`, `source_conditioned_kl_raw_nats` and
`coupling_minus_clock_nats` can be put in one table and read as levels, and a difference between the
rows is attributable to the encoder. That is the whole reason this cell exists, and it is why the
pipeline was bound rather than forked a second time.

Two conditions on that, and both are now load-bearing rather than incidental. The two cells must ship
the **same objective weights** — the same horizon half-life and the same persistence state — because
a level comparison across two differently weighted criteria compares nothing; and the *lag* readouts
must be read knowing that the two cells' local K/V stems are not the same size, 21 steps here against
387 there, so a difference in a lag profile across this edge is about the stems as much as about the
encoders. Both `DESIGN.md` records say so at the point where the difference is priced.

**Against `lag_attn_transformer_fs`, a loss level is *not* comparable**, and the asymmetry is
structural rather than a matter of care. That cell is the same architecture over the **two-sided**
transform: its blocks are 2340 coefficients against this cell's 2940, at the same horizon of 30 steps against
15, and its channel set was never pruned by a warm-up budget at all. A block score is a sum over the
coefficients in a block, so its scale is set by the block before the model is reached; two such
numbers do not become comparable by being placed in adjacent columns. If that edge is ever wanted it
must be a signs-and-orderings table with the level columns removed — and that is a different table,
not this one with a caveat attached.

The same rule holds one step further out and is stated in the shared contract rather than here: the
percentage columns are **budget-local**, because $C_{\mathrm{keep}}$ is whatever the warm-up budget
decided, so two arms of *either* cfs cell at two budgets are non-comparable to each other as well.

## Which alignment a checkpoint was built at

`causal_align_reference` and `causal_align_reference_source` are deliberately **not** among the
`geometry_keys`, and they cannot be: both are config keys that name no constructor parameter, so
`preflight.reconcile` would skip them silently. What reaches the checkpoint is their consequence —
the two shift vectors in `model_kwargs` — and `preflight.check_warmup_budget_matches_checkpoint`
re-resolves both references against the shards this run is about to read and compares all six
resolved tuples. The alignment is therefore checked, just not by name.

**And it is now printed by name as well**, on the console block beside the delay and in
`summary.run_arm`, in three readings kept separate: the *configured* reference label and source
reference, the *built* `lag_kv_source`, and the *resolved* target clock, source clock and
inter-stream offset in seconds. Three because a config naming one arm while the checkpoint carries
another is exactly what the re-resolution guard above catches structurally and what this line makes
visible on the page.

That guard matters here for a specific reason. This package's training config carried
`causal_align_reference: null` for a window and has since been restored to `target_max`, so a
checkpoint trained inside that window is an **unaligned** model. Its target width is identical,
so it would load cleanly and only the numbers would move; the preflight refuses it instead,
naming the disagreeing tuples. The refusal is right and the checkpoint is the thing that does
not belong — the unaligned arm is a legitimate arm, but it is not the arm the cross-cell table
reads, and a summary from it must not be entered in the aligned row.

Three quantities a summary reports also moved underneath, and none is a re-measurement of the
same thing. The alignment shifts are scaled by the impulse-response centroid factor
$\kappa = 0.875$ before quantisation, so the target range is $0$–$85$ steps rather than $0$–$97$ (the
keep-index is scale-invariant, so no width and no parameter total moved with it).
`source_lag_warmth_frac_st` / `_ph` are built from $W'_c + d_c$ rather than $W'_c$ alone, which on a
single-clock aligned configuration is what unpinned `source_lag_warmth_frac_st` from a constant
$1.0000$ it took for any attention distribution at all. And the **source stream now has its own
clock**: `causal_align_reference_source` aligns it onto a reference $113.9$ s nearer than the
target's, which narrows the source keep-index from $47$ channels to $39$ and moves both warmth
columns again. Values from before any of the three are not comparable to values from after it, and a
checkpoint trained at one source reference is refused against a run configured at another by the
re-resolution guard above.

## The gate

From the repository root:

```bash
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_cfs/tests -q -m "not slow"
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_transformer_cfs/tests -q -m slow
```
