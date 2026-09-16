# Lag-Residual Transformer Improvement - Spec and Roadmap

Status: IN_PROGRESS
Last updated: 2026-09-15 (Sprint 3 code delivered: the predictive validation monitor, the
per-epoch clip fraction, the arm profiles selecting on the monitor, and the two operator
runbooks; the matched-arm and factorial runs themselves are with the operator)
Owner: unassigned
Planning style: Implementation milestones (sprints)
Baseline: repository root `C:\Users\mahdi\Desktop\teb_vae_model`, revision `36ab0dc` (2026-09-15),
working tree dirty with untracked documents, fixtures and `tests/test_eval_aggregate.py`.
Inspected scope: `teb_vae/lag_slot_transformer_cfs/` (nets, task, trainer, configs, eval, tests),
the shared seams it binds in `teb_vae/lag_attn_cfs/eval/` (`metrics.py`, `config_schema.py`,
`oracle.py`, `binding.py`) and `teb_vae/lag_attn/config.py`, plus the results export
`output/lag_slot_summary/` of the epoch-1002 run.

This document is the executable roadmap for the improvement programme described in
[IMPROVEMENT_PROPOSAL_2026-09-15.md](IMPROVEMENT_PROPOSAL_2026-09-15.md). The proposal holds the
audit and the reasoning; this document holds the requirements, decisions and tasks. Where the two
disagree, this document wins on *what is built and in what order*, and the proposal's section 4
remains the evidence record. The model itself is explained in [MODEL_EXPLAINED.md](MODEL_EXPLAINED.md)
and its original build roadmap is [SPEC_AND_SPRINTS.md](SPEC_AND_SPRINTS.md), whose `T001`-`T046`
IDs are untouched; task IDs below are local to this file.

---

## 1. Context, outcomes, and scope

### 1.1 Where the model stands

`SeqVaeLagResidualTrfCfs` forecasts the next $H$ stored steps of the kept FHR feature block from a
target-only Gaussian prior $P_t$ and a source-conditioned Gaussian $Q_t$ that differs from it by a
bounded residual $(a_t, b_t)$ summed over $L$ per-lag proposals. The evaluated run (epoch 1002,
seed 42, $L = 91$, `lambda_base: 0.5`, no warm start, batch 256) shows, on 1,959 recordings:

| Quantity | Value | Reading |
| --- | ---: | --- |
| Equal-recording predictive gain $G^{(8)}$ | 0.1564 nats/anchor, 95% interval $[-0.198, 0.531]$ | inconclusive |
| Recordings with positive gain | 46.55% | the mean hides heterogeneity |
| Mean-decoded gain | 10.60 | a different estimator, not predictive |
| Single-draw unweighted gain | 12.12 | a conditional score, not the weighted training loss |
| Suppressing lags 0-14 | +12.83 $[12.16, 13.51]$ | the anchor neighbourhood carries the effect |
| Suppressing lags 15-44 / 45-67 / 68-90 | $-1.49$ / $-0.95$ / $-0.68$, all intervals below zero | every older band hurts this fit |
| Removing lag 0 alone (256-segment profile) | +19.0 $[11.6, 27.8]$ | one contemporaneous coefficient dominates |
| Mixture 50% / 90% / 99% coverage, full branch | 0.640 / 0.937 / 0.990 | too broad in the centre |
| Prior log-variance in floor margin | 55.8% of support | prior scale sits on its bound |
| Effective draws, full branch | 1.46 of 8 | draw concentration; $K = 8$ is not converged |

The source pathway is active and lag-specific, but a predictive advantage over the internal base
is not established, older lags are net harmful, and several reported scores and verdicts do not
mean what their names say.

### 1.2 Outcomes this delivery targets

1. **Scores and verdicts a reader can trust**: every headline number carries its estimator name,
   and a scientific PASS requires a paired interval that excludes zero.
2. **A 25-entry lag bank** (lags $0..24$, oldest centre 96 s before the anchor) as a first-class
   training and evaluation profile, with the 91-entry profile still reproducible.
3. **Independent switches that zero the order-zero scattering coefficient** $S_0$ of the FHR input
   and of the UA input, applied after loader normalisation and before every model path, with
   labels untouched.
4. **A matched comparison** of target-only, 91-entry, 25-entry, capacity-control and mean-only arms
   from one target-only initialisation, selected on a predictive validation monitor.
5. **Bounded objective and uncertainty pilots** (separate variance bounds, multi-draw objective,
   proposal penalty), each promoted or rejected on recorded evidence.

Build acceptance is a passing focused test set plus a fixture-scale end-to-end run for each new
profile. Research acceptance is separate: a shortlisted arm needs three training seeds, $K = 32$
primary draws, and a confirmation population disjoint from development, as
`eval/configs/acceptance_plan.yaml` already requires.

### 1.3 Non-goals

* A larger Transformer, more latent dimensions, or forcing all latent coordinates active. The
  parameter budget is already 98% target/decoder pathway; the evidence points at scoring,
  calibration and lag support, not capacity.
* Replacing the pointwise source representation with a long temporal encoder. The convolution
  comparator exists; its extra reach would break a strict 25-entry support claim.
* Excluding $S_0$ from the *forecast target*. That is a different task from input zeroing and is
  deferred (FR-006).
* Repairing the sufficiency probe's conditioning contract (proposal section 5, last two rows).
  Deferred; its interpretation is marked unavailable until repaired (FR-007).
* Any change to the shared `lag_attn_cfs` verdict code. This cell post-processes the family's
  verdict list in its own `family_results`, leaving sibling cells untouched.

---

## 2. Requirements and acceptance

- FR-001 [required]: **Distinct score labels.** The per-sample table and the results block name the
  weighted objective score, the unweighted single-draw conditional score, the latent-mean score
  and the $K$-draw predictive mixture score as four separate columns. Given the epoch-1002
  export rescored, when a reader opens `conventions`, then each column's estimator, weighting and
  draw count is stated, and a weighted-parity column reproduces the reconstruction terms of
  `compute_loss` on the same batch to $10^{-6}$ relative.
- FR-002 [required]: **Interval-aware verdicts.** The `predictive_improvement` verdict is PASS only
  when the recording-bootstrap interval of the paired predictive gain lies above zero, FAIL when
  it lies below, INCONCLUSIVE otherwise. Given the saved epoch-1002 summary, then the verdict is
  INCONCLUSIVE with both interval ends beside it, and `eval.verify` agrees with the summary.
- FR-003 [required]: **Mixture-based calibration verdict.** `calibration_near_nominal` reads the
  averaged-component-CDF census, resolved at least by horizon step and by stored target block, and
  the single-draw conditional calibration is reported separately under its own name. The
  prior-floor verdict text states the residual KL identity $K_t = \tfrac12\sum_d(a_d^2 + e^{2b_d} - 1 - 2b_d)$
  rather than an inverse-variance inflation.
- FR-004 [required]: **Declared aggregation estimands.** `aggregate_by_recording` reports both the
  legacy equal-segment mean and an anchor-sum-over-anchor-count mean within each recording, both
  then equally weighted over recordings; the headline names which one it reads; a control that
  ran on a subset of batches is compared on matched support.
- FR-005 [required]: **25-entry lag bank profile.** With `max_lag: 24` the model builds 25
  proposals, 25 lag embeddings, 25 exposure and profile entries, $c_L = 25^{-1/2}$ by default, and
  reads source positions $t-24..t$; anchors, warm-up, horizon, kept channels, stride and labels are
  unchanged. A strict load of a 91-entry checkpoint into a 25-entry model refuses by name. The
  25-entry evaluation profile declares an inclusive partition of $0..24$ plus fixed cross-bank
  bands, and the acceptance pass reads band families keyed by bank geometry so a 25-entry run is
  not failed for searching bands the 91-entry family does not name.
- FR-006 [deferred]: **Target-side $S_0$ exclusion.** Excluded from this delivery: removing $S_0$
  from the scored target changes every likelihood, error, calibration and subgroup denominator
  at once and is a different experiment. Reconsider if the input-ablation factorial (S3-T03)
  shows the model relies on $S_0$ *labels* rather than $S_0$ *inputs*.
- FR-007 [deferred]: **Sufficiency probe contract repair.** `oracle.py` conditions on
  `target_state`, omits the metadata clock and runs `probe(states)` without the production
  persistence path, so its $-239$-nat gap is not a bottleneck cost. Excluded from this delivery;
  the summary marks the interpretation unavailable. Reconsider once Sprint 3 needs a bottleneck
  comparison to decide on Sprint 4 pilots.
- FR-008 [required]: **Independent $S_0$ input switches.** Constructor fields
  `zero_fhr_scattering_s0` and `zero_up_scattering_s0`, both default `false`, zero the selected
  coefficient after loader normalisation and before persistence gathering, target/source gates and
  encoders. Given a fixed noise draw and an enabled switch, when the original $S_0$ input is
  perturbed, then no output of `forward`, including `persistence`, changes; unselected channels
  behave as before; the batch tensors and forecast labels are unchanged; both switches `false`
  reproduce legacy outputs bitwise; `zero_up_scattering_s0: true` with `use_up_st: false` raises at
  construction; the flags survive checkpoint reconstruction and are reconciled by the evaluation
  binding; the trivial baselines and attribution read the same permitted-input view.
- FR-009 [required]: **Matched-arm comparison record.** Each arm records its initialisation
  checkpoint, effective loss weights, data budget, external-base drift against the frozen
  target-only reference, predictive/calibration/resource results, and a promote/reject decision.
  No arm is promoted from its training gap alone.
- FR-010 [required]: **Predictive validation monitor.** Validation logs a recording-grouped,
  unweighted $K$-draw mixture NLL for both branches under a fixed RNG bank, usable as the
  checkpoint and early-stopping monitor, while the weighted objective stays as a diagnostic. With
  $K = 1$ it reduces to the unweighted single-draw conditional score.
- FR-011 [required]: **Pilots with recorded decisions.** Separate prior/observation log-variance
  bounds, a small multi-draw predictive objective, and a pre-limiter proposal penalty are each
  implemented behind an explicit config field defaulting to legacy behaviour, tested numerically,
  and piloted one at a time with a recorded promote/reject decision.
- NFR-001 [required]: **Invariants intact.** Existing causality, source-isolation, source-off
  equality, shared-noise pairing and reference-arm tests keep passing; no new preprocessing reads a
  stored input after the anchor or changes the scored target silently.
- NFR-002 [required]: **Measured resources and provenance.** Peak training memory, dense
  evaluation memory and throughput for 91 versus 25 entries are measured at declared batch size
  and draw count with `eval/memory.py`; every summary records scorer version, RNG policy, data,
  config and checkpoint digests, active target axes, ablation flags and lag scale.
- NFR-003 [required]: **Legacy reproducibility.** Old configurations that omit the new fields
  build the same model; the epoch-1002 summary is still read by `eval.verify` and by the
  acceptance pass under its recorded band family.

---

## 3. Current state and evidence

### 3.1 Code facts (verified 2026-09-15)

| Seam | Fact | Source |
| --- | --- | --- |
| Lag count | $L = $ `max_lag` $+ 1$, lag zero included; `lag_scale: null` resolves to $L^{-1/2}$ once from the configured count | `nets/model.py:106,136`, `nets/lag_updates.py:64-88` |
| Lag embedding | `nn.Embedding(n_lags, lag_embed_dim)`; a 91-row table cannot load into a 25-row model | `nets/lag_updates.py:184` |
| Forward input boundary | `forward(y_st, y_ph, u_stream, ...)`; `target = cat([y_st, y_ph])`; persistence is gathered from `target` **before** `target_gate`; the source gate runs on `u_stream` | `nets/model.py:471-560` |
| Constructor forwarding | keywords not in `FORWARDED_EXCLUSIONS_HERE` are forwarded to the core; the driver forwards a config key only when the constructor names it | `nets/model.py:75,285-289` |
| Evaluation geometry check | `GEOMETRY_KEYS` in the binding is reconciled against the checkpoint; every key must be a constructor parameter and a shipped config key | `eval/binding.py:110-135`, `tests/test_eval_binding.py:101-124` |
| Warm-start allowlist | `TRANSFERABLE_PREFIXES` named `clock_proj.` but not `clock_norm.` until S2-T01; both now transfer | `trainer.py` (`TRANSFERABLE_PREFIXES`) |
| Band validation | shared `_validate_occlusion_bands` refuses empty bands and bands past `max_lag`; `band_masks` adds `none` and `all` (= union of declared bands) | `../lag_attn_cfs/eval/config_schema.py:151-212`, `eval/lag_metrics.py:81-123` |
| Band family in acceptance | `exploratory_bands: [anchor, near, mid, far]` was a closed list until S2-T02; the plan (revision 2) now declares `exploratory_band_families` keyed by searched lag steps, and a run is read under its own window's family | `eval/configs/acceptance_plan.yaml`, `eval/acceptance.py` (`band_block`, `arm_lag_window`) |
| Input policy | `zero_fhr_scattering_s0` / `zero_up_scattering_s0` are constructor keywords of the composed model only, applied by `_ablate_input_streams` as the first step of `forward`; `permitted_target_features` is the view every trivial baseline and probe reads; both flags are `GEOMETRY_KEYS` and are in the arm record and the causality disclosure (`effective_inputs`) | `nets/model.py`, `eval/binding.py`, `eval/collect.py` (`arm_record`, `shared_readout`) |
| Overrides merge | the override delta is deep-merged over the checkpoint's `resolved_config.yaml`, which carries no `eval_config`; `base:` is refused | `../lag_attn_cfs/eval/config_schema.py:218-293` |
| Verdict assembly | this cell calls the family's `build_verdicts` inside `family_results` and writes the list itself, so it can post-process verdicts locally | `eval/collect.py:1633-1677` |
| Shared verdict rule | `predictive_improvement` is `PASS if full < base`; `calibration_near_nominal` reads the single-draw 1/2/3-sigma coverage; the prior-floor detail text claims KL inflation | `../lag_attn_cfs/eval/metrics.py:3269-3280,3362-3376,3480-3535` |
| Training-path columns | `shared_readout` scores `nll_full_block`/`nll_base_block` with `masked_raw_block_per_anchor` **without** channel or horizon weights | `eval/collect.py:841-847` |
| Baselines | `baseline_forecasts(target_features, weight, model, anchors)` reads the original declared-width target stream | `eval/collect.py:899`, `../lag_attn_cfs/eval/metrics.py:559` |
| Aggregation | `aggregate_by_recording` averages segment means equally within a recording; anchors are returned only as exposure | `eval/collect.py:1068-1147` |
| Lag profile cohort | the first `caps.lag_profile` segments the fixed-seed shuffled loader hands out; identities not exported | `eval/collect.py:1725-1777,1376-1450` |
| Probe conditioning | reads `target_state`, no clock, `probe(states)`; bias directions recorded but the conditioning gap is not | `../lag_attn_cfs/eval/oracle.py:180-215,662` |
| Validation selection | `default.yaml` keeps `val/total_loss` (the one-draw weighted objective); every arm profile sets `validation_mc_draws: 8` and both selectors on `val/pred_nll_full_mc`, the unweighted $K$-draw mixture score under a fixed noise bank, with `val/total_loss` as the secondary criterion | `task.py` (`_predictive_monitor`), `trainer.py` (`validation_monitor_draws`), `configs/target_only.yaml`, `configs/joint.yaml` |
| Log-variance bound | one `logvar_clamp` feeds both the prior head and the decoder observation head | `nets/core.py:263,476,569,649`, `../lag_attn_rws/nets/heads.py:74`, `../lag_attn/nets/decoders.py:408` |
| $S_0$ identity | `fhr_st[..., 0]` and `up_st[..., 0]` are `kind: st_S0`; the phase blocks have no $S_0$ | `output/lag_slot_summary/band_channel_map.csv` rows 0 and 80 |
| Run-button rule | every runner is enumerated in `tests/test_eval_launch.py::ENTRY_POINTS`; no new runner is planned here | `tests/test_eval_launch.py:37-44` |

### 3.2 Results-export facts

`resolved_config.yaml` of the evaluated run differs from `configs/default.yaml` in
`lambda_base` (0.5 vs 1.0), batch size (256 vs 128), milestones (1000/2000 vs 400/800) and early
stopping (off vs on). The evaluation named 14 shard paths including `fold_1/val/*` and
`fold_1/train/hie_*`; loader counts collapse them to eight basenames. Treat this population as
development evidence, not confirmation. Stage timings: `samples` 2,301 s, `sufficiency` 940 s,
`probe` 627 s, `attribution` 442 s; the $K = 8$ collection itself took about 3,725 s.

### 3.3 Test layout and runtime

Tests are colocated in `tests/`, run from the repository root with `.venv/Scripts/python.exe`.
The whole package ran in 103 s (572 tests, recorded 2026-09-09 in `SPEC_AND_SPRINTS.md`), so a
package run is cheap here; the CLAUDE.md file-level selection rule still applies, and the
`lag_attn_rws` suites must never be pulled in. `tests/test_docs.py` has a pre-existing failure on
the untracked `diagrams/` directory (memory note); it is not caused by this work.

### 3.4 Decisions the user has already made

* Reduce the lag bank to about 100 s / 25 steps.
* Add the ability to zero the FHR order-zero scattering coefficient.
* Pursue the further improvements the proposal recommends, in its priority order.

### 3.5 Assumptions

* "25 time steps" means 25 entries including lag zero, so `max_lag: 24` (oldest centre 96 s).
  The alternative, `max_lag: 25` for an exact 100 s, is one config value; see section 6.
* "Zero the coefficient" means an **input** ablation with labels preserved (FR-008), applied
  identically in training, validation and evaluation. Target exclusion is FR-006, deferred.
* A UA switch is wanted alongside the FHR one, as the proposal recommends; it is independent and
  defaults off.

---

## 4. Proposed approach

### 4.1 Scoring and verdicts (Sprint 1)

All changes are inside this cell's `eval/collect.py` and `eval/verify.py`; shared family code is
not edited.

* **Columns.** Keep `nll_full_block`/`nll_base_block` as they are computed, but state in
  `conventions` that they are *unweighted single-draw conditional* scores and add
  `nll_full_block_weighted`/`nll_base_block_weighted` computed with the model's registered
  `target_channel_weight` and `horizon_weight` buffers, the same mask and the same reduction as
  `compute_loss`. Add `pred_gap_weighted`. Rename nothing that an existing analysis reads by name;
  add the new names beside them and version the results schema (`results.schema_version: 2`).
* **Verdicts.** After `build_verdicts` returns in `family_results`, replace three entries by
  cell-specific verdicts built from the same `Verdict` type: `predictive_improvement` from the
  paired recording-bootstrap interval already computed in `arm_scores_block`;
  `calibration_near_nominal` from the mixture census in `results.mixture_calibration`, resolved by
  horizon and block; `prior_variance_not_pinned` with the residual-KL wording. The family's
  `order_verdicts` guard keeps the registry order, so names are preserved and only the status,
  criterion and detail change. `eval/verify.py::report_predictive_gap` then reads the summary's
  verdict rather than always returning INCONCLUSIVE.
* **Aggregation.** `aggregate_by_recording` additionally accumulates `sum(column * n_anchors)`
  and `sum(n_anchors)` per recording and returns an anchor-weighted-within-recording table.
  `arm_scores_block` reports both under `results.arm_scores` (legacy) and
  `results.arm_scores_anchor_within_recording`. The headline keeps the legacy estimator so the
  epoch-1002 numbers remain comparable, and `conventions` says so. Only 0.445% of segments have
  fewer than the dense anchor count, so the two agree closely on this export; the estimand
  matters for masked or capped runs.
* **Lag profile cohort.** Export the profiled segments' `(guid, epoch, class, subgroup)` to
  `lag_profile_segments.csv`, and draw them class-balanced through the same stratified selection
  the attribution stage uses, so the profile is a declared cohort rather than a loader prefix.

### 4.2 The 25-entry bank (Sprint 2)

No model code changes: `max_lag` already drives $L$, the embedding, $c_L$, the gather, exposure
and profiles. The work is configuration, evaluation profile, acceptance families, transfer
hygiene and tests.

* `configs/lag25.yaml`, `base: joint.yaml`, sets `max_lag: 24`, its own tag and seed. Basing it
  on the warm-started candidate makes `joint.yaml` the matched 91-entry reference by construction.
* `eval/configs/lag25_eval_overrides.yaml` mirrors `eval_overrides.yaml` with
  `num_mc_samples: 32` and the bands

  ```yaml
  occlusion_bands:
    instant: [0, 0]          # the anchor's own stored step
    recent: [1, 4]
    intermediate: [5, 12]
    tail: [13, 24]
    common_head: [0, 14]     # the 91-entry `anchor` band, for cross-bank comparison
    common_tail: [15, 24]    # what survives of the 91-entry `near` band
  ```

  `all` is the union, which is every lag on both partitions. The two `common_*` bands overlap
  the four partition bands; the acceptance family adjustment counts six searched bands.
* `eval/configs/acceptance_plan.yaml` gains `exploratory_band_families`, keyed by
  `searched_lag_steps`: `91: [anchor, near, mid, far]` and `25: [instant, recent, intermediate,
  tail, common_head, common_tail]`. `acceptance.py` selects the family from each run's recorded
  `causality.searched_lag_steps` and refuses a run whose bands are not that family's. `revision`
  is bumped to 2 and `declared_on` updated. Legacy 91-entry records validate under family 91.
* `TRANSFERABLE_PREFIXES` gains `clock_norm.`; the checkpoint-contract test that checks the clock
  transfer is extended to the norm parameters.
* Figures, pages and attribution already derive band names and edges from the summary
  (`eval/figures.py::_declared_bands`, `_shade_bands`); the lag axis length is `model.n_lags`.
  The tasks verify this at fixture scale under both profiles rather than adding code.

### 4.3 The $S_0$ switches (Sprint 2)

* Two constructor keywords on `SeqVaeLagResidualTrfCfs`, forwarded nowhere (added to
  `FORWARDED_EXCLUSIONS_HERE`), stored as attributes, validated at construction
  (`zero_up_scattering_s0` requires `use_up_st`).
* One private helper `_ablate_input_streams(y_st, u_stream)` called first in `forward`. It
  returns new tensors built by multiplying with a channel mask (never an in-place write), so the
  batch storage that also feeds `_build_raw_target` is untouched. Both the persistence gather and
  the gates read the returned tensors. The availability masks are unchanged: this is a value
  ablation, not a missing-sensor event.
* `GEOMETRY_KEYS` gains both flags so the binding reconciles them; `configs/default.yaml` sets
  both `false` so `test_every_geometry_key_is_also_a_shipped_config_key` holds; the disclosure in
  `residual_encoder_disclosure` reports the effective input policy.
* Evaluation: `shared_readout` receives a `permitted_target_features` view (the model's helper
  applied to the declared stream) for `baseline_forecasts`, while `target` (the labels) stays the
  original. Attribution reports the ablated coordinate as `ablated` and asserts zero attribution.
* Profiles: `configs/lag25_s0_fhr.yaml`, `configs/lag25_s0_up.yaml`, `configs/lag25_s0_both.yaml`
  (each `base: lag25.yaml`) and `configs/target_only_s0_fhr.yaml` (`base: target_only.yaml`) so
  every source-enabled ablation has a target-only reference under the same FHR policy.

### 4.4 Training selection and pilots (Sprints 3-4)

* **Predictive validation monitor** in `task.py`: during dense validation, draw $K_{\rm val}$
  paired latent samples from a generator seeded per `(guid, epoch)` and log
  `val/pred_nll_full_mc`, `val/pred_nll_base_mc` and `val/pred_gap_mc` as recording-grouped means
  of the unweighted mixture NLL, reusing `eval/predictive.py::matched_predictive_scores`. New
  config keys `validation_mc_draws` (default `null`, meaning off, which preserves legacy logs) and
  the checkpoint `monitor` pointed at the new column in the new profiles only.
* **Separate variance bounds** (`prior_logvar_clamp`, `obs_logvar_clamp`, both `null` meaning
  "use `logvar_clamp`") threaded through `nets/core.py` to the prior head and decoder, with the
  head-calibration preimages using the bound that applies to each head.
* **Multi-draw objective** (`train_mc_draws`, default 1): encoders once, paired noise, the decoder
  invoked per draw, `logsumexp` over draws per anchor before the weighted reduction; with one
  draw the loss equals the current one bitwise.
* **Proposal penalty** (`proposal_penalty`, default 0.0, with `proposal_penalty_scale_ratio`):
  $\langle \frac1L \sum_\ell (\|c_L r^\mu_\ell\|^2 + \rho \|c_L r^\sigma_\ell\|^2)\rangle$ over
  contributing anchors, zero on unavailable lags, omitted on the mean-only arm.

Each pilot is one config field with a legacy default, so old configs are unchanged.

### 4.5 Alternatives not taken

* Editing the family's `build_verdicts` to use intervals everywhere. It would change three sibling
  cells' summaries and their tests; a cell-local post-process is smaller and reversible.
* Zeroing $S_0$ in the task or loader. Several consumers call the model directly (offline
  controls, attribution, probes, the diagnostic page), so a task-level mask would leak the
  original value into the persistence path and the baselines.
* Cropping the 91-row embedding table to warm-start a 25-entry model. It silently changes
  $c_L$ by $\sqrt{91/25} \approx 1.908$ and conflates support with scale; the strict load refuses
  it and the plan trains the 25-entry arm from the target-only checkpoint instead.

---

## 5. Data, contracts, and failure behaviour

* **Forward contract** is unchanged in keys and shapes; the lag axis is `model.n_lags`. With an
  $S_0$ switch on, `persistence[..., 0]` is exactly zero for the FHR switch.
* **Checkpoint contract**: the two new flags and any new objective fields are stamped in the
  saved config and reconciled by `GEOMETRY_KEYS`; a checkpoint whose flag disagrees with the
  evaluation override refuses. Legacy checkpoints without the keys reconstruct with the defaults.
* **Results schema**: new columns are additive; `results.schema_version` is introduced at 2 and
  `eval.verify` accepts 1 and 2. `results.verdicts` keeps the registry names and order.
* **Acceptance plan**: `exploratory_band_families` replaces `exploratory_bands`; a plan carrying
  only the old key is read as family 91 with a warning, so old records still validate.
* **Failure behaviour**: a band past `max_lag`, an empty band, a UA switch without `use_up_st`, a
  warm start with a mismatched embedding shape, and an unknown `eval_config` key each raise with
  the offending name. Nothing clips, skips or silently defaults.

---

## 6. Decisions, assumptions, and risks

| Item | State | Evidence / rationale | Resolution / owner | Affected work |
| --- | --- | --- | --- | --- |
| 25 entries (`max_lag: 24`) rather than an exact 100 s oldest centre | assumed | user asked for "100s or 25 time steps"; 25 entries is the count reading; the span is 96 s centre-to-centre | record `n_lags` and oldest-centre seconds in every artefact; switch to `max_lag: 25` by editing one value if exact depth is required | FR-005, S2-T01, S2-T02 |
| $S_0$ zeroing is input-only, labels preserved | assumed | proposal section 7.1; target exclusion changes every denominator | FR-006 deferred with a revisit trigger | FR-008, S2-T03, S2-T04 |
| Headline estimator stays equal-segment until confirmation | resolved | keeps the epoch-1002 numbers comparable; both estimators are exported | revisit at S5 when the confirmation profile is frozen | FR-004, S1-T03 |
| Verdicts post-processed in this cell, not in shared code | resolved | `family_results` owns the list it writes | none | FR-002, FR-003, S1-T02 |
| The 25-entry arm warm-starts from the target-only checkpoint | resolved | matches `joint.yaml`; the shipped run was scratch-trained with `lambda_base: 0.5`, which is a different experiment | the runbook also names a scratch 25-entry run as optional context | S3-T02 |
| Original checkpoint and shards are not on this machine | blocking for runs only | proposal section 2 | operator runs S1-T04, S2-T05, S3-T02, S3-T03 on the Linux box; code and tests proceed | all operator tasks |
| Confirmation population disjointness | deferred | the evaluation named `val/` and `train/hie_*` shards | operator supplies GUID manifests before S5; the acceptance pass already checks recording overlap | S5 |
| Draw count for primary comparisons | resolved | acceptance plan says 32; stability at 8/32/128 | new eval profile sets 32 | S2-T02, S3-T02 |

Risks:

* **The 25-entry refit is not better than 91.** Likelihood moderate; impact is a scoped decision,
  not a failure: the noninferiority tolerance is set from development variability before reading
  confirmation results (S3-T02 records it), and the resource saving is measured (S2-T05).
* **Interval-aware verdicts turn the existing PASS into INCONCLUSIVE.** Certain; that is the
  correct reading of the evidence and is disclosed in the change record.
* **Multi-draw training memory.** The decoder per draw multiplies activations; S4 pilots at
  reduced batch size and measures before any production run.
* **`test_docs.py` gates.** Documentation updates to `MODEL_EXPLAINED.md`, `eval/EVAL.md` and
  `eval/FIGURE_GUIDE.md` are deferred to S5 so the docs describe implemented and measured
  behaviour; the untracked-diagrams failure is pre-existing.

lean-limit: verdict post-processing lives in this cell's `family_results`; replace with a
family-level interval verdict when a second cell needs the same rule.

---

## 7. Validation strategy and Definition of Done

Necessary tests only. Existing coverage is inspected per task; a test is added only for a
behaviour no existing test exercises, using the fixtures in `tests/conftest.py`.

Focused commands, all from the repository root with `.venv/Scripts/python.exe`, all **proposed**
(none executed for this plan):

| Scope | Command | Runtime |
| --- | --- | --- |
| Scoring, aggregation, verify | `-m pytest teb_vae/lag_slot_transformer_cfs/tests/test_eval_aggregate.py teb_vae/lag_slot_transformer_cfs/tests/test_eval_binding.py teb_vae/lag_slot_transformer_cfs/tests/test_eval_smoke.py -q` | fast/local, tens of seconds |
| Bank, config, transfer | `-m pytest teb_vae/lag_slot_transformer_cfs/tests/test_config_load.py teb_vae/lag_slot_transformer_cfs/tests/test_checkpoint_contract.py teb_vae/lag_slot_transformer_cfs/tests/test_eval_lag_metrics.py teb_vae/lag_slot_transformer_cfs/tests/test_acceptance.py -q` | fast/local |
| $S_0$ switches | `-m pytest teb_vae/lag_slot_transformer_cfs/tests/test_forward_contract.py teb_vae/lag_slot_transformer_cfs/tests/test_causality.py teb_vae/lag_slot_transformer_cfs/tests/test_scattering_ablation.py teb_vae/lag_slot_transformer_cfs/tests/test_eval_attribution.py -q` | fast/local |
| Training-side | `-m pytest teb_vae/lag_slot_transformer_cfs/tests/test_task.py teb_vae/lag_slot_transformer_cfs/tests/test_objective.py teb_vae/lag_slot_transformer_cfs/tests/test_trainer.py teb_vae/lag_slot_transformer_cfs/tests/test_train_smoke.py -q` | fast/local |
| Fixture end-to-end | `-m teb_vae.lag_slot_transformer_cfs.trainer --config teb_vae/lag_slot_transformer_cfs/configs/tiny.yaml` then `-m teb_vae.lag_slot_transformer_cfs.eval.run --checkpoint CKPT --overrides PROFILE`, with the fixture checkpoint and the profile under test | integration, about a minute |
| Verify on the saved export | `-m teb_vae.lag_slot_transformer_cfs.eval.verify --summary output/lag_slot_summary/summary.json` | fast/local |
| Sprint boundary | `-m pytest teb_vae/lag_slot_transformer_cfs/tests -q` | about two minutes recorded; run once per sprint, not per task |

Production training and rescoring are slow/external and belong to the operator; their outputs
(summaries, acceptance records) are the evidence for the corresponding tasks.

Definition of Done for the delivery: FR-001 to FR-005 and FR-008 to FR-011 evidenced by the
focused commands above and one fixture end-to-end run per new profile; NFR-001 to NFR-003 by the
sprint-boundary package run and `eval.verify` on the legacy export; S3 and S5 outcomes evidenced
by operator run records. Deferred FR-006 and FR-007 stay listed with their triggers.

---

## 8. Delivery and operations

No migration: all new config keys default to legacy behaviour and all new result fields are
additive. Rollback of any sprint is reverting its files. Operator runs follow the runbooks in
S1-T04, S3-T02 and S3-T03; each writes its `resolved_config.yaml`, summary and acceptance record
beside the checkpoint, which is the durable provenance.

---

## Sprint overview

| Sprint | Goal | Usable outcome | Dependencies | Detail |
| --- | --- | --- | --- | --- |
| Sprint 1 | Trustworthy scoring on the existing run | The epoch-1002 export rescored with correct labels, interval verdicts, both estimands and a tail-suppression diagnostic | none for code; original checkpoint for the rescore | Detailed |
| Sprint 2 | 25-entry bank and $S_0$ switches | `lag25.yaml` and the ablation profiles train and score at fixture scale; acceptance reads bank families; resources measured | Sprint 1 score contracts | Detailed |
| Sprint 3 | Matched arms and a predictive monitor | Six arms trained from one target-only checkpoint, selected on the predictive monitor, with a recorded decision; the $S_0$ factorial on the chosen bank | Sprint 2, GPU box and shards | Detailed; code and runbooks delivered, runs with the operator |
| Sprint 4 | Objective and uncertainty pilots | Separate bounds, multi-draw objective and proposal penalty each piloted once with a decision | Sprint 3 evidence | Forecast |
| Sprint 5 | Confirmation and publication | Three seeds, $K = 8/32/128$, disjoint confirmation set, docs updated | Sprint 4 decisions | Forecast |

## Sprint 1: Trustworthy scoring on the existing run

Goal: every number the epoch-1002 export reports carries its estimator's name, the scientific
verdicts read intervals, and the old bank's tail is measured jointly.
Demo: `eval.verify --summary output/lag_slot_summary/summary.json` reports
`predictive_improvement: INCONCLUSIVE [-0.198, 0.531]`; a fixture rescore writes the weighted
parity columns and both aggregation estimands; the tail-diagnostic profile validates its bands.
Definition of Done: the scoring/aggregation focused commands pass; the fixture smoke writes
`results.schema_version: 2`; the legacy export still verifies; the operator runbook for the
rescore is written.
Dependencies: none for code. The production rescore needs the original checkpoint and shards.

#### S1-T01: Score labels and weighted objective-parity columns

Requirements: FR-001, NFR-003
Depends on: none
Description: In `eval/collect.py::shared_readout`, add `nll_full_block_weighted`,
`nll_base_block_weighted` and `pred_gap_weighted`, computed with the model's registered
`target_channel_weight` and `horizon_weight` buffers (read with `getattr`, `None` meaning no
weighting, as `task.py::forecast_rows` already does) on the identical mask and per-anchor
reduction as `compute_loss`. Rewrite `PRED_GAP_CONVENTIONS` to name four estimators: weighted
objective (`*_weighted`), unweighted single-draw conditional (`nll_*_block`, `pred_gap`),
latent-mean (`mean_nll_*`, `mean_pred_gap`) and predictive mixture (`mc_nll_*`, `mc_pred_gap`,
with $K$). Add `results.schema_version = 2` in `family_results`. Update the per-sample and
per-anchor column registries the family's sanity block reads so the new columns are known.
Acceptance criteria:
- On a fixture batch, `nll_full_block_weighted` equals the `nll_full_block` term of
  `compute_loss` to $10^{-6}$ relative, per contributing anchor.
- With a model whose weight buffers are absent, the weighted columns equal the unweighted ones.
- `conventions` names every score column with its estimator and weighting.
- Every existing column keeps its name and value.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/eval/collect.py` - columns, conventions, schema version
- `teb_vae/lag_slot_transformer_cfs/tests/test_eval_smoke.py` - one parity assertion on the fixture batch
Validation: `-m pytest teb_vae/lag_slot_transformer_cfs/tests/test_eval_smoke.py teb_vae/lag_slot_transformer_cfs/tests/test_eval_binding.py -q`, cwd repo root; expected pass. Proposed.
Test rationale: no existing test compares a collected column against `compute_loss`; one parity
assertion closes the gap the proposal found. Nothing else is new behaviour.
Runtime: fast/local.
Evidence: 2026-09-15, working tree at `36ab0dc` plus uncommitted edits. Implemented as
`objective_parity_scores` in `eval/collect.py` (called from `shared_readout`), the
`nll_*_block_weighted` / `pred_gap_weighted` columns, `SCORE_CONVENTIONS`, and
`results.schema_version = 2`. The parity test lives in `tests/test_eval_binding.py` rather than
the smoke file, because it needs a forward and `compute_loss` on the tiny model. Executed:
`.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_eval_binding.py -q`
passed every Sprint 1 case (parity to 1e-6, unweighted-arm identity, conventions); the four
failures in that file are another session's in-progress binding refactor
(`kld_source_null_nats`, new analyses), not this task. Smoke assertion on the column set added
to `tests/test_eval_smoke.py`; fixture run pending (see Resume).

#### S1-T02: Interval-aware, mixture-based and residual-correct verdicts

Requirements: FR-002, FR-003, NFR-003
Depends on: S1-T01
Description: In `eval/collect.py::family_results`, after `build_verdicts`, replace three verdicts
with cell-built ones of the family's `Verdict` type, keeping names and order:
`predictive_improvement` reads the paired recording-bootstrap interval of `mc_pred_gap` (PASS if
`lo > 0`, FAIL if `hi < 0`, else INCONCLUSIVE) and carries `point`, `lo`, `hi`, `n`, `K`;
`calibration_near_nominal` reads `results.mixture_calibration` coverage at 0.5/0.9/0.99 for the
full branch under the family's relative tail tolerance, and the census is extended in
`score_batch` to accumulate per horizon step and per stored target block; the conditional
single-draw calibration keeps its block under `calibration` with a `conditional_single_draw`
label; `prior_variance_not_pinned` keeps its criterion and replaces the detail text with the
residual identity and the note that the prior scale cancels at fixed $(a, b)$. Update
`eval/verify.py::report_predictive_gap` to read the summary's verdict and interval, and
`check_against_reference` to use the same rule. The family's `order_verdicts` guard must still
pass.
Acceptance criteria:
- On the saved `output/lag_slot_summary/summary.json`, `eval.verify` reports
  `predictive_improvement: INCONCLUSIVE` with the recorded interval; on a synthetic summary with
  `lo > 0` it reports PASS and with `hi < 0` FAIL.
- The calibration verdict on the saved export reads the mixture coverage (0.640/0.937/0.990) and
  names the worst level; the conditional block is still present and labelled.
- `results.verdicts` keeps the registry names and order; the headline `verdict_*` scalars agree
  with the list.
- No file under `teb_vae/lag_attn_cfs/` changes.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/eval/collect.py` - verdict post-processing, resolved census
- `teb_vae/lag_slot_transformer_cfs/eval/predictive.py` - census keyed by horizon and block
- `teb_vae/lag_slot_transformer_cfs/eval/verify.py` - interval rule
- `teb_vae/lag_slot_transformer_cfs/tests/test_eval_binding.py` - three verdict cases on synthetic results
- `teb_vae/lag_slot_transformer_cfs/tests/test_eval_smoke.py` - verify on the saved export
Validation: `-m pytest teb_vae/lag_slot_transformer_cfs/tests/test_eval_binding.py teb_vae/lag_slot_transformer_cfs/tests/test_eval_smoke.py -q` then
`-m teb_vae.lag_slot_transformer_cfs.eval.verify --summary output/lag_slot_summary/summary.json`; expected pass and INCONCLUSIVE. Proposed.
Test rationale: the PASS-on-point-estimate rule is the defect; three synthetic cases (above,
spanning, below zero) are the smallest set that would fail on the old rule.
Runtime: fast/local.
Evidence: 2026-09-15. Implemented as `cell_verdicts` and its three builders in
`eval/collect.py`, applied inside `family_results` after the family's `build_verdicts`; the
mixture census in `eval/predictive.py::calibration_census` now also accumulates by horizon step
and by stored block (`resolved`), merged and finished alongside; `eval/verify.py` reads the
interval (`interval_status`), refuses a schema-2 summary whose list disagrees with its
interval, tolerates schema-1 summaries, and pairs the candidate against a reference by
recording (`paired_improvement`, stdlib percentile bootstrap). No file under
`teb_vae/lag_attn_cfs/` was edited for this task. Executed: the four verdict/census/gate cases
in `tests/test_eval_binding.py` pass; `.venv/Scripts/python.exe -m
teb_vae.lag_slot_transformer_cfs.eval.verify --summary output/lag_slot_summary/summary.json`
reports five structural PASS and `predictive_gap_measured: INCONCLUSIVE` with interval
[-0.198, 0.531] over 1959 recordings, noting the summary's own PASS was written under schema 1.

#### S1-T03: Aggregation estimands and a declared lag-profile cohort

Requirements: FR-004, NFR-002
Depends on: S1-T01
Description: Extend `aggregate_by_recording` to also accumulate anchor-weighted sums per column
(`sum(value * n_anchors)` and `sum(n_anchors)` over the segments carrying the column) and return
a second per-recording table; `arm_scores_block`, `paired_margin_block` and `anchor_weighted`
report it under `results.*_anchor_within_recording`. Paired margins for a control that ran on a
subset of batches are computed on the recordings that hold both arms, and the paired count is
recorded. For the single-lag profile, replace the "first segments of the shuffled loader"
selection with the class-balanced stratified draw the attribution stage already uses, and write
`lag_profile_segments.csv` with `guid`, `epoch`, class and subgroup. State in `conventions` which
estimator the headline reads.
Acceptance criteria:
- A hand-built two-batch record with unequal anchor counts recombines to the correct anchor-sum
  estimate and to the legacy equal-segment mean, and the two differ where they should.
- A recording absent from a partial column is absent from both tables.
- The profiled-segment file lists exactly the segments scored with single lags, and its class
  counts follow the balanced draw on the fixture.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/eval/collect.py` - aggregation, blocks, profile selection, export
- `teb_vae/lag_slot_transformer_cfs/tests/test_eval_aggregate.py` - anchor-weighted recombination case
- `teb_vae/lag_slot_transformer_cfs/tests/test_eval_smoke.py` - profile cohort file present and consistent
Validation: `-m pytest teb_vae/lag_slot_transformer_cfs/tests/test_eval_aggregate.py teb_vae/lag_slot_transformer_cfs/tests/test_eval_smoke.py -q`; expected pass. Proposed.
Test rationale: `test_eval_aggregate.py` covers partial columns only; one unequal-anchor case
covers the new estimand. The cohort file is asserted in the existing smoke rather than a new file.
Runtime: fast/local.
Evidence: 2026-09-15. Implemented as `aggregate_by_recording_anchor_weighted` (a second
function, so the existing three-tuple contract is untouched), the
`results.anchor_within_recording` block (arm scores, band and control paired margins) and the
`*_anchor_weighted` headline columns on `per_recording`; `conventions.headline_estimator`
states the headline keeps the legacy estimator. The cohort is `LagProfileCohort` in
`eval/collect.py`: segments are admitted in loader order under the cap and a per-class quota
`ceil(cap / n_classes)` over the classes the split's shards declare (`shard_classes`, read off
the dataset's canonical basenames); rows the cohort did not admit are blanked to NaN in the
lag-margin curve and skipped by the aggregation; identities and composition travel under
`lag_readouts.lag_profile.predictive` and `lag_suppression/lag_profile_segments.csv`. Deviation
from the task text: the attribution stage's post-pass class-balanced draw needs the whole
identity table, which the in-pass profile cannot have, so the quota rule replaces it.
Executed: `tests/test_eval_aggregate.py` (8 cases, incl. anchor-weighted recombination, NaN
row skipping, quota admission, unlabelled fallback, shard classes) passes. Fixture end-to-end
run (22:04, `pytest-3087/slot_eval0`): the collection pass wrote `lag_profile_segments.csv`
with four segments (healthy 1, hie 2, acidosis 1 under quota 2 per class),
`results.anchor_within_recording` with arm scores, band and control margins, the three
`*_anchor_weighted` columns on every per-recording row, and `conventions.headline_estimator`;
`results.schema_version` is 2, the mixture census resolves by horizon (10) and block (2), and
the predictive verdict reads PASS from a fixture interval [0.018, 0.398] over 24 recordings,
consistent with `interval_status`. The pytest assertions of `test_eval_smoke.py` were still
executing (page rendering stages) when this was recorded.

#### S1-T04: Old-bank tail diagnostic profile and the rescore runbook

Requirements: FR-005, NFR-002
Depends on: S1-T02
Description: Add `eval/configs/lag91_tail_diagnostic.yaml`, a copy of `eval_overrides.yaml` with
`num_mc_samples: 32` and bands `anchor, near, mid, far` plus `head_0_24: [0, 24]` and
`tail_25_90: [25, 90]`, so the exact joint removal the proposal asks for is one suppression arm
on the existing checkpoint. The keep-prefix curve is expressed as bands too: `beyond_<l>` for
$\ell \in \{0, 4, 9, 14, 24, 44\}$ removes every lag above $\ell$, so no code path and no new
`eval_config` key (whose set is closed in the shared schema) is needed, and the curve's two ends
are the existing `none` / `all` identities. Mark the sufficiency probe's interpretation
unavailable in a sibling block `results.sufficiency_qualification`, because the family's
registry refuses an extra analysis under a shared name and the marker must not edit shared code;
this is the disclosure NFR-002 asks for while the repair stays deferred under FR-007. Write the
runbook in this task: rescore the epoch-1002 checkpoint under this profile at
$K = 8$, 32 and 128 on the same panel, record effective-draw quantiles for both branches, and
compare the joint tail margin with the sum of the three band margins.
Acceptance criteria:
- The profile validates against `max_lag: 90`; `all` still equals every lag; the prefix arms
  reproduce `none` at $\ell = 90$ and `silence` when the prefix is empty, bitwise on the fixture.
- The sufficiency block carries the unavailable-interpretation marker.
- The runbook names the commands, the panel, the draw counts and the artefacts to keep.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/eval/configs/lag91_tail_diagnostic.yaml` - new profile
- `teb_vae/lag_slot_transformer_cfs/eval/collect.py` - prefix arms, probe marker
- `teb_vae/lag_slot_transformer_cfs/eval/latent_probes.py` - probe marker where the block is assembled
- `teb_vae/lag_slot_transformer_cfs/tests/test_eval_lag_metrics.py` - prefix-arm identities
- `teb_vae/lag_slot_transformer_cfs/eval/EVAL.md` - runbook section
Validation: `-m pytest teb_vae/lag_slot_transformer_cfs/tests/test_eval_lag_metrics.py teb_vae/lag_slot_transformer_cfs/tests/test_eval_smoke.py -q`; expected pass. Production rescore: operator, on the Linux box with the checkpoint named in `summary.json`. Proposed.
Test rationale: the two identities (empty prefix equals silence, full prefix equals none) are
the invariants that make the curve readable; nothing else is new.
Runtime: fast/local for tests; slow/external for the rescore (about an hour at $K = 8$ recorded;
longer at 128).
Evidence: 2026-09-15. Profile written; it loads through `load_eval_overrides` with twelve
bands and `num_mc_samples: 32`. `SUFFICIENCY_QUALIFICATION` written into
`results.sufficiency_qualification` by `collect_tables`. Runbook and the score-convention table
added as sections 7.3 and 7.4 of `eval/EVAL.md`. Executed:
`tests/test_eval_lag_metrics.py` (39 cases incl. the two new: shipped partition carried, cutoff
pair, validation against `max_lag` 90, nested prefix masks, `all` still every lag) passes. The
production rescore is the operator's and has not run.

## Sprint 2: 25-entry bank and $S_0$ switches

Goal: the 25-entry bank and the two ablation switches exist as profiles that train, score and
pass acceptance at fixture scale, with resources measured at production geometry.
Demo: `trainer --config configs/lag25_s0_fhr.yaml` (tiny override) fits; `eval.run` with
`lag25_eval_overrides.yaml` writes a summary whose lag axis has 25 entries and whose disclosure
says the FHR $S_0$ input is ablated; `acceptance.py` reads it under family 25.
Definition of Done: the bank and $S_0$ focused commands pass; the fixture end-to-end run
succeeds for `lag25.yaml` and one ablation profile; the package suite passes at the sprint
boundary; the memory measurement is recorded.
Dependencies: Sprint 1 score contracts (so new runs are scored under the corrected labels).

#### S2-T01: The 25-entry training profile and transfer hygiene

Requirements: FR-005, NFR-003
Depends on: S1-T01
Description: Add `configs/lag25.yaml` (`base: joint.yaml`, `max_lag: 24`, tag
`lag_residual_trf_cfs_lag25`, its own seed, comment stating $L = 25$, $c_L = 0.2$ and the oldest
centre in seconds derived from `raw_per_step`). Add `clock_norm.` to `TRANSFERABLE_PREFIXES`.
Extend the tiny fixture config path so `tiny.yaml` can be layered on `lag25.yaml` for the
fixture fit (a `tiny_lag25.yaml` delta). Record `n_lags` and the oldest-centre seconds in the
trainer's resolved-config dump and in the binding's disclosure.
Acceptance criteria:
- Building `lag25.yaml` through the signature sweep gives `n_lags == 25`, an embedding of 25
  rows, `lag_scale == 0.2`, and `lag_valid` of width 25; anchors remain the production range at
  production geometry.
- Warm-starting a 25-entry model from a target-only checkpoint transfers the clock norm
  parameters and the base forecast equals the donor's bitwise on a fixture batch.
- Loading a 91-entry checkpoint into a 25-entry model with the strict path refuses, naming the
  embedding tensor and both shapes.
- Configs that omit `max_lag` still build $L = 91$.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/configs/lag25.yaml` - new profile
- `teb_vae/lag_slot_transformer_cfs/configs/tiny_lag25.yaml` - fixture delta
- `teb_vae/lag_slot_transformer_cfs/trainer.py` - transfer allowlist, disclosure of lag geometry
- `teb_vae/lag_slot_transformer_cfs/eval/binding.py` - `oldest_lag_seconds` in the disclosure
- `teb_vae/lag_slot_transformer_cfs/tests/test_config_load.py` - the new profiles in the parametrised config lists
- `teb_vae/lag_slot_transformer_cfs/tests/test_checkpoint_contract.py` - clock-norm transfer, 91-to-25 refusal
Validation: `-m pytest teb_vae/lag_slot_transformer_cfs/tests/test_config_load.py teb_vae/lag_slot_transformer_cfs/tests/test_checkpoint_contract.py -q`; expected pass. Proposed.
Test rationale: the shape refusal path exists (`test_a_geometry_disagreement_refuses_rather_than_skipping`) but not for the embedding; the clock-norm transfer is the fix the proposal found and has no test.
Runtime: fast/local.
Evidence: 2026-09-15. `configs/lag25.yaml` (`base: joint.yaml`, `max_lag: 24`, tag
`lag_residual_trf_cfs_lag25`, seed 17) and `configs/tiny_lag25.yaml` (`base: tiny.yaml`, the one
leaf) written; `clock_norm.` added to `TRANSFERABLE_PREFIXES`. Deviation from the task text: the
resolved-config dump is written by the shared driver and already carries `max_lag` and
`raw_per_step`, from which $L$, $c_L$ and the oldest centre follow, so no derived duplicate was
added to it; instead the driver logs the resolved bank (`_log_lag_geometry`: $L$, $c_L$, oldest
centre in seconds) in the run's first lines and the binding discloses `oldest_lag_seconds` in
`summary.json`. Executed: `tests/test_config_load.py` (the short-bank profile builds
$n_{\rm lags} = 25$ through the signature sweep, 25 embedding rows, $c_L = 0.2$, `lag_valid` of
width 25, anchors at the production floor and ceiling; a configuration omitting `max_lag` builds
the constructor default; the fixture delta changes one leaf) and `tests/test_checkpoint_contract.py`
(clock-norm tensors transfer and the warm-started base forecast equals the donor's bitwise; a
wider-window state dict refuses to load into a narrower model naming
`proposal_head.lag_embedding.weight` and both shapes; the target-only transfer is
window-indifferent). Both files pass in the 114-test batch with `test_scattering_ablation.py`
and `test_eval_binding.py`.

#### S2-T02: The 25-entry evaluation profile and geometry-keyed band families

Requirements: FR-005, NFR-002, NFR-003
Depends on: S2-T01
Description: Add `eval/configs/lag25_eval_overrides.yaml` with the bands of section 4.2 and
`num_mc_samples: 32`. In `eval/configs/acceptance_plan.yaml` replace `exploratory_bands` with
`exploratory_band_families` keyed by `searched_lag_steps` (91 and 25), bump `revision` to 2. In
`eval/acceptance.py` select each run's family from its recorded `causality.searched_lag_steps`,
refuse a run whose bands are outside its family, and accept a plan carrying only the legacy key
as family 91 with a warning. Verify on the fixture that figures, pages, traces and attribution
label the lag axis from `n_lags` and the band names from the summary under both profiles.
Acceptance criteria:
- The 25-entry profile validates; a band `[13, 30]` refuses naming `max_lag=24`.
- On a fixture 25-entry summary, `suppress:all` equals `silence` bitwise and the two `common_*`
  bands report margins alongside the four partition bands.
- The acceptance pass reads a fixture 25-entry run under family 25 and the saved 91-entry
  export under family 91; a run whose bands mix families is refused by name.
- Figure and page lag axes carry 25 ticks and the six declared band labels; no code path holds a
  band name or count literal.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/eval/configs/lag25_eval_overrides.yaml` - new profile
- `teb_vae/lag_slot_transformer_cfs/eval/configs/acceptance_plan.yaml` - band families, revision 2
- `teb_vae/lag_slot_transformer_cfs/eval/acceptance.py` - family selection and legacy key
- `teb_vae/lag_slot_transformer_cfs/tests/test_eval_lag_metrics.py` - `shipped_bands` fixture parametrised over both profiles
- `teb_vae/lag_slot_transformer_cfs/tests/test_acceptance.py` - family selection cases
- `teb_vae/lag_slot_transformer_cfs/tests/test_eval_figures.py` - axis length from `n_lags` under the 25-entry summary
Validation: `-m pytest teb_vae/lag_slot_transformer_cfs/tests/test_eval_lag_metrics.py teb_vae/lag_slot_transformer_cfs/tests/test_acceptance.py teb_vae/lag_slot_transformer_cfs/tests/test_eval_figures.py -q`, then the fixture end-to-end run with the new profile; expected pass. Proposed.
Test rationale: the partition and reference-arm tests exist for one profile and are parametrised
rather than duplicated; the family selection is new behaviour with a refusal path.
Runtime: fast/local plus one fixture integration run.
Evidence: 2026-09-15. `eval/configs/lag25_eval_overrides.yaml` written (the six bands of
section 4.2, `num_mc_samples: 32`, `event_lag_window_s` deliberately unchanged);
`acceptance_plan.yaml` revision 2 with `exploratory_band_families: {91: [...], 25: [...]}`,
digest `5653fb573d2961c1` pinned in `tests/test_acceptance.py`. `eval/acceptance.py`: the plan
loader normalises the families and reads a plan carrying only the legacy key as the shipped
window's family (read off `configs/default.yaml`, not remembered) with a `UserWarning`; both
keys together are refused; `run_identity`/`run_descriptor` carry `searched_lag_steps` and the
two input-policy flags; `scoring_mismatch` refuses a pairing across two windows or two input
policies (a target-only run, which searches no window, pairs with either); `band_block` reads
each arm under its window's family and reports `MIXED_LAG_WINDOWS` / `UNDECLARED_FAMILY`, which
`check_declared_bands` fails by name; a summary that records no window is read under the shipped
window with a note. Executed: `tests/test_eval_lag_metrics.py` (`shipped_profile` parametrised
over both deltas: each partition is contiguous from the anchor to its window's last lag and
validates against that window; `all` removes every lag on both; the two `common_*` bands are the
wide profile's `anchor` band and the surviving part of its `near` band and overlap the partition;
the wide profile's bands refuse against the short window naming `max_lag=24`),
`tests/test_acceptance.py` (eight new cases: a short-window run read under family 25, an
undeclared window refused, mixed windows refused, a window-less summary read under the shipped
window, a legacy-key plan read with a warning, both keys refused, the saved epoch-1002 export
read as the shipped window with both flags off, two windows and two input policies unmatched),
`tests/test_eval_figures.py` (a summary over a 25-entry window with six bands draws 25 lag
positions and the six band labels, from the summary alone). All three files pass in the
118-test batch with `test_eval_attribution.py`.

#### S2-T03: The $S_0$ input switches in the model

Requirements: FR-008, NFR-001, NFR-003
Depends on: none
Description: Add `zero_fhr_scattering_s0: bool = False` and `zero_up_scattering_s0: bool = False`
to `SeqVaeLagResidualTrfCfs.__init__`, exclude them from forwarding, validate the UA switch
against `use_up_st`, and store them. Add `_ablate_input_streams(y_st, u_stream)` returning
masked copies (multiply by a registered channel mask; no in-place write) and call it as the first
step of `forward`, before the persistence gather and the gates. Add both flags to `GEOMETRY_KEYS`
and to `configs/default.yaml` as `false`. Add the ablation profiles `configs/lag25_s0_fhr.yaml`,
`configs/lag25_s0_up.yaml`, `configs/lag25_s0_both.yaml` and `configs/target_only_s0_fhr.yaml`.
Acceptance criteria:
- With the FHR switch on and a fixed generator, perturbing `y_st[..., 0]` leaves every forward
  output, including `persistence`, bitwise unchanged; perturbing `y_st[..., 1]` changes outputs.
- With the UA switch on, perturbing `u_stream[..., 0]` leaves outputs unchanged and the source
  channel mask is unchanged; `up_ph[..., 0]` (the first phase channel) is never zeroed.
- Both switches off reproduce the legacy forward bitwise; the input tensors handed to `forward`
  are unmodified afterwards.
- `zero_up_scattering_s0: true` with `use_up_st: false` raises at construction naming both keys.
- A checkpoint round trip preserves the flags; the binding refuses an evaluation override that
  disagrees with the checkpoint's flag.
- Gradient of any output with respect to the ablated coordinate is exactly zero.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/nets/model.py` - keywords, helper, forward call, exclusions
- `teb_vae/lag_slot_transformer_cfs/eval/binding.py` - `GEOMETRY_KEYS`, disclosure of effective input policy
- `teb_vae/lag_slot_transformer_cfs/configs/default.yaml` - both flags `false`
- `teb_vae/lag_slot_transformer_cfs/configs/lag25_s0_fhr.yaml`, `lag25_s0_up.yaml`, `lag25_s0_both.yaml`, `target_only_s0_fhr.yaml` - ablation profiles
- `teb_vae/lag_slot_transformer_cfs/tests/test_scattering_ablation.py` - new focused file for the behaviours above
- `teb_vae/lag_slot_transformer_cfs/tests/test_config_load.py` - profiles in the parametrised lists
Validation: `-m pytest teb_vae/lag_slot_transformer_cfs/tests/test_scattering_ablation.py teb_vae/lag_slot_transformer_cfs/tests/test_forward_contract.py teb_vae/lag_slot_transformer_cfs/tests/test_causality.py teb_vae/lag_slot_transformer_cfs/tests/test_eval_binding.py teb_vae/lag_slot_transformer_cfs/tests/test_config_load.py -q`; expected pass. Proposed.
Test rationale: no existing test covers value ablation, persistence leakage under ablation, or
the UA/phase misidentification; one new file collects them so they are found together.
Runtime: fast/local.
Evidence: 2026-09-15. Implemented in `nets/model.py`: the two keywords (in
`INPUT_ABLATION_KEYWORDS`, excluded from forwarding), the construction-time refusal of the UA
switch without `use_up_st`, two non-persistent mask buffers (`fhr_s0_input_mask` of the first
target block's width, `up_s0_input_mask` of `c_u`), `_ablate_input_streams` as the first step
of `forward`, `permitted_target_features` for the declared stream, and `input_ablation_record`.
`GEOMETRY_KEYS` gained both flags; `configs/default.yaml` sets both `false`; the four ablation
profiles written (`lag25_s0_fhr`, `lag25_s0_up`, `lag25_s0_both` on `lag25.yaml`;
`target_only_s0_fhr` on `target_only.yaml`), each changing exactly its declared leaves.
Executed: `tests/test_scattering_ablation.py`, 13 cases: perturbing the ablated target
coordinate leaves every output including `persistence` bitwise unchanged while its neighbour
changes them; the inputs and the labels are untouched; the gradient with respect to the
ablated coordinate is exactly zero; the permitted view zeroes exactly that coordinate; the
source switch leaves the source channel mask unchanged and never touches the phase block; the
refusal names both keys; a constant substituted stream cannot reintroduce the coefficient;
both switches off reproduce the legacy forward bitwise; the flags survive a round trip and a
disagreeing override is refused by `reconcile_with_checkpoint`; the disclosure names field,
channel and kind. `test_forward_contract.py`, `test_causality.py`, `test_eval_binding.py` and
`test_config_load.py` pass unchanged in behaviour (fast batch of 230 passed; the one failure
was the plan-digest pin, moved by S2-T02).

#### S2-T04: The $S_0$ policy through evaluation, controls and attribution

Requirements: FR-008, NFR-001
Depends on: S2-T03, S1-T01
Description: In `eval/collect.py::score_batch`/`shared_readout`, build a permitted-input view of
the declared target stream through the model's helper and pass it to `baseline_forecasts`, while
the labels keep the original `target_features`. Replacement and permutation controls substitute
the *ablated* source stream so a control cannot reintroduce the coefficient. In
`eval/attribution.py` mark the ablated coordinate `ablated` in the channel tables and assert its
attribution is zero. The binding's disclosure records `effective_inputs` with the ablated
coordinates named by field and channel kind. Run the fixture end-to-end with
`lag25_s0_fhr.yaml` layered on `tiny_lag25.yaml`.
Acceptance criteria:
- On a fixture batch with the FHR switch on, the persistence baseline's block for channel 0 is
  the zero forecast, and the scored target for channel 0 is the original label.
- The observed-zeros and permutation arms reproduce the matched prior bitwise, as they do today.
- The attribution table reports `ablated` for the coordinate with attribution exactly zero.
- `summary.json` names the ablated inputs in `causality` and the acceptance pass carries them
  into the arm record.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/eval/collect.py` - permitted-input view, control streams
- `teb_vae/lag_slot_transformer_cfs/eval/attribution.py` - ablated marker and zero assertion
- `teb_vae/lag_slot_transformer_cfs/eval/binding.py` - disclosure
- `teb_vae/lag_slot_transformer_cfs/eval/acceptance.py` - arm record carries the flags
- `teb_vae/lag_slot_transformer_cfs/tests/test_eval_smoke.py` - one ablated-profile smoke case
- `teb_vae/lag_slot_transformer_cfs/tests/test_eval_attribution.py` - ablated coordinate case
Validation: `-m pytest teb_vae/lag_slot_transformer_cfs/tests/test_eval_smoke.py teb_vae/lag_slot_transformer_cfs/tests/test_eval_attribution.py teb_vae/lag_slot_transformer_cfs/tests/test_controls.py -q`, then the fixture end-to-end run; expected pass. Proposed.
Test rationale: the baseline leakage path (original `target_features` into `baseline_forecasts`)
is the uncovered defect; the smoke case and one attribution case are the smallest checks.
Runtime: fast/local plus one fixture integration run.
Evidence: 2026-09-15. `shared_readout` builds the trivial baselines on
`model.permitted_target_features(target_features)` while `target` (the labels) stays the
original gather; the replacement and permutation controls needed no change because every
substituted stream goes through the forward whose first step is the ablation (stated in
`intervened_branches`); `eval/latent_probes.py` gathers the probe's persistence term from the
same permitted view; `arm_record` carries the two flags and the ablated coordinates; the
binding's disclosure writes `causality.effective_inputs`; `eval/attribution.py` reads the
ablated coordinate back off the written example maps, writes `attribution_ablated_inputs.csv`
with status `ablated`, records `checks.ablated_input_max_abs` and refuses a nonzero value.
Executed: `tests/test_scattering_ablation.py` (the persistence and segment-mean baselines built
on the permitted view forecast a zero level for the ablated channel while the labels keep it,
and the baseline equals the forward's own persistence input), `tests/test_eval_attribution.py`
(integrated gradients through the ablated coordinate are exactly zero on both streams under the
divergence and gap readouts; the stage's block marks the coordinate and reads back zero) -- both
pass in the batches recorded under S2-T02 and S2-T03. The fixture end-to-end run
(`conftest.py`: `slot_short_ablated_run` fits `tiny_lag25.yaml` with `zero_fhr_scattering_s0`
layered on it, `slot_short_collected_run` scores it under `lag25_eval_overrides.yaml` with its
own bands kept; four cases in `tests/test_eval_smoke.py` read the window, the six bands, the
`all`-equals-`silence` identity, the disclosure, the attribution marker and the acceptance
family) was launched with `-k short_bank` and was still executing when this was written; its
result is the remaining S2-T04 evidence and is recorded under Resume when it lands.

#### S2-T05: Measure 91 versus 25 resources at production geometry

Requirements: NFR-002
Depends on: S2-T01
Description: Run `eval/memory.py` at production widths for `max_lag: 90` and `max_lag: 24`,
training batch 256 and 128, dense evaluation at test batch 32 with and without proposals, and
record peak memory and step throughput in `configs/lag25.yaml` comments and in the change record.
State batch size and draw count beside every number. This is a measurement on the GPU box.
Acceptance criteria:
- Both geometries measured under identical settings; numbers recorded with their settings.
- No claim of an end-to-end speed-up beyond what was measured.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/configs/lag25.yaml` - measured numbers in comments
- `teb_vae/lag_slot_transformer_cfs/eval/memory.py` - only if a `max_lag` argument is missing from `RUN_ARGS`
Validation: `-m teb_vae.lag_slot_transformer_cfs.eval.memory` with the two geometries; operator on the GPU box. Proposed.
Test rationale: a measurement, not a behaviour; no test.
Runtime: slow/external, minutes.
Evidence: 2026-09-15, code side only. `eval/memory.py` gained `--max-lag` (`RUN_ARGS["max_lag"]`,
default `None`), which replaces the loaded configuration's `max_lag` before the model is built
and is recorded as `run.max_lag_override`, so one configuration is measured twice with nothing
but the window differing; `tests/test_eval_launch.py` passes with the new argument.
`configs/lag25.yaml` carries the placeholder block the measured numbers go into. The
measurement itself needs the GPU box and has not run; the two invocations are
`-m teb_vae.lag_slot_transformer_cfs.eval.memory --batch-sizes 256,128 --output mem_91.json`
and the same with `--max-lag 24 --output mem_25.json`, each reporting `train`, `eval_dense`
and `eval_dense_proposals` at the swept batches; dense evaluation at the test batch of 32 is
`--batch-sizes 32`.

## Sprint 3: Matched arms and a predictive monitor

Goal: a predictive validation monitor selects checkpoints, six matched arms are trained from one
target-only initialisation and scored under the corrected contracts, the bank length is decided,
and the $S_0$ factorial runs on the chosen bank.
Demo: `metrics_history.csv` of a new run carries `val/pred_gap_mc`; the acceptance record
compares the arms with a recorded decision on the bank length and on the $S_0$ switches.
Definition of Done: the training-side focused commands pass; the fixture tiny fit logs the new
columns; operator run records exist for every arm with the FR-009 fields.
Dependencies: Sprint 2; the integer-operator production shards and the GPU box.

#### S3-T01: Predictive validation monitor and clip-fraction counts

Requirements: FR-010, NFR-003
Depends on: S2-T01
Description: In `task.py`, when `validation_mc_draws` (new hparam, default `null`) is set, the
dense validation step draws that many paired latent samples from a generator seeded from a
digest of `(guid, epoch, run seed)` and scores the unweighted mixture NLL of both branches with
`eval/predictive.py::matched_predictive_scores`, logging `val/pred_nll_full_mc`,
`val/pred_nll_base_mc` and `val/pred_gap_mc` as recording-grouped means reduced across ranks by
the objective's packed all-reduce. Add the columns to the tracked surface. Replace the
last-observation `grad_clip_frac` with a numerator and count so the epoch value is an actual
fraction. `lag25.yaml` and the ablation profiles set `validation_mc_draws: 8` and point
`model_checkpoint.monitor` and `early_stopping.monitor` at `val/pred_nll_full_mc`, with early
stopping re-enabled.
Acceptance criteria:
- With `validation_mc_draws: 1` the new column equals `val/nll_full_block` computed without
  weights on the fixture, per batch.
- Two validation epochs at the same weights log identical `val/pred_nll_full_mc` (fixed RNG bank).
- Legacy configs without the key log exactly the legacy columns.
- `grad_clip_frac` over an epoch equals clipped steps over total steps on the tiny fit.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/task.py` - monitor, clip counts
- `teb_vae/lag_slot_transformer_cfs/trainer.py` - tracked suffixes
- `teb_vae/lag_slot_transformer_cfs/configs/lag25.yaml` and ablation profiles - monitor keys
- `teb_vae/lag_slot_transformer_cfs/tests/test_task.py` - the $K = 1$ identity and the fixed-bank repeat
- `teb_vae/lag_slot_transformer_cfs/tests/test_trainer.py` - surface names
Validation: `-m pytest teb_vae/lag_slot_transformer_cfs/tests/test_task.py teb_vae/lag_slot_transformer_cfs/tests/test_trainer.py teb_vae/lag_slot_transformer_cfs/tests/test_train_smoke.py -q`; expected pass. Proposed.
Test rationale: the $K = 1$ reduction and the fixed bank are the two properties that make the
monitor a valid selector; neither has coverage.
Runtime: fast/local; the tiny fit is under a minute.
Evidence: 2026-09-15. Implemented in `task.py`: `validation_mc_draws` as a saved task
hyperparameter (`VALIDATION_MC_DRAWS_KEY`), `_predictive_monitor` on the dense stages calling
`eval/predictive.py::matched_predictive_scores` with the forward's own persistence tensor,
`validation_noise_generator` (blake2b digest of the run seed and, in batch order, each segment's
recording and floored start time; fixed at fixed batching), `recording_grouped_totals` (sum of
per-recording means and their count, reduced through the objective's `_all_reduce_sum` and
divided once), and the three columns `VALIDATION_MONITOR_SUFFIXES`, absent entirely when the
key is unset. The clip fraction: `on_before_optimizer_step` is overridden to compute the norm on
every optimizer step (`torch.nn.utils.get_total_norm`), count it against the clip, and log the
running fraction on the sampled cadence and always on the epoch's last batch, `on_step` only,
so the bare key the metric history reads during validation holds the epoch's actual fraction;
counters reset in `_on_train_epoch_start_hook`. `trainer.py`: `validation_monitor_draws`
validates the key and refuses a checkpoint or early-stopping monitor naming a monitor column
while the key is null; `_configure_validation_monitor` applies the value onto the task's
hyperparameters by the seed's route and extends `TRACKED_METRICS` **on the instance** with the
three `val/` columns, so a legacy run tracks exactly its legacy columns. Two deviations from the
task text. (1) The key and the monitors live in `target_only.yaml` and `joint.yaml` rather than
in `lag25.yaml` and the ablation profiles: `tests/test_config_load.py` pins `lag25.yaml` to
differ from `joint.yaml` in `max_lag` alone, and every arm of the matched comparison must select
one way, so the two roots carry the rule and every arm inherits it (`lag25.yaml`, the three
ablation profiles, `capacity_control.yaml`, `mean_only.yaml`, `target_only_s0_fhr.yaml`);
`default.yaml` documents the key at `null`; `tiny.yaml` sets two draws so the smoke fit logs the
columns, and `tiny_lag25.yaml` selects on the monitor so a fixture fit exercises the callbacks.
(2) The $K = 1$ identity is stated and tested against the monitor's *own* draw: the forward's
`val/nll_full_block` is scored on a global-RNG draw the monitor cannot reproduce, so the test
recomputes the decoder at the bank's draw, scores it unweighted on the objective's mask and
takes the anchor mean, which on the stub batch equals the recording-grouped mean. Executed:
`tests/test_task.py` (eleven new cases: absent without the key and on the training stage; the
$K = 1$ identity for both branches and the gap; the fixed bank across two passes while
`nll_full_block` moves; the bank keyed on recording and seed; an exact-zero gap on the
target-only arm; a fully masked batch reports zero; a non-positive count refused; the grouped
totals on a hand-built batch; the clip fraction over four steps of which one is on the cadence
is one half, resets per epoch, and is absent without a positive clip), `tests/test_trainer.py`
(the class surface excludes the columns; every shipped profile resolves the draw count it
declares; a monitor on an unlogged column is refused; malformed counts refused),
`tests/test_config_load.py` (the key named in `NON_CONSTRUCTOR_KEYS`): 111 passed.
`tests/test_train_smoke.py` (the three columns present and finite on every validation row, the
gap equal to base minus full, no `train/` twin, the clip fraction inside the unit interval on
every row): 12 passed in 20 s. A two-epoch fit of `tiny_lag25.yaml` through `trainer.main`
selected on `val/pred_nll_full_mc` with `val/total_loss` as the secondary stem, wrote both
checkpoint families, tracked the three columns and stamped `validation_mc_draws: 2` into the
task's hyperparameters. `tests/test_docs.py` passes for every file this sprint touched; its two
standing failures are the concurrent session's (change record).

#### S3-T02: Train and score the matched arms; decide the bank length

Requirements: FR-005, FR-009, NFR-002
Depends on: S2-T02, S2-T05, S3-T01, S1-T04
Description: Operator runbook, in order: (1) `target_only.yaml` seed a and seed b; (2)
`joint.yaml` (91 entries) and `lag25.yaml`, both warm-started from seed a; (3)
`capacity_control.yaml` and `mean_only.yaml` rebased on `lag25.yaml` if the 25-entry arm is
competitive, else on `joint.yaml`; optional (4) a scratch 25-entry run with the shipped run's
`lambda_base: 0.5` for context only. Score every run with the matching evaluation profile at
$K = 32$, then the finalists at 8 and 128. Set the bank-length noninferiority tolerance from the
seed-a/seed-b target-only variability **before** reading the comparison. Read every arm against
the frozen seed-b reference and record the FR-009 fields and the decision in
`eval/EVAL.md`'s run log. Compare `lambda_base` 0.5 and 1.0 on the chosen arm as one extra run.
Acceptance criteria:
- Each run directory holds `resolved_config.yaml`, the summary under the corrected schema, and
  the acceptance record naming its plan digest and band family.
- The bank-length decision cites the predeclared tolerance, the predictive interval, the
  calibration blocks and the measured resources.
- No arm is promoted on the weighted training gap or the mean-decoded gap.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/configs/capacity_control.yaml`, `mean_only.yaml` - `base:` repointed to the chosen bank profile
- `teb_vae/lag_slot_transformer_cfs/eval/EVAL.md` - runbook and run log
Validation: `-m teb_vae.lag_slot_transformer_cfs.eval.acceptance` over the run directory; expected a record with the comparisons and the tolerance. Operator, slow/external. Proposed.
Test rationale: empirical validation; no unit test.
Runtime: slow/external (days of GPU time across runs).
Evidence: 2026-09-15, runbook only. `eval/EVAL.md` section 7.5 states what the monitor is and
which profiles select on it; section 7.6 is the runbook in the order above (two target-only
fits, the tolerance written down before the comparison is read, the two banks at $K = 32$ then
8 and 128, the decision, the two controls rebased on the chosen bank, the `lambda_base` run), the
acceptance command, and the run-log template with every FR-009 field. `capacity_control.yaml`
and `mean_only.yaml` carry the rebase rule in their headers and stay on `joint.yaml` until the
decision exists; rebasing is the one `base:` line. No run has been made: the checkpoint, the
production shards and the GPU box are not on this machine. The decision record is empty.

#### S3-T03: Run the $S_0$ factorial on the chosen bank

Requirements: FR-008, FR-009
Depends on: S3-T02, S2-T04
Description: Train `lag25_s0_fhr.yaml`, `lag25_s0_up.yaml`, `lag25_s0_both.yaml` and
`target_only_s0_fhr.yaml` (rebased if S3-T02 chose 91 entries), warm-starting the source-enabled
ablations from the target-only checkpoint that shares their FHR policy. Score at $K = 32$ with
the resolved axes reported separately for the $S_0$ coordinate and the remaining channels, so
losing level context cannot hide inside the aggregate. Record the decision on each switch.
Acceptance criteria:
- Four run records with the effective-input disclosure and the split $S_0$ / non-$S_0$ scores.
- The decision states whether each switch is kept for Sprint 5 and why.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/eval/EVAL.md` - run log
Validation: acceptance pass over the four runs plus the references; operator, slow/external. Proposed.
Test rationale: empirical validation; no unit test.
Runtime: slow/external.
Evidence: 2026-09-15, runbook only. `eval/EVAL.md` section 7.7: the four cells and the
policy-matched references, which donor warm-starts which arm, the rebase if the bank decision
kept 91 entries, scoring at $K = 32$ with the ablated coordinate's score read beside the rest off
the resolved axes and `causality.effective_inputs`, and one run-log entry per cell carrying the
7.6 fields plus the keep/drop decision per switch. No run has been made.

## Sprint 4: Objective and uncertainty pilots

Forecast coverage: FR-011
Outcome: three config-gated pilots, each numerically tested and run once on the chosen arm at one
seed: separate `prior_logvar_clamp`/`obs_logvar_clamp`; `train_mc_draws` in $\{4, 8\}$ with the
per-anchor `logsumexp` objective; `proposal_penalty` at a validation-chosen strength. Each ends
with a promote/reject decision recorded beside its run. A $\beta$-NLL weighting or a Student-$t$
observation model is considered only if the residual diagnostics after these pilots show heavy
tails; neither is scheduled.
Dependencies: S3-T02 decision on the bank; S3-T01 monitor for selection.
Refine when: the S3 comparison record exists and the calibration blocks by horizon and block
show whether the central overcoverage is floor-limited.
Uncertainty: whether the multi-draw objective fits in memory at the production batch size
(measure at batch 64 first); whether any pilot changes the predictive interval by more than the
seed-to-seed variability measured in S3.

## Sprint 5: Confirmation and publication

Forecast coverage: FR-009, NFR-002, NFR-003
Outcome: the selected configuration, tolerance, scoring population and analysis families are
frozen; three training seeds per shortlisted arm; $K = 8/32/128$ on one panel; a confirmation
population whose GUID manifest is disjoint from development and statistics fitting; the headline
estimator decision revisited; `MODEL_EXPLAINED.md`, `eval/EVAL.md` and `eval/FIGURE_GUIDE.md`
updated to the measured behaviour with rejected variants' provenance retained.
Dependencies: Sprint 4 decisions; GUID manifests from the operator.
Refine when: the shortlist is fixed and the manifests are available.
Uncertainty: independence of the confirmation fold from the pretraining and statistics
populations, which only the manifests can establish.

---

## Review and change record

| Date | Finding | Severity | Disposition | Affected |
| --- | --- | --- | --- | --- |
| 2026-09-15 | The proposal edits shared `lag_attn_cfs` verdict code; this cell already owns the verdict list it writes | medium | scoped to a cell-local post-process; shared code untouched | S1-T02, section 4.5 |
| 2026-09-15 | Proposal warns that deep-merging new band names leaves old keys behind; the override delta merges over a resolved config that carries no `eval_config`, and `base:` is refused | low | no whole-map replacement needed; the acceptance band family is the real seam | S2-T02 |
| 2026-09-15 | `exploratory_bands` in the acceptance plan is a closed list, so a 25-entry run would be failed by name | high | geometry-keyed band families with legacy fallback | S2-T02 |
| 2026-09-15 | Cropping the 91-row embedding to warm-start 25 entries changes $c_L$ by 1.908 | medium | strict refusal kept; the 25-entry arm warm-starts from the target-only checkpoint | S2-T01, S3-T02 |
| 2026-09-15 | `baseline_forecasts` reads the original declared stream, so an $S_0$ ablation would leak into the persistence baseline | high | permitted-input view in `shared_readout` | S2-T04 |
| 2026-09-15 | Switching the headline estimator would break comparability with the epoch-1002 export | medium | both estimands exported; headline unchanged until S5 | S1-T03 |
| 2026-09-15 | S1-T04 was declared dependent on S1-T03, but the profile, the marker and the runbook read nothing the cohort work produces | low | dependency narrowed to S1-T02 | S1-T04 |
| 2026-09-15 | The shared `eval_config` key set is closed, so a `keep_prefix_curve` key would need shared-schema edits | low | the prefix curve is declared as `beyond_<l>` bands; no code | S1-T04 |
| 2026-09-15 | The shared registry refuses an extra analysis under a shared name, so the sufficiency block cannot be post-processed by this cell | low | sibling block `results.sufficiency_qualification` | S1-T04 |
| 2026-09-15 | Another Claude session was concurrently editing `eval/collect.py`, `lag_metrics.py`, `binding.py` and shared figure seams (per-lag maps, new analyses) | medium | all Sprint 1 edits applied as exact-anchor replacements; four `test_eval_binding.py` failures belong to that work; staged files carry both sets of hunks | S1-T01..T04 |
| 2026-09-15 | The resolved-config dump is written by the shared driver and already carries `max_lag` and `raw_per_step`; a derived `n_lags` / oldest-centre duplicate there would be a second value to keep true | low | the driver logs the resolved bank at startup (`_log_lag_geometry`) and the binding discloses `oldest_lag_seconds` in the summary; no shared-driver edit | S2-T01 |
| 2026-09-15 | Two bank lengths, or two input policies, of one arm would be grouped and seed-averaged together by the acceptance pass, and a primary comparison across them would read the bank or the ablation as the leaf it names | medium | `scoring_mismatch` refuses a pairing across windows or policies; `band_block` refuses an arm whose runs span two windows; both fields travel in the run descriptor | S2-T02, S2-T04 |
| 2026-09-15 | The shared attribution pass writes no per-channel table to mark a coordinate in; its example maps are the per-channel artefact | low | the cell's stage reads the ablated coordinate back off the example maps, writes `attribution_ablated_inputs.csv`, records `checks.ablated_input_max_abs`, and refuses a nonzero value | S2-T04 |
| 2026-09-15 | `eval/latent_probes.py` gathered the probe's persistence term from the original stream, so a probe under the FHR ablation would read the coefficient the persistence term never carries | medium | the probe gathers from `permitted_target_features`, the same view the baselines read | S2-T04 |
| 2026-09-15 | The docs gate forbids geometry literals in comments outside three allowlisted configuration files; the new profiles declare a window and cannot say what it resolves to without one | low | `configs/lag25.yaml`, `configs/tiny_lag25.yaml`, `eval/configs/lag25_eval_overrides.yaml` and the Sprint 1 tail profile added to `GEOMETRY_LITERAL_ALLOWLIST` on the allowlist's own ground | S2-T01, S2-T02 |
| 2026-09-15 | `tests/test_docs.py` still fails on two artefacts of the concurrent session: the untracked `diagrams/typeset_math.py` names a markdown document, and a docstring in `tests/test_eval_aggregate.py` carries a geometry literal | low | not this delivery's; reported, left for that session | sprint boundary |
| 2026-09-15 | The task text put `validation_mc_draws` and the monitors in `lag25.yaml` and the ablation profiles, but `test_config_load.py` pins `lag25.yaml` to one leaf against `joint.yaml`, and a comparison whose arms select on two rules is not matched | medium | the key and both monitors live in the two roots, `target_only.yaml` and `joint.yaml`; every arm inherits them; `default.yaml` keeps legacy selection at `null` | S3-T01 |
| 2026-09-15 | The $K = 1$ acceptance criterion compared the monitor with `val/nll_full_block`, but that column is scored on the forward's global-RNG draw, which a fixed bank cannot reproduce without reseeding the process RNG inside validation | low | the identity is tested against the decoder at the bank's own draw, unweighted, on the objective's mask; the gap tolerance is single-precision | S3-T01 |
| 2026-09-15 | A per-epoch clip fraction logged at `on_train_epoch_end` would reach the metric history one row late, because the CSV callback reads `callback_metrics` during validation, before the training epoch is reduced | low | the running fraction is logged `on_step` on the norm's cadence and always on the last batch, where it equals the epoch's fraction; the bare key the CSV reads is that value | S3-T01 |
| 2026-09-15 | A monitor column tracked at class level would be an empty column in every legacy run | low | `TRACKED_METRICS` is extended on the driver instance only when the configuration set the draw count | S3-T01 |
| 2026-09-15 | A checkpoint or early-stopping monitor naming a monitor column with the draw count unset fails inside the framework at the end of the first validation epoch | low | `validation_monitor_draws` refuses it by name before the model is built | S3-T01 |
| 2026-09-15 | A Windows path-length limit broke a fixture fit's checkpoint save when the run root was a long temporary path; the pytest temp root is short enough | low | not a code defect; noted for operators launching fixture fits by hand on Windows | S3-T01 |

Self-review only (correctness, coverage, executability, simplicity passes in sequence). The
structural validator was run on this document; see Resume.

## Resume

Current increment: Sprint 3 code delivered on 2026-09-15 (S3-T01 in code, configuration and
tests; S3-T02 and S3-T03 as runbooks with empty run logs). Next executable work is the
operator's: the S3-T02 runbook in `eval/EVAL.md` section 7.6, in order, then 7.7. The next
*code* task is Sprint 4's first pilot, which is refined only after the S3-T02 decision record
exists. Blockers: none for code. Outstanding checks: the sprint-boundary package run has not
been done for Sprint 2 or Sprint 3; the short-bank end-to-end evaluation fixture
(`test_eval_smoke.py -k short_bank`) now fits `tiny_lag25.yaml` under the monitor selection and
has not been rerun since. Long runs on this machine must be run in the foreground or by the
operator. The production rescore under `lag91_tail_diagnostic.yaml` (S1-T04), the resource
measurement (S2-T05) and all Sprint 3 runs need the original checkpoint, the integer-operator
shards and the GPU box, which are not on this machine.

Latest evidence: focused suites executed 2026-09-15 after the Sprint 2 edits --
`test_config_load.py`, `test_checkpoint_contract.py`, `test_eval_binding.py`,
`test_eval_lag_metrics.py`, `test_forward_contract.py`, `test_causality.py`,
`test_eval_attribution.py`, `test_trainer.py`, `test_eval_launch.py`, `test_acceptance.py`
(non-slow), `test_controls.py`, `test_construct.py`: 230 passed, 1 failed (the plan-digest pin,
moved by the revision-2 plan and re-pinned); the new and extended files afterwards: 114 passed
(`test_scattering_ablation.py`, `test_checkpoint_contract.py`, `test_config_load.py`,
`test_eval_binding.py`), 118 passed (`test_acceptance.py`, `test_eval_lag_metrics.py`,
`test_eval_figures.py`, `test_eval_attribution.py`), `test_scattering_ablation.py` 13 passed.
`test_docs.py` passes its geometry and timeline gates for this delivery's files and still fails
on two artefacts of the concurrent session (see the change record). Sprint 3, 2026-09-15:
`test_task.py`, `test_trainer.py`, `test_config_load.py`: 111 passed; `test_train_smoke.py`:
12 passed; one hand-launched two-epoch fit of `tiny_lag25.yaml` selecting on the monitor.

## Implementation conventions

Keep sprint labels and this file's name out of production identifiers, comments and user-facing
text. Google-style docstrings with `$...$` / `$$...$$` notation; comments explain why. No literal
horizon, anchor, lag-count or channel numbers in comments, docstrings, report strings or guides:
derive them from the model or the resolved config. Lag identities use the feature grid and
filter-delay terms only. Every runnable module keeps the Run-button convention and is listed in
`tests/test_eval_launch.py::ENTRY_POINTS`; none is added by this plan. Necessary tests only:
extend the existing colocated files, add the one new file `tests/test_scattering_ablation.py`,
run file-level selections during a task and the package once per sprint. Do not commit unless
asked.

## Todo checklist

### Sprint 1: Trustworthy scoring on the existing run
- [x] S1-T01: Score labels and weighted objective-parity columns
- [x] S1-T02: Interval-aware, mixture-based and residual-correct verdicts
- [x] S1-T03: Aggregation estimands and a declared lag-profile cohort
- [x] S1-T04: Old-bank tail diagnostic profile and the rescore runbook

### Sprint 2: 25-entry bank and $S_0$ switches
- [x] S2-T01: The 25-entry training profile and transfer hygiene
- [x] S2-T02: The 25-entry evaluation profile and geometry-keyed band families
- [x] S2-T03: The $S_0$ input switches in the model
- [x] S2-T04: The $S_0$ policy through evaluation, controls and attribution (code and unit
  evidence complete; the fixture end-to-end run was executing when recorded, see Resume)
- [ ] S2-T05: Measure 91 versus 25 resources at production geometry (code side delivered:
  `--max-lag`; the measurement is the operator's, on the GPU box)

### Sprint 3: Matched arms and a predictive monitor
- [x] S3-T01: Predictive validation monitor and clip-fraction counts
- [ ] S3-T02: Train and score the matched arms; decide the bank length (runbook and run-log
  template delivered in `eval/EVAL.md` 7.6; the runs and the decision are the operator's)
- [ ] S3-T03: Run the $S_0$ factorial on the chosen bank (runbook delivered in `eval/EVAL.md`
  7.7; the runs are the operator's, after S3-T02's decision)

### Sprint 4: Objective and uncertainty pilots
Forecast only; refine after the S3-T02 decision record exists. No concrete tasks yet.

### Sprint 5: Confirmation and publication
Forecast only; refine after Sprint 4 decisions and the GUID manifests. No concrete tasks yet.
