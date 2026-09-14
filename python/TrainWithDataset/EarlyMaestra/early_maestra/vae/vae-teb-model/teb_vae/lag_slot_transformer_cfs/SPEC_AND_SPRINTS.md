# FHR-Anchored Lag-Residual Transformer CFS - Spec and Roadmap

Status: IN_PROGRESS
Last updated: 2026-09-09 (Sprint 8 machinery complete; sprint 7 complete and measured; sprints 5, 6 and 8 leave nine training runs to the operator)
Owner: unassigned
Planning style: Implementation milestones (sprints)
Baseline: repository root `C:\Users\mahdi\Desktop\teb_vae_model`, revision `6cfa8be` (2026-09-08),
working tree dirty with untracked docs and the `teb_vae/lag_slot_transformer_cfs/` design folder.
Inspected scope is listed in section 3.

The single authoritative planning document for building
`teb_vae/lag_slot_transformer_cfs/`. `DESIGN.md` in the same directory is the architecture and
validation specification and is **not** restated here: every requirement below cites the design
section it comes from, and where the two disagree the design wins on *what* and this document wins
on *where and in what order*. `DESIGN_V1_SUPERSEDED.md` is history and is never a source.

Companion documents: `teb_vae/lag_attn_transformer_cfs/EVAL_DIAGNOSIS_2026-09-08.md` is the
evidence that motivated the design; `teb_vae/CFS_CRWS_MODELS_REFERENCE.md` is the shared data and
timing convention; `teb_vae/lag_attn_cfs/eval/EVAL.md` is the evaluation contract the new binding
narrows.

---

## 1. Context, outcomes, and scope

### 1.1 The problem

The shipped `SeqVaeLagAttnTrfCfs` forecaster works and its uterine-activity pathway does not. On
the held-out causal split at epoch 901 the source-conditioned branch is a **worse** predictive
density than the target-only branch by $14.159$ nats per anchor, recording-bootstrap interval
$[-14.725, -13.586]$, at $K = 8$ Monte Carlo draws. $70.1\%$ of the coupling divergence survives an
observed-zero source control, one latent coordinate of $64$ carries $72.8\%$ of the divergence, and
no occlusion band has a confidently positive predictive contribution.

The design replaces the source convolution stem and the lag cross-attention with three things: a
pointwise information-preserving representation of each available uterine-activity coefficient, a
deterministic proposal from each source lag conditioned on the anchor's fetal-heart-rate state, and
an explicit sum at the latent-parameter fusion boundary. The latent stays one $64$-dimensional
space defined by the target-only prior; uterine activity makes a bounded residual correction to the
parameters of a second Gaussian over that same space.

### 1.2 The smallest useful capability

A trainable, scorable model. Concretely: `SeqVaeLagResidualTrfCfs` constructs from a resolved
config, trains under the anchored objective on the integer-phase causal shards, refuses the
architecture keys that do not apply to it, satisfies every structural gate in design section 11.3,
and produces a matched Monte Carlo predictive gap with recording-level intervals and the
source-specificity controls of design section 8.3. That is sprints 1 through 4.

### 1.3 Goals

- One package that owns its architecture, objective reduction, evaluation binding and lag readouts.
- Zero behavioural change to the four shipped forecasters. Their numbers must remain comparable
  across this work.
- Every claim the design says must not be made is impossible to make from this code: no per-lag
  divergence allocation, no attention-shaped column, no physiological delay readout.

### 1.4 Non-goals for this delivery

- Training runs, checkpoints and their results. The roadmap builds the machinery; running it is a
  separate activity whose first arm is sprint 5.
- The seven-arm comparison of design section 11.1, the ten synthetic instruments of section 11.2
  and the multi-seed acceptance protocol of section 11.3. These are forecast sprints 6 to 8.
- Any change to `teb_vae/lag_attn_transformer_cfs/` or `teb_vae/lag_attn_cfs/nets/`. One additive
  optional field on one shared dataclass is the only shared edit this plan authorises, and
  section 5 records why.
- Student-$t$ observation likelihood, longer horizons, and alternative channel weighting. Design
  section 10 places these after the Gaussian architecture comparison.

### 1.5 Build acceptance versus post-launch measurement

**Build acceptance** is section 8: the structural and numerical gates, the smoke train, the
evaluation smoke, and the repository checks. All of it is measurable without a trained model.

**Post-launch outcomes** are the empirical questions the design closes with, and no code review can
settle them: whether the target-only latent is sufficiently informative, whether uterine activity
improves prediction beyond a capable target-only model, whether any gain survives the controls and
calibration, and whether the lag readout recovers known dependence under the real preprocessing
pipeline. They are measured by sprints 5 to 8 against the acceptance gates of design section 11.3,
and their owner is the model author.

---

## 2. Requirements and acceptance

Design section references are the authority for content. Each requirement is observable.

### 2.1 Architecture

- FR-001 [required]: The source representation is pointwise and parameter-free.
  Given a standardized source stream and its per-channel availability mask, each encoded
  coefficient is $e_{s,j} = [x^{\rm safe}_{s,j}, m_{s,j}]$ with $x^{\rm safe}$ the value where the
  mask is one and zero where it is not. The Jacobian $\partial e_{s,j} / \partial \bar U_{r,k}$ is
  exactly zero for $(r,k) \ne (s,j)$. No temporal convolution, recurrence, state-space block,
  attention, pooling, temporal normalisation or learned cross-lag competition exists on this path.
  Design section 4.3.

- FR-002 [required]: Availability is per channel and indexing is safe.
  $m_{t,\ell,j}$ is the product of the in-range indicator, the warm-up indicator against $W'_j$,
  and the accepted-finite indicator. Validity is evaluated **before** the gather; masked positions
  use an in-range surrogate index; invalid values are filled before any nonlinear operation. A
  negative index never wraps to the end of the sequence. A nonfinite value in a declared-valid
  input is rejected rather than silently zeroed. Design section 3.3.

- FR-003 [required]: One local proposal per lag.
  A learned lag embedding $\zeta_\ell \in \mathbb R^8$ identifies the fixed index. One shared
  multilayer perceptron maps $[h_t, \operatorname{vec}(E_{t,\ell}), \zeta_\ell]$ to a mean proposal
  and a scale proposal, each $\mathbb R^{d_z}$, gated by an external selector $s_{t,\ell}$ and the
  lag-validity indicator $v_{t,\ell}$. Each proposal reads exactly one source stored time. With
  the target state, masks and metadata held fixed, a proposal's derivative with respect to any
  other source lag is zero. The final output weight and bias are zero after all generic
  initialisation. Design section 4.4.

- FR-004 [required]: Fusion is an explicit sum with a bound applied after summation.
  $\bar a_t = c_L \sum_\ell r^\mu_{t,\ell}$, $a_t = a_{\max}\tanh(\bar a_t / a_{\max})$, and the
  same for the scale channel at $b_{\max}$. $c_L$ is fixed at $L^{-1/2}$ and is **not**
  renormalised by the number of available lags. Design section 4.5.

- FR-005 [required]: The full distribution is a prior-relative residual.
  $\mu^q_t = \mu^p_t + \sigma^p_t \odot a_t$ and $\lambda^q_t = \lambda^p_t + 2 b_t$, so
  $|\mu^q_d - \mu^p_d| \le a_{\max}\sigma^p_d$ and
  $e^{-b_{\max}} \le \sigma^q_d/\sigma^p_d \le e^{b_{\max}}$. The bounded log-variance is **not**
  passed through the prior's sigmoid bound a second time. Both signs of $b_t$ are permitted.
  Design section 4.5.

- FR-006 [required]: The prior conditions on an explicit value-free clock.
  $h_t = h^Y_t + W_A \operatorname{LayerNorm}(\chi_t)$ with $\chi_t$ the sinusoidal function of
  stored position alone, and $W_A$ bias-free and zero-initialised after generic initialisation. No
  source value, and no function of a source value, reaches the prior. The same $h_t$ is computed
  once and used by both the prior heads and the proposal head. Design section 4.1.

- FR-007 [required]: Source absence reproduces the prior exactly.
  With every selector zero, or with every source lag unavailable, or at zero-initialised output
  projections, $Q_t = P_t$ for every parameter setting, and the paired predictions are bitwise
  equal. A valid standardized-zero observation carries mask one and is **not** treated as absence;
  no architectural constraint forces it to equal the prior. Design sections 4.6 and 11.3.

- FR-008 [required]: Paired sampling and one shared decoder.
  One $\epsilon^{(k)}_t$ per anchor per draw, shared between branches. The decoder is the existing
  shared horizon decoder invoked twice with the identical persistence input and weights, and
  receives no target state, source value, proposal, mask, posterior parameter or encoder summary.
  Design section 4.7.

- FR-009 [required]: The forward returns the anchor-indexed contract.
  Anchor index and validity, prior and full means and log-variances, paired samples when
  requested, forecast means and log-variances, per-coordinate and total divergence, masks, and
  bounded and unbounded update summaries. Proposals only when requested for diagnostics. The
  `attn_weights`, `attended_source_heads`, `source_kl_lag_map` and `kld_per_t_per_head` keys are
  **absent**, not fabricated. Design sections 4.8 and 12.1.

- FR-010 [required]: Memory is bounded by chunking, not by detaching.
  Anchors, and optionally lags, are evaluated in chunks whose partial sums accumulate without
  detaching gradients. Complete proposals are retained only for explicitly requested diagnostic
  batches. Design sections 4.8 and 12.3.

- FR-011 [required]: Named ablation arms are constructor decisions, not later patches.
  A mean-only arm builds a $d_z$-output final layer and constructs **no** scale-proposal
  parameters. A scalar-lift arm appends a per-channel two-dimensional lift while retaining the
  identity and mask coordinates. Both are config leaves whose default is the recommended starting
  configuration. Design sections 4.3, 4.4 and 10.1.

### 2.2 Objective and training

- FR-012 [required]: The divergence is computed in residual form and agrees with the general
  formula. $K_t = \tfrac12\sum_d [a^2_{t,d} + e^{2b_{t,d}} - 1 - 2b_{t,d}]$, evaluated with
  `expm1` for the variance term and reduced in FP32 or higher. It is zero exactly at $a = b = 0$.
  Design section 5.1.

- FR-013 [required]: The objective averages over the **global** contributing-anchor count.
  Under distributed training the target is the global contributing-anchor mean, with the local
  numerator scaled so that gradient averaging over the world reproduces it. Global numerators and
  denominators are logged. Design section 6.1.

- FR-014 [required]: An empty scored batch contributes a graph-connected zero and records no
  scored data, coordinated across ranks. No denominator is clamped to one and then read as an
  observation. Design section 3.4.

- FR-015 [required]: Channel and horizon weights are resolved and normalised as stated.
  Relative channel weights renormalise to sum to $C_Y$ and horizon weights to sum to $H$; the
  reported phase share of total channel weight follows from the resolved kept-channel set and the
  configured ratio rather than from a quoted percentage. The weighted score is reported as a
  training criterion, never as an unweighted joint log density. Design section 6.1.

- FR-016 [required]: Anchor tiling, forecast masks and the divergence support are the design's.
  The per-segment phase is the existing stable hash; padded slots repeat a legal final index and
  are excluded everywhere; the coverage floor gates reconstruction, source divergence and prior
  regularisation over one anchor set. Design section 3.4.

- FR-017 [required]: The package trains from a resolved configuration with no command line.
  A production config and a tiny smoke config, a task, and a trainer whose `RUN_CONFIG` constant
  makes the module runnable from an IDE Run button, following the repository's runner convention.

- FR-018 [required]: Warm-start transfer is explicit and audited.
  Transferring target encoder, prior and decoder tensors from a target-only checkpoint lists every
  transferred, missing and reinitialised tensor, validates feature order and task geometry, and
  reinitialises the source output projections unless resuming a checkpoint of this exact model
  kind. Joint training after target-only pretraining starts a declared new optimiser, scheduler and
  divergence ramp; an exact resume restores all of it instead. Design section 12.2.

- FR-019 [required]: The model identity is persisted and incompatible keys are refused.
  Model kind and version, architecture and optimizer settings, exact channel order and kept
  indices, warm-up vectors, operator, leg alignment, forecast clock, trim, normalisation
  provenance, source-quality policy, metadata inputs, anchor stride and phase rule, seeds,
  sampling policy, score weights and selection criteria all reach the checkpoint. Source-attention,
  entmax and attention-bias keys raise rather than being accepted and ignored. Design section 12.2.

### 2.3 Evaluation

- FR-020 [required]: Matched Monte Carlo predictive scoring.
  Both branches are scored as $D^{(K)} = -\operatorname{logsumexp}_k \Lambda^{(k)} + \log K$ under
  the same noise draws, target mask, persistence and evaluation mode, giving
  $\widehat G^{(K)}$. Mean negative log likelihood and prior-mean-only decoding are not
  substitutes. Draw counts $K \in \{8, 32, 128\}$ are selectable and the concentration diagnostic
  $1/\sum_k \alpha_k^2$ is reported. Design section 7.

- FR-021 [required]: Proposal suppression and its honest framing.
  For a lag band, selectors are set to zero while the target state, metadata, original masks,
  remaining proposals and $c_L$ are held fixed; sums, limiters, posterior, paired samples and
  decoder predictions are recomputed, giving $J_{\mathcal B}$. It is not normalised to sum to the
  total gain or to the divergence, and the artifacts state that it measures reliance of the fitted
  computation rather than a unique contribution. Design sections 8.1 and 5.3.

- FR-022 [required]: Cancellation and exposure readouts.
  The cancellation ratio $\kappa_t$ is reported alongside its numerator and denominator, for both
  the mean and the scale proposals. Per-lag and per-channel exposure counts available channels and
  scored anchors, and unsupported bins are recorded as missing rather than as measured zero.
  Design sections 5.3 and 8.1.

- FR-023 [required]: Source-specificity controls.
  Explicit source-disabled selectors, which verify the equality invariant only; observed zeros,
  observed constants and mask-only source values with selectors enabled; and correct versus
  compatible cross-recording source with within-source time order preserved and zero same-recording
  pairings. Design section 8.3, items 1 to 3.

- FR-024 [required]: The evaluation binding exposes a reduced, honest registry.
  Analyses that read only latent parameters, forecasts and masks are retained; those that read
  attention-shaped tensors are excluded by name with the exclusion recorded in the run artifacts.
  No column carries a proposal norm under an attention name. Design sections 5.2 and 12.1.

- FR-025 [required]: Calibration uses the predictive mixture.
  Intervals and probability-integral-transform values come from the mixture cumulative
  distribution function or from predictive sampling, never from a mean of conditional standard
  deviations. Recording-level bootstrap is the interval, and equal-recording and anchor-weighted
  summaries are reported separately. Design section 7.

### 2.4 Deferred

- FR-026 [deferred]: Student-$t$ observation likelihood at fixed $\nu_T = 5$ with
  variance-matched initialisation. Design section 10.2 places it after the Gaussian architecture
  comparison, and it changes both branches' observation model at once. Reconsider when the
  Gaussian arm has a matched predictive gap and calibration diagnostics identify the factorised
  Gaussian as the binding error.
- FR-027 [deferred]: Longer horizons and alternative channel weighting. Design section 10.3.
  Reconsider when the first arm's horizon-resolved scores are available on the common ten-step
  task.
- FR-028 [deferred]: Proposal-norm regularisation. Design section 5.3 is explicit that it is an
  ablation and not an identifiability result, and that it must not enter the main objective
  silently. Reconsider only with a declared arm.

### 2.5 Allocated to forecast sprints

In scope for the roadmap, not for the first delivery. Each is allocated to a forecast sprint in
section 7 and is refined into tasks before it is executed.

- FR-029 [required]: The first trained arms exist.
  A competitively trained target-only forecaster on the same task, an independently trained frozen
  target-only predictor held as an external reference, and one jointly trained candidate warm-started
  from the first with its source output projections at zero. The internal base can change during
  joint training, so an improved internal gap alone is insufficient. Design section 6.3.
  Allocated to Sprint 5.

- FR-030 [required]: Mechanism-separating comparisons.
  The retrained attention reference under matched sampling policy and clock, the pointwise-source
  plus attention arm, the mean-only arm, and the target-only capacity control, each with actual
  parameters, compute and memory reported, and each changing one declared thing at a time.
  Design section 11.1. Allocated to Sprint 6.

  Resolved there as **five configurations of this package** forming a chain in which every adjacent
  pair differs in one constructor leaf, rather than as a comparison against the shipped forecaster:
  that model carries none of the four things a comparator must hold fixed. The capacity control is a
  fourth construction that holds the pathway and withholds the source values, not the target-only
  arm, which removes the pathway and its budget with it.

- FR-031 [required]: Synthetic recovery instruments.
  The stored-feature instruments and the raw-signal simulations passed through the actual causal
  feature pipeline, with recovery bands and error criteria predeclared per generator, and power and
  false-positive rates reported over repeated simulations. Design section 11.2.
  Allocated to Sprint 7. **Met**, and unlike the other allocated requirements this one needed no
  production shard: an instrument generates its own data, so its rates are measured rather than
  owed. The false-positive rate is zero over three controls and power is four of seven, with the
  three misses the three the design predicts.

- FR-032 [required]: Empirical acceptance.
  Stable matched held-out improvement over the internal base **and** a competitive independent
  target-only predictor, without base degradation explaining the result; Monte Carlo stability across
  draw counts and sampling seeds; appropriate calibration; informative latent probes; the source
  specificity controls; at least three training seeds for shortlisted models with recording-level
  bootstrap intervals; predeclared primary comparisons and bands; and confirmation on an untouched
  outer fold or reserved final partition. Design section 11.3. Allocated to Sprint 8.

  **Its machinery is built and none of its outcomes is measured**, which is the same standing
  Sprints 5 and 6 have: the protocol reads a directory of finished runs, and there are no trained
  arms to put in one. Two of its clauses needed new artifacts rather than a new reading of existing
  ones -- the latent probes, which no earlier sprint built, and the scored-recording provenance,
  without which a reserved partition can be declared and not checked.

### 2.6 Quality requirements

- NFR-001 [required]: Peak memory at the production tiling fits the target device.
  Workload: batch $128$, padded anchors $32$, lags $91$, latent $64$, FP32, one optimizer step
  with gradients. Each proposal array alone is $91$ MiB and the pair is $182$ MiB before gradients
  and hidden activations; dense $156$-anchor construction is $443.625$ MiB per array. Measurement:
  `torch.cuda.max_memory_allocated` around one training step and one dense evaluation step,
  recorded with the resolved chunk sizes. The design is explicit that removing attention does not
  automatically make this model cheaper, so the number is measured and not assumed.
  Design section 4.8.

- NFR-002 [required]: Numerical agreement.
  Algebraic identities are checked in FP64 at starting tolerance $10^{-10}$ absolute and $10^{-8}$
  relative; model execution tolerances are measured separately in FP32 and recorded rather than
  assumed. Chunked and unchunked forward and backward agree within the declared tolerance.
  Positive divergence is never demanded at initialisation, where zero is intentional.
  Design section 13.1.

- NFR-003 [required]: Distributed correctness without unused-parameter handling.
  Every constructed head is evaluated on safely filled tensors with selector and mask
  multiplication in the graph. No Python branch omits a learned parameter on a rank whose source is
  unavailable. Ranks with no scored anchors participate safely. The global objective and gradient
  match a single-process combined-batch reference. Design section 12.3.

- NFR-004 [required]: The stored uterine-activity timeline is canonical.
  No mechanical shift, sensor delay, acquisition shift, `up_shift_secs` or `tau_pre` term appears
  in any lag formula, plot axis, caption, evaluation record, config comment or document produced by
  this work. Lag identities use only the feature grid and the filter-delay terms. This is both the
  repository rule in `CLAUDE.md` and design section 3.1. Measurement: a grep gate in the package's
  documentation test.

- NFR-005 [required]: Documentation and comment discipline.
  Google-style docstrings with LaTeX for symbols and mathematics. Horizon, anchor, block and
  channel counts in comments, docstrings and report strings are symbolic or read from the model,
  never written as literals.

- NFR-006 [required]: The four shipped forecasters are bitwise unaffected.
  Measurement: the one shared edit this plan authorises is an additive optional dataclass field
  with an empty default, and the existing bindings' test files pass unchanged.

---

## 3. Current state and evidence

All paths below were read at revision `6cfa8be`.

### 3.1 What the new model reuses unchanged

| Component | Path | Why it is reused |
| --- | --- | --- |
| Warm-up mask, lag floor, anchor tiling | `teb_vae/lag_attn_cfs/nets/causal_inputs.py` | Owns `_set_causal_inputs`, `_validate_causal_geometry`, `_build_adapter`, `_build_anchor_index`, `anchor_ceiling`, `_resolve_warmup_readout_constants`. None of these names an attention module. |
| Target gather, channel weights, objective seam | `teb_vae/lag_attn_cfs/nets/causal_feature_target.py`, `teb_vae/lag_attn_fs/nets/feature_target.py` | `_build_forecast_target`, `_anchor_target_values`, `scored_weight`, `_default_decoder_out_channels`, the resolved forecast gaps. |
| Target-only prior head | `teb_vae/lag_attn_rws/nets/heads.py` | `FullLatentPriorHead` already has the bounded mean, the pre-bound raw log-variance and a zero-initialised clock projection at an arbitrary clock width. Its residual bodies are `geometric_schedule(d_model, d_z, 4)`, which at $128 \to 64$ is exactly the $128, 111, 97, 84, 74, 64$ schedule the design names. |
| Shared horizon decoder | `teb_vae/lag_attn/nets/decoders.py` | `HorizonDecoderCore` and `BaselineFutureDecoder`, including the per-step-and-channel persistence weight seeded at `PERSISTENCE_DECAY_HALFLIFE = 5.0`, which is the design's $\omega_{h,c} = 2^{-(h-1)/5}$. |
| Target encoder, gates, adapters, blocks | `teb_vae/lag_attn_transformer_rws/nets/encoders.py`, `teb_vae/lag_attn/nets/delays.py`, `teb_vae/lag_attn/nets/encoders.py`, `teb_vae/lag_attn/nets/blocks.py` | `CausalConvTransformerEncoder`, `ChannelGate`, `AvailabilityInputAdapter`, `ResidualMLP`, `geometric_schedule`, `smooth_bound`, `initialization`, `init_depthwise_`. |
| Per-element and per-anchor score terms | `teb_vae/lag_attn_rws/nets/losses.py` | `raw_sample_score`, `masked_raw_block_per_anchor`, `kld_tensor`, `horizon_decay_weight`. |
| Masks | `teb_vae/lag_attn_rws/nets/raw_masks.py` | `forecast_mask`, `contributing_anchors`, `kl_mask`, including the scatter-by-maximum that keeps padded anchors out of the divergence support. |
| Warm-up budget to constructor keywords | `teb_vae/lag_attn_cfs/model_kwargs.py`, `teb_vae/lag_attn_cfs/warmup_budget.py` | `warmup_model_kwargs` resolves by `inspect.signature` on the model class, so a new class keeping the same keyword names inherits the whole resolution and its refusals. |
| Checkpoint loading | `train/graph_models_utils.py` | Required by the repository rules; `load_checkpoint_strict` and `check_model_class`. |
| Launch-argument merge | `teb_vae/lag_attn_cfs/eval/launch.py` | `resolve_launch_args`, `missing_required`. Standard-library only by design. |

### 3.2 What blocks reuse, and where

Four findings shape the plan. Each was read in the source rather than inferred.

**The architecture parent constructs what the design forbids.**
`SeqVaeLagAttnTrfRws.__init__` at `teb_vae/lag_attn_transformer_rws/nets/model.py` builds
`lag_attn`, `query_proj`, `posterior_head`, `te_analysis`, `source_adapter` and one of
`source_encoder` or `source_kv_stem`. Design section 12.1 forbids constructing unused attention
modules. The new package therefore supplies its own small architecture base; section 5.1 gives the
resolution order that makes the two target-domain mixins still apply.

**The inherited forward calls attention directly.**
`CausalWarmupInputs.forward` poses a query from the prior, calls `self.lag_attn`, calls
`self.posterior_head` and calls `self.te_analysis`. It cannot execute this design unchanged, which
the design itself states. `build_lag_mask` in the same mixin reads `self.lag_attn.L`, and
`_prior_clock` encodes a zeroed source through `self.encode_source_kv`. All three are overridden in
the new package.

**Three objective reductions clamp a local denominator.**
`masked_raw_likelihood`, `masked_source_kl` and `masked_prior_rate` in
`teb_vae/lag_attn_rws/nets/losses.py` each compute `n_anchors = <mask>.sum().clamp_min(1.0)` and
divide a per-rank numerator by it. That is a per-rank mean with a clamped denominator, which
requirements FR-013 and FR-014 both reject. The per-element and per-anchor terms above them are
reusable; only the reduction is not.

**The collection pass reads attention keys unconditionally.**
`evaluate_batch` in `teb_vae/lag_attn_cfs/eval/metrics.py` reads `outputs["source_kl_lag_map"]`,
`outputs["attn_weights"]` and `outputs["kld_per_t_per_head"]` in one contiguous region, and
`Collector` in `collect.py` maps the `attention` analysis to `("attn_weights",)`. Separately,
`merged_analysis_functions` in `teb_vae/lag_attn_cfs/eval/run.py` can only **add** analyses to a
fixed shared registry; a binding has no way to remove one. Sprint 4 addresses both.

### 3.3 Task geometry, as the design fixes it

The first experiment is the current integer-phase, unaligned, stored-clock arm. From
`teb_vae/lag_attn_transformer_cfs/configs/default.yaml` and design section 3.1: stored steps $300$
at $4$ s, horizon $10$, first eligible anchor $134$, candidate source lags $91$, declared target
channels $80$ of which $76$ survive the $134$-step budget, source channels $46$, model width $128$,
latent width $64$, training anchor stride $5$ giving $32$ padded anchors against $156$ dense, block
$760$ coefficients. These numbers belong in the configuration and in this document; NFR-005 keeps
them out of code comments.

### 3.4 Test layout and commands

Tests are colocated per package as `teb_vae/<pkg>/tests/test_<module>.py`, run with

```
.venv/Scripts/python.exe -m pytest teb_vae/<pkg>/tests/test_<file>.py -q
```

from the repository root. `teb_vae/lag_attn_transformer_cfs/tests/` holds 28 files and about
$9{,}700$ lines and is the closest model for the new suite. Per `CLAUDE.md`, whole-package runs are
reserved for changes in modules several packages import, and
`teb_vae/lag_attn_rws/tests` in full takes 20 to 30 minutes. Per the recorded working note, slow
suites on this machine die when detached, so any run beyond the tool timeout is handed to the
operator with a `!` prefix rather than backgrounded.

Fixtures: `teb_vae/lag_attn/tests/fixtures/tiny_shard_causal_int.hdf5` and
`tiny_stats_causal_int.hdf5` are the integer-operator tiny shards that
`teb_vae/lag_attn_transformer_cfs/configs/tiny.yaml` points at. Both are present in the working
tree and untracked.

### 3.5 Evidence status

| Statement | Status |
| --- | --- |
| Every path, class, function and constant named above | Code fact, read at `6cfa8be` |
| The epoch-901 metrics and the diagnosis findings | Reported in `EVAL_DIAGNOSIS_2026-09-08.md`; not rerun here |
| The design's own numerical audit table | Reported in `DESIGN.md` section 13.2; standard-library arithmetic, not a model test |
| The four scoping decisions in section 5 | User decisions, 2026-09-09 |
| Peak memory, chunk sizes, FP32 tolerances, gradient-norm distribution | Unmeasured. Tasks T014, T016 and T020 measure them |

---

## 4. Proposed approach

### 4.1 Package layout

```
teb_vae/lag_slot_transformer_cfs/
  DESIGN.md                      exists
  SPEC_AND_SPRINTS.md            this document
  __init__.py
  nets/
    __init__.py
    pointwise_source.py          FR-001, FR-002, FR-011 scalar lift
    lag_updates.py               FR-003, FR-004, FR-005, FR-011 mean-only
    core.py                      the architecture base: builds only what this design needs
    model.py                     SeqVaeLagResidualTrfCfs, the anchored forward
    objective.py                 FR-012 to FR-015, the global-N reduction
    controls.py                  FR-021, FR-023, the interventions
  task.py                        FR-017
  trainer.py                     FR-017, FR-018
  configs/
    default.yaml                 FR-019
    tiny.yaml
  eval/
    __init__.py
    binding.py                   FR-024, the declaration
    predictive.py                FR-020, FR-025, the scoring loop
    lag_metrics.py               FR-021, FR-022
    memory.py                    NFR-001, NFR-002, entry point
    run.py                       entry point
    verify.py                    entry point, stdlib only
    configs/eval_overrides.yaml
    EVAL.md                      the contract and the gate checklist
  configs/
    default.yaml                 the joint candidate's geometry and optimisation
    target_only.yaml             FR-029, the baseline and the frozen reference
    joint.yaml                   FR-029, the warm-started candidate
    tiny.yaml
  tests/
```

`eval/predictive.py` is one module beyond the list this plan first drew, and it is the natural
split rather than an addition: `binding.py` declares what the pipeline cannot derive about the
model, and folding a draw loop, a concentration diagnostic and a mixture calibration census into a
declaration file would give one module two responsibilities.

### 4.2 The class and its resolution order

```python
class SeqVaeLagResidualTrfCfs(
    CausalWarmupInputs,          # warm-up mask, anchor tiling, adapter and geometry checks
    CausalFeatureForecastTarget, # target gather, channel weights, scored clock
    LagResidualCore,             # new: builds only this design's modules
): ...
```

`LagResidualCore` is a plain `nn.Module` that owns the geometry object, the channel gates, the two
availability adapters, the target conv-Transformer encoder, the prior head with its explicit clock
width, the pointwise source encoder, the lag proposal head, the horizon core and the shared
decoder, and the hooks the two mixins call into: `_build_channel_gate`, `_build_adapter`,
`_default_decoder_out_channels`, `_check_persistence_target` and `kld_tensor`. It builds no
attention module of any kind.

Three mixin members are overridden in `model.py` rather than inherited: `forward`, because the
inherited one calls attention directly; `build_lag_mask`, because the inherited one reads
`self.lag_attn.L` and because this design's mask is per channel; and `_prior_clock`, because the
design replaces the zero-source encoding with the sinusoidal function of stored position.

Two mixin members delegate to `super()` on the shipped arm and must therefore find a real
implementation below them: `CausalFeatureForecastTarget._build_forecast_target` delegates under the
stored clock, and `_build_adapter` delegates on an ungated stream. `FeatureForecastTarget` supplies
the first; `LagResidualCore` supplies the second.

### 4.3 Data flow of one forward

1. Build the anchor set and its validity companion from the stride and the per-sample phase.
2. Concatenate the declared target blocks; gather the anchor's own target vector for persistence
   **before** the gate; apply the target gate; apply the source gate.
3. Encode the target through the availability adapter and the causal conv-Transformer, giving
   $h^Y$.
4. Form the sinusoidal clock and add its zero-initialised projection, giving $h$.
5. Produce the prior mean, bounded log-variance and pre-bound log-variance from $h$.
6. Encode the source pointwise: value-with-mask per channel per stored step, no parameters.
7. Gather, per anchor and lag, the source encoding at $t - \ell$ under the per-channel validity
   mask, with an in-range surrogate index for invalid positions.
8. Run the shared proposal multilayer perceptron over $[h_t, \operatorname{vec}(E_{t,\ell}),
   \zeta_\ell]$, chunked over anchors and optionally lags, multiplied by the selector and the lag
   validity.
9. Sum the proposals with $c_L$, bound the sums, and form the full mean and log-variance as a
   prior-relative residual.
10. Draw one shared $\epsilon$ and produce both latents; gather both at the anchors; call the one
    decoder twice with the identical persistence input.
11. Compute the residual-form divergence per coordinate and its total; assemble the output dict.

Step 5 happens before step 8 because the proposal head conditions on $h$; step 6 is independent of
everything and can be computed once per batch rather than per anchor.

### 4.4 The objective

`nets/objective.py` composes the existing per-element and per-anchor primitives and owns exactly
one thing the shared module does not offer: the reduction. It builds the forecast mask and the
divergence support from the shared mask module, calls `masked_raw_block_per_anchor` for both
branches, computes the residual divergence and the prior rate per anchor, and then reduces once by
the global contributing-anchor count with an all-reduce, scaling the local numerator so that the
distributed gradient average reproduces the global mean. On an empty global support it returns a
graph-connected zero and records no scored data.

The alternative, editing the three shared reductions, was rejected: it would change the effective
objective and the reported nats of the four shipped forecasters, so every existing checkpoint and
result table would stop being comparable to anything trained afterwards.

### 4.5 Evaluation

The binding narrows the shared registry. `ModelBinding` in `teb_vae/lag_attn_cfs/eval/binding.py`
gains one optional field, `excluded_analyses`, defaulting to an empty tuple, and
`merged_analysis_functions` removes those names after merging, refusing a name the registry does not
hold. The existing two bindings pass nothing and are behaviourally unaffected. The excluded set,
with a reason for each, is recorded in the run's `summary.json` so a reader sees which columns are
absent and why.

**The collection pass itself is not reused, and Sprint 4 established that no guard would make it
reusable.** Section 3.2 recorded three unconditional attention-key reads inside `evaluate_batch` and
concluded that guarding them would let this model through. It would not, for a reason that is about
the anchor axis rather than the attention: the readout that function returns declares eight
attention-derived fields as required, and every latent readout in it pairs a dense `(B, T)` support
with a latent produced at every stored step -- `kl_mask` returns dense `(B, T)` explicitly *because*
the tensors it gates are `(B, T, d_z)`. This architecture's latents are `(B, A, d_z)`. Satisfying
the readout would mean fabricating the eight fields, which design section 12.1 forbids by name.

So the package scores through its own pass, and reuses everything that says nothing about an
architecture: the checkpoint and task loading, the override merge and its schema, the per-anchor
block score, the log-mean-likelihood, the derangement, the recording-level bootstrap and the summary
assembly. The one shared edit this plan authorises is the registry field, which is additive with an
inert default -- NFR-006's measurement.

Since Sprint 9 the pass also resolves every arm by horizon step and by stored target block, pairs
every margin per recording, reads the lag axis at every lag -- a latent profile over the whole split
and a capped predictive one -- and draws its figures from the assembled summary alone, so a figure
and its number cannot disagree and the set is redrawable from a finished directory. `EVAL.md` section
4.1 records the order the three lag readouts are read in and `FIGURE_GUIDE.md` each figure.

### 4.6 What this design deliberately does not build

No per-lag divergence allocation, because design section 5.2 proves none exists: the cross terms in
$\|c_L\sum_\ell r_\ell\|^2$ can reinforce or cancel, two scalar proposals of $1$ and $-1$ give a
total of zero against a sum of isolated single-lag divergences of $0.5$ nats, and the bound applied
after summation does not restore additivity. No `source_kl_lag_map` in any form. No attention-shaped
column fed from proposal norms. No physiological delay readout: design section 9 records a content-
lag spread of roughly $1021$ s against a $364$ s search window, and states that a pointwise encoder
cannot undo it.

---

## 5. Decisions, assumptions, and risks

### 5.1 Decisions

| Item | State | Evidence / rationale | Resolution / owner | Affected work |
| --- | --- | --- | --- | --- |
| New architecture base in the new package, composing the two CFS target-domain mixins; no attention module constructed | resolved | User decision 2026-09-09. `SeqVaeLagAttnTrfRws.__init__` builds six modules this design does not use; design section 12.1 forbids constructing them | Author | T004, T005, FR-001 to FR-011 |
| The package owns its objective reduction; shared losses are untouched | resolved | User decision 2026-09-09. The three reductions clamp a local denominator; changing them would move the four shipped forecasters' reported nats | Author | T010, FR-013, FR-014, NFR-006 |
| Reduced evaluation registry plus proposal readouts | resolved | User decision 2026-09-09. Design section 5.2 forbids reporting a proposal norm as an attention allocation | Author | T017 to T021, FR-021, FR-024 |
| Detailed sprints cover model, training, gates, evaluation and controls; arms, synthetics and multi-seed acceptance are forecasts | resolved | User decision 2026-09-09 | Author | Sprints 1 to 4 detailed; 5 to 8 forecast |
| One additive, default-inert edit to shared evaluation code: `ModelBinding.excluded_analyses`, with the registry exclusion in `merged_analysis_functions` | resolved | Both default to current behaviour, and the second half of the original assumption -- an attention-key guard in `evaluate_batch` -- was **withdrawn** in Sprint 4: `BatchReadout` requires eight attention-derived fields and the pass indexes a dense stored-step support against an anchor-indexed latent, so no guard makes it reusable | Verified by T017: both shipped bindings' test files pass, the causal cell's field pin extended by one entry | T017, NFR-006 |
| Mean-only and scalar-lift are constructor arms from the start | assumed | Design sections 4.4 and 12.3 require the mean-only model to construct no scale head, which cannot be retrofitted without changing the module tree and therefore the checkpoint key set | Author | T003, T006, FR-011 |
| Model kind `fhr_lag_residual_cfs_v1`, class `SeqVaeLagResidualTrfCfs`, package `teb_vae/lag_slot_transformer_cfs` | resolved | Stated in `DESIGN.md` front matter | - | T004, T012 |
| The stored uterine-activity timeline is used directly everywhere | resolved | `CLAUDE.md` project rule and design section 3.1 | Enforced by the grep gate in T013 | NFR-004 |

### 5.4 What this repository can build and what it cannot run

Recorded here because Sprint 5 is the first sprint where the two diverge, and every sprint after it
inherits the boundary.

**Buildable and verifiable here**: every module, every configuration, every invariant, and the whole
three-run sequence at fixture scale. `tests/test_arms.py` fits the target-only arm, fits it again at
a second seed, warm-starts the candidate from the first, scores two of them and runs the acceptance
gate over both -- in about seventeen seconds against the committed integer-operator fixture. The
production **geometry** is measurable too, because peak memory and reassociation are properties of
the shapes rather than of the coefficients: `eval/memory.py` reads no data at all.

**Not runnable here**: the training runs themselves. They need the integer-operator production
shards, which this machine does not carry -- `dataset_config` points at `/data1/...`, absent -- and
the seven devices the production configuration describes, against one. That is a property of the
environment rather than of the plan, and no amount of code changes it.

The consequence for how this document should be read: a sprint's machinery being done is not the
same statement as its outcome being delivered. Sprint 5's outcome is three trained checkpoints and
their comparison, and T027 stays open until an operator has them.

### 5.2 Risks

| Risk | Likelihood | Impact | Mitigation | De-risked in |
| --- | --- | --- | --- | --- |
| Peak memory exceeds the device at the production tiling, because the proposal arrays are $182$ MiB before gradients and the design warns that removing attention does not automatically make the model cheaper | medium | high: forces a smaller batch, which changes the arm's comparability | Chunk anchors and optionally lags from the first implementation rather than adding chunking later; measure before the smoke train | T014, T016 |
| Gradients never escape the zero-initialised output projection, leaving the source path permanently inert | low | high: the model would train as a target-only forecaster and report a zero gap | Design section 4.6 predicts hidden-layer gradients may be zero on the first step. A nondegenerate toy asserts that the predictive gradient reaches the final source projection | T009 |
| The two shared evaluation edits are not as inert as intended | low | medium: the shipped cells' numbers move | Both default to current behaviour; the existing bindings' tests are run unchanged as the acceptance evidence | T017 |
| The mixin resolution order silently picks a wrong member, as it already can in the shipped diamond | medium | medium: a wrong number rather than an exception | Assert the linearisation as a list of class names, as `teb_vae/lag_attn_transformer_cfs/tests/test_task.py` already does for the task diamond | T005 |
| Floating-point summation order changes with the chunk size, so a chunk-size edit silently moves results | medium | medium | Accumulate in FP32 or higher, test chunked against unchunked with an explicitly measured tolerance, and record the chunk sizes in the resolved config | T016 |
| The proposal suppression readout is over-interpreted as a lag contribution | medium | high: an unsupportable scientific claim | The parameterisation ambiguity of design section 5.3 is written into the artifact text and the analysis docstring, not only into this document | T019 |

### 5.3 Recorded limitations

```
lean-limit: the objective reduction is duplicated in this package rather than shared;
reconsider when a second model needs the global contributing-anchor mean; likely option
lifting the reduction into teb_vae/lag_attn_rws/nets/losses.py behind an explicit
reduction-policy argument whose default is the current per-rank clamped mean.
```

```
lean-limit: the evaluation binding excludes the attention-dependent analyses by name, so the
cross-cell summary table compares only the shared columns; reconsider when a second
attention-free cell exists; likely option a declared column subset shared by both bindings.
```

```
lean-limit: source availability is the deterministic warm-up schedule, with no measured
source-quality annotation; reconsider when a genuine uterine-activity quality mask exists;
likely option giving its causal metadata to both branches and versioning that conditioning
change, per design section 3.3.
```

---

## 6. Data, contracts, and failure behaviour

### 6.1 The forward contract

| Tensor | Shape |
| --- | --- |
| Declared target and source inputs | $(B, T, C_Y^{\rm declared})$, $(B, T, C_U)$ |
| Kept and encoded target | $(B, T, C_Y)$, $(B, T, d_h)$ |
| Encoded pointwise source | $(B, T, C_U, 2)$ |
| Anchor index and validity | $(B, N_A)$ |
| Gathered source input | $(B, N_A, L, C_U, 2)$ |
| Mean and scale proposals, each | $(B, N_A, L, d_z)$ |
| Prior and full parameters and samples, each | $(B, N_A, d_z)$ |
| Per-coordinate divergence | $(B, N_A, d_z)$ |
| Total divergence | $(B, N_A)$ |
| Base and full output means and log-variances, each | $(B, N_A, H, C_Y)$ |
| Forecast mask | $(B, N_A, H)$ |

The second axis of every latent tensor is an **anchor** axis, not a time axis. Design section 4.8
is explicit that a dense export must map anchors to time. The divergence support remains dense
$(B, T)$ because the shared mask module scatters it back, which is the one place the two conventions
meet.

### 6.2 Refusals

| Condition | Behaviour |
| --- | --- |
| An unknown or incompatible constructor key | `TypeError` from the signature, naming the key |
| `source_attention_blocks`, `source_attention_window`, `use_entmax`, `lag_bias_init`, `alibi_slope_scale`, `lag_kv_source`, `query_uses_logvar`, `d_head`, `posterior_logvar_mode`, `delta_mu_scale`, `delta_logvar_scale` | `ValueError` naming the key and stating that this architecture has no attention or independently bounded posterior. Never accepted and ignored |
| A nonfinite value in a declared-valid input or target | Rejected with an explicit message; not silently multiplied by a zero mask |
| A warm-up vector without its keep-index, a stride leaving a phase with no anchor, a floor below what the kept channels require | The existing mixin refusals, unchanged |
| A mean-only model handed scale-proposal state at load | `ValueError` from the strict state-dict check, naming the unexpected keys |
| Anchor stride above one with no phase | The existing refusal in `_build_anchor_index` |

### 6.3 Checkpoint compatibility

There is none with `SeqVaeLagAttnTrfCfs`, and the model kind makes that explicit. Matching latent
and decoder dimensions do not make the old source fusion semantically compatible. Warm starts
transfer target encoder, prior and decoder tensors only, after validating feature order and task
geometry, and list every transferred, missing and reinitialised tensor.

---

## 7. Sprint overview

| Sprint | Goal | Usable outcome / demo | Dependencies | Detail |
| --- | --- | --- | --- | --- |
| Sprint 1 | The latent-residual computation, verifiable without a model | A `python -c` session builds the pointwise encoder, the proposal head and the fusion, and the tests prove locality, safe indexing, the divergence identity and exact source-off equality | none | Detailed |
| Sprint 2 | The model: an anchored forward end to end | `SeqVaeLagResidualTrfCfs` constructs from the tiny config and returns the full contract on the tiny shard; structural gates pass | Sprint 1 | Detailed |
| Sprint 3 | A run that trains | The trainer runs to completion on the tiny config from the Run button, with the global-N objective and the audited warm start | Sprint 2 | Detailed |
| Sprint 4 | A run that is scored | The evaluation entry point produces a `summary.json` with the matched predictive gap, the proposal readouts and the three source controls | Sprint 3 | Detailed, **done** |
| Sprint 5 | The first real arm | A target-only pretrained checkpoint and one jointly trained candidate, with its evaluation run | Sprint 4 | Detailed, **machinery done; the three runs are the operator's** |
| Sprint 6 | Mechanism-separating comparisons | The arms of design section 11.1 | Sprint 5 | Detailed, **machinery done; the five runs are the operator's** |
| Sprint 7 | Synthetic instruments | The generators of design section 11.2 with predeclared recovery bands | Sprint 5 | Detailed, **done, with its rates measured** |
| Sprint 8 | Acceptance | The protocol of design section 11.3 reads a directory of runs: the seeds, the declared comparisons, the corrected band search, the probes and the reserved partition | Sprints 6 and 7 | Detailed, **machinery done; the runs are the operator's** |

---

## Sprint 1: The latent-residual computation

Goal: the three new computational pieces exist as framework-free modules with their mathematics
proved, before any model is built around them.

Demo: from the repository root, construct `PointwiseSourceEncoder`, `LagProposalHead` and the
fusion on random tensors; show that a proposal's Jacobian with respect to any other source time is
exactly zero, that all-selectors-off reproduces the prior parameters bitwise, and that the residual
divergence matches the general diagonal-Gaussian formula in FP64.

Definition of Done: T001 to T003 done with evidence; the new test files pass; no module in
`nets/` imports Lightning or any training framework, matching the convention the sibling packages
enforce with their own framework-free test.

Dependencies: none.

**Met, 2026-09-09.** `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/ -q`
reports **69 passed in 0.80s**. The framework-free property was checked by grep over `nets/` and
holds: the only third-party import there is `torch`. The forbidden-timeline-term grep over the
package returns nothing, and no source file, docstring or comment in the package references a
Markdown document. No tracked file outside this package was modified, so the four shipped
forecasters are untouched.

### Tasks

#### T001: Package skeleton and the pointwise source encoder

Requirements: FR-001, FR-002, NFR-004, NFR-005
Depends on: none
Description: Create the package and `nets/pointwise_source.py`. The encoder takes the gated source
stream and a per-channel availability specification and returns the value-with-mask encoding, plus
the per-anchor per-lag gather with safe indexing. Availability is the product of the in-range
indicator, the warm-up indicator against the combined source steps the mixin already resolves, and
the accepted-finite indicator. The gather evaluates validity **before** indexing and uses an
in-range surrogate index where invalid, so a negative index cannot wrap. Nonfinite values in a
declared-valid input raise rather than being multiplied by zero. No parameters, no temporal
operator of any kind. Reuse `_combined_source_steps` from `CausalWarmupInputs` for $W'_j$ rather
than re-resolving it.
Acceptance criteria:
- Encoding a stream returns $(B, T, C_U, 2)$ with the value where the mask is one and exactly zero
  where it is zero.
- The gather at anchor $t$ and lag $\ell$ returns the encoding at stored step $t - \ell$, and the
  mask is zero wherever $t - \ell$ is out of range or below the channel's warm-up.
- At an anchor and lag whose index is in range but whose channel has not warmed, the mask is zero
  while the index is still valid: index support and feature warm-up are separate conditions.
- A nonfinite value in a position the mask marks valid raises a `ValueError` naming the channel and
  step.
- `torch.autograd.functional.jacobian` of one encoded coefficient with respect to the stream is
  exactly zero off the diagonal.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/__init__.py` - new, package marker
- `teb_vae/lag_slot_transformer_cfs/nets/__init__.py` - new
- `teb_vae/lag_slot_transformer_cfs/nets/pointwise_source.py` - new, the encoder and the gather
- `teb_vae/lag_slot_transformer_cfs/tests/__init__.py` - new
- `teb_vae/lag_slot_transformer_cfs/tests/conftest.py` - new, repository-root pin and seeded stream
  fixtures, spliced from `teb_vae/lag_attn_transformer_cfs/tests/conftest.py`
- `teb_vae/lag_slot_transformer_cfs/tests/test_pointwise_source.py` - new
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_pointwise_source.py -q`
from the repository root. Expected: all pass. Status: executed.
Test rationale: new module, and the two failures it must catch are silent ones. A negative index
that wraps to the end of the sequence gathers real future data with every shape correct, and a
warm-up condition conflated with an index condition reports a channel as available for up to
$W'_j$ steps before it is. Neither raises anywhere else in the repository.
Runtime: fast/local. Observed as part of the package suite below.
Evidence: 2026-09-09, at `6cfa8be` plus this change. 25 tests pass. One acceptance criterion was
corrected while writing them: the plain arm's state dict is **empty** rather than carrying the
warm-up vector, because that buffer is non-persistent, which is the family's convention for a
budget-shaped tensor and is what keeps a checkpoint trained at one budget from failing to load at
another as a missing key. The test asserts the empty state dict and reads the vector off the module
instead.

#### T002: Lag embeddings, the shared proposal head, and the selector

Requirements: FR-003, FR-011, NFR-005
Depends on: T001
Description: Add `nets/lag_updates.py` with the learned lag embedding table initialised at standard
deviation $0.02$, and the shared multilayer perceptron mapping the concatenation of the target
state, the flattened per-lag source encoding and the lag embedding to the proposal pair. Layers are
three linear maps at the model width with GELU after the first two, a linear final output, and no
dropout or normalisation. The final output weight and bias are zeroed **after** the generic
initialisation pass, following the ordering `SeqVaeLagAttnTrfRws.__init__` already uses for its
delta heads. The mean-only arm builds a $d_z$-wide final layer and constructs no scale parameters
at all. The selector is an argument, never learned and never derived from `self.training`.
Acceptance criteria:
- Output splits into two $d_z$ vectors; under the mean-only arm the head emits one and the module
  tree contains no scale parameter.
- With the target state, masks and metadata fixed, the Jacobian of the proposal at lag $\ell$ with
  respect to the source encoding at any other lag is exactly zero.
- A zero selector at a lag returns an exactly zero proposal at that lag and leaves the others
  unchanged.
- After a generic initialisation pass over the module, the final projection is still exactly zero.
- Two models differing only in the lag embedding produce different proposals for a permuted source
  ordering, which is what the embedding exists for.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/nets/lag_updates.py` - new, embeddings and proposal head
- `teb_vae/lag_slot_transformer_cfs/tests/test_lag_updates.py` - new
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_lag_updates.py -q`.
Expected: all pass. Status: executed.
Test rationale: the zero-initialisation-survives-generic-init check is the one the repository has
already been bitten by, twice, in `_zero_init_delta_heads` and `zero_init_clock`. The locality
check is the architectural claim of the whole design and is not provable anywhere else.
Runtime: fast/local. Observed as part of the package suite below.
Evidence: 2026-09-09, at `6cfa8be` plus this change. 25 tests pass. The generic initialisation pass
was confirmed to refill the output projection, so the re-zeroing hook is a real repair rather than a
precaution, and the same pass was confirmed to leave `nn.Embedding` untouched, which is why the lag
embedding carries no hook of its own. One implementation decision the task did not anticipate: the
input projection is applied per part rather than to a materialised concatenation, which is the same
arithmetic on one `nn.Linear` and avoids building the largest tensor in the architecture. The slice
order is the documented concatenation order and a test asserts the two forms agree.

#### T003: Fusion, the prior-relative residual, and the residual divergence

Requirements: FR-004, FR-005, FR-007, FR-012, NFR-002
Depends on: T002
Description: Complete `nets/lag_updates.py` with the summation at $c_L = L^{-1/2}$, the bounds at
$a_{\max}$ and $b_{\max}$ applied **after** summation, the prior-relative full parameters, and the
residual-form divergence computed with `expm1` and reduced in FP32 or higher. $c_L$ is fixed and
never renormalised by the count of available lags. The bounded log-variance is not passed through
the prior's `smooth_bound` again. Add the cancellation ratio as a diagnostic with its numerator and
denominator returned beside it and a fixed numerical epsilon.
Acceptance criteria:
- For randomised positive prior scales and bounded residuals in FP64, the residual divergence
  agrees with the general diagonal-Gaussian formula computed from the final parameters, at absolute
  tolerance $10^{-10}$ and relative tolerance $10^{-8}$.
- At $a = b = 0$ the divergence is exactly zero and the full parameters equal the prior parameters
  bitwise.
- The mean correction satisfies $|\mu^q_d - \mu^p_d| \le a_{\max}\sigma^p_d$ and the scale ratio
  lies in $[e^{-b_{\max}}, e^{b_{\max}}]$ for arbitrary raw proposals.
- Two proposal sets differing by a zero-sum reallocation give an identical total update and an
  identical divergence, while suppressing one lag gives a different bounded mean. This is the
  design's section 5.3 counterexample, asserted rather than described.
- Masking lags does not change $c_L$.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/nets/lag_updates.py` - fusion and divergence
- `teb_vae/lag_slot_transformer_cfs/tests/test_residual_kl.py` - new
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_residual_kl.py -q`.
Expected: all pass. Status: executed.
Test rationale: the divergence is the model's central quantity and its simplified form is only
equal to the general one under this exact parameterisation; a second application of the sigmoid
bound would break the zero-update equality with no shape changing. The zero-sum reallocation case
is what stops a later reader from reintroducing a per-lag allocation.
Runtime: fast/local. Observed as part of the package suite below.
Evidence: 2026-09-09, at `6cfa8be` plus this change. 19 tests pass. The residual form agrees with
the family's own `kld_tensor` on 2000 randomised FP64 cases at the declared tolerances, evaluated on
the final parameters rather than on a re-derivation. Two measurements worth carrying: the naive
`exp(2b) - 1 - 2b` returns a **negative** divergence at a scale update of $10^{-12}$, where the true
value is $2 \times 10^{-24}$, so `expm1` is load-bearing rather than stylistic; and the implied full
log-variance range is the prior's widened by $2 b_{\max}$ on each side, which is a different range
from the sibling's independently bounded posterior and is the concrete reason the full log-variance
must not be re-bounded.

---

## Sprint 2: The model

Goal: `SeqVaeLagResidualTrfCfs` constructs and runs an anchored forward that satisfies the tensor
contract and the structural gates.

Demo: build the model from `configs/tiny.yaml` on the tiny integer-operator shard, call the forward
at the training stride and again densely, and show the returned key set, the shapes, exact
source-off equality, and that perturbing a future input changes no earlier anchor's forecast.

Definition of Done: T004 to T009 done with evidence; the sprint's test files pass; the model builds
no attention module, which the construction test asserts by walking the module tree.

Dependencies: Sprint 1.

**Met, 2026-09-09.** `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/ -q`
reports **149 passed in 11.80s**. The module-tree walk finds no lag cross-attention, no posterior
head, no attention-attribution head and no source encoder or stem, and exactly one history encoder,
which is the target's. Every parameter is reachable on an all-source-unavailable batch and under
mixed per-channel validity, so a distributed run needs no unused-parameter handling. Five of the six
structural gates of design section 11.3 now have evidence; the sixth, which concerns the objective's
masks and denominators, belongs to Sprint 3. The forbidden-timeline-term and Markdown-reference
greps still return nothing, and no tracked file outside this package has been modified.

### Tasks

#### T004: The architecture base

Requirements: FR-006, FR-008, FR-019, NFR-003
Depends on: T003
Description: Add `nets/core.py` with `LagResidualCore(nn.Module)`. It resolves the trimmed geometry,
builds the two channel gates, the two availability adapters, the target conv-Transformer encoder,
the prior head at the explicit clock width, the pointwise source encoder, the lag proposal head, the
horizon core and the shared decoder, and supplies the hooks the two mixins call: `_build_channel_gate`,
`_build_adapter`, `_default_decoder_out_channels`, `_check_persistence_target`, `kld_tensor` and
`_reparameterize_shared`. It runs the generic initialisation pass, the depthwise repair, the
proposal-head zeroing, the clock-projection zeroing and the FiLM re-zeroing in that order, and the
fresh-model calibration only when configured. It constructs no `lag_attn`, `query_proj`,
`posterior_head`, `te_analysis`, `source_encoder` or `source_kv_stem`. It refuses the attention and
independently-bounded-posterior keys by name.
Acceptance criteria:
- A module-tree walk finds no `LagCrossAttention`, no `PosteriorHead` and no source encoder or stem.
- Every constructed parameter is reachable from a forward on an all-invalid-source batch, so a
  distributed run needs no unused-parameter handling.
- Each refused key raises a `ValueError` naming the key and stating that this architecture has no
  attention path.
- After construction, the proposal head's final projection and the prior's clock projection are both
  exactly zero.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/nets/core.py` - new
- `teb_vae/lag_slot_transformer_cfs/tests/test_construct.py` - new
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_construct.py -q`.
Expected: all pass. Status: executed.
Test rationale: the module-tree walk is the only mechanical proof of the design's central structural
prohibition, and the reachability check is what makes `find_unused_parameters=False` safe. Both are
otherwise silent.
Runtime: fast/local. Observed as part of the package suite below.
Evidence: 2026-09-09. 37 tests pass. Two implementation decisions the task did not anticipate. The
metadata clock's projection is built on the core rather than inside the prior head, because the
conditioning state is read by the proposal head too and a projection reachable only from inside the
prior would have to be applied twice; the head is therefore built with no clock path of its own. And
the refused keywords are **parameters of the model signature**, not merely absent: the driver
forwards a configuration key only when the signature names it, so an omitted key would be dropped in
silence rather than refused. The reachability check needed a loss touching every output head, since
one forecast mean alone leaves the observation log-variance head without gradient and reports a
false failure.

#### T005: The model class, the constructor schema, and the resolution order

Requirements: FR-011, FR-019, NFR-005
Depends on: T004
Description: Add `nets/model.py` with `SeqVaeLagResidualTrfCfs`, composing the two mixins ahead of
`LagResidualCore`. The constructor writes out its own keyword schema in full, because
`trainer._build_model_kwargs` sweeps `inspect.signature`; it keeps every keyword name the warm-up
budget resolver emits, so `warmup_model_kwargs` works unchanged. It calls `_set_causal_inputs`,
`_set_channel_weights` and `_set_target_novelty` before the base and `_validate_causal_geometry` and
`_register_channel_weights` after it, matching the shipped cell. New keywords: the lag count, the
proposal hidden width, the lag embedding width, $c_L$'s convention, $a_{\max}$, $b_{\max}$, the
mean-only flag, the scalar-lift flag and the anchor and lag chunk sizes.
Acceptance criteria:
- The method resolution order equals a written-out list of class names; a reorder fails the test
  rather than training a different model.
- `warmup_model_kwargs` applied to this class returns the four channel tuples and the novelty
  vector, and raises the existing refusal if the class ever loses a keyword.
- Every constructor keyword is either a `model_config.VAE_model` key of the shipped config or is
  documented in the config as absent on purpose.
- The mean-only flag changes the module tree, not only the forward.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/nets/model.py` - new
- `teb_vae/lag_slot_transformer_cfs/tests/test_construct.py` - extend
Validation: same command as T004. Expected: all pass. Status: executed.
Test rationale: the linearisation assertion is established practice in this repository precisely
because the shipped diamond can silently resolve to the wrong side; the same hazard applies here
with a third base.
Runtime: fast/local. Observed as part of the package suite below.
Evidence: 2026-09-09. The resolution order is asserted as a written-out list, and a second test pins
each shared member to the class that must own it, which the list alone does not: an order can be
right while a member is defined on the wrong class. The warm-up budget resolver accepts the class.
The tiny model builds at 21,910 parameters.

#### T006: The anchored forward and the tensor contract

Requirements: FR-009, FR-010, FR-016, NFR-004
Depends on: T005
Description: Implement `forward`, `build_lag_mask` and `_prior_clock` on the model. The forward
follows the eleven steps of section 4.3, chunking the proposal evaluation over anchors and
optionally lags with the partial sums accumulated in the graph and never detached. Proposals are
returned only when the caller requests them. `build_lag_mask` produces the per-channel per-lag
validity from the resolved source warm-up and the lag floor, without reading any attention module.
`_prior_clock` returns the sinusoidal function of stored position, independent of every source
value and of every source-path parameter.
Acceptance criteria:
- The returned dict has exactly the documented key set: it contains the anchor index and validity,
  both branches' latent parameters and samples, both branches' forecast means and log-variances,
  the per-coordinate and total divergence, the masks, the bounded and unbounded update summaries,
  and the persistence input where the decoder was built with it. It contains no `attn_weights`,
  `attended_source_heads`, `source_kl_lag_map` or `kld_per_t_per_head`.
- Every tensor matches section 6.1, at the training stride and densely.
- The clock is bitwise identical for two batches differing only in their source values.
- Perturbing an input at a step after an anchor changes no forecast at that anchor or earlier.
- The last scored label index equals the design's stated endpoint and padded anchors are excluded
  from the mask and from the divergence support.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/nets/model.py` - the forward
- `teb_vae/lag_slot_transformer_cfs/tests/test_forward_contract.py` - new
- `teb_vae/lag_slot_transformer_cfs/tests/test_causality.py` - new
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_forward_contract.py teb_vae/lag_slot_transformer_cfs/tests/test_causality.py -q`.
Expected: all pass. Status: executed.
Test rationale: the absent-keys assertion is what stops the old evaluator contract from being
satisfied by a fabricated tensor, which design section 12.1 explicitly forbids. The future-input
perturbation check is the causality claim and has no other proof.
Runtime: fast/local. Observed as part of the package suite below.
Evidence: 2026-09-09. 30 tests pass. **The chunking test caught a real defect**: the proposal head
flattened its source window against the configured lag count rather than the chunk's own, so any run
with a lag chunk set raised on the first chunk. Fixed, and the chunked-versus-unchunked comparison
now passes at absolute tolerance $10^{-6}$ and relative $10^{-5}$ across four chunk grids. Two
naming decisions: the divergence keys are `kld_per_anchor` and `kld_per_anchor_dim` rather than the
family's `kld_per_t`, because they carry an anchor axis and the family's name would invite exactly
the misreading the design warns about; the latent keys keep the family's `mu_post` and
`logvar_post`, which maximises reuse of the shared readers and is documented in the forward as a key
name rather than a claim that either branch observes a label.

#### T007: Source-off equality and the absence semantics

Requirements: FR-007
Depends on: T006
Description: No new production code beyond what T006 built; this task is the invariant suite. It
asserts the three ways the source can say nothing and the one way it must not be treated as
saying nothing.
Acceptance criteria:
- With every selector zero after the proposal head has been given nonzero weights, the full
  parameters equal the prior parameters and the paired predictions are bitwise equal, in training
  mode as well as evaluation mode.
- With every source lag unavailable, the outputs are finite, no invalid gather occurs, and the
  source residual is exactly zero.
- At zero-initialised output projections, equality holds for arbitrary inputs.
- A valid standardized-zero observation carries mask one, and no assertion in the model or the
  objective forces its output to equal the prior. The test states this as a negative: there is no
  code path that centres $F(h, U) - F(h, 0)$.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/tests/test_invariants.py` - new
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_invariants.py -q`.
Expected: all pass. Status: executed.
Test rationale: source-off equality is the invariant every downstream control is read against, and
the observed-zero case is the specific confusion design section 4.6 spends a paragraph on. The
training-mode half matters because the shipped model draws two dropout masks when one module is
invoked twice.
Runtime: fast/local. Observed as part of the package suite below.
Evidence: 2026-09-09. 13 tests pass. All three absence routes reproduce the prior bitwise, in
training mode as well as evaluation mode. The observed-zero case is asserted as a pair: a stream of
exact zeros carries mask one and moves the full distribution, while the same stream through an
all-cold source leaves the update at exactly zero. That contrast is the evidence that absence and a
real zero are distinct events here, which a one-sided assertion would not give.

#### T008: Shared-noise pairing and the decoder boundary

Requirements: FR-008
Depends on: T006
Description: The invariant suite for the sampling and decoder contract.
Acceptance criteria:
- The paired sample difference equals $\sigma^p \odot [a + (e^b - 1)\odot\epsilon]$ to the measured
  FP32 tolerance.
- Both decoder calls receive the identical persistence input and the identical weights, and the
  decoder receives no target state, source value, proposal, mask, posterior parameter or encoder
  summary. Asserted by intervening on each candidate tensor and showing the forecast does not move.
- Under the mean-only arm the sample difference is deterministic given the anchor.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/tests/test_invariants.py` - extend
Validation: same command as T007. Expected: all pass. Status: executed.
Test rationale: the decoder-bypass check is one of the six structural gates design section 11.3
requires, and it is the only one that can fail through an accidental extra argument.
Runtime: fast/local. Observed as part of the package suite below.
Evidence: 2026-09-09. The decoder boundary is checked by intercepting the module rather than by
reading the forward, because what matters is what it was actually handed: exactly two invocations,
no extra positional or keyword argument, the identical persistence tensor **object** in both, and
the two latents. The paired-sample difference matches the stated formula at $10^{-5}$.

#### T009: Gradient escape from the zero start

Requirements: FR-003, FR-007
Depends on: T006
Description: A bounded check that the zero initialisation is a starting point rather than a fixed
point. On a small nondegenerate toy where the target genuinely depends on the source, one backward
pass must produce a nonzero gradient at the final source projection. Design section 4.6 warns that
hidden source-layer gradients may be zero on the first step because the output projection is zero,
so the assertion is on the final projection only, and no claim is made that every parameter has a
nonzero gradient on every batch.
Acceptance criteria:
- The gradient at the proposal head's final weight is nonzero after one backward pass on the toy.
- The divergence gradient with respect to the residual update is zero at initialisation, which is
  asserted rather than treated as a failure.
- After a handful of optimizer steps on the toy, the hidden layers also carry gradient.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/tests/test_invariants.py` - extend
Validation: same command as T007. Expected: all pass. Status: executed.
Test rationale: named in design section 13.1 as a required check, and it is the difference between
a model that starts at the prior and a model that stays there. Nothing else would detect the latter
until a full training run reported a zero gap.
Runtime: fast/local. Observed as part of the package suite below.
Evidence: 2026-09-09. Three assertions rather than one. The divergence gradient at the residual is
exactly zero at initialisation, recorded as intended rather than left to be found as a failure. The
predictive gradient does reach the final source projection at that same zero start. And on an
eight-step toy whose target depends on the source in a way the target stream cannot supply, the
hidden proposal layers and the lag embedding both carry gradient and the divergence leaves zero.

---

## Sprint 3: A run that trains

Goal: the objective, the task, the trainer and the configurations, so the package trains end to end
from the Run button on the tiny shard.

Demo: set `RUN_CONFIG` to `configs/tiny.yaml`, press Run, and watch a short fit complete with the
metric surface logged and a resolved config written into the run directory.

Definition of Done: T010 to T016 done with evidence; the sprint's test files pass; the tiny fit
completes and writes a checkpoint that reloads into an identical model.

Dependencies: Sprint 2.

**Met, 2026-09-09, with one item carried forward.**
`.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/ -q` reports **286 passed
in 23.34s**, including the two-epoch fit at 13 s. The run resolves the warm-up budget, tiles at the
configured stride, writes both checkpoint criteria and a resolved configuration, and reports this
package's own metric surface with no column naming a tensor the architecture lacks. The sixth
structural gate -- the objective's masks, denominators and divergence support -- now has evidence,
so all six are covered.

**Carried forward:** the production-geometry memory measurement and the reassociation tolerance at
production widths, both of which need the real geometry and a device. They move to Sprint 5 with the
first real arm rather than being claimed here.

### Tasks

#### T010: The objective and its global reduction

Requirements: FR-012, FR-013, FR-014, FR-015, NFR-003
Depends on: T006
Description: Add `nets/objective.py` and route the model to it. It gathers the forecast target
through the inherited `_build_forecast_target` at the anchor index the forward returned, builds the
forecast mask and the divergence support from `teb_vae/lag_attn_rws/nets/raw_masks.py`, calls
`masked_raw_block_per_anchor` for both branches with the resolved channel and horizon weights,
computes the residual divergence and the prior rate per anchor, and reduces **once** by the global
contributing-anchor count. The local numerator is scaled
so that distributed gradient averaging over the world size reproduces the global mean. Global
numerators and denominators are logged. On an empty global support it returns a graph-connected zero
and records no scored data. It reports the coefficient count and mask coverage beside nats per
anchor, and never rescales partial coverage to a complete block. It reuses the existing metric names
where the quantity is the same, and omits the attention-derived ones entirely.

**The routing is load-bearing and is the easiest thing in this plan to get silently wrong.** With
the resolution order of section 4.2, an un-overridden `compute_loss` resolves through
`CausalFeatureForecastTarget.compute_loss` to `FeatureForecastTarget.compute_loss`, which calls the
shared objective directly and never consults this module. The model therefore defines its own
`compute_loss` that applies the inherited `scored_weight` and then calls this module, so neither
mixin's `compute_loss` is reached. Under the stored clock `scored_weight` is the identity object, so
skipping it would be inert today and wrong the moment a forecast clock is configured, which is
exactly the kind of failure that surfaces as a number rather than an exception.
Acceptance criteria:
- Channel weights sum to the kept channel count and horizon weights sum to the horizon, both
  resolved from the model rather than from literals.
- With one rank, the objective equals a hand-computed reference on a small deterministic batch.
- With two simulated ranks holding uneven scored-anchor counts, the objective and its gradient equal
  the single-process combined-batch reference to the measured tolerance. This is the check that
  distinguishes the global mean from the average of per-rank means.
- On a batch with no scored anchor the loss is a finite zero that is connected to the graph, and the
  recorded scored-anchor count is zero rather than one.
- The reconstruction denominator and the divergence support are the same anchor set by construction.
- Neither mixin's `compute_loss` is reached: a test patches the shared objective to raise and shows
  that a training step still succeeds.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/nets/objective.py` - new
- `teb_vae/lag_slot_transformer_cfs/nets/model.py` - the `compute_loss` override
- `teb_vae/lag_slot_transformer_cfs/tests/test_objective.py` - new
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_objective.py -q`.
Expected: all pass. Status: proposed.
Test rationale: the global-versus-per-rank distinction is the whole reason this module exists and is
invisible on one rank, which is where every other test runs. The empty-support case is a
divide-by-zero the shipped code hides with a clamp. The routing check exists because an
un-overridden `compute_loss` trains a working model against the wrong reduction with nothing
raising anywhere.
Runtime: fast/local. Observed as part of the package suite below.
Evidence: 2026-09-09. 13 tests pass. The uneven-rank check is the one that needed real
construction: two shares of one and two samples, a simulated collective, and the latent draw
pinned so slicing the batch does not also change the noise. The average of the two shares'
losses and gradients equals the whole-batch pass; a per-rank mean would over-weight the
smaller share by three halves. The empty-support path gained a fix while being tested: its
graph-connected zero originally reached only the reconstruction, leaving the observation
log-variance head out of the graph on a batch that scored nothing, which is precisely the
distributed hazard the path exists to avoid.

#### T011: The task

Requirements: FR-016, FR-017
Depends on: T010
Description: Add `task.py`. It reuses the anchor phase derivation, the stage-to-geometry resolution
and the forward-input assembly from `SeqVaeLagAttnCfsTask`, and the step-granular learning-rate ramp
from `SeqVaeLagAttnTrfRwsTask`, and replaces the loss call with this package's objective and the
metric surface with this package's key set. The divergence ramp, the checkpoint contract and the
spike-breaker wiring stay the shared ones. The diagnostic page rows that read attention weights are
removed rather than fed a substitute.
Acceptance criteria:
- The method resolution order is asserted as a written-out list of class names.
- The training stage decodes at the configured stride with a per-sample phase; validation and test
  decode densely at stride one and phase zero.
- The logged metric key set contains no attention-derived name and the loss-spike breaker still
  watches the unprefixed main loss key.
- One training step and one validation step run on the tiny batch fixture without a trainer.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/task.py` - new
- `teb_vae/lag_slot_transformer_cfs/tests/test_task.py` - new
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_task.py -q`.
Expected: all pass. Status: proposed.
Test rationale: the shipped task's own test file asserts its linearisation for a documented reason,
and the spike-breaker key is silently dropped if a subclass renames it.
Runtime: fast/local. Observed as part of the package suite below.
Evidence: 2026-09-09. 18 tests pass. The step is written out rather than inherited, because
the shared one reads a saturation key this architecture does not emit and pairs a dense
support with an anchor-indexed latent. Two inherited controls are refused explicitly with the
reason recorded: the permutation control rebuilds the full branch through the lag attention
and the head-structured posterior, and the source-null readout encodes a zeroed stream
through a pathway that holds no parameters. Both would have raised on the first validation
step of the first real run.

#### T012: The configurations

Requirements: FR-015, FR-017, FR-019, NFR-005
Depends on: T005
Description: Add `configs/default.yaml` and `configs/tiny.yaml`. The production config carries the
fully resolved starting settings of design sections 3.1, 4, 6.1 and 6.4: the geometry, the latent and
model widths, the encoder schedule, the lag count, the bounds, the channel and horizon weight ratios,
the divergence ramp and prior-rate weight, the optimizer and its warm-up and milestone schedule, the
epoch cap, the validation cadence, the checkpoint retention and early-stopping rule, the inherited
gradient clip and spike margin marked as unvalidated starting guards, and the model kind. The tiny
config inherits from it and points at the committed integer-operator tiny shard, as the shipped
`tiny.yaml` does. Every key that this architecture does not accept is absent, and the comment says
why rather than leaving it to be inferred.
Acceptance criteria:
- Both configs load and construct the model through the trainer's kwargs sweep.
- Every `model_config.VAE_model` key is a constructor parameter and every constructor parameter
  without a default is a config key.
- The resolved horizon, anchor stride, latent width, lag count and channel counts match design
  section 3.1.
- The config states that the clip and the spike margin are inherited guards to be re-measured, not
  validated thresholds for this head.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/configs/default.yaml` - new
- `teb_vae/lag_slot_transformer_cfs/configs/tiny.yaml` - new
- `teb_vae/lag_slot_transformer_cfs/tests/test_config_load.py` - new
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_config_load.py -q`.
Expected: all pass. Status: proposed.
Test rationale: the key-parity check catches the failure the shipped packages have a test for
already: a config key that names no constructor parameter is silently ignored and the run trains a
different model than the file describes.
Runtime: fast/local. Observed as part of the package suite below.
Evidence: 2026-09-09. 13 tests pass. The configuration is standalone with no inheritance
chain to a sibling, because basing it on one would inherit exactly the keys this architecture
refuses. Key parity is checked in both directions, and a third test asserts that no
configuration sets a refused keyword -- easy to violate by copying a block from the sibling,
and a failure that would otherwise arrive only at a real launch. The diagnostic page block is
deliberately absent and that absence is asserted.

#### T013: Documentation gates

Requirements: NFR-004, NFR-005
Depends on: T012
Description: Add the package documentation test, modelled on
`teb_vae/lag_attn_transformer_cfs/tests/test_docs.py`. Two gates specific to this work: no source
file, config comment, docstring or report string in the package contains a mechanical-shift, sensor
delay, acquisition shift, `up_shift_secs` or `tau_pre` term; and no comment or docstring writes the
horizon, anchor count, block width or channel counts as a literal where a symbolic reference or a
value read from the model is available.
Acceptance criteria:
- The forbidden-term grep returns nothing across the package.
- The geometry-literal gate passes, with an explicit allowlist for the configuration files, which
  are where the numbers legitimately live.
- Every public module, class and function carries a Google-style docstring.
- No module under `nets/` imports Lightning or any training framework, which is the convention
  `teb_vae/lag_attn_transformer_cfs/tests/test_nets_are_framework_free.py` already enforces for the
  shipped cell and which Sprint 1's Definition of Done relies on.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/tests/test_docs.py` - new
- `teb_vae/lag_slot_transformer_cfs/tests/test_nets_are_framework_free.py` - new
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_docs.py teb_vae/lag_slot_transformer_cfs/tests/test_nets_are_framework_free.py -q`.
Expected: all pass. Status: proposed.
Test rationale: both rules are standing repository rules whose violations are invisible at review
time and permanent once written. The shipped packages already carry a documentation test, so this is
extending an established convention rather than introducing one.
Runtime: fast/local. Observed as part of the package suite below.
Evidence: 2026-09-09. 33 tests pass across the two files. The gate scans the whole package
except itself, which it must: the forbidden terms and the document-reference pattern are
written out in it as constants, so a gate that scanned itself could never pass. The
geometry-literal rule allows the two configuration files by name, since they are where the
geometry is declared.

#### T014: Chunking, and the memory measurement

Requirements: FR-010, NFR-001, NFR-002
Depends on: T006, T010
Description: Resolve the anchor and lag chunk sizes as configuration leaves, verify that chunked and
unchunked execution agree, and **measure** peak memory at the production tiling and at the dense
evaluation tiling. Record the measured numbers in this document's evidence field and in the resolved
config, not in a code comment.
Acceptance criteria:
- Chunked and unchunked forward and backward agree within a tolerance measured here and written into
  the test rather than assumed.
- No chunk is detached during training; the gradient of a parameter reached only through the last
  chunk is nonzero.
- Peak allocated memory is recorded for one training step at the configured batch and stride, and
  for one dense evaluation step, with the chunk sizes that produced them.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/nets/model.py` - chunk resolution
- `teb_vae/lag_slot_transformer_cfs/configs/default.yaml` - chunk leaves
- `teb_vae/lag_slot_transformer_cfs/tests/test_chunking.py` - new
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_chunking.py -q`
for the agreement checks. The memory measurement runs on the target device and is recorded manually;
on a machine with no CUDA device the test records that it was skipped and why. Status: proposed.
Test rationale: design section 12.3 states that changing chunk sizes changes floating-point summation
order, so the tolerance is a measurement rather than a convention. A detached chunk would train a
model whose late lags never update, which no other check would catch.
Runtime: fast/local for the agreement checks. The production-geometry memory measurement is
integration-class and is **not** done: see the evidence below.
Evidence: 2026-09-09. 9 tests pass. Chunked and unchunked agree across four chunk grids, and
the tolerance is now pinned to a measurement rather than a guess: the largest observed
absolute difference is $9.5 \times 10^{-7}$, on the decoder output, and the declared tolerance
is ten times that. The first draft asserted the error was well inside a $10^{-6}$ bound and
failed, which is how the real figure was found.

**Two things this task still owes.** Peak memory at the production tiling is unmeasured: it
needs the real geometry and a device, and the smoke run reports only $0.03$ GiB allocated at
the tiny geometry. And the reassociation tolerance is the tiny geometry's; it must be
re-measured at the production widths before a run relies on it. Both are carried into
Sprint 5, where the first real arm supplies the geometry and the device.

#### T015: The trainer, the Run-button convention, and warm-start transfer

Requirements: FR-017, FR-018, FR-019
Depends on: T011, T012
Description: Add `trainer.py` following the repository's runner convention: a repository-root guard
before any absolute import, a module-level `RUN_CONFIG` immediately above the `__main__` guard, no
`required=True`, no non-`None` argparse default, and a working-directory change to the repository
root in the command-line path. Add the warm-start transfer: loading a target-only checkpoint through
`train/graph_models_utils.py`, validating feature order and task geometry, transferring the target
encoder, prior and decoder tensors, reinitialising the source output projections unless resuming this
exact model kind, and logging every transferred, missing and reinitialised tensor name. Joint training
after pretraining starts a declared new optimizer, scheduler and divergence ramp; an exact resume
restores all of them.
Acceptance criteria:
- The module runs from the Run button with only `RUN_CONFIG` edited, and from the command line with
  `--config`.
- The transfer logs three explicit lists and refuses a checkpoint whose kept-channel order or task
  geometry disagrees.
- A transferred model's source output projections are exactly zero, so the first joint step starts at
  the prior.
- Resuming this model kind restores optimizer, scheduler, epoch and random-number state; warm-starting
  a different kind does not, and says so in the log.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/trainer.py` - new
- `teb_vae/lag_slot_transformer_cfs/tests/test_trainer.py` - new
- `teb_vae/lag_slot_transformer_cfs/tests/test_checkpoint_contract.py` - new
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_trainer.py teb_vae/lag_slot_transformer_cfs/tests/test_checkpoint_contract.py -q`.
Expected: all pass. Status: proposed.
Test rationale: `CLAUDE.md` mandates the Run-button convention and points at the test that enforces
it. The transfer audit is the difference between a warm start and a silently half-initialised model,
which design section 12.2 spends its longest paragraph on.
Runtime: fast/local. Observed as part of the package suite below.
Evidence: 2026-09-09. 32 tests pass across the two files. The warm start is a separate
configuration key from the family's strict load, and the two are refused together: one
restores the source pathway and the other deliberately leaves it at zero, so a run doing both
has a starting point neither key describes. **The smoke run found a real defect here**: a
checkpoint written by the task carries the net's keys under `_orig_model.`, not `model.`, so
the original stripper matched nothing and the transfer would have refused every real
checkpoint with a message pointing at the file rather than at the prefix. Fixed, and the
stripping rule is now one helper the smoke test uses too.

#### T016: Distributed reachability and the tiny fit

Requirements: FR-013, FR-017, NFR-002, NFR-003
Depends on: T014, T015
Description: The sprint's integrated acceptance. Assert that every parameter is in the graph on a
rank whose source is entirely unavailable and on a rank with mixed validity, without enabling
unused-parameter handling. Then run a short fit on the tiny config to completion and reload the
checkpoint. Measure the pre-clip gradient-norm distribution and the skipped-batch rate on this
training-only pilot, and record them, because design section 6.4 states the inherited clip of $3500$
and additive spike margin of $2200$ are starting guards rather than validated thresholds for this
head.
Acceptance criteria:
- The all-invalid and mixed-validity reachability checks pass with `find_unused_parameters=False`.
- The tiny fit completes, writes a checkpoint and a resolved config, and the reloaded model produces
  identical inference and identical source-off invariants.
- The pre-clip gradient-norm distribution and the skipped-batch rate are recorded. The spike detector
  is configured knowing that the Gaussian negative log likelihood is not bounded below by zero.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/tests/test_ddp_reachability.py` - new
- `teb_vae/lag_slot_transformer_cfs/tests/test_train_smoke.py` - new
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_ddp_reachability.py teb_vae/lag_slot_transformer_cfs/tests/test_train_smoke.py -q`.
Expected: all pass. Status: proposed. If the smoke fit exceeds the tool timeout, hand the command to
the operator with a `!` prefix rather than backgrounding it; detached long runs are killed on this
machine.
Test rationale: the reachability checks are the only proof that `find_unused_parameters=False` is
safe on this module tree, and a rank whose source is entirely unavailable is exactly the case the
selector-and-mask multiplication exists for. The smoke fit is the sprint's integrated acceptance and
is the only place the objective, the task, the trainer and the configs run together.
Runtime: integration. **Observed at 13 s** for the two-epoch fit on the tiny configuration,
well under the estimate; the whole package suite is 23 s.
Evidence: 2026-09-09. 19 tests pass across the two files. The fit completes through the real
entry point, resolves the warm-up budget to 76 target and 46 source channels, writes both
checkpoint criteria under this package's own stem, and writes a resolved configuration beside
them that records the chunk sizes. Two defects surfaced only here. The tracked metric surface
omitted the gradient-norm, clip-fraction and spike columns, so the numbers the inherited clip
has to be re-derived from were logged and then dropped; they are tracked now. And the tiny
configuration used a squared error, under which the observation log-variance head is outside
the graph on every batch -- harmless on one device and a distributed failure waiting to
happen -- so it now uses the Gaussian score the design specifies.

The pre-clip gradient distribution is recorded in the run's metric history and the clip does
not bind on every step. The distribution at the **production** geometry remains unmeasured,
as T014 records.

---

## Sprint 4: A run that is scored

Goal: a matched predictive gap, the proposal readouts, and the three source controls, produced by an
entry point that follows the repository's launch convention.

Demo: run the evaluation entry point against the tiny fit's checkpoint and read `summary.json`: the
matched Monte Carlo gap with its recording-bootstrap interval, the suppression result for each
declared band with its usable counts, the cancellation ratio with its numerator and denominator, and
the three control margins.

Definition of Done: T017 to T022 done with evidence; the sprint's test files pass; the two shipped
CFS cells' evaluation binding tests pass unchanged, which is NFR-006's measurement.

Dependencies: Sprint 3.

**Met, 2026-09-09, with one planned edit deliberately not made.**
`.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/ -q` reports **386 passed
in 24s**, including a fit-then-score-then-verify run at 14 s. The evaluation entry point produces a
`summary.json` carrying the matched Monte Carlo gap with a recording-bootstrap interval, each band's
suppression margin with its usable anchor and channel counts, the cancellation ratio with both of
its parts for each proposal channel, per-lag and per-source-channel exposure, the three source
control margins with their pairing counts, mixture calibration for both branches, and the run's own
provenance. The acceptance gate passes on what that run wrote.

**The planned guard inside the shared collection pass was not made, and the reason is a finding
rather than a deferral.** Section 3.2 recorded that `evaluate_batch` reads three attention keys
unconditionally and concluded that guarding them would let this model through. It would not. The
readout that function returns declares **eight** attention-derived fields as required, and every
latent readout in it pairs a dense `(B, T)` support with a latent produced at every stored step --
`kl_mask` returns dense `(B, T)` explicitly *because* the tensors it gates are `(B, T, d_z)`. This
architecture's latents are `(B, A, d_z)`, indexed by decoded anchor. No guard reconciles those two
axes, and satisfying the readout would mean fabricating the eight fields, which design section 12.1
forbids by name. So the package scores through its own pass and reuses every piece that says nothing
about an architecture. The registry half of T017 -- the one FR-024 actually needs and the one T020
consumes -- is done and tested.

### Tasks

#### T017: The two shared evaluation seams

Requirements: FR-024, NFR-006
Depends on: T006
Description: Two additive edits with inert defaults. First, `ModelBinding` gains an optional
`excluded_analyses` tuple defaulting to empty, and `merged_analysis_functions` removes those names
after merging, refusing a name that is not registered. Second, the lag block inside `evaluate_batch`
is guarded on the presence of the attention keys, so a model that does not emit them produces the
per-sample and per-anchor tables without those columns instead of raising, and the collector's
analysis-to-key map tolerates their absence. Neither edit changes behaviour for a model that emits
the keys.
Acceptance criteria:
- The two shipped bindings produce the same registry, in the same order, as before the edit.
- A binding excluding an unregistered name raises, naming it.
- A forward dict without the attention keys produces a readout whose remaining columns are unchanged
  from the guarded path.
- `teb_vae/lag_attn_cfs/tests/test_eval_binding.py` and
  `teb_vae/lag_attn_transformer_cfs/tests/test_eval_binding.py` pass unchanged.
Files affected:
- `teb_vae/lag_attn_cfs/eval/binding.py` - one optional field
- `teb_vae/lag_attn_cfs/eval/run.py` - exclusion in `merged_analysis_functions`
- `teb_vae/lag_attn_cfs/eval/metrics.py` - guard the lag block in `evaluate_batch`
- `teb_vae/lag_attn_cfs/eval/collect.py` - tolerate absent attention keys
- `teb_vae/lag_attn_cfs/tests/test_eval_binding.py` - extend with the exclusion cases
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_cfs/tests/test_eval_binding.py teb_vae/lag_attn_transformer_cfs/tests/test_eval_binding.py -q`.
Expected: all pass. Status: executed.
Test rationale: this is the only task in the plan that touches code two shipped models run, so the
regression evidence is the existing test files passing unchanged rather than a new assertion.
Runtime: fast/local. Observed at 8 s.
Evidence: 2026-09-09. 46 tests pass across the two files. **Two of the four planned edits were
made and two were not**, and the split is the task's real finding.

Made: `ModelBinding` gained `excluded_analyses`, defaulting to empty; `merged_analysis_functions`
removes those names **after** the merge, so an exclusion naming an analysis the registry does not
hold refuses instead of doing nothing. That refusal is the one that matters -- a misspelt exclusion
leaves the analysis running while the binding says it was removed.

Not made: the guard inside `evaluate_batch` and the collector's tolerance for absent attention keys.
Section 3.2 identified the three unconditional key reads and concluded a guard would let this model
through the collection pass. It would not: `BatchReadout` declares eight attention-derived fields as
**required**, and the pass indexes a dense `(B, T)` support against a latent produced at every
stored step, while this architecture's latents are anchor-indexed. Guarding three key reads
reconciles neither, and satisfying the readout would mean fabricating the eight fields. A guard no
caller can use, added to a module two shipped forecasters run, is a regression risk for no benefit,
so it was left out and this package scores through its own pass instead.

One planned assertion changed rather than passing untouched: the causal cell's field pin enumerates
`ModelBinding`'s parameters, so a new optional field had to be added to that list. The **behaviour**
of both shipped bindings is unchanged, which is what NFR-006 measures, and the three new exclusion
cases were added beside the collision case they mirror.

#### T018: The interventions

Requirements: FR-021, FR-023
Depends on: T006
Description: Add `nets/controls.py` with four paired interventions, each recomputing the fusion
downstream of a cached deterministic proposal set rather than re-running the whole forward, and each
scored under the same latent noise as its matched arm. Proposal suppression sets the selector to zero
over a lag band and holds the target state, metadata, original masks, remaining proposals and $c_L$
fixed. Source-value replacement substitutes observed zeros, observed constants, or mask-only values,
with the selectors **enabled**, which is what makes it a different question from suppression.
Cross-recording permutation pairs a recording with a different one of compatible acquisition and
availability stratum, preserving within-source time order and verifying zero same-recording pairings.
The selectors-off arm verifies the equality invariant and nothing else, and the module's docstring
says so.
Acceptance criteria:
- Suppression over an empty band reproduces the matched forward bitwise.
- Suppression over all bands reproduces the prior branch.
- The replacement arms leave the availability announcement and the clock untouched, which is asserted
  rather than assumed.
- The permutation verifies zero same-recording pairings and preserves within-source ordering.
- Every arm reuses one latent draw with the matched arm, so a difference is a difference of
  predictions rather than of noise.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/nets/controls.py` - new
- `teb_vae/lag_slot_transformer_cfs/tests/test_controls.py` - new
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_controls.py -q`.
Expected: all pass. Status: executed.
Test rationale: the empty-band and all-band identities are what make every reported margin a
measurement rather than an artifact of a second code path, and they are cheap. The shared-draw
requirement has bitten the shipped occlusion readout already.
Runtime: fast/local. Observed at 6 s.
Evidence: 2026-09-09. 19 tests pass. Four decisions the task did not anticipate.

**The arms return latent parameters, not scores.** The shared-draw requirement is then structural
rather than a convention a caller must honour: every arm is handed to one draw loop together, so two
arms with identical parameters produce bitwise identical scores and a zero margin is a zero.

**Only suppression recomputes from cached proposals.** The task asked all four to. Three of them
change what the head *reads*, and a proposal is a function of the source values it was given, so no
cached set can answer a question about different ones. Those three re-run the forward under a
substituted stream with every other argument matched, and the prior coming back bitwise unchanged is
what the test asserts instead.

**The all-band case is written as a sum over no lags rather than reached by subtraction.** The
subtractive form is what makes an empty band bitwise identical to the matched arm; using it for the
mirror case would leave a residue in the last places and a "source removed" arm whose divergence is
a small positive number rather than zero. Both endpoints are now exact and both are tested.

**`mask_only` and `zeros` are one intervention on the recommended encoder**, by construction rather
than coincidence: the value coordinate is the coefficient, so a zeroed stream leaves exactly the
mask. The identity is asserted, and the arm **refuses** under the scalar lift, where the lift of a
zero is a learned constant -- rather than silently reporting the zeros arm under a second name.

#### T019: Lag readouts

Requirements: FR-021, FR-022
Depends on: T018
Description: Add `eval/lag_metrics.py`. It computes the band suppression margin from the matched and
suppressed marginalised block scores, the cancellation ratio with its numerator and denominator for
both proposal channels, and the per-lag and per-channel exposure counting available channels and
scored anchors. Unsupported bins are recorded as missing, never as measured zero. The margin is not
normalised to sum to the total gain or to the divergence. The artifact text and the module docstring
carry the design's section 5.3 qualification: locality identifies which stored source time a head
can read, suppression measures dependence on the fitted parameterisation, and neither establishes a
unique functional decomposition or a physiological delay.
Acceptance criteria:
- The four declared bands are read from the configuration, and whole-band and joint removals are
  scored before any single-lag result is reported.
- Every band's usable recording and anchor counts appear beside its margin.
- A near-zero cancellation ratio is reported beside its denominator, so a ratio that is small because
  every proposal is near zero is distinguishable from one that is small because they cancel.
- The qualification text is present in the written artifact, asserted by the test.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/eval/__init__.py` - new
- `teb_vae/lag_slot_transformer_cfs/eval/lag_metrics.py` - new
- `teb_vae/lag_slot_transformer_cfs/tests/test_eval_lag_metrics.py` - new
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_eval_lag_metrics.py -q`.
Expected: all pass. Status: executed.
Test rationale: the diagnosis this design responds to reports that every occlusion band's interval
spans zero and that the readout was underpowered for no reason. Recording usable counts beside every
margin is what makes an uninformative band distinguishable from an absent effect, and asserting the
qualification text is what stops the caveat being dropped in a later edit.
Runtime: fast/local. Observed at 7 s.
Evidence: 2026-09-09. 17 tests pass. Three implementation decisions worth carrying.

**Every readout accumulates as sums and is finished once.** A first draft reduced per batch and the
pass would then have reported a mean of per-batch means, weighting a batch holding one segment
equally with a full one. The margins themselves are formed at the **recording** level, from the
bootstrapped arms, rather than per batch -- which is what makes them comparable with the interval
printed beside them.

**The bands come from `eval_config.occlusion_bands`.** The schema's key set is shared and closed, so
this cell reads the lag-attentive cells' band key and reports its own readout under its own name.
The partition is deliberately the same one -- a reader comparing two arms compares the same four
intervals -- and the interventions are different, which the delta and the contract both state.

**Missing is `null`, not NaN and not zero.** A NaN does not survive `json.dump` at all, and a zero is
the exact misreading the readout exists to prevent.

#### T020: The binding and the predictive scoring path

Requirements: FR-020, FR-024, FR-025
Depends on: T017, T019
Description: Add `eval/binding.py` and `eval/configs/eval_overrides.yaml`. The binding names the model
and task classes, the output tag, the geometry keys reconciled against a checkpoint, this
architecture's causality disclosure, the excluded analyses, and this package's extra analyses. The
geometry keys are this architecture's: the attention keys are gone and the lag-residual keys take
their place. The disclosure reports that the additional neural source receptive field is one stored
sample, beside the furthest searched lag, and that causal feature extraction itself still mixes raw
history. The predictive scoring reuses `mc_predictive_block` and `marginalise_block_scores`, which
already implement the log-mean-likelihood with common random numbers and the anchor gather, and adds
the concentration diagnostic and the mixture-based calibration.
Acceptance criteria:
- Every geometry key is both a constructor parameter and a shipped config key, which is the rule the
  shipped bindings' tests already enforce.
- The excluded set is recorded in `summary.json` with the reason.
- The predictive gap uses log-mean-likelihood, common masks and noise, and includes latent
  uncertainty; a mean of conditional standard deviations appears nowhere in the calibration path.
- Draw counts $8$, $32$ and $128$ are selectable and the concentration diagnostic is reported.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/eval/binding.py` - new
- `teb_vae/lag_slot_transformer_cfs/eval/configs/eval_overrides.yaml` - new
- `teb_vae/lag_slot_transformer_cfs/tests/test_eval_binding.py` - new
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_eval_binding.py -q`.
Expected: all pass. Status: executed.
Test rationale: a geometry key that is not both a constructor parameter and a config key is silently
skipped by the reconciler, so a checkpoint and a config can disagree about a setting that changes what
every number means. The shipped bindings document this and test it; the same hazard applies here.
Runtime: fast/local. Observed at 8 s.
Evidence: 2026-09-09. 20 tests pass. Three findings.

**The exclusions split in two, and the split is load-bearing.** Only `attention` and `lag_kl` are in
the shared registry, so only those two can be *removed*; the other five are the lag-attentive cell's
own extras and are absent here because nothing registers them. Declaring all seven on the binding
field refused at once, which is the guard working: an exclusion naming an analysis the registry does
not hold is rejected. All seven still reach the summary with their reasons, beside a record of which
mechanism left each one out.

**The predictive scoring lives in its own module rather than in the binding.** `eval/predictive.py`
is one file beyond this document's planned list, and folding a scoring loop, a concentration
diagnostic and a mixture calibration census into a declaration file would have been worse. It
reuses `marginalise_block_scores` and `masked_raw_block_per_anchor` and deliberately does **not**
reuse `mc_predictive_block`, whose anchor gather is a dense-latent assumption this architecture does
not have; passing it an identity index would work and would leave a load-bearing line reading as
though a gather had happened. The refusal on a stored-clock latent is tested.

**Twenty geometry keys**: the lag-attentive cell's fourteen that survive, plus `coverage_floor`,
which decides the scored population, plus the five that decide what a number *means* here -- the two
residual bounds, the summation scale, and the two arm flags that change the module tree. The
capacity and chunk keys are asserted **absent**: reconciling a chunk size would refuse a correct
run, since an evaluation legitimately tiles differently from the fit it scores.

#### T021: The evaluation entry points

Requirements: FR-017, FR-020
Depends on: T020
Description: Add `eval/run.py` and `eval/verify.py` following the launch convention: a repository-root
guard, a module-level `RUN_ARGS` dictionary keyed by argparse destination immediately above the
`__main__` guard, no `required=True`, no non-`None` argparse default, the merge through
`resolve_launch_args`, the argument sources logged and recorded in the run artifacts, a
working-directory change to the repository root, and `main` returning the exit code. Register both in
the package's launch test tuple.
Acceptance criteria:
- Both modules run from the Run button with only `RUN_ARGS` edited.
- Every `RUN_ARGS` key is a valid argparse destination and a key that is not raises at startup.
- No argument declares `required=True` and no argument has a non-`None` default.
- The launch test's entry-point tuple names exactly the modules in `eval/` carrying a `__main__`
  block, in both directions.
- The argument sources reach the run's artifacts, so a run's provenance is recoverable without a
  shell history.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/eval/run.py` - new
- `teb_vae/lag_slot_transformer_cfs/eval/verify.py` - new
- `teb_vae/lag_slot_transformer_cfs/tests/test_eval_launch.py` - new
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_eval_launch.py -q`.
Expected: all pass. Status: executed.
Test rationale: `CLAUDE.md` mandates the convention and names the enforcing test, which is written
out rather than discovered precisely so a runner that forgot the convention fails instead of going
unseen.
Runtime: fast/local. Observed at 6 s.
Evidence: 2026-09-09. 17 tests pass. The parser of each entry point was factored into a
`build_parser()` of its own, which is the sibling suites' convention and is what lets the four rules
be checked mechanically rather than by reading the file; both `prog` strings name this package, so a
usage line printed at a refusal points at the module the operator actually launched.

`verify.py` is stdlib only and stays that way deliberately: it reads a finished summary, and a gate
that costs a numeric stack to answer "did this run measure what it claims" is one nobody runs
against a summary copied off the box that produced it.

#### T022: Evaluation smoke, and the sprint's integrated acceptance

Requirements: FR-020, FR-021, FR-022, FR-023, FR-024, FR-025
Depends on: T021, T016
Description: Run the evaluation entry point end to end against the tiny fit's checkpoint and assert
the shape of what it produced. Confirm the six structural gates of design section 11.3 as a single
recorded checklist, drawing on the tests from sprints 1 to 3 rather than restating them, and record
which gate each piece of evidence comes from.
Acceptance criteria:
- The run writes a `summary.json` containing the matched gap with a recording-bootstrap interval, the
  per-band suppression margins with usable counts, the cancellation ratio with its two components,
  the per-lag and per-channel exposure, and the three control margins.
- The excluded analyses are named in the summary and no attention-shaped column appears anywhere in
  the output tables.
- Equal-recording and anchor-weighted summaries are reported separately.
- The six structural gates are recorded as passing with a pointer to the test that proves each.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/tests/test_eval_smoke.py` - new
- `teb_vae/lag_slot_transformer_cfs/eval/EVAL.md` - new, the contract and the gate checklist
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_eval_smoke.py -q`.
Expected: all pass. Status: executed.
Runtime: integration. **Observed at 13.9 s** for the fit, the pass and the gate together, far under
the estimate; the whole package suite is 24 s.
Test rationale: the gates are only meaningful as an integrated statement; individually they are
already covered, and this is the one place the full pipeline is exercised against a real checkpoint.
Evidence: 2026-09-09. 14 tests pass. The fit, the evaluation and the acceptance gate run end to end
through the real entry points, against the committed integer-operator fixture shard.

**The two reference identities hold on real weights**: the empty-band arm reproduces the matched
forward and the all-band and silence arms reproduce the target-only prior, each inside the gate's
floating-point allowance. Everything else in the summary is a margin between arms that were pinned
that way.

The evaluation reads a run-specific override delta rather than the committed one, because the
committed delta names the production shards and the production lag bands and the fixture has
neither. Writing one is what an operator does for any other split, so the merge is exercised rather
than bypassed.

The six structural gates are recorded as a checklist in the evaluation contract beside the entry
points, each row naming the test that proves it; a scripted check confirmed that all of the named
tests exist. Two further properties are gated on the run's own output by the acceptance gate and can
genuinely fail: the reference arms being exact, and no key anywhere in the summary naming an
attention distribution or a per-lag divergence allocation.

---

## Sprint 5: The first real arm

Goal: everything the three training runs need, measured and wired, and then the three runs.

Refined 2026-09-09, against its own precondition: the tiny fit and the evaluation smoke both pass,
and T023 below has now measured the peak memory that settles the production batch size.

Demo: from the repository root, fit the target-only arm, fit it again at a second seed, warm-start
the candidate from the first, score the candidate and the reference through one entry point, and
read the two together with the acceptance gate.

Definition of Done: T023 to T027 done with evidence. **T023 to T026 are done; T027 is not**, and
cannot be done from this machine: it needs the integer-operator production shards and the seven-GPU
box the configuration describes, and neither is here. Section 5.4 records the boundary.

Dependencies: Sprint 4, and — for T027 alone — access to the production shards.

### Tasks

#### T023: Peak memory and the reassociation tolerance, at the production geometry

Requirements: NFR-001, NFR-002
Depends on: T014, T016
Description: Add `eval/memory.py`, a runnable pass that measures peak allocated memory for a
training step, a dense evaluation step and a dense evaluation step retaining the proposals, over a
grid of chunk settings and batch sizes, and measures the chunked-versus-unchunked agreement at the
production widths. It reads **no data**: every input is a seeded tensor of the configured shape,
because both quantities are properties of the geometry. Record the numbers in this document and in
the production configuration's own comments, not in a code comment.
Acceptance criteria:
- Peak allocated memory is recorded for each step shape, chunk setting and batch size, with the
  setting that produced it; an out-of-memory result is recorded as a measurement and the sweep
  continues.
- The reassociation difference is measured at the production widths, with a repeat-run floor
  reported beside it so the device's own nondeterminism is not attributed to the chunk size.
- The production configuration's guidance on which setting to reach for is corrected to whatever
  the measurement says, rather than left as the expectation it was written from.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/eval/memory.py` - new
- `teb_vae/lag_slot_transformer_cfs/configs/default.yaml` - the corrected guidance
- `teb_vae/lag_slot_transformer_cfs/tests/test_eval_launch.py` - the new runner joins the tuple
Validation: `python -m teb_vae.lag_slot_transformer_cfs.eval.memory` on the target device,
executed on one 12 GiB device.
Test rationale: the launch convention is enforced by the existing tuple, which the new runner joins;
the measurement itself is a measurement rather than an assertion, and its output is the evidence.
Runtime: about 40 s for the full sweep.

Evidence: 2026-09-09, NVIDIA RTX 4080 Laptop, 12 GiB, ungated at the declared channel widths — an
upper bound on the gated model, since the gate only removes channels and the arrays that dominate
the peak are indexed by anchor, lag and latent width. $3{,}934{,}196$ parameters, $156$ dense
anchors, $91$ candidate lags.

| Step | batch | unchunked | at a quarter of each axis |
| --- | --- | --- | --- |
| training, backward and one optimizer step | $128$ | $8145$ MiB | $8147$ MiB |
| dense evaluation | $128$ | $4447$ MiB | $2309$ MiB |
| dense evaluation, proposals retained | $128$ | $4447$ MiB | $3277$ MiB |
| training | $64$ | $4094$ MiB | $4091$ MiB |
| training | $32$ | $2059$ MiB | $2074$ MiB |

**The finding that changes an operator's behaviour: chunking does nothing for a training step.**
The training peak is flat across the whole grid, to two parts in a thousand, while the dense
evaluation peak falls by $48\%$. The reason is that chunking changes the **order** the proposals are
computed in and not whether their activations are retained: under backward every chunk stays in the
graph until the gradient has flowed. The production configuration told an operator to reach for the
chunk sizes before the batch size, which would have cost hours and achieved nothing; it now says the
opposite, with these numbers beside it.

The training peak scales very nearly linearly in the batch, so the configured $128$ needs about
$8.1$ GiB per device and a $12$ GiB device carries it with room for the loader.

**Reassociation at the production widths**, at a quarter of each axis: the bounded update moves by
$1.2\times10^{-5}$ and the decoder output by $2.4\times10^{-7}$, against a repeat-run floor of
**exactly zero** under the pinned numeric environment. Two defects in the first draft of this
measurement are worth recording, because both would have produced a plausible wrong number: the
forward **samples**, so two runs at different latent draws differ in every decoded coefficient by
four orders of magnitude more than any summation order does; and at the model's zero-initialised
source projection every proposal is exactly zero, so a sum of zeros has no order and the pass would
have reported a tolerance no trained run could meet. The draw is now pinned inside each run and the
source pathway is woken before the comparison.

Note for T027: the update's $1.2\times10^{-5}$ is thirteen times the tiny geometry's figure, which
is what summing $91$ terms instead of $5$ costs. A comparison across chunk settings at production
scale wants an absolute tolerance on the update of about $10^{-4}$, not the fixture's $10^{-5}$.

#### T024: The target-only arm

Requirements: FR-029
Depends on: T005
Description: Add `source_disabled` to the constructor. It builds **no** source pathway at all — no
pointwise encoder, no lag embeddings, no proposal head — so the full distribution is the prior, the
divergence is exactly zero and the two decoded forecasts are bitwise identical. A constructor
decision rather than a flag the forward consults, for the reason the mean-only arm is one. Ship
`configs/target_only.yaml` for it.
Acceptance criteria:
- A module-tree walk finds no source encoder and no proposal head, and the checkpoint's key set
  carries no source tensor.
- The full parameters equal the prior bitwise and the divergence is exactly zero.
- The warm start transfers into a model of the full architecture and leaves its source at zero.
- The production configuration names the new keyword, which the existing key-parity test enforces.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/nets/core.py` - the arm
- `teb_vae/lag_slot_transformer_cfs/nets/model.py` - the keyword and the forward branch
- `teb_vae/lag_slot_transformer_cfs/configs/default.yaml` - the leaf
- `teb_vae/lag_slot_transformer_cfs/configs/target_only.yaml` - new
- `teb_vae/lag_slot_transformer_cfs/trainer.py` - the transfer
- `teb_vae/lag_slot_transformer_cfs/tests/test_checkpoint_contract.py` - extend
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/ -q`.
Expected: all pass. Status: executed.
Test rationale: an arm that built the modules and never used them would be a starved parameter block
under a distributed run and a claim in the manifest that the model reads a source it does not.
Runtime: fast/local.
Evidence: 2026-09-09. The arm builds at no source tensors and its forward reproduces the prior
bitwise. **One change to the warm start fell out of it and is a real correction**: `clock_proj.`
moved from the re-zeroed source prefixes to the transferable ones. The metadata clock is a function
of stored position and conditions the **prior**, so it is no more part of the source pathway than
the target encoder is — and the target-only arm of this same class now trains it, so re-zeroing it
would discard a trained target-only component and start the candidate's prior from a state its own
baseline never occupied. That is precisely the baseline weakness the warm start exists to avoid. A
donor carrying no such tensor reports it as missing and leaves it at its constructed zero.

#### T025: Scoring a source-free checkpoint, and the external-reference verdict

Requirements: FR-029, FR-020
Depends on: T024, T021
Description: The frozen reference has to be scored through the **same** estimator, anchors, mask and
draws as the candidate, or the comparison is between two scoring paths. Make the scoring pass run
against a target-only checkpoint: record the arm, skip every intervention by name with its reason,
and report an empty exposure rather than counts of zero. Then give the acceptance gate an optional
reference summary and the one verdict the internal gap cannot supply.
Acceptance criteria:
- A target-only checkpoint scores without error, and its summary records the arm.
- Every skipped intervention is named with a reason; no margin is reported as zero.
- The gate's two inconclusive verdicts name the arm rather than reporting an absent measurement.
- A candidate whose own base branch has fallen behind the frozen reference **fails**, because that
  is not a threshold question: joint training made the baseline worse and any gap measured against
  it is measuring that.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/eval/run.py` - the source-free path
- `teb_vae/lag_slot_transformer_cfs/eval/binding.py` - the disclosure
- `teb_vae/lag_slot_transformer_cfs/eval/verify.py` - the reference verdict
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_arms.py -q`.
Expected: all pass. Status: executed.
Test rationale: without this the reference cannot be scored at all, and the comparison the design
makes a precondition for reading any source result is unavailable exactly where it is needed.
Runtime: fast/local, inside the integration fixture below.
Evidence: 2026-09-09. The reference scores at a gap of exactly $0.0$ with every intervention named
and skipped. The reference verdict is **reported and not gated** on its margin — where an acceptable
one sits is what T027 measures — but it **is** gated on the base drift, which fails when the
candidate's own base branch scores worse than the frozen reference's.

#### T026: The arm configurations and the runbook

Requirements: FR-029, FR-018
Depends on: T024
Description: Ship `configs/target_only.yaml` and `configs/joint.yaml`, and write the operator
sequence below. The candidate's file sets the warm-start key and turns `head_init_calibration` off,
which is the leaf a warm-started run is most likely to get wrong: the calibration would overwrite
exactly the tensors the transfer just copied.
Acceptance criteria:
- Both configs load and construct through the driver's kwargs sweep.
- The candidate's config sets the warm-start key and disables the calibration.
- The production config documents both checkpoint keys and why they are refused together.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/configs/target_only.yaml` - new
- `teb_vae/lag_slot_transformer_cfs/configs/joint.yaml` - new
- `teb_vae/lag_slot_transformer_cfs/configs/default.yaml` - the documented keys
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_config_load.py -q`.
Expected: all pass. Status: executed.
Test rationale: the existing key-parity check catches the failure that matters, and it is silent
otherwise: a config key naming no constructor parameter is ignored, so the run trains a different
model than the file describes. Both new arms go through it by being configs of this package.
Runtime: fast/local.
Evidence: 2026-09-09. Both arms load. The production config's `core_model_checkpoint` comment said
it was "a target-only checkpoint to warm-start from", which describes the **other** key's job: it is
a strict load of this exact model kind, source pathway included. Corrected, with both keys and the
reason they are refused together written out.

#### T027: The three training runs

Requirements: FR-029
Depends on: T023, T024, T025, T026
Description: Fit the target-only arm; fit it again at a second seed and freeze it; warm-start the
candidate from the first; score the candidate and the reference; read them together.
Acceptance criteria:
- Three checkpoints exist, each with its resolved configuration beside it.
- The candidate's first validation epoch reports a divergence of about zero, which is the evidence
  that the transfer left the source pathway where it belongs.
- Both evaluations run through the same entry point at the same draw count, and the gate reports the
  candidate against the frozen reference.
Files affected:
- the three run directories the fits write and the two evaluation directories that score them. No
  source file changes: everything this task needs is built, and running it is the task.
Validation: the runbook below, and the three checks it names. **Not startable from the machine
this plan was written on:** it needs the integer-operator production shards, which are absent here
-- `dataset_config` points at `/data1/...`, which does not exist -- and the seven devices the
production configuration describes, against the one available. Everything it depends on is done
and is exercised end to end at fixture scale by `tests/test_arms.py`.
Test rationale: the sequence itself is covered by `tests/test_arms.py`, which runs all three fits,
both evaluations and the gate in about seventeen seconds against the committed fixture. What no test
can cover is whether the baseline is COMPETITIVE, which is a property of a converged run on real
data and of nothing else -- so this task's validation is the runbook's three checks rather than an
assertion.
Runtime: days, on the production box.
Evidence: none.

### The runbook

Four commands, from the repository root, in this order. Repoint `dataset_config` in
`configs/default.yaml` at the real shards first; every arm inherits it.

```
# 1. The baseline the candidate starts from.
python -m teb_vae.lag_slot_transformer_cfs.trainer \
    --config teb_vae/lag_slot_transformer_cfs/configs/target_only.yaml

# 2. The frozen external reference: the SAME file, a different seed and tag. Edit the two leaves.
python -m teb_vae.lag_slot_transformer_cfs.trainer \
    --config teb_vae/lag_slot_transformer_cfs/configs/target_only.yaml

# 3. The candidate. Set target_warm_start_checkpoint in joint.yaml to run 1's best checkpoint.
python -m teb_vae.lag_slot_transformer_cfs.trainer \
    --config teb_vae/lag_slot_transformer_cfs/configs/joint.yaml

# 4. Score both, then read them together.
python -m teb_vae.lag_slot_transformer_cfs.eval.run --checkpoint <run 2>/model_checkpoints/<f>.ckpt
python -m teb_vae.lag_slot_transformer_cfs.eval.run --checkpoint <run 3>/model_checkpoints/<f>.ckpt
python -m teb_vae.lag_slot_transformer_cfs.eval.verify \
    --summary <candidate eval>/eval_results/summary.json \
    --reference <reference eval>/eval_results/summary.json
```

Before launching, run the memory pass against the real shards, which resolves the warm-up budget and
therefore measures the **gated** model rather than the upper bound recorded above:

```
python -m teb_vae.lag_slot_transformer_cfs.eval.memory --output memory.json
```

**Three things to check rather than assume**, each of which is cheap and each of which invalidates
the whole run if wrong:

1. Run 1 and run 2 must report `val/source_conditioned_kl_raw` of exactly $0.0$ in every row. A
   target-only arm whose divergence is not identically zero did not build that arm.
2. Run 3's **first** validation epoch must report a divergence of about zero. Anything else means
   the transfer did not leave the source pathway at zero, and every nat of coupling the run later
   reports was inherited rather than earned.
3. Run 3's transfer log lists the tensors it transferred, could not find and left as constructed.
   The middle list must be empty, or the run is training part of its base from random weights while
   the configuration says otherwise.

Uncertainty this sprint does not remove: the training budget, the divergence ramp length and the
early-stopping patience are proposed starting settings rather than measured optima, and whether the
target-only baseline is **competitive** — which the design makes a precondition for reading any
source result — is exactly what run 1 establishes and nothing before it can.

## Sprint 6: Mechanism-separating comparisons

Goal: the arms of design section 11.1 exist as constructions, ship as configurations, score through
the candidate's own estimator, and report their parameters, compute and memory beside their gaps.

Refined 2026-09-09, against its own precondition: Sprint 5's machinery is complete and its three
runs are the operator's, so the arms are built against the pipeline the candidate already runs
rather than against a hoped-for one.

Demo: from the repository root, build all six arms from their configurations, fit and score three
comparators against the committed fixture, and read each summary's `arm` block: the mechanism it
moved, its parameter split, and what its band margins mean.

Definition of Done: T028 to T033 done with evidence; the sprint's test files pass; the four shipped
forecasters remain untouched. **T028 to T032 are done; T033 is not**, and cannot be: it is the
training runs, and it needs the production shards and the box the production configuration
describes. Section 5.4 records the boundary and it has not moved.

Dependencies: Sprint 5, and — for T033 alone — access to the production shards and Sprint 5's own
three runs.

### The decision this sprint opened with

**The attention comparator is a construction in this package, not a configuration arm of the
shipped forecaster.** Section 7's forecast left it open and it is settled here, on evidence rather
than on preference: the comparator has to carry this design's prior-relative posterior equations,
its bounded scale range, its explicit metadata clock and its shared-noise sampling, and the shipped
forecaster has **none** of those. It decodes its base branch at the prior mean rather than under a
shared draw, so its two columns are different estimators; it forms its prior clock by encoding a
zeroed source through the source network rather than reading a function of stored position; and its
posterior log-variance is an independently bounded head rather than a bounded residual on the
prior's, so its divergence is a different quantity with a different range. Making it a
configuration arm there would mean changing the shipped model — which section 1.4 puts out of scope
and NFR-006 measures — and would change four things at once in a comparison built to change one.

### What the arms are, and why they are these five

Two orthogonal constructor axes, and three flags, giving a chain in which **every adjacent pair
differs in one declared leaf**:

| Configuration | `source_stem` | `lag_fusion` | Other | Isolates, against the row above |
| --- | --- | --- | --- | --- |
| `attention_reference.yaml` | `conv` | `attention` | — | — |
| `pointwise_attention.yaml` | `pointwise` | `attention` | — | the source temporal convolution |
| `joint.yaml` (the candidate) | `pointwise` | `local` | — | attention versus the explicit sum |
| `mean_only.yaml` | `pointwise` | `local` | `mean_only_residual` | the variance half of the update |
| `capacity_control.yaml` | `pointwise` | `local` | `source_values_withheld` | the source values, capacity held |
| `target_only.yaml` | — | — | `source_disabled` | the source pathway entire |

The chain is the point. A comparison against the historical checkpoint measures a package of five
simultaneous changes; each adjacent pair here measures one.

**The capacity control is a fourth construction and not the target-only arm**, which Sprint 5's
record already anticipated. Design section 8.3 item 5 asks for an enabled residual head of
comparable trainable capacity reading lag identity and masks and no source values. `source_disabled`
removes the pathway and its whole parameter budget with it; this arm holds the pathway, keeps the
widths, the head, the optimizer state and the checkpoint key set exactly, and withholds the values
at the encoder so no coefficient ever enters the graph. A gain on it is evidence about extra
nonlinear target capacity rather than about uterine activity, which is the confound it exists for.

### Tasks

#### T028: The attention fusion, and the arm axes it introduces

Requirements: FR-030, NFR-005
Depends on: T006
Description: Add `nets/lag_attention.py` with `LagAttentionFusion`: an anchor-indexed cross-attention
from the conditioning state over each lag's source vector, with a learned per-lag key bias carrying
lag identity, plain softmax over the lag axis, and one output projection into the same update pair
the local head emits. Add the two constructor axes `source_stem` and `lag_fusion` and the head count
`lag_attention_heads`, refuse the combinations that are not one declared change, and route the
forward to whichever fusion the arm built. Everything downstream — the limiter, the prior-relative
parameters, the residual divergence, the paired sampling, the shared decoder, the objective and its
reduction — is the recommended arm's, unchanged.
Acceptance criteria:
- The two fusions are different modules under one attribute name, so an arm is a different
  checkpoint key set and a strict load across two arms refuses.
- Every selector zero, and every lag unavailable, reproduce the prior **bitwise** on a woken
  pathway, not merely at the zero initialisation.
- An all-ones selector reproduces the matched forward bitwise, so the empty-band reference identity
  holds under the second suppression path as well as the first.
- The predictive gradient reaches the final projection at the zero start.
- `lag_scale` and `lag_chunk` are refused under attention fusion, each naming why.
- No attention-shaped key is emitted anywhere in the forward.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/nets/lag_attention.py` - new
- `teb_vae/lag_slot_transformer_cfs/nets/core.py` - the axes, the refusals, the construction
- `teb_vae/lag_slot_transformer_cfs/nets/model.py` - the keywords and the attention accumulation
- `teb_vae/lag_slot_transformer_cfs/tests/test_mechanism_arms.py` - new
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_mechanism_arms.py -q`.
Expected: all pass. Status: executed.
Test rationale: the source-off identities are what every reported margin is read against, and they
are trivially true at the zero start — so the assertion has to be made on a woken pathway or it
proves nothing. Under this fusion they hold for a different reason than under the sum, which is
exactly the kind of difference that survives a review and fails a run.
Runtime: fast/local. Observed at 2 s.
Evidence: 2026-09-09. 42 tests pass. Two decisions the task did not anticipate.

**The attention distribution is not exported, and that is the sprint's most consequential choice.**
The weights are real on this arm, unlike anything the recommended arm could offer under that name.
Published beside a predictive comparison they would be read as a lag readout by every reader and
every downstream table, which is the one claim the evidence behind this architecture shows cannot
be supported. Both fusions are therefore interrogated through the **same** interface — suppress a
band with the selector and rescore — and the forbidden-key gate in `verify.py` stays intact and now
runs against the one arm where it could ever fire.

**The update is gated on whether any lag survived, rather than relying on the zero initialisation.**
With every selector off, the scores are all minus infinity, the softmax normalises to NaN, the
sanitiser turns that into zero — and the output projection's *bias* would then make the update a
learned constant rather than zero. The gate makes the identity exact on trained weights, which is
where it has to hold.

#### T029: The convolution source stem

Requirements: FR-030, NFR-005
Depends on: T028
Description: Add `nets/conv_source.py` with `ConvSourceStem`: the family's availability adapter,
resolved by the mixin's own rule, feeding the shipped gated causal convolution stack at the target
encoder's schedule — the same schedule the evaluated source path used, so this arm reproduces the
representation being compared against rather than a differently shaped one. Give it the pointwise
encoder's interface, `forward` and `gather`, so the composing model calls the same two methods on
either representation and holds no case for which. Extract the lag index arithmetic both gathers
share.
Acceptance criteria:
- The stem's reach is read off the module and is greater than one stored step, and the pointwise
  encoder reports no such attribute.
- The per-channel availability is **identical** to the pointwise arm's on the same stream, so two
  arms' exposure tables are comparable.
- The lag index and its validity are decided once, by one expression, for both gathers.
- The scalar lift and the withheld values are both refused alongside this stem, each naming why.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/nets/conv_source.py` - new
- `teb_vae/lag_slot_transformer_cfs/nets/pointwise_source.py` - the shared index, the state gather,
  the encoder's own gather method
- `teb_vae/lag_slot_transformer_cfs/tests/test_mechanism_arms.py` - extend
Validation: same command as T028. Expected: all pass. Status: executed.
Test rationale: two copies of the lag index arithmetic is how a per-channel gather and a per-state
gather come to disagree about which lags exist, and the disagreement is a wrong number rather than a
failure — both shapes stay correct either way. The availability equality is what makes an exposure
table an exposure table rather than a property of the encoder.
Runtime: fast/local. Observed inside the file above.
Evidence: 2026-09-09. One decision worth carrying: **the stem carries the pointwise encoder inside
it**, parameter-free, purely for its availability rule. Whether a channel has warmed up is a
property of the channel and the stored step and does not stop being one because an encoder mixed the
values together; resolving it twice is how two arms come to disagree about which lags exist. It also
means the nonfinite-value refusal fires on this arm exactly as on the recommended one — before a
convolution has spread the value across every window that touches it.

#### T030: The capacity control

Requirements: FR-030
Depends on: T028
Description: Add `source_values_withheld` to the pointwise encoder and to the constructor. The
encoder emits the availability bit and an exact zero in place of every coefficient, so the fusion
receives lag identity and the announcement and no value. Widths, parameter count, optimizer state
and checkpoint key set are the candidate's exactly. Teach the scoring pass to skip the interventions
that are not interventions on it, by name and with the reason.
Acceptance criteria:
- The parameter split and the state-dict key set equal the candidate's, tensor by tensor.
- No source value enters the graph, asserted on what the **encoder emitted** rather than on whether
  a forecast moved.
- The availability bit is still present and still informative, so the arm is not a second
  target-only model.
- The replacement and permutation arms are skipped with a reason; band suppression still runs.
- It is refused alongside the scalar lift and alongside the convolution stem.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/nets/pointwise_source.py` - the flag
- `teb_vae/lag_slot_transformer_cfs/nets/core.py`, `nets/model.py` - the keyword and its refusals
- `teb_vae/lag_slot_transformer_cfs/eval/run.py` - the skipped controls
- `teb_vae/lag_slot_transformer_cfs/tests/test_mechanism_arms.py` - extend
Validation: same command as T028, plus `test_arm_scoring.py`. Expected: all pass. Status: executed.
Test rationale: a model whose update merely happened not to move on one batch would pass a
comparison of forecasts, so the assertion has to be on the encoding. And a replacement margin of
exactly zero on this arm reads as "replacing the source changed nothing" when the truth is that
nothing was replaced — which is a finding-shaped defect, the kind this suite exists to catch.
Runtime: fast/local.
Evidence: 2026-09-09. The arm builds at the candidate's parameter count exactly and its state dict
is key-for-key identical. **The permutation arm needed skipping as well as the replacements**, which
the task did not anticipate: the availability announcement is a function of stored position and the
resolved warm-up alone, so pairing a recording with another one changes nothing this arm reads, and
a recording-specificity margin of zero would state as a measurement something true of the arm by
construction.

#### T031: The arm configurations and the summary that makes them a comparison

Requirements: FR-030, FR-018, NFR-005
Depends on: T029, T030
Description: Ship the four comparator configurations as deltas on the candidate's, each moving one
leaf and each warm-starting from the **same** target-only checkpoint. Record the arm in every
`summary.json`: the mechanism it moved, its parameter split by pathway, and what its band margins
mean. Extend the binding's reconciled geometry keys with the four that decide which arm a checkpoint
is. Extend the encoder disclosure so a comparator reports its own reach rather than the recommended
arm's.
Acceptance criteria:
- Every configuration loads and constructs through the driver's kwargs sweep, and the existing
  key-parity and refused-keyword checks cover them — which means they are discovered rather than
  written out, or an arm added and forgotten would be an arm nothing checked.
- The four arm leaves are reconciled geometry keys, so a checkpoint and a configuration cannot
  disagree about which arm produced a number.
- Each summary's `arm` block names the stem, the fusion, the three flags, the parameter split and
  the suppression semantics.
- The convolution arm's disclosure reports the stem's reach; the pointwise arms report one step.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/configs/mean_only.yaml`, `capacity_control.yaml`,
  `pointwise_attention.yaml`, `attention_reference.yaml` - new
- `teb_vae/lag_slot_transformer_cfs/configs/default.yaml` - the four leaves and the arm chain
- `teb_vae/lag_slot_transformer_cfs/eval/run.py` - the arm record
- `teb_vae/lag_slot_transformer_cfs/eval/binding.py` - the geometry keys and the disclosure
- `teb_vae/lag_slot_transformer_cfs/nets/core.py` - the pathway parameter census
- `teb_vae/lag_slot_transformer_cfs/tests/test_config_load.py` - the configuration set is discovered
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_config_load.py teb_vae/lag_slot_transformer_cfs/tests/test_eval_binding.py -q`.
Expected: all pass. Status: executed.
Test rationale: the key-parity check catches the silent failure a new arm is most likely to
introduce, because an arm is written by copying a sibling: a key naming no constructor parameter is
dropped without a word and the run trains a different model from the one the file describes. The
configuration set became a directory scan for the same reason — an arm the tuple forgot is an arm
nothing checked.
Runtime: fast/local. Observed at 3 s across the two files.
Evidence: 2026-09-09. 45 tests pass across the two files. **One correction fell out of it and is
real**: `source_disabled` was not a reconciled geometry key, although it decides whether a reported
gap is a measurement or is zero by construction. It is one now, alongside the three new arm leaves.
The parameter census lives on the core and uses the **same** two module prefixes the warm start
leaves at zero, deliberately: what a transfer treats as the source pathway and what a comparison
counts as the source pathway have to be one boundary.

#### T032: The comparators through the real pipeline, and the per-arm budget measurement

Requirements: FR-030, NFR-001
Depends on: T031
Description: Fit and score each comparator through the real entry points at fixture scale, and make
the memory pass report an arm's parameters, compute and peak memory so a comparison has budgets
beside its gaps. Hold the lag axis whole on an arm whose aggregation over it is normalised, and say
so in the record rather than dropping half the grid without a word.
Acceptance criteria:
- Each comparator fits, scores and passes the acceptance gate end to end.
- The two reference identities hold on trained weights under the selector path as well as the
  subtractive one.
- Both fusions report the same exposure axes, so two arms' tables are comparable.
- The memory pass records the arm, the parameter split by pathway and a wall time per step, and
  runs on every arm including the target-only one.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/tests/test_arm_scoring.py` - new
- `teb_vae/lag_slot_transformer_cfs/tests/test_arms.py` - the two fit/score helpers become public
- `teb_vae/lag_slot_transformer_cfs/eval/memory.py` - the arm record, the step time, the arms that
  cannot chunk the lag axis
- `teb_vae/lag_slot_transformer_cfs/task.py` - the cancellation readouts become arm-dependent
- `teb_vae/lag_slot_transformer_cfs/eval/EVAL.md` - what changes on a comparator arm, and which
  gate each one suspends by declaration
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_arm_scoring.py -q`.
Expected: all pass. Status: executed.
Test rationale: a comparator that trains for a week and then cannot be scored through the
candidate's estimator is not a comparator, and nothing short of running the pipeline finds that.
Runtime: integration. **Observed at 16 s** for three fits, three evaluations and the gate.
Evidence: 2026-09-09. 12 tests pass. **The fit found a real defect**: the task read
`cancellation_ratio_mean` unconditionally, and an attention arm emits neither cancellation channel —
so every comparator run would have died on its first validation step. Both channels are now reported
only where the arm produces them, matching the forward and matching what the mean-only arm already
did with the scale channel. The tracked metric list stays the union deliberately: a tracked name a
run never emits costs an empty column, while a name a run emits and nothing tracks costs the number.

The memory pass runs on all five arms and the finding it exists to carry is recorded rather than
assumed: on an attention arm the lag axis is held whole, because chunking it would renormalise the
distribution inside each chunk and measure a different model rather than the same one in a different
order. The per-arm peaks at the production geometry are T033's, since they need the device.

#### T033: The comparison itself

Requirements: FR-030
Depends on: T032, T027
Description: Fit each comparator arm from the same target-only baseline Sprint 5's candidate starts
from, score each through the same entry point at the same draw count, and read the five gaps against
one frozen reference.
Acceptance criteria:
- Five checkpoints, each with its resolved configuration beside it, each warm-started from run 1.
- Each arm's first validation epoch reports a divergence of about zero, which is the evidence that
  its transfer left its own source pathway where it belongs.
- The per-arm memory pass is run against the real shards **before** each launch, because the
  attention arms project keys and values at the model width where the candidate holds proposals at
  the latent width, and the candidate's batch size does not carry over by assumption.
- Every arm is read against the same frozen reference, and each pair in the chain is read as one
  declared change.
Files affected:
- the five run directories the fits write and the five evaluation directories that score them. No
  source file changes: everything this task needs is built, and running it is the task.
Validation: the runbook below, and the three checks Sprint 5's runbook already names, applied per
arm. **Not startable from the machine this plan was written on**, for the reason T027 is not: the
integer-operator production shards are absent and the production configuration describes seven
devices against the one available. It additionally depends on T027, since every arm warm-starts
from run 1 and is read against run 2.
Test rationale: the sequence is covered by `tests/test_arm_scoring.py`, which fits and scores three
comparators through the real entry points in about sixteen seconds. What no test can cover is which
mechanism matters, which is a property of converged runs on real data.
Runtime: days, on the production box, per arm.
Evidence: none.

### The runbook

From the repository root, after Sprint 5's runs 1 and 2 exist. Set
`model_config.target_warm_start_checkpoint` in each file below to **run 1's** best checkpoint — the
same one `joint.yaml` uses — and leave `head_init_calibration` off, which each file inherits.

```
# Budgets first, against the real shards, per arm. The attention arms are the ones to check.
for arm in joint mean_only capacity_control pointwise_attention attention_reference; do
  python -m teb_vae.lag_slot_transformer_cfs.eval.memory \
      --config teb_vae/lag_slot_transformer_cfs/configs/$arm.yaml --output memory_$arm.json
done

# Then the four comparators. joint.yaml is Sprint 5's run 3 and is already fitted.
python -m teb_vae.lag_slot_transformer_cfs.trainer \
    --config teb_vae/lag_slot_transformer_cfs/configs/mean_only.yaml
python -m teb_vae.lag_slot_transformer_cfs.trainer \
    --config teb_vae/lag_slot_transformer_cfs/configs/capacity_control.yaml
python -m teb_vae.lag_slot_transformer_cfs.trainer \
    --config teb_vae/lag_slot_transformer_cfs/configs/pointwise_attention.yaml
python -m teb_vae.lag_slot_transformer_cfs.trainer \
    --config teb_vae/lag_slot_transformer_cfs/configs/attention_reference.yaml

# Score each against the SAME frozen reference Sprint 5 produced as run 2.
python -m teb_vae.lag_slot_transformer_cfs.eval.run --checkpoint <arm>/model_checkpoints/<f>.ckpt
python -m teb_vae.lag_slot_transformer_cfs.eval.verify \
    --summary <arm eval>/eval_results/summary.json \
    --reference <reference eval>/eval_results/summary.json
```

**One frozen reference serves every arm, and that is a property of the arms rather than a
convenience.** Every one of them carries the identical target encoder, prior, metadata clock and
decoder, so their base branches are comparable with one independently trained target-only predictor.
What must not be shared is the warm start: each arm starts from run 1, or a difference between two
of them contains a difference of baselines.

**Read the chain, not the table.** Each adjacent pair differs in one leaf, and the differences of
their gaps are what the sprint measures. Two cautions that are properties of the arms and not of the
results:

1. A band-suppression margin does not compare across the two fusions. Removing a lag from a sum
   leaves the other terms standing; removing it from a normalised distribution grows them. Each
   arm's margins are read against its own matched branch, and `arm.suppression_semantics` in every
   summary says which mechanism produced it.
2. A band narrower than the convolution stem's own reach is not resolving what its name says, since
   two lags closer together than that reach are summaries of overlapping windows. The reach is in
   each run's `encoder_disclosure`.

Uncertainty this sprint does not remove: whether any of these differences is larger than the spread
across training seeds. Design section 11.3 asks for at least three seeds on shortlisted models, and
that is Sprint 8's.

## Sprint 7: Synthetic instruments

Goal: generators whose source-to-target dependence is known, criteria declared before them, and a
campaign that reports power and a false-positive rate over repeated seeds rather than a display of
the runs that worked.

Refined 2026-09-09, against its own precondition: the lag readout of T019 now runs on real weights
through two fusions, so the instruments are built against the readout the campaign will actually
score rather than against an expected one.

Demo: from the repository root, run the campaign and read its record -- eleven instruments, each
with the truth it was declared with, the verdict the declared criteria reached, and the two rates
those verdicts aggregate to.

Definition of Done: T034 to T038 done with evidence; the sprint's test file passes; the four shipped
forecasters remain untouched. **All five are done.** Unlike sprints 5 and 6 this one has no task
that needs the production shards: an instrument's whole point is that it generates its own data, so
the campaign runs to completion here and its rates are recorded below.

Dependencies: Sprint 5 for the readouts, and Sprint 6 for the second fusion the campaign supports.

### The uncertainty this sprint opened with, and what survived

`teb_vae/lag_attn_cfs/lag_recovery_check.py` is the prior art, and the answer is that **its band
arithmetic survives and every profile it reads does not.** It scores four readouts -- a per-lag
divergence allocation, its support-corrected form, its clock-excess form, and a per-head attention
distribution -- and this architecture computes none of them, on any arm. What carries over is the
identity $\ell = d - h$ that turns a planted delay into a readable lag band, and the vocabulary of a
band's share of a profile's mass. Both are reimplemented here against the readout this package does
have: **proposal suppression over a declared partition of the lag axis**, which works identically on
both fusions and is the only lag readout either of them offers.

The consequence for a reader: a recovery figure from this campaign and one from that check are not
the same measurement, and the two must not be compared. The campaign records which readout produced
every verdict.

### Tasks

#### T034: The generators and the truth each is declared with

Requirements: FR-031, NFR-005
Depends on: T019
Description: Add `instruments/generators.py` with the ten stored-feature generators the design's own
table names, each returning the tensors the model reads and each carrying a frozen truth record:
whether its source carries information **beyond available target history**, the lag band a planted
delay is directly readable in, whether a detection would also be a directional claim, and what a
reader must know that those flags do not say. The band is derived from the referenced offsets and
the run's horizon, never written down.
Acceptance criteria:
- Every generator produces the declared shapes and finite values, and a build is a function of its
  seed: the same seed twice gives the same segments and two seeds give different ones.
- The planted generators actually carry their plant, measured on the tensors by a linear fit before
  any model is involved: the source at the referenced lag explains variance the target's own recent
  history does not, and a wrong lag explains none.
- The redundant source is nearly perfectly recoverable from target channels the model already holds
  and the planted source is nearly unrecoverable from them, asserted as a contrast.
- The direct support follows the horizon identity, clips at the near edge, and returns nothing where
  the band falls outside the searched window.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/instruments/__init__.py` - new
- `teb_vae/lag_slot_transformer_cfs/instruments/generators.py` - new
- `teb_vae/lag_slot_transformer_cfs/tests/test_instruments.py` - new
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_instruments.py -q`.
Expected: all pass. Status: executed.
Test rationale: the plant check is the one that keeps a low power figure honest. A generator whose
dependence never reached the streams reports no detections at any budget, and the campaign would
then read as a finding about the architecture rather than about the generator.
Runtime: fast/local. Observed at 3 s.
Evidence: 2026-09-09. 20 tests pass. **The synthetic generators deliberately write no shard**, which
the task did not settle and which matters: a stored-feature block synthesised here has no filter
bank behind it, so it has no real warm-up boundary, no group delay and no novelty curve. Written
into a causal shard it would carry those attributes fabricated, and the warm-up resolver reads
exactly them -- every number would then be computed against a boundary nobody measured. The
generators declare what they are instead, and the one instrument that does have a bank behind it is
T035's.

#### T035: The instrument that runs through the real feature operator

Requirements: FR-031
Depends on: T034
Description: Add `instruments/raw_process.py`: synthesise a raw pair in which the target follows the
source at a known delay, pass both through the production causal filter bank in memory at the
integer operator, resolve the warm-up budget from the bank's own channel plan, standardise with the
dataset's fixed transforms over each channel's warm region, and hand the result to the campaign as
one more generator. Its criterion is **relevance and not recovery**: the band is where the plant is,
not where a readout should peak.
Acceptance criteria:
- The coefficients come from the pipeline itself rather than from a reimplementation of it, and the
  widths and per-channel warm-up boundaries are the bank's.
- The delayed envelope is evaluated as a function rather than shifted out of a buffer, so the record
  carries no seam a filter would summarise as an event.
- The scored region standardises to unit scale and the cold region is left off-scale, which is the
  state a real shard is in.
- The record reports the plant, the peak and the distance between them, and records no pass or fail.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/instruments/raw_process.py` - new
Validation: exercised by the campaign below; its channel resolution and its standardisation are
checked by the campaign run's own geometry record. Status: executed.
Test rationale: this is the only instrument that can say anything about preprocessing causality, and
the thing it measures -- how far the operator moves an apparent lag -- is invisible to every other
one by construction.
Runtime: about 1.5 s per build, which is one pass of the real bank.
Evidence: 2026-09-09. The bank resolves 76 declared target channels and 42 declared source channels
at the integer operator, of which the budget keeps 58 and 27; the scored region standardises to
about $0.9$ and the cold region sits four to six times wider, exactly as a real shard does. **The
smallest geometry the bank accepts is its own refusal and is left where it is**: below it the filter
normalisation raises, and an instrument that shrank the geometry until the bank stopped complaining
would be measuring a different operator.

#### T036: The declared criteria and the rates they aggregate to

Requirements: FR-031
Depends on: T034
Description: Add `instruments/criteria.py`: the interval settings, the lag partition, the two
decision rules and the aggregation. Relevance fires when the lower end of a recording-level interval
on the per-segment predictive gap clears zero; recovery fires when the window carrying the largest
margin intersects the declared support **and** that window's own interval clears zero. Power is
measured on the generators that carry source information and the false-positive rate on those that
do not, and the two never share a denominator.
Acceptance criteria:
- A gap whose interval spans zero is not a detection, which is what stops a false-positive rate
  being a measurement of the sign of noise.
- A recovery needs both clauses: a peak in the right window with an indistinguishable margin is not
  one, and a confident margin in the wrong window is not one either.
- The lag partition covers every lag exactly once at widths that do and do not divide the axis.
- A run that found nothing stays in the denominator.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/instruments/criteria.py` - new
- `teb_vae/lag_slot_transformer_cfs/tests/test_instruments.py` - extend
Validation: same command as T034. Expected: all pass. Status: executed.
Test rationale: a point estimate is positive about half the time under no effect, so a rule reading
it alone would report a false-positive rate near one half whatever the readout did. And a partition
that lost a lag in its remainder would leave a plant sitting there unreachable with nothing saying
so.
Runtime: fast/local.
Evidence: 2026-09-09. The window partition is the readout's resolution and the width is a **declared
setting recorded with every verdict**, because two lags closer together than the source encoder's
own reach are summaries of overlapping windows -- so a per-lag readout would report a resolution the
representation does not have, at a hundred extra forwards per batch.

#### T037: The campaign

Requirements: FR-031, FR-017
Depends on: T035, T036
Description: Add `instruments/campaign.py`, following the launch convention. Per run it splits one
generator's segments three ways, fits on the first, stops on the second under the matched
full-branch block score, scores the gap and every window's suppression margin on the third through
this package's own readouts, and applies the criteria the generator was declared with. Register it
in the package's launch test tuple, which grows to cover the new runner directory.
Acceptance criteria:
- The module runs from the Run button with nothing filled in, and every launch rule holds.
- The three splits are disjoint and the scored one is never fitted, asserted on the slices.
- A selection naming no instrument is refused rather than running a smaller campaign.
- The record carries the criteria the verdicts were decided by.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/instruments/campaign.py` - new
- `teb_vae/lag_slot_transformer_cfs/tests/test_eval_launch.py` - the runner directories
- `teb_vae/lag_slot_transformer_cfs/tests/test_instruments.py` - extend
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_instruments.py teb_vae/lag_slot_transformer_cfs/tests/test_eval_launch.py -q`.
Expected: all pass. Status: executed.
Test rationale: the launch tuple is written out rather than discovered precisely so a runner that
forgot the convention fails instead of going unseen -- and a runner landing in a directory the guard
did not walk is the same hole by another route.
Runtime: fast/local for the tests. One campaign run is about 90 s.
Evidence: 2026-09-09. **Three splits, and the second and third were both added after a measurement
rather than by design.** The first arrangement scored the segments it had fitted, and every control
became a detection: the source branch has more capacity to memorise with than the target-only
branch, so an in-sample gap is a measurement of memorisation and it is positive on a generator whose
source carries nothing. Holding out a scoring split turned all three controls negative in one
change. The third split followed the same way: a fit stopped at whichever step scored best on the
set it is then reported on has had its stopping point chosen by the number it reports.

**And the fit split's size was measured too.** At a third of its shipped size the fit memorises
before the source pathway has learned anything -- traced over 3000 steps, the held-out gap on a
planted generator is $+0.33$ at step 500 and decays through zero to $-0.18$ by step 3000 as the
decoder drives its observation variance down. The instrument then has no power at all, for a reason
about data volume rather than about the readout.

#### T038: Run the campaign and record what it says

Requirements: FR-031
Depends on: T037
Description: Run every instrument and record the rates, the misses included.
Acceptance criteria:
- Every instrument produces a verdict, and the record names the truth each was declared with.
- The rates are reported per generator with power and the false-positive rate on separate
  denominators.
- The filter-bank instrument's plant, peak and the distance between them are recorded.
Files affected: none. Running it is the task.
Validation: `python -m teb_vae.lag_slot_transformer_cfs.instruments.campaign --output instruments.json`,
executed at one seed; the shipped three-seed campaign is the operator's, at about 50 minutes.
Test rationale: no test asserts an outcome here, deliberately. Power is a measurement, and a suite
that failed when it came out low would pressure the campaign toward a flattering number -- which is
the one thing an instrument must never be tuned for.
Runtime: about 90 s per run; eleven instruments at one seed is about 17 minutes.

Evidence: 2026-09-09, seed 11, at the shipped criteria and the shipped fit budget.

| Instrument | Source informative | Detected | Peak window | Declared support | Recovered |
| --- | --- | --- | --- | --- | --- |
| `single_delay` | yes | **yes** | $[8, 11]$ | $[8, 11]$ | **yes** |
| `several_delays` | yes | **yes** | $[8, 11]$ | $[8, 19]$ | **yes** |
| `broad_kernel` | yes | **yes** | $[12, 15]$ | $[8, 15]$ | **yes** |
| `common_driver` | yes | **yes** | $[8, 11]$ | $[8, 11]$ | **yes** |
| `state_dependent` | yes | no | $[8, 11]$ | $[8, 11]$ | no |
| `multi_lag_interaction` | yes | no | $[16, 19]$ | $[8, 19]$ | no |
| `informative_zero` | yes | no | $[0, 3]$ | $[8, 11]$ | no |
| `raw_filtered` | yes | **yes** | $[4, 7]$ | $[8, 11]$ | not graded |
| `target_autoregression` | no | no | -- | -- | -- |
| `redundant_source` | no | no | -- | -- | -- |
| `constant_source` | no | no | -- | -- | -- |

**The false-positive rate is $0/3$.** No control fires. That is the property the evidence behind
this architecture most needs: an independent noise source against a capable baseline, a source that
is a deterministic function of available target history, and a source that is constant within a
recording all report a **negative** gap and no detection.

**Power is $4/7$ on the informative generators**, and the three misses are the three the design
predicts rather than a scatter. `multi_lag_interaction` plants a product of two source times, which
is destroyed by removing either and is not separable additively -- a low rate there is a measurement
of a stated representational restriction, which the generator's own truth record says in advance.
`state_dependent` is synergy, where neither stream predicts the effect alone. `informative_zero`
makes a discrete event on three of eight channels the whole signal. Notably `state_dependent`'s peak
window is exactly its declared support: the lag readout points at the right place while the gap does
not clear zero.

**The filter-bank instrument is the sprint's most consequential number.** A delay planted at $12$
stored steps in the raw signals detects clearly -- a gap of $+1.42$ nats per anchor, interval
$[+1.24, +1.60]$ -- and its suppression profile peaks at lags $[4, 7]$, four to eight lags **nearer
the anchor** than the plant, with the window containing the plant carrying about a sixth of the peak
window's margin. The operator moves the apparent lag. That is the design's own claim about content-
lag spread, measured end to end for the first time rather than argued: a readout on real recordings
cannot be read as a physiological delay, and this is how far off it would be.

Two limits on every figure above. They are one seed, and the shipped campaign is three. And they are
rates for the campaign's **reduced fit** -- the production forward, objective, reduction, sampling,
decoder and readouts under a plain optimizer with no learning-rate schedule -- so a rate here is not
a production run's sensitivity.

### The runbook

One command, from the repository root, and nothing has to be filled in:

```
python -m teb_vae.lag_slot_transformer_cfs.instruments.campaign --output instruments.json
```

About fifty minutes for eleven instruments at three seeds. Two flags are worth knowing:
`--skip-raw` drops the only instrument that imports the feature pipeline, and `--only <names>` runs
a subset. `--seeds` widens the repeats, which is the one setting that tightens a rate.

**What must not be changed to make a rate look better.** The effect sizes in `generators.py`, the
criteria in `criteria.py` and the fit budget in `campaign.py` are the instrument. Moving any of them
between two campaigns makes the two incomparable, and moving one after seeing a rate makes that rate
a search result. If a generator needs a different effect size, declare it as a second generator and
report both.

## Sprint 8: Acceptance

Goal: the protocol of design section 11.3, built so that a shortlist can be read the moment one
exists -- several training seeds, one predeclaration, paired recording-level intervals, a corrected
band search, a latent that is probed rather than assumed, and a partition nothing was chosen on.

Refined 2026-09-09 against the one question section 7 recorded for it, which had to be answered
before any task could be written: which partition has already been used to choose something.

Demo: from the repository root, point the protocol at a directory of finished evaluation runs and
read its record -- which arms it found and at how many training seeds, what each declared comparison
says with its interval, which band the search peaked at and what that band's interval is once the
family is paid for, and which verdicts can fail.

Definition of Done: T039 to T042 done with evidence; the sprint's two test files pass; the four
shipped forecasters remain untouched. **T039 to T041 are done.** T042 is the operator's for the
reason T027 and T033 are: a protocol for reading trained arms cannot be run before there are any.

Dependencies: Sprint 6 for the arms a comparison ranges over, Sprint 7 for what a readout can be
asked, and Sprint 5's runbook for how one run is produced.

### The question this sprint opened with, and its answer

**Which partition has already been used to choose something?** Answered, and the answer has two
halves that needed different evidence.

**For the runs that already exist, it is fold 1's test partition, and it is recoverable from their
own artifacts.** The September 8 diagnosis -- the evidence this architecture was designed from --
was scored on `k_fold_cross_validation_dataset/fold_1/test` at the integer operator, $1{,}959$
recordings across eight subgroup shards. That is a path in its resolved configuration, and it is
also, recording by recording, in the `coupling_per_recording.csv` the shipped cell's pass wrote
beside it: those tables are indexed by GUID, so the cohort every choice so far was made on is
enumerable without new machinery. Fold 1's test partition is therefore **spent**: an architecture
was chosen against it, and a confirmation on it would confirm the choice with the evidence that
produced it.

**For the runs this package will produce, no artifact said which recordings it scored**, and that
is what T039 fixes. Every summary now records the files opened, their common parent and a digest of
the recordings that came back, and writes the per-recording table beside it. The consequence is that
disjointness is **checked** rather than declared: two evaluation directories can name different
shards and still overlap, because a cross-validation fold's train partition contains another fold's
test recordings, and the ten folds of this build are drawn from one GUID pool.

**What the reserved partition is, then.** A fold whose test partition no run has been pointed at,
declared by repointing a second evaluation delta at it, and verified against the selection runs'
recordings rather than against their paths. The training shards are not the constraint here: this
package trains on the pre-training dataset, which is built from the leftovers of the classification
cohort, so a fold's test partition is unseen by every fit either way. Selection is the constraint,
and selection is what the recordings answer.

### Tasks

#### T039: The scored-recording provenance and the per-recording table

Requirements: FR-032, NFR-005
Depends on: T020
Description: Make one run say which recordings it scored and carry the values its intervals were
built from. The summary gains a `scored_split` block -- the files, the statistics, their common
parent, the recording count and a digest of the recording set -- and the run block gains the
**training** seed and tag beside the evaluation seed. The pass writes `per_recording.csv` beside the
summary, one row per recording, following the siblings' per-recording exports.
Acceptance criteria:
- The table carries one row per scored recording and every scored column, and the equal-recording
  mean of a column reproduces the summary's point for it.
- The split block names the shards, the statistics file, the common parent and a digest of exactly
  the recordings that were scored.
- The training seed and the evaluation seed are both recorded and are distinguishable.
- The segment and anchor counts come out of the same walk as the means beside them.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/eval/run.py` - the table, the split block, the two seeds
- `teb_vae/lag_slot_transformer_cfs/tests/test_eval_smoke.py` - extend
- `teb_vae/lag_slot_transformer_cfs/eval/EVAL.md` - what a pass writes
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_eval_smoke.py -q`.
Expected: all pass. Status: executed.
Test rationale: the mean of the table against the summary's point is the assertion that matters. Two
paths to one number, and a table built over a different population -- a different empty-segment
rule, a different validity gate -- is otherwise invisible: every row would look reasonable.
Runtime: integration, inside the existing smoke fixture. Observed at 13 s for the whole file.
Evidence: 2026-09-09. 16 tests pass. **The counts had to come out of the same walk as the means**,
which the task did not originally say: a segment that scored no anchor is excluded from the mean,
and a count computed under any other rule would describe a different population from the number it
qualifies. The one design decision here is the split label: it is the common parent of the scored
files rather than a parsed fold name, because a parse encodes one dataset layout into a gate that
has to keep working when the next build names its directories differently.

#### T040: The latent probes

Requirements: FR-032
Depends on: T006
Description: Add `eval/latent_probes.py`, following the launch convention: a pass that fits a frozen
ridge probe from each of six latent readouts -- both distributions' means and scales and one
shared-noise draw of each -- onto the forecast block relative to the anchor's own stored values, on
recordings split by a digest of their identifier, and scores it on the side it was not fitted on.
Register it in the launch tuple.
Acceptance criteria:
- The two sides of the split are disjoint and both are populated, asserted on the assignment rather
  than on the way it was drawn.
- The streaming solve agrees with the same ridge fitted from the rows, in FP64.
- The residual and total sums of squares agree with the ones the rows give.
- A readable target scores near one and noise does not, so a probe reporting one number whatever it
  is handed cannot pass.
- A latent coordinate with no variance contributes nothing rather than dividing by zero.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/eval/latent_probes.py` - new
- `teb_vae/lag_slot_transformer_cfs/tests/test_latent_probes.py` - new
- `teb_vae/lag_slot_transformer_cfs/tests/test_arms.py` - the fixture delta becomes public
- `teb_vae/lag_slot_transformer_cfs/tests/test_eval_launch.py` - the new runner
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_latent_probes.py -q`.
Expected: all pass. Status: executed.
Test rationale: the streaming solve is the one piece here that can be wrong without looking wrong.
It exists so the probe can use every complete-coverage anchor instead of a subsample, and a sign or
a cross term lost in the expansion of the residual sum of squares would produce a plausible
coefficient on every arm.
Runtime: fast/local for the arithmetic; one fit and one pass for the seam. Observed at 11 s.
Evidence: 2026-09-09. 9 tests pass. **The target is anchor-relative and the subtraction uses the
anchor's stored values rather than the decoder's learned persistence weights**, which is the
decision worth recording: a baseline that moved with the model being probed would make two arms'
coefficients incomparable, and comparability across arms is the only use this pass has. The null is
the fitting split's own mean, which is what a probe that had seen only those recordings would
predict -- scored against the scoring split's mean instead, every probe would collect the difference
between two cohorts for free.

#### T041: The predeclaration and the protocol

Requirements: FR-032, FR-017
Depends on: T039, T040
Description: Ship `eval/configs/acceptance_plan.yaml` -- the seed minimum, the primary draw count,
the stability draw counts, the resamples, the five declared comparisons and the bands a search may
range over -- and `eval/acceptance.py`, which reads a directory of finished runs under it. Group by
arm and training seed, average per recording across seeds, pair across arms, bootstrap over
recordings once, correct the band search for its family, attach each probe artifact to its run by
checkpoint, and decide the reserved partition on the recordings. Register it in the launch tuple.
Acceptance criteria:
- The plan's key set is closed and a comparison against an arm this package does not build is
  refused by name.
- Three scorings of one checkpoint count as one training seed.
- A comparison across two draw counts, two evaluation seeds or two splits is refused as unmatched
  rather than read.
- The family-adjusted interval is wider than the nominal one, and the two identity suppression arms
  are not members of the family.
- The confirmation verdict decides on recordings and fails on an overlap whatever the paths say.
- The gate imports no model, no binding and no framework.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/eval/configs/acceptance_plan.yaml` - new
- `teb_vae/lag_slot_transformer_cfs/eval/acceptance.py` - new
- `teb_vae/lag_slot_transformer_cfs/eval/__init__.py` - the module list
- `teb_vae/lag_slot_transformer_cfs/eval/EVAL.md` - reading several runs together
- `teb_vae/lag_slot_transformer_cfs/tests/test_acceptance.py` - new
- `teb_vae/lag_slot_transformer_cfs/tests/test_eval_launch.py` - the new runner
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_acceptance.py -q`.
Expected: all pass. Status: executed.
Test rationale: every defect this file is written against leaves the arithmetic correct and the
statement wrong -- a plan edited after a result, one fit counted three times, two estimators
compared, a peak reported at the interval of a band chosen in advance. None of them is visible in
the output, and the integration fixture is the only place a seam between the scoring pass and the
protocol can show up at all.
Runtime: fast/local for the arithmetic; five fits and five evaluations for the seam. **Observed at
38 s** for the whole file.
Evidence: 2026-09-09. 32 tests pass, including three seeds of one arm fitted, scored and read as
three seeds. **The plan's digest is pinned by a test literal**, which is the mechanism the whole
predeclaration rests on: nothing stops the file being edited and the pin is what stops it being
edited quietly. **The probe artifact is attached to a run by the checkpoint both name**, not by the
directory it sits in -- pointing the probe pass at a finished evaluation directory would rename its
summary aside, so a directory convention would have failed the first time an operator followed it.
And the base-drift verdict is deliberately not the single-run gate's: with a paired interval
available it fails on a **confident** drift and reports an unclear one as unclear, where one summary
against another can only compare two points.

#### T042: The acceptance runs

Requirements: FR-032
Depends on: T041, T027, T033
Description: Fit each shortlisted arm at three training seeds, score every one through the same
entry point at the same draw count, probe each one's latent, read them all under the committed
predeclaration, then repoint a second evaluation delta at a reserved fold and confirm there.
Acceptance criteria:
- Three checkpoints per shortlisted arm, each with its resolved configuration beside it, each
  warm-started from the same target-only baseline.
- Every run scored at the plan's primary draw count, and the finalists rescored at the other two.
- The protocol's record names the committed plan's digest, and the plan is not edited after a rate
  has been seen.
- The confirmation runs' recordings are disjoint from every selection run's, which the protocol
  checks rather than the operator asserts.
Files affected: the run directories the fits write and the evaluation directories that score them.
No source file changes: everything this task needs is built, and running it is the task.
Validation: the runbook below. **Not startable from the machine this plan was written on**, for the
reason T027 is not: the integer-operator production shards are absent and the production
configuration describes seven devices against the one available. It additionally depends on T027 and
T033, since the arms it reads are theirs.
Test rationale: the sequence is covered by `tests/test_acceptance.py`, which fits four arms across
four training seeds, scores them through the real entry point and reads them through the real
protocol in under a minute. What no test can cover is whether an effect survives three seeds on real
recordings, which is a property of converged runs and of nothing else.
Runtime: days per arm, on the production box, plus one evaluation pass per run and per draw count.
Evidence: none.

### The runbook

From the repository root, after Sprint 5's runs 1 and 2 and Sprint 6's comparators exist.

```
# 1. Three seeds of each shortlisted arm. One file per arm; edit general_config.seed and
#    general_config.tag between runs and change nothing else.
python -m teb_vae.lag_slot_transformer_cfs.trainer --config teb_vae/lag_slot_transformer_cfs/configs/joint.yaml

# 2. Score every run into ONE root, at the plan's primary draw count.
python -m teb_vae.lag_slot_transformer_cfs.eval.run --checkpoint <run>/model_checkpoints/<f>.ckpt \
    --num-mc-samples 32 --output-dir output/acceptance/<arm>-<seed>

# 3. Probe each one. Anywhere under the same root: the protocol attaches it by checkpoint.
python -m teb_vae.lag_slot_transformer_cfs.eval.latent_probes --checkpoint <the same>.ckpt \
    --output-dir output/acceptance/probe-<arm>-<seed>

# 4. Read them together, against the frozen reference Sprint 5 produced as run 2.
python -m teb_vae.lag_slot_transformer_cfs.eval.acceptance --runs output/acceptance \
    --reference output/<run-2-eval>/eval_results/summary.json --output acceptance.json
```

Then, and only then, the confirmation: repoint a copy of the evaluation delta at a fold whose test
partition step 2 never opened, score the shortlisted arm's three seeds and the frozen reference
through it into a second root, and read both roots together.

```
python -m teb_vae.lag_slot_transformer_cfs.eval.acceptance --runs output/acceptance \
    --reference output/<run-2-eval>/eval_results/summary.json \
    --confirmation output/acceptance-reserved \
    --confirmation-reference output/<run-2-reserved-eval>/eval_results/summary.json \
    --output acceptance-confirmed.json
```

**Three things must not move between the two.** The predeclaration, whose digest every record names;
the draw count, because two draw counts are two estimators; and the frozen reference, which has to
be the same fit scored on each partition rather than two fits. Changing any of them makes the
confirmation a second development result.

**And one thing the protocol cannot check for you.** It verifies that the confirmation recordings
are disjoint from the selection recordings it was given. It cannot know about a run you did not
hand it, so a fold scored once during development and forgotten is a fold the protocol will report
as reserved. The evaluation directories are the ledger; keep them.

---

## Sprint 9: The evaluation revised

Goal: read one run more finely and more honestly than Sprint 4 built it to be read -- every margin
with the interval a claim about it rests on, every arm on the horizon and block axes, the lag axis
at every lag, and every number drawn as a figure a reader can put in front of a reviewer -- without
changing what a run scores or how it is gated.

Demo: score a checkpoint and open `eval_results/figures/`: the arms and their paired margins, the
gap by horizon step with the band margins beneath it, the lag profile with the declared bands shaded
and the single-lag margins on the segments the cap admitted, the calibration of both branches. Then
run the acceptance pass with `--report` and `--figures` and read the comparisons as a document.

Definition of Done: T043 to T046 done with evidence; the package's fast suite and its slow
evaluation suites pass; the four shipped forecasters remain untouched.

Dependencies: Sprint 4 for the pass, Sprint 8 for the protocol the report and the figures view.

### Tasks

#### T043: Paired margins and the resolved axes

Requirements: FR-032, NFR-005
Depends on: T020, T039
Description: Give every band and control margin a paired interval of the per-recording differences,
and resolve every scored arm by horizon step and by stored target block under the shared draws.
The predictive scorer gains `subset_block_scores` and a `resolve` argument; the pass aggregates
per-recording curves beside its scalar columns, bootstraps them under one resampling for every
position, and writes `horizon_resolved` and `block_resolved` blocks and the `horizon_resolved.csv`
table.
Acceptance criteria:
- Per draw, the horizon steps and the two blocks each sum back to the block score exactly.
- The marginalised steps do not sum to the joint marginal, and the summary says so.
- Every band and control margin carries `margin_interval` with `lo <= point <= hi`, paired over
  exactly the scored recordings; the identity arms' intervals have no width.
- The bands are listed in declaration order.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/eval/predictive.py` - subset scores, resolved branches
- `teb_vae/lag_slot_transformer_cfs/eval/lag_metrics.py` - `paired_margin`, `bootstrap_curve`
- `teb_vae/lag_slot_transformer_cfs/eval/run.py` - curves, the resolved blocks, the table
- `teb_vae/lag_slot_transformer_cfs/tests/test_eval_binding.py`, `test_eval_lag_metrics.py`,
  `test_eval_smoke.py` - extend
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_eval_binding.py teb_vae/lag_slot_transformer_cfs/tests/test_eval_lag_metrics.py -q`,
then the smoke file under `-m slow`. Expected: all pass.
Test rationale: additivity per draw is what pins the subset scores to the block score; the paired
interval on the identity arms having no width is what pins the pairing to the same recordings.
Evidence: 2026-09-13. 51 fast tests and the smoke file pass; the smoke run at the fixture geometry
completes in about 30 s with every arm resolved.

#### T044: The per-lag profiles

Requirements: FR-032
Depends on: T019, T043
Description: Read the lag axis at every candidate lag. A latent profile over the whole split --
proposal norm, update shift and signed divergence drop per lag, averaged over the scored anchors the
lag was live at, vectorised in lag chunks from the cached proposals -- and a predictive profile on
the first `caps.lag_profile` segments, where one suppression arm per lag joins the draw loop under
the same draws as every other arm and reaches the summary as one paired curve. Both under
`lag_readouts.lag_profile`, beside `lag_axis`, and in `lag_profile.csv`. The profile exists on the
local fusion alone and the skip is recorded by name elsewhere.
Acceptance criteria:
- The vectorised profile agrees with removing one lag at a time through `suppressed_parameters`.
- The scale channel is absent on the mean-only arm rather than zero-filled.
- A lag no scored anchor was live at is recorded as missing.
- The predictive profile reports its cap, segment and recording counts, and its status is
  `NOT_REQUESTED` without the cap and `SKIPPED` with the reason on a normalised fusion or a
  target-only checkpoint.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/eval/lag_metrics.py` - `per_lag_latent_totals`,
  `lag_profile_summary`
- `teb_vae/lag_slot_transformer_cfs/eval/run.py` - the single-lag arms, the profile block, the cap
- `teb_vae/lag_slot_transformer_cfs/eval/configs/eval_overrides.yaml` - `caps.lag_profile`
- `teb_vae/lag_slot_transformer_cfs/tests/test_eval_lag_metrics.py`, `test_eval_smoke.py`,
  `test_arms.py` - extend
Validation: as T043. Expected: all pass.
Test rationale: agreement with the one-lag-at-a-time path is the whole correctness of the
vectorised profile; the chunked path reorders nothing else.
Evidence: 2026-09-13. Agreement to $10^{-6}$ on a woken pathway at the fixture geometry; the smoke
run profiles three segments and reports a paired curve over their recordings.

#### T045: The figures and the tables

Requirements: NFR-005
Depends on: T043, T044
Description: Add `eval/figures.py`: seven builders that each take the parsed summary and return a
figure, a renderer that writes them into `figures/` under the run's format and records a manifest
in the summary, and the `FIGURE_GUIDE.md` that describes each. The pass calls the renderer inside a
failure-isolating guard after the tables are written.
Acceptance criteria:
- Every builder draws the candidate summary, a target-only summary and an empty one.
- The written files are exactly the manifest's names, in the run's format.
- Every lag figure labels its axis as stored-coefficient time and carries the qualification.
- A band with nothing to remove is drawn as not measured rather than at zero.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/eval/figures.py` - new
- `teb_vae/lag_slot_transformer_cfs/eval/FIGURE_GUIDE.md` - new
- `teb_vae/lag_slot_transformer_cfs/eval/run.py` - the renderer call and the manifest
- `teb_vae/lag_slot_transformer_cfs/tests/test_eval_figures.py` - new
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_eval_figures.py -q`.
Expected: all pass.
Test rationale: the arms with the fewest blocks are the ones a comparison cannot do without, so
the builders are tested against those summaries rather than only against the full one.
Evidence: 2026-09-13. The figures were rendered from the smoke run and inspected; the band
declaration order was found reversed by the headline's alphabetical columns and fixed at the block.

#### T046: The acceptance report and figures

Requirements: FR-032
Depends on: T041, T045
Description: Give the acceptance pass `--report`, a markdown document rendered from the record, and
`--figures`, a directory the record's three figures are drawn into, both after the record is
assembled and both optional. The plotting stack is imported only under `--figures`.
Acceptance criteria:
- The document carries every number the record does, formatted from the record, and round-trips
  through JSON unchanged.
- The three figures draw the record as the protocol writes it and an empty record.
- The launch dict and the parser agree on the two new keys.
Files affected:
- `teb_vae/lag_slot_transformer_cfs/eval/acceptance.py` - the report, the two flags
- `teb_vae/lag_slot_transformer_cfs/eval/figures.py` - the three acceptance builders
- `teb_vae/lag_slot_transformer_cfs/tests/test_eval_figures.py`, `test_eval_launch.py`
Validation: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/test_eval_figures.py teb_vae/lag_slot_transformer_cfs/tests/test_eval_launch.py teb_vae/lag_slot_transformer_cfs/tests/test_acceptance.py -q -m "not slow"`.
Expected: all pass.
Test rationale: a document that recomputed a number could disagree with the record it stands
beside; reading it from the record is what the round-trip assertion pins.
Evidence: 2026-09-13. 55 fast acceptance and figure tests pass.

---

## 8. Validation strategy and Definition of Done

### 8.1 What is reused

The repository's existing coverage is reused wherever it already proves the behaviour. The warm-up
budget resolution, the anchor tiling arithmetic, the forecast and divergence masks, the target
gather, the shared decoder and the prior head are all covered by the shipped packages' suites and
are not retested here. New tests exist only for the new computation and for the seams where the new
package meets the old.

### 8.2 What is new, and why each is necessary

| File | Uncovered failure it detects |
| --- | --- |
| `test_pointwise_source.py` | A negative lag index wrapping to the end of the record; a warm-up condition conflated with an index condition |
| `test_lag_updates.py` | The generic initialisation pass refilling a zeroed projection; a proposal reading more than one stored source time |
| `test_residual_kl.py` | The simplified divergence diverging from the general formula; a second application of the log-variance bound; a per-lag allocation being reintroduced |
| `test_construct.py` | An attention module being constructed; a parameter unreachable on some batch |
| `test_forward_contract.py` | A fabricated attention-shaped key; an anchor-indexed tensor exported as if its second axis were time |
| `test_causality.py` | A future input changing an earlier anchor's forecast |
| `test_invariants.py` | Source-off inequality; the observed-zero case being forced to the prior; a permanently inert source path |
| `test_objective.py` | The average of per-rank means standing in for the global mean; a clamped denominator read as an observation; `compute_loss` resolving past this package into the shared objective |
| `test_task.py` | The resolution order picking a wrong member; the spike-breaker key being renamed |
| `test_config_load.py` | A config key that names no constructor parameter |
| `test_docs.py`, `test_nets_are_framework_free.py` | A reintroduced timeline-compensation term; a hard-coded geometry literal; a training-framework import reaching the network modules |
| `test_chunking.py` | A detached chunk; an unmeasured summation-order tolerance |
| `test_trainer.py`, `test_checkpoint_contract.py` | A broken Run-button launch; a silently half-initialised warm start |
| `test_ddp_reachability.py`, `test_train_smoke.py` | An unused parameter under a distributed run; a fit that does not complete |
| `test_controls.py` | An intervention that also moves the availability clock; an unpaired latent draw; a suppression whose empty and full bands are not exactly the arms they must reproduce |
| `test_eval_lag_metrics.py` | An uninformative band being reported as an absent effect; the qualification text being dropped; a readout averaged per batch rather than over the split |
| `test_eval_binding.py` | A geometry key silently skipped by the reconciler; a marginalised score that is secretly a mean of per-draw log scores; a calibration built from a mean of conditional standard deviations |
| `test_eval_launch.py` | A runner that forgot the launch convention |
| `test_eval_smoke.py` | The pipeline not producing the numbers the design exists to produce |
| `test_arms.py` | A target-only checkpoint that quietly carries a source pathway; a warm start that copies nothing; a candidate whose source does not begin at zero; a reference that cannot be scored through the candidate's estimator |
| `test_mechanism_arms.py` | An arm that changes a forward branch rather than a module tree; a comparator whose source-off identity holds only at its zero initialisation; a capacity control whose budget is not the candidate's; a band margin differenced across two aggregations |
| `test_arm_scoring.py` | A comparator that fits and cannot be scored; an intervention path that assumes the recommended arm's cached per-lag updates; a control reported as a margin of zero on an arm where it was never an intervention; a summary that cannot say which arm produced it |
| `test_instruments.py` | A generator whose planted dependence never reached its tensors, so a low power figure reads as a property of the readout; a band derived at one geometry and applied at another; a lag partition that loses a lag in its remainder; a criterion read from the run rather than the declaration; a rate whose denominator drops the runs that found nothing; a scored split that was also fitted |
| `test_latent_probes.py` | A probe fitted on the recordings it is scored on; a streaming solve that is not the fit it stands in for; a null that hands every probe the difference between two cohorts; a coefficient clamped at zero, hiding the one outcome that says the latent carries nothing |
| `test_acceptance.py` | A predeclaration edited after a result, which changes no number; three scorings of one checkpoint counted as three training seeds; two arms compared across two estimators or two cohorts; a searched band reported at the interval of a band chosen in advance; a confirmation partition the selection has already been scored on |

Deliberately absent: no test asserts a positive divergence at initialisation, where zero is
intentional; no test enumerates every lag, channel or seed; no new testing framework is introduced;
and no full-package run is part of any task's validation.

### 8.3 Runtime classes and cadence

Every task's own validation is fast and local except a handful. T014's memory measurement is
integration-class and runs once per sprint on the target device. T016's tiny fit and T022's
evaluation smoke are integration-class and run at their sprint boundaries, as are T040's single
fit and T041's four, whose file is the most expensive in the suite at 38 s. Whole-package and
cross-package runs are not part of any task; per `CLAUDE.md` they are reserved for changes in modules
several packages import, which in this plan is T017 alone, and T017's evidence is the two existing
binding test files rather than a package sweep.

Estimated durations are labelled as estimates throughout and none has been observed. Any command
approaching the tool timeout is handed to the operator with a `!` prefix rather than backgrounded,
because detached long runs are killed on this machine.

### 8.4 Definition of Done for the delivery

- Every requirement in sections 2.1 to 2.3 has a done task with recorded evidence. **Met**, with
  one qualification recorded in T017: the shared collection pass is not reusable by this
  architecture at all, so FR-024 is satisfied by the registry exclusion and by this package's own
  honest readouts rather than by narrowing a pass it can run.
- The six structural gates of design section 11.3 are recorded as passing in
  `teb_vae/lag_slot_transformer_cfs/eval/EVAL.md` with a pointer to the test proving each.
- The tiny fit completes and its checkpoint reloads into an identical model.
- The evaluation smoke produces a `summary.json` with the matched gap, the proposal readouts and the
  three control margins.
- The two shipped CFS cells' evaluation binding tests pass unchanged.
- Peak memory, the FP32 execution tolerances and the pre-clip gradient-norm distribution are measured
  numbers recorded in this document, not estimates.

Empirical acceptance is explicitly **not** part of this. Design section 11.3 requires stable matched
held-out improvement over the internal base **and** a competitive independent target-only predictor,
without base degradation explaining the result; that is sprint 8, whose machinery is built and whose
runs are the operator's. What sprint 8 adds to the list above is one more artifact and one more
gate: every run says which recordings it scored, and a record produced under the committed
predeclaration names that declaration's digest.

---

## 9. Review and change record

### 9.1 Review, 2026-09-09

Four perspectives applied sequentially by the author. Material findings and their disposition:

| Finding | Severity | Evidence | Disposition |
| --- | --- | --- | --- |
| The design says to override or refactor the inherited forward without choosing; the choice changes the whole package layout | blocker | `CausalWarmupInputs.forward` calls `self.lag_attn` and `self.posterior_head` directly | Accepted. Resolved as a user decision; section 4.2 records the resolution order and T004 and T005 implement it |
| Three shared objective reductions clamp a local denominator, contradicting two requirements | blocker | `masked_raw_likelihood`, `masked_source_kl` and `masked_prior_rate` each call `clamp_min(1.0)` on a per-rank sum | Accepted. Resolved as a user decision; the package owns its reduction, T010 |
| `merged_analysis_functions` can only add analyses, so a reduced registry is not expressible | major | `teb_vae/lag_attn_cfs/eval/run.py` | Accepted. One additive optional field, T017, with the existing bindings' tests as the regression evidence |
| `evaluate_batch` reads three attention keys unconditionally, so the collection pass cannot run on this model at all | major | `teb_vae/lag_attn_cfs/eval/metrics.py`, one contiguous region | Accepted, and it upgrades T017 from a one-line change to a four-file one. Recorded in section 3.2 rather than discovered during Sprint 4 |
| Mean-only and scalar-lift were initially placed in a later sprint | major | Design sections 4.4 and 12.3 require the mean-only model to construct no scale head | Accepted. Moved into T002 and T003: retrofitting would change the module tree and therefore the checkpoint key set |
| The plan initially carried no memory measurement, only the design's arithmetic | major | Design section 4.8 states that removing attention does not automatically make the model cheaper | Accepted. T014 measures it before the smoke train, and NFR-001 states the workload and the method |
| An early draft proposed one test file per module | minor | Skill guidance on necessary tests only | Accepted. `test_invariants.py` carries T007, T008 and T009 together; `test_construct.py` carries T004 and T005 |
| A reviewer suggested asserting the exact FP32 tolerance for chunked-versus-unchunked agreement up front | minor | Design section 12.3 makes summation order chunk-size dependent | Rejected as stated, accepted in substance: the tolerance is measured in T014 and then written into the test, rather than being predicted here |
| An un-overridden `compute_loss` resolves past this package's objective into the shared one, training a working model against the per-rank clamped reduction with nothing raising | blocker | `CausalFeatureForecastTarget.compute_loss` calls `super()`, and `FeatureForecastTarget.compute_loss` calls the shared objective directly rather than through another `super()` | Accepted. T010 gained the routing paragraph, the `compute_loss` override in its files, and an acceptance criterion that patches the shared objective to raise |
| Sprint 1's Definition of Done cited the framework-free convention, which no task created | minor | `teb_vae/lag_attn_transformer_cfs/tests/test_nets_are_framework_free.py` exists for the shipped cell | Accepted. Added to T013 |
| Validator warnings about angle-bracket text | minor | Lines quoting `n_anchors = <mask>.sum()` and the path pattern `teb_vae/<pkg>/tests/test_<module>.py`, this row itself since Sprint 4, and the instrument runbook's `--only <names>` since Sprint 7 | Rejected: all are quoted code and path patterns, not authoring placeholders. Recorded here so a later reader does not re-investigate |

### 9.10 Changes during Sprint 9, 2026-09-13

| Change | Reason | Affected work |
| --- | --- | --- |
| Every margin carries a paired interval of the per-recording differences | The band and control margins were differences of two point estimates with each arm's own interval beside them. Two arms scored on the same recordings under the same draws differ per recording, and two overlapping arm intervals are wider by exactly the shared variation the pairing removes; the acceptance protocol already paired its comparisons this way and a single run now reads the same quantity | T043, FR-032 |
| Every scored arm is resolved by horizon step and by stored target block | The lag axis is censored by the alignment geometry and the horizon axis is not: a band that informs the first predicted step and not the last is a statement the window carries whatever the lag axis resolves. Each position is the marginal mixture of its own likelihood factors under the shared draws, so the positions do not sum to the block score and the summary says so; the curves are bootstrapped under one resampling for every position so an interval can be followed across steps | T043 |
| The lag axis is read at every lag: a latent profile over the whole split and a predictive profile over a capped subset | Four declared bands are a coarse partition of a window the design asks to read finely, and the design's own order is bands first, single lags after. The latent profile costs one pass of cheap arithmetic over the cached proposals and is vectorised against the one-lag-at-a-time path the band arms use; the predictive profile is one decoder call per lag per draw, which is why it is capped by `caps.lag_profile` rather than taken over the split | T044 |
| The per-lag latent readouts are the proposal norm, the update shift and the signed divergence drop, and none is called an allocation | Design section 5.2 proves no per-lag allocation exists. The shift separates a proposal the limiter has already saturated away from one that moves the update, and the drop is signed because removing a lag can raise the divergence when its proposal was cancelling another's; the qualification travels on every artifact and figure | T044 |
| The pass writes figures and two further tables, all from the assembled summary | A summary with no figure is read by nobody, and a figure drawn from tensors can disagree with the number beside it. Every builder takes the parsed summary, so the set is redrawable from a finished directory and testable against a hand-written one; a figure that fails is recorded and the summary is written regardless | T045 |
| The acceptance pass writes a markdown report and three figures on request | The record is the artifact and both are views of it, built after it is assembled so neither can disagree with it. The figures import the plotting stack only when asked for, so the pass stays stdlib-plus-`numpy` without them | T046 |
| The band suppression block is ordered as the delta declared its bands | The headline holds its columns alphabetically, and a block read from it listed `far` before `near`; the figures took their colours from that order. The exposure carries the declaration order, and the block now follows it | T045 |

### 9.9 Changes during Sprint 8, 2026-09-09

| Change | Reason | Affected work |
| --- | --- | --- |
| The sprint's opening question is answered, and half of it needed no new machinery | Fold 1's test partition is where the September 8 diagnosis measured, so it is the partition this architecture was chosen against, and the shipped cell's own per-recording exports are indexed by recording -- the cohort every choice so far was made on is enumerable from the artifacts that exist. What did not exist was any record of the recordings a run of THIS package scored | T039, Sprint 8's uncertainty |
| A per-recording table is written beside every summary | An interval cannot be taken apart into the vector that produced it, and two things need that vector: a bootstrap over several seeds resamples the recordings once and averages the seeds inside each resample, and disjointness between two partitions is a statement about recordings | T039, FR-032 |
| Disjointness is decided on recordings and never on paths | A cross-validation fold's train partition contains another fold's test recordings, and the ten folds of this build are drawn from one pool. Two evaluation directories can name different files and score the same subjects | T041, FR-032 |
| The split label is the common parent of the scored files rather than a parsed fold name | A parse encodes one dataset layout into a gate that has to keep working when the next build names its directories differently, and the common parent separates two partitions exactly as well | T039 |
| The latent probes are a new pass rather than a block of the scoring one | Design section 11.3 requires informative latent probes and no earlier sprint built any. They read the latent alone and score no predictive density, so folding them into the scoring pass would put two questions on one code path -- and the probe needs a recording split of its own, which the scoring pass has no use for | T040, FR-032 |
| The probe target is the block relative to the anchor's own stored values, not the decoder's learned persistence | A baseline that moved with the model being probed would make two arms' coefficients incomparable, and comparability across arms is the only use the pass has. The null is the fitting split's own mean, which is what a probe that had seen only those recordings would predict | T040 |
| The probe fit is streamed as second moments | It uses every complete-coverage anchor rather than a subsample, at a memory cost that does not grow with the split. The alternative would have put a sampling policy between the split and the number. The suite checks the expansion against the same ridge fitted from the rows, because a lost cross term would produce a plausible coefficient on every arm | T040 |
| The predeclaration is a file whose digest every record names | Nothing prevents a plan being edited; the digest is what stops it being edited quietly, since a comparison added or a seed minimum lowered after a result changes no number in any record. The suite pins the digest, so the pin fails on a revision and the diff is then the record that one happened | T041, FR-032 |
| Multiplicity is applied to the band search and not to the primary comparisons | The five comparisons are declared before any arm was trained; the peak band is chosen on the data its interval is built from. A family-adjusted interval covers every band at once, and a member selected out of a simultaneously covered family keeps that coverage however it was selected. The two identity suppression arms are excluded from the family: neither is a lag the search ranges over | T041, FR-032 |
| Three scorings of one checkpoint are one training seed | A draw-count sweep of one fit would otherwise satisfy a minimum that exists to ask whether an effect survives repeated optimisation | T041 |
| A probe artifact is attached to its run by checkpoint rather than by directory | MEASURED against the writer: pointing the probe pass at a finished evaluation directory renames that directory's summary aside, so the obvious convention would have failed the first time an operator followed it and the arm would have reported an unprobed latent | T041 |
| The protocol's base-drift verdict is not the single-run gate's | That gate compares two point estimates from two runs and fails on the sign alone, which is the only comparison one summary supports. With a paired interval available the protocol fails on a confident drift and reports an unclear one as unclear | T041, T025 |
| The fixture's evaluation delta became a public helper | Both integration files need the same one, and a second copy would be a second definition of which shards, which draw count and which lag bands a fixture run uses -- free to drift in exactly the settings a comparison holds fixed | T040, T032 |

### 9.8 Changes during Sprint 7, 2026-09-09

| Change | Reason | Affected work |
| --- | --- | --- |
| Every task's status moved into one bottom checklist | The roadmap validator's contract changed between sprints: it now requires a single checklist and refuses a task-local status field. All thirty-eight tasks failed, not only the new ones, so the document followed the tool rather than the tool being worked around. No task's state changed, and the four tasks whose `Status:` began a line had it folded into their `Validation` prose | the Todo checklist, section 11 |
| The prior-art recovery check contributes its band arithmetic and none of its profiles | The open uncertainty, settled by reading it: it scores a per-lag divergence allocation, its support-corrected and clock-excess forms, and a per-head attention distribution, and this architecture computes none of them on any arm. What carries over is the identity turning a planted delay into a readable band. The readout here is proposal suppression over a declared lag partition, which works on both fusions | T034, T036, FR-031 |
| The synthetic generators write no shard | A stored-feature block synthesised without a filter bank has no real warm-up boundary, no group delay and no novelty curve, and a causal shard would carry all three fabricated -- which the warm-up resolver then reads. Every downstream number would be computed against a boundary nobody measured | T034, FR-031 |
| The filter-bank instrument runs the bank in memory rather than writing a fixture | Its coefficients are real, so its warm-up is real and is read off the bank's own channel plan; and a committed shard would pin the instrument to one operator and one delay. It also keeps the whole campaign on one code path | T035 |
| A scoring split was added after a measurement, not by design | MEASURED: the first arrangement scored the segments it had fitted, and all three controls became detections. A source branch has more capacity to memorise with than a target-only one, so an in-sample gap is a measurement of memorisation and is positive where the source carries nothing. Holding out a scoring split turned every control negative in one change | T037, FR-031 |
| A third, selection split followed it | A fit stopped at whichever step scored best on the set it is then reported on has had its stopping point chosen by the number it reports. The stopping criterion is the production one -- the matched full-branch block score -- on a split that has no part in either the fit or the verdict | T037 |
| The fit split is the largest of the three, and its size is a measurement | Traced over 3000 steps at a third of the shipped size, the held-out gap on a planted generator is +0.33 at step 500 and decays through zero to -0.18 by step 3000 as the decoder drives its observation variance down. The instrument then has no power for a reason about data volume rather than about the readout | T037 |
| The lag readout is a declared window partition rather than one arm per lag | Two lags closer together than the source encoder's own reach are summaries of overlapping windows, so a per-lag readout would report a resolution the representation does not have -- at a hundred extra forwards per batch at the production window. The width is recorded with every verdict, so a campaign that changed it changed its instrument | T036, FR-022 |
| The filter-bank instrument's band is reported and not graded | Its plant reaches the stored grid through an operator whose filters are far longer than one stored step, so a peak on the band would be the surprising result. The peak, the plant and the DISTANCE between them are all recorded; no pass or fail is | T035, T038 |
| The launch test walks two runner directories | The campaign is launched exactly as the evaluation passes are and would otherwise be checked by none of the convention's rules -- the same hole the written-out tuple exists to close, reached by landing in a directory the guard did not walk | T037, FR-017 |
| No test asserts that an instrument detects its plant | Power is a measurement. A suite that failed when it came out low would pressure the campaign toward a flattering number, which is the one thing an instrument must never be tuned for. The suite checks that the machinery is wired to the declared truth; the record carries the rates | T034, T038 |

### 9.7 Changes during Sprint 6, 2026-09-09

| Change | Reason | Affected work |
| --- | --- | --- |
| The attention comparator is a construction in this package, not a configuration arm of the shipped forecaster | The open decision, settled on evidence. A comparator must carry this design's prior-relative posterior equations, bounded scale range, explicit clock and shared-noise sampling, and the shipped model has NONE of them: it decodes its base at the prior mean, forms its clock by encoding a zeroed source through the source network, and bounds its posterior log-variance independently. A configuration arm there would change the shipped model, which section 1.4 forbids and NFR-006 measures, and would change four things at once | T028, FR-030 |
| The arms are two orthogonal constructor axes plus three flags, forming a chain | The two attention arms of design section 11.1 differ from each other only in how each lag's source vector is FORMED, so one fusion module over two source representations gives both, and every adjacent pair in the chain differs in one leaf. Written as five separate constructions it would be four near-copies free to drift | T028, T029, T031 |
| The attention arm does not export its distribution over lags | The weights are real there, unlike anything the recommended arm could offer under that name -- and published beside a predictive comparison they would be read as a lag readout by every reader and every downstream table, which is the claim the evidence behind this design shows cannot be supported. Both arms are interrogated by suppression through one interface instead, and the forbidden-key gate now runs against the one arm where it could fire | T028, FR-021, FR-024 |
| The attention update is gated on whether any lag survived | Without it, every selector off gives all-minus-infinity scores, a NaN softmax sanitised to zero, and then the output projection's BIAS -- so the update would be a learned constant rather than zero on trained weights. The source-off identity every margin is read against would hold at initialisation and nowhere else | T028, FR-007 |
| Band suppression takes a second path on an attention arm, and the two margins are not comparable | A normalised aggregation has no per-lag term to subtract: removing a lag removes it from the denominator too and the survivors grow. The subtractive control refuses on that arm rather than returning a number with no interpretation, and the run records which mechanism produced every margin | T028, T031, FR-021 |
| The convolution stem carries a parameter-free pointwise encoder for its availability | Whether a channel has warmed up is a property of the channel and the stored step and does not stop being one because an encoder mixed the values. Resolved twice, two arms would disagree about which lags exist -- and an exposure table that changed with the representation would not be an exposure table | T029, FR-022 |
| The lag index arithmetic is extracted and shared by both gathers | Two copies is how a per-channel gather and a per-state gather come to disagree about which lags exist, and the disagreement is a wrong number rather than a failure: both shapes stay correct either way | T029, FR-002 |
| The capacity control withholds values at the ENCODER rather than at an intervention | The question is what a head of this capacity LEARNS when no value ever reaches it, which only a trained arm can answer; an intervention asks a model trained with values a question it never saw. Withholding at the encoder also keeps the widths, the head, the optimizer state and the checkpoint key set identical to the candidate's, which is what "comparable trainable capacity" has to mean | T030, FR-030 |
| The capacity control skips the permutation control as well as the replacements | Not anticipated. The availability announcement is a function of stored position and the resolved warm-up alone, so pairing a recording with another changes nothing that arm reads. A margin of zero would state as a measurement something true of the arm by construction | T030, FR-023 |
| `source_disabled` joins the reconciled geometry keys, with the three new arm leaves | A correction rather than an addition: it already decided whether a reported gap was a measurement or zero by construction, and a checkpoint and a configuration were free to disagree about it | T031, FR-019 |
| Both cancellation channels became arm-dependent in the task's readouts | FOUND BY THE FIT: the task read the mean channel unconditionally and an attention arm emits neither, so every comparator run would have died on its first validation step. The tracked metric list stays the union deliberately -- a tracked name a run never emits costs an empty column, a name a run emits and nothing tracks costs the number | T032, FR-017 |
| The shipped configuration set is discovered rather than written out | The opposite of the entry-point tuple's rule and for the opposite reason: a runner that forgot the launch convention must fail rather than go unseen, while an arm added to the directory and forgotten in a tuple would be an arm nothing checked -- and an arm is written by copying a sibling, which is exactly how a refused keyword arrives | T031, FR-019 |
| The memory pass holds the lag axis whole on an attention arm and says so | Chunking it would renormalise the distribution inside each chunk and measure a different model rather than the same one in a different order. Dropping half the grid without a record would leave a reader comparing two arms over different grids | T032, NFR-001 |
| The two fit and score helpers in `test_arms.py` became public | A private copy in the second integration file would be a second definition of what fitting an arm means, free to drift in the epoch count, the profiler setting or how the checkpoint is found | T032 |

### 9.6 Changes during Sprint 5, 2026-09-09

| Change | Reason | Affected work |
| --- | --- | --- |
| The production configuration's chunking guidance is reversed | MEASURED: the training peak is flat across the whole chunk grid to two parts in a thousand, while the dense evaluation peak falls by 48%. Chunking changes the order the proposals are computed in and not whether their activations are retained, so under backward every chunk stays in the graph. The old comment sent an operator to the chunk sizes before the batch size, which would have cost hours and achieved nothing | T023, NFR-001 |
| The reassociation measurement pins the latent draw and wakes the source pathway | Two defects that each produce a plausible wrong number. The forward SAMPLES, so two runs at different draws differ by four orders of magnitude more than any summation order; and at the zero-initialised source projection every proposal is zero, so a sum of zeros has no order and the pass would report a tolerance no trained run could meet | T023, NFR-002 |
| `source_disabled` builds no source pathway rather than idling one | The same argument the mean-only arm rests on: modules built and never used are starved parameters under a distributed run and a claim in the manifest that the model reads a source it does not. It also makes the checkpoint's key set exactly the transferable half, which is what the warm start needs | T024, FR-029 |
| `clock_proj.` moved from the re-zeroed source prefixes to the transferable ones | The metadata clock is a function of stored position and conditions the PRIOR, so it is no more part of the source pathway than the target encoder is -- and the target-only arm of this same class now trains it. Re-zeroing it would discard a trained target-only component and start the candidate's prior from a state its own baseline never occupied, which is the baseline weakness the warm start exists to avoid | T024, FR-018 |
| The scoring pass runs against a source-free checkpoint | The frozen reference has to be scored through the SAME estimator, anchors, mask and draws as the candidate, or the comparison is between two scoring paths rather than two models. Every intervention is skipped by name with its reason, and the exposure is empty rather than counts of zero | T025, FR-029 |
| The acceptance gate gained an optional reference summary | A candidate's own base branch trains jointly with its source pathway and can degrade, so an improved internal gap can be measuring the degradation. The margin against the reference is reported and not gated; the base DRIFT is gated, because a base that has fallen behind an independently trained one is not a threshold question | T025, FR-029 |
| The production config documents both checkpoint keys | Its `core_model_checkpoint` comment described the OTHER key's job -- it is a strict load of this exact model kind, source pathway included, rather than a target-only warm start | T026, FR-018 |
| The memory pass joined the launch tuple, which refused it first | Exactly the guard working: a runner that lands without joining the tuple is checked by none of the convention's tests. The tuple's required-argument map is now parametrised separately, because an entry point with no required argument is a legitimate shape and this one has none | T023, FR-017 |

### 9.5 Changes during Sprint 4, 2026-09-09

| Change | Reason | Affected work |
| --- | --- | --- |
| The `evaluate_batch` and `collect.py` guards were **not** made | `BatchReadout` declares eight attention-derived fields as required, and the pass pairs a dense stored-step support with a latent produced at every step, while this architecture's latents are anchor-indexed. Guarding three key reads reconciles neither, and satisfying the readout means fabricating tensors the design forbids. A guard no caller can use, in a module two shipped forecasters run, is regression risk for no benefit | T017, FR-024, NFR-006 |
| The interventions return latent parameters rather than scores | Every arm is then handed to one draw loop together, so the shared-noise requirement is structural instead of a convention each call site must honour, and two arms with identical parameters give a margin of exactly zero | T018, FR-021, FR-023 |
| Only suppression recomputes from cached proposals | A proposal is a function of the source values it was given, so no cached set answers a question about different ones. The replacement and permutation arms re-run the forward under a substituted stream with every other argument matched, and the prior returning bitwise unchanged is asserted in place of the caching | T018 |
| Removing every lag is written as a sum over no lags, not as a subtraction | The subtractive form is what makes an empty band bitwise identical to the matched arm; used for the mirror case it leaves a residue in the last places, and the "source removed" arm's divergence is then a small positive number rather than zero. Both endpoints are exact and both are tested | T018, T022 |
| `mask_only` is not scored as a third replacement arm | On the recommended encoder the value coordinate is the coefficient, so a zeroed stream already leaves exactly the mask and the arm is the zeros arm under a second name. It refuses under the scalar lift, where the lift of a zero is a learned constant, rather than silently reporting the zeros arm | T018, FR-023 |
| The binding's exclusions split into two removals and five absences | Only `attention` and `lag_kl` are in the shared registry; the other five are the lag-attentive cell's own extras, which this binding never registers. Naming all seven refused at once, which is the guard working. All seven still reach the summary with their reasons and with which mechanism left each one out | T020, FR-024 |
| `eval/predictive.py` exists, one module beyond the planned list | `binding.py` declares what the pipeline cannot derive about the model; a draw loop, a concentration diagnostic and a mixture calibration census in the same file would give one module two responsibilities | T020, FR-020, FR-025 |
| `mc_predictive_block` is not reused; `marginalise_block_scores` and `masked_raw_block_per_anchor` are | Its anchor gather is a dense-latent assumption this architecture does not have. Passing it an identity index would work and would leave a load-bearing line reading as though a gather had happened; the two pure functions carry no architecture at all | T020, FR-020 |
| Every lag and calibration readout accumulates as sums and is finished once | Reduced per batch, the pass would report a mean of per-batch means, weighting a batch of one segment equally with a full one; and a variance is not recoverable from means, which is why the squared sum travels | T019, T020 |
| The suppression bands are read from `eval_config.occlusion_bands` | The schema's key set is shared and closed. The partition is deliberately the lag-attentive cells' -- a reader comparing two arms compares the same intervals -- and the intervention is not, which the delta and the contract both state | T019, T020 |
| Each entry point factors a `build_parser()` | It is the sibling suites' convention and is what lets the four launch rules be checked mechanically rather than by reading the file. Both `prog` strings name this package, so a usage line points at the module the operator launched | T021, FR-017 |
| The tiny configuration's likelihood moved from a squared error to the Gaussian score | Sprint 3's record already claimed this and the file still said otherwise. It is load-bearing twice: under a squared error the observation log-variance head is outside the graph on every batch, and the evaluation's predictive density and calibration census both read that head, so the smoke fit's checkpoint was not scorable | T012, T016, T022 |
| The evaluation override delta joins the documentation gate's geometry allowlist | It is a configuration file and the lag bands it declares are the geometry of the readout, which is the same ground the two model configurations stand on | T013, T019 |

### 9.3 Changes during Sprint 3, 2026-09-09

| Change | Reason | Affected work |
| --- | --- | --- |
| `compute_loss` accepts the three shape-term weights and refuses a nonzero one | Dropping them from the signature would let the driver forward the key in silence. Accepting and refusing makes a configuration that sets one fail at the first step | T010, FR-015 |
| The empty-support zero is connected through every term, not only the reconstruction | It originally left the observation log-variance head out of the graph on a batch that scored nothing, which is the distributed hazard the path exists to avoid | T010, NFR-003 |
| The task writes its own step and refuses the permutation control and the source-null readout | Both reach modules that were never built, and would have raised on the first validation step of a real run. The controls are rebuilt against the fusion in T018 | T011, T018 |
| The per-epoch diagnostic page is not enabled | Its lag rows draw an attention matrix and a divergence-by-lag map this architecture computes neither of. Enabled, it raises inside a handler that warns and continues, so the figure would never appear and the suite would stay green | T011, T012 |
| The driver tracks the gradient-norm, clip-fraction and spike columns | The first smoke run logged them and then dropped them, so the numbers the inherited clip must be re-derived from reached no artifact | T015, T016 |
| The warm-start stripper handles the `_orig_model.` prefix a task checkpoint carries | The original rule matched nothing, so the transfer would have refused every real checkpoint with a message pointing at the file rather than the prefix | T015, FR-018 |
| The tiny configuration uses the Gaussian score rather than a squared error | Under a squared error the observation log-variance head is outside the graph on every batch, so the smoke run would not have exercised the objective the design specifies | T012, T016 |
| The test builder seeds the global generator before constructing | Two constructions in one process draw different weights, so every comparison that builds a model twice passed or failed on whatever random state preceded it. The Sprint 2 chunking comparison was flaky for exactly this reason and is now a real measurement | T006, T014 |
| The chunking comparison moved from the forward-contract file into its own | The agreement checks, the measured tolerance and the detachment check belong together, and a second copy would be the same comparison at two tolerances | T006, T014 |

### 9.4 Changes during Sprint 2, 2026-09-09

| Change | Reason | Affected work |
| --- | --- | --- |
| The metadata clock's projection lives on `LagResidualCore`, and the prior head is built with no clock path of its own | The conditioning state is read by the proposal head as well as the prior, and a projection inside the head would have to be applied twice. Section 4.2 named the clock as an override; this makes it a module | T004, FR-006 |
| The refused keywords are parameters of the model signature, defaulting to `None` | Section 6.2 required a `ValueError` naming the key. The driver forwards a configuration key only when the signature names it, so an omitted keyword is dropped in silence and the refusal could never fire | T005, FR-019, T012 |
| The proposal head takes an optional lag-slot index, and the source gather an optional lag offset | Lag chunking must identify slot $\ell$ as $\ell$ rather than as its position within a chunk, and the point of chunking is not to build the whole window and slice it | T002, T006, FR-010 |
| The chunked forward reports the cancellation ratio through a shared formula both paths call | Accumulating the two parts separately would be a second definition of one diagnostic, free to pick a different epsilon or norm | T003, T019, FR-022 |
| The divergence keys are named for the anchor axis they carry | The family's `kld_per_t` on an anchor-indexed tensor is precisely the misreading design section 4.8 warns about | T006, FR-009, T010 |
| Anchor and lag chunking landed in T006 rather than T014 | The forward had to walk the two axes anyway; adding the reduction later would have meant writing the loop twice. T014 keeps the measurement, which needs the real geometry and a device | T006, T014, NFR-001 |

### 9.2 Structural validation

`node C:\Users\mahdi\.claude\skills\spec-and-sprints\scripts\validate-roadmap.mjs teb_vae/lag_slot_transformer_cfs/SPEC_AND_SPRINTS.md`
- Re-executed at the end of Sprint 7, 2026-09-09. Result: PASS, 38 requirements, 38 tasks, 1
  forecast requirement, plus the four angle-bracket warnings dispositioned above. Structural checks
  only; the semantic review in section 9.1 is separate and was performed sequentially by the author,
  not independently.
- **The validator's own contract moved between Sprint 6 and Sprint 7**, and the document moved with
  it. It now requires one bottom checklist and refuses a task-local status field, where it
  previously read each task's `Progress` line. Every task in the document failed, not only the new
  ones, so this was a tool change rather than a defect introduced here. The thirty-eight statuses
  were moved into section 12 mechanically and the four tasks whose `Status:` began a line had it
  folded into their `Validation` prose; no task's state changed.

---

## 10. Resume

Current increment: **Sprints 1 to 4 complete; Sprints 5, 6 and 8 machinery complete with their nine
training-run tasks outstanding; Sprint 7 complete, its rates measured**, 2026-09-09. **Every task
this repository can execute is done.** The three that remain are the same task in three sizes: fit
something on the production shards.

Next executable task: **T027, Sprint 5's three training runs**, and it is the operator's rather than
this repository's. Everything it depends on is built, tested and exercised end to end at fixture
scale; what it needs is the integer-operator production shards and the box the production
configuration describes. The runbook is in Sprint 5, four commands and three checks. **T033**, Sprint
6's four comparator runs, follows it and shares the boundary: every comparator warm-starts from run 1
and is read against run 2. **T042**, Sprint 8's seeds and its confirmation, follows both: it reads
whatever those produced, three seeds per shortlisted arm, under the committed predeclaration.

Unresolved decisions: **none.** Sprint 6's is settled -- the attention comparator is a construction
in this package, because the shipped forecaster carries none of the four things a comparator has to
hold fixed. Sprint 7's is settled too: of
`teb_vae/lag_attn_cfs/lag_recovery_check.py`, the band arithmetic survives and every profile it
reads does not, so the instruments score proposal suppression over a declared lag partition instead.
Sprint 8's is settled as well: the partition every choice so far was made on is fold 1's test
partition, which is where the September 8 diagnosis measured, and it is enumerable recording by
recording from that run's own per-recording export. A reserved partition is therefore a fold whose
test partition no run has been pointed at, and the protocol checks that on the recordings rather
than on the paths.

Latest evidence: `.venv/Scripts/python.exe -m pytest teb_vae/lag_slot_transformer_cfs/tests/ -q`,
**572 passed in 103s**, 2026-09-09, including the three-arm sequence at 17s, the three comparator
arms fitted and scored at 16s, the evaluation smoke at 13s, and the acceptance protocol's four fits
across four training seeds at 38s. The two shipped CFS cells' binding tests pass unchanged, 46 in
6s.

What exists now, for a reader picking this up cold: **a model that trains, a pass that scores it,
six arms that build, a measured production geometry, and a protocol that reads several runs
together.** Configuration to fit to checkpoint to `summary.json` to acceptance gate runs end to end
on the committed fixture, for the target-only arm, the source-conditioned candidate and every
comparator alike; each can be read against a frozen reference; and a directory of them can be read
against a predeclaration at several seeds, with the band search corrected and the reserved partition
checked. There are no trained arms and no empirical claim of any kind.

The six arms, and what each adjacent pair isolates: the attention reference against the
pointwise-attention arm isolates the source temporal convolution; that arm against the candidate
isolates attention versus the explicit sum; and the candidate against the mean-only, capacity-control
and target-only arms isolates the variance update, the source values and the pathway entire. Each is
one constructor leaf, so each is one declared change.

And **eleven instruments whose answer is known**, which is the one part of this work that produced
measurements rather than machinery. The readout claims nothing on any of the three controls, finds a
planted delay in four of seven informative generators, and puts its peak window on the declared band
wherever it finds one. The three misses are synergy, a multi-lag product the additive proposals are
stated not to represent, and a discrete-event source. The filter-bank instrument is the one to read
first: a delay planted at twelve stored steps in the RAW signals detects clearly and peaks four to
eight lags nearer the anchor than the plant, which is the content-lag spread measured end to end
instead of argued.

Measured in Sprint 5, and no longer owed:

- **Peak memory at the production geometry.** Training peaks at about $8145$ MiB per device at batch
  $128$ and scales nearly linearly; dense evaluation peaks at $4447$ MiB unchunked and $2309$ MiB at
  a quarter of each axis. Chunking does **nothing** for a training step, which is a measurement and
  contradicted the guidance the configuration shipped with.
- **The reassociation tolerance at the production widths.** The bounded update moves by
  $1.2\times10^{-5}$ and the decoder output by $2.4\times10^{-7}$, on a repeat-run floor of exactly
  zero. The update figure is thirteen times the fixture's, which is what summing $91$ terms instead
  of $5$ costs.

Still owed, and now the only measurement debt: the **pre-clip gradient distribution at the
production geometry**, which needs real data rather than a synthetic batch, so the inherited clip of
$3500$ and the additive spike margin of $2200$ stay starting guards until run 1 reports them.

Carried forward from Sprint 5, and still true:

- **The evaluation costs four extra forwards per batch on a source-conditioned arm and none on a
  target-only one.** At production scale the target half of each of those forwards is identical to
  the matched one, which is the first place to look if the pass is slow.
- **The gate has one verdict that can fail on a real result rather than on a defect**: a candidate
  whose base branch has fallen behind the frozen reference. Every comparator arm inherits it.

Discoveries from Sprint 6 that shape Sprints 7 and 8:

- **One frozen reference serves every arm, and that is a property of the arms.** All six carry the
  identical target encoder, prior, metadata clock and decoder, so their base branches are comparable
  with one independently trained target-only predictor. What must not be shared is the warm start:
  each arm starts from run 1, or a difference between two of them contains a difference of baselines.
- **A band-suppression margin does not compare across the two fusions.** Removing a lag from an
  explicit sum leaves the other terms standing; removing it from a normalised distribution grows
  them. Sprint 7's instruments are calibrated against an observed profile shape, so the shape they
  are calibrated against has to come from one fusion or be read per fusion.
- **The attention arm's distribution over lags is deliberately not exported**, so an instrument that
  expected to read one -- as `lag_recovery_check.py` does on the shipped cell -- has nothing to read
  here. Sprint 7 rebuilt the readout against proposal suppression, which works on both fusions.
- **A band narrower than the convolution stem's reach is not resolving what its name says.** Two lags
  closer together than that reach are summaries of overlapping windows, which is why Sprint 7's lag
  readout is a declared window partition rather than one arm per lag.

Discoveries from Sprint 7 that shape Sprint 8:

- **An in-sample gap is a measurement of memorisation, and it is positive where the source carries
  nothing.** Measured here: scoring the segments a model was fitted to turned all three controls into
  detections, and holding out a scoring split turned every one of them negative in a single change.
  Sprint 8's acceptance protocol rests on the same property at production scale.
- **A readout can point at the right lag while the gap does not clear zero.** The synergy instrument
  peaks exactly on its declared band and is not detected, so the two readouts fail independently and
  a shortlist should not be filtered on one of them alone.
- **The feature operator moves the apparent lag by four to eight stored steps, toward the anchor**,
  at this geometry and this operator. Any lag claim Sprint 8 confirms has to carry that number, and
  it is now a measurement rather than an estimate.
- **The instrument campaign is a fixed instrument.** Its effect sizes, criteria and fit budget must
  not move between two campaigns, and must never move after a rate has been seen. A generator needing
  a different effect size is a second generator reported beside the first.

Discoveries from Sprint 8, which shape how its runs must be produced rather than read:

- **The evidence a protocol can check is the evidence a run recorded.** Before this sprint no
  artifact of this package said which recordings it scored, so a reserved partition could be
  declared and not verified. Every summary now carries the files it opened and a digest of the
  recordings that came back, and the table beside it carries the recordings themselves.
- **Fold 1's test partition is spent.** The architecture was chosen against it. A confirmation there
  would confirm a choice with the evidence that produced it, and the ten folds of this build are
  drawn from one pool, so a second fold's train partition is not automatically clean either. The
  recordings decide, not the paths.
- **A predeclaration is only a declaration while its digest is pinned.** The plan file is a repo
  file and can be edited; what the pin buys is that an edit cannot be quiet. Bumping it deliberately
  is one line and a revision counter; bumping it after a rate has been seen makes that rate a search
  result, exactly as Sprint 7's criteria do.
- **Three runs of one checkpoint are one seed.** Rescoring at three draw counts is Monte Carlo
  stability and is reported as that; it is not evidence that an effect survives repeated fits, and
  the protocol counts training seeds rather than runs for that reason.

Important discoveries to carry forward, all read at `6cfa8be`:
- The architecture parent builds six modules this design forbids, which is why the package needs its
  own base rather than a subclass.
- Three shared objective reductions clamp a local denominator, which is why the package owns its
  reduction.
- The shared collection pass is unavailable to this architecture, and no guard changes that: its
  readout requires eight attention-derived fields and it pairs a dense stored-step support with a
  latent produced at every step, while this architecture's latents are anchor-indexed.
- A checkpoint carries the net's keys under `_orig_model.`, and `strip_task_prefix` is the one rule
  for that.

On resume, recheck the baseline revision before editing: this plan was written against a dirty
working tree.

---

## 11. Implementation conventions

Sprint labels and this document's filename never appear in production identifiers, comments, module
docstrings or any user-facing string. Task identifiers may appear in commit messages and in this
document's change record, nowhere else.

Tests are colocated in `teb_vae/lag_slot_transformer_cfs/tests/`, following the repository's layout.
Add only the tests section 8.2 names; if a task's behaviour turns out to be covered by an existing
check, reuse it and record that instead of adding a file.

Docstrings are Google style with LaTeX for symbols and mathematics, inline `$ $` and display
`$$ $$`. Comments explain why, and carry no horizon, anchor, block or channel literal.

Checkpoint loading goes through `train/graph_models_utils.py`. Nothing is committed unless the
operator asks for a commit.

The bottom checklist is every task's authoritative status and each task's `Evidence` field is its
authoritative record. A task detail carries no status line of its own: two places to record whether
something is done is two places for them to disagree.

---

## Todo checklist

### Sprint 1: The latent-residual computation
- [x] T001: Package skeleton and the pointwise source encoder
- [x] T002: Lag embeddings, the shared proposal head, and the selector
- [x] T003: Fusion, the prior-relative residual, and the residual divergence

### Sprint 2: The model
- [x] T004: The architecture base
- [x] T005: The model class, the constructor schema, and the resolution order
- [x] T006: The anchored forward and the tensor contract
- [x] T007: Source-off equality and the absence semantics
- [x] T008: Shared-noise pairing and the decoder boundary
- [x] T009: Gradient escape from the zero start

### Sprint 3: A run that trains
- [x] T010: The objective and its global reduction
- [x] T011: The task
- [x] T012: The configurations
- [x] T013: Documentation gates
- [x] T014: Chunking, and the memory measurement
- [x] T015: The trainer, the Run-button convention, and warm-start transfer
- [x] T016: Distributed reachability and the tiny fit

### Sprint 4: A run that is scored
- [x] T017: The two shared evaluation seams
- [x] T018: The interventions
- [x] T019: Lag readouts
- [x] T020: The binding and the predictive scoring path
- [x] T021: The evaluation entry points
- [x] T022: Evaluation smoke, and the sprint's integrated acceptance

### Sprint 5: The first real arm
- [x] T023: Peak memory and the reassociation tolerance, at the production geometry
- [x] T024: The target-only arm
- [x] T025: Scoring a source-free checkpoint, and the external-reference verdict
- [x] T026: The arm configurations and the runbook
- [ ] T027: The three training runs

### Sprint 6: Mechanism-separating comparisons
- [x] T028: The attention fusion, and the arm axes it introduces
- [x] T029: The convolution source stem
- [x] T030: The capacity control
- [x] T031: The arm configurations and the summary that makes them a comparison
- [x] T032: The comparators through the real pipeline, and the per-arm budget measurement
- [ ] T033: The comparison itself

### Sprint 7: Synthetic instruments
- [x] T034: The generators and the truth each is declared with
- [x] T035: The instrument that runs through the real feature operator
- [x] T036: The declared criteria and the rates they aggregate to
- [x] T037: The campaign
- [x] T038: Run the campaign and record what it says

### Sprint 8: Acceptance
- [x] T039: The scored-recording provenance and the per-recording table
- [x] T040: The latent probes
- [x] T041: The predeclaration and the protocol
- [ ] T042: The acceptance runs

### Sprint 9: The evaluation revised
- [x] T043: Paired margins and the resolved axes
- [x] T044: The per-lag profiles
- [x] T045: The figures and the tables
- [x] T046: The acceptance report and figures
