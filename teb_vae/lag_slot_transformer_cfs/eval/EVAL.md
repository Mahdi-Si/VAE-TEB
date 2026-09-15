# Evaluating the lag-slot transformer

This guide explains how to evaluate a trained checkpoint, find its results, and decide what those results support. The model forecasts causal features: numerical summaries of the recorded signals that use only present and past information. Its main comparison is between a forecast using target history alone and a forecast that also uses source history.

Read sections 1–4 first for the workflow and the main scores. Sections 5–6 explain implementation limits and checks. Section 7 explains how to compare several trained models. For help reading individual plots, see [the figure guide](FIGURE_GUIDE.md).

## Terms used in this guide

| Term | Meaning |
| --- | --- |
| Recording | One complete recording, which may contain many segments. Statistical intervals resample recordings. |
| Segment | A fixed-length section of a recording supplied to the model. |
| Anchor | The time within a segment from which a forecast starts. |
| Horizon | The future steps predicted from one anchor. |
| Coefficient or feature | A numerical summary produced by the upstream signal transform. The model forecasts these features. |
| Forecast block | All target channels over the whole forecast horizon at one anchor. |
| Latent state | The model's compact, uncertain representation of the history. Its distribution has a mean and a scale. |
| Arm | A model variant or an evaluation condition. For example, an intervention arm removes selected source inputs from a fitted model. |
| Base and full | The target-only branch and the source-conditioned branch, respectively. |
| Lag | How many stored steps before the anchor a source coefficient occurs. |
| Proposal | The contribution a source lag proposes to the latent update. Proposals are summed, then bounded by a limiter. |
| Selector | A switch that enables or removes a lag's contribution. |
| Margin | An intervened arm's score minus the full branch's score. A positive margin means the intervention worsened prediction. |
| Bootstrap interval | An uncertainty interval computed by repeatedly resampling recordings and recalculating the statistic. |
| Gate | A named check that reports whether a required property holds. |

## 1. Run an evaluation

Run this command from the repository root, replacing the checkpoint path:

```bash
python -m teb_vae.lag_slot_transformer_cfs.eval.run --checkpoint output/<run>/model_checkpoints/<name>.ckpt
```

To launch from an IDE, set `RUN_ARGS` at the bottom of `run.py` and use the Run button. A checkpoint is required. The runner checks for it after combining the command-line arguments with `RUN_ARGS`, so either launch method works.

### What the scoring pass does

1. Loads the training run's `resolved_config.yaml` and applies `configs/eval_overrides.yaml`. This preserves the training settings while recording the evaluation changes.
2. Rebuilds the network and task through the checkpoint contract. A checkpoint from another model class is rejected.
3. Evaluates the split once. Each batch gets one dense forward pass that retains the per-lag proposals. The runner constructs the intervention arms and scores them in one Monte Carlo loop, using the same random noise $\epsilon$ for each arm within a draw. It also scores each horizon step and target block. Single-lag interventions run only on the subset allowed by `caps.lag_profile`.
4. Aggregates results per recording and computes bootstrap intervals. Each margin uses paired differences: subtract the two arms' scores for each recording, then resample those differences.
5. Writes `summary.json`, the merged configuration, and the tables below. It draws the summary figures from these saved results. A plotting failure is recorded under `figures.error`; the summary is still saved.

| Output | Contents |
| --- | --- |
| `summary.json` | Scores, uncertainty intervals, checks, run identity, and stage status. |
| `per_recording.csv` | One row per recording, including the values used to compute scalar intervals. |
| `lag_profile.csv` | One row per candidate lag. |
| `horizon_resolved.csv` | One row per arm and forecast step. |
| `figures/` | Plots made from the summary and tables. |
| `recording_traces/` | Detailed outputs for selected recordings, produced in a separate stage. |
| `attribution/` | Input-attribution tables, figures, and traces, produced after the recording traces. |

The `scored_split` block records the input files, standardisation statistics, their common parent directory, and a digest identifying the recordings actually scored. Keep this block and `per_recording.csv`: the multi-run comparison needs the individual recording scores and identities, not just an interval or a file path.

### Detailed traces and input attribution

After scoring, the runner selects up to `eval_config.caps.traces_per_class` recordings from each clinical class using a fixed random seed. Eligible recordings have at least two segments. It rereads every segment of each selected recording in `epoch` order and saves `anchor_trace.parquet`, `segment_summary.csv`, a manifest, one compressed array file and figure per recording, and a summary figure.

At each decoded anchor, the trace includes latent parameters, divergence and its per-coordinate values, the bounded update, and per-lag proposal norms. Its forecast scores come from the forward pass's own single-draw forecasts. Clinical class is recovered from the weight-scaled `target` field loaded by the overrides. If that field is unavailable, the stage records a skip. Batches are checked against their requested rows. The `recording_traces` summary block records selection, status, the manifest, and whether lag quantities were available; they are absent for the normalised-fusion comparator and the target-only arm. Failures appear under `recording_traces.error` without stopping the evaluation.

The attribution stage then uses Captum integrated gradients to relate three outputs to the three input streams: divergence, the mean-decoded forecast gap, and the proposal norm in each configured lag band. It selects up to `eval_config.caps.attribution_segments` segments, one per sampled recording with balanced clinical classes, and attributes a few anchors in each. It compares a source-null baseline, which zeroes the source while holding the target streams fixed, with an all-zero baseline. It also attributes the proposal head's output by lag, removes source bands in turn, keeps one example anchor per class with the input streams and every readout's full maps under both baselines (the divergence, the forecast gap, the full-branch block score and the lag readout on each band) for the map pages, and follows one recording per class through all its segments for both the divergence and the forecast gap.

The attribution files follow the shared family format. See [ATTRIBUTION.md](../../lag_attn_cfs/eval/ATTRIBUTION.md) for the method, baseline choices, checks, and interpretation limits. Here, the lag quantity is a proposal norm. Attribution through the sum and limiter does not allocate the divergence among lags. The `attribution` block records status, structural checks, and cost; a failure is recorded under `attribution.error` and the evaluation completes.

### Verify the saved result

```bash
python -m teb_vae.lag_slot_transformer_cfs.eval.verify --summary <run>/eval_results/summary.json
```

Verification uses only the Python standard library. It reads the saved summary and can run on a computer without the checkpoint, dataset, or PyTorch.

## 2. Understand the scored arms

Each arm first produces latent parameters. All arms are then scored together with shared random draws. Identical parameters therefore give bitwise identical scores, making exact reference checks possible.

| Arm | What changes | What stays fixed |
| --- | --- | --- |
| `base` | Uses the target-only prior. | Target history. |
| `full` | Uses the matched source-conditioned branch. | Target history and the matching source recording. |
| `suppress:<band>` | Sets the selectors for one lag band to zero. | Target state, metadata, original per-channel masks, remaining proposals, and $c_L$. |
| `suppress:none` | Removes nothing. | Everything. |
| `suppress:all` | Removes every declared band at once. | Everything else. |
| `silence` | Sets every selector to zero. | Everything else. |
| `replace:zeros` | Replaces source values with standardised zeros while keeping selectors enabled. | Availability indicators and metadata clock. |
| `replace:constant` | Replaces each channel with its own per-sample time mean. | Availability indicators and metadata clock. |
| `permute` | Uses the source stream from a different recording. | Time order within the substituted source. |
| `lag:<ℓ>` | Removes one candidate lag on the subset allowed by `caps.lag_profile`. | The same quantities as a band-suppression arm. |

For the local additive fusion, band and single-lag suppression reuse cached proposals. Source replacement requires another forward pass because the proposals depend on the source values. Single-lag results are stored as a curve under `lag_readouts.lag_profile.predictive`, rather than as separate arm columns. This profile is skipped for normalised fusion, where each lag removal would require another full forward pass and would change the remaining weights.

Two reference checks verify that interventions reproduce the intended forward computation:

- `suppress:none` matches `full`, so its margin is exactly zero.
- `suppress:all` and `silence` match `base`, so their margins equal the base-minus-full gap exactly.

### 2.1 Differences for comparator models

The evaluator also supports model variants used for comparison. Their behaviour is recorded under `arm` in `summary.json`.

| Variant | Evaluation behaviour |
| --- | --- |
| `source_disabled` | Skips every intervention with a reason, leaves exposure empty, and reports an exactly zero gap. |
| `source_values_withheld` | Skips replacement and permutation because this model does not read source values. Band suppression still measures its use of lag identity and availability. |
| `lag_fusion: attention` | Applies selectors and reruns the forward pass for band suppression. It does not produce cached additive proposals or a cancellation readout. |

Interpret suppression within each fusion type. In an explicit sum, removing a lag leaves the other terms unchanged. In normalised attention, removing a lag also changes the denominator, so the surviving weights increase. These interventions answer related questions but their margins are not directly comparable. Each summary records the distinction under `arm.suppression_semantics`.

The reference identities still hold under both fusions: no suppression reproduces the matched forecast, and complete suppression reproduces the prior. Attention comparators do not publish attention weights as lag evidence; both fusions are assessed through suppression.

## 3. Read the scores and intervals

### Predictive score and draw count

The predictive score is the negative logarithm of the likelihood averaged over $K$ latent draws:

$$
D^{(K)} = -\operatorname{logsumexp}_k(-D^{(k)}) + \log K.
$$

Here, $D^{(k)}$ is the negative log-likelihood for one draw. Lower scores mean better predictions. Scores are reported in nats per anchor: a log-score unit summed over the forecast block and then averaged over anchors. A negative log-density can be negative, so zero is not a universal reference point.

Average the likelihoods before taking their negative logarithm. Averaging the per-draw negative log-likelihoods gives a different quantity. Decoding only the latent mean also gives a different score and does not integrate over latent uncertainty.

The likelihood average is unbiased, but its negative logarithm has upward bias at finite $K$. The biases of two arms need not cancel. The field `draw_concentration_full` reports the effective draw count $1/\sum_k\alpha_k^2$, where $\alpha_k$ is each draw's normalised likelihood weight. A value near one means a few draws dominate. This is a diagnostic, not proof of convergence; rescore finalists at several draw counts.

### Calibration

Calibration asks whether predicted uncertainty matches observed errors. It uses the cumulative distribution of the predictive mixture, accumulated in the same draw loop. An observation is inside a central interval with nominal coverage $q$ exactly when its cumulative probability lies in $[(1-q)/2,(1+q)/2]$. This cumulative probability is the probability integral transform, or PIT.

The calculation uses the mixture distribution itself. Averaging conditional standard deviations does not give mixture intervals, and even a Gaussian with the correct total variance generally has different quantiles from the mixture.

### Recording-level and paired intervals

Neighbouring forecast windows share most of their target steps. Treating their anchors as independent observations would make uncertainty look too small. The evaluator therefore resamples recordings. It reports equal-recording and anchor-weighted summaries separately: these differ when recordings contribute different numbers of anchors.

For a comparison, it first subtracts the two scores within each recording and then bootstraps those differences. Use these paired intervals to judge margins. They appear as `margin_interval` beside a band's `margin_nats`, and as `<control>_margin_interval` for controls. Separate intervals for each arm do not answer whether their difference excludes zero. The predictive gap is also paired.

### Horizon and target-block results

The fields `horizon_resolved` and `block_resolved` score each subset of the forecast separately under the shared draws. For a subset $I$:

$$
D_I^{(K)} = -\operatorname{logsumexp}_k\left(-\sum_{i\in I}d_i^{(k)}\right) + \log K.
$$

These subset scores do not generally add to the joint forecast-block score, because averaging a product of likelihoods differs from multiplying their averages. Each axis includes all arm scores, the gap, and paired margin intervals. One resampling of recordings is shared across positions. Use the horizon curves to see which future steps benefit from the source; use block results to see which target features benefit.

## 4. Interpret the results carefully

- **Band margins are separate interventions.** The limiter acts after the proposals are summed. Two band-removal margins therefore need not add to the margin of removing both bands, the predictive gap, or the divergence.
- **A margin describes this fitted model.** Reallocating proposals as $r_\ell\mapsto r_\ell+k_\ell(h_t)$ with $\sum_\ell k_\ell\equiv0$ leaves the full prediction unchanged but can change a single-lag removal result. Every summary must retain this qualification under `lag_readouts.qualification`; the acceptance gate checks for it.
- **Missing exposure means no measurement.** A band with no available channels has nothing to remove. Its margin is `null`, and its usable anchor and channel counts are reported beside it.
- **A lag is stored-coefficient time.** The recommended model adds only one stored sample of neural source receptive field, but upstream causal feature extraction already combines raw history within each coefficient. Lag readouts therefore do not measure a physiological delay. The encoder disclosure records both kinds of history dependence.
- **A positive internal gap needs an external comparison.** Joint training can weaken the model's own base branch. Compare the full and base branches with an independently trained, frozen target-only model before interpreting the gap as an improvement.
- **The gap has no acceptance threshold here.** It is reported with uncertainty. The first real runs are intended to establish what improvement is achievable; this evaluator does not impose an unmeasured threshold.

### 4.1 Read lag results from broad bands to individual lags

Start with whole-band and joint removal results, then inspect individual lags. A narrow peak alone is weak evidence when removing its surrounding window has no predictive effect. The figures `band_suppression` and `lag_profile` show these related results.

| Readout | Summary location | Meaning and cost |
| --- | --- | --- |
| Band suppression | `lag_readouts.band_suppression` | Paired margin and exposure for each band, plus the `none` and `all` identities. Uses cached subtraction per band on every segment for local fusion. |
| Latent profile | `lag_readouts.lag_profile.latent` | For each available lag: proposal norm $\lVert r_\ell\rVert$, change in bounded update $\lVert a-a^{\setminus\ell}\rVert$, and signed divergence drop $K-K^{\setminus\ell}$. Includes scale proposal norm where supported. Uses cached tensor arithmetic on every segment. |
| Predictive profile | `lag_readouts.lag_profile.predictive` | Paired predictive margin for removing one lag at a time. Requires one decoder call per lag per draw on the first `caps.lag_profile` segments. |

Latent profiles average over anchors where the lag was available. A large proposal can have little effect on the bounded update when the limiter is saturated. The divergence drop can be negative if removing a proposal stops it from cancelling another proposal. None of these quantities allocates a fixed total among lags.

The predictive profile covers a capped subset of the split; its segment and recording counts are saved with it. The `lag_readouts.lag_axis` block records stored steps before the anchor, seconds per step, and the model's input delay. The table `lag_profile.csv` includes all three readouts, one row per lag.

## 5. Analyses this architecture does not provide

Seven analyses in the wider model family require tensors that this architecture does not compute. Their names and required tensors are recorded under `excluded_analyses`; `excluded_analyses_mechanism` explains how they were excluded.

| Analysis | How it is excluded | Required quantity |
| --- | --- | --- |
| `attention` | Removed from the shared registry. | Attention distribution over lags. |
| `lag_kl` | Removed from the shared registry. | Per-lag allocation of divergence. |
| `source_null` | Never registered here. | The lag-attentive model's parameterised source encoding path for a zeroed stream. |
| `occlusion` | Never registered here. | Lag attention used to rebuild the full branch. |
| `lag_clocks` | Never registered here. | Per-lag divergence allocation resolved against clinical time. |
| `lag_kld_scaled` | Never registered here. | That allocation partitioned over lag bands. |
| `lag_high_kl` | Never registered here. | Per-anchor lag map used for anchor selection. |

The evaluator uses its own suppression and source-replacement analyses where appropriate. It does not relabel proposal norms as attention or divergence allocations. The acceptance gate rejects summaries containing the excluded analysis key names.

The shared collection pass is also unsuitable: it expects latent tensors indexed at every stored time step and requires eight attention-derived fields. This model indexes latent tensors by decoded anchor. It reuses architecture-independent components instead: checkpoint and task loading, configuration merging and validation, block scoring, log-mean-likelihood scoring, cross-recording permutation, recording-level bootstrapping, and summary assembly.

## 6. Structural gates and their evidence

These six checks establish required properties of the recommended architecture. The table links each property to tests in the package's own suite. Passing them establishes implementation behaviour; it does not establish predictive usefulness on real recordings.

| Gate | Evidence |
| --- | --- |
| The prior sees no source values and retains every latent coordinate | `test_causality.py::test_the_metadata_clock_carries_no_source_value`, `::test_the_source_reaches_the_full_branch_and_never_the_base_branch`; `test_construct.py::test_the_metadata_clock_is_a_function_of_position_alone`, `::test_the_prior_head_is_built_without_its_own_clock_path` |
| Each source encoding is pointwise; each proposal reads one stored source time | `test_pointwise_source.py::test_the_encoding_is_pointwise_by_jacobian`, `::test_the_scalar_lift_stays_pointwise`; `test_lag_updates.py::test_a_proposal_reads_exactly_one_stored_source_time`, `::test_a_proposal_may_combine_channels_at_its_own_source_time` |
| No future value and no future validity mask enters prediction | `test_causality.py::test_resampling_the_strict_future_leaves_earlier_anchors_bitwise_unchanged`, `::test_no_future_validity_signal_can_enter_the_forward`, `::test_no_lag_reads_a_step_at_or_after_its_anchor` |
| Source-disabled and all-unavailable inputs reproduce the prior; a valid observed zero need not | `test_invariants.py::test_every_selector_off_reproduces_the_prior`, `::test_every_lag_unavailable_reproduces_the_prior_and_stays_finite`, `::test_the_zero_initialised_model_reproduces_the_prior_for_arbitrary_inputs`, `::test_an_observed_standardized_zero_is_not_treated_as_absence` |
| The source has no route around the latent | `test_invariants.py::test_the_decoder_is_invoked_twice_with_the_latent_and_the_persistence_input_only`, `::test_the_two_decoder_calls_share_one_module_and_its_weights`; `test_forward_contract.py::test_no_attention_shaped_key_is_emitted` |
| Shared noise, the divergence, the masks, the anchor indexing and the gradients behave as specified | `test_invariants.py::test_the_paired_sample_difference_matches_the_stated_formula`, `::test_the_predictive_gradient_reaches_the_final_source_projection_at_the_zero_start`, `::test_the_source_pathway_leaves_the_zero_start_after_a_few_steps`; `test_residual_kl.py::test_the_residual_divergence_equals_the_family_formula`; `test_objective.py::test_the_reconstruction_and_divergence_share_one_anchor_set`, `::test_the_objective_optimises_the_global_mean_under_uneven_ranks`; `test_forward_contract.py::test_the_anchor_axis_is_not_the_time_axis` |

The offline `verify.py` also checks the run's own output: reference arms must reproduce their exact scores, and the summary must not claim an attention distribution or per-lag divergence allocation.

### 6.1 Declared differences for comparator models

Comparator variants change a specific architectural choice. Their `arm` block declares which requirement is relaxed, so the comparison can be interpreted correctly.

| Variant | Declared difference | Other gates |
| --- | --- | --- |
| `source_stem: conv` | Source encoding uses a bounded window ending at each step. Its reach is recorded in `encoder_disclosure.source_receptive_field_steps`. | Still apply. |
| `lag_fusion: attention` | Uses learned normalised aggregation over lag keys and values instead of explicit additive proposals. | Still apply. |
| `source_values_withheld` | Withholds source values; it does not relax an architectural requirement. | Still apply. |
| `source_disabled` | Has no source pathway to constrain. | Apply where relevant. |

A band narrower than a convolution stem's receptive field cannot isolate that narrow interval of source history: neighbouring lag encodings summarise overlapping windows. Interpret the comparisons as a sequence of controlled changes. The attention reference versus pointwise attention tests the convolution; pointwise attention versus the candidate tests fusion; the candidate versus mean-only, capacity-control, and target-only variants tests the variance update, source values, and source pathway, respectively.

`tests/test_arm_scoring.py` fits and scores the comparator variants on small fixtures and checks that these differences are recorded.

## 7. Compare several runs

One checkpoint cannot show whether an effect survives retraining, what a simple probe can recover from the latent, or whether a selected result holds on untouched recordings. Use latent probes and the acceptance protocol for these questions. The probe loads a checkpoint; the acceptance pass reads the saved evaluation and probe artifacts.

### 7.1 Latent probes

```bash
python -m teb_vae.lag_slot_transformer_cfs.eval.latent_probes --checkpoint output/<run>/model_checkpoints/<name>.ckpt
```

A probe is a simple predictor trained on a frozen model's latent outputs. This pass fits ridge regression from six readouts: the prior and full distributions' means, scales, and one shared-noise draw from each. It predicts the future forecast block relative to the anchor's stored values and is evaluated on recordings excluded from probe fitting.

Subtracting the anchor values reduces the benefit a probe gets merely from the future resembling the present. Using stored values, rather than learned decoder persistence weights, also makes the target comparable across model variants. A digest of the recording identifier determines the fit/evaluation split, so variants use the same recordings. The fit accumulates second moments without storing all input rows, allowing it to use every anchor with complete coverage.

The output is `latent_probes.json`. The acceptance protocol associates it with a run through checkpoint identity. Probe performance shows what this simple predictor can recover; it does not measure everything the latent contains or what the model's decoder uses. The difference between prior and full probe results is also not a direct measurement of source information.

### 7.2 Acceptance protocol

```bash
python -m teb_vae.lag_slot_transformer_cfs.eval.acceptance --runs output/<development evaluations> --reference output/<frozen target-only>/eval_results/summary.json --output acceptance.json
```

This pass reads evaluation directories under a root and groups runs by model variant and training seed. The file `configs/acceptance_plan.yaml` declares the minimum seeds, primary draw count, bootstrap resamples, five comparisons, and searchable lag bands. The acceptance record stores the plan's digest so later edits are visible. The pass needs only the standard library and the array library used for intervals unless figures are requested.

| Part of the protocol | How to interpret it |
| --- | --- |
| Several seeds | Scores are averaged across training seeds within each recording, then recordings are bootstrapped once. Re-evaluating one checkpoint three times still counts as one training seed. |
| Paired comparisons | Matching split, seed, and draw count allow per-recording differences between variants. Intervals are computed on those differences. |
| Multiple comparisons | The five declared comparisons use nominal intervals. Searched lag bands also receive family-adjusted intervals covering all searched bands together, including a band selected after viewing the results. |
| Reserved confirmation data | Overlap is checked using recording identities. Different file names or folds do not establish that the recordings are untouched. |

Structural checks can fail for a failed single-run gate, an undeclared band search, overlap between selection and confirmation recordings, or a base branch that is confidently worse than the frozen reference. Other measurements are reported with intervals without imposing an acceptance threshold.

To produce a readable report and the three acceptance figures from the assembled record:

```bash
python -m teb_vae.lag_slot_transformer_cfs.eval.acceptance --runs output/<development evaluations> --reference output/<frozen target-only>/eval_results/summary.json --output acceptance.json --report acceptance.md --figures acceptance_figures
```

The report includes variants and seeds, declared comparisons, internal and reference comparisons, controls, band-search intervals, draw-count stability, calibration, probes, and verdicts. The report and figures use the assembled record without recomputing its statistics. See [the figure guide](FIGURE_GUIDE.md) for the plots.

## 8. Settings

Set evaluation choices in `configs/eval_overrides.yaml`. The merged configuration is saved in the run directory, and resolved launch arguments are recorded under `run.argument_sources` in `summary.json`.

| Setting | Meaning |
| --- | --- |
| `eval_config.seed` | Seeds latent draws and cross-recording pairing as independent random streams. |
| `eval_config.num_mc_samples` | Number of latent draws $K$. Finalists are evaluated at $8$, $32$, and $128$. |
| `eval_config.bootstrap_resamples` | Number of recording-level resamples used for intervals. |
| `eval_config.occlusion_bands` | Inclusive lag bands removed by suppression, in stored steps before the anchor. |
| `eval_config.caps.lag_profile` | Segment cap for single-lag predictive scoring. Omitting it skips that profile with a recorded reason. |
| `eval_config.caps.traces_per_class` | Maximum traced recordings per clinical class. Each selected recording is followed through every segment. Omission uses the family default. |
| `eval_config.caps.attribution_segments` | Number of attributed segments, one per sampled recording. Omission uses the family default. |
| `eval_config.figure_format` | Figure file format. A value of `null` uses the family default. |
| `eval_config.max_samples` | Maximum segments scored. A value of `null` evaluates the whole split. |
| `general_config.batch_size.test` | Loader batch size. This also affects how often a batch contains different recordings that can be paired. |

`eval_config.clock_margin_min_nats` is unset and unused by this architecture's verdicts. It belongs to the lag-attentive availability-clock check, which uses that model family's source-null encoding path.

## 9. Scope of this guide

This document describes evaluation machinery and how to interpret its outputs. It does not report completed training experiments or establish empirical acceptance. Those conclusions require real-data runs, comparisons across variants and seeds, synthetic checks, and evaluation on reserved recordings.
