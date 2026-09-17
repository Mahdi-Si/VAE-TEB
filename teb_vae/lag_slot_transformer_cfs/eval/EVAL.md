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
| `recording_traces/` | Detailed outputs for selected recordings, produced in a separate stage: per recording a PDF figure and an interactive plotly page (`<guid>_<subgroup>_trace.html`) with the raw FHR and UA on top, log colour on the lag map and the bounded mean update, and later segments overwriting earlier ones where they overlap. |
| `attribution/` | Input-attribution tables, figures, and traces, produced after the recording traces. |
| `proposal_profile/`, `proposal_clocks/`, `band_clocks/`, `high_kl_anchors/` | The lag structure read off this cell's own sidecars; see section 5. |
| `warmup/`, `source_null/`, `spectral_skill/` | The family's own analyses of the columns this pass writes under the family's names; see section 5 and the family's guide. |

Every table-driven analysis of the family also writes its own subdirectory (`forecast/`, `coupling/`, `latent/`, `calibration/`, `distributions/`, `trajectory/`, `time_to_delivery/`, `second_stage/`, `events/`, `sufficiency/`, `cross_subgroup/`), each with its by-class and by-subgroup variants, exactly as on a lag-attentive cell.

The `scored_split` block records the input files, standardisation statistics, their common parent directory, and a digest identifying the recordings actually scored. Keep this block and `per_recording.csv`: the multi-run comparison needs the individual recording scores and identities, not just an interval or a file path.

### Detailed traces and input attribution

After scoring, the runner selects up to `eval_config.caps.traces_per_class` recordings from each clinical class using a fixed random seed. Eligible recordings have at least two segments. It rereads every segment of each selected recording in `epoch` order and saves `anchor_trace.parquet`, `segment_summary.csv`, a manifest, one compressed array file and figure per recording, and a summary figure.

At each decoded anchor, the trace includes latent parameters, divergence and its per-coordinate values, the bounded update, and per-lag proposal norms. Its forecast scores come from the forward pass's own single-draw forecasts. Clinical class is recovered from the weight-scaled `target` field loaded by the overrides. If that field is unavailable, the stage records a skip. Batches are checked against their requested rows. The `recording_traces` summary block records selection, status, the manifest, and whether lag quantities were available; they are absent for the normalised-fusion comparator and the target-only arm. Failures appear under `recording_traces.error` without stopping the evaluation.

The attribution stage then uses Captum integrated gradients to relate the model's per-anchor outputs to the three input streams: the divergence, the mean-decoded forecast gap, the full-branch block score and its squared-error fidelity, the score at the first and the last horizon step, and the proposal norm in each configured lag band. It selects up to `eval_config.caps.attribution_segments` segments, one per sampled recording with balanced clinical classes, and attributes a few anchors in each. It compares a source-null baseline, which zeroes the source while holding the target streams fixed, with an all-zero baseline. It also attributes the proposal head's output by lag, removes source bands in turn, keeps one example anchor per class with the input streams and every readout's full maps under both baselines (the divergence, the forecast gap, the full-branch block score and the lag readout on each band) for the map pages, and follows one recording per class through all its segments for both the divergence and the forecast gap.

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

## 5. The lag structure, and the family analyses this cell asks its own way

The lag-attentive cells resolve their lag axis through an attention distribution and a per-lag allocation of the divergence. This architecture computes neither. What it computes instead are two per-lag readouts of one fitted parameterisation, which the collection pass writes as sidecars beside the family's tables:

| Sidecar | Contents |
| --- | --- |
| `per_sample_vectors.npz` : `proposal_lag_profile` | Each segment's proposal norm $\lVert r^\mu_{t,\ell} \rVert$ averaged over the anchors at which lag $\ell$ was live; NaN where it never was. |
| `per_sample_vectors.npz` : `divergence_drop_lag_profile` | The signed divergence drop $K_t - K_t^{\setminus \ell}$ averaged the same way. |
| `per_anchor_vectors.npz` : `proposal_lag_map`, `divergence_drop_lag_map` | The same two quantities at every contributing anchor, row-aligned with `per_anchor.parquet`, in half precision. |
| `per_anchor.parquet` : `proposal_argmax_lag` | The lag whose proposal norm was largest at that anchor; `-1` where no lag was live. |
| `per_sample.csv` : `margin_<arm>` | Every intervened arm's paired margin against the matched branch, per segment (`margin_suppress_near`, `margin_replace_zeros`, ...). |
| `per_sample.csv` : `kld_source_null`, `coupling_minus_clock` | The divergence under the zeroed-source arm and what the matched source adds over it, the family's own columns. |

Neither map is an attention distribution and neither allocates the divergence over lags; the summary's `lag_readouts.sidecars` block says so beside the names. Four analyses of this cell read them, and each asks the question one of the absent family analyses asks, named for what it reads rather than for what the family reads:

| Family analysis | Required quantity | This cell's analysis | What it reads |
| --- | --- | --- | --- |
| `attention`, `lag_kl` | Attention over lags; per-lag divergence allocation. | `proposal_profile` | The shape of both profiles per segment (centroid, spread, quantiles, entropy, effective support, the guarded peak, band masses), pooled over recordings and per cohort. |
| `lag_clocks`, `lag_kld_scaled` | That allocation against the clinical clocks; its band masses. | `proposal_clocks` | The same shape statistics and the band masses of both profiles on both clocks, per class, with the two centroids tested per window. |
| `occlusion` (clock page) | Lag attention rebuilding the full branch. | `band_clocks` (beside `lag_suppression` and `resolved_axes`) | Every band and control margin on both clocks, per class, descriptive. |
| `lag_high_kl` | The per-anchor KL lag map. | `high_kl_anchors` | Anchors selected by their own divergence at pooled thresholds; their proposal profiles, their forecast gain against the rest, the overlap with a gain-selected band, contraction enrichment, and both clocks. |

Three further family analyses are registered here **unchanged**, because the columns they read are the same quantities on this cell: `warmup` (the warm-up tertile gaps and the geometry guards; the source-lag warmth fractions are reported absent, since they are an attention mass), `source_null` (the divergence under the zeroed-source arm against the matched one, per recording; its lag-resolved half is reported unmeasured) and `spectral_skill` (the per-channel gap vector on the kept frequency bands).

The six family analyses that cannot run here are recorded under `excluded_analyses` with the tensor each would have needed, `excluded_analyses_mechanism` says how each was left out, and `analogue_analyses` names the analysis above that asks its question. Nothing is relabelled: the acceptance gate still rejects a summary carrying an attention-shaped key anywhere.

Read every lag-structure artifact under the qualification the suppression readouts carry: a reallocation $r_\ell \mapsto r_\ell + k_\ell(h_t)$ with $\sum_\ell k_\ell \equiv 0$ leaves the update, the divergence and every prediction unchanged while changing both profiles at every lag. A centroid says where the fitted head's proposals sit; it does not say the source at that lag was necessary. The `high_kl_anchors` hot-lag set is a top-share selection taken from the same map it summarises, and every artifact of that analysis says so.

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

### 7.3 Rescoring an existing 91-entry checkpoint with its tail removed jointly

The shipped four bands measure the source pathway band by band, and band margins do not add:
proposals interact through the sum, the limiter, the sample and the decoder, so the effect of
removing every lag above a cutoff at once is its own measurement. The profile
`configs/lag91_tail_diagnostic.yaml` declares that measurement as bands -- the cutoff pair
`head_0_24` / `tail_25_90` and a keep-prefix family `beyond_<l>` that removes every lag above
$\ell$ -- beside the shipped partition, at the acceptance plan's primary draw count. It is a copy
of the committed delta with those two differences and is merged over the checkpoint's own resolved
configuration in the same way; repoint its shard paths at the panel the checkpoint's shipped
summary records.

Run it three times on the same panel, at the primary count and the two stability counts:

```bash
python -m teb_vae.lag_slot_transformer_cfs.eval.run --checkpoint <run>/model_checkpoints/<name>.ckpt \
    --overrides teb_vae/lag_slot_transformer_cfs/eval/configs/lag91_tail_diagnostic.yaml
python -m teb_vae.lag_slot_transformer_cfs.eval.run --checkpoint <same> --overrides <same> --num-samples 8
python -m teb_vae.lag_slot_transformer_cfs.eval.run --checkpoint <same> --overrides <same> --num-samples 128
```

Read, in this order, from each `summary.json` under `results`:

1. `lag_readouts.band_suppression.tail_25_90.margin_nats` with its paired interval: the joint
   removal. Compare it with the sum of the `near`, `mid` and `far` margins from the same run; the
   difference is the interaction the bands cannot show.
2. `lag_readouts.band_suppression.beyond_<l>` in increasing $\ell$: the keep-prefix curve. Its
   two ends are the reference identities the readout already carries -- an empty prefix is
   `all`, which equals `silence`, and the whole window is `none`.
3. `arm_scores.draw_concentration_full` and the per-recording `draw_concentration_full` column
   at each draw count. A ranking that moves between 8 and 128 by more than the tail margin's
   interval width is a draw-count effect, not a lag effect.
4. `verdicts` and `eval.verify`: the predictive verdict is read from the paired interval; the
   gate additionally refuses a summary whose verdict list disagrees with the interval it carries.

Keep the three summaries and their `per_recording.csv` tables; the acceptance pass does not read
this profile, because its bands are outside the plan's declared family. Do not shorten
`event_lag_window_s` for this profile: it is an event-conditioned reporting window, not the bank.

### 7.4 What the score columns are

Every score column of a summary is one of four estimators, and `results.conventions.estimators`
names them beside their columns:

| Estimator | Columns | What it is |
| --- | --- | --- |
| weighted objective | `nll_*_block_weighted`, `pred_gap_weighted` | the training objective's reconstruction terms, one shared draw under the configured channel and horizon weights; not a log density |
| single-draw conditional | `nll_*_block`, `pred_gap` on `per_sample.csv` | the same draw scored unweighted, a conditional log density given one latent sample |
| latent mean | `mean_nll_*_block`, `mean_pred_gap` | the decoder at the latent mean, no draw |
| predictive mixture | `mc_nll_*_block`, `mc_pred_gap`; `nll_<arm>`, `pred_gap` on `per_recording.csv`; `pred_gap_mc_nats` | the negative log of the average likelihood over $K$ shared draws; the headline and every scientific verdict read this one |

Two aggregation estimands are reported. `arm_scores` and the headline weight a recording's
segments equally and then recordings equally; `anchor_within_recording` carries the same columns
and paired margins with anchors weighted equally within each recording, and its headline columns
sit on `per_recording.csv` under `*_anchor_weighted`. `results.conventions.headline_estimator`
says which the headline reads. The single-lag predictive profile's cohort is declared under
`lag_readouts.lag_profile.predictive.segments` and written to
`lag_suppression/lag_profile_segments.csv`: segments are admitted in loader order under the cap
and a per-class quota over the classes the split's shards declare, so no class fills the cap
alone.

The family's sufficiency probe is reported with `results.sufficiency_qualification` marking its
interpretation unavailable: the probe conditions on the encoder state alone, without the metadata
clock or the persistence path the production decoder has, so its gap is a probe fit rather than
the bottleneck's cost. That repair is deferred.

### 7.5 Selecting checkpoints on the predictive monitor

A training run selects its checkpoints on whatever `advanced_config.callbacks.model_checkpoint.monitor`
names. Shipped, that is `val/total_loss`: the one-draw weighted objective, which is not the
estimator any summary reports. With `model_config.VAE_model.validation_mc_draws` set to a
positive $K$, every dense validation batch is also scored as the unweighted $K$-draw predictive
mixture of both branches under a noise bank keyed on the batch's segment identities and the run
seed, and three further columns reach `metrics_history.csv`:

| Column | What it is |
| --- | --- |
| `val/pred_nll_full_mc` | the full branch's mixture score, recording-grouped within each batch, batch-global across ranks |
| `val/pred_nll_base_mc` | the same for the base branch, under the identical draws |
| `val/pred_gap_mc` | base minus full: the paired predictive gain the offline headline reads, at the training run's own $K$ |

Two validation passes at the same weights report the same value, which is what lets the column
rank checkpoints; the training-stage columns are never monitored, because a training batch is
tiled and its anchor set changes every epoch. The weighted objective stays logged beside the
monitor as the optimisation diagnostic, and `train/grad_clip_frac` is the fraction of each
epoch's optimizer steps on which the clip bound, counted over every step.

The arm profiles (`target_only.yaml`, `joint.yaml` and everything based on them) set $K = 8$ and
point both the checkpoint and the early-stopping monitor at `val/pred_nll_full_mc`, with the
weighted objective's own optimum kept by the secondary criterion. `default.yaml` keeps the legacy
selection so the shipped run stays reproducible as configured. A monitor naming one of the three
columns while the draw count is null is refused before the model is built.

### 7.6 The matched-arm comparison and the bank-length decision

Six arms, one target-only initialisation, one selection rule, one scoring profile per bank. Run
them in this order on the GPU box against the integer-operator production shards; every run
writes its `resolved_config.yaml` beside its checkpoints, which is the durable record of what it
was.

1. **Two target-only fits.** `target_only.yaml` at its shipped seed (role `-a`, the warm-start
   donor) and once more at a different seed and tag (role `-b`, the frozen reference). Score both
   with `eval_overrides.yaml`. Before reading anything else, record the paired predictive
   difference between the two as the **bank-length noninferiority tolerance**: a shortened bank
   whose interval against the 91-entry candidate lies within that seed-to-seed spread is
   competitive, one whose interval lies below it is not. Write the tolerance down first; a
   tolerance chosen after the comparison is read is not a tolerance.
2. **The two banks.** `joint.yaml` (91 entries) and `lag25.yaml` (25 entries), both with
   `target_warm_start_checkpoint` repointed at the `-a` checkpoint. Score `joint.yaml` with
   `eval_overrides.yaml` and `lag25.yaml` with `lag25_eval_overrides.yaml`, both at the plan's
   primary $K = 32$; then the finalists at 8 and 128 on the same panel, and read the effective-draw
   quantiles of both branches at each count.
3. **The decision.** Read `lag25.yaml` against `joint.yaml` on the arm scores, both paired
   against the `-b` reference: the predictive interval against the predeclared tolerance, the
   mixture calibration by horizon and block, the base drift against the reference, and the
   measured resources at the declared batch and draw counts. Record the decision in the run log
   below. No arm is promoted on `pred_gap_weighted`, on `mean_pred_gap`, or on its own internal
   gap alone.
4. **The two controls.** Set `base:` of `capacity_control.yaml` and `mean_only.yaml` to the
   chosen bank profile -- `lag25.yaml` if the shortened bank was competitive, `joint.yaml`
   otherwise -- repoint their warm start at `-a`, train, and score under the chosen bank's
   evaluation profile.
5. **One extra run.** The chosen arm again with `lambda_base: 0.5`, the shipped run's value,
   so the reconstruction weighting is compared against an external reference rather than assumed.

Then assemble the record:

```bash
python -m teb_vae.lag_slot_transformer_cfs.eval.acceptance --runs output/<development evaluations> \
    --reference output/<target-only -b>/eval_results/summary.json --output acceptance.json --report acceptance.md
```

The acceptance pass groups the runs by arm, reads each under its own window's band family, and
refuses a pairing across two windows or two input policies; it does not compare `lag25.yaml`
with `joint.yaml` directly, because their lag readouts are declared for different windows. That
comparison is the arm-score one above, on the common target set, and it is recorded by hand.

**Run log.** One entry per arm, appended here as the runs complete. Every field is required;
"not measured" is a value.

| Field | What to write |
| --- | --- |
| arm, profile, tag, seed | as configured |
| initialisation | the warm-start checkpoint's path and SHA-256, or "fresh" |
| effective loss weights | `lambda_full`, `lambda_base`, `beta_prior`, the resolved $\beta$ ramp, channel and horizon weights as resolved |
| data budget | shards, epochs trained, batch size, optimizer steps, the selected epoch and what selected it |
| selection | `validation_mc_draws`, the monitor, and `val/pred_nll_full_mc` at the selected epoch |
| external-base drift | `base_minus_reference_nats` against the `-b` reference, with its interval |
| predictive | `mc_pred_gap` and `pred_gap_mc_nats` with intervals at $K = 32$, and at 8 and 128 for a finalist; effective-draw quantiles of both branches |
| calibration | mixture coverage at the three levels, pooled and by horizon and block |
| resources | peak training memory, dense-evaluation memory and throughput at the declared batch and draw counts |
| decision | promote / reject / inconclusive, and the one sentence that says why |

No entries yet: the runs need the GPU box and the production shards.

### 7.7 The input-ablation factorial on the chosen bank

Four cells: neither switch (the chosen bank's own run above), the source switch
(`lag25_s0_up.yaml`), the target switch (`lag25_s0_fhr.yaml`) and both (`lag25_s0_both.yaml`),
plus the target-only reference under the target switch (`target_only_s0_fhr.yaml`, roles `-a` and
`-b` as for the parent). If the bank-length decision kept 91 entries, rebase the three ablation
profiles on `joint.yaml` first; each is a one- or two-leaf delta and moves nothing else.

Warm-start every source-enabled ablation from the target-only checkpoint that shares its FHR
policy: `lag25_s0_up.yaml` from the ordinary `-a` donor, the two arms that zero the FHR
coefficient from `target_only_s0_fhr.yaml`'s `-a`. Read each against the reference that shares its
policy, for the reason the profiles state: a reference that could read the level would credit the
source with closing a gap the ablation itself opened.

Score every cell at $K = 32$ under the chosen bank's evaluation profile. The summary's
`causality.effective_inputs` names the ablated coordinates; read the resolved axes so the
ablated coefficient's own score and the remaining channels' score are reported side by side, and
never let a loss of level context hide inside the aggregate. Record one run-log entry per cell
with the fields of 7.6 and, in the decision, whether each switch is kept for the confirmation
stage and why. Zeroing a coefficient at the input does not establish that its information is
absent from correlated coefficients; the decision is about the forecast under the policy, not
about the coefficient's physiology.

No entries yet: the runs need the GPU box and the production shards.

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
