# Evaluating a lag-residual causal-feature forecaster

What one pass measures, what it deliberately does not decide, and where the evidence for each
structural claim lives.

---

## 1. What a pass does

```
python -m teb_vae.lag_slot_transformer_cfs.eval.run \
    --checkpoint output/<run>/model_checkpoints/<name>.ckpt
```

or, from an IDE's Run button with nothing typed, fill in `RUN_ARGS` at the bottom of `run.py`.
`--checkpoint` is the one value the pass cannot proceed without, and it is enforced after the
launch merge rather than by `required=True`, which would fire before the launch dict was consulted
and make the Run button unusable.

One pass:

1. finds the training run's own `resolved_config.yaml` beside the checkpoint and deep-merges the
   committed override delta over it, so what the model was trained on stays authoritative and the
   divergence is one reviewable file;
2. rebuilds the net and its task through the checkpoint contract, refusing a checkpoint of another
   class by name;
3. walks the evaluation split once. Per batch it runs **one** dense forward with the per-lag
   proposals retained, builds every intervened arm from it, and scores them all in a single draw
   loop under one shared `ε` per replicate;
4. aggregates per recording, bootstraps over recordings, and writes `summary.json` beside the
   merged configuration it ran under, with `per_recording.csv` beside both — one row per
   recording, carrying the values every interval in the summary was built from.

The summary's `scored_split` block names the files the pass opened, the statistics it standardised
with, their common parent and a digest of the recordings that came back. That block and that table
are what the multi-run protocol in section 7 is built on: an interval cannot be taken apart into
the vector that produced it, and a partition cannot be shown to be untouched by anything except the
recordings that were scored.

```
python -m teb_vae.lag_slot_transformer_cfs.eval.verify --summary <run>/eval_results/summary.json
```

reads that file back. It is stdlib only — no torch, no model, no binding — so it runs against a
summary copied off the box that produced it.

---

## 2. The scored arms

Every arm below produces **latent parameters**, never a score of its own. All of them are then
scored together, which is what makes a margin a difference of predictions rather than a difference
of noise: two arms with identical parameters produce bitwise identical scores.

| Arm | What changes | What is held fixed |
| --- | --- | --- |
| `base` | nothing; the target-only prior | — |
| `full` | the matched source-conditioned branch | — |
| `suppress:<band>` | the selectors over one lag band go to zero | target state, metadata, original per-channel masks, remaining proposals, `c_L` |
| `suppress:none` | nothing — the reference arm | everything |
| `suppress:all` | every declared band at once | everything else |
| `silence` | every selector zero | everything else |
| `replace:zeros` | source values become standardized zeros, selectors **enabled** | availability announcement, metadata clock |
| `replace:constant` | each channel becomes its own per-sample time mean | availability announcement, metadata clock |
| `permute` | the source stream is paired with a different recording's | within-source time order |

Only the suppression arms recompute from cached proposals; the rest re-run the forward under a
substituted stream, because a proposal is a function of the source values it was given.

**Two identities pin the intervention path to the forward path**, and they are checked on real
weights by the evaluation smoke test:

- `suppress:none` reproduces the matched arm, so its margin is exactly zero;
- `suppress:all` and `silence` reproduce the target-only prior, so their margins equal the
  base-minus-full gap exactly.

### 2.1 What changes on a comparator arm

The pass scores every arm of this architecture, not only the recommended one, and three of them
change what it can run. Each departure is recorded in `summary.json` under `arm`, so a directory of
runs assembles into a comparison afterwards rather than into a set of numbers with no subjects.

| Arm | What the pass does differently |
| --- | --- |
| `source_disabled` | Every intervention is skipped by name with its reason; the exposure is empty rather than counts of zero; the gap is exactly zero by construction |
| `source_values_withheld` | The replacement and permutation arms are skipped by name: a substituted stream reaches a fusion that reads no values, and the availability announcement does not vary with the recording. Band suppression still runs, because lag identity and availability are real inputs to that arm |
| `lag_fusion: attention` | Band suppression runs through a **selector and a re-run forward** rather than through the cached per-lag updates, which that fusion does not produce. The cancellation readout is absent, because a normalised aggregation sums nothing and therefore has nothing that can cancel |

**A band margin does not compare across the two fusions.** Removing a lag from an explicit sum
leaves every surviving term where it was. Removing it from a normalised distribution removes it from
the denominator too, so the surviving weights grow to fill it. Both answer "what does this model do
when it cannot read these lags" and the two numbers are not on one scale: read each against its own
matched branch, never against the other arm's margin. The `arm.suppression_semantics` field of every
summary says which of the two produced it.

**The two reference identities hold on every arm regardless**, and they are what make each arm's own
margins measurements. `suppress:none` reproduces the matched forward and `suppress:all` and
`silence` reproduce the prior, under both fusions and by different routes -- a sum of zeroed terms
on one, a distribution with no admissible lag on the other.

**The attention arms publish no distribution over lags.** The weights are real there, unlike
anything the recommended arm could offer under that name; published beside a predictive comparison
they would be read as a lag readout, which is the claim the evidence behind this architecture shows
cannot be supported. Both arms are interrogated by suppression instead, through one interface.

---

## 3. What the numbers are

**The predictive score** of each arm is the marginalised density

```
D^(K) = -logsumexp_k(-D^(k)) + log K
```

the negative log of the average likelihood. It is not the mean of the per-draw negative log
likelihoods — that quantity *improves* as the latent becomes less informative — and it is not the
score of a forecast decoded from the prior mean, which is not a predictive density at all.

The likelihood average is unbiased for the model likelihood; its negative logarithm is **upward
biased** at finite `K`, and two arms' biases need not cancel. `draw_concentration_full` reports
`1 / Σ_k α_k²`, how many of the `K` draws a score effectively rests on. It is a warning signal, not
a convergence proof, and a finalist is rescored at more than one draw count.

**Calibration** comes from the mixture cumulative distribution, accumulated across draws inside the
same loop. Central coverage follows from the probability integral transform with no quantile solve:
an observation lies inside the central-`q` interval exactly when its own cumulative probability lies
in `[(1-q)/2, (1+q)/2]`. A mean of conditional standard deviations appears nowhere in that path and
could not: the predictive law is a mixture, and a Gaussian interval built from even the correct
total variance is not a mixture quantile.

**Intervals are bootstrapped over recordings.** Consecutive anchors' forecast windows overlap in all
but one of their steps at the dense geometry, so resampling anchors would report an interval
narrower than the data supports. Equal-recording and anchor-weighted summaries are both reported,
separately, because they differ whenever recordings contribute unequal anchor counts.

---

## 4. What a reader must not take from the output

**The band margins do not decompose anything.** They are not normalised to sum to the gap or to the
divergence and never will be: the limiter is applied after the summation, so the bounded update is
not linear in the proposals, and two bands' margins need not add to the margin of removing both.

**A margin is a property of a fitted parameterisation.** Proposals admit a reallocation
`r_ℓ → r_ℓ + k_ℓ(h_t)` with `Σ_ℓ k_ℓ ≡ 0` that leaves the sum, the divergence and every full-model
prediction identical while changing what removing a single lag does. The qualification travels
verbatim in every summary under `lag_readouts.qualification`, and the acceptance gate fails a run
that lost it.

**An unsupported bin is missing, not zero.** A band whose available-channel count is zero had
nothing to remove, and its margin is recorded as `null`. Every band reports its usable anchor and
channel counts beside its margin for exactly this reason.

**Neither number is a physiological delay.** The additional *neural* source receptive field is one
stored sample; the causal feature extraction upstream of the model still mixes raw history inside
every coefficient, over a span the feature geometry fixes and no model-side change reaches. The
encoder disclosure in every summary states both.

**A positive gap against the internal base is necessary and not sufficient.** The internal base can
change during joint training, so an improved internal gap can come from a weakened baseline. Reading
it requires an independently trained target-only comparator, which no single run of this pass
produces.

**The gap is reported and not gated.** Where an acceptable boundary sits is what the first real runs
measure. A provisional threshold would decide a pass or a fail on exactly the run that was going to
supply the answer, and nobody could tell a healthy model failing from a broken one passing.

---

## 5. What this architecture cannot report, and why

Seven analyses in the wider family read a tensor this architecture does not compute. Two of them are
in the shared registry and this package's binding removes them by name; the other five are the
lag-attentive cell's own extras and are simply never registered here. All seven, with the tensor
each would have needed, reach every summary under `excluded_analyses`, and how each came to be
absent under `excluded_analyses_mechanism`.

| Analysis | Mechanism | What it would have needed |
| --- | --- | --- |
| `attention` | removed | an attention distribution over lags |
| `lag_kl` | removed | a per-lag allocation of the divergence |
| `source_null` | never registered | a source pathway with parameters to encode a zeroed stream through |
| `occlusion` | never registered | the lag attention, to rebuild the full branch through |
| `lag_clocks` | never registered | the per-lag allocation, resolved against the clinical clocks |
| `lag_kld_scaled` | never registered | the same allocation over a partition of the lag axis |
| `lag_high_kl` | never registered | a per-anchor lag map to select anchors against |

None of them is handed a substitute. A proposal norm reported under an attention name is a per-lag
attribution that does not exist, and the acceptance gate refuses a summary carrying any of those key
names.

**The shared collection pass is not used at all**, and the reason is the anchor axis rather than the
attention keys alone. Every latent tensor this architecture produces is indexed by decoded anchor;
the shared pass pairs a dense stored-step support with a latent produced at every step, and its
per-batch readout requires eight attention-derived fields as well. What is reused instead is every
piece that says nothing about an architecture: the checkpoint and task loading, the override merge
and its schema, the per-anchor block score, the log-mean-likelihood, the derangement, the
recording-level bootstrap and the summary assembly.

---

## 6. Structural gates and where each is proved

The six gates the architecture specification requires before any training claim is read. Each row
names the test that proves it; all run in the package's own suite.

| Gate | Evidence |
| --- | --- |
| The prior sees no source values and retains every latent coordinate | `test_causality.py::test_the_metadata_clock_carries_no_source_value`, `::test_the_source_reaches_the_full_branch_and_never_the_base_branch`; `test_construct.py::test_the_metadata_clock_is_a_function_of_position_alone`, `::test_the_prior_head_is_built_without_its_own_clock_path` |
| Each source encoding is pointwise; each proposal reads one stored source time | `test_pointwise_source.py::test_the_encoding_is_pointwise_by_jacobian`, `::test_the_scalar_lift_stays_pointwise`; `test_lag_updates.py::test_a_proposal_reads_exactly_one_stored_source_time`, `::test_a_proposal_may_combine_channels_at_its_own_source_time` |
| No future value and no future validity mask enters prediction | `test_causality.py::test_resampling_the_strict_future_leaves_earlier_anchors_bitwise_unchanged`, `::test_no_future_validity_signal_can_enter_the_forward`, `::test_no_lag_reads_a_step_at_or_after_its_anchor` |
| Source-disabled and all-unavailable inputs reproduce the prior; a valid observed zero need not | `test_invariants.py::test_every_selector_off_reproduces_the_prior`, `::test_every_lag_unavailable_reproduces_the_prior_and_stays_finite`, `::test_the_zero_initialised_model_reproduces_the_prior_for_arbitrary_inputs`, `::test_an_observed_standardized_zero_is_not_treated_as_absence` |
| The source has no route around the latent | `test_invariants.py::test_the_decoder_is_invoked_twice_with_the_latent_and_the_persistence_input_only`, `::test_the_two_decoder_calls_share_one_module_and_its_weights`; `test_forward_contract.py::test_no_attention_shaped_key_is_emitted` |
| Shared noise, the divergence, the masks, the anchor indexing and the gradients behave as specified | `test_invariants.py::test_the_paired_sample_difference_matches_the_stated_formula`, `::test_the_predictive_gradient_reaches_the_final_source_projection_at_the_zero_start`, `::test_the_source_pathway_leaves_the_zero_start_after_a_few_steps`; `test_residual_kl.py::test_the_residual_divergence_equals_the_family_formula`; `test_objective.py::test_the_reconstruction_and_divergence_share_one_anchor_set`, `::test_the_objective_optimises_the_global_mean_under_uneven_ranks`; `test_forward_contract.py::test_the_anchor_axis_is_not_the_time_axis` |

Two further properties are gated on the run's own output rather than on the model, by
`verify.py`, and both can genuinely fail:

- the reference arms are exact, so every reported margin is a difference of predictions;
- no key anywhere in the summary names an attention distribution or a per-lag divergence allocation.

### 6.1 Which gates a comparator arm suspends, by declaration

The gates above are properties of the **recommended** architecture. A comparator arm exists to
violate exactly one of them, which is the only way a difference is attributable to that one thing --
and a suspended gate is worth nothing unless it is declared, so each is written into the run's own
`arm` block rather than left to be inferred from a missing column.

| Arm | Gate it suspends | Every other gate |
| --- | --- | --- |
| `source_stem: conv` | "each source encoding is pointwise": the stem's output at one step summarises a bounded window ending there, and the run discloses that reach as `encoder_disclosure.source_receptive_field_steps` | holds |
| `lag_fusion: attention` | "each proposal reads one stored source time": every lag still enters through its own key and value, but the aggregation over them is a learned normalised one rather than an explicit sum | holds |
| `source_values_withheld` | none. It withholds an input rather than relaxing a property, and every gate above is stronger on it than on the candidate | holds |
| `source_disabled` | none, vacuously: there is no source pathway to constrain | holds |

Two consequences are worth stating because they are what an operator would otherwise discover late.
A band narrower than the convolution stem's own reach is not resolving what its name says, since two
lags closer together than that reach are summaries of overlapping windows. And the arms are read as
a chain, each pair differing in one leaf -- the attention reference against the pointwise-attention
arm isolates the convolution, that arm against the candidate isolates the fusion, and the candidate
against the mean-only, capacity-control and target-only arms isolates the variance update, the
source values and the pathway.

`tests/test_arm_scoring.py` fits and scores each comparator through these entry points at fixture
scale and asserts the departures above are recorded rather than silent.

---

## 7. Several runs together

One pass scores one checkpoint. Three things the design requires cannot be read from one: whether
an effect survives repeated fits, whether the latent carries anything, and whether the result holds
on recordings no choice was made on. Two further passes do those, and both read artifacts rather
than checkpoints.

### 7.1 The latent probes

```
python -m teb_vae.lag_slot_transformer_cfs.eval.latent_probes \
    --checkpoint output/<run>/model_checkpoints/<name>.ckpt
```

Fits a frozen ridge probe from each of six latent readouts — the prior's and the full
distribution's means, scales and one shared-noise draw of each — onto the forecast block **relative
to the anchor's own stored values**, and scores it on recordings the fit never saw. The target is
anchor-relative because a probe scored against the raw block reports how much the future looks like
the present whatever the latent holds; it subtracts the anchor's stored values rather than the
decoder's learned persistence weights, so two arms' figures are comparable.

The split is a digest of the recording identifier, so every arm's probe is fitted on the same
cohort. The fit is streamed as second moments and never holds a row, which is what lets it use every
complete-coverage anchor rather than a subsample. The artifact is `latent_probes.json`, and the
protocol below attaches it to a run by the **checkpoint** both name rather than by the directory it
was written into.

What it does not say: a coefficient here is a lower bound on what the latent holds and not a
measurement of what the decoder uses, and the difference between the prior's and the full
distribution's figures is not a measurement of source information.

### 7.2 The acceptance protocol

```
python -m teb_vae.lag_slot_transformer_cfs.eval.acceptance \
    --runs output/<development evaluations> \
    --reference output/<frozen target-only>/eval_results/summary.json \
    --output acceptance.json
```

Reads every evaluation directory under a root, groups the runs by arm and **training** seed, and
reports what that evidence supports under one predeclaration. Like the single-run gate it needs no
numeric stack beyond the array library its interval already uses.

`configs/acceptance_plan.yaml` is that predeclaration: the seed minimum, the draw count the primary
comparisons are read at, the resamples, the five declared comparisons and the lag bands a search may
range over. Every record names the digest of the plan file it was read from. Nothing prevents the
plan being edited; the digest is what makes an edit visible, and the suite pins it.

Four things it does that a single run cannot:

| | |
| --- | --- |
| Several seeds | Per-recording values are averaged across an arm's training seeds and the interval is drawn over recordings **once**, around that average. Three scorings of one checkpoint are one seed, not three |
| Paired differences | Two arms scored on the same split under the same seed and draw count differ per recording, so every comparison is an interval on the differences rather than two overlapping intervals |
| Multiplicity | The five declared comparisons are reported at the nominal level. The lag bands are a search, so each is additionally reported at a family-adjusted level covering the family at once — and a band selected out of a simultaneously covered family keeps that coverage however it was chosen |
| The reserved partition | A confirmation set is checked against the selection set by the **recordings** both scored, not by the files they name: one fold's train partition holds another fold's test recordings |

Verdicts follow the single-run gate's convention. Structural ones can fail: a run that fails its own
gate, a band searched outside the declaration, a confirmation partition the selection has already
been scored on, and a base branch that has fallen confidently behind the frozen reference. Everything
else is reported with its interval and left ungated, because where an acceptable boundary sits is
what the runs are meant to measure.

---

## 8. Settings

Everything that shapes what a run measures lives in the override delta, `configs/eval_overrides.yaml`,
because that file is deep-merged into the run directory and is therefore the durable record. The
launch dicts are a convenience for getting a run started, and the resolved value of every one of
them reaches `summary.json` under `run.argument_sources`, so a run's provenance is recoverable from
its own output rather than from a shell history.

| Setting | Meaning |
| --- | --- |
| `eval_config.seed` | Seeds the latent draws and the cross-recording pairing, as two independent streams |
| `eval_config.num_mc_samples` | Draws `K`; `8`, `32` and `128` are the counts a finalist is scored at |
| `eval_config.bootstrap_resamples` | Resamples behind every interval, drawn over recordings |
| `eval_config.occlusion_bands` | The lag bands the suppression readout removes, inclusive, in stored steps back from the anchor |
| `eval_config.max_samples` | A cap on the segments the pass sees; `null` evaluates the whole split |
| `general_config.batch_size.test` | The loader's batch size; it also decides how often a batch admits a cross-recording pairing |

`eval_config.clock_margin_min_nats` ships unset and no verdict here reads it: it gates the
lag-attentive cells' availability-clock verdict, whose null arm re-encodes a zeroed stream through a
source pathway with parameters, and this architecture's has none.

---

## 9. What is not here

Training runs and their results. The machinery is what this package builds; running it against real
shards, comparing arms, driving the synthetic instruments and reaching an empirical acceptance are
separate activities, and none of them is settled by any code review.
