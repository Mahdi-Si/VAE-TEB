# Session 7 decision log: `lag_attn_transformer_cfs`, epoch 424 (s6, soft UA mask)

Every experiment has an entry written BEFORE it runs (hypothesis, prediction, metric, split, kill
criterion) and a result block written AFTER it. Killed hypotheses stay in this file.

## Standing rules

- **Split.** $h = \operatorname{crc32}(\mathrm{guid}) \bmod 4$. Code 0 = fit, code 2 = select, codes 1 and 3 = test.
  Every number that informs a decision comes from codes 0 and 2. The test half opens once, in
  Phase 5.
- **Membership.** `acidosis_no_cs` and `hie_no_cs` are unseen (the builder puts only healthy
  recordings in the pre-training set). `healthy_no_bg_no_cs` has unknown membership (its source
  folder on the server is not recorded), so its results carry the label "possibly seen".
- **Statistics.** Stand-in statistics. Report differences between arms, not absolute nats.
- **Intervals.** Recording-level paired bootstrap, 2000 resamples, 95% percentile interval
  (`s7common.paired_bootstrap`). A change wins only if the interval on its gain excludes 0 and no
  official verdict flips to FAIL.
- **Source gain.** $G = \mathrm{NLL}_{\text{base}} - \mathrm{NLL}_{\text{full}}$ per anchor on the scored cells, with mean
  decoding ($z = \mu^p$ against $z = \mu^q$), unless an entry names the Monte-Carlo gap.
- **Timeline.** The stored UP/FHR timeline is canonical.

---

## E0. Phase 0: provenance and soft-mask parameters

**Question.** Does the checkpoint load strictly? Which local recordings did the run see? Did the
three zero-initialised soft-mask parameters move?

**Prediction (from the prior notes, not trusted).** None stated for the soft-mask parameters.
A parameter that stayed within about 0.05 of zero counts as "did not learn".

**Result (2026-11-10).**
- Strict load passes. All 55 shared model keys match `resolved_config.yaml`. 4,176,469 parameters, epoch 424.
- Membership: acidosis 157 recordings unseen (fit 47, select 39, test 71); HIE 15 unseen
  (3, 7, 5); healthy 157 unknown (42, 32, 83). The trainer has no GUID rule; the split is the
  server folder (`pre_training_dataset/train` against `test`), filled by a seeded 90/10 shuffle.
- `source_invalid_key_bias` per head = [+0.104, +0.286, +0.093, +0.045]. It is small and
  POSITIVE: no head learned to discount invalid keys.
- `source_invalid_gate` weight per head = [-0.257, +0.094, -0.084, +0.230]. The invalid mass
  lies in [0, 1], so the largest possible shift of the gate logit is about 0.26.
- `source_adapter.quality_proj` Frobenius norm = 125.7 against 107.7 for the value projection of
  the same shape. The adapter reads the flags strongly.

**Decision.** The attention bias and the gate term of the soft mask did not learn a discount.
Only the adapter's flag input moved. The gate therefore cannot close on invalid attention mass;
whether the adapter flags close it indirectly is tested in E2.1 and E2.2.

---

## E1. Phase 1: UA gaps and interpolation

### E1.1 Gap audit (descriptive)

**Question.** How long are the invalid UA runs, which codes produce them, and where do they fall
relative to the contraction (rise, peak, fall, between)?

**Prediction (prior finding 2 and 3, to re-verify).** Most invalid runs are short: the median is
at most 8 s. TRANSIENT and ISLAND dominate the count. About 12% of contractions touch at least
one invalid sample, and about half of those have at most 8 s invalid.

**Metric.** Run table from `up_quality` with the mask `invalid_bits = 2299`. **Split.** Codes 0 and 2.

### E1.2 Interpolation tolerance (the user's hypothesis)

**Hypothesis (user).** Contractions are step-like, so the builder's bridge serves them well over
short gaps.

**Threshold, fixed before any result.** $X = 10\%$. A gap class (code type, length, contraction
phase, band) is *tolerable* when bridged UA with the flags set to valid costs less than 10% of
the source gain $G$ on the affected anchors, and the upper bound of the 95% interval is below 20%.
Affected anchors are those whose 38-step lag window holds at least one key step that the strict
rule marks invalid.

**Predictions.**
1. Raw level: for gaps of at most 8 s, the bridge RMSE is below 10% of the contraction
   amplitude at every phase.
2. Raw level: a gap at the peak fails first. At 30 s, the peak-amplitude error exceeds 20%.
3. Coefficient level: the fast bands recover within their own window; the slow bands
   (st29-31, ph6-9) stay off for minutes but with a small relative error.
4. Model level: for gaps of at most 16 s, bridged UA with valid flags costs less than 10% of $G$.
5. Model level: the current flags (strict rule) cost more gain than the bridge error itself for
   gaps of at most 16 s.

**Kill criterion.** If bridged UA with valid flags costs 10% of $G$ or more already at 4-8 s in
every phase, the hypothesis is dead and H1 and H2 die with it.

**Method constraint.** The gaps are injected into clean UA and rebuilt with the dataset
builder's own code (`hdf5_dataset/signal_quality.py::bridge` and the builder's coefficient
transform). With no gap, the rebuild must reproduce the stored `up`, `up_st` and `up_ph`; the
experiment is void otherwise.

**Split.** Codes 0 and 2.

### E1.3 Detector audit (added on the user's note that the detectors may be weak)

**Question.** Are the contraction detector (`lag_attn_cfs/eval/events.py::detect_contractions`,
which also sets the training reward through the event weight) and the deceleration detector good
enough to carry the phase analyses and the reward?

**Hypothesis H7.** The contraction detector misses or misplaces a material share of contractions,
so the event weight rewards the wrong anchors and every "contraction phase" table inherits the error.

**Predictions (falsifiable).**
1. Against a detector-free reference (a band-limited UA envelope with a robust baseline,
   thresholded on absolute rise), the detector's recall is below 80% or its onset error has an
   interquartile range above 20 s.
2. The detected rate is unstable under harmless changes: the same contraction seen in two
   overlapping segments is detected in only one of them in more than 15% of cases.
3. The source gain $G$ ranks anchors by a detector-free UA activity measure at least as well as
   by the detector's "seconds since onset" (Spearman correlation, recording-level interval).

**Kill criterion.** If recall is at least 90%, the onset interquartile range is at most 12 s and
the overlap agreement is at least 90%, H7 dies and the detectors stay.

**Consequence if H7 survives.** Every phase-conditioned result in Phases 1-2 is also reported
against the detector-free activity measure, and a reward built from that measure becomes a
candidate change for s7.

**Split.** Codes 0 and 2.

---

## E2. Phase 2: mechanism surgery (interventions on the epoch-424 weights, no training)

All E2 experiments use `p2_surgery.py`: one matched forward per batch, anchors every 5 steps,
mean decoding, the source pathway re-run under each arm. Anchor classes come from the 38-step
lag window of the anchor:

- **fully valid**: every key usable under the strict rule;
- **rho-only**: at least one key unusable, raw UA clean over the whole 152-s window;
- **short bridge**: raw invalid time in the window in (0, 16] s;
- **long gap**: raw invalid time in the window above 16 s.

Each class is also split by the reward: $\omega = 1$ (within 60 s after a detected onset, as
trained) against $\omega = 0$. **Split.** Codes 0 and 2. Unseen and possibly-seen cohorts are
reported separately.

### E2.1 Gate clamp (step 10)

**Hypothesis.** The learned gate is too strict on anchors with imperfect UA.

**Prediction.** On fully valid anchors, clamping $g = 1$ gains nothing (the gate is fitted there).
On rho-only and short-bridge anchors with $\omega = 1$, clamping $g = 1$ lowers $\mathrm{NLL}_{\text{full}}$
(interval excludes 0). Clamping $g = 0$ on anchors with $g \ge 0.2$ loses gain (sanity).

**Metric.** Paired $\Delta\mathrm{NLL}_{\text{full}}$ (clamp minus matched) per anchor class.

**Kill.** If no class gains from $g = 1$, "the gate is too strict" is dead at this checkpoint.

### E2.2 Validity flags (step 11; intervention form of H1 and H3)

**Arms.** (a) matched flags (per channel, $\rho \le 0.1$); (b) all flags valid; (c) slow third of the
channels valid, others matched; (d) fast and middle thirds valid, slow matched; (e) the all-42
rule written into every channel (a step is flagged in every channel when any channel fails).

**Predictions.**
1. H1 form: on rho-only and short-bridge anchors, arm (b) lowers $\mathrm{NLL}_{\text{full}}$ against (a).
2. H3 form: arm (c) recovers at least 70% of the gain of (b) on rho-only anchors, because the
   slow channels cause the rho-only state.
3. Arm (e) is worse than (a) on rho-only anchors: the per-channel flags already beat the
   all-channel rule.
4. Clock test: the source-null KL (values zeroed, flags kept) is at most 20% of the matched KL,
   and the part that the flags add (null KL with real flags minus null KL with valid flags) is
   at most 5% of the matched KL and does not rise toward delivery.

**Kill.** If arm (b) does not lower $\mathrm{NLL}_{\text{full}}$ on rho-only or short-bridge anchors, H1 has no support
at the intervention level. It then needs the Phase 1 tolerance table plus a pilot to survive.

### E2.3 Controls (step 15)

**Predictions.** Shuffled UA (another recording's UA and flags): $G \le 0$. Source-null: $G \le 0$.
Time shift of the UA stream by 60 s: $G$ falls by at least 30%. By 152 s (the whole lag window):
$G$ falls by at least 60%.

**Kill.** If shuffled UA keeps more than 50% of $G$, the gain is not specific to the recording's
UA, and every source claim in this log is void.

### E2 results (2026-11-10; codes 0 and 2; 103,936 anchors, 170 recordings; `out/phase2/surgery_summary.json`)

Unseen (acidosis + HIE) and possibly-seen (healthy) cohorts agree in sign and size on every line
below; the pooled numbers are quoted.

**Re-verified prior claims.** Gate on ($g \ge 0.2$) at 43.7% of anchors, which carry 97.8% of the KL
(prior claim 43% / 97.7%: confirmed). The reward $\omega = 1$ covers 29.7% of anchors as trained
and 34.5% without the `event_weight_ua_valid` rule, so the rule removes 13.9% of the reward mass
(prior claim "11.9% of contractions": same order, confirmed).

**E2.3 controls: PASS.** $G$ = 3.59 [2.97, 4.25] nats per anchor over all anchors and 9.18 [7.31, 11.34]
on $\omega = 1$. Shuffled UA: $G$ = -2.08 [-2.54, -1.51]. Shift by 60 s: -2.45 [-2.82, -2.13]. Shift by
152 s: -2.63 [-3.00, -2.28]. Source-null: 0.06 [-0.08, 0.21]. The gain needs this recording's UA at
the right time. One caveat: on $\omega = 1$ anchors the null posterior still gains 1.16 [0.91, 1.41]
nats (12.6% of $G$ there), a target-only shift that the posterior applies by default.

**E2.2 clock test: prediction 4 half FAILS.** The source-null KL is 1.04 nats against a matched KL
of 1.39: 75% of the all-anchor KL needs no UA values (3.01 against 1.07 on $\omega = 1$; 0.70 against
1.03 on $\omega = 0$, where the null gate rests at 0.33). `lambda_source_null: 1.0` did not remove it.
The flags do NOT act as a clock: they lower the null KL (flag part -0.20 [-0.22, -0.18]), and the
null KL falls toward delivery (1.06 above 4 h, 0.63 in the last 15 min).

**E2.1 gate clamp: hypothesis KILLED on rewarded anchors, ALIVE on dropped contractions.**
$g = 1$ makes $\mathrm{NLL}_{\text{full}}$ worse on every rewarded class (valid +3.50 [2.84, 4.16]; rho-only
+2.25 [0.29, 4.02]; short bridge +1.49 [0.17, 2.85]). It helps only where the reward rule dropped
the contraction: short bridge / dropped -3.06 [-6.06, -0.55], long gap / dropped -1.29 [-2.24, -0.35].

**E2.2 flags.**
1. H1 form on rho-only anchors: KILLED. All flags valid changes nothing there
   (-0.06 [-0.32, 0.17] on $\omega = 1$). The model already uses rho-only anchors ($g$ 0.44 against 0.50,
   $G$ 8.5 against 9.7).
2. H1 form on bridged contractions: SUPPORTED. On dropped-contraction anchors whose gap has ended,
   valid flags lift $G$ from 1.92 [0.84, 3.17] to 7.58 [4.26, 12.57] ($\Delta\mathrm{NLL}$ -5.66 [-10.60, -2.64]; gate 0.16 -> 0.48).
   By invalid time in the lag window: at most 8 s, $G$ 3.87 -> 12.07; 8-16 s, 0.76 -> 6.66; 16-40 s,
   0.30 -> 2.70; above 40 s, no gain (0.25 [-3.55, 4.52]). Clean rewarded anchors give 9.67.
3. Look-ahead warning: on dropped anchors INSIDE a running gap, valid flags give $G$ = 15.14
   [8.99, 21.87], above the clean reference. The bridge is drawn toward samples after the gap, so
   this number contains future UA. A tolerance rule must not mark a running gap valid.
4. H3 (slow bands cause the loss): KILLED. The slow third of the channels recovers 5% of the
   effect; the fast and middle thirds recover 93%.
5. The all-42 rule written into every channel is worse than the per-channel flags
   (+5.30 [3.06, 7.87] on rho-only / $\omega = 1$). The s6 per-channel flags already beat it.

**Decisions.**
- The binding fault is the reward rule, not the gate and not the slow bands: a flagged step
  never earns reward, so the model reads a flag as "no reward here" and closes the gate. H2 (tolerant
  `event_weight_ua_valid`) and H1 in its bridged-contraction form go to the pilots as ONE linked
  change, with the tolerance from Phase 1 and a causal guard for running gaps.
- H3 is dead. No per-band key rule goes to s7.
- New hypothesis H8: the default posterior shift (null KL 75% of KL) is the reason the
  coupling-against-clock verdict is INCONCLUSIVE. Prediction: a larger `lambda_source_null` (1 -> 3)
  cuts the null KL by at least 30% on the select code and keeps at least 90% of $G$ on $\omega = 1$.
  Kill: $G$ on $\omega = 1$ falls by more than 10%, or the null KL falls by less than 15%.

### E2.4 The tolerance rule as an intervention (no training)

**Rule under test.** `s7_tolerance.tolerant_flags`: an invalid run of at most $T_{\mathrm{tol}}$ seconds is
ignored by the flags once it has ended plus the 4-s bridge slope window. Long runs and running
runs keep the builder's flags. Arms: $T_{\mathrm{tol}} \in \{8, 16, 32, 64\}$ s against the matched flags.

**Predictions.**
1. On dropped-contraction anchors whose gap has ended, $T_{\mathrm{tol}} = 16$ s lowers $\mathrm{NLL}_{\text{full}}$ (interval
   excludes 0) and recovers at least 50% of what "all flags valid" recovered there (-5.66 nats).
2. On anchors inside a running gap, the rule changes little (the guard works): $|\Delta\mathrm{NLL}|$ is below
   20% of the all-valid effect there.
3. Over ALL anchors, no arm raises $\mathrm{NLL}_{\text{full}}$ (interval upper bound below +0.1 nat).
4. The gain grows from 8 to 16 s and adds less from 32 to 64 s.

**Kill.** If $T_{\mathrm{tol}} = 16$ s recovers less than 25% of the all-valid effect on ended-gap dropped
anchors, the causal rule is too strict to be worth a training rung in this form.

**Split.** Codes 0 and 2.

---

## E4. Phase 4: pilot fine-tunes (pre-registered 2026-11-10, before any pilot ran)

**Harness.** `p4_pilots.py` on top of `s7_pilot.py`. Start: epoch-424 weights. Fit: split code 0
(92 recordings, all three cohorts). Budget, the same for every arm: 300 optimizer steps of
2 x 64 segments, AdamW, learning rate 1e-4, the training objective of the task at beta = 1 with
the tiled anchors and source dropout. Seeds 0 and 1; a seed fixes the segment order, so two arms
with one seed see the same batches. Score: split code 2 (78 recordings), mean decoding, anchors
every 5 steps.

**Reference.** The `control` arm (no change, same fine-tune) of the same seed. `epoch424` (no
training) is reported beside it, so the effect of fine-tuning itself is visible.

**Primary metric.** Paired $\Delta\mathrm{NLL}_{\text{full}}$ per anchor (arm minus control), averaged over the two seeds per
anchor, recording-level bootstrap. $\mathrm{NLL}_{\text{full}}$ is the log score of the source-conditioned forecast under
the arm's own density, so arms with different densities are comparable.

**Secondary metrics.** Source gain $G$ on all anchors and on anchors within 60 s after any detected
onset (reward without the UA rule, so every arm is read on the same anchors); KL and source-null
KL; coverage at the 1, 2 and 3 sigma levels and the official tail statistic (worst relative
tail error, tolerance 0.5); MSE per block against the own-channel AR(6) and the UA-lag ridge
fitted on code 0.

**A change wins** when the interval of its primary $\Delta\mathrm{NLL}_{\text{full}}$ lies below 0, OR when its own
pre-registered target metric improves with an interval that excludes 0 while $\Delta\mathrm{NLL}_{\text{full}}$ is not worse
(upper bound below +0.5 nat, about 0.05% of the block NLL). It must also keep $G > 0$ and must not
move the tail statistic across 0.5 in the wrong direction.

**Arms and their own predictions.**

| Arm | Hypothesis | Own target metric and prediction | Kill |
|---|---|---|---|
| `tolflags<T>+tolreward<T>` | H1 + H2 linked | $G$ on dropped-contraction anchors (gap ended) rises by at least 2 nats; $\mathrm{NLL}_{\text{full}}$ on all anchors falls | $G$ there rises by less than 1 nat, or $\mathrm{NLL}_{\text{full}}$ rises |
| `tolreward<T>` | H2 alone | same anchors: rises, but by less than the linked arm | no rise |
| `tolflags<T>` | H1 alone | same anchors: rises without training the reward | no rise |
| `rewardfree` | H2 without a tolerance | $G$ on dropped anchors rises, but $G$ inside running gaps rises more than in the linked arm (look-ahead) | - |
| `null3` | H8 | source-null KL falls by at least 30%; $G$ on rewarded anchors keeps at least 90% | see E2 results |
| `decfloor9` | H5a | share of phase cells on the floor falls below 20%; 1-sigma coverage of the phase block moves toward 0.683; $\mathrm{NLL}_{\text{full}}$ falls | $\mathrm{NLL}_{\text{full}}$ rises |
| `studentt<nu>` | H5b | worst relative tail error falls below 0.5; $\mathrm{NLL}_{\text{full}}$ falls | tail error stays above 0.5 |
| `decfloor9+studentt<nu>` | H5 | both of the above | - |
| `phweight3` | H6a | phase MSE at $\tau = 0$ falls against the control; scattering MSE not worse | phase MSE does not fall |

$T$ comes from the Phase 1 tolerance table and E2.4 and is fixed before the pilots start. $\nu$ comes
from the innovation kurtosis measured on code 0 at epoch 424. H3 is dead (E2.2) and gets no arm.
H4, H6b and H7 get an arm only if Phase 2 (adapter RMS, phase latent share) or the detector audit
supports them; their entries are added before they run.

**Stacking.** Winners are added one at a time in the order of their primary effect. A change stays
only if the stack with it beats the stack without it by the same rule. Stop after two
consecutive failures.

### E2.4 first pass: VOID (rule bug, found before any decision used it)

The first pass read the channel windows from the model's warm-up vector, which is counted on the
trimmed grid. Sixteen fast channels had a window of 0 steps, so the long-gap check never applied
to them and their flags turned valid inside long running gaps. Symptoms: a large effect inside
running gaps at 8 s (-10.85 nats) that vanished at 64 s, and a curve that fell with the
tolerance. The rule now reads the builder's own `causal_warmup_steps` (5 to 293 steps) and the
pass was rerun. The numbers of the first pass are not used anywhere.

### E1.3 result: detector audit (2026-11-10; codes 0 and 2; `out/detectors/REPORT_detectors.json`)

**Contraction detector: WEAK as trained, by the pre-registered rule.**
- Recall against the detector-free reference is 0.754 [0.73, 0.78] (prediction 1: below 80%, HOLDS).
  The peak finder alone reaches 0.898 on reference contractions with a fully valid span. The
  shortfall comes from the any-invalid-sample drop rule, which removes 14.1% [12.5, 15.8] of the
  events and of the reward mass; 50.7% [46.9, 54.9] of the dropped events hold at most 8 s invalid.
- Onset error: median 0 s, interquartile range 5.3 s. Overlap agreement 0.985 after re-cutting
  (prediction 2: unstable in more than 15% of cases, REFUTED).
- The UA mask fires at the apex: invalid rate 4.9% there against 2.2% between contractions
  (+2.7 points [1.9, 3.5], mostly TRANSIENT). The quality rule flags real contraction peaks.
- 12% of onsets and 32% of ends sit on the 60-s search bound; the sigma-relative prominence makes a
  quiet segment produce false events and a tone step hide real ones.

**Deceleration detector: WEAK.** Onset interquartile range 27-29 s against a NICHD-rule reference.
Only 12.2% [10.5, 14.0] of its events meet the NICHD definition; 83% are less than 15 bpm deep.
It is used only in evaluation tables, never in the objective.

**Improved detectors (`detectors7.py`).** `detect_contractions_v7` (absolute thresholds, 120-s
bound, tolerant validity): recall 0.817 on code 2 (+8.8 points [7.4, 10.4]), precision 0.831
(+12.2 points), verdict MIXED. `detect_decelerations_v7` is not clearly better.

**Decision.**
- H7 SURVIVES in a narrow form: the detector's fault is the drop rule (the same fault as H2), plus
  the sigma-relative threshold. Prediction 3 (gain ranks by UA activity) is tested on the dump.
- New pilot arm `detv`: reward from `detect_contractions_v7`. Own prediction: $G$ on dropped-contraction
  anchors rises as in the tolerance arm, and $G$ on the anchors that only v7 rewards is positive.
  It needs a scale from the statistics file, which is a larger production change than a tolerance
  key, so it must beat `tolreward<T>` by an interval that excludes 0 to replace it.
- Every "deceleration band" number of the existing evaluation is to be read as "FHR trough of at
  least 10 bpm prominence", not as a NICHD deceleration. No s7 change depends on it.

### E2.4 result (corrected pass, 2026-11-10): the step-level causal rule is KILLED

- Prediction 2 (guard): HOLDS. Inside running gaps the rule changes nothing (+0.07 [-0.03, 0.17]).
- Prediction 3 (no harm overall): HOLDS pooled (at most +0.02 [-0.05, 0.08]); on the unseen cohorts
  16 s gives +0.07 [0.00, 0.15], at the edge.
- Prediction 1 (recover at least 50% at dropped contractions): FAILS. 16 s gives -1.45 [-4.14, 0.17],
  26% of the all-valid effect (-5.66), interval includes 0; on the unseen cohorts -0.09 [-1.08, 1.05].
- Prediction 4 (grows with the tolerance): FAILS. 8, 16, 32 and 64 s give the same effect.

**Why.** The rule validates a step only after the gap has ended, so the steps INSIDE an ended gap
keep their invalid flags for ever. Those are the keys that hold the bridged contraction. A flag
is a property of a step, but look-ahead is a property of an (anchor, key) pair: a key inside a gap
is non-causal only for anchors at which the gap is still running.

**Decision.** The step-level rule does not go to the pilots. It is replaced by E2.5.

### E2.5 Tolerance with per-anchor key removal (no training)

**Rule.** (1) Flags: $v'_{s,c} = v_{s,c} \lor [\text{no long-run sample in the window of channel } c \text{ at step } s]$,
with no exception for steps inside a short run. (2) Keys: for anchor $t$ and key step $s$, the key is
REMOVED from the attention (hard, as in the hard mask mode) when some invalid run $[a, b]$ of any
length has begun by step $s$ and is still running at $t$: $a < D(s+1)$ and $b + g + 1 \ge D(t+1)$.
An anchor with no key left has its gate closed. No anchor can then read a bridged value that
depends on UA after the anchor, for short and for long runs. This also removes a look-ahead path
that the s6 soft mask has today (bridged values of a running gap reach the adapter with a flag).

**Arms.** `tolhard<T>` for $T \in \{8, 16, 32, 64\}$ s, and `hardonly` (key removal, flags unchanged).

**Predictions.**
1. On dropped-contraction anchors whose gap has ended, `tolhard16` recovers at least 50% of the
   all-valid effect (-5.66 nats), interval excludes 0, in the pooled data AND with the same sign
   on the unseen cohorts.
2. On anchors inside a running gap, `tolhard` gives no gain above +/-1 nat (the all-valid arm gave
   -13.45 there through look-ahead).
3. Over all anchors no arm raises $\mathrm{NLL}_{\text{full}}$ (upper bound below +0.1 nat).
4. `hardonly` costs at most 0.1 nat over all anchors: the look-ahead path of s6 is not what the
   matched model lives on.
5. The effect grows from 8 to 32 s.

**Kill.** If `tolhard16` recovers less than 25% pooled, or has the wrong sign on the unseen cohorts,
H1 dies as an input rule and only the reward rule (H2) goes to the pilots.

**Split.** Codes 0 and 2.

### E2.5 result (2026-11-10; codes 0 and 2): the per-anchor rule SURVIVES

| Scope | all flags valid | key removal only | `tolhard8` | `tolhard16` | `tolhard32` | `tolhard64` |
|---|---|---|---|---|---|---|
| dropped contraction, gap ended (2692 anchors) | -5.66 [-10.60, -2.64] | -0.53 [-1.25, 0.01] | -4.12 [-8.73, -1.44] | -4.81 [-9.44, -2.01] | -4.93 [-9.59, -2.09] | -4.94 [-9.59, -2.06] |
| same, unseen cohorts (1650) | -3.63 [-5.67, -1.74] | -0.68 [-1.88, 0.07] | -2.32 [-4.17, -0.64] | -3.05 [-4.96, -1.34] | -3.22 [-5.10, -1.46] | -3.12 [-5.06, -1.25] |
| dropped contraction, inside a running gap (1051) | -13.45 [-19.67, -7.65] | -0.39 [-2.01, 1.32] | -0.60 [-2.37, 1.26] | -1.42 [-3.62, 0.72] | -2.25 [-4.70, 0.21] | -2.44 [-5.07, 0.19] |
| all anchors (103,936) | -0.27 [-0.42, -0.15] | -0.00 [-0.04, 0.03] | -0.10 [-0.22, -0.00] | -0.11 [-0.23, -0.01] | -0.11 [-0.23, -0.01] | -0.10 [-0.21, -0.01] |

Values are $\Delta\mathrm{NLL}_{\text{full}}$ against the matched flags, in nats per anchor; negative is better.

1. Prediction 1: HOLDS. `tolhard16` recovers 85% of the all-valid effect pooled and 84% on the
   unseen cohorts, both intervals exclude 0.
2. Prediction 2: HOLDS within the interval. Inside a running gap the effect is -1.42 [-3.62, 0.72]
   against -13.45 for all-valid; the look-ahead gain is gone. The point estimate is above the
   1-nat band, and it grows with the tolerance (older short runs turn valid), so 16 s is preferred
   over 32 or 64 s.
3. Prediction 3: HOLDS. Every arm lowers $\mathrm{NLL}_{\text{full}}$ over all anchors.
4. Prediction 4: HOLDS. Key removal alone costs nothing (-0.00 [-0.04, 0.03]). The epoch-424 model
   does not live on the look-ahead path.
5. Prediction 5: HOLDS from 8 to 16 s, then saturates.

**Decision.** H1 survives in this form: tolerant flags plus per-anchor key removal. The pilots
run it at 8 s (the conservative length of the synthetic-gap table) and at 16 s (where the
real-anchor effect saturates), each linked with the tolerant reward.

### E3.1 Innovation tails on the fit code (`p3_innovation_tails.py`, code 0; fixes $\nu$ before the pilots)

- Cells off the floor (scattering 98%, phase 50%): a scaled Student-$t$ fits best at $\nu = 8$ with scale
  0.83-0.86; gain 0.017-0.022 nats per cell over the Gaussian density. Excess kurtosis 1.7
  (scattering) and 11 (phase).
- Cells on the decoder floor: 49.5% of the scored phase cells and 1.7% of the scattering cells.
  Their mean $z^2$ is 0.125 (phase) and 0.217 (scattering): the floor binds, and the fitted scale is
  0.19 of the floor in the phase block. A lower floor is worth up to 0.78 nats per floor cell.
- The Student-$t$ arm uses $\nu = 8$. The floor arm uses -9.

### E2.6 Descriptive mechanisms from the per-anchor dump (steps 12-14; codes 0 and 2; 423,085 anchors)

Dump: `out/dump/` (958,392 anchors over all codes; statistics read on codes 0 and 2 only).
Report: `out/phase2/REPORT_descriptive.json`. Unseen and possibly-seen cohorts agree within intervals.

**Prior claims.**
- Gate on at 43.4% [42.4, 44.5] of anchors with 97.7% [97.5, 97.8] of the KL: CONFIRMED.
- Lag 0 carries 48.1% [47.6, 48.7] of the gate-on KL: CONFIRMED.
- Source-adapter RMS 9.6 against a target of 0.98 (finding 5): REFUTED at epoch 424. The source
  adapter output has RMS 10.05 and the target adapter 11.48, ratio 0.875 [0.868, 0.883].
- "The phase block gets nothing from the latent" (finding 5): REFUTED at epoch 424. 28% of the
  variance of the phase forecast mean comes from the latent and 62% from the skip path; the model
  beats a ridge from $\mu^p$ on the phase block (held-out $R^2$ 0.57 against 0.19).
- Decoder floor binding: CONFIRMED for the phase block (56% of scored phase cells on the floor,
  mean $z^2$ 0.17 there; scattering 2.6%). Heavy tails: CONFIRMED (excess kurtosis about 78 in the
  phase block).

**Attention (step 12).** Each head sits on a fixed lag. Head 1 reads lag 0 at 98% of anchors and
carries 68.5% of the KL; head 3 reads about 10 s and carries 28%; heads 0 (about 90 s) and 2
(about 33 s) carry 1.6% each. Only 2.1% of the KL lies beyond lag 10. The attention does not track
the UA peak (share within 2 steps of the peak is at chance for heads 1-3; head 0 reaches 0.172
[0.166, 0.177] against 0.118 by chance). The KL is highest when the anchor is within 4 s of the
contraction peak (KL-weighted offset -7.8 s [-8.6, -7.0]); the gate is 0.60 in the peak phase and
0.18 between contractions. On the stored timeline this is the early-deceleration pattern
(response at the peak), not the late one (30 s or more after it).

**Latent (step 13).** Three active dimensions (20, 28, 61); dimensions 28 and 61 hold 91% of the
KL. Cross-fitted linear probes of $\mu^q - \mu^p$: UA state $R^2$ 0.22 [0.20, 0.24]; FHR state and
clock 0.01 or less.

**Read-outs (step 14).** The implied multi-step 95% interval covers 0.83 (scattering) and 0.86
(phase), down to 0.79-0.81 at the far horizon. A ridge from $\mu^p$ beats the decoder mean on the
scattering block (0.469 against 0.419).

**H7 prediction 3.** The gate does NOT rise with a detector-free UA activity measure: it is lowest
in the top activity quintile (0.22), and inside the rise phase it falls from 0.51 to 0.24 as
activity rises. The model follows the reward window (60 s after onset), not the uterine pressure.

**Decisions.**
- H4 (normalise the source adapter): KILLED before any pilot. There is no scale mismatch to fix.
- H6b (take the phase block off the latent path): KILLED. The latent carries 28% of it.
  H6a (phase weight) stays as a pilot arm.
- New observation, not yet a hypothesis for s7: the model uses UA almost only at lags 0 and 10 s
  and the reward window ends before a late response could show. A longer reward reach is a
  candidate for a later rung; it is not tested here because rung 4 shortened the reach on
  purpose and this session has no evidence against that choice.

### Linear baselines for the pilots (`p4_baselines.py`; fit on code 0, scored on code 2)

MSE per block in normalised units: own-channel AR(6) 0.644 (scattering) and 0.534 (phase);
all-channel ridge 0.573 and 0.542; ridge with UA lags 0.564 and 0.551. The UA lags help the
scattering block (-0.008 [-0.012, -0.004]) and hurt the phase block (+0.009 [0.007, 0.012]); pooled
effect zero. A linear filter finds almost no UA gain, so the model's gain is not linear.

### E1.1 and E1.2 results: gap audit and interpolation tolerance (2026-11-10; codes 0 and 2; `out/phase1/REPORT_phase1.json`)

**Method check.** The rebuild with the builder's own chain reproduces the stored `up` and `up_ph`
exactly and `up_st` to $2.4 \times 10^{-7}$ on 240 segments; the $\rho$ fields match in 239 of 240. The
experiment is valid. A bridged value in a gap $[s, e)$ reads $x[e]$ and the slope over the next 4 s,
so the look-ahead of the bridge is "gap end plus 4 s", as the key-removal rule assumes.

**E1.1 gap audit (9,132 runs without overlap double counts; 2.6% of samples invalid).**
- Length: median 5 s, 90th percentile 33 s. Shares at most 4 / 8 / 16 / 32 / 64 s: 33% / 74% / 83% / 89% / 97%.
  Prediction "median at most 8 s": HOLDS.
- Codes: TRANSIENT is the main code in 92% of runs (80% of invalid time); CODE 2.6%, STEP 1.9%,
  MISSING 1.6%, FLOOR 1.0%, ISLAND 0.2%. Prediction "TRANSIENT and ISLAND dominate": half right.
- Position: 53% between contractions, the rest split evenly over rise, peak and fall.
- 14.0% of contractions touch invalid UA; 51% of those hold at most 8 s. Prior finding 3: CONFIRMED in size.
- Prior finding 2 ("slow channels stay contaminated for minutes after short gaps"): REFUTED for
  short gaps. After a gap of at most 16 s no slow channel passes $\rho = 0.1$. After an 8-s gap the strict
  all-channel rule marks the source unusable for a median of 48 s, and the fast and middle
  channels st0-st18 cause it. The claim is true only for gaps of 32 s or more (6-10 min).

**E1.2 bridge error.**
- Raw level: RMSE below 11% of the contraction amplitude up to 32 s on rise and fall and up to 64 s
  between contractions. At the peak 11.6% at 16 s and 23% at 32 s. The bridge is monotone between
  its edge values, so it cannot draw a crest: peak-amplitude error -2.3%, -5.9%, -10.9%, -28.8%, -65%
  at 4, 8, 16, 32, 64 s.
- Coefficient level: fast channels are off by more than 10% of their sd for a median 24-28 s after
  the gap; middle channels for 32 s after an 8-s gap although $\rho$ never passes 0.1; slow channels
  not at all after gaps of at most 16 s.
- Model level, cost as a share of the source gain on affected anchors:

| Gap | bridged, flags valid | bridged, strict flags |
|---|---|---|
| 4 s | 0.5% [-2.7, 4.3] | 19.7% [10, 29] |
| 8 s | 1.0% [-2.5, 4.6] | 16.3% [8.7, 23] |
| 16 s | 1.0% [-2.2, 4.6] | 11.7% [6.9, 17] |
| 32 s | 4.6% [1.0, 8.8] | 15.8% |
| 64 s | 12.6% [4.8, 22] | 15.2% |
| 128 s | 32% [18, 51] | 20% |

  In the first 60 s after the gap the bridge costs 10%, 23% and 47% at 4, 8 and 16 s, against 101%,
  85% and 68% for the strict flags.

**Tolerance table at $X = 10\%$ (upper bound below 20%).** Rise 4 s (8 s fails on the interval only,
16 s passes, 32 s fails); peak 32 s; fall 64 s; between 64 s; all positions pooled 32 s; fast band 64 s;
middle and slow bands at least 128 s; TRANSIENT (observed mix) 16 s; other codes undecidable
(4-19 recordings each).

**Predictions.** 1 HOLDS on the means. 2 HOLDS (peak worst; -28.8% [-31.4, -26.2] at 32 s).
3 HOLDS in part (fast yes; slow channels are not off after short gaps and not small after long
ones). 4 HOLDS pooled (fails in the first 60 s after the gap). 5 HOLDS. Kill criterion not met.

**Step 9, the step-shape hypothesis.** It holds on flanks and rests and for the middle and slow
bands at any length. It fails at the crest beyond 8-16 s (the bridge cannot overshoot its edges),
on the rise beyond 4-8 s at the model level, in the fast band, and in the first minute after a gap.

**Decision.** $T_{\mathrm{tol}} = 16$ s is fixed for s7. Three independent lines agree: 83% of the runs are
at most 16 s; the synthetic table tolerates TRANSIENT runs up to 16 s (pooled 32 s); and the
real-anchor effect of E2.5 saturates at 16-32 s while the residual inside running gaps grows
with the tolerance. The 8-s arm stays in the pilots as the conservative alternative.

### E4 interim (H8, arm `null3`): KILLED in the "raise the weight" form

`null3` is identical to the control to four decimals on both seeds (KL 1.3108 against 1.3108,
source-null KL 1.0448 against 1.0448), although the loss shows that the weight was applied (total
loss differs by exactly twice the penalty). The reconstruction block is about -1100 nats per
anchor and its gradient swamps a 1-nat KL term. Prediction "null KL falls by at least 30%": FAILS
(0%). This also explains why `lambda_source_null: 1.0` left 75% of the KL in place during the s6
run: the penalty never had a gradient that mattered. A weight large enough to matter would be
in the hundreds and would act on the matched KL too, so it is not tried.

### E2.7 Null-referenced posterior (structural form of H8; intervention, no training)

**Change.** The posterior's shift from the prior is taken relative to the source-null posterior
(UA values zero, flags kept): $\mu' = \mu^p + (\mu^q - \mu^{q,\varnothing})$ and the same for the log-variance.
A zeroed source then gives $q' = p$ exactly, so the source-null KL is 0 by construction and the
coupling-against-clock comparison has no clock term left.

**Predictions.**
1. The KL falls by at least 0.5 nats over all anchors.
2. On rewarded anchors the gain keeps at least 85% ($G' \ge 7.8$ against 9.18).
3. Over all anchors $G'$ is not below $G - 0.3$ nats.
4. $G' - \mathrm{KL}' > G - \mathrm{KL}$ over all anchors.

**Kill.** $G'$ on rewarded anchors below 70% of $G$: the default shift is then doing real work that
the null reference cannot give back without training, and the change needs a pilot to be judged.

**Split.** Codes 0 and 2.

### E2.7 result (2026-11-10; codes 0 and 2): the null-referenced posterior is KILLED as an intervention

| Scope | $G$ matched | $G$ null-referenced | KL matched / null / null-referenced |
|---|---|---|---|
| all anchors | 3.59 [2.97, 4.25] | 2.39 [1.96, 2.83] | 1.39 / 1.04 / 1.65 |
| rewarded | 9.18 [7.31, 11.33] | 6.25 [5.01, 7.59] | 3.01 / 1.07 / 2.25 |
| unrewarded | 1.22 [0.94, 1.50] | 0.75 [0.55, 0.94] | 0.70 / 1.03 / 1.40 |

1. Prediction 1 (KL falls by 0.5 nats): FAILS; the KL rises by 0.26.
2. Prediction 2 (rewarded gain keeps 85%): FAILS; it keeps 68%, below the 70% kill line.
3. Prediction 3 and 4: FAIL ($G$ falls by 1.2; $G - \mathrm{KL}$ falls from 2.20 to 0.73).

**Why.** The null posterior is not a constant offset under the matched one. With a zeroed source
the gate rests at 0.33, but on unrewarded anchors the matched gate is lower (0.20) and the matched
KL (0.70) is below the null KL (1.03). Subtracting the null posterior therefore creates a shift
where the matched posterior had none. A zeroed feature vector is also an input the model never
met in training, so the "null KL" measures the response to an unfamiliar input as much as a
clock.

**Decision.** H8 is dead for s7 in both forms. The INCONCLUSIVE verdict
`coupling_exceeds_availability_clock` stays a known risk of the run. A null-referenced head
could still work when trained from the start, but nothing in this session supports it.

### E4 round 1a results (2026-11-10; select code 2, 78 recordings, 48,020 anchors; two seeds)

Differences are arm minus control, per anchor, with recording-level intervals. `epoch424` (no
training) is +2.87 [+2.42, +3.24] nats worse than the control, so fine-tuning itself is worth
about 3 nats on local data; every arm is read against the control.

| Arm | $\Delta\mathrm{NLL}_{\text{full}}$ | $\Delta G$ | tail statistic | floor share st / ph | 1-sigma coverage st / ph |
|---|---|---|---|---|---|
| control | 0 | 0 ($G$ = 3.62) | 1.14 | 0.02 / 0.50 | 0.724 / 0.856 |
| `null3` | -0.00 [-0.00, +0.00] | +0.00 | 1.14 | 0.02 / 0.50 | 0.724 / 0.856 |
| `phweight3` | +0.09 [+0.01, +0.17] | -0.04 [-0.08, +0.01] | 1.14 | 0.02 / 0.50 | 0.724 / 0.856 |
| `studentt8` | -7.67 [-10.31, -5.46] | -1.60 [-2.06, -1.15] | 0.35 | 0.03 / 0.54 | 0.695 / 0.850 |
| `decfloor9` | -133.25 [-156.01, -110.35] | +0.56 [+0.28, +0.85] | 2.60 | 0.00 / 0.00 | 0.719 / 0.789 |
| `decfloor9+studentt8` | -186.94 [-209.74, -163.48] | -1.27 [-1.72, -0.84] | 0.99 | 0.00 / 0.00 | 0.684 / 0.755 |

Likelihood-free check (MSE of the base branch minus MSE of the full branch, scattering block,
anchors within 60 s after an onset, as a share of the base MSE): control 1.06%, `studentt8` 1.46%,
`decfloor9` 0.06% (interval includes 0), `decfloor9+studentt8` 0.85%. The source gain in nats is
mostly a gain in predictive variance, not in the mean.

Scattering MSE against the own-channel AR(6): control 0.889, `decfloor9` 0.935,
`decfloor9+studentt8` 0.931. Phase MSE ratio 0.753 in every arm.

**Verdicts.**
- H8 (`null3`): KILLED (see the interim entry above).
- H6a (`phweight3`): KILLED. The phase MSE at $\tau = 0$ does not move (ratio to AR(6) 1.069 in both
  arms) and $\mathrm{NLL}_{\text{full}}$ is slightly worse. H6 is dead in both forms.
- H5b (`studentt8`): WINS by the pre-registered rule. $\mathrm{NLL}_{\text{full}}$ falls and the tail statistic passes
  (0.35 against a tolerance of 0.5). $G$ in nats falls by 44%, because a heavy-tailed density pays
  less for large errors; the mean improvement from the source does not fall (it rises).
- H5a (`decfloor9`): WINS on $\mathrm{NLL}_{\text{full}}$ by a wide margin and empties the floor as predicted. Two
  costs: alone it makes the tail statistic worse (2.60), and the scattering MSE rises by 5%
  against AR(6), because cells with a very small variance now carry more weight in the loss.
- H5 (both): the stack is the best arm on $\mathrm{NLL}_{\text{full}}$, reaches nominal 1-sigma coverage on the
  scattering block (0.684) and brings the tail statistic to 0.99, which is better than the
  control but still above 0.5. Calibration is predicted to improve, not yet to pass.

### E4 round 1b results (2026-11-10; select code 2; two seeds; arm minus control)

| Arm | $\Delta\mathrm{NLL}_{\text{full}}$, all anchors | $\Delta G$ at dropped contractions, gap ended | same, unseen cohorts | $\Delta G$ on clean rewarded anchors |
|---|---|---|---|---|
| `tolhard16` | -0.00 [-0.17, +0.18] | +3.19 [+1.24, +5.41] | +3.59 [+1.46, +6.32] | -0.10 [-0.14, -0.07] |
| `tolhard16+tolreward16` | -0.01 [-0.17, +0.15] | +3.16 [+1.21, +5.39] | +3.60 [+1.45, +6.31] | -0.23 [-0.30, -0.16] |
| `tolhard8+tolreward8` | -0.01 [-0.16, +0.16] | +2.84 [+0.99, +5.03] | +3.21 [+1.16, +5.84] | -0.21 [-0.28, -0.16] |
| `tolreward16` | +0.01 [-0.05, +0.06] | +0.04 [-0.15, +0.25] | -0.03 [-0.23, +0.19] | -0.13 [-0.17, -0.08] |
| `detv` | +0.14 [+0.06, +0.22] | -0.05 [-0.14, +0.05] | -0.06 [-0.15, +0.02] | -0.27 [-0.38, -0.18] |
| `rewardfree` | +0.11 [+0.03, +0.18] | -0.01 [-0.18, +0.20] | -0.07 [-0.27, +0.11] | -0.37 [-0.54, -0.24] |

On anchors within 60 s after any onset, `tolhard16` lowers $\mathrm{NLL}_{\text{full}}$ by 0.25 [0.03, 0.48]. Inside
running gaps no tolerance arm shows a gain whose interval excludes 0. Calibration and MSE do not
move in any arm of this round.

**Verdicts.**
- H1 (`tolhard16`: tolerant flags plus per-anchor key removal): WINS on its own target metric
  (at least +2 nats, interval excludes 0, same sign and size on the unseen cohorts) with no cost
  over all anchors. It lifts $G$ at dropped contractions from 1.66 to 4.85. $T_{\mathrm{tol}} = 16$ s gives
  more than 8 s, as E2.5 predicted.
- H2 (`tolreward16`): KILLED by its own criterion (no rise at dropped contractions), alone and on
  top of `tolhard16` (+3.16 against +3.19). In 300 steps the reward does not teach the model what
  the flags already allow. The production key `event_weight_ua_tolerance_s` stays available and
  inert; see the run predictions for when to switch it on.
- H7 (`detv`): KILLED. The session-7 detector makes the forecast worse over all anchors and does
  not raise the gain anywhere. The shipped detector stays.
- `rewardfree`: KILLED (worse over all anchors, lower gain on clean rewarded anchors).

**Stack order for round 2.** `decfloor9`, then `studentt8`, then `tolhard16`.

### E4 round 2 results: stacking (2026-11-10; select code 2; two seeds)

| Step | Stack | Reference | $\Delta\mathrm{NLL}_{\text{full}}$, all anchors | Own target |
|---|---|---|---|---|
| 1 | `decfloor9` | control | -133.25 [-156.01, -110.35] | floor share 0.50 -> 0.00 (phase) |
| 2 | `decfloor9+studentt8` | `decfloor9` | -53.69 [-59.18, -48.66] | tail statistic 2.60 -> 0.99 |
| 3 | `decfloor9+studentt8+tolhard16` | `decfloor9+studentt8` | -0.11 [-0.20, -0.03] | $G$ at dropped contractions +2.30 [+0.84, +3.85]; unseen +2.44 [+0.88, +4.33] |
| - | `...+tolreward16` | stack of step 3 | -0.42 [-0.46, -0.39] | $G$ at dropped contractions -0.02 [-0.07, +0.03] |

Every step beats the stack before it, so the stop condition (two consecutive failures) was not
reached. The stack of step 3 against the control: $\mathrm{NLL}_{\text{full}}$ -187.05 [-209.76, -163.61]; tail
statistic 0.98 against 1.14; scattering 1-sigma coverage 0.684 against 0.724; $G$ 2.39 against 3.62
(-1.23 [-1.68, -0.81], the Student-$t$ effect on nats); source-null KL -0.21 [-0.24, -0.19];
scattering MSE against AR(6) 0.931 against 0.889.

**The reward tolerance again.** On the stack it lowers $\mathrm{NLL}_{\text{full}}$ by 0.42 nats, the same on every
scope (rewarded, unrewarded, clean, dropped), with no change of $G$. Its own pre-registered target
(the gain at dropped contractions) fails for the second time. A uniform shift of both branches is
not the mechanism of H2, so H2 stays killed and the key stays inert in s7. The observation is
recorded as a candidate for a later rung.

**Final stack for s7.** `decoder_logvar_clamp: [-9, 3]`, `forecast_student_nu: 8`,
`source_running_gap_keys: remove`, `source_valid_tolerance_s: 16`.

---

## E5. Phase 5: the single test-half pass (pre-registered 2026-11-10, before the test half was read)

**Production path.** The four mechanisms were ported into the model and task code behind
inert-by-default keys and checked against the session-7 prototypes with zero difference
(`p5_check_production.py`, `p5_check_student_t.py`). The warm start
`lag-attn-trf-cfs-epoch=424-s7init.ckpt` (`init_s7.py`) carries the keys; untrained, it gives the
same KL, gate and source-null KL as the patched pilot stack. The warm start is not
function-preserving in the decoder variance (mean log-variance change 0.51 on the select code).

**Arms, all scored on split codes 1 and 3 (164 recordings), same budget and seeds as the pilots,
fine-tuned on code 0.** `epoch424` (untrained); `control` (epoch 424, fine-tuned, no change);
`s7stack` (the warm start, fine-tuned through the production code, no patch).

**Predictions on the test half.**
1. $\mathrm{NLL}_{\text{full}}$: `s7stack` minus `control` below 0, interval excludes 0, size about -185 nats (between -230
   and -140).
2. $G$ at dropped contractions with the gap ended: rises by at least 1 nat, interval excludes 0, same
   sign on the unseen cohorts.
3. Calibration: the tail statistic of `s7stack` is below that of `control`; the phase floor share
   is below 0.05; the scattering 1-sigma coverage is within 0.02 of 0.683.
4. $G$ over all anchors stays positive for `s7stack` (interval excludes 0), although lower in nats
   than for `control`.
5. Scattering MSE against AR(6) is worse than `control` by at most 6% (the cost seen on the select
   code was 4.7%).
6. Official verdicts (`eval.run` on the seed-0 checkpoints, test half): no verdict that passes for
   `control` fails for `s7stack`. `calibration_near_nominal` is predicted to stay FAIL for both,
   with a smaller tail error for `s7stack`.

**Launch rule.** s7 is recommended only if predictions 1, 2, 4 and 6 hold. If 1 or 6 fails, the
recommendation is "no new run". If 2 fails, the tolerance keys leave s7 and the rest stands only
if 1 and 6 hold without them.

### E5 results: the test half (opened once, 2026-11-10; split codes 1 and 3; 159 recordings, 93,380 anchors; two seeds)

(The pre-registration above says 164 recordings; the test half holds 159: acidosis 71, HIE 5,
healthy 83.)

Differences are `s7stack` minus `control`, per anchor, recording-level intervals.

| Scope | $\Delta\mathrm{NLL}_{\text{full}}$ | $\Delta G$ | $G$ control -> s7 | gate control -> s7 |
|---|---|---|---|---|
| all anchors | -197.30 [-210.23, -184.62] | -0.71 [-1.06, -0.42] | 3.27 -> 2.57 | 0.27 -> 0.27 |
| all, unseen cohorts | -191.80 [-211.66, -172.45] | -0.79 [-1.22, -0.40] | 3.27 -> 2.48 | 0.26 -> 0.27 |
| all, possibly seen | -202.56 [-219.64, -183.74] | -0.63 [-1.21, -0.18] | 3.28 -> 2.65 | 0.28 -> 0.28 |
| within 60 s after an onset | -193.69 [-208.08, -180.39] | -1.36 [-2.25, -0.63] | 7.68 -> 6.32 | 0.46 -> 0.46 |
| dropped contraction, gap ended | -128.64 [-148.05, -108.38] | +1.51 [+0.51, +2.55] | 2.29 -> 3.80 | 0.18 -> 0.36 |
| same, unseen cohorts | -115.92 [-138.02, -91.34] | +1.01 [-0.10, +2.26] | 2.48 -> 3.48 | 0.18 -> 0.35 |
| dropped contraction, inside a running gap | -81.49 [-107.37, -59.20] | +3.17 [+1.02, +5.94] | 2.64 -> 5.81 | 0.14 -> 0.36 |

| Read-out | control | s7stack |
|---|---|---|
| tail statistic (tolerance 0.5) | 1.18 | 0.92 |
| 1-sigma coverage, scattering / phase (nominal 0.683) | 0.722 / 0.857 | 0.683 / 0.758 |
| share of cells on the decoder floor, scattering / phase | 0.02 / 0.51 | 0.00 / 0.00 |
| scattering MSE against AR(6) | 0.884 | 0.931 |
| phase MSE against AR(6) | 0.760 | 0.760 |
| MSE gain of the full branch, scattering, 60 s after onset | 1.17% | 0.95% |
| source-null KL | 1.00 | 0.75 |

`s7stack` $G$ over all anchors: 2.57 [2.15, 2.97]; unseen 2.48 [1.90, 3.05].

**Predictions.**
1. $\mathrm{NLL}_{\text{full}}$: HOLDS (-197.3, inside the predicted range, interval excludes 0).
2. Gain at dropped contractions: HOLDS (+1.51, interval excludes 0; unseen cohorts +1.01 with
   the same sign, interval [-0.10, +2.26]). The size is about half of the select-code value.
3. Calibration: HOLDS (tail statistic lower, floor empty, scattering coverage at nominal).
4. $G$ stays positive: HOLDS.
5. Scattering MSE cost at most 6%: HOLDS (5.3%). The cost is real and repeats.

**Note on anchors inside a running gap.** The gain there rises to 5.81. These anchors cannot read
the keys of the running run (they are removed), so the gain comes from the keys before the run,
which the control ignored because the flagged keys beside them closed its gate (0.14 against
0.36). It stays below the gain on clean rewarded anchors (7.12), unlike the look-ahead case of
E2.2 (15.1 against 9.7).

**Prediction 6, official verdicts** (`eval.run` on the seed-0 fine-tuned checkpoints, test half,
159 recordings; `out/phase5/official/verdicts.json`): HOLDS. Both arms have the same status on all
ten verdicts.

| Verdict | control | s7stack | Values (control -> s7stack) |
|---|---|---|---|
| predictive_improvement | PASS | PASS | Monte-Carlo gap 2.65 -> 2.32 |
| source_margin_positive | PASS | PASS | margin 5.48 -> 4.91 |
| source_specificity | PASS | PASS | shuffle penalty 2.83 -> 2.59 |
| prior_carries_target_state | PASS | PASS | |
| latent_not_collapsed | PASS | PASS | 3 active dimensions in both |
| prior_variance_not_pinned | PASS | PASS | |
| decoder_variance_not_pinned | PASS | PASS | floor share 0.344 -> 0.024 |
| calibration_near_nominal | FAIL | FAIL | worst tail error 1.18 -> 0.89; 1-sigma coverage 0.786 -> 0.719 |
| coupling_exceeds_availability_clock | INCONCLUSIVE | INCONCLUSIVE | KL minus source-null KL 0.21 [0.16, 0.25] -> 0.41 [0.35, 0.46] |
| anchor_geometry_intact | PASS | PASS | |

**Launch rule.** Predictions 1, 2, 4 and 6 hold, so s7 is recommended for launch.

---

## E6. Phase 6: unit tests (run once, at the end)

Selection: the tests of the changed modules in `lag_attn_transformer_cfs`, `lag_attn_cfs`,
`lag_attn_crws` and `lag_attn_rws` (25 files, 714 tests). Result: 699 pass. 11 fail and 4 error;
all 15 also fail on a clean checkout of the last commit (checked in a scratch worktree), so this
session did not cause them: three config-sibling comparisons in `test_config_load.py`, one
tracked-metric test in `test_task.py`, one override-delta test in `test_eval_binding.py`, one
sample-page test, one `lag_attn_crws` forwarded-set test, four metric-reassembly tests in
`lag_attn_rws/tests/test_objective.py`, and four fixture errors in the `lag_attn_cfs` eval tests.
Two failures were caused here and fixed: the s7 config set a constant KL weight, which
`test_sweep_configs.py` refuses for every arm; the config now keeps the default KL warm-up, as
rung 6 did. The new files `test_ua_tolerance.py` and `test_student_t_likelihood.py` pass.

---

## E7. Predictions for the full s7 run (written before the run; the run confirms or refutes them)

All sizes are relative to the rung-6 run at epoch 424 on the same validation data. One rung-6
epoch is 236 optimizer steps; the learning-rate ramp ends after about 9 epochs and the KL warm-up
after 50. A pilot fine-tune was about 4% of one epoch of data, so the forecast effects should show
within the first epochs.

| # | Metric | Direction and size | By | Refuted when |
|---|---|---|---|---|
| 1 | `val/nll_full_block` | lower by 150-220 nats per anchor block | epoch 10 | lower by less than 100 at epoch 20 |
| 2 | `logvar_full_floor_frac` | from about 0.34 to below 0.05 | epoch 5 | above 0.15 at epoch 20 |
| 3 | calibration verdict | tail error between 0.6 and 1.0 (rung 6: 0.99), 1-sigma coverage at most 0.73 (rung 6: 0.79); status still FAIL | first full evaluation | tail error above 1.2 |
| 4 | `val/pred_gap_mean` | positive, 65-85% of the rung-6 value | epoch 60 | at or below 0, or below 40%, after the KL warm-up |
| 5 | gain and gate at contractions with a short invalid UA run (anchors within 60 s after an onset that `event_weight_ua_valid` drops) | gain at least 1.5 times rung 6; gate at least 0.30 (rung 6: 0.16-0.18) | epoch 60 | gate below 0.22: then set `event_weight_ua_tolerance_s: 16.0` |
| 6 | scattering MSE against the own-channel AR(6) | ratio at most 0.95 (rung 6: 0.88-0.89) | epoch 60 | ratio above 1.0: then set `decoder_logvar_clamp: [-8.0, 3.0]` and rebuild the warm start |
| 7 | `coupling_minus_clock` | 0.3-0.5 nats (rung 6: 0.21-0.22); verdict still INCONCLUSIVE | epoch 60 | below 0.15 |
| 8 | `source_conditioned_kl_raw` | rises during the KL warm-up, back to 1.1-1.5 nats | epoch 80 | above 3 nats at epoch 100 |
| 9 | official verdicts | the rung-6 statuses: eight PASS, calibration FAIL, clock INCONCLUSIVE | first full evaluation | any PASS turns FAIL |
| 10 | phase MSE against AR(6) | unchanged (0.75-0.76) | epoch 60 | worse by more than 3% |

## Open risks

- **Stand-in statistics and local data.** Every number here comes from 329 local recordings
  under stand-in normalisation, and from fine-tunes of about 4% of one epoch. The directions are
  tested; the sizes over a full run are predictions.
- **Membership of the healthy shard is unknown.** The unseen cohorts (acidosis, HIE) show the same
  sign and nearly the same size on every result, so the conclusions do not rest on the healthy shard.
- **The scattering MSE cost of the lower floor repeats** (about 5% against AR(6) on both halves).
- **The source gain in nats falls under the Student-t density** (to 79% on the test half), although
  the mean improvement from the source holds on the phase block and falls slightly on the
  scattering block (0.95% against 1.17% of the base MSE after an onset).
- **75% of the KL still needs no UA values** (67% for the stack). Neither a larger penalty weight nor
  a null-referenced posterior fixed it without training from the start.
- **The warm start is not function-preserving** in the decoder variance; the first epochs will show
  a worse forecast NLL than epoch 424 (about 14 nats at step 0) before the fall.
- **The reward still drops contractions with a short invalid run.** The tolerant flags make the
  model use them anyway in a short fine-tune; a long run could unlearn that (prediction 5).
- **The model reads UA almost only at lags 0 and 10 s** and its KL peaks at the contraction peak.
  A late-deceleration response would need a longer reward reach; s7 does not address this.
- **Pre-existing test failures** (15 in the selection) remain in the repository.


## E8. Masked boxes near contractions (pre-registered 2026-11-10, after the s7 commit, before any FHR or model result)

**Origin (user).** Some UA traces jump shortly after what looks like a contraction onset, hold a
flat top (a box), and fall back to the rest level. The quality mask marks the whole box, and the
bridge then draws a monotone line through it. Does the box carry information about the FHR, and
should the transfer-entropy estimate use it?

**What was looked at before this text.** UA only: the lengths, heights and ceiling shares of the
masked runs, and 16 plots of raw UA, stored UA and FHR. No FHR statistic and no model output was
computed.

**Box event (fixed).** A maximal invalid run of `up_quality` on split codes 0 and 2 that meets all
of these conditions:

1. `TRANSIENT` on more than 90% of its samples, no other invalid bit, and no raw value above 127 or below 0;
2. length $16 < L \le 64$ s (longer than the s7 tolerance; the builder closes a box within 60 s);
3. height $h = \operatorname{median}(x_{\text{inside}}) - r \ge 10$ units, where the rest level $r$ is
   the 10th percentile of the valid raw UA within 180 s on both sides;
4. return: the median valid raw UA from 5 to 30 s after the run is at most $r + 0.5\,h$;
5. at least 60 s of the segment on both sides. Events are de-duplicated by recording and absolute start.

The scan gave 648 such runs in about 100 recordings.

### E8.1 Model-free FHR response

**Hypothesis.** A box event is a real uterine event with a distorted shape, so the FHR responds
to it as it responds to a clean contraction.

**Metric.** For an event at time $t_0$, with the valid FHR $f$ smoothed by a 10-s median,
the dip is $D = \bar f_{[t_0-60,\,t_0-10]} - \min_{\tau \in [0, 90]} \tilde f(t_0 + \tau)$ in bpm and
the absolute response is $A = \operatorname{mean}_{\tau \in [0, 90]} |f(t_0+\tau) - \bar f_{[t_0-60,\,t_0-10]}|$.
An event counts when at least 70% of the FHR samples are valid in both windows. Three groups in
the same recordings: box events ($t_0$ = run start), clean contractions (detector events with no
invalid UA sample in their span; $t_0$ = onset) and random times (at least 90 s from every
contraction span and every invalid run). The statistic is the recording-level mean difference,
with a bootstrap over recordings (2000 resamples).

**Predictions.** (1) $D_{\text{box}} - D_{\text{random}} > 0$ and the interval excludes 0.
(2) $D_{\text{box}} - D_{\text{random}}$ is at least half of $D_{\text{clean}} - D_{\text{random}}$.

**Common-cause check.** The share of invalid FHR samples in $[t_0, t_0 + 90]$ s per group. If the
box group exceeds the clean group by more than 10 points, prediction 1 is repeated on events with
at least 95% valid FHR, and that result decides.

**Kill.** The interval of $D_{\text{box}} - D_{\text{random}}$ includes 0 (after the common-cause
rule). Then the mask is right and the boxes stay out.

### E8.2 Model-level value of the box (epoch 424, intervention, no training)

**Hypothesis.** The raw box gives the model forecast gain that the bridge removes.

**Variants** of the UA source of each segment that holds a box event; only box runs change, every
other invalid run keeps its stored values and strict flags:

- `strict`: stored values (bridged), stored flags. This is rung 6.
- `erased`: stored values (bridged), flags recomputed with the box runs removed from the invalid
  mask. The event is erased but the flags are valid. This is what a tolerance of 64 s would do.
- `raw`: values rebuilt by the builder's chain with the box runs removed from the invalid mask, so
  the raw box passes; the same flags as `erased`.
- `slew`: as `raw`, but inside each box run the raw UA is replaced by a causal slew-limited copy,
  $y[n] = y[n-1] + \operatorname{clip}(x[n] - y[n-1], \pm 1)$ per sample (4 units/s), which removes the
  sharp edges and keeps the timing.

**Metric.** Source gain $G = \mathrm{NLL}_{\text{base}} - \mathrm{NLL}_{\text{full}}$ with mean decoding
on affected anchors (the 38-step key window holds at least one step of a box run), by timing
(inside the run, 0-60 s after it, 60-152 s after it). Paired bootstrap over recordings.

**Predictions.** (1) $G_{\text{raw}} - G_{\text{erased}} > 0$ on affected anchors, interval excludes 0.
(2) $G_{\text{raw}} - G_{\text{strict}} > 0$. (3) `slew` is at least as good as `raw`.

**Caveats fixed beforehand.** The network never trained on a box, so a null result here is weak
evidence against the box and a positive result is a lower bound. Inside a run, `erased` reads the
first valid sample after the run through the bridge, so it has a look-ahead advantage over `raw`.

**Kill.** Both `raw` and `slew` fail prediction 1.

### E8 decision rule

- E8.1 killed: no change, and the boxes stay masked.
- E8.1 passes: the boxes are real events. A change of the stored features belongs to the dataset
  builder, and it needs new shards and a pilot fine-tune on them before it enters a rung. s7
  launches unchanged unless a pilot on rebuilt features passes on the select code in this session.

### E8.1 result (2026-11-10; codes 0 and 2; `out/phase6/box_fhr_response.json`): the FHR responds to a box more than to a clean contraction

649 box events in 104 recordings (acidosis 388, healthy 219, HIE 42). Median length 32 s, median
height 74 units above the rest. 61% hold the ceiling value on at least 30% of their samples, and
68% lie in the second stage.

| Group | Events measured | Recordings | Mean dip $D$ (bpm) | Invalid FHR share in $[t_0, t_0+90]$ s |
|---|---|---|---|---|
| box | 524 | 96 | 19.2 | 13.8% |
| clean contraction | 18,920 | 170 | 9.1 | 3.6% |
| random, quiet | 9,169 | 168 | 6.7 | 3.6% |
| random, any | 102,513 | 170 | 7.7 | 3.2% |

| Recording-level difference of $D$ | bpm |
|---|---|
| box minus random quiet | +10.71 [7.80, 13.61] |
| clean minus random quiet | +2.50 [1.60, 3.30] |
| box minus clean | +8.78 [6.15, 11.60] |
| box minus random quiet, events with at least 95% valid FHR | +9.73 [7.07, 12.87] |
| box minus random any, first stage only (204 events) | +4.54 [1.88, 7.51] |
| box minus random any, second stage only (443 events) | +14.16 [10.35, 18.04] |
| box minus random any, ceiling share below 0.3 (250 events) | +7.33 [4.64, 10.17] |

**Predictions.** (1) HOLDS. (2) HOLDS: the box effect is 5.6 times the clean-contraction effect
[3.6, 13.1]. The invalid FHR share of the box group exceeds the clean group by 10.2 points, so the
common-cause rule applies; the comparison on events with at least 95% valid FHR decides, and it
holds. **Not killed.**

**Reading.** The masked boxes are the UA events with the largest FHR response in the data, in
both stages and with or without the ceiling. Most are bearing-down efforts or clamped crests. One
confound stays open: in the second stage the FHR sensor can pick up the maternal heart rate
during a push, which looks like a deceleration. The first-stage result, where this is rare, is
smaller but clear.

### E8.2 result (2026-11-10; codes 0 and 2; 306 segments; `out/phase6/box_model.json`): the raw box gives the epoch-424 model a gain that the bridge removes

Checks: the rebuild reproduces the stored `up_st` exactly; the flags built here equal the loader's;
$\mathrm{NLL}_{\text{base}}$ does not change across variants.

| Scope | Anchors (recordings) | $G$ strict | $G$ erased | $G$ raw | $G$ slew |
|---|---|---|---|---|---|
| affected | 7,747 (87) | 0.77 | 5.10 | 8.39 | 5.00 |
| affected, inside the run | 1,188 (63) | 0.32 | 29.95 | 50.32 | 30.06 |
| affected, 0-60 s after | 2,803 (76) | 0.05 | 0.58 | 1.61 | 0.37 |
| affected, 60-152 s after | 3,756 (81) | 1.44 | 0.62 | 0.18 | 0.52 |
| not affected | 19,994 (101) | 4.73 | 4.67 | 4.75 | 4.74 |

| Difference on affected anchors | nats |
|---|---|
| raw minus erased | +3.29 [1.05, 5.80] |
| raw minus strict | +7.62 [4.47, 10.95] |
| slew minus raw | -3.39 [-5.82, -1.21] |
| raw minus erased, inside the run | +20.37 [9.33, 34.85] |
| raw minus erased, 0-152 s after the run | +0.19 [-0.66, 1.05] |

**Predictions.** (1) HOLDS. (2) HOLDS. (3) REFUTED: the slew-limited copy loses what the raw box
adds, so the sharp shape is the information and smoothing is the wrong treatment. **Not killed.**

**Reading.** The whole effect sits at anchors inside the box. There the raw box gives 50 nats with
no look-ahead, against 30 nats for the bridge, which reads the first valid sample after the run,
and 0.3 nats for rung 6. After the run the variants do not differ. The network never trained on a
box, so these sizes are lower bounds.

**A fault found on the way.** The dataset builder module sets `torch.backends.cudnn.enabled =
False` when it is imported. A probe that builds features and runs the network in one process then
runs the network about 300 times slower. `p1_rebuild.Rebuilder` now restores the switch and turns
cuDNN off only around the builder's transform; the rebuild is still exact. Earlier probes that
used the rebuilder (Phase 1, T4) were slow but their numbers are not affected.

### E8.3 Pilot fine-tune on rebuilt features (pre-registered 2026-11-10, before any pilot result)

**Hypothesis.** Rung 7 forecasts better near a box when it trains on UA features in which the
box runs are valid and hold the raw values.

**Arms** (`p6_box_pilot.py`; rung-7 warm start; the fixed pilot budget; seeds 0 and 1; fit on code
0; scored on code 2):

- `s7ref`: the stored segments.
- `s7box`: the fit and scored segments with a box event are rebuilt by the builder's chain with
  the box runs removed from the invalid mask (values, flags, raw validity and the UA trace that
  the event weight reads).
- `s7ref_onbox`: the `s7ref` network scored on the rebuilt segments (no training on them).

**Predictions** (differences `s7box` minus `s7ref`, recording-level intervals):

1. On affected anchors, $\Delta\mathrm{NLL}_{\text{full}} < 0$ and the interval excludes 0.
2. On affected anchors, $\Delta G > 0$ and the interval excludes 0.
3. No harm: on the anchors that are not affected, the interval of $\Delta\mathrm{NLL}_{\text{full}}$
   includes 0 or lies below it.
4. Training adds to the data effect: `s7box` minus `s7ref_onbox` on affected anchors is negative
   (direction only).

**Kill.** Prediction 1 fails, or the interval of prediction 3 lies above 0.

**What a pass allows.** The official evaluation needs shards, and the test half was opened once
for the s7 decision, so neither can be run for this arm here. A pass therefore supports a builder
change and new shards; it does not put the change into the s7 files of this session.

### E8.3 result (2026-11-11; select code 2; 78 recordings, 48,020 anchors; two seeds; `out/phase6/box_pilot.json`): the pilot PASSES

The self-check of the field construction is exact (all six fields equal the loader's on stored
data). 145 fit segments and 161 scored segments were replaced.

| Scope | Anchors (recordings) | $\Delta\mathrm{NLL}_{\text{full}}$, `s7box` minus `s7ref` | $G$ ref -> box | gate ref -> box |
|---|---|---|---|---|
| all | 48,020 (78) | -0.23 [-0.31, -0.15] | 2.37 -> 2.50 | 0.27 -> 0.28 |
| affected | 1,505 (51) | -4.24 [-6.32, -2.40] | 0.53 -> 4.70 | 0.13 -> 0.35 |
| affected, unseen cohorts | 1,029 (35) | -3.33 [-5.46, -0.94] | 0.76 -> 4.04 | 0.13 -> 0.35 |
| affected, inside the run | 308 (48) | -21.74 [-31.61, -13.74] | 1.61 -> 23.13 | 0.27 -> 0.76 |
| affected, 0-152 s after | 1,197 (51) | +0.27 [-0.54, 1.13] | 0.26 -> -0.04 | 0.10 -> 0.24 |
| not affected | 46,515 (78) | -0.10 [-0.12, -0.08] | 2.43 -> 2.43 | 0.27 -> 0.27 |

Inside the run the scattering MSE of the full branch falls to 0.946 of the reference, so the
gain is in the mean forecast too. The calibration does not move (tail error 0.99 and 1.01
against 0.99 and 1.02; 1-sigma coverage equal to three decimals).

**Predictions.** (1) HOLDS. (2) HOLDS ($\Delta G$ +4.17 [2.33, 6.25]). (3) HOLDS: no harm on the
anchors that are not affected. (4) Direction only, and not clear: `s7box` minus `s7ref_onbox` on
affected anchors is -0.04 [-0.09, 0.02]. **Not killed.**

**Reading.** The rung-7 network uses a box as soon as the data shows it. `s7ref_onbox` (no
training on boxes) reaches the same gain as `s7box`. So a rung-7 run on the present shards loses
nothing that a later switch to rebuilt shards cannot recover. The gain is at anchors inside the
box; after the box the forecast does not change in 300 steps.

**A bound on the look-ahead of the rule.** The rule decides after the run whether the box
returned, so a valid flag inside a running box tells the model one binary fact about the future.
78% of the TRANSIENT-only runs of 16-64 s pass the rule, so that fact carries at most 0.53 nats
of entropy per event, against a gain of 22 nats per anchor inside the run. The raw values
themselves are causal: no bridge writes them.

### E8.4 The builder option (2026-11-11; `p6_box_builder_check.py`; `out/phase6/box_builder_check.json`)

The rule is now in the dataset builder as `QualityConfig.box_valid` (off by default), with a new
valid bit `BOX = 4096`, and both builders take `quality_valid_boxes` (off by default).

- With the option off, `ua_quality` does not call the rule, and the look-ahead stays 180 s.
- With the option on, the bits equal the bits with the option off except in the runs that
  `p6_box_events.box_runs` selects: 0 of 3,712 segments differ. 630 runs turn valid.
- The option on drops the level shift of a neighbouring invalid span in 6 segments. This is
  correct: the valid box samples now count in the builder's return test.
- A re-run of the rules on a stored segment differs from the stored bits in 653 segments (the
  file keeps the raw UA as `float16`); 91.8% of the stored box samples turn valid in the re-run.
- Unit tests: five new tests in `hdf5_dataset/tests/test_signal_quality_rules.py`. The dataset
  tests give 234 passes; two import tests of `test_causal_torch.py` and the collection of
  `test_causal_loader.py` fail on the last commit too.

### E8 decision

Boxes that return to the rest are real UA events with the largest FHR response in the data, and
rung 7 forecasts better with them. The change is a change of the stored features, so it enters
through a new dataset build with `quality_valid_boxes: True`, not through the s7 config.

**The s7 files do not change.** The official verdicts and the test half were not run for this
arm: the official evaluation needs rebuilt shards, and the test half was opened once for the s7
decision. Those two checks belong to the first evaluation on rebuilt shards.

**Predictions for a run on rebuilt shards** (against the same network on the present shards):
the gain on anchors inside a box rises by at least 10 nats; the forecast NLL over all anchors
falls by 0.1 to 0.4 nats; the calibration and the ten s7 predictions of E7 do not move. Refuted
when the gain inside a box rises by less than 5 nats, or the NLL over anchors without a box in
the key window rises.

**Open point.** In the second stage the FHR sensor can record the maternal heart rate during a
push, and that looks like a deceleration. The first-stage boxes show the effect too (+4.5 bpm
[1.9, 7.5]), so the finding does not rest on the second stage, but the size there may be
inflated.

## E9. Final pass before the s7 launch (2026-11-11)

| Check | Result |
|---|---|
| s7 config against the config of the epoch-424 run | differs only in the four rung-7 keys, the two inert keys, the run names and the warm-start path |
| Warm start rebuilt from the committed files with `init_s7 --mode build` | 386 tensors, largest difference 0.0 against the existing file; digest verified; strict load passes |
| Tests of the changed model code (12 files of `lag_attn_transformer_cfs/tests`) | 305 pass; the 5 failures are those recorded in E6 |
| Words that the stored-timeline rule forbids, in every file of the commit and of this pass | none |
| Markdown references in docstrings or comments of the Python files | none |
| Entry points without defaults | none |
| `session7/out/` (9.5 GB) | now in `.gitignore`; it was not ignored, and `git add -A` would have staged it |
| cuDNN switch | `p1_rebuild.Rebuilder` no longer leaves cuDNN off for the process (see E8.2) |
