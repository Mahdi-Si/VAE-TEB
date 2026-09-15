# Figure guide

One entry per figure a scoring pass writes into `eval_results/figures/`, and one per figure the
acceptance pass draws when asked to: what it shows, what its axes are, and how it is misread.
`EVAL.md` beside this file is everything that is not a figure.

Five rules apply to every figure here.

**Every figure is drawn from the summary, never from the tensors.** A figure is a picture of a
number `summary.json` carries -- an interval, a curve, a margin -- so a figure and the number beside
it cannot disagree, and the whole set can be redrawn from a finished directory on a box with no
checkpoint, no shard and no `torch`.

**One axis per panel.** A score and a count never share a frame on two y-axes; a quantity in a
different unit gets its own panel.

**Every interval that exists is drawn, and every margin's interval is the paired one.** Two arms
scored on the same recordings under the same draws differ per recording, so the interval on a
margin is the interval of those differences, drawn once over recordings. Two overlapping intervals
of the two arms are wider by exactly the shared variation the pairing removes and are not what a
claim about a margin rests on.

**Identity is text, colour is family.** On a dot plot the y label names the arm and the colour says
whether it is a matched branch (blue), a reference identity (grey), a lag band (green) or a source
control (amber). On a curve figure the declared lag bands take fixed colours in declaration order --
amber, green, purple, vermilion -- and a legend names them.

**An empty panel says so.** A block the run did not produce -- a target-only arm has no bands and no
lag profile, a normalised fusion has no latent profile -- draws a sentence saying what is absent
rather than an empty frame.

**Every lag axis is stored-coefficient time.** Lag $\ell$ names the source coefficient stored $\ell$
steps before the anchor, and the seconds axis above it is that many stored steps. It is not a
physiological delay: the causal feature extraction upstream of the model mixes raw history inside
every coefficient over a span the feature geometry fixes, and the qualification the lag readouts
must be read with is printed under every lag figure.

The figures share the evaluation packages' publication style, decided once in
`teb_vae/lag_attn/eval/figures.py`: a double-column width, a 7 pt serif type scale, open frames,
unframed legends placed in headroom above the data, and the Okabe-Ito colour-blind-safe palette.
Panels of a multi-panel figure carry bold lowercase letters (**a**, **b**, ...) in row-major order,
which is how the entries below refer to them; a qualification printed under a figure is a footnote
set at 6 pt with its own reserved room, never across an axis label. Panel titles are short noun
phrases: what a panel means is in its axis labels and here.

---

## The scoring pass

### `headline_arms`

Two dot plots.

*Left: each arm's own score.* Every scored arm's equal-recording predictive score in nats per anchor,
lower is better, with its recording-level bootstrap interval. Listed by family: the two matched
branches, the band suppressions, the silence identity, the source controls. The number beside each
row is its point estimate. The intervals here are wide on purpose: they carry the whole
recording-to-recording spread, which is what makes the right panel necessary.

*Right: margins, paired over recordings.* The gap (base − full) first, then every intervened arm's
margin against the full branch with the paired interval of the per-recording differences. A positive
margin means the fitted model predicts worse under the intervention. Three rows are identities that
verify the intervention path rather than measurements: `suppress:none` is exactly zero, and
`suppress:all` and `silence` equal the gap exactly.

**How it is misread.** The gap is necessary and not sufficient: a candidate's own base branch can
weaken during joint training, so a positive gap is read beside an independently trained target-only
reference, which the acceptance figures carry and this one cannot. And the left panel's overlapping
intervals say nothing about the margins on the right, for the pairing reason above.

### `pred_gap_recordings`

Two histograms over recordings. *Left*, the predictive gap per recording with the zero line and the
median: an interval on the mean can exclude zero on a population where a third of the recordings
sit on the other side, and this is where that shows. *Right*, the effective draw count of the full
branch per recording, $1 / \sum_k \alpha_k^2$ over the $K$ draws, with $K$ marked: a mass near one is a
warning that $K$ was too small for those recordings, and a mass near $K$ says the draws agreed, not
that the estimator converged.

### `band_suppression`

*Left*, each declared band's suppression margin with its paired interval, the joint `all` removal
last in grey. *Right*, the exposure behind each margin: how many scored anchor-lag pairs of the band
carried any available channel. A band with nothing to remove is drawn as `not measured` rather than
at zero, because a zero there would read afterwards as a band that did not matter.

**How it is misread.** The margins do not decompose the gap and are not normalised to it: the
limiter is applied after the summation, so two bands' margins need not add to the margin of removing
both. And a margin is a property of this fitted parameterisation -- an exactly zero-sum reallocation
of proposals across lags changes every band's margin while changing no prediction.

### `lag_profile`

Four panels sharing the lag axis, with the declared bands shaded and named along the top of the
first panel and the seconds axis above it.

1. *Exposure.* The fraction of scored anchors at which the lag carried any available channel, and
   the mean fraction of the declared source channels available at those anchors. Both are fractions
   on one axis. A lag deep in the warm-up staircase is low on both, and everything drawn below it is
   read against that.
2. *Latent profile: the proposal and the update.* Per lag, averaged over the scored anchors the lag
   was live at: the norm of the proposal the head emitted, $\lVert r_{t,\ell}\rVert_2$; the shift
   removing that lag alone makes to the bounded mean update,
   $\lVert a_t - a_t^{\setminus\ell}\rVert_2$; and, where the arm has a scale channel, the scale
   proposal's norm. The shift is zero where the limiter has saturated however large the proposal is,
   which is what separates it from the norm.
3. *Divergence drop.* $K_t - K_t^{\setminus\ell}$, signed: removing a lag can raise the divergence
   when its proposal was cancelling another's.
4. *Predictive margin of removing the lag alone.* The finest partition of the suppression readout,
   scored under the same draws as every other arm on the segments the profile cap admitted, with its
   paired interval over the recordings those segments came from. Absent, with the reason, when the
   cap was not set, when the arm's fusion sums no per-lag updates, or when the checkpoint has no
   source pathway.

**How it is misread.** None of the three latent quantities is an allocation over lags, and the
predictive profile is read **after** the band and joint removals of `band_suppression`, never instead
of them: a single-lag peak read off a window whose joint removal does nothing is noise. The profile
is over a capped subset of the split, and its recording count is in the panel title.

### `horizon_resolved`

Three panels sharing the horizon axis. *Top*, the gap by horizon step with its interval. *Middle*,
each declared band's suppression margin by step, paired over recordings, one fixed colour per band.
*Bottom*, the source control margins by step.

Every curve is the marginal mixture of that step's own likelihood factors under the shared draws:
$D^{(K)}_\tau = -\operatorname{logsumexp}_k(-D^{(k)}_\tau) + \log K$. In general
$\log \mathbb E_Z \prod_\tau p_\tau \ne \sum_\tau \log \mathbb E_Z p_\tau$, so the steps do **not** sum to
the joint block score and the axis is read for its shape rather than its total. The horizon axis is
the one a source's timing is expressible on here: a band that informs the first predicted step and
not the last is a statement the window carries whatever the lag axis can resolve.

### `block_resolved`

One dot plot per stored target block -- the scattering coefficients and the phase-harmonic
coefficients, in the order the kept channel axis carries them -- each listing the gap and every
margin with its paired interval, summed over that block's own channels and the horizon. The
channel count each block kept is in its title, because the two blocks are not the same size and a
nat summed over more channels is a larger number for that reason alone. The same subset-mixture
caveat as the horizon figure applies: the two blocks do not sum to the joint score.

### `calibration`

*Left*, the observed central coverage of the mixture predictive law at each nominal level, for both
branches, with the diagonal a calibrated forecast sits on. Below the diagonal is over-confidence,
above it under-confidence, and the three levels separate a body problem from a tail one. *Right*,
the mean and variance of the probability integral transform for both branches against the uniform
reference ($1/2$ and $1/12$). Both branches are drawn because a calibration statement about the
source-conditioned branch alone cannot say whether the source improved it or whether the observation
model was already miscalibrated without it.

---

## The per-recording traces

Drawn into `eval_results/recording_traces/` by the stage that runs after the scoring pass, from the
forwards it re-reads rather than from the summary -- the one exception to the first rule above, and
stated as one: a trace is the whole forward output of every segment of a recording, which no summary
carries. The file layout, the two tables and both figures are the family's shared ones, so a trace
under this architecture reads beside a trace of the same recording under a lag-attentive one.

### `recording_traces_summary`

One panel per per-segment summary column -- the divergence, the single-draw forecast gap, the source
shift of the latent mean, the active coordinate count, the lag centroid of the proposal norm and the
cancellation ratio -- each against hours before delivery, one line per traced recording in its class
colour, one marker per segment, lifted at a break. Up to `eval_config.caps.traces_per_class`
recordings per class, drawn for looking at rather than for testing: a class whose lines sit higher is
a hypothesis and not a finding.

### The per-recording traces: `<class>/<guid>_<subgroup>_trace`

One recording followed through every segment the dataset holds for it, at anchor resolution, on one
shared axis of hours before delivery; the filename carries the GUID **and the subgroup**. The rows:
the divergence per anchor; the single-draw forecast gap off the forward's own forecasts; the
full-branch mean over the latent coordinates and the bounded mean update $a_t$, as heatmaps on a
symmetric scale; the divergence per coordinate; the **proposal norm over lags** with the lag of the
largest proposal drawn over it; the latent norms; the mean log-variances; the lag centre of the
proposal norm; and the cancellation ratio of the mean update. Segment joins are marked by a light
vertical line, and a line row is lifted at every unscored anchor.

**How it is misread.** The lag row is a proposal *norm* -- an update magnitude before the sum and
the limiter, not a distribution over lags and not an allocation of the divergence -- and its
qualification is printed under the figure with the stored-coefficient-time caveat. A step at a
segment join is geometry: each segment is a separate forward with a reset encoder state. The colour
scales are per recording. On the comparator's normalised fusion and on the target-only arm the lag
row is absent, and the block in the summary says so under `lag_family_present`.

---

## The attributions

Drawn into `eval_results/attribution/` by the stage that runs after the traces, from the forwards
it re-reads and differentiates: five fixed-name figures (`attribution_maps`,
`attribution_lag_profile`, `attribution_bands`, `attribution_layer`, `attribution_null`) and
one `traces/<class>/<guid>_<subgroup>_attribution_trace` per traced recording. They are the
family's shared figures and are documented, panel by panel, in the lag-attentive cells' figure
guide under the same names. What differs here: the lag row of every figure is the **proposal
norm**, qualified as the traces qualify it, and the layer figure's left panel is the per-lag
split of the readout on the proposal head's output -- an attribution through the summation and
the limiter, not an allocation -- drawn against the compensated lag axis rather than per head.

---

## The acceptance pass

Drawn into the directory `--figures` names, from the record after it is assembled.

### `acceptance_comparisons`

The declared primary comparisons, each the paired difference of `nll_full` between its left and right
arm, averaged over the arms' training seeds and bootstrapped once over recordings. Negative favours
the left arm. A comparison is blue when it was read at the declared draw count with both arms at
the seed minimum and grey otherwise; its status is in the record.

### `acceptance_arms`

Three dot plots sharing the arm axis: each arm's internal gap; its full branch against the frozen
target-only reference (negative favours the arm), which is the comparison the internal gap cannot
make; and its base branch against that reference, which is the one that can fail -- a confidently
positive value is a base that joint training left behind, and every gap measured against it is then
measuring that.

### `acceptance_bands`

One panel per arm that searched the lag bands. Each band's margin at the nominal level in the band's
colour, with the family-adjusted interval -- the one covering every searched band at once, and the
one a claim about the peak band rests on -- as a wider grey bar behind it. The peak band is named in
the panel title; it was chosen on the same recordings its interval is built from, which is why the
adjusted interval exists.
