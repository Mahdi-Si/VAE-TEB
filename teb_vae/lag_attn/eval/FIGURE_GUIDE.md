# Reading the figures

Every figure an eval run emits, what it shows, and what it does **not** show. For the pipeline's
contract — what each analysis measures, the config schema, the enforced preconditions — see
[`EVAL.md`](EVAL.md).

Read §1 first. Each of those seven traps is a case where a figure looks like it says one thing and
says another, and six of the seven have already been drawn as a wrong conclusion at least once.

---

## 1. Seven traps, before any figure

### 1.1 $K_{\mathrm{shuffled}} \gtrsim K_{\mathrm{true}}$ is normal and is not a failure

`perm_control/kl_overlay.pdf` puts the true per-step KL against the KL under a deranged source. The
intuitive reading — "a wrong source should move the posterior less" — is **backwards**. A
mismatched UP stream is out of distribution, and an out-of-distribution input typically moves the
posterior *more*, not less. A run where $K_{\mathrm{shuffled}}$ sits at or above
$K_{\mathrm{true}}$ is the ordinary healthy case.

Source specificity is proven in **prediction space**, by the ordering

$$L_{\mathrm{feat}} < L_{\mathrm{base}} < L_{\mathrm{feat,\ shuffled}}$$

which is what `perm_control/losses.pdf` draws and what the `source_specificity` verdict reports.
The KL-space panel is labelled as diagnostic and **cannot flip that verdict**.

### 1.2 The seconds axis is the stored timeline, with no offset

Every lag figure gets a secondary axis in seconds, $\mathrm{seconds}(\ell) = 4\ell$. The stored UP/FHR timeline is canonical: the dataset builder shifts the UP channel when it writes the shards, that shift is part of how the stored signals are, and nothing downstream adds it back, subtracts it, budgets it or interprets it.
The former `eval_config.up_shift_secs` key — which undid that shift on every axis, and whose sign
was corrected once before the key itself was removed on 2026-09-05 — no longer exists; a config
naming it is refused. Figures drawn before that date carry a secondary axis on a "raw-file"
timeline offset by the key's value, and `preflight.json`'s `lag_seconds_convention` says which.

**What this means for a figure you already have.** A run whose `preflight.json` records
`"sign_verified": false` drew its seconds axes under the old convention: subtract $40$ s from every
number on them. The **model-lag** axis (the left one) was exact throughout and needs no correction —
read it first regardless.

The **training-callback** figures are a separate case and still carry no offset at all:
`plotting.py` reads `delta_up_seconds` off an attribute the model does not have, so their seconds
axis is the bare $4\ell$. Not wrong in sign, but not a lead either.

### 1.3 `te_lag_map` needs `head_structured_latent`

Without it the map is still drawn, but it is a **diagnostic**, not an attribution: the per-lag
decomposition is not rigorous and the numbers do not add up to a defensible per-lag transfer
entropy. Every affected title and CSV carries `te_lag_map_label`, which reads `attribution` or
`diagnostic`. Check it before quoting a per-lag number.

### 1.4 `kld_raw` needs `causal_norm`

Without a causal normalisation the KL is not a transfer entropy at all. The `te_lag` analysis
**refuses to run** rather than drawing a mislabelled figure, so if you have no `te_lag/` directory
this is the first thing to check — `preflight.json` records `causal_norm` and what it blocks.

### 1.5 Under `kld_support='anchor'`, the last $H_d$ anchors are *supposed* to look decoupled

`latent/kt_curve.pdf` and the $K_t$ row of every sample page fall away at the right-hand edge. Under
`kld_support='anchor'` those final anchors sit outside the supervised support, so they collapse
toward the prior with nothing pulling back. It reads as a real drop in coupling at the end of every
recording. It is not one. Read the interior.

### 1.6 `attended_source` is diagnostic-only in production

The posterior consumes `attended_source_heads`, and $W_o$ is frozen (`frozen_attn_proj`). So
`attended_source` — the projected combination — is not on the path that produces the latent, and a
figure of it describes a quantity the model does not use. Nothing in this pipeline draws it as a
headline; if you add one, label it.

### 1.7 `measure_transfer_entropy(reduce_mean=True)` returns $\bar{K}_t / d_z$, not $\bar{K}_t$

A factor of $d_z = 24$. Nothing in this pipeline calls it that way — the analyses reduce
`kld_per_t` themselves — but a number pulled from that helper into a figure caption will be 24×
too small, and it looks entirely plausible.

---

## 2. Forecast — `forecast/`

### `horizon_error.pdf`
Masked MSE against horizon step $h$, median with an inter-quartile ribbon over samples.

**Read:** the slope. Error is *supposed* to rise with $h$ — a step further into the future is
harder. **A flat profile is the signature of a model predicting a constant**, which can post a
perfectly respectable aggregate MSE. This is the single most diagnostic panel in the run.

**Median and IQR, not mean and SD:** these profiles are right-skewed, a handful of poorly forecast
recordings sit far above the rest, and a mean band would be pulled off the bulk and could extend
below zero on a non-negative quantity.

### `anchor_error.pdf`
The same, against anchor position $t$. Localises a forecast that degrades partway through a
recording rather than averaging it in. Expect the warm-up prefix to be absent (masked, not zero).

### `distributions.pdf`
Per-sample masked MSE and $R^2$ histograms, with the median marked and a reference line at
$R^2 = 0$ — the point where the model does no better than predicting each channel's mean. Mass at
or below zero is the number to look at, not the mean.

**These are per-sample means**, each sample divided by its own mask sum. They do **not** equal
`compute_loss`'s pooled figure unless every sample has the same mask density, which real recordings
never have. The pooled, training-reconcilable form is in `scalars/test_metrics.csv`.

### `heatmaps.pdf`
Three $(c_y, T)$ panels: mean forecast, mean target, RMS residual. The dashed line is the
scattering / phase-harmonic boundary, drawn from the batch rather than a literal.

**The residual panel is an RMS, not a signed mean.** A signed mean cancels: a channel the model
over-predicts as often as it under-predicts averages to zero and reads as perfectly forecast, which
is the opposite of the truth. A bright row is a badly forecast channel.

Forecast and target share one colour range so they are directly comparable; the residual has its
own.

---

## 2b. Frequency band — `frequency_band/`

Two partition subdirectories, `clinical/` and `by_kind/`, each carrying the same two figures, plus
a `per_channel/` directory. **Rows and ticks run high frequency to low** on every panel here, so
the top of a figure is always the fastest structure.

Every label carries its frequency range explicitly — `deceleration (0.008-0.04 Hz, 22 ch)` — so no
panel requires looking a band name up. A clinical band shows its *defining* range; a harmonic kind,
which is not a frequency range, shows the range its channels actually occupy. A label whose
channels carry no centre frequency at all says `no centre frequency` rather than being drawn at
$0$ Hz, which would assert a frequency the provenance does not determine.

### `band_violins.pdf`
Per-sample masked MSE and $R^2$, one violin per band, each with a thin box plot inside it — the
bar is $Q_1$ to $Q_3$, the hairline runs to the furthest sample within $1.5$ inter-quartile ranges
of it, the dot is the median — and a reference line at $R^2 = 0$.

**Read:** which bands the model actually predicts. Violins rather than boxes because these
distributions are routinely bimodal — a set of recordings forecast well and a tail that is not —
and a box renders both as the same five numbers.

**A band with no finite value keeps its slot** as an empty position rather than being dropped, so
the categories stay aligned with their labels.

### `band_horizon.pdf`
Three panels: band-by-horizon as overlaid lines, then the same as a heatmap, then band-by-anchor as
a heatmap. The line panel is for comparing bands against each other; the heatmaps are for finding
the one cell that is wrong.

**Read:** whether the horizon degradation is uniform across frequency. A band whose error rises
much faster with $h$ than the others is the one bounding the usable forecast horizon.

The warm-up anchors are absent (masked, not zero), as in `forecast/anchor_error.pdf`.

### `per_channel/per_channel_frequency.pdf`
Per-channel pooled MSE against centre frequency, on a **log** frequency axis, split into a
scattering panel and a phase-harmonic panel.

**Log $x$ is not cosmetic:** the channels are a geometric filter bank spanning $5 \times 10^{-4}$
to $1.5$ Hz, so on a linear axis the entire slow half of the bank collapses onto the left-hand tick.

**Two panels, because a phase-harmonic channel is a *pair*.** A scattering channel is described by
one centre frequency; a phase channel is described by $(\xi_i, \xi_j)$ and its ratio
$p = \xi_j / \xi_i$, and plotting it at $\xi_j$ alone throws away the half of its identity that
distinguishes it from the scattering channel at the same frequency. The phase panel therefore
colours by $p$, so the dual-frequency identity survives into the figure.

**Channels with no centre frequency are omitted, and the count is in the legend.** At the
production geometry that is $14$ of the $43$ scattering channels — the three fastest and eleven
slowest order-1 filters, which no selected phase pair references. A panel silently missing them
would look complete.

---

## 3. Uplift — `uplift/uplift.pdf`

$L_{\mathrm{base}} - L_{\mathrm{full}}$ per sample, absolute and relative, with the fraction of
samples where the residual pathway helped.

**Read:** the mass to the right of zero. A distribution centred on zero means the source pathway
contributes nothing on this split — which is a finding, not an error. Read it beside
`residual/residual.pdf`: a dead residual with a positive uplift is contradictory, and means one of
the two is measuring something other than what its name says.

Scored under the **checkpoint's own** objective, so this is comparable across checkpoints only when
they were trained with the same one. `summary.json` records it.

---

## 4. Residual — `residual/residual.pdf`

`residual_ratio`, the masked RMS of `delta_mu_src` relative to the full forecast, per sample and
per anchor, against the `health_probe_floor` reference line.

**Read:** whether the distribution clears the floor. Below it, the residual head is contributing
essentially nothing.

**The floor is a priori.** It was set to `0.01` before any real checkpoint existed. A distribution
sitting just below it is not evidence of collapse — it is evidence the floor has not been
calibrated. The check that actually separates "collapsed" from "never loaded" is the weight-space
one in `preflight.json`, not this figure.

---

## 5. Attention — `attention/`

### `attention.pdf`
Argmax-lag distribution, entropy against its **attainable** ceiling, and head diversity.

**The ceiling is not $\log L$.** Causal masking gives anchor $t$ only $\min(t + 1, L)$ valid lags,
so at the production geometry ($L = 91$, warmup $30$) sixty of the two hundred and forty supported
anchors cannot reach $\log 91$ at all. The bound the panel draws is the support-weighted attainable
one, $\sum_t s_t \log\min(t{+}1, L) / \sum_t s_t$, reported as `mean_attainable_entropy_nats`;
$\log L$ survives as the separately named `max_possible_entropy_nats`, the width of the window
rather than a reachable value. Attention uniform over every causally available lag — zero lag
structure — scores $4.398$ against $\log 91 = 4.511$, so read against $\log L$ that degenerate case
looks like mild concentration. `per_sample.csv` carries the per-sample `attainable_entropy` if you
want to check the panel against the verdict.

**Two degenerate readings, failing in opposite directions.** An argmax pinned at lag $0$ means the
attention never looks back and the lag machinery is inert. An entropy at the **attainable** ceiling
means the weights are uniform, so the argmax is whichever lag won a rounding contest and any "peak"
is noise wearing a hat. `summary.json`'s `sanity.argmax_lag` checks both, dividing by the attainable
ceiling — against $\log L$ its uniformity branch could never fire at production geometry.

### `attention_heatmaps.pdf`
Per-sample $\alpha$ over $(t, \ell)$ for a capped stratified draw. Rows sum to $1$ — from an
`eval()`-mode pass. **Under `train()` dropout is live inside the attention, the rows do not sum to
1, and the `te_lag_map` identity silently stops holding.** Emitted only when samples were retained.

---

## 6. Lag-resolved TE — `te_lag/`

### `te_lag.pdf`
`te_lag_map`: the attribution of $K_t$ across lags. Includes the summation-identity check — the sum
over lags against `kld_per_t` on the same support — as a *measured* deviation rather than an
assumption. A large deviation means the map and the KL disagree and neither should be quoted.

**Column-normalised where drawn per sample.** The per-step KL varies over orders of magnitude
across a recording, so a raw map is dominated by a few bright columns and the lag *selection* is
invisible everywhere else. Columns whose KL is effectively zero are left blank rather than
amplified into a confident-looking pattern.

See traps 1.2, 1.3 and 1.4 before reading a lag off this.

### `per_head_lag_profile.pdf`
The per-head decomposition $K_t = \sum_m K_t^{(m)}$ — the readout only this model supports.
**Emitted only when `head_structured_latent` is true.** Absent otherwise, with the reason in
`summary.json` under `te_lag.per_head.reason`.

**Read:** whether the heads specialise by lag or all look alike. All-alike is not a bug, but it
means the head structure is buying nothing.

---

## 7. Latent — `latent/`

### `per_dim_kl.pdf`
Mean KL per latent dimension against the active threshold, with `kld_active_frac`.

**Read:** how many dimensions clear the threshold. A handful of active dimensions out of $d_z = 24$
is the posterior-collapse signature. **Exactly zero everywhere is the trap:** the posterior delta
heads are zero-initialised, so $K_t \equiv 0$ at initialisation — an all-zero panel is as
consistent with "the checkpoint never loaded" as with "the latent collapsed", and only
`preflight.json`'s weight-space check tells them apart.

### `per_dim_violin.pdf`
The same per dimension, as a distribution over samples rather than a mean — a dimension active on a
few recordings and dead on the rest looks identical to a uniformly weak one in the bar panel.

### `kt_curve.pdf`
$K_t$ against $t$. **See trap 1.5** about the right-hand edge under `kld_support='anchor'`.

**The saturation readouts** in `summary.json` come in `_raw` and `_masked` pairs and routinely
disagree. `_raw` is the model's own in-forward reading over *every* element including warm-up and
padding; `_masked` is recomputed over the supervised support. A model saturated in its padding but
not in its supervised region has a padding problem, not a hyperparameter problem. The flag applies
to `_masked`.

---

## 8. Calibration — `calibration/`

All three are emitted only when the checkpoint was trained with `likelihood='gaussian_nll'` **and**
`sigma_obs='learned'`. Under any other objective there is no learned predictive variance to
calibrate and the analysis records a clean skip.

### `reliability.pdf`
The PIT diagram. Under a perfectly calibrated predictive Gaussian the PIT values are uniform, so
the curve is the diagonal. A sagging curve means over-confidence (intervals too narrow); a bulging
one means under-confidence.

### `coverage.pdf`
Observed central-interval coverage at $1/2/3\sigma$ against nominal. Note the $2\sigma$ nominal is
$0.9545$, **not** $0.95$ — $0.95$ is $\pm 1.96\sigma$, and the difference is large enough to read
as a real miscalibration if you compare against the wrong one.

### `sharpness.pdf`
The predictive $\sigma$ distribution against the homoscedastic reference. **Read together with
`reliability.pdf`:** calibration alone is trivially achievable by predicting a huge constant
variance. The learned variance is worth having only if it is *both* calibrated and sharper than the
constant, which is what `mean_nll_gain` and `learned_variance_beats_homoscedastic` report.

---

## 9. Permutation controls — `perm_control/`

### `losses.pdf`
**The source-specificity verdict, and the one that counts.** Three losses side by side:
$L_{\mathrm{feat}}$, $L_{\mathrm{base}}$, $L_{\mathrm{feat,\ shuffled}}$.

- $L_{\mathrm{feat}} < L_{\mathrm{base}}$ — the source helps.
- $L_{\mathrm{base}} < L_{\mathrm{feat,\ shuffled}}$ — it helps *because it is the matching
  source*, not merely because an extra pathway adds capacity. A model that had learned a generic
  smoother would be unharmed by shuffling.

Both together give `source_specific`. Only the first gives `influential_not_specific`. Neither
gives `no_uplift`, and too few samples to derange gives `undetermined`.

### `kl_overlay.pdf`
$K_{\mathrm{true}}$ against $K_{\mathrm{shuffled}}$ per step. **See trap 1.1** — this panel cannot
prove or disprove specificity, and the direction most readers expect is the wrong one.

**Cross-referencing the CSV: mind the normalisation.** These are the $d_z$-summed per-step KL, and
`per_sample.csv` names them `kld_true_per_t` / `kld_shuffled_per_t` for that reason. They are
$d_z = 24$ times `scalars`' `kld_raw` and $24\times$ `latent`'s `kld_mean` — three quantities with
KL in the name, differing by a factor of twenty-four, inside one run directory. Compare the
**ratio** (`kld_shuffled_ratio`, and `mean_kld_shuffled_ratio` in `summary.json`), never the levels
across analyses.

The derangement is fixed-point-free by construction, so no sample is ever paired with its own
source. A batch of fewer than two samples cannot be deranged; those rows carry no control columns
at all rather than zeros, and `n_shuffle_penalty_scored` reports how many samples actually
contributed to `positive_shuffle_penalty_frac`.

---

## 10. Lag-band ablation — `lag_ablation/`

### `forecast_degradation.pdf`
Forecast MSE per band relative to the unmasked baseline. **Read:** which band, when it is the only
one kept, best preserves the forecast — that is where the causally useful source history lives.

**The bar you want is the SHORTEST one.** The mask *keeps* rather than removes, so each bar is that
band running alone. A **small** $\Delta$ means the band alone nearly reproduced the full forecast
and is therefore **sufficient**; a **large** one means it was not enough, which says the rest of the
window carried what it lacks — *not* that the band mattered more. Read as a removal ablation the
ranking inverts exactly, which on a model whose UP influence lives at short lags would publish the
longest lags as the important ones. The title says "with ONLY this lag band kept" for that reason,
and `summary.json` names both ends, `most_sufficient_band` ($\min$) and `least_sufficient_band`
($\max$), plus a `semantics` string that travels with the numbers. There is no "most damaging band":
the phrase has no correct reading under a keep-mask.

**This is sufficiency, not necessity.** Necessity would need a keep-mask over the band's
*complement*, and nothing in the pipeline constructs one.

### `kl_change.pdf`
The same for the KL, and the same sign convention.

**Both are scored on one common dead-anchor-safe support, and this is what makes them
comparable.** Keeping only a band whose lowest lag is $\ell$ leaves the first $\ell$ anchors with no
valid source history at all; scoring a band over anchors it cannot possibly serve measures the
deadness, not the band. Every band and the baseline are therefore scored on the identical anchor
range, starting after the widest band's dead prefix — `summary.json` records `common_scoring_start`
and
`anchors_excluded` per band. The per-band KL is recomputed from `kld_tensor` on that same support
rather than read from `compute_loss`'s band-unaware `kld_raw`.

**A band's absolute number is not comparable across runs** with different band definitions, because
the common support depends on the widest band configured.

---

## 11. Per-sample pages — `samples/sample<index>_<guid>_epoch<epoch>.pdf`

One page per selected recording. Every row shares one physical-time axis in seconds, so a vertical
line cuts every panel at the same instant and a feature seen in the attention can be traced
straight down into the forecast. That alignment is the entire point of the page.

Rows, in order:

| Row | Shows | Read for |
| --- | --- | --- |
| Raw FHR / UP | The loaded traces, twin-axis | Context: is this a recording with real contractions? **Omitted entirely when `fhr`/`up` are not in `load_fields`** — the page is one row shorter, not one row blank. |
| Forecast | Overlap-averaged $\mu_{\mathrm{full}}$, all $c_y$ channels | Structure. Flat bands are channels the model gave up on. |
| Target | $Y$, same colour range | Direct comparison — **the shared range is deliberate**; scaling the two independently makes a badly-scaled forecast look well-scaled. |
| Forecast residual | $\mu_{\mathrm{full}} - Y$, own range | Which channels and which times carry the error. |
| Latent $z$ | One seeded draw | Whether the latent varies at all. |
| Per-dimension KL | $(d_z, T)$ | Which dimensions carry information, and when. |
| $K_t$ | Per-step KL, with attention entropy on the twin axis | Coupling over time. **See trap 1.5** about the right edge. |
| Lag attention | Mean $\alpha$ over heads, with the argmax overlaid | Where the model looks. **See trap 1.2** about the seconds axis. |
| TE lag attribution | Column-normalised `te_lag_map` | Which lag carries the coupling. Title states `attribution` or `diagnostic` — **see trap 1.3**. |

The heatmaps draw with `interpolation='none'`, so one data cell is one cell. `'nearest'` would
resample at 600 dpi and can merge adjacent channels — on a per-channel diagnostic that is the one
artifact these rows exist to rule out.

A page missing from the directory is not a silent loss: `summary.json`'s `samples.failures` names
every sample whose render failed and why.

---

## 11b. By-class and by-subgroup variants — `<analysis>/<stem>_by_class.pdf`, `<analysis>/<stem>_by_subgroup.pdf`

One violin panel per headline metric, split across cohorts, beside every analysis's pooled output.
Class colours come from the repository's shared table, so an eval figure and a training figure of
the same cohort are the same colour.

**Read:** whether the cohorts separate at all — and then read `cross_subgroup/` before saying they
do. Eight cohorts with a mean will always produce a highest and a lowest, and a violin figure is
exactly the artifact that makes an accidental separation look convincing.

**A group with no finite value keeps its slot** as an empty position, so the categories stay
aligned with their labels.

**These appear only when the split holds more than one cohort.** On the healthy-only pretraining
split, and on any single-shard run, they are absent by design and `summary.json` records the skip
with its reason. Their absence is not a failure.

---

## 11c. Cross-subgroup statistics — `cross_subgroup/cross_subgroup.pdf`

Two panels, because the two questions are different and are routinely confused.

**Upper — is there anything there?** $-\log_{10}$ of the Holm-adjusted $p$ per metric, with the
$\alpha$ line marked. Bars to the right of the line are the metrics whose omnibus Kruskal-Wallis
survived correction across the whole family.

**Lower — does it matter?** Cliff's delta for every pair of every surviving metric. Positive means
the *left* group of the pair runs higher; the colour scale is symmetric about zero.

**A metric can clear the upper panel and be negligible on the lower, and at eight subgroups that is
the common case rather than the exceptional one.** A $p$-value is not an effect size: with a few
thousand recordings per cohort, a difference of no clinical consequence reaches significance
readily. Quote $\delta$ and its magnitude label, never the $p$ alone.

**Only metrics that survived Holm appear in the lower panel.** A metric absent from it was not
tested pairwise, which is different from having been tested and found equal — `significance.csv`
distinguishes the two.

---

## 11d. KLD trajectory to delivery — `kld_time_to_delivery/`

The per-segment KL $\overline{K}$ — the same quantity the latent figures show — cut by **time to
delivery** instead of pooled over gestation. Two PDFs.

**`trajectory.pdf` — where the coupling lives in time.** Two panels: the upper by clinical class,
the lower by subgroup. Each draws one median line per group across $0.5$ h time-before-delivery
windows, with an inter-quartile band, and the **time axis is inverted so delivery (0 h) sits at the
right** — the eye reads left-to-right *toward* delivery. The $y$ axis is $\overline{K}$ in nats. A
rising line toward the right is a cohort whose source→target coupling strengthens as delivery
nears; a flat line is one where it does not.

**`significance.pdf` — is a class difference real, window by window.** Upper: $-\log_{10}$ of the
Holm-adjusted $p$ from the per-window Kruskal-Wallis across classes, against the $\alpha$ line, on
the same inverted time axis — bars above the line are the windows in which the classes' $\overline{K}$
separates after correction across every window. Lower: Cliff's delta for each surviving class pair
in those windows; positive means the *left* class of the pair runs higher, the scale symmetric about
zero.

**Three traps.** The **pooled** class test in `kld_time_to_delivery.json` is *not* the trajectory
answer — it ignores time and is flagged `confounded_by_time`, because the classes do not cover the
axis equally. A window can clear the upper panel with a **negligible** Cliff's delta — quote the
effect size, not the $p$. And an **empty or skipped** figure is an ordinary outcome: the class test
self-skips below two clinical classes (the single-class pretraining split), and the whole analysis
skips on a split carrying no `epoch` field or no cohort labels.

---

## 12. Figures this pipeline does not emit

Not oversights — each was considered and declined, and knowing that saves looking for them.

- **Raw-FHR reconstruction.** The model forecasts *features*, not raw signal. Raw FHR and UP appear
  only as the sample pages' context strip.
- **Latent interpolation.** A no-op in the predecessor too, for the same reason.
- **Cross-run or cross-checkpoint comparison.** Every analysis writes CSVs to a timestamped
  directory; comparing two runs is a `pandas` merge.
- **Cross-*run* comparison of the grouped variants.** Each run's grouped tables are CSVs in a
  timestamped directory; comparing two checkpoints' by-subgroup results is a `pandas` merge.
- **Interactive Plotly.** `plotly` is installed, but static PDF is the repository's figure
  convention and the figure tests assert it.

---

## 13. Coverage

`summary.json`'s `artifacts.figures` lists every PDF a run actually emitted. A test reads that
manifest and asserts each filename appears in this guide, so a new figure with no entry here fails
the suite rather than shipping undocumented. That is why the manifest exists: a hardcoded filename
list would pass by construction.
