# UA quality mask, cleaning and scattering: research and design

Date: 2026-10-25. Status: Phase 0 and Phase 1 are implemented; the builder does not use them yet.
The rules in `hdf5_dataset/signal_quality.py` and the profile
`hdf5_dataset/new_pipeline/profile_signal_quality.py` exist. A local run on the two `.mat` records
changed several rules; section 4.3 describes the rules as implemented, and section 2.6 gives the
local evidence for each change.

Scope: the dataset-creation pipeline (`hdf5_dataset/new_pipeline/create_new_pipeline.py`) and the
fields it writes. The model gate in `teb_vae/lag_attn_transformer_cfs` is a later step; this document
only lists what the gate will need.

Evidence sources:
- the 11 sample pages in `tmp/new-k-fold-docs/samples`, with the UP and FHR polylines extracted from the PDF vectors;
- two local real shards: `output/causal_scattering/hie_cs_alighned/hie_cs_alighned.hdf5` (2617 segments)
  and `output/poster_figures/hie_cs_int/hie_cs_causal_int.hdf5` (339 segments), both hie_cs only;
- two local real `.mat` records;
- a code review of the pipeline, the MIMO adaptor, the loader, the model, and the colleague's `ctg_rules` repository;
- a literature search (section 3).

Numbers from the local shards come from one subgroup. Run the profile in section 7 on the production data
before you fix any threshold.

---

## 1. Answer

1. **Most UA "above 100" is not a truncated contraction.** It is a constant invalid code, 128, that
   the sample page clips at its 0–100 axis. In the shards, every 128 run starts with a jump of 30 units
   or more, and 76% start from a value of 5 or less. The 255 code behaves the same. Mask these values
   as missing. They carry no information.
2. **A run at exactly 100 is usually a real contraction crest that the device or the export clamped.**
   The median value just before the run is 95–97, and only 3–6% of the runs start with a jump. Keep
   these samples as valid but *censored*: the timing is correct and the amplitude is "at least 100". Do
   not reconstruct the peak.
3. **The largest source of flat regions is a defect in the MIMO fill step, not the signals.**
   `MimoSequence.interpolate_input` fills every zero run with a constant, in both channels at once. A gap
   in one channel therefore overwrites the valid data in the other channel. It has no gap-length limit,
   and it can mis-pair gap edges. Fix this first, because every later step depends on knowing which
   samples are real.
4. **Build a per-sample quality code for UA and for FHR.** Clean only inside the bad spans, with a
   smooth bridge, so that a bad span does not spread into the valid coefficients. Then carry the mask
   to the 330-step grid *per channel*, with the causal support of each filter.
5. **Valid UA is not the same as informative UA.** Mask only invalid UA. Keep valid quiet UA: the
   coupling literature uses rest periods as the contrast for contraction responses. Express
   "informative" as a soft contraction weight, not as a hard mask.
6. **Use the scattering transform to audit the cleaning, not to detect the artifacts.** The raw-domain
   rules find the same events with sample-level timing and without circularity.

---

## 2. What the data show

### 2.1 The values above 100

Counts from the two local shards. "Entry" is the UP value one sample before the run starts.

| Code | Share of samples | Run length (median / p90 / max) | Entry from $\le 5$ | Entry jump $\ge 30$ | Reading |
|---|---|---|---|---|---|
| 100 | 0.82–1.06% | 6–9 s / 24–31 s / 92–179 s | 0% | 3–6% | clamped crest, mostly real |
| 127 | 0.001–0.08% | 2–4 s / 2–16 s / 39 s | 0% | 15–50% | device ceiling (Philips range 0–127); some artifact |
| 128 | 0.08–0.20% | 1–5 s / 18–58 s / 246 s | 41–76% | 100% | invalid code |
| 255 | 0.003–0.008% | 18–28 s / 28–34 s / 35 s | 0% | 100% | invalid code |
| (100, 127) | 0.003–0.23% | — | — | — | real values in recordings with a 0–127 range |

The sample set over-represents the problem: the 128 code covers 142–566 s of a 1200-s window in 10 of
the 11 samples. In 8 of those 10 samples, the largest $K_t$ peak (source-conditioned KL) comes 1–42 s
after the onset of a 128 block. Every lag-argmax excursion away from lag 0–1 points 0–11 s after a
block edge. **The model currently learns "coupling" from the invalid-code edges.** Sample 59096 is the
clearest case: $K_t$ averages 24.5 nats after the block onset and 0.5 nats elsewhere.

The colleague's MATLAB code treats raw UA $\ge 127$ as invalid, and its viewer treats 127 as the PeriGen
lost-signal code. The local data show 127 runs that start smoothly from about 120, so in these
recordings 127 is mostly the device ceiling. The rule in section 4.3 therefore treats 127 like 100
(censored if the entry is smooth) and treats every value $\ge 128$ as invalid.

The sample page hides all of this. Its UP axis stops at 100, so a 128 block shows only as two vertical
lines. The axis is also labelled "mmHg", but TOCO reports relative units.

### 2.2 The fill defect

`MimoSequence.interpolate_input` (`mimo/MIMO_Sequence_Trainer/mimo/sequence/mimo_sequence.py:2096-2139`)
runs because the build passes `min_domain_start`. Its defects:

- It treats every value $\le 10^{-4}$ in either channel as a gap of *both* channels.
- It fills a gap with the constant $s_a + (s_a - s_{b+1})/\text{len}$. That value is not an interpolation,
  and it is the source of the negative UP values (down to $-30.75$) in the shards.
- It has no gap-length limit.
- A gap at the segment start can mis-pair the transitions, and the fill then overwrites valid samples.

On one real record with the build's arguments, it changed up to 3196 FHR samples in one segment, and
it turned five kept segments into UP $\equiv 0$.

The upstream MATLAB repair produces interior FHR fills of up to 256 samples (64 s), and the shortest
break between runs is 255–272 samples. The sample pages show FHR constant holds of mostly 58–64 s, often
inside UP blocks. Inference: these holds are breaks between runs that the MIMO step filled with a
constant.

**A valid UA rest at 0 also triggers the fill.** Some monitors rest at exactly 0 between contractions.
The fill reads each such rest as a gap and writes a constant into *both* channels. The UA rest becomes a
plateau of 8–12 units, and the valid FHR becomes a flat line for the whole rest (record
`A97E3DEA…`, segment 8: four rests of 35–130 s). Today's flat-FHR gate then drops the segment. In that
record, the gate dropped 16 of 80 pre-delivery segments, and all 16 pass every gate without the fill.
In those 16 segments, the fill changed 655 s of FHR per segment on average, and 54% of their UA samples
are exact zeros. The loss is therefore not random: it removes recordings whose monitor rests at 0.

### 2.3 Other UA problems in the samples

| Type | Samples | Duration | Signature |
|---|---|---|---|
| Baseline step, then relaxation | 7 of 11 | step < 1 s; decay 60–150 s | $\lvert\Delta x\rvert \ge 15$ in 1 s, then a new level or a monotone decay |
| Hold at the last value | 4 of 11 | 17–84 s | exactly constant $\ge 30$ s, with an edge jump or an FHR hold with the same edges |
| Floor (loss of contact) | 5 of 11 | 20–130 s | $x \le 2$ for $\ge 20$ s, often just before a 128 block |
| Spike clusters (pushing, movement) | 1 of 11 | 30–50 s | $\ge 2$ peaks above 100 within 40 s; rise $\ge 20$ units/s |
| Spikes, notches, ripple | 5 of 11 | 1–5 s | $\lvert\Delta x\rvert \ge 10$ in 1 s, back within 5 s |

Natural UA plateaus reach about 24 s of exactly constant value in quiet periods, because UP is integer
data at 1 Hz. A hold rule needs an edge condition, or a long minimum.

Two more types appear in the local `.mat` records:

| Type | Duration | Signature |
|---|---|---|
| Short pulse | 3 s | 0 → 6 → 17 → 11 → 0, about every 300 s; no 1-s step reaches 15 units |
| Clamped rest | 30–150 s | exactly constant at the floor the uterus decays into (0, 10 or 25), between smooth contractions |

A clamped rest is a quiet uterus below the device's zero, not a frozen sensor. The contractions rise
smoothly out of it and decay back into it.

### 2.4 UP is 1 Hz data held over 4 samples

In the shard, 95% of UP changes fall on sample index $n \equiv 1 \pmod 4$. FHR changes fall evenly on all
four phases. So UP is a 1 Hz signal expanded by zero-order hold, while FHR is true 4 Hz data. The
staircase puts energy at 1 Hz and its harmonics. The `up_st` channels with $\xi_c > 0.5$ Hz (about ch1–ch7)
therefore see mostly staircase edges, not physiology. Contractions lie at about 0.004–0.05 Hz (ch20–ch35),
and breathing ripple at about 0.2–0.5 Hz.

### 2.5 How far one bad sample spreads

The transform is causal. A bad sample at raw index $n$ affects only the steps $k$ with
$n \le 16k \le n + S_c - 1$, where $S_c$ is the support of channel $c$. Earlier steps are never affected.

| `up_st` channel | $\xi_c$ (Hz) | 95% support |
|---|---|---|
| ch1 | 1.49 | 24 s |
| ch10 | 0.31 | 35 s |
| ch19 | 0.066 | 92 s |
| ch25 | 0.023 | 224 s |
| ch30 | 0.0098 | 504 s |
| ch35 | 0.0041 | 1172 s |

The `up_ph` pairs have supports of 224–596 s. A per-step scalar mask is therefore the wrong shape. A
5-s spike would hide 20 min of ch35 but only 24 s of ch1. The mask must be per channel.

### 2.6 What the local profile run changed

The first version of the rules ran on the two local records (80 pre-delivery segments, hie_cs). The
example pages showed five defects. Each fix below is in `signal_quality.py`.

| Defect seen | Fix |
|---|---|
| A 15-unit jump inside a steep contraction rise opened a transient that lasted until the signal came back to the pre-jump level. Whole contractions (up to 90 s) became flat bridges. | A jump marks only the discontinuity (about 3 s). A wider span needs specific evidence: a box, a step or a spike. |
| A box closed on a 1-s spike that touched the old level, and 45 s of real low UA became invalid. | A box closes only if the signal stays at the old level for 5 s after the opposite jump. |
| Clamped rests at a floor of 10 or 25 were `HOLD`; 45% of one segment was invalid. | A constant run at the minimum of the past 120 s, with no edge jump and no joint FHR hold, is `CENSORED` (valid) for up to 300 s. |
| A spike that began where a rest ended counted as the rest's end edge. | The end edge also needs the 10-s median after the run to stay 10 units away. |
| The jump out of a frozen run passed the step test, because a frozen run is exactly constant. The correction would shift the rest of the segment by 31 units. | A step that leaves a `HOLD` becomes a plain jump mark. |

Against the exact repair mask from raw `HR2`, the FHR `HOLD`/`LINEAR`/`JUMP` rules had a precision of
0.46 and a recall of 0.11 (890 repaired samples). They also flag smooth, low-variability FHR, which is
itself a sign of the outcome. So when the raw channel is read, `HOLD` and `LINEAR` no longer invalidate
FHR.

After the fixes, on the kept segments: UA valid 99.3%, `up_st` coefficients usable 97.8%, and no
detected contraction lies mostly on invalid UA (before: 0.15 per segment). These numbers come from two
records of one subgroup. The production profile decides the thresholds.

---

## 3. What the literature says

- **TOCO range and meaning.** The Philips Avalon FM20–FM50 manual specifies TOCO as 0–127 units. It
  resets the baseline to 20 units on connection and on reset, and to 0 after more than 5 s below zero.
  The paper scale ends at 100. FIGO 2015 states that TOCO measures contraction *frequency* reliably,
  but not intensity, duration or basal tone (Ayres-de-Campos et al., IJGO 131:13–24). Against IUPC,
  TOCO peak timing is close (−1.2 ± 4.5 s), so timing survives when amplitude does not
  (Euliano et al., AJOG 2013, 208:66.e1).
- **Contraction detection.** Matonia et al. (2006) remove breathing with a 0.04 Hz low-pass filter,
  estimate basal tone as the histogram mode in 4-min windows, and accept a contraction above basal + 10
  units for more than 30 s with an amplitude above 20 units. Euliano et al. reject durations under 45 s
  or over 180 s, and amplitudes under 10 units. FIGO describes a contraction as bell-shaped and 45–120 s long.
- **UA quality.** No large-scale validated TOCO quality index exists. Petrozziello et al. (IEEE Access
  2019) used an AR-model index and kept UA in only 24% of 35,429 CTGs. They also found that more
  signal loss gave more high-risk outputs, whatever the real outcome was: the missingness itself became a
  feature.
- **Gap repair.** Linear repair of FHR lowers short-term variability. At 50% missing samples, STV fell
  by 16–19% (Spilka et al., CinC 2012). Published rules bridge gaps under 15 s and exclude longer ones
  (Warrick & Hamilton, MBE 2020). Better imputation did not improve a CNN classifier on 51,449 births
  (Asfaw et al., 2023).
- **Declipping.** Audio declipping relies on sparse harmonic structure. A contraction crest is one smooth
  pulse, and TOCO amplitude is unreliable anyway, so a rebuilt crest adds a guess on top of an
  unreliable value. Masking the crest, or flagging it as censored, keeps the timing and the bound.
- **Missing-data-aware transforms.** Normalized convolution estimates $(a * (c\,x))/(a * c)$ (Knutsson &
  Westin, 1993). It suits a low-pass stage but not a band-pass wavelet, which loses its zero mean on
  gapped data (Frick et al., 1997). The cone of influence of an edge grows with scale (Torrence &
  Compo, 1998). No published work covers scattering transforms with gaps.
- **UA→FHR coupling.**
  - Kraskov transfer entropy at lags up to 90 s was significant in only 35–44% of 20-min epochs
    (Warrick & Hamilton, 2020). Expect many valid windows with no detectable coupling.
  - Contraction-triggered averaging (BPRSA) tolerates gaps and found coupling in 86% of labours
    (Casati et al., PLoS ONE 2014).
  - Splitting HRV into contraction and rest periods separated asphyxia cases, but pooled HRV did not
    (Warmerdam et al., 2016, 2018). **Rest periods are information, not noise.**
  - The deceleration area 80–140 s after contraction onset correlated best with scalp pH
    (Beguin et al., 1975).

---

## 4. Design

### 4.1 Principles

1. **Mark, do not invent.** A filled sample is always flagged, whatever method filled it.
2. **Stay causal.** A mask value at step $k$ may depend only on raw samples up to $16k$, with one
   declared exception: a bounded lookahead for rules that need a confirmation time (section 4.3).
3. **Clean only inside masked spans.** A bridge may use the next valid sample, because every step that
   could see the bridge before that sample is masked. Thus no future value reaches a valid step.
4. **Validity is hard; informativeness is soft.** Invalid UA is removed. Quiet valid UA stays.
5. **The GUID pool does not change.** The new step must not change the prescreen
   (`eligible_2h`/`eligible_3h`), so the minimization folds stay valid.

### 4.2 Step 0: fix the fill path

1. Replace `MimoSequence.interpolate_input` with the identity during the build. The `mimo` package is not
   tracked in this repository, so do it from the builder.
2. Record the per-sample presence mask before any fill. Upstream `Repaired HR2` has no zeros inside a
   recorded run, and `Aligned UA` uses the same runs. Thus "pre-fill FHR $\ne 0$" marks the recorded
   runs exactly. It is the same rule that `weight` already uses at every 16th sample.
3. Remove `_sanitize_signals`' clip of UP to $[-50, 500]$ (`create_new_pipeline.py:1870`). The new codes
   replace it.

Optional: also read the raw `HR2` channel from the `.mat`. After a 1-sample offset, `Repaired HR2`
equals `HR2` at 99.5% of shared samples, so a direct comparison gives an exact FHR repair mask. Raw
`UA` does not align with `Aligned UA` this simply, so the UA repair mask must come from rules.

### 4.3 Step 1: per-sample quality codes

One `uint8` bit field per signal and per sample. A sample can carry several bits.

**UA codes (`up_quality`)**

As implemented (`QUALITY_VERSION` = 1). A "jump" is $\lvert x_n - x_{n-4}\rvert \ge 15$: 15 units
within one stored 1-s step.

| Bit | Name | Rule (4 Hz, $x$ = UP) | Class | Lookahead |
|---|---|---|---|---|
| 0 | `MISSING` | outside a recorded run, or a padding tail (`weight` = 0) | invalid | none |
| 1 | `CODE` | $x > 127.5$ or $x < 0$ (128, 255) | invalid | none |
| 2 | `CENSORED` | a ceiling run ($x$ = 100 or 127) outside an artifact, for up to 120 s; or a clamped rest: a constant run of $\ge 30$ s at the minimum of the past 120 s (+1 unit), with no edge jump and no joint FHR hold, for up to 300 s | valid | none |
| 3 | `TRANSIENT` | a lone jump: the discontinuity and one stored step on each side; a box: a jump undone within 60 s by an opposite jump that lands, and stays for 5 s, within 10 units of the pre-jump level; a spike: height $\ge 12$ units above both 3-s sides, at half height plus 1 s on each side (also dips); after a gap, a value 10 units from the pre-gap level, until the signal returns or settles; a ceiling run past 120 s | invalid | 65 s (box) |
| 4 | `STEP` | a jump from a settled level (range $\le 6$ over the last 5 s) to a settled 20-s window that starts within 10 s, $\ge 10$ units away in the jump's direction, with no return to the old level within 40 s; not out of a `HOLD` | invalid | 40 s |
| 5 | `HOLD` | a constant run of $\ge 30$ s with a start jump $\ge 10$, an end jump $\ge 10$ that stays 10 s (runs under 60 s), or a joint FHR hold; any other constant run of $\ge 60$ s that is not a clamped rest; a clamped rest past 300 s | invalid | 70 s |
| 6 | `FLOOR` | $x \le 2$ for $\ge 20$ s that is all exact zeros or starts within 30 s after `CODE`/`MISSING`; the part within 60 s before a `CODE`/`MISSING` that follows within 30 s | invalid | 60 s |
| 7 | `SHIFTED` | valid, but the level correction after a `STEP` changed the value | valid | — |

Every invalid sample is bridged (section 4.4), so a separate `BRIDGED` bit is not needed.

Evidence for the thresholds:
- A contraction rises 30–80 units in 30–60 s, so the physiological slope is at most about 3 units/s.
  A 15-unit jump in 1 s is at least 4× above that. But a steep rise can still contain one jump, so a
  jump alone marks only itself.
- A contraction crest falls about 4 units within 3 s on each side; a spike needs 12.
- Every 128 and 255 run in the shards starts with a jump of 30 or more.
- The 120-s ceiling limit follows the 180-s contraction-duration limit of Euliano et al., minus the flanks.
- The 30-s and 60-s hold limits sit above the 24-s natural plateaus seen in the samples. The 300-s rest
  limit is the builder's flat-UA gate.

The lookahead of the whole rule set is 70 s; `QualityConfig.lookahead_s` computes it, and
`as_attrs()` writes it as `quality_lookahead_s`.

**FHR codes (`fhr_quality`)**

| Bit | Name | Rule | Class |
|---|---|---|---|
| 0 | `MISSING` | outside a recorded run, or a padding tail | invalid |
| 1 | `REPAIRED` | the raw channel is absent, 0, 255 or different at this sample | invalid |
| 3 | `JUMP` | $\lvert\Delta \mathrm{FHR}\rvert \ge 20$ bpm within 0.25 s, $\pm 0.5$ s | invalid |
| 5 | `HOLD` | exactly constant for $\ge 15$ s (natural integer runs reach about 11 s) | invalid only without the raw channel |
| 6 | `LINEAR` | monotone, rise $\ge 2$ bpm, 95th-percentile residual of a line fit $\le 0.75$ bpm, for $\ge 15$ s | invalid only without the raw channel |

The raw channel is the repaired channel's name without "Repaired " (`HR2` for `Repaired HR2`). The
profile places each segment on the repaired channel's grid first (offset with the most exact matches,
$\pm 4$ samples), then reads the raw channel at one record-wide offset ($\pm 1$ sample). On the two
local records the placement matched 100% and the raw channel 98–99% of present samples.

Suspected maternal heart-rate capture (flat 70–110 bpm with jumps at both ends) is a later bit. It needs
its own validation.

The FHR mask matters for the model even before the gate exists. A linear repair across a gap is drawn
toward the FHR value *after* the gap. A forecasting model that sees the ramp can therefore "predict" the
future. Masking repaired FHR in the target loss and the KL support removes this leak.

### 4.4 Step 2: cleaning before the transform

1. **Bridge every invalid span.** Use a shape-preserving cubic (PCHIP) from the last valid sample before
   the span to the first valid sample after it. At the window start, hold the first valid value backward;
   at the window end, hold the last valid value forward. Every replaced sample is invalid. The
   bridge does not add information. It stops a span edge from exciting every channel with a broadband
   step, which is the arc that the sample pages show at every 128 edge.
2. **Correct the level after a `STEP`.** Subtract the level difference from every sample after the
   settle point. The difference is estimated from samples inside the masked relaxation tail, so no valid
   step sees a future value. This is defensible because TOCO basal tone is arbitrary: the monitor sets
   it to 20 on reset. A causal basal-tone removal (a running 4-min mode) is a stronger variant for a
   later ablation. One caveat from the local data: a fast drop to about 10 can be a nurse's re-zero,
   which *removes* an offset. The correction then carries the old offset forward. Only `up_st` ch0
   (the low-pass level) feels this, because the wavelets have zero mean; the profile counts steps so
   that you can judge how often it matters.
3. **Leave censored crests as they are.** Do not declip.
4. **Handle the 1 Hz staircase.** The default is to keep the channels in the file and let the model's
   channel gate drop `up_st` ch1–ch7 ($\xi_c > 0.5$ Hz). The alternative is to rebuild UP at 4 Hz by
   linear interpolation of the 1 Hz samples before the transform. That costs a 0.75-s lookahead and
   breaks strict causality, so I do not recommend it as the default.
5. **Do not add a general low-pass filter** to the model input. A causal low-pass filter has a group
   delay, which shifts the stored timeline. Any filtering for *detection* (for example the 0.04 Hz
   low-pass for contraction detection) stays inside the detector.

Write the cleaned UA to `up`, so that `up` and `up_st` describe the same signal, and keep the original
in `up_raw` as `uint8`. Do the same for FHR (`fhr`, `fhr_raw`).

### 4.5 Step 3: carry the mask to the coefficient grid

For a coefficient channel $c$, let $w_c = |\psi_c| * \phi$ be the causal magnitude kernel. It bounds how
much raw sample $n - j$ can affect the coefficient at raw index $n$. Let $b_n = 1$ for every invalid or
bridged sample. Before the window start, the edge pad repeats sample 0, so $b_{n<0} = b_0$. The
contamination of channel $c$ at step $k$ is

$$\rho_c[k] = \frac{\sum_{j \ge 0} w_c[j]\, b_{16k - j}}{\sum_{j \ge 0} w_c[j]} \in [0, 1].$$

- $\rho_c$ is causal by construction, because it uses only samples at or before $16k$.
- For a phase pair, use the kernel of the slower leg.
- Compute $\rho$ in the same GPU batch as the transform, as one more causal convolution of $b$ with the
  existing filter bank.
- A coefficient is valid if $\rho_c[k] \le \varepsilon$. Use $\varepsilon = 0.1$ as the default.
- Store $\rho$ as `uint8` ($\mathrm{round}(255\rho)$), so that a loader can change $\varepsilon$ without
  a rebuild.

A short bridged gap is handled without a special rule. A 10-s gap fills a large part of the 24-s
support of ch1 but less than 1% of the 1172-s support of ch35. So $\rho$ masks the fine channels around
the gap and keeps the slow channels valid.

### 4.6 Step 4: informative UA

- **Contraction detector.** Run it on cleaned, valid UA only:
  - basal tone = the causal running mode over the last 4 min of valid samples;
  - a contraction is $x - \text{basal} \ge 10$ for at least 30 s, with an amplitude of at least 15 units
    and a duration of 45–180 s;
  - censored samples count as above the threshold.
  
  Validate it against the PeriGen contraction markings (start, peak, end), which the colleague's
  repository already reads from the Patterns XML.
- **Do not hard-mask quiet UA.** Instead, give the later gate a soft prior: contraction onset to
  156 s after it, which is the model's lag range of 39 steps × 4 s.
- **The existing contraction weight reads raw UP.** `lag_attn_rws/event_weight.py`
  (`contraction_anchor_weight`) currently runs its detector on raw `up`, with only `weight` as the
  validity input. It therefore finds "contractions" at the 128 blocks. It must take the cleaned `up` and
  the UA mask.
- **Per-segment scalars** for filtering, stratification and shortcut checks: `up_valid_frac`,
  `up_censored_frac`, `fhr_valid_frac` and `n_contractions`.
- **Do not reject a segment because its UA is bad.** The FHR in that segment is still useful. Mask the UA
  instead.

### 4.7 New HDF5 content

| Field or attribute | Shape and type | Approximate size per sample (raw, before LZF) |
|---|---|---|
| `up_quality`, `fhr_quality` | (N, 5280) uint8 | 5.3 KB each; mostly zeros |
| `up_raw`, `fhr_raw` | (N, 5280) uint8 | 5.3 KB each |
| `up_st_rho`, `up_ph_rho` | (N, 36, 330), (N, 10, 330) uint8 | 15 KB together; mostly zeros |
| `fhr_st_rho`, `fhr_ph_rho` | (N, 36, 330), (N, 44, 330) uint8 | 26 KB together; mostly zeros |
| `up_valid_frac`, `up_censored_frac`, `fhr_valid_frac`, `n_contractions` | (N,) | — |
| attrs `quality_version`, `quality_lookahead_s`, the thresholds, the kernel digest | file attributes | — |

`up` and `fhr` keep their shape and become the cleaned signals. Compare with `fhr_st` alone, which is
47 KB per sample as float32. The masks compress well, because they are mostly zeros inside per-sample
chunks.

### 4.8 Integration points

Line numbers are from the review and can drift.

| Where | Change |
|---|---|
| `create_new_pipeline.py:2942-2945` | Patch out the MIMO fill and keep the pre-fill presence mask. |
| after `:2945`, before `_sanitize_signals` (`:1855-1874`) | Compute the codes; bridge and correct the signals. |
| `:2968-2971`, `:3012`, `:3115-3174` | Carry the new arrays through dedup, `valid_indices` and the `b_*` lists. |
| `create_initial_hdf5` (`:687-1010`), `append_samples_batch` (`:1063-1159`) | Create and write the fields. A field that is created but never written stays zero without an error, so test this. |
| `_validate_geometry` (`:1637-1718`), causal attrs (`:622-684`) | Add the quality attributes, so that the loader's equality check covers them. |
| segment gates (`:2975-3006`) | Compute the flat gates on the pre-fill signals: the fill writes the flats that the gate rejects (section 2.2). Decide after the production profile, which reports both gate results. Change only the build's per-segment gate. The prescreen (`_quality_filter_segments`, `:1877`) uses the same gates for `estimated_valid_hours_6h` and so for `eligible_2h`/`eligible_3h`; changing it there changes the GUID pool and invalidates the minimization folds. |
| `build_fold_datasets.py` `is_complete` (`:464-478`) | Also compare `quality_version`; otherwise old files count as complete. |
| `calculate_dataset_stats.py` | Compute the coefficient statistics over valid steps only ($\rho \le \varepsilon$). Artifacts inflate the variance today. |
| `hdf5_dataset.py` (`:1792-1855`) | Add a mask group: trim it like the coefficient blocks, read it without the float cast, never normalise it, and give a default when a file lacks it. |
| `lag_attn_rws/sample_page.py` | Show UP up to 130, mark the codes and add a mask row. |

The model side (later): the lag attention collapses a per-sample mask to `m_lag[0]` and broadcasts it
over the batch (`lag_attn/nets/attention.py:229, 274`). A per-sample key mask is therefore dropped today
without an error. Also, if the mask reaches the posterior but not the prior, it repeats the "clock" leak
of `LAG_READOUT_DIAGNOSIS.md` §5.

---

## 5. Scattering-based detection

The scattering transform does see these artifacts. A jump or spike excites every first-order channel
at once (the vertical arcs in panel d of the sample pages), while a contraction excites only the
0.004–0.05 Hz channels. A per-step score such as "the share of channels above their 95th percentile" or
"the energy above 0.1 Hz over the energy in the contraction band" flags transients.

I recommend it as an audit, not as the primary detector, for three reasons:
- The raw rules give sample-level timing. The scattering view is smeared by $T = 16$ and by the
  channel supports.
- The raw rules are simple and causal.
- Deciding validity from the coefficients of the corrupted signal is circular.

Use the score after cleaning instead. A broadband event inside a step that the mask calls valid is a
detector miss. Count these misses per release. A learned detector (one-class, trained on audited clean
frames) is an option only if the audit shows misses that the rules cannot catch.

---

## 6. Risks

- **The missingness shortcut.** Signal loss correlates with outcome (Petrozziello et al.), and the
  128-block rate may differ between subgroups and between labour stages. A mask can then teach the
  classifier the label without any physiology.
  - Report the mask statistics per subgroup in the fold EDA.
  - Evaluate per quality stratum.
  - During training, mask random valid UA spans, so that "masked" does not mean "abnormal".
- **Lookahead.** The box, step, hold and floor rules mark a span before their confirmation time ends,
  up to 70 s ahead. The attribute `quality_lookahead_s` declares the bound. If strict causality is
  required, mark from the confirmation point only, and accept that the start of each span stays unmasked.
- **FHR rules as a label shortcut.** `HOLD` and `LINEAR` also fire on smooth, low-variability FHR, and low
  variability is a sign of hypoxia. With the raw channel they do not invalidate. Without it, report
  their rate per subgroup before you trust them.
- **Fill-made flats bias the GUID pool.** The prescreen drops segments, and so records, whose UA rests at
  0 (section 2.2). The current pool already carries this bias. The profile's default draw
  (`eligible_only`) therefore undercounts the effect; run it once with `eligible_only` off as well.
- **Changed statistics.** The cleaned signals and the masked statistics change the normalisation. Do not
  mix old and new files in one training run; the `quality_version` check prevents this.
- **Over-masking.** The 1172-s support of ch35 means that one long gap can hide that channel for most of
  a window. That is honest, but measure how often it happens before you train.

---

## 7. Validation plan

1. **Profile the production data (read only)** with `hdf5_dataset/new_pipeline/profile_signal_quality.py`.
   Edit its `RUN_ARGS` and press Run on the production box. It draws `records_per_subgroup` records per
   subgroup from `guid_screening_results.csv` and writes, per subgroup:
   - the share of each code value (100, 101–126, 127, 128, 255) and every code run with its entry jump
     and the rule's decision (`ua_code_runs.csv`, figures 01–02);
   - what the MIMO fill changes, and the builder's gates with and without it (figures 03 and 10);
   - the share of each quality bit, the valid shares, and the usable coefficients per channel
     (figures 04–07, `channel_validity.csv`);
   - contractions found today and after cleaning (figure 08), and the FHR repair share with the rules'
     precision and recall (figure 09);
   - example pages per category, with the coefficients before and after cleaning (`examples/`).
2. **Unit-test every rule** on synthetic signals: a bell contraction clipped at 100, a 128 block entered
   from 1, a step with relaxation, a hold, and a spike cluster.
3. **Hand-label about 150 windows**, stratified by detector output. The target is precision $\ge 0.9$ for
   the invalid classes, and a censored-versus-artifact accuracy on the 100 runs.
4. **Compare the contraction detector** with the PeriGen contraction markings.
5. **Run the scattering audit** of section 5 on the cleaned build.
6. **Check the model downstream.** $K_t$ at the old 128 edges should fall to about 0. $K_t$ at clean
   contractions with decelerations should stay. The lag-argmax should no longer point at block edges.

---

## 8. Phased plan and open decisions

| Phase | Work | Output |
|---|---|---|
| 0 | A profile script (read only), run on the production box | numbers that confirm or change section 4.3 |
| 1 | `hdf5_dataset/signal_quality.py`: pure NumPy codes, bridge, $\rho$ kernels, with unit tests | a tested module |
| 2 | Builder integration (section 4.8), with `quality_version` = 1 | a new dataset variant |
| 3 | Loader mask group, masked statistics, sample-page rows | a loadable, visible mask |
| 4 | Model: key mask in the lag attention, gate prior, contraction weight on cleaned UA | a separate task |

Status: Phases 0 and 1 exist (the unit tests are not written yet). Phase 2 waits for the production
profile.

Decisions taken (2026-10-25, the recommended option each time):
1. Store $\rho$ in the file.
2. `up` and `fhr` become the cleaned signals, with `*_raw` beside them.
3. Level correction after a `STEP`: on (`clean_segment(..., correct_steps=True)`).
4. The UP staircase: drop ch1–ch7 in the model, not in the data.
5. Read raw `HR2` for an exact FHR repair mask.
6. A declared lookahead, now 70 s (section 4.3).

Open after the profile:
7. Compute the build's flat gates on the pre-fill signals (section 4.8).
8. Confirm the thresholds of section 4.3 per subgroup, mainly the spike height and the clamped-rest test.
