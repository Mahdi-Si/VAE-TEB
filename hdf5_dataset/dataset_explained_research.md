# CTG Dataset Reference

The authoritative reference for the Cardiotocography (CTG) dataset used for
fetal outcome prediction: what the signals are, how a raw recording becomes
model-ready segments, the exact mathematics of the scattering and
phase-harmonic features, which coefficients are kept and why, the on-disk
schema, and how to load it in PyTorch.

Every number here was read from the code or measured by running it at the
production geometry ($J = 11$, $Q = 4$, $T = 16$, `shape` $= 5280$,
$f_s = 4$ Hz). Where a figure is *derived* rather than *enforced by a
constant*, that is said explicitly.

**A shard is described by three independent axes, not one.** A width or a
coefficient value quoted without them is ambiguous, so every table below that
depends on one of them says which:

| Axis | Values | Resolved default | Recorded as |
|---|---|---|---|
| `transform` | `two_sided` \| `causal` | `two_sided` | root `transform` (absent = legacy two-sided) |
| `leg_alignment` | `none` \| `envelope` | `none` | root `causal_leg_alignment` (absent = `none`) |
| `phase_operator` | `ratio_power_v0` \| `integer_harmonic_v1` | `ratio_power_v0` | root `causal_phase_operator` (absent = `ratio_power_v0`) |

The latter two apply to the causal arm only. "Resolved default" is what
`main()` substitutes for an unset value (`create_new_pipeline.py:3757-3763`);
it is **not** what the checked-in `RUN_ARGS` launch dict selects, which is a
causal, `envelope`, `integer_harmonic_v1` build (§11.3). Legacy-operator
shards are what the resolved defaults have always produced, and two shipped
model packages already declare the integer widths (§8), so this document gives
both geometries side by side rather than choosing one.

**Code map**

| Concern | File |
|---|---|
| Dataset creation (current) | `hdf5_dataset/new_pipeline/create_new_pipeline.py` |
| Dataset creation (superseded) | `hdf5_dataset/create_hdf5_dataset.py` |
| Transform implementation (two-sided) | `hdf5_dataset/kymatio_phase_scattering.py` |
| Transform implementation (causal) | `hdf5_dataset/causal_scattering.py`, `hdf5_dataset/causal_scattering_torch.py` |
| Causal transform mathematics | `documents/docs-md-files/datasets/CAUSAL_SCATTERING_PHASE_HARMONIC_MATH.md` |
| Normalisation statistics | `hdf5_dataset/calculate_dataset_stats.py` |
| PyTorch segment dataset | `hdf5_dataset/hdf5_dataset.py` |
| PyTorch GUID-sequence dataset | `hdf5_dataset/guid_hdf5_dataset.py` |
| Length-bucketed sampling | `hdf5_dataset/length_bucket_sampler.py` |
| Channel-selection rationale | `documents/docs-md-files/datasets/PHASE_HARMONIC_CHANNEL_SELECTION.md` |
| Pipeline self-check | `hdf5_dataset/smoke_check_channel_selection.py` |
| Diagonal-redundancy measurement | `hdf5_dataset/check_phase_diagonal_redundancy.py` |
| Dataset tests | `hdf5_dataset/tests/`, `hdf5_dataset/test_causal_scattering.py` |

> Two pointers in this table were wrong until now and are worth flagging,
> because four code sites still carry the old ones. The channel-selection and
> causal-mathematics documents live under `documents/docs-md-files/datasets/`,
> not under `hdf5_dataset/`; `create_new_pipeline.py:171`, `:1177`, `:1315-1317`
> and `check_phase_diagonal_redundancy.py:3` still cite the `hdf5_dataset/`
> path, which resolves to nothing. `new_pipeline/kfold_dataset_modes.md`, cited
> by §6.3 in earlier revisions, does not exist anywhere in the repository; the
> reference has been dropped rather than repaired.

---

## 1. Clinical Context

Electronic Fetal Monitoring (EFM), or Cardiotocography, simultaneously records
**Fetal Heart Rate (FHR)** and **Uterine Pressure (UP)** to assess fetal
oxygenation during labour. Both are sampled at $f_s = 4$ Hz.

FHR is not a directly sampled waveform. It is an instantaneous *rate* derived
from beat detection and resampled onto a uniform 4 Hz grid — a fact that sets a
hard ceiling on its meaningful bandwidth (§7.5). UP comes from a
tocodynamometer, a surface pressure transducer whose trace above the
contraction band is dominated by maternal movement and respiration rather than
uterine activity.

The physiological relationship of interest is directional: contractions
transiently reduce placental perfusion, and the fetal response appears in FHR
as decelerations and as changes in variability, delayed by 20–120 s. The
dataset is built so a model can learn **UP $\rightarrow$ FHR** coupling, which
is why the source (UP) and target (FHR) feature blocks are kept separable.

Each recording is one delivery, identified by a **GUID** derived from the
`.mat` filename stem, e.g. `56357ED5E7DD412F8897C60A9D926C9D`.

---

## 2. Notation and Key Concepts

| Symbol | Meaning |
|---|---|
| $f_s = 4$ Hz | Sampling rate of both raw signals |
| $x(t)$ | A raw signal (FHR or UP) |
| $\psi_i$ | Analytic Morlet wavelet $i$ of the first-order bank |
| $\xi_i$ | Centre frequency of $\psi_i$ in **cycles per sample** (kymatio's normalised unit) |
| $\xi_i f_s$ | The same centre frequency in Hz |
| $\phi$ | Low-pass averaging filter. Its Gaussian scale is $\sigma_{\text{low}} = 6.249875 \times 10^{-3}$ cycles/sample, so its time width is $1/(2\pi\sigma_{\text{low}}) = 25.47$ samples $= 6.37$ s — wider than the 4 s decimation step $T$ that follows it |
| $z_i = x * \psi_i$ | Complex wavelet coefficient at scale $i$ |
| $p = \xi_j / \xi_i$ | Harmonic ratio of a wavelet pair, $p \ge 1$. The exponent actually applied under `ratio_power_v0`; under `integer_harmonic_v1` the exponent is $k_{ij} = \operatorname{round}(p)$ instead (§7.3) |
| $k$ | Harmonic **step** on the power grid, $p = 2^{k/Q}$. Not to be confused with $k_{ij}$, the integer harmonic |
| $k_{ij}$ | Integer harmonic of a pair, $\operatorname{round}(\xi_j/\xi_i) \in \{2, 4\}$ at the production bands |
| $T = 16$ | Decimation factor, and the invariance scale in samples (4 s at 4 Hz). **Not** $\phi$'s time width |
| $J = 11$, $Q = 4$ | Octaves and wavelets per octave |
| $W_c$ | Causal warm-up of channel $c$, in decimated steps |
| $\tau_g$ | Group delay of a causal filter, $n/(2\pi b)$ |
| $c_y$, $c_u$ | Target-encoder and source-encoder input widths. Both are per-variant *and* per-operator (§8) |

| Term | Definition |
|---|---|
| **GUID** | One infant / one delivery / one monitoring session. Many segments share a GUID. |
| **Segment** | A fixed 5280-sample (22 min) window. 330 time steps after decimation; **300 (20 min)** after trimming at the production `trim_minutes = 1.0`, which is the only value any shipped config uses (§3.3). |
| **Epoch** (`domain_start`) | Segment start time in **seconds relative to delivery**. Negative = before delivery. |
| **Subgroup** | One of 8 outcome × delivery-mode × blood-gas categories (§5.2). One HDF5 file each. |
| **Fold** | One of 10 cross-validation splits, named `fold_1` … `fold_10`. |
| **TLO** | Time from Labour Onset, seconds. NaN when the GUID is absent from the labour-onset CSV. |
| **Weight** | Per-timestep validity mask, **binary** $\{0, 1\}$ in this pipeline. |

---

## 3. From Recording to Segments

### 3.1. Constants

Defined at `create_new_pipeline.py:140-146`:

$$\begin{aligned}
\texttt{BASE\_BLOCK\_SIZE} &= 3520 \\
\texttt{SIGNAL\_LENGTH} &= \lfloor 3520 \times 1.5 \rfloor = 5280 \ \text{samples} = 1320\ \text{s} = 22\ \text{min} \\
\texttt{OVERLAP\_PERCENTAGE} &= 1/11 \\
\texttt{STEP\_SIZE} &= \lfloor 5280 \times (1 - 1/11) \rfloor = 4800\ \text{samples} = 1200\ \text{s} = 20\ \text{min} \\
\texttt{SEQUENCE\_LENGTH} &= 5280 / 16 = 330\ \text{time steps} \\
\texttt{SEGMENT\_DURATION\_SEC} &= 5280 / 4 = 1320.0\ \text{s} \\
\texttt{STEP\_DURATION\_SEC} &= 4800 / 4 = 1200.0\ \text{s}
\end{aligned}$$

$5280 \times (1 - 1/11) = 4800.0$ exactly, so the `int()` truncation in
`STEP_SIZE` loses nothing.

> **Two of these constants are dead.** `SEQUENCE_LENGTH` (`:143`) and
> `STEP_SIZE` (`:145`) are defined and never read: the build recomputes the
> sequence width inline as `sequence_length = SIGNAL_LENGTH // 16` (`:3379`),
> and the 4800-sample stride is produced inside MIMO from
> `overlap_percentage`, not from `STEP_SIZE`. Only `SEGMENT_DURATION_SEC` and
> `STEP_DURATION_SEC` are consumed (`:149`, `:1993`). Editing `STEP_SIZE`
> changes nothing about the dataset.

### 3.2. Segmentation

Segmentation itself is **not** implemented in `create_new_pipeline.py`; it is
delegated to the MIMO sequence layer
(`mimo/MIMO_Sequence_Trainer/mimo/sequence/mimo_sequence.py`). The pipeline
passes `base_length=3520` and the $1.5\times$ expansion to 5280 happens inside
MIMO (`mimo_sequence.py:405`).

A GUID's recording is tiled by a sliding window:

- **Window length:** 5280 samples (22 min)
- **Stride between consecutive segment starts:** 4800 samples (20 min)
- **Overlap:** $5280 - 4800 = 480$ samples (2 min)

The overlap fraction $1/11$ is of the *segment* (5280), not of the base block.
Tail handling (`mimo_sequence.py:466-473`) has three branches on the remaining
length $L$, and the boundaries are inclusive at the top:

| Condition | Emitted |
|---|---|
| $L \ge 7040$ ($= 2 \times 3520$) | a full 5280 window |
| $5280 < L < 7040$ | $\lfloor L/2/16 \rfloor \times 16$ samples |
| $L \le 5280$ | all of $L$ |

so a remainder of exactly 7040 takes the *first* branch. Short tail segments
use no overlap.

**The five stages MIMO actually runs.** `prepare_data` is called once per
record from `_run_mimo_pipeline` (screening, `create_new_pipeline.py:1798-1812`)
and from `create_hdf5_dataset_from_records_list` (build, `:2905-2913`), with
identical flags — `batch_size=1`, `do_evaluate=True`, `align_left=True`,
`do_split=True`, `do_pad=True`, `do_reflect=True`, `base_length=3520`,
`do_equalize=True`, `do_merge=True`, `max_domain_start=[\infty, \infty]`,
`overlap_percentage=1/11`. Only `min_domain_start` differs between the two.
In order:

1. **`split_long`** — windowing, preceded by equalisation padding.
2. **`merge_sequences` — skipped.** Its guard is
   `do_merge and overlap_percentage == 0.0` (`mimo_sequence.py:2152-2154`) and
   the pipeline always passes $1/11$, so `do_merge=True` has no effect. This
   is *why* duplicate `domain_start` values survive to §3.8.
3. **`filter_by_domain_start`** then **`interpolate_input`** (§3.5).
4. **`generate_block_sequences(align_left=True, do_reflect=True)`**, which also
   builds the sample weights (§3.6).
5. **`pad_number_of_sequences(1)`** — a no-op, since $n \bmod 1 = 0$.

**Equalisation puts segment starts on a global grid.** `do_equalize=True`
zero-pads each continuous input section on both sides
(`mimo_sequence.py:437-451`), by
$\texttt{pad\_eq\_len\_l} = \lfloor \texttt{domain\_start}/\texttt{domain\_incr} \rfloor \bmod 5280$
on the left and $5280 - (L \bmod 5280)$ on the right, so boundaries land on
multiples of 5280 samples. The target is padded over the same spans with the
pad label, so equalisation fill carries weight 0.

**Block layout.** `align_left=True` places the real samples at block indices
$[0, \texttt{thisLength})$ of the 5280-wide block and puts all fill on the
right. With `do_reflect=True` the region
$[\texttt{thisLength}, \texttt{thisLength} + \texttt{pad\_length})$ carries the
segment mirrored end-first, where
$\texttt{pad\_length} = \min(5280 - \texttt{thisLength}, \texttt{thisLength})$;
anything beyond that stays zero (`mimo_sequence.py:583-627`). So reflection
padding enters the data **here**, at block assembly, as well as inside the
two-sided transform (§3.3).

**A third, unadvertised filter.** `filter_by_domain_start`
(`mimo_sequence.py:761-784`) keeps a segment only if its `domain_start` is in
window *and* its decimated target is not all-pad. Labels are
`['PAD','HIE','ACIDOSIS','HEALTHY']` after `add_pad_label`, so a segment whose
sampled FHR is zero at every one of its 330 decimated positions is dropped
here, before the quality filters of §3.7 ever see it.

**Channel order inside MIMO is UP first.** The adaptor's channel map is
`{'UP': 0, 'FHR': 1}` (`mimo_adaptor.py:224`) and the pipeline unpacks
accordingly: `fhr = block_input[:, :, 1]`, `up = block_input[:, :, 0]`
(`create_new_pipeline.py:1836-1837`, `:2930-2931`). This is the opposite of
the order the two-sided transform is fed in (§7.4), which is worth knowing
before reading either.

### 3.3. Decimation and Trimming

The scattering transform decimates by $T = 16$, taking each segment from 5280
raw samples to **330 time steps** (one step = 4 s).

Trimming exists for a different reason on each arm, and the shipped causal
build does not use the one usually quoted:

- **Two-sided.** The transform reflection-pads to $N = 8192$ (1456 raw samples
  = 364 s each side, §7.2), so the outermost coefficients are dominated by
  mirrored signal. Trimming removes those edge artefacts.
- **Causal.** No reflection is applied at all — the causal chain prepends
  $32767$ samples of left-only assumed history and refuses reflection by name
  (`causal_scattering.py:947-974`). Trimming there is **not** an artefact
  removal; it is what makes consecutive segments tile (below), and per-channel
  edge validity is handled instead by `causal_warmup_steps` (§10.3).

With `trim_minutes` $= 1.0$, using the shared helper `decimated_trim_steps`
(`hdf5_dataset.py:445-466`, with `RAW_SAMPLING_HZ = 4` and `DECIMATION = 16` at
`:441-442`):

$$\texttt{trim\_samples\_raw} = \lfloor 4 \cdot 60 \cdot 1.0 \rfloor = 240,
\qquad
\texttt{trim\_samples\_decimated} = \lfloor 240 / 16 \rfloor = 15$$

so a trimmed segment is $330 - 2 \times 15 = \mathbf{300}$ time steps
(20 min) and the raw arrays become $5280 - 2 \times 240 = 4800$ samples.

Because the stride (20 min) equals the trimmed window (20 min), consecutive
segments from one GUID tile the timeline continuously after trimming, with no
gaps and no overlap: a segment starting at $s$ keeps raw span
$[s + 240,\ s + 5040)$ and the next start is $s + 4800$. This is why
`trim_minutes = 1.0` is the correct value and not a free parameter: it is what
makes the segments tile.

> **Tiling is a property of *retained* segments only.** Deduplication (§3.8),
> the mean-weight gate, the flat-region gate (§3.7) and the post-delivery drop
> (§3.9) all remove segments from the sequence. Two stored segments that are
> adjacent in the file are not necessarily adjacent on the clock; use `epoch`,
> or `SignalSequenceDataset`'s `delta_t` / `segment_indices` (§10.2), rather
> than assuming contiguity.

#### Where trimming happens — and where it must not

**The model input is a 20-minute (300-step) segment. Trimming is the data
layer's job, applied at load time by `CombinedHDF5Dataset.__getitem__`
(`hdf5_dataset.py:1810-1825`). The model performs no trimming and must not.**

The HDF5 stores the untrimmed 330 / 5280 geometry; the loader slices it:

| Field class | Slice | Result |
|---|---|---|
| `fhr`, `up` | `[240:-240]` | 5280 → **4800** |
| `fhr_st`, `fhr_ph`, `fhr_up_ph`, `up_st`, `up_ph` | `[:, 15:-15]` | 330 → **300** |
| `target`, `weight` | `[15:-15]` | 330 → **300** |

Metadata (`guid`, `epoch`, labels, TLO) is never trimmed.

Set it in the loader's `dataset_kwargs`:

```yaml
dataset_kwargs:
  trim_minutes: 1.0     # -> 300 steps (20 min); required for model input
```

Three failure modes worth guarding against:

1. **`trim_minutes` defaults to `None`, i.e. no trim.** Omit it and the loader
   silently yields 330 steps. A model config declaring `sequence_length: 300`
   does **not** check this — it is a declaration, not an assertion.
2. **The stats file must be built with the same `trim_minutes`.** Statistics
   are accumulated over the trimmed region (§9.3), so a mismatch normalises
   with the wrong $\mu$ and $\sigma$. The loader only emits a
   `warnings.warn` on mismatch (`hdf5_dataset.py:1503-1507`) — it does not fail.
   Worse, the two sites do not share the arithmetic: `calculate_dataset_stats.py:124-129`
   reimplements it with bare literals (`int(4 * 60 * trim)`, `// 16`) instead
   of calling `decimated_trim_steps`, whose own docstring says it exists so the
   conversion is written once. The two agree for every value in the repository;
   the duplication is a drift risk, not a live bug.
3. **The repository is now consistent on the value**, which it was not when
   this section was first written. Measured across the tree:

   | Location | Value | Yields |
   |---|---|---|
   | all ten `teb_vae/*/configs/default.yaml` | `1.0` | 300 steps (20 min) ✓ |
   | `calculate_dataset_stats.py` `RUN_ARGS` | `1.0` | stats over 300 steps ✓ |
   | `train/config.yaml`, and two `mimo/.../seq_vae_teb` configs | `2.0` | 270 steps (18 min) |

   The `tiny.yaml` / smoke / sweep variants set no value and inherit `1.0`
   through `base: default.yaml`. **No config anywhere sets `null`.** The three
   `2.0` files are the legacy MIMO-era stack. The former warning about the
   statistics calculator shipping `trim = 2` no longer applies — its `RUN_ARGS`
   ships `1.0` — but the pairing rule still does: build the stats file at the
   same `trim_minutes` the loader will use.

### 3.4. UP Time Shift

The MIMO adaptor is constructed with `up_shift_secs=-20` at **both** call
sites — the screening adaptor (`create_new_pipeline.py:1787`) and the build
adaptor (`create_new_pipeline.py:2897`) — and the shift is applied inside
`read_input_common_2d` (`mimo_adaptor.py:265-285`) as

$$\Delta = \operatorname{round}\!\left(\frac{-20}{\texttt{domain\_incr}}\right)
\ \text{native samples} \;=\; -80 \ \text{samples on a } 4\ \text{Hz record}$$

i.e. **the UP trace is advanced 20 s earlier** relative to FHR, zero-filled at
the tail. Three details of *how* it is applied:

- It uses the record's **native** `domain_incr`, and it runs before the
  per-segment resampling to $f_s = 4$ Hz (`mimo_adaptor.py:299-302` and
  `:318-324`). The displacement is therefore $-20$ s whatever the source rate;
  $-80$ samples is the 4 Hz instance, not the general case.
- It rewrites the whole per-record UP channel before any segment is sliced
  (`:306-308`), so it is not a per-segment operation.
- Because $\Delta < 0$ the branch taken is
  `up_shifted[:-abs_shift] = up_signal[abs_shift:]` on an array pre-filled by
  `np.zeros_like`, so the **last 80 samples of the record's UP channel are
  zero**. FHR is untouched. If $|\Delta|$ exceeded the record length the whole
  UP channel would become zeros.

**Downstream convention (binding, 2026-09-05):** The stored UP/FHR timeline is canonical: the dataset builder shifts the UP channel when it writes the shards, that shift is part of how the stored signals are, and nothing downstream adds it back, subtracts it, budgets it or interprets it. Every
lag-resolved readout, lag formula, plot axis, simulation and evaluation record is expressed on
the stored grid as written, with no "raw-file" or "sensor-timeline" correction. The former
`MECHANICAL_SHIFT_SECONDS`, `lag_original_sensor_seconds` and `eval_config.up_shift_secs`
consumers were removed for that reason. The source carries an unresolved
`# TODO: Check with Phil for global UP shift`
(`mimo/EarlyMaestra/early_maestra/adaptor/mimo_adaptor.py:265`, immediately
above the shift block); resolving it is a dataset-creation question and
would produce a new dataset version, never a downstream correction.

The dataset test suite enforces the same rule on itself:
`hdf5_dataset/tests/test_phase_operator.py:3-4` and
`test_preprocessing_availability.py:8-11` both state in their module
docstrings that the shift is part of the signal and is outside their audit,
which is what stops it re-entering through a test fixture.

### 3.5. Signal Sanitisation

`_sanitize_signals` (`create_new_pipeline.py:1843-1862`), called after the
MIMO block extraction and before deduplication (`:1962` screening, `:2935`
build), in order:

1. `interpolate_bad_values` (`:447-475`) on FHR, then on UP: linear
   interpolation over non-finite values, with flat extrapolation at the edges;
   an all-bad row becomes all zeros.
2. Clipping: FHR to $[0, 500]$ bpm, UP to $[-50, 500]$ mmHg.
3. Denormal flush: `arr[(arr != 0) & (np.abs(arr) < tiny)] = 0.0` with
   `tiny = np.finfo(np.float32).tiny`.

**Availability audit.** `hdf5_dataset/tests/test_preprocessing_availability.py`
asks which of these steps read samples from *after* the time they repair — the
question that matters if any of this is ever run online rather than
retrospectively. Result: only the **interior** non-finite bridge is
retrospective, because `np.interp` bridges between two valid knots and the
right knot lies after every repaired sample. A leading non-finite run reads the
first valid sample (a future read confined to the record head); a trailing run
holds the last valid value and is prefix-equivalent; the clip and the denormal
flush are pointwise. The same file records that the retrospective branch is
probably dead on the real path anyway: the adaptor zero-fills any segment
containing a NaN and marks its targets as pad (`mimo_adaptor.py:308-313`), so
no NaN reaches `_sanitize_signals`.

> **A separate, earlier gap-fill happens inside MIMO and is not this function.**
> `MimoSequence.interpolate_input` (`mimo_sequence.py:2096-2139`) runs at stage
> 3 of §3.2, before block generation, and fills gaps with a **constant**,
> `signal[left] + (signal_l - signal_r)/\texttt{gap\_len}`, not with a linear
> ramp; the line carries `# todo: Debug this and find out what is going on`.
> Validity is computed on the 2-D $(L, 2)$ array, so the row indices come from
> either channel and a gap in one channel rewrites both. Every stored segment
> passes through it. The exact spans it rewrites are **not** settled by reading
> the code — the `indices.size == sequence_length` early-out compares a count of
> up to $2L$ against $L$ and essentially never fires — so this document states
> only that a constant-valued fill is applied to both channels of the affected
> rows. Establish the behaviour empirically before relying on the contents of a
> dropout.

### 3.6. Weight and Target

`weight` comes from MIMO verbatim (`mimo_sequence.py:814-815`) and is
**binary**, not fractional. It is driven by the **FHR channel alone**, and it
is a *point sample* rather than an aggregate: the per-sample target is set to
the record's class index only where `curr_input[:, FHR] != 0`
(`mimo_adaptor.py:329-336`), and `split_long` decimates that target at
`ctd_i = np.arange(cur_segment_length, step=16)` — raw indices $0, 16, 32,
\ldots$ So

$$w[j] = 0 \iff \text{the raw FHR sample at index } 16j \text{ was exactly zero}$$

(or index $16j$ lies in equalisation, reflection or block padding). **A UP
dropout never lowers the weight.** Formally:

$$w(t) = \begin{cases}
0 & \text{if the one-hot block target at } t \text{ equals the pad label} \\
1 & \text{otherwise}
\end{cases}$$

Weight 0 marks padding, reflection fill, equalisation fill, and NaN-blanked
input. `target` is then formed at `create_new_pipeline.py:3155-3156`:

$$\texttt{target}(t) = \texttt{class\_id} \times w(t)$$

so $\texttt{target}(t) \in \{0, \texttt{class\_id}\}$, and is 0 exactly where
the weight is 0. Filtering by outcome uses equality against the class id.

The `class_id` here is `pre_defined_target`, the **partition-level** class the
caller passes (1 HEALTHY, 2 ACIDOSIS, 3 HIE) — not the class MIMO parsed from
the `.mat`. MIMO's own parsed class only affects the weight mask and the
all-pad drop of §3.2. The four pre-training shards pass
`pre_defined_target=1` unconditionally (`:3543`).

> **Weight 0 does not mean the stored signal is zero.** MIMO's
> `interpolate_input` (§3.5) runs on the input *after* the targets were built
> from the pre-interpolation FHR, so interior dropouts and equalisation fill
> carry machine-generated constant values at steps whose weight is 0. Only the
> block right-padding beyond the reflection region is exactly zero. Use
> `weight`, never `fhr == 0`, to find invalid steps.

> The `weight` field is typed as float and documented elsewhere as lying in
> $[0, 1]$. In *this* pipeline it only ever takes the values 0 and 1 —
> MIMO's fractional variants (`apply_class_weights_to_sample_weights`,
> `add_sample_weights_forgetting_factor`) are not on the code path used here.

### 3.7. Segment Quality Filtering

Applied identically in screening (`_quality_filter_segments`, `:1865-1904`)
and in the dataset build (inline, `:2960-2994`), with
`WEIGHT_THRESHOLD = 0.90` at `:157` and `FLAT_TOLERANCE = 1e-9` at `:158`:

- **Mean-weight test:** a segment is discarded if
  $\overline{w} < \texttt{WEIGHT\_THRESHOLD} = 0.90$. Since $w$ is binary this
  means at least $\lceil 0.9 \times 330 \rceil = 297$ of 330 steps must be
  labelled. A segment that passes is stored with its zeros intact.
- **Flat-region test** (`find_flat_regions`, `:478-480`, tolerance $10^{-9}$,
  default minimum run 20 samples = 5 s). Discard if any of:
  - longest flat FHR run $> 480$ samples (120 s)
  - longest flat UP run $> 1200$ samples (300 s)
  - total FHR flat time counting only runs $\ge 240$ samples (60 s) exceeds
    1200 samples (300 s)

**What this implies for tail segments.** A short final segment's fill carries
weight 0, so a raw tail of length $L$ passes the mean-weight gate only if
$\lfloor L/16 \rfloor \ge 297$, i.e. $L \ge 4752$ samples $= 1188$ s $= 19.8$
min — and then only if all 297 sampled FHR values are non-zero. Tails shorter
than 19.8 min are always discarded.

### 3.8. Deduplication

`deduplicate_segments` (`:511-549`) groups segments by identical
`domain_start` — the same wall-clock window emitted more than once — and keeps
the member with the highest mean sample weight (`:539-541`); Python's `max`
returns the first maximal element, so ties go to the lowest index. It runs
after sanitisation and **before** quality filtering at both call sites
(`:1962` → `:1964` → `:1976`; `:2935` → `:2948` → `:2960`), so the survivor may
still be rejected afterwards.

Duplicates exist in the first place because MIMO's `merge_sequences` stage is
skipped whenever `overlap_percentage != 0` (§3.2), which it always is here.

### 3.9. GUID Eligibility Screening

**Discovery comes first, and it is folder-driven.** For each of the eight
entries of `FOLDER_TO_SUBGROUP` (`:95-104`) the pipeline lists
`<records_base_path>/<FOLDER>/EFMOut`, skipping the folder with a warning if it
does not exist, and takes every `.mat` file in `sorted(os.listdir(...))` order
(`:2055-2072`). Nothing else is scanned: a GUID with no `.mat` under one of
those eight folders is never seen, and a GUID's **subgroup is decided purely by
which folder its file was found in** — no predicate is evaluated against the
data (§5.2).

A GUID string is the `.mat` filename stem. `_normalize_guid` (`:435-444`)
strips whitespace, uppercases and removes hyphens, but **only** to key the CSV
lookups; the value stored in the shard's `guid` field is the raw stem, with its
original hyphens and case (`:3110`, `:3157`).

**The labour-onset CSV.** `load_csv_metadata` (`:1712-1756`) reads it once and
builds two dicts keyed by the normalised GUID, from the columns `trace_guid`,
`labor_onset_hours` and `second_stage_onset_hours`, multiplying by 3600. Values
are **negative hours relative to delivery** (onset precedes birth). A row whose
value is NaN or an empty string is skipped and counted as missing, so a GUID can
be present in the CSV and still have no TLO.

`prescreen_guid_6h` (`:1907-2027`) then runs MIMO over the last 6 hours plus one
segment buffer:

$$\texttt{MIN\_DOMAIN\_START\_SCREENING} = -(21600 + 1320) = -22920\ \text{s}$$

(`MIN_DOMAIN_START_SCREENING` at `:149`). After sanitisation, dedup and
quality filtering, segments are kept only if

$$-22920 < \texttt{domain\_start} < 0$$

(strict at both ends — a segment starting exactly at $-22920$ is excluded;
the screen adds this strict re-test itself, `:1981-1988`, which the *build*
does not — see §3.10). Valid-signal duration is then *estimated* by treating
the survivors as contiguous (`:1991-1999`):

$$\hat{H} = \frac{(n - 1) \cdot 1200 + 1320}{3600} \ \text{hours}$$

and the GUID is flagged (`:2020-2021`, thresholds
`MIN_VALID_HOURS_UNHEALTHY = 2.0` and `MIN_VALID_HOURS_HEALTHY = 3.0` at
`:152-153`)

$$\texttt{eligible\_2h} = \hat{H} \ge 2.0, \qquad
\texttt{eligible\_3h} = \hat{H} \ge 3.0$$

Two things a reader should not mis-read:

- **The "minimum 6 / 9 segments" figures are derived, not enforced.** No such
  constant exists. They follow from the formula: $n = 6 \Rightarrow \hat H =
  2.033$ h (passes 2 h) and $n = 9 \Rightarrow \hat H = 3.033$ h (passes 3 h),
  while $n = 5$ and $n = 8$ fail.
- $\hat H$ does not check that the surviving segments are actually adjacent,
  so a GUID with gaps is not penalised.

**Post-delivery segments** ($\texttt{domain\_start} \ge 0$) are dropped
unconditionally, at screening (`:1981-1988`) and again at build
(`:2988-2992`). There is **no percentage cap** — the count is written to the
screening CSV but never read by any selection logic.

**Failure is a row, not an abort.** Any exception inside `prescreen_guid_6h`
produces an error row (`_error`, `:1931-1955`): all counts 0,
$\hat H = 0$, both eligibility flags `False`, `error = True`, `error_msg` the
exception string. `select_classification_guids` then excludes every such row
via `no_err = screening_df["error"] == False` (`:2220`).

**Missing TLO is never an eligibility gate.** A GUID with TLO = NaN is
discovered, screened and can be selected: the unhealthy pools take all eligible
GUIDs with no TLO condition, the BG-healthy draw has none, and only the no-BG
healthy draw applies a ratio — which it relaxes when a sub-pool is short
(§6.2). NaN TLO also gets its own stratification bin (§6.3).

**Parallelism and output.** `num_workers` defaults to
$\min(\texttt{os.cpu\_count()}, 8)$ (`:2048-2049`), one `ProcessPoolExecutor`
task per `.mat` file, falling back to a serial loop at $\le 1$. Results are
collected in completion order, so the CSV's row order is **not** the discovery
order. One row per discovered file is written to
`<output_base_path>/guid_screening_results.csv` (`:2001-2025`) with the columns

```
guid, subgroup, record_path, n_total_segments, n_after_dedup,
n_valid_segments_6h, n_low_weight, n_flat_region, n_duplicate,
estimated_valid_hours_6h, has_tlo, tlo_hours, has_second_stage,
second_stage_hours, domain_start_min, domain_start_max, n_post_delivery,
eligible_2h, eligible_3h, error, error_msg
```

where `domain_start_min` / `domain_start_max` are over the *surviving*
in-window starts (NaN when none survive) and `n_post_delivery` counts only
post-delivery segments that had already passed the weight and flat-region
tests. `--screening-csv-path` re-loads this file and skips step 1 entirely.

> No test covers any of this. `hdf5_dataset/tests/` contains no reference to
> `prescreen`, `eligible_2h` or `MIN_DOMAIN_START_*`, so the thresholds, the
> $\hat H$ formula and the strict-vs-inclusive bounds are unprotected against
> regression.

### 3.10. Recording Window — Which Hours Actually Reach the Model

Three filters compose, and they use **different** windows:

| Stage | Bound | Window before delivery |
|---|---|---|
| Eligibility screening | $\texttt{MIN\_DOMAIN\_START\_SCREENING} = -22920$ s | **6.37 h** (6 h + one segment) |
| Dataset extraction | $\texttt{MIN\_DOMAIN\_START\_DATASET} = -44640$ s | **12.4 h** |
| Loader (`epoch_min` / `epoch_max`) | per-config | whatever the config says |

**The asymmetry is deliberate and easy to misread.** A GUID is *judged*
eligible on how much valid signal it has in the last **6.37 h** (§3.9), but
once it qualifies, **every** segment back to **12.4 h** is written. So the
dataset contains up to 12.4 h per GUID while the quality bar was set on the
final 6.4 h.

`MIN_DOMAIN_START_DATASET` is the bare literal $-44640$ at `:150`, commented
`# ~12.4 hours`; the tilde is inaccurate, since $44640/3600 = 12.4$ **exactly**.
The code does not say where the literal came from.

`max_domain_start` is $+\infty$, but post-delivery segments are dropped
unconditionally (§3.9), so the effective extraction range is

$$-44640 \le \texttt{epoch} < 0$$

**inclusive at the low end**, unlike the screening window. The only lower bound
at build time is MIMO's own `domain_start >= min_domain_start`
(`mimo_sequence.py:776-779`); the builder adds no strict re-test, and its
quality loop (`:2957-2994`) checks only weight, flatness and
$\texttt{domain\_start} \ge 0$.

At a 1200 s stride the 44640 s window admits
$\lfloor 44640/1200 \rfloor = 37$ stride positions, and up to **38** segment
starts depending on where the 5280-sample equalisation grid falls (§3.2). Both
the current and the legacy writer use $-44640$.

**The 12.4 h bound applies to the pre-training shards too.** Every partition —
classification and pre-training alike — goes through the same MIMO call at
`:2910` / `:2925`. The two builds differ in how their GUID lists are chosen
(§6.4), not in the window they extract.

**The loader narrows this further.** `CombinedHDF5Dataset` applies
`epoch_min` / `epoch_max` at index-build time
(`hdf5_dataset.py:758-761`), inclusive on both sides:

$$\texttt{epoch\_min} \le \texttt{epoch} \le \texttt{epoch\_max}$$

Since `epoch` is negative, `epoch_min` sets how far **back** to go and
`epoch_max` sets how **close to delivery** to stop. Measured against a
full-coverage GUID (38 segments spanning the whole 12.4 h):

| Filter | Meaning | Segments kept |
|---|---|---:|
| none | full 12.4 h | 38 / 38 |
| `epoch_min: -23000` | last 6.4 h only | 19 / 38 |
| `epoch_max: -23000` | exclude the last 6.4 h | 19 / 38 |
| `epoch_max: -48000` | more than 13.3 h back | **0 / 38** |

> **`epoch_max: -48000` selects nothing.** It appears in
> `config_lag_attn_v{1,2,3}.yaml` (and the synthetic v4 configs). It asks for
> segments earlier than 13.33 h before delivery, but the dataset floor is
> 12.4 h, so `CombinedHDF5Dataset` raises
> `ValueError: No samples match the specified filters.` The intent was
> probably `-4800` (exclude the final 80 minutes) or `epoch_min: -48000`
> (a no-op). Verify before the next training run — this is outside
> `hdf5_dataset/` and has not been changed.

---

## 4. Reproducibility

Seeds are set at import (`:82-88`): `matplotlib.use("Agg")`,
`torch.backends.cudnn.enabled = False`, `random.seed(42)`, `np.random.seed(42)`,
`torch.manual_seed(42)`, `torch.cuda.manual_seed(42)`. `RANDOM_STATE = 42`
(`:162`) is the seed passed to `StratifiedKFold`, `KFold`, `train_test_split`,
`np.random.RandomState` and `random.Random`.

GUID selection uses a local `random.Random(42)`. The pre-training 90/10 split
(§6.4) uses the **module-level** `random` — but it is the only consumer of it:
`random.shuffle(pretrain_bg_cs)` (`:3508`) and `random.shuffle(pretrain_bg_no_cs)`
(`:3513`) are the sole module-level draws in the whole file, every other RNG use
being a local `rng`, `rng_state` or a scikit-learn `random_state`. Since the
step reorder (§6.4) they are also the *first* draws of the run, because the
k-fold build now follows them.

What that split does still depend on is the **incoming order of the pool**,
which differs between the fresh path (the remainder of an `rng.Random(42)`
shuffle, `:2255-2262`) and the resume path (the `sorted` output of
`_discover_mat_files`, `:3441-3444`). Two runs of the same kind reproduce each
other; a fresh run and a resumed run do not.

> One caveat this repository cannot settle: `EarlyMaestraMimoAdaptor` and the
> kymatio bank live outside it. If either draws from the module-level `random`
> during mask computation, the pre-training shuffle becomes order-dependent
> again. Nothing in `create_new_pipeline.py` does.

---

## 5. Outcome Labels and Subgroups

### 5.1. Three-Class Labels

| Class ID | Name | Description |
|---|---|---|
| 1 | HEALTHY | Normal outcome |
| 2 | ACIDOSIS | Metabolic acidosis at birth |
| 3 | HIE | Hypoxic-ischaemic encephalopathy |

Binary framing: HEALTHY $\rightarrow$ 0; ACIDOSIS and HIE $\rightarrow$ 1.

### 5.2. Eight Subgroups

Each GUID falls in one subgroup by outcome, caesarean section (CS), and blood
gas availability (BG) — `SUBGROUP_META` at `:129-138`:

| # | Subgroup | Class | CS | BG |
|---|---|---|---|---|
| 1 | `healthy_no_bg_no_cs` | HEALTHY | No | No |
| 2 | `healthy_no_bg_cs` | HEALTHY | Yes | No |
| 3 | `healthy_bg_cs` | HEALTHY | Yes | Yes |
| 4 | `healthy_bg_no_cs` | HEALTHY | No | Yes |
| 5 | `acidosis_cs` | ACIDOSIS | Yes | Yes |
| 6 | `acidosis_no_cs` | ACIDOSIS | No | Yes |
| 7 | `hie_cs` | HIE | Yes | Yes |
| 8 | `hie_no_cs` | HIE | No | Yes |

**The predicate in that table is descriptive, not evaluated.** A GUID's
subgroup is decided entirely by which `StudyGroup` folder its `.mat` file was
discovered in, through `FOLDER_TO_SUBGROUP` (`:95-104`):

| Folder | Subgroup |
|---|---|
| `HEALTHY_NO_BG_NoCS` | `healthy_no_bg_no_cs` |
| `HEALTHY_NO_BG_CS` | `healthy_no_bg_cs` |
| `HEALTHY_NO_ACIDOSIS_CS` | `healthy_bg_cs` |
| `HEALTHY_NO_ACIDOSIS_NoCS` | `healthy_bg_no_cs` |
| `ACIDOSIS_NO_HIE_CS` | `acidosis_cs` |
| `ACIDOSIS_NO_HIE_NoCS` | `acidosis_no_cs` |
| `HIE_CS` | `hie_cs` |
| `HIE_NoCS` | `hie_no_cs` |

Nothing reconciles a GUID appearing under two folders; it would produce two
screening rows with different subgroups and could enter two pools. Whether that
happens is a property of the data directory.

Blood gas is the objective confirmation of outcome, so the `no_bg` healthy
subgroups are the weakest-labelled part of the cohort and are used mainly as
balancing filler (§6.2).

---

## 6. Cohort Construction

### 6.1. Eligibility Gates

`select_classification_guids` (`:2182-2469`), with `error == False` (§3.9) and:

| Pool | Gate |
|---|---|
| 4 unhealthy subgroups | `eligible_2h` |
| BG healthy (2 subgroups) | `eligible_3h` |
| no-BG healthy (2 subgroups) | `eligible_3h` |

### 6.2. Balancing

Not a proportional downsample of healthy subgroups. The mechanism is:

1. **All** eligible unhealthy GUIDs are taken.
2. BG healthy is split by `HEALTHY_BG_CLS_FRACTION = 0.10` (`:154`) — 10 % to
   classification, **90 % reserved for VAE pre-training** (`:2255-2262`), per
   CS/NoCS independently, after an `rng.shuffle`, with no TLO constraint. The
   share is $n_{\text{cls}} = \max(\operatorname{round}(0.10\,|\text{pool}|), 1)$
   for a non-empty pool — **a floor of one GUID**, so a single-GUID pool is
   consumed entirely by classification and leaves the pre-training pool empty.
3. The residual imbalance is closed with no-BG healthy:
   $$\texttt{deficit} = n_{\text{unhealthy}}^{\text{trainval}} - n_{\text{BG}}^{\text{trainval}}$$
   split between the two no-BG subgroups in proportion to their eligible pool
   sizes ($\texttt{frac\_cs} = |{\rm no\_bg\_cs}| / |{\rm no\_bg\ eligible}|$),
   each capped by its pool, with a two-step compensation pass when a cap bites
   (`:2300-2331`). If $\texttt{deficit} \le 0$ or the no-BG pools are empty,
   both no-BG train/val lists are empty (`:2343-2345`).

Net effect: roughly 1:1 healthy:unhealthy in train/val.

4. **TLO constraint on no-BG:** `HEALTHY_NO_BG_TLO_RATIO = 0.75` (`:156`) —
   75 % of sampled no-BG GUIDs must have a known labour-onset time. The
   sampler is `_sample_with_tlo_constraint` (`:2144-2179`):
   $n_{\text{with}} = \operatorname{round}(0.75\,n)$,
   $n_{\text{without}} = n - n_{\text{with}}$, with two-sided relaxation when
   either sub-pool is short, both sub-pools shuffled by the same seeded `rng`.
   It governs **two** draws, not one: the train/val no-BG draw (`:2336`) and
   the population-proportional test draw of §6.3 (`:2392`).

> `TLO_WITH_RATIO = 0.75` at `:163` is dead code — never referenced. Only
> `HEALTHY_NO_BG_TLO_RATIO` is live.

### 6.3. K-Fold Splitting

`N_FOLDS = 10`, `VAL_RATIO = 1/9`, `RANDOM_STATE = 42`. Folds are named
`fold_1` … `fold_10` (**1-indexed**).

Each subgroup is split **independently** and the folds recombined, so subgroup
proportions are preserved exactly. Within a subgroup, stratification is on the
**labour-duration bin**, not on the subgroup: `_compute_duration_bins`
(`:2470-2512`, with `N_DURATION_BINS = 3` at `:164`) takes quantile tertiles of
$|\texttt{tlo\_hours}|$ via `np.digitize`, giving labels $\{0, 1, 2\}$ for known
durations plus a fourth label $3$ for unknown (NaN) TLO. With no known duration
at all it returns all zeros, degenerating to plain `KFold`.

Boundaries are recomputed **per subgroup for the outer split** — once, at
`:2571`, outside the fold loop — and **again per subgroup and per fold for the
inner train/val split**, which re-bins only that fold's train+val subset
(`:2645-2648`). They are never global.

Fallback ladder, in the order the code tests it (`:2567-2609`): if a subgroup
has fewer than `N_FOLDS` GUIDs, leave-one-out cycling, where folds beyond the
GUID count get an empty held-out set; otherwise `StratifiedKFold(shuffle=True,
random_state=42)` when every non-empty duration bin has $\ge 10$ members, with
a `try`/`except ValueError` fallback to `KFold`; otherwise `KFold` directly.
The leave-one-out branch shuffles with a single `np.random.RandomState(42)`
shared across subgroups.

The inner train/val split is itself stratified when feasible (`:2641-2673`):
`train_test_split(test_size=1/9, shuffle=True, random_state=42, stratify=tv_bins)`
when the smallest non-empty bin has $\ge 2$ members, otherwise unstratified;
and with fewer than two train+val entries, val is empty and everything goes to
train. `random_state` is 42 for every fold and every subgroup.

Two test modes:

| | **holdout** | **augmented** (default) |
|---|---|---|
| Test set | One fixed pool, shared by all folds | Per-fold, from the KFold test split |
| Split per fold | 90 train / 10 val | 80 train / 10 val / 10 test |
| `TEST_HOLDOUT_FRACTION = 0.10` | used | unused |
| Test augmentation | — | extra healthy GUIDs added per fold |

**The mechanism of each mode.** In **holdout** mode a fixed test pool is taken
off the *front* of the already-shuffled lists — $\max(\operatorname{round}(0.10\,|{\rm paths}|), 1)$
per subgroup, for the four unhealthy subgroups and the two BG-healthy
classification lists (`:2270-2288`, the same floor-of-one as §6.2); the
remainder is train/val. In **augmented** mode nothing is held out and every
selected GUID goes to the core pool (`:2289-2294`).

**The population-proportional no-BG test draw runs in both modes** (step F,
`:2348-2394`), not only in augmented. Each healthy subgroup's share $p_{sg}$ is
computed over the whole *eligible* healthy pool; $n_{\rm bg\ test}$ is the
realised BG test count in holdout mode and
$\max(\operatorname{round}(n_{\rm bg\ core}/10), 1)$ in augmented mode; then
$\texttt{total\_healthy\_test} = \operatorname{round}(n_{\rm bg\ test} / p_{\rm bg})$
and each no-BG subgroup contributes
$\min(\operatorname{round}(\texttt{total\_healthy\_test} \cdot p_{sg}), |{\rm remaining}|)$,
drawn under the 0.75 TLO constraint from the pool *minus* every GUID already
used in train/val — so an augmentation GUID can never leak into training. Note
that $p_{sg}$ comes from the eligible population while $n_{\rm bg\ test}$ comes
from the selected 10 %, so the result is proportional in *composition*, not in
size.

> **The augmentation set is drawn once, not per fold.** `test_data` is fixed at
> `:2547` and the same GUIDs are appended to every one of the ten folds' test
> partitions (`:2679-2685`). The ten test sets are therefore **not disjoint** in
> their healthy no-BG part, and a per-fold metric averaged over folds
> double-counts those GUIDs. Nothing in the code says whether that was intended.

### 6.4. Pre-Training Split — and Why It Is Built First

The 90 % of BG-healthy GUIDs not taken for classification become the VAE
pre-training pool, split 90/10 train/test by GUID (truncating `int`), per
CS/NoCS, with no val split and no stratification. Output under
`pre_training_dataset/` (`:3518-3523`, the tuple being
`(filename, records, cs_label, bg_label)`):

| File | `cs_label` | `bg_label` | `pre_defined_target` |
|---|---|---|---|
| `train_dataset_cs.hdf5` | True | True | 1 |
| `train_dataset_no_cs.hdf5` | False | True | 1 |
| `test_dataset_cs.hdf5` | True | True | 1 |
| `test_dataset_no_cs.hdf5` | False | True | 1 |

`bg_label` is True for all four because the pool is drawn from the BG-healthy
subgroups. These four **bypass `_build_hdf5_for_partition` entirely** — the code
says so at `:3526-3528` — and go instead directly through the same
`create_hdf5_for_masks` and `create_hdf5_dataset_from_records_list` the
classification shards use (`:3531`, `:3535-3551`), with `run_guid_analysis=False`.
That is deliberate: threading a variant through the partition path alone would
produce a directory whose classification and pre-training files disagree.

**Step order changed on 2026-09-06 (commit `d97869b`).** The pre-training build
is now **step 4** and the k-fold classification build **step 5**, so
`pre_training_dataset/` is written *before* `k_fold_cross_validation_dataset/`.
The two steps are independent — both consume only the masks, the CSV maps and
the GUID lists that steps 1–3 (or the resumed pickle) already fixed, and
neither reads what the other writes (`:3296-3303`, `:3496-3499`). The
pre-training set goes first because it is the smaller build and the one VAE
pre-training waits on, so an interrupted run still leaves a complete
unsupervised dataset behind.

The order is pinned by
`hdf5_dataset/tests/test_causal_pipeline.py::test_the_pre_training_dataset_is_built_before_the_folds`,
which logs both call sites and asserts all four pre-training files are written
before the first fold partition.

> **Two in-file banners still carry the old numbers.** `# Step 4: HDF5 dataset
> creation from records list` (`:2701`) and `# Step 5: Main orchestrator`
> (`:3209`) are section dividers, not pipeline steps, and now contradict the
> step-4/step-5 banners at `:3494` and `:3556`. Grepping for "Step 4" finds two
> different things.

> **Known inconsistency on the resume path.** With `classification_pickle_path`
> supplied, the pre-training pool is rebuilt at `:3397-3444` as *every* `.mat`
> file under `HEALTHY_NO_ACIDOSIS_CS/EFMOut` and `HEALTHY_NO_ACIDOSIS_NoCS/EFMOut`
> minus those appearing in any fold partition or in the test dict — with **no
> `eligible_3h` test at all**, so screening-ineligible and errored GUIDs are
> included. A resumed run therefore yields a larger and differently-composed
> pre-training set than a fresh run, from the same inputs.

**Legacy pickles are accepted.** A pickle that is a flat dict of folds, with no
`test_mode` key, is wrapped as
`{"test_mode": "augmented", "folds": <the dict>, "test_augmentation": {}}` with
a warning (`:3406-3415`). Note that the mode which then drives the build is
`cv_result["test_mode"]` (`:3559`), which is **not** put through
`validate_test_mode` — only the `test_mode` *argument* is. On a resume the
validated argument and the mode the build follows can differ.

### 6.5. Directory Layout

```
<output_base_path>/
├── guid_screening_results.csv                  ← fresh run only, and only when
│                                                 --screening-csv-path is unset
├── classification_guid_selection_summary.json  ← fresh run only
├── classification_dataset_records.pickle       ← fresh run only
├── pre_training_dataset/                       ← written FIRST (step 4)
│   ├── train_dataset_cs.hdf5      test_dataset_cs.hdf5
│   └── train_dataset_no_cs.hdf5   test_dataset_no_cs.hdf5
└── k_fold_cross_validation_dataset/            ← written SECOND (step 5)
    ├── test/                     ← holdout mode only, shared by all folds
    │   └── <subgroup>.hdf5
    ├── fold_1/
    │   ├── train/<subgroup>.hdf5
    │   ├── val/<subgroup>.hdf5
    │   ├── test/<subgroup>.hdf5  ← augmented mode only
    │   └── fold_eda/             ← fold_1 only, both modes
    └── fold_2/ … fold_10/
```

A partition may contain fewer than 8 files: empty subgroups are skipped.

**The three top-level files are conditional.** The pickle and the summary JSON
are written only on the fresh path (`:3466-3489`); a **resumed run writes no
run-level artefact at all**. The screening CSV is written only when
`screening_csv_path` was also unset (`:3446-3459`). The JSON carries
`test_mode`, the selection statistics, and per-subgroup
`{count, paths_sample: <first 3>}` for the trainval and test pools.

**`fold_eda/` is produced for the first fold only, in both test modes**, by
`fold_eda_analysis.run_fold_eda` (`:3590-3598`, `:3614-3622`), imported lazily
inside a `try`/`except` that only logs on failure. In holdout mode it is passed
the shared `test/` directory; in augmented mode, none. It writes
`fold_eda_report.md` and seven PNGs: `tlo_distribution`, `tlo_presence_ratio`,
`duration_bin_distribution`, `subgroup_duration_histograms`,
`subgroup_composition`, `segment_coverage`, `class_balance`.

**No `*_guid_analysis/` directory is produced, despite the code asking for one.**
`run_guid_analysis` is passed True for exactly one partition of the whole run —
the `train` partition of the first fold, in both modes (`:3587`, `:3611`) — and
the builder does collect a per-GUID `GuidTrackingEntry` (`:417-429`) recording
every segment's fate. But the call site is
`from guid_analysis import run_guid_analysis` (`:3199`), and `guid_analysis.py`
lives in `hdf5_dataset/`, while the module's repo-root guard puts only the
repository root on `sys.path` and the script directory is
`hdf5_dataset/new_pipeline/`. The `ImportError` is swallowed by the surrounding
`except Exception` and logged as "GUID analysis failed", so a run silently
produces nothing. (`from fold_eda_analysis import run_fold_eda` has the same
shape but *does* resolve, because that module is in `new_pipeline/`.) This is a
code defect, not a documentation gap; until it is fixed, treat the analysis
directory as unavailable.

---

## 7. Signal Processing

The **two-sided** features come from `KymatioPhaseScattering1D` with
$J = 11$, $Q = 4$, $T = 16$, `max_order=1`, `shape=5280`, `tukey_alpha=None`,
reflection padding.

The **causal** features do not. The causal writer deliberately never
instantiates that class; it runs `CausalTorchBank` over a one-sided gammatone
bank instead (`create_new_pipeline.py:2831-2838`, whose comment says so). A
temporary `KymatioPhaseScattering1D` *is* built on both paths inside
`compute_scattering_masks` (`:1376-1383`), but only to enumerate the 903 wavelet
pairs and size the scattering block — never to transform data on the causal
arm.

### 7.1. The Wavelet Filter Bank

The bank contains **42 first-order analytic Morlet wavelets**, emitted in
**descending** centre frequency. Kymatio stores $\xi$ in cycles per sample;
Hz follows as $\xi f_s$ with $f_s = 4$.

Over most of the bank the spacing is constant-$Q$ geometric:

$$\frac{\xi_i}{\xi_{i+1}} = 2^{1/Q} = 2^{1/4} \approx 1.1892$$

The ladder holds for filters **0 through 38**: measured,
$\xi_k/\xi_{k+1} = 1.1892$ for $k = 0 \ldots 37$, and $\sigma_{38}$ is exactly
$\sigma_{37}/2^{1/4}$. Kymatio's `sigma_min` floor,
$\sigma = 4.8828125 \times 10^{-5}$ cycles/sample, first binds at **filter 39**,
so exactly **three** filters (39, 40, 41) are off the ladder, with ratios
1.333, 1.5 and 2.0. Equivalently: $\xi/\sigma = 9.633$ for indices 0–38, then
7.910, 5.273, 2.637.

| Filter | $\xi$ | Hz | Period |
|---:|---:|---:|---:|
| 0 | 0.372885 | 1.4915 | 0.7 s |
| 1 | 0.313558 | 1.2542 | 0.8 s |
| 2 | 0.263669 | 1.0547 | 0.9 s |
| … | | | |
| 21 | 0.009799 | 0.0392 | 25.5 s |
| 24 | 0.005826 | 0.0233 | 42.9 s |
| 27 | 0.003464 | 0.0139 | 72.2 s |
| 30 | 0.002060 | 0.0082 | 121.4 s |
| 33 | 0.001225 | 0.0049 | 204.1 s |
| 38 | 0.000515 | 0.0021 | 485.5 s |
| 41 | 0.000129 | 0.0005 | 1941.8 s |

Clinical band membership:

| Band | Frequency | Filters | Count |
|---|---|---|---:|
| Beat-to-beat variability | 0.25 – 1.5 Hz | 0 – 10 | 11 |
| LF / MF variability | 0.04 – 0.25 Hz | 11 – 20 | 10 |
| Decelerations | 0.008 – 0.04 Hz | 21 – 30 | 10 |
| Baseline trend | < 0.008 Hz | 31 – 41 | 11 |

### 7.2. The Scattering Transform

First-order scattering coefficients are

$$S_1 x(t, i) = \big| x * \psi_i \big| * \phi \,(t)$$

together with the zeroth-order term

$$S_0 x(t) = x * \phi \,(t)$$

The modulus $|\cdot|$ extracts the envelope, discarding phase; the low-pass
$\phi$ makes the representation locally translation-invariant and permits
decimation by $T = 16$ (one step = 4 s). $T$ is the **decimation and invariance
scale**, not $\phi$'s width: $\phi$ is kymatio's Gaussian at
$\sigma_{\text{low}} = 6.249875 \times 10^{-3}$ cycles/sample (against the
nominal $\sigma_0/T = 0.1/16 = 6.25 \times 10^{-3}$, agreeing to $2\times10^{-5}$
relative), whose time width $1/(2\pi\sigma_{\text{low}})$ is 25.47 samples
$= 6.37$ s. The output is therefore

$$\underbrace{1}_{S_0} + \underbrace{42}_{S_1} = \mathbf{43}\ \text{channels}
\ \times\ 330\ \text{time steps}$$

This width is **fixed by the filter bank**, not by any selection — the whole
scattering block is stored unmasked, for both FHR (`fhr_st`) and UP (`up_st`)
— **on the two-sided variant only**. The causal variant drops the seven
channels whose warm-up exceeds the 330-step segment and stores 36 (§8.1).

**Padding geometry, two-sided.** `min_to_pad` $= 243$; $J_{\rm pad} =
\lceil \log_2(5280 + 486) \rceil = 13$; $N_{\rm padded} = 8192$; and
$8192 - 5280 = 2912$ split as $\texttt{pad\_left} = \texttt{pad\_right} = 1456$
raw samples $= 364$ s each side (`causal_scattering.py:318-340`, reproducing
`kymatio_phase_scattering.py:100-113`). The reflection is applied iteratively in
chunks because torch's reflect pad refuses widths above $n - 1$.

**Padding geometry, causal.** Fundamentally different: the causal chain
prepends $n_{\rm taps} - 1 = 32767$ samples ($8191.75$ s) of assumed history on
the **left only**, in mode `edge` (replicate $x[0]$) or `zero`
(`causal_scattering.py:947-1006`). Reflection is **refused by name**, because it
mirrors the signal forward in time and would reintroduce exactly the future
dependence the causal kernels exist to remove. A zero-mean $\psi$ annihilates a
constant exactly, so `edge` costs nothing in the passband.

**Decimation differs too.** Kymatio decimates each band by $2^{k_1}$,
$k_1 = \min(j_1, \log_2 T)$, *before* taking the modulus, then smooths with a
$\phi$ pre-folded to that rate. Sampling commutes with the modulus, but the
modulus of an already-sampled band creates content above the reduced Nyquist,
so the convention aliases: up to $2.6\times10^{-2}$ relative on the $k_1 = 4$
bands, while $S_0$ and the $k_1 \le 2$ bands agree with full-rate to about
$3\times10^{-9}$ (measured and recorded at `causal_scattering.py:1286-1296`).
The causal cascade instead stays at full rate throughout and subsamples once at
the end by plain `[..., ::16]`.

**The causal bank itself.** A complex gammatone of order
$n = \texttt{GAMMATONE\_ORDER} = 4$, 32768 taps:

$$\psi_k[\tau] = a_k[\tau]\left(e^{\mathrm{i}2\pi\xi_k\tau} - \kappa_k\right),
\qquad a_k[\tau] = \tau^{\,n-1}e^{-2\pi b_k\tau}\,\mathbb{1}_{\tau>0}$$

with $\xi_k$ taken from the production Morlet bank, $\kappa_k$ the zero-mean
correction, and kernels $L^1$-normalised to 1. The rate is matched to the Morlet
at **half power** on both sides:

$$b_k = \sigma_k\,\frac{\sqrt{\ln 2}}{\sqrt{2^{1/n}-1}} = 1.9140091\,\sigma_k
\quad (n = 4)$$

with measured $-3$ dB widths in $[0.994, 1.056]$ of the Gaussian target. The
causal low-pass is the same envelope at
$b_\phi = 1.9140091\,\sigma_{\text{low}} = 1.19623\times10^{-2}$, normalised to
sum 1 so its DC gain matches the production Gaussian's; its group delay is
$\tau_g^\phi = n/(2\pi b_\phi) = 53.22$ samples $= 13.30$ s — the 13.3 s that
appears as the $S_0$ delay in §8.1.

Two properties follow that a consumer should know:

- **A causal filter cannot be exactly analytic** (Paley–Wiener), so the bank
  carries a measured analyticity defect
  $\epsilon_{\rm anl} = |\hat\psi(-\xi)|/\|\hat\psi\|_\infty$: median
  $9.48\times10^{-5}$, max $3.07\times10^{-2}$.
- **The reported staleness is the phase group delay** $\tau_g = n/(2\pi b)$, the
  envelope's *mean*. The envelope's mode $(n-1)/(2\pi b)$ and its energy
  centroid $(2n-1)/(4\pi b)$ are different numbers, 25 % apart at $n = 4$. Over
  the bank $\tau_g$ ranges 2.15 s to 1702.97 s.

**Warm-up and delay compose differently.** Along the cascade both **add**
($W(S_0) = W_\phi$, $W(S_1^k) = W_k + W_\phi$); across a phase pair the warm-up
takes the **maximum** ($W(\Phi_{ij}) = \max(W_i, W_j) + W_\phi$), then is
ceiling-rounded to decimated steps. The warm-up itself is
$W_{0.95}(h)$, the leading tap count enclosing 95 % of the kernel's energy
(`CAUSAL_WARMUP_QUANTILE = 0.95`): measured $W_\phi = 80$ samples $= 20.0$ s,
with wavelet supports from 14 samples (3.5 s) to 10085 samples (2521.25 s), and
$W_{0.95}/\tau_g = 1.484$ median.

> **95 % is an initialisation policy, not independence from the prehistory.**
> `hdf5_dataset/tests/test_phase_operator.py:452,491` pins the consequence: for
> a constant input the output error at the threshold sample is exactly the
> kernel's remaining $L^1$ tail — 15.3 % of the input on the low-pass — and the
> composed slow-channel envelope keeps 15.2 % of its mass beyond its rounded
> warm-up.

Scattering is stable to deformation and invariant to translation, which is what
makes it a good front end. The cost is that the modulus destroys **relative
phase between scales** — which is exactly what the phase block restores.

### 7.3. The Phase-Harmonic Operator

Two **operator versions** ship, and they are a variant axis of their own,
recorded on every causal file as the root attribute `causal_phase_operator`.
Everything in this subsection up to "Two operator versions" below describes the
construction both share.

For a complex signal $z$, the phase-harmonic operator of order $p$ is

$$[z]^p = |z| \, e^{\mathrm{i} p \arg z}$$

It preserves the modulus and multiplies the instantaneous phase by $p$. The
phase-harmonic correlation of a wavelet pair $(\psi_i, \psi_j)$ with
$\xi_i \le \xi_j$ is

$$C_{i,j,p}(t) = \phi * \Big( [z_i]^p \cdot \overline{z_j} \Big)(t),
\qquad z_i = x * \psi_i, \qquad p = \frac{\xi_j}{\xi_i}$$

and only the real part is stored.

**Why $p = \xi_j / \xi_i$.** Advancing the phase at the slow scale $i$ by the
frequency ratio makes it run at the same rate as the fast scale $j$. The
product with $\overline{z_j}$ then has a *stationary* phase — and therefore
survives the $\phi$ average — precisely when the two scales are **phase-locked
at ratio $p$**. If they are independent, the phase of the product rotates and
$\phi$ averages it toward zero.

Read by harmonic step $k$, where $p = 2^{k/Q} = 2^{k/4}$:

| $k$ | $p$ | What a large value means |
|---:|---:|---|
| 0 | 1.00 | Same scale — reduces to local energy, $\phi * \lvert z_i\rvert^2$ |
| 4 | 2.00 | One octave — the waveform at scale $i$ is **non-sinusoidal**; a sharp or asymmetric wave carries harmonic content at $2f$ locked to $f$ |
| 6 | 2.83 | 1.5 octaves — intermediate cross-scale locking |
| 8 | 4.00 | Two octaves — coupling between **well-separated rhythms**, one band modulating another |

So octave coefficients measure **waveform shape** (departure from a sine) and
wide-span coefficients measure **cross-band coupling**. Neither survives the
scattering modulus.

**Pair enumeration** (`kymatio_phase_scattering.py:134-160`). All ordered pairs
with $\xi_j \ge \xi_i$ are enumerated, $i$ outer and $j$ inner:

$$N_{\text{pairs}} = \frac{n(n+1)}{2} = \frac{42 \times 43}{2} = \mathbf{903}$$

exposed as `i_idx`, `j_idx`, `powers`, and `autoc_idx` (the 42 diagonal
positions where $i = j$). **`i` indexes the lower-frequency wavelet and `j` the
higher**, so `powers` $\ge 1$ always. A phase block is produced at full
903-pair width and then reduced by a boolean mask.

#### Two operator versions: `ratio_power_v0` and `integer_harmonic_v1`

Everything above resolves the exponent as the raw frequency ratio. That is one
of two shipped versions, named by the constants `PHASE_OPERATOR_LEGACY` and
`PHASE_OPERATOR_INTEGER` and enumerated in
`PHASE_OPERATORS = ('ratio_power_v0', 'integer_harmonic_v1')`
(`causal_scattering.py:184-186`). `validate_phase_operator` (`:193-211`)
refuses anything else, naming both.

**Why a second version exists.** For non-integer $p$, $[y]^p$ is
**discontinuous across the principal-angle branch**. Two inputs straddling the
negative real axis by $\epsilon$ produce images

$$\big|\,[y]^p_{+} - [y]^p_{-}\big| \;=\; 2\,|y|\,\big|\sin \pi p\big|$$

apart, however small $\epsilon$ is. At the legacy $k = 6$ family,
$p = 2^{3/2} = 2.8284271$, that is $2|\sin(\pi\,2^{3/2})| = 1.0266$ times
$|y|$ — a jump of order the signal itself. Polar construction does not help: it
avoids exponentiating the modulus and does nothing whatever for the angle. For
**integer** $p$ the jump is exactly zero (measured $< 10^{-6}$), because

$$|y|\,e^{\mathrm{i}k\operatorname{Arg}y} \;=\; |y|^{1-k}\,y^{\,k}$$

is continuous — and this is the property the phase-harmonic construction
(Zhang & Mallat) actually establishes.
`hdf5_dataset/tests/test_phase_operator.py:83-141` pins the counterexample
permanently, at branch separation $10^{-8}$.

**What each version does.** The single shared resolver is
`resolve_phase_power(pairs, xi, op)` (`causal_scattering.py:270-300`), used by
the numpy chain and its torch twin alike:

| | `ratio_power_v0` | `integer_harmonic_v1` |
|---|---|---|
| Exponent | the raw float64 ratio $p_{ij} = \xi_j/\xi_i$ | $k_{ij} = \operatorname{round}(\xi_j/\xi_i)$, an exact integer |
| Admissible $k$-steps | $\texttt{PHASE\_K\_STEPS} = (4, 6, 8)$ | $\texttt{PHASE\_K\_STEPS\_INTEGER} = (4, 8)$ |
| Integer harmonics reached | — | $k_{ij} \in \{2, 4\}$ at the production bands |
| `fhr_ph` / `up_ph` channels | **66 / 15** | **44 / 10** |
| $c_y$ / $c_u$ (causal) | 102 / 51 | 80 / 46 |
| Branch continuity | no | yes |
| Root attribute | `'ratio_power_v0'`, or absent | `'integer_harmonic_v1'` |
| Per-block attrs | `sel_phase_operator` | `sel_phase_operator`, `sel_harmonic` |

`harmonic_index` (`:231-266`) computes $k_{ij}$ as int64 and **raises** unless
every pair satisfies $|\xi_j/\xi_i - k_{ij}| < \tau\,k_{ij}$ with $k_{ij} \ge 2$,
at $\tau = \texttt{PHASE\_REL\_TOL} = 0.05$; its refusal ends "refuses a
fractional exponent rather than silently reintroducing the branch
discontinuity."

**The version selects the pairs, not just the exponent.**
`phase_k_steps_for(op)` (`:214-228`) validates the operator and returns its step
tuple, so the version and the admissible steps are one decision rather than two;
`_build_phase_selection` then refuses a `k_steps` tuple that is not that
operator's own (`create_new_pipeline.py:1255-1261`).

**So the substantive change is the channel axis, not the values.** At the
production bands the integer selection is *exactly* the legacy selection with
the $k = 6$ family removed — every integer pair is a legacy pair:

| Field | $k=4$ | $k=6$ | $k=8$ | `ratio_power_v0` | `integer_harmonic_v1` |
|---|---:|---:|---:|---:|---:|
| `fhr_ph` | 24 | 22 | 20 | 66 | **44** (24 + 20) |
| `up_ph` | 7 | 5 | 3 | 15 | **10** (7 + 3) |

and on the pairs both keep, the two exponents differ only by float64 round-off:
the measured maximum of $|\xi_j/\xi_i - k_{ij}|/k_{ij}$ over the 44 integer
`fhr_ph` pairs is $6.66\times10^{-16}$. What the integer exponent buys on the
surviving channels is the *guarantee* of branch continuity, not a different
number.

**It is a causal-arm version only.** `transform_sample`,
`compute_scattering_masks` and `create_initial_hdf5` all refuse a non-legacy
operator on the two-sided variant, on the ground that production's kymatio
operator *is* `ratio_power_v0` by definition and a two-sided integer block would
match no shard on disk (`create_new_pipeline.py:1367-1373`, `:773-778`). There
is no 44/10 two-sided geometry.

**Backward compatibility.** `resolve_phase_operator(attrs)`
(`hdf5_dataset.py:100-124`) returns
`attrs.get('causal_phase_operator', 'ratio_power_v0')`: **absence is the legacy
operator**, because every causal shard written before the version existed was
built with it. A file list mixing operators is refused
("Mixed causal phase operators in one dataset"), as is a statistics file built
under the other one ("Statistics/dataset phase-operator mismatch") — and the
latter is checked *before* the width check, so the message names the operator
rather than a width.

**An integer build is a separate dataset, not a replacement.** It needs its own
`--output-base-path` and its own statistics file, because its channel axis is a
different one (§11.3).

> The legacy branch is retained **only** so shards, statistics and checkpoints
> already on disk keep their original meaning (`causal_scattering.py:174-183`).
> A future integer $k = 3$ or $(m, n)$ family would be defined as a *separate
> version*, never as a relaxation of `integer_harmonic_v1` — `harmonic_index`'s
> $k \ge 2$ floor is explicitly not negotiable.

#### The causal arm multiplies the two legs at different physical instants

Everything above is written for the two-sided bank, where both legs are zero-delay and the product
compares band $i$ with band $j$ **at one instant**. On the one-sided bank it does not: each leg
carries its own group delay $\tau_g = \gamma/(2\pi b_k)$, so multiplying them at the same *stored*
index compares band $i$ at $t - \tau_i$ with band $j$ at $t - \tau_j$. The gap is

$$\Delta_{ij} \;=\; \tau_i - \tau_j \;=\; \tau_i\Big(1 - \frac{1}{p_{ij}}\Big),$$

which grows with the harmonic ratio the block exists to sweep. The range is
$3.61$–$291.64$ s on `fhr_ph` and $68.74$–$291.64$ s on `up_ph`; the *medians*
depend on the operator version, because the $k = 6$ family the integer operator
drops sits in the middle of the distribution:

| Field | range | median, `ratio_power_v0` | median, `integer_harmonic_v1` |
|---|---|---:|---:|
| `fhr_ph` | 3.61 – 291.64 s | 39.12 s (66 pairs) | 38.66 s (44 pairs) |
| `up_ph` | 68.74 – 291.64 s | 163.49 s (15 pairs) | 150.49 s (10 pairs) |

Every `up_ph` channel is skewed by at least $68.7$ s against a lag search of
$360$ s, under either operator.

$\Delta_{ij}$ comes from `pair_leg_skew` (`causal_scattering.py:1113-1126`), the
difference of the bank's stored per-filter group delays; the low-pass delay
$\tau_\phi$ is common to both legs and cancels. A negative skew is refused by
name — "column 0 must index the lower frequency, so the faster leg would have to
be advanced rather than delayed, reading its own future."

`causal_scattering.py` therefore offers the causal phase block two **leg
alignments**, selected by `leg_alignment` and recorded as the root attribute
`causal_leg_alignment`. This is a **different axis from the phase operator
above**; the two are orthogonal (`leg_alignment_shift` reads only the bank's
group delays and $\xi$, never the exponent; `resolve_phase_power` never reads
the shift) and compose as two independent attributes on one file:

| mode | what it does |
|---|---|
| `'none'` | Multiplies the legs at one stored index. The API default at every level, and what a shard written before the mode existed holds. |
| `'envelope'` | Delays the faster leg by $s_{ij} = \operatorname{round}(\Delta_{ij} f_s)$ raw samples and multiplies it by $e^{\,i2\pi\xi_j s_{ij}}$, putting both legs on the slow leg's clock. |

Two mechanical details that are easy to get wrong:

- **The shift is per *pair*, not per filter, and must be applied after the
  per-pair gather.** One fast filter serves up to three slow partners at three
  harmonic ratios and needs a different $s_{ij}$ in each: on the legacy bank 22
  of the 24 distinct `fhr_ph` fast legs are reused, 20 of them by three partners
  each. Shifting the response array *before* the gather would satisfy at most one
  pair per reused filter and be silently wrong for the rest — with every shape
  still correct.
- **The delayed leg's leading samples are edge-replicated**, not zero-filled:
  the source index is clipped at zero, so every tap the shift pushes before the
  start reads the response's first sample — the same assumed history the causal
  convolution ahead of it already ran on.

The shift and phasor are computed once in the shared numpy bank in float64 and
passed into the torch chain rather than recomputed there: the phasor's angle
reaches 9.7 turns, and a single-precision evaluation would lose four digits a
float64 evaluation rounded once does not.

The phasor is not optional. A gammatone's *phase* delay at its own centre frequency is exactly
zero (measured $\le 0.3^\circ$ on the shipped bank), so a plain shift moves the carrier as well as
the envelope and injects a spurious rotation. The largest rotation over the stored pairs is
$9.695$ turns on `fhr_ph` and $9.616$ on `up_ph` (identical under both operator versions); the
$9.6$ turns quoted in the source is the worked example at $\xi_j = 0.033$ Hz,
$\Delta_{ij} = 291.6$ s, not the maximum. Measured, the shift without the phasor scores **worse**
than no alignment at all: the median `fhr_ph` correlation against the centred block at the
predicted delay moves from $+0.049$ to $-0.432$. Measured against the centred block at the
predicted delay, over twelve segments: `fhr_ph` $+0.07 \to +0.80$, `up_ph` $+0.09 \to +0.73$.

> Those correlation figures, and the ones in the next paragraph, are quoted from
> the docstrings that recorded them (`causal_scattering.py:1141-1150` and the
> comparison tool). They derive from `output/causal_scattering/per_channel.csv`,
> which is git-ignored and not present in a fresh checkout —
> `hdf5_dataset/tests/test_preprint_numbers.py` *skips* without it. Regenerate
> the measurement (§11.3 step 3) before relying on a digit.

The alignment costs nothing. $W_j + s_{ij} \le W_i$ holds for all $81$ stored pairs of the legacy
selection (tightest slack $8$ raw samples), so the composed warm-up, the stored widths, the channel
identities and the drop rule are all unchanged — and the composed delay
$\max(\tau_i,\tau_j) + \tau_\phi$ that the shard already records stops being a misprediction and
becomes correct. The integer selection is a strict subset of those pairs, so the property holds a
fortiori on its 54; derived, the slack minima are again 8 raw samples on `fhr_ph` and 132 on
`up_ph`. Only the legacy case is pinned by a test.

The integer sample shift itself ranges 14–1167 raw samples on `fhr_ph` and 275–1167 on `up_ph`,
identical under both operator versions. Keep the two quantities distinct: the maximum *skew* is
$\Delta_{ij} = 291.6418$ s, and the *shift* it rounds to is
$s_{ij} = \operatorname{round}(291.6418 \times 4) = 1167$ samples $= 291.75$ s.

What the alignment does not fix is the $p = 2^{6/4}$ family: $[y]^p$ uses the principal argument, so
a causal leg and a centred one differ by a per-channel constant rotation there. That rotation is
**measured and recorded** in the comparison tool's `coherence_deg_envelope` column and applied
nowhere. **Under `integer_harmonic_v1` that family does not exist at all** — the $k = 6$ pairs are
precisely the ones the integer operator drops — so on an integer build this caveat is empty, and
the branch discontinuity behind it is gone rather than tolerated.

Because an aligned file has exactly the widths, warm-ups and delays of an unaligned one, the mode
is recorded as the root attribute `causal_leg_alignment` (§8.1) — nothing else on the file reveals
it, and the loader refuses a file list or a statistics pairing that mixes the two.

**The alignment applies to `fhr_ph` and `up_ph`, and to both of them.** It is a property of the
transform a shard was built with, not of one block. It does not touch `fhr_st` / `up_st` (real
scattering, no pairs), and it cannot touch `fhr_up_ph` — the causal variant does not produce that
block at all.

> **What is verifiable about existing shards.** `'none'` is the API default at
> every level and the value a file lacking the attribute resolves to. The two
> committed tiny fixtures (`teb_vae/lag_attn/tests/fixtures/tiny_shard_causal.hdf5`
> and `tiny_shard_causal_int.hdf5`) both carry `'envelope'`, and the checked-in
> `RUN_ARGS` selects `'envelope'`. Whether any *production* shard is aligned is
> not determinable from the repository; the refusal text at
> `create_new_pipeline.py:315-320` still describes `'none'` as "every shard on
> disk was built", which was true when it was written.

### 7.4. Cross-Channel Phase Correlation

For two-channel input the same construction runs across channels. The
**intent**, which the selector implements and the code comments state, is

$$C^{\text{cross}}_{i,j,p}(t) = \phi * \Big( \big[\, \mathrm{UP} * \psi_i \,\big]^p \cdot \overline{\big(\mathrm{FHR} * \psi_j\big)} \Big)(t)$$

with $i$ indexing the **UP** filter and $j$ the **FHR** filter, so the block
would encode UP $\rightarrow$ FHR directionality. That makes the $i$/$j$
semantics of `fhr_up_ph` different from the self-phase blocks: not
low-vs-high frequency but source-vs-target.

> **Defect: the executed code binds the channels the other way round, and the
> band masks assume the intent.** The builder stacks the batch **FHR-first** —
> `np.stack([valid_fhr, valid_up], axis=1)` (`create_new_pipeline.py:3029-3031`)
> — and calls the cross pass with `phase_channels=[0, 1]`, which
> `_setup_phase_channels` slices in that order. The transform then takes
> `signal_i = filtered_signals[:, 0, i_idx, :]` and
> `signal_j = filtered_signals[:, 1, j_idx, :]` and accelerates `signal_i`
> (`kymatio_phase_scattering.py:334-339`). So the **accelerated leg is FHR** and
> the **conjugated leg is UP** — the reverse of the formula above.
>
> Meanwhile the selector applies the UP band mask to `i_idx` and the FHR band
> mask to `j_idx` (`kymatio_phase_scattering.py:696-714`), i.e. it assumes
> $i = \mathrm{UP}$. The two disagree. Because kymatio's enumeration guarantees
> $\xi_i \le \xi_j$, the practical consequence is worst in Band B (§7.6): the
> slow "contraction band" leg is bound to **FHR** and the fast 0.04–0.25 Hz leg
> to **UP** — a band in which §7.5 argues the tocodynamometer carries maternal
> movement and respiration rather than uterine activity.
>
> The pipeline's own comment (`create_new_pipeline.py:1392-1395`, "UP filter vs
> FHR filter") repeats the intent, so code comment, selector and this document
> agreed with each other and disagreed with the executed code.
>
> **It is inert for current work.** `fhr_up_ph` is produced on the two-sided
> arm only, and the lag-attn models deliberately do not consume it (§8). The
> only consumer is the legacy `seq_vae_teb` stack under `mimo/`. Treat any
> existing `fhr_up_ph` block as FHR-accelerated / UP-conjugated with the band
> roles swapped, and settle the binding before the block is used again.

### 7.5. Self-Phase Channel Selection

Both self-phase blocks are selected by one rule: a band in **true Hz**, crossed
with a set of harmonic steps on the $2^{k/Q}$ power grid. A pair is kept when

$$\xi_i f_s \ge f_{\min} \quad\wedge\quad \xi_j f_s \le f_{\max}
\quad\wedge\quad \exists\, k : \big| p - 2^{k/Q} \big| < \tau \cdot 2^{k/Q}$$

with **relative** tolerance $\tau = 0.05$. (Both endpoints are required in
band; since $\xi_i \le \xi_j$ this is equivalent to the two one-sided tests
above.)

**The $k$-steps are set by the phase-operator version** (§7.3), not by a
free constant: `compute_scattering_masks` takes them from
`phase_k_steps_for(phase_operator)` (`create_new_pipeline.py:1375`).

| Field | Band | $k$-steps, `ratio_power_v0` | $k$-steps, `integer_harmonic_v1` | Channels |
|---|---|---|---|---|
| `fhr_ph` | 0.008 – 1.00 Hz | $\{4, 6, 8\}$ → $p =$ 2.00, 2.83, 4.00 | $\{4, 8\}$ → $k_{ij} =$ 2, 4 | **66** / **44** |
| `up_ph` | 0.008 – 0.05 Hz | $\{4, 6, 8\}$ | $\{4, 8\}$ | **15** / **10** |

Measured breakdown by harmonic step (the $k = 6$ column is exactly what the
integer operator drops):

| Field | $k=4$ | $k=6$ | $k=8$ | `ratio_power_v0` | `integer_harmonic_v1` |
|---|---:|---:|---:|---:|---:|
| `fhr_ph` | 24 | 22 | 20 | 66 | 44 |
| `up_ph` | 7 | 5 | 3 | 15 | 10 |

`fhr_ph` channels by the band of their anchor $\xi_i$, which is likewise
operator-dependent:

| Anchor band | `ratio_power_v0` | `integer_harmonic_v1` |
|---|---:|---:|
| Deceleration, 0.008–0.04 Hz | 30 | 20 |
| LF/MF variability, 0.04–0.25 Hz | 30 | 20 |
| Beat-to-beat, 0.25–1.0 Hz | 6 | 4 |

`up_ph` lies entirely in the deceleration band under both (15, resp. 10).

**An empty selection is refused at selection time**, not at file-creation time,
naming the field, the band and the octave span the widest step needs
(`:1263-1274`) — because a zero-width HDF5 dataset otherwise fails inside h5py
with "All chunk dimensions must be positive", naming neither.

**Why each bound:**

- **Lower edge 0.008 Hz — the analysis-window floor.** A Gabor wavelet at $\xi$
  has Gaussian time envelope of standard deviation $\sigma_t = 1/(2\pi\sigma)$.
  Below roughly 0.008 Hz its $\pm 3\sigma_t$ support exceeds the 1200 s trimmed
  segment, so the convolution is dominated by reflection padding rather than
  signal. Filter 30 (0.0082 Hz, $\pm 2\sigma$ span 744 s) is the last that
  comfortably fits; filter 33 (0.0049 Hz, span 1252 s) does not.
- **FHR upper edge 1.0 Hz — the beat-series Nyquist limit.** FHR is a rate
  derived from beat detection at 110–160 bpm (1.83–2.67 beats/s), giving a
  Nyquist limit of 0.92–1.33 Hz. Filters 0–2 (1.49, 1.25, 1.05 Hz) sit at or
  above it and describe interpolation, not physiology.
- **UP upper edge 0.05 Hz — the contraction band.** A contraction is a 45–90 s
  pulse recurring every 2–3 min; essentially all uterine energy — fundamental,
  duration, rise and fall — lies below 0.05 Hz. Above it the tocodynamometer
  carries maternal movement, respiration, transducer noise and baseline drift.
  This matches the cap the cross-channel selector already used.
- **$k = 4$** is the clinically load-bearing step. A deceleration is strongly
  asymmetric, and the asymmetry is the discriminator: variable decelerations
  have an abrupt onset, late decelerations a gradual and delayed one. Octave
  phase-locking measures exactly that.
- **$k = 8$** captures sympathovagal balance — baroreflex activity near
  0.04–0.15 Hz against higher-frequency vagal activity near 0.2–0.5 Hz, roughly
  two octaves apart, a relationship that shifts under hypoxia.
- **$k = 0$ (the diagonal) is excluded.** For $i = j$, $p = 1$ the operator is
  the identity, so
  $$C_{i,i,1} = \phi * \big(z_i \overline{z_i}\big) = \phi * |z_i|^2$$
  while the scattering channel stored alongside it is $\phi * |z_i|$. If
  $|z_i|$ is roughly constant across the 4 s window then
  $\phi * |z_i|^2 \approx (\phi * |z_i|)^2$, and after the pipeline's
  asinh/log transforms plus standardisation the two are near-collinear —
  median $|r| = 0.967$ measured on a synthetic FHR-like signal. The unique
  content is the within-window envelope variance, a genuine but small
  second-order quantity.

**Relative, not absolute, tolerance.** The power grid is geometric, so a fixed
absolute window is far too permissive at large $p$ and too strict at small $p$.
This matters historically: the legacy selector used an absolute tolerance of
0.1 against literal ratios $[2, 3]$, and since $\log_2 3 = 1.585$ needs
$k = 6.34$ — off-grid, nearest reachable 2.828 and 3.364 — it matched
**exactly zero** harmonic-3 coefficients despite several documents claiming
otherwise.

**Configuration.** `FHR_PHASE_BAND_HZ`, `UP_PHASE_BAND_HZ` and
`PHASE_POWER_REL_TOL` at the top of `create_new_pipeline.py` are live; the
$k$-steps come from `causal_scattering.PHASE_K_STEPS` (`:163`) and
`PHASE_K_STEPS_INTEGER` (`:190`) through the operator version.

> **`PHASE_HARMONIC_K_STEPS` (`create_new_pipeline.py:213`) is dead code.**
> Nothing on any execution path reads it — `compute_scattering_masks` resolves
> its steps from the operator at `:1375`, and the only remaining mentions are a
> stale comment at `:1202`, `smoke_check_channel_selection.py` and prose in
> `check_phase_diagonal_redundancy.py`. **Setting it to `(0, 4, 6, 8)` changes
> nothing**, and the keep-the-diagonal variant it used to switch on has no
> supported mechanism today: `_build_phase_selection` would refuse a step tuple
> that is not the operator's own. Restoring the diagonal now means defining a
> third operator version — and $k = 0$ gives $p = 1$, which
> `integer_harmonic_v1` would admit only if `harmonic_index`'s $k \ge 2$ floor
> were relaxed, which its docstring explicitly forbids. The old switch's
> `fhr_ph` = 94 / `up_ph` = 26 figures still check out arithmetically (66 + 28
> and 15 + 11 in-band diagonals); there is simply no supported way to build
> them.

`check_phase_diagonal_redundancy.py` re-measures the redundancy on a real shard.
It computes the Pearson $r$, per first-order filter, between $\operatorname{asinh}$
of the diagonal phase channel $C_{i,i,1}$ (taken from the full 903-pair axis via
`autoc_idx`) and $\log(\max(S_1^{(i)},0) + 10^{-6})$ of the scattering channel
stored beside it, pooled over segments and time after a symmetric trim, and
reports the median $|r|$ overall and in-band against a verdict threshold of 0.90.
It re-runs the two-sided transform on raw `fhr`/`up` rather than reading the
stored block, so it never has to reconstruct the selector's ordering.

> Three caveats on that tool: its printed verdict still tells the operator to set
> the dead `PHASE_HARMONIC_K_STEPS`; it hard-codes the two-sided
> `KymatioPhaseScattering1D` and the legacy operator, so its verdict does not
> describe a causal or integer build; and it uses `required=True` on `--hdf5`,
> which makes it unlaunchable from an IDE Run button under the repository's
> runner convention.

Full rationale and rejected alternatives:
`documents/docs-md-files/datasets/PHASE_HARMONIC_CHANNEL_SELECTION.md` — but read
it as a **pre-change recommendation document**, not as a description of this
dataset. Its "Today" columns (`fhr_ph` 44, `up_ph` 58, $c_y$ 87, $c_u$ 101)
describe the legacy `select_fhr_phase_coefficients` selection the pipeline no
longer uses; its "Recommended" row (66 / 15) is what shipped; it mentions neither
the causal variant, the warm-up drop rule nor the integer operator; and its
"Lean $k \in \{4,8\}$" row states `up_ph` = 12 where the measured value at that
band and those steps is **10** — which matters now, because $\{4,8\}$ is exactly
the integer operator's step set.

> **Historical note.** The legacy selector `select_fhr_phase_coefficients`
> compares its `min_freq` argument against the *normalised* $\xi$ without
> dividing by $f_s$, while the cross-channel selector does convert. Its
> nominal 0.006 / 0.002 Hz floors were therefore really 0.024 / 0.008 Hz,
> which is why the deceleration band had almost no phase representation. The
> current pipeline bypasses that selector rather than fixing it, because four
> other call sites depend on its present behaviour.

### 7.6. Cross-Phase Channel Selection

`fhr_up_ph` is produced by the **two-sided variant only** — the causal build
sets its width to `None`, refuses a causal file given `n_cross_phase_channels`,
and fails `_validate_geometry` if the dataset exists on one. Read this section
together with the binding defect recorded in §7.4.

It retains the two-band selector, unchanged, giving **79** channels
(fs = 4 Hz), decomposing as **Band A 50 + Band B 29** — disjoint, because their
FHR intervals are $[0.008, 0.04)$ and $[0.04, 0.25]$. The selector's own
metadata dict reports the split and is carried through as
`masks['cross_metadata']`.

**Band A — deceleration response**
- UP frequency $< 0.05$ Hz (contraction band)
- FHR frequency $\in [0.008, 0.04)$ Hz (deceleration band)
- $k \in \{0, 1, 2, 3, 4\}$, i.e. $p \in \{1.00, 1.19, 1.41, 1.68, 2.00\}$
- Fine $k$ resolution near the fundamental captures varying contraction-to-
  deceleration delays (20–120 s).

**Band B — variability coupling**
- UP frequency $< 0.05$ Hz
- FHR frequency $\in [0.04, 0.25]$ Hz (LF and MF variability)
- $k \in \{8, 12, 16\}$, i.e. $p \in \{4, 8, 16\}$
- Large ratios link slow contraction dynamics to fast variability; reduced
  variability during contractions is a marker of compromise.

The two masks are OR-ed, so Band A and Band B channels are **interleaved** in
pair-index order, not concatenated.

**This block keeps its $k = 0$ term.** Unlike the self-phase diagonal,
cross-channel $k = 0$ pairs UP at frequency $f$ against FHR at the same $f$ —
a genuine coupling measurement, not a duplicate of a stored energy channel. The
redundancy argument of §7.5 does not apply to it.

### 7.7. Per-Channel Provenance (`sel_*` attrs)

`fhr_ph` and `up_ph` each carry attributes describing every stored channel:

| Attr | Shape | Meaning |
|---|---|---|
| `sel_i` | $(C,)$ int32 | Index of the **lower**-frequency wavelet |
| `sel_j` | $(C,)$ int32 | Index of the **higher**-frequency wavelet |
| `sel_xi_i_hz` | $(C,)$ float32 | $\xi_i$ in Hz |
| `sel_xi_j_hz` | $(C,)$ float32 | $\xi_j$ in Hz |
| `sel_power` | $(C,)$ float32 | $p = \xi_j / \xi_i \ge 1$ |
| `sel_band_hz` | $(2,)$ float32 | The band the selection was built from |
| `sel_k_steps` | $(K,)$ int32 | The harmonic steps admitted. $K = 3$ under `ratio_power_v0`, $K = 2$ under `integer_harmonic_v1`; sourced from `phase_k_steps_for` (`:1375`), **never** from the dead `PHASE_HARMONIC_K_STEPS` |
| `sel_phase_operator` | UTF-8 str | The operator version the block's exponents follow (§7.3). Written unconditionally by the current writer |
| `sel_harmonic` | $(C,)$ int32 | The integer harmonic $k_{ij} \in \{2,4\}$ actually used as the exponent. Written **only** under `integer_harmonic_v1` |

Arrays are ordered to match the channel axis: `sel_xi_i_hz[c]` describes
channel `c` of the stored block. Read these rather than re-deriving the
selection — a consumer that rebuilds the mask with mismatched parameters gets a
silently wrong channel map.

Three traps in reading them:

- **`sel_i` is the lower-*frequency* filter, which is the numerically *larger*
  index.** Kymatio emits `center_freqs` in descending order, so on `fhr_ph`
  `sel_i` spans indices 7–30 while `sel_j` spans 3–26. A consumer that reads
  `sel_i < sel_j` as "lower frequency" gets every pair backwards.
- **`sel_power` is provenance, not the exponent, on an integer shard.** It
  remains the floating-point ratio $\xi_j/\xi_i$ under both operators; the
  exponent actually applied under `integer_harmonic_v1` is `sel_harmonic`. A
  consumer that reads `sel_power` and uses it as the exponent reproduces a
  different operator than the one that built the data.
- **The last two rows are absent from older shards.** The write rule is
  unconditional (`:617`) but the read rule is what matters: absence resolves to
  `ratio_power_v0` (`hdf5_dataset.py:100-124`), and every shard written before
  the operator version existed carries neither attribute — verifiable on the
  committed `tiny_shard_causal.hdf5`, whose `fhr_ph` has exactly the seven
  pre-change keys.

> The "write-only" note that stood here is no longer true of the whole set.
> `sel_phase_operator` and `sel_harmonic` **are** read:
> `causal_scattering.assert_matches_shard` (`:2070-2125`) uses them to refuse a
> rebuilt pair list that is not the shard's, and the loader and statistics path
> read the operator to refuse a mixed file list. The other six still have no
> reader inside `hdf5_dataset/`, and the `band_partition.py` migration remains
> outstanding (§11).

### 7.8. What the Geometry Cannot Represent

Two clinically relevant phenomena are below the window floor and no selection
choice recovers them:

- **Contraction repetition rate.** Normal labour is 3–4 contractions per
  10 min — a 150–200 s period, 0.005–0.0067 Hz. The wavelets there are
  1250–1500 s long against a 1200 s window. Only tachysystole (5+ per 10 min,
  0.0083 Hz) crosses the threshold.
- **Prolonged decelerations** (2–10 min) fall in the same category for FHR.

Mitigation, **on the two-sided variant**: `up_st` is stored complete at 43
channels (one $S_0$ plus 42 first-order wavelets) down to 0.000515 Hz, so the
contraction *envelope and timing* are present in the scattering block even
though the narrowband phase estimate at that frequency is not reliable.

**On the causal variant this mitigation is weaker.** The drop rule removes the
seven slowest wavelets, leaving 36 channels whose lowest surviving filter is
index 35, not 41 — precisely the filters that would have covered the
contraction repetition rate. A causal build is therefore blind to it in the
scattering block as well. The model is not blind to contraction timing; it lacks the
phase-coherence description of it. Recovering that properly needs a longer
analysis window or a lower $Q$ at the bottom of the bank — both segmentation
changes affecting the whole pipeline.

### 7.9. How the Transform Is Invoked

The writer has two transform paths, selected by `transform`.

**Two-sided path.** `phase_corr` and `cross_phase_corr` are mutually exclusive
in a single Kymatio forward call, so each batch of segments requires **four
passes**, with input stacked as $(B, 2, 5280)$, FHR at channel 0 and UP at
channel 1:

**with FHR at stack channel 0 and UP at channel 1**
(`create_new_pipeline.py:3029-3031`) — the opposite of MIMO's own block order
(§3.2), and the source of the cross-phase binding defect in §7.4:

| Pass | Arguments | Yields |
|---|---|---|
| 1 | `compute_phase=True, scattering_channel=0, phase_channels=[0]` | `fhr_st` (from `scattering`) and `fhr_ph` |
| 2 | `compute_cross_phase=True, phase_channels=[0,1]` | `fhr_up_ph` |
| 3 | `compute_phase=True, phase_channels=[1]` | `up_ph` |
| 4 | `scattering_channel=1` | `up_st` |

Each phase output arrives at full 903-pair width and is reduced by boolean
indexing with its mask, which preserves ascending pair order — the order the
`sel_*` metadata describes. Batching is controlled by `scatter_batch_size`
(default 16 in the function, 128 in `__main__`), with per-segment retry on
`RuntimeError`; segments that fail individually are recorded and dropped.

**Causal path.** The causal build does not instantiate
`KymatioPhaseScattering1D` for writing. `compute_scattering_masks` builds the
causal bank and a `CausalChannelPlan` once, then `CausalTorchBank` caches the
kernel spectra on the selected device. A batch is transformed by one call to
`transform_batch_numpy`, returning the four stored causal blocks directly:
`fhr_st`, `fhr_ph`, `up_st`, and `up_ph`. `fhr_up_ph` is absent by design.

The causal path still uses the same `PhaseChannelSelection` objects for
`fhr_ph` and `up_ph`, so the data channel order and the `sel_*` provenance
remain tied to one selector. The scattering blocks are gathered through the
channel plan, which drops the seven never-valid channels per block and supplies
the `causal_warmup_steps` and `causal_delay_s` attributes written beside them.
Per-segment retry on `RuntimeError` is shared with the two-sided path; this is
valid because the causal chain is batch-invariant.

The causal build path additionally uses a complex FFT on the wavelet leg (an
`irfft` would discard the negative-frequency content that *is* the measured
analyticity defect of §7.2), and `rfft`/`irfft` only for the two real-input
$\phi$ smoothings. Its FFT length is
$2^{\lceil \log_2(H + N) \rceil} = 2^{16}$ for $H = 32767$ and $N = 5280$,
shorter than the numpy reference's.

**Writes are per record, not per transform batch.** `scatter_batch_size`
controls only the transform; `append_samples_batch` is called once per record
(`:3165`) after the whole record's segments have been accumulated in Python
lists and is flushed in one resize-and-write. Peak host memory therefore scales
with a GUID's segment count (up to 37–38 segments across all five blocks), not
with `scatter_batch_size`.

> **A quiet build hides two MIMO notices.** Both `prepare_data` call sites run
> inside `suppress_stdout_stderr()` on the non-verbose branch (`:1813`, `:2915`,
> helper at `:387-396`), which swallows the adaptor's own prints — including
> `Subsampling by factor ...` (`mimo_adaptor.py:301`) and
> `<record>: adding all pad file: NaN data found in input` (`:310`). A record
> silently blanked to all-pad, or one that was resampled from a non-4 Hz source,
> therefore leaves **no trace whatever** in a production build; `verbose=True`
> is the only way to see them. Note also that neither the native rate nor a
> resample flag is stored, so whether a shard's record was resampled is not
> recoverable from the shard.

`_validate_geometry` runs **before** any transform work and raises on a
pair-axis mismatch, a missing dataset, an unexpected dataset for the variant,
or a width mismatch — see §11.

---

## 8. HDF5 Schema

One file per subgroup per partition. Storage geometry is **pre-trim**:
`len_signal` $= 5280$, `len_sequence` $= 330$. Every dataset is resizable
(leading `maxshape` `None`) and chunked at 32 samples along the batch axis with
LZF compression — **except `guid`**, which is chunked `(32,)` with no
compression. Files are created `h5py.File(path, 'w', libver='latest')` and
appended in mode `'a'`, also `libver='latest'`. `create_initial_hdf5`
`os.remove`s the path first, so creation always overwrites.

> **`libver='latest'` is required, not cosmetic.** Its dense attribute storage
> is what admits an attribute past HDF5's 64 KB object-header limit, which the
> `causal_novelty_curve` written by the current writer exceeds at the legacy
> phase width: $66 \times 331 \times 4 = 87{,}384$ B $= 85.3$ KiB. Under the
> integer operator the same curve is $44 \times 331 \times 4 = 58{,}256$ B
> $= 56.9$ KiB and would fit — so the requirement bites only at the wider
> geometry, but the writer always sets it.

The three variant axes of the front matter all show up here. **The channel
widths of the two phase blocks depend on the phase operator**, so the table
gives three geometries:

| Field | dtype | Two-sided | Causal, `ratio_power_v0` | Causal, `integer_harmonic_v1` | Description |
|---|---|---|---|---|---|
| `fhr` | float32 | $(5280,)$ | $(5280,)$ | $(5280,)$ | Raw FHR (bpm) |
| `up` | float32 | $(5280,)$ | $(5280,)$ | $(5280,)$ | Raw UP (mmHg), shifted 20 s earlier (§3.4) |
| `fhr_st` | float32 | $(43, 330)$ | $(36, 330)$ | $(36, 330)$ | FHR scattering, $S_0$ + $S_1$ |
| `fhr_ph` | float32 | $(66, 330)$ | $(66, 330)$ | $(44, 330)$ | FHR self-phase (+ `sel_*` attrs) |
| `fhr_up_ph` | float32 | $(79, 330)$ | absent | absent | UP↔FHR cross-phase (§7.4) |
| `up_st` | float32 | $(43, 330)$ | $(36, 330)$ | $(36, 330)$ | UP scattering |
| `up_ph` | float32 | $(15, 330)$ | $(15, 330)$ | $(10, 330)$ | UP self-phase (+ `sel_*` attrs) |
| `target` | float32 | $(330,)$ | $(330,)$ | $(330,)$ | $\texttt{class\_id} \times w(t)$ |
| `weight` | float32 | $(330,)$ | $(330,)$ | $(330,)$ | Binary validity mask |
| `epoch` | float32 | scalar | scalar | scalar | Seconds relative to delivery (negative = before) |
| `cs_label` | uint8 | scalar | scalar | scalar | 1 if caesarean section |
| `bg_label` | uint8 | scalar | scalar | scalar | 1 if blood gas available |
| `time_from_labor_onset` | float32 | scalar | scalar | scalar | Seconds since labour onset; NaN if unknown |
| `second_stage_onset` | float32 | scalar | scalar | scalar | Seconds since second-stage onset; NaN if unknown |
| `guid` | UTF-8 str | scalar | scalar | scalar | Recording identifier |

**Total feature channels per time step:** two-sided
$43 + 66 + 79 + 43 + 15 = 246$; causal legacy $36 + 66 + 36 + 15 = 153$;
causal integer $36 + 44 + 36 + 10 = \mathbf{126}$.

**Field notes**

- `epoch` $/ 60$ gives minutes before delivery.
- `time_from_labor_onset` $= \texttt{domain\_start} - t_{\text{onset}}$ where
  $t_{\text{onset}} = 3600 \cdot \texttt{labor\_onset\_hours}$ is negative
  (§3.9). It is therefore **elapsed seconds since labour onset, positive after
  onset** — not the CSV value. NaN propagates through the subtraction whenever
  the normalised GUID is absent from the map. Worked example: onset at $-17$ h
  $= -61200$ s and $\texttt{domain\_start} = -3600$ s give
  $\texttt{tlo} = 57600$ s $= 16$ h since onset. Divide by 3600 for hours;
  `second_stage_onset` follows the same convention.
- Every coefficient block is a first-class dataset with its own per-channel
  statistics. `up_ph` is **not** sliced from `fhr_up_ph`.

**Model input groupings**

- **VAE-TEB target encoder:** `fhr_st` + `fhr_ph`, so
  $c_y = 43 + 66 = 109$ two-sided, $36 + 66 = 102$ causal legacy,
  $36 + 44 = \mathbf{80}$ causal integer.
- **VAE-TEB source encoder (lag-attn v1):** `up_st` + `up_ph`, so
  $c_u = 43 + 15 = 58$ two-sided, $36 + 15 = 51$ causal legacy,
  $36 + 10 = \mathbf{46}$ causal integer. `fhr_up_ph` is deliberately *not*
  consumed — the v1 spec requires a source-pure UP pathway so the model must
  infer coupling rather than being handed it. (Given §7.4, that is also what
  keeps the cross-phase binding defect off every current training path.)
- **Causal multimodal transformer:** `fhr_st` and `up_st` as separate streams.
- After trimming, models receive $(300, C)$ — transposed from the stored
  $(C, 330)$ at load time.

**All three geometries are live simultaneously.** Measured across the ten
shipped packages' `configs/default.yaml`:

| Geometry | $c_y$ / $c_u$ | Packages |
|---|---|---|
| Two-sided | 109 / 58 | `lag_attn`, `lag_attn_fs`, `lag_attn_rws`, `lag_attn_transformer_fs`, `lag_attn_transformer_rws` |
| Causal, `ratio_power_v0` | 102 / 51 | `lag_attn_crws`, `lag_attn_transformer_crws` |
| Causal, `integer_harmonic_v1` | 80 / 46 | `lag_attn_cfs`, `lag_attn_transformer_cfs` |

The `lag_attn_cfs` config names `integer_harmonic_v1` explicitly in its comment,
so the integer build is not hypothetical — two packages already declare its
widths. **No config anywhere carries 87 or 101**; the note that stood here about
stale $c_y = 87$ / $c_u = 101$ was itself stale in both directions and has been
removed.

### 8.1. The Two Transform Variants

Every stored coefficient is a weighted average of raw samples. In the
**two-sided** variant that average is taken over samples on *both* sides of
step $t$, so a channel read as "the past up to $t$" carries part of $t$'s own
future — up to 965 s of it on the slowest `fhr_st` channel. The **causal**
variant runs the same cascade through a strictly one-sided gammatone bank
matched to the production Morlets at half power, so a coefficient at $t$ is a
function of $\{x(s) : s \le t\}$ and of nothing else. The mathematics is in
`CAUSAL_SCATTERING_PHASE_HARMONIC_MATH.md`.

Three things follow, and all three are visible in the file:

**Seven channels per scattering block are dropped.** A one-sided filter's
output depends on the assumed pre-recording history until its **warm-up**
$W_{0.95}$ — the leading delay enclosing 95 % of the kernel's energy — has
passed. Seven wavelets per block have $W$ between 348 and 636 steps against a
330-step segment, so their boundary never closes: they carry no signal at any
step of any segment and are dropped at write time rather than masked later.
Both phase selections are band-limited at 0.008 Hz, which excludes those
filters entirely, so both phase blocks keep their full width and the drop is a
clean channel-axis operation rather than a re-selection.

**`fhr_up_ph` is not produced.** No model loads it and it is the one block with
no `sel_*` attributes to verify channel identity against.

**Every channel is stale.** Beyond its warm-up a causal channel still lags by
its composed group delay — 13.3 s ($S_0$) to 791.0 s (slowest survivor) on
`fhr_st`, 20.5–402.2 s on `fhr_ph`, 150.8–402.2 s on `up_ph`. The pipeline
records that delay per channel and **does not compensate for it**;
`read_causal_warmup` surfaces it beside the warm-up for a consumer that aligns
channels in time.

**"Forecast" means something different per channel.** The same group delay puts
most of a coefficient's weight before the anchor, so the share of it drawn from
samples the anchor has *not* seen ranges from all of it on the fast channels to
$2.6\%$ on the slowest kept one over a 120 s horizon. That is not a leak — every
one of those coefficients still depends on samples after the anchor — but a
block score summed over both mixes two different claims, so the whole
envelope-mass curve is measured at write time and stored as `causal_novelty_curve`
— for every window rather than at one horizon, because the forecast horizon is a
model-side choice the dataset must not bake in (older shards carry the scalar
`causal_novelty_frac` at 120 s instead).

**Attributes.** A file is self-describing. Root:

| Attribute | Variant | Meaning |
|---|---|---|
| `transform` | both | `'two_sided'` or `'causal'` |
| `source_pickle_path` | both | Fold pickle the run resumed from, or `'<fresh run>'` |
| `source_guid_digest` | both | SHA-256 hex digest of the sorted, newline-joined GUID stems the shard **will** be built from. Computed at file creation, before any record is read, so a GUID whose every segment is later filtered out still contributes to it |
| `causal_kernel_taps` | causal | 32768 |
| `gammatone_order` | causal | 4 |
| `causal_warmup_quantile` | causal | 0.95 |
| `causal_leg_alignment` | causal | `'none'` or `'envelope'` — which phase-harmonic **leg alignment** built the two phase blocks (§7.3). **Absent reads as `'none'`**, which every causal shard written before the mode existed is. |
| `causal_phase_operator` | causal | `'ratio_power_v0'` or `'integer_harmonic_v1'` — which phase-harmonic **operator version** the two phase blocks' exponents follow (§7.3), and hence their width. **Absent reads as `'ratio_power_v0'`.** A two-sided file never carries it |

> **These two attributes are different axes and were once described with the
> same words.** Four code sites still say "which phase-harmonic *operator*
> built the two phase blocks" of `causal_leg_alignment` —
> `create_new_pipeline.py:843-845` and its `--leg-alignment` help at
> `:3686-3689`, `hdf5_dataset.py:70-72`, and `calculate_dataset_stats.py:105-107`
> and `:591-593`. That wording predates `causal_phase_operator` and now makes
> the two read as synonyms; it should say "leg alignment". This document has
> been corrected; the code has not.

Per coefficient block, beside the `sel_*` provenance of §7.7. A causal file
carries that provenance identically **under the legacy operator**; under
`integer_harmonic_v1` the phase selections themselves change, so the widths
change, `sel_k_steps` becomes $(4, 8)$, and `sel_harmonic` appears:

| Attribute | Variant | Meaning |
|---|---|---|
| `causal_warmup_steps` | causal | $(C,)$ int32, warm-up per channel in **untrimmed** decimated steps |
| `causal_delay_s` | causal | $(C,)$ float32, composed group delay per channel |
| `causal_novelty_curve` | causal | $(C, T+1)$ float32, share of each channel's composed envelope mass within $w$ stored steps, $w = 0..T$; horizon-free (legacy shards: `causal_novelty_frac`, $(C,)$ at the 120 s horizon) |

`causal_warmup_steps` is stored untrimmed because that is the storage geometry
every other field uses; a consumer reading the file at a given `trim_minutes`
rebases it itself (§10.3). The warm-up is an attribute rather than a stored
per-sample mask because it is a property of the *filter bank*: identical for
every segment in every file, so a $(C, 330)$ boolean array per sample would
replicate one constant tens of thousands of times — about 76 KB per sample
against about 600 bytes per file.

### 8.2. Compatibility

Every dataset and stats file written before this variant existed carries **no**
`transform` attribute, and none was rewritten to add one. The rule, applied
identically by the loader, the statistics calculator and every consumer:

| Attribute | Absent means | Values |
|---|---|---|
| `transform` | **legacy two-sided** — the normal state of every older file | `'two_sided'`, `'causal'` |
| `causal_leg_alignment` | `'none'` | `'none'`, `'envelope'` |
| `causal_phase_operator` | `'ratio_power_v0'` | `'ratio_power_v0'`, `'integer_harmonic_v1'` |

All three resolutions are applied identically by the loader, the statistics
calculator and every consumer (`hdf5_dataset.py:56-124`). On a `'causal'` file
`causal_warmup_steps` is **mandatory** on every block, and a warm-up vector
whose length differs from the block width is a hard `ValueError`;
`causal_delay_s` and `causal_novelty_frac` are read tolerantly;
`causal_novelty_curve` is read tolerantly but its row count *is* checked
against the block width.

- Absence is not a defect and produces no warning. Reading an existing file is
  the common case.
- Absent and `'two_sided'` are the same variant. Coherence checks compare
  *resolved* values, so a `paths` list mixing an old shard with a newly built
  two-sided one is accepted.
- **A `paths` list is refused on five properties**, not one
  (`_check_layouts_agree`, `hdf5_dataset.py:285-338`): resolved `transform`,
  resolved `causal_leg_alignment`, resolved `causal_phase_operator`, the block
  set, and every block's width. `causal_warmup_quantile` is read into the layout
  but **not** compared — two shards built at different quantiles are caught only
  indirectly, and only if their widths actually differ.
- **Append-time tolerance is deliberate.** `append_samples_batch` resizes every
  dataset the file happens to have, writes `up_st` / `up_ph` only when both the
  batch and the dataset exist, and writes the two TLO fields only when those
  datasets exist — so an older file lacking them is still appendable.
  `fhr_up_ph` is the exception, guarded in **both** directions: dataset without
  batch, and batch without dataset, each raise.
- A file claiming `'causal'` without warm-up attributes is corrupt and raises;
  a file with neither is simply legacy. The two are distinguishable precisely
  because `transform` is now written on both variants.
- A newly built two-sided file is **not** byte-identical to an old one: it
  gains the root attributes above. Every stored *array* is bit-for-bit
  unchanged, and no attribute is renamed, retyped or removed.

---

## 9. Normalisation and Statistics

### 9.1. Per-Field Transforms

Each block gets the transform suited to its distribution, applied per channel
and followed by standardisation:

| Field(s) | Transform | Rationale |
|---|---|---|
| `fhr`, `up` | none | raw signals, approximately Gaussian |
| `fhr_st`, `up_st` ch 0 | none | the $S_0$ low-pass average |
| `fhr_st`, `up_st` ch 1 and up | $\log(x + \varepsilon)$ | scattering moduli are positive and heavy-tailed |
| `fhr_ph`, `fhr_up_ph`, `up_ph` | $\operatorname{asinh}(x)$ | phase correlations are signed and heavy-tailed |

with $\varepsilon = 10^{-6}$ and

$$\operatorname{asinh}(x) = \ln\!\left(x + \sqrt{x^2 + 1}\right)$$

which compresses tails like $\log$ but is defined and odd-symmetric on the
whole real line — necessary because phase correlations take both signs.

The log branch clamps first, $\log(\max(x, 0) + \varepsilon)$, so a negative
input maps to $\log \varepsilon \approx -13.8$ rather than NaN.

**The configuration is width-agnostic**, which is why the same file works for
every variant: `log_norm_channels_config = {'fhr_st': 'all_except_0', 'up_st':
'all_except_0'}` and `asinh_norm_channels_config = {'fhr_ph': 'all',
'fhr_up_ph': 'all', 'up_ph': 'all'}` are **strings**, expanded against the width
read from each HDF5 field (`calculate_dataset_stats.py:135-197`). So
"channels 1 and up" is 1–42 on a two-sided block and 1–35 on a causal one; the
code never sees a literal 42. Channels in neither list become
`regular_channels` (no transform), and on an overlap log wins with a
`warnings.warn` — neither branch is reachable with the shipped configs.

### 9.2. Standardisation

Then, per channel $c$:

$$\hat{x}_c = \frac{g(x_c) - \mu_c}{\sigma_c + 10^{-8}}$$

where $g$ is that channel's transform. Note the two distinct epsilons: the
$10^{-6}$ inside the log, and the $10^{-8}$ added to the **standard
deviation** (not the variance) in the denominator.

### 9.3. Computing the Statistics

`DatasetStatsCalculator` accumulates **transformed** values — the transform is
applied before summation, so $\mu_c$ and $\sigma_c$ are moments of $g(x_c)$,
exactly what §9.2 needs.

Two raw moments are accumulated in float64:

$$\mu_c = \frac{1}{N_c}\sum g(x_c), \qquad
\sigma^2_c = \frac{1}{N_c}\sum g(x_c)^2 - \mu_c^2$$

with $N_c$ counted **per channel**, excluding non-finite values. Negative
variance from catastrophic cancellation is clamped to 0; channels with
$N_c = 0$ get $\mu_c = \sigma^2_c = 0$. Results are stored as float32.

Trimming is applied before accumulation using the same *formulas* as the loader
(§3.3) — though not the same code: `calculate_dataset_stats.py:124-129`
reimplements the arithmetic with bare literals rather than calling
`decimated_trim_steps`. The two agree for every value in the repository. The
`trim_minutes` used is recorded in the stats file and the loader warns on
mismatch.

**On a causal file the warm-up is excluded too.** Accumulating over $t < W_c$
would fold the assumed pre-recording history into the constants every model
normalises with, so the source file's `causal_warmup_steps` is read, rebased
for the trim, and those steps left out per channel:

$$\mu_c = \frac{1}{N_c}\sum_{t \ge W_c} g(x_c(t)), \qquad
N_c = n_{\text{samples}} \cdot \big(T_{\text{trimmed}} - \max(W_c - \text{trim},\ 0)\big)$$

The per-channel counting already in place is what makes this a change to *which
samples enter the sum* rather than to the structure. A channel left with no
valid step **raises**, naming field and channel: its $N_c = 0$ would reach
`std = 0`, and `normalize_tensor_data` divides by $\sigma_c + 10^{-8}$, which
would inflate that channel by $10^{8}$ with no exception anywhere. A two-sided
or legacy file has no such attribute and accumulates exactly as it always has.
`plot_histograms` applies the same exclusion to its own collection loop, so the
distributions it draws describe the same region as the constants beside them.

**The exclusion is implemented by NaN-blanking, not by slicing.**
`_blank_warmup_region` (`:262-300`) writes
`data[:, channel, :int(invalid_steps)] = np.nan` per channel, and the
per-channel `isfinite` filter already in place then drops those steps from
$N_c$. That is why there is no second exclusion path — and why the displayed
$N_c$ above is an **upper bound**: any step whose transformed value is
non-finite for its own reasons is dropped by the same mask, and nothing
separates the two causes. A raw $(B, T)$ field or a two-sided layout is
returned untouched.

A mixed file list is **refused**, on the same five properties as the loader
(§8.2) — resolved variant, leg alignment, phase operator, block set and block
widths — because `_resolve_valid_region` delegates to the loader's own
`_resolve_dataset_layout`. The reason it must be refused is the blow-up above:
field shapes are taken from the first file alone, so a 36-wide file fed into a
43-wide accumulator would leave seven channels at $N_c = 0$.

> `calculate_stats` takes the per-file sample count from
> `f[next(iter(self.stats_fields))]`, i.e. hard-wired to `fhr` by list order,
> while every other read in the method is guarded with
> `if field not in f: continue`. A shard without an `fhr` dataset raises a bare
> `KeyError` rather than a named refusal. The pipeline always writes `fhr`.

### 9.4. Stats File Layout

Written `libver='latest'`. Root attrs: `created_at` (ISO timestamp),
`trim_minutes` ($-1.0$ = no trim), `transform`, **`causal_leg_alignment`**,
**`causal_phase_operator`**, `log_epsilon` (a hard literal $10^{-6}$, not the
instance value), `description`, plus an optional `metadata` group.

The three variant attributes are stamped from the **source dataset's** resolved
layout and are written on **both** variants, defaulting to the legacy values
when the source is two-sided or legacy — so absence means "written before the
attribute existed", never "unlabelled".

Per field group:

| | Single-channel (`fhr`, `up`) | Multi-channel |
|---|---|---|
| Datasets | `mean`, `variance`, `std` (scalars) | `mean`, `variance`, `std`, shape $(C,)$ |
| Attrs | `shape`, `count`, `mean_scalar`, `variance_scalar`, `std_scalar` | `shape`, `count`, `n_channels`, `regular_channels`, `log_channels`, `asinh_channels`, `uses_log_transform`, `uses_asinh_transform`, and on a causal source `causal_warmup_steps` |

`causal_warmup_steps` is recorded here in the same untrimmed coordinates the
dataset stores it in, so one attribute name means one thing in both files.

**Pairing is checked before normalisation, not inside it.**
`_check_stats_pairing` (`hdf5_dataset.py:1405-1484`) raises on **four**
mismatches: resolved `transform`, resolved `causal_leg_alignment`, resolved
`causal_phase_operator`, and per-block `n_channels`. The leg-alignment check
matters because it is the *only* one that can fire between an aligned and an
unaligned shard — their widths are identical. That check runs **outside** the
broad `except Exception` that wraps the stats load, because that handler
answers a failure by disabling normalisation — so a mispaired stats file caught
inside it would silently degrade a run to *unnormalised* training data rather
than stopping it. A stats file that cannot even be opened returns early and is
left to the recovering load below. A legacy stats file paired with a legacy or
two-sided dataset resolves to the same values and is silent.

The per-channel transform assignment lives **in the stats file**, and the
loader adopts it wholesale, overwriting its own defaults. A field whose stats
group lacks those attrs silently gets no transform.

> **Statistics must be recomputed whenever the channel selection changes.**
> The pipeline and the loader both derive widths from the data, but a stats
> file stores fixed-width arrays — an old one will broadcast-fail against new
> data. Compute stats on the **training** partition of a fold and use that same
> file for train, val and test.

`plot_histograms` renders raw-vs-normalised distributions per channel, clipped
to the 5th–95th percentile, through the *same* `normalize_tensor_data` code
path the loader uses. It recomputes statistics internally, so enabling it
doubles the pass over the data.

---

## 10. PyTorch Data Loading

Two layers: `CombinedHDF5Dataset` for independent segments, and
`SignalSequenceDataset` for whole GUID sequences.

### 10.1. Segment-Level — `CombinedHDF5Dataset`

Returns one segment per `__getitem__` as an `AttributeDict` (dict with
attribute access: `sample.fhr_st` $\equiv$ `sample["fhr_st"]`).

Per-item order of operations:

1. FIFO cache probe (returns by reference, not a copy)
2. Read `f[name][sample_idx]` — one HDF5 read per field
3. **Trim**, by field class:
   | Fields | Slice | Amount |
   |---|---|---|
   | `fhr`, `up` | `[a:-a]` | 240 |
   | `fhr_st`, `fhr_ph`, `fhr_up_ph`, `up_st`, `up_ph` | `[:, a:-a]` | 15 |
   | `target`, `weight` | `[a:-a]` | 15 |
   Metadata (`guid`, `epoch`, labels, TLO) is never trimmed.
4. Tensor conversion, optional pinning, dtype cast
5. **Normalise** — on the $(C, T)$ layout, before transposing
6. **Transpose** $(C, T) \rightarrow (T, C)$ for the coefficient fields the
   file actually carries (those with `tensor.dim() == 2`)
7. Optional `<block>_valid` masks (§10.3)
8. Three unconditional **provenance keys**, added with `setdefault`:
   `source_file` (normpath'd), `source_file_basename`, `source_file_index`
9. FIFO insert

`SignalSequenceDataset` classifies all three provenance keys as meta fields and
does **not** stack them, so they exist per segment but vanish at the GUID level.

Coherence and statistics are resolved at **construction**, in this order
(`hdf5_dataset.py:1243-1264`): `_resolve_dataset_layout` →`_resolve_validity`
(rebase, print, dead-channel refusal) → the `emit_validity_mask` guard →
`_load_normalization_stats` (which calls `_check_stats_pairing` first) →
`_build_index`. All of it happens *before* the index is built, not during it.

Constructor arguments:

| Arg | Default | Effect |
|---|---|---|
| `paths` | required | One path or a list; a flat index is built across files |
| `load_fields` | `None` | Restrict which datasets are read |
| `allowed_guids` | `None` | GUID whitelist |
| `cs_label` / `bg_label` | `None` | Vectorised equality filters |
| `epoch_min` / `epoch_max` | `None` | Inclusive bounds on `epoch` |
| `label` | `None` | Keep sample if any timestep equals this class id |
| `cache_size` | `2000` | FIFO sample cache; 0 disables |
| `pin_memory` | `True` | Pin each tensor for faster host→device copy |
| `dtype` | `torch.float32` | `float16` supported |
| `stats_path` | `None` | Stats HDF5; `None` disables normalisation |
| `normalize_fields` | `None` | Restrict which fields are normalised |
| `trim_minutes` | `None` | **Must be 1.0** for production (§3.3) — the default performs no trimming |
| `emit_validity_mask` | `False` | Add a `<block>_valid` boolean field per sample (§10.3). Causal datasets only |

Multiprocessing-safe: `__getstate__` / `__setstate__` drop locks, open HDF5
handles and the sample cache so each worker reopens lazily. The validity-mask
cache is deliberately kept: it is a small immutable filter-bank constant, not
sample data.

> **Caveat.** The class docstring says target and weight are always included,
> but `load_fields` is used verbatim — passing `load_fields=['fhr_st']` yields
> a sample with no `target` or `weight`.

**Factory:** `create_optimized_dataloader(hdf5_files, batch_size=32,
num_workers=4, rank=0, world_size=1, …)` adds `DistributedSampler` when
`world_size > 1`, spawn multiprocessing, persistent workers, and
`attribute_dict_collate`. Leave `shuffle=None` — passing `shuffle=True`
alongside a distributed sampler raises.

### 10.2. GUID-Level — `SignalSequenceDataset`

In `guid_hdf5_dataset.py`. Wraps `CombinedHDF5Dataset` and groups by GUID,
returning one **sequence** per item with segments sorted by ascending epoch.
`__len__` is the number of unique GUIDs.

Each item stacks every tensor field to $(S_i, \ldots)$ — e.g. `fhr_st` becomes
$(S_i, 300, 43)$ two-sided, $(S_i, 300, 36)$ causal, with `fhr_ph` at
$(S_i, 300, 66)$ or $(S_i, 300, 44)$ by operator — and adds derived fields the
flat dataset has no notion of:

- `epoch` $(S_i,)$
- `delta_t` $(S_i,)$, with `delta_t[0] = 0` and the rest
  $\texttt{epoch}_{s} - \texttt{epoch}_{s-1}$ (asserted non-negative)
- `segment_indices` $(S_i,)$, $\text{round}\big((\texttt{epoch} -
  \texttt{epoch}_0) / 1200\big)$ — position on the GUID timeline
- `num_segments`, `guid`, `cs_label`, `bg_label`

Caching here is a true LRU, distinct from the inner FIFO.

`sequence_collate_fn` right-pads to $S_{\max}$ and emits `mask` $(B, S_{\max})$
bool and `lengths` $(B,)$. Pad values: `target` $\rightarrow -1.0$, `weight`
$\rightarrow 0.0$, `segment_indices` $\rightarrow -1$, everything else $0.0$.

**Factory:** `create_sequence_dataloader(hdf5_files, batch_size=4,
segment_duration=1200.0, guid_cache_size=128, …)`.
`estimate_class_weights` derives inverse-frequency weights from the first
segment of each GUID, labelling it `int(target_max > 1)`, normalising the
weights to sum to `num_classes` and replacing an infinite weight (an empty
class) with 1. **It reads `target` directly from HDF5 and bypasses the loader**,
so its per-segment max is taken over the untrimmed 330 steps, not the 300 the
model sees — a class present only in the trimmed-off edge still labels the GUID.
`_build_index`'s `label` filter reads untrimmed for the same reason.

`length_bucket_sampler.py` provides `LengthBucketSampler` and
`VariableBatchBucketSampler`, **and a factory that wires the first of them in**:
`create_bucketed_sequence_dataloader(hdf5_files, batch_size=8,
bucket_ranges=None, ...)` builds a `SignalSequenceDataset`, pairs it with a
`LengthBucketSampler` over `dataset.guid_lengths`, uses `sequence_collate_fn`
and returns `(DataLoader, dataset)`. It is `create_sequence_dataloader` in
`guid_hdf5_dataset.py` that wires no sampler, and it carries an open TODO
saying so.

`LengthBucketSampler` defaults to `bucket_ranges = [[1,5],[6,12],[13,20],[21,40]]`,
`shuffle=True`, `seed=42`, per-epoch seed `seed + epoch`. Its `__iter__`
**auto-increments** the internal epoch counter, so a reshuffle happens without a
`set_epoch` call; items longer than the final range overflow into the last
bucket rather than being dropped; and `__len__` is the item count, not the batch
count. `VariableBatchBucketSampler` is a *batch* sampler taking
`bucket_batch_sizes` as ordered `[((lo, hi), batch_size), ...]`, rejecting an
empty list or a non-positive batch size, with `__len__` the number of batches.
Nothing in the repository instantiates it.

A simpler alternative, `build_guid_filtered_dataloader` in `hdf5_dataset.py`,
yields one variable-size batch per GUID via `GuidBatchSampler` without
stacking, padding or epoch sorting. Eligibility is strict:
`count > min_samples`.

### 10.3. The Causal Valid Region

On a causal dataset the loader exposes where each channel stops reporting the
assumed pre-recording history and starts reporting the recording. On a
two-sided or legacy one every one of these is `None` or raises, silently and by
design — a two-sided block has no warm-up, and the only mask it could produce
is all-`True`, which asserts exactly the claim the causal variant exists to
stop making.

```python
ds = CombinedHDF5Dataset(paths=..., trim_minutes=1.0)

ds.transform                 # 'causal' | 'two_sided' (absent attribute → two_sided)
ds.causal_warmup_steps       # legacy operator: {'fhr_st': (36,), 'fhr_ph': (66,),
                             #                  'up_st': (36,), 'up_ph': (15,)}
                             # integer operator: fhr_ph (44,), up_ph (10,)
                             # or None on a two-sided dataset
                             # the key set is exactly the blocks the file stores
ds.channel_valid_mask('fhr_st')   # (300, 36) bool in the model's (T, C) layout, cached

ds = CombinedHDF5Dataset(..., emit_validity_mask=True)   # opt-in, causal only
sample.fhr_st_valid          # (300, 36) bool; collates to (B, 300, 36)
```

**Rebasing.** The stored vector is untrimmed (§8.1); the property returns it in
*this dataset's* coordinates, $W' = \max(W - \text{trim},\ 0)$. At
`trim_minutes = 1.0` a stored warm-up of 20 reports 5, and the slowest
surviving channel's 293 reports 278 — leaving **22 valid steps of 300**. That
is 7 % of the window, so the per-channel valid-step counts are **printed** at
index build rather than merely being available. A channel with no valid step at
all raises, naming field and channel: an all-`False` column normalises to zeros
a model cannot distinguish from real coefficients.

**Mask emission is off by default.** The mask is a dataset constant, so paying
for it per sample, per worker, per collate and per host-to-device copy is only
worth it for a model that consumes it batched; anything else reads
`channel_valid_mask` once. When on, the mask fields bypass tensor conversion,
normalisation and the transpose — a float32 cast would turn a boolean mask into
numbers that are all truthy. `SignalSequenceDataset` stacks them to
$(S_i, 300, C)$ like any other tensor field and `sequence_collate_fn` pads them
to `False` without a per-field entry, because `torch.full(..., 0.0,
dtype=torch.bool)` already is `False`. The mask cache is **kept** through
`__getstate__`, unlike the sample cache: a few tens of KB of immutable
filter-bank constant, identical in every worker. **The emitted tensor is that
shared cached constant, not a per-sample copy** — treat it as read-only.

**A loader-free reader exists, and it is stricter.**
`read_causal_warmup(paths, trim_minutes) -> CausalWarmup`
(`hdf5_dataset.py:526-705`) is the supported way to resolve a channel budget
*before* building a loader. Unlike the loader's own scan it raises on a missing
file rather than skipping it, reads **every** file rather than only the first,
requires and width-checks `causal_delay_s`, and additionally requires the shards
to agree on `causal_warmup_quantile`, the stored block length, the warm-up
vectors themselves, the delay vectors and the novelty curve. The returned
`CausalWarmup` carries, beside the rebased `warmup_steps`: `paths`,
`trim_minutes`, `trim_steps`, `quantile`, `delay_s` (seconds, **unrebased and
uncompensated** — the trim moves where a window starts, not how far back in
physical time a coefficient's content sits), `novelty_frac` (the legacy
per-channel scalar, empty on a current build), `novelty_curve` (the horizon-free
$(C, W+1)$ table every current build writes, empty on a legacy shard),
`leg_alignment`, `phase_operator`, and `kept_steps` per block.

**Coherence.** A `paths` list must agree on resolved variant, resolved leg
alignment, resolved phase operator, block set and per-block widths, checked with
one attribute read per file at construction (§8.2). A mixed list is otherwise
accepted silently and fails much later in two different ways:
`default_collate` raises something opaque on the first mixed batch, while
`SignalSequenceDataset` iterates the first segment's keys and instead **drops**
the field the other file did not have.

**The group delay is not compensated.** Beyond its warm-up a causal channel is
still stale by `causal_delay_s` and nothing here shifts it back (§8.1).

### 10.4. Optional Fields

| Field | Present when | If absent |
|---|---|---|
| `up_st` | `n_up_st_channels > 0` at creation | Loader skips silently |
| `up_ph` | `up_ph_selection` passed at creation | Loader skips silently — regenerate to use the lag-attn v1 model |
| `fhr_up_ph` | Two-sided variant only | Loader skips silently; `__getitem__` skips any field the file lacks before every membership test, which is why an absent cross-phase block needs no special case |
| `second_stage_onset` | New pipeline only | Loader skips silently |
| `time_from_labor_onset` | All recent datasets | NaN per-GUID |

The **loader** tolerates absence; the **writer** does not — `_validate_geometry`
requires every block the *variant* declares, because a missing dataset would
otherwise cause `append_samples_batch` to silently discard computed
coefficients. `fhr_up_ph` is therefore mandatory on a two-sided build and legal
to omit on a causal one, and the symmetric guard also refuses a cross-phase
batch handed to a file with no cross-phase dataset.

`time_from_labor_onset` and `second_stage_onset` flow through generically: the
dataset iterates HDF5 keys, `SignalSequenceDataset` stacks them to $(S_i,)$,
and `sequence_collate_fn` right-pads to $(B, S_{\max})$. They are metadata and
are not normalised. NaN entries are preserved and must be handled downstream.

> An earlier revision named a learned `missing_embedding` in a `TLOEmbedding`
> module as the downstream handler. **Neither symbol exists anywhere in the
> repository**; the claim has been removed rather than re-anchored. A consumer
> of these fields must handle NaN itself.

---

## 11. Validation and Regeneration

### 11.1. Built-In Guards

`_validate_geometry` runs once per output file, before any transform work, and
raises `ValueError` on **four** conditions:

1. **Pair-axis mismatch** — the mask spans a different number of wavelet pairs
   than the transform produces (a $J$ / $Q$ / `shape` divergence). Unchecked
   this raises `IndexError` inside the per-record handler, so every record
   fails identically, the run reports success, and empty files ship. **This
   check alone** is skipped when no transform model is supplied, which is the
   causal path; the other three run on both arms.
2. **A missing coefficient dataset** — which would be silently dropped. The
   required set is per variant (§10.4).
3. **An unexpected dataset** — one that exists in the file but which this build
   does not produce (expected width `None`), e.g. `fhr_up_ph` on a causal file.
4. **A width mismatch** between the stored dataset and what will be written.

The guard takes *resolved* expected widths: from the transform model on the
two-sided path, where scattering widths are derived as
$1 + |\texttt{center\_freqs}|$ rather than assumed, and from the channel plan on
the causal path. Either way a filter-bank change is caught here.

**Four pre-flight validators**, not one, run as the first four statements of
`create_new_pipeline` — before `os.makedirs` and before the CSV is read, so a
typo does not leave an output directory behind and fail later on something
unrelated. Each returns its argument so it can wrap an assignment, and each
names the valid values in its refusal:

| Validator | Refuses anything outside |
|---|---|
| `validate_transform` (`:253-270`) | `('two_sided', 'causal')` |
| `validate_leg_alignment` (`:298-322`) | `('none', 'envelope')` |
| `validate_phase_operator` (`causal_scattering.py:193-211`) | `('ratio_power_v0', 'integer_harmonic_v1')` |
| `validate_test_mode` (`:273-295`) | `('augmented', 'holdout')` |

They are not redundant with argparse's `choices=`: that guards the command line
only, and a value set in `RUN_ARGS` never passes through the parser (§11.3).
`validate_test_mode` in particular exists because its consumer is an
`if actual_mode == 'holdout'` with an implicit `else` (`:3559`, `:3592`), so a
misspelling would silently produce an *augmented* build.

Six further refusals live in `create_initial_hdf5` (`:763-813`): a non-legacy
`phase_operator` on a two-sided file; a `PhaseChannelSelection` whose own
operator disagrees with the one being recorded as the root attribute (so a file
cannot misdescribe its own channel axis); a causal file with `channel_plan=None`;
a causal file given `n_cross_phase_channels`; a causal file with
`up_ph_selection=None`; and a per-block width that disagrees with
`channel_plan[name].n_channels`. Three more sit elsewhere:
`append_samples_batch` guards `fhr_up_ph` symmetrically (dataset without batch,
and batch without dataset, each raise); `_write_causal_attrs` refuses a novelty
curve whose row count is not the block's channel count; and
`create_hdf5_dataset_from_records_list` refuses a `transform` that disagrees
with the variant the precomputed masks were built for.

**Out-of-memory is a retry, not a refusal.** On both arms a `RuntimeError` on a
batch re-runs that batch one segment at a time; a segment that fails a second
time is dropped, recorded in `scatter_failed`, logged, and the record still
writes. A per-record `except Exception` logs the traceback and continues.

> One gap: on a resume the mode that actually drives the build is
> `cv_result["test_mode"]` (`:3559`), which is never validated — only the
> `test_mode` *argument* is (§6.4).

### 11.2. Regeneration Runbook

1. **Verify the pipeline** —
   `python hdf5_dataset/smoke_check_channel_selection.py`. Twelve registered
   checks covering the selection, HDF5 geometry, `sel_*` attrs and **four**
   guards (width mismatch fails loudly rather than as an h5py broadcast error;
   a missing coefficient dataset is rejected; a mask built for a different
   filter bank is rejected; an empty band selection is rejected readably). Needs
   no data.

   > **Two of the twelve currently fail**, and both for the same reason: they
   > monkeypatch the dead `PHASE_HARMONIC_K_STEPS` (§7.5). "keep-diagonal
   > variant is a one-constant switch" reports
   > `keep-diagonal fhr_ph: expected 94, got 66`, and "channel-count mismatch
   > fails loudly" reports `expected a ValueError, none was raised` because its
   > deliberately stale 94-wide file now comes out 66 wide, leaving no mismatch
   > to detect. Expect **10/12 passed** until the constant is either deleted or
   > restored. Neither failure indicates a problem with the dataset.

2. **The diagonal question is currently un-gateable.**
   `python hdf5_dataset/check_phase_diagonal_redundancy.py --hdf5 <shard>` still
   measures the redundancy (§7.5), but the action its verdict names — setting
   `PHASE_HARMONIC_K_STEPS = (0, 4, 6, 8)` for `fhr_ph` 94 / `up_ph` 26 — has no
   effect, because nothing reads that constant. Restoring the diagonal now means
   defining a third operator version. The tool also cannot be pointed at the
   causal bank or the integer operator, so its verdict does not describe the
   shipped build.
3. **Generate** — run `create_new_pipeline`. It logs the resolved layout
   (variant, leg alignment, **phase operator**, per-block widths, $c_y$/$c_u$,
   dropped channels, warm-up and delay ranges) before any work; confirm they
   match intent. Every number in those lines is derived from the channel plan,
   which `test_no_channel_count_is_a_literal_in_the_formatted_layout` enforces.
4. **Recompute statistics** with `calculate_dataset_stats.py`. Mandatory (§9.4).
5. **Point the model configs at the geometry you built.** The $c_y$ / $c_u$
   migration this step used to describe has landed: no config carries 87 or 101
   any more, and all three live geometries are declared explicitly (§8). What
   remains is to make sure the package you train matches the shard you built —
   109/58 two-sided, 102/51 causal legacy, 80/46 causal integer.
   - `band_partition.py` rebuilds the `fhr_ph` channel map by re-running the
     legacy selector and truncating to `n_ph_channels`. It will raise on the
     count mismatch, which is fine — but do **not** "fix" it by loosening
     `min_freq` until the count exceeds 66, because the truncation would then
     silently produce wrong band labels. Read the `sel_*` attrs instead (§7.7).

### 11.3. Building the Causal Variant

Both builders now launch from a command line **or** from an IDE's Run button with nothing typed —
edit the `RUN_ARGS` dict above each `__main__` guard and press Run. Every resolved value is logged
with where it came from, so a run's provenance is readable from its own output rather than
reconstructed from a shell history.

Step 1 runs on the **production box only**: it reads `.mat` records through `early_maestra`, which
is installed nowhere else, so `create_new_pipeline.py` cannot even be imported on a development
machine except through the stub `smoke_check_channel_selection.py` applies. Steps 2 to 4 run
anywhere the shards do.

The parser exposes 13 flags, one per `create_new_pipeline` keyword —
`--records-base-path`, `--output-base-path`, `--tlo-csv-path`, `--test-mode`,
`--verbose`, `--scatter-batch-size`, `--num-workers`, `--screening-csv-path`,
`--classification-pickle-path`, `--transform`, `--leg-alignment`, `--device`,
`--phase-operator`. **No argument is `required=True` and none carries a
non-`None` default**; `main()` applies the real defaults after the merge
(`test_mode or 'augmented'`, `verbose` True when None, `scatter_batch_size or
128`, `transform or 'two_sided'`, `leg_alignment or 'none'`,
`phase_operator or 'ratio_power_v0'`), and required-ness is enforced after the
merge by `missing_required(values, ('records_base_path', 'output_base_path',
'tlo_csv_path'))`, whose refusal names both ways to supply the value. `RUN_ARGS`
has the same 13 keys.

```bash
# 1. The shard. --phase-operator and --leg-alignment select the variant;
#    omitting either gives the LEGACY value, not the one RUN_ARGS ships.
python hdf5_dataset/new_pipeline/create_new_pipeline.py \
    --transform causal \
    --leg-alignment envelope \
    --phase-operator integer_harmonic_v1 \
    --output-base-path /data1/fetal-heart-tracing/HDF5_Datasets/new_pipeline_6h_causal \
    --classification-pickle-path /data1/.../new_pipeline_6h/classification_dataset_records.pickle \
    --records-base-path /data/deid/datafabric/fetal-heart-tracing/StudyGroup2022_v4/ \
    --tlo-csv-path /path/to/complete_labor_onset.csv \
    --device cuda:3

# 2. Its statistics, at the loader's own trim.
python hdf5_dataset/calculate_dataset_stats.py \
    --input-files <the shards> \
    --output-file output/hie_cs_causal_stats.hdf5 \
    --trim-minutes 1.0

# 3. The review directory: fidelity, leakage, budgets, and the leg-alignment diagnostics.
python -m hdf5_dataset.compare_causal_scattering \
    --shard output/hie_cs.hdf5 --output-dir output/causal_scattering

# 4. The dataset tests. Eight modules, not two.
pytest hdf5_dataset/test_causal_scattering.py hdf5_dataset/tests/ -q
```

**What those flags mean if you omit them.** The command line resolves an unset
`--leg-alignment` to `'none'` and an unset `--phase-operator` to
`'ratio_power_v0'` — so the shortest invocation builds the *legacy* causal
variant, which is **not** the build the checked-in `RUN_ARGS` and both CFS
`default.yaml` files expect. That launch dict ships
`transform='causal'`, `leg_alignment='envelope'`,
`phase_operator='integer_harmonic_v1'` (`:3843`, `:3851`, `:3861`) and an
`output_base_path` ending `..._causal_int`. Editing the dict and pressing Run in
an IDE is the supported no-command-line path; every resolved value is logged
with where it came from.

**The test suite.** `hdf5_dataset/tests/` holds six modules plus `conftest.py`,
and two more sit at package level:

| Module | Tests | Covers |
|---|---:|---|
| `tests/test_causal_pipeline.py` | 41 | the writer and the build guards; the step order of §6.4 |
| `tests/test_causal_loader.py` | 62 | the loading and statistics stack |
| `tests/test_causal_torch.py` | 52 | transform geometry, torch-vs-numpy gate, channel plan |
| `tests/test_phase_operator.py` | 16 | the two operator versions, and the timing / start-up counterexamples |
| `tests/test_preprint_numbers.py` | 3 | the preprint fidelity table against the measurement CSV |
| `tests/test_preprocessing_availability.py` | 7 | the retrospective-vs-online sanitiser audit (§3.5) |
| `test_causal_scattering.py` | 26 | the two load-bearing causality proofs, and the arm-B-reproduces-the-shard gate |
| `test_hdf5_dataset.py` | 0 | a plotting script that skips itself at collection |

`conftest.py` deliberately **raises** rather than skips when the committed
fixture `data/causal_fixture.hdf5` is missing — eight real 5280-sample segments
extracted verbatim from a production shard, 336 KB, tracked. Only three gates
may skip: `requires_shard`, `requires_measurements` and `requires_cuda`.

Two design choices in `test_causal_pipeline.py` are worth knowing before adding
to it: the two-sided build is pinned against an **in-process recomputation**
rather than a stored hash (a hash needs a committed baseline, and its classic
failure is pasting in the new number), and root attributes are checked
**additively** against a documented set rather than by whole-file equality — so
adding an attribute is not a test failure, while changing a coefficient is. A
mutation test proves the comparison can actually fail.

`test_phase_operator.py` additionally pins two things this document relies on:
that $\kappa = 0.875$ is a **convention** and not a universal delay (a slow
narrowband modulation through the real wavelet and low-pass arrives 49.5 s later
than $\kappa$ predicts), and that the 95 % warm-up is an initialisation policy
rather than independence from the assumed prehistory (§7.2).

**The review directory** (step 3) contains eight top-level figures
(`figures/01_filters_time_domain.pdf` … `08_survivorship.pdf`), per-block
galleries `figures/09_gallery_<block>.pdf`, per-sample panels under
`figures/samples/`, `per_channel.csv`, `summary.json` and a generated
`REPORT.md`. `summary.json` carries `stored_widths`, `argument_sources`, a
`validation` block reported before anything else, a `headline` block, the
leakage block and the survivorship rows. `per_channel.csv` has one row per
(block, channel) with 23 columns — reach, delay, bandwidths, gains, warm-up,
the four correlation columns, and the seven leg-alignment columns, which are
`nan` on the scattering blocks by construction. It describes the **banks, not a
shard**, so it is the same width under either variant and includes the seven
channels a causal build drops.

`--trim-minutes` **must match the loader's**. Statistics are accumulated over the trimmed window
only, so a mismatch normalises with constants drawn from a window nothing serves — and it produces
a `warnings.warn` at load time and nothing else.

**Resume from the two-sided run's fold pickle.** This is the supported
invocation and the reason the pipeline records `source_pickle_path` and
`source_guid_digest` on every shard: reusing that pickle guarantees the causal
dataset contains exactly the same GUIDs, folds, partitions and segments as the
two-sided one, which is what makes a later comparison valid segment for
segment. Two shards built from one pickle carry the same digest whatever their
variant, so the claim is checkable after the fact rather than asserted. A fresh
run records the *absence* of a pickle explicitly rather than as an empty
string.

**`--leg-alignment envelope` is a further variant, not a replacement.** It changes no width, no
warm-up and no stored delay — only the values inside the two phase blocks (§7.3) — so it must go to
its own `--output-base-path` and needs its own statistics file. The two are told apart by the root
attribute `causal_leg_alignment` alone, and the loader refuses a file list that mixes them or a
statistics file built at the other setting, naming both.

**`--phase-operator integer_harmonic_v1` is likewise a separate dataset.** Unlike the alignment it
*does* change widths — `fhr_ph` 66 → 44 and `up_ph` 15 → 10, hence $c_y$ 102 → 80 and $c_u$ 51 → 46
— so it needs its own `--output-base-path` and its own statistics file, and the loader refuses a
mixed-operator file list or a statistics file built under the other operator, checking the operator
**before** the width so the message names the real cause. The two axes are orthogonal and compose
freely: four causal builds are reachable, and each is a distinct dataset.

The build logs its resolved layout before any work — variant, leg alignment, per-block widths,
$c_y$/$c_u$, the dropped channels and the warm-up and delay ranges — and
`smoke_check_channel_selection.py` prints those same lines for both variants
with no data at all.

Then recompute statistics against the causal shards (step 2 above); a causal
dataset paired with two-sided constants is refused at load, but only if the
stats file exists to be paired.

**Nothing is migrated.** The two-sided dataset is untouched and remains in use;
rollback is deleting a directory. Existing files are read exactly as they are
today under the legacy rule of §8.2, and a causal shard written before
`causal_leg_alignment` existed reads as `'none'`, which is what it is.

Downstream work this does **not** do: `c_y`/`c_u` in the model configs
(109 → 102, 58 → 51), the $L_{95}$ reach guard in `channel_reach.py` — which is
meaningless on causal data, where future energy is exactly zero and the
quantile returns round-off — and any consumer of per-channel staleness or
per-channel novelty, which must read `causal_delay_s` and `causal_novelty_curve`
(§8.1).
---

## 12. Quick Reference

| Property | Value |
|---|---|
| Sampling rate $f_s$ | 4 Hz |
| Raw segment length | 5280 samples = 1320 s = 22 min |
| Stride between segments | 4800 samples = 1200 s = 20 min |
| Overlap | 480 samples = 120 s = 2 min |
| Decimation factor $T$ | 16 (one step = 4 s) |
| Stored sequence length | 330 steps |
| Trim (`trim_minutes=1.0`) | 240 raw samples / 15 decimated steps each end |
| **Model input window** | **300 steps = 20 min (raw 4800)** — trimmed by the loader, not the model |
| Scattering geometry | $J = 11$, $Q = 4$, $T = 16$, `max_order=1` |
| Wavelet filters | 42 first-order (+1 order-0 channel) |
| Phase pairs before masking | 903 |
| `fhr_st` / `up_st` channels | 43 each (unmasked); **36 causal** |
| `fhr_ph` channels | 66 (`ratio_power_v0`, $k \in \{4,6,8\}$); **44** (`integer_harmonic_v1`, $k \in \{4,8\}$) — 0.008–1.0 Hz |
| `fhr_up_ph` channels | 79 — two-band cross-phase; **absent causal**; binding defect §7.4 |
| `up_ph` channels | 15 (`ratio_power_v0`); **10** (`integer_harmonic_v1`) — 0.008–0.05 Hz |
| Total feature channels | 246 two-sided; 153 causal legacy; **126 causal integer** |
| Target encoder width $c_y$ | 109 two-sided; 102 causal legacy; **80 causal integer** |
| Source encoder width $c_u$ | 58 two-sided; 51 causal legacy; **46 causal integer** |
| `transform` | `two_sided` \| `causal`; absent = legacy two-sided |
| `causal_leg_alignment` | `none` \| `envelope`; absent = `none`; `RUN_ARGS` ships `envelope` |
| `causal_phase_operator` | `ratio_power_v0` \| `integer_harmonic_v1`; absent = `ratio_power_v0`; `RUN_ARGS` ships the integer operator |
| Causal gammatone order / taps / warm-up quantile | 4 / 32768 / 0.95 |
| Causal warm-up range (untrimmed steps) | `fhr_st` 5–293, `fhr_ph` 8–149, `up_st` 5–293, `up_ph` 56–149 |
| Causal group delay (uncompensated) | `fhr_st` 13.3–791.0 s, `fhr_ph` 20.5–402.2 s, `up_ph` 150.8–402.2 s |
| UP time shift | $-20$ s ($-80$ samples at 4 Hz; applied at the record's native rate) |
| Log epsilon / z-score epsilon | $10^{-6}$ / $10^{-8}$ |
| Folds | 10, named `fold_1` … `fold_10` |
| Partitions per fold | 3 (train / val / test) in **augmented** mode; 2 (train / val) in **holdout** mode, plus one shared `test/` built outside the fold loop |
| Build order | pre-training shards first (step 4), then the k-fold shards (step 5) |
| Subgroups per partition | up to 8 |
| Split ratios | 80/10/10 (augmented), 90/10 + fixed test (holdout) |
| Segment quality gate | mean weight $\ge 0.90$ |
| Screening window | last 6.37 h before delivery ($-22920$ s) |
| Extraction window | last **12.4 h** before delivery ($-44640$ s) |
| Max segments per GUID | 37 stride positions, up to 38 starts depending on the equalisation grid |
| Eligibility | $\ge 2$ h valid (unhealthy), $\ge 3$ h (healthy), judged on the 6.37 h window |
| Pre-training reserve | 90 % of BG-healthy GUIDs |
| Seeds | 42 throughout |
| Numeric / label / string dtype | float32 / uint8 / UTF-8 |
| HDF5 chunking | 32 samples, LZF |
| Recommended `cache_size` | 2000 (segment), 128 (GUID) |
| Recommended DataLoader `num_workers` | 0 (debug), 4 (training) — unrelated to the pipeline's own `--num-workers`, which affects prescreening only and defaults to $\min(\texttt{cpu\_count}, 8)$ |
