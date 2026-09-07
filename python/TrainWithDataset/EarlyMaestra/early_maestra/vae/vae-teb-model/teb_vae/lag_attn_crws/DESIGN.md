# `lag_attn_crws` — the as-built design record

The causal-input raw-target lag-attention VAE-TEB: what it is, what it consumes, what it returns,
what it optimises, what it measures, which of its members are another package's objects reached by
reference, and every place the built package differs from the design it was built from.

Companion documents, none of them restated here: `teb_vae/lag_attn_rws/DESIGN.md` records the raw
target, the objective and the architecture every cell of the raw-target row shares, and
`teb_vae/lag_attn_rws/model_explained.md` the latent factorisation; `teb_vae/lag_attn_cfs/DESIGN.md`
records the causal input machinery this package composes — the warm-up budget, the anchor tiling,
the derived phase, the source compromise and the availability-clock control — and the dataset facts
behind it; `teb_vae/lag_attn_transformer_rws/DESIGN.md` records the conv-Transformer encoder the
twin cell of this row composes it over; `hdf5_dataset/dataset_explained_research.md` section 8.1
defines the causal dataset variant and `hdf5_dataset/CAUSAL_SCATTERING_PHASE_HARMONIC_MATH.md`
carries the warm-up and group-delay mathematics. `RESULTS.md` in this directory carries the
pre-registered criteria and, once there are runs, the measurements.

**What this document is for.** Reading it should leave a reader able to say what a reported number
of this model means and what it does not — and, because almost every member of this package is
another package's object, *whose* code a given behaviour is. A model assembled by reference rather
than by copy has one failure mode a copied one does not: a change two packages away can move it with
nothing in this directory changing, and this record is where the reader finds out which members
those are.

---

## 1. What the model is

The seventh cell of an encoder-by-target grid, and the first of the one row in which **neither side
of the objective contains its own future**:

```
                          conv-LSTM encoders          conv-Transformer encoders
  raw FHR target          lag_attn_rws                lag_attn_transformer_rws
  two-sided feature       lag_attn_fs                 lag_attn_transformer_fs
  causal feature          lag_attn_cfs                lag_attn_transformer_cfs
  causal in / raw out     lag_attn_crws  <- this      lag_attn_transformer_crws
```

At each admitted 4-second anchor $t$ the model forecasts the next **two minutes of raw FHR** —
$H \cdot R = 30 \times 16 = 480$ raw samples — twice: once from a target-only latent and once from a
source-conditioned one, through one shared decoder invoked twice. The gap between the two forecasts,
and the KL between the two latents resolved across lags, are the coupling readout. The inputs are the
**causal** dataset variant's one-sided scattering and phase-harmonic coefficients, so an input at
decimated step $t$ is a function of $\{x(s) : s \le t\}$ and of nothing else.

Structurally this is `lag_attn_rws` told where its inputs begin. The two-sided cells read
coefficients that at step $t$ average raw samples on **both** sides of $t$ — up to $965$ s into
$t$'s own future on the slowest channel — so their forecast of what follows $t$ is computed from
inputs that have already seen it, and `causal_reach_budget_s` mitigates that with a per-channel
*shift* rather than removing it. `lag_attn_cfs` removes it at source but forecasts a **causal
feature**, which reintroduces a different caveat: a stored causal coefficient still lags by its own
composed group delay, so any lag-resolved reading of that cell is over stored-coefficient time on
both sides of the attention. Pairing causal inputs with the raw target removes both — the target is
the signal itself, with no warm-up, no group delay and no channel selection — and it makes
`lag_attn_rws` the direct control: same target, same objective, same horizon family, differing only
in whether the inputs contain the answer.

**What this cell can claim that no other cell can**, and the only thing it claims: its forecast
claim and its lag claim are simultaneously exact on the target side. The forecast is of raw samples
that no input has seen; the anchor is at the instant it is at. What is *not* claimed is a physical
lag from an unaligned run: an input coefficient still carries its own composed group delay, and only
the shift onto a common clock (§3) collapses that into the single constant the lead-time identity of
§11 needs. This cell's alignment reference is chosen so that identity can reach the $20$ to $120$ s
contraction-to-deceleration band at all, which at the causal-feature cells' reference it cannot; §14
records what still stands between the identity and a number.

**It is an experiment, not a remedy.** `lag_attn_rws/PRIOR_SCALE_AND_PREDICTIVE_GAP.md` §5
establishes that the held-out predictive gain is negative because the source pathway does not
generalise — a failure in the source encoder, the lag attention and the posterior fusion, none of
which an input-representation change touches. This model is expected to reproduce it. **The sign of
`pred_gap` is a criterion nowhere in this document.**

The whole model is

```python
class SeqVaeLagAttnCrws(CausalRawInputs, SeqVaeLagAttnRws): ...
```

with **a constructor and nothing else** — `vars(SeqVaeLagAttnCrws)` carries `__init__` and no other
callable — linearising as
`SeqVaeLagAttnCrws -> CausalRawInputs -> CausalWarmupInputs -> SeqVaeLagAttnRws -> Module -> object`.
§6 records why the constructor is the one exception, and why there is one mixin here where the
causal-feature cells have two.

At the shipped configuration the model holds **4,589,907 parameters**, against **4,218,476** for the
encoder-axis comparison `lag_attn_transformer_crws`. With every architecture switch of §17 at its
off-state it holds **5,081,146** against that cell's **4,989,804**, which is bitwise the pair that
shipped before this revision and is the row on which the input-representation comparison against
`lag_attn_rws` (**5,094,458**) is still readable. §13 carries the arithmetic, decomposes every delta, and states the unaligned arm the
channel alignment (§3) moved this cell off — which at *this* row's reference is the arm with **more**
parameters, because the alignment here drops sixty target-stream channels and thirty-four source
ones rather than four.

**Four mechanisms of this cell are configuration rather than architecture**, each gated by a key
whose off-state reproduces the pre-revision model bitwise: where the lag attention's keys and values
come from, whether the prior is told the source's arrival clock, whether the reconstruction weights
the horizon axis, and how the lag bias is seeded. §17 is the inventory. **Two of the six the
feature-target cells carry are deliberately absent from this row** — the decoder's persistence
residual and the second alignment reference — and §14 records why each.

## 2. Input contract

Read from the **causal** HDF5 shards through `train/data_module.py::GraphDataModule`, at
`trim_minutes: 1.0`. The three tensors the encoders receive are the causal-feature cells' exactly —
same fields, same widths, same warm-up boundaries — and the target is the raw-signal cells' exactly.
Nothing about the input contract is this cell's own except the pairing.

| Field | Shape | Role |
| --- | --- | --- |
| `fhr` | $(B, 4800)$ | **the reconstruction target**, z-scored; $T \cdot R = 300 \times 16$ raw samples after the loader's symmetric trim |
| `fhr_st`, `fhr_ph` | $(B, 300, 36)$, $(B, 300, 66)$ | the target-feature **input** stream, concatenated to $(B, 300, 102)$ — an input here, not a target |
| `up_st`, `up_ph` | $(B, 300, 36)$, $(B, 300, 15)$ | the source stream, concatenated to $(B, 300, 51)$ |
| `weight` | $(B, 300)$ | per-step validity on the decimated grid; every mask the objective builds reads it |
| `up` | $(B, 4800)$ | **not read by the model** — the raw context row of the diagnostic page only |
| `guid`, `epoch` | — | figure titles, run provenance, **and the tile phase** (§4) |

$c_y = 36 + 66 = 102$ and $c_u = 36 + 15 = 51$, against the two-sided $109$ and $58$: seven
scattering channels per block were dropped at write time because their one-sided warm-up outruns the
stored segment. The causal transform does not touch the raw arrays, so `fhr` is stored at
$(N, 5280)$ and trims to exactly the $4800$ samples the raw geometry expects.

**`fhr` must be in `load_fields` and in `normalize_fields`, and `weight` in `load_fields`.** The
first two are the shared entry point's own guard, called from this driver's pre-flight against
`LagAttnCrwsTrainer.TARGET_FIELDS = ("fhr",)` — inherited from the raw-signal driver and
deliberately **not** re-pointed, since re-pointing it is exactly the edit the causal-feature driver
makes and the edit this one must not. An unnormalised raw target makes the Gaussian NLL meaningless
against a unit-scale variance model with the loader raising nothing. The third is this cell's own
guard and no shared one covers it: `weight` is not a field the target is *built* from, so the shared
guard has no reason to ask about it, but gaps in the raw trace are stored as $0$ bpm — roughly
$-11\sigma$ once z-scored, not a detectable sentinel — so without it the run scores pad as signal at
full weight.

**`guid` and `epoch` must be in `load_fields`.** The tile phase is keyed on the pair, and
`load_fields` is honoured literally with no forced additions; without them every segment of every
recording is decoded at one tile grid forever, with $A_{\max}$ a geometry constant either way, so no
shape, no count and no metric would differ. `trainer.preflight` refuses it.

**`fhr_up_ph` is absent from the variant and must stay absent from every config.** A coefficient
mixing both signals would put the source's own signal into the target-only branch's inputs and
destroy the target-only / source-conditioned separation the coupling readout rests on. In
`load_fields` the loader would raise, but only after every rank had initialised; in
`normalize_fields` it is silently ignored and reads as though the block were being handled.

**The stored warm-up region is real values, not zeros and not NaN**, normalised with constants that
excluded exactly that region. `lag_attn_cfs/DESIGN.md` §2 records the three ways that is misread;
what matters here is that the same availability adapter that masks it for that cell masks it for
this one, because the inputs are the same tensors.

## 3. Geometry: the floor is a policy, not a constraint

`TrimmedRawGeometry` is reused unchanged:

$$T = 300, \qquad H = 30, \qquad R = 16, \qquad T_{\mathrm{valid}} = T - H = 270, \qquad F = 134,
\qquad S = H = 30.$$

Anchors live in $[F, T_{\mathrm{valid}})$, which is $136$ of them; the tiling admits
$A_{\max} = \lceil (T_{\mathrm{valid}} - F)/S \rceil = 5$ per sample at phase $\varphi \le 15$ and
$4$ otherwise, mean exactly $136/30 = 4.533$.

**None of it is forced by the target.** In the causal-feature cells the pairing of the anchor floor
with the resolved warm-up budget, $F \ge B - 1$, is a **validity requirement**: the target there is
a stored coefficient, honest only from its own $W'_c$, so a floor one step too low scores assumed
pre-recording history as signal with every shape correct. A raw sample is honest at every step. The
objective here would be sound at *any* floor, and every value above is a deliberate choice to hold
the geometry fixed against `lag_attn_cfs` so that the two cells differ in exactly one variable — what
the decoder emits.

### The alignment reference is this row's own, and it is not the feature cells'

The alignment *machinery* is `lag_attn_cfs`'s, reached by import and not restated here: every
surviving channel of **both** input streams is gathered with a shift

$$d_c = \operatorname{round}\!\Bigl(\kappa\,\frac{\tau_{\mathrm{ref}} - \tau_c}{\Delta}\Bigr) \ge 0,
\qquad \kappa \;=\; 1 - \frac{1}{2\gamma} \;=\; 0.875, \qquad \Delta = 4\ \mathrm{s},$$

so that a channel vector at one encoder step describes one physical instant rather than the thirteen
minutes its composed group delays span. `lag_attn_cfs/DESIGN.md` §3 derives $\kappa$ — the ratio
between the impulse response's energy centroid, which is the delay a channel's content actually sits
at, and the envelope mean $\tau_g$ that `causal_delay_s` ships — and carries the measurement behind
it. What is **this row's own** is the reference: `causal_align_reference: 42.21`, against
`target_max` ($402.1604$ s) in all four feature-target cells.

**The reason is that a raw target has $\tau^y \equiv 0$, so the reference does not cancel.** A peak
at lag $\ell$, horizon element $h$, is a physical lead of

$$\tau_{\mathrm{phys}}(\ell, h) \;=\; \Delta\,(\ell + 1 + h) \;+\; \kappa\,\tau_{\mathrm{ref}}$$

(§11), on the canonical stored timeline — the dataset builder's UP shift is part of the signal and
no term undoes it (an earlier revision subtracted $\tau_{\mathrm{pre}} = 20$ s here; superseded
2026-09-05) — which is **minimised at $\ell = h = 0$ and grows with the lag**. At $402.1604$ s the
smallest lead the attention can express is $355.9$ s, so the $20$ to $120$ s contraction-to-deceleration band
this model exists to find is unreachable at *every* lag index — and raising `max_lag` cannot help,
because a longer search only makes the lead longer. In the feature-target cells the same reference
appears on both sides of the identity and cancels, which is why they keep `target_max` and reach the
band at $\ell + h \in [9, 34]$.

At $42.21$ s the picture inverts. The key is a **name for a channel** rather than a number: the
resolver snaps it to the shard's own float32 $42.206562$ s — the filter at $0.1109$ Hz — and refuses
a literal further than $\Delta/2$ from any kept channel, so a rebuilt bank moves the reference
instead of leaving it resolving against a transform the shards no longer have. Because the shift
cancels each channel's *realised* delay, the aligned stream's common **effective** delay is
$\kappa\,\tau_{\mathrm{ref}} = 36.93$ s and not $42.21$ s, and that effective value is the constant
the identity above takes. The reachable lead is then $[40.9, 516.9]$ s: $\ell + h = 0$ reads
$40.9$ s and $\ell + h = 20$ reads $120.9$ s, so $79$ of the band's $100$ s sit inside the lag
axis where none of it did (the first $20.9$ s of the band lie below lag $0$).

### What it costs: two rules, cutting in sequence

**The warm-up budget and the alignment are different rules with different reasons**, and a reader
who took either for the other would misread every count here and every warmth column of §10.

Measured against the committed causal fixture, the **budget** keeps **98 of 102** target-stream
input channels — `fhr_st` $32/36$ and `fhr_ph` $66/66$ — dropping the four `fhr_st` channels at
$W' \in \{162, 194, 233, 278\}$, whose one-sided wait outruns what a segment can spend. A threshold
of $151$ keeps the identical channels; the `fhr_ph` block tops out at $134$ and the `fhr_st`
staircase has a channel at exactly $134$ with the next at $162$, so that boundary lands on the
$\approx 0.008$ Hz frequency edge rather than on an arbitrary cut. **All $51$ source channels survive
the budget** and its resolved source keep-index is the identity.

The **alignment** then drops every channel whose own composed delay *exceeds* the reference, because
reaching it would need a negative shift — reading that channel from a later stored step, i.e. from
its own future. That is a correctness requirement rather than a policy, and it is what distinguishes
these drops from the budget's. At $42.2066$ s it takes a further $60$ target-stream channels and $34$
source ones, leaving **38 of 102** on the target-feature input stream — `fhr_st` $17/36$ and
`fhr_ph` $21/66$ — and leaving $17$ of $51$ on the source, `up_st` $17/36$ and **`up_ph` $0/15$**.
Neither of the budget's own boundaries decides anything at this reference: the alignment cuts far
below both.

**The whole `up_ph` block is gone, and that is what the lag axis costs.** Its fastest channel sits at
$150.79$ s, while the band's lower edge needs $\kappa\,\tau_{\mathrm{ref}} \lesssim 36$ s — so any
reference that puts the coupling band inside the lag axis is far below every `up_ph` channel and
keeps none of them. The same arithmetic is why the resolver refuses `use_up_st: false` here, naming
the $150.7859$ s fastest channel: without `up_st` the source stream would have no channels at all.
Contraction *morphology* leaves the source stream entirely and what remains is `up_st`, the
contraction envelope. §8 is the record, and §10 records what it does to `source_lag_warmth_frac_ph`.

### The floor, and why nothing binds it any more

**The pairing is retained anyway, re-justified as the declared input-warmth policy**: every kept
input channel of **both streams** is warm **by the first forecast step**. Every part of that wording
is exact and no obvious paraphrase is.

*By the first forecast step* rather than at the anchor: a channel is honest from $W'_c$, and the
earliest target step an anchor reads is $t + 1$, so this half asks only for $F \ge B - 1$ with
$B = \max_{c \in \mathrm{kept}} W'_c$.

*Both streams*, because the channel alignment shifts both and the constructor checks both. A channel
gathered with a shift of $d_c$ steps is honest only from $W'_c + d_c$, so the second half asks for
$F \ge \max_c(W'_c + d_c)$. `CausalRawInputs._check_anchor_floor` refuses a floor below either
requirement with a message that says which stream and which reading it enforces, and the driver's
pre-flight delegates to that same function so the two cannot come to disagree.

**At this reference both halves have gone slack, and that is the largest single change to this
document.** The survivors are the *fast* channels, so the slowest kept target-stream channel waits
one step: $B = 1$. The shifts span $d_c \in [0, 6]$ and $\max_c(W'_c + d_c) = 6$ on **both** streams.
The requirement is therefore $\max(B - 1, 6) = 6$ steps, and the shipped `warmup_period: 134` clears
it more than twenty times over. Where the feature-target cells' floor is *decided* by the second half
— it is what puts them at $134$ — **nothing decides this one**: the floor has become a retained
anchor-cost policy rather than a number the data computes. It is left at $134$ deliberately all the
same. Lowering it toward $7$ would roughly double the anchors per sample, which is a training-cost
change to be taken with a run rather than with a key, and holding it is what keeps this cell and
`lag_attn_cfs` differing in exactly one variable.

Before the alignment only the target stream was fed to the check: the source was ungated, kept
channels waiting $162$, $194$, $233$ and $278$ steps by design, and a policy read across it would
have pushed $F$ to $277$ and cost about $143$ of the $136$ anchors. The alignment retires that
asymmetry by dropping the channels slower than the reference rather than by raising the floor to
cover them, and at this reference it retires it completely: the slowest source channel the model
still sees is honest at step $6$.

**What the policy costs is anchors, and it is the largest single lever this design leaves on the
table.** A raw target could begin at the model's own $30$-step warm-up: $240$ anchors against $136$,
a $76\%$ increase in supervision. Nothing but the policy prevents it, and the choice is about what a
run *claims* rather than about what the data supports — which is why it is the first `lean-limit:`
note of §14 and not a shipped arm. **No floor arm ships**, and the reason is no longer a refusal: at
this reference the constructor and the pre-flight admit any floor at or above $6$, so an arm would be
a decision about how much supervision to buy rather than an edit to a guard.

**Shortening the horizon buys back anchors.** $T_{\mathrm{valid}} = T - H$, so the shipped
$H = 30$ admits $136$ anchors against $151$ at $H = 15$. The horizon is one of the two levers on the
warm-up cost rather than an independent preference, which is why `sweep_horizon_15.yaml` — the arm
that gives the raw-signal sibling's $480$-sample block back up in exchange for those anchors — is an
arm and not the default.

## 4. Anchor tiling, and the anchored raw target

The tiled anchor set, the derived phase and its refusals are the causal-feature cell's, reached by
inheritance and unchanged: `lag_attn_cfs/DESIGN.md` §4 is the record. In one paragraph — the model
decodes

$$\mathcal{A}(\varphi) = \{\, F + \varphi + kS \;:\; k \ge 0,\; F + \varphi + kS < T_{\mathrm{valid}} \,\},
\qquad S = H = 15,$$

so windows partition the timeline and no raw sample is scored twice in one step; the phase is
$\mathrm{blake2b}(\texttt{guid} \Vert \lfloor \texttt{domain\_start} \rfloor \Vert
\texttt{train\_epoch} \Vert \texttt{seed}) \bmod S$, derived per segment and never drawn, so it
consumes no RNG, needs no collective and survives a resume; validation and test decode every valid
anchor at stride $1$ and phase $0$; the stride and the phase are **forward arguments** resolved from
the stage string rather than from `self.training`; and short rows repeat their last valid anchor and
mark it invalid, so `anchor_index` is $(B, A_{\max})$ with `anchor_valid` beside it and no rank can
disagree on shape.

**One thing about the padding is newly load-bearing here.** In the feature cells a padded slot
holding a *distinct legal* anchor would score a target block twice while `kl_mask`'s scatter
deduplicated it. Here it would make the raw gather below pull one raw window twice — the same
denominator divergence, on a different tensor — so `tests/test_anchors.py` asserts directly that a
padded slot's gathered window equals its row's last valid window and contributes zero to the loss.

### The anchored raw target

This is the one genuinely new piece of arithmetic in the package. `nets/raw_targets.py`'s
`build_future_index` returns the $(T_{\mathrm{valid}}, H, R)$ grid of raw sample indices and the
architecture caches it as the non-persistent `future_index` buffer; `build_future_target` gathers
that grid at *every* anchor because no raw-target cell ever tiled. Gathering it at an anchor set is
one index into the grid's first axis,

$$\mathrm{idx}[b, a, \tau, r] = \texttt{future\_index}\big[\,\mathcal{A}[b, a],\, \tau,\, r\,\big],
\qquad X^{+} = \texttt{fhr}.\texttt{gather}\big(1,\ \mathrm{idx}.\texttt{reshape}(B, -1)\big).\texttt{reshape}(B, A, H, R),$$

and `gather_anchored_future_target` in `nets/causal_raw_inputs.py` is that expression.

**`gather`, not `index_select`, and the distinction is the whole of the difference from the dense
builder.** `build_future_target` may use `index_select` because its index is the *shared* grid — the
same rows for every sample. Here the anchor set is **per sample**, because the tile phase is derived
per segment, so the index is $(B, A, H, R)$ and an `index_select` on dimension $1$ would return
$(B, B \cdot A \cdot H \cdot R)$ and fail the reshape rather than the values. Three things pin it:
at `anchors = arange(t_valid)` it equals `build_future_target` under `torch.equal`; a per-sample
anchor set with two different rows yields two different windows; and it reads the model's own cached
buffer — asserted by `data_ptr` identity — rather than rebuilding a grid that could disagree with
the one the forward decoded at. Extending `nets/raw_targets.py` with an anchor parameter was rejected
by direction: it would edit a module four shipped cells score through.

The gather bounds-checks its own anchors, and that is not redundant with the objective's check:
advanced indexing on a negative index *wraps* rather than raising, so an anchor of $-1$ would
silently gather the last legal window with every shape correct. Uniqueness among the valid entries
is deliberately **not** checked there — a padded slot is a duplicate by design — and is refused by
`forecast_mask` on the same call, where the two per-anchor denominators it protects are built.

## 5. Loss, and what its nats are summed over

$$\mathcal{L} = \lambda_{\mathrm{full}} D_1 + \lambda_{\mathrm{base}} D_0
+ \beta(e)\,\mathrm{KL}_{\mathrm{train}} + \beta_p\,R_p
+ \lambda_{\mathrm{ms}} \mathcal{L}_{\mathrm{ms}}
+ \lambda_{\mathrm{deriv}} \mathcal{L}_{\mathrm{deriv}}
+ \lambda_{\mathrm{boundary}} \mathcal{L}_{\mathrm{boundary}}$$

in nats per anchor, computed by `lag_attn_rws/nets/losses.py` — **the same code, not a copy of it**.
`CausalRawInputs.compute_loss` is the architecture's own adapter with one line changed: the raw
window is gathered at `anchor_index` (§4) rather than at every anchor, and everything else — every
term, every reduction, `block_width = geometry.r`, the coverage floor, the log-variance clamp — is
delegated to the shared objective. The **mask** side needs nothing: the shared objective reads
`anchor_index` and `anchor_valid` off the forward dict itself and threads them into `forecast_mask`
and `kl_mask`, so the reconstruction support and the KL support are the decoded set by construction
rather than by agreement.

**A forward dict carrying no anchor set falls back to the dense builder, and that is what makes a
stripped anchor set a shape refusal rather than a mis-scored batch**: the dense target carries
$T_{\mathrm{valid}} = 270$ anchors against a forecast carrying $A_{\max}$, and the objective raises.
`tests/test_objective.py` asserts the model at `anchor_stride: 1` equals the shared objective given
an explicit dense anchor index of $[F, T_{\mathrm{valid}})$ key by key under `torch.equal`, and that
stripping the anchor keys raises.

**`block_width` is `geometry.r` $= 16$**, at every anchor set, because a horizon token still emits
$R$ raw samples whatever the anchor set is. It feeds only the four log-variance diagnostics and not
the loss, and it is the raw grid's width here where it is $C_{\mathrm{keep}}$ in the feature cells.

**Two of the three shape weights transfer from the raw-signal sibling and one does not.**
`lambda_ms` and `lambda_deriv` ship at the sibling's $0.1$: they price the envelope and slope of a
raw *waveform*, this block is raw samples on the same grid, and the coarsest pooling rate of $16$
still divides the halved block $15$ times — so zeroing them, as the causal-feature cell must, would
be a second confound here. `lambda_boundary` is refused at any non-zero value, unconditionally: the
boundary term is a slicing identity over **adjacent** anchors, and this cell always decodes a set
whose entries are $S$ apart, so at any weight it would join two windows a whole horizon apart and
score the gap between them as an error. The shared objective raises on the combination; the driver's
pre-flight moves the failure to before a run directory exists.

### The reconstruction is weighted per horizon step

`horizon_weight_halflife_steps` resolves a geometric decay in $\tau$,

$$w_\tau \;\propto\; 2^{-\tau/\lambda}, \qquad \sum_\tau w_\tau = H, \qquad \lambda = 15.0
\text{ shipped},$$

registered as a non-persistent buffer and threaded through the objective **exactly as
`channel_weight` is** in the feature-target cells — `raw_sample_score` ->
`masked_raw_block_per_anchor` -> `masked_raw_likelihood` -> `compute_loss` — applied on the horizon
axis. At $\lambda = 15.0$ and $H = 30$ the resolved vector runs $1.8063$ at $\tau = 0$ to $0.4729$ at
$\tau = 29$, a $3.8\times$ spread. This row carries **no channel weight** — its block's last axis is
$R = 16$ raw samples of one trace rather than a set of wavelet channels — so the horizon weight is
the only weight in this objective.

**The renormalisation to $\sum_\tau w_\tau = H$ is what keeps the rest of the configuration valid.**
Without it the block would shrink by $1.81\times$ against an unmoved KL, and `gradient_clip_val`,
`additive_margin` and $\beta$'s standing against the reconstruction would all go out of date
silently. Only the *distribution* over horizon steps moves. `null` — the off-state — registers no
buffer at all and the delegation sites read `None`, so the block score is **bitwise** the one this
cell scored before the mechanism existed; a vector of the wrong length is refused by name rather
than broadcast onto another axis.

**What it costs is a unit rather than a scale**, and it is the caution `losses.py` already carries
for the feature cells' channel weight: a weighted block score is not a log-density, so $\beta = 1$
stops being the exact ELBO. This row has no evaluation pipeline (§14), so there is no unweighted
second reading of the same quantity here — which means a `pred_gap` from this cell's
`metrics_history.csv` is a weighted number and there is nothing beside it that is not.

### What the nats are, and are not, comparable to

- **Comparable to `lag_attn_transformer_crws` and to nothing else.** The reconstruction sums
  $H \cdot R = 240$ raw samples against the raw-signal cells' $480$ and the feature cells' $1470$ or
  $2340$ coefficients, and the decoded anchors per training step are about $10.1$ against a dense
  $240$; every loss-scale constant stated in nats was re-derived rather than transferred (§15). The
  twin cell of this row ships the identical geometry, so a loss *level* is comparable across the
  encoder edge and across no other.
- **The headline `pred_gap` is not comparable to `lag_attn_rws`'s as a level either.** It is the
  same quantity over the same target from the same objective, and it is what the cell exists to
  produce — but its block is half the sibling's, so what carries across the input-representation edge
  is a sign, a trajectory, the bottleneck-health columns and the ordering of arms.
  `sweep_horizon_15.yaml` restores the block and the horizon; it does not restore the anchor count.
- **Comparable across warm-up budgets within this model, unlike the feature cells.** The decoder
  emits $R$ raw samples at every budget, so the block does not move when the budget does. Two arms at
  two budgets have **mutually unloadable checkpoints** all the same — the input adapters are built at
  the surviving widths — and the class stamp cannot separate them; only the stamped keep-indices do.

## 6. One mixin, one architecture, one constructor

The causal-feature cells split their target domain into two mixins because a stored-coefficient
target changes the decoder's width, its gather and its readouts. A raw target changes none of them,
and so this cell composes **one** mixin over the architecture, and the mixin is a subclass:

```python
class CausalRawInputs(CausalWarmupInputs): ...
```

`CausalWarmupInputs` — the causal-feature cell's input half — holds seven members, and **five of them
are already target-domain-free**: `_set_causal_inputs`, `_build_adapter`, `build_lag_mask`,
`_build_anchor_index` and `forward`, whose only trace of a target is the argument *names* `y_st` and
`y_ph`. Those five are inherited untouched, and `tests/test_causal_raw_inputs.py` asserts each is
absent from `vars(CausalRawInputs)` and resolves to that mixin's function object by identity. Not one
line of the tiled forward, the warm-up adapter or the floored lag mask is copied. Exactly two members
are overridden, because exactly two are target-coupled:

- `_check_anchor_floor` — the same inequality, restated as the input-warmth policy of §3.
  Overridden rather than `_validate_causal_geometry`, which would re-copy the stride-versus-span
  refusal and its message and let the two drift.
- `_resolve_warmup_readout_constants` — resolves the two **source**-block warmth patterns and
  nothing else. The target warm fraction and the warm tertiles the causal-feature cells resolve are
  partitions of kept target channels, and this target has none.

**The width hook is the load-bearing absence.** Neither `CausalRawInputs` nor `SeqVaeLagAttnCrws`
defines `_default_decoder_out_channels`, so it resolves to the architecture's, which returns
`raw_per_step`, and the decoder is built at $16$ against a $(B, A, H, 16)$ target by construction.
Compose the causal-feature cell's `CausalFeatureForecastTarget` in by mistake and the decoder is
built at $C_{\mathrm{keep}} = 98$: `raw_sample_score` then computes $(\text{target} - \mu)^2$ on
shapes that do not broadcast, three frames below the decision that caused it.
`tests/test_construct.py` builds that wrong composition and reads `mean_head.out_features == 98`,
so the reason the feature mixin is excluded is a passing test rather than a comment.

**The order of the bases is load-bearing.** The mixin comes first, which is what makes the tiled
`forward` win method resolution over the architecture's dense one, `_build_adapter` build each
stream's availability terms from the resolved warm-up **plus** each channel's alignment shift,
$W'_c + d_c$, rather than from the gate's shift vector alone (which the architecture's own version
reads, and which is all zeros only on an unaligned arm), and
the anchored `compute_loss` win over the dense raw one. Reversed, the model would decode the dense
range, return no anchor set at all, and score a $(B, T_{\mathrm{valid}}, H, R)$ target against it.
The linearisation is pinned as a list of class names.

**The constructor is the one member of the class, and only because of the driver.**
`trainer._build_model_kwargs` builds a run's kwargs by sweeping
`inspect.signature(MODEL_CLS.__init__)`, so a `**kwargs` signature would forward four keys and
silently build an all-defaults model. The signature is the architecture's written out in full, with
`target_delays` and `source_delays` **removed** and `target_warmup_steps`, `source_warmup_steps`,
`anchor_stride` and `lag_floor` in their place; the forwarded set is captured from `locals()` minus
the mixin's four keywords rather than written out a second time, so a keyword added to the
architecture reaches it rather than being forwarded at its default with nothing raising. Removing the
two delay keywords is the point: a warm-up is a leading *mask* and `ChannelDelay` is a *shift*
($\mathrm{out}[t, c] = x[t - \delta_c, c]$), so a warm-up routed under a delay name would train a
different model with every shape intact.

**One limit of the width absence, stated because the twin cell does not share it.**
`decoder_out_channels` is still a keyword of this signature — it is the architecture's, and the
signature is written out in full — and the driver forwards `model_config.VAE_model` keys onto the
constructor by name. So `decoder_out_channels: 98` in a config really does build the decoder at $98$
against a $(B, A, H, 16)$ target. It fails loudly, inside `raw_sample_score` on the first batch, but
at the batch rather than at the config. `SeqVaeLagAttnTrfCrws` cannot be misconfigured that way at
all, because its architecture declares no such keyword; `lag_attn_rws` has the identical exposure,
which is why no guard was added here alone.

### The binding record

Every member this package needs from a sibling and does not change is **bound by reference in a
class body**, or imported, rather than copied. A bound member is read once at class creation, the
owning class never learns it has a second consumer, and there is no second definition to drift — which
is what makes "no edit to any existing package" a structural property rather than a tested one. Two
of them are `staticmethod`s and **must be re-wrapped when bound**: `Owner.some_staticmethod` returns
the plain function, the descriptor having already resolved, and a plain function assigned in a class
body becomes an *instance* method that receives `self` as its first argument. That fails three frames
from the binding, which is why the pins below check each member is callable through `self` at the
arity its owner declares and not merely `is` the same object.

| Bound member | Owner | Pinned by |
| --- | --- | --- |
| `SOURCE_BLOCK_SPLIT`, `TARGET_BLOCK_SPLIT` | `CausalFeatureForecastTarget` | `tests/test_causal_raw_inputs.py` |
| `_resolve_block_warm_steps` — re-wrapped as `staticmethod` | `CausalFeatureForecastTarget` | `tests/test_causal_raw_inputs.py` |
| `_anchors_per_sample`, `_source_lag_warmth` | `CausalFeatureForecastTarget` | `tests/test_causal_raw_inputs.py`, `tests/test_metrics.py` |
| `anchor_phase`, `_phase_field` — re-wrapped as `staticmethod` —, `resolve_anchor_geometry`, `_build_forward_inputs` | `SeqVaeLagAttnCfsTask` | `tests/test_task.py` |
| `_mu_gap_rms`, `_added_metrics` | `SeqVaeLagAttnCfsTask` | `tests/test_task.py` |
| `input_stream_panels` — a `property`, which the binding carries as the descriptor —, `input_budget_figure` | `SeqVaeLagAttnCfsTask` | `tests/test_task.py`, `tests/test_sample_page.py` |
| `WARMUP_MODEL_KWARGS`, `warmup_model_kwargs` | `lag_attn_cfs/model_kwargs.py` | `tests/test_warmup_budget.py` |
| `resolve_warmup_budget`, `WarmupBudget` | `lag_attn_cfs/causal_warmup.py` | `tests/test_warmup_budget.py`, `tests/test_preflight.py` |
| `_tiling_anchors`, `_draw_anchor_overlay` | `lag_attn_cfs/sample_page.py` | `tests/test_sample_page.py` |
| `_horizon_receptive_field` | `lag_attn_cfs/trainer.py` | `tests/test_trainer.py` |
| `_check_raw_target_normalized`, the shared `main` | `lag_attn_rws/trainer.py` | `tests/test_preflight.py`, `tests/test_trainer.py` |

`TARGET_BLOCK_SPLIT` is still needed although there is no target block being forecast: the target
feature block is an **input** here, and the page's stream panels split it. Two task members are
**not** bound and are written out at three lines each — `__init__` and `compute_loss_and_metrics` —
because both call zero-argument `super()`, which closes over the class that *defines* it: bound onto
a class outside that hierarchy it resolves `super(SeqVaeLagAttnCfsTask, self)` against an instance
that is not one and raises `TypeError` on the first step of the first run.

## 7. The warm-up is a mask, and it lives inside the availability adapter

The stored coefficients inside $[0, W'_c)$ are zeroed after normalisation and the fact is carried by
the availability channel, and both happen inside `AvailabilityInputAdapter`:

$$m_{t,c} = \mathbb{1}[t \ge W'_c], \qquad
e_t = W_x (x_t \odot m_t) + W_m (m_t - \mathbf{1}) + \mathbb{1}\Big[\textstyle\sum_c m_{t,c} = 0\Big] e_{\mathrm{start}}.$$

`lag_attn_cfs/DESIGN.md` §7 is the record; the model reaches it through the inherited
`_build_adapter`, which announces a channel at $W'_c + d_c$ — its resolved warm-up **plus** its
gate's alignment shift — rather than at either alone, because a gathered-and-delayed channel is
honest only once both have passed. Two facts are restated because §13 rests on them.

**Both streams build a start embedding, and it covers exactly one step.**
$e_{\mathrm{start}}$ exists only when *every* channel of a stream waits at least one step. Both
streams have a channel at $W' = 0$ — `fhr_ph` on the target-feature stream and `up_st` on the source
— so **before the alignment** both indicators were identically zero and neither vector was
constructed. The shift changes that: the adapter is built against $W'_c + d_c$, whose minimum is
$1$ on both streams at the shipped reference, so both vectors exist and each indicator is true on
step $0$ alone. The two-sided reach guard builds both for the same structural reason — every channel
it keeps is shifted by at least its own delay — which is why the start-embedding term of the
input-representation delta in §13 is $0$ rather than $-256$.

**`use_up_st: false` together with a warm-up budget is still refused, and the reason has hardened.**
The driver's refusal was written for a construction-time change: dropping the first stored source
block leaves the unaligned source's minimum warm-up at $41$ and flips the start embedding into
existence — a parameter reached only by the leading steps of a segment, which under
`find_unused_parameters=False` is a DDP hazard. At this cell's reference the same configuration
fails harder and earlier: `up_ph`'s fastest channel is $150.79$ s, far above the $42.2066$ s
reference, so without `up_st` the alignment would drop **every** source channel and the resolver
refuses by name (§3).

**No gradient flows from inside the warm-up.** The masked positions are exactly zero before the
input linear, and `tests/test_construct.py` asserts it rather than inferring it from the mask.

## 8. The source, and the constraint this reference dissolves

Lag attention searches $L = 91$ lags, so at anchor $F = 134$ it reads source states back to step
$44$. In the causal-feature cells that is the design's sorest point: `up_ph`'s band is $0.008$ to
$0.05$ Hz, its slowest channels wait up to $278$ steps, and at step $44$ most of the block is still
inside its own warm-up. Neither gating nor shortening fixes it there — a source budget that made
every reachable lag warm would cost almost the whole `up_ph` block, and a shorter lag search would
not cover the $20$ to $120$ s contraction-to-deceleration delay the model exists to find.
`lag_attn_cfs/DESIGN.md` §8 carries that arithmetic.

**This cell pays that bill in full and up front, and gets a warm source in exchange.** The rule that
gates the stream is still not the budget: **the warm-up budget gates no source channel at all** —
all $51$ survive it and its resolved keep-index is the identity — while the alignment removes the
$34$ channels whose composed delay exceeds the $42.2066$ s reference, for the unrelated reason that
they could reach it only by being read from a later stored step. $17$ are kept, `up_st` $17/36$ and
`up_ph` $0/15$: the second stored source block is dropped *whole*, because its fastest channel is
$150.79$ s and every reference that puts the coupling band inside the lag axis sits far below it
(§3). That is the compromise this row makes instead of the feature cells' — theirs keeps a cold
block, this one keeps no block at all.

**What it buys is that every lag the attention can reach is warm.** The kept source channels are
honest by $\max_c(W'_c + d_c) = 6$, and the readout's own half-warm threshold for the `up_st` block
lands at step $5$ of $300$. The deepest step any lag reaches from the anchor floor is $F - (L - 1) =
44$, which is $39$ steps past that threshold, and the warmth readouts are evaluated at the decoded
anchors — all of them at or above $F$ — so no reachable lag is cold.

**The two warmth columns are therefore read differently here than anywhere else in the family, and
at the shipped reference neither of them varies.** In the causal-feature cells **a small value there
is the expected finding, not a failure**, because the number sizes the compromise those cells make.
Here `source_lag_warmth_frac_st` is exactly $1.0$ **for a real reason** — its block genuinely is warm
at every reachable lag — and `source_lag_warmth_frac_ph` is $1.0$ over **zero channels**, because
`_resolve_block_warm_steps` reports an empty block as warm at every step by deliberate design: a
constraint over no channels holds, and reporting a block that does not exist as permanently cold
would put a zero in the CSV that reads as a measurement rather than as an absence. Read the second as
the absence it is. Both regain information on the unaligned arm, which is where the compromise the
columns were built for still exists; the only record of the second source block in a shipped run is
its own $0/15$ in the startup budget summary.

The residual is still *made measurable* rather than assumed away, and the mechanism is retained
unchanged for the unaligned arm's sake: `lag_floor` generalises the lag validity mask from
$\mathbb{1}[t - \ell \ge 0]$ to $\mathbb{1}[t - \ell \ge F_u]$ and ships at $0$, where the mask is
bitwise the sibling's, and the availability mechanism announces per step when each channel arrives.
The input-warmth policy of §3 is still stated over the target stream alone, and unaligned that still
matters: a floor that cleared the unaligned source would sit at $277$.

**The availability clock, and the control that isolates it.** The source availability pattern
$m^u_{t,c}$ is a deterministic function of $t$, identical for every sample, and it enters
$q(z \mid Y, U)$ but not $p(z \mid Y)$ — so the posterior can be pushed off the prior by the clock
alone, inflating `source_conditioned_kl_raw` with no source information in it. The permutation
control cannot detect that: it deranges `source_state` across the batch, and every row carries the
same pattern. The source-null arm, `kld_source_null`, re-runs the source gate, adapter and encoder
from a zeroed stream — zero being the channel mean over the region the model reads, since the
normalisation constants excluded the warm-up — and reports the KL that remains. It costs one source
*encode* and no decode, draws no noise, and its adapter output depends on $m_t$ alone, so the null
state is one $(1, T, d_{\mathrm{model}})$ tensor broadcast over the batch.
`source_conditioned_kl_raw - kld_source_null` is the part attributable to source variation; **if the
two are equal, the coupling readout is measuring a clock.** At this reference the clock is nearly
flat — every kept channel arrives by step $6$ — which makes the arm cheaper to interpret and no less
necessary.

**The null control re-encodes through whichever module `lag_kv_source` selected**, not through a
deep source encoder it may not have built. That clause is what keeps it a control after §8.1: it
probes the tensor the attention reads rather than one nothing consumes, and the permutation control
permutes that same tensor for the same reason. Neither control's signature moved — each resolves the
path off model attributes — so no call site in the task layer changed.

### 8.1 Where the keys and values come from

**`lag_kv_source` decides which source representation the lag attention's keys *and* values are
built from.** Under `encoder` (the constructor default) they are the deep source history state,
which the LSTM makes a function of the whole causal prefix — so by the data-processing inequality
the lag-$0$ entry already contains whatever any later-lag entry carries, and an attention
distribution pinned there is reporting a representation that made every other lag redundant rather
than an absence of delay. Under the shipped `conv_stem` they are a `CausalConvStem` over the
availability adapter's output; under `adapter` the adapter's output directly, one step of content.
**Both** K and V move together, because localising keys while leaving deep values would leave the
lag-$0$ value informationally dominant and the degeneracy standing under a better-looking map.

Under a local arm the deep source encoder is **not built at all** — nothing else in the model
consumes it, so building one would be a starved parameter block in DDP's expectation set and a claim
in the manifest that the model attends over a state it does not have. The stem reuses this parent
encoder's own convolution schedule and block classes, with an output norm, so the shared encoder's
validation is untouched and the arm removes rather than replaces; `causalize_norms` follows the stem
wherever it followed the encoder. **What that schedule costs is stated rather than assumed**: at the
shipped `encoder_extra_dilations: [8, 16]` and `encoder_extra_kernel: 15` the stem reaches $387$
steps, longer than the $91$-lag window and longer than the $300$-step sequence, so on this parent
`conv_stem` removes unbounded *recurrence* and leaves a convolution memory that is still effectively
whole-prefix. A lag profile read off this arm must be read beside that number; the conv-Transformer
twin, whose stem reaches $21$ steps, is where the localisation argument gets an honest test.

**`alibi_slope_scale` is the other half of what a lag profile can express.** `lag_bias_init:
alibi_decay` builds a learnable $(\text{num heads}, L)$ parameter and the scale is what it is
*seeded* with: the shipped $0.0$ seeds it flat, so the profile the model reports is what training
put there, and $1.0$ seeds a monotone decay towards lag $0$. The pair is what separates "the data
says lag 0" from "the initialisation says lag 0". `lag_bias_init: normal` is not the flat arm — it
builds no bias parameter at all.

### 8.2 The prior is told the arrival clock too

**`prior_availability_input` makes the clock symmetric rather than subtracted.** The asymmetry above
is that a deterministic function of $t$ enters $q$ and not $p$; the mechanism gives the same function
to $p$, so the term cancels in the divergence instead of being measured out of it afterwards.
`FullLatentPriorHead.forward(h_y, clock=None)` takes it through the head's **own** `LayerNorm` and a
zero-initialised linear projection — deliberately not shared with the adapter's `mask_proj`, because
the pattern is what must be shared and sharing the map would couple the two pathways' gradients. The
projection is re-zeroed in this parent's post-initialisation zeroing block, beside
`_zero_init_delta_heads`, because the generic `initialization(self)` pass runs after module
construction and would otherwise refill a constructor-only zero; with the re-zero in place the prior
is bitwise the flag-off prior at initialisation and the exact-zero-KL start survives.

**What the prior is given is not the announcement.** The per-channel staircase is *constant on every
scored anchor* — the anchor floor is defined to clear the last step at which it changes — and a
constant through a `LayerNorm` and a linear map is an offset the prior head's biases already span.
What survives past the floor is the source pathway's **memory** of the arrivals, so the clock is the
encode of a **zeroed** source stream through the configured K/V path, computed at batch $1$ and
broadcast: the same tensor the source-null control feeds the posterior, **detached** so no gradient
couples the prior to the source pathway, and **forced out of train mode** so dropout cannot make it
a fresh draw per step rather than a function of $t$.

**The invariant is restated, not weakened.** "The prior never sees the source" becomes **"the prior
sees no function of the source *values*"**, and that is what the source-purity tests assert. On this
row the mechanism has less to do than on the feature-target cells for a stated reason: every kept
source channel here arrives by step $6$, so there is very little arrival transient for the encode of
silence to carry past the floor. `lag_attn_cfs/DESIGN.md` §8.2 carries the bound on what any
prior-side clock can reach, which is a property of the shared posterior parameterisation and holds
identically here; §14 records it as a limitation with its trigger.

## 9. Forward return dict

`SeqVaeLagAttnCrws.forward(y_st, y_ph, u_stream, anchor_phase=None, anchor_stride=None)` — the
signature is the causal-feature cell's exactly, because the three input tensors are — returns
**twenty-two keys**: the raw-signal family's twenty, plus the two the anchor axis needs.

- `mu_base`, `logvar_base`, `mu_full`, `logvar_full` — $(B, A_{\max}, H, R)$, so $(B, 5, 30, 16)$
  at the shipped training geometry, $(B, 136, 30, 16)$ at the dense evaluation one, and $16$ wide at
  every budget.
- **`anchor_index`** $(B, A_{\max})$ `long` and **`anchor_valid`** $(B, A_{\max})$ `bool` — the
  decoded anchors and which of them are real. The dtypes are part of the contract: the first gathers
  and scatters, and the second multiplies into a float mask.
- `mu_prior`, `logvar_prior`, `raw_logvar_prior`, `mu_post`, `logvar_post`, `z_prior`, `z_post` —
  $(B, 300, 64)$; `target_state`, `source_state` — $(B, 300, 128)$; `attended_source_heads`
  $(B, 300, 4, 32)$; `attn_weights` $(B, 300, 4, 91)$; `kld_per_t`, `kld_per_t_per_head`,
  `source_kl_lag_map`, `mu_prior_sat_frac`, `delta_mu_sat_frac` — all unchanged in shape.
  **`source_state` is the lag attention's K/V tensor** rather than the deep source state by
  definition (§8.1): under `lag_kv_source: encoder` those are the same object, and under the shipped
  `conv_stem` it is the stem's output. It is the thing the model reads, which is what makes both
  controls' re-pointing a consequence rather than a second decision.

**No `decoder_state` and no `delta_mu_src`**: the decoder receives the latent and nothing else, so
there is no bypass to report. **And no `persistence`**, because this row deliberately does not take
the feature-target cells' persistence residual (§14) — so the twenty-two keys are the whole contract
at every configuration, where in those cells the count moves with a switch. A three-argument call
and a four-argument call with a zero phase agree bitwise at stride $1$.

## 10. What is measured

Four readouts this package adds — three on both stages, one of which is a guard rather than a result,
and one on the evaluation stages alone — and **eight the causal-feature cells carry that this one
drops rather than re-points**.

| Metric | Stages | What it separates |
| --- | --- | --- |
| `anchors_per_sample` | train, val | the tiling actually firing; $[4, 5]$ in train, $136$ in val — a **guard**: a row off that band means the geometry broke, not that the model learned something |
| `source_lag_warmth_frac_st` | train, val | attention mass landing on lags where the first stored source block is warm — exactly $1.0$ at the shipped reference, because every lag the attention reaches is past that block's step-$5$ threshold (§8) |
| `source_lag_warmth_frac_ph` | train, val | the same for the second block, which this cell's reference drops whole — so it is $1.0$ over **zero channels**, an absence rather than a measurement (§8) |
| `kld_source_null` | val | the KL floor the availability clock induces with no source content (§8) |

The first three are merged onto the shared objective's metric dict inside `compute_loss`, from the
two readouts bound from the causal-feature cell (§6); `kld_source_null` is task-side, through the
bound `_added_metrics`, because it needs the source stream, which is not in the forward dict.

**Both warmth columns are constants at the shipped reference, and that is a geometry fact rather
than a bug.** The pattern each is read against is built from $W'_c + d_c$ — the warm-up *plus* the
alignment shift — because a gathered-and-delayed channel is honest only once both have passed. On an
earlier build the source patterns were resolved from $W'_c$ alone while the availability mask and
the anchor floor both used $W'_c + d_c$; since $d_c \ge 0$ the omission could only ever report the
source as *warmer* than it is, and it made `source_lag_warmth_frac_st` identically $1.0$ for any
attention distribution — a column that could not vary and therefore measured nothing.
`lag_attn_cfs/tests/test_metrics.py::test_the_shipped_source_warmth_pattern_is_not_vacuous` is the
guard, asserted on the pattern rather than on the fraction because the pattern is the mechanism.

**The fraction here is still $1.0$, and for a different reason.** The corrected `up_st` pattern of
this cell is `False` for steps $0$ to $4$ and `True` from step $5$ — it is not vacuous — but the
readout is evaluated **at the decoded anchors**, all of which are at or above $F = 134$, and the
deepest step any of $L = 91$ lags reaches from there is $44$. So no reachable lag is cold and the
column is exactly $1.0$ as a statement about the geometry, which a different reference would move.
`source_lag_warmth_frac_ph` is $1.0$ over zero channels and is an absence rather than a measurement
(§8).

**Dropped, and why.** The causal-feature cell's surface is $93$ entries; this one's is $77$, and
the sixteen missing are eight columns on both stages. Four are the feature target's own gap splits:
`pred_gap_tau_first` and `pred_gap_tau_last` resolve the gap by horizon step, which the raw-signal
family never tracked, and `pred_gap_st` and `pred_gap_ph` split it by stored target block, and a raw
target has no blocks. Four are the causal-feature cell's: `pred_gap_warm_lo`, `pred_gap_warm_mid`
and `pred_gap_warm_hi` are tertiles of $W'_c$ over kept target channels, and there are none — the
block's last axis counts raw samples, which have no warm-up, no filter and no order to rank by; and
`target_warm_frac` is the share of scored target coefficients past their own warm-up, which would be
vacuously $1.0$, and a vacuous constant column reads as a measurement. None of the eight is tracked,
and `tests/test_metric_tracking.py` asserts the surface in both directions.

**The parent's own `pred_gap` is untouched and is the headline readout**, which is the point of the
cell: it is the same quantity `lag_attn_rws` reports, over the same raw target, computed from inputs
that do not contain the answer.

**One inherited readout is re-pointed rather than left alone.** `_mu_gap_rms` rebuilds
`forecast_mask` and `kl_mask` itself with no anchors, and its own docstring promises it uses "the
KL's own anchor support … so the two cannot drift". Under tiling that stated invariant *fails*:
`mu_post_prior_gap_rms` would average the latent belief shift over all $136$ anchors while the
`source_conditioned_kl_raw` printed beside it averages over $\approx 10$. The bound override takes
the same anchor set, which restores the property the function already claims.

`LagAttnCrwsTrainer.TRACKED_METRICS` carries **77** entries: the raw-signal driver's $70$, this
package's three on both stages, and the source-null KL on validation alone. A `train/kld_source_null`
is deliberately absent — it is a readout that never enters the objective, so the column would be NaN
in every row of every run.

## 11. What is drawn

The page is the raw-signal sibling's — `build_diagnostic_figure` owns the GridSpec, the shared
physical-time axis, the caption and the five latent-and-attention rows — and this cell replaces one
seam and borrows two. **Nine titled rows**: the sibling's seven and the two input rows.

| # | row | whose it is |
|---|-----|-------------|
| 1 | `raw` | the shared raw-context row, drawn once — this model's target *is* the raw trace |
| 2 | `forecast` | **this cell's**: the base and full forecasts tiled into non-overlapping $480$-sample windows, walked off `anchor_index` and `anchor_valid` from the forward dict; window edges; the anchor overlay marking the floor, the decoded anchors and the training tile grid over the dense set the page is produced at; and the lag caveat as a footnote |
| 3–4 | `input_target`, `input_source` | the causal-feature cell's `causal_stream_panels`, **imported** — the streams are the same three tensors, and the shipped builder consults the production two-sided Morlet bank, which did not produce these coefficients, inside a handler that warns and continues |
| 5–9 | `latent`, `kld_dims`, `kld_total`, `lag_attn`, `kl_lag_map` | the shared builder's, unedited |

**Why the forecast rows are re-pointed, and the failure they prevent.** The shipped raw rows tile
through `concat_single_forecasts`, which walks `range(warmup, t_valid, horizon)` and reads the
forecast at each **anchor**; this model's forecast is $(A_{\max}, H, R)$ indexed by *position in the
decoded set*. At the shipped geometry that is $136$ positions read at anchors $134 \dots 269$, so
the first read is already out of range and the page dies inside a handler that warns and continues —
a run with an empty diagnostics directory and one log line, which the first real fit confirmed. At a
smaller floor it would draw a real forecast at the wrong time with no exception anywhere in it.
`tests/test_sample_page.py` compares the drawn curve at the first tile against the forward's own
`mu_full` at the anchor `anchor_index` names, so an implementation reading position $k$ as anchor
$k$ fails even though every shape, axis and colour is right. Raw samples no drawn window covers
render as gaps, never as fabricated continuation. `row_axes` and `finalise_time_axis` are reused
rather than reimplemented, and the anchor walk and the overlay are the causal-feature page's own
functions reached by import.

**Two rows, not the causal-feature page's eight.** That page adds six stitched field rows because
three of $98$ target channels drawn as lanes is not a picture of that forecast. This decoder emits
$R = 16$ raw samples of **one** signal, so the forecast is a curve on the raw grid and the shipped
two-row layout already is the picture; nothing is reserved through `forecast_extra_rows` and the
task declares no such seam.

**The run-level warm-up budget figure** is the causal-feature cell's, reached through the bound
`input_budget_figure`, and is written once per run under the stem `causal_warmup_budget` — distinct
from the shipped `causal_input_budget`, which describes the two-sided reach guard, so a directory
holding both is readable. It is drawn from the resolved budget rather than from the network, because
its subject is the channels the budget **dropped**, whose $W'_c$ the checkpoint does not carry; the
driver hands the task that budget after building the model, and a task with none refuses the figure
by name rather than drawing an empty one.

**The lag axis caption is one-sided, and it is this cell's own string.** The two lag panels are the
shared builder's and the six shipped models must not gain a caption about a transform they do not
use, so the caveat is a page footnote: lag axes are stored-coefficient time on the input side, not
physical delay — the raw target carries no group delay, so the anchor is exact, but a causal input
coefficient still lags by its own composed group delay, and the correction to a physical lag has no
target-side term to subtract. Stating the sibling's two-sided correction here would be wrong in the
direction that reads as more careful. `tests/test_lag_consistency.py` asserts the page's lag axis,
the model's reported lag and a hand-computed physical lag agree, that filling the panel's delay slot
with the warm-up did not silently move the axis, and that a non-zero `lag_floor` moves the mask and
not the axis.

**And under the channel alignment the caption states a lead time in seconds, which is a claim this
cell alone can make.** Unaligned, the bias between a source coefficient and a target one is
$\tau^u_c - \tau^y_{c'}$, indexed by a channel *pair* and therefore not a number a pooled source
state can carry. With every source channel shifted onto one reference $\tau^u_{\mathrm{ref}}$ that
index collapses to a single constant, and because the target here is a raw sample —
$\tau^y \equiv 0$ exactly, not approximately — the footnote can give the identity outright: a peak
at lag $\ell$, horizon element $h$, is a physical lead time of
$\Delta(\ell + 1 + h) + \kappa\tau^u_{\mathrm{ref}}$ s on the canonical stored timeline, with no
dataset-shift term. The arithmetic is
`teb_vae/lag_attn/nets/lag_report.py::physical_lag_seconds`, which the caption interpolates $\Delta$
and $\kappa$ from so that the two cannot state different constants.

The caption gives the identity and **not** a number, because $\tau^u_{\mathrm{ref}}$ is a decision
of the run rather than a property of the page: the rows are handed arrays, a geometry and the
loader's statistics, and the model's own `source_delay_steps` is the largest *stored-step* shift —
$6$ at the shipped reference, attained by the channel furthest from it — and is emphatically not it.
The resolved value travels in the run's own resolved-config dump and preflight disclosure as
`source_reference_delay_s`, which the caption names so a reader can complete it. The causal-feature
cells keep the refusal, narrowed rather than withdrawn: their $\tau^y_{\mathrm{ref}}$ is nonzero,
so the same expression there is a lag between two coefficient epochs and not between two signals.

**The constant that completes the identity is the *effective* reference, not the logged one.** The
shift cancels each channel's realised centroid delay onto $\kappa\,\tau^u_{\mathrm{ref}}$, so what
the expression above takes is $\kappa\,\tau^u_{\mathrm{ref}} = 36.93$ s at the shipped
$\tau^u_{\mathrm{ref}} = 42.2066$ s. Nothing in the tree applies $\kappa$ to the disclosed number
and `physical_lag_seconds` uses whatever it is handed, so a reader who substitutes
`source_reference_delay_s` verbatim overstates every lead by
$(1 - \kappa)\,\tau^u_{\mathrm{ref}} = 5.3$ s. §14 records that as a limitation rather than a
defect of the page.

**And this is where the reference choice of §3 becomes legible.** At an effective $36.93$ s the
reachable lead spans $[40.9, 516.9]$ s, so a peak at $\ell + h = 0$ reads as $40.9$ s and one at
$\ell + h = 20$ as $120.9$ s — most of the $20$ to $120$ s contraction-to-deceleration band is on
the axis. At the feature cells' $402.1604$ s the same axis would begin at $355.9$ s and the band
would not be on it at any lag index at all.

## 12. Configuration, and DDP

`configs/` ships exactly `default.yaml`, `tiny.yaml`, `smoke_causal.yaml`,
`sweep_anchor_stride_1.yaml` and `sweep_horizon_15.yaml`, and the directory listing itself is
asserted. Each is written out in full rather than inheriting: a `base:` chain would be the smaller
file and the worse record, because it hides which settings this run shares with the model it is
compared against, and everything outside the input representation being identical is the whole
value of the comparison.

**`default.yaml` is pinned leaf-for-leaf against `lag_attn_rws/configs/default.yaml`** outside a
declared allow-list of **twenty-eight** exemptions, in both directions — an exemption that is no longer a
divergence fails as loudly as a divergence that is not exempt. Five are identity (the run tag, the
output tree, the MLflow experiment, run name and variant tag); four are the dataset (the causal
shards, their statistics, and `epoch` in `load_fields`); eight are the geometry the transform forces
or the tiling adds (`c_y`, `c_u`, `warmup_period`, `horizon`, `causal_reach_budget_s` — present and
required `null` —, `causal_warmup_budget_steps`, `anchor_stride`, `lag_floor`) and two more
are the channel alignment, which has no two-sided counterpart at all
(`causal_align_reference` — the one exempted key whose *value* also differs from the
causal-feature cells', at `42.21` against their `target_max`, for the reason §3 gives — and
`causal_leg_alignment`); one is the objective
(`lambda_boundary`, $0.05 \to 0.0$); **seven are mechanisms rather than values** — the four
architecture switches this row takes (`lag_kv_source`, `prior_availability_input`,
`horizon_weight_halflife_steps` and the flat `alibi_slope_scale`) and the three training controls
beside them (`early_stopping.enabled` and its `patience`, and `model_checkpoint.secondary_monitor`),
each a key the two-sided sibling's constructor does not have and its config never carries, so the
divergence is *this mechanism exists here* rather than *this number was chosen differently*; and
**two are measurements** — `gradient_clip_val` and
`spike_breaker.additive_margin`, declared `RETUNED` and asserted to have moved *down*, because a
smaller block cannot want a larger threshold. `ema_floor` and `horizon_embed_std` were re-derived
and came back to the sibling's values, and are listed as `MEASURED_TO_MATCH` so the equality reads
as a measurement rather than an oversight. `tests/test_config_load.py` also asserts
`anchor_stride == horizon` in the default and in every arm, since nothing in the shipped code ties
the two, and that every key reaches a constructor argument or a task-level consumer.

`tiny.yaml` points at the committed causal fixture, shrinks `d_model`, `d_z`, `d_head` and
`max_lag`, ships `likelihood: mse`, and retains the real geometry — the sequence length, the channel
counts, the budget, the floor, the horizon and the trim. `smoke_causal.yaml` carries the shipped
widths and `gaussian_nll` against the same fixture with the clip parked at $10^{9}$, and is the
instrumented run the two `RETUNED` constants were measured on: four windows in one batch, so its
$600$ epochs are $600$ optimizer steps. The two arms each move the smallest possible key set with a
declared, exact delta: `sweep_anchor_stride_1.yaml` trains densely, and `sweep_horizon_15.yaml`
restores the raw-signal sibling's horizon and its $480$-sample block, pinning `horizon_depth` on the
family's $\mathrm{RF} \ge H + 1$ criterion. **No floor arm** (§3). `emit_validity_mask` appears in
no shipped config: the mask is a filter-bank constant, and the resolved vectors reach the model
through `model_kwargs`, which also puts them in the checkpoint.

**DDP.** Production runs under plain `"ddp"` with `find_unused_parameters=False` under
`gaussian_nll`; `mse` starves the decoder's log-variance heads — at width $R$, which is the one
width no budget can move, so both guard states starve the same tensor at the same size — and selects
`ddp_find_unused_parameters_true`. The availability terms are unconditional in the forward and every
branch is a construction-time decision on a module's existence; the per-segment phase is derived per
rank from that rank's own samples with no collective; $A_{\max}$ is a geometry constant, so no rank
can disagree on shape; `broadcast_buffers=False` because every buffer — the two source-warmth
patterns among them, both non-persistent — is a deterministic function of the config; and
`static_graph` is deliberately absent, because the spike breaker's skipped-batch backward is
structurally different from the recorded one. Two evaluations of one checkpoint on one shard must
produce an identical `metrics_history.csv` row set, and `tests/test_train_smoke.py` asserts it
across two CPU subprocesses under different `PYTHONHASHSEED` values, pinned to one intra-op thread
because a reduction split across workers accumulates in the order they finish.

**A run's own artifacts state both of the two independently-toggleable mechanisms.** The shard
variant and the stream reference are separate decisions and each is its own configuration key —
`causal_leg_alignment` names which phase-harmonic operator built the phase blocks and is compared
against the shards' own root attribute; `causal_align_reference` names the clock the input
channels are shifted onto and resolves to $\tau_{\mathrm{ref}}$ from the data — at the shipped
`42.21` the budget summary reads `reference 42.2066 s`, `fhr_st 17/36, fhr_ph 21/66; 38/102
channels`, `up_st 17/36, up_ph 0/15; 17/51 channels`, and `shift 0-6 steps, honest by 0-6` on each
stream. Both are leaves of
`model_config.VAE_model`, so both land verbatim in the resolved-config artifact the run writes
beside its checkpoints, and the resolved *consequences* land in the startup log's budget summary
(the reference in seconds, the shift range and the surviving counts per block) and in the
checkpoint's own `model_kwargs`, which carry the two shift vectors. No tracked metric is added
for the reference: it is a constant of the configuration, so a per-step column of it would be the
same number in every row.

## 13. Parameter budget

Measured on constructed models in one process, not predicted: both cells of this row at the shipped
warm-up budget and alignment reference and ungated, and the two raw-signal cells they are compared
against at the shipped reach budget and ungated. `tests/test_docs.py` re-measures every total below
by constructing the models rather than comparing against literals, so a legitimate change to a shared
imported component re-costs this table instead of failing an unrelated assertion — and it evaluates
the stated decomposition parameter name by parameter name, because a table carrying the right delta
beside a wrong decomposition of it is exactly the half a search for the number cannot see.

**Two rows per cell, and both are the record.** The **shipped** row is the revised default: local
K/V, the prior clock, the weighted horizon axis and the flat lag-bias seed. The **off-state** row is
every switch of §17 at its inert value, which is bitwise the model that shipped before this revision
— and it is the row on which the input-representation comparison against `lag_attn_rws` is still
readable, because that cell is a raw-input model and never takes the new keys.

| | conv-LSTM encoders | conv-Transformer encoders |
| --- | ---: | ---: |
| **causal in / raw out, shipped** (budget $134$, aligned to $42.2066$ s; $38$ of $102$ target, $17$ of $51$ source) | $\mathbf{4{,}589{,}907}$ | $4{,}218{,}476$ |
| causal in / raw out, shipped but unaligned ($98$ of $102$ target, $51$ of $51$ source) | $4{,}613{,}715$ | $4{,}242{,}284$ |
| causal in / raw out, shipped but ungated ($102$ and $51$) | $4{,}595{,}155$ | $4{,}223{,}724$ |
| causal in / raw out, `lag_kv_source: encoder` at the shipped reference | $5{,}097{,}786$ | $5{,}006{,}444$ |
| **off-state, budget $134$ and aligned** | $5{,}081{,}146$ | $4{,}989{,}804$ |
| off-state, budget $134$ and unaligned | $5{,}104{,}954$ | $5{,}013{,}612$ |
| off-state, ungated | $5{,}086{,}394$ | $4{,}995{,}052$ |
| raw target, reach budget $120$ s ($78$ of $109$) | $5{,}094{,}458$ | $5{,}003{,}116$ |
| raw target, ungated ($109$) | $5{,}088{,}186$ | $4{,}996{,}844$ |

**Ungated means the whole guard**, the warm-up mask and the common clock together, and that is
forced rather than chosen: a shift vector is positional over the *survivors*, so a stream that has
lost its keep-index has no width for one to be positional against.

### What the revision costs, factorised

**The four architecture switches cost $-491{,}239$ on this cell** and $-771{,}328$ on the
conv-Transformer one, and each term is a module rather than a rounding:

| Term | conv-LSTM | conv-Transformer | What it is |
| --- | ---: | ---: | --- |
| the deep source encoder, not built | $-1{,}312{,}231$ | $-888{,}960$ | `lag_kv_source` is not `encoder`, and nothing else consumed it |
| the local K/V stem, built | $+804{,}352$ | $+100{,}992$ | each parent encoder's own convolution blocks plus an output norm |
| the prior's clock | $+16{,}640$ | $+16{,}640$ | $2 \times 128$ for its own `LayerNorm` and $128 \times 128$ for the bias-free projection |
| the horizon weight | $0$ | $0$ | a non-persistent buffer, so not a parameter and not a state-dict key |
| the flat lag-bias seed | $0$ | $0$ | the same $(\text{num heads}, L)$ parameter, seeded differently |

$-1{,}312{,}231 + 804{,}352 + 16{,}640 = -491{,}239$, and $-888{,}960 + 100{,}992 + 16{,}640 =
-771{,}328$. A delta that does not decompose into exactly these means something else moved. **There
is no persistence term on this row**, because this row does not take that key at all (§14) — the
feature-target cells' figure carries a fifth term of $+2{,}940$ where this one carries none.

**The two stems are very differently sized, and that is a schedule rather than a design difference.**
Each reuses *its own* parent encoder's convolution schedule, so the two arms differ in what is
removed rather than in two independently chosen front ends. Here that schedule is
$(3, 5, 11, 15, 15)$ at dilations $(1, 2, 4, 8, 16)$, reaching
$1 + \sum_b (k_b - 1) r_b = 387$ steps; on the conv-Transformer cell it is $(5, 9)$ at $(1, 2)$,
reaching $21$. §8.1 states the consequence.

**The alignment costs $-23{,}808$ here and $-768$ in the four feature-target cells, and the whole of
the difference is the reference.** Those cells align onto $402.1604$ s and lose four source channels;
this row aligns onto $42.2066$ s (§3) and loses sixty target-stream channels and thirty-four source
ones. The arithmetic:

$$\underbrace{-\,60 \times 128 \times 2}_{\text{target adapter}}
\;\underbrace{-\,34 \times 128 \times 2}_{\text{source adapter}}
\;+\; \underbrace{2 \times 128}_{\text{start embeddings}} \;=\; -23{,}808,$$

the two linears per adapter being its input linear and its availability projection, and the start
embeddings being the vectors both adapters build for the first time in this family once the shifted
warm-up starts above zero (§7). The unaligned row is kept as a named comparison arm rather than
deleted, because it is what `causal_align_reference: null` still builds — and at this reference it is
the arm with **more** parameters, which is the opposite of the feature cells' ordering. **The number
did not move with the revision**, which is the check that the two are independent: the alignment
narrows adapters and the switches replace a source module, and neither touches the other's tensors.

### The two axes

**The encoder axis: $91{,}342$ at the off-state row**, guarded against guarded — the conv-Transformer
encoders are that much smaller at a fixed target. It is **identical** to the reduction the same two
encoders buy in the raw-signal pair at its own budget, on the ungated arm, and in the two-sided and
causal-feature pairs, which is what a difference living entirely in the two history encoders must
look like — and it is what "the grid's two axes are independent" means numerically.

**At the shipped configuration the same axis is $371{,}431$**, identical across the shipped,
unaligned and ungated rows, so it is still the two history stacks alone. It moved because *both*
stacks moved: the target encoders differ by $-331{,}929$ as they always did, and the two local stems
by $+703{,}360$ where the two deep source encoders differed by $+423{,}271$. The comparison against
the other three rows of the grid is therefore only meaningful on the off-state row, where every cell
carries the same source module.

**The input-representation axis: $-13{,}312$** against `lag_attn_rws`, guarded against guarded **on
the off-state row**, and identical in both encoder families. **It changed sign with the reference**:
it was $+9{,}728$ while this cell carried $98$ target and $47$ source channels against the reach
guard's $78$ and $29$, and the same decomposition now runs the other way. It decomposes into
**exactly one surviving term**, measured parameter by parameter, and the decoder head is deliberately
not one of them:

| Term | Value | What it is |
| --- | ---: | --- |
| the horizon embedding | $0$ | `nn.Parameter(torch.zeros(horizon, decoder_hidden))`; both cells forecast $30$ steps, so the two are the same size |
| the two input adapters | $-13{,}312$ | $128 \times (38 - 78)$ and $128 \times (17 - 29)$ on the input linear *and* the availability projection of each stream |
| the two start embeddings | $0$ | the reach guard builds both and, under the alignment, so does this cell (§7) |
| the decoder's output head | $0$ | `raw_per_step` in both cells, so `mean_head.out_features == 16` on each side |

$0 + (-13{,}312) = -13{,}312$, and every parameter name whose count differs between the two guarded
models is one of the four adapter weights; a delta that does not decompose into exactly them means
something else moved. **Ungated against ungated the axis is $-1{,}792$**, which is
$128 \times (102 - 109) + 128 \times (51 - 58)$ on the two input linears at the narrower stored
widths; the horizon-embedding term is zero there for the same reason. The ungated arm did **not**
move with the reference, because an ungated stream has no keep-index for a reference to narrow.
**It is read on the off-state row deliberately:** `lag_attn_rws` is a raw-input model and never takes
the new keys, so at the shipped configuration the difference between the two is dominated by
mechanisms one of them does not have, and no input-representation reading survives it.

**Two of those terms used to be nonzero, and both went to zero for stated reasons.** The horizon
embedding contributed $-3{,}840 = -15 \times 256$ while this cell forecast one minute against the
raw-signal sibling's two; both now forecast $30$ steps. The start embeddings contributed $-256$
while the reach guard built both and a warm-up whose fastest channel waits zero steps built neither;
the alignment's shift makes the combined minimum $\min_c(W'_c + d_c) = 1$ on both streams, so this
cell builds both too. They are computed rather than deleted so that either divergence reappears as a
failing sum.

**The guard changed sign too, and both numbers are right.** Guarded minus ungated is

$$\underbrace{128 \times 38}_{\text{target availability}}
+ \underbrace{128 \times 17}_{\text{source availability}}
+ \underbrace{2 \times 128}_{\text{start embeddings}}
- \underbrace{128 \times 64}_{\text{target input linear}}
- \underbrace{128 \times 34}_{\text{source input linear}} \;=\; -5{,}248$$

on this model, against $+6{,}272$ on `lag_attn_rws`. The sign is decided by how much the guard
narrows: the reach budget drops $31$ of $109$ target and $29$ of $58$ source channels, few enough
that the two availability projections it adds outweigh the narrowing of the two input linears; here
the budget and the alignment together drop $64$ of $102$ and $34$ of $51$, so the narrowing dominates
and the guarded model is *smaller* than the ungated one. It was $+17{,}792$ at the feature cells'
reference, which is the same expression at $98$ and $47$. Unlike the feature cells, nothing in this
target domain widens a head: every parameter the guard moves is under an adapter, which
`tests/test_docs.py` asserts by name. **The expression is the same on the shipped and off-state rows
alike** — five terms, no persistence weight — because this row does not take that key.

## 14. Deliberate limitations

- **The nats are row-local.** §5. Comparable to `lag_attn_transformer_crws` and to nothing else,
  including the direct control; recorded, not fixed.
- **`_mu_gap_rms` is overridden onto the tiled anchor set.** §10. The alternative — leaving it dense
  and renaming the column — would put two differently-denominated numbers on one page.
- **Group-delay compensation is per channel, and the constant that completes it is not the one the
  run logs.** `causal_delay_s` is read by the warm-up resolver, which turns it into the per-channel
  shift §3 describes, so each stream's entries describe one instant. Because the target here is raw,
  $\tau^y \equiv 0$ and the residual bias on a lag reading is a single known constant rather than a
  channel-pair-dependent range. But that constant is the **effective** reference
  $\kappa\,\tau^u_{\mathrm{ref}} = 36.93$ s, not the $42.2066$ s the resolved-config dump and the
  preflight disclosure carry as `source_reference_delay_s`: the shift cancels each channel's realised
  centroid delay, not its envelope mean. Nothing in the tree applies $\kappa$ to the disclosed
  reference and `physical_lag_seconds` takes whatever it is handed, so substituting the logged number
  overstates a lead by $(1 - \kappa)\,\tau^u_{\mathrm{ref}} = 5.3$ s. No readout divides either out
  yet, so a lag-resolved caption still carries the one-sided caveat (§11).
- **The transform side and the model side are on different delay conventions, on purpose.**
  `hdf5_dataset/causal_scattering.py::leg_alignment_shift` — the shift baked into the *stored*
  coefficients — still uses the unscaled $\tau_g$, while the model-side alignment
  (`channel_alignment_delays` and `_align_stream`) carries $\kappa$. Applying $\kappa$ on the
  transform side would change stored coefficients and require a dataset rebuild, which is not in
  scope, so the two are deliberately on different conventions. Stated as a limitation rather than
  left to be discovered as a discrepancy.
- **Every local measurement is in-sample.** `dataset_kwargs` is shared between the two loaders and
  cannot carry a per-split GUID filter, so a dev-box run validates the objective's optimisation
  behaviour and says nothing about generalisation.
- **No mixed precision** (`precision: "32-true"`) and **`compile: false`**, the latter permanently:
  the `nn.LSTM` encoders, the checkpointed attention region and the data-dependent boolean mask
  indexing behind `kld_active_frac` each break TorchInductor independently.
- **No warm start.** `core_model_checkpoint` stays `null`: a blob from any sibling carries a
  different `model_class` stamp, and a blob from a raw-signal cell has adapters at other widths —
  $78$ and $29$ against this cell's $38$ and $17$; the guard refuses each, correctly.
- **No run checker ships.** The causal-feature cell ships `check_run.py` because its headline runs are
  blocked on the production box and its criteria must be scored by code. No production run is in
  scope here, so a checker would have nothing to read; the geometry guard `anchors_per_sample` is
  asserted on the fixture fit by `tests/test_train_smoke.py` and read by hand from
  `metrics_history.csv` until a checker exists.
- **Checkpoint compatibility with the pre-revision model is deliberately broken.** The constructor
  and state-dict changes of §17 mean a pre-revision blob does not load into a shipped-configuration
  model. `load_checkpoint_strict` refuses rather than partially loads and `check_model_class` still
  guards the class, so the failure is by name; no migration shim is built, because the off-state arm
  exists for exactly the case where the old weights are wanted.
- **The K/V receptive field is the lag resolution floor, and on this parent it is not local.** §8.1
  and §13. The stem reuses this parent's own convolution schedule, which reaches $387$ steps —
  longer than the lag window and longer than the sequence — so here `conv_stem` removes unbounded
  recurrence rather than whole-prefix memory. Giving the stem a schedule of its own would be a new
  decision rather than a reuse, and it is not taken.

> lean-limit: no persistence residual in this row; replace with an anchored raw persistence input
> when this cell's fast-step NLL shows on a trained revised run the same suppression signature the
> feature-target cells' residual answers.

**The exclusion is deliberate and it is about the object, not about the effort.** What the residual
answers in the feature-target cells is a *feature-domain* measurement — that the channels the model
abandons are the fast phase-harmonic ones, whose persistence $R^2$ is negative from horizon step $4$
— and a raw persistence input is a different object: this block's last axis is $R = 16$ raw samples
of one trace, with no channel axis for an $(H, C)$ weight to be positional against, and the anchor's
"own value" is a $16$-sample vector rather than a per-channel scalar. Both raw-target constructors
therefore refuse `persistence_residual` **by name at construction** rather than accepting it and
building a weight no forward reaches: the parents carry a hook of exactly the shape `_prior_clock_dim`
already has, which raises here and is overridden to a no-op by the feature-target mixin. Without it
a config setting the key would build an $(H, R)$ parameter that DDP expects a gradient for and fail
three frames deep on the first batch instead of at the key.

> lean-limit: the prior's clock cancels no part of the KL's **mean** term, because the posterior is
> a bounded residual on the prior and that term is a function of the delta head alone; replace with
> a delta defined as $D(a) - D(a^\varnothing)$ -- and `posterior_logvar_mode` back to `residual` for
> the variance half -- when the owner accepts a change to the posterior parameterisation the whole
> coupling readout is defined on.

`lag_attn_cfs/DESIGN.md` §8.2 carries the decomposition. It is a property of the shared posterior
parameterisation rather than of any target domain, so it holds identically here — and on this row it
binds a mechanism that had little to do in the first place, since every kept source channel arrives
by step $6$ and the arrival transient the clock exists to describe is correspondingly short (§8.2).

**The second alignment reference is deliberately not taken here, and this row needs none.** The
feature-target cells gained `causal_align_reference_source` because one reference resolved from the
*target* stream was being applied to both, censoring a physiological delay off the lag axis. Here the
target is raw with zero delay, so the single `causal_align_reference: 42.21` is **already**
source-only: there is no second clock to separate from, the inter-stream offset is the reference
itself, and §3's analysis of what this row's reference buys is untouched. Adding the key would be a
second way to say what one key already says.

> lean-limit: the anchor floor is $134$ by policy rather than by validity, costing $104$ of $240$
> available anchors; replace with $F = 30$ and a re-derived stride when a run shows the anchor count
> rather than the source pathway is the binding constraint on `pred_gap`.

A raw target is honest at every step, so anchors could begin at the model's own $30$-step warm-up:
$240$ against $136$. What the floor buys is that every kept target-stream input channel is past its
own warm-up at every scored step; what it costs is $40\%$ of the supervision. At this cell's
reference the purchase is nearly free to give up — the policy's own requirement is $6$ steps (§3), so
the constructor would admit $F = 30$ with no edit to any guard — which makes the note a decision
about supervision rather than a refusal to unpick.

> lean-limit: the driver's four config-shaped pre-flight refusals and the task's two
> `super()`-calling members are written out here rather than shared with the causal-feature cell,
> each pinned by its own refusal or behaviour test rather than by identity; replace with a shared
> home when a third consumer of any of them appears.

`_check_no_cross_channel_block`, `_check_boundary_term_is_off`, `_check_phase_key_fields` and
`_check_source_block_kept` are module-level functions of `trainer.py` with the causal-feature
driver's behaviour, and `SeqVaeLagAttnCrwsTask.__init__` and `compute_loss_and_metrics` are the
three-line members §6 explains cannot be bound. Every other sibling member this package needs is
bound (§6), and the anchored raw gather is the one piece of arithmetic that is genuinely new; its
one test that matters is elementwise equality with `build_future_target` at the dense anchor set.
Two consumers with a test between them is a defensible cost; three would not be.

> lean-limit: no `eval/` package, so every number is a scalar from one run's own
> `metrics_history.csv` and no reported difference carries an uncertainty; replace with a
> `ModelBinding` against `teb_vae/lag_attn_rws/eval` when a result from these cells is to be
> reported as a measurement rather than as a demonstration.

Deferred whole, as the causal-feature cell deferred it, and stated here so no number from this row
is read as though it had a confidence interval. Unlike that cell the binding is genuinely available:
the target is raw, which is exactly what `teb_vae/lag_attn_rws/eval` evaluates, and
`eval/binding.py::ModelBinding` is a frozen dataclass that names no model class, with
`teb_vae/lag_attn_transformer_rws/eval/` the working precedent at six modules. The one known
obstacle is recorded rather than solved: `eval/metrics.py::model_inputs` builds its own three-tensor
call and deliberately bypasses `_build_forward_inputs`, so it would forward with no phase and no
stride and silently score the default geometry. `tests/test_lag_consistency.py` asserts the absence
of an `eval/` directory, so a pipeline arriving here without its lag read site pinned fails against a
stated intention rather than becoming the third consumer that went unnoticed the first time.

## 15. Deviation record

Where the built package differs from the design it was built from, and why.

- **One mixin, not two, and the second is a subclass rather than a sibling.** An earlier shape had a
  `CausalRawForecastTarget` mixin beside `CausalRawInputs` holding the objective and the readouts. A
  raw target has no width hook, no target-channel readouts and one line of gather to add, so the
  second class would have held one method; `compute_loss` lives on `CausalRawInputs`, whose `vars()`
  is pinned to exactly eight names, and the gather is a **module-level function** so that set stays
  eight and the dense-equivalence test can call it directly.
- **`_check_anchor_floor` is the override, not `_validate_causal_geometry`.** §6. The narrower
  override leaves the stride-versus-span refusal with one expression.
- **The task binds six step-path members from the causal-feature task, not four.** `_mu_gap_rms`
  and `_added_metrics` are target-agnostic too, and binding them removes the last copyable body;
  the two that cannot be bound are named in §6 with the reason.
- **`shipped_warmup_kwargs` in the test conftest takes the model class as an argument**, defaulting
  to this package's model, because the mapping it wraps refuses a class that cannot take the warm-up
  vectors — and a test *about* that refusal names its own class rather than whichever a default had
  chosen.
- **`lambda_ms` and `lambda_deriv` are inherited at $0.1$, not zeroed.** §5. Only `lambda_boundary`
  moves, and it is refused rather than merely set.
- **The gradient clip moved and the additive margin moved, both down, and both are measurements.**
  `gradient_clip_val` $5000 \to 1000$: the smallest round value above the pre-clip norm's measured
  $q_{99}$ of $953$ on the instrumented run, well below its maximum of $1420$. `additive_margin`
  $1.0\mathrm{e}{+3} \to 5.0\mathrm{e}{+2}$: about $2.0\times$ the worst excursion above the
  breaker's own EMA in the noisiest regime the fixture can produce ($248$, at batch $1$), and inside
  the $\approx 7.6\mathrm{e}{+2}$ magnitude the objective can reach — so the additive test can still
  fire, which is the check the sibling's $1.0\mathrm{e}{+3}$ would *fail* at this block size. The
  batch-$1$ excursion needed its own harness: `metrics_history.csv` is per epoch, and at four
  optimizer steps per epoch the per-step statistic the breaker compares is not recoverable from it.
  `ema_floor` and `horizon_embed_std` stay at the sibling's values because the first is a switch
  rather than a scale and the second is a per-pair quantity that does not depend on the token count.
  `RESULTS.md` carries the percentiles; every one of them is **provisional**, since four in-sample
  windows are a thinner tail than a production run's.
- **The determinism check pins one intra-op thread and bounds drift per column.** Drawing the page
  is the heaviest thing a run does outside the fit, and it exposed two float32 bounds that were
  tighter than the numbers supporting them: `train/source_conditioned_kl_raw` takes two values a
  part in $10^{5}$ apart depending on how a reduction is split across workers, so the subprocess
  runs at `OMP_NUM_THREADS=1`; and the `benchmark: true` drift bound is
  $\mathrm{atol} + \mathrm{rtol} \cdot |x|$ per column, because `train/grad_norm` moves by $0.13$ on
  a magnitude of $441$ while `train/pred_gap` moves by $6 \cdot 10^{-4}$ on a magnitude below $1$ —
  cancellation between two $243$-nat block scores — and one absolute bound is vacuous for the first
  and unmeetable for the second.
- **The page's window edges are deduplicated before drawing**, because at the evaluation resolution
  consecutive drawn windows abut and the shared boundary would otherwise be two lines at two alphas.
- **`anchor_stride` defaults to $1$, not to $H$.** The inert value, so a model constructed without an
  opinion behaves like the rest of the family; the tiling is a configuration decision and every
  shipped config states it.
- **`causal_reach_budget_s` is present and required `null`.** Not merely unnecessary but undefined
  here: it prunes channels by the forward reach of a *two-sided* Morlet, measured on a bank that did
  not produce these coefficients, and a delay is a shift. It is present rather than absent so the
  parity comparison against the raw-signal sibling stays leaf-for-leaf.
- **$\beta = 1.0$ / $\beta_p = 0.1$ is carried across unmeasured at this scale**, from a block that
  halved and an anchor count that fell by $24\times$. Resolved by the first real fit's
  `kld_active_frac` and `logvar_prior_floor_frac`, both of which the fixture fit already reports
  finite; a sweep is the follow-up, not a re-tune here.

### What moved in this revision, against the record it replaces

Four mechanisms arrived and two were deliberately declined. The four are §17's; what belongs here is
the sentences of the earlier record they replace, and the two declines with their reasons.

- **The lag attention's keys and values are no longer the deep source state**, and that encoder is
  no longer built. The earlier record described it as the module every arm carries; §8.1 replaces
  that, and §13's off-state row is where the earlier parameter totals still live.
- **The prior is conditioned on a function of $t$.** "The prior never sees the source" is restated
  as "the prior sees no function of the source *values*" (§8.2), which is what the source-purity
  tests now assert.
- **The reconstruction weights the horizon axis** (§5), so this row's objective has a weight for the
  first time — and, with no evaluation pipeline here, no unweighted second reading of the same
  quantity beside it.
- **The lag bias ships seeded flat.** `alibi_slope_scale: 0.0` against the constructor default
  $1.0$; no code moved, the shipped configuration did, and the decaying seed is a named arm.
- **The persistence residual is declined**, because it is a feature-domain answer to a
  feature-domain measurement and a raw persistence input is a different object (§14). Both
  raw-target constructors refuse the key by name.
- **The second alignment reference is declined**, because this row's target is raw with zero delay,
  so its single `causal_align_reference: 42.21` is already source-only (§14). §3's analysis of what
  this reference buys is untouched.
- **The training controls are on.** Early stopping is enabled at patience $50$ on `val/total_loss`,
  and a second `ModelCheckpoint` on `val/nll_full_block` is built behind a single optional monitor
  key, because the composite optimum and the best conditioned forecast are different epochs.

## 16. Running it

From the repository root.

```bash
# Production, 7 ranks. TEB_RUN_STAMP is required so ranks 1..N-1 share rank 0's run directory,
# and the rank count must equal len(general_config.cuda_devices). The shard paths in default.yaml
# are REPOINT_ME placeholders until the causal production shards exist.
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_crws.trainer --config teb_vae/lag_attn_crws/configs/default.yaml

# Local smoke: one epoch, one device, the committed causal fixture.
python -m teb_vae.lag_attn_crws.trainer --config teb_vae/lag_attn_crws/configs/tiny.yaml

# The instrumented run the two loss-scale constants were measured on: shipped widths, the
# committed fixture, the clip parked at 1e9.
python -m teb_vae.lag_attn_crws.trainer --config teb_vae/lag_attn_crws/configs/smoke_causal.yaml

# The two arms.
python -m teb_vae.lag_attn_crws.trainer --config teb_vae/lag_attn_crws/configs/sweep_anchor_stride_1.yaml
python -m teb_vae.lag_attn_crws.trainer --config teb_vae/lag_attn_crws/configs/sweep_horizon_15.yaml
```

`RUN_CONFIG` near the bottom of `trainer.py` names the config used when the module is launched with
no command line, so the entry point works from an IDE's Run button with the only operator action being
to edit a value inside the file; a `--config` on the command line always wins, and a relative path
resolves against the repository root rather than the working directory. Note that a Run-button launch
of `default.yaml` is a *single* process whose seven `cuda_devices` make the framework spawn DDP
workers underneath it. There is no `eval` entry point and no `check_run` entry point (§14).

The gate:

```bash
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_crws/tests -q -m "not slow"
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_crws/tests -q -m slow
```

And the standing check that no shipped module moved: `scripts/print_objective_metrics.py` prints
every objective metric of the four shipped forecasters in about a minute, and its `MODELS` tuple is
deliberately not extended with this cell — it exists to prove shared-seam edits inert for those four,
and adding a fifth would change the recorded hash and destroy that gate's whole value.

---

## 17. The switch inventory

Four mechanisms, each a key whose **off-state reproduces the pre-revision model bitwise**. That is
the safety rail rather than a convenience: it is what makes any single mechanism removable by one
YAML line without reverting code, what makes the arms comparable, and what the construction tests
assert against — state-dict key for state-dict key, parameter for parameter.

| Key | Shipped | Off-state | What it decides |
|---|---|---|---|
| `lag_kv_source` | `conv_stem` | `encoder` | which source representation the lag attention's keys **and** values are built from, and whether the deep source encoder is built at all (§8.1) |
| `prior_availability_input` | `true` | `false` | whether the prior head is conditioned on the source pathway's encode of silence (§8.2) |
| `horizon_weight_halflife_steps` | `15.0` | `null` | whether the reconstruction weights the horizon axis, and with what half-life (§5) |
| `alibi_slope_scale` | `0.0` | `1.0` | whether the learnable lag bias is seeded flat or with a monotone decay towards lag $0$ (§8.1) |

All four are constructor keywords, so all four land in the checkpoint's `model_kwargs` and are what
a checkpoint is rebuilt through. **Two keys the feature-target cells carry are deliberately absent**
— `persistence_residual` and `causal_align_reference_source` — and §14 records why each; both are
refused by name rather than silently ignored, so a config carrying one fails at the key.

**Three properties of the table are load-bearing.**

**An unknown value is refused by name.** `lag_kv_source` names its three admissible values in the
refusal. A silently accepted value is the failure mode the refusal exists to prevent.

**A key absent from this cell's constructor is dropped in silence.** The driver builds a run's
kwargs by sweeping `inspect.signature(MODEL_CLS.__init__)`, so a key present in the parent and
forgotten in this cell's own signature would leave the arm training as the baseline with nothing in
its log saying so. All four are therefore written out in this cell's signature as well as the
parent's, and the resolved-config artifact carries the configured values beside the startup log's
resolved consequences.

**Every new parameter is reachable in the graph.** The clock projection is added unconditionally in
the prior head's forward when the flag is on, and the lag bias exists whether or not its seed is
flat, so `find_unused_parameters=False` still holds. The horizon weight is a **non-persistent
buffer** and therefore absent from the state dict, which is deliberate: a checkpoint stays loadable
across horizons.
