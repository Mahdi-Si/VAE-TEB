# `lag_attn_fs`: open tasks

Owner: Mahdi Si

The `lag_attn_fs` package is built, and `DESIGN.md` is its as-built record. This document holds
only the tasks that are not yet done. All of them are changes to the lag-attention preprint, so that
the preprint documents the feature-domain variant and the backward half of two-sidedness
(`DESIGN.md` §8).

The preprint is at `documents/latex_paper_lag_attn_rws/`. Paths below are relative to that
directory. Every task runs the build itself, and T07 is the final clean build.

## T01: Notation and macros

**Description.** Add the symbols that the new material needs, in the Project block of
`math_commands.tex` and in the tables of `appendix/notation.tex`.

**Acceptance criteria.**
- Resolve the symbol collisions explicitly. `R` is already both the decimation factor and the raw
  samples per token, `H` is the horizon, `\nheads` is $H_e$, `C` is not a symbol yet, and
  `\gA`/`\gB` are bound to the anchor set and the reach budget. The kept-channel count needs a name
  that collides with none of them. `appendix/parameters.tex` already uses $c^{\mathrm{kept}}$.
- Use every new macro at least once.
- Add rows to `tab:sym-signals` and `tab:sym-objective`, not only macro definitions.
- The build reports no `Command ... already defined`.

**Files.** `math_commands.tex`, `appendix/notation.tex`.

**Validation.** Run `latexmk`, then grep that each new macro appears in a section file.

## T02: `ss:filtersupport` in `reach.tex`

**Description.** Add a subsection between `ss:reachtrainability` and `ss:reachlimits` that covers
the backward half of two-sidedness.

**Acceptance criteria.**
- The subsection contains the blend fraction $b(\tau, \rho_c)$ with `\highlight` at its definition,
  the blend values of the production filter bank for the kept set and the full set, the
  not-a-leak argument, the zero-conditional-MI display that names the shared-decoder cancellation,
  and a rule to report the forecast gap by horizon step.
- The subsection does **not** restate the eight things that `reach.tex` already covers: the
  two-sidedness itself, the $L_{95}$ definition and its numbers, the quantile-not-support caveat,
  the delay construction, `tab:budgets`, the trainability argument, the one-sided-support
  requirement and the mixture-of-coordinates consequence.
- Amend `rem:lagreference`. It attributes the neighbourhood width only to the encoder's receptive
  field and never to the filter's own support, which is a second, independent source of the same
  blur.
- `\cref{ss:filtersupport}` resolves.

**Files.** `sections/reach.tex`, `sections/readouts.tex`, `sections/limits.tex`.

**Validation.** Run `latexmk`, then grep for the eight restated phrases.

## T03: `fig_filter_support`

**Description.** Add a TikZ figure that shows the two-sided coefficient window across the anchor at
several horizon steps, with the blend falling to zero.

**Acceptance criteria.**
- The file stem is `fig_filter_support`, because `fig_support` is the anchor-support cascade.
- The figure uses `\input{figstyle}` and the shared palette, with no per-figure style.
- The figure ships as `.tex`, `.pdf` and `.png`, and the `.pdf` is not older than the `.tex`.
- The caption is self-contained, and the text interprets the float.

**Files.** `assets/figures/fig_filter_support.{tex,pdf,png}`.

**Validation.** Build the figure standalone, then run `latexmk`.

## T04: `s:featuredomain`, and the front matter

**Description.** Add a self-contained section for the feature-domain variant, one abstract sentence,
a `tab:scope` row and the body-order comment.

**Acceptance criteria.**
- The section names the shared modules to state what does not change. It states what does change:
  the target, the decoder width and the block cardinality.
- The section states the unit consequence of $H \cdot C_{\mathrm{keep}}$ against $H \cdot R$
  explicitly, because `ss:units` currently claims a cross-architecture comparability that this
  variant breaks.
- The section states that `\archA`/`\archB` remain **encoder** names and that the feature domain is a
  target-domain axis, so that it does not contradict `appendix/notation.tex`.
- The section notes that `architecture.tex` records the `logvar_clamp` interval as inherited from a
  decoder that emitted feature coefficients.
- The abstract's statement that the model "predicts raw FHR at $4$~Hz" becomes conditional on the
  variant. The title does not change.
- `s:featuredomain` appears in the table of contents, and the `\cref` to `ss:units` resolves.

**Files.** `sections/featuredomain.tex`, `main.tex`, `sections/introduction.tex`.

**Validation.** Run `latexmk`, then grep the abstract for the unconditional claim.

## T05: `appendix/configuration.tex`

**Description.** Add the variant's keys to the config tables.

**Acceptance criteria.**
- `tab:config-geometry` records that `raw_per_step` remains a geometry input and is no longer the
  decoder width.
- `a:absentkeys` gets no entry for `raw_per_step`, because the key is not absent.
- The per-architecture `\midrule` blocks take a variant block additively.

**Files.** `appendix/configuration.tex`.

**Validation.** Run `latexmk`.

## T06: The citation, and the `filemap` decision

**Description.** `references.bib` holds `schreiber2000measuring`, but no file cites it, while
`main.tex`, `sections/introduction.tex`, `sections/readouts.tex` and `sections/limits.tex` discuss
transfer entropy. Cite it. Separately, decide what to do with `appendix/filemap.tex`. No file
`\input`s it, it has two dangling `\cref`s, and it is the only place that lists module names.

**Acceptance criteria.**
- The text cites Schreiber at the definition of transfer entropy.
- Either revive `filemap` (fix the dangling references, add a row for `lag_attn_fs`, wire it into
  `main.tex`) or delete it, and record the choice.
- Add no invented citations.

**Files.** `sections/limits.tex` or `sections/readouts.tex`, `appendix/filemap.tex`, `main.tex`.

**Validation.** Run `latexmk`, then check that the bibliography lists Schreiber.

## T07: The clean build

**Description.** Rebuild the document fully and report the result.

**Acceptance criteria.**
- Run `latexmk -C && latexmk` from the preprint directory, as that directory's `CLAUDE.md` says. A
  partial rebuild after a new citation reports a stale "Label(s) may have changed".
- The build has no undefined reference or citation, and no overfull box wider than about 15 pt.
- Report the page count as a delta against the page count before T01, and report any warning that
  is not a `latexfont` warning.
- Do not enable `-shell-escape`.

**Files.** None.

**Validation.** Run `latexmk -C && latexmk` in `documents/latex_paper_lag_attn_rws/`.

## Todo checklist

- [ ] T01: Notation and macros
- [ ] T02: `ss:filtersupport` in `reach.tex`
- [ ] T03: `fig_filter_support`
- [ ] T04: `s:featuredomain`, and the front matter
- [ ] T05: `appendix/configuration.tex`
- [ ] T06: The citation, and the `filemap` decision
- [ ] T07: The clean build

The code must not mention these tasks or this document.
