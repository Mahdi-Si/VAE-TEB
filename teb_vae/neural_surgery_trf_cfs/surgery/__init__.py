r"""Gradient-free weight surgery of the epoch-524 checkpoint, in four rounds.

* ``round1_closed_form``: GLS mean head and persistence weights, WLS $\phi_c$, per-channel affine
  variance map and a lower decoder log-variance floor. Also the shared head state, solvers and
  the test-half evaluation used by every later round.
* ``round2_neurons``: scalar edits of the encoder and source path (all reverted), single-unit
  removal in the decoder, then a second head solve.
* ``round3_irls``: full Fisher-scoring re-solve of ``logvar_head`` and a lower floor; a KL-optimal
  prior-variance shift was measured and rejected.
* ``round4_latent``: undo the shared-clamp side effect on the latent variances and match the
  posterior variance to the prior's.
* ``make_v4a``: the round-4 checkpoint without the posterior variance match.
* ``tail_diagnostic``: where the 3-sigma innovation excess sits and whether any global variance
  inflation passes the calibration verdict.
"""
