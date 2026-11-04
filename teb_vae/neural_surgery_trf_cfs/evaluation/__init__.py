r"""Official evaluation of the original and the edited checkpoints on the test half.

* ``make_test_half_overrides``: an override file that keeps only the test recordings
  ($\operatorname{crc32}(\mathrm{guid}) \bmod 4 \in \{1, 3\}$).
* ``run_official_eval``: runs ``teb_vae.lag_attn_transformer_cfs.eval.run`` on each checkpoint.
* ``compare_official_eval``: puts the verdicts and the main readouts side by side.
"""
