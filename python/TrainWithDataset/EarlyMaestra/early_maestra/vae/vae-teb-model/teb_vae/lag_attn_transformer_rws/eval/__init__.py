r"""The evaluation pipeline for the causal conv-Transformer lag-attention VAE.

**Almost none of it is here, and that is the design.** ``SeqVaeLagAttnTrfRws`` is
``SeqVaeLagAttnRws`` with both history encoders replaced and nothing else changed -- the same
objective, the same geometry, the same data contract, the same prior head, the same lag
cross-attention, the same shared horizon decoder. Two architectures are only comparable if they
are measured by one implementation, so this package supplies what the sibling pipeline cannot
derive and imports the rest.

Imported from ``teb_vae.lag_attn_rws.eval``:

* the runner (``run.main``), the collection pass, the preflight guards, the loader probe;
* seventeen of the eighteen registered analyses, every statistic, every identity check and every
  figure primitive;
* the configuration schema, the cohort grid, the report and figure seams.

Local, because each is a fact about *this* model that no shared module can compute:

* :mod:`.binding` -- ``TRF_BINDING``: the classes to rebuild, the constructor keys reconciled
  against a checkpoint, this encoder's own causality disclosure, this package's override delta,
  and the one analysis only this architecture can have;
* :mod:`.encoder_attention` and :mod:`.analyses.encoder_attention` -- the eighteenth analysis: it
  recomputes the encoder self-attention the model attends through but never materialises, and
  reports what each head attends to, how far, and how that compares with the lag range. There is
  nothing for it to answer in a model whose history encoders are recurrent, which is why it is
  here rather than in the shared registry;
* :mod:`.run` and :mod:`.probe` -- thin entry points carrying this package's ``prog=`` strings;
* :mod:`.verify` -- the sibling's acceptance gate delegated to unchanged, plus what is genuinely
  local: the sweep axes *this* model ships arms for, and the cross-model table that puts runs of
  both architectures side by side under the selection rule they must be read with;
* ``configs/eval_overrides.yaml`` -- the holdout split and the evaluation-only settings.

Launch from the repository root::

    python -m teb_vae.lag_attn_transformer_rws.eval.probe --config <run>/model_checkpoints/resolved_config.yaml
    python -m teb_vae.lag_attn_transformer_rws.eval.run --checkpoint <run>/model_checkpoints/<name>.ckpt
    python -m teb_vae.lag_attn_transformer_rws.eval.verify <run>/eval_results/summary.json
    python -m teb_vae.lag_attn_transformer_rws.eval.verify --runs <dir-of-runs> --out RESULTS_arms.md
"""
