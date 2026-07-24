"""Offline evaluation of a trained raw-signal lag-attention VAE checkpoint.

Two modules. :mod:`~teb_vae.lag_attn_rws.eval.metrics` computes the readouts and turns them into
explicit verdicts; :mod:`~teb_vae.lag_attn_rws.eval.run` is the command line that points those at
a checkpoint and writes ``summary.json``.

The scope is deliberately narrow: the minimum outputs the model's specification names, and
nothing else. Every number here comes from the same loss functions the training objective uses,
reached through the same task class, because an evaluation that re-implements the objective it is
evaluating measures its own re-implementation.
"""
