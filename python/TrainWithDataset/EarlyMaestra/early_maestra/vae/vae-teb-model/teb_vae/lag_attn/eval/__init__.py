r"""Offline evaluation of a trained :class:`~teb_vae.lag_attn.nets.model.SeqVaeLagAttn`.

Everything here is a *consumer* of the model's contracts, never a co-author of them: nothing
in this package modifies ``nets/``, ``task.py`` or ``trainer.py``. A run answers, for one
checkpoint: does the forecast work, does the source pathway carry information, is that
information source-specific, at what lag does it arrive, and is the latent actually used.

Run it from the repository root:

.. code-block:: bash

    python -m teb_vae.lag_attn.eval.run \
        --config teb_vae/lag_attn/eval/configs/eval.yaml \
        --checkpoint /path/to/lag-attn-epoch=412.ckpt

The package is self-contained by requirement, not by convention: ``teb_vae`` may not import
``model``, and ``train/tests/test_layering.py`` AST-walks every file here to enforce it,
lazy in-function imports included. The tree it replaces is a historical reference, never an
import source.

Layout, flat by design -- variants are configs, not subclasses:

* ``runner.py``    checkpoint to model, batch dispatch, the one inference-mode context
* ``preflight.py`` hard-fail guards, interpretation preconditions, the load health probe
* ``report.py``    ``summary.json`` assembly and the fail-soft step wrapper
* ``numerics.py``  the pinned numeric environment (fp32, no TF32, seeded)
* ``config_schema.py`` validation of the ``eval_config`` block, before anything is built
* ``analyses/``    one module per question the pipeline answers
* ``configs/``     ``eval.yaml``, chained off the training config
* ``tests/``       hermetic, against the committed tiny fixture, in the fast gate

No module here re-exports another's names. An analysis imports what it needs by module, so
the import graph says what actually depends on what.
"""
