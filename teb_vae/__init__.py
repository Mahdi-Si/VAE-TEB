"""The model layer: model families built on the ``train/`` framework.

Layering in this repository runs one way::

    utils/  <-  train/  <-  teb_vae/

``utils/`` holds leaf helpers and imports no other first-party package. ``train/`` is the
model-agnostic training framework and may use ``utils/`` but never a model. ``teb_vae/`` is the
top layer: it uses both, and nothing above it imports it.

``teb_vae`` must not import ``model``. It is the tree that replaces ``model/``, not a consumer of
it, and the two are forked rather than chained -- a fix here must never depend on the tree it
supersedes. ``train/tests/test_layering.py`` asserts this by walking imports, so an upward or
backward import fails a test rather than passing review.

Each family is a subpackage (``teb_vae.lag_attn``). Code shared by two families would live in
``teb_vae/shared/``; it does not exist because nothing needs it yet.

There is no editable install and no packaging. ``teb_vae`` resolves from the repository root on
``sys.path``, exactly as ``train``, ``utils`` and ``model`` do, so entry points and pytest are run
from the repository root.
"""
