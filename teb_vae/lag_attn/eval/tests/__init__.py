"""Tests for the evaluation pipeline.

This file is load-bearing rather than decorative. The repository has no root
``conftest.py`` and no ``pytest.ini``, so pytest runs in its default *prepend* import mode,
which keys a test module by its bare basename when the containing directory is not a
package. ``test_config.py`` already exists elsewhere in this repository; without this
``__init__.py`` the two would collide and collection would fail with an import-file mismatch
naming neither culprit clearly.

The suite is hermetic: it builds its own tiny checkpoint and reads only the committed
``teb_vae/lag_attn/tests/fixtures/tiny_shard.hdf5``, so it runs in the fast gate.
"""
