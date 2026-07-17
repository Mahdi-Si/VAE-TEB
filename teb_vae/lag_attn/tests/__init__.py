"""Tests for the lag-attention VAE.

This file is not incidental. There is no repository-root ``conftest.py`` or ``pytest.ini``, so
without it pytest's prepend import mode would key modules by bare basename and collide with
same-named modules elsewhere in the repository (``test_perm_kl``, ``test_config``, ``test_trainer``
all exist under ``model/``), raising ``import file mismatch`` on a repository-wide run.

It also completes the package chain up to ``teb_vae``, which makes pytest walk to the first
non-package ancestor -- the repository root -- and put that on ``sys.path``, which is what the
absolute imports here need.
"""
