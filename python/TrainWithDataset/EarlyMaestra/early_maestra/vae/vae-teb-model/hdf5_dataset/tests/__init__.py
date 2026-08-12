"""Tests for the dataset package: the causal transform, the writer, the loader and the statistics.

Everything here runs on the committed fixture under ``data/``, so the suite is meaningful on a
clean checkout and on the production box, neither of which has the git-ignored ``output/``
directory. Run from the repository root::

    .venv/Scripts/python.exe -m pytest hdf5_dataset -q

``hdf5_dataset/test_causal_scattering.py`` stays where it is: it is the filter-design suite, it
passes, and moving it would be unrelated churn.
"""
