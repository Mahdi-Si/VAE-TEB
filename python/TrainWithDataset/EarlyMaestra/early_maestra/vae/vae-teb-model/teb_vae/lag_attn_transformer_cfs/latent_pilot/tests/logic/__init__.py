"""The synthetic logic subset: fast, hermetic, and import-isolated.

Import isolation is the property that makes this directory runnable anywhere, and it is a rule about
what these tests may import rather than about what they assert. No module reachable from here may
construct a model, open a checkpoint, read an HDF5 file or import the production data pipeline, and
no ``conftest.py`` above this directory may do so on its behalf. A temporary directory holding a tiny
configuration file is acceptable; a generated dataset fixture is not.
"""
