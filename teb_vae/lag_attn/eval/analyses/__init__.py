"""One module per question the pipeline answers.

Each analysis is a plain function taking a runner and a loader and writing its own outputs
into a subdirectory of the run directory. They do not import one another: an analysis that
needs something another produced takes it as an argument, so the call site in ``run.py`` shows
the whole dependency graph.

``probe`` is the exception in ordering, not in kind. It runs first, before any other analysis,
because it is the pipeline's only real input validator -- and because the per-file composition
it records is what every stratified capped draw stratifies over.
"""
