"""The dataset builder, as a package.

Empty, and present so that ``hdf5_dataset.new_pipeline.create_new_pipeline`` is an importable
dotted name. Without it the directory is not a subpackage of the regular package
``hdf5_dataset`` at all, and the launch-convention test -- which imports every runner by
module path -- could not reach the builder to check it.
"""
