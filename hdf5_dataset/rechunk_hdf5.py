r"""Re-chunk an existing HDF5 shard to one sample per chunk, for random single-sample reads.

The training loader serves one random sample per read, and HDF5 decompresses a whole chunk to
deliver any part of it. A shard whose datasets are chunked $n$ samples deep therefore decompresses
$n$ samples for every one it serves, and under a shuffled sampler over hundreds of thousands of
samples the per-file chunk cache almost never hits. This tool copies a shard dataset by dataset
with a chunk of one sample along the leading axis for every dataset of two or more dimensions,
so a random read costs exactly the bytes it returns.

**One-dimensional datasets keep the source chunking.** ``epoch``, ``guid`` and the labels are read
whole when the loader builds its sample index, and a one-element chunk would turn that single
column read into one B-tree lookup per sample.

**Nothing downstream reads the chunk layout.** The loader, the warm-up resolvers and the
statistics pairing read dataset names, shapes, dtypes and attributes, all of which this copy
preserves verbatim; the root attributes, every per-dataset attribute and the resizable
``maxshape`` are carried over unchanged, so the converted file is the same dataset under the
same names and stays appendable.

Compression is a parameter. Uncompressed per-sample chunks measure fastest to read; LZF per-sample
chunks read within a few percent of that and are about a fifth smaller. Pass ``--compression lzf``
when disk is the constraint.

Run from the command line::

    python hdf5_dataset/rechunk_hdf5.py SRC DST [--compression lzf] [--rows-per-pass N]

or edit :data:`SRC`, :data:`DST` and :data:`COMPRESSION` below and run this file from the IDE.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Union

import h5py

#: Run-button inputs: the shard to convert, where to write it, and the codec (``None`` or
#: ``"lzf"``). Edit these and run the file with no arguments.
SRC = r"C:\path\to\shard.hdf5"
DST = r"C:\path\to\rechunked\shard.hdf5"
COMPRESSION: Optional[str] = None

#: Samples copied per pass, bounding memory at a few tens of megabytes per dataset.
ROWS_PER_PASS = 1024


def rechunk(
    src: Union[str, Path],
    dst: Union[str, Path],
    *,
    compression: Optional[str] = None,
    rows_per_pass: int = ROWS_PER_PASS,
) -> Path:
    """Copy a shard with one-sample chunks on every dataset of two or more dimensions.

    Args:
        src: The shard to read.
        dst: The file to write. Refused if it already exists.
        compression: h5py compression name for the numeric datasets, or ``None`` for none. The
            object-typed ``guid`` dataset is never compressed, matching the writer.
        rows_per_pass: Samples copied per pass along the leading axis.

    Returns:
        The destination path.

    Raises:
        FileExistsError: If ``dst`` exists, naming it.
    """
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise FileExistsError(f"refusing to overwrite existing file {dst}; delete it or pick another path")
    with h5py.File(src, "r") as source, h5py.File(dst, "w", libver="latest") as target:
        target.attrs.update(source.attrs)
        for name, dataset in source.items():
            chunks = (1, *dataset.shape[1:]) if dataset.ndim >= 2 else dataset.chunks
            copy = target.create_dataset(
                name,
                shape=dataset.shape,
                dtype=dataset.dtype,
                maxshape=dataset.maxshape,
                chunks=chunks,
                compression=None if dataset.dtype.kind == "O" else compression,
            )
            copy.attrs.update(dataset.attrs)
            for start in range(0, dataset.shape[0], rows_per_pass):
                stop = min(start + rows_per_pass, dataset.shape[0])
                copy[start:stop] = dataset[start:stop]
    return dst


def main(argv: Optional[list] = None) -> int:
    """Command-line entry; with no arguments, converts :data:`SRC` to :data:`DST`.

    Args:
        argv: Arguments to parse, or ``None`` for ``sys.argv``.

    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(description="Re-chunk an HDF5 shard to one sample per chunk.")
    parser.add_argument("src", nargs="?", default=SRC)
    parser.add_argument("dst", nargs="?", default=DST)
    parser.add_argument("--compression", default=COMPRESSION, choices=[None, "lzf", "gzip"])
    parser.add_argument("--rows-per-pass", type=int, default=ROWS_PER_PASS)
    args = parser.parse_args(argv)
    written = rechunk(
        args.src, args.dst, compression=args.compression, rows_per_pass=args.rows_per_pass
    )
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
