"""Copy an eval output folder without its figures into a sibling ``<name>_send`` folder.

The copy keeps the folder structure and every table, record and array; figures and pages are left
out. Zip the ``_send`` folder to hand a run over for reading.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

#: Figures, pages and their assets: drawn from the data that is copied, so not needed to read a run.
FIGURE_SUFFIXES = frozenset({".png", ".pdf", ".svg", ".jpg", ".jpeg", ".gif", ".eps", ".html", ".js"})
#: Working files that are not results (the lag analysis's resume file).
SKIP_NAMES = frozenset({"progress.state", "progress.state.tmp"})
#: Files above this size are listed in SKIPPED.txt instead of copied.
MAX_BYTES = 500 * 2**20


def pack_for_sending(out_dir: str | os.PathLike) -> Path:
    """Copy ``out_dir`` to ``<out_dir>_send`` without figures, and return the new folder.

    A file already in the ``_send`` folder is overwritten; nothing else there is removed.
    """
    src = Path(out_dir).resolve()
    dst = src.with_name(src.name + "_send")
    too_big = []

    def ignore(folder: str, names: list[str]) -> set[str]:
        skipped = set()
        for name in names:
            path = Path(folder, name)
            if path.is_dir():
                continue
            if path.suffix.lower() in FIGURE_SUFFIXES or name in SKIP_NAMES:
                skipped.add(name)
            elif path.stat().st_size > MAX_BYTES:
                skipped.add(name)
                too_big.append(f"{path.relative_to(src)}\t{path.stat().st_size / 2**20:.0f} MB")
        return skipped

    shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)
    if too_big:
        (dst / "SKIPPED.txt").write_text(
            "Files left out because they are larger than "
            f"{MAX_BYTES // 2**20} MB (path in {src.name}, size):\n" + "\n".join(too_big) + "\n"
        )
    return dst


if __name__ == "__main__":
    import sys

    print(pack_for_sending(sys.argv[1]))
