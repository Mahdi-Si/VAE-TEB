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

    The ``_send`` folder is rebuilt from scratch on every call, so a re-run never keeps a stale
    table or ``SKIPPED.txt`` from an earlier run. A file that cannot be read (for example a broken
    symbolic link) is left out and listed in ``SKIPPED.txt``. A folder left empty by the filter
    (for example a ``figures/`` folder) is removed.

    Args:
        out_dir: The evaluation output folder.

    Returns:
        The ``<out_dir>_send`` folder.

    Raises:
        ValueError: If ``out_dir`` is not an existing folder.
    """
    src = Path(out_dir).resolve()
    if not src.is_dir():
        raise ValueError(f"not a folder: {src}")
    dst = src.with_name(src.name + "_send")
    skipped_lines = []

    def ignore(folder: str, names: list[str]) -> set[str]:
        skipped = set()
        for name in names:
            path = Path(folder, name)
            if path.is_dir():
                continue
            if path.suffix.lower() in FIGURE_SUFFIXES or name in SKIP_NAMES:
                skipped.add(name)
                continue
            try:
                size = path.stat().st_size
            except OSError as error:
                skipped.add(name)
                skipped_lines.append(f"{path.relative_to(src)}\tunreadable ({type(error).__name__})")
                continue
            if size > MAX_BYTES:
                skipped.add(name)
                skipped_lines.append(f"{path.relative_to(src)}\t{size / 2**20:.0f} MB")
        return skipped

    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)
    for folder, _dirs, _files in sorted(os.walk(dst), key=lambda entry: -len(entry[0])):
        if folder != str(dst) and not os.listdir(folder):
            os.rmdir(folder)
    if skipped_lines:
        (dst / "SKIPPED.txt").write_text(
            "Files left out because they are larger than "
            f"{MAX_BYTES // 2**20} MB or cannot be read (path in {src.name}, reason):\n"
            + "\n".join(skipped_lines) + "\n"
        )
    return dst

if __name__ == "__main__":
    import sys

    print(pack_for_sending(sys.argv[1]))
