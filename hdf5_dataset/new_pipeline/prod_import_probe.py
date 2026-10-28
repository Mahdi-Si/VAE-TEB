r"""Report how this interpreter resolves the imports of the dataset build.

Run it on the production box with the interpreter that runs ``build_fold_datasets.py``, from this
folder, with the IDE's Run button or ``python prod_import_probe.py``. It needs no argument. It
writes ``prod_import_report.txt`` next to this file, with:

* the interpreter, the working directory, ``PYTHONPATH`` and the active environment;
* ``sys.path`` in order, as the script starts;
* the ``.pth`` files of every site-packages folder;
* the file each module name of the build resolves to, after the repository root is added to
  ``sys.path`` exactly as ``build_fold_datasets.py`` adds it;
* the result of a full import of ``create_new_pipeline`` and ``build_fold_datasets``, with the
  traceback of a failure, and the ``signal_quality`` module that the import used.

A name such as ``signal_quality`` or ``launch`` that resolves to an old flat copy, instead of the
file in the repository, shadows the repository code. The report lists paths and module names only,
never data.
"""
import importlib
import importlib.util
import os
import site
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: ``hdf5_dataset/new_pipeline/prod_import_probe.py`` -> up two.
REPO_ROOT = HERE.parents[1]
#: Every module name the build imports, under each name it may be found by.
MODULES = [
    "early_maestra",
    "early_maestra.adaptor.mimo_adaptor",
    "Variational_AutoEncoder",
    "Variational_AutoEncoder.seqvae_teb",
    "Variational_AutoEncoder.seqvae_teb.hdf5_dataset",
    "Variational_AutoEncoder.seqvae_teb.hdf5_dataset.causal_scattering",
    "hdf5_dataset",
    "hdf5_dataset.signal_quality",
    "hdf5_dataset.new_pipeline.create_new_pipeline",
    "teb_vae",
    "teb_vae.lag_attn_rws.eval.launch",
    "signal_quality",
    "launch",
    "create_new_pipeline",
    "quality_report",
    "kymatio",
]
#: The build modules imported in full.
BUILD_MODULES = [
    "hdf5_dataset.new_pipeline.create_new_pipeline",
    "hdf5_dataset.new_pipeline.build_fold_datasets",
]
REPORT_NAME = "prod_import_report.txt"


def _origin(name: str) -> str:
    """Return where ``name`` resolves: its file, its package folders, or why it does not."""
    try:
        spec = importlib.util.find_spec(name)
    except Exception as exc:  # noqa: BLE001 -- every failure is part of the report
        return f"ERROR {type(exc).__name__}: {exc}"
    if spec is None:
        return "NOT FOUND"
    return str(spec.origin or list(spec.submodule_search_locations or []))


def main() -> Path:
    """Write the report and return its path."""
    lines = []
    out = lines.append

    out(f"executable: {sys.executable}")
    out(f"version: {sys.version}")
    out(f"cwd: {os.getcwd()}")
    out(f"this file: {Path(__file__).resolve()}")
    out(f"repo root: {REPO_ROOT}")
    for name in ("PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV"):
        out(f"{name}: {os.environ.get(name)}")

    out("\n== sys.path at start ==")
    out("\n".join(f"[{index}] {entry}" for index, entry in enumerate(sys.path)))

    out("\n== .pth files ==")
    folders = list(getattr(site, "getsitepackages", lambda: [])()) + [site.getusersitepackages()]
    for folder in folders:
        if not Path(folder).is_dir():
            continue
        for pth in sorted(Path(folder).glob("*.pth")):
            rows = pth.read_text(errors="replace").splitlines()[:20]
            out(f"{pth}:\n    " + "\n    ".join(rows))

    # The same step build_fold_datasets.py takes when it runs as a script.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    out("\n== module origins (after adding the repo root, as the build does) ==")
    for name in MODULES:
        out(f"{name}: {_origin(name)}")

    out("\n== full import of the build modules ==")
    for name in BUILD_MODULES:
        try:
            module = importlib.import_module(name)
            out(f"{name}: OK {module.__file__}")
        except Exception:  # noqa: BLE001 -- the traceback is the point
            out(f"{name}: FAILED\n{traceback.format_exc()}")
    quality = sys.modules.get("signal_quality") or sys.modules.get("hdf5_dataset.signal_quality")
    if quality is not None:
        out(f"signal_quality used: {quality.__file__}, "
            f"QUALITY_VERSION={getattr(quality, 'QUALITY_VERSION', '?')}")

    report = HERE / REPORT_NAME
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(f"wrote {main()}")
