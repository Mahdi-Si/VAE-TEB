from teb_vae.lag_attn_rws.eval import send_pack


def test_send_folder_keeps_data_and_drops_figures_and_oversized_files(tmp_path, monkeypatch):
    out = tmp_path / "eval_results"
    (out / "forecast").mkdir(parents=True)
    (out / "summary.json").write_text("{}")
    (out / "forecast" / "table.csv").write_text("a\n1\n")
    (out / "forecast" / "plot.pdf").write_bytes(b"%PDF")
    (out / "dashboard.html").write_text("<html>")
    (out / "big.npz").write_bytes(b"x" * 64)
    monkeypatch.setattr(send_pack, "MAX_BYTES", 32)

    dst = send_pack.pack_for_sending(out)

    assert dst == tmp_path / "eval_results_send"
    files = sorted(p.relative_to(dst).as_posix() for p in dst.rglob("*") if p.is_file())
    assert files == ["SKIPPED.txt", "forecast/table.csv", "summary.json"]
    assert "big.npz" in (dst / "SKIPPED.txt").read_text()


def test_a_rerun_rebuilds_the_send_folder_and_drops_figure_only_folders(tmp_path, monkeypatch):
    """A second pack must not keep the first run's stale files, and a folder that held only
    figures must not survive as an empty folder."""
    out = tmp_path / "eval_results"
    (out / "figures").mkdir(parents=True)
    (out / "figures" / "plot.png").write_bytes(b"png")
    (out / "big.npz").write_bytes(b"x" * 64)
    monkeypatch.setattr(send_pack, "MAX_BYTES", 32)
    dst = send_pack.pack_for_sending(out)
    (dst / "stale.csv").write_text("old")
    assert (dst / "SKIPPED.txt").exists()

    (out / "big.npz").unlink()
    (out / "summary.json").write_text("{}")
    dst = send_pack.pack_for_sending(out)

    assert sorted(p.relative_to(dst).as_posix() for p in dst.rglob("*")) == ["summary.json"]


def test_an_unreadable_file_is_listed_rather_than_aborting_the_copy(tmp_path, monkeypatch):
    """A file whose ``stat`` fails (a broken symbolic link, say) is listed and the rest copied.

    The failure is injected rather than made with ``os.symlink``, which needs a privilege on
    Windows, so the test runs on every platform.
    """
    import errno
    from pathlib import Path

    out = tmp_path / "eval_results"
    out.mkdir()
    (out / "summary.json").write_text("{}")
    (out / "gone.npz").write_bytes(b"x")
    real_stat = Path.stat

    def stat(self, *args, **kwargs):
        if self.name == "gone.npz":
            raise FileNotFoundError(errno.ENOENT, "dangling link", str(self))
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    dst = send_pack.pack_for_sending(out)
    monkeypatch.undo()

    assert (dst / "summary.json").exists()
    assert not (dst / "gone.npz").exists()
    assert "gone.npz" in (dst / "SKIPPED.txt").read_text()
