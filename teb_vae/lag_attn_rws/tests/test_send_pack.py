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
