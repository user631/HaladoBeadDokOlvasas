from __future__ import annotations

import json
from pathlib import Path

from docproc.cli import main


def test_load_directory(sample_dir: Path, capsys):
    assert main(["load", str(sample_dir)]) == 0
    output = capsys.readouterr().out
    for name in ("level.txt", "adatok.csv", "jelentes.pdf", "jegyzet.txt"):
        assert name in output
    assert "Összesen: 4 dokumentum" in output


def test_load_with_preview(sample_dir: Path, capsys):
    assert main(["load", str(sample_dir / "level.txt"), "--preview", "50"]) == 0
    assert "Tisztelt Ügyfelünk!" in capsys.readouterr().out


def test_load_json(sample_dir: Path, capsys):
    assert main(["load", str(sample_dir), "--no-recursive", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {item["név"] for item in payload} == {"level.txt", "adatok.csv", "jelentes.pdf"}


def test_load_missing_path_returns_error(tmp_path: Path, capsys):
    assert main(["load", str(tmp_path / "nincs")]) == 2
    assert "Hiba" in capsys.readouterr().err


def test_load_empty_directory(tmp_path: Path, capsys):
    assert main(["load", str(tmp_path)]) == 1


def test_stats_table(sample_dir: Path, capsys):
    assert main(["stats", str(sample_dir)]) == 0
    output = capsys.readouterr().out
    assert "olvashatóság" in output
    assert "Leggyakoribb szavak a teljes gyűjteményben" in output
    assert "Numerikus oszlopok – adatok.csv" in output


def test_stats_json(sample_dir: Path, capsys):
    assert main(["stats", str(sample_dir / "level.txt"), "--json", "--top", "3"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["level.txt"]["language"] in {"hu", "unknown"}
    assert len(payload["level.txt"]["top_words"]) <= 3


def test_stats_plot(sample_dir: Path, tmp_path: Path, capsys):
    figures_dir = tmp_path / "abrak"
    assert main(["stats", str(sample_dir), "--plot", "--figures-dir", str(figures_dir), "--prefix", "teszt"]) == 0
    names = {path.name for path in figures_dir.glob("*.png")}
    assert names == {
        "teszt_gyakori_szavak.png",
        "teszt_szohosszok.png",
        "teszt_mondathosszok.png",
        "teszt_terjedelem.png",
        "teszt_olvashatosag.png",
    }


def test_formats(capsys):
    assert main(["formats"]) == 0
    assert ".pdf" in capsys.readouterr().out
