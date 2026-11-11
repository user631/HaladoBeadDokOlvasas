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


def test_search(sample_dir: Path, capsys):
    assert main(["search", str(sample_dir), "számla"]) == 0
    output = capsys.readouterr().out
    assert "1 találat" in output
    assert "level.txt" in output
    assert "[számla]" in output


def test_search_without_hits(sample_dir: Path, capsys):
    assert main(["search", str(sample_dir), "nemletezoszo"]) == 1
    assert "0 találat" in capsys.readouterr().out


def test_search_invalid_query(sample_dir: Path, capsys):
    assert main(["search", str(sample_dir), "(számla"]) == 2
    assert "zárójel" in capsys.readouterr().err


def test_search_regex(sample_dir: Path, capsys):
    assert main(["search", str(sample_dir), r"B\w+a", "--regex"]) == 0
    assert "adatok.csv:3" in capsys.readouterr().out


def test_count(sample_dir: Path, capsys):
    assert main(["count", str(sample_dir), "számla", "jegyzet"]) == 0
    output = capsys.readouterr().out
    assert "összesen" in output
    assert "jegyzet" in output


def test_extract(tmp_path: Path, capsys):
    path = tmp_path / "szamla.txt"
    path.write_text("Számlaszám: 2026/0142\nE-mail: info@example.com\nFizetendő: 138 049 Ft", encoding="utf-8")
    csv_path = tmp_path / "ki" / "adatok.csv"
    assert main(["extract", str(path), "--csv", str(csv_path)]) == 0
    output = capsys.readouterr().out
    assert "info@example.com" in output
    assert "138 049 HUF" in output
    assert csv_path.exists()


def test_extract_selected_kinds_json(tmp_path: Path, capsys):
    path = tmp_path / "a.txt"
    path.write_text("Kapcsolat: info@example.com, +36 1 234 5678", encoding="utf-8")
    assert main(["extract", str(path), "--kinds", "phone", "--keywords", "0", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["a.txt"]["entities"] == {"phone": ["+36 1 234 5678"]}
    assert payload["a.txt"]["keywords"] == []


def test_extract_unknown_kind(sample_dir: Path, capsys):
    assert main(["extract", str(sample_dir), "--kinds", "nincs"]) == 2


def test_formats(capsys):
    assert main(["formats"]) == 0
    assert ".pdf" in capsys.readouterr().out
