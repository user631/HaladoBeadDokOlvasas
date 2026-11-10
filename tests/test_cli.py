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


def test_formats(capsys):
    assert main(["formats"]) == 0
    assert ".pdf" in capsys.readouterr().out
