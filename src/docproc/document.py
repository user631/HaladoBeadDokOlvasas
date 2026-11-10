from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class Document:
    path: Path
    text: str = field(repr=False)
    file_type: str
    pages: list[str] = field(default_factory=list, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)
    table: pd.DataFrame | None = field(default=None, repr=False, compare=False)
    loaded_at: datetime = field(default_factory=datetime.now, compare=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size if self.path.exists() else 0

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def line_count(self) -> int:
        return len(self.text.splitlines())

    @property
    def page_count(self) -> int:
        return len(self.pages) if self.pages else 1

    def preview(self, length: int = 200) -> str:
        compact = " ".join(self.text.split())
        if len(compact) <= length:
            return compact
        return compact[:length].rstrip() + "..."

    def summary(self) -> dict[str, Any]:
        return {
            "név": self.name,
            "típus": self.file_type,
            "méret (bájt)": self.size_bytes,
            "oldalak": self.page_count,
            "sorok": self.line_count,
            "szavak": self.word_count,
            "karakterek": self.char_count,
        }
