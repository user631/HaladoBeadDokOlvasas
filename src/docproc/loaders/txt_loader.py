from __future__ import annotations

import codecs
from pathlib import Path

from docproc.config import TEXT_ENCODINGS
from docproc.document import Document
from docproc.loaders.base import BaseLoader
from docproc.loaders.registry import register_loader


def decode_bytes(raw: bytes, encodings: tuple[str, ...] = TEXT_ENCODINGS) -> tuple[str, str]:
    if raw.startswith(codecs.BOM_UTF8):
        return raw[len(codecs.BOM_UTF8):].decode("utf-8", errors="replace"), "utf-8-sig"
    for encoding in encodings:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8 (hibás karakterek cserélve)"


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


@register_loader(".txt", ".text", ".md")
class TxtLoader(BaseLoader):
    def __init__(self, encodings: tuple[str, ...] = TEXT_ENCODINGS):
        self.encodings = encodings

    def _load(self, path: Path) -> Document:
        text, encoding = decode_bytes(path.read_bytes(), self.encodings)
        text = normalize_newlines(text)
        metadata = {
            "encoding": encoding,
            "empty_lines": sum(1 for line in text.splitlines() if not line.strip()),
        }
        return Document(path=path, text=text, file_type="txt", metadata=metadata)
