from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from docproc.document import Document
from docproc.exceptions import DocumentLoadError, EmptyDocumentError
from docproc.loaders.base import BaseLoader
from docproc.loaders.registry import register_loader

_HYPHENATED_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


@register_loader(".pdf")
class PdfLoader(BaseLoader):
    def _load(self, path: Path) -> Document:
        try:
            reader = PdfReader(path)
        except (PdfReadError, ValueError, OSError) as exc:
            raise DocumentLoadError(path, f"sérült vagy nem olvasható PDF ({exc})") from exc

        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentLoadError(path, "jelszóval védett PDF")

        pages = [self.clean_text(page.extract_text() or "") for page in reader.pages]
        text = "\n\n".join(page for page in pages if page)
        if not text.strip():
            raise EmptyDocumentError(path, "a PDF nem tartalmaz kinyerhető szöveget (szkennelt dokumentum?)")

        metadata = self.read_metadata(reader)
        metadata["page_count"] = len(pages)
        pages_without_text = [number for number, page in enumerate(pages, start=1) if not page]
        if pages_without_text:
            metadata["pages_without_text"] = pages_without_text
        return Document(path=path, text=text, file_type="pdf", pages=pages, metadata=metadata)

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = _HYPHENATED_BREAK.sub(r"\1\2", text)
        lines = [_MULTI_SPACE.sub(" ", line).strip() for line in text.split("\n")]
        return _MULTI_NEWLINE.sub("\n\n", "\n".join(lines)).strip()

    @staticmethod
    def read_metadata(reader: PdfReader) -> dict[str, Any]:
        try:
            info = reader.metadata
        except PdfReadError:
            return {}
        if info is None:
            return {}
        metadata: dict[str, Any] = {}
        for key in ("title", "author", "subject", "creator", "producer"):
            value = getattr(info, key, None)
            if value:
                metadata[key] = str(value)
        try:
            created = info.creation_date
        except (PdfReadError, ValueError):
            created = None
        if created is not None:
            metadata["creation_date"] = created.isoformat()
        return metadata
