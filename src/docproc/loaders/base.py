from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from docproc.document import Document
from docproc.exceptions import DocProcError, DocumentLoadError, EmptyDocumentError


class BaseLoader(ABC):
    extensions: tuple[str, ...] = ()

    def load(self, path: str | Path) -> Document:
        path = Path(path)
        if not path.is_file():
            raise DocumentLoadError(path, "a fájl nem található")
        try:
            document = self._load(path)
        except DocProcError:
            raise
        except Exception as exc:
            raise DocumentLoadError(path, str(exc)) from exc
        if not document.text.strip():
            raise EmptyDocumentError(path)
        document.metadata.setdefault("loader", type(self).__name__)
        return document

    @abstractmethod
    def _load(self, path: Path) -> Document:
        raise NotImplementedError
