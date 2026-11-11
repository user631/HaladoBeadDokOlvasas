from pathlib import Path


class DocProcError(Exception):
    pass


class UnsupportedFormatError(DocProcError):
    def __init__(self, path: Path):
        self.path = Path(path)
        suffix = self.path.suffix or "(nincs kiterjesztés)"
        super().__init__(f"Nem támogatott fájlformátum: {suffix} – {self.path}")


class DocumentLoadError(DocProcError):
    def __init__(self, path: Path, reason: str):
        self.path = Path(path)
        self.reason = reason
        super().__init__(f"Nem sikerült beolvasni: {self.path} – {reason}")


class QuerySyntaxError(DocProcError):
    pass


class EmptyDocumentError(DocumentLoadError):
    def __init__(self, path: Path, reason: str = "a dokumentum nem tartalmaz kinyerhető szöveget"):
        super().__init__(path, reason)
