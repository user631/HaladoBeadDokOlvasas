from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from pathlib import Path

from docproc.document import Document
from docproc.exceptions import DocProcError, DocumentLoadError, UnsupportedFormatError
from docproc.loaders.base import BaseLoader

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[BaseLoader]] = {}


def _normalize_extension(extension: str) -> str:
    extension = extension.strip().lower()
    return extension if extension.startswith(".") else f".{extension}"


def register_loader(*extensions: str) -> Callable[[type[BaseLoader]], type[BaseLoader]]:
    if not extensions:
        raise ValueError("Legalább egy fájlkiterjesztést meg kell adni.")
    normalized = tuple(_normalize_extension(ext) for ext in extensions)

    def decorator(cls: type[BaseLoader]) -> type[BaseLoader]:
        if not (isinstance(cls, type) and issubclass(cls, BaseLoader)):
            raise TypeError(f"{cls!r} nem a BaseLoader leszármazottja.")
        for extension in normalized:
            _REGISTRY[extension] = cls
        cls.extensions = normalized
        return cls

    return decorator


def unregister_loader(extension: str) -> None:
    _REGISTRY.pop(_normalize_extension(extension), None)


def supported_extensions() -> list[str]:
    return sorted(_REGISTRY)


def is_supported(path: str | Path) -> bool:
    return Path(path).suffix.lower() in _REGISTRY


def get_loader(path: str | Path) -> BaseLoader:
    path = Path(path)
    loader_class = _REGISTRY.get(path.suffix.lower())
    if loader_class is None:
        raise UnsupportedFormatError(path)
    return loader_class()


def load_document(path: str | Path) -> Document:
    return get_loader(path).load(path)


def iter_files(directory: str | Path, recursive: bool = True) -> Iterator[Path]:
    directory = Path(directory)
    pattern = "**/*" if recursive else "*"
    for path in sorted(directory.glob(pattern)):
        hidden = any(part.startswith((".", "~$")) for part in path.relative_to(directory).parts)
        if hidden or not path.is_file():
            continue
        if is_supported(path):
            yield path


def iter_documents(
    source: str | Path,
    recursive: bool = True,
    skip_errors: bool = True,
) -> Iterator[Document]:
    source = Path(source)
    if source.is_file():
        yield load_document(source)
        return
    if not source.is_dir():
        raise DocumentLoadError(source, "a megadott útvonal nem létezik")
    for path in iter_files(source, recursive=recursive):
        try:
            yield load_document(path)
        except DocProcError as exc:
            if not skip_errors:
                raise
            logger.warning("%s", exc)
