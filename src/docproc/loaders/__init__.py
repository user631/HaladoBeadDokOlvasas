from docproc.loaders.base import BaseLoader
from docproc.loaders.registry import (
    get_loader,
    is_supported,
    iter_documents,
    iter_files,
    load_document,
    register_loader,
    supported_extensions,
    unregister_loader,
)
from docproc.loaders.txt_loader import TxtLoader
from docproc.loaders.csv_loader import CsvLoader
from docproc.loaders.pdf_loader import PdfLoader

__all__ = [
    "BaseLoader",
    "CsvLoader",
    "PdfLoader",
    "TxtLoader",
    "get_loader",
    "is_supported",
    "iter_documents",
    "iter_files",
    "load_document",
    "register_loader",
    "supported_extensions",
    "unregister_loader",
]
