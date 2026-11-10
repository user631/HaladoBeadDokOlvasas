from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import pandas as pd

from docproc.config import CSV_DELIMITERS
from docproc.document import Document
from docproc.exceptions import DocumentLoadError
from docproc.loaders.base import BaseLoader
from docproc.loaders.registry import register_loader
from docproc.loaders.txt_loader import decode_bytes, normalize_newlines

_HU_NUMBER = re.compile(r"-?(?:\d{1,3}(?:[  ]\d{3})+|\d+)(?:,\d+)?")
_EN_NUMBER = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


@register_loader(".csv", ".tsv")
class CsvLoader(BaseLoader):
    sample_lines = 20

    def _load(self, path: Path) -> Document:
        raw_text, encoding = decode_bytes(path.read_bytes())
        raw_text = normalize_newlines(raw_text)
        if not raw_text.strip():
            return Document(path=path, text="", file_type="csv")

        delimiter = "\t" if path.suffix.lower() == ".tsv" else self.detect_delimiter(raw_text)
        try:
            table = pd.read_csv(io.StringIO(raw_text), sep=delimiter, dtype=str, skipinitialspace=True)
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise DocumentLoadError(path, f"hibás CSV szerkezet: {exc}") from exc

        table.columns = [str(column).strip() for column in table.columns]
        hungarian_style = delimiter == ";"
        for column in table.columns:
            table[column] = self.convert_numeric(table[column].str.strip(), hungarian_style)
        metadata = {
            "encoding": encoding,
            "delimiter": delimiter,
            "row_count": len(table),
            "column_count": len(table.columns),
            "columns": list(table.columns),
            "numeric_columns": table.select_dtypes("number").columns.tolist(),
            "missing_values": int(table.isna().sum().sum()),
        }
        return Document(
            path=path,
            text=self.table_to_text(table),
            file_type="csv",
            metadata=metadata,
            table=table,
        )

    @classmethod
    def detect_delimiter(cls, text: str) -> str:
        sample = "\n".join(text.splitlines()[: cls.sample_lines])
        try:
            return csv.Sniffer().sniff(sample, delimiters=CSV_DELIMITERS).delimiter
        except csv.Error:
            counts = {delimiter: sample.count(delimiter) for delimiter in CSV_DELIMITERS}
            best = max(counts, key=counts.get)
            return best if counts[best] else ","

    @staticmethod
    def convert_numeric(series: pd.Series, hungarian_style: bool) -> pd.Series:
        values = series.dropna()
        values = values[values != ""]
        if values.empty:
            return series
        pattern = _HU_NUMBER if hungarian_style else _EN_NUMBER
        if not values.str.fullmatch(pattern).all():
            return series
        if hungarian_style:
            cleaned = series.str.replace(r"[  ]", "", regex=True).str.replace(",", ".", regex=False)
        else:
            cleaned = series.str.replace(",", "", regex=False)
        return pd.to_numeric(cleaned.replace("", None))

    @staticmethod
    def table_to_text(table: pd.DataFrame) -> str:
        lines = [" | ".join(table.columns)]
        for row in table.itertuples(index=False):
            cells = ("" if pd.isna(value) else str(value) for value in row)
            lines.append(" | ".join(cells))
        return "\n".join(lines)
