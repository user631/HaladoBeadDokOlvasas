from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from docproc.config import RAW_DATA_DIR
from docproc.loaders import iter_documents

NEWSGROUP_CATEGORIES = {
    "comp.graphics": "technológia",
    "rec.autos": "autók",
    "rec.sport.hockey": "sport",
    "sci.med": "orvostudomány",
    "soc.religion.christian": "vallás",
    "talk.politics.misc": "politika",
}
MIN_TEXT_LENGTH = 50

_EMAIL_OR_URL = re.compile(r"\S+@\S+|https?://\S+|www\.\S+")
_QUOTE_MARKERS = re.compile(r"^[>|:]+.*$", re.MULTILINE)
_REPEATED_CHARS = re.compile(r"([^\w\s])\1{3,}")
_WHITESPACE = re.compile(r"\s+")


def clean_newsgroup_text(text: str) -> str:
    text = _QUOTE_MARKERS.sub(" ", text)
    text = _EMAIL_OR_URL.sub(" ", text)
    text = _REPEATED_CHARS.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def prepare_dataset(frame: pd.DataFrame, min_length: int = MIN_TEXT_LENGTH) -> pd.DataFrame:
    frame = frame.copy()
    frame["text"] = frame["text"].map(clean_newsgroup_text)
    frame = frame[frame["text"].str.len() >= min_length]
    frame = frame.drop_duplicates(subset="text")
    return frame.reset_index(drop=True)


def load_newsgroups(
    categories: dict[str, str] | None = None,
    data_home: Path = RAW_DATA_DIR,
    clean: bool = True,
) -> pd.DataFrame:
    from sklearn.datasets import fetch_20newsgroups

    categories = categories or NEWSGROUP_CATEGORIES
    data_home.mkdir(parents=True, exist_ok=True)
    bunch = fetch_20newsgroups(
        subset="all",
        categories=list(categories),
        remove=("headers", "footers", "quotes"),
        data_home=str(data_home),
        shuffle=True,
        random_state=42,
    )
    frame = pd.DataFrame(
        {
            "text": bunch.data,
            "label": [categories[bunch.target_names[target]] for target in bunch.target],
            "source": [bunch.target_names[target] for target in bunch.target],
        }
    )
    return prepare_dataset(frame) if clean else frame


def load_folder_dataset(directory: str | Path, min_length: int = MIN_TEXT_LENGTH) -> pd.DataFrame:
    directory = Path(directory)
    rows = []
    for label_dir in sorted(path for path in directory.iterdir() if path.is_dir()):
        for document in iter_documents(label_dir):
            rows.append({"text": document.text, "label": label_dir.name, "source": str(document.path)})
    frame = pd.DataFrame(rows, columns=["text", "label", "source"])
    if frame.empty:
        raise ValueError(f"A(z) {directory} mappában nincs címkézett dokumentum (almappa = kategória).")
    frame["text"] = frame["text"].map(lambda text: _WHITESPACE.sub(" ", text).strip())
    return frame[frame["text"].str.len() >= min_length].reset_index(drop=True)


def dataset_summary(frame: pd.DataFrame) -> pd.DataFrame:
    words = frame["text"].str.split().str.len()
    summary = (
        frame.assign(words=words)
        .groupby("label")["words"]
        .agg(documents="count", mean_words="mean", median_words="median", min_words="min", max_words="max")
        .round(1)
        .sort_values("documents", ascending=False)
    )
    summary["share"] = (summary["documents"] / summary["documents"].sum()).round(3)
    return summary
