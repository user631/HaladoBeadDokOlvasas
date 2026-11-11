from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

RESOURCES_DIR = Path(__file__).resolve().parent / "resources"
SUPPORTED_LANGUAGES = ("hu", "en")
UNKNOWN_LANGUAGE = "unknown"

_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_WORD = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*")
_WORD_OR_NUMBER = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*")
_WHITESPACE = re.compile(r"\s+")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_BULLET = re.compile(r"^(?:[-•*–]|\d+[.)])\s+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+(?=[\"'„(]?[A-ZÁÉÍÓÖŐÚÜŰ0-9])")
_TERMINAL_PUNCTUATION = (".", "!", "?", "…", ":", ";", ",")
_HUNGARIAN_ONLY_CHARS = set("őűŐŰ")

ABBREVIATIONS = frozenset({
    "pl", "stb", "kb", "ill", "ún", "ld", "vö", "dr", "id", "ifj", "özv", "prof", "tel", "sz",
    "kft", "bt", "zrt", "nyrt", "kht", "u", "krt", "s.k", "jan", "febr", "márc", "ápr", "máj",
    "jún", "júl", "aug", "szept", "okt", "nov", "dec", "mr", "mrs", "ms", "inc", "ltd", "co",
    "vs", "etc", "e.g", "i.e", "no",
})

_HU_SUFFIXES = tuple(sorted(
    {
        "ban", "ben", "ból", "ből", "ról", "ről", "tól", "től", "nak", "nek", "val", "vel",
        "hoz", "hez", "höz", "nál", "nél", "ért", "kor", "ig", "ra", "re", "ba", "be", "on",
        "en", "ön", "ok", "ek", "ak", "ök", "ot", "et", "at", "öt", "i", "t", "k",
    },
    key=len,
    reverse=True,
))
_HU_LENGTHENED_VOWELS = {"á": "a", "é": "e"}
_EN_SUFFIXES = (("ies", "y"), ("ing", ""), ("ed", ""), ("ly", ""), ("es", ""), ("s", ""))


@lru_cache(maxsize=None)
def load_stopwords(language: str) -> frozenset[str]:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Nem támogatott nyelv: {language}")
    path = RESOURCES_DIR / f"stopwords_{language}.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    return frozenset(line.strip().lower() for line in lines if line.strip())


def stopwords_for(language: str) -> frozenset[str]:
    if language in SUPPORTED_LANGUAGES:
        return load_stopwords(language)
    return frozenset().union(*(load_stopwords(lang) for lang in SUPPORTED_LANGUAGES))


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_accents: bool = False,
) -> str:
    text = unicodedata.normalize("NFC", text)
    if remove_urls:
        text = _URL.sub(" ", text)
    if remove_emails:
        text = _EMAIL.sub(" ", text)
    if lowercase:
        text = text.lower()
    if remove_accents:
        text = strip_accents(text)
    return normalize_whitespace(text)


def tokenize(text: str, keep_numbers: bool = False) -> list[str]:
    pattern = _WORD_OR_NUMBER if keep_numbers else _WORD
    return pattern.findall(text)


def iter_tokens(text: str, keep_numbers: bool = True) -> Iterator[tuple[str, int, int]]:
    pattern = _WORD_OR_NUMBER if keep_numbers else _WORD
    for match in pattern.finditer(text):
        yield match.group(), match.start(), match.end()


def split_paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in _PARAGRAPH_BREAK.split(text) if paragraph.strip()]


def _starts_new_segment(previous: str, line: str) -> bool:
    if _BULLET.match(line) or previous.endswith(":"):
        return True
    looks_like_heading = len(previous) < 70 and not previous.endswith(_TERMINAL_PUNCTUATION)
    return looks_like_heading and line[:1].isupper()


def _join_lines(paragraph: str) -> list[str]:
    segments: list[str] = []
    current: list[str] = []
    for raw_line in paragraph.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if current and _starts_new_segment(current[-1], line):
            segments.append(" ".join(current))
            current = []
        current.append(line)
    if current:
        segments.append(" ".join(current))
    return segments


def _ends_with_abbreviation(candidate: str) -> bool:
    words = candidate.split()
    if not words:
        return False
    token = words[-1].rstrip(".").lstrip("(\"'„").lower()
    if not token:
        return False
    if token.isdigit():
        return len(words) == 1
    return token in ABBREVIATIONS or (len(token) == 1 and token.isalpha())


def _split_segment(segment: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(segment):
        candidate = segment[start:match.start()]
        if _ends_with_abbreviation(candidate):
            continue
        sentences.append(candidate.strip())
        start = match.end()
    tail = segment[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for paragraph in split_paragraphs(text):
        for segment in _join_lines(paragraph):
            sentences.extend(_split_segment(normalize_whitespace(segment)))
    return sentences


def language_scores(tokens: list[str]) -> dict[str, float]:
    lowered = [token.lower() for token in tokens]
    total = len(lowered) or 1
    scores = {
        language: sum(token in load_stopwords(language) for token in lowered) / total
        for language in SUPPORTED_LANGUAGES
    }
    if any(char in _HUNGARIAN_ONLY_CHARS for token in tokens for char in token):
        scores["hu"] += 0.1
    return scores


def detect_language(text: str, min_tokens: int = 5, min_score: float = 0.05) -> str:
    tokens = tokenize(text)
    if len(tokens) < min_tokens:
        return UNKNOWN_LANGUAGE
    scores = language_scores(tokens)
    best = max(scores, key=scores.get)
    return best if scores[best] >= min_score else UNKNOWN_LANGUAGE


def _strip_hungarian_suffix(token: str) -> str:
    for suffix in _HU_SUFFIXES:
        if not token.endswith(suffix) or (suffix == "i" and token.endswith("ni")):
            continue
        min_stem = 4 if len(suffix) == 1 else 3
        if len(token) - len(suffix) >= min_stem:
            stem = token[: -len(suffix)]
            last = stem[-1]
            return stem[:-1] + _HU_LENGTHENED_VOWELS.get(last, last)
    return token


def stem_hungarian(token: str, max_passes: int = 2) -> str:
    for _ in range(max_passes):
        stripped = _strip_hungarian_suffix(token)
        if stripped == token:
            break
        token = stripped
    return token


def stem_english(token: str) -> str:
    if token.endswith("ss"):
        return token
    for suffix, replacement in _EN_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)] + replacement
    return token


def stem(token: str, language: str) -> str:
    if language == "hu":
        return stem_hungarian(token)
    if language == "en":
        return stem_english(token)
    return token


@dataclass
class TextAnalysis:
    language: str
    paragraphs: list[str]
    sentences: list[str]
    words: list[str]
    content_words: list[str]


@dataclass
class TextPreprocessor:
    language: str = "auto"
    lowercase: bool = True
    remove_stopwords: bool = True
    min_token_length: int = 2
    keep_numbers: bool = False
    use_stemming: bool = False
    extra_stopwords: frozenset[str] = field(default_factory=frozenset)

    def resolve_language(self, text: str) -> str:
        return detect_language(text) if self.language == "auto" else self.language

    def stopwords(self, language: str) -> frozenset[str]:
        return stopwords_for(language) | {word.lower() for word in self.extra_stopwords}

    def tokens(self, text: str, language: str | None = None) -> list[str]:
        language = language or self.resolve_language(text)
        stopwords = self.stopwords(language) if self.remove_stopwords else frozenset()
        cleaned = clean_text(text, lowercase=self.lowercase)
        result = []
        for token in tokenize(cleaned, keep_numbers=self.keep_numbers):
            if len(token) < self.min_token_length or token.lower() in stopwords:
                continue
            result.append(stem(token, language) if self.use_stemming else token)
        return result

    def __call__(self, text: str) -> list[str]:
        return self.tokens(text)

    def analyze(self, text: str) -> TextAnalysis:
        language = self.resolve_language(text)
        return TextAnalysis(
            language=language,
            paragraphs=split_paragraphs(text),
            sentences=split_sentences(text),
            words=[word.lower() for word in tokenize(clean_text(text, lowercase=False))],
            content_words=self.tokens(text, language=language),
        )
