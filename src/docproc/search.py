from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Union

import pandas as pd

from docproc.document import Document
from docproc.exceptions import QuerySyntaxError
from docproc.preprocessing import iter_tokens, normalize_whitespace, stem, strip_accents

OPERATORS = {"AND": "AND", "ÉS": "AND", "OR": "OR", "VAGY": "OR", "NOT": "NOT", "NEM": "NOT"}
RANKINGS = ("bm25", "tfidf")
DEFAULT_MARKERS = ("[", "]")
_QUERY_TOKEN = re.compile(r'"[^"]*"?|\(|\)|[^\s()"]+')


@dataclass(frozen=True)
class TermNode:
    text: str


@dataclass(frozen=True)
class PrefixNode:
    prefix: str


@dataclass(frozen=True)
class PhraseNode:
    words: tuple[str, ...]


@dataclass(frozen=True)
class AndNode:
    left: QueryNode
    right: QueryNode


@dataclass(frozen=True)
class OrNode:
    left: QueryNode
    right: QueryNode


@dataclass(frozen=True)
class NotNode:
    operand: QueryNode


QueryNode = Union[TermNode, PrefixNode, PhraseNode, AndNode, OrNode, NotNode]


class QueryParser:
    def __init__(self, query: str):
        self.tokens = _QUERY_TOKEN.findall(query)
        self.position = 0

    def parse(self) -> QueryNode:
        if not self.tokens:
            raise QuerySyntaxError("Üres keresőkifejezés.")
        node = self._parse_or()
        if self.position < len(self.tokens):
            raise QuerySyntaxError(f"Váratlan elem a keresőkifejezésben: {self.tokens[self.position]}")
        return node

    def _peek(self) -> str | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def _advance(self) -> str | None:
        token = self._peek()
        self.position += 1
        return token

    @staticmethod
    def _operator(token: str | None) -> str | None:
        return OPERATORS.get(token) if token else None

    def _parse_or(self) -> QueryNode:
        node = self._parse_and()
        while self._operator(self._peek()) == "OR":
            self._advance()
            node = OrNode(node, self._parse_and())
        return node

    def _parse_and(self) -> QueryNode:
        node = self._parse_not()
        while True:
            token = self._peek()
            if token is None or token == ")" or self._operator(token) == "OR":
                return node
            if self._operator(token) == "AND":
                self._advance()
            node = AndNode(node, self._parse_not())

    def _parse_not(self) -> QueryNode:
        if self._operator(self._peek()) == "NOT":
            self._advance()
            return NotNode(self._parse_not())
        return self._parse_atom()

    def _parse_atom(self) -> QueryNode:
        token = self._advance()
        if token is None:
            raise QuerySyntaxError("A keresőkifejezés hiányosan ér véget.")
        if token == "(":
            node = self._parse_or()
            if self._advance() != ")":
                raise QuerySyntaxError("Hiányzó záró zárójel.")
            return node
        if token == ")":
            raise QuerySyntaxError("Váratlan záró zárójel.")
        if self._operator(token):
            raise QuerySyntaxError(f"Hiányzó keresőszó a(z) {token} operátor előtt.")
        if token.startswith('"'):
            if len(token) < 2 or not token.endswith('"'):
                raise QuerySyntaxError("Lezáratlan idézőjel.")
            return self._words_node(token[1:-1], token)
        if token.endswith("*") and len(token) > 1:
            return PrefixNode(token[:-1])
        return self._words_node(token, token)

    @staticmethod
    def _words_node(text: str, original: str) -> QueryNode:
        words = tuple(word for word, _, _ in iter_tokens(text))
        if not words:
            raise QuerySyntaxError(f"Érvénytelen keresőszó: {original}")
        return TermNode(words[0]) if len(words) == 1 else PhraseNode(words)


def parse_query(query: str) -> QueryNode:
    return QueryParser(query).parse()


@dataclass
class SearchHit:
    document: Document
    score: float
    term_counts: dict[str, int] = field(default_factory=dict)
    snippets: list[str] = field(default_factory=list)


@dataclass
class RegexMatch:
    document: Document
    line_number: int
    text: str
    snippet: str


class SearchIndex:
    def __init__(
        self,
        stemming: bool = True,
        fold_accents: bool = True,
        language: str = "hu",
        ranking: str = "bm25",
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if ranking not in RANKINGS:
            raise ValueError(f"Ismeretlen rangsorolás: {ranking} (lehetséges: {', '.join(RANKINGS)})")
        self.stemming = stemming
        self.fold_accents = fold_accents
        self.language = language
        self.ranking = ranking
        self.k1 = k1
        self.b = b
        self.documents: list[Document] = []
        self.postings: dict[str, dict[int, list[int]]] = defaultdict(dict)
        self.token_spans: list[list[tuple[int, int]]] = []

    @classmethod
    def from_documents(cls, documents: Iterable[Document], **options) -> SearchIndex:
        index = cls(**options)
        for document in documents:
            index.add(document)
        return index

    def __len__(self) -> int:
        return len(self.documents)

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)

    @property
    def average_length(self) -> float:
        if not self.token_spans:
            return 0.0
        return sum(len(spans) for spans in self.token_spans) / len(self.token_spans)

    def normalize(self, word: str) -> str:
        word = word.lower()
        if self.stemming:
            word = stem(word, self.language)
        if self.fold_accents:
            word = strip_accents(word)
        return word

    def _normalize_prefix(self, prefix: str) -> str:
        prefix = prefix.lower()
        return strip_accents(prefix) if self.fold_accents else prefix

    def add(self, document: Document) -> int:
        doc_id = len(self.documents)
        self.documents.append(document)
        spans: list[tuple[int, int]] = []
        for position, (word, start, end) in enumerate(iter_tokens(document.text)):
            self.postings[self.normalize(word)].setdefault(doc_id, []).append(position)
            spans.append((start, end))
        self.token_spans.append(spans)
        return doc_id

    def document_frequency(self, term: str) -> int:
        return len(self.postings.get(term, {}))

    def term_frequency(self, term: str, doc_id: int) -> int:
        return len(self.postings.get(term, {}).get(doc_id, ()))

    def idf(self, term: str) -> float:
        total = len(self.documents)
        df = self.document_frequency(term)
        if self.ranking == "bm25":
            return math.log(1 + (total - df + 0.5) / (df + 0.5))
        return math.log((1 + total) / (1 + df)) + 1

    def score(self, doc_id: int, terms: Iterable[str]) -> float:
        length = len(self.token_spans[doc_id])
        average = self.average_length or 1.0
        total = 0.0
        for term in set(terms):
            tf = self.term_frequency(term, doc_id)
            if tf == 0:
                continue
            if self.ranking == "bm25":
                saturation = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / average))
                total += self.idf(term) * saturation
            else:
                total += (1 + math.log(tf)) * self.idf(term) / math.sqrt(max(length, 1))
        return total

    def _prefix_terms(self, prefix: str) -> list[str]:
        normalized = self._normalize_prefix(prefix)
        return sorted(term for term in self.postings if term.startswith(normalized))

    def phrase_positions(self, words: Iterable[str]) -> dict[int, list[int]]:
        terms = [self.normalize(word) for word in words]
        postings = [self.postings.get(term, {}) for term in terms]
        if not terms or not all(postings):
            return {}
        candidates = set.intersection(*(set(posting) for posting in postings))
        result: dict[int, list[int]] = {}
        for doc_id in candidates:
            following = [set(posting[doc_id]) for posting in postings[1:]]
            starts = [
                position
                for position in postings[0][doc_id]
                if all(position + offset in positions for offset, positions in enumerate(following, start=1))
            ]
            if starts:
                result[doc_id] = starts
        return result

    def _evaluate(self, node: QueryNode) -> set[int]:
        match node:
            case TermNode(text):
                return set(self.postings.get(self.normalize(text), {}))
            case PrefixNode(prefix):
                documents: set[int] = set()
                for term in self._prefix_terms(prefix):
                    documents |= set(self.postings[term])
                return documents
            case PhraseNode(words):
                return set(self.phrase_positions(words))
            case AndNode(left, right):
                return self._evaluate(left) & self._evaluate(right)
            case OrNode(left, right):
                return self._evaluate(left) | self._evaluate(right)
            case NotNode(operand):
                return set(range(len(self.documents))) - self._evaluate(operand)
        raise QuerySyntaxError(f"Ismeretlen lekérdezéselem: {node!r}")

    def _positive_terms(self, node: QueryNode, negated: bool = False) -> list[str]:
        match node:
            case TermNode(text):
                return [] if negated else [self.normalize(text)]
            case PrefixNode(prefix):
                return [] if negated else self._prefix_terms(prefix)
            case PhraseNode(words):
                return [] if negated else [self.normalize(word) for word in words]
            case AndNode(left, right) | OrNode(left, right):
                return self._positive_terms(left, negated) + self._positive_terms(right, negated)
            case NotNode(operand):
                return self._positive_terms(operand, not negated)
        return []

    def snippets(
        self,
        doc_id: int,
        terms: Iterable[str],
        max_snippets: int = 2,
        context: int = 60,
        markers: tuple[str, str] = DEFAULT_MARKERS,
    ) -> list[str]:
        text = self.documents[doc_id].text
        spans = self.token_spans[doc_id]
        positions = sorted({position for term in set(terms) for position in self.postings.get(term, {}).get(doc_id, ())})
        result: list[str] = []
        covered_until = -1
        for position in positions:
            start, end = spans[position]
            if start < covered_until:
                continue
            window_start = max(0, start - context)
            window_end = min(len(text), end + context)
            parts: list[str] = []
            cursor = window_start
            for other in positions:
                other_start, other_end = spans[other]
                if other_start >= cursor and other_end <= window_end and other_start >= window_start:
                    parts.append(text[cursor:other_start])
                    parts.append(f"{markers[0]}{text[other_start:other_end]}{markers[1]}")
                    cursor = other_end
            parts.append(text[cursor:window_end])
            prefix = "…" if window_start > 0 else ""
            suffix = "…" if window_end < len(text) else ""
            result.append(prefix + normalize_whitespace("".join(parts)) + suffix)
            covered_until = window_end
            if len(result) >= max_snippets:
                break
        return result

    def search(
        self,
        query: str,
        top_k: int | None = 10,
        max_snippets: int = 2,
        context: int = 60,
        markers: tuple[str, str] = DEFAULT_MARKERS,
    ) -> list[SearchHit]:
        node = parse_query(query)
        terms = list(dict.fromkeys(self._positive_terms(node)))
        hits = []
        for doc_id in self._evaluate(node):
            counts = {term: self.term_frequency(term, doc_id) for term in terms}
            hits.append(
                SearchHit(
                    document=self.documents[doc_id],
                    score=round(self.score(doc_id, terms), 4),
                    term_counts={term: count for term, count in counts.items() if count},
                    snippets=self.snippets(doc_id, terms, max_snippets, context, markers),
                )
            )
        hits.sort(key=lambda hit: (-hit.score, hit.document.name))
        return hits[:top_k] if top_k else hits

    def count(self, keyword: str, doc_id: int) -> int:
        words = [word for word, _, _ in iter_tokens(keyword)]
        if not words:
            return 0
        if len(words) == 1:
            return self.term_frequency(self.normalize(words[0]), doc_id)
        return len(self.phrase_positions(words).get(doc_id, ()))

    def keyword_counts(self, keywords: Iterable[str]) -> pd.DataFrame:
        keywords = list(keywords)
        rows = {
            document.name: {keyword: self.count(keyword, doc_id) for keyword in keywords}
            for doc_id, document in enumerate(self.documents)
        }
        return pd.DataFrame.from_dict(rows, orient="index", columns=keywords)


def regex_search(
    documents: Iterable[Document],
    pattern: str,
    ignore_case: bool = True,
    context: int = 40,
    markers: tuple[str, str] = DEFAULT_MARKERS,
) -> list[RegexMatch]:
    try:
        compiled = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        raise QuerySyntaxError(f"Hibás reguláris kifejezés: {exc}") from exc
    matches = []
    for document in documents:
        text = document.text
        for match in compiled.finditer(text):
            if not match.group():
                continue
            before = text[max(0, match.start() - context):match.start()]
            after = text[match.end():match.end() + context]
            snippet = normalize_whitespace(f"{before}{markers[0]}{match.group()}{markers[1]}{after}")
            line_number = text.count("\n", 0, match.start()) + 1
            matches.append(RegexMatch(document, line_number, match.group(), snippet))
    return matches
