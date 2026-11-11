from __future__ import annotations

from pathlib import Path

import pytest

from docproc import Document
from docproc.exceptions import QuerySyntaxError
from docproc.search import (
    AndNode,
    NotNode,
    OrNode,
    PhraseNode,
    PrefixNode,
    SearchIndex,
    TermNode,
    parse_query,
    regex_search,
)


def make_document(name: str, text: str) -> Document:
    return Document(path=Path(name), text=text, file_type="txt")


@pytest.fixture
def documents() -> list[Document]:
    return [
        make_document("szerzodes.txt", "A bérleti szerződés szerint a havi bérleti díj 425 000 Ft. A szerződés két évre szól."),
        make_document("szamla.txt", "Számla a bérleti díjról. A számlát 15 napon belül kell kiegyenlíteni."),
        make_document("ajanlat.txt", "Árajánlat webáruház fejlesztésére. Az ajánlat 30 napig érvényes."),
        make_document("hirlevel.txt", "Szerződéskötés előtt kérjen ajánlatot! Új webáruházban is vásárolhat."),
    ]


@pytest.fixture
def index(documents) -> SearchIndex:
    return SearchIndex.from_documents(documents)


def names(hits) -> list[str]:
    return [hit.document.name for hit in hits]


class TestQueryParser:
    def test_single_term(self):
        assert parse_query("szerződés") == TermNode("szerződés")

    def test_implicit_and(self):
        assert parse_query("bérleti díj") == AndNode(TermNode("bérleti"), TermNode("díj"))

    def test_operator_precedence(self):
        assert parse_query("a OR b AND c") == OrNode(TermNode("a"), AndNode(TermNode("b"), TermNode("c")))

    def test_parentheses_and_not(self):
        assert parse_query("(a OR b) NOT c") == AndNode(OrNode(TermNode("a"), TermNode("b")), NotNode(TermNode("c")))

    def test_hungarian_operators(self):
        assert parse_query("a VAGY b ÉS NEM c") == OrNode(TermNode("a"), AndNode(TermNode("b"), NotNode(TermNode("c"))))

    def test_phrase_and_prefix(self):
        assert parse_query('"bérleti díj" szerz*') == AndNode(PhraseNode(("bérleti", "díj")), PrefixNode("szerz"))

    def test_lowercase_operator_is_a_word(self):
        assert parse_query("kovács és társa") == AndNode(AndNode(TermNode("kovács"), TermNode("és")), TermNode("társa"))

    @pytest.mark.parametrize("query", ["", "   ", "(a OR b", "a )", "AND a", "a OR", '"nyitott', '""', "NOT"])
    def test_invalid_queries(self, query: str):
        with pytest.raises(QuerySyntaxError):
            parse_query(query)


class TestSearchIndex:
    def test_index_size(self, index: SearchIndex):
        assert len(index) == 4
        assert index.vocabulary_size > 20
        assert index.average_length > 0

    def test_term_search_uses_stemming(self, index: SearchIndex):
        assert names(index.search("számlák")) == ["szamla.txt"]

    def test_accent_insensitive(self, index: SearchIndex):
        assert names(index.search("szerzodes")) == ["szerzodes.txt"]

    def test_exact_mode(self, documents):
        exact = SearchIndex.from_documents(documents, stemming=False, fold_accents=False)
        assert exact.search("szerzodes") == []
        assert names(exact.search("szerződés")) == ["szerzodes.txt"]

    def test_and_or_not(self, index: SearchIndex):
        assert set(names(index.search("bérleti AND díj"))) == {"szerzodes.txt", "szamla.txt"}
        assert set(names(index.search("webáruház OR számla"))) == {"ajanlat.txt", "hirlevel.txt", "szamla.txt"}
        assert names(index.search("bérleti NOT számla")) == ["szerzodes.txt"]

    def test_phrase_requires_adjacency(self, index: SearchIndex):
        assert names(index.search('"havi bérleti díj"')) == ["szerzodes.txt"]
        assert index.search('"díj havi"') == []

    def test_prefix(self, index: SearchIndex):
        assert set(names(index.search("szerz*"))) == {"szerzodes.txt", "hirlevel.txt"}

    def test_ranking_prefers_more_occurrences(self, index: SearchIndex):
        hits = index.search("szerződés OR szerződéskötés")
        assert hits[0].document.name == "szerzodes.txt"
        assert hits[0].score > hits[1].score

    def test_tfidf_ranking(self, documents):
        hits = SearchIndex.from_documents(documents, ranking="tfidf").search("bérleti")
        assert names(hits) == ["szerzodes.txt", "szamla.txt"]

    def test_unknown_ranking(self):
        with pytest.raises(ValueError):
            SearchIndex(ranking="pagerank")

    def test_pure_not_query(self, index: SearchIndex):
        hits = index.search("NOT bérleti")
        assert set(names(hits)) == {"ajanlat.txt", "hirlevel.txt"}
        assert all(hit.score == 0 for hit in hits)

    def test_top_k(self, index: SearchIndex):
        assert len(index.search("a OR az OR új", top_k=2)) == 2

    def test_snippets_highlight(self, index: SearchIndex):
        hit = index.search("havi", context=10)[0]
        assert hit.snippets == ["…szerint a [havi] bérleti d…"]

    def test_term_counts(self, index: SearchIndex):
        hit = index.search("szerződés")[0]
        assert hit.term_counts == {"szerzodes": 2}

    def test_keyword_counts(self, index: SearchIndex):
        table = index.keyword_counts(["bérleti díj", "számla", "webáruház"])
        assert table.loc["szerzodes.txt", "bérleti díj"] == 1
        assert table.loc["szamla.txt", "számla"] == 2
        assert table["webáruház"].sum() == 2


class TestRegexSearch:
    def test_finds_matches_with_line_numbers(self):
        document = make_document("a.txt", "első sor\nAdószám: 12345678-2-43\nharmadik")
        matches = regex_search([document], r"\d{8}-\d-\d{2}")
        assert len(matches) == 1
        assert matches[0].line_number == 2
        assert matches[0].text == "12345678-2-43"
        assert "[12345678-2-43]" in matches[0].snippet

    def test_case_sensitivity(self, documents):
        assert len(regex_search(documents, r"SZÁML")) == 2
        assert len(regex_search(documents, r"SZÁML", ignore_case=False)) == 0

    def test_invalid_pattern(self, documents):
        with pytest.raises(QuerySyntaxError):
            regex_search(documents, r"(nyitott")
