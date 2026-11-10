from __future__ import annotations

import pytest

from docproc.preprocessing import (
    TextPreprocessor,
    clean_text,
    detect_language,
    load_stopwords,
    normalize_whitespace,
    split_paragraphs,
    split_sentences,
    stem,
    stopwords_for,
    strip_accents,
    tokenize,
)


class TestCleaning:
    def test_normalize_whitespace(self):
        assert normalize_whitespace("  egy \t ketto\n\nharom ") == "egy ketto harom"

    def test_strip_accents(self):
        assert strip_accents("Árvíztűrő tükörfúrógép") == "Arvizturo tukorfurogep"

    def test_clean_text_removes_urls_and_emails(self):
        text = "Írjon a info@example.com címre vagy nézze meg: https://example.com/oldal Köszönjük!"
        assert clean_text(text) == "írjon a címre vagy nézze meg: köszönjük!"

    def test_clean_text_keeps_case_when_requested(self):
        assert clean_text("Budapest Szeged", lowercase=False) == "Budapest Szeged"


class TestTokenize:
    def test_words_only(self):
        assert tokenize("A számla 2026-ban 138 049 Ft volt.") == ["A", "számla", "ban", "Ft", "volt"]

    def test_keep_numbers(self):
        assert tokenize("2026-ban 138 Ft", keep_numbers=True) == ["2026-ban", "138", "Ft"]

    def test_hyphenated_words_stay_together(self):
        assert tokenize("e-mail és átadás-átvétel") == ["e-mail", "és", "átadás-átvétel"]


class TestSentences:
    def test_basic_split(self):
        assert split_sentences("Ez az első. Ez a második! Ez a harmadik?") == [
            "Ez az első.",
            "Ez a második!",
            "Ez a harmadik?",
        ]

    def test_abbreviations_do_not_split(self):
        text = "A Kovács és Társa Bt. Szegeden működik. Pl. Kiss Anna dr. Nagy Béla mellett dolgozik."
        assert split_sentences(text) == [
            "A Kovács és Társa Bt. Szegeden működik.",
            "Pl. Kiss Anna dr. Nagy Béla mellett dolgozik.",
        ]

    def test_dates_inside_sentence(self):
        text = "A szerződés 2026. december 1. napjától érvényes. Utána meghosszabbítható."
        assert split_sentences(text) == [
            "A szerződés 2026. december 1. napjától érvényes.",
            "Utána meghosszabbítható.",
        ]

    def test_numbered_heading_is_separate(self):
        text = "1. A szerződés tárgya\nA bérbeadó bérbe adja az irodát."
        assert split_sentences(text) == ["1. A szerződés tárgya", "A bérbeadó bérbe adja az irodát."]

    def test_wrapped_lines_are_joined(self):
        text = "Ez egy hosszú mondat, amelyet a PDF a sor végén\nkettévágott, de egy mondat."
        assert split_sentences(text) == ["Ez egy hosszú mondat, amelyet a PDF a sor végén kettévágott, de egy mondat."]

    def test_bullets_are_separate(self):
        text = "Jelen vannak:\n- Kovács Péter\n- Nagy Eszter"
        assert split_sentences(text) == ["Jelen vannak:", "- Kovács Péter", "- Nagy Eszter"]

    def test_paragraphs(self):
        assert split_paragraphs("Első bekezdés.\n\n\nMásodik\nbekezdés.") == ["Első bekezdés.", "Második\nbekezdés."]


class TestLanguage:
    def test_hungarian(self):
        assert detect_language("A bérlő köteles a helyiséget rendeltetésszerűen használni és megóvni.") == "hu"

    def test_english(self):
        assert detect_language("The tenant is required to use the office and to keep it in good condition.") == "en"

    def test_too_short_is_unknown(self):
        assert detect_language("Szeged") == "unknown"

    def test_stopwords_loaded(self):
        assert "és" in load_stopwords("hu")
        assert "the" in load_stopwords("en")
        assert stopwords_for("unknown") >= load_stopwords("hu") | load_stopwords("en")

    def test_unsupported_language(self):
        with pytest.raises(ValueError):
            load_stopwords("de")


class TestStemming:
    @pytest.mark.parametrize(
        ("word", "expected"),
        [
            ("szerződésben", "szerződés"),
            ("számlát", "számla"),
            ("számlákat", "számla"),
            ("irodában", "iroda"),
            ("bérlőnek", "bérlő"),
            ("fizetési", "fizetés"),
            ("átadni", "átadni"),
            ("kert", "kert"),
        ],
    )
    def test_hungarian(self, word: str, expected: str):
        assert stem(word, "hu") == expected

    @pytest.mark.parametrize(
        ("word", "expected"),
        [("processing", "process"), ("categories", "category"), ("documents", "document"), ("class", "class")],
    )
    def test_english(self, word: str, expected: str):
        assert stem(word, "en") == expected

    def test_unknown_language_is_unchanged(self):
        assert stem("szerződésben", "unknown") == "szerződésben"


class TestTextPreprocessor:
    def test_removes_stopwords_and_short_tokens(self):
        preprocessor = TextPreprocessor(language="hu")
        assert preprocessor("A bérlő és a bérbeadó a szerződést aláírta.") == [
            "bérlő",
            "bérbeadó",
            "szerződést",
            "aláírta",
        ]

    def test_stemming(self):
        preprocessor = TextPreprocessor(language="hu", use_stemming=True)
        assert preprocessor("számlák számlát számlában") == ["számla", "számla", "számla"]

    def test_extra_stopwords(self):
        preprocessor = TextPreprocessor(language="hu", extra_stopwords=frozenset({"Kft"}))
        assert "kft" not in preprocessor("Napfény Irodaszer Kft. számlája")

    def test_keep_stopwords(self):
        preprocessor = TextPreprocessor(language="hu", remove_stopwords=False, min_token_length=1)
        assert preprocessor("a ház és a kert") == ["a", "ház", "és", "a", "kert"]

    def test_analyze(self):
        text = "A számla kiállítása megtörtént. A vevő fizetett.\n\nKöszönjük a megrendelést!"
        analysis = TextPreprocessor().analyze(text)
        assert analysis.language == "hu"
        assert len(analysis.paragraphs) == 2
        assert len(analysis.sentences) == 3
        assert analysis.words[:2] == ["a", "számla"]
        assert "a" not in analysis.content_words
