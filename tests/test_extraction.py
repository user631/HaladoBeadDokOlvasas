from __future__ import annotations

from pathlib import Path

import pytest

from docproc import Document
from docproc.extraction import (
    ENTITY_LABELS,
    InformationExtractor,
    KeywordExtractor,
    available_kinds,
    bank_account_is_valid,
    extract_information,
    format_number,
    iban_is_valid,
    results_to_frame,
)


def values(text: str, kind: str) -> list[str]:
    return [entity.value for entity in InformationExtractor([kind]).extract_text(text)]


class TestContactData:
    def test_emails(self):
        text = "Írjon a Kiss.Anna@Example.com címre vagy az info@ceg.example.hu-ra."
        assert values(text, "email") == ["kiss.anna@example.com", "info@ceg.example.hu"]

    def test_urls(self):
        text = "Részletek: https://palyazat.example.com/felhivas. Vagy www.example.com, köszönjük."
        assert values(text, "url") == ["https://palyazat.example.com/felhivas", "https://www.example.com"]

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Tel.: +36 1 234 5678", "+36 1 234 5678"),
            ("Mobil: +36 30 123 4567", "+36 30 123 4567"),
            ("Hívjon: 06-30-123-4567", "+36 30 123 4567"),
            ("Telefon: 06 76 456 789", "+36 76 456 789"),
            ("Telefon: +36/20/987-6543", "+36 20 987 6543"),
        ],
    )
    def test_phones(self, text: str, expected: str):
        assert values(text, "phone") == [expected]

    def test_phone_not_in_tax_number(self):
        assert values("Adószám: 87654321-1-06 123 4567", "phone") == []


class TestDatesAndAmounts:
    def test_numeric_dates(self):
        text = "Kiállítva: 2026.10.15., esedékes 2026-10-29, rögzítve 2026/11/03."
        assert values(text, "date") == ["2026-10-15", "2026-10-29", "2026-11-03"]

    def test_textual_dates(self):
        text = "A szerződés 2026. december 1. napjától 2028. nov. 30-ig érvényes, átadás 2026. november 28-án."
        assert sorted(values(text, "date")) == ["2026-11-28", "2026-12-01", "2028-11-30"]

    def test_invalid_dates_are_skipped(self):
        assert values("Hibás dátum: 2026.02.30.", "date") == []

    def test_invoice_number_is_not_a_date(self):
        assert values("Számlaszám: 2026/0142", "date") == []

    def test_amounts(self):
        text = "Bérleti díj 425 000 Ft, óvadék 1 275 000 Ft-ot, előleg 1250,50 EUR és 300 000 000 forintot."
        assert values(text, "amount") == ["425 000 HUF", "1 275 000 HUF", "1 250,50 EUR", "300 000 000 HUF"]

    def test_amount_requires_currency(self):
        assert values("A helyiség 85 négyzetméter, 2026-ban.", "amount") == []

    def test_percentages(self):
        assert values("ÁFA (27%), előleg 30 %, kamat 12,5%", "percentage") == ["27%", "30%", "12,5%"]

    def test_format_number(self):
        assert format_number(1250000.0) == "1 250 000"
        assert format_number(980.25) == "980,25"


class TestFinancialIdentifiers:
    def test_iban_validation(self):
        assert iban_is_valid("HU42 1177 3016 1111 1018 0000 0000")
        assert not iban_is_valid("HU43 1177 3016 1111 1018 0000 0000")

    def test_iban_extraction_across_line_break(self):
        text = "Számla: HU42 1177\n3016 1111 1018 0000 0000 E-mail: x@example.com"
        assert values(text, "iban") == ["HU42 1177 3016 1111 1018 0000 0000"]

    def test_invalid_iban_is_skipped(self):
        assert values("IBAN: HU43 1177 3016 1111 1018 0000 0000", "iban") == []

    def test_bank_account(self):
        assert bank_account_is_valid("11773016-11111018")
        assert not bank_account_is_valid("11773016-11111019")
        assert values("Számlaszám: 11773016-11111018-00000000, hibás: 11773016-11111019", "bank_account") == [
            "11773016-11111018-00000000"
        ]

    def test_tax_numbers(self):
        text = "Adószám: 12345678-2-43, hibás megyekód: 12345678-2-99, hibás áfakód: 12345678-7-43"
        assert values(text, "tax_number") == ["12345678-2-43"]


class TestNamesAndAddresses:
    def test_postal_addresses(self):
        text = "Székhely: 1117 Budapest, Kőrösi utca 12. Telephely: 6720 Szeged, Tisza Lajos körút 41/B."
        assert values(text, "postal_address") == ["1117 Budapest, Kőrösi utca 12", "6720 Szeged, Tisza Lajos körút 41/B"]

    def test_companies(self):
        text = "A Kovács és Társa Bt. megrendelte a Napfény Irodaszer Kft.-től, a Duna Ingatlankezelő Zrt. pedig nem."
        assert values(text, "company") == ["Kovács és Társa Bt.", "Napfény Irodaszer Kft.", "Duna Ingatlankezelő Zrt."]

    def test_company_does_not_cross_lines(self):
        assert values("Eladó\nNapfény Irodaszer Kft.", "company") == ["Napfény Irodaszer Kft."]

    def test_fields(self):
        text = "Számlaszám: 2026/0142\nKiállítás dátuma: 2026.10.15. Fizetési határidő: 2026.10.29.\nidőpont 11:45-kor"
        assert values(text, "field") == [
            "Számlaszám: 2026/0142",
            "Kiállítás dátuma: 2026.10.15.",
            "Fizetési határidő: 2026.10.29.",
        ]


class TestInformationExtractor:
    def test_all_kinds_have_labels(self):
        assert set(available_kinds()) == set(ENTITY_LABELS)

    def test_unknown_kind(self):
        with pytest.raises(ValueError):
            InformationExtractor(["szemelyi_szam"])

    def test_entities_are_sorted_and_positioned(self):
        text = "Kapcsolat: anna@example.com, +36 1 234 5678"
        entities = InformationExtractor(["email", "phone"]).extract_text(text)
        assert [entity.kind for entity in entities] == ["email", "phone"]
        assert all(text[entity.start:entity.end] == entity.text for entity in entities)

    def test_result_helpers(self):
        document = Document(path=Path("a.txt"), text="Számlaszám: 12\nE-mail: a@example.com, a@example.com", file_type="txt")
        result = InformationExtractor().extract(document)
        assert result.values("email") == ["a@example.com"]
        assert result.fields()["Számlaszám"] == "12"
        records = result.to_records()
        assert records[0]["document"] == "a.txt"
        assert {"kind", "label", "value", "start", "end"} <= set(records[0])


class TestKeywordExtractor:
    def test_distinctive_terms_rank_higher(self):
        texts = [
            "A bérlő fizeti a bérleti díjat. A bérlő köteles a bérleti díjat időben fizetni.",
            "A számla tartalmazza a fizetendő összeget. A számla határideje 15 nap.",
            "A pályázat támogatja a kisvállalkozásokat. A pályázat határideje december.",
        ]
        keywords = KeywordExtractor(top_n=3).fit_extract(texts)
        assert keywords[0][0][0] in {"bérlő", "bérleti"}
        assert "bérleti díjat" in [term for term, _ in keywords[0]]
        assert keywords[1][0][0] == "számla"
        assert keywords[2][0][0] == "pályázat"

    def test_single_bigram_occurrence_is_ignored(self):
        extractor = KeywordExtractor().fit(["havi díj fizetése"])
        assert "havi díj" not in dict(extractor.extract("havi díj fizetése"))

    def test_without_fit_uses_frequency(self):
        keywords = KeywordExtractor(top_n=1).extract("ajánlat ajánlat határidő")
        assert keywords[0][0] == "ajánlat"


def test_extract_information_and_frame(sample_dir: Path):
    from docproc import iter_documents

    results = extract_information(iter_documents(sample_dir), keyword_count=3)
    assert len(results) == 4
    assert all(len(result.keywords) <= 3 for result in results)
    frame = results_to_frame(results)
    assert list(frame.columns) == ["document", "kind", "label", "value", "text", "start", "end"]
