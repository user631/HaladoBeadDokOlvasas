from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import date

import pandas as pd

from docproc.document import Document
from docproc.preprocessing import TextPreprocessor

UPPER = "A-ZÁÉÍÓÖŐÚÜŰ"
LOWER = "a-záéíóöőúüű"

HUNGARIAN_MONTHS = {
    "január": 1, "február": 2, "március": 3, "április": 4, "május": 5, "június": 6,
    "július": 7, "augusztus": 8, "szeptember": 9, "október": 10, "november": 11, "december": 12,
    "jan": 1, "febr": 2, "márc": 3, "ápr": 4, "máj": 5, "jún": 6,
    "júl": 7, "aug": 8, "szept": 9, "okt": 10, "nov": 11, "dec": 12,
}
_MONTH_ALTERNATION = "|".join(sorted(HUNGARIAN_MONTHS, key=len, reverse=True))
CURRENCIES = {"ft": "HUF", "huf": "HUF", "forint": "HUF", "eur": "EUR", "€": "EUR", "euró": "EUR", "euro": "EUR", "usd": "USD", "$": "USD"}
TAX_COUNTY_CODES = set(range(2, 21)) | set(range(22, 45)) | {51}
LEGAL_FORMS = ("Kft", "Bt", "Zrt", "Nyrt", "Kkt", "Kht")
_LEADING_FUNCTION_WORDS = {"A", "Az", "Egy", "És", "Is"}

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b")
URL = re.compile(r"\b(?:https?://|www\.)[^\s<>\"'|]+", re.IGNORECASE)
PHONE = re.compile(r"(?<![\w+-])(?:\+36|06)[ /-]?(?:1|[2-9]\d)[ /-]?\d{3}[ -]?\d{3,4}(?![\d-])")
DATE_NUMERIC = re.compile(
    r"(?<![\d.])(?P<year>(?:19|20)\d{2}) ?[.\-/] ?(?P<month>1[0-2]|0?[1-9]) ?[.\-/] ?(?P<day>3[01]|[12]\d|0?[1-9])(?!\d)\.?"
)
DATE_TEXT = re.compile(
    rf"(?<![\d.])(?P<year>(?:19|20)\d{{2}})\. ?(?P<month>{_MONTH_ALTERNATION})\.? ?(?P<day>3[01]|[12]\d|0?[1-9])(?!\d)\.?",
    re.IGNORECASE,
)
AMOUNT = re.compile(
    rf"(?<![\d,.])(?P<number>\d{{1,3}}(?:[  .]\d{{3}})+|\d+)(?:,(?P<decimals>\d{{1,2}}))? ?"
    rf"(?P<currency>Ft|HUF|forint[{LOWER}]*|EUR|€|eur[oó][{LOWER}]*|USD|\$)(?![\w])",
    re.IGNORECASE,
)
PERCENTAGE = re.compile(r"(?<![\d,])\d+(?:,\d+)? ?%")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ \n]?[A-Z0-9]{4}){2,7}(?:[ \n]?\d{1,3})?(?![\w])")
BANK_ACCOUNT = re.compile(r"(?<![\d-])\d{8}-\d{8}(?:-\d{8})?(?![\d-])")
TAX_NUMBER = re.compile(r"(?<![\d-])(?P<base>\d{8})-(?P<vat>[1-5])-(?P<county>\d{2})(?![\d-])")
POSTAL_ADDRESS = re.compile(
    rf"\b(?P<zip>[1-9]\d{{3}}) (?P<city>[{UPPER}][{LOWER}]+(?:[- ][{UPPER}][{LOWER}]+)*), "
    rf"(?P<street>(?:[{UPPER}0-9][\w{LOWER}.-]* ){{1,4}})"
    rf"(?P<type>utca|útja|út|tere|tér|körút|krt\.|u\.|köz|sor|fasor|park|sétány|rakpart|lakótelep) "
    rf"(?P<number>\d+(?:/[A-Za-z0-9]+|[A-Za-z])?)\.?"
)
COMPANY = re.compile(
    rf"(?<![\w{LOWER}])(?P<name>[{UPPER}][\w{LOWER}&.-]*(?: (?:és|and|&|[{UPPER}][\w{LOWER}&.-]*)){{0,5}}) "
    rf"(?P<form>{'|'.join(LEGAL_FORMS)})\.?(?![\w{LOWER}])"
)
FIELD_KEY = re.compile(rf"(?:^|(?<=\s))(?P<key>[{UPPER}][\w{LOWER}-]*(?: [\w{LOWER}()-]+){{0,3}}):(?=\s|$)")


@dataclass(frozen=True)
class Entity:
    kind: str
    text: str
    value: str
    start: int
    end: int


Extractor = Callable[[str], Iterator[Entity]]
_EXTRACTORS: dict[str, Extractor] = {}
ENTITY_LABELS: dict[str, str] = {}


def extractor(kind: str, label: str) -> Callable[[Extractor], Extractor]:
    def decorator(function: Extractor) -> Extractor:
        _EXTRACTORS[kind] = function
        ENTITY_LABELS[kind] = label
        return function

    return decorator


def available_kinds() -> list[str]:
    return list(_EXTRACTORS)


def _entity(kind: str, match: re.Match, value: str, group: int | str = 0) -> Entity:
    return Entity(kind, match.group(group), value, match.start(group), match.end(group))


def format_number(value: float) -> str:
    formatted = f"{value:,.2f}".replace(",", " ").replace(".", ",")
    return formatted[:-3] if formatted.endswith(",00") else formatted


def iban_is_valid(iban: str) -> bool:
    compact = iban.replace(" ", "").upper()
    if not 15 <= len(compact) <= 34 or not compact[:2].isalpha() or not compact[2:4].isdigit():
        return False
    rearranged = compact[4:] + compact[:4]
    digits = "".join(str(int(char, 36)) for char in rearranged)
    return int(digits) % 97 == 1


def giro_group_is_valid(digits: str) -> bool:
    weights = (9, 7, 3, 1)
    return sum(int(digit) * weights[index % 4] for index, digit in enumerate(digits)) % 10 == 0


def bank_account_is_valid(account: str) -> bool:
    groups = account.split("-")
    return giro_group_is_valid(groups[0]) and giro_group_is_valid("".join(groups[1:]))


@extractor("email", "E-mail-címek")
def extract_emails(text: str) -> Iterator[Entity]:
    for match in EMAIL.finditer(text):
        yield _entity("email", match, match.group().lower())


@extractor("url", "Webcímek")
def extract_urls(text: str) -> Iterator[Entity]:
    for match in URL.finditer(text):
        url = match.group().rstrip(".,;:!?)")
        normalized = url if url.lower().startswith("http") else f"https://{url}"
        yield Entity("url", url, normalized, match.start(), match.start() + len(url))


@extractor("phone", "Telefonszámok")
def extract_phones(text: str) -> Iterator[Entity]:
    for match in PHONE.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        national = digits[2:]
        area_length = 1 if national.startswith("1") else 2
        area, subscriber = national[:area_length], national[area_length:]
        if len(subscriber) not in (6, 7):
            continue
        yield _entity("phone", match, f"+36 {area} {subscriber[:3]} {subscriber[3:]}")


@extractor("date", "Dátumok")
def extract_dates(text: str) -> Iterator[Entity]:
    for pattern in (DATE_NUMERIC, DATE_TEXT):
        for match in pattern.finditer(text):
            month = match.group("month")
            month_number = int(month) if month.isdigit() else HUNGARIAN_MONTHS[month.lower()]
            try:
                parsed = date(int(match.group("year")), month_number, int(match.group("day")))
            except ValueError:
                continue
            yield _entity("date", match, parsed.isoformat())


@extractor("amount", "Pénzösszegek")
def extract_amounts(text: str) -> Iterator[Entity]:
    for match in AMOUNT.finditer(text):
        number = re.sub(r"[  .]", "", match.group("number"))
        decimals = match.group("decimals") or "0"
        value = float(f"{number}.{decimals}")
        currency_key = match.group("currency").lower()
        currency = CURRENCIES.get(currency_key) or ("HUF" if currency_key.startswith("forint") else "EUR")
        yield _entity("amount", match, f"{format_number(value)} {currency}")


@extractor("percentage", "Százalékok")
def extract_percentages(text: str) -> Iterator[Entity]:
    for match in PERCENTAGE.finditer(text):
        yield _entity("percentage", match, match.group().replace(" ", ""))


@extractor("iban", "IBAN-számlaszámok")
def extract_ibans(text: str) -> Iterator[Entity]:
    for match in IBAN.finditer(text):
        compact = re.sub(r"\s", "", match.group())
        if iban_is_valid(compact):
            grouped = " ".join(compact[index:index + 4] for index in range(0, len(compact), 4))
            yield _entity("iban", match, grouped)


@extractor("bank_account", "Bankszámlaszámok")
def extract_bank_accounts(text: str) -> Iterator[Entity]:
    for match in BANK_ACCOUNT.finditer(text):
        if bank_account_is_valid(match.group()):
            yield _entity("bank_account", match, match.group())


@extractor("tax_number", "Adószámok")
def extract_tax_numbers(text: str) -> Iterator[Entity]:
    for match in TAX_NUMBER.finditer(text):
        if int(match.group("county")) in TAX_COUNTY_CODES:
            yield _entity("tax_number", match, match.group())


@extractor("postal_address", "Postacímek")
def extract_postal_addresses(text: str) -> Iterator[Entity]:
    for match in POSTAL_ADDRESS.finditer(text):
        address = re.sub(r"\s+", " ", match.group()).rstrip(".")
        yield Entity("postal_address", address, address, match.start(), match.start() + len(address))


@extractor("company", "Cégnevek")
def extract_companies(text: str) -> Iterator[Entity]:
    for match in COMPANY.finditer(text):
        words = match.group("name").split(" ")
        offset = 0
        while len(words) > 1 and words[0] in _LEADING_FUNCTION_WORDS:
            offset += len(words.pop(0)) + 1
        name = " ".join(words)
        start = match.start("name") + offset
        yield Entity("company", text[start:match.end()], f"{name} {match.group('form')}.", start, match.end())


@extractor("field", "Mezők (kulcs: érték)")
def extract_fields(text: str) -> Iterator[Entity]:
    line_start = 0
    for line in text.split("\n"):
        keys = list(FIELD_KEY.finditer(line))
        for index, key in enumerate(keys):
            value_end = keys[index + 1].start() if index + 1 < len(keys) else len(line)
            value = line[key.end():value_end].strip(" \t,;")
            if value and len(value) <= 150:
                start = line_start + key.start()
                end = line_start + key.end() + len(line[key.end():value_end].rstrip(" \t,;"))
                yield Entity("field", text[start:end], f"{key.group('key')}: {value}", start, end)
        line_start += len(line) + 1


@dataclass
class ExtractionResult:
    document: Document
    entities: list[Entity] = field(default_factory=list)
    keywords: list[tuple[str, float]] = field(default_factory=list)

    def values(self, kind: str) -> list[str]:
        return list(dict.fromkeys(entity.value for entity in self.entities if entity.kind == kind))

    def by_kind(self) -> dict[str, list[str]]:
        kinds = dict.fromkeys(entity.kind for entity in self.entities)
        return {kind: self.values(kind) for kind in kinds}

    def fields(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for value in self.values("field"):
            key, _, content = value.partition(": ")
            result.setdefault(key, content)
        return result

    def to_records(self) -> list[dict]:
        return [{"document": self.document.name, "label": ENTITY_LABELS[entity.kind], **asdict(entity)} for entity in self.entities]


class InformationExtractor:
    def __init__(self, kinds: Iterable[str] | None = None):
        self.kinds = list(kinds) if kinds else available_kinds()
        unknown = set(self.kinds) - set(_EXTRACTORS)
        if unknown:
            raise ValueError(f"Ismeretlen adattípus: {', '.join(sorted(unknown))}")

    def extract_text(self, text: str) -> list[Entity]:
        entities = [entity for kind in self.kinds for entity in _EXTRACTORS[kind](text)]
        return sorted(entities, key=lambda entity: (entity.start, entity.kind))

    def extract(self, document: Document) -> ExtractionResult:
        return ExtractionResult(document=document, entities=self.extract_text(document.text))

    def extract_all(self, documents: Iterable[Document]) -> list[ExtractionResult]:
        return [self.extract(document) for document in documents]


class KeywordExtractor:
    def __init__(
        self,
        preprocessor: TextPreprocessor | None = None,
        top_n: int = 10,
        use_bigrams: bool = True,
        min_bigram_count: int = 2,
    ):
        self.preprocessor = preprocessor or TextPreprocessor(min_token_length=3)
        self.top_n = top_n
        self.use_bigrams = use_bigrams
        self.min_bigram_count = min_bigram_count
        self.document_frequency: Counter[str] = Counter()
        self.document_count = 0

    def terms(self, text: str) -> Counter[str]:
        tokens = self.preprocessor.tokens(text)
        counts = Counter(tokens)
        if self.use_bigrams:
            bigrams = Counter(f"{first} {second}" for first, second in zip(tokens, tokens[1:]) if first != second)
            counts.update({bigram: count for bigram, count in bigrams.items() if count >= self.min_bigram_count})
        return counts

    def fit(self, texts: Iterable[str]) -> KeywordExtractor:
        self.document_frequency = Counter()
        self.document_count = 0
        for text in texts:
            self.document_frequency.update(set(self.terms(text)))
            self.document_count += 1
        return self

    def idf(self, term: str) -> float:
        return math.log((1 + self.document_count) / (1 + self.document_frequency.get(term, 0))) + 1

    def extract(self, text: str, top_n: int | None = None) -> list[tuple[str, float]]:
        counts = self.terms(text)
        total = sum(count for term, count in counts.items() if " " not in term) or 1
        scores = {term: count / total * self.idf(term) for term, count in counts.items()}
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [(term, round(score, 4)) for term, score in ranked[: top_n or self.top_n]]

    def fit_extract(self, texts: Iterable[str], top_n: int | None = None) -> list[list[tuple[str, float]]]:
        texts = list(texts)
        self.fit(texts)
        return [self.extract(text, top_n) for text in texts]


def extract_information(
    documents: Iterable[Document],
    kinds: Iterable[str] | None = None,
    keyword_count: int = 10,
) -> list[ExtractionResult]:
    documents = list(documents)
    results = InformationExtractor(kinds).extract_all(documents)
    if keyword_count > 0:
        keywords = KeywordExtractor(top_n=keyword_count).fit_extract(document.text for document in documents)
        for result, document_keywords in zip(results, keywords):
            result.keywords = document_keywords
    return results


def results_to_frame(results: Iterable[ExtractionResult]) -> pd.DataFrame:
    records = [record for result in results for record in result.to_records()]
    columns = ["document", "kind", "label", "value", "text", "start", "end"]
    return pd.DataFrame(records, columns=columns)
