from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from docproc.config import FIGURES_DIR, MODELS_DIR
from docproc.exceptions import DocProcError
from docproc.loaders import iter_documents, supported_extensions


def cmd_load(args: argparse.Namespace) -> int:
    documents = list(iter_documents(args.path, recursive=not args.no_recursive))
    if not documents:
        print("Nem található beolvasható dokumentum.")
        return 1

    if args.json:
        payload = [{**document.summary(), "metaadatok": document.metadata} for document in documents]
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 0

    table = pd.DataFrame([document.summary() for document in documents])
    print(table.to_string(index=False))
    print(f"\nÖsszesen: {len(documents)} dokumentum, {table['szavak'].sum()} szó, {table['oldalak'].sum()} oldal")

    if args.preview > 0:
        for document in documents:
            print(f"\n--- {document.name} ---")
            print(document.preview(args.preview))
    return 0


STATS_COLUMNS = {
    "file_type": "típus",
    "language": "nyelv",
    "word_count": "szavak",
    "unique_word_count": "egyedi",
    "sentence_count": "mondatok",
    "avg_sentence_length": "átl. mondathossz",
    "lexical_diversity": "diverzitás",
    "lix": "LIX",
    "readability": "olvashatóság",
}


def _format_frequencies(frequencies: list[tuple[str, int]]) -> str:
    return ", ".join(f"{word} ({count})" for word, count in frequencies)


def _save_stat_figures(documents, stats: pd.DataFrame, preprocessor, args: argparse.Namespace) -> list[Path]:
    from docproc import visualization as viz
    from docproc.statistics import corpus_word_frequencies, sentence_lengths

    output_dir: Path = args.figures_dir
    prefix = args.prefix or (args.path.stem if args.path.is_file() else args.path.name)
    analyses = [preprocessor.analyze(document.text) for document in documents if document.file_type != "csv"]
    all_words = [word for analysis in analyses for word in analysis.words]
    all_sentence_lengths = [length for analysis in analyses for length in sentence_lengths(analysis)]
    subtitle = f"{len(documents)} dokumentum: {prefix}"

    figures = {
        "gyakori_szavak": viz.plot_top_words(
            corpus_word_frequencies(documents, preprocessor, top_n=args.top), subtitle=subtitle
        ),
        "szohosszok": viz.plot_word_length_distribution(all_words, subtitle=subtitle),
        "mondathosszok": viz.plot_sentence_length_distribution(all_sentence_lengths, subtitle=subtitle),
    }
    if len(documents) > 1:
        figures["terjedelem"] = viz.plot_corpus_overview(stats, subtitle=subtitle)
        figures["olvashatosag"] = viz.plot_readability(stats)
    return [viz.save_figure(figure, output_dir / f"{prefix}_{name}.png") for name, figure in figures.items()]


def cmd_stats(args: argparse.Namespace) -> int:
    from docproc.preprocessing import TextPreprocessor
    from docproc.statistics import corpus_word_frequencies, describe_table, document_statistics

    documents = list(iter_documents(args.path, recursive=not args.no_recursive))
    if not documents:
        print("Nem található beolvasható dokumentum.")
        return 1

    preprocessor = TextPreprocessor(use_stemming=args.stem)
    per_document = {document.name: document_statistics(document, preprocessor, top_n=args.top) for document in documents}
    stats = pd.DataFrame(
        [{"name": name, "file_type": document.file_type, **per_document[name].to_dict()}
         for name, document in zip(per_document, documents)]
    ).set_index("name")

    if args.json:
        payload = {
            name: {"file_type": document.file_type, **per_document[name].to_dict(include_top_words=True)}
            for name, document in zip(per_document, documents)
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(stats[list(STATS_COLUMNS)].rename(columns=STATS_COLUMNS).to_string())

    print("\nLeggyakoribb tartalmas szavak dokumentumonként:")
    for name, document_stats in per_document.items():
        print(f"  {name}: {_format_frequencies(document_stats.top_words)}")

    if len(documents) > 1:
        print("\nLeggyakoribb szavak a teljes gyűjteményben:")
        print("  " + _format_frequencies(corpus_word_frequencies(documents, preprocessor, top_n=args.top)))

    for document in documents:
        description = describe_table(document)
        if description is not None:
            print(f"\nNumerikus oszlopok – {document.name}:")
            print(description.to_string())

    if args.plot:
        print("\nMentett ábrák:")
        for path in _save_stat_figures(documents, stats, preprocessor, args):
            print(f"  {path}")
    return 0


def _highlight_markers() -> tuple[str, str]:
    if sys.stdout.isatty():
        return "\033[1;33m", "\033[0m"
    return "[", "]"


def _index_options(args: argparse.Namespace) -> dict:
    return {"stemming": not args.exact, "fold_accents": not args.exact}


def cmd_search(args: argparse.Namespace) -> int:
    from docproc.search import SearchIndex, regex_search

    documents = list(iter_documents(args.path, recursive=not args.no_recursive))
    if not documents:
        print("Nem található beolvasható dokumentum.")
        return 1
    markers = _highlight_markers()

    if args.regex:
        matches = regex_search(documents, args.query, ignore_case=not args.case_sensitive, context=args.context, markers=markers)
        print(f"{len(matches)} találat a(z) /{args.query}/ mintára ({len(documents)} dokumentumban keresve)\n")
        for match in matches[: args.top] if args.top else matches:
            print(f"{match.document.name}:{match.line_number}  {match.snippet}")
        return 0 if matches else 1

    index = SearchIndex.from_documents(documents, ranking=args.ranking, **_index_options(args))
    hits = index.search(args.query, top_k=args.top, context=args.context, markers=markers)
    print(
        f"{len(hits)} találat a(z) „{args.query}” kifejezésre "
        f"({len(index)} dokumentum, {index.vocabulary_size} kifejezés az indexben)\n"
    )
    for rank, hit in enumerate(hits, start=1):
        counts = ", ".join(f"{term}×{count}" for term, count in hit.term_counts.items())
        print(f"{rank}. {hit.document.name}  (pontszám: {hit.score:.3f}" + (f"; {counts})" if counts else ")"))
        for snippet in hit.snippets:
            print(f"   {snippet}")
        print()
    return 0 if hits else 1


def cmd_count(args: argparse.Namespace) -> int:
    from docproc.search import SearchIndex

    documents = list(iter_documents(args.path, recursive=not args.no_recursive))
    if not documents:
        print("Nem található beolvasható dokumentum.")
        return 1
    index = SearchIndex.from_documents(documents, **_index_options(args))
    table = index.keyword_counts(args.keywords)
    table.loc["összesen"] = table.sum()
    print(table.to_string())
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    from docproc.extraction import ENTITY_LABELS, extract_information, results_to_frame

    documents = list(iter_documents(args.path, recursive=not args.no_recursive))
    if not documents:
        print("Nem található beolvasható dokumentum.")
        return 1
    kinds = [kind.strip() for kind in args.kinds.split(",")] if args.kinds else None
    try:
        results = extract_information(documents, kinds=kinds, keyword_count=args.keywords)
    except ValueError as exc:
        print(f"Hiba: {exc}", file=sys.stderr)
        return 2

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        results_to_frame(results).to_csv(args.csv, index=False, encoding="utf-8-sig")

    if args.json:
        payload = {
            result.document.name: {
                "entities": result.by_kind(),
                "keywords": [{"term": term, "score": score} for term, score in result.keywords],
            }
            for result in results
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    for result in results:
        print(f"=== {result.document.name} ===")
        grouped = result.by_kind()
        if not grouped and not result.keywords:
            print("  (nem található kinyerhető adat)")
        for kind, values in grouped.items():
            print(f"  {ENTITY_LABELS[kind]}:")
            for value in values:
                print(f"    - {value}")
        if result.keywords:
            print("  Kulcsszavak: " + ", ".join(term for term, _ in result.keywords))
        print()
    if args.csv:
        print(f"A kinyert adatok mentve: {args.csv}")
    return 0


def cmd_classify(args: argparse.Namespace) -> int:
    from docproc.classification import VECTORIZER_NAMES, DocumentClassifier
    from docproc.preprocessing import detect_language

    try:
        classifier = DocumentClassifier.load(args.model)
    except FileNotFoundError as exc:
        print(f"Hiba: {exc}", file=sys.stderr)
        return 2
    documents = list(iter_documents(args.path, recursive=not args.no_recursive))
    if not documents:
        print("Nem található beolvasható dokumentum.")
        return 1

    metadata = classifier.metadata
    model_language = metadata.get("nyelv")
    rows = []
    for document in documents:
        scores = classifier.classify(document, top_k=args.top)
        language = detect_language(document.text)
        rows.append(
            {
                "név": document.name,
                "kategória": scores[0][0],
                "valószínűség": scores[0][1],
                "további jelöltek": ", ".join(f"{label} ({score:.2f})" for label, score in scores[1:]),
                "nyelv": language,
                "eltérő nyelv": bool(model_language and language not in (model_language, "unknown")),
                "pontszámok": scores,
            }
        )

    if args.json:
        payload = [{key: value for key, value in row.items() if key != "további jelöltek"} for row in rows]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if metadata:
        f1 = metadata.get("teszt_metrikák", {}).get("f1_macro")
        vectorizer = VECTORIZER_NAMES.get(metadata.get("vektorizáló"), metadata.get("vektorizáló", "?"))
        print(f"Modell: {metadata.get('modell_neve', '?')} + {vectorizer}"
              + (f" (teszt makro F1: {f1:.3f})" if f1 is not None else ""))
        print(f"Kategóriák: {', '.join(classifier.labels)}\n")
    table = pd.DataFrame(rows).drop(columns=["pontszámok", "eltérő nyelv"])
    table["valószínűség"] = table["valószínűség"].map(lambda value: f"{value:.2f}")
    print(table.to_string(index=False))
    if any(row["eltérő nyelv"] for row in rows):
        print(f"\nFigyelem: egyes dokumentumok nyelve eltér a modell tanítási nyelvétől ({model_language}), "
              "ezeknél a besorolás nem megbízható.")
    return 0


def cmd_formats(args: argparse.Namespace) -> int:
    print("Támogatott formátumok: " + ", ".join(supported_extensions()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docproc", description="Automatikus dokumentumfeldolgozó rendszer")
    parser.add_argument("-v", "--verbose", action="store_true", help="részletes naplózás")
    subparsers = parser.add_subparsers(dest="command", required=True)

    load = subparsers.add_parser("load", help="dokumentumok beolvasása és összesítése")
    load.add_argument("path", type=Path, help="fájl vagy mappa")
    load.add_argument("--no-recursive", action="store_true", help="az almappákat nem járja be")
    load.add_argument("--preview", type=int, default=0, metavar="N", help="az első N karakter megjelenítése")
    load.add_argument("--json", action="store_true", help="kimenet JSON formátumban")
    load.set_defaults(func=cmd_load)

    stats = subparsers.add_parser("stats", help="szövegstatisztikák és ábrák")
    stats.add_argument("path", type=Path, help="fájl vagy mappa")
    stats.add_argument("--no-recursive", action="store_true", help="az almappákat nem járja be")
    stats.add_argument("--top", type=int, default=10, metavar="N", help="a leggyakoribb N szó")
    stats.add_argument("--stem", action="store_true", help="szótövezés a szógyakoriságokhoz")
    stats.add_argument("--json", action="store_true", help="kimenet JSON formátumban")
    stats.add_argument("--plot", action="store_true", help="ábrák mentése PNG formátumban")
    stats.add_argument("--figures-dir", type=Path, default=FIGURES_DIR, metavar="DIR", help="az ábrák mappája")
    stats.add_argument("--prefix", default=None, help="az ábrafájlok névelőtagja")
    stats.set_defaults(func=cmd_stats)

    search = subparsers.add_parser("search", help="keresés a dokumentumokban")
    search.add_argument("path", type=Path, help="fájl vagy mappa")
    search.add_argument("query", help='keresőkifejezés, pl. \'szerződés AND (bérleti OR "havi díj") NOT számla\'')
    search.add_argument("--top", type=int, default=10, metavar="K", help="legfeljebb K találat")
    search.add_argument("--regex", action="store_true", help="a kifejezés reguláris kifejezés")
    search.add_argument("--case-sensitive", action="store_true", help="kis- és nagybetű megkülönböztetése (regex)")
    search.add_argument("--exact", action="store_true", help="szótövezés és ékezetfüggetlenség nélkül")
    search.add_argument("--ranking", choices=("bm25", "tfidf"), default="bm25", help="rangsorolási módszer")
    search.add_argument("--context", type=int, default=60, metavar="N", help="kontextus hossza karakterben")
    search.add_argument("--no-recursive", action="store_true", help="az almappákat nem járja be")
    search.set_defaults(func=cmd_search)

    count = subparsers.add_parser("count", help="kulcsszavak előfordulása dokumentumonként")
    count.add_argument("path", type=Path, help="fájl vagy mappa")
    count.add_argument("keywords", nargs="+", help="kulcsszavak vagy többszavas kifejezések")
    count.add_argument("--exact", action="store_true", help="szótövezés és ékezetfüggetlenség nélkül")
    count.add_argument("--no-recursive", action="store_true", help="az almappákat nem járja be")
    count.set_defaults(func=cmd_count)

    extract = subparsers.add_parser("extract", help="információk kinyerése (e-mail, telefon, dátum, összeg, ...)")
    extract.add_argument("path", type=Path, help="fájl vagy mappa")
    extract.add_argument(
        "--kinds",
        help="csak a megadott adattípusok, vesszővel elválasztva "
             "(email,url,phone,date,amount,percentage,iban,bank_account,tax_number,postal_address,company,field)",
    )
    extract.add_argument("--keywords", type=int, default=10, metavar="N", help="kulcsszavak száma (0 = kikapcsolva)")
    extract.add_argument("--json", action="store_true", help="kimenet JSON formátumban")
    extract.add_argument("--csv", type=Path, metavar="FÁJL", help="a kinyert adatok mentése CSV-be")
    extract.add_argument("--no-recursive", action="store_true", help="az almappákat nem járja be")
    extract.set_defaults(func=cmd_extract)

    classify = subparsers.add_parser("classify", help="dokumentumok kategorizálása a betanított modellel")
    classify.add_argument("path", type=Path, help="fájl vagy mappa")
    classify.add_argument("--model", type=Path, default=MODELS_DIR / "kategorizalo.joblib", help="a modellfájl útvonala")
    classify.add_argument("--top", type=int, default=3, metavar="K", help="a K legvalószínűbb kategória")
    classify.add_argument("--json", action="store_true", help="kimenet JSON formátumban")
    classify.add_argument("--no-recursive", action="store_true", help="az almappákat nem járja be")
    classify.set_defaults(func=cmd_classify)

    formats = subparsers.add_parser("formats", help="támogatott fájlformátumok listája")
    formats.set_defaults(func=cmd_formats)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    try:
        return args.func(args)
    except DocProcError as exc:
        print(f"Hiba: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
