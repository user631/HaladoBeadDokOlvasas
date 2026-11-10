from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from docproc.config import FIGURES_DIR
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
