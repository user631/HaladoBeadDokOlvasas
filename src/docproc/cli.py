from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

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
