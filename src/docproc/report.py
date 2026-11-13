from __future__ import annotations

import base64
import io
import json
from collections.abc import Iterable
from html import escape
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from docproc import visualization as viz
from docproc.extraction import ENTITY_LABELS
from docproc.pipeline import CorpusReport, DocumentReport

FORMATS = ("json", "csv", "html")

HTML_STYLE = """
:root { --surface: #fcfcfb; --card: #ffffff; --text: #0b0b0b; --muted: #52514e; --line: #e4e3df; --accent: #2a78d6; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--text); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 32px 24px 64px; }
h1 { margin: 0 0 4px; font-size: 28px; }
h2 { margin: 40px 0 12px; font-size: 20px; border-bottom: 1px solid var(--line); padding-bottom: 6px; }
h3 { margin: 0; font-size: 16px; }
.muted { color: var(--muted); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-top: 24px; }
.tile { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; }
.tile .value { font-size: 26px; font-weight: 600; }
.tile .label { color: var(--muted); font-size: 13px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; background: var(--card); }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.figures { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }
.figures img { width: 100%; border: 1px solid var(--line); border-radius: 8px; background: var(--card); }
details { background: var(--card); border: 1px solid var(--line); border-radius: 8px; margin: 10px 0; padding: 0 16px; }
summary { cursor: pointer; padding: 12px 0; display: flex; gap: 12px; align-items: baseline; flex-wrap: wrap; }
summary .tag { font-size: 12px; color: var(--muted); }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 6px 0 12px; }
.chip { background: #eef4fc; color: #0f3d75; border-radius: 999px; padding: 2px 10px; font-size: 12px; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; margin: 0 0 12px; font-size: 13px; }
dt { color: var(--muted); }
dd { margin: 0; }
.note { color: var(--muted); font-style: italic; }
.cluster { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 12px 16px; margin: 10px 0; }
"""


def export_json(report: CorpusReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def export_csv(report: CorpusReport, directory: str | Path) -> list[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    documents_path = directory / "dokumentumok.csv"
    entities_path = directory / "kinyert_adatok.csv"
    report.statistics_frame().to_csv(documents_path, index=False, encoding="utf-8-sig")
    report.entities_frame().to_csv(entities_path, index=False, encoding="utf-8-sig")
    return [documents_path, entities_path]


def figure_to_base64(figure: Figure) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110, facecolor=viz.SURFACE)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def report_figures(report: CorpusReport) -> list[tuple[str, Figure]]:
    figures: list[tuple[str, Figure]] = []
    if report.corpus_terms:
        figures.append(("Leggyakoribb szavak", viz.plot_top_words(report.corpus_terms[:15], subtitle=f"{len(report.documents)} dokumentum")))
    if len(report.documents) > 1:
        frame = pd.DataFrame(
            {
                "file_type": [item.document.file_type for item in report.documents],
                "word_count": [item.statistics.word_count for item in report.documents],
                "lix": [item.statistics.lix for item in report.documents],
            },
            index=[item.name for item in report.documents],
        )
        figures.append(("Dokumentumok terjedelme", viz.plot_corpus_overview(frame)))
        figures.append(("Olvashatóság", viz.plot_readability(frame)))
    if report.clustering is not None:
        titles = {cluster: f"K{cluster}: {', '.join(word for word, _ in terms[:3])}" for cluster, terms in report.clustering.terms.items()}
        figures.append(("Csoportosítás", viz.plot_cluster_map(report.clustering.coordinates, report.clustering.labels, titles, title="Dokumentumcsoportok")))
    return figures


def _table(frame: pd.DataFrame) -> str:
    header = "".join(f"<th>{escape(str(column))}</th>" for column in frame.columns)
    rows = []
    for record in frame.itertuples(index=False):
        cells = []
        for value in record:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                cells.append("<td></td>")
            elif isinstance(value, (int, float)):
                cells.append(f'<td class="num">{value:g}</td>' if isinstance(value, float) else f'<td class="num">{value}</td>')
            else:
                cells.append(f"<td>{escape(str(value))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _summary_frame(report: CorpusReport) -> pd.DataFrame:
    frame = report.statistics_frame()
    empty = [column for column in ("kategória", "valószínűség", "klaszter") if frame[column].replace("", None).isna().all()]
    return frame.drop(columns=empty)


def _tile(value: object, label: str) -> str:
    return f'<div class="tile"><div class="value">{escape(str(value))}</div><div class="label">{escape(label)}</div></div>'


def _document_section(item: DocumentReport) -> str:
    stats = item.statistics
    top = item.top_category
    category = (
        f"{escape(top[0])} ({top[1]:.0%})" + "".join(f", {escape(label)} ({score:.0%})" for label, score in item.category[1:])
        if top else f'<span class="note">{escape(item.category_note or "–")}</span>'
    )
    details = [
        ("Típus", f"{item.document.file_type.upper()}, {item.document.page_count} oldal"),
        ("Nyelv", stats.language),
        ("Terjedelem", f"{stats.word_count} szó, {stats.sentence_count} mondat, {stats.paragraph_count} bekezdés"),
        ("Olvashatóság", f"LIX {stats.lix} ({stats.readability})"),
        ("Lexikai diverzitás", f"{stats.lexical_diversity}"),
    ]
    rows = "".join(f"<dt>{escape(label)}</dt><dd>{escape(str(value))}</dd>" for label, value in details)
    rows += f"<dt>Kategória</dt><dd>{category}</dd>"
    if item.cluster is not None:
        rows += f"<dt>Csoport</dt><dd>K{item.cluster}</dd>"
    chips = "".join(f'<span class="chip">{escape(term)}</span>' for term, _ in item.extraction.keywords)
    entity_rows = "".join(
        f"<tr><td>{escape(ENTITY_LABELS[kind])}</td><td>{'<br>'.join(escape(value) for value in values)}</td></tr>"
        for kind, values in item.extraction.by_kind().items()
    )
    entities = (
        f"<table><thead><tr><th>adattípus</th><th>értékek</th></tr></thead><tbody>{entity_rows}</tbody></table>"
        if entity_rows else '<p class="note">Nem található kinyerhető adat.</p>'
    )
    return (
        f"<details><summary><h3>{escape(item.name)}</h3>"
        f'<span class="tag">{escape(item.document.file_type.upper())} · {stats.word_count} szó · {len(item.extraction.entities)} kinyert adat</span></summary>'
        f"<dl>{rows}</dl>"
        + (f"<div><strong>Kulcsszavak</strong></div><div class=\"chips\">{chips}</div>" if chips else "")
        + f"{entities}<p></p></details>"
    )


def build_html(report: CorpusReport, include_figures: bool = True) -> str:
    summary = report.summary()
    tiles = "".join(
        [
            _tile(summary["dokumentumok"], "dokumentum"),
            _tile(summary["oldalak összesen"], "oldal"),
            _tile(f"{summary['szavak összesen']:,}".replace(",", " "), "szó"),
            _tile(summary["kinyert adatok"], "kinyert adat"),
            _tile(summary["hibás fájlok"], "hibás fájl"),
        ]
    )
    parts = [
        "<!DOCTYPE html>",
        '<html lang="hu"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Dokumentumfeldolgozási riport – {escape(report.source.name or str(report.source))}</title>",
        f"<style>{HTML_STYLE}</style></head><body><main>",
        "<h1>Dokumentumfeldolgozási riport</h1>",
        f'<div class="muted">Forrás: {escape(str(report.source))} · Készült: {report.created_at:%Y.%m.%d. %H:%M}</div>',
        f'<div class="tiles">{tiles}</div>',
        "<h2>Összesítés</h2>",
        _table(_summary_frame(report)) if report.documents else '<p class="note">Nincs feldolgozott dokumentum.</p>',
    ]
    if report.model_info:
        info = report.model_info
        f1 = info.get("teszt_metrikák", {}).get("f1_macro")
        parts.append(
            f'<p class="muted">Kategorizáló modell: {escape(str(info.get("modell_neve", "?")))}, '
            f'tanítási nyelv: {escape(str(info.get("nyelv", "?")))}'
            + (f", teszt makro F1: {f1:.3f}" if f1 is not None else "")
            + "</p>"
        )
    if include_figures:
        figures = report_figures(report)
        if figures:
            images = "".join(f'<img alt="{escape(title)}" src="data:image/png;base64,{figure_to_base64(figure)}">' for title, figure in figures)
            parts.append(f'<h2>Ábrák</h2><div class="figures">{images}</div>')
    if report.clustering is not None:
        parts.append(f"<h2>Csoportosítás</h2><p class=\"muted\">K-Means, {report.clustering.k} csoport, sziluett-érték: {report.clustering.silhouette}</p>")
        for cluster, terms in report.clustering.terms.items():
            members = [item.name for item in report.documents if item.cluster == cluster]
            parts.append(
                f'<div class="cluster"><strong>K{cluster}</strong> – {escape(", ".join(word for word, _ in terms[:8]))}'
                f'<div class="muted">{escape(", ".join(members))}</div></div>'
            )
    parts.append("<h2>Dokumentumok</h2>")
    parts.extend(_document_section(item) for item in report.documents)
    if report.errors:
        parts.append("<h2>Hibás fájlok</h2>")
        parts.append(_table(pd.DataFrame(report.errors)))
    parts.append("</main></body></html>")
    return "\n".join(parts)


def export_html(report: CorpusReport, path: str | Path, include_figures: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_html(report, include_figures), encoding="utf-8")
    return path


def export_report(report: CorpusReport, output_dir: str | Path, formats: Iterable[str] = FORMATS) -> list[Path]:
    formats = list(formats)
    unknown = set(formats) - set(FORMATS)
    if unknown:
        raise ValueError(f"Ismeretlen formátum: {', '.join(sorted(unknown))} (lehetséges: {', '.join(FORMATS)})")
    output_dir = Path(output_dir)
    created: list[Path] = []
    if "json" in formats:
        created.append(export_json(report, output_dir / "riport.json"))
    if "csv" in formats:
        created.extend(export_csv(report, output_dir))
    if "html" in formats:
        created.append(export_html(report, output_dir / "riport.html"))
    return created
