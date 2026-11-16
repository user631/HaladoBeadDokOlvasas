from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docproc import visualization as viz
from docproc.classification import VECTORIZER_NAMES, DocumentClassifier
from docproc.clustering import cluster_documents
from docproc.config import DATA_DIR, MODELS_DIR, SAMPLES_DIR
from docproc.exceptions import QuerySyntaxError
from docproc.extraction import ENTITY_LABELS
from docproc.loaders import supported_extensions
from docproc.pipeline import CorpusReport, DocumentPipeline, DocumentReport
from docproc.preprocessing import detect_language
from docproc.report import build_html, report_figures, summary_frame
from docproc.search import SearchIndex, regex_search
from docproc.statistics import describe_table

MODEL_PATH = MODELS_DIR / "kategorizalo.joblib"
UPLOAD_OPTION = "Saját fájlok feltöltése"
DEMO_SOURCES = {
    "Magyar minta dokumentumok": SAMPLES_DIR,
    "Angol szövegek (kategorizálás)": DATA_DIR / "demo_en",
}
HIGHLIGHT = ("\x00", "\x01")
SEARCH_HELP = """
| Szintaxis | Jelentés |
|---|---|
| `szerződés bérleti` | mindkét szó szerepel |
| `a OR b`, `a VAGY b` | bármelyik szó szerepel |
| `NOT a`, `NEM a` | a szó nem szerepel |
| `"bérleti díj"` | pontos kifejezés |
| `szerz*` | előtagra illeszkedés |
| `( ... )` | csoportosítás |
"""


def model_mtime() -> float:
    return MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0.0


@st.cache_resource(show_spinner=False)
def load_classifier(mtime: float) -> DocumentClassifier | None:
    return DocumentClassifier.load(MODEL_PATH) if mtime else None


def build_pipeline(use_classifier: bool, use_clustering: bool, keyword_count: int) -> DocumentPipeline:
    return DocumentPipeline(
        classifier=load_classifier(model_mtime()) if use_classifier else None,
        model_path=None,
        use_classifier=use_classifier,
        use_clustering=use_clustering,
        keyword_count=keyword_count,
    )


@st.cache_data(show_spinner="Dokumentumok feldolgozása…")
def process_folder(folder: str, use_classifier: bool, use_clustering: bool, keyword_count: int, mtime: float) -> CorpusReport:
    return build_pipeline(use_classifier, use_clustering, keyword_count).process(folder)


@st.cache_data(show_spinner="Feltöltött fájlok feldolgozása…")
def process_uploads(
    files: tuple[tuple[str, bytes], ...],
    use_classifier: bool,
    use_clustering: bool,
    keyword_count: int,
    mtime: float,
) -> CorpusReport:
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        for name, data in files:
            (folder / Path(name).name).write_bytes(data)
        report = build_pipeline(use_classifier, use_clustering, keyword_count).process(folder)
    report.source = Path("feltöltött fájlok")
    return report


def highlighted_html(text: str) -> str:
    return escape(text).replace(HIGHLIGHT[0], "<mark>").replace(HIGHLIGHT[1], "</mark>")


def show_figure(figure) -> None:
    st.pyplot(figure, clear_figure=True)


def sidebar() -> tuple[CorpusReport | None, bool]:
    st.sidebar.title("DocProc")
    st.sidebar.caption("Automatikus dokumentumfeldolgozás: PDF, TXT, CSV")
    source = st.sidebar.radio("Dokumentumok forrása", [UPLOAD_OPTION, *DEMO_SOURCES], index=1)

    classifier = load_classifier(model_mtime())
    with st.sidebar.expander("Beállítások", expanded=False):
        use_classifier = st.checkbox("Kategorizálás", value=classifier is not None, disabled=classifier is None)
        if classifier is None:
            st.caption("Nincs betanított modell. Tanítás: `python scripts/train.py`")
        use_clustering = st.checkbox("Csoportosítás", value=True)
        keyword_count = st.slider("Kulcsszavak dokumentumonként", 0, 20, 10)

    if source == UPLOAD_OPTION:
        extensions = [extension.lstrip(".") for extension in supported_extensions()]
        uploads = st.sidebar.file_uploader("Fájlok", type=extensions, accept_multiple_files=True)
        if not uploads:
            return None, use_classifier
        files = tuple(sorted((upload.name, upload.getvalue()) for upload in uploads))
        return process_uploads(files, use_classifier, use_clustering, keyword_count, model_mtime()), use_classifier

    folder = DEMO_SOURCES[source]
    if not folder.exists():
        st.sidebar.error(f"A mappa nem található: {folder}. Generálás: `python scripts/generate_samples.py`")
        return None, use_classifier
    return process_folder(str(folder), use_classifier, use_clustering, keyword_count, model_mtime()), use_classifier


def overview_tab(report: CorpusReport) -> None:
    summary = report.summary()
    columns = st.columns(5)
    columns[0].metric("Dokumentumok", summary["dokumentumok"])
    columns[1].metric("Oldalak", summary["oldalak összesen"])
    columns[2].metric("Szavak", f"{summary['szavak összesen']:,}".replace(",", " "))
    columns[3].metric("Kinyert adatok", summary["kinyert adatok"])
    columns[4].metric("Hibás fájlok", summary["hibás fájlok"])
    for error in report.errors:
        st.warning(error["hiba"])

    st.subheader("Összesítés")
    st.dataframe(summary_frame(report), hide_index=True, width="stretch")

    figures = report_figures(report)
    for start in range(0, len(figures), 2):
        cells = st.columns(2)
        for cell, (_, figure) in zip(cells, figures[start:start + 2]):
            with cell:
                show_figure(figure)


def category_text(item: DocumentReport) -> str:
    if item.top_category:
        label, score = item.top_category
        return f"{label} ({score:.0%})"
    return item.category_note or "–"


def document_tab(report: CorpusReport) -> None:
    names = [item.name for item in report.documents]
    selected = st.selectbox("Dokumentum", names)
    item = next(entry for entry in report.documents if entry.name == selected)
    stats = item.statistics

    columns = st.columns(5)
    columns[0].metric("Szavak", stats.word_count)
    columns[1].metric("Mondatok", stats.sentence_count)
    columns[2].metric("LIX", stats.lix, help=f"olvashatóság: {stats.readability}")
    columns[3].metric("Lexikai diverzitás", stats.lexical_diversity)
    columns[4].metric("Nyelv", stats.language)

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**Típus:** {item.document.file_type.upper()}, {item.document.page_count} oldal  \n"
                    f"**Kategória:** {category_text(item)}"
                    + (f"  \n**Csoport:** K{item.cluster}" if item.cluster is not None else ""))
        if item.extraction.keywords:
            st.markdown("**Kulcsszavak:** " + " ".join(f":blue-background[{term}]" for term, _ in item.extraction.keywords))
        st.markdown("**Kinyert adatok**")
        grouped = item.extraction.by_kind()
        if grouped:
            rows = [{"adattípus": ENTITY_LABELS[kind], "érték": value} for kind, values in grouped.items() for value in values]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.caption("Nem található kinyerhető adat.")
    with right:
        if item.category:
            show_figure(viz.plot_probabilities(item.category))
        if stats.top_words:
            show_figure(viz.plot_top_words(stats.top_words, subtitle=item.name))

    if item.document.table is not None:
        st.markdown("**Táblázat**")
        st.dataframe(item.document.table, hide_index=True, width="stretch")
        description = describe_table(item.document)
        if description is not None:
            st.markdown("**Numerikus oszlopok**")
            st.dataframe(description, width="stretch")
    with st.expander("Kinyert szöveg"):
        st.text(item.document.text)
    with st.expander("Metaadatok"):
        st.json(item.document.metadata)


def search_tab(report: CorpusReport) -> None:
    documents = [item.document for item in report.documents]
    query = st.text_input("Keresőkifejezés", placeholder='pl. szerződés AND (bérleti OR "havi díj") NOT számla')
    columns = st.columns(4)
    mode = columns[0].radio("Mód", ["Kulcsszavas", "Reguláris kifejezés"], horizontal=False)
    ranking = columns[1].selectbox("Rangsorolás", ["bm25", "tfidf"], format_func=str.upper, disabled=mode != "Kulcsszavas")
    exact = columns[2].checkbox("Pontos egyezés", help="szótövezés és ékezetfüggetlenség nélkül")
    top = columns[3].slider("Találatok száma", 1, 50, 10)
    with st.expander("Keresési szintaxis"):
        st.markdown(SEARCH_HELP)

    if query:
        try:
            if mode == "Kulcsszavas":
                index = SearchIndex.from_documents(documents, ranking=ranking, stemming=not exact, fold_accents=not exact)
                hits = index.search(query, top_k=top, markers=HIGHLIGHT)
                st.caption(f"{len(hits)} találat · {len(index)} dokumentum · {index.vocabulary_size} kifejezés az indexben")
                for rank, hit in enumerate(hits, start=1):
                    counts = ", ".join(f"{term}×{count}" for term, count in hit.term_counts.items())
                    st.markdown(f"**{rank}. {escape(hit.document.name)}** · pontszám: {hit.score:.3f}" + (f" · {escape(counts)}" if counts else ""))
                    for snippet in hit.snippets:
                        st.markdown(f"<div style='margin:0 0 6px 16px'>{highlighted_html(snippet)}</div>", unsafe_allow_html=True)
            else:
                matches = regex_search(documents, query, markers=HIGHLIGHT)
                st.caption(f"{len(matches)} találat")
                for match in matches[:top]:
                    st.markdown(f"**{escape(match.document.name)}:{match.line_number}** – {highlighted_html(match.snippet)}", unsafe_allow_html=True)
        except QuerySyntaxError as exc:
            st.error(str(exc))

    st.divider()
    keywords = st.text_input("Kulcsszavak számlálása (vesszővel elválasztva)", placeholder="pl. számla, bérleti díj, határidő")
    terms = [term.strip() for term in keywords.split(",") if term.strip()]
    if terms:
        index = SearchIndex.from_documents(documents, stemming=not exact, fold_accents=not exact)
        table = index.keyword_counts(terms)
        table.loc["összesen"] = table.sum()
        st.dataframe(table, width="stretch")


def extraction_tab(report: CorpusReport) -> None:
    frame = report.entities_frame()
    if frame.empty:
        st.info("A dokumentumokból nem sikerült adatot kinyerni.")
        return
    columns = st.columns(2)
    labels = columns[0].multiselect("Adattípusok", sorted(frame["label"].unique()))
    names = columns[1].multiselect("Dokumentumok", sorted(frame["document"].unique()))
    if labels:
        frame = frame[frame["label"].isin(labels)]
    if names:
        frame = frame[frame["document"].isin(names)]
    counts = frame.groupby("label").size().sort_values(ascending=False)
    st.caption(" · ".join(f"{label}: {count}" for label, count in counts.items()))
    st.dataframe(frame[["document", "label", "value", "text"]], hide_index=True, width="stretch")
    st.download_button("Táblázat letöltése (CSV)", frame.to_csv(index=False).encode("utf-8-sig"), "kinyert_adatok.csv", "text/csv")


def clustering_tab(report: CorpusReport) -> None:
    if len(report.documents) < 3:
        st.info("A csoportosításhoz legalább 3 dokumentum szükséges.")
        return
    automatic = st.checkbox("Automatikus klaszterszám (sziluett-érték alapján)", value=True)
    k = None if automatic else st.slider("Klaszterek száma", 2, max(2, min(10, len(report.documents) - 1)), 2)
    languages = Counter(item.statistics.language for item in report.documents)
    language = languages.most_common(1)[0][0]
    try:
        result = cluster_documents(
            [item.document.text for item in report.documents],
            k=k,
            k_range=range(2, min(9, len(report.documents))),
            language=language if language in ("hu", "en") else None,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    columns = st.columns(2)
    columns[0].metric("Klaszterek", result.k)
    columns[1].metric("Sziluett-érték", result.silhouette)
    for cluster, description in result.describe(n_terms=8).items():
        members = [item.name for item, label in zip(report.documents, result.labels) if label == cluster]
        with st.container(border=True):
            st.markdown(f"**K{cluster}** ({len(members)} dokumentum) – {description}")
            st.caption(", ".join(members))
    titles = {cluster: f"K{cluster}: {', '.join(word for word, _ in terms[:3])}" for cluster, terms in result.terms.items()}
    left, right = st.columns(2)
    with left:
        show_figure(viz.plot_cluster_map(result.coordinates, result.labels, titles, title="Dokumentumcsoportok", columns=2))
    if result.k_scores is not None and len(result.k_scores) > 1:
        with right:
            show_figure(viz.plot_k_selection(result.k_scores, chosen_k=result.k))


def classification_tab(use_classifier: bool) -> None:
    classifier = load_classifier(model_mtime())
    if classifier is None:
        st.info("Nincs betanított modell. Futtasd: `python scripts/train.py`")
        return
    metadata = classifier.metadata
    metrics = metadata.get("teszt_metrikák", {})
    columns = st.columns(4)
    columns[0].metric("Modell", metadata.get("modell_neve", "?"))
    columns[1].metric("Vektorizáló", VECTORIZER_NAMES.get(metadata.get("vektorizáló"), "?"))
    columns[2].metric("Teszt makro F1", f"{metrics.get('f1_macro', 0):.3f}")
    columns[3].metric("Pontosság", f"{metrics.get('accuracy', 0):.3f}")
    st.caption(f"Adathalmaz: {metadata.get('adathalmaz', '?')} · tanítási nyelv: {metadata.get('nyelv', '?')} · "
               f"kategóriák: {', '.join(classifier.labels)}")
    if not use_classifier:
        st.caption("A feltöltött dokumentumok kategorizálása a beállításokban ki van kapcsolva.")

    text = st.text_area("Saját szöveg kategorizálása", height=150,
                        placeholder="Illessz be egy angol nyelvű szöveget, pl. egy hírt vagy fórumbejegyzést.")
    if text.strip():
        language = detect_language(text)
        if metadata.get("nyelv") and language not in (metadata["nyelv"], "unknown"):
            st.warning(f"A szöveg nyelve ({language}) eltér a modell tanítási nyelvétől ({metadata['nyelv']}), a besorolás nem megbízható.")
        scores = classifier.predict_scores([text], top_k=len(classifier.labels))[0]
        st.success(f"Becsült kategória: **{scores[0][0]}** ({scores[0][1]:.0%})")
        chart, _ = st.columns([3, 2])
        with chart:
            show_figure(viz.plot_probabilities(scores))


def report_tab(report: CorpusReport) -> None:
    st.markdown("A teljes feldolgozás eredménye letölthető riportként.")
    columns = st.columns(4)
    columns[0].download_button("HTML riport", build_html(report).encode("utf-8"), "riport.html", "text/html")
    columns[1].download_button(
        "JSON",
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=str).encode("utf-8"),
        "riport.json",
        "application/json",
    )
    columns[2].download_button("Dokumentumok (CSV)", report.statistics_frame().to_csv(index=False).encode("utf-8-sig"), "dokumentumok.csv", "text/csv")
    columns[3].download_button("Kinyert adatok (CSV)", report.entities_frame().to_csv(index=False).encode("utf-8-sig"), "kinyert_adatok.csv", "text/csv")


def main() -> None:
    st.set_page_config(page_title="DocProc – dokumentumfeldolgozó", layout="wide")
    report, use_classifier = sidebar()
    st.title("Automatikus dokumentumfeldolgozó")
    if report is None:
        st.info("Tölts fel PDF, TXT vagy CSV fájlokat a bal oldali sávban, vagy válassz egy minta gyűjteményt.")
        return
    if not report.documents:
        st.warning("Egyik fájlt sem sikerült beolvasni.")
        for error in report.errors:
            st.caption(error["hiba"])
        return

    tabs = st.tabs(["Áttekintés", "Dokumentum", "Keresés", "Kinyert adatok", "Csoportosítás", "Kategorizálás", "Riport"])
    with tabs[0]:
        overview_tab(report)
    with tabs[1]:
        document_tab(report)
    with tabs[2]:
        search_tab(report)
    with tabs[3]:
        extraction_tab(report)
    with tabs[4]:
        clustering_tab(report)
    with tabs[5]:
        classification_tab(use_classifier)
    with tabs[6]:
        report_tab(report)


main()
