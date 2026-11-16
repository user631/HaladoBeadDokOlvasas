from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from docproc.config import MODELS_DIR, PROJECT_ROOT, SAMPLES_DIR

APP_PATH = PROJECT_ROOT / "app" / "streamlit_app.py"
MODEL_PATH = MODELS_DIR / "kategorizalo.joblib"

pytestmark = pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="a demó dokumentumok nincsenek legenerálva")


@pytest.fixture
def app() -> AppTest:
    return AppTest.from_file(str(APP_PATH), default_timeout=120).run()


def metric_values(app: AppTest) -> dict[str, str]:
    return {metric.label: metric.value for metric in app.metric}


def test_overview_with_samples(app: AppTest):
    assert not app.exception
    assert len(app.tabs) == 7
    values = metric_values(app)
    assert values["Dokumentumok"] == "7"
    assert values["Hibás fájlok"] == "0"


def test_keyword_search(app: AppTest):
    app.text_input[0].set_value("szerződés AND bérleti").run()
    assert not app.exception
    results = [block.value for block in app.markdown if "pontszám" in block.value]
    assert results and "berleti_szerzodes.pdf" in results[0]


def test_invalid_query_shows_error(app: AppTest):
    app.text_input[0].set_value("(számla").run()
    assert any("zárójel" in error.value for error in app.error)


def test_regex_search(app: AppTest):
    mode = next(radio for radio in app.radio if radio.label == "Mód")
    mode.set_value("Reguláris kifejezés").run()
    app.text_input[0].set_value(r"\d{8}-\d-\d{2}").run()
    assert not app.exception
    assert any("találat" in caption.value for caption in app.caption)


def test_keyword_counts(app: AppTest):
    app.text_input[1].set_value("számla, bérleti díj").run()
    assert not app.exception
    assert any(list(frame.value.columns) == ["számla", "bérleti díj"] for frame in app.dataframe)


def test_english_source(app: AppTest):
    app.sidebar.radio[0].set_value("Angol szövegek (kategorizálás)").run()
    assert not app.exception
    assert metric_values(app)["Dokumentumok"] == "6"


def test_upload_without_files(app: AppTest):
    app.sidebar.radio[0].set_value("Saját fájlok feltöltése").run()
    assert not app.exception
    assert any("Tölts fel" in info.value for info in app.info)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="nincs betanított modell (python scripts/train.py)")
def test_classify_free_text(app: AppTest):
    app.text_area[0].set_value("The goalie made a great save and the team won the hockey game in overtime.").run()
    assert not app.exception
    assert any("sport" in message.value for message in app.success)
