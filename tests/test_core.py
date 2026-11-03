import json
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from cdss import DATA, ARTIFACTS, ROOT, advisor, rules
from cdss.engine import DEMO_PATIENTS, get_engine
from cdss.features import FEATURES, Imputer, clean_raw, engineer
from cdss.schemas import PatientInput, field_errors

E = get_engine()
FULL = pd.read_csv(DATA / "Task4_clean_dataset.csv")
FEAT = pd.read_csv(DATA / "Task5_Feature_Matrix.csv")


def _typed(d):
    d = d.copy()
    for c in ("fbs", "exang"):
        d[c] = d[c].astype(bool)
    return d


def test_feature_parity_and_order():
    X = engineer(_typed(FULL))
    assert list(X.columns) == FEATURES
    assert np.abs(X.values - FEAT[FEATURES].values).max() < 1e-9


def test_formulas_by_hand():
    r = pd.DataFrame([dict(age=60, sex="male", cp="asymptomatic", trestbps=150, chol=250, fbs=True, restecg="normal",
                           thalch=120, exang=True, oldpeak=2.0, slope="flat", ca=1, thal="fixed defect")])
    f = engineer(r).iloc[0]
    assert f["cardiac_risk_score"] == 8 and f["hr_reserve"] == 40 and f["rate_pressure_product"] == 18000
    assert f["oldpeak_slope_risk"] == 4.0 and f["cp_asymptomatic"] == 1


def test_model_parity_with_reported_metrics():
    te = FEAT[FEAT.split == "test"]
    from sklearn.metrics import roc_auc_score
    meta = json.loads((ARTIFACTS / "meta.json").read_text())["test_metrics"]
    assert abs(roc_auc_score(te.target, E.p_xgb(te)) - meta["xgb"]["roc_auc"]) < 0.005
    assert abs(roc_auc_score(te.target, E.p_dnn(te)) - meta["dnn"]["roc_auc"]) < 0.005  # numpy DNN == torch DNN


def test_rules_tristate():
    base = dict(age=40, sex="male", cp="asymptomatic", trestbps=120, chol=200, thalch=150, oldpeak=0.0,
                slope=None, ca=None, thal=None)
    assert rules.evaluate(base)["verdict"] == "undetermined"        # not evaluated on imputed values
    assert rules.evaluate({**base, "thal": "reversable defect"})["fired"] == ["R1"]
    r = rules.evaluate({**base, "cp": "typical angina", "slope": "upsloping", "thal": "normal", "ca": 0})
    assert r["verdict"] == "absent"                                  # R7 default
    assert rules.evaluate({**base, "oldpeak": 3.0})["verdict"] == "present"   # R4 fires regardless of unknowns
    assert rules.evaluate({**base, "oldpeak": 2.8})["state"]["R4"] is False   # strict >


def test_rule_boundaries():
    b = dict(age=58, sex="female", cp="atypical angina", trestbps=153, chol=200, thalch=150, oldpeak=0,
             slope="upsloping", ca=0, thal="normal")
    assert rules.evaluate(b)["fired"] == ["R5", "R6"]


def test_validation_errors_are_field_level():
    with pytest.raises(ValidationError) as ei:
        PatientInput(**{**DEMO_PATIENTS["Clear absent"], "chol": 0, "age": 10, "cp": "none"})
    assert {"chol", "age", "cp"} <= set(field_errors(ei.value))
    assert PatientInput(**{**DEMO_PATIENTS["Clear absent"], "ca": "?"}).ca is None


@pytest.mark.parametrize("name", list(DEMO_PATIENTS))
def test_demo_patients_end_to_end(name):
    r = E.predict(PatientInput(**DEMO_PATIENTS[name]))
    assert set(r) >= {"models", "verdict", "rules", "confidence", "explanation", "counterfactual", "advice", "summary", "disclaimer"}
    assert 0 <= r["models"]["xgb"] <= 1
    assert "not a diagnosis" in r["summary"]


def test_demo_outcomes():
    ok = lambda n: E.predict(PatientInput(**DEMO_PATIENTS[n]))
    assert ok("Clear present")["verdict"] == "present" and ok("Clear absent")["verdict"] == "absent"
    d = ok("Rule vs ML disagreement")
    assert d["rules"]["agrees_with_ml"] is False and any("rule base" in w for w in d["confidence"]["warnings"])


def test_imputation_warning_and_cap():
    r = E.predict(PatientInput(**DEMO_PATIENTS["Rule vs ML disagreement"]))
    assert r["imputed"] == ["slope", "ca", "thal"] and r["confidence"]["band"] != "high"
    assert any("Reduced reliability" in w for w in r["confidence"]["warnings"])


def test_deterministic():
    p = PatientInput(**DEMO_PATIENTS["Borderline / missing data"])
    a, b = E.predict(p), E.predict(p)
    assert a["models"] == b["models"] and a["counterfactual"] == b["counterfactual"]


def test_astar_plan():
    raw = dict(DEMO_PATIENTS["Borderline / missing data"])
    plan = advisor.astar_plan(raw, ["slope", "ca", "thal"])
    assert plan["tests"] == ["slope", "ca"] and plan["cost"] == 12


def test_cleaning_steps():
    raw = pd.DataFrame({"id": [1, 2, 2], "age": [50, 60, 60], "sex": ["Male", "Female", "Female"], "dataset": ["A"] * 3,
                        "cp": ["asymptomatic"] * 3, "trestbps": [0, 130, 130], "chol": [0, 200, 200], "fbs": ["?", "True", "True"],
                        "restecg": ["normal"] * 3, "thalch": [150] * 3, "exang": ["False"] * 3, "oldpeak": [1.0] * 3,
                        "slope": ["?"] * 3, "ca": [np.nan] * 3, "thal": ["?"] * 3, "num": [0, 2, 2]})
    c = clean_raw(raw)
    assert len(c) == 2 and c["trestbps"].isna().sum() == 1 and c["target"].tolist() == [0, 1] and c["sex"][0] == "male"


def test_dashboard_examples_and_how_it_works():
    from streamlit.testing.v1 import AppTest
    for name in DEMO_PATIENTS:
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        assert not at.exception
        at.button(key=f"ex_{name}").click().run()
        assert not at.exception, name
        assert any("model-estimated chance" in m.value for m in at.markdown)
    at.segmented_control(key="view").set_value("How it works").run()
    assert not at.exception


def test_dashboard_guided_intake_flow():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.text_area(key="w_intro").set_value("I'm 58, a man, pressure in my chest when I climb stairs that eases with rest").run()
    at.button[0].click().run()
    assert not at.exception and at.session_state["step"] == 1
    assert at.session_state["ans"]["age"] == 58
    for _ in range(5):  # Continue through all five steps with defaults
        btn = [b for b in at.button if b.label in ("Continue", "Analyse my answers")][0]
        btn.click().run()
        assert not at.exception
    assert at.session_state["stage"] == "result"


def test_orchestrator_trace_and_analysis():
    from cdss.orchestrator import assess
    c = assess(PatientInput(**DEMO_PATIENTS["Borderline / missing data"]))
    agents = [t["agent"] for t in c["result"]["trace"]]
    assert agents[0] == "Intake agent" and agents[-1] == "Synthesizer" and len(agents) == 10
    a = c["analysis"]
    assert a["tier"] in ("lower", "mid", "higher") and len(a["domains"]) == 4 and len(a["cards"]) == 13
    assert abs(sum(d["impact"] for d in a["domains"]) - sum(c["result"]["shap_all"].values())) < 1e-9
    assert set(a["curves"]) == {"chol", "trestbps", "thalch"}


def test_glossary_status_and_chat():
    from cdss import chat
    from cdss.glossary import status
    from cdss.orchestrator import assess
    assert status("chol", {"chol": 250})[0] == "flag" and status("chol", {"chol": 180})[0] == "ok"
    assert status("thalch", {"thalch": 100, "age": 60})[0] == "flag"   # 100/160 = 62%
    assert status("ca", {"ca": None})[0] == "missing"
    p = chat.parse_intro("I'm 58, a man, pressure in my chest when I climb stairs that eases with rest")
    assert p["age"] == 58 and p["sex"] == "male" and p["pressure"] and p["exertion"] and p["relief"]
    assert chat.parse_intro("52 year old woman, no chest pain") == {"age": 52, "sex": "female", "pain": False}
    c = assess(PatientInput(**DEMO_PATIENTS["Clear present"]))
    assert "thallium" in chat.answer("What does thallium mean?", c).lower()
    assert "emergency" in chat.answer("I have severe chest pain right now", c).lower()
    for q in chat.SUGGESTIONS:
        assert len(chat.answer(q, c)) > 40


def test_no_constant_imputed_chol_artifact():
    imputed = FULL[FULL.chol_missing == 1] if "chol_missing" in FULL else FULL.iloc[0:0]
    assert len(imputed) == 0 or imputed.chol.nunique() > 20  # imputed values must not collapse to one constant


def test_clinic_round_trip_links(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("SYMBIOFORGE_CLINIC_URL", "http://localhost:8600")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.query_params["from"] = "clinic"
    at.run()
    assert not at.exception and at.session_state["from_clinic"]
    assert any("Back to Meridian" in m.value and "localhost:8600/index.html" in m.value for m in at.markdown)
    at.button(key="ex_Clear present").click().run()
    assert not at.exception and at.session_state["stage"] == "result"
    at2 = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()      # no ?from=clinic: no back link
    assert not any("Back to Meridian" in m.value for m in at2.markdown)
    monkeypatch.delenv("SYMBIOFORGE_CLINIC_URL")                                 # no clinic address configured: no dead links
    at3 = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at3.query_params["from"] = "clinic"
    at3.run()
    assert not at3.exception and not any("Back to Meridian" in m.value for m in at3.markdown)


def test_clinic_site_wiring():
    import re
    c = ROOT / "clinic"
    cfg = (c / "config.js").read_text()
    assert "localhost:8501" in cfg and "https://symbioforge.streamlit.app" in cfg
    for page in ("index.html", "services.html", "heart-check.html", "book.html"):
        html = (c / page).read_text(encoding="utf-8")
        assert "config.js" in html and "app.js" in html and "styles.css" in html, page
        for href in re.findall(r'href="([^"#?]+\.html)"', html):
            assert (c / href).exists(), (page, href)
    hc = (c / "heart-check.html").read_text(encoding="utf-8")
    assert "data-app-link" in hc and "<iframe" not in hc


def test_how_it_works_sections_render_fast_and_light():
    import time
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    at.segmented_control(key="view").set_value("How it works").run()
    assert not at.exception
    for sec in ("Architecture", "Prediction internals", "Evaluation", "Data (EDA)", "Limits and decisions"):
        t0 = time.perf_counter()
        at.segmented_control(key="how_section").set_value(sec).run()
        assert not at.exception, sec
        assert len(at.markdown) > 2, sec
        assert time.perf_counter() - t0 < 8, sec                                # generous bound; typically well under 1 s
    assert not at.dataframe and not at.metric                                   # no heavy data-grid / metric components


def test_lite_helpers_escape_and_render():
    from ui import lite
    h = lite.table([["<script>alert(1)</script>", 1.5]], ["a", "b"], num_cols=(1,))
    assert "<script>" not in h and "&lt;script&gt;" in h and "class='n'" in h
    b = lite.bars([("x", 0.5), ("y", -0.25)])
    assert "<b style='left:50%" in b and "+0.50" in b and "-0.25" in b
    svg = lite.curve([1, 2, 3, 4], [0.1, 0.4, 0.6, 0.9], 3, "Cholesterol")
    assert svg.startswith("<svg") and "<path" in svg and svg.count("<circle") == 1
    assert "data:image/png;base64," in lite.img(ROOT / "reports" / "figures" / "roc.png", "ROC", 300)


def test_result_page_uses_light_charts_only():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    at.button(key="ex_Clear present").click().run()
    html = " ".join(m.value for m in at.markdown)
    assert "lt-bars" in html and "<svg" in html and "plotly" not in html.lower()
    assert not at.dataframe and not at.metric
