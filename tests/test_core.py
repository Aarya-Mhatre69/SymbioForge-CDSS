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
        at.button(key=f"ex_{name}_2").click().run()
        assert at.session_state["step"] == 1 and "ctx" not in at.session_state          # safety screen first, no result yet
        [b for b in at.button if b.label == "Continue"][0].click().run()
        assert not at.exception, name
        assert at.session_state["stage"] == "result"
        assert any("model-estimated chance" in m.value or "Cannot estimate" in m.value or "outside what the model" in m.value for m in at.markdown)
    at.segmented_control(key="view").set_value("How it works").run()
    assert not at.exception


def test_dashboard_guided_intake_flow():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.text_area(key="w_intro").set_value("I'm 58, a man, pressure in my chest when I climb stairs that eases with rest").run()
    at.button[0].click().run()
    assert not at.exception and at.session_state["step"] == 1
    assert at.session_state["ans"]["age"] == 58
    for _ in range(5):  # safety check + four stage-1 steps with defaults
        btn = [b for b in at.button if b.label in ("Continue", "See my result")][0]
        btn.click().run()
        assert not at.exception
    assert at.session_state["stage"] == "result" and at.session_state["ctx"]["stage"] == 1
    # add stress-test results -> full analysis
    [b for b in at.button if b.label == "I have stress-test results"][0].click().run()
    assert at.session_state["stage"] == "intake" and at.session_state["step"] == 6
    for _ in range(2):
        btn = [b for b in at.button if b.label in ("Continue", "Analyse my answers")][0]
        btn.click().run()
        assert not at.exception
    assert at.session_state["ctx"]["stage"] == 2


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


def test_plausibility_guard():
    from cdss import plausibility
    from cdss.orchestrator import assess
    assert plausibility.check({"age": 90, "thalch": 200})["hard"]            # far above age-expected max heart rate
    assert not plausibility.check({"age": 52, "trestbps": 130, "chol": 210})["outside"]
    assert not plausibility.reliable(plausibility.check({"age": 100}))      # beyond the training min-max: withheld
    assert not plausibility.reliable(plausibility.check({"age": 20}))
    assert plausibility.reliable(plausibility.check({"age": 52, "trestbps": 130, "chol": 210}))
    pl = plausibility.check({"age": 52, "chol": 440})                        # edge-of-range only: still estimated, band capped
    assert plausibility.reliable(pl)
    conf = plausibility.apply({"band": "high", "warnings": []}, pl)
    assert conf["band"] in ("moderate", "high")
    from cdss.report import build_report
    from cdss import chat
    import re as _re
    for bad in ({**DEMO_PATIENTS["Clear present"], "age": 90, "thalch": 200}, {**DEMO_PATIENTS["Clear present"], "age": 100}):
        c = assess(PatientInput(**bad))
        assert c["analysis"]["tier"] == "unsure"
        html = _re.sub(r"<[^>]+>", " ", build_report(c["result"], c["analysis"]))
        assert "withheld" in html.lower() and "about" not in html.split("Model details")[0]
        assert "P(disease)" not in html and "new P=" not in html and "P ranges" not in html
        assert chat.answer("what is the second opinion?", c) == chat.WITHHELD and chat.answer("what if my cholesterol were lower?", c) == chat.WITHHELD


def test_stage1_model_tiers_and_guard():
    from cdss.stage1 import Stage1Input, get_stage1
    s1 = get_stage1()
    keys = ("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg")
    hi = s1.assess(Stage1Input(**{k: DEMO_PATIENTS["Clear present"][k] for k in keys}))
    lo = s1.assess(Stage1Input(**{k: DEMO_PATIENTS["Clear absent"][k] for k in keys}))
    from cdss.stage1 import tier_for
    assert tier_for(0.84, ["present"]) == "higher" and tier_for(0.84, ["absent", "present"]) == "mid" and tier_for(0.10, ["absent"]) == "lower"
    # 'Clear present' reports no chest pain: the pain-free presentation is capped at 'mid' by policy (referral-bias protection)
    assert hi["analysis"]["tier"] == "mid" and "capped" in hi["analysis"]["headline"] and lo["analysis"]["tier"] == "lower" and hi["stage"] == 1
    assert "p" in hi["analysis"] and "%" not in hi["analysis"]["headline"]       # tier only, no percentage in the narrative
    typ = s1.assess(Stage1Input(age=35, sex="female", cp="typical angina", trestbps=130, chol=210, fbs=False, restecg="normal"))
    assert typ["analysis"]["tier"] in ("mid", "higher")                           # classic angina is never 'lower'
    silent = s1.assess(Stage1Input(age=62, sex="male", cp="asymptomatic", trestbps=150, chol=260, fbs=True, restecg="lv hypertrophy"))
    assert silent["analysis"]["tier"] != "higher"                                  # pain-free presentation is capped
    odd = s1.assess(Stage1Input(**{**{k: DEMO_PATIENTS["Clear present"][k] for k in keys}, "chol": 590, "trestbps": 215}))
    assert odd["analysis"]["tier"] == "unsure"


def test_counterfactual_only_lowers_modifiable_factors():
    from cdss.engine import get_engine
    r = get_engine().predict(PatientInput(**DEMO_PATIENTS["Borderline / missing data"]))
    act = r["counterfactual"]["actionable"]
    if act:
        assert all(v["to"] < v["from"] for v in act["changes"].values())


def test_clinic_queue_requires_consent_and_orders_by_urgency(tmp_path, monkeypatch):
    from cdss import queue as q
    from cdss.orchestrator import assess
    from cdss.stage1 import Stage1Input, get_stage1
    monkeypatch.setattr(q, "DB", tmp_path / "q.db")
    keys = ("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg")
    lo = get_stage1().assess(Stage1Input(**{k: DEMO_PATIENTS["Clear absent"][k] for k in keys}))
    hi = get_stage1().assess(Stage1Input(**{k: DEMO_PATIENTS["Clear present"][k] for k in keys}))
    with pytest.raises(PermissionError):
        q.submit(lo, "x", consent=False)
    assert q.cases() == []
    q.submit(lo, "low", True)
    q.submit(hi, "high", True)
    rows = q.cases()
    assert [r["name"] for r in rows] == ["high", "low"]
    q.mark_reviewed(rows[0]["id"])
    assert q.cases("new")[0]["name"] == "low"
    # an unreliable ('unsure') case must not outrank an older 'higher' case just because it is newer
    unsure = {**lo, "analysis": {**lo["analysis"], "tier": "unsure", "tier_info": {"label": "x"}}, "result": {**lo["result"], "confidence": {"warnings": ["not shown as reliable"]}}}
    hi2 = {**hi, "analysis": {**hi["analysis"], "tier": "higher"}}
    q.submit(hi2, "older-higher", True)
    q.submit(unsure, "newer-unsure", True)
    assert [r["name"] for r in q.cases("new")][:2] == ["older-higher", "newer-unsure"]


def test_red_flag_stops_the_flow():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.button[0].click().run()                       # Start
    at.checkbox(key="w_rf0").check().run()
    [b for b in at.button if b.label == "Continue"][0].click().run()
    assert at.session_state["stage"] == "emergency" and "ctx" not in at.session_state
    assert any("urgent medical help" in m.value for m in at.markdown)


def test_edit_my_answers_returns_to_safety_screen():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.button(key="ex_Clear present_2").click().run()
    [b for b in at.button if b.label == "Continue"][0].click().run()
    [b for b in at.button if b.label == "Edit my answers"][0].click().run()
    assert at.session_state["step"] == 1 and at.session_state["stage"] == "intake"
    assert "screened" not in at.session_state or not at.session_state["screened"]


def test_clinician_view_is_gated():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.query_params["view"] = "clinician"
    at.run()
    assert any("Passcode" in t.label for t in at.text_input) and not any("Load sample cases" in b.label for b in at.button)
    at.text_input(key="clin_pw").set_value("wrong").run()
    [b for b in at.button if b.label == "Open queue"][0].click().run()
    assert not at.session_state["clin_ok"] if "clin_ok" in at.session_state else True
    at.text_input(key="clin_pw").set_value("demo").run()
    [b for b in at.button if b.label == "Open queue"][0].click().run()
    assert at.session_state["clin_ok"] and any("Load sample cases" in b.label for b in at.button)


def test_ranges_file_is_required(monkeypatch):
    from cdss import plausibility
    monkeypatch.setattr(plausibility, "_RANGES", None)
    monkeypatch.setattr(plausibility, "ARTIFACTS", ROOT / "nonexistent")
    with pytest.raises(FileNotFoundError):
        plausibility.check({"age": 50})


def test_clinician_view_renders():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.query_params["view"] = "clinician"
    at.run()
    assert not at.exception
