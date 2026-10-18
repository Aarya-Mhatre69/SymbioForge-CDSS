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


def test_dashboard_runs_for_every_demo():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    for name in DEMO_PATIENTS:
        at.selectbox(key="demo").select(name).run()
        assert not at.exception, name


def test_no_constant_imputed_chol_artifact():
    imputed = FULL[FULL.chol_missing == 1] if "chol_missing" in FULL else FULL.iloc[0:0]
    assert len(imputed) == 0 or imputed.chol.nunique() > 20  # imputed values must not collapse to one constant
