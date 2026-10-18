"""Orchestrator: runs the eight Task 1-8 workflows through the engine, then synthesises a patient-level analytical overview.
engine.predict() produces the raw evidence (and an honest trace); this module turns it into a story."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .advisor import COST
from .engine import Engine, get_engine
from .features import FEATURES, engineer
from .glossary import FIELDS, TEST_PLAIN, status, value_text
from .schemas import PatientInput

DOMAINS = {
    "Chest pain and symptoms": ["cp_asymptomatic", "cp_atypical angina", "exang"],
    "Exercise stress test": ["thalch", "hr_reserve", "rate_pressure_product", "oldpeak", "oldpeak_slope_risk"],
    "Cholesterol": ["chol", "chol_age_ratio"],
    "Overall risk profile": ["cardiac_risk_score", "age"],
}
FEATURE_PLAIN = {
    "cp_asymptomatic": "Chest-pain pattern (silent presentation)", "exang": "Chest pain during exercise",
    "oldpeak_slope_risk": "ECG stress dip combined with its shape", "cardiac_risk_score": "Count of classic risk factors",
    "thalch": "Peak heart rate in the stress test", "cp_atypical angina": "Atypical chest-pain pattern",
    "oldpeak": "ST depression on the stress ECG", "chol": "Cholesterol", "hr_reserve": "Heart-rate reserve",
    "rate_pressure_product": "Heart workload at peak (BP x heart rate)", "age": "Age", "chol_age_ratio": "Cholesterol relative to age",
}
TIERS = {
    "lower": dict(label="Lower-risk pattern", color="#2f855a", icon="●"),
    "mid": dict(label="Intermediate pattern", color="#b7791f", icon="●"),
    "higher": dict(label="Higher-risk pattern", color="#c53030", icon="●"),
}
CURVES = {"chol": range(120, 361, 10), "trestbps": range(90, 191, 5), "thalch": range(80, 201, 5)}
MODIFIABLE = ("trestbps", "chol", "fbs")
BOOL_FEATS = {"exang", "cp_asymptomatic", "cp_atypical angina"}


def tier_for(p: float, conformal: list[str]) -> str:
    if len(conformal) == 2 or 0.30 <= p < 0.60:
        return "mid"
    return "higher" if p >= 0.60 else "lower"


def domain_impacts(shap_all: dict) -> list[dict]:
    rows = [{"domain": d, "impact": sum(shap_all[f] for f in fs)} for d, fs in DOMAINS.items()]
    return sorted(rows, key=lambda r: -abs(r["impact"]))


def drivers(r: dict, n: int = 4) -> list[dict]:
    out = []
    for f, v in sorted(r["shap_all"].items(), key=lambda kv: -abs(kv[1]))[:n]:
        val = r["features"][f]
        if f == "cardiac_risk_score":
            shown = f"{int(val)} of 8"
        elif f in BOOL_FEATS:
            shown = "yes" if val == 1 else "no"
        else:
            shown = f"{val:.3g}"
        out.append({"feature": f, "label": FEATURE_PLAIN[f], "value": shown, "shap": v,
                    "direction": "raises" if v > 0 else "lowers"})
    return out


def measurement_cards(raw_in: dict) -> list[dict]:
    cards = []
    for f in ("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal"):
        lvl, note = status(f, raw_in)
        cards.append({"field": f, "label": FIELDS[f]["label"], "value": value_text(f, raw_in.get(f)), "level": lvl, "note": note,
                      "normal": FIELDS[f]["normal"], "what": FIELDS[f]["what"], "matters": FIELDS[f]["matters"]})
    return cards


def what_if_curves(e: Engine, raw: dict) -> dict:
    out = {}
    for f, xs in CURVES.items():
        rows = [{**raw, f: x} for x in xs]
        out[f] = {"x": list(xs), "p": [float(v) for v in e.p_xgb(engineer(pd.DataFrame(rows)))], "current": raw[f]}
    return out


def doctor_questions(r: dict, cards: list[dict]) -> list[str]:
    q = []
    flagged = {c["field"]: c for c in cards if c["level"] == "flag"}
    if "chol" in flagged:
        q.append("My cholesterol is high. What target should I aim for, given my other risk factors?")
    if "trestbps" in flagged:
        q.append("My resting blood pressure is high. How should it be monitored or managed?")
    if "fbs" in flagged:
        q.append("Should I be tested for diabetes or pre-diabetes?")
    if "thalch" in flagged or "oldpeak" in flagged or "exang" in flagged:
        q.append("My stress-test results look unusual. Should I be seen by a cardiologist?")
    for rk in r["advice"].get("ranking", []):
        if rk["can_flip_verdict"]:
            q.append(f"Would a {TEST_PLAIN[rk['test']].split(' (')[0].lower()} change how my risk is assessed?")
    if r["rules"]["agrees_with_ml"] is False:
        q.append("Different parts of this tool disagreed about my risk. Can you review my full picture?")
    q.append("Given my results, how often should my heart health be checked?")
    return q[:5]


def next_steps(r: dict, tier: str) -> dict:
    steps = []
    if tier == "higher":
        steps.append("Book an appointment with a doctor or cardiologist soon and bring these results.")
    elif tier == "mid":
        steps.append("Discuss these results with a doctor; the picture is not clear-cut and a clinician can weigh it properly.")
    else:
        steps.append("Keep up routine check-ups. Low estimates are not a guarantee, especially if symptoms change.")
    tests = []
    for rk in r["advice"].get("ranking", []):
        tests.append({"key": rk["test"], "name": TEST_PLAIN[rk["test"]], "cost": rk["cost"], "can_flip": rk["can_flip_verdict"],
                      "range": (rk["p_min"], rk["p_max"])})
    return {"steps": steps, "tests": tests, "plan": r["advice"].get("plan", {})}


def headline(r: dict, tier: str, domains: list[dict]) -> str:
    p = r["models"]["xgb"]
    top = domains[0]
    word = {"lower": "lower", "mid": "intermediate", "higher": "higher"}[tier]
    s = (f"Your answers resemble patterns seen in people with {word} likelihood of heart disease "
         f"(model estimate about {p:.0%}). The biggest influence was your {top['domain'].lower()}, which "
         f"{'raises' if top['impact'] > 0 else 'lowers'} the estimate.")
    if len(r["confidence"]["conformal_set"]) == 2:
        s += " The model could not clearly separate the two outcomes for answers like yours, so this is genuinely uncertain."
    if r["imputed"]:
        s += f" {len(r['imputed'])} optional test(s) were not available, so treat this as a preliminary view."
    return s


def build_analysis(e: Engine, r: dict) -> dict:
    t0 = time.perf_counter()
    raw_filled = {**r["inputs"], **{k: e.mode[k] for k in r["imputed"]}}
    tier = tier_for(r["models"]["xgb"], r["confidence"]["conformal_set"])
    domains = domain_impacts(r["shap_all"])
    cards = measurement_cards(r["inputs"])
    curves = what_if_curves(e, raw_filled)
    steps = next_steps(r, tier)
    a = {
        "tier": tier, "tier_info": TIERS[tier], "p": r["models"]["xgb"], "domains": domains, "drivers": drivers(r),
        "cards": cards, "curves": curves, "next": steps, "questions": doctor_questions(r, cards),
        "headline": headline(r, tier, domains),
        "completeness": {"given": 13 - len(r["imputed"]), "of": 13},
        "modifiable": [c for c in cards if c["field"] in MODIFIABLE and c["level"] in ("flag", "watch")],
        "second_opinions": {
            "ml": {"name": "Machine-learning model", "p": r["models"]["xgb"], "says": "higher" if r["models"]["xgb"] >= 0.5 else "lower"},
            "dl": {"name": "Deep-learning model", "p": r["models"]["dnn"], "says": "higher" if r["models"]["dnn"] >= 0.5 else "lower"},
            "rules": {"name": "Expert rules", "verdict": r["rules"]["verdict"], "fired": r["rules"]["fired"]},
        },
    }
    r["trace"].append({"agent": "Synthesizer", "task": "Task 9 integration", "ms": (time.perf_counter() - t0) * 1000,
                       "note": "combined all findings into the patient overview"})
    return a


def assess(patient: PatientInput, engine: Engine | None = None) -> dict:
    e = engine or get_engine()
    r = e.predict(patient)
    return {"result": r, "analysis": build_analysis(e, r), "patient": patient.model_dump()}
