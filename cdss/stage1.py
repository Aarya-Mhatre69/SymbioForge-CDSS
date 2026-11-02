"""Stage 1 ("home checkup") model: uses only what a person or a home-visit technician can supply
(history, blood pressure, blood tests, resting ECG). No stress test or imaging. Output is a TIER, never a percentage:
the training data are patients referred for cardiac work-up (55% disease), so a probability would be wrong for the general public."""
from __future__ import annotations

import json
import time
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import BaseModel, Field

from . import ARTIFACTS, DISCLAIMER, plausibility
from .glossary import FIELDS, status, value_text
from .schemas import CP, Sex, RestECG

S1_FEATURES = ["age", "male", "trestbps", "chol", "fbs", "cp_typical angina", "cp_atypical angina", "cp_non-anginal",
               "cp_asymptomatic", "ecg_normal", "ecg_lv hypertrophy", "ecg_st-t abnormality"]
S1_DOMAINS = {
    "Chest pain and symptoms": ["cp_typical angina", "cp_atypical angina", "cp_non-anginal", "cp_asymptomatic"],
    "Blood pressure and sugar": ["trestbps", "fbs"],
    "Cholesterol": ["chol"],
    "Resting ECG": ["ecg_normal", "ecg_lv hypertrophy", "ecg_st-t abnormality"],
    "Age and sex": ["age", "male"],
}
S1_PLAIN = {"age": "Age", "male": "Sex", "trestbps": "Resting blood pressure", "chol": "Cholesterol", "fbs": "High fasting blood sugar",
            "cp_typical angina": "Classic angina pattern", "cp_atypical angina": "Atypical chest-pain pattern",
            "cp_non-anginal": "Non-anginal chest pain", "cp_asymptomatic": "No chest pain (silent presentation)",
            "ecg_normal": "Normal resting ECG", "ecg_lv hypertrophy": "ECG: thickened heart wall", "ecg_st-t abnormality": "ECG: strain pattern"}
TIERS = {
    "lower": dict(label="Lower-risk pattern", color="#2f855a", icon="●"),
    "mid": dict(label="Worth a check-up", color="#b7791f", icon="●"),
    "higher": dict(label="See a clinician soon", color="#c53030", icon="●"),
    "unsure": dict(label="Cannot assess reliably", color="#667085", icon="●"),
}
LOW, HIGH = 0.30, 0.60


class Stage1Input(BaseModel):
    age: int = Field(ge=18, le=100)
    sex: Sex
    cp: CP
    trestbps: int = Field(ge=80, le=220)
    chol: int = Field(ge=100, le=600)
    fbs: bool
    restecg: RestECG


def features(d: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame({"age": d["age"], "male": (d["sex"] == "male").astype(int), "trestbps": d["trestbps"], "chol": d["chol"],
                      "fbs": d["fbs"].astype(bool).astype(int)}, index=d.index)
    for c in ("typical angina", "atypical angina", "non-anginal", "asymptomatic"):
        X["cp_" + c] = (d["cp"] == c).astype(int)
    for c in ("normal", "lv hypertrophy", "st-t abnormality"):
        X["ecg_" + c] = (d["restecg"] == c).astype(int)
    return X[S1_FEATURES].astype(float)


def tier_for(p: float, conformal: list[str]) -> str:
    if len(conformal) == 2 or LOW <= p < HIGH:
        return "mid"
    return "higher" if p >= HIGH else "lower"


class Stage1:
    def __init__(self):
        self.meta = json.loads((ARTIFACTS / "stage1_meta.json").read_text())
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(ARTIFACTS / "stage1_xgb.json"))

    def p(self, d: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(features(d))[:, 1]

    def assess(self, inp: Stage1Input) -> dict:
        trace: list[dict] = []

        def step(agent, task, fn, note):
            t = time.perf_counter()
            res = fn()
            trace.append({"agent": agent, "task": task, "ms": (time.perf_counter() - t) * 1000, "note": note(res)})
            return res

        raw = inp.model_dump()
        X = step("Feature agent", "Task 5 feature engineering", lambda: features(pd.DataFrame([raw])), lambda r: "built the 12 stage-1 features")
        p = step("Stage-1 risk model", "Task 6 machine learning", lambda: float(self.model.predict_proba(X)[0, 1]), lambda r: "tier computed (no percentage shown)")
        contrib = step("Explainer agent (SHAP)", "Task 8 explainability",
                       lambda: self.model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0], lambda r: "exact per-feature contributions")
        shap = {k: float(v) for k, v in zip(S1_FEATURES, contrib[:-1])}
        q = self.meta["conformal"]["q"]
        cset = [lab for lab, pc in (("absent", 1 - p), ("present", p)) if 1 - pc <= q] or ["absent", "present"]
        conf_row = next(b for b in self.meta["bands"] if b["lo"] <= max(p, 1 - p) < b["hi"])
        acc = conf_row["accuracy"]
        band = "high" if acc is not None and acc >= 0.90 else "moderate" if acc is not None and acc >= 0.75 else "low"
        pl = plausibility.check(raw)
        conf = {"band": band, "warnings": [], "validation_accuracy_in_band": acc, "validation_n_in_band": conf_row["n"], "conformal_set": cset,
                "conformal_level": 1 - self.meta["conformal"]["alpha"]}
        conf = step("Trust agent", "Task 8 trust score", lambda: plausibility.apply(conf, pl), lambda r: f"confidence {r['band']}")
        reliable = plausibility.reliable(pl)
        tier = tier_for(p, cset) if reliable else "unsure"
        policy = None
        if tier == "lower" and raw["cp"] == "typical angina":
            tier, policy = "mid", "Classic angina symptoms always warrant a proper check-up, whatever the other numbers show."
        elif tier == "higher" and raw["cp"] == "asymptomatic":
            tier, policy = "mid", ("The model reads a pain-free presentation as a risk sign because it learned from patients who were referred for testing; "
                                   "that does not hold for the general public, so this result is capped at 'worth a check-up'.")
        domains = sorted(({"domain": d, "impact": sum(shap[f] for f in fs)} for d, fs in S1_DOMAINS.items()), key=lambda r: -abs(r["impact"]))
        drivers = []
        for f, v in sorted(shap.items(), key=lambda kv: -abs(kv[1]))[:4]:
            val = X.iloc[0][f]
            shown = ("yes" if val == 1 else "no") if f.startswith(("cp_", "ecg_", "fbs", "male")) else f"{val:.3g}"
            drivers.append({"feature": f, "label": S1_PLAIN[f], "value": shown, "shap": v, "direction": "raises" if v > 0 else "lowers"})
        cards = []
        for f in ("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg"):
            lvl, note = status(f, raw)
            cards.append({"field": f, "label": FIELDS[f]["label"], "value": value_text(f, raw[f]), "level": lvl, "note": note, "normal": FIELDS[f]["normal"],
                          "what": FIELDS[f]["what"], "matters": FIELDS[f]["matters"]})
        analysis = self._narrative(tier, domains, cards, conf, reliable)
        if policy:
            analysis["headline"] += " " + policy
            analysis["policy"] = policy
        analysis.update(tier=tier, tier_info=TIERS[tier], domains=domains, drivers=drivers, cards=cards, p=p,
                        completeness={"given": 7, "of": 7}, reliable=reliable)
        trace.append({"agent": "Synthesizer", "task": "Task 9 integration", "ms": 0.0, "note": "combined the findings into the stage-1 overview"})
        return {"stage": 1, "result": {"inputs": raw, "confidence": conf, "plausibility": pl, "trace": trace, "disclaimer": DISCLAIMER,
                                       "shap_all": shap, "models": {"stage1": p}}, "analysis": analysis, "patient": raw}

    @staticmethod
    def _narrative(tier, domains, cards, conf, reliable) -> dict:
        flagged = {c["field"]: c for c in cards if c["level"] == "flag"}
        top = domains[0]
        if not reliable:
            head = ("Some of your values are outside what this tool can judge reliably, so it is not giving a risk tier. "
                    "The measurement cards below show how each value compares with common reference ranges; please discuss them with a clinician.")
        else:
            head = {"lower": "Your answers do not stand out compared with patients who had heart-disease work-ups.",
                    "mid": "Some of your answers resemble patients who were later found to have heart disease, so a proper check-up is worthwhile.",
                    "higher": "Your answers strongly resemble patients who were later found to have heart disease. Please see a clinician soon."}[tier]
            pick = max(domains, key=lambda d: d["impact"]) if tier in ("mid", "higher") else min(domains, key=lambda d: d["impact"])
            head += (f" The factor that counted most {'toward' if tier in ('mid', 'higher') else 'against'} this was your {pick['domain'].lower()}. "
                     "This is a screening flag, not a finding: most people flagged by a home-level check turn out to be fine, and the next step is a clinic check.")
        steps = []
        if tier in ("mid", "higher") or not reliable:
            steps.append("Book a clinic visit so a clinician can review these results and decide on further tests.")
            steps.append("An exercise stress test would add the most clarity, because it shows how the heart behaves under strain.")
        else:
            steps.append("Keep up routine check-ups; low results are not a guarantee, especially if symptoms change.")
        qs = []
        if "chol" in flagged:
            qs.append("My cholesterol is high. What target should I aim for, given my other risk factors?")
        if "trestbps" in flagged:
            qs.append("My resting blood pressure is high. How should it be monitored or managed?")
        if "fbs" in flagged:
            qs.append("Should I be tested for diabetes or pre-diabetes?")
        qs.append("Should I have an exercise stress test, and how soon?")
        qs.append("Given my results, how often should my heart health be checked?")
        tests = [{"key": "stress", "name": "Exercise stress test (peak heart rate and ECG changes under strain)", "can_flip": tier != "lower"}]
        return {"headline": head, "questions": qs[:5], "next": {"steps": steps, "tests": tests, "plan": {"tests": []}}}


@lru_cache(maxsize=1)
def get_stage1() -> Stage1:
    return Stage1()
