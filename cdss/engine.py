"""Inference engine: validation -> imputation -> features -> models -> rules -> explanation -> confidence."""
from __future__ import annotations

import itertools
import json
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import xgboost as xgb

from . import ARTIFACTS, DISCLAIMER, advisor, rules
from .features import FEATURES, OPTIONAL, engineer
from .schemas import TRAIN_AGE_RANGE, PatientInput

THRESHOLD = 0.5
LABELS = {
    "cp_asymptomatic": "asymptomatic chest pain", "exang": "exercise-induced angina",
    "oldpeak_slope_risk": "ST depression x slope", "cardiac_risk_score": "cardiac risk score",
    "thalch": "max heart rate", "cp_atypical angina": "atypical angina", "oldpeak": "ST depression",
    "chol": "cholesterol", "hr_reserve": "heart-rate reserve", "rate_pressure_product": "rate-pressure product",
    "age": "age", "chol_age_ratio": "cholesterol / age",
}
MODIFIABLE = {"chol", "trestbps"}
PLAUSIBLE = 0.30  # modifiable factors may move at most +-30% in the actionable search
CF_GRID = {
    "chol": list(range(120, 361, 20)), "trestbps": list(range(100, 181, 10)), "fbs": [False, True],
    "thalch": list(range(100, 191, 10)), "exang": [False, True], "oldpeak": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    "cp": ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
    "restecg": ["normal", "lv hypertrophy", "st-t abnormality"],
}
CF_RANGE = {"chol": 240, "trestbps": 100, "thalch": 120, "oldpeak": 6}


class DNN:
    """Dependency-free forward pass (Dense/ReLU/BatchNorm-inference/Sigmoid) from exported weights."""

    def __init__(self, spec: dict):
        self.mean, self.scale = np.array(spec["scaler_mean"]), np.array(spec["scaler_scale"])
        self.layers = spec["layers"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        h = (X - self.mean) / self.scale
        for l in self.layers:
            t = l["type"]
            if t == "Dense":
                h = h @ np.array(l["W"]) + np.array(l["b"])
            elif t == "ReLU":
                h = np.maximum(h, 0)
            elif t == "BN":
                h = np.array(l["gamma"]) * (h - np.array(l["mean"])) / np.sqrt(np.array(l["var"]) + l["eps"]) + np.array(l["beta"])
            elif t == "Sigmoid":
                h = 1 / (1 + np.exp(-h))
        return h.ravel()


class Engine:
    def __init__(self, artifacts=ARTIFACTS):
        self.meta = json.loads((artifacts / "meta.json").read_text())
        self.xgb = xgb.XGBClassifier()
        self.xgb.load_model(str(artifacts / "xgb.json"))
        self.dnn = DNN(json.loads((artifacts / "dnn.json").read_text()))
        self.mode = self.meta["imputer"]["mode"]

    # -- model helpers --
    def p_xgb(self, F: pd.DataFrame) -> np.ndarray:
        return self.xgb.predict_proba(F[FEATURES])[:, 1]

    def p_dnn(self, F: pd.DataFrame) -> np.ndarray:
        return self.dnn.predict(F[FEATURES].to_numpy(float))

    def impute(self, raw: dict) -> tuple[dict, list[str]]:
        r, imputed = dict(raw), []
        for f in OPTIONAL:
            if r.get(f) is None:
                r[f] = self.mode[f]
                imputed.append(f)
        return r, imputed

    def prob_fn(self, raw: dict) -> float:
        """P(present) for a raw dict (impute remaining optional fields first). Used by the advisor."""
        r, _ = self.impute(raw)
        return float(self.p_xgb(engineer(pd.DataFrame([r])))[0])

    # -- explanation --
    def shap_top(self, F: pd.DataFrame, n=5) -> list[dict]:
        """Exact TreeSHAP via XGBoost's native contributions (log-odds units)."""
        vals, base = self.shap_all(F)
        top = sorted(vals.items(), key=lambda kv: -abs(kv[1]))[:n]
        row = F.iloc[0]
        return [{"feature": k, "label": LABELS[k], "value": float(row[k]), "shap": float(v),
                 "direction": "raises risk" if v > 0 else "lowers risk"} for k, v in top], base

    def shap_all(self, F: pd.DataFrame) -> tuple[dict, float]:
        contrib = self.xgb.get_booster().predict(xgb.DMatrix(F[FEATURES]), pred_contribs=True)[0]
        return {k: float(v) for k, v in zip(FEATURES, contrib[:-1])}, float(contrib[-1])

    def counterfactual(self, raw: dict, imputed: list[str], verdict: str) -> dict:
        """Smallest change set that flips the verdict. 'actionable' = only modifiable fields; 'any' = up to 2 fields."""
        target_present = verdict != "present"
        fields = [f for f in CF_GRID if f not in imputed]

        def search(allowed: set[str], max_k: int, capped: bool = False):
            opts = {f: [v for v in CF_GRID[f] if v != raw[f] and (not capped or abs(v - raw[f]) <= PLAUSIBLE * raw[f])]
                    for f in fields if f in allowed}
            opts = {f: v for f, v in opts.items() if v}
            rows, meta = [], []
            for k in range(1, max_k + 1):
                for combo in itertools.combinations(opts, k):
                    for vals in itertools.product(*(opts[f] for f in combo)):
                        r = dict(raw)
                        r.update(zip(combo, vals))
                        rows.append(r)
                        meta.append(dict(zip(combo, vals)))
            if not rows:
                return None
            p = self.p_xgb(engineer(pd.DataFrame(rows)))
            ok = [i for i, pi in enumerate(p) if (pi >= THRESHOLD) == target_present]
            if not ok:
                return None

            def cost(i):
                ch = meta[i]
                return (len(ch), sum(abs(v - raw[f]) / CF_RANGE[f] if f in CF_RANGE and not isinstance(v, bool) else 1
                                     for f, v in ch.items()))
            best = min(ok, key=cost)
            return {"changes": {f: {"from": raw[f], "to": v} for f, v in meta[best].items()}, "new_probability": float(p[best])}

        act = search(MODIFIABLE, 2, capped=True) if verdict == "present" else None  # risk-lowering only
        anyc = search(set(fields), 2)
        return {"found": bool(act or anyc), "actionable": act, "any": anyc,
                "note": None if (act or anyc) else "robust: no change of up to 2 fields flips the verdict"}

    # -- confidence --
    def confidence(self, p: float, imputed: list, age: int, disagree: bool, rule_disagree: bool) -> dict:
        n_imputed = len(imputed)
        c = max(p, 1 - p)
        band_row = next(b for b in self.meta["bands"] if b["lo"] <= c < b["hi"])
        acc = band_row["accuracy"]  # band label comes from validation accuracy in this confidence range
        band = "high" if acc is not None and acc >= 0.90 else "moderate" if acc is not None and acc >= 0.75 else "low"
        warnings = []
        if band == "low":
            warnings.append("Low confidence: in validation, estimates this uncertain were right only "
                            f"{acc:.0%} of the time. Defer to clinical judgement.")
        if n_imputed:
            warnings.append(f"{n_imputed} of 3 optional fields not provided ({', '.join(imputed)}); filled with the training mode.")
        if n_imputed >= 2 or disagree:
            if band == "high":
                band = "moderate"
        if n_imputed >= 2:
            warnings.append("Reduced reliability: 2+ imputed fields, so confidence cannot be 'high'.")
        if not TRAIN_AGE_RANGE[0] <= age <= TRAIN_AGE_RANGE[1]:
            warnings.append(f"Age {age} is outside the training range {TRAIN_AGE_RANGE[0]}-{TRAIN_AGE_RANGE[1]}.")
        if disagree:
            warnings.append("XGBoost and the DNN disagree on the verdict.")
        if rule_disagree:
            warnings.append("The explicit rule base disagrees with the ML verdict.")
        q = self.meta["conformal"]["q"]
        pset = [lab for lab, pc in (("absent", 1 - p), ("present", p)) if 1 - pc <= q]
        return {"band": band, "probability_confidence": c, "validation_accuracy_in_band": band_row["accuracy"],
                "validation_n_in_band": band_row["n"], "conformal_set": pset or ["absent", "present"],
                "conformal_level": 1 - self.meta["conformal"]["alpha"], "warnings": warnings}

    # -- main --
    def predict(self, patient: PatientInput, explain: bool = True) -> dict:
        trace: list[dict] = []

        def step(agent: str, task: str, fn, note):
            t = time.perf_counter()
            res = fn()
            trace.append({"agent": agent, "task": task, "ms": (time.perf_counter() - t) * 1000, "note": note(res)})
            return res

        raw_in = patient.model_dump()
        rule_in = dict(raw_in)  # unobserved stay None for the rule engine
        raw, imputed = step("Intake agent", "Task 4 data preparation", lambda: self.impute(raw_in),
                            lambda r: f"validated 13 fields; {len(r[1])} optional field(s) missing, filled and flagged")
        F = step("Feature agent", "Task 5 feature engineering", lambda: engineer(pd.DataFrame([raw])),
                 lambda r: "built the 12 model features in fixed order")
        px = step("Risk model (XGBoost)", "Task 6 machine learning", lambda: float(self.p_xgb(F)[0]),
                  lambda r: f"P(disease) = {r:.0%}")
        pd_ = step("Second-opinion model (DNN)", "Task 7 deep learning", lambda: float(self.p_dnn(F)[0]),
                   lambda r: f"P(disease) = {r:.0%}")
        verdict = "present" if px >= THRESHOLD else "absent"
        disagree = (pd_ >= THRESHOLD) != (px >= THRESHOLD)
        rr = step("Clinical rules agent", "Tasks 1-2 knowledge + rule base", lambda: rules.evaluate(rule_in),
                  lambda r: f"verdict {r['verdict']}; fired {', '.join(r['fired']) or 'none'}; unknown {', '.join(r['unknown']) or 'none'}")
        rule_disagree = rr["verdict"] != "undetermined" and rr["verdict"] != verdict
        conf = step("Trust agent", "Task 8 trust score + conformal", lambda: self.confidence(px, imputed, patient.age, disagree, rule_disagree),
                    lambda r: f"confidence {r['band']}; {len(r['warnings'])} warning(s)")
        out = {
            "inputs": raw_in, "imputed": imputed, "features": F.iloc[0].to_dict(),
            "models": {"xgb": px, "dnn": pd_}, "verdict": verdict, "model_disagreement": disagree,
            "rules": {"fired": rr["fired"], "unknown": rr["unknown"], "state": rr["state"], "verdict": rr["verdict"],
                      "agrees_with_ml": None if rr["verdict"] == "undetermined" else not rule_disagree},
            "confidence": conf, "disclaimer": DISCLAIMER, "trace": trace,
        }
        if explain:
            out["shap_all"], _ = step("Explainer agent (SHAP)", "Task 8 explainability", lambda: self.shap_all(F),
                                      lambda r: "exact per-feature contributions")
            out["explanation"], out["base_value"] = self.shap_top(F)
            out["counterfactual"] = step("What-if agent", "Task 8 counterfactuals", lambda: self.counterfactual(raw, imputed, verdict),
                                         lambda r: "found a flip" if r["found"] else "robust, no small change flips it")
            out["advice"] = step("Test planner (A*)", "Task 3 state-space search", lambda: advisor.advise({**rule_in}, patient.missing(), self.prob_fn),
                                 lambda r: f"plan: {', '.join(r['plan']['tests']) or 'no extra tests needed'}")
            out["summary"] = summarize(out)
        return out


def summarize(r: dict) -> str:
    v, p = r["verdict"], r["models"]["xgb"]
    s = [f"Model estimate: {p:.0%} probability of cardiovascular disease ({'above' if v == 'present' else 'below'} the 50% threshold), "
         f"confidence {r['confidence']['band']}."]
    ups = [e["label"] for e in r["explanation"] if e["shap"] > 0][:3]
    dns = [e["label"] for e in r["explanation"] if e["shap"] < 0][:2]
    if ups:
        s.append("Main factors raising risk: " + ", ".join(ups) + ".")
    if dns:
        s.append("Factors lowering risk: " + ", ".join(dns) + ".")
    rl = r["rules"]
    if rl["verdict"] == "undetermined":
        s.append(f"The rule base cannot decide yet (rules {', '.join(rl['unknown'])} need missing tests).")
    else:
        s.append(f"Rule base: {rl['verdict']}" + (f" (fired {', '.join(rl['fired'])})" if rl["fired"] else "") +
                 (", agrees with the model." if rl["agrees_with_ml"] else ", DISAGREES with the model."))
    if r["model_disagreement"]:
        s.append("The two ML models disagree; treat as uncertain.")
    if r["confidence"]["warnings"]:
        s.append("Cautions: " + " ".join(r["confidence"]["warnings"]))
    s.append(r["disclaimer"])
    return " ".join(s)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return Engine()


DEMO_PATIENTS = {
    "Clear present": dict(age=63, sex="male", cp="asymptomatic", trestbps=150, chol=280, fbs=True, restecg="lv hypertrophy",
                          thalch=105, exang=True, oldpeak=3.2, slope="flat", ca=2, thal="reversable defect"),
    "Clear absent": dict(age=38, sex="female", cp="non-anginal", trestbps=115, chol=190, fbs=False, restecg="normal",
                         thalch=175, exang=False, oldpeak=0.0, slope="upsloping", ca=0, thal="normal"),
    "Borderline / missing data": dict(age=52, sex="male", cp="non-anginal", trestbps=140, chol=250, fbs=False,
                                      restecg="normal", thalch=130, exang=True, oldpeak=1.5, slope=None, ca=None, thal=None),
    "Rule vs ML disagreement": dict(age=57, sex="male", cp="atypical angina", trestbps=135, chol=240, fbs=False,
                                    restecg="st-t abnormality", thalch=135, exang=False, oldpeak=1.2, slope=None, ca=None, thal=None),
}
