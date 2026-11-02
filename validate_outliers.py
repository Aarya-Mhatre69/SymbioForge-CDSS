"""Outlier and edge-case validation (run after train.py and train_stage1.py): python validate_outliers.py
1. Out-of-fold accuracy on outlier rows vs normal rows (both models).
2. Sweeps of every numeric input across its valid range: smoothness and clinical direction.
3. Extreme corner combinations: finite outputs, and how often the plausibility guard withholds the estimate.
Writes reports/outlier_validation.json and reports/outlier_validation.md."""
from __future__ import annotations

import itertools
import json
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from cdss import ARTIFACTS, DATA, REPORTS, plausibility
from cdss.engine import DEMO_PATIENTS, get_engine
from cdss.features import engineer
from cdss.schemas import PatientInput
from cdss.stage1 import Stage1Input, features as s1_features, get_stage1

SEED = 42
full = pd.read_csv(DATA / "Task4_clean_dataset.csv")
for c in ("fbs", "exang"):
    full[c] = full[c].astype(bool)
y = full.target.values
meta = json.loads((ARTIFACTS / "meta.json").read_text())
meta1 = json.loads((ARTIFACTS / "stage1_meta.json").read_text())
e, s1 = get_engine(), get_stage1()
NUM = ["age", "trestbps", "chol", "thalch", "oldpeak"]
BOUNDS = {"age": (18, 100), "trestbps": (80, 220), "chol": (100, 600), "thalch": (60, 220), "oldpeak": (-3.0, 7.0)}
R = {}

# ------------------------------------------------------------------ 1. OOF outlier analysis
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
p_full = {k: v for k, v in meta["xgb_params"].items()}
p_full["eval_metric"] = "logloss"
oof_full = cross_val_predict(xgb.XGBClassifier(**p_full), engineer(full), y, cv=cv, method="predict_proba")[:, 1]
p1 = {k: v for k, v in meta1["params"].items()}
oof_s1 = cross_val_predict(xgb.XGBClassifier(**p1), s1_features(full), y, cv=cv, method="predict_proba")[:, 1]

observed = {f: full[f][full[f"{f}_missing"] == 0] if f"{f}_missing" in full else full[f] for f in NUM}
iqr_out = pd.DataFrame(index=full.index)
for f in NUM:
    q1, q3 = observed[f].quantile([.25, .75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    flag = (full[f] < lo) | (full[f] > hi)
    if f"{f}_missing" in full:
        flag &= full[f"{f}_missing"] == 0  # imputed values are not real outliers
    iqr_out[f] = flag
any_out = iqr_out.any(axis=1).values


def grp(mask, p):
    yy, pp = y[mask], p[mask]
    return {"n": int(mask.sum()), "accuracy": float(accuracy_score(yy, pp >= .5)), "auc": float(roc_auc_score(yy, pp)) if len(set(yy)) > 1 else None,
            "prevalence": float(yy.mean())}


R["oof"] = {}
for name, p in (("full_model", oof_full), ("stage1_model", oof_s1)):
    R["oof"][name] = {"overall": grp(np.ones(len(y), bool), p), "any_outlier": grp(any_out, p), "no_outlier": grp(~any_out, p),
                      "per_field": {f: grp(iqr_out[f].values, p) for f in NUM if iqr_out[f].sum() >= 15}}
R["outlier_counts"] = {f: int(iqr_out[f].sum()) for f in NUM}

# ------------------------------------------------------------------ 2. sweeps
anchors = {k: dict(v) for k, v in DEMO_PATIENTS.items()}
EXPECT = {"age": +1, "trestbps": +1, "chol": +1, "oldpeak": +1, "thalch": -1}  # expected direction of risk
sweeps = []
for aname, a in anchors.items():
    base = {k: v for k, v in a.items()}
    for f in NUM:
        lo, hi = BOUNDS[f]
        grid = np.linspace(lo, hi, 40)
        if f != "oldpeak":
            grid = grid.round().astype(int)
        for model in ("full", "stage1"):
            if model == "stage1" and f in ("thalch", "oldpeak"):
                continue
            rows = []
            for v in grid:
                d = {**base, f: (float(v) if f == "oldpeak" else int(v))}
                if model == "full":
                    rows.append({**d, **{k: (e.mode[k] if d.get(k) is None else d[k]) for k in ("slope", "ca", "thal")}})
                else:
                    rows.append(d)
            df = pd.DataFrame(rows)
            for c in ("fbs", "exang"):
                df[c] = df[c].astype(bool)
            p = e.p_xgb(engineer(df)) if model == "full" else s1.model.predict_proba(s1_features(df))[:, 1]
            rho = spearmanr(grid, p).correlation
            sweeps.append({"anchor": aname, "field": f, "model": model, "swing": float(p.max() - p.min()), "max_jump": float(np.abs(np.diff(p)).max()),
                           "spearman": None if np.isnan(rho) else float(rho), "against_expected": bool(not np.isnan(rho) and rho * EXPECT[f] < -0.3)})
R["sweeps"] = {
    "n": len(sweeps),
    "jumps_over_0.25": [s for s in sweeps if s["max_jump"] > 0.25],
    "against_clinical_direction": [s for s in sweeps if s["against_expected"]],
    "worst_jump": max(s["max_jump"] for s in sweeps),
}

# ------------------------------------------------------------------ 3. corners
corners, finite, withheld, hard = 0, 0, 0, 0
for aname, a in anchors.items():
    for combo in itertools.product(*[BOUNDS[f] for f in NUM]):
        d = {**a, **dict(zip(NUM, combo))}
        d["age"], d["trestbps"], d["chol"], d["thalch"] = int(d["age"]), int(d["trestbps"]), int(d["chol"]), int(d["thalch"])
        out = e.predict(PatientInput(**d), explain=False)
        corners += 1
        finite += int(np.isfinite(out["models"]["xgb"]) and np.isfinite(out["models"]["dnn"]))
        withheld += int(not out["estimate_reliable"])
        hard += int(bool(out["plausibility"]["hard"]))
R["corners"] = {"tested": corners, "finite_outputs": finite, "estimate_withheld": withheld, "with_hard_implausibility": hard}

# examples of the guard in action
R["guard_examples"] = [
    {"input": "age 90, peak HR 200", "result": plausibility.check({"age": 90, "thalch": 200})},
    {"input": "cholesterol 520", "result": plausibility.check({"chol": 520})},
    {"input": "age 25 (below the training minimum of 28)", "result": plausibility.check({"age": 25})},
    {"input": "typical values: age 52, BP 130, chol 210", "result": plausibility.check({"age": 52, "trestbps": 130, "chol": 210})},
]

(REPORTS / "outlier_validation.json").write_text(json.dumps(R, indent=2, default=float))

# ------------------------------------------------------------------ markdown
f3 = lambda x: "n/a" if x is None else f"{x:.3f}"
md = ["# Outlier and Edge-Case Validation", "", "Generated by `validate_outliers.py`. Purpose: check where the models can be trusted before any claim is made.", "",
      "## 1. Accuracy on outlier rows (5-fold out-of-fold predictions on all rows)",
      f"Outliers are values beyond 1.5 x IQR of observed (not imputed) values. Counts: {R['outlier_counts']}.", "",
      "| Model | Group | n | Accuracy | AUC | Prevalence |", "|---|---|---|---|---|---|"]
for m, d in R["oof"].items():
    for g in ("overall", "no_outlier", "any_outlier"):
        v = d[g]
        md.append(f"| {m} | {g} | {v['n']} | {f3(v['accuracy'])} | {f3(v['auc'])} | {v['prevalence']:.2f} |")
    for f, v in d["per_field"].items():
        md.append(f"| {m} | outlier in {f} | {v['n']} | {f3(v['accuracy'])} | {f3(v['auc'])} | {v['prevalence']:.2f} |")
md += ["", "## 2. Sweeps across valid ranges",
       f"{R['sweeps']['n']} sweeps (4 anchor patients x numeric fields x models). Worst single-step jump in estimated probability: {R['sweeps']['worst_jump']:.2f}.",
       f"Sweeps with a step jump above 0.25: {len(R['sweeps']['jumps_over_0.25'])}. Sweeps moving against the expected clinical direction (Spearman < -0.3 for risk-increasing fields, > 0.3 for peak heart rate): {len(R['sweeps']['against_clinical_direction'])}.", ""]
if R["sweeps"]["against_clinical_direction"]:
    md += ["| Anchor | Field | Model | Spearman |", "|---|---|---|---|"]
    md += [f"| {s['anchor']} | {s['field']} | {s['model']} | {s['spearman']:.2f} |" for s in R["sweeps"]["against_clinical_direction"]]
md += ["", "## 3. Extreme corner combinations",
       f"{corners} combinations of minimum and maximum valid values for age, blood pressure, cholesterol, peak heart rate and ST depression across 4 anchor patients: "
       f"{finite} finite outputs; the plausibility guard withheld the estimate in {withheld} ({hard} had a hard physiological flag).", "",
       "## 4. Guard examples"]
md += [f"- {g['input']}: hard={g['result']['hard']}, beyond training range={g['result']['beyond']}, edge of range={g['result']['outside']}" for g in R["guard_examples"]]
(REPORTS / "outlier_validation.md").write_text("\n".join(md), encoding="utf-8")
print(json.dumps({"outlier_counts": R["outlier_counts"],
                  "full_any_outlier": R["oof"]["full_model"]["any_outlier"], "full_no_outlier": R["oof"]["full_model"]["no_outlier"],
                  "s1_any_outlier": R["oof"]["stage1_model"]["any_outlier"], "s1_no_outlier": R["oof"]["stage1_model"]["no_outlier"],
                  "sweeps": {k: (v if not isinstance(v, list) else len(v)) for k, v in R["sweeps"].items()}, "corners": R["corners"]}, indent=1, default=lambda o: round(float(o), 3)))
