"""Train the stage-1 (home checkup) model, calibrate its trust bands, store training ranges, and evaluate it.
Run after train.py:  python train_stage1.py  ->  artifacts/stage1_*.json, artifacts/ranges.json, reports/stage1_evaluation.json
Uses the SAME split as train.py (data/Task4_clean_dataset.csv, column `split`)."""
from __future__ import annotations

import json
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from cdss import ARTIFACTS, DATA, REPORTS
from cdss.stage1 import HIGH, LOW, features

SEED, ALPHA = 42, 0.10
full = pd.read_csv(DATA / "Task4_clean_dataset.csv")
for c in ("fbs", "exang"):
    full[c] = full[c].astype(bool)
tr, va, te = (full[full.split == s] for s in ("train", "val", "test"))
X = {k: features(d) for k, d in (("tr", tr), ("va", va), ("te", te))}

# ---- training ranges (observed values only, train split) -> used by both models for out-of-range warnings
ranges = {}
for f in ("age", "trestbps", "chol", "thalch", "oldpeak"):
    obs = tr[f][tr[f"{f}_missing"] == 0] if f"{f}_missing" in tr else tr[f]
    ranges[f] = {"p01": float(np.percentile(obs, 1)), "p99": float(np.percentile(obs, 99)), "min": float(obs.min()), "max": float(obs.max())}
(ARTIFACTS / "ranges.json").write_text(json.dumps(ranges, indent=2))

# ---- model selection by 5-fold CV AUC on train only
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
best, bp = -1, None
for d in (2, 3):
    for lr in (0.03, 0.05, 0.1):
        for n in (100, 200, 300):
            p = dict(n_estimators=n, max_depth=d, learning_rate=lr, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_lambda=2.0,
                     random_state=SEED, eval_metric="logloss")
            s = cross_val_score(xgb.XGBClassifier(**p), X["tr"], tr.target, cv=cv, scoring="roc_auc").mean()
            if s > best:
                best, bp = s, p
model = xgb.XGBClassifier(**bp).fit(X["tr"], tr.target)
model.save_model(str(ARTIFACTS / "stage1_xgb.json"))

# ---- calibration on val
pv = model.predict_proba(X["va"])[:, 1]
conf = np.maximum(pv, 1 - pv)
correct = (pv >= 0.5).astype(int) == va.target.values
edges = [0.5, 0.65, 0.75, 0.85, 0.95, 1.0001]
bands = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (conf >= lo) & (conf < hi)
    bands.append({"lo": lo, "hi": hi, "n": int(m.sum()), "accuracy": float(correct[m].mean()) if m.any() else None})
score = np.where(va.target.values == 1, 1 - pv, pv)
n = len(score)
q = float(np.quantile(score, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n), method="higher"))
meta = {"params": {k: v for k, v in bp.items()}, "bands": bands, "conformal": {"alpha": ALPHA, "q": q, "n_cal": n}, "cv_auc": best}
(ARTIFACTS / "stage1_meta.json").write_text(json.dumps(meta, indent=2, default=float))

# ---- test evaluation + base-rate analysis
pt, yt = model.predict_proba(X["te"])[:, 1], te.target.values


def sens_spec(p, y, thr):
    yh = p >= thr
    return float((yh & (y == 1)).sum() / (y == 1).sum()), float((~yh & (y == 0)).sum() / (y == 0).sum())


def ppv(sens, spec, prev):
    return sens * prev / (sens * prev + (1 - spec) * (1 - prev))


ev = {"test_n": int(len(yt)), "auc": float(roc_auc_score(yt, pt)), "accuracy_at_0.5": float(accuracy_score(yt, pt >= 0.5)),
      "brier": float(brier_score_loss(yt, pt)), "cv_auc_train": best, "flag_thresholds": {"check_up": LOW, "see_clinician": HIGH}}
for name, thr in (("check_up_or_higher", LOW), ("see_clinician", HIGH), ("0.5", 0.5)):
    se, sp = sens_spec(pt, yt, thr)
    ev[f"flag_{name}"] = {"threshold": thr, "sensitivity": se, "specificity": sp,
                          "ppv_by_prevalence": {f"{pv_:.0%}": float(ppv(se, sp, pv_)) for pv_ in (0.55, 0.20, 0.10, 0.05, 0.02)}}
# comparison against the full (stress-test) model on the same test rows
from cdss.engine import get_engine
from cdss.features import engineer
e = get_engine()
ev["full_model_auc"] = float(roc_auc_score(yt, e.p_xgb(engineer(te))))
tiers = np.where(pt < LOW, "lower", np.where(pt < HIGH, "mid", "higher"))
ev["tier_outcomes"] = {t: {"n": int((tiers == t).sum()), "disease_rate": float(yt[tiers == t].mean()) if (tiers == t).any() else None} for t in ("lower", "mid", "higher")}
(REPORTS / "stage1_evaluation.json").write_text(json.dumps(ev, indent=2, default=float))
print(json.dumps({k: ev[k] for k in ("auc", "full_model_auc", "tier_outcomes")}, indent=1, default=lambda o: round(float(o), 3)))
print("flag >=0.30:", {k: (round(v, 3) if not isinstance(v, dict) else {a: round(b, 2) for a, b in v.items()}) for k, v in ev["flag_check_up_or_higher"].items()})
print("ranges:", {k: (round(v["p01"], 1), round(v["p99"], 1)) for k, v in ranges.items()})
