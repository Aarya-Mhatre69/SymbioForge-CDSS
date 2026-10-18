"""Task 10 evaluation + Task 6-8 figures. Run: python evaluate.py  ->  reports/evaluation.json, reports/figures/*.png"""
from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from cdss import REPORTS, rules
from cdss.engine import DEMO_PATIENTS, get_engine
from cdss.features import FEATURES, engineer
from cdss.schemas import PatientInput
from pydantic import ValidationError

FIG = REPORTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
E = get_engine()
full = pd.read_csv("data/Task4_clean_dataset.csv")
for c in ("fbs", "exang"):
    full[c] = full[c].astype(bool)
X = engineer(full)
test = full[full.split == "test"].copy()
Xt, yt = X.loc[test.index], test["target"].values
px, pdn = E.p_xgb(Xt), E.p_dnn(Xt)
pens = (px + pdn) / 2
rng = np.random.default_rng(42)


def m(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    return {"n": int(len(y)), "accuracy": accuracy_score(y, yh), "precision": precision_score(y, yh, zero_division=0),
            "recall": recall_score(y, yh, zero_division=0),
            "roc_auc": roc_auc_score(y, p) if len(set(y)) > 1 else None, "brier": brier_score_loss(y, p)}


def boot_auc(y, p, n=1000):
    s = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1:
            s.append(roc_auc_score(y[i], p[i]))
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


R = {"models": {"xgb": m(yt, px), "dnn": m(yt, pdn), "mean_ensemble": m(yt, pens)}}
R["models"]["xgb"]["auc_ci95"] = boot_auc(yt, px)
R["models"]["dnn"]["auc_ci95"] = boot_auc(yt, pdn)

# rule engine on raw (unobserved stay None) vs ML
def rule_state(row):
    d = row.to_dict()
    for f in ("slope", "ca", "thal", "trestbps", "thalch", "oldpeak"):
        if d[f"{f}_missing"] == 1:
            d[f] = None
    return rules.evaluate(d)["verdict"]

test["rule"] = test.apply(rule_state, axis=1)
test["ml"] = np.where(px >= 0.5, "present", "absent")
test["xgb_p"], test["dnn_p"] = px, pdn
det = test[test.rule != "undetermined"]
agree = det[det.rule == det.ml]
dis = det[det.rule != det.ml]
R["rules"] = {
    "coverage_determinate": len(det) / len(test),
    "rule_accuracy_on_determinate": float(((det.rule == "present").astype(int) == det.target).mean()),
    "ml_accuracy_on_determinate": float(((det.ml == "present").astype(int) == det.target).mean()),
    "agreement_rate": len(agree) / max(1, len(det)),
    "ml_accuracy_when_agree": float(((agree.ml == "present").astype(int) == agree.target).mean()) if len(agree) else None,
    "ml_accuracy_when_disagree": float(((dis.ml == "present").astype(int) == dis.target).mean()) if len(dis) else None,
    "n_disagree": int(len(dis)),
}
dm = (test.xgb_p >= .5) != (test.dnn_p >= .5)
R["model_disagreement"] = {"rate": float(dm.mean()), "xgb_acc_when_disagree": float(((test.xgb_p[dm] >= .5).astype(int) == test.target[dm]).mean()) if dm.any() else None,
                           "xgb_acc_when_agree": float(((test.xgb_p[~dm] >= .5).astype(int) == test.target[~dm]).mean())}

# ablation: imputed vs fully observed
n_imp = test[["slope_missing", "ca_missing", "thal_missing"]].sum(axis=1).values
R["ablation_imputation"] = {}
for name, mask in (("0 imputed", n_imp == 0), ("1 imputed", n_imp == 1), ("2+ imputed", n_imp >= 2)):
    if mask.sum() > 10 and len(set(yt[mask])) > 1:
        R["ablation_imputation"][name] = m(yt[mask], px[mask])
# Ablation: neutralise the 3 optional fields for everyone (as if never provided)
Xn = engineer(test.assign(slope=E.mode["slope"], ca=E.mode["ca"], thal=E.mode["thal"]))
R["ablation_imputation"]["all optional fields neutralised"] = m(yt, E.p_xgb(Xn))

# trust: bands, calibration, conformal
frac, mean_p = calibration_curve(yt, px, n_bins=6, strategy="quantile")
R["calibration"] = {"brier_xgb": brier_score_loss(yt, px), "brier_dnn": brier_score_loss(yt, pdn),
                    "brier_baseline_prevalence": brier_score_loss(yt, np.full(len(yt), yt.mean())),
                    "reliability_points": [{"mean_pred": float(a), "frac_pos": float(b)} for a, b in zip(mean_p, frac)]}
q = E.meta["conformal"]["q"]
sets = [[lab for lab, pc in (("absent", 1 - p), ("present", p)) if 1 - pc <= q] for p in px]
covered = [("present" if y else "absent") in s for s, y in zip(sets, yt)]
R["conformal"] = {"target": 1 - E.meta["conformal"]["alpha"], "empirical_coverage": float(np.mean(covered)),
                  "frac_single": float(np.mean([len(s) == 1 for s in sets])), "frac_both": float(np.mean([len(s) == 2 for s in sets])),
                  "frac_empty": float(np.mean([len(s) == 0 for s in sets]))}
conf = np.maximum(px, 1 - px)
R["trust_bands_test"] = [{"band": f"{lo:.2f}-{min(hi,1):.2f}", "n": int(((conf >= lo) & (conf < hi)).sum()),
                          "accuracy": float((((px >= .5).astype(int) == yt)[(conf >= lo) & (conf < hi)]).mean()) if ((conf >= lo) & (conf < hi)).any() else None}
                         for lo, hi in zip([.5, .65, .75, .85, .95], [.65, .75, .85, .95, 1.0001])]

# fairness
fair = {}
for col, groups in (("sex", test.sex), ("age_band", pd.cut(test.age, [0, 45, 55, 65, 120], labels=["<=45", "46-55", "56-65", ">65"])),
                    ("site", test.dataset)):
    fair[col] = {}
    for g in pd.Series(groups).dropna().unique():
        mk = (groups == g).values
        if mk.sum() >= 8:
            fair[col][str(g)] = {**m(yt[mk], px[mk]), "prevalence": float(yt[mk].mean())}
R["fairness"] = fair

# robustness + latency
bad = {"chol": 0, "trestbps": 0, "age": 5, "cp": "x", "oldpeak": 99, "ca": 7, "thalch": 10, "sex": "m"}
rejected = 0
for k, v in bad.items():
    try:
        PatientInput(**{**DEMO_PATIENTS["Clear absent"], k: v})
    except ValidationError:
        rejected += 1
fuzz_ok, base = 0, DEMO_PATIENTS["Clear absent"]
for _ in range(300):
    p = dict(base, age=int(rng.integers(18, 101)), trestbps=int(rng.integers(80, 221)), chol=int(rng.integers(100, 601)),
             thalch=int(rng.integers(60, 221)), oldpeak=float(rng.uniform(-3, 7)), exang=bool(rng.integers(2)), fbs=bool(rng.integers(2)),
             cp=str(rng.choice(["typical angina", "atypical angina", "non-anginal", "asymptomatic"])),
             slope=rng.choice([None, "flat", "upsloping", "downsloping"]), ca=rng.choice([None, 0, 1, 2, 3]),
             thal=rng.choice([None, "normal", "fixed defect", "reversable defect"]))
    p["slope"] = None if p["slope"] is None else str(p["slope"])
    p["ca"] = None if p["ca"] is None else int(p["ca"])
    p["thal"] = None if p["thal"] is None else str(p["thal"])
    r = E.predict(PatientInput(**p), explain=False)
    fuzz_ok += int(0 <= r["models"]["xgb"] <= 1 and 0 <= r["models"]["dnn"] <= 1)
lat = []
for name in list(DEMO_PATIENTS) * 5:
    t = time.perf_counter()
    E.predict(PatientInput(**DEMO_PATIENTS[name]))
    lat.append((time.perf_counter() - t) * 1000)
R["robustness"] = {"invalid_inputs_rejected": f"{rejected}/{len(bad)}", "fuzz_valid_inputs_ok": f"{fuzz_ok}/300"}
R["latency_ms_full_pipeline"] = {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)), "max": float(max(lat))}

# global SHAP (mean |contribution|, test set)
contrib = E.xgb.get_booster().predict(xgb.DMatrix(Xt[FEATURES]), pred_contribs=True)[:, :-1]
gs = pd.Series(np.abs(contrib).mean(0), index=FEATURES).sort_values()
R["global_shap"] = gs.sort_values(ascending=False).to_dict()

# figures
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(5, 4))
for n_, p_ in (("XGBoost", px), ("DNN", pdn), ("Mean ensemble", pens)):
    f, t_, _ = roc_curve(yt, p_)
    ax.plot(f, t_, label=f"{n_} (AUC {roc_auc_score(yt, p_):.3f})")
ax.plot([0, 1], [0, 1], "k:", lw=1)
ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC, held-out test")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / "roc.png", dpi=150); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(7, 3))
for a, (n_, p_) in zip(axs, (("XGBoost", px), ("DNN", pdn))):
    cm = confusion_matrix(yt, (p_ >= .5).astype(int))
    a.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        a.text(j, i, v, ha="center", va="center")
    a.set(title=n_, xticks=[0, 1], yticks=[0, 1], xticklabels=["absent", "present"], yticklabels=["absent", "present"], xlabel="predicted")
fig.tight_layout(); fig.savefig(FIG / "confusion.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.5, 4))
ax.plot(mean_p, frac, "o-", label="XGBoost")
ax.plot([0, 1], [0, 1], "k:", label="perfect")
ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction present", title=f"Reliability (Brier {R['calibration']['brier_xgb']:.3f})")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / "calibration.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 4))
ax.barh([k for k in gs.index], gs.values, color="#2b6cb0")
ax.set(xlabel="mean |SHAP| (log-odds)", title="Global feature importance")
fig.tight_layout(); fig.savefig(FIG / "shap_global.png", dpi=150); plt.close(fig)

h = pd.read_csv(REPORTS / "Task7_training_history.csv")
fig, axs = plt.subplots(1, 2, figsize=(8, 3))
axs[0].plot(h.loss, label="train"); axs[0].plot(h.val_loss, label="val"); axs[0].set(title="Loss", xlabel="epoch"); axs[0].legend(frameon=False)
axs[1].plot(h.accuracy, label="train"); axs[1].plot(h.val_accuracy, label="val"); axs[1].set(title="Accuracy", xlabel="epoch"); axs[1].legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / "dnn_training.png", dpi=150); plt.close(fig)

(REPORTS / "evaluation.json").write_text(json.dumps(R, indent=2, default=float))
print(json.dumps({k: R[k] for k in ("models", "rules", "model_disagreement", "calibration", "conformal", "robustness", "latency_ms_full_pipeline")},
                 indent=1, default=lambda o: round(float(o), 3))[:6000])
print("ablation:", {k: (round(v["accuracy"], 3), round(v["roc_auc"], 3), v["n"]) for k, v in R["ablation_imputation"].items()})
print("fair sex:", {k: (round(v["roc_auc"], 3), round(v["recall"], 3), v["n"]) for k, v in fair["sex"].items()})
print("fair age:", {k: (round(v["roc_auc"] or 0, 3), v["n"]) for k, v in fair["age_band"].items()})
print("fair site:", {k: (round(v["roc_auc"] or 0, 3), v["n"], round(v["prevalence"], 2)) for k, v in fair["site"].items()})
