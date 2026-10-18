"""Rebuild Tasks 4-8 artifacts from data/heart_dataset.csv. Run: python train.py
Split 70/15/15 stratified (seed 42). Imputer, scaler, models fit on TRAIN only; VAL calibrates trust bands and
conformal threshold; TEST is touched only for reporting."""
from __future__ import annotations

import json
import os
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from cdss import ARTIFACTS, DATA, REPORTS
from cdss.features import FEATURES, Imputer, clean_raw, engineer

SEED = 42
ALPHA = 0.10  # conformal miscoverage


def metrics(y, p):
    yh = (p >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y, yh), "precision": precision_score(y, yh), "recall": recall_score(y, yh),
            "f1": f1_score(y, yh), "roc_auc": roc_auc_score(y, p)}


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    ARTIFACTS.mkdir(exist_ok=True)
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)

    # ---- Task 4: clean, split, impute ----
    raw = pd.read_csv(DATA / "heart_dataset.csv")
    clean = clean_raw(raw)
    idx_tr, idx_tmp = train_test_split(clean.index, test_size=0.30, stratify=clean["target"], random_state=SEED)
    idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, stratify=clean.loc[idx_tmp, "target"], random_state=SEED)
    imp = Imputer().fit(clean.loc[idx_tr])
    full = imp.transform(clean)
    full["split"] = "train"
    full.loc[idx_va, "split"] = "val"
    full.loc[idx_te, "split"] = "test"
    full.to_csv(DATA / "Task4_clean_dataset.csv", index=False)

    # ---- Task 5: features ----
    X = engineer(full)
    y = full["target"].values
    feat = X.copy()
    feat["target"] = y
    feat["split"] = full["split"]
    feat.to_csv(DATA / "Task5_Feature_Matrix.csv", index=False)
    tr, va, te = (full.index[full["split"] == s] for s in ("train", "val", "test"))
    Xtr, Xva, Xte = X.loc[tr], X.loc[va], X.loc[te]
    ytr, yva, yte = y[tr], y[va], y[te]
    mi = pd.Series(mutual_info_classif(Xtr, ytr, random_state=SEED), index=FEATURES).sort_values(ascending=False)
    mi.to_csv(REPORTS / "feature_mutual_info.csv", header=["mutual_info"])

    # ---- Task 6: model comparison (5-fold CV on TRAIN, then TEST) ----
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    base_xgb = dict(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=SEED, eval_metric="logloss")
    candidates = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "DecisionTree": DecisionTreeClassifier(max_depth=4, random_state=SEED),
        "RandomForest": RandomForestClassifier(300, random_state=SEED),
        "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED)),
        "GradientBoosting": GradientBoostingClassifier(random_state=SEED),
        "XGBoost(context)": xgb.XGBClassifier(**base_xgb),
    }
    # small grid for XGBoost, selected by CV AUC on train only
    best_cv, best_params = -1, None
    for d in (2, 3, 4):
        for lr in (0.03, 0.05, 0.1):
            for n in (100, 200, 400):
                for cs in (0.7, 1.0):
                    p = dict(base_xgb, max_depth=d, learning_rate=lr, n_estimators=n, subsample=0.8,
                             colsample_bytree=cs, min_child_weight=3, reg_lambda=2.0)
                    s = cross_val_score(xgb.XGBClassifier(**p), Xtr, ytr, cv=cv, scoring="roc_auc").mean()
                    if s > best_cv:
                        best_cv, best_params = s, p
    candidates["XGBoost(tuned)"] = xgb.XGBClassifier(**best_params)
    comp = []
    for name, m in candidates.items():
        cvs = cross_val_score(m, Xtr, ytr, cv=cv, scoring="roc_auc")
        m.fit(Xtr, ytr)
        comp.append({"model": name, "cv_auc_mean": cvs.mean(), "cv_auc_std": cvs.std(),
                     **{f"test_{k}": v for k, v in metrics(yte, m.predict_proba(Xte)[:, 1]).items()}})
    comp = pd.DataFrame(comp).sort_values("cv_auc_mean", ascending=False)
    comp.to_csv(REPORTS / "Task6_model_comparison.csv", index=False)
    pick = comp[comp["model"].str.startswith("XGBoost")].iloc[0]["model"]  # SHAP/trust layer needs a tree model
    xgb_model = candidates[pick]
    xgb_model.save_model(str(ARTIFACTS / "xgb.json"))

    # ---- Task 7: DNN (context.md §8 architecture) in PyTorch (TensorFlow cannot load in this env) ----
    import torch
    from torch import nn
    torch.manual_seed(SEED)
    sc = StandardScaler().fit(Xtr)
    T = lambda a: torch.tensor(np.asarray(a, dtype="float64").tolist(), dtype=torch.float32)  # list route: torch2.3 vs numpy2 interop
    Ttr, Tva, Tte = T(sc.transform(Xtr)), T(sc.transform(Xva)), T(sc.transform(Xte))
    ytr_t, yva_t = T(ytr).unsqueeze(1), T(yva).unsqueeze(1)
    net = nn.Sequential(
        nn.Linear(12, 64), nn.ReLU(), nn.BatchNorm1d(64, eps=1e-3), nn.Dropout(0.3),
        nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32, eps=1e-3), nn.Dropout(0.3),
        nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(16, 1), nn.Sigmoid())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
    lossf = nn.BCELoss()
    best, best_state, wait, hist = 1e9, None, 0, []
    for ep in range(200):
        net.train()
        perm = torch.randperm(len(Ttr))
        for i in range(0, len(perm), 32):
            ix = perm[i:i + 32]
            if len(ix) < 2:
                continue
            opt.zero_grad()
            lossf(net(Ttr[ix]), ytr_t[ix]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            tl = lossf(net(Ttr), ytr_t).item()
            vl = lossf(net(Tva), yva_t).item()
            ta = ((net(Ttr) >= 0.5).float() == ytr_t).float().mean().item()
            va_ = ((net(Tva) >= 0.5).float() == yva_t).float().mean().item()
        hist.append({"epoch": ep + 1, "loss": tl, "val_loss": vl, "accuracy": ta, "val_accuracy": va_})
        sched.step(vl)
        if vl < best - 1e-6:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= 20:
                break
    net.load_state_dict(best_state)
    net.eval()
    torch.save(net.state_dict(), ARTIFACTS / "dnn.pt")
    pd.DataFrame(hist).to_csv(REPORTS / "Task7_training_history.csv", index=False)
    arr = lambda t: t.detach().double().tolist()
    L = []
    for m in net:
        if isinstance(m, nn.Linear):
            L.append({"type": "Dense", "W": arr(m.weight.T), "b": arr(m.bias)})
        elif isinstance(m, nn.BatchNorm1d):
            L.append({"type": "BN", "gamma": arr(m.weight), "beta": arr(m.bias), "mean": arr(m.running_mean),
                      "var": arr(m.running_var), "eps": m.eps})
        elif isinstance(m, nn.ReLU):
            L.append({"type": "ReLU"})
        elif isinstance(m, nn.Sigmoid):
            L.append({"type": "Sigmoid"})
    (ARTIFACTS / "dnn.json").write_text(json.dumps({
        "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(), "layers": L}))
    pte_d = np.array(net(Tte).detach().double().tolist()).ravel()

    # ---- Task 8/trust: bands + conformal on VAL (XGB) ----
    pva = xgb_model.predict_proba(Xva)[:, 1]
    conf = np.maximum(pva, 1 - pva)
    correct = ((pva >= 0.5).astype(int) == yva)
    edges = [0.5, 0.65, 0.75, 0.85, 0.95, 1.0001]
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        bands.append({"lo": lo, "hi": hi, "n": int(m.sum()), "accuracy": float(correct[m].mean()) if m.any() else None})
    score = np.where(yva == 1, 1 - pva, pva)  # LAC nonconformity = 1 - p(true class)
    n = len(score)
    q = float(np.quantile(score, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n), method="higher"))

    pte_x = xgb_model.predict_proba(Xte)[:, 1]
    meta = {
        "primary_model": pick, "xgb_params": {k: v for k, v in xgb_model.get_params().items() if k in best_params},
        "imputer": imp.to_dict(), "bands": bands, "conformal": {"alpha": ALPHA, "q": q, "n_cal": n},
        "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
        "dataset": {"rows_raw": len(raw), "rows_clean": len(clean), "prevalence": float(y.mean())},
        "test_metrics": {"xgb": metrics(yte, pte_x), "dnn": metrics(yte, pte_d)},
        "dnn_epochs": len(hist), "cv_best_auc": best_cv,
    }
    (ARTIFACTS / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
    print(comp.round(3).to_string(index=False))
    print("primary:", pick)
    print("test xgb", {k: round(v, 3) for k, v in meta["test_metrics"]["xgb"].items()})
    print("test dnn", {k: round(v, 3) for k, v in meta["test_metrics"]["dnn"].items()}, "epochs", meta["dnn_epochs"])


if __name__ == "__main__":
    main()
