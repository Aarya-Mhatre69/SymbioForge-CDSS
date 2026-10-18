"""Single source of truth for feature formulas (context.md §6) and training-time cleaning (§5)."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "cp_asymptomatic", "exang", "oldpeak_slope_risk", "cardiac_risk_score", "thalch",
    "cp_atypical angina", "oldpeak", "chol", "hr_reserve", "rate_pressure_product",
    "age", "chol_age_ratio",
]
RAW_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalch", "exang",
            "oldpeak", "slope", "ca", "thal"]
OPTIONAL = ["slope", "ca", "thal"]
NUMERIC = ["trestbps", "chol", "thalch", "oldpeak"]
BOOL_CATS = ["fbs", "restecg", "exang"]
SLOPE_SEVERITY = {"upsloping": 0, "flat": 1, "downsloping": 2}


def engineer(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw (fully imputed, lowercase categories, bool fbs/exang) -> the 12 model features, in order."""
    r = raw
    fbs = r["fbs"].astype(bool)
    exang = r["exang"].astype(bool)
    slope_sev = r["slope"].map(SLOPE_SEVERITY).astype(float)
    risk = (
        (r["age"] > 55).astype(int)
        + (r["sex"] == "male").astype(int)
        + (r["chol"] > 240).astype(int)
        + (r["trestbps"] > 140).astype(int)
        + fbs.astype(int)
        + exang.astype(int)
        + (r["ca"] >= 1).astype(int)
        + (r["thal"] != "normal").astype(int)
    )
    out = pd.DataFrame({
        "cp_asymptomatic": (r["cp"] == "asymptomatic").astype(int),
        "exang": exang.astype(int),
        "oldpeak_slope_risk": r["oldpeak"] * (slope_sev + 1),
        "cardiac_risk_score": risk,
        "thalch": r["thalch"],
        "cp_atypical angina": (r["cp"] == "atypical angina").astype(int),
        "oldpeak": r["oldpeak"],
        "chol": r["chol"],
        "hr_reserve": (220 - r["age"]) - r["thalch"],
        "rate_pressure_product": r["trestbps"] * r["thalch"],
        "age": r["age"],
        "chol_age_ratio": r["chol"] / r["age"],
    }, index=r.index)
    return out[FEATURES].astype(float)


# ---- Task 4 cleaning (training time) --------------------------------------------------

def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Steps 1-5 of context.md §5. Imputation is separate (fit on train only: no leakage)."""
    d = df.copy()
    d = d.replace("?", np.nan)
    for c in ("chol", "trestbps"):
        d.loc[d[c] == 0, c] = np.nan
    d = d.drop(columns=[c for c in ("id",) if c in d.columns])
    for c in ("sex", "cp", "restecg", "slope", "thal"):
        d[c] = d[c].astype("string").str.strip().str.lower().astype(object)
    for c in ("fbs", "exang"):
        d[c] = d[c].map({True: True, False: False, "True": True, "False": False})
    d = d.drop_duplicates().reset_index(drop=True)
    d["target"] = (d["num"] > 0).astype(int)
    return d.drop(columns=["num"])


class Imputer:
    """Step 6. Numeric: site-grouped median (fallback: seeded draw from observed train values). Categorical: mode. slope/ca/thal: flag + mode."""

    def fit(self, d: pd.DataFrame) -> "Imputer":
        self.global_median = {c: float(d[c].median()) for c in NUMERIC}
        self.pool = {c: d[c].dropna().to_numpy() for c in NUMERIC}  # observed train values, for fallback draws
        self.site_median = d.groupby("dataset")[NUMERIC].median().to_dict("index")
        self.mode = {c: d[c].mode().iloc[0] for c in BOOL_CATS + OPTIONAL}
        self.mode["ca"] = float(self.mode["ca"])
        for c in ("fbs", "exang"):
            self.mode[c] = bool(self.mode[c])
        return self

    def transform(self, d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        for c in OPTIONAL:
            d[f"{c}_missing"] = d[c].isna().astype(int)
        rng = np.random.default_rng(42)
        for c in NUMERIC:
            d[f"{c}_missing"] = d[c].isna().astype(int)
            site = d["dataset"].map(lambda s: self.site_median.get(s, {}).get(c, np.nan))
            d[c] = d[c].fillna(site)
            # A site with no observed values (e.g. chol at Switzerland) must not collapse to one constant:
            # a constant becomes a site proxy the model can learn. Draw from the observed train distribution instead.
            gap = d[c].isna()
            d.loc[gap, c] = rng.choice(self.pool[c], gap.sum())
        for c in BOOL_CATS + OPTIONAL:
            d[c] = d[c].fillna(self.mode[c])
        for c in ("fbs", "exang"):
            d[c] = d[c].astype(bool)
        return d

    def to_dict(self) -> dict:
        return {"global_median": self.global_median, "mode": self.mode}
