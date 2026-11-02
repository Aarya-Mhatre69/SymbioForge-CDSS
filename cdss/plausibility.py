"""Input plausibility and out-of-range checks. A model estimate is only claimed when the inputs look physiologically
sensible and sit inside the range the model was trained on; otherwise the system says so and falls back to
guideline status cards.

hard    : physiologically suspicious combinations or values; confidence capped at 'low', estimate withheld
beyond  : outside the min-max of the training data (the model never saw such values); estimate withheld
outside : inside min-max but beyond the 1st-99th percentile; warning, confidence capped at 'moderate',
          estimate withheld only when two or more occur"""
from __future__ import annotations

import json

from . import ARTIFACTS

LABEL = {"age": "age", "trestbps": "resting blood pressure", "chol": "cholesterol", "thalch": "peak heart rate", "oldpeak": "ST depression"}
_RANGES: dict | None = None


def ranges() -> dict:
    """Training ranges. Fails closed: a missing file is an error, never a silent 'no checks'."""
    global _RANGES
    if _RANGES is None:
        p = ARTIFACTS / "ranges.json"
        if not p.exists():
            raise FileNotFoundError(f"{p} is missing; run train_stage1.py. Refusing to run without range checks.")
        _RANGES = json.loads(p.read_text())
    return _RANGES


def check(p: dict) -> dict:
    """p: raw input dict (any subset of fields). Returns {'hard': [...], 'beyond': [...], 'outside': [...]} of human-readable notes."""
    hard, beyond, outside = [], [], []
    age, hr = p.get("age"), p.get("thalch")
    if age is not None and hr is not None and hr > (220 - age) + 20:
        hard.append(f"Peak heart rate {hr} bpm is far above the age-expected maximum (about {220 - age} bpm). Please re-check it.")
    if p.get("trestbps") is not None and p["trestbps"] < 90:
        hard.append(f"Resting blood pressure {p['trestbps']} mmHg is unusually low. Please re-check it.")
    if p.get("chol") is not None and p["chol"] > 450:
        hard.append(f"Cholesterol {p['chol']} mg/dl is extremely high. Please re-check it.")
    for f, r in ranges().items():
        v = p.get(f)
        if v is None:
            continue
        name = LABEL.get(f, f).capitalize()
        if v < r["min"] or v > r["max"]:
            beyond.append(f"{name} ({v}) is outside the range this model was trained on ({r['min']:g} to {r['max']:g}), so it cannot judge it reliably.")
        elif v < r["p01"] or v > r["p99"]:
            outside.append(f"{name} ({v}) is at the edge of the range seen in training ({r['p01']:g} to {r['p99']:g}).")
    return {"hard": hard, "beyond": beyond, "outside": outside}


def reliable(pl: dict) -> bool:
    return not pl["hard"] and not pl["beyond"] and len(pl["outside"]) < 2


def apply(conf: dict, pl: dict) -> dict:
    """Add warnings to a confidence dict and cap its band."""
    order = ["low", "moderate", "high"]
    cap = "low" if (pl["hard"] or pl["beyond"]) else "moderate" if pl["outside"] else None
    if cap and order.index(conf["band"]) > order.index(cap):
        conf["band"] = cap
    conf["warnings"] = list(conf["warnings"]) + pl["hard"] + pl["beyond"] + pl["outside"]
    if not reliable(pl):
        conf["warnings"].append("The model estimate is not shown as reliable for these inputs; rely on the measurement cards and a clinician.")
    return conf
