"""Snapshot of one full analysis for the 'How it works' view, so that view needs no model run. Run after train.py: python build_demo_trace.py"""
import json

from cdss import REPORTS
from cdss.demo import DEMO_PATIENTS
from cdss.orchestrator import assess
from cdss.schemas import PatientInput

c = assess(PatientInput(**DEMO_PATIENTS["Borderline / missing data"]))
r = c["result"]
snap = {
    "example": "Borderline, tests missing",
    "trace": r["trace"], "rules": r["rules"], "features": r["features"], "shap_all": r["shap_all"], "models": r["models"],
    "verdict": r["verdict"], "confidence": {k: r["confidence"][k] for k in ("band", "conformal_set", "conformal_level")},
}
(REPORTS / "demo_trace.json").write_text(json.dumps(snap, indent=1, default=float), encoding="utf-8")
print("saved", REPORTS / "demo_trace.json")
