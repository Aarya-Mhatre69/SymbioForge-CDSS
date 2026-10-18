# SymbioForge CDSS

Cardiovascular clinical decision support prototype (CELBC710, Tasks 9-10). Decision support only, not a diagnosis.

Raw patient inputs -> validation -> 12 engineered features -> XGBoost (primary) + DNN (second opinion) -> Task 2 rule cross-check -> SHAP explanation -> trust bands + conformal set -> counterfactual hint -> A* next-test advice -> printable report.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Artifacts (`artifacts/`) and evaluation outputs (`reports/`) are committed, so the app runs without retraining.

## Rebuild / verify
```bash
pip install -r requirements-train.txt
python train.py        # Tasks 4-8: clean, split 70/15/15, features, model comparison, DNN, trust calibration
python eda.py          # Tasks 4-5: EDA, data quality, correlation report
python evaluate.py     # Task 10: metrics, calibration, fairness, ablation, robustness, latency, figures
python -m pytest -q    # 17 tests: feature parity, model parity, rules, validation, A*, dashboard smoke test
```

## Deploy (Streamlit Community Cloud)
Push the repo, create an app pointing at `app.py` (this folder is the repo root), Python 3.11+. Uses `requirements.txt` only (no torch/TensorFlow at runtime; the DNN runs in numpy from `artifacts/dnn.json`). No secrets needed, nothing is stored.

## Layout
| Path | Purpose |
|---|---|
| `cdss/features.py` | Only place feature formulas live; Task 4 cleaning + imputer |
| `cdss/schemas.py` | Input validation with field-level errors |
| `cdss/rules.py` | Task 2 rules R1-R7, three-valued (fired / not fired / unknown) |
| `cdss/advisor.py` | Task 3 A* next-test planner |
| `cdss/engine.py` | Inference pipeline, SHAP, counterfactual, confidence, summary |
| `cdss/report.py` | Print-ready HTML report |
| `app.py` | Streamlit dashboard |
| `docs/` | `PROJECT_REPORT.md` (summary of everything), `DELIVERABLES.md` (task checklist) |
