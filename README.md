# SymbioForge CDSS

**Cardiovascular Clinical Decision Support System.** A lab-practical prototype (CELBC710, Tasks 9-10) that takes a patient's clinical attributes and returns a risk estimate together with the evidence needed to judge whether to trust it.

> **Decision support only, not a diagnosis.** Trained on a public research dataset; not clinically validated.

## What it does

For one patient at a time, the system returns:

| Output | How |
|---|---|
| Risk estimate and verdict (present / absent) | XGBoost on 12 engineered features |
| Second opinion | Deep neural network; the UI flags when the two models disagree |
| Rule-base cross-check | Task 2 expert rules R1-R7, evaluated three-valued (fired / not fired / unknown), never on guessed values |
| Per-patient explanation | Exact TreeSHAP contributions plus a plain-language summary |
| Confidence | Trust band from validation accuracy, plus a conformal prediction set (90%) |
| Reliability warnings | Missing optional tests, out-of-range age, model or rule disagreement |
| "What would flip it" | Model-sensitivity search over plausible changes (not clinical advice) |
| Next-test advice | A* search (Task 3) over the missing tests, ranked by whether they could change the verdict |
| Report | Print-ready HTML (print to PDF) |

## Technology

| Layer | Tools |
|---|---|
| Language | Python 3.11+ |
| UI / dashboard | Streamlit, Plotly |
| Models | XGBoost (primary, saved as portable JSON), PyTorch-trained DNN served in plain NumPy |
| Data / ML | pandas, NumPy, scikit-learn (CV, metrics, scaling) |
| Validation | Pydantic v2 (field-level errors) |
| Explainability | XGBoost native TreeSHAP, conformal prediction, counterfactual search |
| Reasoning | Rule-based forward chaining, A* state-space search |
| Evaluation | pytest (17 tests), matplotlib |
| Cost | Zero: all open source, runs locally |

## Flowchart

```mermaid
flowchart TD
    A[Patient input<br/>13 attributes, 3 optional] --> B{Validation<br/>Pydantic}
    B -- invalid --> B1[Field-level errors]
    B -- valid --> C[Fill missing optional fields<br/>with training mode and flag them]
    C --> D[Feature engineering<br/>12 features, fixed order]
    D --> E[XGBoost<br/>primary verdict]
    D --> F[DNN<br/>second opinion]
    A --> G[Rule base R1-R7<br/>observed values only]
    E --> H[Verdict, P disease]
    F --> I{Models agree?}
    G --> J{Rules agree with ML?}
    H --> K[SHAP explanation]
    H --> L[Confidence band and<br/>conformal set]
    H --> M[Counterfactual<br/>sensitivity search]
    G --> N[A* next-test advisor]
    I --> L
    J --> L
    K --> O[Summary and report]
    L --> O
    M --> O
    N --> O
    O --> P[Streamlit dashboard<br/>6 tabs + downloadable report]
```

## Run it (PowerShell)

From the project folder:

```powershell
cd C:\Users\aarya\OneDrive\Desktop\CODING\SymbioForge
```

**1. Create and activate a virtual environment (optional but recommended)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
If activation is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

**2. Install runtime dependencies**
```powershell
pip install -r requirements.txt
```

**3. Start the dashboard**
```powershell
streamlit run app.py
```
Open http://localhost:8501. Pick a scripted case under "Load demo patient", or enter your own values.

The trained models (`artifacts/`) and evaluation outputs (`reports/`) are committed, so no training is needed to run the app.

## Verify and rebuild (optional)

```powershell
pip install -r requirements-train.txt   # adds torch, matplotlib, pytest
python -m pytest -q                     # 17 tests
python train.py                         # re-clean, split, train XGBoost + DNN, calibrate trust bands (several minutes)
python eda.py                           # EDA and correlation report
python evaluate.py                      # test-set metrics, calibration, fairness, ablation, robustness, latency
python build_docs.py                    # regenerate docs/PROJECT_REPORT.md from the results
```
Run the rebuild scripts in that order. `train.py` overwrites `artifacts/`, `data/Task4*` and `data/Task5*`.

## Project layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `cdss/features.py` | Feature formulas (single source of truth) and Task 4 cleaning |
| `cdss/schemas.py` | Input validation |
| `cdss/rules.py` | Task 2 rule base, three-valued logic |
| `cdss/advisor.py` | Task 3 A* next-test planner |
| `cdss/engine.py` | Inference pipeline, SHAP, counterfactuals, confidence, summary |
| `cdss/report.py` | HTML report generator |
| `train.py`, `eda.py`, `evaluate.py`, `build_docs.py` | Reproducible build and evaluation scripts |
| `data/` | Input dataset, cleaned dataset, feature matrix |
| `artifacts/` | `xgb.json`, `dnn.json` (served), `dnn.pt` (training checkpoint), `meta.json` |
| `reports/` | Evaluation JSON, EDA report, model comparison, figures, sample report |
| `docs/` | `PROJECT_REPORT.md` (full summary), `DELIVERABLES.md` (task checklist) |
| `SPEC.md`, `ARCHITECTURE.md`, `PLAN.md`, `context.md` | Requirements, design and decision log, plan, prior-task reference |

## Deploy

Push to GitHub, then create an app on Streamlit Community Cloud with main file `app.py` (Python 3.11+, `requirements.txt` only; no secrets, nothing stored).

## Limitations

Historical data from four hospitals; small test set (138 patients), so model differences are within noise; no external validation; counterfactuals are model sensitivity, not treatment effects. See `docs/PROJECT_REPORT.md` for the full responsible-AI discussion.
