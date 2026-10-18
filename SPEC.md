# SPEC.md — SymbioForge CDSS (Tasks 9–10)

## 1. Goal
One working prototype that takes raw patient attributes and returns: a risk prediction, an agreement check against the explicit rule base, a per-patient explanation, a confidence indicator, and a readable summary report.

## 2. Non-goals
- Not a diagnostic device; no clinical claims.
- No retraining, re-tuning, re-cleaning, or re-mining (see `context.md` §12).
- No user accounts or real patient data storage.

## 3. Users
- **Primary:** clinician/evaluator in a viva demo, entering one patient at a time.
- **Secondary:** examiner reading the generated report and architecture.

## 4. Functional requirements
| ID | Requirement |
|---|---|
| FR1 | Accept raw inputs (schema below), validate types, ranges, enums. Return field-level errors. |
| FR2 | Handle optional/missing inputs explicitly (see §6), never silently. |
| FR3 | Build the 12-feature vector using the exact formulas and order in `context.md` §6. |
| FR4 | Run XGBoost (raw features) and DNN (`scaler.transform` first). Return both probabilities. |
| FR5 | Final verdict = XGBoost probability at threshold 0.5 (see ARCHITECTURE D3'); DNN is a second opinion. Flag model disagreement. |
| FR6 | Evaluate the Task 2 rule base R1–R7 on raw inputs, three-valued (rules with missing inputs are `unknown`, never evaluated on imputed values). Return fired rules and rule verdict. Flag ML/rule disagreement. |
| FR7 | SHAP (TreeExplainer on XGBoost): top-N contributing features with direction and magnitude. |
| FR8 | Confidence band from model probability (Task 8 calibration), plus a "low-information input" warning when imputation is heavy. |
| FR9 | Counterfactual hint: smallest set of modifiable-input changes that flips the verdict (or "robust: none found"). |
| FR10 | Next-test suggestion via Task 3 A* when key inputs (ca, thal, slope) are missing. |
| FR11 | Human-readable clinical summary + downloadable report (PDF or print-ready HTML). |
| FR12 | UI flow: inputs → engineered features → prediction → explanation → confidence. |

## 5. Input schema
| Field | Type | Accepted | Required |
|---|---|---|---|
| age | int | 18–100 (warn outside 28–77, the training range) | yes |
| sex | enum | male / female | yes |
| cp | enum | typical angina / atypical angina / non-anginal / asymptomatic | yes |
| trestbps | int | 80–220 mmHg | yes |
| chol | int | 100–600 mg/dl | yes |
| fbs | bool | true/false | yes |
| restecg | enum | normal / lv hypertrophy / st-t abnormality | yes |
| thalch | int | 60–220 bpm | yes |
| exang | bool | true/false | yes |
| oldpeak | float | -3.0–7.0 | yes |
| slope | enum | upsloping / flat / downsloping | optional |
| ca | int | 0–3 | optional |
| thal | enum | normal / fixed defect / reversable defect | optional |

Zero values for `chol` and `trestbps` are rejected (they were disguised missing in training data).

## 6. Missing-input policy
Training imputed `slope`, `ca`, `thal` with the mode and added `*_missing` flags. At inference:
- Missing optional field → mode-impute exactly as in training, mark as `imputed`, and show it in the UI.
- `cardiac_risk_score` and `oldpeak_slope_risk` depend on these fields; if any are imputed, display a "reduced reliability" warning.
- Never present a confident verdict on a record with 2+ imputed fields without that warning.

## 7. Non-functional requirements
- Single prediction latency < 500 ms locally (SHAP included).
- Deterministic: same input → same output.
- Runs with one command (`docker compose up` or `uvicorn`). Zero cost.
- Every engineered feature traceable to its formula in code.

## 8. Acceptance criteria
1. **Feature parity:** pipeline run on `Task4_clean_dataset.csv` reproduces `Task5_Feature_Matrix.csv` (max abs diff < 1e-9).
2. **Model parity:** loaded models reproduce Task 6/7 reported metrics on the same split (±0.005).
3. **Rule parity:** rule engine reproduces the Task 2 rule base outputs on `heart_clean.csv`.
4. Invalid inputs return clear field errors; no 500s from bad input.
5. Demo: 3 scripted patients (clear present, clear absent, borderline/disagreement) run end to end.

## 9. Open questions
See `PLAN.md` → "Open questions".
