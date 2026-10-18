# ARCHITECTURE.md

## 1. Shape
Python backend (FastAPI) + static frontend (HTML/JS). Backend owns all logic; frontend only renders.

```
Browser UI ──HTTP/JSON──> FastAPI
                            │
        ┌───────────────────┼────────────────────────┐
        ▼                   ▼                        ▼
   validation          features.py              rules.py (R1–R7)
   (pydantic)          (12-feature vector)             │
                            │                          │
                  ┌─────────┴─────────┐                │
                  ▼                   ▼                │
            xgb_model.pkl      dnn.h5 + scaler.pkl     │
                  │                   │                │
                  └────────┬──────────┘                │
                           ▼                           ▼
                      ensemble.py ──────────> agreement.py
                           │
              ┌────────────┼─────────────┐
              ▼            ▼             ▼
          explain.py   confidence.py   advisor.py
          (SHAP, CF)   (trust bands)   (A* next test)
                           │
                           ▼
                     report.py (summary + PDF/HTML)
```

## 2. Repo layout
```
cdss/
  app/
    main.py            # FastAPI app, routes
    schemas.py         # pydantic request/response models
    features.py        # raw -> 12-feature vector (single source of formulas)
    rules.py           # Task 2 rule base, forward chaining
    models.py          # load/cache XGB, DNN, scaler
    ensemble.py        # combine probabilities, disagreement flag
    explain.py         # SHAP + counterfactual
    confidence.py      # calibration bands, reliability warnings
    advisor.py         # A* next-test suggestion (Task 3)
    report.py
  artifacts/           # .pkl / .h5 / reference CSVs (read-only)
  web/                 # index.html, app.js, styles.css
  tests/
    test_feature_parity.py
    test_model_parity.py
    test_rules_parity.py
    test_api.py
  docs/                # context.md, SPEC.md, ARCHITECTURE.md, PLAN.md, prompt.md
  Dockerfile
  requirements.txt     # pinned versions
```

## 3. API contract
| Method | Path | Purpose |
|---|---|---|
| POST | `/predict` | Full pipeline; returns validation echo, features, probabilities, verdict, rules, explanation, confidence, warnings |
| POST | `/explain` | SHAP + counterfactual only |
| POST | `/advise` | Next-test suggestion for partial input |
| POST | `/report` | Returns report (HTML/PDF) |
| GET | `/health` | Model versions, artifact hashes |
| GET | `/meta` | Feature list, schema, thresholds (UI renders from this) |

`/predict` response (abridged):
```json
{
  "inputs": {...}, "imputed": ["ca"],
  "features": {"cp_asymptomatic": 1, "...": 0},
  "models": {"xgb": 0.87, "dnn": 0.91, "ensemble": 0.89},
  "verdict": "present", "model_disagreement": false,
  "rules": {"fired": ["R1"], "verdict": "present", "agrees_with_ml": true},
  "explanation": {"top": [{"feature": "cp_asymptomatic", "shap": 0.84}]},
  "counterfactual": {"found": false, "note": "robust"},
  "confidence": {"band": "high", "warnings": []},
  "disclaimer": "Decision support only."
}
```

## 4. Key design rules
- **One feature function.** `features.py` is the only place formulas live; training-parity test guards it.
- **Artifacts are immutable and hashed.** `/health` reports hashes so the demo proves which models ran.
- **Stateless.** No patient data persisted. Logs hold no inputs.
- **Pinned dependencies.** XGBoost/Keras/scikit-learn versions must match the training environment, or pickles may misbehave.

## 5. Decision Log
| # | Decision | Why | Trade-off |
|---|---|---|---|
| D1 | FastAPI backend instead of pure client-side | SHAP TreeExplainer and counterfactual search are Python-native; porting to JS adds risk with no user value | Needs a running server; mitigated by Docker one-liner |
| D2 | Keep `dnn_weights.json` browser demo as optional offline fallback | Already proven; useful if demo environment has no backend | Duplicate logic; kept out of main path |
| D3 | Ensemble = mean of XGB and DNN probabilities | Simple, explainable; DNN has best AUC, XGB best on accuracy | Not tuned; revisit in Task 10 with calibration check |
| D4 | SHAP explains XGBoost only | Matches Task 8; TreeExplainer is exact and fast | DNN has no per-prediction explanation; state this in UI |
| D3' | Supersedes D3: verdict = XGBoost only; DNN shown as a second opinion with a disagreement flag | Mean ensemble did not beat XGBoost on test; SHAP explains XGBoost, so verdict and explanation stay consistent (reviewer finding) | Single-model verdict |
| D5 | Rules are three-valued (fired / not fired / unknown) | A rule must not fire on an imputed thal/slope/ca | Many patients get `undetermined` |
| D6 | DNN trained in PyTorch, served in numpy from `dnn.json` | TensorFlow would not load on the dev machine; no heavy runtime dependency | Not a Keras .h5 |
| D7 | Numeric imputation fallback = seeded draw from observed train values, not the global median | Switzerland has no observed chol; a constant filler became a learnable disease proxy (reviewer finding) | Departs from context.md section 5 step 6 for that fallback only |
| D8 | Confidence band labelled from validation accuracy per confidence range (>=0.90 high, >=0.75 moderate, else low) | Fixed probability cut-offs ignored calibration (reviewer finding) | Bands are small (n ~ 20-50) |
| D9 | Counterfactuals renamed "model sensitivity"; modifiable = chol, trestbps only, capped at +-30%, shown only to lower risk | `fbs` is a lab result; large changes implied treatment effects (reviewer finding) | Fewer suggestions |
