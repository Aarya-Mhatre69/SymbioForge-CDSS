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
| D10 | Two-stage flow: stage 1 (home checkup: history, BP, blood tests, resting ECG) shows a tier only; stage 2 adds stress-test results for the full analysis | The full model needs stress-test inputs nobody has at home; a stage-1 model without them keeps AUC 0.868 vs 0.896 | Slightly weaker model; stage 2 stays available |
| D11 | Stage 1 never shows a percentage | Training data are referred patients (55% disease). At 5-10% prevalence most flags would be false alarms (PPV about 10-19% at the check-up threshold) | Less precise-looking output, deliberately |
| D12 | Red-flag safety screen before any model; urgent symptoms end the flow with an emergency message | A public-facing tool must not process emergencies as screening | One extra step |
| D13 | Plausibility guard: the estimate is withheld for a hard physiological flag, for any value beyond the training min-max, or for two values at the edge of the range (1st-99th percentile); one edge value only caps confidence at 'moderate'. Fails closed if ranges.json is missing | Outlier rows were less accurate in out-of-fold tests (AUC 0.77 vs 0.88, small n); trees extrapolate silently | About 2% of real patients get no estimate |
| D14 | Clinician queue (SQLite, demo): nothing stored without explicit consent; disagreement bumps a case one tier up (bumped case first on ties); clinician view behind a demo passcode | Gives 'flag if concerning' a recipient; conservative triage | Demo has no authentication |
| D15 | What-if curves carry a caution when the model's relationship is weak or opposite to clinical expectation | Out-of-fold sweeps showed blood pressure and cholesterol effects are tiny or inverted in this historical data | Curves are labelled as behaviour, not advice |
| D16 | Stage-1 policy: typical angina is never 'lower'; a pain-free presentation is capped at 'worth a check-up' | The data are referral-biased (79% of pain-free patients had disease), which does not hold for the general public (reviewer finding) | Hard-coded clinical rule on top of the model |
| D17 | Red-flag screen cannot be skipped: example buttons and 'Edit my answers' route through it; every model result requires it | Reviewer found two bypass paths | One extra click for demos |
| D18 | Every probability output (opinion cards, report, chat, what-if) is gated on the plausibility guard | Reviewer found percentages leaking when the estimate was withheld | None |
| D19 | The clinic website is the entry point and redirects to the app (no iframe); the app links back and offers booking; the return URL comes from configuration, never from the query string | A visitor meets the clinic first; redirect avoids embedding limits and keeps the app full-width; fixed return URL avoids open redirects; nothing sensitive travels in URLs | Two servers to run (one command: run_demo.py) |
