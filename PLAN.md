# PLAN.md

Status key: [ ] todo · [~] in progress · [x] done

## Status: Phases 0-5 prototype complete (see docs/PROJECT_REPORT.md). Remaining: public deploy, EDA report, repeated-CV.

## Phase 0 — Audit
- [ ] Confirm artifacts load: XGB pkl, DNN h5, scaler pkl, in a clean env
- [ ] Record training env versions (python, xgboost, keras/tensorflow, sklearn, shap) and pin them
- [ ] Verify Task 5 feature parity on `Task4_clean_dataset.csv` → `Task5_Feature_Matrix.csv`
- [ ] Check for leakage: was feature selection / scaler fitted before the train/test split? (affects Task 10 metrics)
- **Exit:** parity tests green, env pinned, leakage question answered.

## Phase 1 — Core pipeline (no UI)
- [ ] `schemas.py` validation + missing-input policy
- [ ] `features.py`, `rules.py`, `models.py`, `ensemble.py`
- [ ] Parity tests for features, models, rules
- **Exit:** CLI/pytest can take a raw patient dict → verdict + rules.

## Phase 2 — Explanation & confidence
- [ ] SHAP top-N per prediction
- [ ] Confidence bands + reliability warnings
- [ ] ML vs rule agreement, XGB vs DNN disagreement flags
- [ ] Counterfactual search over modifiable fields
- **Exit:** `/predict` returns the full response contract.

## Phase 3 — API + UI
- [ ] FastAPI routes, `/meta`, `/health`
- [ ] Frontend: input form → features → prediction → explanation → confidence
- [ ] Next-test advisor (`/advise`) for partial inputs
- **Exit:** 3 demo patients run end to end in the browser.

## Phase 4 — Report & packaging
- [ ] Report (print-ready HTML → PDF)
- [ ] Dockerfile, one-command run, README
- **Exit:** fresh machine runs it in under 5 minutes.

## Phase 5 — Task 10 evaluation
- [ ] End-to-end metrics on held-out data, ensemble vs single models
- [ ] Calibration curve + Brier score
- [ ] Ablation: with vs without rule cross-check, with vs without imputed fields
- [ ] Fairness re-check (sex; age bands)
- [ ] Latency and robustness tests (bad inputs, extremes)
- [ ] Limitations section
- **Exit:** evaluation report written.

## Proposed additions (my recommendation, your call)
1. **Parity tests as the backbone.** Cheap, catches the most likely failure (train/serve skew).
2. **Reliability warning on imputed `ca`/`thal`/`slope`.** Training data had 66%/53%/34% missing; `cardiac_risk_score` silently depends on them. Biggest honesty gap in a naive build.
3. **Disagreement flags** (XGB vs DNN, ML vs rules). Turns a weakness into a visible safety feature, and demos well.
4. **Next-test advisor** via Task 3 A*. Reuses finished work and makes the system more than a classifier.
5. **Calibration check** (Brier, reliability curve) before showing probabilities as "confidence".
6. **Out-of-range warning** for inputs outside training distribution (e.g. age < 28).

## Backlog (not building unless approved)
- (done) Conformal prediction sets
- Per-site distribution shift analysis
- Offline browser mode using exported weights for both models
- Audit log of model/artifact versions per prediction

## Open questions
1. Is the deliverable judged on a live demo, a report, or both? (Sets how much UI polish vs evaluation depth.)
2. Which Python/package versions were used for training? (Pickle compatibility.)
3. Was feature selection done before the train/test split?
4. Are Colab/Kaggle notebooks the only environment, or do you have a local machine for Docker?
5. Deadline?
