# Deliverables checklist (from the General Project Task Template)

Legend: [x] produced here · [~] regenerated on the supplied dataset, needs your prior-task version/report · [ ] not produced (needs you)

## Tasks 1-3 (your earlier practicals; inputs only, reused here)
| # | Deliverable | Status |
|---|---|---|
| 1 | Entity/Attribute/Relationship tables, Ontology, Knowledge Graph, Validation Report | [ ] already submitted in your Task 1 report; nothing rebuilt |
| 2 | Rule Base | [x] `cdss/rules.py` (R1-R7, from context.md) |
| 2 | Rule Tree, Decision Table, Inference Trace, Validation Cases | [~] Inference trace shown live in dashboard (rule list per patient); tables remain in your Task 2 report |
| 3 | State-space diagram, BFS/DFS/A* results, heuristic table, performance report | [~] A* implemented in `cdss/advisor.py`; BFS/DFS comparison remains in your Task 3 report |

## Tasks 4-8 (re-created on `data/heart_dataset.csv` so the system is self-contained)
| # | Deliverable | Status |
|---|---|---|
| 4 | Clean dataset | [x] `data/Task4_clean_dataset.csv` (918 rows, 0 missing, 3 missing flags) |
| 4 | EDA, statistical summary, visualization, data quality report | [x] `reports/EDA_report.md` + `figures/eda_*.png` (`python eda.py`); also the dashboard's Data (EDA) tab |
| 5 | Feature matrix, selected features | [x] `data/Task5_Feature_Matrix.csv`, 12 features per context.md; `reports/feature_mutual_info.csv` |
| 5 | Correlation matrix, importance report | [x] `figures/eda_correlation.png`, mutual information + SHAP importance in `reports/` |
| 6 | Model files, evaluation, comparison | [x] `artifacts/xgb.json`, `reports/Task6_model_comparison.csv` (7 models, 5-fold CV) |
| 6 | ROC curves, confusion matrix | [x] `reports/figures/roc.png`, `confusion.png` |
| 7 | DNN model file, training history, curves | [x] `artifacts/dnn.pt` + `dnn.json`, `reports/Task7_training_history.csv`, `figures/dnn_training.png` (PyTorch, since TensorFlow would not load on this machine) |
| 7 | Hyperparameter report | [x] PROJECT_REPORT.md section 3 (selected XGBoost params, DNN config) |
| 8 | SHAP report | [x] per-patient in dashboard; global `figures/shap_global.png` |
| 8 | LIME report | [ ] dropped (SHAP is exact for trees; LIME adds nothing here) |
| 8 | Counterfactual analysis | [x] in dashboard and report (`engine.counterfactual`) |
| 8 | Dashboard, trust score | [x] Confidence tab + `evaluation.json` trust bands, conformal coverage |
| 8 | Fairness | [x] `evaluation.json["fairness"]` (sex, age band, site) |

## Task 9 - System Integration & Deployment
| Deliverable | Status |
|---|---|
| Application (running instance) | [x] `streamlit run app.py` |
| Integrated source code | [x] `cdss/`, `app.py`, `train.py`, `evaluate.py` |
| Dashboard | [x] Two views: guided chat Health check (overview, measurements, what-if, next steps, follow-up chat) and How it works (architecture + live trace, internals, evaluation, EDA, limits) |
| Report generator | [x] `cdss/report.py`, download button; sample `reports/sample_report.html` |
| Deployment documentation | [x] README.md + PROJECT_REPORT.md section 8 |
| Public URL | [ ] needs you: push to GitHub and connect Streamlit Community Cloud |

## Task 10 - Evaluation, Responsible AI, Documentation (all in `docs/PROJECT_REPORT.md`)
| Deliverable | Status |
|---|---|
| Evaluation report | [x] section 5 + `reports/evaluation.json` |
| Responsible AI report | [x] section 6 |
| User manual | [x] section 7 |
| SRS / SDD | [x] SPEC.md (requirements), ARCHITECTURE.md (design), section 2 |
| Testing report | [x] section 4 (20 automated tests) |
| Final project report | [x] PROJECT_REPORT.md |
| Deployment guide | [x] section 8 |
