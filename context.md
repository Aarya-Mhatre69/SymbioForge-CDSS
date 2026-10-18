# SymbioForge CDSS — System Context Reference
**Project:** Intelligent Clinical Decision Support System (CDSS) for Cardiovascular Disease Diagnosis
**Course:** CELBC710, Computational Intelligence Laboratory, Semester VII
**Purpose of this file:** Everything needed to build Task 9 (System Integration) and Task 10 (Evaluation/Deployment) without re-deriving any prior work. Treat this as the single source of truth for formulas, file names, column encodings, model hyperparameters, and rule definitions used across Tasks 1–8.

---

## 1. Domain & Dataset

- **Source:** UCI Heart Disease dataset, 4 collection sites (Cleveland, Hungary, Switzerland, VA Long Beach).
- **Two dataset versions were used across tasks — do not mix them up:**
  - **Cleveland-only, 297 rows** (`heart_clean.csv`) — used in Tasks 1, 2, 3. Clean, near-zero missing values.
  - **Full multi-site, 918 rows** (`Task4_clean_dataset.csv`) — used in Tasks 4–8. Larger, required real data-cleaning work.
- **Raw columns (as originally provided):**
  `id, age, sex, dataset, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal, num`
- **Target:** `num` (original 0–4 severity scale) binarized everywhere to `target`: `1` = disease present (num > 0), `0` = disease absent.

### Raw column encodings (exact strings used throughout)
| Column | Type | Values |
|---|---|---|
| `sex` | nominal | `"Male"` / `"Female"` (raw) → lowercased to `"male"`/`"female"` during cleaning |
| `cp` | nominal | `"typical angina"`, `"atypical angina"`, `"non-anginal"`, `"asymptomatic"` |
| `fbs` | boolean | `True`/`False` (fasting blood sugar > 120 mg/dl) |
| `restecg` | nominal | `"normal"`, `"lv hypertrophy"`, `"st-t abnormality"` |
| `exang` | boolean | `True`/`False` (exercise-induced angina) |
| `slope` | nominal | `"upsloping"`, `"flat"`, `"downsloping"` |
| `ca` | numeric | 0–3 (number of major vessels via fluoroscopy) |
| `thal` | nominal | `"normal"`, `"fixed defect"`, `"reversable defect"` |

### Known data-quality issues (already handled, don't re-fix)
- `chol == 0` and `trestbps == 0` are **disguised missing values** (concentrated in Switzerland/VA Long Beach), not real readings — converted to NaN before any imputation.
- ARFF file exports use `"?"` as the missing marker for nominal columns — must be replaced with NaN before processing (scipy's arff loader does NOT do this automatically for string/nominal columns, only numeric).
- `slope` (33.6%), `ca` (66.4%), `thal` (52.8%) have high missingness — imputed with mode, accompanied by `*_missing` indicator flags.

---

## 2. Task 1 — Knowledge Representation

- **Entities:** Patient, Symptom/Clinical Finding, Risk Factor, Diagnostic Test Result, Diagnosis.
- **Method:** Apriori association rule mining (minSupport=0.15, minConfidence=0.80) on discretized Cleveland data (297 rows).
- **Discretization bins used:**
  - age: `<40 / 40-50 / 50-60 / >60`
  - trestbps: `<120 / 120-140 / 140-160 / >160`
  - chol: `<200 / 200-240 / >240`
  - thalch: `<100 / 100-140 / 140-160 / >160`
  - oldpeak: `0 / 0-1 / 1-2 / >2`
- **Output:** 142 diagnosis-relevant rules mined; top 12 by confidence/lift form the knowledge graph.
- **Key finding (confirmed in every later task):** `cp=asymptomatic` is the single strongest signal for disease presence.

---

## 3. Task 2 — Rule-Based Expert System (THE CANONICAL RULE BASE)

Trained via RIPPER (JRip-equivalent) on the 297-patient Cleveland dataset. **This exact rule base is reused as-is in Task 3 and should be reused again in Task 9's reasoning/explanation layer:**

```
R1: cp = asymptomatic  AND  thal = reversable defect           → present
R2: slope = flat  AND  sex = male  AND  thalch < 116            → present
R3: slope = flat  AND  sex = male  AND  ca = 1                  → present
R4: oldpeak > 2.8                                                → present
R5: trestbps > 152.8  AND  age in [58, 60]                      → present
R6: age in [56, 58]  AND  cp = atypical angina                  → present
R7 (default): none of R1–R6 fire                                 → absent
```
Evaluated top-to-bottom (forward chaining). Resubstitution accuracy 85.5%, 10-fold CV accuracy 78.8% (±9.3%).

A parallel Decision Tree (max_depth=4) was trained as a cross-check / "Rule Tree"; 10-fold CV accuracy 81.5% (±5.1%).

---

## 4. Task 3 — State-Space Search (Diagnostic Test Ordering)

- **Formulation:** state = set of tests performed; action = perform one more test; goal = a Task 2 rule fires (present) or all 6 are eliminated (absent).
- **Test costs (clinical invasiveness proxy):** age=1, sex=1, cp=1, trestbps=2, thalch=4, oldpeak=4, slope=4, thal=6, ca=8.
- **Algorithms compared:** BFS, DFS, A*. **A\* is the recommended strategy** — same optimal cost as BFS but fewer node expansions; DFS is consistently worst on cost.
- Relevant if Task 9 wants to recommend "what test to order next" rather than requiring all inputs upfront.

---

## 5. Task 4 — Data Cleaning Pipeline (apply this exact sequence to any new raw data)

1. Replace `"?"` → NaN (ARFF artifact).
2. Set `chol == 0` and `trestbps == 0` → NaN (disguised missing).
3. Drop `id`, `dataset` columns (not clinical features).
4. Drop exact duplicate rows.
5. Binarize target: `target = 1 if num > 0 else 0`; drop `num`.
6. Impute: numeric columns (trestbps, chol, thalch, oldpeak) → site-grouped median, fallback to global median. Categorical low-missingness (fbs, restecg, exang) → mode. High-missingness (slope, ca, thal) → add `{col}_missing` indicator flag, then mode-impute.
7. Outlier policy: IQR-flagged values were **retained** (not removed/capped) — they are clinically plausible extremes, not entry errors.

Output: `Task4_clean_dataset.csv` (918 rows, 0 missing values, 20 columns including 3 missing-indicator flags).

---

## 6. Task 5 — Feature Engineering (EXACT FORMULAS — required for any new prediction)

```python
rate_pressure_product = trestbps * thalch
hr_reserve = (220 - age) - thalch
chol_age_ratio = chol / age
bp_chol_interaction = (trestbps * chol) / 1000          # NOT in final 12 selected features
cardiac_risk_score = sum of 8 binary conditions:
    age > 55
    sex == 'male'
    chol > 240
    trestbps > 140
    fbs == True
    exang == True
    ca >= 1
    thal != 'normal'
slope_severity = {'upsloping': 0, 'flat': 1, 'downsloping': 2}[slope]
oldpeak_slope_risk = oldpeak * (slope_severity + 1)
```

**Encoding for modeling:**
- `sex` → `sex_male` (1 if male else 0)
- `fbs`, `exang` → 0/1 int
- `cp`, `restecg`, `slope`, `thal` → one-hot encoded (e.g. `cp_asymptomatic`, `cp_atypical angina`, `cp_non-anginal`, `cp_typical angina`)

**Final 12 selected features (ranked by averaged Random Forest + Mutual Information rank)** — **this exact list and this exact order is the required input to the Task 6 XGBoost model and the Task 7 DNN:**
```
['cp_asymptomatic', 'exang', 'oldpeak_slope_risk', 'cardiac_risk_score', 'thalch',
 'cp_atypical angina', 'oldpeak', 'chol', 'hr_reserve', 'rate_pressure_product',
 'age', 'chol_age_ratio']
```
Output: `Task5_Feature_Matrix.csv` (918 rows × 12 features + target).

---

## 7. Task 6 — Classical ML Models

5 models trained (75/25 stratified split, 10-fold CV). **Best model: XGBoost.**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.813 | 0.809 | 0.866 | 0.837 | 0.889 |
| Decision Tree | 0.778 | 0.797 | 0.803 | 0.800 | 0.858 |
| Random Forest | 0.800 | 0.800 | 0.850 | 0.824 | 0.877 |
| SVM (RBF) | 0.796 | 0.790 | 0.858 | 0.823 | 0.875 |
| **XGBoost** | **0.826** | **0.837** | **0.850** | **0.844** | **0.891** |

**XGBoost hyperparameters:** `n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='logloss'`.
**No feature scaling needed for XGBoost** (tree-based). LR and SVM used `StandardScaler`.

**Model file:** `Task6_Model_XGBoost_BEST.pkl` — **this is a raw `XGBClassifier` object (NOT a dict/bundle)**, loadable directly via `pickle.load()`. Call `.predict(X)` / `.predict_proba(X)` directly with a DataFrame of the 12 features in the exact order above.

---

## 8. Task 7 — Deep Neural Network

**Architecture (Sequential Keras model):**
```
Input(12) → Dense(64, relu) → BatchNorm → Dropout(0.3)
          → Dense(32, relu) → BatchNorm → Dropout(0.3)
          → Dense(16, relu) → Dropout(0.2)
          → Dense(1, sigmoid)
```
3,843 total params (3,649 trainable, 192 non-trainable).

**Training config:** Adam (lr=0.001), binary cross-entropy, batch_size=32, max 200 epochs, EarlyStopping(patience=20, monitor=val_loss, restore_best_weights=True), ReduceLROnPlateau(factor=0.5, patience=8). Train/Val/Test = 70/15/15 (642/138/138). Actually stopped at epoch 34, restored weights from epoch 14.

**Test performance:** Accuracy 0.819, Precision 0.840, Recall 0.829, F1 0.834, **ROC-AUC 0.904 (best of all models across Tasks 6 and 7)**.

**CRITICAL:** The DNN requires features to be scaled with the **fitted StandardScaler** before prediction — raw/unscaled input will give wrong results. Always apply `scaler.transform(X)` first.

**Files:**
- `Task7_DNN_Model.h5` — Keras model, load via `keras.models.load_model()`.
- `Task7_DNN_Scaler.pkl` — fitted `StandardScaler`, load via `pickle.load()`.
- `dnn_weights.json` (if present) — raw weight arrays + `scaler_mean`/`scaler_scale`, used for the dependency-free browser demo (`Task7_DNN_Live_Demo.html`). Useful if Task 9's frontend needs a no-backend prediction path.

**Verified manual forward-pass formula** (ReLU, BatchNorm inference-mode, sigmoid output) — matches Keras to 6 decimal places:
```
bn(x) = gamma * (x - moving_mean) / sqrt(moving_var + 1e-3) + beta
```

---

## 9. Task 8 — Explainability (XAI)

Explains the **Task 6 XGBoost model** (not the DNN).

- **SHAP (TreeExplainer):** Global ranking — `cp_asymptomatic` dominant, then `chol`, `cardiac_risk_score`, `oldpeak`. Matches Task 1/2/5 findings independently (4 different methods agree).
- **LIME:** Cross-validates SHAP's local explanations on the same patients — independent agreement.
- **Counterfactual analysis:** High-confidence predictions (e.g. P=0.998) are robust — no single feature change flips them; requires ~4 simultaneous changes. Use this logic if Task 9 wants a "what would need to change" patient-facing feature.
- **Trust score:** Confidence-weighted accuracy ratio = 0.878. Model is well-calibrated (66%→96% accuracy as confidence band increases) — safe to surface the model's own probability as a trust/confidence indicator in the UI.
- **Fairness:** Equal ROC-AUC across sex (0.887 male / 0.885 female); an apparent precision gap was traced to test-set sample size and the dataset's known under-representation of female "present" cases, not differential model treatment — but flagged as a limitation worth re-checking on more data before real deployment.

**Dashboard reference:** `Task8_Explainability_Dashboard.html` — self-contained, can be reused/embedded as a reference UI pattern for Task 9.

---

## 10. File Inventory (all in outputs so far)

| File | From Task | Purpose |
|---|---|---|
| `heart_clean.csv` | 1 | Cleveland-only, 297 rows, used for KR/rules/search |
| `Task4_clean_dataset.csv` | 4 | Full 918-row cleaned dataset |
| `Task5_Feature_Matrix.csv` | 5 | 918 rows × 12 features + target — **the standard ML input** |
| `Task6_Model_XGBoost_BEST.pkl` | 6 | Best classical model (raw XGBClassifier) |
| `Task6_Model_{LogisticRegression,DecisionTree,RandomForest,SVM}.pkl` | 6 | Other trained models |
| `Task7_DNN_Model.h5` | 7 | Trained Keras DNN |
| `Task7_DNN_Scaler.pkl` | 7 | StandardScaler fitted for the DNN |
| `Task7_DNN_Live_Demo.html` | 7 | Dependency-free browser demo (weights embedded) |
| `Task8_Explainability_Dashboard.html` | 8 | SHAP/LIME/trust/fairness dashboard |

---

## 11. What Task 9 (System Integration) Needs To Do

Per the course's general task template, Task 9 = integrate Tasks 1–8 into one working prototype:

1. **Input validation** — accept raw clinical attributes (the columns in Section 1), validate types/ranges.
2. **Feature engineering pipeline** — apply Section 6's exact formulas to go from raw input → 12-feature vector, in the exact column order given.
3. **Model execution** — run both XGBoost (Task 6, no scaling) and/or DNN (Task 7, requires `scaler.transform` first) on the 12-feature vector.
4. **Rule-based cross-check (optional but recommended)** — also evaluate the Task 2 rule base (Section 3) against the raw inputs, so the system can show "ML model agrees/disagrees with the explicit rule base" — good for a viva demo of system coherence.
5. **XAI explanation integration** — generate a SHAP explanation (or reuse precomputed logic from Task 8) for the specific prediction, not just a bare probability.
6. **Recommendation/output engine** — combine prediction + confidence (trust score logic from Task 8) + top contributing factors into a human-readable clinical summary.
7. **Report generation / UI** — a dashboard (web page) showing: patient inputs → engineered features → prediction (probability + verdict) → explanation → confidence.

**Recommended minimal architecture:** a single Python backend (Flask/FastAPI) wrapping the feature-engineering function + both models, with a simple HTML/JS frontend — OR, given everything already proven to work standalone in the browser (`Task7_DNN_Live_Demo.html` embeds the full DNN in JS with zero backend), a **pure client-side HTML/JS app** is also fully viable and avoids deployment complexity for a student project. The DNN weight-extraction pattern used there can be reused for XGBoost too if a no-backend demo is preferred (export tree structure to JSON, implement tree traversal in JS).

---

## 12. Things To Not Re-Derive

- Don't re-run Apriori/rule mining — the rule base in Section 3 is final and validated.
- Don't re-tune model hyperparameters — Section 7/8's configs are already the result of comparison/tuning.
- Don't reinvent feature formulas — Section 6 is exact and verified against the saved feature matrix.
- Don't re-clean data — Section 5's pipeline is the agreed, documented process; apply it as-is to any new patient record.
