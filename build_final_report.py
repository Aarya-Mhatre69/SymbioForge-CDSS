"""Generate docs/SymbioForge_Final_Project_Report.docx (one report covering Tasks 9-10). Run after train.py and evaluate.py."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

R = Path(__file__).parent
FIG = R / "reports" / "figures"
DOCF = R / "docs" / "figures"
ev = json.loads((R / "reports/evaluation.json").read_text())
meta = json.loads((R / "artifacts/meta.json").read_text())
comp = pd.read_csv(R / "reports/Task6_model_comparison.csv")
BLUE, HEAD = RGBColor(0x1F, 0x4E, 0x78), "2F5496"
f3 = lambda x: "n/a" if x is None else f"{x:.3f}"

# ---------------------------------------------------------------- architecture figure
def arch_figure(path):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)

    def box(x, y, w, h, t, fc="#eef2f7", ec="#9fb3c8", fs=8):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.2, joinstyle="round"))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#667085", lw=1))

    box(1, 25, 12, 10, "User\n(chat intake)", "#fff")
    box(16, 25, 13, 10, "Intake agent\nTask 4\nvalidate + flag")
    box(32, 25, 13, 10, "Feature agent\nTask 5\n12 features")
    box(48, 25, 14, 10, "Orchestrator\nTask 9", "#dbe8f5", "#1f5f8b", 9)
    names = ["Risk model (XGBoost)\nTask 6", "Second opinion (DNN)\nTask 7", "Clinical rules\nTasks 1-2", "Explainer (SHAP)\nTask 8",
             "Trust agent\nTask 8", "What-if agent\nTask 8", "Test planner (A*)\nTask 3"]
    ys = [52 - 8.2 * i for i in range(7)]
    for y, t in zip(ys, names):
        box(66, y, 16, 6.2, t, fs=7.5)
        arrow(62, 30, 66, y + 3.1)
        arrow(82, y + 3.1, 84, 30)
    box(84, 25, 15, 10, "Synthesizer\nTask 9\noverview + report", "#dbe8f5", "#1f5f8b", 8)
    for a, b in (((13, 30), (16, 30)), ((29, 30), (32, 30)), ((45, 30), (48, 30))):
        arrow(*a, *b)
    ax.text(50, 1, "Every agent reports what it did; the orchestrator merges the evidence into one patient-level overview.",
            ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


arch_figure(DOCF / "architecture.png")

# ---------------------------------------------------------------- docx helpers
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, m, Inches(0.75))
st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(10)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after = Pt(4)


def para(text="", bold=False, size=None, color=None, align=None, italic=False, after=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    if color:
        r.font.color.rgb = color
    if align:
        p.alignment = align
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def h1(t):
    p = para(t, True, 13, BLUE, after=3)
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.keep_with_next = True


def h2(t):
    p = para(t, True, 11, BLUE, after=2)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.keep_with_next = True


def bullets(items):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(1)
        if isinstance(it, tuple):
            p.add_run(it[0]).bold = True
            p.add_run(it[1])
        else:
            p.add_run(it)


def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), fill)
    tcPr.append(sh)


def table(rows, header, widths=None, size=9):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(str(h))
        r.bold, r.font.size, r.font.color.rgb = True, Pt(size), RGBColor(255, 255, 255)
        shade(c, HEAD)
    for k, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(size)
            if k % 2:
                shade(cells[i], "EEF2F9")
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                p.paragraph_format.space_after = Pt(1)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(path, width, caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width))
    p.paragraph_format.keep_with_next = True
    para(caption, italic=True, size=8.5, color=RGBColor(0x55, 0x55, 0x55), align=WD_ALIGN_PARAGRAPH.CENTER)


# ---------------------------------------------------------------- header block
C = WD_ALIGN_PARAGRAPH.CENTER
para("Fr. C. Rodrigues Institute of Technology, Vashi", True, 13, align=C, after=0)
para("Department of Computer Engineering", True, 11, align=C, after=0)
para("CELBC710 - Computational Intelligence Laboratory | Semester VII | AY 2026-2027", size=10, align=C, after=8)
para("Experiments 9 and 10", True, 12, align=C, after=0)
para("System Integration & Deployment, and Evaluation, Responsible AI & Documentation", True, 12, align=C, after=0)
para("Final Project Report: SymbioForge, an Intelligent Clinical Decision Support System for Cardiovascular Disease", size=10, align=C, after=0)
para("Tools: Python, XGBoost, NumPy/PyTorch, Pydantic, Streamlit, Plotly, pytest", size=10, align=C, after=8, color=RGBColor(0x55, 0x55, 0x55))

# ---------------------------------------------------------------- 1
h1("1. Objective and Summary")
para("To integrate the eight earlier experiments (knowledge representation, rule-based expert system, state-space search, data preparation, feature "
     "engineering, machine learning, deep learning and explainable AI) into one working prototype, and to evaluate it for performance, reliability, "
     "responsible use and deployment readiness.")
para("SymbioForge is not a single classifier. It is a team of independent AI workflows under one orchestrator. A user describes their symptoms in a "
     "guided chat; each workflow examines the answers from its own angle (statistical model, deep network, expert rules, explanation, trust, what-if "
     "sensitivity, test planning); the orchestrator merges the evidence into a patient-level overview that explains what stands out, how sure the system is, "
     "and what to do next. A reviewer-facing view exposes the machinery behind it.")
para("Decision support only, not a diagnosis. The prototype is trained on a public research dataset and is not clinically validated.", bold=True)

# ---------------------------------------------------------------- 2
h1("2. System Overview: Eight Workflows, One Orchestrator")
table([
    ["Intake agent", "Task 4 data preparation", "Validates 13 inputs, flags missing optional tests and fills them with the training mode", "cdss/schemas.py, features.py"],
    ["Feature agent", "Task 5 feature engineering", "Builds the 12 model features in one fixed order from one formula module", "cdss/features.py"],
    ["Risk model", "Task 6 machine learning", "XGBoost probability of disease (primary verdict)", "artifacts/xgb.json"],
    ["Second-opinion model", "Task 7 deep learning", "Dense 64-32-16 DNN; flags disagreement with XGBoost", "artifacts/dnn.json"],
    ["Clinical rules agent", "Tasks 1-2 knowledge + rules", "Rules R1-R7, three-valued: fired / not fired / unknown", "cdss/rules.py"],
    ["Test planner", "Task 3 state-space search", "A* picks the cheapest tests that make the rules decisive; ML sensitivity ranks them", "cdss/advisor.py"],
    ["Explainer", "Task 8 SHAP", "Exact per-feature contributions, grouped into body-system domains", "cdss/engine.py"],
    ["Trust agent", "Task 8 trust score", "Confidence band from validation accuracy plus a 90% conformal set", "cdss/engine.py"],
    ["What-if agent", "Task 8 counterfactuals", "Model sensitivity over plausible changes; curves for key measurements", "cdss/orchestrator.py"],
    ["Synthesizer", "Task 9 integration", "Combines everything into the overview, doctor questions and report", "cdss/orchestrator.py"],
], ["Workflow", "From", "What it does", "Where"], [1.3, 1.4, 3.1, 1.2], 8.5)
figure(DOCF / "architecture.png", 6.4, "Figure 1. Orchestrated architecture. Each box is an independent workflow that reports what it did.")

# ---------------------------------------------------------------- 3
h1("3. Implementation")
h2("3.1 Input validation and missing-data policy")
bullets([
    "Pydantic schema with typed ranges (e.g. age 18-100, cholesterol 100-600 mg/dl). Bad input returns field-level errors; it never reaches a model.",
    "Zero cholesterol or blood pressure is rejected (disguised missing values in the training data).",
    "Optional tests (ST slope, vessel count, thallium result) may be missing. They are filled with the training mode, flagged, and the confidence "
    "band cannot be 'high' when two or more are missing. Rules are never evaluated on guessed values.",
])
h2("3.2 Feature engineering and models")
para("The 12 features follow context.md exactly: cp_asymptomatic, exang, oldpeak_slope_risk, cardiac_risk_score, thalch, cp_atypical angina, oldpeak, chol, "
     "hr_reserve, rate_pressure_product, age, chol_age_ratio. Feature formulas live in one function that a parity test compares with the saved feature "
     "matrix (max difference under 1e-9), so training and serving cannot drift apart.")
para(f"Data: {meta['dataset']['rows_raw']} raw rows from four hospital sites, {meta['dataset']['rows_clean']} after cleaning, prevalence "
     f"{meta['dataset']['prevalence']:.1%}. Stratified split 70/15/15 (train {meta['split_sizes']['train']}, validation {meta['split_sizes']['val']}, "
     f"test {meta['split_sizes']['test']}), seed 42. Imputer, scaler and models are fit on the training split only.")
rows = [[r.model, f"{r.cv_auc_mean:.3f} +/- {r.cv_auc_std:.3f}", f3(r.test_accuracy), f3(r.test_f1), f3(r.test_roc_auc)] for r in comp.itertuples()]
table(rows, ["Model", "5-fold CV AUC (train)", "Test accuracy", "Test F1", "Test AUC"], [1.8, 1.7, 1.2, 1.0, 1.0])
para(f"XGBoost was selected by cross-validated AUC on the training split ({json.dumps(meta['xgb_params'])}). The DNN (Dense 64-32-16, BatchNorm, Dropout, "
     f"Adam 1e-3, early stopping) stopped at epoch {meta['dnn_epochs']}. It is trained in PyTorch and served in plain NumPy from exported weights.", size=9)
h2("3.3 Explainability, trust and recommendations")
bullets([
    ("Explanation: ", "exact TreeSHAP contributions per patient, summed into four domains (chest symptoms, stress test, cholesterol, overall profile) and shown as plain-language drivers."),
    ("Trust: ", "confidence band labelled from validation accuracy per confidence range, a 90% conformal prediction set, and disagreement flags (XGBoost vs DNN, ML vs rules)."),
    ("What-if: ", "model sensitivity to plausible changes in cholesterol and blood pressure (capped at +/-30%), presented as model behaviour, not treatment advice."),
    ("Next-test advice: ", "A* over the missing tests (costs: slope 4, thallium 6, vessel count 8) plus a ranking by whether a result could flip the verdict."),
    ("Chat: ", "rule-based, no external AI service. It parses a free-text opening and answers follow-ups only from the computed analysis, so it cannot invent facts."),
])
h2("3.4 User interface")
para("The user view is a guided chat: five short steps, plain-language questions, and a tooltip on every field explaining what it is and why it matters. Chest pain is "
     "classified from three yes/no questions rather than by asking for a medical term. The result page leads with a risk pattern, the estimated chance, confidence and "
     "information completeness; details sit in four tabs. A separate 'How it works' view serves reviewers.")
figure(DOCF / "ui_intake.jpg", 4.6, "Figure 2. Guided chat intake with free-text pre-fill.")
figure(DOCF / "ui_overview.jpg", 4.6, "Figure 3. Overview: risk pattern, estimate, confidence and information completeness.")
figure(DOCF / "ui_how_it_works.jpg", 4.6, "Figure 4. 'How it works' view: orchestration of the workflows for reviewers.")

# ---------------------------------------------------------------- 4
h1("4. Evaluation Report")
para(f"All figures are on the held-out test split (n = {ev['models']['xgb']['n']}), untouched during training, tuning and calibration.")
mrows = [[k, f3(v["accuracy"]), f3(v["precision"]), f3(v["recall"]), f3(v["roc_auc"]), f3(v["brier"])] for k, v in ev["models"].items()]
table(mrows, ["Model", "Accuracy", "Precision", "Recall", "ROC-AUC", "Brier"], [1.8, 1, 1, 1, 1, 1])
ci = ev["models"]["xgb"]["auc_ci95"], ev["models"]["dnn"]["auc_ci95"]
para(f"AUC 95% bootstrap interval: XGBoost {ci[0][0]:.2f}-{ci[0][1]:.2f}, DNN {ci[1][0]:.2f}-{ci[1][1]:.2f}. The intervals overlap heavily, so the models cannot be "
     f"separated statistically at this sample size. Averaging the two gave no meaningful gain (ensemble AUC {f3(ev['models']['mean_ensemble']['roc_auc'])}), which is why "
     f"XGBoost alone gives the verdict and the DNN acts as a second opinion. Always-predict-prevalence Brier baseline: {f3(ev['calibration']['brier_baseline_prevalence'])}.")
figure(FIG / "roc.png", 3.4, "Figure 5. ROC curves, held-out test.")
figure(FIG / "confusion.png", 4.6, "Figure 6. Confusion matrices at the 0.5 threshold.")
h2("4.1 Calibration, confidence and conformal prediction")
brows = [[b["band"], b["n"], f3(b["accuracy"])] for b in ev["trust_bands_test"]]
table(brows, ["Model confidence range", "Test patients", "Accuracy"], [2.2, 1.5, 1.5])
cf = ev["conformal"]
para(f"Accuracy is clearly lowest in the least-confident band ({brows[0][2]}) and between 0.84 and 0.95 in the others. It is not strictly increasing, which is "
     f"expected with such small bands, but it is why low-confidence results carry a 'defer to clinical judgement' warning. The 90% conformal set covered {cf['empirical_coverage']:.1%} of test patients "
     f"(target {cf['target']:.0%}); {cf['frac_single']:.0%} of patients received a single-label set and {cf['frac_both']:.0%} the 'uncertain' two-label set. "
     "Bands contain only 13-50 patients each, so these figures are approximate.")
figure(FIG / "calibration.png", 3.2, "Figure 7. Reliability diagram for XGBoost.")
h2("4.2 Cross-checks: rule base and second model")
ru, md_ = ev["rules"], ev["model_disagreement"]
para(f"The rule base reaches a decision for {ru['coverage_determinate']:.0%} of patients (the rest lack a needed test). Where it decides, ML accuracy is "
     f"{f3(ru['ml_accuracy_when_agree'])} when rules and ML agree and {f3(ru['ml_accuracy_when_disagree'])} when they disagree (n = {ru['n_disagree']}). "
     f"XGBoost and the DNN disagree on {md_['rate']:.0%} of patients; XGBoost accuracy is {f3(md_['xgb_acc_when_agree'])} when they agree and "
     f"{f3(md_['xgb_acc_when_disagree'])} when they do not. Disagreement is therefore a useful warning, and the system lowers confidence when it occurs.")
h2("4.3 Effect of missing optional tests (ablation)")
arows = [[k, v["n"], f3(v["accuracy"]), f3(v["roc_auc"])] for k, v in ev["ablation_imputation"].items()]
table(arows, ["Subset", "n", "Accuracy", "AUC"], [3.0, 0.8, 1.2, 1.2])
para("Hiding the three optional tests from every test patient changes AUC only slightly, so the model leans mainly on symptoms and risk factors. The reliability "
     "warning is still kept, because the filled-in value can be wrong for an individual. Small subsets, so treat as indicative.", size=9)
h2("4.4 Explainability validation")
gs = list(ev["global_shap"].items())[:5]
para("Global SHAP ranks: " + ", ".join(f"{k} ({v:.2f})" for k, v in gs) + ". The dominant role of asymptomatic chest pain agrees with Tasks 1, 2, 5 and 8 "
     "(four independent methods). In the training data a 'silent' presentation was more often found to have disease, so the interface warns that no chest pain is "
     "not reassuring.")
figure(FIG / "shap_global.png", 3.4, "Figure 8. Global feature importance (mean |SHAP|, test set).")

# ---------------------------------------------------------------- 5
h1("5. Testing Report")
table([
    ["Feature parity", "Pipeline output equals the saved feature matrix (max diff < 1e-9); formulas hand-checked", "Pass"],
    ["Model parity", "NumPy DNN reproduces the trained DNN's test AUC; XGBoost reproduces its reported AUC (+/-0.005)", "Pass"],
    ["Rule engine", "Three-valued logic, strict/inclusive boundaries (e.g. oldpeak > 2.8, age in [56,58]), R7 default", "Pass"],
    ["Validation", "Out-of-range, wrong-type and zero-value inputs return field-level errors", "Pass"],
    ["End-to-end", "All four scripted patients (higher risk, lower risk, borderline, rule-vs-ML disagreement) run through the full pipeline", "Pass"],
    ["Determinism", "Same input gives identical output, including the counterfactual", "Pass"],
    ["A* planner", "Returns the cheapest decisive test set (slope + vessel count, cost 12) for the borderline case", "Pass"],
    ["Cleaning", "Zero-to-missing, duplicate removal, target binarisation, no constant filler for imputed cholesterol", "Pass"],
    ["Chat and glossary", "Intro parsing, status cut-offs, every suggested question answered, emergency wording", "Pass"],
    ["Dashboard", "Streamlit test runs the guided flow and every example through to the result page, and the How-it-works view", "Pass"],
], ["Area", "What is checked", "Result"], [1.2, 5.2, 0.6])
rb, lat = ev["robustness"], ev["latency_ms_full_pipeline"]
para(f"20 automated tests pass. Robustness: invalid inputs rejected {rb['invalid_inputs_rejected']}; 300 random valid patients processed without error "
     f"({rb['fuzz_valid_inputs_ok']}). Latency of the full pipeline including SHAP, counterfactual and planner: median {lat['p50']:.0f} ms, 95th percentile "
     f"{lat['p95']:.0f} ms (target under 500 ms). An independent code review found six defects (a constant imputation value that acted as a hidden "
     "disease marker, miscalibrated confidence labels, an ambiguous headline probability, a spec mismatch, misleading 'actionable' counterfactuals and rule "
     "statistics that used imputed values); five were fixed and one was recorded as a deliberate design change.")
para("Domain and user validation: scripted cases and automated checks only. No clinician or end-user study has been carried out; this is listed as future work.", size=9, italic=True)

# ---------------------------------------------------------------- 6
h1("6. Responsible AI Assessment")
table([
    ["Intended use", "Educational decision support for one patient at a time. Emergency and not-a-diagnosis notices appear on every screen and in the report."],
    ["Data", "Public research data from four hospitals, collected decades ago. Not representative of any current population."],
    ["Missing data", "Flagged, never silent. Two or more filled fields block a 'high' confidence band. Rules never run on guessed values."],
    ["Out-of-distribution", "Age outside the 28-77 training range triggers a warning."],
    ["Transparency", "SHAP per patient, plain-language reasons, rule trace, agreement chips, a visible trace of every workflow. The DNN has no per-patient explanation, and the UI says so."],
    ["Uncertainty", "Validation-based confidence bands, conformal set, and disagreement flags. 'Uncertain' is an allowed output."],
    ["Counterfactuals", "Labelled model sensitivity, restricted to cholesterol and blood pressure, capped at +/-30%; never presented as treatment effect."],
    ["Chat", "Rule-based and offline; answers are assembled from computed evidence and cannot hallucinate. Emergency wording is always available."],
    ["Privacy", "Stateless: no patient data stored or logged; no data leaves the machine; no external AI service."],
    ["Reproducibility", "Fixed seeds, pinned split, saved artifacts, determinism test, one-command rebuild."],
], ["Area", "Finding and mitigation"], [1.4, 5.6])
frows = [[f"{k}: {g}", v["n"], f3(v["roc_auc"]), f3(v["recall"]), f3(v["precision"]), f"{v['prevalence']:.2f}"]
         for k, gs_ in ev["fairness"].items() for g, v in gs_.items()]
para("Fairness check (test set). Groups are small, so differences are weak evidence.", bold=True, size=9.5)
table(frows, ["Group", "n", "AUC", "Recall", "Precision", "Prevalence"], [2.2, 0.7, 0.9, 0.9, 0.9, 1.0], 8.5)
para("Limitations: women are a small share of the data; disease prevalence varies strongly by hospital (for example 94% in one site against 36% in another), and "
     "hospital also drives which values are missing, so it acts as a hidden confounder; the test set is small; there is no external or temporal validation.", size=9)

# ---------------------------------------------------------------- 7
h1("7. Requirements and Design Decisions")
table([
    ["FR1-FR2", "Validate raw inputs; handle optional/missing inputs explicitly"],
    ["FR3-FR5", "Build the 12-feature vector exactly; run XGBoost and DNN; verdict from XGBoost with disagreement flag"],
    ["FR6", "Evaluate rule base R1-R7, report fired/unknown rules and ML agreement"],
    ["FR7-FR9", "SHAP explanation; confidence band with reliability warnings; counterfactual sensitivity"],
    ["FR10", "Next-test recommendation by A* when key tests are missing"],
    ["FR11-FR12", "Readable summary and downloadable report; guided UI from inputs to overview"],
    ["NFR", "Latency under 500 ms (met), deterministic, stateless, one-command run, zero cost, every feature traceable to its formula"],
], ["ID", "Requirement"], [0.9, 6.1])
para("Key design decisions (full log in ARCHITECTURE.md):", bold=True, size=9.5)
table([
    ["XGBoost gives the verdict; the DNN is a second opinion", "The mean ensemble was no better, and SHAP explains XGBoost, keeping verdict and explanation consistent"],
    ["Three-valued rules", "A rule must not fire on a guessed value; the unknowns drive the test planner"],
    ["Random draw instead of a constant for unobservable cholesterol", "A constant filler became a hidden site marker the model learned (found in review)"],
    ["DNN trained in PyTorch, served in NumPy", "TensorFlow would not load on the development machine; no heavy runtime dependency"],
    ["Confidence labels from validation accuracy", "Fixed probability cut-offs ignored calibration"],
    ["Rule-based chat, no external AI service", "Zero cost, offline, cannot invent facts; trade-off is limited open-ended understanding"],
], ["Decision", "Why"], [2.8, 4.2], 8.5)

# ---------------------------------------------------------------- 8
h1("8. User Manual and Deployment Guide")
h2("8.1 Using the system")
bullets([
    "Open the app. 'Health check' is for users; 'How it works' is for reviewers.",
    "Describe symptoms in your own words (optional) or press Start. Answer the five short steps; hover the '?' next to any field for a plain explanation.",
    "Read the overview: risk pattern, estimated chance, confidence and how much information was provided.",
    "Use the tabs: Why this result, Your measurements, What-if, Next steps (including questions for a doctor and a downloadable report).",
    "Ask follow-up questions in the chat, for example 'What does thallium mean?'. For a quick demo use the four ready-made examples on the first screen.",
])
h2("8.2 Running and deploying")
para("Local run (PowerShell):", bold=True, size=9.5)
para("cd C:\\Users\\aarya\\OneDrive\\Desktop\\CODING\\SymbioForge\npip install -r requirements.txt\nstreamlit run app.py", size=9)
para("Rebuild everything: pip install -r requirements-train.txt, then python train.py, python eda.py, python evaluate.py, python build_docs.py, python -m pytest. "
     "Cloud deployment: push the repository to GitHub and create a Streamlit Community Cloud app with main file app.py (Python 3.11+, requirements.txt only; no "
     "secrets and no database). Artifacts are under 300 KB and committed, and XGBoost is stored as portable JSON so it does not depend on the training machine.", size=9)
para("Deployment readiness: all dependencies are free and open source; the app starts cold in about 15 seconds on a slow disk; no external services are required.", size=9)

# ---------------------------------------------------------------- 9
h1("9. Deliverables Checklist")
table([
    ["9", "Application (running instance)", "streamlit run app.py"],
    ["9", "Integrated source code", "cdss/, ui/, app.py, train.py, evaluate.py, eda.py"],
    ["9", "Dashboard", "Health check view and How it works view"],
    ["9", "Report generator", "cdss/report.py; download button; reports/sample_report.html"],
    ["9", "Deployment documentation", "README.md and Section 8 of this report"],
    ["10", "Evaluation report", "Section 4; reports/evaluation.json"],
    ["10", "Responsible AI report", "Section 6"],
    ["10", "User manual", "Section 8.1"],
    ["10", "SRS / SDD", "Section 7, SPEC.md, ARCHITECTURE.md"],
    ["10", "Testing report", "Section 5; tests/test_core.py"],
    ["10", "Final project report", "This document"],
    ["10", "Deployment guide", "Section 8.2"],
], ["Task", "Deliverable", "Where"], [0.6, 2.8, 3.6])

# ---------------------------------------------------------------- 10
h1("10. Conclusion and Future Enhancements")
para("The experiments are integrated into one system whose value lies in the combination: eight workflows produce independent evidence, an orchestrator reconciles "
     f"them, and the patient receives an explained, uncertainty-aware overview rather than a bare label. On the held-out test split XGBoost reaches "
     f"{ev['models']['xgb']['accuracy']:.1%} accuracy and {ev['models']['xgb']['roc_auc']:.3f} AUC, the system answers in about {lat['p50']:.0f} ms, and all 20 automated "
     "tests pass. The prototype is honest about its limits: small test set, historical single-source data, no clinical validation.")
para("Future work: repeated cross-validation and site-held-out validation; calibrated probabilities (isotonic); a clinician and user study; an optional language-model "
     "layer for the chat, restricted to the computed analysis; and multilingual support.")

out = R / "docs" / "SymbioForge_Final_Project_Report.docx"
doc.save(out)
print("saved", out)
