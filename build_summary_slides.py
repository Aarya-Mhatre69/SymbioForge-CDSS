"""Generate docs/SymbioForge_Summary.pptx (5 slides) from the real artifacts and reports. Run after train.py and evaluate.py."""
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

R = Path(__file__).parent
FIG, DOCF = R / "reports" / "figures", R / "docs" / "figures"
ev = json.loads((R / "reports/evaluation.json").read_text())
meta = json.loads((R / "artifacts/meta.json").read_text())
comp = pd.read_csv(R / "reports/Task6_model_comparison.csv")

BG, CARD, LINE = RGBColor(0xFA, 0xF9, 0xF7), RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0xE6, 0xE1, 0xD8)
INK, MUTED, ACC = RGBColor(0x1F, 0x29, 0x33), RGBColor(0x66, 0x70, 0x85), RGBColor(0x1F, 0x5F, 0x8B)
SOFT, GREEN, AMBER, RED = RGBColor(0xF1, 0xEE, 0xE9), RGBColor(0x2F, 0x85, 0x5A), RGBColor(0xB7, 0x79, 0x1F), RGBColor(0xC5, 0x30, 0x30)
WHITE = RGBColor(255, 255, 255)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
f3 = lambda x: f"{x:.3f}"
pct = lambda x: f"{x:.0%}"


def slide():
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    return s


def text(s, x, y, w, h, items, anchor=MSO_ANCHOR.TOP):
    """items: list of (text, size, bold, color) or (text, size, bold, color, space_after)."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    for i, it in enumerate(items):
        t, size, bold, color = it[:4]
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = t
        r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), bold, color, "Calibri"
        p.space_after = Pt(it[4] if len(it) > 4 else 3)
    return tb


def card(s, x, y, w, h, fill=CARD, line=LINE, radius=0.05):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = radius
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = line
    sh.line.width = Pt(0.75)
    sh.shadow.inherit = False
    return sh


def pic(s, path, x, y, w, h):
    """Fit an image inside the box, centred, keeping its aspect."""
    iw, ih = Image.open(path).size
    sc = min(w / iw, h / ih)
    pw, ph = iw * sc, ih * sc
    s.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def header(s, label, title, n):
    text(s, 0.5, 0.28, 8, 0.3, [(label.upper(), 10.5, True, ACC)])
    text(s, 0.5, 0.55, 12.3, 0.7, [(title, 28, True, INK)])
    text(s, 12.2, 7.08, 0.7, 0.3, [(f"{n} / 5", 9.5, False, MUTED)])
    text(s, 0.5, 7.08, 11, 0.3, [("SymbioForge  ·  Decision support only, not a diagnosis. Research prototype, not clinically validated.", 9, False, MUTED)])


def table(s, x, y, w, rows, header_row, widths, size=9.5, row_h=0.27):
    shape = s.shapes.add_table(len(rows) + 1, len(header_row), Inches(x), Inches(y), Inches(w), Inches(row_h * (len(rows) + 1)))
    tbl = shape.table
    tbl.first_row = True
    for j, wd in enumerate(widths):
        tbl.columns[j].width = Inches(wd)
    for i in range(len(rows) + 1):
        tbl.rows[i].height = Inches(row_h)
        for j in range(len(header_row)):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.015)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            val = header_row[j] if i == 0 else rows[i - 1][j]
            c.fill.fore_color.rgb = ACC if i == 0 else (CARD if i % 2 else RGBColor(0xF3, 0xF6, 0xFA))
            p = c.text_frame.paragraphs[0]
            p.text = ""
            r = p.add_run()
            r.text = str(val)
            r.font.size, r.font.name = Pt(size), "Calibri"
            r.font.bold = i == 0
            r.font.color.rgb = WHITE if i == 0 else INK
    return shape


def kpi(s, x, y, w, h, big, label, color=ACC):
    card(s, x, y, w, h)
    text(s, x + 0.12, y + 0.1, w - 0.24, h * 0.55, [(big, 30, True, color)], MSO_ANCHOR.MIDDLE)
    text(s, x + 0.12, y + h * 0.58, w - 0.24, h * 0.4, [(label, 10.5, False, MUTED)])


# ================================================================= slide 1: start
s = slide()
text(s, 0.5, 0.3, 9, 0.3, [("CELBC710 · Computational Intelligence Laboratory · Experiments 9 and 10", 10.5, False, MUTED)])
text(s, 0.5, 0.72, 8, 0.9, [("SymbioForge", 50, True, INK)])
text(s, 0.5, 1.62, 12, 0.5, [("Heart-health decision support: eight AI workflows, one orchestrator, one explained overview", 20, False, ACC)])

card(s, 0.5, 2.35, 6.35, 3.2)
text(s, 0.72, 2.45, 5.9, 3.1, [
    ("The problem", 14, True, ACC, 4),
    ("A bare yes/no from a single model is hard to trust and impossible to act on.", 12.5, False, INK, 2),
    ("People do not know what their own numbers mean.", 12.5, False, INK, 2),
    ("A health tool that cannot say \"I am unsure\" is unsafe.", 12.5, False, INK, 9),
    ("Our answer", 14, True, ACC, 4),
    ("Run independent workflows (ML, deep learning, expert rules, SHAP, trust, what-if, test planner) and reconcile them.", 12.5, False, INK, 2),
    ("Return an explained, uncertainty-aware overview and a next step, not a label.", 12.5, False, INK, 2),
    ("Reach it through a clinic website: one link, one flow.", 12.5, False, INK, 2),
])
kx, ky, kw, kh = 7.15, 2.35, 2.75, 1.5
kpi(s, kx, ky, kw, kh, f3(ev["models"]["xgb"]["roc_auc"]), f"ROC-AUC, held-out test (n = {ev['models']['xgb']['n']})")
kpi(s, kx + kw + 0.2, ky, kw, kh, "10", "AI workflows run and traced for every patient")
kpi(s, kx, ky + kh + 0.2, kw, kh, f"{ev['latency_ms_full_pipeline']['p50']:.0f} ms", "median time for the full pipeline incl. explanation")
kpi(s, kx + kw + 0.2, ky + kh + 0.2, kw, kh, "25", "automated tests passing")

text(s, 0.5, 5.72, 12, 0.3, [("What a visitor experiences", 12, True, MUTED)])
steps = ["Clinic website", "Guided chat", "Overview + explanation", "Next steps + chat", "Book a home checkup"]
cw = 12.35 / len(steps)
for i, st_ in enumerate(steps):
    sh = s.shapes.add_shape(MSO_SHAPE.PENTAGON if i == 0 else MSO_SHAPE.CHEVRON, Inches(0.5 + i * cw), Inches(6.05), Inches(cw + 0.12), Inches(0.75))
    sh.fill.solid()
    sh.fill.fore_color.rgb = ACC if i % 2 == 0 else RGBColor(0x2B, 0x73, 0xA3)
    sh.line.fill.background()
    sh.shadow.inherit = False
    tf = sh.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = st_
    r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(12.5), True, WHITE, "Calibri"
text(s, 12.2, 7.08, 0.7, 0.3, [("1 / 5", 9.5, False, MUTED)])
text(s, 0.5, 7.08, 11, 0.3, [("Decision support only, not a diagnosis. Research prototype trained on a public dataset; not clinically validated.", 9, False, MUTED)])

# ================================================================= slide 2: how it works
s = slide()
header(s, "How it works", "Eight workflows, one orchestrator", 2)
card(s, 0.5, 1.4, 7.35, 4.35)
pic(s, DOCF / "architecture.png", 0.6, 1.5, 7.15, 4.1)
rows = [["Intake agent", "T4", "validate; flag and fill missing tests"],
        ["Feature agent", "T5", "12 features from one formula module"],
        ["Risk model", "T6", "XGBoost probability (the verdict)"],
        ["Second opinion", "T7", "DNN; flags disagreement"],
        ["Clinical rules", "T1-2", "R1-R7: fired / not fired / unknown"],
        ["Test planner", "T3", "A* picks the cheapest decisive test"],
        ["Explainer", "T8", "exact SHAP contributions"],
        ["Trust + what-if", "T8", "bands, conformal set, sensitivity"],
        ["Synthesizer", "T9", "overview, doctor questions, report"]]
table(s, 8.05, 1.4, 4.8, rows, ["Workflow", "From", "What it does"], [1.25, 0.55, 3.0], 9.5, 0.4)
cards = [("918 patients", f"4 hospitals, {meta['dataset']['rows_clean']} after cleaning, 12 engineered features"),
         ("Clean evaluation", f"70/15/15 split (seed 42); imputer, scaler and models fit on train only"),
         ("No train/serve skew", "one feature function; parity test < 1e-9 against the saved matrix"),
         ("Honest about gaps", "missing tests are flagged, confidence is capped, rules never run on guesses")]
cw = (12.35 - 3 * 0.18) / 4
for i, (h_, b_) in enumerate(cards):
    x = 0.5 + i * (cw + 0.18)
    card(s, x, 5.95, cw, 1.0)
    text(s, x + 0.1, 6.0, cw - 0.2, 0.95, [(h_, 12.5, True, ACC, 2), (b_, 10, False, INK)])

# ================================================================= slide 3: evidence
s = slide()
header(s, "Evidence", "Accurate, and honest about uncertainty", 3)
figs = [(FIG / "roc.png", f"ROC curves, held-out test (AUC XGBoost {f3(ev['models']['xgb']['roc_auc'])}, DNN {f3(ev['models']['dnn']['roc_auc'])})", 0.5, 4.2),
        (FIG / "calibration.png", f"Reliability: predicted vs observed (Brier {f3(ev['calibration']['brier_xgb'])})", 4.85, 3.5),
        (FIG / "shap_global.png", "What drives predictions (mean |SHAP|, test set)", 8.5, 4.35)]
for path, cap, x, w in figs:
    card(s, x, 1.35, w, 3.05)
    pic(s, path, x + 0.08, 1.4, w - 0.16, 2.6)
    text(s, x + 0.08, 4.02, w - 0.16, 0.36, [(cap, 9, False, MUTED)])
rows = [[r.model.replace("XGBoost(tuned)", "XGBoost (tuned)").replace("XGBoost(context)", "XGBoost (task 6)"), f"{r.cv_auc_mean:.3f}", f3(r.test_accuracy), f3(r.test_roc_auc)]
        for r in comp.itertuples()]
text(s, 0.5, 4.55, 5.8, 0.3, [("Seven models compared (5-fold CV on train, then held-out test)", 11, True, ACC)])
table(s, 0.5, 4.85, 5.75, rows, ["Model", "CV AUC", "Test acc.", "Test AUC"], [2.3, 1.1, 1.15, 1.2], 9.5, 0.25)
ru, md_, cf = ev["rules"], ev["model_disagreement"], ev["conformal"]
kw = 2.1
kpi(s, 6.5, 4.6, kw, 1.15, pct(ru["ml_accuracy_when_agree"]), "ML accuracy when rules agree", GREEN)
kpi(s, 6.5 + kw + 0.12, 4.6, kw, 1.15, pct(md_["rate"]), "XGBoost and DNN disagree; confidence is lowered", AMBER)
kpi(s, 6.5 + 2 * (kw + 0.12), 4.6, kw, 1.15, f"{cf['empirical_coverage']:.0%}", f"conformal coverage (target {cf['target']:.0%}); {pct(cf['frac_both'])} flagged 'uncertain'")
ci = ev["models"]["xgb"]["auc_ci95"]
card(s, 6.5, 5.9, 3 * kw + 0.24, 1.05, fill=SOFT)
text(s, 6.62, 5.95, 3 * kw, 1.0, [
    ("What we conclude", 11.5, True, ACC, 2),
    (f"AUC 95% interval {ci[0]:.2f}-{ci[1]:.2f}: XGBoost and the DNN cannot be separated at n = 138, and averaging gave no gain. So XGBoost gives the verdict, the DNN is a second opinion, "
     "and every disagreement is shown to the user.", 10, False, INK),
])

# ================================================================= slide 4: product
s = slide()
header(s, "The product", "From a clinic website to a plain-language overview", 4)
shots = [(DOCF / "ui_intake.jpg", "Guided chat: plain-language questions, a tooltip on every field, free-text pre-fill"),
         (DOCF / "ui_overview.jpg", "Overview: risk pattern, estimate, confidence and how much information was given"),
         (DOCF / "ui_how_it_works.jpg", "How it works view for reviewers: live trace of the workflows")]
sw = (12.35 - 2 * 0.2) / 3
for i, (path, cap) in enumerate(shots):
    x = 0.5 + i * (sw + 0.2)
    card(s, x, 1.35, sw, 3.3)
    pic(s, path, x + 0.08, 1.42, sw - 0.16, 2.78)
    text(s, x + 0.1, 4.2, sw - 0.2, 0.42, [(cap, 9.5, False, MUTED)])
card(s, 0.5, 4.85, 6.1, 2.1)
text(s, 0.68, 4.9, 5.8, 2.05, [
    ("What the user gets", 13, True, ACC, 3),
    ("Plain-language questions; chest pain classified from three yes/no questions.", 11, False, INK, 1),
    ("Four tabs: why this result, their measurements vs reference ranges, what-if curves, next steps.", 11, False, INK, 1),
    ("Three independent opinions (ML, deep learning, expert rules).", 11, False, INK, 1),
    ("Ranked next tests, questions for the doctor, a downloadable report.", 11, False, INK, 1),
    ("A chat that answers only from the computed analysis: no external AI service, no invented facts.", 11, False, INK, 1),
])
card(s, 6.8, 4.85, 6.05, 2.1, fill=SOFT)
text(s, 6.98, 4.9, 5.75, 2.05, [
    ("One public link, free to run", 13, True, ACC, 3),
    ("Clinic website (GitHub Pages) is the front door; its button redirects into the Streamlit Cloud app, which links back and offers booking.", 11, False, INK, 3),
    ("Stateless: nothing stored, no API keys, no data leaves the page.", 11, False, INK, 3),
    ("Local demo with one command: python run_demo.py", 11, False, INK, 3),
    ("Live app: symbioforge.streamlit.app", 11, True, ACC, 1),
])

# ================================================================= slide 5: responsible + conclusion
s = slide()
header(s, "Responsible by design", "What we guard against, what we learned, where it goes next", 5)
rows = [["Not a diagnosis", "Notice on every screen and report; emergency wording always available"],
        ["Missing tests", "Flagged, never silent; 2+ missing blocks 'high' confidence; rules never run on guesses"],
        ["Disagreement", "ML vs DNN vs rules shown; any conflict lowers confidence"],
        ["Explainable", "Exact SHAP per patient; DNN limits stated in the UI"],
        ["Uncertainty", "Validation-based bands and a 90% conformal set; 'uncertain' is allowed"],
        ["Counterfactuals", "Labelled model sensitivity, never treatment advice"],
        ["Privacy", "Stateless; no patient data stored or sent anywhere"]]
table(s, 0.5, 1.4, 6.3, rows, ["Area", "How it is handled"], [1.45, 4.85], 9.5, 0.36)
fr = [[f"Sex: {g}", v["n"], f3(v["roc_auc"]), f3(v["recall"])] for g, v in ev["fairness"]["sex"].items()]
fr += [[f"Site: {g}", v["n"], f3(v["roc_auc"]), f3(v["recall"])] for g, v in list(ev["fairness"]["site"].items())[:4]]
text(s, 0.5, 4.5, 6.3, 0.3, [("Fairness check (test set; groups are small, so weak evidence)", 11, True, ACC)])
table(s, 0.5, 4.8, 6.3, fr, ["Group", "n", "AUC", "Recall"], [2.4, 1.0, 1.4, 1.5], 9, 0.235)

card(s, 7.0, 1.4, 5.85, 1.8)
text(s, 7.15, 1.45, 5.55, 1.75, [
    ("What independent review caught", 12.5, True, ACC, 3),
    ("Two review passes found defects that tests had missed. The most telling: a constant used to fill missing cholesterol acted as a hidden hospital marker, "
     "and the model learned it. Removing it lowered test AUC slightly (0.899 to 0.896), which is the honest number.", 10.5, False, INK),
])
card(s, 7.0, 3.35, 5.85, 1.3, fill=SOFT)
pic(s, FIG / "eda_missing_by_site.png", 7.05, 3.38, 2.6, 1.24)
text(s, 9.7, 3.4, 3.1, 1.2, [("Why we are careful", 11, True, ACC, 2), ("Missingness and disease rate both depend on the hospital, so the data hide a confounder.", 10, False, INK)])
card(s, 7.0, 4.8, 5.85, 2.15, fill=ACC, line=ACC)
text(s, 7.2, 4.86, 5.5, 2.1, [
    ("Conclusion", 14, True, WHITE, 3),
    ("SymbioForge turns a classifier into an explained, uncertainty-aware overview by reconciling eight workflows, and it is reached through a clinic front door "
     "with a single public link.", 11, False, WHITE, 4),
    ("Honest limits: small test set (138), historical single-source data, no clinical validation.", 10.5, False, WHITE, 4),
    ("Next: clinician and user study, site-held-out validation, calibrated probabilities, an optional language-model layer limited to computed evidence.", 10.5, False, WHITE, 1),
])

out = R / "docs" / "SymbioForge_Summary.pptx"
prs.save(out)
print("saved", out)
