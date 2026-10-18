"""SymbioForge CDSS dashboard. Run: streamlit run app.py"""
from __future__ import annotations

import json

import plotly.graph_objects as go
import streamlit as st
from pydantic import ValidationError

from cdss import DISCLAIMER, REPORTS
from cdss.engine import DEMO_PATIENTS, LABELS, get_engine
from cdss.report import build_report
from cdss.rules import RULES
from cdss.schemas import PatientInput, field_errors

st.set_page_config(page_title="SymbioForge CDSS", page_icon="❤", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.6rem;max-width:1200px}
.verdict{border-radius:10px;padding:1rem 1.2rem;border:1px solid var(--bd);background:var(--bg)}
.verdict h2{margin:0;font-size:1.6rem}.verdict p{margin:.2rem 0 0;opacity:.85}
.pill{display:inline-block;padding:.15rem .6rem;border-radius:99px;font-size:.8rem;margin-right:.3rem;border:1px solid #8884}
.ok{background:#2f855a22}.bad{background:#c5303022}.unk{background:#dd6b2022}
.disc{font-size:.8rem;opacity:.7;margin-top:1rem}
</style>""", unsafe_allow_html=True)

E = get_engine()
CP = ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
NA = "not available"
DEFAULTS = {"age": 55, "sex": "male", "cp": "asymptomatic", "trestbps": 130, "chol": 240, "fbs": False,
            "restecg": "normal", "thalch": 140, "exang": False, "oldpeak": 1.0, "slope": NA, "ca": NA, "thal": NA}


def load_demo():
    d = DEMO_PATIENTS[st.session_state.demo]
    for k, v in d.items():
        st.session_state[k] = NA if v is None else v


for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# ---------------- sidebar: inputs ----------------
with st.sidebar:
    st.title("Patient")
    st.selectbox("Load demo patient", list(DEMO_PATIENTS), key="demo", on_change=load_demo, index=None,
                 placeholder="Choose a scripted case")
    st.number_input("Age", 18, 100, key="age")
    st.radio("Sex", ["male", "female"], key="sex", horizontal=True)
    st.selectbox("Chest pain type", CP, key="cp")
    c1, c2 = st.columns(2)
    c1.number_input("Resting BP (mmHg)", 80, 220, key="trestbps")
    c2.number_input("Cholesterol (mg/dl)", 100, 600, key="chol")
    c1.number_input("Max heart rate", 60, 220, key="thalch")
    c2.number_input("ST depression", -3.0, 7.0, step=0.1, key="oldpeak")
    st.selectbox("Resting ECG", ["normal", "lv hypertrophy", "st-t abnormality"], key="restecg")
    c1.checkbox("Fasting sugar > 120", key="fbs")
    c2.checkbox("Exercise angina", key="exang")
    st.caption("Optional tests. Leave unavailable if not done; the system will say how that affects reliability.")
    st.selectbox("ST slope", [NA, "upsloping", "flat", "downsloping"], key="slope")
    st.selectbox("Major vessels (fluoroscopy)", [NA, 0, 1, 2, 3], key="ca")
    st.selectbox("Thalassemia", [NA, "normal", "fixed defect", "reversable defect"], key="thal")

raw = {k: (None if st.session_state[k] == NA else st.session_state[k]) for k in DEFAULTS}
st.title("SymbioForge CDSS")
st.caption("Cardiovascular decision support · " + DISCLAIMER)

try:
    patient = PatientInput(**raw)
except ValidationError as ex:
    st.error("Please fix these fields:\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in field_errors(ex).items()))
    st.stop()


@st.cache_data(show_spinner=False)
def run(p: str):
    return E.predict(PatientInput(**json.loads(p)))


r = run(json.dumps(patient.model_dump(), sort_keys=True))
c, rl = r["confidence"], r["rules"]
present = r["verdict"] == "present"
col = "#c53030" if present else "#2f855a"
st.markdown(f"""<div class="verdict" style="--bd:{col}66;--bg:{col}14">
<h2 style="color:{col}">Disease {r['verdict']} · P(disease) = {r['models']['xgb']:.0%}</h2>
<p>Confidence <b>{c['band']}</b> · prediction set at {c['conformal_level']:.0%}: <b>{' / '.join(c['conformal_set'])}</b></p></div>""",
            unsafe_allow_html=True)
agree = {True: ("rules agree", "ok"), False: ("rules DISAGREE", "bad"), None: ("rules undetermined", "unk")}[rl["agrees_with_ml"]]
st.markdown(f"""<div style="margin:.6rem 0"><span class="pill {agree[1]}">{agree[0]}</span>
<span class="pill {'bad' if r['model_disagreement'] else 'ok'}">{'XGB and DNN disagree' if r['model_disagreement'] else 'XGB and DNN agree'}</span>
<span class="pill {'unk' if r['imputed'] else 'ok'}">{len(r['imputed'])} of 3 optional fields imputed</span></div>""", unsafe_allow_html=True)
if c["warnings"]:
    st.warning("\n".join(f"- {w}" for w in c["warnings"]), icon="⚠")

tabs = st.tabs(["Prediction", "Explanation", "Confidence", "Next test", "Report", "Data (EDA)", "System & responsible AI"])

with tabs[0]:
    a, b = st.columns([1, 1])
    with a:
        st.subheader("Models")
        m1, m2 = st.columns(2)
        m1.metric("XGBoost (primary)", f"{r['models']['xgb']:.0%}")
        m2.metric("DNN (second opinion)", f"{r['models']['dnn']:.0%}")
        st.write(r["summary"])
    with b:
        st.subheader("Rule base cross-check")
        for k, txt in RULES.items():
            if k == "R7":
                continue
            s = rl["state"][k]
            st.markdown(f"{'🟢 fired' if s else '⚪ not fired' if s is False else '🟠 unknown'} · **{k}** · {txt.split(' -> ')[0]}")
        st.caption(f"Rule verdict: **{rl['verdict']}**. Unknown rules need tests that were not provided; they are never evaluated on guessed values.")
    st.subheader("Engineered features")
    st.dataframe({"feature": list(r["features"]), "value": [round(v, 3) for v in r["features"].values()]}, hide_index=True, width="stretch")

with tabs[1]:
    st.subheader("Why this estimate (SHAP, XGBoost)")
    ex = list(reversed(r["explanation"]))
    fig = go.Figure(go.Bar(x=[x["shap"] for x in ex], y=[f"{x['label']} = {x['value']:.3g}" for x in ex], orientation="h",
                           marker_color=["#c53030" if x["shap"] > 0 else "#2b6cb0" for x in ex]))
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="push toward present  →  (log-odds)",
                      template="plotly_white", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")
    st.caption("The DNN has no per-patient explanation; explanations describe the XGBoost model only.")
    st.subheader("What would flip the verdict")
    cf = r["counterfactual"]
    if not cf["found"]:
        st.info(cf["note"])
    else:
        for title, x in (("Model sensitivity: modifiable factors only (±30%)", cf["actionable"]), ("Model sensitivity: smallest change of any inputs", cf["any"])):
            if x:
                st.markdown(f"**{title}** → new P = {x['new_probability']:.0%}")
                for f, v in x["changes"].items():
                    st.markdown(f"- {LABELS.get(f, f)}: {v['from']} → {v['to']}")
            else:
                st.markdown(f"**{title}**: none found")
        st.caption("Hypothetical model sensitivity, not clinical advice or a treatment effect. Test results (ECG, oldpeak) are not things a patient can change.")

with tabs[2]:
    st.subheader("How much to trust this")
    t1, t2, t3 = st.columns(3)
    t1.metric("Model confidence", f"{c['probability_confidence']:.0%}")
    t2.metric("Validation accuracy at this confidence",
              "n/a" if c["validation_accuracy_in_band"] is None else f"{c['validation_accuracy_in_band']:.0%}",
              help=f"{c['validation_n_in_band']} validation patients in this band")
    t3.metric("Band", c["band"])
    st.markdown("**Trust bands (validation set)**")
    st.dataframe([{"confidence": f"{b['lo']:.2f}-{min(b['hi'], 1):.2f}", "patients": b["n"],
                   "accuracy": None if b["accuracy"] is None else round(b["accuracy"], 3)} for b in E.meta["bands"]],
                 hide_index=True, width="stretch")
    st.caption(f"Conformal prediction: the set contains the true class about {c['conformal_level']:.0%} of the time on average. "
               "'present / absent' means the model cannot separate the cases and the patient needs human review.")

with tabs[3]:
    st.subheader("Next-test advisor")
    adv = r["advice"]
    if not adv["needed"]:
        st.success("All optional tests were provided.")
    else:
        pl = adv["plan"]
        if pl["tests"]:
            st.markdown(f"**Task 3 A\\* plan:** to make the rule base decisive whatever the outcomes, order **{' + '.join(pl['tests'])}** "
                        f"(cost {pl['cost']}, {pl['expanded']} states expanded).")
        else:
            st.markdown("The rule base is already decisive; further tests only refine the ML estimate.")
        st.dataframe([{"test": x["test"], "cost": x["cost"], "P(present) min": round(x["p_min"], 2),
                       "P(present) max": round(x["p_max"], 2), "can flip verdict": x["can_flip_verdict"],
                       "spread per cost": round(x["value_per_cost"], 3)} for x in adv["ranking"]], hide_index=True, width="stretch")
        st.caption("Ranked by whether a result could flip the verdict, then by probability spread per unit of test cost.")

with tabs[4]:
    html = build_report(r)
    st.download_button("Download report (HTML, print to PDF)", html, "cdss_report.html", "text/html", type="primary")
    st.components.v1.html(html, height=700, scrolling=True)

with tabs[5]:
    eda = REPORTS / "EDA_report.md"
    if eda.exists():
        st.subheader("Exploratory data analysis")
        g1, g2 = st.columns(2)
        g1.image(str(REPORTS / "figures" / "eda_correlation.png"), caption="Correlation of the 12 model features")
        g2.image(str(REPORTS / "figures" / "eda_missing_by_site.png"), caption="Missingness depends on the hospital site")
        st.image(str(REPORTS / "figures" / "eda_target_by_category.png"))
        with st.expander("Full EDA report"):
            st.markdown("\n".join(l for l in eda.read_text(encoding="utf-8").splitlines() if not l.startswith("![")))
    else:
        st.info("Run `python eda.py` to generate the EDA report.")

with tabs[6]:
    ev_path = REPORTS / "evaluation.json"
    st.subheader("Evaluation (held-out test set)")
    if ev_path.exists():
        ev = json.loads(ev_path.read_text())
        rows = [{"model": k, **{m: round(v[m], 3) for m in ("accuracy", "precision", "recall", "roc_auc", "brier")}}
                for k, v in ev["models"].items()]
        st.dataframe(rows, hide_index=True, width="stretch")
        g1, g2 = st.columns(2)
        g1.image(str(REPORTS / "figures" / "roc.png"))
        g2.image(str(REPORTS / "figures" / "calibration.png"))
        st.caption(f"Test n = {ev['models']['xgb']['n']}; AUC 95% CI {ev['models']['xgb']['auc_ci95'][0]:.2f}-{ev['models']['xgb']['auc_ci95'][1]:.2f}. "
                   "Small test set: differences between models are within noise.")
    else:
        st.info("Run `python evaluate.py` to generate evaluation results.")
    st.subheader("Known limitations")
    st.markdown("""
- Trained on a public research dataset (four hospitals, 1980s–90s), not on current or local patients; not clinically validated.
- Test set is small (138), so every metric carries wide uncertainty.
- Female patients and some sites are under-represented; see the fairness table in the evaluation report.
- `ca`/`thal`/`slope` are often missing in training; when absent they are filled with the most common value and flagged.
- Counterfactuals are model sensitivity, not treatment advice.
""")
    st.caption(f"Primary model: {E.meta['primary_model']} · artifacts: xgb.json, dnn.json · deterministic, stateless, no patient data stored.")

st.markdown(f"<div class='disc'>{DISCLAIMER}</div>", unsafe_allow_html=True)
