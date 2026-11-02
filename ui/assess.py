"""User-facing flow: safety check -> stage 1 (home checkup, tier only) -> optional stage 2 (stress-test add-on, full analysis) -> chat."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from pydantic import ValidationError

from cdss import DISCLAIMER, chat
from cdss import queue as clinic_queue
from cdss.engine import DEMO_PATIENTS
from cdss.glossary import FIELDS, TEST_PLAIN
from cdss.orchestrator import assess
from cdss.report import build_report, build_report_stage1
from cdss.schemas import PatientInput, field_errors
from cdss.stage1 import Stage1Input, get_stage1
from . import style
from .config import CLINIC_NAME, CLINIC_URL

RAW = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal"]
S1 = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg"]
NA = "Not done / don't know"
ECG_OPTS = {"Normal": "normal", "Thickened heart wall (LV hypertrophy)": "lv hypertrophy", "Strain pattern (ST-T abnormality)": "st-t abnormality"}
EXAMPLES = [("Higher-risk case", "Clear present", 2), ("Lower-risk case", "Clear absent", 2), ("Borderline, tests missing", "Borderline / missing data", 2),
            ("Rules vs model disagree", "Rule vs ML disagreement", 2), ("Home-checkup example", "Borderline / missing data", 1)]
SAFETY = ("Not a diagnosis. If you have severe or worsening chest pain, pain spreading to the arm or jaw, breathlessness or fainting, "
          "call your local emergency number now.")
RED_FLAGS = ["Severe chest pain or pressure right now, or in the last hour",
             "Chest pain spreading to the arm, jaw, neck or back",
             "Shortness of breath at rest",
             "Fainting or feeling about to faint",
             "Cold sweat or nausea together with chest discomfort"]


def tip(field: str) -> str:
    f = FIELDS[field]
    return f"{f['what']}\n\n{f['matters']}\n\nTypical range: {f['normal']}"


def _init(key, val):
    st.session_state.setdefault(key, val)


def derive_cp(pain: bool, pressure: bool, exertion: bool, relief: bool) -> str:
    if not pain:
        return "asymptomatic"
    n = int(pressure) + int(exertion) + int(relief)
    return "typical angina" if n == 3 else "atypical angina" if n == 2 else "non-anginal"


def pain_from_cp(cp: str) -> dict:
    return {"asymptomatic": dict(pain=False, pressure=False, exertion=False, relief=False),
            "typical angina": dict(pain=True, pressure=True, exertion=True, relief=True),
            "atypical angina": dict(pain=True, pressure=True, exertion=True, relief=False),
            "non-anginal": dict(pain=True, pressure=True, exertion=False, relief=False)}[cp]


# ---------------------------------------------------------------- intake steps
def r_safety(P):
    if st.session_state.get("pending"):
        st.caption("The same safety check applies before an example is run.")
    st.caption("Please tick anything that is happening to you now. If none apply, leave all boxes empty and continue.")
    for k, txt in enumerate(RED_FLAGS):
        st.checkbox(txt, key=f"w_rf{k}")


def c_safety():
    return {"redflag": [t for k, t in enumerate(RED_FLAGS) if st.session_state.get(f"w_rf{k}")]}


def s_safety(a):
    return "None of those are happening."


def r_about(P):
    _init("w_age", int(P.get("age", 55)))
    _init("w_sex", P.get("sex", "male"))
    st.number_input("Age", 18, 100, key="w_age", help=tip("age"))
    st.radio("Sex", ["male", "female"], key="w_sex", horizontal=True, format_func=str.capitalize, help=tip("sex"))


def c_about():
    return {"age": int(st.session_state.w_age), "sex": st.session_state.w_sex}


def s_about(a):
    return f"{a['age']} years old, {a['sex']}."


def r_pain(P):
    pr = P.get("pain_raw") or {}
    _init("w_pain", "Yes, I have" if pr.get("pain") else "No chest pain")
    st.radio("Have you had chest pain or discomfort recently?", ["No chest pain", "Yes, I have"], key="w_pain", horizontal=True, help=tip("cp"))
    if st.session_state.w_pain == "Yes, I have":
        yn = ["Yes", "No"]
        for k, txt in (("pressure", "Is it a pressure, tightness or squeezing feeling behind the breastbone?"),
                       ("exertion", "Is it brought on by physical effort or emotional stress?"),
                       ("relief", "Does it ease within a few minutes of resting?")):
            _init(f"w_{k}", "Yes" if pr.get(k) else "No")
            st.radio(txt, yn, key=f"w_{k}", horizontal=True)
        cp = derive_cp(True, *(st.session_state[f"w_{k}"] == "Yes" for k in ("pressure", "exertion", "relief")))
        st.caption(f"Doctors would call this pattern **{cp}**.")


def c_pain():
    pain = st.session_state.w_pain == "Yes, I have"
    d = {k: (pain and st.session_state.get(f"w_{k}") == "Yes") for k in ("pressure", "exertion", "relief")}
    return {"cp": derive_cp(pain, **d), "pain_raw": {"pain": pain, **d}}


def s_pain(a):
    return "No chest pain." if not a["pain_raw"]["pain"] else f"Yes, chest discomfort: pattern **{a['cp']}**."


def r_vitals(P):
    _init("w_trestbps", int(P.get("trestbps", 130)))
    _init("w_chol", int(P.get("chol", 210)))
    _init("w_fbs", bool(P.get("fbs", False)))
    st.caption("Typical values are pre-filled. **Replace them with your own** or the result will not mean much. "
               "A home-visit technician can measure all of these.")
    c1, c2 = st.columns(2)
    c1.number_input("Resting blood pressure (mmHg, top number)", 80, 220, key="w_trestbps", help=tip("trestbps"))
    c2.number_input("Total cholesterol (mg/dl)", 100, 600, key="w_chol", help=tip("chol"))
    st.checkbox("My fasting blood sugar is above 120 mg/dl", key="w_fbs", help=tip("fbs"))


def c_vitals():
    return {"trestbps": int(st.session_state.w_trestbps), "chol": int(st.session_state.w_chol), "fbs": bool(st.session_state.w_fbs)}


def s_vitals(a):
    return f"Blood pressure {a['trestbps']} mmHg, cholesterol {a['chol']} mg/dl, high fasting sugar: {'yes' if a['fbs'] else 'no'}."


def r_ecg(P):
    inv = {v: k for k, v in ECG_OPTS.items()}
    _init("w_restecg", inv[P.get("restecg", "normal")])
    st.caption("A resting ECG takes a few minutes and a home-visit technician can record it.")
    st.selectbox("Resting ECG result", list(ECG_OPTS), key="w_restecg", help=tip("restecg"))


def c_ecg():
    return {"restecg": ECG_OPTS[st.session_state.w_restecg]}


def s_ecg(a):
    return f"Resting ECG: {a['restecg']}."


def r_stress(P):
    _init("w_thalch", int(P.get("thalch", 150)))
    _init("w_oldpeak", float(P.get("oldpeak", 1.0)))
    _init("w_exang", "Yes" if P.get("exang") else "No")
    st.caption("From a treadmill stress test done at a clinic. Pre-filled with typical values: replace them with your results.")
    c1, c2 = st.columns(2)
    c1.number_input("Highest heart rate reached (bpm)", 60, 220, key="w_thalch", help=tip("thalch"))
    c2.number_input("ST depression (mm)", -3.0, 7.0, step=0.1, key="w_oldpeak", help=tip("oldpeak"))
    st.radio("Did exercise bring on chest pain during the test?", ["No", "Yes"], key="w_exang", horizontal=True, help=tip("exang"))


def c_stress():
    return {"thalch": int(st.session_state.w_thalch), "oldpeak": float(st.session_state.w_oldpeak), "exang": st.session_state.w_exang == "Yes"}


def s_stress(a):
    return f"Peak heart rate {a['thalch']} bpm, ST depression {a['oldpeak']} mm, exercise chest pain: {'yes' if a['exang'] else 'no'}."


def r_optional(P):
    for f, opts in (("slope", ["upsloping", "flat", "downsloping"]), ("ca", [0, 1, 2, 3]),
                    ("thal", ["normal", "fixed defect", "reversable defect"])):
        _init(f"w_{f}", NA if P.get(f) is None else P.get(f))
        st.selectbox(FIELDS[f]["label"] + f" ({FIELDS[f]['short'].lower()})", [NA] + opts, key=f"w_{f}", help=tip(f))


def c_optional():
    return {f: (None if st.session_state[f"w_{f}"] == NA else st.session_state[f"w_{f}"]) for f in ("slope", "ca", "thal")}


def s_optional(a):
    n = sum(a.get(f) is None for f in ("slope", "ca", "thal"))
    return "None of these tests done." if n == 3 else f"{3 - n} of 3 specialised tests provided."


STEPS = [
    ("Safety check", "First, a quick safety check. Is any of this happening to you right now?", r_safety, c_safety, s_safety),
    ("About you", "Let's start with the basics. How old are you, and what is your sex?", r_about, c_about, s_about),
    ("Chest symptoms", "Now about any chest discomfort. The answers to three short questions are how doctors classify chest pain.", r_pain, c_pain, s_pain),
    ("Check-up numbers", "A few numbers from a recent check-up or blood test.", r_vitals, c_vitals, s_vitals),
    ("Resting ECG", "Last for the home-checkup stage: your resting ECG result.", r_ecg, c_ecg, s_ecg),
    ("Stress test", "Now the optional second stage. If you have had a treadmill stress test, add the results for a fuller analysis.", r_stress, c_stress, s_stress),
    ("Specialised tests", "Last, three specialised tests. Most people have not had these, and that is fine: choose 'Not done' and I will tell you which one would help most.",
     r_optional, c_optional, s_optional),
]
N1 = 5  # steps 0..4 are stage 1


# ---------------------------------------------------------------- state helpers
def reset():
    for k in list(st.session_state):
        if k.startswith("w_") or k in ("stage", "step", "ans", "parsed", "ctx", "chat", "understood", "sent_id", "screened", "pending"):
            del st.session_state[k]


def _clear_widgets():
    for k in list(st.session_state):
        if k.startswith("w_"):
            del st.session_state[k]


def _finish(ctx):
    st.session_state.ctx = ctx
    st.session_state.stage = "result"
    st.session_state.chat = []
    st.session_state.pop("sent_id", None)
    _clear_widgets()
    st.rerun()


def go_result1(raw: dict):
    if not st.session_state.get("screened"):  # no model result without the red-flag screen
        st.session_state.step, st.session_state.stage = 1, "intake"
        st.rerun()
    try:
        inp = Stage1Input(**{k: raw.get(k) for k in S1})
    except ValidationError as ex:
        st.error("Please check these answers:\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in field_errors(ex).items()))
        return
    _finish(get_stage1().assess(inp))


def go_result2(raw: dict):
    if not st.session_state.get("screened"):  # no model result without the red-flag screen
        st.session_state.step, st.session_state.stage = 1, "intake"
        st.rerun()
    try:
        patient = PatientInput(**{k: raw.get(k) for k in RAW})
    except ValidationError as ex:
        st.error("Please check these answers:\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in field_errors(ex).items()))
        return
    ctx = assess(patient)
    ctx["stage"] = 2
    _finish(ctx)


def load_example(name: str, stage: int):
    """Examples go through the safety screen too: remember the choice and ask the red-flag question first."""
    ss = st.session_state
    ss.pending = (name, stage)
    ss.step, ss.screened = 1, False
    _clear_widgets()
    st.rerun()


def _run_example(name: str, stage: int):
    d = dict(DEMO_PATIENTS[name])
    d["pain_raw"] = pain_from_cp(d["cp"])
    st.session_state.ans = {**d, "redflag": []}
    (go_result1 if stage == 1 else go_result2)(d)


# ---------------------------------------------------------------- views
def emergency(flags):
    st.markdown("<div class='safety' style='font-size:1.05rem;padding:1.1rem 1.2rem'><b>Please get urgent medical help now.</b><br>"
                "Call your local emergency number or go to the nearest emergency department. Do not wait for an online tool.<br><br>"
                + "You told us: " + "; ".join(f.lower() for f in flags) + ".</div>", unsafe_allow_html=True)
    st.caption("SymbioForge stopped here on purpose: with these symptoms, a screening tool is not the right place to be.")
    if st.button("Start over"):
        reset()
        st.rerun()


def intake():
    ss = st.session_state
    ss.setdefault("step", 0)
    ss.setdefault("ans", {})
    ss.setdefault("parsed", {})
    step = ss.step
    if step == 0:
        with st.chat_message("assistant"):
            st.markdown("Hello, I'm **SymbioForge**. I'll ask a few simple questions about your heart health, run them through several AI workflows "
                        "working together, and give you an overview of what stands out and what to do next. "
                        "The first stage needs only things a home-visit technician can measure. If you also have stress-test results, you can add them afterwards.")
            st.markdown("**In your own words:** what's been going on? (optional, for example *\"I'm 58, a man, and I get a tight chest when I climb stairs\"*)")
            txt = st.text_area("Your description", key="w_intro", label_visibility="collapsed", height=90,
                               placeholder="Describe your symptoms, or leave this blank and just answer the questions")
            if st.button("Start", type="primary"):
                parsed = chat.parse_intro(txt or "")
                ss.parsed = parsed
                if parsed:
                    pr = {k: parsed[k] for k in ("pain", "pressure", "exertion", "relief") if k in parsed}
                    ss.ans = {**{k: parsed[k] for k in ("age", "sex") if k in parsed}, **({"pain_raw": pr} if "pain" in pr else {})}
                    ss.understood = chat.understood(parsed)
                ss.step = 1
                _clear_widgets()
                st.rerun()
        st.markdown("<div class='steplabel' style='margin-top:1rem'>Or try a ready-made example:</div>", unsafe_allow_html=True)
        cols = st.columns(len(EXAMPLES))
        for c, (label, key, stg) in zip(cols, EXAMPLES):
            if c.button(label, use_container_width=True, key=f"ex_{key}_{stg}"):
                load_example(key, stg)
        return

    i = step - 1
    if i < N1:
        style.stepper(i, N1, STEPS[i][0])
    else:
        style.stepper(i - N1, len(STEPS) - N1, "Add stress-test results: " + STEPS[i][0])
    if step == 2 and ss.get("understood"):
        with st.chat_message("assistant"):
            st.markdown(f"Thanks. Here is what I picked up: **{ss.understood}**. I've pre-filled those answers; please check them.")
    for j in range(i):  # answered steps as chat history
        with st.chat_message("assistant"):
            st.markdown(STEPS[j][1])
        with st.chat_message("user"):
            st.markdown(STEPS[j][4](ss.ans))
    label, q, render, collect, _ = STEPS[i]
    with st.chat_message("assistant"):
        st.markdown(q)
        render(ss.ans)
    last1, last = i == N1 - 1, i == len(STEPS) - 1
    b1, b2, _ = st.columns([1, 2, 5])
    if b1.button("Back"):
        if i == N1:
            ss.stage = "result"
        else:
            ss.step -= 1
        _clear_widgets()
        st.rerun()
    if b2.button("Analyse my answers" if last else "See my result" if last1 else "Continue", type="primary"):
        ss.ans = {**ss.ans, **collect()}
        if i == 0:
            if ss.ans["redflag"]:
                ss.stage, ss.screened = "emergency", False
                ss.pop("pending", None)
            else:
                ss.screened = True
                pending = ss.pop("pending", None)
                if pending:
                    _run_example(*pending)
                ss.step += 1
            _clear_widgets()
            st.rerun()
        elif last1:
            go_result1(ss.ans)
        elif last:
            go_result2(ss.ans)
        else:
            ss.step += 1
            _clear_widgets()
            st.rerun()


def _domain_chart(domains):
    ds = list(reversed(domains))
    fig = go.Figure(go.Bar(x=[d["impact"] for d in ds], y=[d["domain"] for d in ds], orientation="h",
                           marker_color=["#c0504d" if d["impact"] > 0 else "#2b6cb0" for d in ds]))
    fig.update_layout(height=210, margin=dict(l=0, r=10, t=6, b=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(title="lowers the estimate  ←   →  raises the estimate", zeroline=True, zerolinecolor="#999", showgrid=False, showticklabels=False),
                      yaxis=dict(showgrid=False))
    return fig


def _curve_chart(c, field):
    f = FIELDS[field]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c["x"], y=c["p"], mode="lines", line=dict(color="#1f5f8b", width=2.5, shape="hv"), name="estimate"))
    fig.add_hline(y=0.5, line_dash="dot", line_color="#999")
    cur = min(range(len(c["x"])), key=lambda k: abs(c["x"][k] - c["current"]))
    fig.add_trace(go.Scatter(x=[c["current"]], y=[c["p"][cur]], mode="markers", marker=dict(size=12, color="#c0504d"), name="you"))
    fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                      xaxis_title=f"{f['label']} ({f['unit']})" if f["unit"] else f["label"], yaxis=dict(title="Model-estimated chance", tickformat=".0%", range=[0, 1]))
    return fig


def _hero(ctx):
    r, a = ctx["result"], ctx["analysis"]
    t, c, comp = a["tier_info"], r["confidence"], a["completeness"]
    band_w = {"high": 90, "moderate": 60, "low": 30}[c["band"]]
    if ctx.get("stage") == 1 or not a["reliable"]:
        big = f"<div class='big' style='font-size:1.9rem'>{t['label']}</div>"
        sub = ("Screening result from your history, blood pressure, blood tests and resting ECG" if ctx.get("stage") == 1
               else "Some values are outside what the model can judge")
        big = big.replace("</div>", f"<small>{sub}</small></div>")
        tier_line = f"<div class='tier'>{t['icon']} {'Stage 1: home checkup' if ctx.get('stage') == 1 else 'Full analysis'}</div>"
    else:
        big = f"<div class='big'>≈ {a['p']:.0%}<small>model-estimated chance of heart disease</small></div>"
        tier_line = f"<div class='tier'>{t['icon']} {t['label']}</div>"
    st.markdown(f"""<div class="hero" style="--c:{t['color']}">{tier_line}{big}<p>{a['headline']}</p>
<div class="meters">
<div class="meter"><label>How sure is the model?</label><div class="bar"><div style="width:{band_w}%"></div></div><div class="v">{c['band'].capitalize()} confidence</div></div>
<div class="meter"><label>Information provided</label><div class="bar"><div style="width:{comp['given'] / comp['of'] * 100:.0f}%"></div></div><div class="v">{comp['given']} of {comp['of']} inputs</div></div>
</div></div>""", unsafe_allow_html=True)
    if ctx.get("stage") == 1:
        st.caption("Stage 1 shows a tier, not a percentage: the model learned from patients already referred for heart tests, "
                   "so a percentage would not be valid for the general public.")
    if c["warnings"]:
        with st.expander(f"Things to keep in mind ({len(c['warnings'])})"):
            for w in c["warnings"]:
                st.markdown(f"- {w}")


def _tab_why(ctx):
    a = ctx["analysis"]
    st.markdown("**Which parts of your profile moved the estimate**")
    st.plotly_chart(_domain_chart(a["domains"]), width="stretch", config={"displayModeBar": False})
    st.markdown("**The biggest individual influences**")
    html = "".join(f"<div class='drv'><div class='dot' style='background:{'#c0504d' if d['shap'] > 0 else '#2b6cb0'}'></div>"
                   f"<div><b>{d['label']}</b>: {d['value']}<br><span class='note' style='color:#667085;font-size:.85rem'>"
                   f"This {d['direction']} the estimate.</span></div></div>" for d in a["drivers"])
    st.markdown(html, unsafe_allow_html=True)
    if ctx.get("stage") != 1 and not a["reliable"]:
        st.info("The model opinions are hidden because some of your values are outside the range the models can judge reliably.")
    if ctx.get("stage") != 1 and a["reliable"]:
        so = a["second_opinions"]
        rules_txt = {"present": "points to disease", "absent": "finds no sign", "undetermined": "cannot decide with the tests provided"}[so["rules"]["verdict"]]
        st.markdown("**Three independent opinions**")
        st.markdown(f"""<div class='opin'>
<div class='card'><h4>{so['ml']['name']}</h4><div class='big'>{so['ml']['p']:.0%}</div><div class='note'>leans {so['ml']['says']} risk</div></div>
<div class='card'><h4>{so['dl']['name']}</h4><div class='big'>{so['dl']['p']:.0%}</div><div class='note'>leans {so['dl']['says']} risk</div></div>
<div class='card'><h4>{so['rules']['name']}</h4><div class='big' style='font-size:1.05rem'>{rules_txt.capitalize()}</div><div class='note'>{('rules ' + ', '.join(so['rules']['fired']) + ' fired') if so['rules']['fired'] else 'no rule fired'}</div></div></div>""",
                    unsafe_allow_html=True)
        st.caption("When all three agree the result is more dependable. When they disagree, the tool says so and lowers its confidence.")


def _tab_measurements(ctx):
    st.caption("Each card compares one of your answers with common reference ranges. Open 'What is this?' for a plain explanation.")
    cols = st.columns(2)
    for k, cd in enumerate(ctx["analysis"]["cards"]):
        with cols[k % 2]:
            st.markdown(f"<div class='card'><h4>{cd['label']}{style.chip(cd['level'])}</h4><div class='val'>{cd['value']}</div>"
                        f"<div class='note'>{cd['note']}</div></div>", unsafe_allow_html=True)
            with st.expander("What is this?"):
                st.markdown(f"{cd['what']}\n\n**Why it matters:** {cd['matters']}\n\n**Typical range:** {cd['normal']}")


def _tab_whatif(ctx):
    a, r = ctx["analysis"], ctx["result"]
    if not a["reliable"]:
        st.info("What-if curves are hidden because some values are outside what the model can judge reliably.")
        return
    st.markdown("**See how the estimate would move**")
    st.caption("Everything else is held at your answers. This shows how the model behaves; it is not a prediction of what treatment would do.")
    pick = st.radio("Change", ["chol", "trestbps", "thalch"], horizontal=True, label_visibility="collapsed", format_func=lambda f: FIELDS[f]["label"], key="w_curve")
    st.plotly_chart(_curve_chart(a["curves"][pick], pick), width="stretch", config={"displayModeBar": False})
    st.caption("The red dot is you; the dotted line is the 50% mark.")
    if a["curves"][pick]["caution"]:
        st.warning(a["curves"][pick]["caution"], icon="⚠")
    cf = r["counterfactual"]
    if cf["actionable"]:
        ch = "; ".join(f"{FIELDS[k]['label'].lower()} {v['from']} to {v['to']}" for k, v in cf["actionable"]["changes"].items())
        st.info(f"In the model, {ch} would bring the estimate to about {cf['actionable']['new_probability']:.0%}.")
    elif not cf["found"]:
        st.info("Your estimate is robust: no small, realistic change in these measurements flips it.")


def _send_to_clinic(ctx):
    ss = st.session_state
    with st.expander("Send this overview to the clinic (demo)"):
        if ss.get("sent_id"):
            st.success("Sent. A clinician will see it in the queue (demo, fictional data only).")
            return
        st.caption("Demo only: no real clinic receives anything. Use fictional details.")
        name = st.text_input("Name or nickname (optional)", key="w_clinic_name")
        ok = st.checkbox("I agree to share this overview and my answers with the clinic.", key="w_clinic_consent")
        if st.button("Send to clinic", disabled=not ok):
            ss.sent_id = clinic_queue.submit(ctx, name, ok)
            st.rerun()


def _tab_next(ctx):
    ss = st.session_state
    a, r = ctx["analysis"], ctx["result"]
    nx = a["next"]
    st.markdown("**What to do**")
    for s in nx["steps"]:
        st.markdown(f"- {s}")
    st.markdown("- Share this overview with your doctor; the questions below can start the conversation.")
    if ctx.get("stage") == 1:
        st.markdown("**Add stress-test results for a fuller analysis**")
        st.caption("If you have had a treadmill stress test, you can add its results; the full analysis then uses more models and shows a probability.")
        if st.button("I have stress-test results", type="primary"):
            ss.step = N1 + 1
            ss.stage = "intake"
            _clear_widgets()
            st.rerun()
    elif nx["tests"]:
        st.markdown("**Tests that could add clarity** (ranked)")
        for tst in nx["tests"]:
            flip = " This result could change the outcome." if tst["can_flip"] else ""
            st.markdown(f"- {tst['name']}.{flip}")
        if nx["plan"].get("tests"):
            st.caption("To settle the expert rules whatever the outcome, the cheapest route is: " + ", ".join(TEST_PLAIN[x].split(" (")[0] for x in nx["plan"]["tests"]) + ".")
    st.markdown("**Questions to ask your doctor**")
    for qn in a["questions"]:
        st.markdown(f"- {qn}")
    if ss.get("from_clinic"):
        b1, b2 = st.columns(2)
        b1.link_button("Book a home checkup", f"{CLINIC_URL}/book.html", type="primary", use_container_width=True)
        b2.link_button(f"Back to {CLINIC_NAME}", f"{CLINIC_URL}/index.html", use_container_width=True)
    d1, d2 = st.columns(2)
    html = build_report_stage1(ctx) if ctx.get("stage") == 1 else build_report(r, a)
    d1.download_button("Download report", html, "symbioforge_report.html", "text/html", type="primary")
    if d2.button("Start over"):
        reset()
        st.rerun()
    _send_to_clinic(ctx)


def _chat(ctx):
    ss = st.session_state
    r = ctx["result"]
    st.divider()
    st.subheader("Ask SymbioForge")
    st.caption("Ask about your result, a term, or what to do next. Answers come from your own analysis, not from guesswork.")
    ss.setdefault("chat", [])
    for role, msg in ss.chat:
        with st.chat_message(role):
            st.markdown(msg)
    sg = st.columns(len(chat.SUGGESTIONS))
    asked = None
    for k, (col, s) in enumerate(zip(sg, chat.SUGGESTIONS)):
        if col.button(s, key=f"sg{k}", use_container_width=True):
            asked = s
    typed = st.chat_input("Ask a question, for example: what does ECG mean?")
    asked = asked or typed
    if asked:
        ss.chat += [("user", asked), ("assistant", chat.answer(asked, ctx))]
        st.rerun()
    with st.expander(f"How this was produced: {len(r['trace'])} AI workflows ran"):
        total = sum(x["ms"] for x in r["trace"])
        st.markdown("".join(f"<div class='agent'><span><b>{x['agent']}</b> <small>{x['task']}</small></span><span>{x['note']}</span></div>"
                            for x in r["trace"]), unsafe_allow_html=True)
        st.caption(f"Total {total:.0f} ms. See the 'How it works' view for the architecture.")
    if st.button("Edit my answers"):
        raw = {**ctx["patient"], "pain_raw": pain_from_cp(ctx["patient"]["cp"])}
        reset()
        ss.ans, ss.step, ss.stage = raw, 1, "intake"
        st.rerun()


def result():
    ctx = st.session_state.ctx
    stage1 = ctx.get("stage") == 1
    st.markdown(f"<div class='safety'>{SAFETY}</div>", unsafe_allow_html=True)
    _hero(ctx)
    names = ["Why this result", "Your measurements", "Next steps"] if stage1 else ["Why this result", "Your measurements", "What-if", "Next steps"]
    tabs = st.tabs(names)
    with tabs[0]:
        _tab_why(ctx)
    with tabs[1]:
        _tab_measurements(ctx)
    if stage1:
        with tabs[2]:
            _tab_next(ctx)
    else:
        with tabs[2]:
            _tab_whatif(ctx)
        with tabs[3]:
            _tab_next(ctx)
    _chat(ctx)


def render():
    ss = st.session_state
    ss.setdefault("stage", "intake")
    if ss.stage == "emergency":
        emergency(ss.ans.get("redflag", []))
    elif ss.stage == "result" and "ctx" in ss:
        result()
    else:
        intake()
    st.markdown(f"<div class='disc'>{DISCLAIMER}</div>", unsafe_allow_html=True)
