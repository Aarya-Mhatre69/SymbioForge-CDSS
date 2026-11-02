"""Demo clinician triage queue. No login, fictional data only."""
from __future__ import annotations

import datetime as dt
import hmac
import os
from html import escape

import streamlit as st

from cdss import queue as q
from cdss.engine import DEMO_PATIENTS
from cdss.orchestrator import assess
from cdss.stage1 import TIERS, Stage1Input, get_stage1

S1 = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg"]


def _seed():
    s1 = get_stage1()
    for name, key, stage in (("Sample patient A", "Clear present", 2), ("Sample patient B", "Borderline / missing data", 1),
                             ("Sample patient C", "Rule vs ML disagreement", 2), ("Sample patient D", "Clear absent", 1)):
        d = DEMO_PATIENTS[key]
        ctx = s1.assess(Stage1Input(**{k: d[k] for k in S1})) if stage == 1 else {**assess_full(d), "stage": 2}
        q.submit(ctx, name, True)


def assess_full(d):
    from cdss.schemas import PatientInput
    return assess(PatientInput(**d))


def _gate() -> bool:
    """Demo passcode gate (env SYMBIOFORGE_CLINICIAN_PASSCODE, default 'demo'). A real deployment needs proper authentication."""
    if st.session_state.get("clin_ok"):
        return True
    st.markdown("### Clinician queue (demo)")
    st.caption("Clinicians only. Demo passcode: `demo` (set SYMBIOFORGE_CLINICIAN_PASSCODE to change it). This is not real authentication.")
    pw = st.text_input("Passcode", type="password", key="clin_pw")
    if st.button("Open queue"):
        if hmac.compare_digest(pw.encode(), os.environ.get("SYMBIOFORGE_CLINICIAN_PASSCODE", "demo").encode()):
            st.session_state.clin_ok = True
            st.rerun()
        else:
            st.error("Wrong passcode.")
    return False


def render():
    if not _gate():
        return
    st.markdown("### Clinician queue (demo)")
    st.markdown("<div class='safety'>Demo only: no login, fictional data. In a real deployment this page would need authentication, encryption, "
                "consent management and an audit log.</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 2, 3])
    if c1.button("Load sample cases"):
        _seed()
        st.rerun()
    if c2.button("Clear queue"):
        q.clear()
        st.rerun()
    show = c3.radio("Show", ["New", "All"], horizontal=True, label_visibility="collapsed")
    rows = q.cases("new" if show == "New" else None)
    if not rows:
        st.info("The queue is empty. Patients who choose 'Send to clinic' after their assessment appear here, most urgent first.")
        return
    st.caption(f"{len(rows)} case(s), sorted by urgency (a model or rule disagreement moves a case up one tier), then newest first.")
    for r in rows:
        t = TIERS[r["tier"]]
        when = dt.datetime.fromtimestamp(r["created"]).strftime("%d %b %H:%M")
        st.markdown(f"<div class='card' style='border-left:5px solid {t['color']}'><h4>{escape(r['name'])} "
                    f"<span class='chip' style='background:{t['color']}22;color:{t['color']}'>{t['label']}</span>"
                    f"{'' if r['status'] == 'new' else '<span class=chip>reviewed</span>'}</h4>"
                    f"<div class='note'>Stage {r['stage']} · {when}</div></div>", unsafe_allow_html=True)
        with st.expander("Open case"):
            st.markdown(r["headline"])
            if r["alerts"]:
                st.warning("Needs review: " + " ".join(r["alerts"]), icon="⚠")
            if r["flags"]:
                st.markdown("**Outside reference range:**\n" + "\n".join(f"- {f}" for f in r["flags"]))
            st.markdown("**Main influences:**\n" + "\n".join(f"- {d}" for d in r["drivers"]))
            st.markdown("**Suggested next steps:**\n" + "\n".join(f"- {s}" for s in r["steps"]))
            st.caption("Inputs: " + ", ".join(f"{k}={v}" for k, v in r["inputs"].items() if v is not None))
            if r["status"] == "new" and st.button("Mark reviewed", key=f"rev{r['id']}"):
                q.mark_reviewed(r["id"])
                st.rerun()
