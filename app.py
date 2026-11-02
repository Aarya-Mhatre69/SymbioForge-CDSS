"""SymbioForge CDSS. Run: streamlit run app.py
Query params: ?embed=true (used when embedded in the demo clinic site), ?view=clinician (demo triage queue)."""
import streamlit as st

from ui import assess, clinician, mechanics, style

st.set_page_config(page_title="SymbioForge", page_icon="❤", layout="wide")
style.inject()
if st.query_params.get("embed") == "true":
    st.markdown("<style>.block-container{padding-top:1.2rem!important}</style>", unsafe_allow_html=True)

st.markdown("<div class='brand'><b>SymbioForge</b><span>Heart-health decision support</span></div>", unsafe_allow_html=True)
if st.query_params.get("view") == "clinician":
    clinician.render()
else:
    view = st.segmented_control("View", ["Health check", "How it works"], default="Health check", key="view", label_visibility="collapsed")
    if view == "How it works":
        mechanics.render()
    else:
        assess.render()
