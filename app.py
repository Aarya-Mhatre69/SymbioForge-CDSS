"""SymbioForge CDSS. Run: streamlit run app.py
Query params: ?from=clinic (arrived from the demo clinic site: show a link back), ?view=clinician (demo triage queue), ?embed=true (compact layout)."""
from html import escape

import streamlit as st

from ui import assess, clinician, mechanics, style
from ui.config import CLINIC_NAME, CLINIC_URL

st.set_page_config(page_title="SymbioForge", page_icon="❤", layout="wide")
style.inject()
if st.query_params.get("embed") == "true":
    st.markdown("<style>.block-container{padding-top:1.2rem!important}</style>", unsafe_allow_html=True)

if st.query_params.get("from") == "clinic":
    st.session_state["from_clinic"] = True  # remembered for the session; the target URL is fixed in ui/config.py
back = (f"<a class='backlink' href='{CLINIC_URL}/index.html'>&larr; Back to {escape(CLINIC_NAME)}</a>" if st.session_state.get("from_clinic") else "")
st.markdown(f"<div class='brand'><b>SymbioForge</b><span>Heart-health decision support</span><span style='margin-left:auto'>{back}</span></div>", unsafe_allow_html=True)
if st.query_params.get("view") == "clinician":
    clinician.render()
else:
    view = st.segmented_control("View", ["Health check", "How it works"], default="Health check", key="view", label_visibility="collapsed")
    if view == "How it works":
        mechanics.render()
    else:
        assess.render()
