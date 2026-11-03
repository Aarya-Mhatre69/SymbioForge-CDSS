"""SymbioForge CDSS. Run: streamlit run app.py
Optional query param ?from=clinic (arrived from the demo clinic site): shows a link back to the clinic."""
from html import escape

import streamlit as st

from ui import assess, mechanics, style
from ui.config import CLINIC_NAME, clinic_url

st.set_page_config(page_title="SymbioForge", page_icon="❤", layout="wide")
style.inject()

if st.query_params.get("from") == "clinic":
    st.session_state["from_clinic"] = True  # remembered for the session; the target URL is fixed in ui/config.py
CLINIC = clinic_url()
back = (f"<a class='backlink' href='{escape(CLINIC)}/index.html'>&larr; Back to {escape(CLINIC_NAME)}</a>" if CLINIC and st.session_state.get("from_clinic") else "")
st.markdown(f"<div class='brand'><b>SymbioForge</b><span>Heart-health decision support</span><span style='margin-left:auto'>{back}</span></div>", unsafe_allow_html=True)
view = st.segmented_control("View", ["Health check", "How it works"], default="Health check", key="view", label_visibility="collapsed")
if view == "How it works":
    mechanics.render()
else:
    assess.render()
