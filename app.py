"""SymbioForge CDSS. Run: streamlit run app.py"""
import streamlit as st

from ui import assess, mechanics, style

st.set_page_config(page_title="SymbioForge", page_icon="❤", layout="wide")
style.inject()

st.markdown("<div class='brand'><b>SymbioForge</b><span>Heart-health decision support</span></div>", unsafe_allow_html=True)
view = st.segmented_control("View", ["Health check", "How it works"], default="Health check", key="view", label_visibility="collapsed")
if view == "How it works":
    mechanics.render()
else:
    assess.render()
