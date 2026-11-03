"""Where the demo clinic site lives, for the 'back to clinic' and 'book' links.
Set it with the SYMBIOFORGE_CLINIC_URL environment variable (run_demo.py does this for local runs) or the CLINIC_URL Streamlit secret
(Streamlit Community Cloud: app settings -> Secrets). Never taken from the page URL, so there are no open redirects.
If unset, the links are simply not shown."""
import os

CLINIC_NAME = "Meridian Heart & Family Clinic (demo)"


def clinic_url() -> str | None:
    url = os.environ.get("SYMBIOFORGE_CLINIC_URL")
    if not url:
        try:
            import streamlit as st
            url = st.secrets.get("CLINIC_URL")
        except Exception:
            url = None
    return url.rstrip("/") if url else None
