# Meridian Heart & Family Clinic (fictional demo)

A static presentation shell showing where the SymbioForge heart-health app could live inside a clinic website. The clinic, phone numbers and addresses are fictional placeholders.

- Open: double-click `index.html`, or run `python -m http.server 8600` inside this folder and visit http://localhost:8600.
- Start the SymbioForge Streamlit app separately (default http://localhost:8501) so the embedded check loads.
- Change the app URL: edit `data-app-url` in `heart-check.html` and `clinicians.html`.
- Booking form data stays in this browser's localStorage (`meridian_bookings`); nothing is sent anywhere.
- No login, no real data, no external resources. Demo only, not medical advice.
