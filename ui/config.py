"""Where the demo clinic site lives. Fixed by environment, never taken from the URL (no open redirects)."""
import os

CLINIC_URL = os.environ.get("SYMBIOFORGE_CLINIC_URL", "http://localhost:8600").rstrip("/")
CLINIC_NAME = "Meridian Heart & Family Clinic (demo)"
