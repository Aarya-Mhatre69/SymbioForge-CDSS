"""One command for the whole demo: python run_demo.py
Starts the SymbioForge app (8501) and the demo clinic website (8600), then opens the CLINIC site, which is the front door.
Press Ctrl+C to stop both."""
from __future__ import annotations

import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_PORT, CLINIC_PORT = 8501, 8600


def up(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait(port: int, name: str, timeout: float = 90) -> bool:
    t = time.time()
    while time.time() - t < timeout:
        if up(port):
            return True
        time.sleep(0.5)
    print(f"{name} did not start on port {port} within {timeout:.0f}s")
    return False


def main() -> int:
    procs = []
    if up(CLINIC_PORT) or up(APP_PORT):
        print(f"Port {CLINIC_PORT} or {APP_PORT} is already in use; stop the other process first.")
        return 1
    try:
        procs.append(subprocess.Popen([sys.executable, "-m", "http.server", str(CLINIC_PORT), "--directory", str(ROOT / "clinic")],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        procs.append(subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.port", str(APP_PORT),
                                       "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                                      cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        print("Starting the demo clinic website and the SymbioForge app ...")
        if not (wait(CLINIC_PORT, "Clinic site") and wait(APP_PORT, "SymbioForge app")):
            return 1
        url = f"http://localhost:{CLINIC_PORT}/index.html"
        print(f"\nDemo is ready.\n  Clinic website (start here): {url}\n  SymbioForge app:             http://localhost:{APP_PORT}\n"
              f"  Clinician queue (passcode: demo): http://localhost:{APP_PORT}/?view=clinician\n\nPress Ctrl+C to stop both.")
        webbrowser.open(url)
        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()
        print("Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
