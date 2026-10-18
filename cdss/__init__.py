"""SymbioForge CDSS: decision support for cardiovascular disease. Not a diagnosis."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
DATA = ROOT / "data"
DISCLAIMER = "Decision support only, not a diagnosis. Prototype trained on a public research dataset; not validated for clinical use."
