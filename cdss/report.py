"""Print-ready HTML clinical summary report (FR11). Browser 'Print -> Save as PDF' gives the PDF."""
from __future__ import annotations

import datetime as dt
from html import escape as e

CSS = """body{font:14px/1.5 system-ui,Segoe UI,sans-serif;max-width:760px;margin:2rem auto;color:#1a202c;padding:0 1rem}
h1{font-size:1.4rem;margin:0}h2{font-size:1rem;border-bottom:1px solid #cbd5e0;padding-bottom:.2rem;margin-top:1.6rem}
table{border-collapse:collapse;width:100%}td,th{padding:.25rem .5rem;border-bottom:1px solid #edf2f7;text-align:left}
.v{font-size:1.2rem;font-weight:600}.warn{background:#fffaf0;border-left:3px solid #dd6b20;padding:.4rem .7rem;margin:.3rem 0}
.disc{margin-top:2rem;font-size:.85rem;border:1px solid #cbd5e0;padding:.6rem}@media print{body{margin:0}}"""


def build_report(r: dict) -> str:
    c, rl = r["confidence"], r["rules"]
    inp = "".join(f"<tr><td>{e(k)}</td><td>{e(str(v)) if v is not None else '<i>not provided</i>'}</td></tr>"
                  for k, v in r["inputs"].items())
    exp = "".join(f"<tr><td>{e(x['label'])}</td><td>{x['value']:.3g}</td><td>{x['shap']:+.2f}</td><td>{e(x['direction'])}</td></tr>"
                  for x in r["explanation"])
    warns = "".join(f"<div class='warn'>{e(w)}</div>" for w in c["warnings"]) or "<p>No reliability warnings.</p>"
    cf = r["counterfactual"]
    if cf["found"]:
        def fmt(x):
            return "; ".join(f"{k}: {v['from']} to {v['to']}" for k, v in x["changes"].items()) + f" (new P={x['new_probability']:.0%})"
        cft = (f"<p>Model sensitivity, modifiable factors only (+-30%): {e(fmt(cf['actionable']))}</p>" if cf["actionable"] else
               "<p>No plausible change to modifiable factors alone flips the estimate.</p>")
        cft += f"<p>Model sensitivity, smallest change of any inputs: {e(fmt(cf['any']))}</p>" if cf["any"] else ""
    else:
        cft = f"<p>{e(cf['note'])}</p>"
    adv = ""
    if r["advice"]["needed"]:
        adv = "<h2>Suggested next tests</h2><ul>" + "".join(
            f"<li>{e(x['test'])} (cost {x['cost']}): P ranges {x['p_min']:.0%} to {x['p_max']:.0%}"
            f"{' - could change the verdict' if x['can_flip_verdict'] else ''}</li>" for x in r["advice"]["ranking"]) + "</ul>"
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>CDSS report</title><style>{CSS}</style></head><body>
<h1>SymbioForge CDSS - Clinical decision support report</h1><small>Generated {dt.datetime.now():%Y-%m-%d %H:%M}</small>
<div class='disc'><b>{e(r['disclaimer'])}</b></div>
<h2>Result</h2><p class='v'>Disease {e(r['verdict'])} - P(disease) = {r['models']['xgb']:.0%} (XGBoost), {r['models']['dnn']:.0%} (DNN)</p>
<p>Confidence: <b>{e(c['band'])}</b>. Conformal set at {c['conformal_level']:.0%}: {e(' / '.join(c['conformal_set']))}.
Validation accuracy in this confidence band: {c['validation_accuracy_in_band'] if c['validation_accuracy_in_band'] is None else format(c['validation_accuracy_in_band'], '.0%')} (n={c['validation_n_in_band']}).</p>
<p>{e(r['summary'])}</p><h2>Warnings</h2>{warns}
<h2>Rule base cross-check</h2><p>Verdict: {e(rl['verdict'])}. Fired: {e(', '.join(rl['fired']) or 'none')}. Unknown (missing inputs): {e(', '.join(rl['unknown']) or 'none')}.</p>
<h2>Top contributing factors (SHAP, XGBoost)</h2><table><tr><th>Factor</th><th>Value</th><th>SHAP</th><th>Effect</th></tr>{exp}</table>
<h2>What would change the estimate</h2>{cft}{adv}<h2>Inputs</h2><table>{inp}</table>
<div class='disc'>{e(r['disclaimer'])} Probabilities come from models trained on the public UCI Heart Disease dataset and do not replace clinical judgement.</div>
</body></html>"""
