"""'How it works' view for the mentor / examiner: architecture, model internals, evaluation, data, limits."""
from __future__ import annotations

import json

import plotly.graph_objects as go
import streamlit as st

from cdss import REPORTS, ROOT
from cdss.engine import DEMO_PATIENTS, LABELS, get_engine
from cdss.orchestrator import assess
from cdss.rules import RULES
from cdss.schemas import PatientInput

DOT = """
digraph G {
  rankdir=TB; bgcolor="transparent"; nodesep=.3; ranksep=.45;
  node [shape=box, style="rounded,filled", fillcolor="#eef2f7", color="#9fb3c8", fontname="Helvetica", fontsize=12, margin="0.14,0.08"];
  edge [color="#8a99a8", arrowsize=.7];
  user [label="User\\n(chat intake)", fillcolor="#fff"];
  intake [label="Intake agent\\nTask 4: validate,\\nflag missing"];
  feat [label="Feature agent\\nTask 5: 12 features"];
  orch [label="Orchestrator\\nTask 9: runs the workflows,\\nmerges the evidence", fillcolor="#dbe8f5", color="#1f5f8b"];
  xgb [label="Risk model\\nTask 6: XGBoost"];
  dnn [label="Second opinion\\nTask 7: DNN"];
  rules [label="Clinical rules\\nTasks 1-2: R1-R7"];
  shap [label="Explainer\\nTask 8: SHAP"];
  trust [label="Trust agent\\nTask 8: bands,\\nconformal set"];
  whatif [label="What-if agent\\nTask 8: counterfactuals"];
  plan [label="Test planner\\nTask 3: A* search"];
  synth [label="Synthesizer\\nTask 9: overview,\\nquestions, report", fillcolor="#dbe8f5", color="#1f5f8b"];
  out [label="Overview + chat\\n(Streamlit)", fillcolor="#fff"];
  user -> intake -> feat -> orch;
  orch -> xgb; orch -> dnn; orch -> rules;
  xgb -> shap; xgb -> trust; xgb -> whatif; rules -> plan;
  dnn -> trust; rules -> trust;
  shap -> synth; trust -> synth; whatif -> synth; plan -> synth; xgb -> synth; dnn -> synth; rules -> synth;
  synth -> out;
}
"""


def _demo_ctx():
    if "ctx" in st.session_state:
        return st.session_state.ctx, "your current assessment"
    return assess(PatientInput(**DEMO_PATIENTS["Borderline / missing data"])), "the built-in example 'Borderline, tests missing'"


def render():
    e = get_engine()
    st.markdown("### How SymbioForge works")
    st.caption("For reviewers: how the eight earlier workflows are orchestrated into one system, and the evidence behind it.")
    ctx, which = _demo_ctx()
    r = ctx["result"]
    t = st.tabs(["Architecture", "Prediction internals", "Evaluation", "Data (EDA)", "Limits and decisions"])

    with t[0]:
        st.markdown(
            "SymbioForge is not one classifier. It is **a team of AI workflows from Tasks 1-8 under a single orchestrator**. "
            "Each workflow is independent and reports what it did; the orchestrator merges their evidence into a patient-level overview.")
        st.graphviz_chart(DOT, width="stretch")
        st.markdown(f"**Live trace for {which}**")
        st.dataframe([{"agent": x["agent"], "from": x["task"], "ms": round(x["ms"], 1), "result": x["note"]} for x in r["trace"]],
                     hide_index=True, width="stretch")
        st.caption("Design rules: one feature function (no train/serve skew), rules never evaluate guessed inputs, "
                   "stateless (no patient data stored), deterministic output.")

    with t[1]:
        c1, c2, c3 = st.columns(3)
        c1.metric("XGBoost P(disease)", f"{r['models']['xgb']:.1%}")
        c2.metric("DNN P(disease)", f"{r['models']['dnn']:.1%}")
        c3.metric("Verdict (XGBoost at 0.5)", r["verdict"])
        st.markdown("**Rule base trace** (three-valued: fired / not fired / unknown)")
        st.dataframe([{"rule": k, "definition": RULES[k].split(" -> ")[0],
                       "state": "fired" if s else "not fired" if s is False else "unknown"} for k, s in r["rules"]["state"].items()],
                     hide_index=True, width="stretch")
        st.markdown("**Engineered features** (exact order fed to the models)")
        st.dataframe([{"feature": k, "meaning": LABELS[k], "value": round(v, 3), "SHAP": round(r["shap_all"][k], 3)}
                      for k, v in r["features"].items()], hide_index=True, width="stretch")
        ex = sorted(r["shap_all"].items(), key=lambda kv: abs(kv[1]))
        fig = go.Figure(go.Bar(x=[v for _, v in ex], y=[LABELS[k] for k, _ in ex], orientation="h",
                               marker_color=["#c0504d" if v > 0 else "#2b6cb0" for _, v in ex]))
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="SHAP (log-odds)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
        c = r["confidence"]
        st.markdown(f"**Trust layer:** band `{c['band']}`, conformal set `{' / '.join(c['conformal_set'])}` at {c['conformal_level']:.0%}.")
        st.dataframe([{"confidence range": f"{b['lo']:.2f}-{min(b['hi'], 1):.2f}", "validation patients": b["n"],
                       "accuracy": None if b["accuracy"] is None else round(b["accuracy"], 3)} for b in e.meta["bands"]],
                     hide_index=True, width="stretch")

    with t[2]:
        p = REPORTS / "evaluation.json"
        if p.exists():
            ev = json.loads(p.read_text())
            st.dataframe([{"model": k, **{m: round(v[m], 3) for m in ("accuracy", "precision", "recall", "roc_auc", "brier")}}
                          for k, v in ev["models"].items()], hide_index=True, width="stretch")
            g1, g2 = st.columns(2)
            g1.image(str(REPORTS / "figures" / "roc.png"))
            g2.image(str(REPORTS / "figures" / "calibration.png"))
            ci = ev["models"]["xgb"]["auc_ci95"]
            st.caption(f"Held-out test n = {ev['models']['xgb']['n']}; XGBoost AUC 95% CI {ci[0]:.2f}-{ci[1]:.2f}. Model differences are within noise.")
            ru = ev["rules"]
            st.markdown(f"**Cross-checks.** Rules decide {ru['coverage_determinate']:.0%} of patients; when rules and ML agree the ML is right "
                        f"{ru['ml_accuracy_when_agree']:.0%} of the time, versus {ru['ml_accuracy_when_disagree']:.0%} when they disagree. "
                        f"That is why disagreement lowers confidence.")
            st.markdown("**Fairness (test set)**")
            st.dataframe([{"group": f"{k}: {g}", "n": v["n"], "AUC": round(v["roc_auc"], 3) if v["roc_auc"] else None,
                           "recall": round(v["recall"], 3), "prevalence": round(v["prevalence"], 2)}
                          for k, gs in ev["fairness"].items() for g, v in gs.items()], hide_index=True, width="stretch")
        else:
            st.info("Run `python evaluate.py` to generate evaluation results.")

    with t[3]:
        if (REPORTS / "EDA_report.md").exists():
            a, b = st.columns(2)
            a.image(str(REPORTS / "figures" / "eda_correlation.png"), caption="Correlation of the 12 model features")
            b.image(str(REPORTS / "figures" / "eda_missing_by_site.png"), caption="Missingness depends on the hospital")
            st.image(str(REPORTS / "figures" / "eda_target_by_category.png"))
            with st.expander("Full EDA report"):
                st.markdown("\n".join(ln for ln in (REPORTS / "EDA_report.md").read_text(encoding="utf-8").splitlines() if not ln.startswith("![")))
        else:
            st.info("Run `python eda.py` to generate the EDA report.")

    with t[4]:
        st.markdown("""**Limitations**
- Historical research data from four hospitals, not current or local patients; not clinically validated.
- Small test set (138): every metric is approximate.
- Women and some sites are under-represented; site prevalence differs a lot and confounds missing data.
- Optional tests (`slope`, `ca`, `thal`) are often missing in training; absent values are filled with the most common value and flagged.
- Counterfactuals describe the model, not treatment effects. The chat layer is rule-based and can only answer from computed evidence.
""")
        arch = (ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
        if "## 5. Decision Log" in arch:
            st.markdown("**Decision log**")
            st.markdown(arch.split("## 5. Decision Log", 1)[1])
