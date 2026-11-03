"""'How it works' view for the mentor / examiner: architecture, model internals, evaluation, data, limits.
Fast by design: one section is rendered at a time, nothing runs a model (a stored snapshot is used), and only light HTML is drawn."""
from __future__ import annotations

import json

import streamlit as st

from cdss import ARTIFACTS, REPORTS, ROOT
from cdss.demo import LABELS
from cdss.rules import RULES

from . import lite

FIG = REPORTS / "figures"
SECTIONS = ["Architecture", "Prediction internals", "Evaluation", "Data (EDA)", "Limits and decisions"]
WORKFLOWS = [
    ("Intake agent", "Task 4", "validates 13 inputs; flags and fills missing optional tests"),
    ("Feature agent", "Task 5", "builds the 12 model features from one formula module"),
    ("Risk model (XGBoost)", "Task 6", "probability of disease; gives the verdict"),
    ("Second opinion (DNN)", "Task 7", "independent deep-learning estimate; flags disagreement"),
    ("Clinical rules", "Tasks 1-2", "rules R1-R7: fired / not fired / unknown"),
    ("Test planner (A*)", "Task 3", "picks the cheapest tests that make the rules decisive"),
    ("Explainer (SHAP)", "Task 8", "exact per-feature contributions"),
    ("Trust agent", "Task 8", "confidence band and 90% conformal set"),
    ("What-if agent", "Task 8", "sensitivity to plausible changes"),
    ("Synthesizer", "Task 9", "overview, doctor questions, report"),
]


@st.cache_data(show_spinner=False)
def _json(name: str, base: str = "reports"):
    p = (REPORTS if base == "reports" else ARTIFACTS) / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data(show_spinner=False)
def _text(path: str) -> str:
    from pathlib import Path
    return Path(path).read_text(encoding="utf-8")


def _snapshot():
    """Use the user's own latest analysis if there is one, otherwise the stored example."""
    ctx = st.session_state.get("ctx")
    if ctx:
        r = ctx["result"]
        return {"trace": r["trace"], "rules": r["rules"], "features": r["features"], "shap_all": r["shap_all"], "models": r["models"],
                "verdict": r["verdict"], "confidence": r["confidence"]}, "your current assessment"
    d = _json("demo_trace.json")
    return d, f"the built-in example '{d['example']}'" if d else ("", "")


def _architecture(snap, which):
    st.markdown("SymbioForge is not one classifier. It is **a team of AI workflows from Tasks 1-8 under a single orchestrator**. "
                "Each workflow is independent and reports what it did; the orchestrator merges their evidence into a patient-level overview.")
    arch = ROOT / "docs" / "figures" / "architecture.png"
    if arch.exists():
        lite.show(lite.img(arch, "Orchestrated architecture: each box is an independent workflow", 1000))
    lite.show(lite.table([[a, b, c] for a, b, c in WORKFLOWS], ["Workflow", "From", "What it does"]))
    if snap:
        st.markdown(f"**Live trace for {which}**")
        lite.show(lite.table([[x["agent"], x["task"], f"{x['ms']:.0f}", x["note"]] for x in snap["trace"]],
                             ["Agent", "From", "ms", "Result"], num_cols=(2,)))
    st.caption("Design rules: one feature function (no train/serve skew), rules never evaluate guessed inputs, "
               "stateless (no patient data stored), deterministic output.")


def _internals(snap):
    if not snap:
        st.info("Run `python build_demo_trace.py` to create the example snapshot.")
        return
    lite.show(lite.kpis([(f"{snap['models']['xgb']:.1%}", "XGBoost P(disease)", "#1f5f8b"), (f"{snap['models']['dnn']:.1%}", "DNN P(disease)", "#1f5f8b"),
                         (snap["verdict"], "Verdict (XGBoost at 0.5)", "#b7791f")]))
    st.markdown("**Rule base trace** (three-valued: fired / not fired / unknown)")
    lite.show(lite.table([[k, RULES[k].split(" -> ")[0], "fired" if s else "not fired" if s is False else "unknown"]
                          for k, s in snap["rules"]["state"].items()], ["Rule", "Definition", "State"]))
    st.markdown("**What pushed the estimate** (SHAP, log-odds)")
    items = sorted(snap["shap_all"].items(), key=lambda kv: -abs(kv[1]))
    lite.show(lite.bars([(LABELS[k], v) for k, v in items], axis="lowers the estimate  ←   →  raises the estimate"))
    st.markdown("**Engineered features** (exact order fed to the models)")
    lite.show(lite.table([[k, LABELS[k], f"{v:.3f}", f"{snap['shap_all'][k]:+.3f}"] for k, v in snap["features"].items()],
                         ["Feature", "Meaning", "Value", "SHAP"], num_cols=(2, 3)))
    c = snap["confidence"]
    st.markdown(f"**Trust layer:** band `{c['band']}`, conformal set `{' / '.join(c['conformal_set'])}` at {c['conformal_level']:.0%}.")
    meta = _json("meta.json", "artifacts")
    if meta:
        lite.show(lite.table([[f"{b['lo']:.2f}-{min(b['hi'], 1):.2f}", b["n"], "-" if b["accuracy"] is None else f"{b['accuracy']:.3f}"]
                              for b in meta["bands"]], ["Confidence range", "Validation patients", "Accuracy"], num_cols=(1, 2)))


def _evaluation():
    ev = _json("evaluation.json")
    if not ev:
        st.info("Run `python evaluate.py` to generate evaluation results.")
        return
    lite.show(lite.table([[k] + [f"{v[m]:.3f}" for m in ("accuracy", "precision", "recall", "roc_auc", "brier")] for k, v in ev["models"].items()],
                         ["Model", "Accuracy", "Precision", "Recall", "ROC-AUC", "Brier"], num_cols=(1, 2, 3, 4, 5)))
    lite.show("<div class='lt-2'>" + lite.img(FIG / "roc.png", "ROC curves, held-out test", 520) + lite.img(FIG / "calibration.png", "Reliability diagram", 520) + "</div>")
    ci = ev["models"]["xgb"]["auc_ci95"]
    st.caption(f"Held-out test n = {ev['models']['xgb']['n']}; XGBoost AUC 95% CI {ci[0]:.2f}-{ci[1]:.2f}. Model differences are within noise.")
    ru = ev["rules"]
    st.markdown(f"**Cross-checks.** Rules decide {ru['coverage_determinate']:.0%} of patients; when rules and ML agree the ML is right "
                f"{ru['ml_accuracy_when_agree']:.0%} of the time, versus {ru['ml_accuracy_when_disagree']:.0%} when they disagree. That is why disagreement lowers confidence.")
    st.markdown("**Fairness (test set)**")
    lite.show(lite.table([[f"{k}: {g}", v["n"], f"{v['roc_auc']:.3f}" if v["roc_auc"] else "-", f"{v['recall']:.3f}", f"{v['prevalence']:.2f}"]
                          for k, gs in ev["fairness"].items() for g, v in gs.items()], ["Group", "n", "AUC", "Recall", "Prevalence"], num_cols=(1, 2, 3, 4)))


def _eda():
    if not (REPORTS / "EDA_report.md").exists():
        st.info("Run `python eda.py` to generate the EDA report.")
        return
    lite.show("<div class='lt-2'>" + lite.img(FIG / "eda_correlation.png", "Correlation of the 12 model features", 520)
              + lite.img(FIG / "eda_missing_by_site.png", "Missingness depends on the hospital", 520) + "</div>")
    lite.show(lite.img(FIG / "eda_target_by_category.png", "Disease rate by category", 900))
    with st.expander("Full EDA report"):
        st.markdown("\n".join(ln for ln in _text(str(REPORTS / "EDA_report.md")).splitlines() if not ln.startswith("![")))


def _limits():
    st.markdown("""**Limitations**
- Historical research data from four hospitals, not current or local patients; not clinically validated.
- Small test set (138): every metric is approximate.
- Women and some sites are under-represented; site prevalence differs a lot and confounds missing data.
- Optional tests (`slope`, `ca`, `thal`) are often missing in training; absent values are filled with the most common value and flagged.
- Counterfactuals describe the model, not treatment effects. The chat layer is rule-based and can only answer from computed evidence.
""")
    arch = _text(str(ROOT / "ARCHITECTURE.md"))
    if "## 5. Decision Log" in arch:
        st.markdown("**Decision log**")
        st.markdown(arch.split("## 5. Decision Log", 1)[1])


def render():
    st.markdown("### How SymbioForge works")
    st.caption("For reviewers: how the eight earlier workflows are orchestrated into one system, and the evidence behind it.")
    sec = st.segmented_control("Section", SECTIONS, default=SECTIONS[0], key="how_section", label_visibility="collapsed") or SECTIONS[0]
    snap, which = _snapshot()
    if sec == "Architecture":
        _architecture(snap, which)
    elif sec == "Prediction internals":
        _internals(snap)
    elif sec == "Evaluation":
        _evaluation()
    elif sec == "Data (EDA)":
        _eda()
    else:
        _limits()
