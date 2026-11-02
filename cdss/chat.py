"""Deterministic, zero-cost conversational layer: parses the opening message and answers follow-up questions
from the assessment. No LLM; every answer is assembled from computed evidence, so nothing is invented."""
from __future__ import annotations

import re

from .glossary import FIELDS, TEST_PLAIN, lookup

SUGGESTIONS = ["Why is my result like this?", "What should I do next?", "How sure is this?",
               "What if my cholesterol or blood pressure were lower?", "What does ST depression mean?"]
SAFETY = ("If you have severe or worsening chest pain, pain spreading to the arm, jaw or back, breathlessness, sweating, or fainting, "
          "call your local emergency number now. Do not wait for an online tool.")


def parse_intro(text: str) -> dict:
    """Very light extraction from free text; everything is shown back to the user to confirm."""
    t = text.lower()
    out: dict = {}
    m = re.search(r"\b(\d{2})\s*(?:-?\s*(?:year|yr|y/o|yo)s?\b|\s*(?:year|yr)s?\s*old)", t) or re.search(r"\bi(?:'m| am)\s+(\d{2})\b", t) or re.search(r"\bage[d]?\s*(\d{2})\b", t)
    if m and 18 <= int(m.group(1)) <= 100:
        out["age"] = int(m.group(1))
    if re.search(r"\b(female|woman|girl|she|her)\b", t):
        out["sex"] = "female"
    elif re.search(r"\b(male|man|boy|he|his)\b", t):
        out["sex"] = "male"
    if re.search(r"\b(no chest pain|without chest pain|no pain)\b", t):
        out["pain"] = False
    elif re.search(r"chest|angina|tightness|pressure|squeez", t):
        out["pain"] = True
        out["pressure"] = bool(re.search(r"pressure|tight|squeez|heavy|crushing|weight on", t))
        out["exertion"] = bool(re.search(r"exert|climb|stairs|walk|run|exercis|effort|stress|physical", t))
        out["relief"] = bool(re.search(r"(rest|stop|sit|sitting).{0,30}(eas|better|relie|go away|settle|pass)|eas\w*.{0,25}rest|relie\w*.{0,25}rest|rest.{0,20}help", t))
    return out


def understood(parsed: dict) -> str:
    bits = []
    if "age" in parsed:
        bits.append(f"age {parsed['age']}")
    if "sex" in parsed:
        bits.append(parsed["sex"])
    if parsed.get("pain") is True:
        feats = [n for k, n in (("pressure", "pressure or tightness"), ("exertion", "brought on by exertion"), ("relief", "eased by rest")) if parsed.get(k)]
        bits.append("chest discomfort" + (f" ({', '.join(feats)})" if feats else ""))
    elif parsed.get("pain") is False:
        bits.append("no chest pain")
    return ", ".join(bits)


def _pct(x: float) -> str:
    return f"{x:.0%}"


def answer(q: str, ctx: dict) -> str:
    """ctx = assessment dict from orchestrator.assess."""
    t = q.lower()
    r, a = ctx["result"], ctx["analysis"]
    if re.search(r"emergenc|urgent|right now|severe|911|112|ambulance", t):
        return SAFETY
    topic = lookup(t) if re.search(r"what (is|does|are)|mean|explain|\bwhy do you ask\b|\bwhat's\b", t) else None
    if topic:
        f = FIELDS[topic]
        return f"**{f['label']}**: {f['what']} {f['matters']} Typical range: {f['normal']}. Measured by: {f['how']}."
    if re.search(r"what if|lower|reduce|improv|change|cholesterol|blood pressure|\bbp\b", t) and not topic:
        mods = a["modifiable"]
        cf = r["counterfactual"]
        parts = []
        if mods:
            parts.append("The factors that lifestyle and treatment can influence in your results are: " +
                         ", ".join(f"{m['label'].lower()} ({m['note'].lower()})" for m in mods) + ".")
        else:
            parts.append("None of your modifiable measurements (blood pressure, cholesterol, blood sugar) were outside the comfortable range.")
        act = cf.get("actionable")
        if act:
            parts.append("In the model, " + "; ".join(f"moving {FIELDS[k]['label'].lower()} from {v['from']} to {v['to']}" for k, v in act["changes"].items()) +
                         f" would lower the estimate to about {_pct(act['new_probability'])}. This is how the model behaves, not a promise of what treatment would do.")
        elif cf.get("found") is False:
            parts.append("The estimate is robust: no small, plausible change in these measurements moves it across the 50% line.")
        parts.append("The What-if tab shows how the estimate moves across the whole range.")
        return " ".join(parts)
    if re.search(r"next|should i|do now|advice|recommend|test", t):
        nx = a["next"]
        s = nx["steps"][0]
        if nx["tests"]:
            s += " Tests that could add the most clarity: " + "; ".join(
                f"{x['name'].split(' (')[0]}" + (" (could change the verdict)" if x["can_flip"] else "") for x in nx["tests"]) + "."
        return s + " " + SAFETY
    if re.search(r"sure|confiden|reliable|trust|accura|certain", t):
        c = r["confidence"]
        s = (f"Confidence is **{c['band']}**. Among validation patients with a similar level of model certainty, about "
             f"{_pct(c['validation_accuracy_in_band'])} were classified correctly ({c['validation_n_in_band']} patients, so this is approximate). ")
        s += f"{a['completeness']['given']} of {a['completeness']['of']} inputs were provided. "
        if len(c["conformal_set"]) == 2:
            s += "The model could not separate the two outcomes for someone with your answers, so this result is genuinely uncertain. "
        s += " ".join(c["warnings"])
        return s
    if re.search(r"second opinion|rules?\b|disagree|other model|deep", t):
        so = a["second_opinions"]
        return (f"Three independent methods looked at your answers. The machine-learning model gives {_pct(so['ml']['p'])}; the deep-learning model "
                f"gives {_pct(so['dl']['p'])}; the expert rule base says '{so['rules']['verdict']}'"
                + (f" (rules {', '.join(so['rules']['fired'])} fired)" if so["rules"]["fired"] else "") + ". Agreement makes a result more dependable; disagreement is a reason for caution.")
    if re.search(r"why|result|driv|reason|explain|because|factor", t):
        d = a["drivers"]
        lines = [f"- {x['label']} ({x['value']}) {x['direction']} the estimate." for x in d]
        extra = ""
        if any(x["feature"] == "cp_asymptomatic" for x in d):
            extra = " Note: in the training data, having no chest pain was linked to more disease found on testing, so it counts against you here."
        return a["headline"] + "\n\nMain influences:\n" + "\n".join(lines) + extra
    return ("I can explain your result, your measurements, what to do next, how sure the result is, what-if changes, or any term "
            "(for example 'What does thallium mean?'). " + "Try one of the suggested questions below.")
