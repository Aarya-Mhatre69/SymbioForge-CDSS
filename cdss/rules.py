"""Task 2 rule base R1-R7, forward chaining with three-valued logic.
A rule whose inputs are missing is UNKNOWN, never evaluated on imputed values."""
from __future__ import annotations

from typing import Optional

RULES = {
    "R1": "cp = asymptomatic AND thal = reversable defect -> present",
    "R2": "slope = flat AND sex = male AND thalch < 116 -> present",
    "R3": "slope = flat AND sex = male AND ca = 1 -> present",
    "R4": "oldpeak > 2.8 -> present",
    "R5": "trestbps > 152.8 AND age in [58, 60] -> present",
    "R6": "age in [56, 58] AND cp = atypical angina -> present",
    "R7": "default: none of R1-R6 fire -> absent",
}
# which raw fields each rule can depend on (used by the advisor)
RULE_FIELDS = {
    "R1": {"cp", "thal"}, "R2": {"slope", "sex", "thalch"}, "R3": {"slope", "sex", "ca"},
    "R4": {"oldpeak"}, "R5": {"trestbps", "age"}, "R6": {"age", "cp"},
}


def _and(*conds: Optional[bool]) -> Optional[bool]:
    if any(c is False for c in conds):
        return False
    if any(c is None for c in conds):
        return None
    return True


def _eq(v, target) -> Optional[bool]:
    return None if v is None else v == target


def _cmp(v, op, t) -> Optional[bool]:
    if v is None:
        return None
    return v < t if op == "<" else v > t


def evaluate(p: dict) -> dict:
    """p: raw patient dict, unobserved optional fields = None. Returns per-rule state + verdict."""
    g = p.get
    state = {
        "R1": _and(_eq(g("cp"), "asymptomatic"), _eq(g("thal"), "reversable defect")),
        "R2": _and(_eq(g("slope"), "flat"), _eq(g("sex"), "male"), _cmp(g("thalch"), "<", 116)),
        "R3": _and(_eq(g("slope"), "flat"), _eq(g("sex"), "male"), _eq(g("ca"), 1)),
        "R4": _cmp(g("oldpeak"), ">", 2.8),
        "R5": _and(_cmp(g("trestbps"), ">", 152.8), 58 <= g("age") <= 60),
        "R6": _and(56 <= g("age") <= 58, _eq(g("cp"), "atypical angina")),
    }
    fired = [r for r, s in state.items() if s is True]
    unknown = [r for r, s in state.items() if s is None]
    if fired:
        verdict = "present"
    elif unknown:
        verdict = "undetermined"
    else:
        verdict = "absent"  # R7 default
    return {"state": state, "fired": fired, "unknown": unknown, "verdict": verdict}


def determinate(p: dict) -> bool:
    return evaluate(p)["verdict"] != "undetermined"
