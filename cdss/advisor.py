"""Next-test advisor (FR10). Task 3 formulation: state = set of tests performed, action = perform one more,
goal = rule verdict is determinate whatever the outcomes. A* with test costs from context.md §4."""
from __future__ import annotations

import heapq
import itertools

from . import rules

COST = {"slope": 4, "thal": 6, "ca": 8}
VALUES = {
    "slope": ["upsloping", "flat", "downsloping"],
    "thal": ["normal", "fixed defect", "reversable defect"],
    "ca": [0, 1, 2, 3],
}


def _resolved(raw: dict, tested: frozenset, missing: list[str]) -> bool:
    """Verdict determinate for EVERY outcome of the tested fields (untested ones stay unknown)."""
    tested = [t for t in tested]
    for combo in itertools.product(*(VALUES[t] for t in tested)):
        p = dict(raw)
        for t in missing:
            p[t] = None
        p.update(dict(zip(tested, combo)))
        if not rules.determinate(p):
            return False
    return True


def astar_plan(raw: dict, missing: list[str]) -> dict:
    """Cheapest set of tests that makes the rule verdict determinate. Heuristic: cheapest remaining test (admissible)."""
    if not missing:
        return {"tests": [], "cost": 0, "expanded": 0}
    start = frozenset()
    heap = [(min(COST[m] for m in missing), 0, 0, start)]
    seen, expanded, tick = set(), 0, 0
    while heap:
        _, g, _, s = heapq.heappop(heap)
        if s in seen:
            continue
        seen.add(s)
        expanded += 1
        if _resolved(raw, s, missing):
            return {"tests": sorted(s, key=COST.get), "cost": g, "expanded": expanded}
        rest = [m for m in missing if m not in s]
        for m in rest:
            ns = s | {m}
            tick += 1
            left = [x for x in rest if x != m]
            h = min((COST[x] for x in left), default=0)
            heapq.heappush(heap, (g + COST[m] + h, g + COST[m], tick, ns))
    return {"tests": [], "cost": 0, "expanded": expanded}


def advise(raw: dict, missing: list[str], prob_fn) -> dict:
    """prob_fn(dict)->P(present) from the primary model. Ranks each missing test by how far it could move the verdict."""
    if not missing:
        return {"needed": False, "ranking": [], "plan": {"tests": [], "cost": 0, "expanded": 0}}
    ranking = []
    for m in missing:
        probs = []
        for v in VALUES[m]:
            p = dict(raw)
            p[m] = v
            probs.append(float(prob_fn(p)))
        flips = (min(probs) < 0.5) != (max(probs) < 0.5)
        ranking.append({
            "test": m, "cost": COST[m], "p_min": min(probs), "p_max": max(probs),
            "spread": max(probs) - min(probs), "can_flip_verdict": bool(flips),
            "value_per_cost": (max(probs) - min(probs)) / COST[m],
        })
    ranking.sort(key=lambda r: (-r["can_flip_verdict"], -r["value_per_cost"]))
    return {"needed": True, "ranking": ranking, "plan": astar_plan(raw, missing)}
