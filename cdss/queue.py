"""Demo clinician queue (SQLite). A patient may explicitly send their overview to the clinic; nothing is stored otherwise.
DEMO ONLY: no authentication, fictional data. A real deployment would need consent management, encryption, access control and audit logging."""
from __future__ import annotations

import json
import sqlite3
import time

from . import DATA

DB = DATA / "queue.db"
ORDER = {"higher": 0, "unsure": 1, "mid": 2, "lower": 3}


def _conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    c.execute("""CREATE TABLE IF NOT EXISTS cases (id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL, name TEXT, stage INTEGER,
                 tier TEXT, tier_label TEXT, headline TEXT, flags TEXT, steps TEXT, drivers TEXT, inputs TEXT, status TEXT DEFAULT 'new', alerts TEXT DEFAULT '[]')""")
    try:
        c.execute("ALTER TABLE cases ADD COLUMN alerts TEXT DEFAULT '[]'")
    except sqlite3.OperationalError:
        pass
    return c


def submit(ctx: dict, name: str, consent: bool) -> int:
    """Store an overview. Refuses without explicit consent."""
    if not consent:
        raise PermissionError("Patient consent is required before sending anything to the clinic.")
    a = ctx["analysis"]
    flags = [f"{c['label']}: {c['value']} ({c['note']})" for c in a["cards"] if c["level"] == "flag"]
    alerts = [w for w in ctx["result"]["confidence"]["warnings"] if "disagree" in w.lower() or "not shown as reliable" in w.lower()]
    drivers = [f"{d['label']} ({d['value']}) {d['direction']} the pattern" for d in a["drivers"]]
    with _conn() as c:
        cur = c.execute("INSERT INTO cases (created,name,stage,tier,tier_label,headline,flags,steps,drivers,inputs,alerts) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (time.time(), (name or "Anonymous (demo)")[:60], ctx.get("stage", 2), a["tier"], a["tier_info"]["label"], a["headline"],
                         json.dumps(flags), json.dumps(a["next"]["steps"]), json.dumps(drivers), json.dumps(ctx["patient"]), json.dumps(alerts)))
        return cur.lastrowid


def cases(status: str | None = None) -> list[dict]:
    with _conn() as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM cases" + (" WHERE status=?" if status else ""), (status,) if status else ())]
    for r in rows:
        for k in ("flags", "steps", "drivers", "inputs", "alerts"):
            r[k] = json.loads(r[k])
    # a model/rule disagreement bumps a case one tier up (conservative triage); on a tie the bumped case goes first, then newest first
    def key(r):
        dis = any("disagree" in a.lower() for a in r["alerts"])
        return (r["status"] != "new", ORDER.get(r["tier"], 9) * 2 - (2 if dis else 0), not dis, -r["created"])
    return sorted(rows, key=key)


def mark_reviewed(case_id: int):
    with _conn() as c:
        c.execute("UPDATE cases SET status='reviewed' WHERE id=?", (case_id,))


def clear():
    with _conn() as c:
        c.execute("DELETE FROM cases")
