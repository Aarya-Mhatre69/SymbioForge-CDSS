"""Shared look: warm neutral surface, one blue accent, semantic green/amber/red only for status."""
import streamlit as st

CSS = """
<style>
:root{--ink:#1f2933;--muted:#667085;--line:#e6e1d8;--card:#ffffff;--accent:#1f5f8b;--soft:#f1eee9}
.block-container{max-width:980px;padding-top:4.6rem;padding-bottom:5rem}
h1,h2,h3{letter-spacing:-.01em}
#MainMenu,footer{visibility:hidden}
.brand{display:flex;align-items:baseline;gap:.6rem;margin-bottom:.2rem}
.brand b{font-size:1.45rem;letter-spacing:-.02em}.brand span{color:var(--muted);font-size:.92rem}
.backlink{font-size:.85rem;color:var(--accent)!important;text-decoration:none}.backlink:hover{text-decoration:underline}
.stepper{display:flex;gap:6px;margin:.6rem 0 1rem}
.stepper i{flex:1;height:4px;border-radius:4px;background:var(--line)}.stepper i.on{background:var(--accent)}
.steplabel{color:var(--muted);font-size:.82rem;margin-top:-.4rem;margin-bottom:.4rem}
.safety{background:#fff6f5;border:1px solid #f3cfcb;color:#7a2a24;border-radius:10px;padding:.55rem .85rem;font-size:.86rem;margin:.2rem 0 1rem}
.hero{background:var(--card);border:1px solid var(--line);border-left:6px solid var(--c);border-radius:14px;padding:1.2rem 1.4rem;margin:.4rem 0 .8rem}
.hero .tier{color:var(--c);font-weight:600;font-size:.95rem;letter-spacing:.01em}
.hero .big{font-size:2.6rem;font-weight:650;letter-spacing:-.03em;line-height:1.1;margin:.15rem 0}
.hero .big small{font-size:.95rem;font-weight:400;color:var(--muted);letter-spacing:0;margin-left:.4rem}
.hero p{margin:.4rem 0 .8rem;color:#344054;line-height:1.55}
.meters{display:flex;gap:2rem;flex-wrap:wrap}
.meter{min-width:200px;flex:1}.meter label{font-size:.76rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.bar{height:7px;border-radius:7px;background:var(--soft);margin:.3rem 0 .2rem;overflow:hidden}.bar>div{height:100%;border-radius:7px;background:var(--accent)}
.meter .v{font-size:.9rem;font-weight:550}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:.8rem 1rem;margin-bottom:.6rem}
.card h4{margin:0 0 .15rem;font-size:.95rem}.card .val{font-size:1.15rem;font-weight:600}
.card .note{color:var(--muted);font-size:.84rem}
.chip{display:inline-block;padding:.1rem .55rem;border-radius:99px;font-size:.72rem;font-weight:600;margin-left:.4rem;vertical-align:middle}
.chip.ok{background:#e6f4ec;color:#22633f}.chip.watch{background:#fbf0d9;color:#8a5a0b}.chip.flag{background:#fbe5e3;color:#a12a22}
.chip.info{background:#eceff3;color:#4a5565}.chip.missing{background:#eceff3;color:#4a5565}
.opin{display:flex;gap:.6rem;flex-wrap:wrap;margin:.4rem 0 1rem}
.opin .card{flex:1;min-width:200px;margin:0}.opin .big{font-size:1.4rem;font-weight:600}
.drv{display:flex;gap:.6rem;align-items:flex-start;padding:.45rem 0;border-bottom:1px solid var(--line)}
.drv:last-child{border:0}.drv .dot{width:9px;height:9px;border-radius:50%;margin-top:.45rem;flex:none}
.agent{display:flex;justify-content:space-between;gap:1rem;padding:.35rem 0;border-bottom:1px solid var(--line);font-size:.88rem}
.agent small{color:var(--muted)}
.disc{font-size:.78rem;color:var(--muted);margin-top:1.4rem}
div[data-testid="stChatMessage"]{background:transparent}
.stButton>button{border-radius:10px}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def stepper(i: int, n: int, label: str):
    st.markdown("<div class='stepper'>" + "".join(f"<i class='{'on' if k <= i else ''}'></i>" for k in range(n)) + "</div>"
                f"<div class='steplabel'>Step {i + 1} of {n} · {label}</div>", unsafe_allow_html=True)


def chip(level: str) -> str:
    text = {"ok": "In range", "watch": "Worth watching", "flag": "Outside range", "info": "Info", "missing": "Not provided"}[level]
    return f"<span class='chip {level}'>{text}</span>"
