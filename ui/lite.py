"""Lightweight rendering helpers: plain HTML/SVG instead of heavy front-end components (data grid, plotly, graphviz).
Each page then loads only a few small JavaScript chunks, which matters on a shared free-tier server."""
from __future__ import annotations

import base64
import io
from html import escape

import streamlit as st


@st.cache_data(show_spinner=False, max_entries=64)
def _img_b64(path: str, max_w: int) -> tuple[str, str]:
    """Downscale, optimise and base64-encode an image once; returns (mime, data)."""
    from PIL import Image

    im = Image.open(path)
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    if path.lower().endswith((".jpg", ".jpeg")):
        im.convert("RGB").save(buf, "JPEG", quality=82, optimize=True)
        mime = "image/jpeg"
    else:
        im.convert("P", palette=Image.ADAPTIVE, colors=128).save(buf, "PNG", optimize=True)
        mime = "image/png"
    return mime, base64.b64encode(buf.getvalue()).decode()


def img(path, caption: str | None = None, max_w: int = 900) -> str:
    mime, data = _img_b64(str(path), max_w)
    cap = f"<div class='cap'>{escape(caption)}</div>" if caption else ""
    return f"<figure class='lt-fig'><img alt='{escape(caption or '')}' src='data:{mime};base64,{data}'/>{cap}</figure>"


def table(rows: list[list], header: list[str], num_cols: tuple[int, ...] = ()) -> str:
    """rows: list of lists of plain strings (escaped here). Wrap a cell in a tuple ('html', text) to pass markup."""
    def cell(v, i, tag="td"):
        txt = v[1] if isinstance(v, tuple) else escape(str(v))
        cls = " class='n'" if i in num_cols else ""
        return f"<{tag}{cls}>{txt}</{tag}>"
    head = "".join(cell(h, i, "th") for i, h in enumerate(header))
    body = "".join("<tr>" + "".join(cell(v, i) for i, v in enumerate(r)) + "</tr>" for r in rows)
    return f"<div class='lt-wrap'><table class='lt'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def kpi(big: str, label: str, color: str = "#1f5f8b") -> str:
    return f"<div class='lt-kpi'><div class='v' style='color:{color}'>{escape(big)}</div><div class='l'>{escape(label)}</div></div>"


def kpis(items: list[tuple[str, str, str]]) -> str:
    return "<div class='lt-kpis'>" + "".join(kpi(*i) for i in items) + "</div>"


def bars(items: list[tuple[str, float]], pos="#c0504d", neg="#2b6cb0", axis="lowers  ←   →  raises") -> str:
    """Horizontal diverging bars centred on zero (replaces a plotly bar chart)."""
    m = max((abs(v) for _, v in items), default=1) or 1
    rows = []
    for label, v in items:
        w = abs(v) / m * 48
        side = f"left:50%;width:{w:.1f}%;background:{pos}" if v > 0 else f"left:{50 - w:.1f}%;width:{w:.1f}%;background:{neg}"
        rows.append(f"<div class='br'><div class='bl'>{escape(label)}</div><div class='bt'><i class='mid'></i><b style='{side}'></b></div>"
                    f"<div class='bv'>{v:+.2f}</div></div>")
    return f"<div class='lt-bars'>{''.join(rows)}<div class='ax'>{escape(axis)}</div></div>"


def curve(xs: list[float], ps: list[float], current: float, xlabel: str) -> str:
    """Step-line chart as inline SVG with the 50% line and a marker for the patient (replaces a plotly line chart)."""
    W, H, L, R, T, B = 640, 250, 46, 12, 10, 38
    x0, x1 = min(xs), max(xs)
    sx = lambda x: L + (x - x0) / ((x1 - x0) or 1) * (W - L - R)
    sy = lambda p: T + (1 - p) * (H - T - B)
    path = f"M{sx(xs[0]):.1f},{sy(ps[0]):.1f}"
    for i in range(1, len(xs)):
        path += f" H{sx(xs[i]):.1f} V{sy(ps[i]):.1f}"
    cur = min(range(len(xs)), key=lambda k: abs(xs[k] - current))
    grid = "".join(f"<line x1='{L}' x2='{W - R}' y1='{sy(v):.1f}' y2='{sy(v):.1f}' stroke='#eee'/>"
                   f"<text x='{L - 6}' y='{sy(v) + 4:.1f}' text-anchor='end' font-size='11' fill='#667085'>{v:.0%}</text>" for v in (0, .25, .5, .75, 1))
    ticks = "".join(f"<text x='{sx(x):.1f}' y='{H - 20}' text-anchor='middle' font-size='11' fill='#667085'>{x:g}</text>"
                    for x in (xs[0], xs[len(xs) // 2], xs[-1]))
    return (f"<svg viewBox='0 0 {W} {H}' class='lt-svg' role='img' aria-label='Estimated chance versus {escape(xlabel)}'>{grid}"
            f"<line x1='{L}' x2='{W - R}' y1='{sy(.5):.1f}' y2='{sy(.5):.1f}' stroke='#999' stroke-dasharray='4 4'/>"
            f"<path d='{path}' fill='none' stroke='#1f5f8b' stroke-width='2.5'/>"
            f"<circle cx='{sx(xs[cur]):.1f}' cy='{sy(ps[cur]):.1f}' r='6' fill='#c0504d'/>{ticks}"
            f"<text x='{(L + W - R) / 2}' y='{H - 4}' text-anchor='middle' font-size='12' fill='#344054'>{escape(xlabel)}</text></svg>")


def show(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)
