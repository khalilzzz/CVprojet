# Cours : hors cours (restitution) : rapport HTML autonome de chaque run
"""Self-contained HTML report of one run: figures are embedded, tables are plain HTML."""
from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd

CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3de;--real:#2a78d6;--fake:#eb6834}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,Segoe UI,Arial,sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:25px;margin:0 0 4px} h2{font-size:19px;margin:38px 0 8px} h3{font-size:15px;margin:22px 0 6px}
p,li{max-width:78ch} .muted{color:var(--ink2)} .meta{color:var(--ink2);font-size:13px}
img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;margin:8px 0}
.scroll{overflow-x:auto}
table{border-collapse:collapse;font-size:13px;margin:8px 0;font-variant-numeric:tabular-nums}
th,td{padding:5px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{color:var(--ink2);font-weight:600} th:first-child,td:first-child{text-align:left}
td.real{color:var(--real);font-weight:600} td.fake{color:var(--fake);font-weight:600}
details{margin:6px 0} summary{cursor:pointer;font-weight:600}
code{background:#f1f0ec;padding:1px 5px;border-radius:4px;font-size:13px}
"""


def _img(path: Path, alt: str) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{data}">'


def _table(df: pd.DataFrame, digits: int = 3) -> str:
    if df is None or df.empty:
        return '<p class="muted">No data.</p>'
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for _, r in df.iterrows():
        cells = []
        for c, v in r.items():
            if isinstance(v, float):
                txt = "-" if pd.isna(v) else f"{v:.{digits}f}"
            else:
                txt = html.escape(str(v))
            cls = f' class="{v}"' if c == "label" and v in ("real", "fake") else ""
            cells.append(f"<td{cls}>{txt}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


class Report:
    """Collects sections, then writes one HTML file."""

    def __init__(self, title: str, subtitle: str):
        self.title, self.parts = title, [f"<h1>{html.escape(title)}</h1>",
                                         f'<p class="meta">{html.escape(subtitle)}</p>']

    def h2(self, text: str):
        self.parts.append(f"<h2>{html.escape(text)}</h2>")

    def h3(self, text: str):
        self.parts.append(f"<h3>{html.escape(text)}</h3>")

    def p(self, text: str, muted: bool = False):
        self.parts.append(f'<p class="muted">{text}</p>' if muted else f"<p>{text}</p>")

    def ul(self, items: list[str]):
        self.parts.append("<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")

    def figure(self, path: Path, alt: str):
        self.parts.append(_img(path, alt))

    def table(self, df: pd.DataFrame, digits: int = 3):
        self.parts.append(_table(df, digits))

    def details(self, summary: str, inner: list[str]):
        self.parts.append(f"<details><summary>{html.escape(summary)}</summary>{''.join(inner)}</details>")

    def img_html(self, path: Path, alt: str) -> str:
        return _img(path, alt)

    def write(self, out: Path) -> Path:
        doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
               f'<meta name="viewport" content="width=device-width,initial-scale=1">'
               f"<title>{html.escape(self.title)}</title><style>{CSS}</style></head>"
               f"<body><main>{''.join(self.parts)}</main></body></html>")
        out = Path(out)
        out.write_text(doc, encoding="utf-8")
        return out
