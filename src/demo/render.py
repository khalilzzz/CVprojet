# Cours : Sessions 1-7 : dessin des méthodes sur les images de la démo (points, droites épipolaires, points de fuite, flot, spectre)
"""Drawing of the demo frames (demo.py).

Top: the real clip (left) and the generated clip (right), each with the overlay of the
method of the current chapter. Bottom: the title of the chapter and the table of every
value (real / generated / gap / curve over the clip). Pure functions of the measurements of
src/demo/measure.py: nothing is computed here except what is drawn.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from src.demo.measure import INDICATORS

W, H = 1920, 1080
PW, PH = 960, 540                      # one clip panel
S = 16                                 # sub-pixel shift of the OpenCV drawings (2^4)

# colours, BGR (OpenCV)
BG = (23, 19, 16)
ROW_HL = (52, 42, 34)
RED = (45, 45, 240)
GREY = (150, 150, 150)
EPI = (110, 230, 110)
YELLOW = (40, 220, 255)
REAL_C = (232, 150, 60)                # blue
FAKE_C = (40, 145, 250)                # orange
VP_COLORS = [(0, 210, 255), (255, 210, 0), (250, 100, 240)]   # yellow, cyan, magenta
TEXT = (240, 240, 240)
DIM = (140, 136, 130)

CHAPTERS = {
    "title": ("Réel ou IA ? Les méthodes du cours, image par image",
              "À gauche la vidéo réelle, à droite la vidéo générée à partir de sa première image. "
              "Chaque valeur est recalculée sur chaque image par notre code.", "Titre"),
    "v1": ("V1 · Géométrie épipolaire (Sessions 3-4)",
           "Points SIFT appariés entre t−10 et t. Rouge : compatibles avec la matrice fondamentale F "
           "(RANSAC + 8 points normalisé), gris : rejetés. Vert : droites épipolaires de F.", "V1 épipolaire"),
    "tv": ("V1 · Test à trois vues (Session 7)",
           "Points suivis sur t−24, t−12 et t (trait rouge). Triangulation depuis les deux premières vues, "
           "caméra de la 3e vue par DLT + RANSAC. Jaune : erreur de reprojection ×20 ; gris : rejetés par RANSAC.", "Trois vues"),
    "i2": ("I2 · Lignes parallèles et points de fuite (Sessions 1-3)",
           "Segments LSD groupés par RANSAC : une couleur = une famille de droites parallèles en 3D, "
           "qui convergent vers un point de fuite (pointillés). Gris : segments sans point de fuite.",
           "I2 points de fuite"),
    "v3": ("V3 · Cohérence temporelle (Session 6)",
           "Flot optique de Farnebäck : l'image t−1 est recalée sur t. Couleur : détail fin qui change malgré "
           "la compensation du mouvement. Blanc : flot incohérent (occlusion).", "V3 flot"),
    "i1": ("I1 · Empreinte fréquentielle (Sessions 1-2)",
           "Fond : résidu de bruit (image − filtre médian). Encart : son spectre de Fourier moyenné sur les "
           "images vues ; un point isolé brillant = motif périodique d'upsampling.", "I1 spectre"),
    "summary": ("Bilan · médiane sur tout le clip",
                "Même scène, même traitement. Verdict : erreur à trois vues comparée à mouvement égal, "
                "intervalle à 95 % par bootstrap. Comparaison de deux clips, pas un détecteur.", "Bilan"),
}


# ---------------------------------------------------------------- helpers
@lru_cache(maxsize=None)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans",
                                                             weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


def rgb(c):
    return (c[2], c[1], c[0])


def fmt(v: float, d: int) -> str:
    return "—" if not np.isfinite(v) else f"{v:.{d}f}".replace(".", ",")


def fit_frame(img: np.ndarray, w: int = PW, h: int = PH):
    """Image scaled into a w x h panel, letterboxed. Returns (panel, (scale, ox, oy), (nw, nh))."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = int(round(iw * s)), int(round(ih * s))
    out = np.zeros((h, w, 3), np.uint8)
    ox, oy = (w - nw) // 2, (h - nh) // 2
    out[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out, (s, ox, oy), (nw, nh)


def _p(pt, g) -> tuple[int, int]:
    """Image point -> panel point, sub-pixel (shift S)."""
    s, ox, oy = g
    return int(round((pt[0] * s + ox) * S)), int(round((pt[1] * s + oy) * S))


def _subsample(idx: np.ndarray, n: int) -> np.ndarray:
    return idx if len(idx) <= n else idx[np.linspace(0, len(idx) - 1, n).round().astype(int)]


def _line_ends(l: np.ndarray, iw: int, ih: int):
    """Two far points of the image line a u + b v + c = 0 (image coordinates)."""
    a, b, c = l
    if abs(b) > abs(a):
        return (-iw, -(a * -iw + c) / b), (2 * iw, -(a * 2 * iw + c) / b)
    return (-(b * -ih + c) / a, -ih), (-(b * 2 * ih + c) / a, 2 * ih)


def _dashed(img, p, q, color, dash=12, gap=9, thickness=1):
    p, q = np.asarray(p, float), np.asarray(q, float)
    L = np.linalg.norm(q - p)
    if L < 1:
        return
    u = (q - p) / L
    s = 0.0
    while s < L:
        a, b = p + u * s, p + u * min(s + dash, L)
        cv2.line(img, (int(a[0] * S), int(a[1] * S)), (int(b[0] * S), int(b[1] * S)), color,
                 thickness, cv2.LINE_AA, shift=4)
        s += dash + gap


# ---------------------------------------------------------------- overlays of the methods
def draw_v1(img, m: dict, g, shape, d: dict) -> None:
    v = m.get("v1")
    if not v or v.get("F") is None:
        return
    x, xp, mask, F = v["x"], v["xp"], v["mask"], v["F"]
    ih, iw = shape
    inl = np.flatnonzero(mask)
    # a few epipolar lines of F in frame t (x'^T F x = 0: the line of x is F x)
    for j in _subsample(inl[np.argsort(xp[inl, 0])], int(d["n_epipolar_lines"])):
        p, q = _line_ends(F @ np.r_[x[j], 1.0], iw, ih)
        ok, a, b = cv2.clipLine((0, 0, PW * S, PH * S), _p(p, g), _p(q, g))
        if ok:
            cv2.line(img, a, b, EPI, 1, cv2.LINE_AA, shift=4)
    for j in _subsample(np.flatnonzero(~mask), int(d["max_drawn_matches"]) // 3):
        u, w = _p(xp[j], g)
        cv2.drawMarker(img, (u // S, w // S), GREY, cv2.MARKER_TILTED_CROSS, 6, 1, cv2.LINE_AA)
    for j in _subsample(inl, int(d["max_drawn_matches"])):
        cv2.line(img, _p(x[j], g), _p(xp[j], g), (60, 60, 200), 1, cv2.LINE_AA, shift=4)
        cv2.circle(img, _p(xp[j], g), 3 * S, RED, -1, cv2.LINE_AA, shift=4)


def draw_three_view(img, m: dict, g, d: dict) -> None:
    tv = m.get("tv")
    if not tv:
        return
    k = float(d["error_magnify"])
    inl = np.asarray(tv["inliers"], bool)
    for j in _subsample(np.flatnonzero(~inl), int(d["max_drawn_matches"]) // 3):
        u, w = _p(tv["a3"][j], g)
        cv2.drawMarker(img, (u // S, w // S), GREY, cv2.MARKER_TILTED_CROSS, 6, 1, cv2.LINE_AA)
    for j in _subsample(np.flatnonzero(inl), int(d["max_drawn_matches"]) * 2 // 3):
        a1, a2, a3, pr = tv["a1"][j], tv["a2"][j], tv["a3"][j], tv["proj"][j]
        cv2.polylines(img, [np.array([_p(a1, g), _p(a2, g), _p(a3, g)], np.int32)], False,
                      (70, 70, 210), 1, cv2.LINE_AA, shift=4)
        e = (pr - a3) * k
        if np.all(np.isfinite(e)):
            cv2.line(img, _p(a3, g), _p(a3 + e, g), YELLOW, 2, cv2.LINE_AA, shift=4)
        cv2.circle(img, _p(a3, g), 3 * S, RED, -1, cv2.LINE_AA, shift=4)


def draw_i2(img, m: dict, g, shape) -> None:
    f = m.get("i2")
    if not f:
        return
    seg, lab, vps = f["segments"], f["labels"], f["vps"]
    ih, iw = shape
    over = img.copy()
    for k, vp in enumerate(vps):
        if abs(vp[2]) < 1e-12:
            continue
        u = vp[:2] / vp[2]
        if not (-2 * iw < u[0] < 3 * iw and -2 * ih < u[1] < 3 * ih):
            continue                                       # far away: lines nearly parallel in the image
        fam = np.flatnonzero(lab == k)
        length = np.hypot(seg[fam, 2] - seg[fam, 0], seg[fam, 3] - seg[fam, 1])
        vp_px = np.array(_p(u, g), float) / S
        for j in fam[np.argsort(-length)][:10]:
            mid = 0.5 * (seg[j, :2] + seg[j, 2:])
            _dashed(over, np.array(_p(mid, g), float) / S, vp_px, VP_COLORS[k % 3])
        if 0 <= vp_px[0] < PW and 0 <= vp_px[1] < PH:
            cv2.circle(over, (int(vp_px[0] * S), int(vp_px[1] * S)), 10 * S, VP_COLORS[k % 3], 2,
                       cv2.LINE_AA, shift=4)
    cv2.addWeighted(over, 0.55, img, 0.45, 0, dst=img)
    for j in range(len(seg)):
        c, th = (VP_COLORS[lab[j] % 3], 3) if lab[j] >= 0 else (GREY, 1)
        cv2.line(img, _p(seg[j, :2], g), _p(seg[j, 2:], g), c, th, cv2.LINE_AA, shift=4)


def draw_v3(img, m: dict, g, size) -> None:
    v = m.get("v3")
    if not v:
        return
    s, ox, oy = g
    nw, nh = size
    region = img[oy:oy + nh, ox:ox + nw]
    heat = cv2.applyColorMap(cv2.resize(v["hres"], (nw, nh)), cv2.COLORMAP_INFERNO)
    out = cv2.addWeighted(region, 0.30, heat, 0.85, 0)
    bad = cv2.resize(v["bad"], (nw, nh)) > 127
    out[bad] = (0.4 * out[bad] + 0.6 * 255).astype(np.uint8)
    region[:] = out


def draw_i1(img, gray: np.ndarray, s1: dict, g, size, labels: list) -> None:
    s, ox, oy = g
    nw, nh = size
    r = np.abs(gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32))
    r = cv2.resize(np.clip(r * 12, 0, 255).astype(np.uint8), (nw, nh), interpolation=cv2.INTER_AREA)
    region = img[oy:oy + nh, ox:ox + nw]
    region[:] = cv2.addWeighted(region, 0.25, cv2.cvtColor(r, cv2.COLOR_GRAY2BGR), 0.9, 0)
    # spectrum inset and radial profile, bottom right of the clip
    sz = 170
    x0, y0 = ox + nw - sz - 12, oy + nh - sz - 12
    inset = cv2.applyColorMap(cv2.resize(s1["inset"], (sz, sz)), cv2.COLORMAP_MAGMA)
    img[y0:y0 + sz, x0:x0 + sz] = inset
    cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + sz, y0 + sz), TEXT, 1)
    pw_, ph_ = 190, sz
    px0 = x0 - pw_ - 14
    cv2.rectangle(img, (px0, y0), (px0 + pw_, y0 + ph_), (30, 30, 30), -1)
    prof = np.asarray(s1["prof"], float)
    if np.isfinite(prof).all() and np.ptp(prof) > 0:
        yy = y0 + ph_ - 10 - (prof - prof.min()) / np.ptp(prof) * (ph_ - 34)
        xx = px0 + 8 + np.linspace(0, pw_ - 16, len(prof))
        cv2.polylines(img, [np.stack([xx, yy], 1).round().astype(np.int32)], False, (120, 200, 255),
                      2, cv2.LINE_AA)
    for xl, wl, text in ((x0, sz, "spectre FFT du résidu"), (px0, pw_, "profil radial")):
        cv2.rectangle(img, (xl - 1, y0 - 26), (xl + wl, y0 - 1), (30, 30, 30), -1)
        labels.append(((xl + wl // 2, y0 - 13), text, 15, TEXT, False, "mm"))


# ---------------------------------------------------------------- whole frame
class Renderer:
    """Builds the demo frames of one real / generated pair."""

    def __init__(self, names: tuple[str, str], frames: tuple[list, list], meas: tuple[dict, dict],
                 series: tuple[dict, dict], fps: float, d: dict, chapter_keys: list[str],
                 verdict: dict | None = None):
        self.names, self.frames, self.meas, self.series = names, frames, meas, series
        self.verdict = verdict                                   # src/demo/verdict.compare, shown in the summary
        self.n = min(len(frames[0]), len(frames[1]))
        self.fps, self.d, self.chapter_keys = fps, d, chapter_keys
        self.grays = tuple([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in fr[: self.n]] for fr in frames)
        self.ranges = {}
        for key, *_ in INDICATORS:
            v = np.concatenate([series[0][key][0][: self.n], series[1][key][0][: self.n]])
            v = v[np.isfinite(v)]
            if len(v):
                lo, hi = np.percentile(v, [2, 98])
                pad = 0.1 * (hi - lo) if hi > lo else max(abs(hi), 1e-3) * 0.1
                self.ranges[key] = (lo - pad, hi + pad)

    # one clip panel with the overlay of the chapter
    def panel(self, side: int, t: int, chapter: str, labels: list) -> np.ndarray:
        frame = self.frames[side][t]
        img, g, size = fit_frame(frame)
        m = self.meas[side]["frames"][t]
        shape = frame.shape[:2]
        if chapter == "v1":
            draw_v1(img, m, g, shape, self.d)
        elif chapter == "tv":
            draw_three_view(img, m, g, self.d)
        elif chapter == "i2":
            draw_i2(img, m, g, shape)
        elif chapter == "v3":
            draw_v3(img, m, g, size)
        elif chapter == "i1":
            sub = []
            draw_i1(img, self.grays[side][t], self.meas[side]["i1"][t], g, size, sub)
            labels += [((x + side * PW, y), *rest) for (x, y), *rest in sub]
        elif chapter == "title" or (chapter == "summary" and self.verdict is not None):
            img[:] = (img * 0.45).astype(np.uint8)
        return img

    def frame(self, t: int, chapter: str) -> np.ndarray:
        t = int(np.clip(t, 0, self.n - 1))
        tv = self.n - 1 if chapter == "summary" else t          # the summary shows the final values
        canvas = np.full((H, W, 3), BG, np.uint8)
        labels: list = []                                       # (xy, text, size, colour, bold, anchor)
        for side in (0, 1):
            canvas[:PH, side * PW:(side + 1) * PW] = self.panel(side, t, chapter, labels)
        cv2.line(canvas, (PW, 0), (PW, PH), (0, 0, 0), 3)
        for side, (tag, col) in enumerate((("RÉEL", REAL_C), ("IA", FAKE_C))):
            x0 = side * PW + 16
            cv2.rectangle(canvas, (x0, 14), (x0 + 74, 50), col, -1)
            labels += [((x0 + 37, 32), tag, 22, (255, 255, 255), True, "mm"),
                       ((x0 + 88, 32), self.names[side], 19, TEXT, False, "lm")]
        if chapter == "title":
            labels += [((W // 2, PH // 2 - 30), "Réel ou IA ?", 64, (255, 255, 255), True, "mm"),
                       ((W // 2, PH // 2 + 36), "Les méthodes du cours appliquées image par image", 28,
                        (230, 230, 230), False, "mm")]
        if chapter == "summary" and self.verdict is not None:
            self._verdict(canvas, labels)
        self._bottom(canvas, t, tv, chapter, labels)
        return self._text(canvas, labels)

    def _bottom(self, canvas, t: int, tv: int, chapter: str, labels: list) -> None:
        title, desc, _ = CHAPTERS[chapter]
        labels.append(((40, 572), title, 30, TEXT, True, "lm"))
        labels.append(((W - 40, 572), f"t = {fmt(t / self.fps, 2)} s", 20, DIM, False, "rm"))
        y = 606
        for line in _wrap(desc, font(18), W - 80):
            labels.append(((40, y), line, 18, (200, 200, 200), False, "lm"))
            y += 24
        # chapter chips
        x = 40
        for key in self.chapter_keys:
            name = CHAPTERS[key][2]
            wtxt = font(15).getlength(name)
            cur = key == chapter
            cv2.rectangle(canvas, (x, 650), (int(x + wtxt + 20), 674), TEXT if cur else (70, 66, 62),
                          -1 if cur else 1, cv2.LINE_AA)
            labels.append(((int(x + 10), 662), name, 15, BG if cur else DIM, cur, "lm"))
            x = int(x + wtxt + 32)
        # table
        y0, rh = 694, 32
        cols = dict(real=930, fake=1060, gap=1185, sx0=1225, sx1=W - 40)
        labels += [((40, y0), "indicateur", 15, DIM, False, "lm"),
                   ((cols["real"], y0), "réel", 15, REAL_C, True, "rm"),
                   ((cols["fake"], y0), "IA", 15, FAKE_C, True, "rm"),
                   ((cols["gap"], y0), "écart IA", 15, DIM, False, "rm"),
                   ((cols["sx0"], y0), "valeur image par image au fil du clip", 15, DIM, False, "lm")]
        for r, (key, label, chap, dec, kind) in enumerate(INDICATORS):
            yc = y0 + 22 + r * rh + rh // 2
            cur = chap == chapter or chapter == "summary"
            if cur:
                cv2.rectangle(canvas, (24, yc - rh // 2 + 2), (W - 24, yc + rh // 2 - 2), ROW_HL, -1)
            vr, vf = self.series[0][key][1][tv], self.series[1][key][1][tv]
            labels += [((40, yc), label, 18, TEXT if cur else DIM, False, "lm"),
                       ((cols["real"], yc), fmt(vr, dec), 21, REAL_C if cur else DIM, cur, "rm"),
                       ((cols["fake"], yc), fmt(vf, dec), 21, FAKE_C if cur else DIM, cur, "rm")]
            if np.isfinite(vr) and np.isfinite(vf) and abs(vr) > 1e-9:
                labels.append(((cols["gap"], yc), f"{(vf - vr) / abs(vr) * 100:+.0f} %", 17,
                               TEXT if cur else DIM, False, "rm"))
            self._spark(canvas, key, tv, yc, rh, cols["sx0"], cols["sx1"], cur)

    def _verdict(self, canvas, labels: list) -> None:
        """Verdict box over the two (darkened) clip panels."""
        from src.demo.verdict import describe
        v = self.verdict
        colour = {"fake_less_coherent": FAKE_C, "real_less_coherent": REAL_C}.get(v["code"], GREY)
        x0, x1, y0, y1 = 150, W - 150, 120, 440
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (32, 27, 23), -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, 3, cv2.LINE_AA)
        labels += [((W // 2, y0 + 40), "VERDICT", 22, DIM, True, "mm"),
                   ((W // 2, y0 + 95), v["text"], 38, colour, True, "mm"),
                   ((W // 2, y0 + 160), describe(v["main"]) + "  (à mouvement égal)", 24, TEXT, False, "mm")]
        y = y0 + 205
        for r in v["support"]:
            mark = {"fake_higher": "IA plus haute", "real_higher": "réel plus haut",
                    "same": "pas d'écart", "n/a": "—"}[r["call"]]
            labels.append(((W // 2, y), f"appui · {describe(r)} · {mark}", 19, (200, 200, 200), False, "mm"))
            y += 30
        ov = v["main"]["overlap"]
        if np.isfinite(ov):
            labels.append(((W // 2, y1 - 28), f"recouvrement des mouvements : {ov * 100:.0f} % des images "
                           f"(minimum {v['min_overlap'] * 100:.0f} %)", 16, DIM, False, "mm"))

    def _spark(self, canvas, key, t, yc, rh, x0, x1, cur) -> None:
        if key not in self.ranges:
            return
        lo, hi = self.ranges[key]
        top, bot = yc - rh // 2 + 6, yc + rh // 2 - 6
        cv2.line(canvas, (x0, bot), (x1, bot), (60, 56, 52), 1)
        for side, col in ((0, REAL_C), (1, FAKE_C)):
            c = self.series[side][key][0][: t + 1]
            xs = x0 + np.arange(len(c)) * (x1 - x0) / max(self.n - 1, 1)
            ys = bot - (np.clip(c, lo, hi) - lo) / (hi - lo) * (bot - top)
            colour = col if cur else tuple(int(0.5 * v) for v in col)
            ok = np.isfinite(ys)
            # draw each run of finite values as one polyline
            for run in np.split(np.arange(len(c)), np.flatnonzero(np.diff(ok.astype(int))) + 1):
                if len(run) and ok[run[0]]:
                    pts = np.stack([xs[run], ys[run]], 1)
                    cv2.polylines(canvas, [(pts * S).round().astype(np.int32)], False, colour,
                                  2 if cur else 1, cv2.LINE_AA, shift=4)
            if len(c) and ok[-1] and cur:
                cv2.circle(canvas, (int(xs[-1] * S), int(ys[-1] * S)), 4 * S, colour, -1, cv2.LINE_AA, shift=4)

    @staticmethod
    def _text(canvas: np.ndarray, labels: list) -> np.ndarray:
        im = Image.fromarray(canvas[..., ::-1])
        dr = ImageDraw.Draw(im)
        for (x, y), text, size, colour, bold, anchor in labels:
            dr.text((x, y), text, font=font(size, bold), fill=rgb(colour), anchor=anchor)
        return np.asarray(im)[..., ::-1].copy()


def _wrap(text: str, f: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if f.getlength(trial) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    return lines + [cur] if cur else lines
