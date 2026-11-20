# Cours : Session 7 (erreur de reprojection à trois vues) et Session 3 (mouvement image) : verdict comparatif de la démo
"""Comparative verdict of the demo: which of the two clips agrees less with a real camera?

The demo always shows a real clip and a generated clip of the same scene, so the verdict
compares them instead of judging one clip alone (pixel errors depend on the scene, the
resolution and the motion, so a fixed threshold would not mean much).

Main judge: the three-view reprojection error (Session 7). All errors grow with the image
motion, and generated clips often move more or less than the real one, so the two clips are
compared at equal motion: on the frames of both clips,

    log(error) = a + b * log(motion) + c * [generated clip]

and exp(c) is the generated / real ratio at equal motion. Its 95 % interval comes from a
moving-block bootstrap of the frames (neighbouring frames are not independent).

Outcomes:
- the motions of the two clips barely overlap: no verdict (the fit would extrapolate);
- interval above 1: the generated clip agrees less with a rigid camera;
- interval below 1: the real clip agrees less (it happens: moving objects, rolling shutter);
- otherwise: no measurable difference.

Supporting indicators (Sampson distance, V3 texture change) get the same fit and are shown as
agreeing or not with the main judge. They do not change the verdict: on our own pairs, the
Sampson distance also measures the rendering (vid4 control) and V3 changes direction between
pairs.

Checked on our 6 pairs vid5 to vid10 (scratchpad audit of 2026-11-20): main judge above 1 on
vid6, vid7 and vid9, not significant on vid10, below 1 on vid8, motions not comparable on vid5.
This is a measurement on two clips, not a detector.
"""
from __future__ import annotations

import numpy as np

DEFAULTS = dict(main="tv_err", support=["v1_sampson", "v3_hf"], motion="v1_motion",
                min_motion_px=0.3, min_overlap=0.3, min_frames=24, n_boot=1000, block=12,
                seed=0)

LABELS = dict(tv_err="erreur à trois vues", v1_sampson="distance de Sampson",
              v3_hf="changement de texture V3")


def _clean(values: np.ndarray, motion: np.ndarray, min_motion: float):
    ok = np.isfinite(values) & np.isfinite(motion) & (values > 0) & (motion > min_motion)
    return np.log(values[ok]), np.log(motion[ok])


def _fit(lr, mr, lf, mf) -> np.ndarray:
    """Least squares of log(value) on [1, log(motion), is_generated]."""
    y = np.r_[lr, lf]
    X = np.c_[np.ones(len(y)), np.r_[mr, mf], np.r_[np.zeros(len(lr)), np.ones(len(lf))]]
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _blocks(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    block = min(block, n)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, k)
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def motion_overlap(mr: np.ndarray, mf: np.ndarray) -> float:
    """Share of the frames of both clips whose (log) motion lies in the range common to both."""
    lo, hi = max(mr.min(), mf.min()), min(mr.max(), mf.max())
    if hi <= lo:
        return 0.0
    m = np.r_[mr, mf]
    return float(np.mean((m >= lo) & (m <= hi)))


def ratio_at_equal_motion(real: dict, fake: dict, key: str, c: dict,
                          rng: np.random.Generator) -> dict:
    """Generated / real ratio of indicator `key` at equal image motion, with its 95 % interval.
    real, fake: indicator series of measure.indicator_series ({key: (per-frame curve, ...)})."""
    lr, mr = _clean(real[key][0], real[c["motion"]][0], c["min_motion_px"])
    lf, mf = _clean(fake[key][0], fake[c["motion"]][0], c["min_motion_px"])
    out = dict(key=key, n_real=len(lr), n_fake=len(lf), ratio=np.nan, lo=np.nan, hi=np.nan,
               overlap=np.nan, raw_ratio=np.nan)
    if min(len(lr), len(lf)) < c["min_frames"]:
        return out
    out["overlap"] = motion_overlap(mr, mf)
    out["raw_ratio"] = float(np.exp(np.median(lf) - np.median(lr)))
    out["ratio"] = float(np.exp(_fit(lr, mr, lf, mf)[2]))
    boot = []
    for _ in range(int(c["n_boot"])):
        i, j = _blocks(len(lr), c["block"], rng), _blocks(len(lf), c["block"], rng)
        boot.append(_fit(lr[i], mr[i], lf[j], mf[j])[2])
    out["lo"], out["hi"] = (float(v) for v in np.exp(np.percentile(boot, [2.5, 97.5])))
    return out


def _call(r: dict) -> str:
    if not np.isfinite(r["ratio"]):
        return "n/a"
    return "fake_higher" if r["lo"] > 1 else "real_higher" if r["hi"] < 1 else "same"


def compare(real: dict, fake: dict, cfg: dict | None = None) -> dict:
    """Verdict on one real / generated pair. Returns a dict with the outcome code, the sentence
    to show, the main ratio and the supporting ratios."""
    c = dict(DEFAULTS, **(cfg or {}))
    rng = np.random.default_rng(c["seed"])
    main = ratio_at_equal_motion(real, fake, c["main"], c, rng)
    support = [ratio_at_equal_motion(real, fake, k, c, rng) for k in c["support"]]
    for r in [main] + support:
        r["call"] = _call(r)
    if not np.isfinite(main["ratio"]):
        code, text = "no_data", "Pas de verdict : trop peu d'images mesurées à trois vues"
    elif main["overlap"] < c["min_overlap"]:
        code, text = "not_comparable", "Pas de verdict : les deux caméras ne bougent pas assez pareil"
    elif main["call"] == "fake_higher":
        code, text = "fake_less_coherent", "La vidéo IA respecte moins la géométrie d'une vraie caméra"
    elif main["call"] == "real_higher":
        code, text = "real_less_coherent", "La vidéo réelle est la moins cohérente (objets mobiles ?)"
    else:
        code, text = "no_difference", "Pas de différence mesurable de géométrie entre les deux"
    agree = [r for r in support if main["call"] in ("fake_higher", "real_higher") and r["call"] == main["call"]]
    return dict(code=code, text=text, main=main, support=support, n_agree=len(agree),
                min_overlap=c["min_overlap"])


def describe(r: dict) -> str:
    """'erreur à trois vues : IA = 1,54 × réel [1,40 ; 1,65]' (French decimal comma)."""
    if not np.isfinite(r["ratio"]):
        return f"{LABELS.get(r['key'], r['key'])} : non mesurable"
    f = lambda v: f"{v:.2f}".replace(".", ",")  # noqa: E731
    return f"{LABELS.get(r['key'], r['key'])} : IA = {f(r['ratio'])} × réel [{f(r['lo'])} ; {f(r['hi'])}]"


def rows(verdict: dict) -> list[dict]:
    """Flat rows for the CSV written next to the demo video."""
    out = []
    for role, r in [("main", verdict["main"])] + [("support", s) for s in verdict["support"]]:
        out.append(dict(role=role, indicator=r["key"], ratio_at_equal_motion=r["ratio"],
                        ci_low=r["lo"], ci_high=r["hi"], raw_ratio=r["raw_ratio"],
                        motion_overlap=r["overlap"], n_real=r["n_real"], n_fake=r["n_fake"],
                        call=r["call"]))
    return out
