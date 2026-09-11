"""Image-aligned cue rendering. Display legibility and driver benefit are unvalidated."""
from __future__ import annotations
import math
import numpy as np
import cv2

# Risk bands. Colours are BGR. Shape coding is redundant with colour.
BAND_HIGH, BAND_MED = 0.55, 0.30
COL_HIGH = (56, 62, 232)      # red
COL_MED = (40, 168, 244)      # amber
COL_LOW = (150, 214, 120)     # muted green
COL_TEXT = (245, 245, 245)
COL_PANEL = (32, 30, 28)


def band(risk: float) -> str:
    return "high" if risk >= BAND_HIGH else ("med" if risk >= BAND_MED else "low")


def band_colour(risk: float):
    return {"high": COL_HIGH, "med": COL_MED, "low": COL_LOW}[band(risk)]


def _dashed_rect(img, p1, p2, colour, thickness, dash, gap, alpha=1.0):
    x1, y1 = p1
    x2, y2 = p2
    layer = img if alpha >= 0.999 else img.copy()
    pts = [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
           ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]
    for (ax, ay), (bx, by) in pts:
        L = math.hypot(bx - ax, by - ay)
        if L < 1:
            continue
        ux, uy = (bx - ax) / L, (by - ay) / L
        s = 0.0
        while s < L:
            e = min(s + dash, L)
            cv2.line(layer, (int(ax + ux * s), int(ay + uy * s)),
                     (int(ax + ux * e), int(ay + uy * e)), colour, thickness,
                     cv2.LINE_AA)
            s = e + gap
    if alpha < 0.999:
        cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, img)


def _corner_bracket(img, box, colour, thickness, frac=0.28):
    x1, y1, x2, y2 = [int(v) for v in box]
    lx = max(6, int((x2 - x1) * frac))
    ly = max(6, int((y2 - y1) * frac))
    for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                             (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (px, py), (px + dx * lx, py), colour, thickness, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * ly), colour, thickness, cv2.LINE_AA)


def _ground_marker(img, box, colour, unc):
    """Image-space ellipse at the base of a detected box."""
    x1, y1, x2, y2 = [int(v) for v in box]
    cx, cy = (x1 + x2) // 2, y2
    a = max(8, (x2 - x1) // 2)
    b = max(3, a // 4)
    layer = img.copy()
    cv2.ellipse(layer, (cx, cy), (a, b), 0, 0, 360, colour, 2, cv2.LINE_AA)
    cv2.addWeighted(layer, 0.85 - 0.35 * unc, img, 0.15 + 0.35 * unc, 0, img)


def _label(img, anchor, text, colour, scale=0.44):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = int(anchor[0]), int(anchor[1])
    x = max(2, min(x, img.shape[1] - tw - 12))
    y = max(th + 8, y)
    box = img[y - th - 7:y + 4, x:x + tw + 10]
    if box.size:
        img[y - th - 7:y + 4, x:x + tw + 10] = cv2.addWeighted(
            box, 0.28, np.full_like(box, COL_PANEL, dtype=np.uint8), 0.72, 0)
    cv2.line(img, (x, y - th - 7), (x, y + 4), colour, 2, cv2.LINE_AA)
    cv2.putText(img, text, (x + 6, y - 1), cv2.FONT_HERSHEY_SIMPLEX, scale,
                COL_TEXT, 1, cv2.LINE_AA)


def draw_cue(img, cue):
    """Render one selected cue in place."""
    risk = cue["risk"]
    unc = cue.get("uncertainty", 0.0)
    col = band_colour(risk)
    level = cue.get("level", "full")
    x1, y1, x2, y2 = [int(v) for v in cue["box"]]
    th = 2 if level != "full" else 3

    if level == "tick":
        _ground_marker(img, cue["box"], col, unc)
        return
    if level == "bracket":
        _corner_bracket(img, cue["box"], col, th)
        _ground_marker(img, cue["box"], col, unc)
        return

    # full cue: uncertainty is encoded as stroke discontinuity and halo softness
    if unc > 0.30:
        dash = int(max(4, 22 * (1.0 - unc)))
        gap = int(max(3, 16 * unc))
        _dashed_rect(img, (x1, y1), (x2, y2), col, th, dash, gap,
                     alpha=1.0 - 0.35 * unc)
        pad = int(3 + 7 * unc)
        _dashed_rect(img, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), col, 1,
                     dash, gap * 2, alpha=0.45 * (1 - unc) + 0.15)
    else:
        _corner_bracket(img, cue["box"], col, th, frac=0.34)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 1, cv2.LINE_AA)
    _ground_marker(img, cue["box"], col, unc)
    if cue.get("text"):
        conf_txt = "?" if unc > 0.55 else ""
        _label(img, (x1, y1 - 4), cue["text"] + conf_txt, col)


def risk_ribbon(img, cues, cam_w):
    """Optional bottom-image summary bar ordered by priority."""
    h, w = img.shape[:2]
    bar_h = max(10, h // 40)
    y0 = h - bar_h - 4
    strip = img[y0:y0 + bar_h, :]
    img[y0:y0 + bar_h, :] = cv2.addWeighted(
        strip, 0.40, np.full_like(strip, COL_PANEL, dtype=np.uint8), 0.60, 0)
    cv2.line(img, (0, y0), (w, y0), (110, 110, 110), 1, cv2.LINE_AA)
    for c in sorted(cues, key=lambda d: d["risk"]):
        cx = int((c["box"][0] + c["box"][2]) / 2)
        half = max(5, int(22 * c["risk"]))
        hh = int((bar_h - 4) * (0.40 + 0.60 * min(c["risk"] / 0.8, 1.0)))
        yb = y0 + bar_h - 2
        cv2.rectangle(img, (cx - half, yb - hh), (cx + half, yb),
                      band_colour(c["risk"]), -1, cv2.LINE_AA)


def status_panel(img, n_cues, n_objects, load, top_text, title="PRISM"):
    h, w = img.shape[:2]
    pw, ph = int(w * 0.30), 46
    roi = img[6:6 + ph, 6:6 + pw]
    img[6:6 + ph, 6:6 + pw] = cv2.addWeighted(
        roi, 0.30, np.full_like(roi, COL_PANEL, dtype=np.uint8), 0.70, 0)
    cv2.putText(img, f"{title}  cues {n_cues}/{n_objects}", (14, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_TEXT, 1, cv2.LINE_AA)
    cv2.putText(img, top_text[:38], (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (180, 215, 245), 1, cv2.LINE_AA)
    # load meter
    bx = 6 + pw - 68
    cv2.rectangle(img, (bx, 14), (bx + 58, 22), (90, 90, 90), 1)
    cv2.rectangle(img, (bx + 1, 15), (bx + 1 + int(56 * min(load, 1.0)), 21),
                  COL_HIGH if load > 0.85 else COL_MED if load > 0.6 else COL_LOW, -1)


def render_frame(frame, cues, n_objects=0, load=0.0, top_text="", ribbon=True,
                 panel=True, title="PRISM"):
    out = frame.copy()
    ranked = sorted(cues, key=lambda d: d["risk"])
    # Preserve allocated detail; full cues may carry their textual justification.
    n_lab = max(0, len(ranked) - 2)
    for i, c in enumerate(ranked):
        c = dict(c)
        if i < n_lab and c.get("level") != "full":
            c.pop("text", None)
        draw_cue(out, c)
    if ribbon and cues:
        risk_ribbon(out, cues, frame.shape[1])
    if panel:
        status_panel(out, len(cues), n_objects, load, top_text, title=title)
    return out


def overlay_mask(frame_shape, cues, ribbon=True, panel=True):
    """Binary mask of the pixels the interface paints, used for the ink metrics."""
    blank = np.zeros(frame_shape, dtype=np.uint8)
    painted = render_frame(blank, cues, ribbon=ribbon, panel=panel)
    return (painted.sum(axis=2) > 0).astype(np.uint8)
